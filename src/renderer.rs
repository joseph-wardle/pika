//! What one Hydra render delegate owns on the Rust side: its scene store,
//! the GPU copy of its meshes, its Vulkan device and the target it reads
//! frames back from.

use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ash::vk;

use crate::geometry::Geometry;
use crate::gpu;
use crate::raster::{self, Raster};
use crate::shader;
use crate::smooth::Smooth;
use crate::store::{self, Store};
use crate::target::Target;
use crate::transform::Camera;

/// The render buffer formats Pika writes, as Hydra names them. Color is
/// half or full float, as the host asks: husk asks for `HdFormatFloat32Vec4`,
/// and hosts that take the delegate's default get `HdFormatFloat16Vec4`,
/// which halves the bytes read back. Depth is `HdFormatFloat32` and normals
/// `HdFormatFloat32Vec3`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Float16Vec4,
    Float32Vec4,
    Float32,
    Float32Vec3,
}

impl Format {
    /// The value `pika.h` passes for this format.
    pub fn from_abi(value: u32) -> Result<Self, String> {
        match value {
            1 => Ok(Self::Float16Vec4),
            2 => Ok(Self::Float32Vec4),
            3 => Ok(Self::Float32),
            4 => Ok(Self::Float32Vec3),
            _ => Err(format!("unknown pika_format {value}")),
        }
    }

    fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Float16Vec4 => 8,
            Self::Float32Vec4 => 16,
            Self::Float32 => 4,
            Self::Float32Vec3 => 12,
        }
    }
}

/// A render buffer Hydra bound to one AOV: `width` × `height` pixels of
/// `format`, bottom row first.
pub struct Aov<'a> {
    pub format: Format,
    pub pixels: &'a mut [u8],
}

/// One frame of a render pass, as Hydra's render pass state gives it.
pub struct Frame<'a> {
    pub world_to_view: [f64; 16],
    /// Conformed to the framing: it maps the data window to clip space.
    pub projection: [f64; 16],
    /// Every bound render buffer's size.
    pub width: u32,
    pub height: u32,
    /// The data window, the pixels Pika writes: x, y, width, height, with y
    /// counted down from the top row, as CameraUtilFraming counts it.
    pub window: [u32; 4],
    pub clear_color: [f32; 4],
    pub color: Option<Aov<'a>>,
    pub depth: Option<Aov<'a>>,
    pub normal: Option<Aov<'a>>,
    /// The render tags the pass draws; none means every tag.
    pub render_tags: Vec<String>,
}

/// The depth AOV's clear value: OpenGL's far plane.
const FAR: f32 = 1.0;

// One instance per process: plugins are never unloaded, and each render
// delegate's device borrows it.
static INSTANCE: OnceLock<Result<gpu::Instance, String>> = OnceLock::new();

pub struct Renderer {
    pub store: Store,
    // Field order is drop order: the target waits for idle, then the
    // pipelines and buffers are destroyed before their device.
    target: Option<Target>,
    raster: Raster,
    smooth: Smooth,
    geometry: Geometry,
    device: gpu::Device<'static>,
    pub device_name: String,
    /// `PIKA_STORE_DUMP`: the file each sync's store is appended to, for the
    /// fixture checks. Writing it changes nothing about rendering.
    dump: Option<(String, File)>,
    /// `PIKA_STATS`: the file each sync's stats and census are appended to,
    /// one line each, for hydra/census.py.
    stats: Option<(String, File)>,
    sync_start: Option<Instant>,
    /// When the last sync began, so each sync's line can say how long the
    /// host took from one sync to the next.
    last_start: Option<Instant>,
    syncs: Vec<Sync>,
    frames: Vec<FrameTimes>,
    /// The size and color format of the last frame, for the report.
    last: Option<(u32, u32, Format)>,
}

/// One Hydra sync, from the render delegate's `Update` to its
/// `CommitResources`: the prim syncs, including Hydra's own pulls.
struct Sync {
    time: Duration,
    stats: store::Stats,
}

/// Where one frame's time went: the GPU uploads, draws and copy to host
/// memory, then the CPU copy into the host's render buffers.
struct FrameTimes {
    submit: Duration,
    copy: Duration,
}

impl Renderer {
    pub fn new() -> Result<Self, String> {
        let instance = INSTANCE
            .get_or_init(gpu::Instance::new)
            .as_ref()
            .map_err(Clone::clone)?;
        let index = gpu::choose(&instance.devices).ok_or(
            "no suitable Vulkan device: Pika needs a GPU with Vulkan 1.3 and a graphics+compute \
             queue. Check the driver with `vulkaninfo --summary`.",
        )?;
        let validation = match instance.validation_messages() {
            Some(_) => "on (sync)",
            None => "not found",
        };
        let dump = output("PIKA_STORE_DUMP")?;
        let stats = output("PIKA_STATS")?;
        let device = instance.open(index)?;
        let mesh = shader::compile("mesh")?;
        let smooth = shader::compile("smooth")?;
        Ok(Self {
            store: Store::new(),
            target: None,
            raster: Raster::new(&device, &mesh.spirv)?,
            smooth: Smooth::new(&device, &smooth.spirv)?,
            geometry: Geometry::new(),
            device,
            device_name: format!("{}, validation {validation}", instance.devices[index].name),
            dump,
            stats,
            sync_start: None,
            last_start: None,
            syncs: Vec::new(),
            frames: Vec::new(),
            last: None,
        })
    }

    pub fn begin_sync(&mut self) {
        self.sync_start = Some(Instant::now());
    }

    /// Records the sync's time and stats, and appends them and the store to
    /// their files.
    pub fn end_sync(&mut self) -> Result<(), String> {
        let start = self
            .sync_start
            .take()
            .ok_or("a sync ended that never began: the delegate must call pika_sync_begin")?;
        let sync = Sync {
            time: start.elapsed(),
            stats: self.store.end_sync(),
        };
        let period = self.last_start.replace(start).map(|last| start - last);
        if let Some((path, file)) = &mut self.stats {
            let line = stats_line(self.syncs.len() + 1, &sync, period, &self.store.census()?);
            file.write_all(line.as_bytes())
                .map_err(|e| format!("cannot write PIKA_STATS file {path}: {e}"))?;
        }
        self.syncs.push(sync);
        if let Some((path, file)) = &mut self.dump {
            let text = format!("sync {}\n{}", self.syncs.len(), self.store.dump());
            file.write_all(text.as_bytes())
                .map_err(|e| format!("cannot write PIKA_STORE_DUMP file {path}: {e}"))?;
        }
        Ok(())
    }

    /// Draws the store's meshes into the frame's data window, and copies
    /// each bound AOV into its render buffer.
    pub fn render(&mut self, frame: Frame) -> Result<(), String> {
        let [x, y, w, h] = frame.window;
        if w == 0 || h == 0 || x + w > frame.width || y + h > frame.height {
            return Err(format!(
                "the data window ({x}, {y}) {w}×{h} is not inside the {}×{} render buffers",
                frame.width, frame.height
            ));
        }
        let expect = |aov: &Option<Aov>, name, formats: &[Format]| match aov {
            Some(a) if !formats.contains(&a.format) => Err(format!(
                "the {name} AOV is {:?}; Pika writes it as {formats:?}",
                a.format
            )),
            Some(a)
                if a.pixels.len()
                    != frame.width as usize
                        * frame.height as usize
                        * a.format.bytes_per_pixel() =>
            {
                Err(format!(
                    "the {name} render buffer is {} bytes, not {}×{} {:?}",
                    a.pixels.len(),
                    frame.width,
                    frame.height,
                    a.format
                ))
            }
            _ => Ok(()),
        };
        expect(
            &frame.color,
            "color",
            &[Format::Float16Vec4, Format::Float32Vec4],
        )?;
        expect(&frame.depth, "depth", &[Format::Float32])?;
        expect(&frame.normal, "normal", &[Format::Float32Vec3])?;
        let color = frame
            .color
            .as_ref()
            .map_or(Format::Float16Vec4, |a| a.format);
        let formats = [
            match color {
                Format::Float32Vec4 => (raster::COLOR_FORMATS[1], 16),
                _ => (raster::COLOR_FORMATS[0], 8),
            },
            raster::DEPTH_AOV,
            raster::NORMAL_AOV,
        ];
        let camera = Camera::new(&frame.world_to_view, &frame.projection)?;

        let target = match &mut self.target {
            Some(t) if t.fits(&formats, w, h) => t,
            slot => {
                // Free the old target before allocating its replacement.
                *slot = None;
                slot.insert(Target::new(&self.device, &formats, w, h)?)
            }
        };
        let moved = self.store.take_moved();
        let cb = target.begin()?;
        self.geometry.update(
            &self.device,
            &self.store,
            &moved,
            &frame.render_tags,
            &self.smooth,
            cb,
        )?;
        let clears = [FAR, 0.0].map(|v| vk::ClearColorValue { float32: [v; 4] });
        target.begin_rendering(&[
            vk::ClearColorValue {
                float32: frame.clear_color,
            },
            clears[0],
            clears[1],
        ]);
        if let Some(geometry) = self.geometry.bound() {
            self.raster
                .draw(cb, formats[0].0, target.area(), camera, &geometry);
        }
        let read = [&frame.color, &frame.depth, &frame.normal].map(Option::is_some);
        let submit = target.finish(&read)?;

        let start = Instant::now();
        for (i, aov) in [frame.color, frame.depth, frame.normal]
            .into_iter()
            .enumerate()
        {
            if let Some(aov) = aov {
                place(
                    target.pixels(i),
                    formats[i].1,
                    aov,
                    frame.width,
                    frame.height,
                    frame.window,
                );
            }
        }
        self.frames.push(FrameTimes {
            submit,
            copy: start.elapsed(),
        });
        self.last = Some((w, h, color));
        Ok(())
    }

    /// Sync and frame timings so far, as median and maximum per stage, with
    /// the bytes Hydra handed over.
    pub fn report(&self) -> String {
        let syncs = if self.syncs.is_empty() {
            "no syncs".into()
        } else {
            let stored = |s: &Sync| [s.stats.mesh, s.stats.instancer];
            let bytes: usize = self.syncs.iter().flat_map(stored).map(|b| b.pulled).sum();
            let redundant: usize = self
                .syncs
                .iter()
                .flat_map(stored)
                .map(|b| b.redundant)
                .sum();
            let curves: usize = self.syncs.iter().map(|s| s.stats.curve_bytes).sum();
            format!(
                "{} syncs: {}; {bytes} bytes pulled, {redundant} redundant; \
                 {curves} bytes of basis curves pulled and dropped",
                self.syncs.len(),
                percentiles(self.syncs.iter().map(|s| s.time)),
            )
        };
        let Some((width, height, color)) = self.last else {
            return format!("{syncs}; no frames rendered");
        };
        format!(
            "{syncs}; {} frames, last at {width}×{height} with {color:?} color: \
             upload, draw and readback {}; copy into the render buffers {}",
            self.frames.len(),
            percentiles(self.frames.iter().map(|f| f.submit)),
            percentiles(self.frames.iter().map(|f| f.copy)),
        )
    }
}

/// Copies a target attachment's rows, read back bottom row first with
/// `stride` bytes per pixel, into an AOV's render buffer of `width` ×
/// `height`, at the data window. Each pixel keeps the buffer's bytes per
/// pixel of its value, dropping the attachment's extra channel.
fn place(src: &[u8], stride: usize, aov: Aov, width: u32, height: u32, window: [u32; 4]) {
    let [x, y, w, h] = window.map(|v| v as usize);
    let size = aov.format.bytes_per_pixel();
    // The window's bottom row, counted up from the buffer's bottom row.
    let bottom = height as usize - y - h;
    for row in 0..h {
        let src = &src[row * w * stride..(row + 1) * w * stride];
        let start = ((bottom + row) * width as usize + x) * size;
        let dst = &mut aov.pixels[start..start + w * size];
        if stride == size {
            dst.copy_from_slice(src);
            continue;
        }
        for (d, s) in dst.chunks_exact_mut(size).zip(src.chunks_exact(stride)) {
            d.copy_from_slice(&s[..size]);
        }
    }
}

/// The file an environment variable names, created empty; `None` when the
/// variable is unset.
fn output(variable: &str) -> Result<Option<(String, File)>, String> {
    let Ok(path) = std::env::var(variable) else {
        return Ok(None);
    };
    let file =
        File::create(&path).map_err(|e| format!("cannot create {variable} file {path}: {e}"))?;
    Ok(Some((path, file)))
}

/// One sync as `key value` pairs. The period, from the last sync's start to
/// this one's, is left out of the first sync.
fn stats_line(
    index: usize,
    sync: &Sync,
    period: Option<Duration>,
    census: &store::Census,
) -> String {
    let ms = |t: Duration| t.as_secs_f64() * 1e3;
    let s = &sync.stats;
    let mut line = format!("sync {index} time_ms {:.3}", ms(sync.time));
    if let Some(period) = period {
        line += &format!(" period_ms {:.3}", ms(period));
    }
    line += &format!(
        " inserted {} removed {} mesh_bytes {} mesh_redundant_bytes {} instancer_bytes {} \
         instancer_redundant_bytes {} curve_bytes {} meshes {} mesh_instances {} triangles {} \
         prototype_triangles {} geometry_bytes {} materials {} instancers {} curves {}",
        s.inserted,
        s.removed,
        s.mesh.pulled,
        s.mesh.redundant,
        s.instancer.pulled,
        s.instancer.redundant,
        s.curve_bytes,
        census.meshes,
        census.mesh_instances,
        census.triangles,
        census.prototype_triangles,
        census.geometry_bytes,
        census.materials,
        census.instancers,
        census.curves,
    );
    for (prim_type, count) in &census.counted {
        line += &format!(" counted_{prim_type} {count}");
    }
    line + "\n"
}

/// The median and maximum of a non-empty set of times.
fn percentiles(times: impl Iterator<Item = Duration>) -> String {
    let mut ms: Vec<f64> = times.map(|t| t.as_secs_f64() * 1e3).collect();
    ms.sort_by(f64::total_cmp);
    format!(
        "p50 {:.2} ms, max {:.2} ms",
        ms[ms.len() / 2],
        ms[ms.len() - 1]
    )
}
