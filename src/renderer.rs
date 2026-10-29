//! What one Hydra render delegate owns on the Rust side: its scene store,
//! the GPU copy of its meshes, its Vulkan device, the target it reads
//! frames back from and the stats it measures.

use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ash::vk;

use crate::commands::Commands;
use crate::cull::{self, Cull};
use crate::geometry::{self, Geometry};
use crate::gpu;
use crate::raster::{self, Raster};
use crate::shader;
use crate::smooth::Smooth;
use crate::stats::{self, Stats};
use crate::store::{Moved, Store};
use crate::subdiv::Subdiv;
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
    pub stats: Stats,
    // Field order is drop order: the commands and target wait for idle, then
    // the pipelines and buffers are destroyed before their device.
    commands: Commands,
    target: Option<Target>,
    /// The camera's records and visible instances.
    view: Option<cull::View>,
    raster: Raster,
    cull: Cull,
    subdiv: Subdiv,
    smooth: Smooth,
    geometry: Geometry,
    device: gpu::Device<'static>,
    pub device_name: String,
    /// `PIKA_STORE_DUMP`: the file each sync's store is appended to, for the
    /// fixture checks. Writing it changes nothing about rendering.
    dump: Option<(String, File)>,
    syncs: usize,
    /// The frame on the GPU, waiting to be placed.
    pending: Option<Pending>,
    /// The buffers the last frame was placed into, and what it was drawn from.
    placed: Option<(Buffers, Inputs)>,
}

/// What a frame's image depends on besides the store and its buffers.
#[derive(Clone, PartialEq)]
struct Inputs {
    world_to_view: [f64; 16],
    projection: [f64; 16],
    clear_color: [f32; 4],
    render_tags: Vec<String>,
}

/// The render buffers a frame is placed into, known only by size, data
/// window and, for each AOV, format, address and length. The host's
/// buffers are taken to hold the last frame placed into ones that match.
#[derive(Clone, Copy, PartialEq)]
struct Buffers {
    width: u32,
    height: u32,
    window: [u32; 4],
    aovs: [Option<(Format, usize, usize)>; 3],
}

impl Buffers {
    fn of(frame: &Frame) -> Self {
        let aov = |a: &Option<Aov>| {
            (a.as_ref()).map(|a| (a.format, a.pixels.as_ptr() as usize, a.pixels.len()))
        };
        Self {
            width: frame.width,
            height: frame.height,
            window: frame.window,
            aovs: [aov(&frame.color), aov(&frame.depth), aov(&frame.normal)],
        }
    }
}

/// A frame submitted to the GPU, with what placing it and its stats need.
struct Pending {
    buffers: Buffers,
    inputs: Inputs,
    /// Each attachment's format and bytes per pixel, and whether it was read.
    formats: [(vk::Format, usize); 3],
    read: [bool; 3],
    period: Option<Duration>,
    record: Duration,
    update: geometry::Update,
    /// Whether the camera's view was culled, so its counts are this frame's.
    culled: bool,
    moved: usize,
    transforms: usize,
    latent: bool,
}

impl Renderer {
    pub fn new() -> Result<Self, String> {
        let instance = INSTANCE
            .get_or_init(gpu::Instance::new)
            .as_ref()
            .map_err(Clone::clone)?;
        let index = gpu::choose(&instance.devices).ok_or(
            "no suitable Vulkan device: Pika needs a GPU with Vulkan 1.3 and a graphics+compute \
             queue with timestamps. Check the driver with `vulkaninfo --summary`.",
        )?;
        let validation = match instance.validation_messages() {
            Some(_) => "on (sync)",
            None => "not found",
        };
        let device_name = format!("{}, validation {validation}", instance.devices[index].name);
        let dump = output("PIKA_STORE_DUMP")?;
        let stats = Stats::new(&device_name)?;
        let device = instance.open(index)?;
        let mesh = shader::compile("mesh")?;
        let cull = shader::compile("cull")?;
        let subdiv = shader::compile("subdiv")?;
        let smooth = shader::compile("smooth")?;
        Ok(Self {
            store: Store::new(),
            stats,
            commands: Commands::new(&device)?,
            target: None,
            view: None,
            raster: Raster::new(&device, &mesh.spirv)?,
            cull: Cull::new(&device, &cull.spirv)?,
            subdiv: Subdiv::new(&device, &subdiv.spirv)?,
            smooth: Smooth::new(&device, &smooth.spirv)?,
            geometry: Geometry::new(),
            device,
            device_name,
            dump,
            syncs: 0,
            pending: None,
            placed: None,
        })
    }

    pub fn begin_sync(&mut self) {
        self.stats.begin_sync();
    }

    /// Records the sync's time and stats, and appends the store to its file.
    pub fn end_sync(&mut self) -> Result<(), String> {
        let pulled = self.store.end_sync();
        let census = self.store.census()?;
        self.stats.end_sync(pulled, census)?;
        self.syncs += 1;
        if let Some((path, file)) = &mut self.dump {
            let text = format!("sync {}\n{}", self.syncs, self.store.dump());
            file.write_all(text.as_bytes())
                .map_err(|e| format!("cannot write PIKA_STORE_DUMP file {path}: {e}"))?;
        }
        Ok(())
    }

    /// Draws the store's meshes into the frame's data window, and copies
    /// each bound AOV into its render buffer. Returns whether the buffers
    /// now hold this frame.
    ///
    /// While frames change, each call hands back the frame before and leaves
    /// this one on the GPU, so its work overlaps the host's until the next
    /// call: drawing and waiting in one call would put every GPU pass in
    /// series with the host's own frame time. A call that changes nothing
    /// waits for the frame on the GPU and returns true. A frame drawn into
    /// buffers that do not hold the last frame is drawn and waited for at
    /// once.
    pub fn render(&mut self, frame: Frame) -> Result<bool, String> {
        let period = self.stats.begin_render();
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
        let buffers = Buffers::of(&frame);
        let inputs = Inputs {
            world_to_view: frame.world_to_view,
            projection: frame.projection,
            clear_color: frame.clear_color,
            render_tags: frame.render_tags.clone(),
        };
        let mut aovs = [frame.color, frame.depth, frame.normal];

        let moved = self.store.take_moved();
        if let Some(pending) = self.pending.take() {
            self.collect(pending, &buffers, &mut aovs)?;
        }
        let placed = self.placed.as_ref();
        if moved == Moved::default() && placed == Some(&(buffers, inputs.clone())) {
            return Ok(true);
        }
        let latent = placed.is_some_and(|(b, _)| *b == buffers);

        let start = Instant::now();
        let target = match &mut self.target {
            Some(t) if t.fits(&formats, w, h) => t,
            slot => {
                // Free the old target before allocating its replacement.
                *slot = None;
                slot.insert(Target::new(&self.device, &formats, w, h)?)
            }
        };
        let cb = self.commands.begin()?;
        let update = self.geometry.update(
            &self.device,
            &self.store,
            &moved,
            &inputs.render_tags,
            &self.subdiv,
            &self.smooth,
            cb,
        )?;
        self.commands.mark("update")?;
        let bound = self.geometry.bound();
        let mut view = match (&bound, &mut self.view) {
            (None, _) => None,
            (Some(g), Some(v)) if v.fits(g) => Some(v),
            (Some(g), slot) => {
                *slot = None;
                Some(slot.insert(cull::View::new(&self.device, g)?))
            }
        };
        if let (Some(geometry), Some(view)) = (&bound, view.as_deref_mut()) {
            self.cull.record(cb, &camera, geometry, view);
        }
        self.commands.mark("cull")?;
        let clears = [FAR, 0.0].map(|v| vk::ClearColorValue { float32: [v; 4] });
        target.begin_rendering(
            cb,
            &[
                vk::ClearColorValue {
                    float32: inputs.clear_color,
                },
                clears[0],
                clears[1],
            ],
        );
        if let (Some(geometry), Some(view)) = (&bound, view.as_deref()) {
            (self.raster).draw(cb, formats[0].0, target.area(), camera, geometry, view);
        }
        target.end_rendering(cb);
        self.commands.mark("draw")?;
        let read = aovs.each_ref().map(Option::is_some);
        target.readback(cb, &read);
        self.commands.mark("readback")?;
        self.commands.submit()?;
        let pending = Pending {
            buffers,
            inputs,
            formats,
            read,
            period,
            record: start.elapsed(),
            update,
            culled: view.is_some(),
            moved: moved.points.len(),
            transforms: moved.transforms.len(),
            latent,
        };
        if latent {
            self.pending = Some(pending);
            return Ok(false);
        }
        self.collect(pending, &buffers, &mut aovs)?;
        Ok(true)
    }

    /// Waits for the pending frame and copies it into `aovs`, if they are
    /// the buffers it was drawn for, then records its stats.
    fn collect(
        &mut self,
        pending: Pending,
        buffers: &Buffers,
        aovs: &mut [Option<Aov>; 3],
    ) -> Result<(), String> {
        let target = (self.target.as_mut()).ok_or("a frame is pending without its target")?;
        let times = self.commands.wait()?;
        let drawn = match (&mut self.view, pending.culled) {
            (Some(view), true) => view.drawn(),
            _ => [0; 2],
        };
        let placing = Instant::now();
        let [x, y, w, h] = pending.buffers.window;
        let fits = pending.buffers == *buffers;
        if fits {
            for (i, aov) in aovs.iter_mut().enumerate() {
                if let Some(aov) = aov {
                    let (width, height) = (buffers.width, buffers.height);
                    let window = [x, y, w, h];
                    place(
                        target.pixels(i),
                        pending.formats[i].1,
                        aov,
                        width,
                        height,
                        window,
                    );
                }
            }
        }
        let place = placing.elapsed();
        let read_bytes = (pending.formats.iter().zip(pending.read))
            .filter(|(_, r)| *r)
            .map(|(f, _)| (f.1 * w as usize * h as usize) as u64)
            .sum();
        let (vram_used, vram_budget) = self.device.budget();
        let memory = stats::Memory {
            geometry: self.geometry.memory(),
            views: self
                .view
                .as_ref()
                .map(cull::View::memory)
                .unwrap_or_default(),
            targets: target.memory(),
            vram_used,
            vram_budget,
        };
        self.stats.frame(stats::Frame {
            width: w,
            height: h,
            period: pending.period,
            record: pending.record,
            times,
            place,
            update: pending.update,
            read_bytes,
            drawn,
            memory,
            moved: pending.moved,
            transforms: pending.transforms,
            latent: pending.latent,
        })?;
        self.placed = fits.then_some((pending.buffers, pending.inputs));
        Ok(())
    }
}

/// Copies a target attachment's rows, read back bottom row first with
/// `stride` bytes per pixel, into an AOV's render buffer of `width` ×
/// `height`, at the data window. Each pixel keeps the buffer's bytes per
/// pixel of its value, dropping the attachment's extra channel.
fn place(src: &[u8], stride: usize, aov: &mut Aov, width: u32, height: u32, window: [u32; 4]) {
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
