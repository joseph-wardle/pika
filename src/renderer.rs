//! What one Hydra render delegate owns on the Rust side.

use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ash::vk;

use crate::commands::Commands;
use crate::cull::Cull;
use crate::geometry::{self, Geometry};
use crate::gpu;
use crate::light::{self, Tiles};
use crate::link::Links;
use crate::raster::{self, Raster};
use crate::shade::Shade;
use crate::shader;
use crate::shadow::{Caster, Shadows};
use crate::stats::{self, Stats};
use crate::store::{Moved, Store};
use crate::target::Target;
use crate::texture::Needs;
use crate::transform::Camera;
use crate::view::View;

/// Color is half or full float, as the host asks; the delegate's default, half, halves readback.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Float16Vec4,
    Float32Vec4,
    Float32,
    Float32Vec3,
}

impl Format {
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "HdFormatFloat16Vec4" => Ok(Self::Float16Vec4),
            "HdFormatFloat32Vec4" => Ok(Self::Float32Vec4),
            "HdFormatFloat32" => Ok(Self::Float32),
            "HdFormatFloat32Vec3" => Ok(Self::Float32Vec3),
            _ => Err(format!("{name} is not a render buffer format Pika writes")),
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

/// A render buffer Hydra bound to one AOV, bottom row first.
pub struct Aov<'a> {
    pub format: Format,
    pub pixels: &'a mut [u8],
}

pub struct Frame<'a> {
    pub world_to_view: [f64; 16],
    /// Conformed to the framing: it maps the data window to clip space.
    pub projection: [f64; 16],
    pub width: u32,
    pub height: u32,
    /// x, y, width, height, with y counted down from the top row, as CameraUtilFraming counts it.
    pub window: [u32; 4],
    pub clear_color: [f32; 4],
    pub color: Option<Aov<'a>>,
    pub depth: Option<Aov<'a>>,
    pub normal: Option<Aov<'a>>,
    /// The render tags the pass draws; none means every tag.
    pub render_tags: Vec<String>,
    /// `pika:textureCap`, 0 for none.
    pub texture_cap: u32,
    /// `pika:shadowPool`, in MB.
    pub shadow_pool: u32,
}

/// OpenGL's far plane.
const FAR: f32 = 1.0;

/// How long a call with nothing else to draw waits for a material build; keeps usdview responsive.
const WAIT: Duration = Duration::from_millis(16);

// One instance per process: plugins are never unloaded, and each delegate's device borrows it.
static INSTANCE: OnceLock<Result<gpu::Instance, String>> = OnceLock::new();

pub struct Renderer {
    pub store: Store,
    pub stats: Stats,
    // Field order is drop order: the commands and target wait for idle before the device goes.
    commands: Commands,
    target: Option<Target>,
    view: Option<View>,
    raster: Raster,
    shade: Shade,
    shadows: Shadows,
    cull: Cull,
    geometry: Geometry,
    device: gpu::Device<'static>,
    pub device_name: String,
    /// `PIKA_STORE_DUMP`, for the fixture checks; writing it changes nothing about rendering.
    dump: Option<(String, File)>,
    syncs: usize,
    pending: Option<Pending>,
    placed: Option<(Buffers, Inputs)>,
}

/// What a frame's image depends on besides the store and its buffers.
#[derive(Clone, PartialEq)]
struct Inputs {
    world_to_view: [f64; 16],
    projection: [f64; 16],
    clear_color: [f32; 4],
    render_tags: Vec<String>,
    texture_cap: u32,
    shadow_pool: u32,
}

/// The host's buffers are taken to hold the last frame placed into ones that match.
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

struct Pending {
    buffers: Buffers,
    inputs: Inputs,
    formats: [(vk::Format, usize); 3],
    read: [bool; 3],
    period: Option<Duration>,
    record: Duration,
    update: geometry::Update,
    /// Whether the view was culled, so its counts are this frame's.
    culled: bool,
    lights: usize,
    moved: usize,
    transforms: usize,
    latent: bool,
    compiled: usize,
    baked: usize,
    loaded: u64,
    texture_read: Duration,
    pending: usize,
    pending_textures: usize,
    lowered_textures: usize,
    need: Duration,
    needs: Needs,
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
        let tangent = shader::compile("tangent")?;
        let shadow = shader::compile("shadow")?;
        Ok(Self {
            store: Store::new(),
            stats,
            commands: Commands::new(&device)?,
            target: None,
            view: None,
            raster: Raster::new(&device, &mesh.spirv)?,
            shade: Shade::new(&device)?,
            shadows: Shadows::new(&device, &shadow.spirv)?,
            cull: Cull::new(&device, &cull.spirv)?,
            geometry: Geometry::new(&device, &subdiv.spirv, &smooth.spirv, &tangent.spirv)?,
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

    /// Returns whether the buffers now hold this frame, with no material build still out.
    /// While frames change, each call returns the frame before and leaves this one on the GPU,
    /// so GPU work overlaps the host's own frame time instead of running in series with it.
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
            texture_cap: frame.texture_cap,
            shadow_pool: frame.shadow_pool,
        };
        let mut aovs = [frame.color, frame.depth, frame.normal];

        let moved = self.store.take_moved();
        if let Some(pending) = self.pending.take() {
            self.collect(pending, &buffers, &mut aovs)?;
        }
        let placed = self.placed.as_ref();
        // A swap waiting for a frame's bytes needs no build to land.
        if moved == Moved::default()
            && placed == Some(&(buffers, inputs.clone()))
            && self.shade.settled()
            && !self.shade.builder.wait(WAIT)?
        {
            return Ok(self.shade.settled());
        }
        let latent = placed.is_some_and(|(b, _)| *b == buffers);

        let start = Instant::now();
        self.shadows.reserve(&self.device, inputs.shadow_pool)?;
        let target = match &mut self.target {
            Some(t) if t.fits(&formats, w, h) => t,
            slot => {
                // Free the old target before allocating its replacement.
                *slot = None;
                slot.insert(Target::new(&self.device, &formats, w, h)?)
            }
        };
        let cb = self.commands.begin()?;
        let update = (self.geometry).update(
            &self.device,
            &self.store,
            &moved,
            &inputs.render_tags,
            &mut self.commands,
        )?;
        let estimating = Instant::now();
        let footprints = (self.geometry).footprints(&camera.view_projection, [w, h]);
        let need = estimating.elapsed();
        let shading = (self.shade).update(
            &self.device,
            &self.store,
            &moved,
            inputs.texture_cap,
            &footprints,
            cb,
        )?;
        self.commands.mark("texture")?;
        let bound = self.geometry.bound();
        // A headlight is the host's to send, so a store holding no light draws black.
        let links = Links::new(&self.store);
        let mut lights = Vec::new();
        let mut casters = Vec::new();
        for l in self.store.lights().values() {
            let Some(resolved) = light::resolve(l) else {
                continue;
            };
            let link = links.light(l);
            if l.params.is_some_and(|p| p.shadow) {
                let shadow = link.shadow;
                casters.push(Caster {
                    light: lights.len(),
                    shadow,
                });
            }
            lights.push(light::Light {
                link: link.light,
                ..resolved
            });
        }
        if let Some(g) = &bound {
            let (device, geometry) = (&self.device, &self.geometry);
            let projection = &inputs.projection;
            (self.shadows).fit(
                device,
                geometry,
                g,
                &camera,
                projection,
                [w, h],
                &mut lights,
                &casters,
            )?;
        }
        let tiles = Tiles::new(&lights, &camera, [w, h]);
        let capacity = light::capacity(lights.len(), [w, h]);
        let mut view = match (&bound, &mut self.view) {
            (None, _) => None,
            (Some(g), Some(v)) if v.fits(g, capacity) => Some(v),
            (Some(g), slot) => {
                *slot = None;
                Some(slot.insert(View::new(&self.device, g, capacity)?))
            }
        };
        if let (Some(geometry), Some(view)) = (&bound, view.as_deref_mut()) {
            let shadows = self.shadows.table();
            view.begin(cb, &camera, geometry, &lights, &tiles, [w, h], shadows);
            self.cull.record(cb, geometry, view);
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
        let drawn = bound.as_ref().zip(view.as_deref());
        if let Some((geometry, view)) = drawn {
            (self.raster).draw(cb, formats[0].0, target.area(), geometry, view);
        }
        target.end_rendering(cb);
        self.commands.mark("draw")?;
        // Between the prepass and shading, which reads them: pages follow the depth drawn.
        if let Some((geometry, view)) = drawn {
            self.shadows.record(cb, geometry, view, target);
        }
        self.commands.mark("shadow")?;
        target.resume_rendering(cb);
        if let Some((geometry, view)) = drawn {
            let groups = self.geometry.groups();
            (self.shade).draw(cb, formats[0].0, target.area(), geometry, groups, view);
        }
        target.end_rendering(cb);
        self.commands.mark("shade")?;
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
            lights: lights.len(),
            moved: moved.points.len(),
            transforms: moved.transforms.len(),
            latent,
            compiled: shading.compiled,
            baked: shading.baked,
            loaded: shading.loaded,
            texture_read: shading.read,
            pending: shading.pending,
            pending_textures: shading.pending_textures,
            lowered_textures: shading.lowered_textures,
            need,
            needs: shading.needs,
        };
        if latent {
            self.pending = Some(pending);
            return Ok(false);
        }
        self.collect(pending, &buffers, &mut aovs)?;
        Ok(self.shade.settled())
    }

    /// Places the pending frame only if `aovs` are the buffers it was drawn for.
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
        let shadow = self.shadows.counts();
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
        let local = self
            .device
            .heap(!0, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let vram = self.device.heaps()[local];
        let (vram_used, vram_budget) = (vram.used, vram.allowed);
        let memory = stats::Memory {
            geometry: self.geometry.memory(),
            views: self.view.as_ref().map(View::memory).unwrap_or_default(),
            targets: target.memory(),
            materials: self.shade.memory(),
            shadows: self.shadows.memory(),
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
            lights: pending.lights,
            shadow,
            memory,
            moved: pending.moved,
            transforms: pending.transforms,
            latent: pending.latent,
            compiled: pending.compiled,
            baked: pending.baked,
            loaded: pending.loaded,
            texture_read: pending.texture_read,
            pending: pending.pending,
            pending_textures: pending.pending_textures,
            lowered_textures: pending.lowered_textures,
            need: pending.need,
            needs: pending.needs,
        })?;
        self.placed = fits.then_some((pending.buffers, pending.inputs));
        Ok(())
    }
}

/// Copies attachment rows, read back bottom row first, into the AOV's data window, dropping
/// the attachment's extra channel.
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

fn output(variable: &str) -> Result<Option<(String, File)>, String> {
    let Ok(path) = std::env::var(variable) else {
        return Ok(None);
    };
    let file =
        File::create(&path).map_err(|e| format!("cannot create {variable} file {path}: {e}"))?;
    Ok(Some((path, file)))
}
