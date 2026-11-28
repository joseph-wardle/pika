//! Everything one Hydra render delegate owns on the Rust side.

use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;
use std::time::Instant;

use ash::vk;

use crate::bake::Bake;
use crate::commands::Commands;
use crate::cull::Cull;
use crate::dome::Dome;
use crate::film::Film;
use crate::geometry::Geometry;
use crate::gpu;
use crate::light::{self, Tiles};
use crate::raster::{self, Raster};
use crate::shade::Shade;
use crate::shader;
use crate::shadow::Shadows;
use crate::stats::{self, Stats};
use crate::store::{Moved, Store};
use crate::target::Target;
use crate::transform::Camera;
use crate::view::View;

/// Colour is half or full float, as the host asks. The delegate defaults to half, which halves
/// the readback.
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

/// A render buffer Hydra bound to one AOV, stored bottom row first.
pub struct Aov<'a> {
    pub format: Format,
    pub pixels: &'a mut [u8],
}

pub struct Frame<'a> {
    pub world_to_view: [f64; 16],
    /// Conformed to the framing, so it maps the data window to clip space.
    pub projection: [f64; 16],
    pub width: u32,
    pub height: u32,
    /// x, y, width and height, with y counted down from the top row as CameraUtilFraming does.
    pub window: [u32; 4],
    pub clear_color: [f32; 4],
    pub color: Option<Aov<'a>>,
    pub depth: Option<Aov<'a>>,
    pub normal: Option<Aov<'a>>,
    /// The render tags the pass draws. An empty list means every tag.
    pub render_tags: Vec<String>,
    /// `pika:textureCap`, or 0 for no cap.
    pub texture_cap: u32,
    /// `pika:shadowPool`, in MB.
    pub shadow_pool: u32,
    /// `pika:sunThreshold`, or 0 for no sun.
    pub sun_threshold: f32,
    /// Hydra's `domeLightCameraVisibility`: whether the lit dome's sky is drawn behind the scene.
    pub dome_light_camera_visibility: bool,
}

impl Frame<'_> {
    fn check(&self) -> Result<(), String> {
        let [x, y, w, h] = self.window;
        if w == 0 || h == 0 || x + w > self.width || y + h > self.height {
            return Err(format!(
                "the data window ({x}, {y}) {w}×{h} is not inside the {}×{} render buffers",
                self.width, self.height
            ));
        }
        let pixels = self.width as usize * self.height as usize;
        let expect = |aov: &Option<Aov>, name, formats: &[Format]| match aov {
            Some(a) if !formats.contains(&a.format) => Err(format!(
                "the {name} AOV is {:?}; Pika writes it as {formats:?}",
                a.format
            )),
            Some(a) if a.pixels.len() != pixels * a.format.bytes_per_pixel() => Err(format!(
                "the {name} render buffer is {} bytes, not {}×{} {:?}",
                a.pixels.len(),
                self.width,
                self.height,
                a.format
            )),
            _ => Ok(()),
        };
        expect(
            &self.color,
            "color",
            &[Format::Float16Vec4, Format::Float32Vec4],
        )?;
        expect(&self.depth, "depth", &[Format::Float32])?;
        expect(&self.normal, "normal", &[Format::Float32Vec3])
    }

    /// Each attachment's format and bytes per pixel, in the AOVs' order.
    fn formats(&self) -> [(vk::Format, usize); 3] {
        let color = match self.color.as_ref().map(|a| a.format) {
            Some(Format::Float32Vec4) => (raster::COLOR_FORMATS[1], 16),
            _ => (raster::COLOR_FORMATS[0], 8),
        };
        [color, raster::DEPTH_AOV, raster::NORMAL_AOV]
    }
}

/// The colour AOV's place among a frame's three AOVs: colour, depth and normal.
const COLOR: usize = 0;

/// OpenGL's far plane.
const FAR: f32 = 1.0;

// One instance per process, because plugins are never unloaded and each delegate's device
// borrows it.
static INSTANCE: OnceLock<Result<gpu::Instance, String>> = OnceLock::new();

pub struct Renderer {
    pub store: Store,
    pub stats: Stats,
    // Fields drop in order, so the commands and target wait for idle before the device goes.
    commands: Commands,
    target: Option<Target>,
    view: Option<View>,
    raster: Raster,
    shade: Shade,
    shadows: Shadows,
    cull: Cull,
    dome: Dome,
    film: Film,
    geometry: Geometry,
    device: gpu::Device<'static>,
    pub device_name: String,
    /// `PIKA_STORE_DUMP`, for the fixture checks. Writing it changes nothing about rendering.
    dump: Option<(String, File)>,
    syncs: usize,
    /// The frame the last call submitted, when it left placing it to this call.
    pending: Option<Pending>,
    /// The buffers the last frame was placed in and what it was drawn from. A call with both the
    /// same draws the next still sample. A call with the same buffers may place its frame late.
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
    sun_threshold: f32,
    dome_light_camera_visibility: bool,
}

impl Inputs {
    fn of(frame: &Frame) -> Self {
        Self {
            world_to_view: frame.world_to_view,
            projection: frame.projection,
            clear_color: frame.clear_color,
            render_tags: frame.render_tags.clone(),
            texture_cap: frame.texture_cap,
            shadow_pool: frame.shadow_pool,
            sun_threshold: frame.sun_threshold,
            dome_light_camera_visibility: frame.dome_light_camera_visibility,
        }
    }
}

/// The host's buffers are assumed to hold the last frame placed into matching ones.
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

/// A frame that was submitted and not yet placed in the host's buffers.
struct Pending {
    buffers: Buffers,
    inputs: Inputs,
    formats: [(vk::Format, usize); 3],
    read: [bool; 3],
    /// Whether the view was culled, so its counts belong to this frame.
    culled: bool,
    recorded: stats::Recorded,
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
        let film = shader::compile("film")?;
        let dome = shader::compile("dome")?;
        let sky = shader::compile("sky")?;
        let bake = Bake::new()?;
        Ok(Self {
            store: Store::new(),
            stats,
            commands: Commands::new(&device)?,
            target: None,
            view: None,
            raster: Raster::new(&device, &mesh.spirv)?,
            shade: Shade::new(&device, bake.clone())?,
            shadows: Shadows::new(&device, &shadow.spirv)?,
            cull: Cull::new(&device, &cull.spirv)?,
            dome: Dome::new(&device, &dome.spirv, &sky.spirv, bake)?,
            film: Film::new(&device, &film.spirv)?,
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

    /// Returns whether the buffers now hold this frame, with no build still out and all its samples
    /// drawn. While frames change, each call places the frame before and leaves this one on the
    /// GPU, so GPU work overlaps the host's own frame time.
    pub fn render(&mut self, frame: Frame) -> Result<bool, String> {
        let period = self.stats.begin_render();
        frame.check()?;
        let [_, _, w, h] = frame.window;
        let formats = frame.formats();
        let camera = Camera::new(&frame.world_to_view, &frame.projection)?;
        let buffers = Buffers::of(&frame);
        let inputs = Inputs::of(&frame);
        let mut aovs = [frame.color, frame.depth, frame.normal];

        let moved = self.store.take_moved();
        if let Some(pending) = self.pending.take() {
            self.collect(pending, &buffers, &mut aovs)?;
        }
        let placed = self.placed.as_ref();
        let unchanged = moved == Moved::default()
            && placed == Some(&(buffers, inputs.clone()))
            && self.shade.settled();
        if unchanged && self.film.converged() {
            return Ok(true);
        }
        // The buffers hold the frame before, so this one may be placed by the next call.
        let late = placed.is_some_and(|(b, _)| *b == buffers);

        let start = Instant::now();
        let mut laps = stats::Laps::start();
        self.shadows.reserve(&self.device, inputs.shadow_pool)?;
        let target = match &mut self.target {
            Some(t) if t.fits(&formats, w, h) => t,
            slot => {
                // Free the old target before allocating the new one.
                *slot = None;
                slot.insert(Target::new(
                    &self.device,
                    &formats,
                    &[raster::MOTION],
                    w,
                    h,
                )?)
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
        laps.lap("geometry");
        let footprints = (self.geometry).footprints(&camera.view_projection, [w, h]);
        laps.lap("need");
        let shading = (self.shade).update(
            &self.device,
            &self.store,
            &moved,
            inputs.texture_cap,
            &footprints,
            cb,
        )?;
        self.commands.mark("texture")?;
        laps.lap("textures");
        // Asked again, because the update above may have finished a texture swap. A swap draws the
        // old image until the new one is complete, but finishing must start the sum again.
        let sample = (self.film).begin(unchanged && self.shade.settled());
        // Texture needs use the unjittered camera, since jitter moves a pixel by less than one.
        let unjittered = camera;
        let jitter = self.film.jitter();
        let camera = unjittered.jittered(jitter, [w, h]);
        // With no frame before, nothing has moved.
        let previous_view_projection =
            (self.film.previous_view_projection()).unwrap_or(unjittered.view_projection);
        let addresses = self.geometry.addresses();
        let (dome, sun) =
            (self.dome).update(&self.device, &self.store, cb, inputs.sun_threshold)?;
        if inputs.dome_light_camera_visibility {
            (self.dome).show(&self.device, cb, inputs.texture_cap)?;
        }
        let (mut lights, casters) = light::gather(&self.store, sun);
        laps.lap("lights");
        if let Some(g) = &addresses {
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
        laps.lap("fit");
        let tiles = Tiles::new(&lights, &camera, [w, h]);
        laps.lap("tiles");
        let capacity = light::capacity(lights.len(), [w, h]);
        let mut view = match (&addresses, &mut self.view) {
            (None, _) => None,
            (Some(g), Some(v)) if v.fits(g, capacity) => Some(v),
            (Some(g), slot) => {
                *slot = None;
                Some(slot.insert(View::new(&self.device, g, capacity)?))
            }
        };
        if let (Some(geometry), Some(view)) = (&addresses, view.as_deref_mut()) {
            let shadows = self.shadows.table();
            view.begin(
                cb,
                &camera,
                jitter,
                &previous_view_projection,
                geometry,
                &lights,
                &tiles,
                [w, h],
                self.film.pattern(),
                shadows,
                dome,
            );
            self.cull.record(cb, geometry, view);
        }
        self.commands.mark("cull")?;
        laps.lap("view");
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
        let drawn = addresses.as_ref().zip(view.as_deref());
        if let Some((geometry, view)) = drawn {
            (self.raster).draw(cb, formats[COLOR].0, target.area(), geometry, view);
        }
        target.end_rendering(cb);
        self.commands.mark("draw")?;
        laps.lap("draw");
        // This sits between the prepass and shading, which reads the shadows, because pages follow
        // the depth drawn.
        if let Some((geometry, view)) = drawn {
            let moving = self.geometry.moved_instances(&moved, &update);
            let commands = &mut self.commands;
            (self.shadows).record(commands, geometry, view, target, moving.as_deref())?;
        } else {
            Shadows::skip(&mut self.commands)?;
        }
        laps.lap("shadows");
        target.resume_rendering(cb);
        if let Some((geometry, view)) = drawn {
            let groups = self.geometry.groups();
            (self.shade).draw(cb, formats[COLOR].0, target.area(), geometry, groups, view);
        }
        if inputs.dome_light_camera_visibility {
            let previous = &previous_view_projection;
            (self.dome).draw(
                cb,
                formats[COLOR].0,
                target.area(),
                &camera,
                previous,
                jitter,
            );
        }
        target.end_rendering(cb);
        self.commands.mark("shade")?;
        laps.lap("shade");
        (self.film).record(&self.device, cb, target, &unjittered)?;
        self.commands.mark("film")?;
        laps.lap("film");
        // A still view's depth and normal come from its sample 0, at the pixels' centres, as EEVEE's
        // data passes keep the nearest sample. While anything changes, they come from each frame.
        let read = [0, 1, 2].map(|i| aovs[i].is_some() && (i == COLOR || sample == 0));
        target.readback(cb, &read);
        self.commands.mark("readback")?;
        self.commands.submit()?;
        laps.lap("submit");
        let pending = Pending {
            buffers,
            inputs,
            formats,
            read,
            culled: view.is_some(),
            recorded: stats::Recorded {
                period,
                record: start.elapsed(),
                update,
                shading,
                laps: laps.laps,
                sample,
                lights: lights.len(),
                moved: moved.points.len(),
                transforms: moved.transforms.len(),
                late,
            },
        };
        if late {
            self.pending = Some(pending);
            return Ok(false);
        }
        self.collect(pending, &buffers, &mut aovs)?;
        Ok(self.shade.settled() && self.film.converged())
    }

    /// Places the pending frame, but only if `aovs` are the buffers it was drawn for.
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
                // An AOV that wasn't read back keeps what the buffer holds.
                let Some(aov) = aov.as_mut().filter(|_| pending.read[i]) else {
                    continue;
                };
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
        let mut targets = target.memory();
        targets += self.film.memory();
        let mut views = self.view.as_ref().map(View::memory).unwrap_or_default();
        views += self.dome.memory();
        let memory = stats::Memory {
            geometry: self.geometry.memory(),
            views,
            targets,
            materials: self.shade.memory(),
            shadows: self.shadows.memory(),
            vram_used,
            vram_budget,
        };
        self.stats.frame(stats::Frame {
            width: w,
            height: h,
            recorded: pending.recorded,
            times,
            place,
            read_bytes,
            drawn,
            shadow,
            memory,
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
