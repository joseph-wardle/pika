//! What one Hydra render delegate owns on the Rust side: its scene store,
//! its Vulkan device and the color target it reads frames back from.

use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ash::vk;

use crate::gpu;
use crate::store::{self, Store};
use crate::target::Target;

/// The color formats Pika renders, as Hydra names them. Each host picks one:
/// husk asks for `HdFormatFloat32Vec4`, and hosts that take the delegate's
/// default get `HdFormatFloat16Vec4`, which halves the bytes read back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Float16Vec4,
    Float32Vec4,
}

impl Format {
    /// The value `pika.h` passes for this format.
    pub fn from_abi(value: u32) -> Result<Self, String> {
        match value {
            1 => Ok(Self::Float16Vec4),
            2 => Ok(Self::Float32Vec4),
            _ => Err(format!("unknown pika_format {value}")),
        }
    }

    fn vk(self) -> vk::Format {
        match self {
            Self::Float16Vec4 => vk::Format::R16G16B16A16_SFLOAT,
            Self::Float32Vec4 => vk::Format::R32G32B32A32_SFLOAT,
        }
    }

    fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Float16Vec4 => 8,
            Self::Float32Vec4 => 16,
        }
    }
}

/// The skeleton's whole image. Exactly representable in both formats, and not
/// black, so a buffer that was never written cannot pass for it.
pub const CLEAR: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

// One instance per process: plugins are never unloaded, and each render
// delegate's device borrows it.
static INSTANCE: OnceLock<Result<gpu::Instance, String>> = OnceLock::new();

pub struct Renderer {
    pub store: Store,
    // Field order is drop order: the target is destroyed before its device.
    target: Option<Target>,
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
    frames: Vec<Frame>,
}

/// One Hydra sync, from the render delegate's `Update` to its
/// `CommitResources`: the prim syncs, including Hydra's own pulls.
struct Sync {
    time: Duration,
    stats: store::Stats,
}

/// Where one frame's time went: the GPU clear and copy to host memory, then
/// the CPU copy into the host's render buffer.
struct Frame {
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
        Ok(Self {
            store: Store::new(),
            target: None,
            device: instance.open(index)?,
            device_name: format!("{}, validation {validation}", instance.devices[index].name),
            dump,
            stats,
            sync_start: None,
            last_start: None,
            syncs: Vec::new(),
            frames: Vec::new(),
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

    /// Renders a `width` × `height` frame into `pixels`, the host's render
    /// buffer of `format`.
    pub fn render(
        &mut self,
        format: Format,
        width: u32,
        height: u32,
        pixels: &mut [u8],
    ) -> Result<(), String> {
        let expected = width as usize * height as usize * format.bytes_per_pixel();
        if pixels.len() != expected {
            return Err(format!(
                "the render buffer is {} bytes; {width}×{height} {format:?} needs {expected}",
                pixels.len()
            ));
        }
        let target = match &mut self.target {
            Some(t) if (t.format, t.width, t.height) == (format.vk(), width, height) => t,
            slot => {
                // Free the old target before allocating its replacement.
                *slot = None;
                slot.insert(Target::new(
                    &self.device,
                    format.vk(),
                    format.bytes_per_pixel(),
                    width,
                    height,
                )?)
            }
        };

        let clear = vk::ClearValue {
            color: vk::ClearColorValue { float32: CLEAR },
        };
        target.begin(clear)?;
        let submit = target.finish()?;
        let start = Instant::now();
        pixels.copy_from_slice(target.pixels());
        self.frames.push(Frame {
            submit,
            copy: start.elapsed(),
        });
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
        let (Some(target), false) = (&self.target, self.frames.is_empty()) else {
            return format!("{syncs}; no frames rendered");
        };
        format!(
            "{syncs}; {} frames, last at {}×{} {:?}: clear and readback {}; copy into the render buffer {}",
            self.frames.len(),
            target.width,
            target.height,
            target.format,
            percentiles(self.frames.iter().map(|f| f.submit)),
            percentiles(self.frames.iter().map(|f| f.copy)),
        )
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
