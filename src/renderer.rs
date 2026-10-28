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
    sync_start: Option<Instant>,
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
        let dump = match std::env::var("PIKA_STORE_DUMP") {
            Ok(path) => {
                let file = File::create(&path)
                    .map_err(|e| format!("cannot create PIKA_STORE_DUMP file {path}: {e}"))?;
                Some((path, file))
            }
            Err(_) => None,
        };
        Ok(Self {
            store: Store::new(),
            target: None,
            device: instance.open(index)?,
            device_name: format!("{}, validation {validation}", instance.devices[index].name),
            dump,
            sync_start: None,
            syncs: Vec::new(),
            frames: Vec::new(),
        })
    }

    pub fn begin_sync(&mut self) {
        self.sync_start = Some(Instant::now());
    }

    /// Records the sync's time and stats, and appends the store to the dump.
    pub fn end_sync(&mut self) -> Result<(), String> {
        let start = self
            .sync_start
            .take()
            .ok_or("a sync ended that never began: the delegate must call pika_sync_begin")?;
        self.syncs.push(Sync {
            time: start.elapsed(),
            stats: self.store.end_sync(),
        });
        let Some((path, file)) = &mut self.dump else {
            return Ok(());
        };
        let text = format!("sync {}\n{}", self.syncs.len(), self.store.dump());
        file.write_all(text.as_bytes())
            .map_err(|e| format!("cannot write PIKA_STORE_DUMP file {path}: {e}"))
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
            let bytes: usize = self.syncs.iter().map(|s| s.stats.bytes).sum();
            let redundant: usize = self.syncs.iter().map(|s| s.stats.redundant_bytes).sum();
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
