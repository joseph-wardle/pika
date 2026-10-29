//! What Pika measures: one record per Hydra sync and one per rendered
//! frame. Each record lists itself as `key value` pairs, which are appended
//! to the file `PIKA_STATS` names (one line per record, for hydra/stats.py
//! and hydra/census.py), summarised when the renderer closes, and handed to
//! the host through Hydra's render stats. A key ending in `_ms` is
//! milliseconds, written with three decimals; every other value is a count,
//! written whole.

use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

use crate::commands::Times;
use crate::geometry::Update;
use crate::gpu;
use crate::store::{Census, Stats as Pulled};

/// The records so far, their file, and the clocks the periods come from.
pub struct Stats {
    file: Option<(String, File)>,
    sync_start: Option<Instant>,
    /// When the last sync began, so each sync can say how long the host
    /// took from one sync to the next.
    last_sync: Option<Instant>,
    /// When the last render call came, so each frame can say the host's
    /// frame period.
    last_render: Option<Instant>,
    pub syncs: Vec<Sync>,
    pub frames: Vec<Frame>,
}

/// One Hydra sync, from the render delegate's `Update` to its
/// `CommitResources`: the prim syncs, including Hydra's own pulls.
pub struct Sync {
    pub time: Duration,
    /// From the last sync's start to this one's; none for the first.
    pub period: Option<Duration>,
    pub pulled: Pulled,
    pub census: Census,
}

/// Where one frame's time went, and what it rewrote and read back.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// From the render call before to the one that drew this frame; none
    /// for the first.
    pub period: Option<Duration>,
    /// The CPU's work before the submit: updating the geometry and recording.
    pub record: Duration,
    pub times: Times,
    /// The CPU copy into the host's render buffers.
    pub place: Duration,
    pub update: Update,
    pub read_bytes: u64,
    /// The instances and triangles the camera's view drew.
    pub drawn: [u32; 2],
    pub memory: Memory,
    /// Meshes whose points moved.
    pub moved: usize,
    /// Meshes whose transform moved.
    pub transforms: usize,
    /// The frame was placed a call later than it was drawn.
    pub latent: bool,
}

/// What each owner holds on the GPU, and what the driver says the process
/// uses and may use of the device's memory.
#[derive(Clone, Copy, Default)]
pub struct Memory {
    pub geometry: gpu::Memory,
    pub views: gpu::Memory,
    pub targets: gpu::Memory,
    pub vram_used: u64,
    pub vram_budget: u64,
}

/// Over 1080p at 24 fps: the gate's frame period.
pub const BUDGET_MS: f64 = 41.67;

fn ms(t: Duration) -> f64 {
    t.as_secs_f64() * 1e3
}

impl Stats {
    /// Opens the `PIKA_STATS` file, if the variable is set, and writes the
    /// host and device every measurement below ran on.
    pub fn new(device_name: &str) -> Result<Self, String> {
        let mut stats = Self {
            file: None,
            sync_start: None,
            last_sync: None,
            last_render: None,
            syncs: Vec::new(),
            frames: Vec::new(),
        };
        if let Ok(path) = std::env::var("PIKA_STATS") {
            let file = File::create(&path)
                .map_err(|e| format!("cannot create PIKA_STATS file {path}: {e}"))?;
            stats.file = Some((path, file));
            let host = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default();
            stats.append(format!("host {}\ndevice {device_name}\n", host.trim()))?;
        }
        Ok(stats)
    }

    pub fn begin_sync(&mut self) {
        self.sync_start = Some(Instant::now());
    }

    /// Records the sync that `begin_sync` started, with what the store
    /// pulled during it and holds after it.
    pub fn end_sync(&mut self, pulled: Pulled, census: Census) -> Result<(), String> {
        let start = self
            .sync_start
            .take()
            .ok_or("a sync ended that never began: the delegate must call pika_sync_begin")?;
        let sync = Sync {
            time: start.elapsed(),
            period: self.last_sync.replace(start).map(|last| start - last),
            pulled,
            census,
        };
        self.append(line("sync", self.syncs.len() + 1, &sync.pairs()))?;
        self.syncs.push(sync);
        Ok(())
    }

    /// Marks a render call; returns the time since the one before.
    pub fn begin_render(&mut self) -> Option<Duration> {
        let now = Instant::now();
        self.last_render.replace(now).map(|last| now - last)
    }

    pub fn frame(&mut self, frame: Frame) -> Result<(), String> {
        self.append(line("frame", self.frames.len() + 1, &frame.pairs()))?;
        self.frames.push(frame);
        Ok(())
    }

    fn append(&mut self, text: String) -> Result<(), String> {
        let Some((path, file)) = &mut self.file else {
            return Ok(());
        };
        file.write_all(text.as_bytes())
            .map_err(|e| format!("cannot write PIKA_STATS file {path}: {e}"))
    }

    /// Every time as p50/p95/max over the syncs and the frames so far, and
    /// the last frame's memory, in a few lines.
    pub fn summary(&self) -> String {
        let percentiles = |values: Vec<f64>| -> String {
            let (p50, p95, max) = percentiles(values);
            format!("{p50:.2}/{p95:.2}/{max:.2}")
        };
        let mut text = format!("{} syncs, {} frames", self.syncs.len(), self.frames.len());
        let Some(last) = self.frames.last() else {
            return text;
        };
        text += &format!(
            ", last at {}×{}; ms as p50/p95/max:",
            last.width, last.height
        );
        let syncs: Vec<_> = self.syncs.iter().map(|s| s.pairs()).collect();
        let frames: Vec<_> = self.frames.iter().map(|f| f.pairs()).collect();
        for (records, prefix) in [(&syncs, "sync_"), (&frames, "")] {
            let Some(keys) = records.last() else {
                continue;
            };
            for (key, _) in keys.iter().filter(|(k, _)| k.ends_with("_ms")) {
                let values = records
                    .iter()
                    .filter_map(|r| r.iter().find(|(k, _)| k == key).map(|(_, v)| *v))
                    .collect();
                let name = key.strip_suffix("_ms").unwrap_or(key);
                text += &format!("\n  {prefix}{name} {}", percentiles(values));
            }
        }
        let periods = self.frames.iter().filter_map(|f| f.period).map(ms);
        let (misses, run) = misses(periods);
        text += &format!("\n  frames over {BUDGET_MS} ms: {misses}, longest run {run}");
        let m = &last.memory;
        text += &format!(
            "\n  memory: geometry {}, views {}, targets {}, host {}; vram {} used of {}",
            size(m.geometry.device),
            size(m.views.device),
            size(m.targets.device),
            size(m.geometry.host + m.views.host + m.targets.host),
            size(m.vram_used),
            size(m.vram_budget),
        );
        text
    }

    /// What the host shows: the counts of records, then the last sync's
    /// and the last frame's pairs, with `gpuMemoryUsed`, the key usdview's
    /// HUD reads.
    pub fn host(&self) -> Vec<(String, f64)> {
        let mut pairs = vec![
            ("syncs".into(), self.syncs.len() as f64),
            ("frames".into(), self.frames.len() as f64),
        ];
        if let Some(sync) = self.syncs.last() {
            pairs.extend(
                sync.pairs()
                    .into_iter()
                    .map(|(k, v)| (format!("sync_{k}"), v)),
            );
        }
        if let Some(frame) = self.frames.last() {
            pairs.extend(frame.pairs());
            pairs.push(("gpuMemoryUsed".into(), frame.memory.vram_used as f64));
        }
        pairs
    }
}

impl Sync {
    pub fn pairs(&self) -> Vec<(String, f64)> {
        let p = &self.pulled;
        let c = &self.census;
        let mut pairs = vec![("time_ms".to_string(), ms(self.time))];
        if let Some(period) = self.period {
            pairs.push(("period_ms".into(), ms(period)));
        }
        let counts = [
            ("inserted", p.inserted),
            ("removed", p.removed),
            ("mesh_bytes", p.mesh.pulled),
            ("mesh_redundant_bytes", p.mesh.redundant),
            ("instancer_bytes", p.instancer.pulled),
            ("instancer_redundant_bytes", p.instancer.redundant),
            ("curve_bytes", p.curve_bytes),
            ("meshes", c.meshes),
            ("mesh_instances", c.mesh_instances),
            ("triangles", c.triangles),
            ("prototype_triangles", c.prototype_triangles),
            ("refined_triangles", c.refined_triangles),
            ("refined_prototype_triangles", c.refined_prototype_triangles),
            ("geometry_bytes", c.geometry_bytes),
            ("refined_bytes", c.refined_bytes),
            ("materials", c.materials),
            ("instancers", c.instancers),
            ("curves", c.curves),
        ];
        pairs.extend(counts.map(|(k, v)| (k.to_string(), v as f64)));
        pairs.extend((c.counted.iter()).map(|(t, n)| (format!("counted_{t}"), *n as f64)));
        pairs
    }
}

impl Frame {
    /// `layout` and `draws` are 1 when they were rewritten, and `latent`
    /// when the frame was placed a call late.
    pub fn pairs(&self) -> Vec<(String, f64)> {
        let m = &self.memory;
        let mut pairs = vec![
            ("width".to_string(), f64::from(self.width)),
            ("height".into(), f64::from(self.height)),
        ];
        if let Some(period) = self.period {
            pairs.push(("period_ms".into(), ms(period)));
        }
        pairs.extend([
            ("record_ms".into(), ms(self.record)),
            ("wait_ms".into(), ms(self.times.wait)),
            ("place_ms".into(), ms(self.place)),
        ]);
        pairs.extend(
            (self.times.spans.iter()).map(|(label, t)| (format!("gpu_{label}_ms"), ms(*t))),
        );
        let counts = [
            ("uploaded_bytes", self.update.bytes),
            ("read_bytes", self.read_bytes),
            ("drawn_instances", u64::from(self.drawn[0])),
            ("drawn_triangles", u64::from(self.drawn[1])),
            ("geometry_bytes", m.geometry.device),
            ("view_bytes", m.views.device),
            ("target_bytes", m.targets.device),
            (
                "host_bytes",
                m.geometry.host + m.views.host + m.targets.host,
            ),
            ("vram_used_bytes", m.vram_used),
            ("vram_budget_bytes", m.vram_budget),
            ("moved_meshes", self.moved as u64),
            ("moved_transforms", self.transforms as u64),
            ("layout", u64::from(self.update.layout)),
            ("draws", u64::from(self.update.draws)),
            ("latent", u64::from(self.latent)),
        ];
        pairs.extend(counts.map(|(k, v)| (k.to_string(), v as f64)));
        pairs
    }
}

/// One record as a line: its kind and index, then its pairs.
fn line(kind: &str, index: usize, pairs: &[(String, f64)]) -> String {
    let mut text = format!("{kind} {index}");
    for (key, value) in pairs {
        match key.ends_with("_ms") {
            true => text += &format!(" {key} {value:.3}"),
            false => text += &format!(" {key} {value:.0}"),
        }
    }
    text + "\n"
}

/// p50, p95 and max of a non-empty set, by nearest rank.
fn percentiles(mut values: Vec<f64>) -> (f64, f64, f64) {
    values.sort_by(f64::total_cmp);
    let rank = |p: f64| values[((p * values.len() as f64) as usize).min(values.len() - 1)];
    (rank(0.5), rank(0.95), values[values.len() - 1])
}

/// How many periods exceed the budget, and the longest run of consecutive
/// ones that do.
fn misses(periods: impl Iterator<Item = f64>) -> (usize, usize) {
    let (mut misses, mut run, mut longest) = (0, 0, 0);
    for period in periods {
        run = if period > BUDGET_MS { run + 1 } else { 0 };
        misses += usize::from(period > BUDGET_MS);
        longest = longest.max(run);
    }
    (misses, longest)
}

/// Bytes in the unit that reads best.
fn size(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(period: Option<f64>) -> Frame {
        let ms = |t: f64| Duration::from_secs_f64(t / 1e3);
        Frame {
            width: 1620,
            height: 1080,
            period: period.map(ms),
            record: ms(1.5),
            times: Times {
                wait: ms(5.25),
                spans: vec![
                    ("update", ms(0.875)),
                    ("cull", ms(0.125)),
                    ("draw", ms(2.0)),
                    ("readback", ms(1.375)),
                ],
            },
            place: ms(4.5),
            update: Update {
                bytes: 4096,
                layout: false,
                draws: true,
            },
            read_bytes: 20_995_200,
            drawn: [1274, 11_550_000],
            memory: Memory {
                geometry: gpu::Memory {
                    device: 1 << 30,
                    host: 1 << 20,
                },
                views: gpu::Memory {
                    device: 40_000,
                    host: 8,
                },
                targets: gpu::Memory {
                    device: 14_000_000,
                    host: 21_000_000,
                },
                vram_used: 2_000_000_000,
                vram_budget: 15_000_000_000,
            },
            moved: 338,
            transforms: 12,
            latent: true,
        }
    }

    #[test]
    fn frame_line_is_key_value_pairs_named_by_mark() {
        let expected = "frame 7 width 1620 height 1080 period_ms 29.000 record_ms 1.500 \
             wait_ms 5.250 place_ms 4.500 gpu_update_ms 0.875 gpu_cull_ms 0.125 \
             gpu_draw_ms 2.000 gpu_readback_ms 1.375 uploaded_bytes 4096 \
             read_bytes 20995200 drawn_instances 1274 drawn_triangles 11550000 \
             geometry_bytes 1073741824 view_bytes 40000 target_bytes 14000000 \
             host_bytes 22048584 vram_used_bytes 2000000000 vram_budget_bytes 15000000000 \
             moved_meshes 338 moved_transforms 12 layout 0 draws 1 latent 1\n";
        assert_eq!(line("frame", 7, &frame(Some(29.0)).pairs()), expected);
        // A first frame has no period, so it is not the reference's line
        // with the period zeroed.
        let first = line("frame", 7, &frame(None).pairs());
        assert!(!first.contains("period_ms"), "{first}");
        assert_ne!(
            first,
            expected.replace("period_ms 29.000", "period_ms 0.000")
        );
    }

    #[test]
    fn host_pairs_carry_the_last_frame_and_usdviews_key() {
        let mut stats = Stats {
            file: None,
            sync_start: None,
            last_sync: None,
            last_render: None,
            syncs: Vec::new(),
            frames: vec![frame(None), frame(Some(29.0))],
        };
        let get = |stats: &Stats, key: &str| {
            (stats.host().into_iter())
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
        };
        assert_eq!(get(&stats, "frames"), Some(2.0));
        assert_eq!(get(&stats, "period_ms"), Some(29.0));
        assert_eq!(get(&stats, "gpuMemoryUsed"), Some(2e9));
        // The control: without frames the frame keys are absent, not zero.
        stats.frames.clear();
        assert_eq!(get(&stats, "frames"), Some(0.0));
        assert_eq!(get(&stats, "period_ms"), None);
        assert_eq!(get(&stats, "gpuMemoryUsed"), None);
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let values: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentiles(values.clone()), (51.0, 96.0, 100.0));
        // Not interpolated: the p95 of 1..=100 would be 95.05.
        assert_ne!(percentiles(values).1, 95.05);
        assert_eq!(percentiles(vec![3.0]), (3.0, 3.0, 3.0));
    }

    #[test]
    fn misses_count_periods_over_budget_and_their_longest_run() {
        let periods = [30.0, 42.0, 50.0, 30.0, 45.0, 41.67];
        assert_eq!(misses(periods.into_iter()), (3, 2));
        // The control: the same periods a hair under the budget miss nothing.
        assert_eq!(misses(periods.iter().map(|p| p - 9.0)), (0, 0));
    }

    #[test]
    fn summary_lists_each_time_once() {
        let stats = Stats {
            file: None,
            sync_start: None,
            last_sync: None,
            last_render: None,
            syncs: Vec::new(),
            frames: vec![frame(None), frame(Some(29.0)), frame(Some(31.0))],
        };
        let summary = stats.summary();
        assert!(
            summary.starts_with("0 syncs, 3 frames, last at 1620×1080"),
            "{summary}"
        );
        assert!(
            summary.contains("\n  period 31.00/31.00/31.00"),
            "{summary}"
        );
        assert!(summary.contains("\n  gpu_draw 2.00/2.00/2.00"), "{summary}");
        assert_eq!(summary.matches("gpu_draw").count(), 1);
        assert!(
            summary.contains("frames over 41.67 ms: 0, longest run 0"),
            "{summary}"
        );
        assert!(
            summary.contains("vram 2.00 GB used of 15.00 GB"),
            "{summary}"
        );
    }
}
