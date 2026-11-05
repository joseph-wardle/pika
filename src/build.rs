//! Material builds, off the render thread. A job carries one material's
//! document and network to one of `WORKERS` threads, which generates and
//! compiles its surface and bakes and reads its textures through the cache
//! (`bake`, `shader`), and the result waits for the frame that installs it.
//! Each job for a material is a later generation than the one before, and
//! `drain` hands out only the latest's result, so an edit's build never
//! replaces a newer edit's.

use std::collections::BTreeMap;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use crate::abi;
use crate::bake::{Bake, Baked, Manifest};
use crate::material::{self, Document, Source};
use crate::shader;
use crate::store::Network;
use crate::surface;

// slangc takes one core for a third of a second, and the bake tool spreads
// a texture over every core, so a few jobs at a time keep the cores busy
// without piling up decoded 4K images.
const WORKERS: usize = 4;

struct Job {
    id: u64,
    generation: u64,
    document: Document,
    network: Network,
    cap: u32,
    /// Whether the renderer holds the document's surface, so the job skips
    /// its compile.
    held: bool,
}

/// A finished job, with what the frame installing it needs.
pub struct Built {
    pub id: u64,
    generation: u64,
    pub document: Document,
    pub network: Network,
    pub cap: u32,
    pub result: Result<Output, String>,
}

impl Built {
    /// `network`'s build, run now on this thread: Pika's own networks are
    /// built before the first frame, with no textures and so no cap.
    pub fn now(bake: &Bake, document: Document, network: Network) -> Self {
        let result = run(bake, &document, &network, 0, false);
        Self {
            id: 0,
            generation: 0,
            document,
            network,
            cap: 0,
            result,
        }
    }
}

pub struct Output {
    pub manifest: Manifest,
    /// The surface's SPIR-V and reflection, unless the renderer holds it.
    pub surface: Option<shader::Surface>,
    /// What the textures were baked from, and the textures, in the
    /// manifest's order.
    pub sources: Vec<Source>,
    pub textures: Vec<Baked>,
}

pub struct Builder {
    jobs: Sender<Job>,
    done: Receiver<Built>,
    latest: Latest,
    /// What `wait` received, for `drain`.
    arrived: Vec<Built>,
}

/// The generation of each material's latest job while it is out.
#[derive(Default)]
struct Latest {
    jobs: BTreeMap<u64, u64>,
    generation: u64,
}

impl Builder {
    pub fn new(bake: Bake) -> Self {
        let (jobs, queue) = mpsc::channel();
        let queue = Arc::new(Mutex::new(queue));
        let (finished, done) = mpsc::channel();
        for _ in 0..WORKERS {
            let (bake, queue, finished) = (bake.clone(), queue.clone(), finished.clone());
            std::thread::spawn(move || work(&bake, &queue, &finished));
        }
        Self {
            jobs,
            done,
            latest: Latest::default(),
            arrived: Vec::new(),
        }
    }

    /// Queues a build of material `id`'s `network`, whose document is
    /// `document`, under the texture cap `cap`; `held` says the renderer
    /// holds the document's surface already.
    pub fn submit(
        &mut self,
        id: u64,
        document: Document,
        network: Network,
        cap: u32,
        held: bool,
    ) -> Result<(), String> {
        let generation = self.latest.submit(id);
        let job = Job {
            id,
            generation,
            document,
            network,
            cap,
            held,
        };
        (self.jobs.send(job)).map_err(|_| "the build workers have exited".to_string())
    }

    /// Waits up to `timeout` for a build to finish, and says whether one
    /// did; at once when none is out.
    pub fn wait(&mut self, timeout: Duration) -> Result<bool, String> {
        if self.latest.jobs.is_empty() {
            return Ok(false);
        }
        match self.done.recv_timeout(timeout) {
            Ok(built) => {
                self.arrived.push(built);
                Ok(true)
            }
            Err(RecvTimeoutError::Timeout) => Ok(false),
            Err(RecvTimeoutError::Disconnected) => Err("the build workers have exited".into()),
        }
    }

    /// The finished builds that are their material's latest; a stale one
    /// is dropped here.
    pub fn drain(&mut self) -> Vec<Built> {
        let mut built = std::mem::take(&mut self.arrived);
        built.extend(self.done.try_iter());
        built.retain(|b| self.latest.landed(b.id, b.generation));
        built
    }

    /// Drops material `id`'s job, if one is out: its result will be stale.
    pub fn forget(&mut self, id: u64) {
        self.latest.jobs.remove(&id);
    }

    pub fn retain(&mut self, keep: impl Fn(&u64) -> bool) {
        self.latest.jobs.retain(|id, _| keep(id));
    }

    /// The materials whose build is out.
    pub fn len(&self) -> usize {
        self.latest.jobs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.latest.jobs.is_empty()
    }
}

impl Latest {
    fn submit(&mut self, id: u64) -> u64 {
        self.generation += 1;
        self.jobs.insert(id, self.generation);
        self.generation
    }

    /// Whether the job of `generation` is material `id`'s latest, which
    /// then lands; an older one, or one forgotten, does not.
    fn landed(&mut self, id: u64, generation: u64) -> bool {
        if self.jobs.get(&id) != Some(&generation) {
            return false;
        }
        self.jobs.remove(&id);
        true
    }
}

/// A worker: builds each job from the queue until the builder is dropped.
/// One dropped mid-build finishes it; the cache's temporary names keep a
/// build cut short by the process exiting from leaving a partial entry.
fn work(bake: &Bake, queue: &Mutex<Receiver<Job>>, finished: &Sender<Built>) {
    loop {
        let job = queue.lock().unwrap_or_else(PoisonError::into_inner).recv();
        let Ok(job) = job else {
            return;
        };
        // A panic here would leave the material pending for good, with no
        // frame to report it: it is the build's failure, warned of like one.
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            run(bake, &job.document, &job.network, job.cap, job.held)
        }))
        .unwrap_or_else(|payload| {
            Err(format!(
                "its build panicked: {}",
                abi::panic_message(payload.as_ref())
            ))
        });
        let built = Built {
            id: job.id,
            generation: job.generation,
            document: job.document,
            network: job.network,
            cap: job.cap,
            result,
        };
        if finished.send(built).is_err() {
            return;
        }
    }
}

/// Generates `document`'s stages and, unless `held`, compiles its surface,
/// then bakes and reads each texture `network` gives it, under the cap
/// `cap`.
fn run(
    bake: &Bake,
    document: &Document,
    network: &Network,
    cap: u32,
    held: bool,
) -> Result<Output, String> {
    let generated = bake.generate(document)?;
    let surface = if held {
        None
    } else {
        Some(shader::surface(&generated.dir)?)
    };
    let sources = sources(&generated.manifest, document, network)?;
    let mut textures = Vec::new();
    for source in &sources {
        let file = Path::new(&source.file);
        textures.push(bake.texture(file, &source.space, source.encoding, cap)?);
    }
    Ok(Output {
        manifest: generated.manifest,
        surface,
        sources,
        textures,
    })
}

/// The texture each of `manifest`'s textures reads, from `network`'s node
/// the name stands for, by `document`'s node order.
pub fn sources(
    manifest: &Manifest,
    document: &Document,
    network: &Network,
) -> Result<Vec<Source>, String> {
    let mut sources = Vec::new();
    for name in &manifest.textures {
        let (index, input) = surface::node_input(name)?;
        let path = (document.nodes.get(index))
            .ok_or_else(|| format!("{name} names a node the document lacks"))?;
        let node = &network.nodes[path];
        let source = material::source(node, material::store_input(node, input))
            .map_err(|e| format!("{path}'s {input} {e}"))?;
        sources.push(source);
    }
    Ok(sources)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_latest_job_lands() {
        let mut latest = Latest::default();
        let first = latest.submit(7);
        let second = latest.submit(7);
        assert!(!latest.landed(7, first), "the older edit's build is stale");
        assert_eq!(latest.jobs.len(), 1);
        assert!(latest.landed(7, second));
        assert!(latest.jobs.is_empty());
        assert!(!latest.landed(7, second), "a build lands once");

        let other = latest.submit(8);
        latest.jobs.remove(&8);
        assert!(
            !latest.landed(8, other),
            "a forgotten material's build is stale"
        );
    }
}
