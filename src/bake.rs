//! Runs the bake tools, cached, and builds materials on worker threads. A
//! build's result lands only while it is its material's latest job.

use std::collections::BTreeMap;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard, Once, PoisonError};
use std::time::{Duration, UNIX_EPOCH};

use crate::cache;
use crate::material::{self, Document, Network, Source};
use crate::shader;
use crate::texture::{self, Texture};
use crate::uniforms::{self, Manifest};

const GENERATE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/target/bake/generate");
const TEXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/target/bake/texture");
// Shadows Houdini's library files; the generator inlines them.
const LIBRARIES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bake/libraries");

pub struct Generated {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

pub struct Baked {
    /// Its cache file, or `fallback` and its texel.
    pub key: String,
    pub texture: Texture,
    /// The tool ran rather than the cache answering.
    pub fresh: bool,
}

#[derive(Clone)]
pub struct Bake {
    materials: PathBuf,
    textures: PathBuf,
    /// The host's OCIO config, or empty for OpenImageIO's default.
    config: String,
    generate: Tool,
    texture: Tool,
}

/// A bake tool and its cache key: its bytes and the libraries it lists, so a
/// rebuilt tool or a Houdini update misses its own cache alone.
#[derive(Clone)]
struct Tool {
    path: &'static str,
    hash: u64,
}

impl Tool {
    fn new(path: &'static str, more: &[u8]) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| {
            format!("cannot read the bake tool {path}: {e}. Build it with `make -C bake`.")
        })?;
        let mut tool = Self { path, hash: 0 };
        let output = (tool.command().arg("libraries").output())
            .map_err(|e| format!("could not run the bake tool {path}: {e}"))?;
        if !output.status.success() {
            return Err(failed(&output));
        }
        tool.hash = cache::fnv1a(&[&bytes, more, &output.stdout]);
        Ok(tool)
    }

    fn command(&self) -> Command {
        let mut command = Command::new(self.path);
        // A host's library path beats the runpath of the libraries the tool loads:
        // a Houdini 21 host would mix its codecs with Houdini 22's MaterialX.
        command.env_remove("LD_LIBRARY_PATH");
        command
    }

    fn run(&self, command: &mut Command) -> Result<(), String> {
        let output = (command.output())
            .map_err(|e| format!("could not run the bake tool {}: {e}", self.path))?;
        match output.status.success() {
            true => Ok(()),
            false => Err(failed(&output)),
        }
    }
}

impl Bake {
    pub fn new() -> Result<Self, String> {
        let mut libraries = Vec::new();
        for file in files(Path::new(LIBRARIES))? {
            libraries.extend_from_slice(file.as_os_str().as_encoded_bytes());
            libraries.extend(
                std::fs::read(&file).map_err(|e| format!("cannot read {}: {e}", file.display()))?,
            );
        }
        let generate = Tool::new(GENERATE, &libraries)?;
        let texture = Tool::new(TEXTURE, &[])?;
        let root = cache::root()?;
        let (materials, textures) = (root.join("materials"), root.join("textures"));
        for dir in [&materials, &textures] {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("cannot create the cache {}: {e}", dir.display()))?;
        }
        let mounts = std::fs::read_to_string("/proc/self/mounts").unwrap_or_default();
        if cache::on_nfs(&mounts, &textures) {
            static WARNED: Once = Once::new();
            WARNED.call_once(|| {
                eprintln!(
                    "pika: warning: the cache {} is on NFS, from which textures upload slowly; \
                     set PIKA_CACHE to a local disk",
                    root.display()
                )
            });
        }
        Ok(Self {
            materials,
            textures,
            config: std::env::var("OCIO").unwrap_or_default(),
            generate,
            texture,
        })
    }

    pub fn generate(&self, document: &Document) -> Result<Generated, String> {
        let key = cache::fnv1a(&[document.text.as_bytes(), &self.generate.hash.to_le_bytes()]);
        let dir = self.materials.join(format!("{key:016x}"));
        match generated(&dir) {
            Ok(manifest) => return Ok(Generated { manifest, dir }),
            // Left partial by a run that never finished.
            Err(why) if dir.exists() => cache::discard(&dir, &why)?,
            Err(_) => {}
        }
        // Run beside `dir` and rename into place, so `dir` exists only whole.
        let tmp = dir.with_extension(cache::unique());
        let path = tmp.join("document.mtlx");
        let run = || -> Result<(), String> {
            std::fs::create_dir_all(&tmp)
                .map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
            std::fs::write(&path, &document.text)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            let tool = &self.generate;
            tool.run(tool.command().arg(&path).arg(&tmp))?;
            match std::fs::rename(&tmp, &dir) {
                Ok(()) => Ok(()),
                Err(_) if dir.join("manifest").is_file() => Ok(()),
                Err(e) => Err(format!("cannot move {} into place: {e}", tmp.display())),
            }
        };
        let result = run();
        if result.is_err() || tmp.is_dir() {
            let _ = std::fs::remove_dir_all(&tmp);
        }
        result?;
        Ok(Generated {
            manifest: generated(&dir)?,
            dir,
        })
    }

    pub fn texture(&self, source: &Source) -> Result<Baked, String> {
        let set = tiles(Path::new(&source.file))?;
        let path = self.texture_path(source, &set)?;
        let key = path.to_string_lossy().into_owned();
        if path.is_file() {
            match texture::read(&path) {
                Ok(texture) => {
                    return Ok(Baked {
                        key,
                        texture,
                        fresh: false,
                    });
                }
                Err(why) => cache::discard(&path, &why)?,
            }
        }
        let dir = path.parent().unwrap();
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let tmp = path.with_extension(cache::unique());
        let mut command = self.texture.command();
        command
            .args([&source.space, source.encoding])
            .arg(&self.config)
            .arg(&tmp);
        for (number, tile) in &set {
            command.arg(number.to_string()).arg(tile);
        }
        if let Err(why) = self.texture.run(&mut command) {
            let _ = std::fs::remove_file(&tmp);
            return Err(why);
        }
        std::fs::rename(&tmp, &path)
            .map_err(|e| format!("cannot move {} into place: {e}", tmp.display()))?;
        Ok(Baked {
            texture: texture::read(&path)?,
            key,
            fresh: true,
        })
    }

    /// Mirrors the source's path so an asset's bakes can be found and deleted.
    /// Not keyed on the cap: the bake holds every level.
    pub fn texture_path(&self, source: &Source, set: &[(u32, PathBuf)]) -> Result<PathBuf, String> {
        let mut state = Vec::new();
        for (_, file) in set {
            let meta = std::fs::metadata(file)
                .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
            let modified = meta
                .modified()
                .map_err(|e| format!("{} has no modification time: {e}", file.display()))?
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default();
            state.extend_from_slice(file.file_name().unwrap().as_encoded_bytes());
            state.extend_from_slice(&meta.len().to_le_bytes());
            state.extend_from_slice(&modified.as_secs().to_le_bytes());
            state.extend_from_slice(&modified.subsec_nanos().to_le_bytes());
        }
        let key = cache::fnv1a(&[
            &state,
            source.space.as_bytes(),
            b"\0",
            source.encoding.as_bytes(),
            b"\0",
            self.config.as_bytes(),
            b"\0",
            &self.texture.hash.to_le_bytes(),
        ]);
        let source = Path::new(&source.file);
        let dir = source.parent().unwrap_or(Path::new("."));
        let dir = std::fs::canonicalize(dir)
            .map_err(|e| format!("cannot find the texture {}: {e}", source.display()))?;
        let mirror: PathBuf = (dir
            .join(source.file_name().unwrap_or_default())
            .components())
        .filter(|c| matches!(c, Component::Normal(_)))
        .collect();
        Ok(self.textures.join(mirror).join(format!("{key:016x}")))
    }
}

fn files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut found = Vec::new();
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
            .path();
        if path.is_dir() {
            found.extend(files(&path)?);
        } else {
            found.push(path);
        }
    }
    found.sort();
    Ok(found)
}

/// A single file is tile 0; a `<UDIM>` pattern is tiles 1001 to 1100, as USD reads a set.
pub fn tiles(source: &Path) -> Result<Vec<(u32, PathBuf)>, String> {
    let text = source.to_string_lossy();
    // The delegate passes a path that did not resolve as empty.
    if text.is_empty() {
        return Err("its path is empty or did not resolve".into());
    }
    let Some((before, after)) = text.split_once("<UDIM>") else {
        if !source.is_file() {
            return Err(format!("cannot find the texture {text}"));
        }
        return Ok(vec![(0, source.to_owned())]);
    };
    let set: Vec<(u32, PathBuf)> = (1001..=1100)
        .map(|number| (number, PathBuf::from(format!("{before}{number}{after}"))))
        .filter(|(_, tile)| tile.is_file())
        .collect();
    if set.is_empty() {
        return Err(format!(
            "no tiles of {text} exist: UDIM tiles are numbered 1001 to 1100"
        ));
    }
    Ok(set)
}

fn generated(dir: &Path) -> Result<Manifest, String> {
    let path = dir.join("manifest");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    for name in ["pixel.slang", "surface.slang"] {
        if !dir.join(name).is_file() {
            return Err(format!("{} has no {name}", dir.display()));
        }
    }
    uniforms::manifest(&text)
}

/// On one line, so a warning quoting it stays one.
fn failed(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    format!(
        "the bake tool failed: {}",
        stderr.trim().replace('\n', "; ")
    )
}

// slangc takes one core and the texture tool all of them; a few jobs keep
// the cores busy without piling up decoded 4K images.
const WORKERS: usize = 4;

struct Job {
    id: u64,
    generation: u64,
    document: Document,
    network: Network,
}

pub struct Built {
    pub id: u64,
    generation: u64,
    pub document: Document,
    pub network: Network,
    pub result: Result<Output, String>,
}

impl Built {
    pub fn missing(&self) -> &[String] {
        match &self.result {
            Ok(output) => &output.missing,
            Err(_) => &[],
        }
    }

    /// Builds on this thread, without textures: Pika's own networks, before the first frame.
    pub fn now(bake: &Bake, document: Document, network: Network) -> Self {
        let result = build(bake, &document, &network, &|| true);
        Self {
            id: 0,
            generation: 0,
            document,
            network,
            result,
        }
    }
}

pub struct Output {
    pub manifest: Manifest,
    pub compiled: shader::Compiled,
    /// Both in the manifest's order.
    pub sources: Vec<Source>,
    pub textures: Vec<Baked>,
    /// Warnings for texture inputs whose file is missing, drawn as fallbacks.
    pub missing: Vec<String>,
}

pub struct Builder {
    jobs: Sender<Job>,
    done: Receiver<Built>,
    latest: Latest,
    arrived: Vec<Built>,
}

/// Each material's latest generation while its job is out; workers skip stale jobs.
#[derive(Default)]
struct Latest {
    jobs: Arc<Mutex<BTreeMap<u64, u64>>>,
    generation: u64,
}

impl Builder {
    pub fn new(bake: Bake) -> Self {
        let (jobs, queue) = mpsc::channel();
        let queue = Arc::new(Mutex::new(queue));
        // Bounded: a finished build holds its baked files until a frame installs it.
        let (finished, done) = mpsc::sync_channel(WORKERS);
        let latest = Latest::default();
        for _ in 0..WORKERS {
            let (bake, queue, finished) = (bake.clone(), queue.clone(), finished.clone());
            let jobs = latest.jobs.clone();
            std::thread::spawn(move || work(&bake, &queue, &finished, &jobs));
        }
        Self {
            jobs,
            done,
            latest,
            arrived: Vec::new(),
        }
    }

    pub fn submit(&mut self, id: u64, document: Document, network: Network) -> Result<(), String> {
        let generation = self.latest.submit(id);
        let job = Job {
            id,
            generation,
            document,
            network,
        };
        (self.jobs.send(job)).map_err(|_| "the build workers have exited".to_string())
    }

    pub fn wait(&mut self, timeout: Duration) -> Result<bool, String> {
        if !self.arrived.is_empty() {
            return Ok(true);
        }
        if self.latest.jobs().is_empty() {
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

    /// The finished builds still their material's latest; stale ones are dropped.
    pub fn drain(&mut self) -> Vec<Built> {
        let mut arrived = std::mem::take(&mut self.arrived);
        arrived.extend(self.done.try_iter());
        (arrived.into_iter())
            .filter(|b| self.latest.landed(b.id, b.generation))
            .collect()
    }

    /// Drops material `id`'s job: its result will not land.
    pub fn forget(&mut self, id: u64) {
        self.latest.jobs().remove(&id);
    }

    pub fn retain(&mut self, keep: impl Fn(&u64) -> bool) {
        self.latest.jobs().retain(|id, _| keep(id));
    }

    pub fn len(&self) -> usize {
        self.latest.jobs().len()
    }

    pub fn is_empty(&self) -> bool {
        self.latest.jobs().is_empty()
    }
}

impl Latest {
    fn jobs(&self) -> MutexGuard<'_, BTreeMap<u64, u64>> {
        self.jobs.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn submit(&mut self, id: u64) -> u64 {
        self.generation += 1;
        self.jobs().insert(id, self.generation);
        self.generation
    }

    /// Whether this is material `id`'s latest job; each lands at most once.
    fn landed(&mut self, id: u64, generation: u64) -> bool {
        let mut jobs = self.jobs();
        if jobs.get(&id) != Some(&generation) {
            return false;
        }
        jobs.remove(&id);
        true
    }
}

fn current(jobs: &Mutex<BTreeMap<u64, u64>>, id: u64, generation: u64) -> bool {
    (jobs.lock().unwrap_or_else(PoisonError::into_inner)).get(&id) == Some(&generation)
}

/// A dropped builder finishes the job in hand; the cache's temporary names
/// keep an exit mid-build from leaving a partial entry.
fn work(
    bake: &Bake,
    queue: &Mutex<Receiver<Job>>,
    finished: &SyncSender<Built>,
    jobs: &Mutex<BTreeMap<u64, u64>>,
) {
    loop {
        let job = queue.lock().unwrap_or_else(PoisonError::into_inner).recv();
        let Ok(job) = job else {
            return;
        };
        let wanted = || current(jobs, job.id, job.generation);
        if !wanted() {
            continue;
        }
        // A panic would leave the material pending for good; report it as a failure.
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            build(bake, &job.document, &job.network, &wanted)
        }))
        .unwrap_or_else(|payload| {
            let message = (payload.downcast_ref::<&str>().copied())
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown payload");
            Err(format!("its build panicked: {message}"))
        });
        let built = Built {
            id: job.id,
            generation: job.generation,
            document: job.document,
            network: job.network,
            result,
        };
        if finished.send(built).is_err() {
            return;
        }
    }
}

/// Textures are read before the compile, so a bad one fails the build first.
fn build(
    bake: &Bake,
    document: &Document,
    network: &Network,
    wanted: &dyn Fn() -> bool,
) -> Result<Output, String> {
    let generated = bake.generate(document)?;
    let sources = sources(&generated.manifest, document, network)?;
    let mut textures = Vec::new();
    let mut missing = Vec::new();
    for source in &sources {
        if !wanted() {
            return Err("its build was superseded".into());
        }
        let (path, input) = (&source.node, &source.input);
        if source.file.is_empty() {
            missing.push(format!(
                "{path}'s {input}: its path is empty or did not resolve"
            ));
            textures.push(fallback(&source.default));
            continue;
        }
        let baked = (bake.texture(source)).map_err(|e| format!("{path}'s {input}: {e}"))?;
        textures.push(baked);
    }
    let compiled = shader::surface(&generated.dir)?;
    Ok(Output {
        manifest: generated.manifest,
        compiled,
        sources,
        textures,
        missing,
    })
}

/// A 1×1 texel of `default`: the Slang image implementation never reads the
/// node's own `default` input.
fn fallback(default: &[f32]) -> Baked {
    let mut texel = [0.0; 4];
    let n = default.len().min(4);
    texel[..n].copy_from_slice(&default[..n]);
    Baked {
        key: format!("fallback {texel:?}"),
        texture: texture::solid(texel),
        fresh: false,
    }
}

pub fn sources(
    manifest: &Manifest,
    document: &Document,
    network: &Network,
) -> Result<Vec<Source>, String> {
    (manifest.textures.iter())
        .map(|name| {
            let (path, node, input) = document.reader(network, name)?;
            material::source(network, path, node, input)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_files_are_listed_in_order() {
        let dir = std::env::temp_dir().join(format!("pika_libraries_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("b/c")).unwrap();
        std::fs::write(dir.join("b/c/z.slang"), b"z").unwrap();
        std::fs::write(dir.join("a.slang"), b"a").unwrap();
        std::fs::write(dir.join("b/y.slang"), b"y").unwrap();
        assert_eq!(
            files(&dir).unwrap(),
            [
                dir.join("a.slang"),
                dir.join("b/c/z.slang"),
                dir.join("b/y.slang")
            ]
        );
        assert!(files(&dir.join("nowhere")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn key(
        bake: &Bake,
        source: &Path,
        space: &str,
        encoding: &'static str,
    ) -> Result<PathBuf, String> {
        let source = Source {
            node: String::new(),
            input: String::new(),
            file: source.to_str().unwrap().to_owned(),
            space: space.to_owned(),
            encoding,
            default: Vec::new(),
            transformed: false,
        };
        bake.texture_path(&source, &tiles(Path::new(&source.file))?)
    }

    #[test]
    fn texture_paths_mirror_the_source_and_key_its_state() {
        let dir = std::env::temp_dir().join(format!("pika_bake_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("assets")).unwrap();
        let source = dir.join("assets/rock.png");
        std::fs::write(&source, b"png").unwrap();
        let bake = |config: &str, texture, generate| Bake {
            materials: dir.join("materials"),
            textures: dir.join("textures"),
            config: config.into(),
            generate: Tool {
                path: GENERATE,
                hash: generate,
            },
            texture: Tool {
                path: TEXTURE,
                hash: texture,
            },
        };
        let path = key(&bake("", 1, 1), &source, "srgb_texture", "color").unwrap();
        // temp_dir may be a symlink; the mirror follows the real path.
        let real = std::fs::canonicalize(&source).unwrap();
        assert_eq!(
            path.parent().unwrap(),
            dir.join("textures").join(real.strip_prefix("/").unwrap())
        );
        assert_eq!(path.file_name().unwrap().len(), 16);
        assert_eq!(
            path,
            key(&bake("", 1, 1), &source, "srgb_texture", "color").unwrap()
        );
        // A rebuilt generate tool leaves the textures baked.
        assert_eq!(
            path,
            key(&bake("", 1, 2), &source, "srgb_texture", "color").unwrap()
        );
        for (space, encoding, config, tool) in [
            ("raw", "color", "", 1),
            ("srgb_texture", "data", "", 1),
            ("srgb_texture", "color", "studio.ocio", 1),
            ("srgb_texture", "color", "", 2),
        ] {
            assert_ne!(
                path,
                key(&bake(config, tool, 1), &source, space, encoding).unwrap()
            );
        }

        // Negative control: the same file saved again is another texture.
        let file = std::fs::File::options().write(true).open(&source).unwrap();
        file.set_modified(UNIX_EPOCH + std::time::Duration::from_secs(1_000_000))
            .unwrap();
        assert_ne!(
            path,
            key(&bake("", 1, 1), &source, "srgb_texture", "color").unwrap()
        );
        assert!(
            key(
                &bake("", 1, 1),
                Path::new("/nowhere/rock.png"),
                "raw",
                "data"
            )
            .is_err()
        );

        // A set is keyed on its tiles: one added or saved again misses.
        let pattern = dir.join("assets/wall.<UDIM>.png");
        let error = tiles(&pattern).unwrap_err();
        assert!(error.contains("wall.<UDIM>.png"), "{error}");
        std::fs::write(dir.join("assets/wall.1001.png"), b"png").unwrap();
        std::fs::write(dir.join("assets/wall.1012.png"), b"png").unwrap();
        std::fs::write(dir.join("assets/wall.1101.png"), b"png").unwrap();
        assert_eq!(
            tiles(&pattern).unwrap(),
            [
                (1001, dir.join("assets/wall.1001.png")),
                (1012, dir.join("assets/wall.1012.png"))
            ]
        );
        assert_eq!(tiles(&source).unwrap(), [(0, source.clone())]);
        let error = tiles(Path::new("")).unwrap_err();
        assert!(error.contains("did not resolve"), "{error}");
        let set = key(&bake("", 1, 1), &pattern, "raw", "data").unwrap();
        assert_eq!(
            set.parent().unwrap(),
            dir.join("textures")
                .join(real.parent().unwrap().strip_prefix("/").unwrap())
                .join("wall.<UDIM>.png")
        );
        std::fs::write(dir.join("assets/wall.1002.png"), b"png").unwrap();
        let added = key(&bake("", 1, 1), &pattern, "raw", "data").unwrap();
        assert_ne!(set, added);
        let file = std::fs::File::options()
            .write(true)
            .open(dir.join("assets/wall.1012.png"))
            .unwrap();
        file.set_modified(UNIX_EPOCH + std::time::Duration::from_secs(2_000_000))
            .unwrap();
        assert_ne!(
            added,
            key(&bake("", 1, 1), &pattern, "raw", "data").unwrap()
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn only_the_latest_job_lands() {
        let mut latest = Latest::default();
        let first = latest.submit(7);
        let second = latest.submit(7);
        assert!(!latest.landed(7, first), "the older edit's build is stale");
        assert_eq!(latest.jobs().len(), 1);
        assert!(latest.landed(7, second));
        assert!(latest.jobs().is_empty());
        assert!(!latest.landed(7, second), "a build lands once");

        let other = latest.submit(8);
        latest.jobs().remove(&8);
        assert!(
            !latest.landed(8, other),
            "a forgotten material's build is stale"
        );
        let third = latest.submit(9);
        assert!(current(&latest.jobs, 9, third));
        let fourth = latest.submit(9);
        assert!(
            !current(&latest.jobs, 9, third),
            "a resubmitted job is skipped"
        );
        assert!(current(&latest.jobs, 9, fourth));
    }

    #[test]
    fn a_frame_drains_the_latest_builds() {
        let network = material::clay();
        let mut builder = Builder::new(Bake {
            materials: "materials".into(),
            textures: "textures".into(),
            config: String::new(),
            generate: Tool {
                path: GENERATE,
                hash: 1,
            },
            texture: Tool {
                path: TEXTURE,
                hash: 1,
            },
        });
        let mut finished = |id: u64, stale: bool| {
            let generation = builder.latest.submit(id);
            if stale {
                builder.latest.submit(id);
            }
            builder.arrived.push(Built {
                id,
                generation,
                document: material::document(&network).unwrap().unwrap(),
                network: network.clone(),
                result: Err(String::new()),
            });
        };
        finished(1, false);
        finished(2, true);
        finished(3, false);
        let ids = |built: &[Built]| built.iter().map(|b| b.id).collect::<Vec<_>>();

        assert!(
            builder.wait(Duration::ZERO).unwrap(),
            "builds wait to be installed"
        );
        assert_eq!(ids(&builder.drain()), [1, 3], "the stale one does not land");
        assert_eq!(builder.len(), 1, "material 2's latest build is still out");
        assert!(builder.drain().is_empty());
        assert!(!builder.wait(Duration::ZERO).unwrap());
    }
}
