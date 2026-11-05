//! The bake tool (`bake/bake.cpp`) as the renderer runs it: a material's
//! document in, the pixel stage Houdini's MaterialX generates, the vertex
//! stage feeding it and the manifest describing them out; a texture as
//! authored in, the file the renderer uploads out (`texture`). Both are
//! kept under the cache so nothing is baked twice.

use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;

use crate::cache;
use crate::material::Document;
use crate::texture::{self, Texture};

// Built by `make -C bake` from the checkout, like the shaders.
const TOOL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/target/bake/bake");
// The library files that shadow Houdini's, which the generator inlines.
const LIBRARIES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bake/libraries");

/// What the pixel stage expects of the renderer, as the tool lists it:
/// its public uniforms and textures.
#[derive(Debug, Default, PartialEq)]
pub struct Manifest {
    pub materialx: String,
    pub uniforms: Vec<Uniform>,
    pub textures: Vec<String>,
}

/// A public uniform: its variable, its MaterialX type and the initial
/// value the generator gave it, as MaterialX writes values ("0, 0, 1").
#[derive(Debug, PartialEq)]
pub struct Uniform {
    pub name: String,
    pub kind: String,
    pub initial: String,
}

pub fn manifest(text: &str) -> Result<Manifest, String> {
    let mut m = Manifest::default();
    for line in text.lines() {
        let mut words = line.splitn(3, ' ');
        let (Some(kind), Some(second)) = (words.next(), words.next()) else {
            return Err(format!("the manifest line {line:?} has no value"));
        };
        let rest = words.next().unwrap_or("");
        match kind {
            "materialx" => m.materialx = second.to_owned(),
            "texture" => m.textures.push(second.to_owned()),
            "uniform" => {
                let mut rest = rest.splitn(2, ' ');
                let name = rest.next().unwrap_or("").to_owned();
                if name.is_empty() {
                    return Err(format!("the manifest line {line:?} names no variable"));
                }
                m.uniforms.push(Uniform {
                    name,
                    kind: second.to_owned(),
                    initial: rest.next().unwrap_or("").to_owned(),
                });
            }
            _ => {
                return Err(format!(
                    "the manifest line {line:?} is not one the renderer knows"
                ));
            }
        }
    }
    Ok(m)
}

/// The tool's output for one document: the directory holding its
/// `pixel.slang`, `surface.slang` and `manifest`.
pub struct Generated {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

/// A texture as the renderer uploads it, read from the cache.
pub struct Baked {
    pub path: PathBuf,
    /// The image, or UDIM pattern, it was baked from.
    pub source: PathBuf,
    pub texture: Texture,
    /// Whether the bake tool ran now rather than the cache answering.
    pub fresh: bool,
}

#[derive(Clone)]
pub struct Bake {
    /// The cache's `materials` directory, one entry per document and tool.
    materials: PathBuf,
    /// The cache's `textures` directory, mirroring each source's path.
    textures: PathBuf,
    /// The host's OCIO config, or empty for OpenImageIO's default, through
    /// which the tool converts textures to the working space.
    config: String,
    /// A hash of the tool's own bytes, its library files and the listing
    /// it generates from (`bake libraries`): a rebuilt or edited tool, or a
    /// Houdini update beneath it, regenerates and re-bakes everything.
    tool: u64,
}

impl Bake {
    pub fn new() -> Result<Self, String> {
        let mut bytes = std::fs::read(TOOL).map_err(|e| {
            format!("cannot read the bake tool {TOOL}: {e}. Build it with `make -C bake`.")
        })?;
        for file in files(Path::new(LIBRARIES))? {
            bytes.extend_from_slice(file.as_os_str().as_encoded_bytes());
            bytes.extend(
                std::fs::read(&file).map_err(|e| format!("cannot read {}: {e}", file.display()))?,
            );
        }
        let output = tool()
            .arg("libraries")
            .output()
            .map_err(|e| format!("could not run the bake tool {TOOL}: {e}"))?;
        if !output.status.success() {
            return Err(failed(&output));
        }
        bytes.extend_from_slice(&output.stdout);
        let root = cache::root()?;
        let (materials, textures) = (root.join("materials"), root.join("textures"));
        for dir in [&materials, &textures] {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("cannot create the cache {}: {e}", dir.display()))?;
        }
        Ok(Self {
            materials,
            textures,
            config: std::env::var("OCIO").unwrap_or_default(),
            tool: cache::fnv1a(&[&bytes]),
        })
    }

    /// Generates `document`'s pixel stage, unless the cache holds it.
    pub fn generate(&self, document: &Document) -> Result<Generated, String> {
        let key = cache::fnv1a(&[document.text.as_bytes(), &self.tool.to_le_bytes()]);
        let dir = self.materials.join(format!("{key:016x}"));
        match generated(&dir) {
            Ok(manifest) => return Ok(Generated { manifest, dir }),
            // Left partial by a run that never finished: the tool runs into
            // a directory with nothing left of the old.
            Err(why) if dir.exists() => cache::discard(&dir, &why)?,
            Err(_) => {}
        }
        // The tool runs into a directory beside `dir`, moved into place
        // once complete, so `dir` exists only whole; a run that fails, or
        // that another process beat to it, leaves nothing behind.
        let tmp = dir.with_extension(cache::unique());
        let path = tmp.join("document.mtlx");
        let run = || -> Result<(), String> {
            std::fs::create_dir_all(&tmp)
                .map_err(|e| format!("cannot create {}: {e}", tmp.display()))?;
            std::fs::write(&path, &document.text)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            let output = (tool().arg("generate").arg(&path).arg(&tmp).output())
                .map_err(|e| format!("could not run the bake tool {TOOL}: {e}"))?;
            if !output.status.success() {
                return Err(failed(&output));
            }
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

    /// Bakes the image `source`, or the UDIM set its `<UDIM>` pattern
    /// names, declared in the colour space `space`, with the levels above
    /// `cap` texels dropped, in one of the tool's encodings, unless the
    /// cache holds it, and reads it.
    pub fn texture(
        &self,
        source: &Path,
        space: &str,
        encoding: &str,
        cap: u32,
    ) -> Result<Baked, String> {
        let set = tiles(source)?;
        let path = self.texture_path(source, &set, space, encoding, cap)?;
        if path.is_file() {
            match texture::read(&path) {
                Ok(texture) => {
                    return Ok(Baked {
                        path,
                        source: source.to_owned(),
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
        let mut command = tool();
        command
            .arg("texture")
            .args([space, encoding, &cap.to_string(), &self.config])
            .arg(&tmp);
        for (number, tile) in &set {
            command.arg(number.to_string()).arg(tile);
        }
        let output =
            (command.output()).map_err(|e| format!("could not run the bake tool {TOOL}: {e}"))?;
        if !output.status.success() {
            let _ = std::fs::remove_file(&tmp);
            return Err(failed(&output));
        }
        std::fs::rename(&tmp, &path)
            .map_err(|e| format!("cannot move {} into place: {e}", tmp.display()))?;
        Ok(Baked {
            texture: texture::read(&path)?,
            path,
            source: source.to_owned(),
            fresh: true,
        })
    }

    /// Where `source`'s bake lives, its files being `set` (`tiles`): under
    /// `textures`, mirroring the source's own path so one asset's bakes can
    /// be found and deleted, named by a hash of each file's name, size and
    /// modification time, the parameters, the config and the tool, so an
    /// edited source, a tile added or a changed setting misses.
    pub fn texture_path(
        &self,
        source: &Path,
        set: &[(u32, PathBuf)],
        space: &str,
        encoding: &str,
        cap: u32,
    ) -> Result<PathBuf, String> {
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
            space.as_bytes(),
            b"\0",
            encoding.as_bytes(),
            b"\0",
            self.config.as_bytes(),
            b"\0",
            &cap.to_le_bytes(),
            &self.tool.to_le_bytes(),
        ]);
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

/// Every file under `dir`, in path order.
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

/// The files `source` names, each with its tile number: itself, numbered
/// 0, or for a `<UDIM>` pattern the tiles 1001 to 1100 that exist, as USD
/// reads a set. The tool bakes what it is given; the key here is what
/// makes a bake miss.
pub fn tiles(source: &Path) -> Result<Vec<(u32, PathBuf)>, String> {
    let text = source.to_string_lossy();
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

/// What the entry `dir` holds: its manifest, with the pixel stage beside it.
fn generated(dir: &Path) -> Result<Manifest, String> {
    let path = dir.join("manifest");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    for name in ["pixel.slang", "surface.slang"] {
        if !dir.join(name).is_file() {
            return Err(format!("{} has no {name}", dir.display()));
        }
    }
    manifest(&text)
}

/// The tool, ready for its job's arguments.
fn tool() -> Command {
    let mut command = Command::new(TOOL);
    // A Houdini 21 host's library path holds an older MaterialX of the
    // same soname; the tool's rpath finds Houdini 22's without it.
    command.env_remove("LD_LIBRARY_PATH");
    command
}

/// The tool's complaint, on one line so a warning quoting it stays one.
fn failed(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    format!(
        "the bake tool failed: {}",
        stderr.trim().replace('\n', "; ")
    )
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// The bake tool wrote this for bake/fixtures/network.mtlx.
    pub const FIXTURE: &str = include_str!("../bake/fixtures/network.manifest");

    #[test]
    fn reads_the_fixture_manifest() {
        let m = manifest(FIXTURE).unwrap();
        assert_eq!(m.materialx, "1.39.5");
        assert_eq!(m.textures, ["n1_file"]);
        assert_eq!(m.uniforms.len(), 21);
        assert_eq!(
            m.uniforms[4],
            Uniform {
                name: "n1_fallback".into(),
                kind: "color4".into(),
                initial: "0, 0, 0, 1".into(),
            }
        );
        assert_eq!(m.uniforms[0].initial, "", "a struct uniform has no value");

        assert!(manifest("uniform float").is_err());
        assert!(manifest("varying vector2 i_geomprop_st").is_err());
        assert!(manifest("sampler n1_file").is_err());
    }

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
        encoding: &str,
        cap: u32,
    ) -> Result<PathBuf, String> {
        bake.texture_path(source, &tiles(source)?, space, encoding, cap)
    }

    #[test]
    fn texture_paths_mirror_the_source_and_key_its_state() {
        let dir = std::env::temp_dir().join(format!("pika_bake_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("assets")).unwrap();
        let source = dir.join("assets/rock.png");
        std::fs::write(&source, b"png").unwrap();
        let bake = |config: &str, tool| Bake {
            materials: dir.join("materials"),
            textures: dir.join("textures"),
            config: config.into(),
            tool,
        };
        let path = key(&bake("", 1), &source, "srgb_texture", "color", 4096).unwrap();
        // temp_dir may be a symlink; the mirror follows the real path.
        let real = std::fs::canonicalize(&source).unwrap();
        assert_eq!(
            path.parent().unwrap(),
            dir.join("textures").join(real.strip_prefix("/").unwrap())
        );
        assert_eq!(path.file_name().unwrap().len(), 16);
        assert_eq!(
            path,
            key(&bake("", 1), &source, "srgb_texture", "color", 4096).unwrap()
        );
        for (space, encoding, cap, config, tool) in [
            ("raw", "color", 4096, "", 1),
            ("srgb_texture", "data", 4096, "", 1),
            ("srgb_texture", "color", 1024, "", 1),
            ("srgb_texture", "color", 4096, "studio.ocio", 1),
            ("srgb_texture", "color", 4096, "", 2),
        ] {
            assert_ne!(
                path,
                key(&bake(config, tool), &source, space, encoding, cap).unwrap()
            );
        }

        // The control: the same file saved again is another texture.
        let file = std::fs::File::options().write(true).open(&source).unwrap();
        file.set_modified(UNIX_EPOCH + std::time::Duration::from_secs(1_000_000))
            .unwrap();
        assert_ne!(
            path,
            key(&bake("", 1), &source, "srgb_texture", "color", 4096).unwrap()
        );
        assert!(
            key(
                &bake("", 1),
                Path::new("/nowhere/rock.png"),
                "raw",
                "data",
                4096
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
        let set = key(&bake("", 1), &pattern, "raw", "data", 4096).unwrap();
        assert_eq!(
            set.parent().unwrap(),
            dir.join("textures")
                .join(real.parent().unwrap().strip_prefix("/").unwrap())
                .join("wall.<UDIM>.png")
        );
        std::fs::write(dir.join("assets/wall.1002.png"), b"png").unwrap();
        let added = key(&bake("", 1), &pattern, "raw", "data", 4096).unwrap();
        assert_ne!(set, added);
        let file = std::fs::File::options()
            .write(true)
            .open(dir.join("assets/wall.1012.png"))
            .unwrap();
        file.set_modified(UNIX_EPOCH + std::time::Duration::from_secs(2_000_000))
            .unwrap();
        assert_ne!(
            added,
            key(&bake("", 1), &pattern, "raw", "data", 4096).unwrap()
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
