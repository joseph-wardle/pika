//! The bake tool (`bake/bake.cpp`) as the renderer runs it: a material's
//! document in, the pixel stage Houdini's MaterialX generates and the
//! manifest describing it out; a texture as authored in, the file the
//! renderer uploads out (`texture`). Both are kept under the cache so
//! nothing is baked twice.

use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;

use crate::cache;
use crate::material::Document;

// Built by `make -C bake` from the checkout, like the shaders.
const TOOL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/target/bake/bake");
// The library files that shadow Houdini's, which the generator inlines.
const LIBRARIES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bake/libraries");

/// What the pixel stage expects of the renderer, as the tool lists it:
/// its public uniforms, textures and the varyings the vertex stage fills.
#[derive(Debug, Default, PartialEq)]
pub struct Manifest {
    pub materialx: String,
    pub uniforms: Vec<Uniform>,
    pub textures: Vec<String>,
    pub varyings: Vec<Varying>,
}

/// A public uniform: its variable, its MaterialX type and the initial
/// value the generator gave it, as MaterialX writes values ("0, 0, 1").
#[derive(Debug, PartialEq)]
pub struct Uniform {
    pub name: String,
    pub kind: String,
    pub initial: String,
}

#[derive(Debug, PartialEq)]
pub struct Varying {
    pub name: String,
    pub kind: String,
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
            "uniform" | "varying" => {
                let mut rest = rest.splitn(2, ' ');
                let name = rest.next().unwrap_or("").to_owned();
                if name.is_empty() {
                    return Err(format!("the manifest line {line:?} names no variable"));
                }
                let kind_name = second.to_owned();
                match kind {
                    "uniform" => m.uniforms.push(Uniform {
                        name,
                        kind: kind_name,
                        initial: rest.next().unwrap_or("").to_owned(),
                    }),
                    _ => m.varyings.push(Varying {
                        name,
                        kind: kind_name,
                    }),
                }
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
/// `pixel.slang` and `manifest`.
pub struct Generated {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

pub struct Bake {
    /// The cache's `materials` directory, one entry per document and tool.
    materials: PathBuf,
    /// The cache's `textures` directory, mirroring each source's path.
    textures: PathBuf,
    /// The host's OCIO config, or empty for OpenImageIO's default, through
    /// which the tool converts textures to the working space.
    config: String,
    /// A hash of the tool's own bytes and its library files: a rebuilt or
    /// edited tool regenerates everything.
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
        if !dir.join("manifest").is_file() {
            self.run(document, &dir)?;
        }
        let text = std::fs::read_to_string(dir.join("manifest"))
            .map_err(|e| format!("cannot read {}/manifest: {e}", dir.display()))?;
        Ok(Generated {
            manifest: manifest(&text)?,
            dir,
        })
    }

    /// Runs the tool into a directory beside `dir`, then moves it into
    /// place, so `dir` exists only once it is complete.
    fn run(&self, document: &Document, dir: &Path) -> Result<(), String> {
        let tmp = dir.with_extension(format!("tmp{}", std::process::id()));
        let result = run(document, &tmp).and_then(|()| match std::fs::rename(&tmp, dir) {
            Ok(()) => Ok(()),
            // Another process generated it meanwhile.
            Err(_) if dir.join("manifest").is_file() => Ok(()),
            Err(e) => Err(format!("cannot move {} into place: {e}", tmp.display())),
        });
        if result.is_err() || tmp.is_dir() {
            let _ = std::fs::remove_dir_all(&tmp);
        }
        result
    }

    /// Bakes the image `source`, or the UDIM set its `<UDIM>` pattern
    /// names, declared in the colour space `space`, with the levels above
    /// `cap` texels dropped, in one of the tool's encodings, unless the
    /// cache holds it; returns the baked file and whether it was baked now.
    pub fn texture(
        &self,
        source: &Path,
        space: &str,
        encoding: &str,
        cap: u32,
    ) -> Result<(PathBuf, bool), String> {
        let path = self.texture_path(source, space, encoding, cap)?;
        if path.is_file() {
            return Ok((path, false));
        }
        let dir = path.parent().unwrap();
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        let output = Command::new(TOOL)
            .arg("texture")
            .arg(source)
            .args([space, encoding, &cap.to_string(), &self.config])
            .arg(&tmp)
            .env_remove("LD_LIBRARY_PATH")
            .output()
            .map_err(|e| format!("could not run the bake tool {TOOL}: {e}"))?;
        if !output.status.success() {
            let _ = std::fs::remove_file(&tmp);
            return Err(failed(&output));
        }
        std::fs::rename(&tmp, &path)
            .map_err(|e| format!("cannot move {} into place: {e}", tmp.display()))?;
        Ok((path, true))
    }

    /// Where `source`'s bake lives: under `textures`, mirroring the source's
    /// own path so one asset's bakes can be found and deleted, named by a
    /// hash of each of its files' name, size and modification time, the
    /// parameters, the config and the tool, so an edited source, a tile
    /// added or a changed setting misses.
    pub fn texture_path(
        &self,
        source: &Path,
        space: &str,
        encoding: &str,
        cap: u32,
    ) -> Result<PathBuf, String> {
        let mut state = Vec::new();
        for file in tiles(source)? {
            let meta = std::fs::metadata(&file)
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

/// The files `source` names: itself, or for a `<UDIM>` pattern the tiles
/// 1001 to 1100 that exist, as USD reads a set. The tool lists them by the
/// same rule (`bake/texture.cpp`); the key here is what makes a bake miss.
pub fn tiles(source: &Path) -> Result<Vec<PathBuf>, String> {
    let text = source.to_string_lossy();
    let Some((before, after)) = text.split_once("<UDIM>") else {
        if !source.is_file() {
            return Err(format!("cannot find the texture {text}"));
        }
        return Ok(vec![source.to_owned()]);
    };
    let set: Vec<PathBuf> = (1001..=1100)
        .map(|number| PathBuf::from(format!("{before}{number}{after}")))
        .filter(|tile| tile.is_file())
        .collect();
    if set.is_empty() {
        return Err(format!(
            "no tiles of {text} exist: UDIM tiles are numbered 1001 to 1100"
        ));
    }
    Ok(set)
}

fn run(document: &Document, dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let path = dir.join("document.mtlx");
    std::fs::write(&path, &document.text)
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    // A Houdini 21 host's library path holds an older MaterialX of the
    // same soname; the tool's rpath finds Houdini 22's without it.
    let output = Command::new(TOOL)
        .arg("generate")
        .arg(&path)
        .arg(dir)
        .env_remove("LD_LIBRARY_PATH")
        .output()
        .map_err(|e| format!("could not run the bake tool {TOOL}: {e}"))?;
    if !output.status.success() {
        return Err(failed(&output));
    }
    Ok(())
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
        assert_eq!(
            m.varyings[0],
            Varying {
                name: "i_geomprop_st".into(),
                kind: "vector2".into(),
            }
        );
        assert_eq!(m.varyings.len(), 5);

        assert!(manifest("uniform float").is_err());
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
        let path = bake("", 1)
            .texture_path(&source, "srgb_texture", "color", 4096)
            .unwrap();
        // temp_dir may be a symlink; the mirror follows the real path.
        let real = std::fs::canonicalize(&source).unwrap();
        assert_eq!(
            path.parent().unwrap(),
            dir.join("textures").join(real.strip_prefix("/").unwrap())
        );
        assert_eq!(path.file_name().unwrap().len(), 16);
        assert_eq!(
            path,
            bake("", 1)
                .texture_path(&source, "srgb_texture", "color", 4096)
                .unwrap()
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
                bake(config, tool)
                    .texture_path(&source, space, encoding, cap)
                    .unwrap()
            );
        }

        // The control: the same file saved again is another texture.
        let file = std::fs::File::options().write(true).open(&source).unwrap();
        file.set_modified(UNIX_EPOCH + std::time::Duration::from_secs(1_000_000))
            .unwrap();
        assert_ne!(
            path,
            bake("", 1)
                .texture_path(&source, "srgb_texture", "color", 4096)
                .unwrap()
        );
        assert!(
            bake("", 1)
                .texture_path(Path::new("/nowhere/rock.png"), "raw", "data", 4096)
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
                dir.join("assets/wall.1001.png"),
                dir.join("assets/wall.1012.png")
            ]
        );
        let set = bake("", 1)
            .texture_path(&pattern, "raw", "data", 4096)
            .unwrap();
        assert_eq!(
            set.parent().unwrap(),
            dir.join("textures")
                .join(real.parent().unwrap().strip_prefix("/").unwrap())
                .join("wall.<UDIM>.png")
        );
        std::fs::write(dir.join("assets/wall.1002.png"), b"png").unwrap();
        let added = bake("", 1)
            .texture_path(&pattern, "raw", "data", 4096)
            .unwrap();
        assert_ne!(set, added);
        let file = std::fs::File::options()
            .write(true)
            .open(dir.join("assets/wall.1012.png"))
            .unwrap();
        file.set_modified(UNIX_EPOCH + std::time::Duration::from_secs(2_000_000))
            .unwrap();
        assert_ne!(
            added,
            bake("", 1)
                .texture_path(&pattern, "raw", "data", 4096)
                .unwrap()
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
