//! The bake tool (`bake/bake.cpp`) as the renderer runs it: a material's
//! document in, the pixel stage Houdini's MaterialX generates and the
//! manifest describing it out, kept under the cache so no document is
//! generated twice.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cache;
use crate::material::Document;

// Built by `make -C bake` from the checkout, like the shaders.
const TOOL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/target/bake/bake");

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
    /// A hash of the tool's own bytes: a rebuilt tool regenerates everything.
    tool: u64,
}

impl Bake {
    pub fn new() -> Result<Self, String> {
        let bytes = std::fs::read(TOOL).map_err(|e| {
            format!("cannot read the bake tool {TOOL}: {e}. Build it with `make -C bake`.")
        })?;
        let materials = cache::root()?.join("materials");
        std::fs::create_dir_all(&materials)
            .map_err(|e| format!("cannot create the cache {}: {e}", materials.display()))?;
        Ok(Self {
            materials,
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
        return Err(format!(
            "the bake tool failed:\n{}",
            String::from_utf8_lossy(&output.stderr).trim_end()
        ));
    }
    Ok(())
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
}
