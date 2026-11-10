//! Slang shaders compiled by the `slangc` on `PATH`; material surfaces are cached.

use std::io::{Cursor, ErrorKind};
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use crate::cache;
use crate::json;

// Read at run time from the source tree, so Pika must run from a checkout.
const SHADERS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");

const FLAGS: [&str; 5] = [
    "-target",
    "spirv",
    "-fvk-use-entrypoint-name",
    "-warnings-as-errors",
    "all",
];

pub struct Shader {
    pub slangc_version: String,
    pub spirv: Vec<u32>,
}

pub fn compile(name: &str) -> Result<Shader, String> {
    let path = format!("{SHADERS}/{name}.slang");
    let output = slangc(&[&[path.as_str()], &FLAGS[..]].concat())?;
    if !output.status.success() {
        return Err(format!(
            "slangc failed on {path}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(Shader {
        slangc_version: version()?.to_owned(),
        spirv: words(&output.stdout, &path)?,
    })
}

pub struct Compiled {
    pub shader: Shader,
    pub reflection: String,
    /// slangc ran rather than the cache answering.
    pub fresh: bool,
}

/// Compiles `<dir>/surface.slang`, keyed on every input its SPIR-V depends on.
pub fn surface(dir: &Path) -> Result<Compiled, String> {
    let read = |name: &str| {
        std::fs::read(dir.join(name))
            .map_err(|e| format!("cannot read {}/{name}: {e}", dir.display()))
    };
    let version = version()?;
    let sources = [
        read("surface.slang")?,
        read("pixel.slang")?,
        fixed("frame.slang")?,
        fixed("vertex.slang")?,
        fixed("view.slang")?,
    ];
    let flags = [FLAGS.join(" "), SURFACE_FLAGS.join(" ")].concat();
    let mut parts: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    parts.extend([flags.as_bytes(), version.as_bytes()]);
    let key = format!("{:016x}", cache::fnv1a(&parts));
    let spv = dir.join(format!("{key}.spv"));
    let json = dir.join(format!("{key}.json"));
    let cached = compiled(&spv, &json);
    let fresh = cached.is_err();
    let (spirv, reflection) = match cached {
        Ok(pair) => pair,
        Err(why) => {
            // Absent, or cut short by a write that never finished.
            if spv.exists() {
                cache::discard(&spv, &why)?;
            }
            compile_surface(dir, &key, &spv, &json)?;
            compiled(&spv, &json)?
        }
    };
    Ok(Compiled {
        shader: Shader {
            slangc_version: version.to_owned(),
            spirv,
        },
        reflection,
        fresh,
    })
}

fn compiled(spv: &Path, json: &Path) -> Result<(Vec<u32>, String), String> {
    let spirv = std::fs::read(spv).map_err(|e| format!("cannot read {}: {e}", spv.display()))?;
    let reflection = std::fs::read_to_string(json)
        .map_err(|e| format!("cannot read {}: {e}", json.display()))?;
    json::parse(&reflection).map_err(|e| format!("{}: {e}", json.display()))?;
    Ok((words(&spirv, &spv.display().to_string())?, reflection))
}

fn compile_surface(dir: &Path, key: &str, spv: &Path, json: &Path) -> Result<(), String> {
    let path = dir.join("surface.slang");
    // Renamed into place, the SPIR-V last, so others see a whole pair or none.
    let unique = cache::unique();
    let tmp = |name: &str| dir.join(format!("{key}.{unique}.{name}"));
    let (out, reflected) = (tmp("spv"), tmp("json"));
    let mut args = vec![
        path.to_str().unwrap(),
        "-I",
        SHADERS,
        "-o",
        out.to_str().unwrap(),
    ];
    args.extend(FLAGS);
    args.extend(SURFACE_FLAGS);
    args.extend(["-reflection-json", reflected.to_str().unwrap()]);
    let output = slangc(&args)?;
    if !output.status.success() {
        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(&reflected);
        return Err(format!(
            "slangc failed on {}:\n{}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    std::fs::rename(&reflected, json)
        .and_then(|()| std::fs::rename(&out, spv))
        .map_err(|e| format!("cannot write {}: {e}", spv.display()))
}

fn fixed(name: &str) -> Result<Vec<u8>, String> {
    std::fs::read(format!("{SHADERS}/{name}"))
        .map_err(|e| format!("cannot read {SHADERS}/{name}: {e}"))
}

// The generator declares textures in its uniform block, which slangc warns
// of. bake/check.sh passes the same flags.
const SURFACE_FLAGS: [&str; 2] = ["-warnings-disable", "31106,31107"];

/// `read_spv` checks only the magic, which an entry cut short passes, to crash the
/// driver: the instructions must end on the last word, with OpFunctionEnd.
fn words(bytes: &[u8], path: &str) -> Result<Vec<u32>, String> {
    let words = ash::util::read_spv(&mut Cursor::new(bytes))
        .map_err(|e| format!("{path} is not SPIR-V: {e}"))?;
    const HEADER: usize = 5;
    const FUNCTION_END: u32 = 56;
    let (mut at, mut last) = (HEADER, 0);
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 {
            break;
        }
        last = words[at] & 0xffff;
        at += count;
    }
    if at != words.len() || last != FUNCTION_END {
        return Err(format!("{path} is cut short"));
    }
    Ok(words)
}

fn version() -> Result<&'static str, String> {
    static VERSION: OnceLock<Result<String, String>> = OnceLock::new();
    VERSION
        .get_or_init(|| {
            // slangc prints its version to stderr.
            slangc(&["-v"]).map(|o| String::from_utf8_lossy(&o.stderr).trim().to_owned())
        })
        .as_deref()
        .map_err(Clone::clone)
}

fn slangc(args: &[&str]) -> Result<std::process::Output, String> {
    Command::new("slangc").args(args).output().map_err(|e| {
        if e.kind() == ErrorKind::NotFound {
            "slangc is not on PATH. Install Slang 2026.9.2 \
             (https://github.com/shader-slang/slang/releases) and put its bin/ on PATH."
                .into()
        } else {
            format!("could not run slangc: {e}")
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn spirv_must_be_whole() {
        // The header, OpCapability Shader, OpFunctionEnd.
        let module = [
            0x0723_0203,
            0x0001_0000,
            0,
            1,
            0,
            0x0002_0011,
            1,
            0x0001_0038,
        ];
        assert_eq!(words(&bytes(&module), "m").unwrap(), module);
        // Cut mid-instruction, before the last one, and after the header.
        for cut in [6, 7, 5] {
            let error = words(&bytes(&module[..cut]), "m").unwrap_err();
            assert!(error.contains("cut short"), "{cut}: {error}");
        }
        assert!(words(&bytes(&module)[..30], "m").is_err());
        assert!(words(b"not spirv at all", "m").is_err());
    }
}
