//! Slang shaders, compiled to SPIR-V by the `slangc` on `PATH`: the fixed
//! ones in `shaders/` when a renderer or check starts, and each material's
//! surface, whose SPIR-V the cache keeps.

use std::io::{Cursor, ErrorKind};
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use crate::cache;

// Read at run time from the source tree, so Pika must run from a checkout.
const SHADERS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");

const FLAGS: [&str; 5] = [
    "-target",
    "spirv",
    "-fvk-use-entrypoint-name",
    "-warnings-as-errors",
    "all",
];

/// A compiled shader, with entry points named as in its source.
pub struct Shader {
    pub slangc_version: String,
    pub spirv: Vec<u32>,
}

/// Compiles `shaders/<name>.slang`.
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

/// A material's surface: its SPIR-V and the layout slangc reflected.
pub struct Surface {
    pub shader: Shader,
    pub reflection: String,
    /// Whether slangc ran now rather than the cache answering.
    pub compiled: bool,
}

/// Compiles `<dir>/surface.slang`, the vertex stage written beside the
/// pixel stage it includes, unless `dir` holds the SPIR-V for these
/// sources, flags and slangc: the key hashes everything the SPIR-V depends
/// on, so a stale entry is impossible.
pub fn surface(dir: &Path) -> Result<Surface, String> {
    let read = |name: &str| {
        std::fs::read(dir.join(name))
            .map_err(|e| format!("cannot read {}/{name}: {e}", dir.display()))
    };
    let version = version()?;
    let sources = [
        read("surface.slang")?,
        read("pixel.slang")?,
        fixed("vertex.slang")?,
        fixed("view.slang")?,
    ];
    let flags = [FLAGS.join(" "), SURFACE_FLAGS.join(" ")].concat();
    let mut parts: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
    parts.extend([flags.as_bytes(), version.as_bytes()]);
    let key = format!("{:016x}", cache::fnv1a(&parts));
    let spv = dir.join(format!("{key}.spv"));
    let json = dir.join(format!("{key}.json"));
    let compiled = !(spv.is_file() && json.is_file());
    if compiled {
        let path = dir.join("surface.slang");
        // Both land under temporary names and are renamed into place, the
        // SPIR-V last, so another process sees a complete pair or none.
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
        std::fs::rename(&reflected, &json)
            .and_then(|()| std::fs::rename(&out, &spv))
            .map_err(|e| format!("cannot write {}: {e}", spv.display()))?;
    }
    let spirv = std::fs::read(&spv).map_err(|e| format!("cannot read {}: {e}", spv.display()))?;
    let reflection = std::fs::read_to_string(&json)
        .map_err(|e| format!("cannot read {}: {e}", json.display()))?;
    Ok(Surface {
        shader: Shader {
            slangc_version: version.to_owned(),
            spirv: words(&spirv, &spv.display().to_string())?,
        },
        reflection,
        compiled,
    })
}

/// A fixed shader's source, which the surface includes.
fn fixed(name: &str) -> Result<Vec<u8>, String> {
    std::fs::read(format!("{SHADERS}/{name}"))
        .map_err(|e| format!("cannot read {SHADERS}/{name}: {e}"))
}

// The generator declares textures inside its uniform block, and slangc
// warns that it moves them to bindings of their own (31106, 31107).
const SURFACE_FLAGS: [&str; 2] = ["-warnings-disable", "31106,31107"];

fn words(bytes: &[u8], path: &str) -> Result<Vec<u32>, String> {
    ash::util::read_spv(&mut Cursor::new(bytes))
        .map_err(|e| format!("slangc wrote invalid SPIR-V for {path}: {e}"))
}

/// slangc's version, asked once per process.
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
