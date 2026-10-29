//! Slang shaders, compiled to SPIR-V by the `slangc` on `PATH` when a
//! renderer or check starts.

use std::io::{Cursor, ErrorKind};
use std::process::Command;

// Read at run time from the source tree, so Pika must run from a checkout.
const SHADERS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");

/// A compiled shader, with entry points named as in its source.
pub struct Shader {
    pub slangc_version: String,
    pub spirv: Vec<u32>,
}

/// Compiles `shaders/<name>.slang`.
pub fn compile(name: &str) -> Result<Shader, String> {
    let version = slangc(&["-v"])?;
    // slangc prints its version to stderr.
    let slangc_version = String::from_utf8_lossy(&version.stderr).trim().to_owned();

    let path = format!("{SHADERS}/{name}.slang");
    let output = slangc(&[
        &path,
        "-target",
        "spirv",
        "-fvk-use-entrypoint-name",
        "-warnings-as-errors",
        "all",
    ])?;
    if !output.status.success() {
        return Err(format!(
            "slangc failed on {path}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let spirv = ash::util::read_spv(&mut Cursor::new(&output.stdout))
        .map_err(|e| format!("slangc wrote invalid SPIR-V for {path}: {e}"))?;
    Ok(Shader {
        slangc_version,
        spirv,
    })
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
