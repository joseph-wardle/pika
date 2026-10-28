//! The first-frame check: draws a known rectangle offscreen and compares the
//! read-back pixels with the image it must produce.

use std::io::{Cursor, ErrorKind};
use std::process::Command;

// Read at run time from the source tree, so `cargo run` must run from a checkout.
const SHADER_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/check.slang");

/// `check.slang` compiled to SPIR-V, with entry points `vs` and `fs`.
pub struct Shader {
    pub slangc_version: String,
    pub spirv: Vec<u32>,
}

/// Compiles `check.slang` with the `slangc` on `PATH`.
pub fn compile() -> Result<Shader, String> {
    let version = slangc(&["-v"])?;
    // slangc prints its version to stderr.
    let slangc_version = String::from_utf8_lossy(&version.stderr).trim().to_owned();

    let output = slangc(&[
        SHADER_PATH,
        "-target",
        "spirv",
        "-fvk-use-entrypoint-name",
        "-warnings-as-errors",
        "all",
    ])?;
    if !output.status.success() {
        return Err(format!(
            "slangc failed on {SHADER_PATH}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let spirv = ash::util::read_spv(&mut Cursor::new(&output.stdout))
        .map_err(|e| format!("slangc wrote invalid SPIR-V for {SHADER_PATH}: {e}"))?;
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
