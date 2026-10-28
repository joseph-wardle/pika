//! The first-frame check: draws a known rectangle offscreen and compares the
//! read-back pixels with the image it must produce.

use std::io::{Cursor, ErrorKind};
use std::process::Command;

/// Width and height of the target. A power of two keeps the shader's pixel to
/// NDC mapping exact, and positions must fit the 8-bit channels.
#[cfg_attr(not(test), expect(dead_code, reason = "the GPU path will use it"))]
pub const SIZE: i32 = 64;

/// The drawn rectangle in pixels, `[x0, y0, x1, y1]` with exclusive ends.
/// Off-centre and not square, so a flipped or transposed image fails.
#[cfg_attr(not(test), expect(dead_code, reason = "the GPU path will use it"))]
pub const RECT: [i32; 4] = [16, 8, 56, 40];

/// `RECT` moved 1 px right, the negative control. Because each pixel holds its
/// own position, only the two edge columns differ: 2 × height pixels.
#[cfg_attr(not(test), expect(dead_code, reason = "the GPU path will use it"))]
pub const SHIFTED: [i32; 4] = [17, 8, 57, 40];

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

/// The image a draw of `rect` must produce: tightly packed RGBA8_UINT rows,
/// `[x, y, 1, 255]` inside the rectangle and zero outside.
#[cfg_attr(not(test), expect(dead_code, reason = "the GPU path will use it"))]
pub fn expected(rect: [i32; 4]) -> Vec<u8> {
    let [x0, y0, x1, y1] = rect;
    let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let inside = (x0..x1).contains(&x) && (y0..y1).contains(&y);
            let pixel = if inside {
                [x as u8, y as u8, 1, 255]
            } else {
                [0; 4]
            };
            pixels.extend_from_slice(&pixel);
        }
    }
    pixels
}

/// How many RGBA8 pixels differ between two images of the same size.
#[cfg_attr(not(test), expect(dead_code, reason = "the GPU path will use it"))]
pub fn mismatches(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len(), "images differ in size");
    let (a, b) = (a.as_chunks::<4>().0, b.as_chunks::<4>().0);
    a.iter().zip(b).filter(|(p, q)| p != q).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(image: &[u8], x: i32, y: i32) -> &[u8] {
        let i = ((y * SIZE + x) * 4) as usize;
        &image[i..i + 4]
    }

    #[test]
    fn expected_covers_rect_with_exclusive_ends() {
        let image = expected(RECT);
        assert_eq!(image.len(), (SIZE * SIZE * 4) as usize);
        assert_eq!(pixel(&image, 16, 8), [16, 8, 1, 255]);
        assert_eq!(pixel(&image, 55, 39), [55, 39, 1, 255]);
        assert_eq!(pixel(&image, 15, 8), [0; 4]);
        assert_eq!(pixel(&image, 16, 7), [0; 4]);
        assert_eq!(pixel(&image, 56, 39), [0; 4]);
        assert_eq!(pixel(&image, 55, 40), [0; 4]);
    }

    #[test]
    fn identical_images_match() {
        assert_eq!(mismatches(&expected(RECT), &expected(RECT)), 0);
    }

    #[test]
    fn shifted_rect_differs_in_two_columns() {
        let height = (RECT[3] - RECT[1]) as usize;
        assert_eq!(mismatches(&expected(RECT), &expected(SHIFTED)), 2 * height);
    }

    #[test]
    fn unwritten_readback_fails_every_pixel() {
        // The GPU path prefills the readback buffer with 0xCD before each draw.
        let unwritten = vec![0xCD; (SIZE * SIZE * 4) as usize];
        assert_eq!(
            mismatches(&expected(RECT), &unwritten),
            (SIZE * SIZE) as usize
        );
    }
}
