//! Paused accumulation: while nothing changes, each frame draws the next sample of the soft
//! shadows' pattern and shows the mean of the frames so far, until `SAMPLES`. Only shadows
//! vary between samples, so the mean of the frames is the frame of the shadows' mean.

use ash::vk;

use crate::gpu;
use crate::light::{self, Light};
use crate::target::Target;

/// Samples a still view converges at, as EEVEE renders take: shading is linear in each
/// light's visibility, so the mean of 64 frames is each penumbra's mean of 64 rays.
pub const SAMPLES: u32 = 64;

pub struct Accumulation {
    compute: gpu::Compute,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    sum: vk::DeviceAddress,
    color: vk::DeviceAddress,
    first: u32,
    count: u32,
    stride: u32,
    sample: u32,
    words: u32,
    unused: u32,
}

/// The samples a still view needs: `SAMPLES` when some light's shadow is soft (as
/// `shaders/frame.slang`'s `visibility` tells), else 0, since every frame would be the same.
pub fn samples(lights: &[Light]) -> u32 {
    let soft = |l: &Light| {
        l.shadow != light::NO_SHADOW
            && (l.radius > 0.0 || l.shape == light::RECT || l.shape == light::DISK)
    };
    match lights.iter().any(soft) {
        true => SAMPLES,
        false => 0,
    }
}

impl Accumulation {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
        let compute = gpu::Compute::new(device, spirv, c"cs", size_of::<Push>())?;
        Ok(Self { compute })
    }

    /// Records outside rendering, after shading: adds the color attachment to the target's
    /// sum, which `sample` 0 starts, and leaves the mean in the attachment.
    pub fn record(&self, cb: vk::CommandBuffer, target: &Target, sample: u32) {
        target.copy_out(cb, 0, target.copy.raw);
        let pixels = target.width * target.height;
        let words = (target.copy.size / u64::from(pixels) / 4) as u32;
        let all = 0..target.width * target.height;
        (self.compute).record(cb, &[all], |first, count, stride| Push {
            sum: target.sum.address,
            color: target.copy.address,
            first,
            count,
            stride,
            sample,
            words,
            unused: 0,
        });
        target.copy_in(cb, 0, target.copy.raw);
    }
}
