//! Still accumulation: while nothing changes, each frame draws the next sample of the soft
//! shadows' pattern and shows the mean of the frames so far, until `SAMPLES`. Only shadows
//! vary between samples, so the mean of the frames is the frame of the shadows' mean.

use ash::vk;

use crate::gpu::{self, Buffer, Memory};
use crate::light::{self, Light};
use crate::target::Target;

/// Samples a still view converges at, as EEVEE renders take: shading is linear in each
/// light's visibility, so the mean of 64 frames is each penumbra's mean of 64 rays.
pub const SAMPLES: u32 = 64;

pub struct Accumulation {
    compute: gpu::Compute,
    /// Allocated by the first still frame, so a view with no soft shadow holds none.
    sum: Option<Sum>,
    /// Frames the sum holds, drawn while nothing changed; the next frame's sample.
    pub drawn: u32,
    /// What the last frame's lights converge at (`samples`).
    pub wanted: u32,
}

struct Sum {
    /// Per pixel, four floats: the color of each frame since the view was last still.
    colors: Buffer,
    /// The color attachment's pixels, copied out for the sum and back.
    copy: Buffer,
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
        Ok(Self {
            compute,
            sum: None,
            drawn: 0,
            wanted: 0,
        })
    }

    pub fn converged(&self) -> bool {
        self.drawn >= self.wanted
    }

    /// Records outside rendering, after a still frame's shading: adds the color attachment to
    /// the sum, which sample 0 starts, and leaves the mean in the attachment.
    pub fn record(
        &mut self,
        device: &gpu::Device,
        cb: vk::CommandBuffer,
        target: &Target,
    ) -> Result<(), String> {
        let pixels = u64::from(target.width * target.height);
        let bytes = pixels * target.color_bytes;
        let sum = match &mut self.sum {
            Some(s) if s.copy.size == bytes && s.colors.size == pixels * 16 => s,
            slot => {
                // Free the old sum before allocating its replacement.
                *slot = None;
                let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
                let copied = storage
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST;
                slot.insert(Sum {
                    colors: Buffer::device_local(device, pixels * 16, storage)?,
                    copy: Buffer::device_local(device, bytes, copied)?,
                })
            }
        };
        target.copy_out(cb, 0, sum.copy.raw);
        let sample = self.drawn;
        let words = (target.color_bytes / 4) as u32;
        let all = 0..target.width * target.height;
        (self.compute).record(cb, &[all], |first, count, stride| Push {
            sum: sum.colors.address,
            color: sum.copy.address,
            first,
            count,
            stride,
            sample,
            words,
            unused: 0,
        });
        target.copy_in(cb, 0, sum.copy.raw);
        self.drawn += 1;
        Ok(())
    }

    pub fn memory(&self) -> Memory {
        let mut memory = Memory::default();
        if let Some(s) = &self.sum {
            memory += s.colors.memory();
            memory += s.copy.memory();
        }
        memory
    }
}
