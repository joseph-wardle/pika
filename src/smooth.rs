//! The smooth normal pass: a compute pipeline writing each point's normal
//! from the faces around it, for meshes whose normals are not authored.

use std::ops::Range;

use ash::vk;

use crate::gpu;

pub struct Smooth {
    compute: gpu::Compute,
}

/// The buffers `shaders/smooth.slang` reads and writes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Buffers {
    pub points: vk::DeviceAddress,
    pub offsets: vk::DeviceAddress,
    pub corners: vk::DeviceAddress,
    pub normals: vk::DeviceAddress,
}

/// `shaders/smooth.slang`'s push constants.
#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    buffers: Buffers,
    first: u32,
    count: u32,
    stride: u32,
    unused: u32,
}

impl Smooth {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
        let compute = gpu::Compute::new(device, spirv, size_of::<Push>())?;
        Ok(Self { compute })
    }

    /// Records into `cb`, outside rendering, the normals of the points in
    /// `ranges`. The buffers must be live and as large as their table says.
    pub fn record(&self, cb: vk::CommandBuffer, buffers: Buffers, ranges: &[Range<u32>]) {
        (self.compute).record(cb, ranges, |first, count, stride| Push {
            buffers,
            first,
            count,
            stride,
            unused: 0,
        });
    }
}
