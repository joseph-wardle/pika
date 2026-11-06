//! The stencil pass: a compute pipeline writing each refined value of a
//! subdivision surface, a point's xyz or a texcoord's xy, as the weighted
//! sum of the cage values its stencil names. OpenSubdiv built the stencils,
//! in the delegate.

use std::ops::Range;

use ash::vk;

use crate::gpu;

pub struct Subdiv {
    compute: gpu::Compute,
}

/// The addresses of the buffers `shaders/subdiv.slang` reads and writes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Addresses {
    pub cage: vk::DeviceAddress,
    pub offsets: vk::DeviceAddress,
    pub indices: vk::DeviceAddress,
    pub weights: vk::DeviceAddress,
    pub refined: vk::DeviceAddress,
}

/// `shaders/subdiv.slang`'s push constants.
#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    addresses: Addresses,
    first: u32,
    count: u32,
    stride: u32,
    width: u32,
}

impl Subdiv {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
        let compute = gpu::Compute::new(device, spirv, size_of::<Push>())?;
        Ok(Self { compute })
    }

    /// Records into `cb`, outside rendering, the refined values in `ranges`,
    /// each `width` floats (3 for points, 2 for texcoords). The buffers
    /// must be live and as large as their table says.
    pub fn record(
        &self,
        cb: vk::CommandBuffer,
        addresses: Addresses,
        width: u32,
        ranges: &[Range<u32>],
    ) {
        (self.compute).record(cb, ranges, |first, count, stride| Push {
            addresses,
            first,
            count,
            stride,
            width,
        });
    }
}
