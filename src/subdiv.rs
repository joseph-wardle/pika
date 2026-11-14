//! The stencil pass: each refined subdivision value as the weighted sum of
//! the cage values its OpenSubdiv stencil names.

use std::ops::Range;

use ash::vk;

use crate::gpu;

pub struct Subdiv {
    compute: gpu::Compute,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Addresses {
    pub cage: vk::DeviceAddress,
    pub offsets: vk::DeviceAddress,
    pub indices: vk::DeviceAddress,
    pub weights: vk::DeviceAddress,
    pub refined: vk::DeviceAddress,
}

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
        let compute = gpu::Compute::new(device, spirv, c"cs", size_of::<Push>())?;
        Ok(Self { compute })
    }

    /// `width` is floats per value: 3 for points, 2 for texcoords.
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
