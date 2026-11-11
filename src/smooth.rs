//! The smooth normal pass, for meshes whose normals are not authored.

use std::ops::Range;

use ash::vk;

use crate::gpu;

pub struct Smooth {
    compute: gpu::Compute,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Addresses {
    pub points: vk::DeviceAddress,
    pub offsets: vk::DeviceAddress,
    pub corners: vk::DeviceAddress,
    pub normals: vk::DeviceAddress,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    addresses: Addresses,
    first: u32,
    count: u32,
    stride: u32,
    unused: u32,
}

impl Smooth {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
        let compute = gpu::Compute::new(device, spirv, c"cs", size_of::<Push>())?;
        Ok(Self { compute })
    }

    pub fn record(&self, cb: vk::CommandBuffer, addresses: Addresses, ranges: &[Range<u32>]) {
        (self.compute).record(cb, ranges, |first, count, stride| Push {
            addresses,
            first,
            count,
            stride,
            unused: 0,
        });
    }
}
