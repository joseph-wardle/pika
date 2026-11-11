//! The tangent pass: each vertex's tangent and handedness, for normal maps.

use std::ops::Range;

use ash::vk;

use crate::gpu;

pub struct Tangent {
    compute: gpu::Compute,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Addresses {
    pub points: vk::DeviceAddress,
    pub normals: vk::DeviceAddress,
    pub smooth: vk::DeviceAddress,
    pub vertices: vk::DeviceAddress,
    pub texcoord: vk::DeviceAddress,
    pub vertex_texcoord: vk::DeviceAddress,
    pub offsets: vk::DeviceAddress,
    pub corners: vk::DeviceAddress,
    pub indices: vk::DeviceAddress,
    pub tangents: vk::DeviceAddress,
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

impl Tangent {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
        let compute = gpu::Compute::new(device, spirv, c"cs", size_of::<Push>())?;
        Ok(Self { compute })
    }

    /// The points, normals and texcoords it reads must be written already.
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
