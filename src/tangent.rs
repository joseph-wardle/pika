//! The tangent pass: a compute pipeline writing each vertex's tangent and
//! handedness from the triangles around it, for normal maps.

use std::ops::Range;

use ash::vk;

use crate::gpu;

pub struct Tangent {
    compute: gpu::Compute,
}

/// The addresses of the buffers `shaders/tangent.slang` reads and writes.
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

/// `shaders/tangent.slang`'s push constants.
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
        let compute = gpu::Compute::new(device, spirv, size_of::<Push>())?;
        Ok(Self { compute })
    }

    /// Records into `cb`, outside rendering, the tangents of the vertices
    /// in `ranges`. The buffers must be live and as large as their table
    /// says, and the points, normals and texcoords read written already.
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
