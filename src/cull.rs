//! The cull pass: keeps each mesh instance whose box may meet a view's frustum.

use ash::vk;

use crate::geometry::Bound;
use crate::gpu;
use crate::view::View;

pub struct Cull {
    // The owner's device table, which outlives this (field order).
    device: ash::Device,
    compute: gpu::Compute,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    view: vk::DeviceAddress,
    first: u32,
    count: u32,
    stride: u32,
    unused: u32,
}

impl Cull {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
        let compute = gpu::Compute::new(device, spirv, size_of::<Push>())?;
        Ok(Self {
            device: device.raw.clone(),
            compute,
        })
    }

    /// Call after `view` began its frame.
    pub fn record(&self, cb: vk::CommandBuffer, geometry: &Bound, view: &View) {
        if geometry.count == 0 {
            return;
        }
        let instances = 0..geometry.instance_count;
        (self.compute).record(cb, &[instances], |first, count, stride| Push {
            view: view.constants(),
            first,
            count,
            stride,
            unused: 0,
        });
        let culled = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::DRAW_INDIRECT
                    | vk::PipelineStageFlags2::VERTEX_SHADER
                    | vk::PipelineStageFlags2::HOST,
            )
            .dst_access_mask(
                vk::AccessFlags2::INDIRECT_COMMAND_READ
                    | vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::HOST_READ,
            );
        // SAFETY: `cb` is recording outside rendering.
        unsafe {
            self.device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().memory_barriers(&[culled]),
            )
        };
    }
}
