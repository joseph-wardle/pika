//! The smooth normal pass: a compute pipeline writing each point's normal
//! from the faces around it, for meshes whose normals are not authored.

use ash::vk;

use crate::gpu;

/// Vulkan's guaranteed minimum of groups in one dispatch dimension.
const MAX_GROUPS: u64 = 65535;
/// `numthreads` in `shaders/smooth.slang`.
const GROUP_SIZE: u64 = 64;

pub struct Smooth {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

/// `shaders/smooth.slang`'s push constants.
#[repr(C)]
pub struct Push {
    pub points: vk::DeviceAddress,
    pub offsets: vk::DeviceAddress,
    pub corners: vk::DeviceAddress,
    pub normals: vk::DeviceAddress,
    pub count: u32,
    pub stride: u32,
}

impl Smooth {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
        let d = &device.raw;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(size_of::<Push>() as u32)];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push);
        // SAFETY: `d` is live and `layout_info` outlives the call.
        let layout = unsafe { d.create_pipeline_layout(&layout_info, None) }
            .map_err(|e| format!("vkCreatePipelineLayout failed: {e}"))?;
        let mut s = Self {
            device: d.clone(),
            layout,
            pipeline: vk::Pipeline::null(),
        };
        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: `d` is live and `module_info` outlives the call.
        let module = unsafe { d.create_shader_module(&module_info, None) }
            .map_err(|e| format!("vkCreateShaderModule failed: {e}"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"cs");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(s.layout);
        // SAFETY: `d` is live, `module` and the layout were created on it,
        // and `info` outlives the call.
        let built = unsafe { d.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
        // SAFETY: pipeline creation is complete, so the module is no longer used.
        unsafe { d.destroy_shader_module(module, None) };
        s.pipeline = built
            .map(|p| p[0])
            .map_err(|(_, e)| format!("vkCreateComputePipelines failed: {e}"))?;
        Ok(s)
    }

    /// Records into `cb` the normals of `count` points, outside rendering.
    /// The addresses must be of live buffers as large as the table says.
    pub fn record(&self, cb: vk::CommandBuffer, mut push: Push) {
        if push.count == 0 {
            return;
        }
        let groups = (push.count as u64).div_ceil(GROUP_SIZE).min(MAX_GROUPS);
        push.stride = (groups * GROUP_SIZE) as u32;
        // SAFETY: `Push` is plain `repr(C)` data without padding.
        let bytes = unsafe {
            std::slice::from_raw_parts((&raw const push).cast::<u8>(), size_of::<Push>())
        };
        let d = &self.device;
        // SAFETY: `cb` is recording outside rendering, and the pipeline and
        // layout were created on `d`.
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            d.cmd_dispatch(cb, groups as u32, 1, 1);
        }
    }
}

impl Drop for Smooth {
    fn drop(&mut self) {
        // SAFETY: both handles were created on `device` or are null, and the
        // owner's target waited for idle before this drops (field order).
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
