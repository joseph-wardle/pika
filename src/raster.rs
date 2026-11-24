//! The prepass. It draws the culled instances into depth, the depth and normal AOVs, and
//! motion.

use ash::vk;

use crate::geometry::Addresses;
use crate::gpu;
use crate::target;
use crate::view::View;

/// Comes after colour, in the target order of `shaders/mesh.slang`. Devices needn't render to
/// three-channel formats, so normals are drawn with a fourth channel.
pub const DEPTH_AOV: (vk::Format, usize) = (vk::Format::R32_SFLOAT, 4);
pub const NORMAL_AOV: (vk::Format, usize) = (vk::Format::R32G32B32A32_SFLOAT, 16);
/// Each pixel's offset to where its surface was in the frame before, which `film` reprojects
/// with. Half floats keep it within 1/64 px below 32 px, where history matters most.
pub const MOTION: vk::Format = vk::Format::R16G16_SFLOAT;
pub const MOTION_ATTACHMENT: usize = 3;

/// In the order of `pipelines`.
pub const COLOR_FORMATS: [vk::Format; 2] = [
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::R32G32B32A32_SFLOAT,
];

pub struct Raster {
    // A copy of the owner's device table. The owner's device outlives this, by field order.
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
}

impl Raster {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
        let d = &device.raw;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(size_of::<vk::DeviceAddress>() as u32)];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push);
        // SAFETY: the device is alive and `layout_info` outlives the call.
        let layout = unsafe { d.create_pipeline_layout(&layout_info, None) }
            .map_err(|e| format!("vkCreatePipelineLayout failed: {e}"))?;
        let mut r = Self {
            device: d.clone(),
            layout,
            pipelines: Vec::new(),
        };
        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: the device is alive and `module_info` outlives the call.
        let module = unsafe { d.create_shader_module(&module_info, None) }
            .map_err(|e| format!("vkCreateShaderModule failed: {e}"))?;
        let depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::GREATER);
        let built: Vec<_> = (COLOR_FORMATS.iter())
            .map(|&format| {
                let colors = [
                    (format, vk::ColorComponentFlags::empty()),
                    (DEPTH_AOV.0, vk::ColorComponentFlags::RGBA),
                    (NORMAL_AOV.0, vk::ColorComponentFlags::RGBA),
                    (
                        MOTION,
                        vk::ColorComponentFlags::R | vk::ColorComponentFlags::G,
                    ),
                ];
                gpu::graphics(
                    d,
                    r.layout,
                    module,
                    [c"vs", c"fs"],
                    &colors,
                    target::DEPTH_FORMAT,
                    depth,
                )
            })
            .collect();
        // SAFETY: the pipeline is built, so nothing uses the shader module any more.
        unsafe { d.destroy_shader_module(module, None) };
        // Kept before the error returns, so `drop` destroys them.
        r.pipelines = built.iter().filter_map(|p| p.clone().ok()).collect();
        built.into_iter().collect::<Result<Vec<_>, _>>()?;
        Ok(r)
    }

    /// Records inside rendering into `color` (one of `COLOR_FORMATS`), `DEPTH_AOV`, `NORMAL_AOV`
    /// and `MOTION`, after the cull pass.
    pub fn draw(
        &self,
        cb: vk::CommandBuffer,
        color: vk::Format,
        area: vk::Rect2D,
        geometry: &Addresses,
        view: &View,
    ) {
        if geometry.count == 0 {
            return;
        }
        let index = COLOR_FORMATS.iter().position(|&f| f == color);
        let pipeline = self.pipelines[index.expect("the renderer chooses a color format")];
        let push = view.constants().to_ne_bytes();
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: area.extent.width as f32,
            height: area.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let d = &self.device;
        // SAFETY: recording inside rendering with this pipeline's formats and handles from this device.
        // The view's constants point at buffers as large as the draws read.
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            d.cmd_set_viewport(cb, 0, &[viewport]);
            d.cmd_set_scissor(cb, 0, &[area]);
            let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
            d.cmd_push_constants(cb, self.layout, stages, 0, &push);
            d.cmd_bind_index_buffer(cb, geometry.indices, 0, vk::IndexType::UINT32);
            d.cmd_draw_indexed_indirect(
                cb,
                view.records(),
                0,
                geometry.count,
                size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
        }
    }
}

impl Drop for Raster {
    fn drop(&mut self) {
        // SAFETY: every handle was created on this device, and the owner's target waited for the
        // device to go idle before this drops, by field order.
        unsafe {
            for &p in &self.pipelines {
                self.device.destroy_pipeline(p, None);
            }
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
