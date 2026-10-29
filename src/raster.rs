//! The mesh pass: one pipeline per color format, drawing every mesh instance
//! with a single multi-draw into the color, depth and normal AOVs.

use ash::vk;

use crate::geometry::Bound;
use crate::gpu;
use crate::target;
use crate::transform::Camera;

/// The AOV attachments after color, in `shaders/mesh.slang`'s target order.
/// Vulkan devices need not render to three-channel formats, so normals are
/// drawn with a fourth.
pub const DEPTH_AOV: (vk::Format, usize) = (vk::Format::R32_SFLOAT, 4);
pub const NORMAL_AOV: (vk::Format, usize) = (vk::Format::R32G32B32A32_SFLOAT, 16);

/// The color formats a pipeline is built for, in the order of `pipelines`.
pub const COLOR_FORMATS: [vk::Format; 2] = [
    vk::Format::R16G16B16A16_SFLOAT,
    vk::Format::R32G32B32A32_SFLOAT,
];

pub struct Raster {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
}

/// `shaders/mesh.slang`'s push constants.
#[repr(C)]
struct Push {
    camera: Camera,
    points: vk::DeviceAddress,
    normals: vk::DeviceAddress,
    vertices: vk::DeviceAddress,
    instances: vk::DeviceAddress,
}

impl Raster {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
        let d = &device.raw;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(size_of::<Push>() as u32)];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push);
        // SAFETY: `d` is live and `layout_info` outlives the call.
        let layout = unsafe { d.create_pipeline_layout(&layout_info, None) }
            .map_err(|e| format!("vkCreatePipelineLayout failed: {e}"))?;
        let mut r = Self {
            device: d.clone(),
            layout,
            pipelines: Vec::new(),
        };
        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: `d` is live and `module_info` outlives the call.
        let module = unsafe { d.create_shader_module(&module_info, None) }
            .map_err(|e| format!("vkCreateShaderModule failed: {e}"))?;
        let built: Vec<_> = (COLOR_FORMATS.iter())
            .map(|&format| pipeline(d, r.layout, module, format))
            .collect();
        // SAFETY: pipeline creation is complete, so the module is no longer used.
        unsafe { d.destroy_shader_module(module, None) };
        // Kept before the error returns, so `drop` destroys them.
        r.pipelines = built.iter().filter_map(|p| p.clone().ok()).collect();
        built.into_iter().collect::<Result<Vec<_>, _>>()?;
        Ok(r)
    }

    /// Records the draws into `cb`, inside rendering into a target of
    /// `color` (one of `COLOR_FORMATS`), `DEPTH_AOV` and `NORMAL_AOV`.
    pub fn draw(
        &self,
        cb: vk::CommandBuffer,
        color: vk::Format,
        area: vk::Rect2D,
        camera: Camera,
        geometry: &Bound,
    ) {
        if geometry.count == 0 {
            return;
        }
        let index = COLOR_FORMATS.iter().position(|&f| f == color);
        let pipeline = self.pipelines[index.expect("the renderer chooses a color format")];
        let push = Push {
            camera,
            points: geometry.points,
            normals: geometry.normals,
            vertices: geometry.vertices,
            instances: geometry.instances,
        };
        // SAFETY: `Push` is plain `repr(C)` data without padding.
        let push = unsafe {
            std::slice::from_raw_parts((&raw const push).cast::<u8>(), size_of::<Push>())
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: area.extent.width as f32,
            height: area.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let d = &self.device;
        // SAFETY: `cb` is recording inside rendering of the formats this
        // pipeline was built for, and every handle was created on `d`. The
        // addresses are of live buffers, as large as the draws read.
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            d.cmd_set_viewport(cb, 0, &[viewport]);
            d.cmd_set_scissor(cb, 0, &[area]);
            let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
            d.cmd_push_constants(cb, self.layout, stages, 0, push);
            d.cmd_bind_index_buffer(cb, geometry.indices, 0, vk::IndexType::UINT32);
            d.cmd_draw_indexed_indirect(
                cb,
                geometry.records,
                0,
                geometry.count,
                size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
        }
    }
}

impl Drop for Raster {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device`, and the owner's
        // target waited for idle before this drops (field order).
        unsafe {
            for &p in &self.pipelines {
                self.device.destroy_pipeline(p, None);
            }
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// The mesh pipeline for one color format: two-sided, reversed depth.
fn pipeline(
    d: &ash::Device,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    color: vk::Format,
) -> Result<vk::Pipeline, String> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(module)
            .name(c"vs"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(module)
            .name(c"fs"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    // Nothing is culled. USD's front faces wind counter-clockwise in a
    // right-handed frame, and the camera's y is not flipped (rows go bottom
    // up), so they wind clockwise in Vulkan's framebuffer coordinates.
    // hydra/images/behind.usda fails if this is reversed.
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::GREATER);
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA); 3];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
    let formats = [color, DEPTH_AOV.0, NORMAL_AOV.0];
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&formats)
        .depth_attachment_format(target::DEPTH_FORMAT);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    // SAFETY: `d` is live, `module` and `layout` were created on it, and
    // `info` and what it points to outlive the call.
    unsafe { d.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
        .map(|p| p[0])
        .map_err(|(_, e)| format!("vkCreateGraphicsPipelines ({color:?}) failed: {e}"))
}
