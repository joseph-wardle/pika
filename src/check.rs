//! The first-frame check: draws a known rectangle offscreen and compares the
//! read-back pixels with the image it must produce.

use ash::vk;

use pika::gpu;
use pika::shader::Shader;
use pika::target::{self, Target};

/// Width and height of the target. A power of two keeps the shader's pixel to
/// NDC mapping exact, and positions must fit the 8-bit channels.
pub const SIZE: i32 = 64;

/// The drawn rectangle in pixels, `[x0, y0, x1, y1]` with exclusive ends.
/// Off-centre and not square, so a flipped or transposed image fails.
pub const RECT: [i32; 4] = [16, 8, 56, 40];

/// `RECT` moved 1 px right, the negative control. Because each pixel holds its
/// own position, only the two edge columns differ: 2 × height pixels.
pub const SHIFTED: [i32; 4] = [17, 8, 57, 40];

/// Mismatches a draw of `SHIFTED` must give against `expected(RECT)`.
pub const SHIFTED_MISMATCHES: usize = 2 * (RECT[3] - RECT[1]) as usize;

const FORMAT: vk::Format = vk::Format::R8G8B8A8_UINT;

/// The image a draw of `rect` must produce: tightly packed RGBA8_UINT rows,
/// `[x, y, 1, 255]` inside the rectangle and zero outside.
pub fn expected(rect: [i32; 4]) -> Vec<u8> {
    let [x0, y0, x1, y1] = rect;
    let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let inside = (x0..x1).contains(&x) && (y0..y1).contains(&y);
            let pixel = if inside {
                [x as u8, y as u8, 1, 255]
            } else {
                [0; 4]
            };
            pixels.extend_from_slice(&pixel);
        }
    }
    pixels
}

/// How many RGBA8 pixels differ between two images of the same size.
pub fn mismatches(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len(), "images differ in size");
    let (a, b) = (a.as_chunks::<4>().0, b.as_chunks::<4>().0);
    a.iter().zip(b).filter(|(p, q)| p != q).count()
}

/// The check pipeline drawing into a `SIZE`² RGBA8_UINT target.
pub struct Offscreen<'a> {
    // Field order is drop order: the target waits for idle before the
    // pipeline is destroyed.
    target: Target,
    device: &'a gpu::Device<'a>,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl<'a> Offscreen<'a> {
    pub fn new(device: &'a gpu::Device<'a>, shader: &Shader) -> Result<Self, String> {
        let target = Target::new(device, &[(FORMAT, 4)], SIZE as u32, SIZE as u32)?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .size(size_of::<Push>() as u32)];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push);
        // SAFETY: `device.raw` is live and `layout_info` outlives the call.
        let layout = unsafe { device.raw.create_pipeline_layout(&layout_info, None) }
            .map_err(|e| format!("vkCreatePipelineLayout failed: {e}"))?;
        let mut o = Self {
            target,
            device,
            layout,
            pipeline: vk::Pipeline::null(),
        };
        o.pipeline = pipeline(&device.raw, o.layout, &shader.spirv, o.target.area())?;
        Ok(o)
    }

    /// Clears the target, draws `rect` and returns the read-back pixels as
    /// tightly packed RGBA8 rows.
    pub fn draw(&mut self, rect: [i32; 4]) -> Result<Vec<u8>, String> {
        // A draw or copy that never lands leaves 0xCD, which no pixel can be.
        self.target.pixels(0).fill(0xCD);
        let cb = self.target.begin()?;
        self.target
            .begin_rendering(&[vk::ClearColorValue::default()]);
        let push = Push {
            rect,
            size: [SIZE, SIZE],
        };
        let d = &self.device.raw;
        // SAFETY: `cb` is recording inside `begin`'s rendering, and the
        // pipeline and layout were created on `d` for the target's format.
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            d.cmd_push_constants(
                cb,
                self.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &push.bytes(),
            );
            d.cmd_draw(cb, 6, 1, 0, 0);
        }
        self.target.submit(&[true])?;
        self.target.wait()?;
        Ok(self.target.pixels(0).to_vec())
    }
}

impl Drop for Offscreen<'_> {
    fn drop(&mut self) {
        let d = &self.device.raw;
        // SAFETY: both handles were created on `d` or are null, which destroy
        // calls accept, and no submit is pending: `draw` waits for each one.
        unsafe {
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
        }
    }
}

/// `check.slang`'s push constants, laid out as Slang lays out `Push`.
#[repr(C)]
struct Push {
    rect: [i32; 4],
    size: [i32; 2],
}

impl Push {
    fn bytes(&self) -> Vec<u8> {
        self.rect
            .iter()
            .chain(&self.size)
            .flat_map(|v| v.to_ne_bytes())
            .collect()
    }
}

/// The check pipeline: no vertex input, no blending (integer formats cannot
/// blend), two-sided, rendering to one `FORMAT` attachment. The target's
/// depth buffer is attached but unused.
fn pipeline(
    d: &ash::Device,
    layout: vk::PipelineLayout,
    spirv: &[u32],
    area: vk::Rect2D,
) -> Result<vk::Pipeline, String> {
    let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
    // SAFETY: `d` is live and `module_info` outlives the call.
    let module = unsafe { d.create_shader_module(&module_info, None) }
        .map_err(|e| format!("vkCreateShaderModule failed: {e}"))?;
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
    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: SIZE as f32,
        height: SIZE as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [area];
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let formats = [FORMAT];
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&formats)
        .depth_attachment_format(target::DEPTH_FORMAT);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default();
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .depth_stencil_state(&depth)
        .layout(layout)
        .push_next(&mut rendering);
    // SAFETY: `d` is live, `module` and `layout` were created on it, and
    // `info` and what it points to outlive the call.
    let pipeline = unsafe { d.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
    // SAFETY: pipeline creation is complete, so the module is no longer used.
    unsafe { d.destroy_shader_module(module, None) };
    pipeline
        .map(|p| p[0])
        .map_err(|(_, e)| format!("vkCreateGraphicsPipelines failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(image: &[u8], x: i32, y: i32) -> &[u8] {
        let i = ((y * SIZE + x) * 4) as usize;
        &image[i..i + 4]
    }

    #[test]
    fn expected_covers_rect_with_exclusive_ends() {
        let image = expected(RECT);
        assert_eq!(image.len(), (SIZE * SIZE * 4) as usize);
        assert_eq!(pixel(&image, 16, 8), [16, 8, 1, 255]);
        assert_eq!(pixel(&image, 55, 39), [55, 39, 1, 255]);
        assert_eq!(pixel(&image, 15, 8), [0; 4]);
        assert_eq!(pixel(&image, 16, 7), [0; 4]);
        assert_eq!(pixel(&image, 56, 39), [0; 4]);
        assert_eq!(pixel(&image, 55, 40), [0; 4]);
    }

    #[test]
    fn identical_images_match() {
        assert_eq!(mismatches(&expected(RECT), &expected(RECT)), 0);
    }

    #[test]
    fn shifted_rect_differs_in_two_columns() {
        assert_eq!(
            mismatches(&expected(RECT), &expected(SHIFTED)),
            SHIFTED_MISMATCHES
        );
    }

    #[test]
    fn unwritten_readback_fails_every_pixel() {
        // The GPU path prefills the readback buffer with 0xCD before each draw.
        let unwritten = vec![0xCD; (SIZE * SIZE * 4) as usize];
        assert_eq!(
            mismatches(&expected(RECT), &unwritten),
            (SIZE * SIZE) as usize
        );
    }
}
