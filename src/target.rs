//! The images a frame renders into, a depth buffer, and the host-visible
//! buffers the CPU reads them back from.

use std::time::{Duration, Instant};

use ash::vk;

use crate::gpu::{self, Buffer};

/// Pika's depth is reversed: 1 at the near plane, 0 at the far plane and
/// beyond, so the depth buffer is cleared to 0.
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// A color image and the buffer it is read back into.
struct Attachment {
    format: vk::Format,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    readback: Buffer,
}

/// `width` × `height` color images of the given formats, in attachment
/// order, and a depth buffer. Every handle starts null, so a failed `new`
/// can drop a partly built value.
pub struct Target {
    // A copy of the owner's device table. The owner keeps the `gpu::Device`
    // alive longer than this target (field order), which `drop` relies on.
    device: ash::Device,
    queue: vk::Queue,
    pub width: u32,
    pub height: u32,
    colors: Vec<Attachment>,
    depth: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_view: vk::ImageView,
    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    fence: vk::Fence,
}

impl Target {
    /// `formats` pairs each color attachment's format with its bytes per
    /// pixel, which size its readback buffer.
    pub fn new(
        device: &gpu::Device,
        formats: &[(vk::Format, usize)],
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let d = &device.raw;
        let mut t = Self {
            device: d.clone(),
            queue: device.queue,
            width,
            height,
            colors: Vec::new(),
            depth: vk::Image::null(),
            depth_memory: vk::DeviceMemory::null(),
            depth_view: vk::ImageView::null(),
            pool: vk::CommandPool::null(),
            commands: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
        };
        for &(format, bytes_per_pixel) in formats {
            let bytes = width as u64 * height as u64 * bytes_per_pixel as u64;
            // Uncached host memory reads at a small fraction of cached speed.
            let host = vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED;
            let readback = Buffer::host(device, bytes, vk::BufferUsageFlags::TRANSFER_DST, host)?;
            let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC;
            let (image, memory, view) = image(device, format, usage, COLOR, width, height)?;
            t.colors.push(Attachment {
                format,
                image,
                memory,
                view,
                readback,
            });
        }
        let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        (t.depth, t.depth_memory, t.depth_view) =
            image(device, DEPTH_FORMAT, usage, DEPTH, width, height)?;

        let pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(device.queue_family);
        // SAFETY (this and every call below in `new`): `d` is a live device,
        // each info struct outlives its call, and each handle passed in was
        // created on `d` just above.
        t.pool = unsafe { d.create_command_pool(&pool_info, None) }
            .map_err(|e| format!("vkCreateCommandPool failed: {e}"))?;
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(t.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        t.commands = unsafe { d.allocate_command_buffers(&alloc_info) }
            .map_err(|e| format!("vkAllocateCommandBuffers failed: {e}"))?[0];
        t.fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| format!("vkCreateFence failed: {e}"))?;
        Ok(t)
    }

    /// Whether the target has these color formats and this size.
    pub fn fits(&self, formats: &[(vk::Format, usize)], width: u32, height: u32) -> bool {
        let same = self
            .colors
            .iter()
            .map(|a| a.format)
            .eq(formats.iter().map(|f| f.0));
        same && (self.width, self.height) == (width, height)
    }

    /// Attachment `index`'s read-back pixels as tightly packed rows, first
    /// row first. No GPU work is pending outside `finish`, so they are safe
    /// to read or overwrite here.
    pub fn pixels(&mut self, index: usize) -> &mut [u8] {
        self.colors[index].readback.bytes()
    }

    /// Starts a frame's command buffer, for uploads before `begin_rendering`.
    pub fn begin(&mut self) -> Result<vk::CommandBuffer, String> {
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the pool and buffer were created on this live device, and
        // `finish` waited for the last submit, so neither is in use.
        unsafe {
            self.device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(|e| format!("vkResetCommandPool failed: {e}"))?;
            self.device
                .begin_command_buffer(self.commands, &begin)
                .map_err(|e| format!("vkBeginCommandBuffer failed: {e}"))?;
        }
        Ok(self.commands)
    }

    /// Begins dynamic rendering into every attachment, each color cleared to
    /// its value in `clears`, the depth buffer to the far plane. Record draws,
    /// then call `finish`.
    pub fn begin_rendering(&mut self, clears: &[vk::ClearColorValue]) {
        let d = &self.device;
        let cb = self.commands;
        let mut barriers: Vec<_> = (self.colors.iter())
            .map(|a| {
                vk::ImageMemoryBarrier2::default()
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(a.image)
                    .subresource_range(COLOR)
            })
            .collect();
        let depth_stages = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        barriers.push(
            vk::ImageMemoryBarrier2::default()
                .dst_stage_mask(depth_stages)
                .dst_access_mask(
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .image(self.depth)
                .subresource_range(DEPTH),
        );
        let attachments: Vec<_> = (self.colors.iter().zip(clears))
            .map(|(a, &color)| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(a.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { color })
            })
            .collect();
        let depth = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth_view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });
        let rendering = vk::RenderingInfo::default()
            .render_area(self.area())
            .layer_count(1)
            .color_attachments(&attachments)
            .depth_attachment(&depth);
        // SAFETY: `cb` is recording (`begin`), every handle was created on
        // `d`, and the info structs outlive the calls.
        unsafe {
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
            d.cmd_begin_rendering(cb, &rendering);
        }
    }

    /// Ends the frame begun by `begin_rendering`: copies each attachment that
    /// `read` marks to its readback buffer, submits and waits. Returns how
    /// long the submit took to complete.
    pub fn finish(&mut self, read: &[bool]) -> Result<Duration, String> {
        let d = &self.device;
        let cb = self.commands;
        let read: Vec<&Attachment> = (self.colors.iter().zip(read))
            .filter(|(_, r)| **r)
            .map(|(a, _)| a)
            .collect();
        let to_copy: Vec<_> = (read.iter())
            .map(|a| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(a.image)
                    .subresource_range(COLOR)
            })
            .collect();
        // Host-coherent memory still needs this for the copies to be visible to the host.
        let to_host: Vec<_> = (read.iter())
            .map(|a| {
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)
                    .buffer(a.readback.raw)
                    .size(vk::WHOLE_SIZE)
            })
            .collect();
        // A zero row length and height mean tightly packed rows.
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            });
        // SAFETY (this and every call below in `finish`): `cb` is recording
        // inside the rendering `begin_rendering` started, every handle was
        // created on `d`, and the info structs outlive the calls.
        unsafe {
            d.cmd_end_rendering(cb);
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&to_copy),
            );
            for a in &read {
                d.cmd_copy_image_to_buffer(
                    cb,
                    a.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    a.readback.raw,
                    &[region],
                );
            }
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().buffer_memory_barriers(&to_host),
            );
            d.end_command_buffer(cb)
                .map_err(|e| format!("vkEndCommandBuffer failed: {e}"))?;

            let submit = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
            let submits = [vk::SubmitInfo2::default().command_buffer_infos(&submit)];
            d.reset_fences(&[self.fence])
                .map_err(|e| format!("vkResetFences failed: {e}"))?;
            let start = Instant::now();
            d.queue_submit2(self.queue, &submits, self.fence)
                .map_err(|e| format!("vkQueueSubmit2 failed: {e}"))?;
            d.wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| format!("vkWaitForFences failed: {e}"))?;
            Ok(start.elapsed())
        }
    }

    /// The whole image, for render areas, viewports and scissors.
    pub fn area(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.width,
                height: self.height,
            },
        }
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        let d = &self.device;
        // SAFETY: every handle was created on `d` or is null, which destroy
        // calls accept, and waiting for idle means none is still in use.
        // The readback buffers drop with `colors`, after this.
        unsafe {
            let _ = d.device_wait_idle();
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.pool, None);
            for a in &self.colors {
                d.destroy_image_view(a.view, None);
                d.destroy_image(a.image, None);
                d.free_memory(a.memory, None);
            }
            d.destroy_image_view(self.depth_view, None);
            d.destroy_image(self.depth, None);
            d.free_memory(self.depth_memory, None);
        }
    }
}

const COLOR: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

const DEPTH: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::DEPTH,
    ..COLOR
};

/// A device-local 2D image with its memory and a view of `range`.
fn image(
    device: &gpu::Device,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    range: vk::ImageSubresourceRange,
    width: u32,
    height: u32,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), String> {
    let d = &device.raw;
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .usage(usage);
    // SAFETY (this and every call below): `d` is a live device, each info
    // struct outlives its call, and each handle passed in was created on `d`
    // just above. On an error, the handles made so far are destroyed here.
    let image = unsafe { d.create_image(&info, None) }
        .map_err(|e| format!("vkCreateImage ({width}×{height} {format:?}) failed: {e}"))?;
    let needs = unsafe { d.get_image_memory_requirements(image) };
    let flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    let memory = match gpu::allocate(device, needs, flags, vk::MemoryAllocateFlags::empty()) {
        Ok(m) => m,
        Err(e) => {
            unsafe { d.destroy_image(image, None) };
            return Err(e);
        }
    };
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(range);
    let view = unsafe { d.bind_image_memory(image, memory, 0) }
        .map_err(|e| format!("vkBindImageMemory failed: {e}"))
        .and_then(|()| {
            unsafe { d.create_image_view(&view_info, None) }
                .map_err(|e| format!("vkCreateImageView failed: {e}"))
        });
    match view {
        Ok(view) => Ok((image, memory, view)),
        Err(e) => {
            unsafe {
                d.destroy_image(image, None);
                d.free_memory(memory, None);
            }
            Err(e)
        }
    }
}
