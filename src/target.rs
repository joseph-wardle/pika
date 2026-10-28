//! A color image to render into and the host-visible buffer the CPU reads it
//! back from.

use std::time::Instant;

use ash::vk;

use crate::gpu;

/// A `width` × `height` color image of `format`, cleared and read back once per
/// frame. Every handle starts null, so a failed `new` can drop a partly built value.
pub struct Target {
    // A copy of the owner's device table. The owner keeps the `gpu::Device`
    // alive longer than this target (field order), which `drop` relies on.
    device: ash::Device,
    queue: vk::Queue,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    bytes: usize,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    view: vk::ImageView,
    readback: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    mapped: *mut u8,
    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    fence: vk::Fence,
}

// SAFETY: `mapped` points into memory this target owns, and it is only
// dereferenced through `&mut self`, so moving the target to another thread is sound.
unsafe impl Send for Target {}

impl Target {
    /// `bytes_per_pixel` must match `format`; it sizes the readback buffer.
    pub fn new(
        device: &gpu::Device,
        format: vk::Format,
        bytes_per_pixel: usize,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let d = &device.raw;
        let mut t = Self {
            device: d.clone(),
            queue: device.queue,
            width,
            height,
            format,
            bytes: width as usize * height as usize * bytes_per_pixel,
            image: vk::Image::null(),
            image_memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            readback: vk::Buffer::null(),
            readback_memory: vk::DeviceMemory::null(),
            mapped: std::ptr::null_mut(),
            pool: vk::CommandPool::null(),
            commands: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
        };

        let image_info = vk::ImageCreateInfo::default()
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
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC);
        // SAFETY (this and every call below in `new`): `d` is a live device,
        // each info struct outlives its call, and each handle passed in was
        // created on `d` just above.
        t.image = unsafe { d.create_image(&image_info, None) }
            .map_err(|e| format!("vkCreateImage ({width}×{height} {format:?}) failed: {e}"))?;
        let needs = unsafe { d.get_image_memory_requirements(t.image) };
        t.image_memory = allocate(device, needs, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        unsafe { d.bind_image_memory(t.image, t.image_memory, 0) }
            .map_err(|e| format!("vkBindImageMemory failed: {e}"))?;
        let view_info = vk::ImageViewCreateInfo::default()
            .image(t.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(COLOR);
        t.view = unsafe { d.create_image_view(&view_info, None) }
            .map_err(|e| format!("vkCreateImageView failed: {e}"))?;

        let buffer_info = vk::BufferCreateInfo::default()
            .size(t.bytes as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST);
        t.readback = unsafe { d.create_buffer(&buffer_info, None) }
            .map_err(|e| format!("vkCreateBuffer failed: {e}"))?;
        let needs = unsafe { d.get_buffer_memory_requirements(t.readback) };
        // Uncached host memory reads at a small fraction of cached speed.
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;
        t.readback_memory = allocate(device, needs, host)?;
        unsafe { d.bind_buffer_memory(t.readback, t.readback_memory, 0) }
            .map_err(|e| format!("vkBindBufferMemory failed: {e}"))?;
        // Freeing the memory in `drop` also unmaps it.
        t.mapped = unsafe {
            d.map_memory(
                t.readback_memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| format!("vkMapMemory failed: {e}"))?
        .cast();

        let pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(device.queue_family);
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

    /// The read-back pixels as tightly packed rows. No GPU work is pending
    /// outside `finish`, so they are safe to read or overwrite here.
    pub fn pixels(&mut self) -> &mut [u8] {
        // SAFETY: `mapped` is the whole of the readback memory, at least
        // `bytes` long, and `finish` waits for every submit before returning.
        unsafe { std::slice::from_raw_parts_mut(self.mapped, self.bytes) }
    }

    /// Starts a frame: begins dynamic rendering into the image, cleared to
    /// `clear`. Record draws into the returned command buffer, then call `finish`.
    pub fn begin(&mut self, clear: vk::ClearValue) -> Result<vk::CommandBuffer, String> {
        let d = &self.device;
        let cb = self.commands;
        // SAFETY (this and every call below in `begin`): `d` is a live device,
        // all handles were created on it in `new`, `cb` is not in use because
        // the last submit was waited on, and each info struct outlives its call.
        unsafe {
            d.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(|e| format!("vkResetCommandPool failed: {e}"))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            d.begin_command_buffer(cb, &begin)
                .map_err(|e| format!("vkBeginCommandBuffer failed: {e}"))?;

            let to_attachment = vk::ImageMemoryBarrier2::default()
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(self.image)
                .subresource_range(COLOR);
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_attachment]),
            );

            let attachment = [vk::RenderingAttachmentInfo::default()
                .image_view(self.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear)];
            let rendering = vk::RenderingInfo::default()
                .render_area(self.area())
                .layer_count(1)
                .color_attachments(&attachment);
            d.cmd_begin_rendering(cb, &rendering);
        }
        Ok(cb)
    }

    /// Ends the frame begun by `begin`: copies the image to the readback
    /// buffer, submits and waits. Returns how long the submit took to complete.
    pub fn finish(&mut self) -> Result<std::time::Duration, String> {
        let d = &self.device;
        let cb = self.commands;
        // SAFETY (this and every call below in `finish`): as in `begin`, and
        // `cb` is recording, inside the rendering `begin` started.
        unsafe {
            d.cmd_end_rendering(cb);
            let to_copy = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(self.image)
                .subresource_range(COLOR);
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_copy]),
            );
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
            d.cmd_copy_image_to_buffer(
                cb,
                self.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback,
                &[region],
            );
            // Host-coherent memory still needs this for the copy to be visible to the host.
            let to_host = vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)
                .buffer(self.readback)
                .size(vk::WHOLE_SIZE);
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[to_host]),
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
        unsafe {
            let _ = d.device_wait_idle();
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.pool, None);
            d.destroy_buffer(self.readback, None);
            d.free_memory(self.readback_memory, None);
            d.destroy_image_view(self.view, None);
            d.destroy_image(self.image, None);
            d.free_memory(self.image_memory, None);
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

fn allocate(
    device: &gpu::Device,
    needs: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory, String> {
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(needs.size)
        .memory_type_index(gpu::memory_type(
            &device.memory_types,
            needs.memory_type_bits,
            flags,
        )?);
    // SAFETY: `device.raw` is live and `info` outlives the call.
    unsafe { device.raw.allocate_memory(&info, None) }
        .map_err(|e| format!("vkAllocateMemory ({flags:?}) failed: {e}"))
}
