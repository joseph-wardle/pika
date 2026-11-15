//! The images a frame renders into, its depth buffer and the buffers read back from them.

use ash::vk;

use crate::gpu::{self, Buffer, Memory};

/// Pika's depth is reversed: 1 at the near plane, 0 at the far plane and
/// beyond, so the depth buffer is cleared to 0.
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

struct Attachment {
    format: vk::Format,
    image: vk::Image,
    memory: vk::DeviceMemory,
    bytes: u64,
    view: vk::ImageView,
    /// None for an attachment only the GPU reads.
    readback: Option<Buffer>,
}

/// Every handle starts null, so a failed `new` can drop a partly built value.
pub struct Target {
    // A copy of the owner's device table; the owner outlives this (field order), as `drop` needs.
    device: ash::Device,
    pub width: u32,
    pub height: u32,
    colors: Vec<Attachment>,
    depth: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_bytes: u64,
    depth_view: vk::ImageView,
    /// Per pixel, the first color attachment's bytes.
    pub color_bytes: u64,
}

impl Target {
    /// `formats` pairs each read-back color format with its bytes per pixel; the color
    /// attachments `unread`, which only the GPU reads, follow them.
    pub fn new(
        device: &gpu::Device,
        formats: &[(vk::Format, usize)],
        unread: &[vk::Format],
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let mut t = Self {
            device: device.raw.clone(),
            width,
            height,
            colors: Vec::new(),
            depth: vk::Image::null(),
            depth_memory: vk::DeviceMemory::null(),
            depth_bytes: 0,
            depth_view: vk::ImageView::null(),
            color_bytes: formats.first().map_or(0, |f| f.1 as u64),
        };
        let read = formats.iter().map(|&(f, bytes)| (f, Some(bytes)));
        for (format, bytes_per_pixel) in read.chain(unread.iter().map(|&f| (f, None))) {
            // Uncached host memory reads at a small fraction of cached speed.
            let host = vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED;
            let readback = match bytes_per_pixel {
                Some(b) => {
                    let bytes = width as u64 * height as u64 * b as u64;
                    let usage = vk::BufferUsageFlags::TRANSFER_DST;
                    Some(Buffer::host(device, bytes, usage, host)?)
                }
                None => None,
            };
            let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST;
            let (image, memory, bytes, view) = image(device, format, usage, COLOR, width, height)?;
            t.colors.push(Attachment {
                format,
                image,
                memory,
                bytes,
                view,
                readback,
            });
        }
        let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        (t.depth, t.depth_memory, t.depth_bytes, t.depth_view) =
            image(device, DEPTH_FORMAT, usage, DEPTH, width, height)?;
        Ok(t)
    }

    pub fn fits(&self, formats: &[(vk::Format, usize)], width: u32, height: u32) -> bool {
        let read = self.colors.iter().filter(|a| a.readback.is_some());
        let same = read.map(|a| a.format).eq(formats.iter().map(|f| f.0));
        same && (self.width, self.height) == (width, height)
    }

    /// Tightly packed rows, safe to read only between the frame's wait and its next submit.
    pub fn pixels(&mut self, index: usize) -> &mut [u8] {
        let readback = self.colors[index].readback.as_mut();
        readback.expect("only motion has no readback").bytes()
    }

    pub fn memory(&self) -> Memory {
        let mut memory = Memory {
            device: self.depth_bytes,
            host: 0,
        };
        for a in &self.colors {
            memory.device += a.bytes;
            if let Some(r) = &a.readback {
                memory += r.memory();
            }
        }
        memory
    }

    /// Each color is cleared to its value in `clears`, motion to none, the depth to the far plane.
    pub fn begin_rendering(&self, cb: vk::CommandBuffer, clears: &[vk::ClearColorValue]) {
        let d = &self.device;
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
        let none = vk::ClearColorValue::default();
        let attachments: Vec<_> = (self.colors.iter().zip(clears.iter().chain([&none])))
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
            .store_op(vk::AttachmentStoreOp::STORE)
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
        // SAFETY: `cb` is recording outside rendering, every handle was
        // created on `d`, and the info structs outlive the calls.
        unsafe {
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
            d.cmd_begin_rendering(cb, &rendering);
        }
    }

    /// Records after `end_rendering`: copies color attachment `index` row by row into `buffer`,
    /// for compute passes to read and write before `resume_rendering` or `copy_in`.
    pub fn copy_out(&self, cb: vk::CommandBuffer, index: usize, buffer: vk::Buffer) {
        let attachment_image = self.colors[index].image;
        let barrier =
            |src: (vk::PipelineStageFlags2, vk::AccessFlags2), dst: (_, _), layouts: [_; 2]| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(src.0)
                    .src_access_mask(src.1)
                    .dst_stage_mask(dst.0)
                    .dst_access_mask(dst.1)
                    .old_layout(layouts[0])
                    .new_layout(layouts[1])
                    .image(attachment_image)
                    .subresource_range(COLOR)
            };
        let attachment = (
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        let copy = (
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        );
        let to_copy = barrier(
            attachment,
            copy,
            [
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ],
        );
        let back = barrier(
            copy,
            (
                attachment.0,
                attachment.1 | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            ),
            [
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ],
        );
        let copied = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
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
        let d = &self.device;
        // SAFETY: `cb` is recording outside rendering, every handle was created on `d`, and
        // `buffer` holds `width`×`height` of the attachment's pixels (the caller's).
        unsafe {
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[to_copy]),
            );
            d.cmd_copy_image_to_buffer(
                cb,
                attachment_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                &[region],
            );
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default()
                    .memory_barriers(&[copied])
                    .image_memory_barriers(&[back]),
            );
        }
    }

    /// Records after compute passes that wrote `buffer`, `copy_out`'s of attachment `index`:
    /// copies it back into the attachment, as `readback` or `resume_rendering` then find it.
    pub fn copy_in(&self, cb: vk::CommandBuffer, index: usize, buffer: vk::Buffer) {
        let written = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
        let image = self.colors[index].image;
        let to_copy = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(COLOR);
        let back = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(COLOR);
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
        let d = &self.device;
        // SAFETY: `cb` is recording outside rendering, every handle was created on `d`, and
        // `buffer` holds `width`×`height` of the attachment's pixels (the caller's).
        unsafe {
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default()
                    .memory_barriers(&[written])
                    .image_memory_barriers(&[to_copy]),
            );
            d.cmd_copy_buffer_to_image(
                cb,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&[back]),
            );
        }
    }

    /// Renders on into what the last rendering stored, colors and depth alike.
    pub fn resume_rendering(&self, cb: vk::CommandBuffer) {
        let stages = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        let access = vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
        let stored = vk::MemoryBarrier2::default()
            .src_stage_mask(stages)
            .src_access_mask(access)
            .dst_stage_mask(stages)
            .dst_access_mask(
                access
                    | vk::AccessFlags2::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            );
        let attachments: Vec<_> = (self.colors.iter())
            .map(|a| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(a.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
            })
            .collect();
        let depth = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth_view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::DONT_CARE);
        let rendering = vk::RenderingInfo::default()
            .render_area(self.area())
            .layer_count(1)
            .color_attachments(&attachments)
            .depth_attachment(&depth);
        let d = &self.device;
        // SAFETY: `cb` is recording outside rendering, after a rendering of these attachments
        // that stored them; the info structs outlive the calls.
        unsafe {
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().memory_barriers(&[stored]),
            );
            d.cmd_begin_rendering(cb, &rendering);
        }
    }

    pub fn end_rendering(&self, cb: vk::CommandBuffer) {
        // SAFETY: `cb` is recording inside that rendering.
        unsafe { self.device.cmd_end_rendering(cb) };
    }

    /// Records after `end_rendering`; host-visible once the frame is waited for.
    pub fn readback(&self, cb: vk::CommandBuffer, read: &[bool]) {
        let d = &self.device;
        let read: Vec<(&Attachment, &Buffer)> = (self.colors.iter().zip(read))
            .filter(|(_, r)| **r)
            .filter_map(|(a, _)| Some((a, a.readback.as_ref()?)))
            .collect();
        let to_copy: Vec<_> = (read.iter())
            .map(|(a, _)| {
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
            .map(|(_, buffer)| {
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)
                    .buffer(buffer.raw)
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
        // SAFETY: `cb` is recording outside rendering, every handle was
        // created on `d`, and the info structs outlive the calls.
        unsafe {
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&to_copy),
            );
            for (a, buffer) in &read {
                d.cmd_copy_image_to_buffer(
                    cb,
                    a.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    buffer.raw,
                    &[region],
                );
            }
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().buffer_memory_barriers(&to_host),
            );
        }
    }

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
        // SAFETY: every handle is `d`'s or null, which destroy calls accept, and after waiting for
        // idle none is in use. The readback buffers drop with `colors`, after this.
        unsafe {
            let _ = d.device_wait_idle();
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

fn image(
    device: &gpu::Device,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    range: vk::ImageSubresourceRange,
    width: u32,
    height: u32,
) -> Result<(vk::Image, vk::DeviceMemory, u64, vk::ImageView), String> {
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
    // SAFETY (this and every call below): `d` is live, each info outlives its call, and each
    // handle was created on `d` above. On an error, the handles made so far are destroyed here.
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
        Ok(view) => Ok((image, memory, needs.size, view)),
        Err(e) => {
            unsafe {
                d.destroy_image(image, None);
                d.free_memory(memory, None);
            }
            Err(e)
        }
    }
}
