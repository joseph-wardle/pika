//! One view of the scene: what its passes read, the camera and every buffer
//! address, written once per frame into a buffer whose address is each
//! pass's push constant; and the draw records and visible instances the
//! cull pass fills for it. Every view (the camera now; shadow views and
//! probe faces later) is culled against its own frustum into its own
//! `View`, never reusing another's.

use ash::vk;

use crate::geometry::Bound;
use crate::gpu::{self, Buffer};
use crate::transform::Camera;

/// `shaders/view.slang`'s `View`.
#[repr(C)]
struct Constants {
    camera: Camera,
    points: vk::DeviceAddress,
    normals: vk::DeviceAddress,
    smooth: vk::DeviceAddress,
    vertices: vk::DeviceAddress,
    instances: vk::DeviceAddress,
    owners: vk::DeviceAddress,
    boxes: vk::DeviceAddress,
    records: vk::DeviceAddress,
    visible: vk::DeviceAddress,
    drawn: vk::DeviceAddress,
}

pub struct View {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    constants: Buffer,
    records: Buffer,
    visible: Buffer,
    /// The instances and triangles drawn, written by the cull pass.
    drawn: Buffer,
}

impl View {
    /// A view's buffers for `geometry`'s records and instances.
    pub fn new(device: &gpu::Device, geometry: &Bound) -> Result<Self, String> {
        let records = geometry.count as u64 * size_of::<vk::DrawIndexedIndirectCommand>() as u64;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        Ok(Self {
            device: device.raw.clone(),
            constants: Buffer::device_local(
                device,
                size_of::<Constants>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            )?,
            records: Buffer::device_local(
                device,
                records,
                vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )?,
            visible: Buffer::device_local(
                device,
                geometry.instance_count as u64 * 4,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?,
            drawn: Buffer::host(
                device,
                8,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                host,
            )?,
        })
    }

    /// Whether the buffers hold `geometry`'s records and instances.
    pub fn fits(&self, geometry: &Bound) -> bool {
        let records = geometry.count as u64 * size_of::<vk::DrawIndexedIndirectCommand>() as u64;
        self.records.size >= records && self.visible.size >= geometry.instance_count as u64 * 4
    }

    /// Records into `cb`, outside rendering, the start of the view's frame:
    /// its constants from `camera` and `geometry`, its records copied from
    /// `geometry`'s with no instances, and a barrier making both visible to
    /// its passes. `geometry` must fit, and no GPU work may be pending.
    pub fn begin(&mut self, cb: vk::CommandBuffer, camera: &Camera, geometry: &Bound) {
        self.drawn.bytes().fill(0);
        let constants = Constants {
            camera: *camera,
            points: geometry.points,
            normals: geometry.normals,
            smooth: geometry.smooth,
            vertices: geometry.vertices,
            instances: geometry.instances,
            owners: geometry.owners,
            boxes: geometry.boxes,
            records: self.records.address,
            visible: self.visible.address,
            drawn: self.drawn.address,
        };
        // SAFETY: `Constants` is plain `repr(C)` data without padding.
        let bytes = unsafe {
            std::slice::from_raw_parts((&raw const constants).cast::<u8>(), size_of::<Constants>())
        };
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: geometry.count as u64 * size_of::<vk::DrawIndexedIndirectCommand>() as u64,
        };
        let written = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::VERTEX_SHADER
                    | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            )
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
        let d = &self.device;
        // SAFETY: `cb` is recording outside rendering, every buffer was
        // created on `d`, and the region lies inside both record buffers, as
        // `fits` checked. Vulkan has no empty copies.
        unsafe {
            d.cmd_update_buffer(cb, self.constants.raw, 0, bytes);
            if geometry.count > 0 {
                d.cmd_copy_buffer(cb, geometry.records, self.records.raw, &[region]);
            }
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().memory_barriers(&[written]),
            );
        }
    }

    /// The address of the view's constants, for its passes.
    pub fn constants(&self) -> vk::DeviceAddress {
        self.constants.address
    }

    pub fn records(&self) -> vk::Buffer {
        self.records.raw
    }

    pub fn memory(&self) -> gpu::Memory {
        let mut memory = self.constants.memory();
        memory += self.records.memory();
        memory += self.visible.memory();
        memory += self.drawn.memory();
        memory
    }

    /// The instances and triangles the last frame drew. No GPU work may be
    /// pending.
    pub fn drawn(&mut self) -> [u32; 2] {
        let b = self.drawn.bytes();
        [0, 1].map(|i| u32::from_ne_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
    }
}
