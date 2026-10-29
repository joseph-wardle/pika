//! The cull pass: a compute pipeline that, for one view, keeps each mesh
//! instance whose box may be inside the view's frustum, and the buffers it
//! writes for that view. Every view (the camera now; shadow views and probe
//! faces later) is culled against its own frustum into its own `View`,
//! never reusing another's.

use ash::vk;

use crate::geometry::Bound;
use crate::gpu::{self, Buffer};
use crate::transform::Camera;

pub struct Cull {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    compute: gpu::Compute,
}

/// One view's draw records, with the instances the cull pass found in it,
/// and those instances, through which the mesh pass reads them.
pub struct View {
    records: Buffer,
    visible: Buffer,
    /// The instances and triangles drawn, written by the cull pass.
    drawn: Buffer,
}

/// `shaders/cull.slang`'s push constants.
#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    view_projection: [[f32; 4]; 4],
    instances: vk::DeviceAddress,
    owners: vk::DeviceAddress,
    boxes: vk::DeviceAddress,
    records: vk::DeviceAddress,
    visible: vk::DeviceAddress,
    drawn: vk::DeviceAddress,
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

    /// Records into `cb`, outside rendering, `view`'s records and visible
    /// instances from `camera`, and barriers making them visible to the mesh
    /// pass. `view` must fit `geometry`, and no GPU work may be pending.
    pub fn record(
        &self,
        cb: vk::CommandBuffer,
        camera: &Camera,
        geometry: &Bound,
        view: &mut View,
    ) {
        view.drawn.bytes().fill(0);
        if geometry.count == 0 {
            return;
        }
        let d = &self.device;
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: geometry.count as u64 * size_of::<vk::DrawIndexedIndirectCommand>() as u64,
        };
        let copied = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
        // SAFETY: `cb` is recording outside rendering, both buffers were
        // created on `d`, and the region lies inside each, as `View::fits` checked.
        unsafe {
            d.cmd_copy_buffer(cb, geometry.records, view.records.raw, &[region]);
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().memory_barriers(&[copied]),
            );
        }
        let instances = 0..geometry.instance_count;
        (self.compute).record(cb, &[instances], |first, count, stride| Push {
            view_projection: camera.view_projection,
            instances: geometry.instances,
            owners: geometry.owners,
            boxes: geometry.boxes,
            records: view.records.address,
            visible: view.visible.address,
            drawn: view.drawn.address,
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
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().memory_barriers(&[culled]),
            )
        };
    }
}

impl View {
    /// A view's buffers for `geometry`'s records and instances.
    pub fn new(device: &gpu::Device, geometry: &Bound) -> Result<Self, String> {
        let records = geometry.count as u64 * size_of::<vk::DrawIndexedIndirectCommand>() as u64;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        Ok(Self {
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

    pub fn records(&self) -> vk::Buffer {
        self.records.raw
    }

    pub fn visible(&self) -> vk::DeviceAddress {
        self.visible.address
    }

    /// The instances and triangles the last frame drew. No GPU work may be
    /// pending.
    pub fn drawn(&mut self) -> [u32; 2] {
        let b = self.drawn.bytes();
        [0, 1].map(|i| u32::from_ne_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
    }
}
