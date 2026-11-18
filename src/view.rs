//! One view's constants, lights, tiles and cull outputs. Every view is culled against
//! its own frustum into its own `View`, never reusing another's.

use ash::vk;

use crate::geometry::Addresses;
use crate::gpu::{self, Buffer};
use crate::light::{Light, Tiles};
use crate::transform::Camera;

/// `shaders/view.slang`'s `View`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Constants {
    camera: Camera,
    previous_view_projection: [[f32; 4]; 4],
    lights: vk::DeviceAddress,
    light_tiles: vk::DeviceAddress,
    tile_lights: vk::DeviceAddress,
    extent: [u32; 2],
    jitter: [f32; 2],
    pattern: u32,
    unused: u32,
    points: vk::DeviceAddress,
    normals: vk::DeviceAddress,
    smooth: vk::DeviceAddress,
    vertices: vk::DeviceAddress,
    texcoord: vk::DeviceAddress,
    vertex_texcoord: vk::DeviceAddress,
    tangents: vk::DeviceAddress,
    instances: vk::DeviceAddress,
    previous_instances: vk::DeviceAddress,
    previous_points: vk::DeviceAddress,
    owners: vk::DeviceAddress,
    boxes: vk::DeviceAddress,
    records: vk::DeviceAddress,
    visible: vk::DeviceAddress,
    drawn: vk::DeviceAddress,
    shadows: vk::DeviceAddress,
}

pub struct View {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    constants: Buffer,
    /// The lights, then the tiles, then the tiles' lights.
    lights: Buffer,
    staging: Buffer,
    records: Buffer,
    visible: Buffer,
    drawn: Buffer,
}

impl View {
    pub fn new(device: &gpu::Device, geometry: &Addresses, lights: u64) -> Result<Self, String> {
        let records = geometry.count as u64 * size_of::<vk::DrawIndexedIndirectCommand>() as u64;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        Ok(Self {
            device: device.raw.clone(),
            constants: Buffer::device_local(
                device,
                size_of::<Constants>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            )?,
            lights: Buffer::device_local(
                device,
                lights,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            )?,
            staging: Buffer::host(device, lights, vk::BufferUsageFlags::TRANSFER_SRC, host)?,
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

    pub fn fits(&self, geometry: &Addresses, lights: u64) -> bool {
        let records = geometry.count as u64 * size_of::<vk::DrawIndexedIndirectCommand>() as u64;
        self.records.size >= records
            && self.visible.size >= geometry.instance_count as u64 * 4
            && self.lights.size >= lights
    }

    /// Records outside rendering. `geometry` and the lights must fit; no GPU work may be pending.
    /// `pattern` is the frame's index into the shadow pattern's sequence (`Film::pattern`).
    /// `camera` is jittered by `jitter` (`film::jitter`); `previous_view_projection` is the frame before's
    /// view-projection, unjittered, which motion is measured from.
    /// `shadows` addresses `shaders/view.slang`'s `Shadow`s.
    #[allow(clippy::too_many_arguments)]
    pub fn begin(
        &mut self,
        cb: vk::CommandBuffer,
        camera: &Camera,
        jitter: [f64; 2],
        previous_view_projection: &[[f32; 4]; 4],
        geometry: &Addresses,
        lights: &[Light],
        tiles: &Tiles,
        extent: [u32; 2],
        pattern: u32,
        shadows: vk::DeviceAddress,
    ) {
        self.drawn.bytes().fill(0);
        let parts = [
            gpu::bytes(lights),
            gpu::bytes(&tiles.tiles),
            gpu::bytes(&tiles.lights),
        ];
        // Each part's offset is a multiple of 8, as its addresses need.
        let staging = self.staging.bytes();
        let mut offsets = [0; 3];
        let mut end = 0;
        for (part, offset) in parts.iter().zip(&mut offsets) {
            *offset = end;
            staging[end..end + part.len()].copy_from_slice(part);
            end += part.len();
        }
        let address = |i: usize| self.lights.address + offsets[i] as u64;
        let constants = Constants {
            camera: *camera,
            previous_view_projection: *previous_view_projection,
            lights: address(0),
            light_tiles: address(1),
            tile_lights: address(2),
            extent,
            jitter: jitter.map(|v| v as f32),
            pattern,
            unused: 0,
            points: geometry.points,
            normals: geometry.normals,
            smooth: geometry.smooth,
            vertices: geometry.vertices,
            texcoord: geometry.texcoord,
            vertex_texcoord: geometry.vertex_texcoord,
            tangents: geometry.tangents,
            instances: geometry.instances,
            previous_instances: geometry.previous_instances,
            previous_points: geometry.previous_points,
            owners: geometry.owners,
            boxes: geometry.boxes,
            records: self.records.address,
            visible: self.visible.address,
            drawn: self.drawn.address,
            shadows,
        };
        let bytes = gpu::bytes(std::slice::from_ref(&constants));
        let light_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: end as u64,
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
        // SAFETY: `cb` is recording outside rendering, buffers are from `d`, and the regions
        // lie inside them, as `fits` checked. Vulkan has no empty copies; tiles are never empty.
        unsafe {
            d.cmd_update_buffer(cb, self.constants.raw, 0, bytes);
            d.cmd_copy_buffer(cb, self.staging.raw, self.lights.raw, &[light_region]);
            if geometry.count > 0 {
                d.cmd_copy_buffer(cb, geometry.records, self.records.raw, &[region]);
            }
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().memory_barriers(&[written]),
            );
        }
    }

    pub fn constants(&self) -> vk::DeviceAddress {
        self.constants.address
    }

    pub fn records(&self) -> vk::Buffer {
        self.records.raw
    }

    pub fn memory(&self) -> gpu::Memory {
        let mut memory = self.constants.memory();
        memory += self.lights.memory();
        memory += self.staging.memory();
        memory += self.records.memory();
        memory += self.visible.memory();
        memory += self.drawn.memory();
        memory
    }

    /// The instances and triangles the last frame drew. No GPU work may be pending.
    pub fn drawn(&mut self) -> [u32; 2] {
        let b = self.drawn.bytes();
        [0, 1].map(|i| u32::from_ne_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
    }
}
