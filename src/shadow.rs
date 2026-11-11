//! Shadow maps: each shadowed distant light gets a plain `SIZE`² map, orthographic, drawn by
//! atomic min of light depth, so it needs no depth attachment. Its square covers the casters
//! the camera sees, so a receiver that casts nothing and lies outside them is lit.

use ash::vk;

use crate::cull::Cull;
use crate::geometry::{Bound, Geometry};
use crate::gpu::{self, Buffer};
use crate::light::{self, Light, Tiles};
use crate::transform::{Camera, Instance};
use crate::view::View;

/// Texels a side: 64 MB a map.
pub const SIZE: u32 = 4096;

/// `shaders/view.slang`'s `Shadow`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Shadow {
    pub view_projection: [[f32; 4]; 4],
    map: vk::DeviceAddress,
    size: u32,
    pub texel: f32,
}

/// A shadowed distant light, by its index among the view's lights.
pub struct Caster {
    pub light: usize,
    /// Its shadow-link bit, 0 when every mesh casts.
    pub shadow: u32,
}

struct Map {
    depth: Buffer,
    view: View,
    /// The light's link bit, for the cull.
    shadow: u32,
}

pub struct Shadows {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    table: Option<Buffer>,
    shadows: Vec<Shadow>,
    maps: Vec<Map>,
}

impl Shadows {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
        let d = &device.raw;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(size_of::<vk::DeviceAddress>() as u32)];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push);
        // SAFETY: `d` is live and `layout_info` outlives the call.
        let layout = unsafe { d.create_pipeline_layout(&layout_info, None) }
            .map_err(|e| format!("vkCreatePipelineLayout failed: {e}"))?;
        let mut s = Self {
            device: d.clone(),
            layout,
            pipeline: vk::Pipeline::null(),
            table: None,
            shadows: Vec::new(),
            maps: Vec::new(),
        };
        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: `d` is live and `module_info` outlives the call.
        let module = unsafe { d.create_shader_module(&module_info, None) }
            .map_err(|e| format!("vkCreateShaderModule failed: {e}"))?;
        let depth = vk::PipelineDepthStencilStateCreateInfo::default();
        let built = gpu::graphics(
            d,
            layout,
            module,
            [c"vs", c"fs"],
            &[],
            vk::Format::UNDEFINED,
            depth,
        );
        // SAFETY: pipeline creation is complete, so the module is no longer used.
        unsafe { d.destroy_shader_module(module, None) };
        s.pipeline = built?;
        Ok(s)
    }

    /// Fits each caster's map and points its light at it; the lights are then ready for
    /// the camera's view, whose shadows are at `table`.
    pub fn fit(
        &mut self,
        device: &gpu::Device,
        geometry: &Geometry,
        bound: &Bound,
        camera: &Camera,
        lights: &mut [Light],
        casters: &[Caster],
    ) -> Result<(), String> {
        self.shadows.clear();
        let mut kept = 0;
        for c in casters {
            let toward = lights[c.light].position;
            let Some((matrix, texel)) = fit(toward, camera, geometry.instances(), c.shadow) else {
                continue;
            };
            if self.maps.len() == kept {
                self.maps.push(Map::new(device, bound)?);
            }
            let map = &mut self.maps[kept];
            if !map.view.fits(bound, light::capacity(0, [1, 1])) {
                map.view = View::new(device, bound, light::capacity(0, [1, 1]))?;
            }
            map.shadow = c.shadow;
            lights[c.light].shadow = kept as u32;
            self.shadows.push(Shadow {
                view_projection: matrix,
                map: map.depth.address,
                size: SIZE,
                texel,
            });
            kept += 1;
        }
        self.maps.truncate(kept);
        let bytes = (kept.max(1) * size_of::<Shadow>()) as u64;
        if self.table.as_ref().is_none_or(|t| t.size < bytes) {
            self.table = None;
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
            self.table = Some(Buffer::device_local(device, bytes, usage)?);
        }
        Ok(())
    }

    /// The camera's view reads each light's `Shadow` here.
    pub fn table(&self) -> vk::DeviceAddress {
        self.table.as_ref().map_or(0, |t| t.address)
    }

    /// Records outside rendering, after `fit`, the maps the camera's view reads; no GPU work
    /// may be pending.
    pub fn record(&mut self, cb: vk::CommandBuffer, geometry: &Bound, cull: &Cull) {
        let (Some(table), false) = (&self.table, self.shadows.is_empty()) else {
            return;
        };
        let d = &self.device;
        // SAFETY: `Shadow` is plain `repr(C)` data without padding.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.shadows.as_ptr().cast::<u8>(),
                size_of_val(&*self.shadows),
            )
        };
        let cleared = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
        // SAFETY: `cb` is recording outside rendering; the table holds every shadow (`fit`)
        // and each map `SIZE`² words. The view's begin orders the table's update before
        // the shaders that read it.
        unsafe {
            d.cmd_update_buffer(cb, table.raw, 0, bytes);
            for map in &self.maps {
                d.cmd_fill_buffer(cb, map.depth.raw, 0, vk::WHOLE_SIZE, u32::MAX);
            }
            d.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().memory_barriers(&[cleared]),
            );
        }
        let stride = size_of::<Shadow>() as u64;
        for (i, (map, shadow)) in self.maps.iter_mut().zip(&self.shadows).enumerate() {
            let camera = Camera {
                view_projection: shadow.view_projection,
                eye: [0.0; 4],
            };
            let tiles = Tiles::new(&[], &camera, [1, 1]);
            // A light's view has no screen, so no tiles to speak of.
            let own = table.address + i as u64 * stride;
            map.view
                .begin(cb, &camera, geometry, &[], &tiles, [1, 1], own);
            cull.record(cb, geometry, &map.view, map.shadow);
            draw(d, cb, self.layout, self.pipeline, geometry, &map.view);
        }
        let drawn = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ);
        // SAFETY: `cb` is recording outside rendering.
        unsafe {
            d.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&[drawn]))
        };
    }

    pub fn memory(&self) -> gpu::Memory {
        let mut memory = gpu::Memory::default();
        for map in &self.maps {
            memory += map.depth.memory();
            memory += map.view.memory();
        }
        if let Some(t) = &self.table {
            memory += t.memory();
        }
        memory
    }
}

impl Drop for Shadows {
    fn drop(&mut self) {
        // SAFETY: both handles were created on `device`, and the owner's target waited
        // for idle before this drops (field order).
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

impl Map {
    fn new(device: &gpu::Device, bound: &Bound) -> Result<Self, String> {
        let words = SIZE as u64 * SIZE as u64;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        Ok(Self {
            depth: Buffer::device_local(device, words * 4, usage)?,
            view: View::new(device, bound, light::capacity(0, [1, 1]))?,
            shadow: 0,
        })
    }
}

/// Draws the culled casters with no attachments: the fragments write the map themselves.
fn draw(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    geometry: &Bound,
    view: &View,
) {
    if geometry.count == 0 {
        return;
    }
    let area = vk::Rect2D {
        offset: vk::Offset2D::default(),
        extent: vk::Extent2D {
            width: SIZE,
            height: SIZE,
        },
    };
    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: SIZE as f32,
        height: SIZE as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let rendering = vk::RenderingInfo::default()
        .render_area(area)
        .layer_count(1);
    let push = view.constants().to_ne_bytes();
    // SAFETY: `cb` is recording outside rendering, and the pipeline was built for rendering
    // with no attachments. The view's constants address live buffers, as large as the draws
    // read, and its shadow's map holds `SIZE`² words.
    unsafe {
        d.cmd_begin_rendering(cb, &rendering);
        d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
        d.cmd_set_viewport(cb, 0, &[viewport]);
        d.cmd_set_scissor(cb, 0, &[area]);
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        d.cmd_push_constants(cb, layout, stages, 0, &push);
        d.cmd_bind_index_buffer(cb, geometry.indices, 0, vk::IndexType::UINT32);
        d.cmd_draw_indexed_indirect(
            cb,
            view.records(),
            0,
            geometry.count,
            size_of::<vk::DrawIndexedIndirectCommand>() as u32,
        );
        d.cmd_end_rendering(cb);
    }
}

/// The orthographic world-to-clip rows over the light-space square that holds the boxes of
/// the casters the camera sees, deep enough for every caster, and a texel's width; `None`
/// when the camera sees no caster. Depth is 0 toward the light.
fn fit<'a>(
    toward: [f32; 3],
    camera: &Camera,
    instances: impl Iterator<Item = (&'a Instance, &'a [f32; 6])>,
    shadow: u32,
) -> Option<([[f32; 4]; 4], f32)> {
    let forward = toward.map(|v| -v);
    let up = if forward[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let right = normalize(cross(up, forward));
    let up = cross(forward, right);
    let axes = [right, up, forward];
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    let mut seen_lo = [f32::INFINITY; 2];
    let mut seen_hi = [f32::NEG_INFINITY; 2];
    for (instance, b) in instances {
        if instance.shadow() & shadow != shadow {
            continue;
        }
        let world = instance.world();
        let corners: [[f32; 3]; 8] = std::array::from_fn(|c| {
            let p = [0, 1, 2].map(|k| if c >> k & 1 == 1 { b[3 + k] } else { b[k] });
            world.map(|row| row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + row[3])
        });
        let seen = seen(&camera.view_projection, &corners);
        for p in &corners {
            let q = axes.map(|a| dot(a, *p));
            for k in 0..3 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
            if seen {
                for k in 0..2 {
                    seen_lo[k] = seen_lo[k].min(q[k]);
                    seen_hi[k] = seen_hi[k].max(q[k]);
                }
            }
        }
    }
    if seen_lo[0] > seen_hi[0] {
        return None;
    }
    let side = (seen_hi[0] - seen_lo[0]).max(seen_hi[1] - seen_lo[1]);
    // A texel more each way, so the outermost casters' edges fall inside.
    let texel = side.max(f32::MIN_POSITIVE) / (SIZE - 2) as f32;
    let half = side / 2.0 + texel;
    let centre = [0, 1].map(|k| (seen_lo[k] + seen_hi[k]) / 2.0);
    // Padded so the nearest and farthest casters are not clipped by rounding.
    let pad = (hi[2] - lo[2]).max(texel) * 1e-3;
    let (near, depth) = (lo[2] - pad, hi[2] - lo[2] + 2.0 * pad);
    let row =
        |a: [f32; 3], scale: f32, offset: f32| [a[0] * scale, a[1] * scale, a[2] * scale, offset];
    let matrix = [
        row(right, 1.0 / half, -centre[0] / half),
        row(up, 1.0 / half, -centre[1] / half),
        row(forward, 1.0 / depth, -near / depth),
        [0.0, 0.0, 0.0, 1.0],
    ];
    Some((matrix, texel))
}

/// As `shaders/cull.slang` decides it: false only when every corner is outside one plane.
fn seen(view_projection: &[[f32; 4]; 4], corners: &[[f32; 3]; 8]) -> bool {
    let mut inside = 0u32;
    for p in corners {
        let [x, y, z, w] = view_projection.map(|r| r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + r[3]);
        inside |= (x >= -w) as u32
            | ((x <= w) as u32) << 1
            | ((y >= -w) as u32) << 2
            | ((y <= w) as u32) << 3
            | ((z >= 0.0) as u32) << 4
            | ((z <= w) as u32) << 5;
    }
    inside == 0x3f
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt();
    a.map(|v| v / l)
}
