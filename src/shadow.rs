//! Virtual shadow maps, drawn by `shaders/shadow.slang`.
//!
//! A distant light gets a clipmap around the camera and a local light a cube of faces. Only the
//! tiles the camera's pixels read get a page from the shared pool. Pages hold only casters that
//! aren't moving. Pages a moving caster touches get a live copy each frame with the movers drawn
//! in, and shading reads that instead.

use std::mem::offset_of;

use ash::vk;

use crate::commands::Commands;
use crate::geometry::{Addresses, Geometry};
use crate::gpu::{self, Buffer};
use crate::light::{self, Caster, Light};
use crate::link;
use crate::target::Target;
use crate::transform::{Camera, Instance, invert};
use crate::view::View;

/// Shared with the shaders. A test checks they match.
const LEVELS: u32 = 16;
const TILES: u32 = 32;
const PAGE: u32 = 256;
const LOCAL_LEVELS: u32 = 6;
const FACES: u32 = 6;
const SPAN: u32 = TILES * PAGE;
const PAGE_BYTES: u64 = PAGE as u64 * PAGE as u64 * 4;
/// Must match `Owner` in `shaders/shadow.slang`.
const OWNER_BYTES: u64 = 12;
/// Draws a frame can list into pages, and as many again into live copies.
const CAPACITY: u32 = 1 << 19;
/// Past this many tile walks, the moving pass walks the tiles itself.
const WALKS: u32 = 1 << 18;
/// One `vkCmdUpdateBuffer` writes at most 64 KiB. Past that, every page is drawn again.
const MOVED: u32 = 65536 / 4;
const SPANS: [&str; 4] = ["shadow_list", "shadow_pages", "shadow_copy", "shadow_live"];
const STARTED: u32 = 0x8000_0000;
const STOPPED: u32 = 0x4000_0000;
/// Padding on a distant light's depth range, so casters that move a little keep their pages.
const DEPTH_PAD: f64 = 0.25;
const TILE_BYTES: u64 = (LEVELS * TILES * TILES) as u64 * 4;
const BOUND_BYTES: u64 = (LEVELS * FACES) as u64 * 32;
const LAST_BYTES: u64 = 32;
const STATE_BYTES: u64 = size_of::<State>() as u64;

/// Must match `Shadow` in `shaders/view.slang`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Shadow {
    light_space: [[f32; 4]; 3],
    origin: [[i32; 2]; LEVELS as usize],
    finest: i32,
    levels: u32,
    pixel: f32,
    link: u32,
    faces: u32,
    near: f32,
    far: f32,
    reset: u32,
    tiles: vk::DeviceAddress,
    pool: vk::DeviceAddress,
}

impl Shadow {
    /// Whether a page drawn for `self` serves `other`. The camera only changes which tiles are held.
    fn draws_as(&self, other: &Self) -> bool {
        self.light_space == other.light_space
            && self.finest == other.finest
            && self.levels == other.levels
            && self.link == other.link
            && self.faces == other.faces
            && self.near == other.near
            && self.far == other.far
    }
}

/// Must match `Constants` in `shaders/shadow.slang`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Constants {
    unproject: [[f32; 4]; 4],
    camera: vk::DeviceAddress,
    points: vk::DeviceAddress,
    vertices: vk::DeviceAddress,
    instances: vk::DeviceAddress,
    owners: vk::DeviceAddress,
    boxes: vk::DeviceAddress,
    records: vk::DeviceAddress,
    shadows: vk::DeviceAddress,
    pages: vk::DeviceAddress,
    tiles: vk::DeviceAddress,
    flags: vk::DeviceAddress,
    bounds: vk::DeviceAddress,
    depth: vk::DeviceAddress,
    page_owners: vk::DeviceAddress,
    state: vk::DeviceAddress,
    indirect: vk::DeviceAddress,
    draws: vk::DeviceAddress,
    pool: vk::DeviceAddress,
    free: vk::DeviceAddress,
    drawn: vk::DeviceAddress,
    copies: vk::DeviceAddress,
    last: vk::DeviceAddress,
    moved: vk::DeviceAddress,
    walks: vk::DeviceAddress,
    count: u32,
    pool_pages: u32,
    fresh: u32,
    frame: u32,
}

/// Must match `State` in `shaders/shadow.slang`.
#[repr(C)]
#[derive(Clone, Copy)]
struct State {
    page_draws: u32,
    held: u32,
    wanted: u32,
    dropped: u32,
    past: u32,
    drawn: u32,
    triangles: u32,
    live_draws: u32,
    freed: u32,
    taken: u32,
    lives: u32,
    walked: u32,
    marked: [u32; (LEVELS * LEVELS) as usize],
    parents: [u32; (LEVELS * LEVELS) as usize],
    both: [u32; LEVELS as usize],
    ranked: u32,
    room: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    constants: vk::DeviceAddress,
    first: u32,
    count: u32,
    stride: u32,
    pad: u32,
}

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Counts {
    /// Pages the camera's pixels asked for, before any level was dropped.
    pub wanted: u32,
    pub held: u32,
    pub drawn: u32,
    pub live: u32,
    pub draws: u32,
    pub triangles: u32,
}

struct Pool {
    pages: u32,
    texels: Buffer,
    owners: Buffer,
    free: Buffer,
    drawn: Buffer,
    copies: Buffer,
}

struct Maps {
    table: Buffer,
    pages: Buffer,
    tiles: Buffer,
    flags: Buffer,
    bounds: Buffer,
    depth: Buffer,
    last: Buffer,
}

impl Maps {
    fn new(
        device: &gpu::Device,
        shadows: u64,
        pixels: u64,
        instances: u64,
    ) -> Result<Self, String> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = |bytes| Buffer::device_local(device, bytes, usage);
        Ok(Self {
            table: buffer(shadows * size_of::<Shadow>() as u64)?,
            pages: buffer(shadows * TILE_BYTES)?,
            tiles: buffer(shadows * TILE_BYTES)?,
            flags: buffer(shadows * TILE_BYTES)?,
            bounds: buffer(shadows * BOUND_BYTES)?,
            depth: buffer(pixels * 4)?,
            last: buffer(instances * LAST_BYTES)?,
        })
    }

    fn holds(&self, shadows: u64, pixels: u64, instances: u64) -> bool {
        self.pages.size >= shadows * TILE_BYTES
            && self.depth.size >= pixels * 4
            && self.last.size >= instances * LAST_BYTES
    }
}

/// A shadow as last recorded, keyed by its light's id.
#[derive(Clone, Copy)]
struct Held {
    id: u64,
    shadow: Shadow,
    axes: [[f64; 3]; 3],
    depth: [f64; 2],
}

pub struct Shadows {
    // A copy of the owner's device table. The owner's device outlives this, by field order.
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    mark: gpu::Compute,
    count: gpu::Compute,
    choose: gpu::Compute,
    assign: gpu::Compute,
    release: gpu::Compute,
    rank: gpu::Compute,
    gather: gpu::Compute,
    moving: gpu::Compute,
    remember: gpu::Compute,
    clear: gpu::Compute,
    cast: gpu::Compute,
    cast_moving: gpu::Compute,
    walk: gpu::Compute,
    copy: gpu::Compute,
    pool: Option<Pool>,
    constants: Buffer,
    maps: Option<Maps>,
    moved: Buffer,
    walks: Buffer,
    /// The instances that were moving in the last recorded frame, which its pages leave out.
    movers: Vec<u32>,
    indirect: Buffer,
    draws: Buffer,
    state: Buffer,
    counts: Buffer,
    fitted: Vec<Held>,
    held: Vec<Held>,
    coarsest: [Option<i32>; 2],
    /// The pool or the maps are new, so every tile must let go of its page before the next record.
    clean: bool,
    frame: u32,
    unproject: [[f32; 4]; 4],
    extent: [u32; 2],
    recorded: bool,
    warned: [bool; 2],
}

impl Shadows {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
        let limits = device.limits();
        let most = (limits
            .max_framebuffer_width
            .min(limits.max_framebuffer_height))
        .min(limits.max_viewport_dimensions[0].min(limits.max_viewport_dimensions[1]));
        if most < SPAN {
            return Err(format!(
                "the device draws at most {most} pixels across; shadows need {SPAN}"
            ));
        }
        let d = &device.raw;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let compute = |name| gpu::Compute::new(device, spirv, name, size_of::<Push>());
        let draw_bytes = size_of::<vk::DrawIndexedIndirectCommand>() as u64;
        let indirect = storage | vk::BufferUsageFlags::INDIRECT_BUFFER;
        let copied = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let mut s = Self {
            device: d.clone(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            mark: compute(c"mark")?,
            count: compute(c"count")?,
            choose: compute(c"choose")?,
            assign: compute(c"assign")?,
            release: compute(c"release")?,
            rank: compute(c"rank")?,
            gather: compute(c"gather")?,
            moving: compute(c"moving")?,
            remember: compute(c"remember")?,
            clear: compute(c"clear")?,
            cast: compute(c"cast")?,
            cast_moving: compute(c"cast_moving")?,
            walk: compute(c"walk")?,
            copy: compute(c"copy")?,
            pool: None,
            constants: Buffer::device_local(
                device,
                size_of::<Constants>() as u64,
                storage | vk::BufferUsageFlags::TRANSFER_DST,
            )?,
            maps: None,
            moved: Buffer::device_local(
                device,
                u64::from(MOVED) * 4,
                storage | vk::BufferUsageFlags::TRANSFER_DST,
            )?,
            walks: Buffer::device_local(device, u64::from(WALKS) * 16, storage)?,
            movers: Vec::new(),
            indirect: Buffer::device_local(device, 2 * CAPACITY as u64 * draw_bytes, indirect)?,
            draws: Buffer::device_local(device, 2 * CAPACITY as u64 * 16, storage)?,
            state: Buffer::device_local(device, STATE_BYTES, indirect | copied)?,
            counts: Buffer::host(
                device,
                STATE_BYTES,
                vk::BufferUsageFlags::TRANSFER_DST,
                host,
            )?,
            fitted: Vec::new(),
            held: Vec::new(),
            coarsest: [None; 2],
            clean: true,
            frame: 0,
            unproject: [[0.0; 4]; 4],
            extent: [0; 2],
            recorded: false,
            warned: [false; 2],
        };
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(size_of::<Push>() as u32)];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push);
        // SAFETY: the device is alive and `layout_info` outlives the call.
        s.layout = unsafe { d.create_pipeline_layout(&layout_info, None) }
            .map_err(|e| format!("vkCreatePipelineLayout failed: {e}"))?;
        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: the device is alive and `module_info` outlives the call.
        let module = unsafe { d.create_shader_module(&module_info, None) }
            .map_err(|e| format!("vkCreateShaderModule failed: {e}"))?;
        let depth = vk::PipelineDepthStencilStateCreateInfo::default();
        let built = gpu::graphics(
            d,
            s.layout,
            module,
            [c"vs", c"fs"],
            &[],
            vk::Format::UNDEFINED,
            depth,
        );
        // SAFETY: the pipeline is built, so nothing uses the shader module any more.
        unsafe { d.destroy_shader_module(module, None) };
        s.pipeline = built?;
        Ok(s)
    }

    /// Allocates a pool of `megabytes` plus as many live copies. Call it before the textures are
    /// fitted to the memory budget.
    pub fn reserve(&mut self, device: &gpu::Device, megabytes: u32) -> Result<(), String> {
        let pages = megabytes.saturating_mul(4);
        if self.pool.as_ref().is_some_and(|p| p.pages == pages) {
            return Ok(());
        }
        if pages == 0 {
            return Err("pika:shadowPool is 0 MB; shadows need at least one 0.25 MB page".into());
        }
        // The shaders index the pool and its live copies with 32-bit texel offsets.
        if u64::from(pages) * 2 * u64::from(PAGE * PAGE) > 1 << 32 {
            return Err(format!(
                "pika:shadowPool is {megabytes} MB, but Pika addresses at most {} MB",
                (1u64 << 32) / (2 * u64::from(PAGE * PAGE)) / 4
            ));
        }
        // Free the old pool before allocating the new one.
        self.pool = None;
        self.clean = true;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let filled = storage | vk::BufferUsageFlags::TRANSFER_DST;
        let list = |device| Buffer::device_local(device, pages as u64 * 4, storage);
        self.pool = Some(Pool {
            pages,
            texels: Buffer::device_local(device, 2 * pages as u64 * PAGE_BYTES, storage)?,
            owners: Buffer::device_local(device, pages as u64 * OWNER_BYTES, filled)?,
            free: list(device)?,
            drawn: list(device)?,
            copies: list(device)?,
        });
        Ok(())
    }

    /// Fits each distant light's clipmap to the camera and each local light's cube to the light.
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &mut self,
        device: &gpu::Device,
        geometry: &Geometry,
        addresses: &Addresses,
        camera: &Camera,
        projection: &[f64; 16],
        extent: [u32; 2],
        lights: &mut [Light],
        casters: &[Caster],
    ) -> Result<(), String> {
        let pool = (self.pool.as_ref()).ok_or("shadows fit before their pool was reserved")?;
        let unproject = invert(&camera.view_projection.map(|r| r.map(f64::from)))
            .ok_or("the camera's view-projection matrix is singular")?;
        let centre = match camera.eye {
            [x, y, z, 1.0] => [x, y, z].map(f64::from),
            _ => {
                let h = unproject.map(|r| r[2] * 0.5 + r[3]);
                [h[0] / h[3], h[1] / h[3], h[2] / h[3]]
            }
        };
        // Hydra's projection matrix is OpenGL's, for row vectors. Clip w is the view depth for a
        // perspective camera and 1 for an orthographic one, and element [5] scales view y to clip y.
        let pixel = 2.0 / (extent[1] as f64 * projection[5].abs());
        let near = match projection[15] {
            0.0 => projection[14] / (projection[10] - 1.0),
            _ => 1.0,
        };
        let (distant, local): (Vec<_>, Vec<_>) = casters
            .iter()
            .partition(|c| lights[c.light].shape == light::DISTANT);
        let held = |id| self.held.iter().find(|h| h.id == id);
        let mut fitted = Vec::new();
        let mut need = i32::MIN;
        for c in distant {
            let toward = lights[c.light].position.map(f64::from);
            let instances = geometry.instances();
            let Some((axes, depth, reach)) = light_space(toward, centre, instances, c.link) else {
                continue;
            };
            // A level's tiles reach 15 of its 32 tiles from its centre, which snaps to a tile.
            let level = (reach * TILES as f64 / 15.0)
                .max(f64::MIN_POSITIVE)
                .log2()
                .ceil();
            need = need.max(level as i32);
            let depth = depth_span(held(c.id).map(|h| (h.axes, h.depth)), axes, depth);
            fitted.push((c, axes, depth));
        }
        // Every distant shadow gets the same levels, which keeps a level's pages the same size.
        let [coarsest, finest] = clip_levels(self.coarsest[1], need, near, pixel);
        self.coarsest[0] = (!fitted.is_empty()).then_some(coarsest);
        let levels = coarsest.saturating_sub(finest) as u32 + 1;
        let count = (fitted.len() + local.len()).max(1) as u64;
        let pixels = u64::from(extent[0]) * u64::from(extent[1]);
        let instances = u64::from(addresses.instance_count).max(1);
        if !(self.maps.as_ref()).is_some_and(|m| m.holds(count, pixels, instances)) {
            // Free the old maps before allocating the new ones.
            self.maps = None;
            self.maps = Some(Maps::new(device, count, pixels, instances)?);
            self.clean = true;
        }
        let tiles = self.maps.as_ref().map_or(0, |m| m.tiles.address);
        let shadow = |j: usize, rows: [[f64; 4]; 3], c: &Caster| Shadow {
            light_space: rows.map(|r| r.map(|v| v as f32)),
            origin: [[0; 2]; LEVELS as usize],
            finest: 0,
            levels: LOCAL_LEVELS,
            pixel: pixel as f32,
            link: c.link,
            faces: 0,
            near: 0.0,
            far: 0.0,
            reset: 0,
            tiles: tiles + j as u64 * TILE_BYTES,
            pool: pool.texels.address,
        };
        let mut made = Vec::new();
        for (c, axes, depth) in fitted {
            let span = depth[1] - depth[0];
            let row = |a: [f64; 3], scale: f64, offset: f64| {
                [a[0] * scale, a[1] * scale, a[2] * scale, offset]
            };
            let rows = [
                row(axes[0], 1.0, 0.0),
                row(axes[1], 1.0, 0.0),
                row(axes[2], 1.0 / span, -depth[0] / span),
            ];
            let across = [0, 1].map(|k| dot3(&rows[k], centre));
            let origin = std::array::from_fn(|i| {
                let width = f64::from(finest + i as i32).exp2() / TILES as f64;
                across.map(|v| (v / width).floor() as i32 - (TILES / 2) as i32)
            });
            lights[c.light].shadow = made.len() as u32;
            let s = Shadow {
                origin,
                finest,
                levels,
                ..shadow(made.len(), rows, c)
            };
            made.push(Held {
                id: c.id,
                shadow: s,
                axes,
                depth,
            });
        }
        for c in local {
            let l = &lights[c.light];
            let one_sided = l.shape == light::RECT || l.shape == light::DISK;
            let s = Shadow {
                faces: if one_sided { 5 } else { 6 },
                // A caster inside the light's own shape would hide all of it.
                near: l.radius.max(l.reach * 1e-4),
                far: l.reach,
                ..shadow(made.len(), local_space(l), c)
            };
            lights[c.light].shadow = made.len() as u32;
            made.push(Held {
                id: c.id,
                shadow: s,
                axes: [[0.0; 3]; 3],
                depth: [0.0; 2],
            });
        }
        self.fitted = made;
        self.unproject = unproject.map(|r| r.map(|v| v as f32));
        self.extent = extent;
        Ok(())
    }

    pub fn table(&self) -> vk::DeviceAddress {
        self.maps.as_ref().map_or(0, |m| m.table.address)
    }

    /// Records the shadow passes, after `fit` and the depth prepass. No GPU work may be pending.
    /// `moved` lists the instances that moved since the last frame, ascending, or is `None` when the
    /// draws were rewritten, which redraws every page.
    pub fn record(
        &mut self,
        commands: &mut Commands,
        geometry: &Addresses,
        camera: &View,
        target: &Target,
        moved: Option<&[u32]>,
    ) -> Result<(), String> {
        self.recorded = false;
        let (Some(pool), Some(maps)) = (&self.pool, &self.maps) else {
            return Self::skip(commands);
        };
        if self.fitted.is_empty() || geometry.count == 0 {
            return Self::skip(commands);
        }
        self.recorded = true;
        let cb = commands.buffer();
        // 0 marks an instance that hasn't moved since a fresh frame.
        self.frame = self.frame.wrapping_add(1).max(1);
        // Tiles are stored per shadow, so pages are kept only while the same lights cast in the same
        // order.
        let ids = |h: &[Held]| h.iter().map(|h| h.id).collect::<Vec<_>>();
        let clean = self.clean || ids(&self.fitted) != ids(&self.held);
        let faces = (self.fitted.iter())
            .map(|h| h.shadow.levels * h.shadow.faces.max(1))
            .sum();
        // Instances are only added by rewriting the draws, which always comes with `moved` as `None`.
        let listed = moved
            .filter(|_| !clean)
            .map(|m| (m, moved_entries(m, &self.movers)))
            .filter(|(_, e)| e.len() <= MOVED as usize);
        let fresh = listed.is_none();
        let (moving, moved) = listed.unwrap_or((&[], Vec::new()));
        self.movers = moving.to_vec();
        let moved = &moved[..];
        let shadows: Vec<_> = (self.fitted.iter().enumerate())
            .map(|(j, h)| {
                let kept = (self.held.get(j)).is_some_and(|o| o.shadow.draws_as(&h.shadow));
                Shadow {
                    reset: u32::from(fresh || !kept),
                    ..h.shadow
                }
            })
            .collect();
        let constants = Constants {
            unproject: self.unproject,
            camera: camera.constants(),
            points: geometry.points,
            vertices: geometry.vertices,
            instances: geometry.instances,
            owners: geometry.owners,
            boxes: geometry.boxes,
            records: geometry.record_words,
            shadows: maps.table.address,
            pages: maps.pages.address,
            tiles: maps.tiles.address,
            flags: maps.flags.address,
            bounds: maps.bounds.address,
            depth: maps.depth.address,
            page_owners: pool.owners.address,
            state: self.state.address,
            indirect: self.indirect.address,
            draws: self.draws.address,
            pool: pool.texels.address,
            free: pool.free.address,
            drawn: pool.drawn.address,
            copies: pool.copies.address,
            last: maps.last.address,
            moved: self.moved.address,
            walks: self.walks.address,
            count: shadows.len() as u32,
            pool_pages: pool.pages,
            fresh: u32::from(fresh),
            frame: self.frame,
        };
        let constants = gpu::bytes(std::slice::from_ref(&constants));
        let (table, moved_bytes) = (gpu::bytes(&shadows), gpu::bytes(moved));
        let tile_bytes = shadows.len() as u64 * TILE_BYTES;
        let d = &self.device;
        // SAFETY: recording outside a render pass. `fit` and this function sized every buffer for these
        // shadows and instances, and `moved` holds at most `MOVED`.
        unsafe {
            d.cmd_update_buffer(cb, self.constants.raw, 0, constants);
            d.cmd_update_buffer(cb, maps.table.raw, 0, table);
            d.cmd_fill_buffer(cb, self.state.raw, 0, STATE_BYTES, 0);
            if !moved_bytes.is_empty() {
                d.cmd_update_buffer(cb, self.moved.raw, 0, moved_bytes);
            }
            if clean {
                d.cmd_fill_buffer(cb, maps.pages.raw, 0, tile_bytes, !0);
                d.cmd_fill_buffer(cb, pool.owners.raw, 0, pool.pages as u64 * OWNER_BYTES, !0);
            }
            d.cmd_fill_buffer(cb, maps.flags.raw, 0, tile_bytes, 0);
            let bound_bytes = shadows.len() as u64 * BOUND_BYTES;
            d.cmd_fill_buffer(cb, maps.bounds.raw, 0, bound_bytes, !0);
        }
        self.clean = false;
        self.held = std::mem::take(&mut self.fitted);
        self.coarsest[1] = self.coarsest[0];
        target.copy_out(cb, 1, maps.depth.raw);
        let written = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
        barrier(d, cb, written);
        let push = |first, count, stride| Push {
            constants: self.constants.address,
            first,
            count,
            stride,
            pad: 0,
        };
        let computed = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
        let [w, h] = self.extent;
        let pixels = 0..w * h;
        let entries = 0..shadows.len() as u32 * LEVELS * TILES * TILES;
        let pool_pages = 0..pool.pages;
        let texels = 0..pool.pages * PAGE * PAGE;
        let casting = 0..geometry.instance_count * LEVELS * shadows.len() as u32;
        let walks = 0..walk_bound(moved, faces).min(WALKS.into()) as u32 * 64;
        let entries_moved = 0..moved.len() as u32 * LEVELS * shadows.len() as u32;
        let remembered = 0..if fresh {
            geometry.instance_count
        } else {
            moving.len() as u32
        };
        let thread = 0..1;
        self.mark.record(cb, &[pixels], push);
        barrier(d, cb, computed);
        self.count.record(cb, std::slice::from_ref(&entries), push);
        barrier(d, cb, computed);
        self.choose.record(cb, std::slice::from_ref(&thread), push);
        barrier(d, cb, computed);
        self.release
            .record(cb, std::slice::from_ref(&entries), push);
        barrier(d, cb, computed);
        // A single item is one group, and `rank` needs whole groups.
        self.rank.record(cb, &[thread], push);
        barrier(d, cb, computed);
        self.gather.record(cb, &[pool_pages], push);
        barrier(d, cb, computed);
        self.assign.record(cb, &[entries], push);
        barrier(d, cb, computed);
        self.moving
            .record(cb, std::slice::from_ref(&entries_moved), push);
        barrier(d, cb, computed);
        self.walk.record(cb, &[walks], push);
        barrier(d, cb, computed);
        // After `moving`, which reads the boxes this replaces, and before the casts, which read the frame
        // it stamps.
        self.remember.record(cb, &[remembered], push);
        barrier(d, cb, computed);
        // These three write to separate memory, and the casts count with atomics, so they need no
        // barrier between them.
        self.clear.record(cb, std::slice::from_ref(&texels), push);
        self.cast.record(cb, &[casting], push);
        self.cast_moving.record(cb, &[entries_moved], push);
        let listed = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::DRAW_INDIRECT
                    | vk::PipelineStageFlags2::VERTEX_SHADER
                    | vk::PipelineStageFlags2::FRAGMENT_SHADER
                    | vk::PipelineStageFlags2::COPY,
            )
            .dst_access_mask(
                vk::AccessFlags2::INDIRECT_COMMAND_READ
                    | vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE
                    | vk::AccessFlags2::TRANSFER_READ,
            );
        barrier(d, cb, listed);
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: STATE_BYTES,
        };
        // SAFETY: recording outside a render pass, and both buffers hold a `State`.
        unsafe { d.cmd_copy_buffer(cb, self.state.raw, self.counts.raw, &[region]) };
        let to_host = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ);
        barrier(d, cb, to_host);
        commands.mark(SPANS[0])?;
        self.draw(cb, geometry, 0, offset_of!(State, page_draws));
        commands.mark(SPANS[1])?;
        let pages_drawn = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ);
        barrier(d, cb, pages_drawn);
        self.copy.record(cb, &[texels], push);
        commands.mark(SPANS[2])?;
        let copied = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
        barrier(d, cb, copied);
        self.draw(cb, geometry, CAPACITY, offset_of!(State, live_draws));
        commands.mark(SPANS[3])?;
        let drawn = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ);
        barrier(d, cb, drawn);
        Ok(())
    }

    /// Records `record`'s spans with nothing in them, so every frame reports the same spans.
    pub fn skip(commands: &mut Commands) -> Result<(), String> {
        SPANS.iter().try_for_each(|m| commands.mark(m))
    }

    /// Draws the listed casters from draw `first` on, as many as the state word at `counted` says.
    /// There are no attachments. The fragment shader writes the pool itself.
    fn draw(&self, cb: vk::CommandBuffer, geometry: &Addresses, first: u32, counted: usize) {
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: SPAN,
                height: SPAN,
            },
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: SPAN as f32,
            height: SPAN as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let rendering = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1);
        let push = Push {
            constants: self.constants.address,
            first: 0,
            count: 0,
            stride: 0,
            pad: 0,
        };
        let push = gpu::bytes(std::slice::from_ref(&push));
        let stride = size_of::<vk::DrawIndexedIndirectCommand>() as u32;
        let d = &self.device;
        // SAFETY: recording outside a render pass, and `new` checked the device can draw `SPAN` texels
        // across with no attachments. Each list holds `CAPACITY` draws.
        unsafe {
            d.cmd_begin_rendering(cb, &rendering);
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            d.cmd_set_viewport(cb, 0, &[viewport]);
            d.cmd_set_scissor(cb, 0, &[area]);
            let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
            d.cmd_push_constants(cb, self.layout, stages, 0, push);
            d.cmd_bind_index_buffer(cb, geometry.indices, 0, vk::IndexType::UINT32);
            d.cmd_draw_indexed_indirect_count(
                cb,
                self.indirect.raw,
                u64::from(first) * u64::from(stride),
                self.state.raw,
                counted as u64,
                CAPACITY,
                stride,
            );
            d.cmd_end_rendering(cb);
        }
    }

    /// The last frame's counts, once its GPU work is done. Warns once if pages or draws ran out.
    pub fn counts(&mut self) -> Counts {
        if !self.recorded {
            return Counts::default();
        }
        // SAFETY: `counts` holds a `State`, which is plain `u32`s, so any bytes are valid.
        let s = unsafe {
            self.counts
                .bytes()
                .as_ptr()
                .cast::<State>()
                .read_unaligned()
        };
        let State {
            held: after,
            wanted,
            dropped,
            past,
            drawn,
            triangles,
            lives: live,
            ..
        } = s;
        let draws = s.page_draws.min(CAPACITY) + s.live_draws.min(CAPACITY);
        let pages = self.pool.as_ref().map_or(0, |p| p.pages);
        if wanted > pages && !self.warned[0] {
            self.warned[0] = true;
            let levels = self.held.iter().map(|h| h.shadow.levels).max().unwrap_or(0);
            let consequence = if after > pages {
                "some go unshadowed".to_owned()
            } else {
                // The coarsest level is never dropped.
                let n = levels.saturating_sub(1).saturating_sub(dropped);
                format!("the farthest {n} of their {levels} levels drop a LOD")
            };
            eprintln!(
                "pika: warning: shadows want {wanted} pages and pika:shadowPool holds {pages}; \
                 {consequence}"
            );
        }
        if past > 0 && !self.warned[1] {
            self.warned[1] = true;
            eprintln!(
                "pika: warning: shadow casters met levels to draw {draws} times, past the \
                 {CAPACITY} draws Pika lists; some shadows are missing"
            );
        }
        Counts {
            wanted,
            held: after.min(pages),
            drawn,
            live,
            draws,
            triangles,
        }
    }

    pub fn memory(&self) -> gpu::Memory {
        let mut memory = self.constants.memory();
        if let Some(m) = &self.maps {
            for b in [
                &m.table, &m.pages, &m.tiles, &m.flags, &m.bounds, &m.depth, &m.last,
            ] {
                memory += b.memory();
            }
        }
        if let Some(p) = &self.pool {
            for b in [&p.texels, &p.owners, &p.free, &p.drawn, &p.copies] {
                memory += b.memory();
            }
        }
        for b in [
            &self.indirect,
            &self.draws,
            &self.state,
            &self.counts,
            &self.moved,
            &self.walks,
        ] {
            memory += b.memory();
        }
        memory
    }
}

impl Drop for Shadows {
    fn drop(&mut self) {
        // SAFETY: both handles were created on this device or are null, and the owner's target waited
        // for the device to go idle before this drops, by field order.
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

fn barrier(d: &ash::Device, cb: vk::CommandBuffer, b: vk::MemoryBarrier2) {
    // SAFETY: the command buffer is recording outside a render pass.
    unsafe { d.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&[b])) };
}

/// Lists instances moving `now`, marked `STARTED` if they weren't moving `before`, then those
/// that stopped, marked `STOPPED`. Both inputs are ascending.
fn moved_entries(now: &[u32], before: &[u32]) -> Vec<u32> {
    let started = now.iter().map(|&i| match before.binary_search(&i) {
        Ok(_) => i,
        Err(_) => i | STARTED,
    });
    let stopped = (before.iter())
        .filter(|i| now.binary_search(i).is_err())
        .map(|&i| i | STOPPED);
    started.chain(stopped).collect()
}

/// The depth range a distant light's z covers. It is kept while the axes stay and the casters
/// fit in `held`, so their pages are kept too.
fn depth_span(
    held: Option<([[f64; 3]; 3], [f64; 2])>,
    axes: [[f64; 3]; 3],
    depth: [f64; 2],
) -> [f64; 2] {
    match held {
        Some((a, h)) if a == axes && h[0] <= depth[0] && depth[1] <= h[1] => h,
        _ => {
            let pad = (depth[1] - depth[0]).max(1e-6) * DEPTH_PAD;
            [depth[0] - pad, depth[1] + pad]
        }
    }
}

/// The distant clipmaps' coarsest and finest levels, as log2 of their width. The coarsest holds
/// back by one level so pages survive camera moves. The finest spans one pixel at `near`.
fn clip_levels(held: Option<i32>, need: i32, near: f64, pixel: f64) -> [i32; 2] {
    let coarsest = match held {
        Some(c) if c >= need && c <= need.saturating_add(1) => c,
        _ => need,
    };
    let want = (near.max(f64::MIN_POSITIVE) * pixel * f64::from(SPAN)).log2();
    let finest = (want.floor() as i32).clamp(coarsest.saturating_sub(LEVELS as i32 - 1), coarsest);
    [coarsest, finest]
}

/// The most tile walks `moved` can list: one per shadow face for each entry, and two for a start.
fn walk_bound(moved: &[u32], faces: u32) -> u64 {
    let walks: u64 = (moved.iter())
        .map(|e| 1 + u64::from(e & STARTED != 0))
        .sum();
    walks * u64::from(faces)
}

/// A distant light's axes, with z pointing away from it, the casters' depth along z, and how far
/// any mesh reaches from `centre`. `None` when nothing casts.
fn light_space<'a>(
    toward: [f64; 3],
    centre: [f64; 3],
    instances: impl Iterator<Item = (&'a Instance, &'a [f32; 6])>,
    link: u32,
) -> Option<([[f64; 3]; 3], [f64; 2], f64)> {
    let forward = toward.map(|v| -v);
    let up = if forward[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let right = normalize(cross(up, forward));
    let up = cross(forward, right);
    let axes = [right, up, forward];
    let c = axes.map(|a| dot(a, centre));
    let mut depth = [f64::INFINITY, f64::NEG_INFINITY];
    let mut reach: f64 = 0.0;
    for (instance, b) in instances {
        let world = instance.world().map(|r| r.map(f64::from));
        let casts = link::reaches(link, instance.shadow_link());
        for corner in 0..8 {
            let p = [0, 1, 2].map(|k| f64::from(b[k + 3 * (corner >> k & 1)]));
            let p = world.map(|row| row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + row[3]);
            let q = axes.map(|a| dot(a, p));
            reach = reach.max((q[0] - c[0]).abs()).max((q[1] - c[1]).abs());
            if casts {
                depth = [depth[0].min(q[2]), depth[1].max(q[2])];
            }
        }
    }
    if depth[0] > depth[1] {
        return None;
    }
    Some((axes, depth, reach))
}

/// World-to-light rows for a local light: its frame about its centre. For a rect or disk light,
/// z points out of the lit side, so its cube needs no -z face.
fn local_space(l: &Light) -> [[f64; 4]; 3] {
    let c = l.position.map(f64::from);
    let axes = if l.shape == light::RECT || l.shape == light::DISK {
        let (u, v) = (l.u.map(f64::from), l.v.map(f64::from));
        let z = normalize(cross(u, v));
        let x = normalize(u);
        [x, cross(z, x), z]
    } else {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    };
    axes.map(|a| [a[0], a[1], a[2], -dot(a, c)])
}

fn dot3(row: &[f64; 4], p: [f64; 3]) -> f64 {
    row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + row[3]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let l = dot(a, a).sqrt();
    a.map(|v| v / l)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pages hold only the casters that aren't moving, so an instance's start and stop must each be
    /// listed once, and one moving in both frames must be listed as neither.
    #[test]
    fn moved_entries_mark_starts_and_stops() {
        let before = [2, 5, 9];
        let now = [1, 5, 7];
        assert_eq!(
            moved_entries(&now, &before),
            [1 | STARTED, 5, 7 | STARTED, 2 | STOPPED, 9 | STOPPED]
        );
        assert_eq!(moved_entries(&now, &now), now);
        // A fresh frame's pages hold every caster, so everything that moves after it starts.
        assert_eq!(moved_entries(&[3], &[]), [3 | STARTED]);
        // Negative control: with the lists swapped, the starts become stops.
        assert_ne!(moved_entries(&before, &now), moved_entries(&now, &before));
    }

    /// Reads a shader's `static const uint name = value;`, where value is a number or `a << b`.
    fn slang_const(source: &str, name: &str) -> Option<u32> {
        let number = |v: &str| {
            let v = v.trim();
            match v.strip_prefix("0x") {
                Some(hex) => u32::from_str_radix(hex, 16).ok(),
                None => v.parse().ok(),
            }
        };
        source.lines().find_map(|line| {
            let line = line.trim().strip_prefix("static const uint ")?;
            let (n, value) = line.strip_suffix(';')?.split_once('=')?;
            if n.trim() != name {
                return None;
            }
            match value.split_once("<<") {
                Some((a, b)) => Some(number(a)? << number(b)?),
                None => number(value),
            }
        })
    }

    /// The shaders keep their own copies of these. One out of step reads and writes the wrong
    /// memory without any error.
    #[test]
    fn the_shaders_mirror_the_constants() {
        let view = include_str!("../shaders/view.slang");
        let shadow = include_str!("../shaders/shadow.slang");
        let mirrored = [
            (view, "LEVELS", LEVELS),
            (view, "TILES", TILES),
            (view, "PAGE", PAGE),
            (shadow, "FACES", FACES),
            (shadow, "CAPACITY", CAPACITY),
            (shadow, "WALKS", WALKS),
            (shadow, "STARTED", STARTED),
            (shadow, "STOPPED", STOPPED),
        ];
        for (source, name, value) in mirrored {
            assert_eq!(slang_const(source, name), Some(value), "{name}");
        }
        // Negative controls: a constant out of step, and one the shaders don't define.
        let stepped = "static const uint TILES = 1 << 4;";
        assert_ne!(slang_const(stepped, "TILES"), Some(TILES));
        assert_eq!(slang_const(view, "MOVED"), None);
    }

    /// The walk pass's groups are sized by this bound, so it must count both walks of a start.
    #[test]
    fn walk_bound_counts_a_starts_two_walks() {
        let faces = 16 + 6 * 6;
        assert_eq!(walk_bound(&[], faces), 0);
        assert_eq!(walk_bound(&[4, 7 | STARTED, 9 | STOPPED], faces), 4 * 52);
        // Negative control: a stop walks once, as a moving instance does.
        assert_ne!(
            walk_bound(&[9 | STOPPED], faces),
            walk_bound(&[9 | STARTED], faces)
        );
    }

    /// A distant light's depth range must stay while the casters fit and change when they leave.
    #[test]
    fn a_held_depth_span_keeps_while_it_holds_the_casters() {
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let fresh = depth_span(None, axes, [2.0, 6.0]);
        assert_eq!(fresh, [1.0, 7.0]);
        // Casters moving a little within the padding keep it.
        assert_eq!(depth_span(Some((axes, fresh)), axes, [1.5, 6.5]), fresh);
        // Negative controls: moving past its end, or turning the light, gives a new range.
        assert_eq!(
            depth_span(Some((axes, fresh)), axes, [2.0, 8.0]),
            [0.5, 9.5]
        );
        let turned = [axes[1], axes[0], axes[2]];
        assert_eq!(
            depth_span(Some((axes, fresh)), turned, [2.5, 5.5]),
            [1.75, 6.25]
        );
    }

    /// The coarsest level holds back by one, so a camera crossing a level's edge doesn't redraw
    /// every page, and the levels run down to one pixel at the near plane.
    #[test]
    fn clip_levels_hold_the_coarsest_and_reach_a_pixel() {
        // A texel of the finest level spans one pixel at the near plane, 2^-7 / SPAN.
        let pixel = (-7.0f64).exp2() / f64::from(SPAN);
        assert_eq!(clip_levels(None, 3, 1.0, pixel), [3, -7]);
        assert_eq!(clip_levels(Some(4), 3, 1.0, pixel), [4, -7]);
        // Negative controls: two levels coarser, or too fine to hold every mesh, is not kept.
        assert_eq!(clip_levels(Some(5), 3, 1.0, pixel), [3, -7]);
        assert_eq!(clip_levels(Some(2), 3, 1.0, pixel), [3, -7]);
        // At most `LEVELS` levels, and never finer than the coarsest.
        let [c, f] = clip_levels(None, 3, 1e-9, pixel);
        assert_eq!(c - f + 1, LEVELS as i32);
        assert_eq!(clip_levels(None, 3, 1e9, pixel), [3, 3]);
    }

    /// A rect light's cube can skip its -z face only if z points out of the lit side.
    #[test]
    fn a_rect_lights_frame_faces_its_lit_side() {
        let mut l = Light::sphere([1.0, 2.0, 3.0], 0.0, [1.0; 3]);
        l.shape = light::RECT;
        // Lit on the u × v side, which is -y here.
        l.u = [1.0, 0.0, 0.0];
        l.v = [0.0, 0.0, 0.5];
        let rows = local_space(&l);
        let to = |p: [f64; 3]| rows.map(|r| dot3(&r, p));
        assert!(to([1.0, 2.0, 3.0]).iter().all(|v| v.abs() < 1e-12));
        let below = to([1.2, 1.0, 3.1]);
        assert!(below[2] > 0.9, "{below:?}");
        // Negative control: the unlit side is behind the frame.
        assert!(to([1.0, 3.0, 3.0])[2] < 0.0);
        for (i, a) in rows.iter().enumerate() {
            for (j, b) in rows.iter().enumerate() {
                let d: f64 = (0..3).map(|k| a[k] * b[k]).sum();
                assert!((d - f64::from(u8::from(i == j))).abs() < 1e-12);
            }
        }
    }
}
