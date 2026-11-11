//! Virtual shadow maps (`shaders/shadow.slang`): for a distant light, a clipmap of up to
//! `LEVELS` square levels around the camera, each `TILES`² tiles; for a sphere, rect, disk or
//! cylinder light, a cube of faces around its centre to its reach, each face `LOCAL_LEVELS`
//! LODs of `TILES`² tiles and coarser. Only the tiles the camera's pixels read hold a page of
//! the one pool, and a page is drawn only when it is new, its light's map changed, or a caster
//! moved over it; a frame that rewrites the draws draws every page.

use ash::vk;

use crate::geometry::{Bound, Geometry};
use crate::gpu::{self, Buffer};
use crate::light::{self, Light, Tiles};
use crate::target::Target;
use crate::transform::{Camera, Instance};
use crate::view::View;

/// `shaders/view.slang`'s constants.
const LEVELS: usize = 16;
const LOCAL_LEVELS: u32 = 6;
const TILES: u32 = 32;
/// Faces' bounds per shadow and level.
const FACES: u64 = 6;
const PAGE: u32 = 256;
/// A pool page's bytes, as `pika:shadowPool` counts them.
const PAGE_BYTES: u64 = PAGE as u64 * PAGE as u64 * 4;
/// Draws a frame lists, one per caster instance and page it meets: 14 MB of lists.
const CAPACITY: u32 = 1 << 19;
/// `Pages`' state words (`shaders/shadow.slang` `MARKED`): eight the host reads, two more,
/// then the counts per level.
const STATE: usize = 10 + 2 * LEVELS * LEVELS + 3 * LEVELS;
const READ: u64 = 8 * 4;
/// Moved instances a frame lists, as one `vkCmdUpdateBuffer` writes them; past it, every page
/// is drawn anew.
const MOVED: usize = 65536 / 4;
/// A distant light's depth spans its casters' and this much of their span past each end, so
/// casters moving a little keep its pages.
const DEPTH_PAD: f64 = 0.25;

/// `shaders/view.slang`'s `Shadow`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Shadow {
    light_space: [[f32; 4]; 3],
    origin: [[i32; 2]; LEVELS],
    level: i32,
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
    /// Whether a page drawn for `self` shows the same for `other`: the camera moves only which
    /// tiles are held.
    fn draws_as(&self, other: &Self) -> bool {
        self.light_space == other.light_space
            && self.level == other.level
            && self.levels == other.levels
            && self.link == other.link
            && self.faces == other.faces
            && self.near == other.near
            && self.far == other.far
    }
}

/// `shaders/shadow.slang`'s `Pages`.
#[repr(C)]
struct Pages {
    unproject: [[f32; 4]; 4],
    camera: vk::DeviceAddress,
    casters: vk::DeviceAddress,
    shadows: vk::DeviceAddress,
    tiles: vk::DeviceAddress,
    marks: vk::DeviceAddress,
    bounds: vk::DeviceAddress,
    depth: vk::DeviceAddress,
    owners: vk::DeviceAddress,
    state: vk::DeviceAddress,
    draws: vk::DeviceAddress,
    pairs: vk::DeviceAddress,
    pool: vk::DeviceAddress,
    free: vk::DeviceAddress,
    drawn: vk::DeviceAddress,
    last: vk::DeviceAddress,
    moved: vk::DeviceAddress,
    count: u32,
    pages: u32,
    capacity: u32,
    fresh: u32,
    frame: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    pages: vk::DeviceAddress,
    first: u32,
    count: u32,
    stride: u32,
    pad: u32,
}

/// A shadowed light, by its index among the view's lights.
pub struct Caster {
    /// Its light's id in the store, which keeps its pages from frame to frame.
    pub id: u64,
    pub light: usize,
    /// Its shadow-link bit, 0 when every mesh casts.
    pub shadow: u32,
}

/// What the last frame's shadows held.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Counts {
    /// Pages the camera's pixels marked, before any level dropped.
    pub wanted: u32,
    pub held: u32,
    /// Pages cleared and drawn: new, their light's map changed, or a caster moved over them.
    pub drawn: u32,
    /// One per caster instance and page it meets.
    pub draws: u32,
    pub triangles: u32,
    /// Of those, the triangles of casters that did not move, drawn into pages only a moving
    /// caster changed: what a static page layer would save.
    pub still_triangles: u32,
}

struct Pool {
    pages: u32,
    texels: Buffer,
    /// Per page: its shadow, its tile and the tile's place, or free.
    owners: Buffer,
    /// The free pages, and the pages drawn, listed each frame.
    free: Buffer,
    drawn: Buffer,
}

/// A shadow as last recorded, by its light's id; a distant light's with its axes and the depth
/// its z spans.
#[derive(Clone, Copy)]
struct Held {
    id: u64,
    shadow: Shadow,
    axes: [[f64; 3]; 3],
    depth: [f64; 2],
}

pub struct Shadows {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    mark: gpu::Compute,
    count: gpu::Compute,
    choose: gpu::Compute,
    assign: gpu::Compute,
    release: gpu::Compute,
    gather: gpu::Compute,
    moving: gpu::Compute,
    remember: gpu::Compute,
    clear: gpu::Compute,
    cast: gpu::Compute,
    pool: Option<Pool>,
    constants: Buffer,
    table: Option<Buffer>,
    /// Per shadow, its tiles' pages, kept from frame to frame, and their flags this frame.
    tiles: Option<Buffer>,
    marks: Option<Buffer>,
    /// Per shadow, level and face, the bounds of the tiles drawn this frame.
    bounds: Option<Buffer>,
    depth: Option<Buffer>,
    /// Per instance, its box when it last moved.
    last: Option<Buffer>,
    moved: Buffer,
    view: Option<View>,
    draws: Buffer,
    pairs: Buffer,
    /// `Pages`' state, and its host copy.
    state: Buffer,
    counts: Buffer,
    shadows: Vec<Shadow>,
    /// This frame's shadows as `fit` made them, and as the last recorded frame drew them.
    fitted: Vec<Held>,
    held: Vec<Held>,
    /// The distant shadows' coarsest level, fitted and drawn.
    coarsest: [Option<i32>; 2],
    /// Every tile lets its page go before the next record: the pool or tiles are new, or the
    /// shadows are not those held.
    clean: bool,
    frame: u32,
    unproject: [[f32; 4]; 4],
    extent: [u32; 2],
    recorded: bool,
    warned: [bool; 2],
}

impl Shadows {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
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
            gather: compute(c"gather")?,
            moving: compute(c"moving")?,
            remember: compute(c"remember")?,
            clear: compute(c"clear")?,
            cast: compute(c"cast")?,
            pool: None,
            constants: Buffer::device_local(
                device,
                size_of::<Pages>() as u64,
                storage | vk::BufferUsageFlags::TRANSFER_DST,
            )?,
            table: None,
            tiles: None,
            marks: None,
            bounds: None,
            depth: None,
            last: None,
            moved: Buffer::device_local(
                device,
                (MOVED * 4) as u64,
                storage | vk::BufferUsageFlags::TRANSFER_DST,
            )?,
            view: None,
            draws: Buffer::device_local(device, CAPACITY as u64 * draw_bytes, indirect)?,
            pairs: Buffer::device_local(device, CAPACITY as u64 * 8, storage)?,
            state: Buffer::device_local(device, (STATE * 4) as u64, indirect | copied)?,
            counts: Buffer::host(device, READ, vk::BufferUsageFlags::TRANSFER_DST, host)?,
            shadows: Vec::new(),
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
        // SAFETY: `d` is live and `layout_info` outlives the call.
        s.layout = unsafe { d.create_pipeline_layout(&layout_info, None) }
            .map_err(|e| format!("vkCreatePipelineLayout failed: {e}"))?;
        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: `d` is live and `module_info` outlives the call.
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
        // SAFETY: pipeline creation is complete, so the module is no longer used.
        unsafe { d.destroy_shader_module(module, None) };
        s.pipeline = built?;
        Ok(s)
    }

    /// Holds a pool of `megabytes` (`pika:shadowPool`) in quarter-megabyte pages. Call before
    /// textures fit the memory budget, so they fit beside it.
    pub fn reserve(&mut self, device: &gpu::Device, megabytes: u32) -> Result<(), String> {
        let pages = megabytes.saturating_mul(4);
        if self.pool.as_ref().is_some_and(|p| p.pages == pages) {
            return Ok(());
        }
        if pages == 0 {
            return Err("pika:shadowPool is 0 MB; shadows need at least one 0.25 MB page".into());
        }
        let layers = device.limits().max_framebuffer_layers;
        if pages > layers {
            return Err(format!(
                "pika:shadowPool is {megabytes} MB, {pages} pages, but the device draws into \
                 at most {layers}; set it to {} MB or less",
                layers / 4
            ));
        }
        // Free the old pool before allocating its replacement.
        self.pool = None;
        self.clean = true;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let filled = storage | vk::BufferUsageFlags::TRANSFER_DST;
        let list = |device| Buffer::device_local(device, pages as u64 * 4, storage);
        self.pool = Some(Pool {
            pages,
            texels: Buffer::device_local(device, pages as u64 * PAGE_BYTES, storage)?,
            owners: Buffer::device_local(device, pages as u64 * 16, filled)?,
            free: list(device)?,
            drawn: list(device)?,
        });
        Ok(())
    }

    /// Fits each distant caster's clipmap to the camera and the scene and each local one's
    /// cube to its light, and points its light at it; the lights are then ready for the
    /// camera's view, whose shadows are at `table`.
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &mut self,
        device: &gpu::Device,
        geometry: &Geometry,
        bound: &Bound,
        camera: &Camera,
        projection: &[f64; 16],
        extent: [u32; 2],
        lights: &mut [Light],
        casters: &[Caster],
    ) -> Result<(), String> {
        self.shadows.clear();
        let pool = (self.pool.as_ref()).ok_or("shadows fit before their pool was reserved")?;
        let unproject = invert(&camera.view_projection.map(|r| r.map(f64::from)))
            .ok_or("the camera's view-projection matrix is singular")?;
        // The clipmaps centre on the eye, or on an orthographic camera's middle.
        let centre = match camera.eye {
            [x, y, z, 1.0] => [x, y, z].map(f64::from),
            _ => {
                let h = unproject.map(|r| r[2] * 0.5 + r[3]);
                [h[0] / h[3], h[1] / h[3], h[2] / h[3]]
            }
        };
        // Hydra's projection is row-vector OpenGL: clip w is the view depth of a perspective
        // camera and 1 for an orthographic one, and [5] scales view y to clip y.
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
            let Some((axes, depth, reach)) = light_space(toward, centre, instances, c.shadow)
            else {
                continue;
            };
            // A level's tiles reach 15 of its 32 from its centre, which snaps to a tile.
            let level = (reach * TILES as f64 / 15.0)
                .max(f64::MIN_POSITIVE)
                .log2()
                .ceil();
            need = need.max(level as i32);
            // The held span while it still holds the casters, so their pages keep.
            let depth = match held(c.id) {
                Some(h) if h.axes == axes && h.depth[0] <= depth[0] && depth[1] <= h.depth[1] => {
                    h.depth
                }
                _ => {
                    let pad = (depth[1] - depth[0]).max(1e-6) * DEPTH_PAD;
                    [depth[0] - pad, depth[1] + pad]
                }
            };
            fitted.push((c, axes, depth));
        }
        // The held coarsest level while it holds every mesh and is at most one coarser.
        let coarsest = match self.coarsest[1] {
            Some(c) if c >= need && c <= need.saturating_add(1) => c,
            _ => need,
        };
        self.coarsest[0] = (!fitted.is_empty()).then_some(coarsest);
        // Every distant shadow has the same levels, which keeps a level's pages at one size.
        let want = (near.max(f64::MIN_POSITIVE) * pixel * (TILES * PAGE) as f64).log2();
        let finest =
            (want.floor() as i32).clamp(coarsest.saturating_sub(LEVELS as i32 - 1), coarsest);
        let levels = coarsest.saturating_sub(finest) as u32 + 1;
        let tile_bytes = LEVELS as u64 * TILES as u64 * TILES as u64 * 4;
        let count = (fitted.len() + local.len()).max(1) as u64;
        self.clean |= grow(device, &mut self.tiles, count * tile_bytes)?;
        grow(device, &mut self.marks, count * tile_bytes)?;
        grow(device, &mut self.bounds, count * LEVELS as u64 * FACES * 16)?;
        grow(device, &mut self.table, count * size_of::<Shadow>() as u64)?;
        grow(
            device,
            &mut self.depth,
            extent[0] as u64 * extent[1] as u64 * 4,
        )?;
        let tiles = self.tiles.as_ref().map_or(0, |t| t.address);
        // A local light's, which a distant one's levels then replace.
        let shadow = |j: usize, rows: [[f64; 4]; 3], c: &Caster| Shadow {
            light_space: rows.map(|r| r.map(|v| v as f32)),
            origin: [[0; 2]; LEVELS],
            level: 0,
            levels: LOCAL_LEVELS,
            pixel: pixel as f32,
            link: c.shadow,
            faces: 0,
            near: 0.0,
            far: 0.0,
            reset: 0,
            tiles: tiles + j as u64 * tile_bytes,
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
                level: finest,
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
        // Tiles are by shadow, so pages keep only while the same lights cast in the same order.
        let ids = |h: &[Held]| h.iter().map(|h| h.id).collect::<Vec<_>>();
        self.clean |= ids(&made) != ids(&self.held);
        self.shadows = (made.iter().enumerate())
            .map(|(j, h)| {
                let kept = self
                    .held
                    .get(j)
                    .is_some_and(|o| o.shadow.draws_as(&h.shadow));
                Shadow {
                    reset: u32::from(!kept),
                    ..h.shadow
                }
            })
            .collect();
        self.fitted = made;
        let capacity = light::capacity(0, [1, 1]);
        if !self.view.as_ref().is_some_and(|v| v.fits(bound, capacity)) {
            self.view = None;
            self.view = Some(View::new(device, bound, capacity)?);
        }
        self.unproject = unproject.map(|r| r.map(|v| v as f32));
        self.extent = extent;
        Ok(())
    }

    /// The camera's view reads each light's `Shadow` here.
    pub fn table(&self) -> vk::DeviceAddress {
        self.table.as_ref().map_or(0, |t| t.address)
    }

    /// Records outside rendering, after `fit` and after the prepass drew `target`'s depth for
    /// `camera`, before the passes that read the shadows; no GPU work may be pending. `moved`
    /// lists the instances whose transform or points moved since the last frame, `None` when
    /// the draws were rewritten, and every page is drawn anew.
    pub fn record(
        &mut self,
        device: &gpu::Device,
        cb: vk::CommandBuffer,
        geometry: &Bound,
        camera: &View,
        target: &Target,
        moved: Option<&[u32]>,
    ) -> Result<(), String> {
        self.recorded = false;
        if self.shadows.is_empty() || geometry.count == 0 {
            return Ok(());
        }
        let instances = geometry.instance_count as u64;
        let grown = grow(device, &mut self.last, instances.max(1) * 32)?;
        let (Some(pool), Some(table), Some(tiles), Some(marks), Some(bounds), Some(depth)) = (
            &self.pool,
            &self.table,
            &self.tiles,
            &self.marks,
            &self.bounds,
            &self.depth,
        ) else {
            return Ok(());
        };
        let (Some(view), Some(last)) = (&mut self.view, &self.last) else {
            return Ok(());
        };
        self.recorded = true;
        self.frame = self.frame.wrapping_add(1);
        let moved = moved.filter(|m| !grown && !self.clean && m.len() <= MOVED);
        let fresh = moved.is_none();
        let moved = moved.unwrap_or(&[]);
        if fresh {
            for s in &mut self.shadows {
                s.reset = 1;
            }
        }
        let pages = Pages {
            unproject: self.unproject,
            camera: camera.constants(),
            casters: view.constants(),
            shadows: table.address,
            tiles: tiles.address,
            marks: marks.address,
            bounds: bounds.address,
            depth: depth.address,
            owners: pool.owners.address,
            state: self.state.address,
            draws: self.draws.address,
            pairs: self.pairs.address,
            pool: pool.texels.address,
            free: pool.free.address,
            drawn: pool.drawn.address,
            last: last.address,
            moved: self.moved.address,
            count: self.shadows.len() as u32,
            pages: pool.pages,
            capacity: CAPACITY,
            fresh: u32::from(fresh),
            frame: self.frame,
            pad: 0,
        };
        let mut state = [0; STATE];
        state[3] = LEVELS as u32;
        // SAFETY: `Pages`, `Shadow`, the state and the instances are plain `repr(C)` data
        // without padding.
        let (pages, shadows, state, moved_bytes) = unsafe {
            (
                std::slice::from_raw_parts((&raw const pages).cast::<u8>(), size_of::<Pages>()),
                std::slice::from_raw_parts(
                    self.shadows.as_ptr().cast::<u8>(),
                    size_of_val(&*self.shadows),
                ),
                std::slice::from_raw_parts(state.as_ptr().cast::<u8>(), size_of_val(&state)),
                std::slice::from_raw_parts(moved.as_ptr().cast::<u8>(), size_of_val(moved)),
            )
        };
        let tile_bytes = self.shadows.len() as u64 * LEVELS as u64 * (TILES * TILES * 4) as u64;
        let d = &self.device;
        // SAFETY: `cb` is recording outside rendering; `fit` and this sized each buffer for
        // these shadows and instances, and `moved` holds at most `MOVED` of them. The view's
        // begin and the depth's copy below order these writes before the compute passes.
        unsafe {
            d.cmd_update_buffer(cb, self.constants.raw, 0, pages);
            d.cmd_update_buffer(cb, table.raw, 0, shadows);
            d.cmd_update_buffer(cb, self.state.raw, 0, state);
            if !moved_bytes.is_empty() {
                d.cmd_update_buffer(cb, self.moved.raw, 0, moved_bytes);
            }
            if self.clean {
                d.cmd_fill_buffer(cb, tiles.raw, 0, tile_bytes, !0);
                d.cmd_fill_buffer(cb, pool.owners.raw, 0, pool.pages as u64 * 16, !0);
            }
            d.cmd_fill_buffer(cb, marks.raw, 0, tile_bytes, 0);
            d.cmd_fill_buffer(
                cb,
                bounds.raw,
                0,
                self.shadows.len() as u64 * LEVELS as u64 * FACES * 16,
                !0,
            );
        }
        self.clean = false;
        self.held = std::mem::take(&mut self.fitted);
        self.coarsest[1] = self.coarsest[0];
        // The casters' view only pulls their vertices and gives their draws.
        let nowhere = Camera {
            view_projection: [[0.0; 4]; 4],
            eye: [0.0; 4],
        };
        let screen = Tiles::new(&[], &nowhere, [1, 1]);
        view.begin(cb, &nowhere, geometry, &[], &screen, [1, 1], table.address);
        target.copy_depth(cb, depth.raw);
        let written = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
        barrier(d, cb, written);
        let push = |first, count, stride| Push {
            pages: self.constants.address,
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
        let entries = 0..self.shadows.len() as u32 * LEVELS as u32 * TILES * TILES;
        let pool_pages = 0..pool.pages;
        let texels = 0..pool.pages * PAGE * PAGE;
        let casting = 0..geometry.instance_count * LEVELS as u32;
        let remembered = 0..if fresh {
            geometry.instance_count
        } else {
            moved.len() as u32
        };
        let thread = 0..1;
        self.mark.record(cb, &[pixels], push);
        barrier(d, cb, computed);
        self.count.record(cb, std::slice::from_ref(&entries), push);
        barrier(d, cb, computed);
        self.choose.record(cb, &[thread], push);
        barrier(d, cb, computed);
        self.release
            .record(cb, std::slice::from_ref(&entries), push);
        barrier(d, cb, computed);
        self.gather.record(cb, &[pool_pages], push);
        barrier(d, cb, computed);
        self.assign.record(cb, &[entries], push);
        barrier(d, cb, computed);
        if !moved.is_empty() {
            let moving = 0..moved.len() as u32 * LEVELS as u32;
            self.moving.record(cb, std::slice::from_ref(&moving), push);
            barrier(d, cb, computed);
        }
        self.remember.record(cb, &[remembered], push);
        barrier(d, cb, computed);
        self.clear.record(cb, &[texels], push);
        self.cast.record(cb, &[casting], push);
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
            size: READ,
        };
        // SAFETY: `cb` is recording outside rendering and both buffers hold the state.
        unsafe { d.cmd_copy_buffer(cb, self.state.raw, self.counts.raw, &[region]) };
        let to_host = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ);
        barrier(d, cb, to_host);
        self.draw(cb, geometry, pool.pages);
        let drawn = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ);
        barrier(d, cb, drawn);
        Ok(())
    }

    /// Draws the listed casters with no attachments, a pool page per layer: the fragments
    /// write the pool themselves.
    fn draw(&self, cb: vk::CommandBuffer, geometry: &Bound, pages: u32) {
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: PAGE,
                height: PAGE,
            },
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: PAGE as f32,
            height: PAGE as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let rendering = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(pages);
        let push = Push {
            pages: self.constants.address,
            first: 0,
            count: 0,
            stride: 0,
            pad: 0,
        };
        // SAFETY: `Push` is plain `repr(C)` data without padding.
        let push = unsafe {
            std::slice::from_raw_parts((&raw const push).cast::<u8>(), size_of::<Push>())
        };
        let stride = size_of::<vk::DrawIndexedIndirectCommand>() as u32;
        let d = &self.device;
        // SAFETY: `cb` is recording outside rendering, and the pipeline was built for
        // rendering with no attachments. `Pages` addresses live buffers, the lists hold
        // `CAPACITY` draws counted by the state's first word, and every listed page is a
        // layer under `pages`.
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
                self.draws.raw,
                0,
                self.state.raw,
                0,
                CAPACITY,
                stride,
            );
            d.cmd_end_rendering(cb);
        }
    }

    /// The last frame's counts, warning once when its pages or draws ran out. Its GPU work
    /// must be done.
    pub fn counts(&mut self) -> Counts {
        if !self.recorded {
            return Counts::default();
        }
        let b = self.counts.bytes();
        let [draws, after, wanted, dropped, past, drawn, triangles, still] =
            std::array::from_fn(|i| u32::from_ne_bytes(b[4 * i..4 * i + 4].try_into().unwrap()));
        let pages = self.pool.as_ref().map_or(0, |p| p.pages);
        if wanted > pages && !self.warned[0] {
            self.warned[0] = true;
            let levels = self.shadows.iter().map(|s| s.levels).max().unwrap_or(0);
            let consequence = if after > pages {
                "some go unshadowed".to_owned()
            } else {
                // The coarsest level cannot drop.
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
                "pika: warning: shadow casters met pages {draws} times, past the {CAPACITY} \
                 draws Pika lists; some shadows are missing"
            );
        }
        Counts {
            wanted,
            held: after.min(pages),
            drawn,
            draws: draws.min(CAPACITY),
            triangles,
            still_triangles: still,
        }
    }

    pub fn memory(&self) -> gpu::Memory {
        let mut memory = self.constants.memory();
        let held = [
            &self.table,
            &self.tiles,
            &self.marks,
            &self.bounds,
            &self.depth,
            &self.last,
        ];
        for b in held.into_iter().flatten() {
            memory += b.memory();
        }
        if let Some(p) = &self.pool {
            for b in [&p.texels, &p.owners, &p.free, &p.drawn] {
                memory += b.memory();
            }
        }
        if let Some(v) = &self.view {
            memory += v.memory();
        }
        for b in [
            &self.draws,
            &self.pairs,
            &self.state,
            &self.counts,
            &self.moved,
        ] {
            memory += b.memory();
        }
        memory
    }
}

impl Drop for Shadows {
    fn drop(&mut self) {
        // SAFETY: both handles were created on `device` or are null, and the owner's target
        // waited for idle before this drops (field order).
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

fn barrier(d: &ash::Device, cb: vk::CommandBuffer, b: vk::MemoryBarrier2) {
    // SAFETY: `cb` is recording outside rendering.
    unsafe { d.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&[b])) };
}

/// Replaces `slot` with a device-local storage buffer when it holds fewer than `bytes`; true
/// when it did, and the buffer's contents are undefined.
fn grow(device: &gpu::Device, slot: &mut Option<Buffer>, bytes: u64) -> Result<bool, String> {
    if slot.as_ref().is_some_and(|b| b.size >= bytes) {
        return Ok(false);
    }
    *slot = None;
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    *slot = Some(Buffer::device_local(device, bytes, usage)?);
    Ok(true)
}

/// A distant light's axes: x and y across it, z away from it. With them, the depth along z of
/// the casters carrying `shadow`, and how far across the light from `centre` any mesh reaches,
/// which the coarsest level must hold; `None` when nothing casts.
fn light_space<'a>(
    toward: [f64; 3],
    centre: [f64; 3],
    instances: impl Iterator<Item = (&'a Instance, &'a [f32; 6])>,
    shadow: u32,
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
        let casts = instance.shadow() & shadow == shadow;
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

/// World-to-light rows for a local light: its frame about its centre, z out of a rect or
/// disk light's lit side, whose cube then needs no -z face.
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

/// Gauss–Jordan with partial pivoting; `None` when `m` is singular.
fn invert(m: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let mut a = *m;
    let mut inverse: [[f64; 4]; 4] =
        std::array::from_fn(|i| std::array::from_fn(|j| f64::from(u8::from(i == j))));
    for col in 0..4 {
        let pivot = (col..4).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        if a[pivot][col] == 0.0 {
            return None;
        }
        a.swap(col, pivot);
        inverse.swap(col, pivot);
        let scale = 1.0 / a[col][col];
        for j in 0..4 {
            a[col][j] *= scale;
            inverse[col][j] *= scale;
        }
        for row in (0..4).filter(|&r| r != col) {
            let f = a[row][col];
            for j in 0..4 {
                a[row][j] -= f * a[col][j];
                inverse[row][j] -= f * inverse[col][j];
            }
        }
    }
    Some(inverse)
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

    #[test]
    fn inverse_undoes_the_matrix() {
        let m = [
            [2.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, -1.0, 2.0],
            [0.5, 4.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ];
        let inverse = invert(&m).unwrap();
        for (i, row) in m.iter().enumerate() {
            for j in 0..4 {
                let v: f64 = row.iter().zip(&inverse).map(|(a, r)| a * r[j]).sum();
                assert!(
                    (v - f64::from(u8::from(i == j))).abs() < 1e-12,
                    "{i} {j}: {v}"
                );
            }
        }
        // Negative control: a repeated row has no inverse.
        assert!(invert(&[m[0], m[0], m[2], m[3]]).is_none());
    }

    /// A rect light's cube needs no -z face only if z points out of its lit side.
    #[test]
    fn a_rect_lights_frame_faces_its_lit_side() {
        let mut l = Light::sphere([1.0, 2.0, 3.0], 0.0, [1.0; 3]);
        l.shape = light::RECT;
        // Lit on the u × v side: -y.
        l.u = [1.0, 0.0, 0.0];
        l.v = [0.0, 0.0, 0.5];
        let rows = local_space(&l);
        let to = |p: [f64; 3]| rows.map(|r| dot3(&r, p));
        assert!(to([1.0, 2.0, 3.0]).iter().all(|v| v.abs() < 1e-12));
        let below = to([1.2, 1.0, 3.1]);
        assert!(below[2] > 0.9, "{below:?}");
        // Negative control: the unlit side is behind the frame.
        assert!(to([1.0, 3.0, 3.0])[2] < 0.0);
        // The frame is orthonormal.
        for (i, a) in rows.iter().enumerate() {
            for (j, b) in rows.iter().enumerate() {
                let d: f64 = (0..3).map(|k| a[k] * b[k]).sum();
                assert!((d - f64::from(u8::from(i == j))).abs() < 1e-12);
            }
        }
    }
}
