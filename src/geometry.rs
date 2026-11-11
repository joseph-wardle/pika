//! The GPU copy of the store's meshes, laid out afresh only when a mesh is
//! remodelled; a playing frame writes only what moved.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ash::vk;

use crate::commands::Commands;
use crate::gpu::{self, Buffer};
use crate::link::Links;
use crate::need::{self, Footprint};
use crate::smooth::{self, Smooth};
use crate::store::{
    self, Instancer, Interpolation, Mesh, Moved, Normals, Refined, Scheme, Store, Texcoord,
    Topology,
};
use crate::subdiv::{self, Subdiv};
use crate::tangent::{self, Tangent};
use crate::transform::{Instance, instance_transforms, multiply};

/// A vertex's normal index when its normal is its point's smooth normal.
const SMOOTH: u32 = u32::MAX;
/// A vertex's texcoord index when its mesh has none.
const NONE: u32 = u32::MAX;

struct Range {
    first_point: u64,
    /// Refined points when refined.
    point_count: u32,
    /// `None` when its points are drawn as they are.
    first_cage: Option<u64>,
    first_normal: u64,
    normal_count: usize,
    /// Refined values when refined.
    first_texcoord: u64,
    texcoord_count: usize,
    /// `None` when its values are drawn as they are.
    first_cage_texcoord: Option<u64>,
    first_vertex: i32,
    vertex_count: u32,
    index_count: u32,
    /// By material, in order, none empty.
    parts: Vec<Part>,
    /// The fraction of its cage box's size a refined point can lie outside it;
    /// 0 when not refined.
    spread: f32,
    /// Texcoord units per unit of its own space; recomputed when its texcoord moves.
    density: Option<f32>,
}

/// One material's triangles of a mesh: its faces in no GeomSubset, or one subset's.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Part {
    /// `None` for the faces in no subset.
    subset: Option<usize>,
    first_index: u32,
    index_count: u32,
}

struct Layout {
    ranges: BTreeMap<u64, Range>,
    point_count: u32,
    vertex_count: u32,
    /// They come first in `points`, so refined point p's stencil is the pth.
    refined_count: u32,
    points: Buffer,
    cage: Buffer,
    stencils: Stencils,
    normals: Buffer,
    /// Zero where its mesh's normals are authored.
    smooth: Buffer,
    adjacency: Adjacency,
    /// They come first in `texcoord`, so refined value t's stencil is the tth.
    refined_texcoord_count: u32,
    texcoord: Buffer,
    cage_texcoord: Buffer,
    texcoord_stencils: Stencils,
    /// Point and normal index (or `SMOOTH`) per unique point, normal and texcoord.
    vertices: Buffer,
    /// Texcoord index (or `NONE`) per vertex.
    vertex_texcoord: Buffer,
    /// Tangent and handedness packed in 4 bytes (`shaders/view.slang`).
    tangents: Buffer,
    vertex_adjacency: Adjacency,
    /// Counted from the mesh's first vertex.
    indices: Buffer,
}

struct Stencils {
    /// One per refined value, then the end of the last stencil.
    offsets: Buffer,
    indices: Buffer,
    weights: Buffer,
}

impl Stencils {
    fn new(device: &gpu::Device, tables: &StencilTables) -> Result<Self, String> {
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        Ok(Self {
            offsets: Buffer::device_local(device, size(&tables.offsets), storage)?,
            indices: Buffer::device_local(device, size(&tables.indices), storage)?,
            weights: Buffer::device_local(device, size(&tables.weights), storage)?,
        })
    }

    fn addresses(&self, cage: &Buffer, refined: &Buffer) -> subdiv::Addresses {
        subdiv::Addresses {
            cage: cage.address,
            offsets: self.offsets.address,
            indices: self.indices.address,
            weights: self.weights.address,
            refined: refined.address,
        }
    }
}

struct Adjacency {
    /// One per point (or vertex), then the end of the last one's corners.
    offsets: Buffer,
    /// Per corner, grouped by point: the previous and next point of its face;
    /// grouped by vertex: its position in the triangle indices.
    corners: Buffer,
}

#[derive(Clone, Debug, PartialEq)]
struct Row {
    record: u32,
    instances: std::ops::Range<usize>,
}

type Rows = BTreeMap<u64, Vec<Row>>;

#[derive(Default)]
struct DrawTables {
    instances: Vec<Instance>,
    owners: Vec<u32>,
    /// Low corner, then high, in the mesh's own space.
    boxes: Vec<[f32; 6]>,
    /// No record has instances until the cull pass counts a view's.
    records: Vec<vk::DrawIndexedIndirectCommand>,
    groups: Vec<Group>,
    rows: Rows,
}

/// Consecutive records of one material, drawn with one multi-draw.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Group {
    /// `None` for meshes with no material bound.
    pub material: Option<u64>,
    pub first: u32,
    pub count: u32,
}

struct Draws {
    instances: Buffer,
    owners: Buffer,
    boxes: Buffer,
    records: Buffer,
    /// Kept on the host for `footprints`; a patch writes both.
    tables: DrawTables,
    tags: Vec<String>,
}

pub struct Geometry {
    layout: Option<Layout>,
    draws: Option<Draws>,
    /// Kept while frames keep uploading, dropped by the first that does not.
    staging: Vec<Buffer>,
    subdiv: Subdiv,
    smooth: Smooth,
    tangent: Tangent,
}

pub struct Bound {
    pub points: vk::DeviceAddress,
    pub normals: vk::DeviceAddress,
    pub smooth: vk::DeviceAddress,
    pub vertices: vk::DeviceAddress,
    pub texcoord: vk::DeviceAddress,
    pub vertex_texcoord: vk::DeviceAddress,
    pub tangents: vk::DeviceAddress,
    pub instances: vk::DeviceAddress,
    pub owners: vk::DeviceAddress,
    pub boxes: vk::DeviceAddress,
    pub indices: vk::Buffer,
    /// Without instances; each view copies them.
    pub records: vk::Buffer,
    pub count: u32,
    pub instance_count: u32,
}

pub struct Update {
    pub bytes: u64,
    pub layout: bool,
    pub draws: bool,
}

struct Upload<'a> {
    bytes: &'a [u8],
    buffer: vk::Buffer,
    offset: u64,
}

impl<'a> Upload<'a> {
    fn table(bytes: &'a [u8], buffer: &Buffer) -> Self {
        Self {
            bytes,
            buffer: buffer.raw,
            offset: 0,
        }
    }
}

impl Geometry {
    pub fn new(
        device: &gpu::Device,
        subdiv: &[u32],
        smooth: &[u32],
        tangent: &[u32],
    ) -> Result<Self, String> {
        Ok(Self {
            layout: None,
            draws: None,
            staging: Vec::new(),
            subdiv: Subdiv::new(device, subdiv)?,
            smooth: Smooth::new(device, smooth)?,
            tangent: Tangent::new(device, tangent)?,
        })
    }

    /// Empty `tags` draws every tag. Marks every step each frame, run or not.
    /// No GPU work may be pending: buffers are replaced and staging rewritten.
    pub fn update(
        &mut self,
        device: &gpu::Device,
        store: &Store,
        moved: &Moved,
        tags: &[String],
        commands: &mut Commands,
    ) -> Result<Update, String> {
        let rebuilt = moved.layout || self.layout.is_none();
        let tables = match rebuilt {
            true => Some(self.rebuild(device, store)?),
            false => None,
        };
        let layout = self.layout.as_mut().expect("rebuilt above");
        for id in &moved.texcoord {
            if let Some(range) = layout.ranges.get_mut(id) {
                range.density = need::density(&store.meshes()[id]);
            }
        }
        let layout = &*layout;

        let kept = (self.draws.as_ref()).filter(|d| !rebuilt && !moved.draws && d.tags == tags);
        let patched = kept.and_then(|d| patch(store, &layout.ranges, &d.tables.rows, moved, tags));
        if patched.is_none() {
            rewrite(device, store, layout, tags, &mut self.draws)?;
        }
        let draws = self.draws.as_ref().expect("rewritten above");

        let mut copies = Vec::new();
        if let Some(t) = &tables {
            copies.extend(table_uploads(t, layout));
        }
        for (&id, range) in &layout.ranges {
            let mesh = &store.meshes()[&id];
            if rebuilt || moved.points.contains(&id) {
                copies.push(point_upload(mesh, range, layout));
            }
            if rebuilt || moved.normals.contains(&id) {
                copies.extend(normal_upload(mesh, range, layout));
            }
            if rebuilt || moved.texcoord.contains(&id) {
                copies.extend(texcoord_upload(mesh, range, layout));
            }
        }
        match &patched {
            Some(p) => copies.extend(patch_uploads(p, draws)),
            None => copies.extend(draw_uploads(draws)),
        }
        let bytes = upload(device, &mut self.staging, &copies, commands.buffer())?;
        commands.mark("upload")?;
        if let (Some(p), Some(d)) = (&patched, self.draws.as_mut()) {
            p.apply(&mut d.tables);
        }
        let layout = self.layout.as_ref().expect("rebuilt above");

        let rewrite = match rebuilt {
            true => Rewrite {
                refined: whole(layout.refined_count),
                points: whole(layout.point_count),
                texcoords: whole(layout.refined_texcoord_count),
                vertices: whole(layout.vertex_count),
            },
            false => moved_ranges(&layout.ranges, moved),
        };
        self.refine(device, commands, layout, &rewrite)?;
        Ok(Update {
            bytes,
            layout: rebuilt,
            draws: patched.is_none(),
        })
    }

    fn rebuild(&mut self, device: &gpu::Device, store: &Store) -> Result<Tables, String> {
        // Free the old buffers before allocating their replacements.
        self.layout = None;
        let (layout, tables) = build(device, store)?;
        self.layout = Some(layout);
        Ok(tables)
    }

    fn refine(
        &self,
        device: &gpu::Device,
        commands: &mut Commands,
        layout: &Layout,
        rewrite: &Rewrite,
    ) -> Result<(), String> {
        let cb = commands.buffer();
        if !rewrite.texcoords.is_empty() {
            let addresses =
                (layout.texcoord_stencils).addresses(&layout.cage_texcoord, &layout.texcoord);
            self.subdiv.record(cb, addresses, 2, &rewrite.texcoords);
        }
        if !rewrite.refined.is_empty() {
            let addresses = layout.stencils.addresses(&layout.cage, &layout.points);
            self.subdiv.record(cb, addresses, 3, &rewrite.refined);
            barrier(device, cb, vk::PipelineStageFlags2::COMPUTE_SHADER);
        }
        commands.mark("subdiv")?;
        if !rewrite.points.is_empty() {
            let addresses = smooth::Addresses {
                points: layout.points.address,
                offsets: layout.adjacency.offsets.address,
                corners: layout.adjacency.corners.address,
                normals: layout.smooth.address,
            };
            self.smooth.record(cb, addresses, &rewrite.points);
        }
        commands.mark("smooth")?;
        let summed = !(rewrite.texcoords.is_empty() && rewrite.points.is_empty());
        if !rewrite.vertices.is_empty() {
            if summed {
                barrier(device, cb, vk::PipelineStageFlags2::COMPUTE_SHADER);
            }
            let addresses = tangent::Addresses {
                points: layout.points.address,
                normals: layout.normals.address,
                smooth: layout.smooth.address,
                vertices: layout.vertices.address,
                texcoord: layout.texcoord.address,
                vertex_texcoord: layout.vertex_texcoord.address,
                offsets: layout.vertex_adjacency.offsets.address,
                corners: layout.vertex_adjacency.corners.address,
                indices: layout.indices.address,
                tangents: layout.tangents.address,
            };
            self.tangent.record(cb, addresses, &rewrite.vertices);
        }
        if summed || !rewrite.vertices.is_empty() {
            barrier(device, cb, vk::PipelineStageFlags2::VERTEX_SHADER);
        }
        commands.mark("tangent")
    }

    pub fn bound(&self) -> Option<Bound> {
        let (layout, draws) = (self.layout.as_ref()?, self.draws.as_ref()?);
        Some(Bound {
            points: layout.points.address,
            normals: layout.normals.address,
            smooth: layout.smooth.address,
            vertices: layout.vertices.address,
            texcoord: layout.texcoord.address,
            vertex_texcoord: layout.vertex_texcoord.address,
            tangents: layout.tangents.address,
            instances: draws.instances.address,
            owners: draws.owners.address,
            boxes: draws.boxes.address,
            indices: layout.indices.raw,
            records: draws.records.raw,
            count: draws.tables.records.len() as u32,
            instance_count: draws.tables.instances.len() as u32,
        })
    }

    pub fn groups(&self) -> &[Group] {
        self.draws.as_ref().map_or(&[], |d| &d.tables.groups)
    }

    /// An instance outside the view adds nothing; a material with none inside has no entry.
    pub fn footprints(
        &self,
        view_projection: &[[f32; 4]; 4],
        size: [u32; 2],
    ) -> BTreeMap<u64, Footprint> {
        let mut footprints = BTreeMap::new();
        let (Some(layout), Some(draws)) = (&self.layout, &self.draws) else {
            return footprints;
        };
        let t = &draws.tables;
        let mut materials = vec![None; t.records.len()];
        for g in &t.groups {
            materials[g.first as usize..(g.first + g.count) as usize].fill(g.material);
        }
        for (id, rows) in &t.rows {
            let density = layout.ranges[id].density;
            for row in rows {
                let Some(material) = materials[row.record as usize] else {
                    continue;
                };
                let bounds = &t.boxes[row.record as usize];
                for instance in &t.instances[row.instances.clone()] {
                    let world = instance.world();
                    let seen = match density {
                        Some(d) => need::uv_per_pixel(view_projection, size, world, bounds, d),
                        None => need::uv_per_pixel(view_projection, size, world, bounds, 1.0)
                            .map(|_| f32::INFINITY),
                    };
                    let Some(uv_per_pixel) = seen else {
                        continue;
                    };
                    let footprint = Footprint {
                        uv_per_pixel,
                        unestimated: density.is_none(),
                    };
                    (footprints.entry(material))
                        .or_insert(Footprint::NONE)
                        .add(footprint);
                }
            }
        }
        footprints
    }

    /// Each drawn instance with its mesh's box, in the mesh's own space.
    pub fn instances(&self) -> impl Iterator<Item = (&Instance, &[f32; 6])> {
        self.draws.iter().flat_map(|d| {
            let t = &d.tables;
            (t.instances.iter().zip(&t.owners)).map(|(i, &r)| (i, &t.boxes[r as usize]))
        })
    }

    pub fn memory(&self) -> gpu::Memory {
        let layout = self.layout.iter().flat_map(|l| {
            [
                &l.points,
                &l.cage,
                &l.stencils.offsets,
                &l.stencils.indices,
                &l.stencils.weights,
                &l.normals,
                &l.smooth,
                &l.adjacency.offsets,
                &l.adjacency.corners,
                &l.texcoord,
                &l.cage_texcoord,
                &l.texcoord_stencils.offsets,
                &l.texcoord_stencils.indices,
                &l.texcoord_stencils.weights,
                &l.vertices,
                &l.vertex_texcoord,
                &l.tangents,
                &l.vertex_adjacency.offsets,
                &l.vertex_adjacency.corners,
                &l.indices,
            ]
        });
        let draws =
            (self.draws.iter()).flat_map(|d| [&d.instances, &d.owners, &d.boxes, &d.records]);
        let mut memory = gpu::Memory::default();
        for b in layout.chain(draws).chain(&self.staging) {
            memory += b.memory();
        }
        memory
    }
}

fn table_uploads<'a>(t: &'a Tables, l: &Layout) -> [Upload<'a>; 13] {
    [
        Upload::table(bytes(&t.vertices), &l.vertices),
        Upload::table(bytes(&t.vertex_texcoord), &l.vertex_texcoord),
        Upload::table(bytes(&t.indices), &l.indices),
        Upload::table(bytes(&t.corner_offsets), &l.adjacency.offsets),
        Upload::table(bytes(&t.corners), &l.adjacency.corners),
        Upload::table(bytes(&t.vertex_offsets), &l.vertex_adjacency.offsets),
        Upload::table(bytes(&t.vertex_corners), &l.vertex_adjacency.corners),
        Upload::table(bytes(&t.stencils.offsets), &l.stencils.offsets),
        Upload::table(bytes(&t.stencils.indices), &l.stencils.indices),
        Upload::table(bytes(&t.stencils.weights), &l.stencils.weights),
        Upload::table(
            bytes(&t.texcoord_stencils.offsets),
            &l.texcoord_stencils.offsets,
        ),
        Upload::table(
            bytes(&t.texcoord_stencils.indices),
            &l.texcoord_stencils.indices,
        ),
        Upload::table(
            bytes(&t.texcoord_stencils.weights),
            &l.texcoord_stencils.weights,
        ),
    ]
}

fn point_upload<'a>(mesh: &'a Mesh, range: &Range, layout: &Layout) -> Upload<'a> {
    let (buffer, first) = match range.first_cage {
        Some(first) => (layout.cage.raw, first),
        None => (layout.points.raw, range.first_point),
    };
    Upload {
        bytes: bytes(mesh.points.as_deref().unwrap_or_default()),
        buffer,
        offset: first * 12,
    }
}

fn normal_upload<'a>(mesh: &'a Mesh, range: &Range, layout: &Layout) -> Option<Upload<'a>> {
    if range.normal_count == 0 {
        return None;
    }
    let normals = authored(mesh)?;
    Some(Upload {
        bytes: bytes(&normals.values),
        buffer: layout.normals.raw,
        offset: range.first_normal * 12,
    })
}

fn texcoord_upload<'a>(mesh: &'a Mesh, range: &Range, layout: &Layout) -> Option<Upload<'a>> {
    let t = texcoord(mesh).filter(|t| !t.values.is_empty())?;
    let (buffer, first) = match range.first_cage_texcoord {
        Some(first) => (layout.cage_texcoord.raw, first),
        None => (layout.texcoord.raw, range.first_texcoord),
    };
    Some(Upload {
        bytes: bytes(&t.values),
        buffer,
        offset: first * 8,
    })
}

/// The least `maxMemoryAllocationSize` Vulkan requires; a first upload can pass
/// the 4 GiB NVIDIA reports.
const STAGING: u64 = 1 << 30;

fn upload(
    device: &gpu::Device,
    staging: &mut Vec<Buffer>,
    copies: &[Upload],
    cb: vk::CommandBuffer,
) -> Result<u64, String> {
    let lengths: Vec<u64> = copies.iter().map(|c| c.bytes.len() as u64).collect();
    let total: u64 = lengths.iter().sum();
    if !reuses(staging.iter().map(|s| s.size).sum(), total) {
        // Every buffer but the last is STAGING bytes, as `pieces` lays them.
        staging.clear();
        // Write-combined memory: written once in order, never read.
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mut left = total;
        while left > 0 {
            let size = left.min(STAGING);
            staging.push(Buffer::host(
                device,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                host,
            )?);
            left -= size;
        }
    }
    if total == 0 {
        return Ok(0);
    }
    let d = &device.raw;
    for p in pieces(&lengths, STAGING) {
        let c = &copies[p.copy];
        let s = &mut staging[p.buffer];
        let from = &c.bytes[p.start as usize..(p.start + p.size) as usize];
        s.bytes()[p.at as usize..(p.at + p.size) as usize].copy_from_slice(from);
        let region = vk::BufferCopy {
            src_offset: p.at,
            dst_offset: c.offset + p.start,
            size: p.size,
        };
        // SAFETY: `cb` records outside rendering; both buffers are `d`'s and hold the region.
        unsafe { d.cmd_copy_buffer(cb, s.raw, c.buffer, &[region]) };
    }
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COPY)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(
            vk::PipelineStageFlags2::COMPUTE_SHADER
                | vk::PipelineStageFlags2::VERTEX_SHADER
                | vk::PipelineStageFlags2::INDEX_INPUT
                | vk::PipelineStageFlags2::COPY,
        )
        .dst_access_mask(
            vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::INDEX_READ
                | vk::AccessFlags2::TRANSFER_READ,
        );
    // SAFETY: `cb` is recording outside any rendering.
    unsafe {
        d.cmd_pipeline_barrier2(
            cb,
            &vk::DependencyInfo::default().memory_barriers(&[barrier]),
        )
    };
    Ok(total)
}

/// Never keeps more than twice a frame's bytes: playback must not keep a first
/// upload's gigabytes, but reusing between its own frames saves allocating.
fn reuses(staged: u64, total: u64) -> bool {
    total <= staged && staged <= 2 * total
}

struct Piece {
    copy: usize,
    start: u64,
    buffer: usize,
    at: u64,
    size: u64,
}

/// Splits a copy where a staging buffer ends.
fn pieces(lengths: &[u64], chunk: u64) -> Vec<Piece> {
    let mut out = Vec::new();
    let mut at = 0;
    for (copy, &length) in lengths.iter().enumerate() {
        let mut start = 0;
        while start < length {
            let size = (chunk - at % chunk).min(length - start);
            out.push(Piece {
                copy,
                start,
                buffer: (at / chunk) as usize,
                at: at % chunk,
                size,
            });
            (start, at) = (start + size, at + size);
        }
    }
    out
}

#[derive(Default)]
struct Tables {
    vertices: Vec<[u32; 2]>,
    vertex_texcoord: Vec<u32>,
    indices: Vec<u32>,
    /// One per point, then the end of the last one's corners.
    corner_offsets: Vec<u32>,
    corners: Vec<[u32; 2]>,
    /// One per vertex, then the end of the last one's corners.
    vertex_offsets: Vec<u32>,
    /// Each corner's position in `indices`.
    vertex_corners: Vec<u32>,
    stencils: StencilTables,
    texcoord_stencils: StencilTables,
}

/// Entries count from the tables' start, cage values from the cage buffer's.
#[derive(Default)]
struct StencilTables {
    /// One per refined value, then the end of the last stencil.
    offsets: Vec<u32>,
    indices: Vec<u32>,
    weights: Vec<f32>,
}

/// Refined meshes first. A mesh whose data does not fit together is left out, with a warning.
fn build(device: &gpu::Device, store: &Store) -> Result<(Layout, Tables), String> {
    let mut ranges = BTreeMap::new();
    let mut tables = Tables::default();
    let (mut points, mut cage, mut normals) = (0u64, 0u64, 0u64);
    let (mut texcoords, mut cage_texcoords) = (0u64, 0u64);
    let (mut refined_count, mut refined_texcoord_count) = (0, 0);
    for refined_pass in [true, false] {
        for (&id, mesh) in store.meshes() {
            let (Some(t), Some(p)) = (&mesh.topology, &mesh.points) else {
                continue;
            };
            if t.refined.is_some() != refined_pass {
                continue;
            }
            let first_vertex = tables.vertices.len();
            let first_index = tables.indices.len();
            let first_corner = tables.corners.len();
            let first_vertex_corner = tables.vertex_corners.len();
            let first_entry = tables.stencils.indices.len();
            let first_texcoord_entry = tables.texcoord_stencils.indices.len();
            let normal_count = authored(mesh).map_or(0, |n| n.values.len());
            let (texcoord_count, cage_texcoord_count) = match texcoord(mesh) {
                Some(t) => match &t.refined {
                    Some(r) => (r.stencils.count(), t.values.len()),
                    None => (t.values.len(), 0),
                },
                None => (0, 0),
            };
            let first_cage = refined_pass.then_some(cage);
            let first = Firsts {
                point: points as u32,
                cage: cage as u32,
                normal: normals as u32,
                texcoord: texcoords as u32,
                cage_texcoord: cage_texcoords as u32,
            };
            let (drawn, parts) = match triangulate(mesh, first, &mut tables) {
                Ok(drawn) => drawn,
                Err(e) => {
                    eprintln!("pika: warning: {}: {e}; not drawn", mesh.path);
                    tables.vertices.truncate(first_vertex);
                    tables.vertex_texcoord.truncate(first_vertex);
                    tables.indices.truncate(first_index);
                    tables.corner_offsets.truncate(points as usize);
                    tables.corners.truncate(first_corner);
                    tables.vertex_offsets.truncate(first_vertex);
                    tables.vertex_corners.truncate(first_vertex_corner);
                    if refined_pass {
                        tables.stencils.offsets.truncate(points as usize);
                        tables
                            .texcoord_stencils
                            .offsets
                            .truncate(texcoords as usize);
                    }
                    tables.stencils.indices.truncate(first_entry);
                    tables.stencils.weights.truncate(first_entry);
                    tables
                        .texcoord_stencils
                        .indices
                        .truncate(first_texcoord_entry);
                    tables
                        .texcoord_stencils
                        .weights
                        .truncate(first_texcoord_entry);
                    continue;
                }
            };
            ranges.insert(
                id,
                Range {
                    first_point: points,
                    point_count: drawn as u32,
                    first_cage,
                    first_normal: normals,
                    normal_count,
                    first_texcoord: texcoords,
                    texcoord_count,
                    first_cage_texcoord: refined_pass.then_some(cage_texcoords),
                    first_vertex: first_vertex as i32,
                    vertex_count: (tables.vertices.len() - first_vertex) as u32,
                    index_count: (tables.indices.len() - first_index) as u32,
                    parts,
                    spread: t.refined.as_ref().map_or(0.0, spread),
                    density: need::density(mesh),
                },
            );
            points += drawn as u64;
            normals += normal_count as u64;
            texcoords += texcoord_count as u64;
            cage_texcoords += cage_texcoord_count as u64;
            if refined_pass {
                cage += p.len() as u64;
            }
        }
        if refined_pass {
            refined_count = points as u32;
            refined_texcoord_count = texcoords as u32;
            (tables.stencils.offsets).push(tables.stencils.indices.len() as u32);
            (tables.texcoord_stencils.offsets).push(tables.texcoord_stencils.indices.len() as u32);
        }
    }
    tables.corner_offsets.push(tables.corners.len() as u32);
    tables
        .vertex_offsets
        .push(tables.vertex_corners.len() as u32);
    let vertices = tables.vertices.len() as u64;
    let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let index = vk::BufferUsageFlags::INDEX_BUFFER | storage;
    let written = vk::BufferUsageFlags::STORAGE_BUFFER;
    let layout = Layout {
        point_count: points as u32,
        vertex_count: vertices as u32,
        refined_count,
        points: Buffer::device_local(device, points * 12, storage)?,
        cage: Buffer::device_local(device, cage * 12, storage)?,
        stencils: Stencils::new(device, &tables.stencils)?,
        normals: Buffer::device_local(device, normals * 12, storage)?,
        smooth: Buffer::device_local(device, points * 12, written)?,
        adjacency: Adjacency {
            offsets: Buffer::device_local(device, size(&tables.corner_offsets), storage)?,
            corners: Buffer::device_local(device, size(&tables.corners), storage)?,
        },
        refined_texcoord_count,
        texcoord: Buffer::device_local(device, texcoords * 8, storage)?,
        cage_texcoord: Buffer::device_local(device, cage_texcoords * 8, storage)?,
        texcoord_stencils: Stencils::new(device, &tables.texcoord_stencils)?,
        vertices: Buffer::device_local(device, size(&tables.vertices), storage)?,
        vertex_texcoord: Buffer::device_local(device, size(&tables.vertex_texcoord), storage)?,
        tangents: Buffer::device_local(device, vertices * 4, written)?,
        vertex_adjacency: Adjacency {
            offsets: Buffer::device_local(device, size(&tables.vertex_offsets), storage)?,
            corners: Buffer::device_local(device, size(&tables.vertex_corners), storage)?,
        },
        indices: Buffer::device_local(device, size(&tables.indices), index)?,
        ranges,
    };
    Ok((layout, tables))
}

struct Rewrite {
    refined: Vec<std::ops::Range<u32>>,
    points: Vec<std::ops::Range<u32>>,
    texcoords: Vec<std::ops::Range<u32>>,
    vertices: Vec<std::ops::Range<u32>>,
}

fn moved_ranges(ranges: &BTreeMap<u64, Range>, moved: &Moved) -> Rewrite {
    let mut rewrite = Rewrite {
        refined: Vec::new(),
        points: Vec::new(),
        texcoords: Vec::new(),
        vertices: Vec::new(),
    };
    for range in moved.points.iter().filter_map(|id| ranges.get(id)) {
        let first = range.first_point as u32;
        let drawn = first..first + range.point_count;
        if range.first_cage.is_some() {
            rewrite.refined.push(drawn.clone());
        }
        rewrite.points.push(drawn);
    }
    for range in moved.texcoord.iter().filter_map(|id| ranges.get(id)) {
        if range.first_cage_texcoord.is_some() {
            let first = range.first_texcoord as u32;
            rewrite
                .texcoords
                .push(first..first + range.texcoord_count as u32);
        }
    }
    let shaped = (moved.points.iter())
        .chain(&moved.normals)
        .chain(&moved.texcoord);
    for range in shaped
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter_map(|id| ranges.get(id))
    {
        let first = range.first_vertex as u32;
        rewrite.vertices.push(first..first + range.vertex_count);
    }
    rewrite
}

fn whole(count: u32) -> Vec<std::ops::Range<u32>> {
    (count > 0).then_some(0..count).into_iter().collect()
}

fn barrier(device: &gpu::Device, cb: vk::CommandBuffer, stage: vk::PipelineStageFlags2) {
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(stage)
        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ);
    // SAFETY: `cb` is recording outside any rendering.
    unsafe {
        device.raw.cmd_pipeline_barrier2(
            cb,
            &vk::DependencyInfo::default().memory_barriers(&[barrier]),
        )
    };
}

/// Authored normals only when not subdivided, as USD specifies; Hydra hands
/// them over on subdivided meshes too.
fn authored(mesh: &Mesh) -> Option<&Normals> {
    match (&mesh.topology, &mesh.normals) {
        (Some(t), Some(Some(n))) if t.scheme == Scheme::None => Some(n),
        _ => None,
    }
}

fn texcoord(mesh: &Mesh) -> Option<&Texcoord> {
    mesh.texcoord.as_ref()?.as_ref()
}

#[derive(Clone, Copy)]
struct Firsts {
    point: u32,
    cage: u32,
    normal: u32,
    texcoord: u32,
    cage_texcoord: u32,
}

struct Faces<'a> {
    counts: &'a [i32],
    corners: &'a [i32],
    holes: &'a [i32],
    points: usize,
}

/// A vertex is a unique point, normal and texcoord. Faces fan from their first
/// corner; holes are left out.
fn triangulate(
    mesh: &Mesh,
    first: Firsts,
    tables: &mut Tables,
) -> Result<(usize, Vec<Part>), String> {
    let (Some(topology), Some(points)) = (&mesh.topology, &mesh.points) else {
        return Ok((0, Vec::new()));
    };
    let faces = match &topology.refined {
        Some(r) => {
            stencils(&r.stencils, points.len(), first.cage, &mut tables.stencils)?;
            Faces {
                counts: &r.face_vertex_counts,
                corners: &r.face_vertex_indices,
                holes: &r.holes,
                points: r.point_count(),
            }
        }
        None => Faces {
            counts: &topology.face_vertex_counts,
            corners: &topology.face_vertex_indices,
            holes: &topology.holes,
            points: points.len(),
        },
    };
    let (counts, corners) = (faces.counts, faces.corners);
    let total: i64 = counts.iter().map(|&n| n as i64).sum();
    if counts.iter().any(|&n| n < 0) || total != corners.len() as i64 {
        return Err(format!(
            "face vertex counts sum to {total}, but there are {} face vertex indices",
            corners.len()
        ));
    }
    if let Some(&bad) = corners
        .iter()
        .find(|&&p| p < 0 || p as usize >= faces.points)
    {
        return Err(format!(
            "face vertex index {bad} is not one of {} points",
            faces.points
        ));
    }
    if let Some(&bad) = (faces.holes.iter()).find(|&&f| f < 0 || f as usize >= counts.len()) {
        return Err(format!(
            "hole index {bad} is not one of {} faces",
            counts.len()
        ));
    }
    let expected = |interpolation: Interpolation| match interpolation {
        Interpolation::Vertex | Interpolation::Varying => points.len(),
        Interpolation::FaceVarying => corners.len(),
        Interpolation::Uniform => counts.len(),
        Interpolation::Constant => 1,
    };
    let normals = authored(mesh);
    if let Some(n) = normals
        && n.values.len() != expected(n.interpolation)
    {
        return Err(format!(
            "{} normals for {:?} interpolation, which needs {}",
            n.values.len(),
            n.interpolation,
            expected(n.interpolation)
        ));
    }
    let texcoord = texcoord(mesh);
    if let Some(t) = texcoord {
        let refined = topology.refined.is_some();
        let cage = first.cage_texcoord;
        refined_texcoord(
            t,
            refined,
            corners.len(),
            cage,
            &mut tables.texcoord_stencils,
        )?;
    }
    if let Some(t) = texcoord
        && t.refined.is_none()
        && t.values.len() != expected(t.interpolation)
    {
        return Err(format!(
            "{} {} values for {:?} interpolation, which needs {}",
            t.values.len(),
            t.name,
            t.interpolation,
            expected(t.interpolation)
        ));
    }
    match normals {
        Some(_) => (tables.corner_offsets).extend(std::iter::repeat_n(
            tables.corners.len() as u32,
            faces.points,
        )),
        None => point_corners(&faces, topology.left_handed, first.point, tables),
    }

    // Welded by value (bitwise, as the delegate welds), so the tangent pass sums both
    // sides of an edge; a seam keeps them apart.
    let welded_texcoord: Vec<u32> = match texcoord {
        Some(t) if t.refined.is_none() && t.interpolation == Interpolation::FaceVarying => {
            let mut first_of: HashMap<[u32; 2], u32> = HashMap::new();
            (t.values.iter().enumerate())
                .map(|(c, v)| *first_of.entry(v.map(f32::to_bits)).or_insert(c as u32))
                .collect()
        }
        _ => Vec::new(),
    };

    let first_vertex = tables.vertices.len();
    let first_index = tables.indices.len();
    let mut welded: HashMap<[u32; 3], u32> = HashMap::new();
    let mut vertex = |point: i32, normal: Option<usize>, texcoord: Option<usize>| {
        let triple = [
            first.point + point as u32,
            normal.map_or(SMOOTH, |n| first.normal + n as u32),
            texcoord.map_or(NONE, |t| first.texcoord + t as u32),
        ];
        *welded.entry(triple).or_insert_with(|| {
            tables.vertices.push([triple[0], triple[1]]);
            tables.vertex_texcoord.push(triple[2]);
            (tables.vertices.len() - 1 - first_vertex) as u32
        })
    };
    let index =
        |interpolation: Interpolation, point: i32, c: usize, face: usize| match interpolation {
            Interpolation::Vertex | Interpolation::Varying => point as usize,
            Interpolation::FaceVarying => c,
            Interpolation::Uniform => face,
            Interpolation::Constant => 0,
        };
    let texcoord_index = |t: &Texcoord, point: i32, c: usize, face: usize| match &t.refined {
        Some(r) => r.face_vertex_indices[c] as usize,
        None if t.interpolation == Interpolation::FaceVarying => welded_texcoord[c] as usize,
        None => index(t.interpolation, point, c, face),
    };
    let mut holes = vec![false; counts.len()];
    for &f in faces.holes {
        holes[f as usize] = true;
    }
    let mut starts = Vec::with_capacity(counts.len());
    let mut corner = 0;
    for &count in counts {
        starts.push(corner);
        corner += count as usize;
    }
    let mut face_vertices = Vec::new();
    let mut parts: Vec<Part> = Vec::new();
    for (subset, face) in ordered_faces(topology, &faces, &holes)? {
        if parts.last().is_none_or(|p| p.subset != subset) {
            parts.push(Part {
                subset,
                first_index: tables.indices.len() as u32,
                index_count: 0,
            });
        }
        let (start, end) = (starts[face], starts[face] + counts[face] as usize);
        face_vertices.clear();
        for (c, &point) in corners.iter().enumerate().take(end).skip(start) {
            let normal = normals.map(|n| index(n.interpolation, point, c, face));
            let texcoord = texcoord.map(|t| texcoord_index(t, point, c, face));
            face_vertices.push(vertex(point, normal, texcoord));
        }
        for i in 1..face_vertices.len().saturating_sub(1) {
            tables
                .indices
                .extend([face_vertices[0], face_vertices[i], face_vertices[i + 1]]);
        }
        parts.last_mut().unwrap().index_count = tables.indices.len() as u32;
    }
    for part in &mut parts {
        part.index_count -= part.first_index;
    }
    vertex_corners(first_vertex, first_index, tables);
    Ok((faces.points, parts))
}

/// A face in two subsets goes in the first. OpenSubdiv's uniform refinement lists a
/// cage face's children together in cage order: one per corner under catmullClark
/// and bilinear, four under loop.
fn ordered_faces(
    topology: &Topology,
    faces: &Faces,
    holes: &[bool],
) -> Result<Vec<(Option<usize>, usize)>, String> {
    let subsets = &topology.subsets;
    let mut parts: Vec<Vec<usize>> = vec![Vec::new(); subsets.len() + 1];
    let mut part = vec![None; faces.counts.len()];
    let cage = &topology.face_vertex_counts;
    if !subsets.is_empty() {
        let mut starts = Vec::with_capacity(cage.len() + 1);
        let mut start = 0;
        for &count in cage {
            starts.push(start);
            start += match (&topology.refined, topology.scheme) {
                (None, _) => 1,
                (Some(_), Scheme::Loop) => 4,
                (Some(_), _) => count.max(0) as usize,
            };
        }
        starts.push(start);
        if start != faces.counts.len() {
            return Err(format!(
                "{} faces drawn for a cage of {} faces, which refine to {start}",
                faces.counts.len(),
                cage.len()
            ));
        }
        for (k, subset) in subsets.iter().enumerate() {
            for &f in &subset.faces {
                if f < 0 || f as usize >= cage.len() {
                    return Err(format!(
                        "subset face {f} is not one of {} faces",
                        cage.len()
                    ));
                }
                for p in &mut part[starts[f as usize]..starts[f as usize + 1]] {
                    p.get_or_insert(k);
                }
            }
        }
    }
    for (face, &k) in part.iter().enumerate() {
        if !holes[face] {
            parts[k.map_or(0, |k| k + 1)].push(face);
        }
    }
    Ok((parts.into_iter().enumerate())
        .flat_map(|(k, faces)| faces.into_iter().map(move |f| (k.checked_sub(1), f)))
        .collect())
}

fn vertex_corners(first_vertex: usize, first_index: usize, tables: &mut Tables) {
    let count = tables.vertices.len() - first_vertex;
    let mut uses = vec![0u32; count];
    for &v in &tables.indices[first_index..] {
        uses[v as usize] += 1;
    }
    let mut slot = Vec::with_capacity(count);
    let mut at = tables.vertex_corners.len() as u32;
    for n in uses {
        tables.vertex_offsets.push(at);
        slot.push(at);
        at += n;
    }
    tables.vertex_corners.resize(at as usize, 0);
    for (i, &v) in tables.indices[first_index..].iter().enumerate() {
        let s = &mut slot[v as usize];
        tables.vertex_corners[*s as usize] = (first_index + i) as u32;
        *s += 1;
    }
}

fn refined_texcoord(
    t: &Texcoord,
    refined: bool,
    corners: usize,
    first_cage: u32,
    tables: &mut StencilTables,
) -> Result<(), String> {
    let r = match (&t.refined, refined) {
        (Some(r), true) => r,
        (None, false) => return Ok(()),
        (Some(_), false) => {
            return Err(format!("its {} is refined, but its cage is drawn", t.name));
        }
        (None, true) => return Err(format!("its {} is not refined with it", t.name)),
    };
    stencils(&r.stencils, t.values.len(), first_cage, tables)?;
    if r.face_vertex_indices.len() != corners {
        return Err(format!(
            "{} refined {} indices for {corners} refined corners",
            r.face_vertex_indices.len(),
            t.name
        ));
    }
    let count = r.stencils.count();
    if let Some(&bad) = (r.face_vertex_indices.iter()).find(|&&i| i < 0 || i as usize >= count) {
        return Err(format!(
            "refined {} index {bad} is not one of {count} refined values",
            t.name
        ));
    }
    Ok(())
}

fn stencils(
    s: &store::Stencils,
    cage_count: usize,
    first_cage: u32,
    tables: &mut StencilTables,
) -> Result<(), String> {
    let (offsets, entries) = (&s.offsets, s.indices.len());
    let ordered = offsets.first() == Some(&0) && offsets.is_sorted();
    if !ordered || offsets.last() != Some(&(entries as u32)) || s.weights.len() != entries {
        return Err(format!(
            "its stencil offsets do not divide its {entries} stencil entries"
        ));
    }
    if let Some(&bad) = (s.indices.iter()).find(|&&p| p < 0 || p as usize >= cage_count) {
        return Err(format!(
            "stencil index {bad} is not one of {cage_count} values"
        ));
    }
    // The cull pass's boxes rely on every stencil being an affine
    // combination (see `spread`), as subdivision's are.
    if let Some(p) = (offsets.windows(2)).position(|w| {
        let sum: f32 = s.weights[w[0] as usize..w[1] as usize].iter().sum();
        (sum - 1.0).abs() > 1e-3
    }) {
        return Err(format!("stencil {p}'s weights do not sum to 1"));
    }
    let first_entry = tables.indices.len() as u32;
    let starts = &offsets[..offsets.len() - 1];
    (tables.offsets).extend(starts.iter().map(|&o| first_entry + o));
    (tables.indices).extend(s.indices.iter().map(|&p| first_cage + p as u32));
    (tables.weights).extend_from_slice(&s.weights);
    Ok(())
}

/// Weights sum to 1, so only negative ones (Catmull-Clark at valence 2) push a
/// refined point outside the cage box, by at most their sum times its size.
fn spread(refined: &Refined) -> f32 {
    (refined.stencils.offsets.windows(2))
        .map(|w| {
            let weights = &refined.stencils.weights[w[0] as usize..w[1] as usize];
            weights
                .iter()
                .filter(|&&w| w < 0.0)
                .map(|w| -w)
                .sum::<f32>()
        })
        .fold(0.0, f32::max)
}

/// Low corner, then high; holds every point drawn.
fn mesh_box(mesh: &Mesh, spread: f32) -> [f32; 6] {
    let points = mesh.points.as_deref().unwrap_or_default();
    if points.is_empty() {
        return [0.0; 6];
    }
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let grow: [f32; 3] = std::array::from_fn(|k| spread * (hi[k] - lo[k]));
    [
        lo[0] - grow[0],
        lo[1] - grow[1],
        lo[2] - grow[2],
        hi[0] + grow[0],
        hi[1] + grow[1],
        hi[2] + grow[2],
    ]
}

/// As Hydra's Hd_VertexAdjacency: swapped on left-handed meshes, hole faces kept
/// (a hole is not drawn but shapes the surface around it).
fn point_corners(faces: &Faces, left_handed: bool, first_point: u32, tables: &mut Tables) {
    let indices = faces.corners;
    let mut uses = vec![0u32; faces.points];
    for &p in indices {
        uses[p as usize] += 1;
    }
    let mut slot = Vec::with_capacity(faces.points);
    let mut at = tables.corners.len() as u32;
    for n in uses {
        tables.corner_offsets.push(at);
        slot.push(at);
        at += n;
    }
    tables.corners.resize(at as usize, [0; 2]);
    let mut first = 0;
    for &count in faces.counts {
        let face = &indices[first..first + count as usize];
        for (j, &point) in face.iter().enumerate() {
            let prev = first_point + face[(j + face.len() - 1) % face.len()] as u32;
            let next = first_point + face[(j + 1) % face.len()] as u32;
            let s = &mut slot[point as usize];
            tables.corners[*s as usize] = match left_handed {
                false => [prev, next],
                true => [next, prev],
            };
            *s += 1;
        }
        first += count as usize;
    }
}

/// Empty `tags` draws every tag. Grouped by material, the unbound parts first.
fn draws(store: &Store, ranges: &BTreeMap<u64, Range>, tags: &[String]) -> DrawTables {
    let instancers = by_path(store);
    let links = Links::new(store);
    let materials: HashMap<&str, u64> = (store.materials().iter())
        .map(|(&id, m)| (m.path.as_str(), id))
        .collect();
    let mut ordered: Vec<(Option<u64>, u64, &Part)> = Vec::new();
    for (&id, range) in ranges {
        let mesh = &store.meshes()[&id];
        for part in &range.parts {
            ordered.push((part_material(mesh, part, &materials), id, part));
        }
    }
    ordered.sort_by_key(|&(material, id, part)| (material, id, part.first_index));
    let mut t = DrawTables {
        instances: Vec::new(),
        owners: Vec::new(),
        boxes: Vec::new(),
        records: Vec::new(),
        groups: Vec::new(),
        rows: Rows::new(),
    };
    for (material, id, part) in ordered {
        let (mesh, range) = (&store.meshes()[&id], &ranges[&id]);
        let first = t.instances.len();
        t.instances
            .extend(mesh_instances(&instancers, &links, mesh, range, tags));
        if t.instances.len() == first {
            continue;
        }
        let record = t.records.len() as u32;
        match t.groups.last_mut() {
            Some(g) if g.material == material => g.count += 1,
            _ => t.groups.push(Group {
                material,
                first: record,
                count: 1,
            }),
        }
        t.owners.resize(t.instances.len(), record);
        t.rows.entry(id).or_default().push(Row {
            record,
            instances: first..t.instances.len(),
        });
        t.boxes.push(mesh_box(mesh, range.spread));
        t.records.push(vk::DrawIndexedIndirectCommand {
            index_count: part.index_count,
            instance_count: 0,
            first_index: part.first_index,
            vertex_offset: range.first_vertex,
            first_instance: first as u32,
        });
    }
    t
}

/// A subset binding none uses the mesh's; `None` when Hydra has not made it.
fn part_material(mesh: &Mesh, part: &Part, materials: &HashMap<&str, u64>) -> Option<u64> {
    let subset = part
        .subset
        .and_then(|k| mesh.topology.as_ref()?.subsets.get(k));
    let path = match subset {
        Some(s) if !s.material.is_empty() => Some(s.material.as_str()),
        _ => mesh.material.as_deref(),
    };
    path.and_then(|p| materials.get(p).copied())
}

fn rewrite(
    device: &gpu::Device,
    store: &Store,
    layout: &Layout,
    tags: &[String],
    slot: &mut Option<Draws>,
) -> Result<(), String> {
    let t = draws(store, &layout.ranges, tags);
    let draws = match slot.take() {
        Some(d)
            if d.instances.size >= size(&t.instances)
                && d.owners.size >= size(&t.owners)
                && d.boxes.size >= size(&t.boxes)
                && d.records.size >= size(&t.records) =>
        {
            d
        }
        _ => {
            let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
            let copied = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            Draws {
                instances: Buffer::device_local(device, size(&t.instances), storage)?,
                owners: Buffer::device_local(device, size(&t.owners), storage)?,
                boxes: Buffer::device_local(device, size(&t.boxes), storage)?,
                records: Buffer::device_local(device, size(&t.records), copied)?,
                tables: DrawTables::default(),
                tags: Vec::new(),
            }
        }
    };
    *slot = Some(Draws {
        tables: t,
        tags: tags.to_vec(),
        ..draws
    });
    Ok(())
}

fn draw_uploads(d: &Draws) -> [Upload<'_>; 4] {
    let t = &d.tables;
    [
        Upload::table(bytes(&t.instances), &d.instances),
        Upload::table(bytes(&t.owners), &d.owners),
        Upload::table(bytes(&t.boxes), &d.boxes),
        Upload::table(bytes(&t.records), &d.records),
    ]
}

struct Patch {
    instances: Vec<(usize, Vec<Instance>)>,
    boxes: Vec<(u32, [f32; 6])>,
}

impl Patch {
    fn apply(&self, t: &mut DrawTables) {
        for (first, instances) in &self.instances {
            t.instances[*first..first + instances.len()].copy_from_slice(instances);
        }
        for (record, b) in &self.boxes {
            t.boxes[*record as usize] = *b;
        }
    }
}

/// `None` when an instance count changed, so every record must be rewritten.
fn patch(
    store: &Store,
    ranges: &BTreeMap<u64, Range>,
    rows: &Rows,
    moved: &Moved,
    tags: &[String],
) -> Option<Patch> {
    let instancers = by_path(store);
    let links = Links::new(store);
    let mut instances = Vec::new();
    for id in &moved.transforms {
        let Some(range) = ranges.get(id) else {
            continue;
        };
        let moved = mesh_instances(&instancers, &links, &store.meshes()[id], range, tags);
        let rows = rows.get(id).map_or(&[][..], Vec::as_slice);
        if moved.len() != rows.first().map_or(0, |r| r.instances.len()) {
            return None;
        }
        for row in rows {
            instances.push((row.instances.start, moved.clone()));
        }
    }
    let mut boxes = Vec::new();
    for id in &moved.points {
        let (Some(rows), Some(range)) = (rows.get(id), ranges.get(id)) else {
            continue;
        };
        let b = mesh_box(&store.meshes()[id], range.spread);
        boxes.extend(rows.iter().map(|r| (r.record, b)));
    }
    Some(Patch { instances, boxes })
}

fn patch_uploads<'a>(p: &'a Patch, d: &Draws) -> Vec<Upload<'a>> {
    let instances = p.instances.iter().map(|(first, instances)| Upload {
        bytes: bytes(instances),
        buffer: d.instances.raw,
        offset: (first * size_of::<Instance>()) as u64,
    });
    let boxes = p.boxes.iter().map(|(record, b)| Upload {
        bytes: bytes(b),
        buffer: d.boxes.raw,
        offset: *record as u64 * 24,
    });
    instances.chain(boxes).collect()
}

fn mesh_instances(
    instancers: &HashMap<&str, &Instancer>,
    links: &Links,
    mesh: &Mesh,
    range: &Range,
    tags: &[String],
) -> Vec<Instance> {
    let (Some(true), Some(instancing), Some(transform), Some(topology)) = (
        mesh.visible,
        &mesh.instancing,
        &mesh.transform,
        &mesh.topology,
    ) else {
        return Vec::new();
    };
    let tagged = (mesh.render_tag.as_ref()).is_some_and(|t| tags.contains(t));
    if range.index_count == 0 || !(tagged || tags.is_empty()) {
        return Vec::new();
    }
    let matrices = match instancing {
        None => vec![*transform],
        Some(i) => match instance_transforms(instancers, i) {
            Ok(m) => m.iter().map(|m| multiply(transform, m)).collect(),
            Err(e) => {
                eprintln!("pika: warning: {}: {e}; not drawn", mesh.path);
                return Vec::new();
            }
        },
    };
    let link = links.mesh(mesh);
    (matrices.iter())
        .filter_map(|m| Instance::new(m, topology.left_handed, link))
        .collect()
}

fn by_path(store: &Store) -> HashMap<&str, &Instancer> {
    (store.instancers().values())
        .map(|i| (i.path.as_str(), i))
        .collect()
}

fn size<T>(values: &[T]) -> u64 {
    size_of_val(values) as u64
}

/// Only for padding-free plain values: floats, integers, the shaders' `repr(C)` records.
fn bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: callers pass padding-free plain data, so every byte is initialized.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{RefinedTexcoord, Stencils, Subset};

    fn mesh(scheme: Scheme, normals: Option<Normals>) -> Mesh {
        Mesh {
            path: "/m".into(),
            topology: Some(Topology {
                scheme,
                face_vertex_counts: vec![4, 3],
                face_vertex_indices: vec![0, 1, 2, 3, 1, 4, 2],
                left_handed: false,
                holes: vec![],
                subsets: vec![],
                refined: None,
            }),
            points: Some(vec![[0.0; 3]; 5]),
            normals: Some(normals),
            texcoord: None,
            transform: None,
            visible: None,
            instancing: None,
            material: None,
            render_tag: None,
            categories: None,
        }
    }

    fn normals(interpolation: Interpolation, count: usize) -> Option<Normals> {
        Some(Normals {
            interpolation,
            values: vec![[0.0, 0.0, 1.0]; count],
        })
    }

    fn tables(mesh: &Mesh) -> Result<Tables, String> {
        let mut t = Tables {
            vertices: vec![[7, 7]],
            vertex_texcoord: vec![7],
            stencils: StencilTables {
                offsets: vec![0],
                indices: vec![0, 0],
                weights: vec![0.5, 0.5],
            },
            ..Tables::default()
        };
        let first = Firsts {
            point: 10,
            cage: 30,
            normal: 20,
            texcoord: 40,
            cage_texcoord: 50,
        };
        triangulate(mesh, first, &mut t)?;
        Ok(t)
    }

    /// Whether every copy's bytes come back with each piece moved `shift` bytes back.
    fn round_trips(lengths: &[u64], chunk: u64, shift: u64) -> bool {
        let copies: Vec<Vec<u8>> = (lengths.iter().enumerate())
            .map(|(c, &n)| (0..n).map(|i| (c as u8) << 5 | i as u8).collect())
            .collect();
        let mut pieces = pieces(lengths, chunk);
        for p in &mut pieces {
            p.at = p.at.saturating_sub(shift);
        }
        let total: u64 = lengths.iter().sum();
        let mut staging = vec![vec![0; chunk as usize]; total.div_ceil(chunk) as usize];
        for p in &pieces {
            let from = &copies[p.copy][p.start as usize..(p.start + p.size) as usize];
            staging[p.buffer][p.at as usize..(p.at + p.size) as usize].copy_from_slice(from);
        }
        let mut out: Vec<Vec<u8>> = lengths.iter().map(|&n| vec![0; n as usize]).collect();
        for p in &pieces {
            let from = &staging[p.buffer][p.at as usize..(p.at + p.size) as usize];
            out[p.copy][p.start as usize..(p.start + p.size) as usize].copy_from_slice(from);
        }
        out == copies
    }

    #[test]
    fn staging_splits_copies_at_buffer_ends() {
        let lengths = [3, 5, 0, 20, 8, 1];
        let p = pieces(&lengths, 8);
        assert!(p.iter().all(|p| p.size > 0 && p.at + p.size <= 8));
        assert_eq!(p.last().unwrap().buffer, 4);
        assert!(round_trips(&lengths, 8, 0));
        // Negative control: pieces moved 4 bytes back overlap.
        assert!(!round_trips(&lengths, 8, 4));
    }

    #[test]
    fn staging_is_kept_only_while_it_fits_closely() {
        assert!(reuses(10, 10) && reuses(20, 10) && reuses(0, 0));
        // Negative controls: too small, kept too large, and nothing to copy.
        assert!(!reuses(9, 10));
        assert!(!reuses(583_000_000, 5_400_000));
        assert!(!reuses(5, 0));
    }

    #[test]
    fn faces_fan_and_vertices_weld() {
        let t = tables(&mesh(Scheme::None, normals(Interpolation::Vertex, 5))).unwrap();
        assert_eq!(
            t.vertices[1..],
            [[10, 20], [11, 21], [12, 22], [13, 23], [14, 24]]
        );
        assert_eq!(t.indices, [0, 1, 2, 0, 2, 3, 1, 4, 2]);
        assert_eq!((t.corner_offsets, t.corners.len()), (vec![0; 5], 0));

        let t = tables(&mesh(Scheme::None, normals(Interpolation::FaceVarying, 7))).unwrap();
        assert_eq!(t.vertices.len() - 1, 7);
        assert_eq!(t.vertices[5..], [[11, 24], [14, 25], [12, 26]]);
        let t = tables(&mesh(Scheme::None, normals(Interpolation::Uniform, 2))).unwrap();
        assert_eq!(t.vertices.len() - 1, 4 + 3);
        let t = tables(&mesh(Scheme::None, None)).unwrap();
        assert_eq!(t.vertices[1], [10, SMOOTH]);
        assert_eq!(t.vertices.len() - 1, 5);
        assert_eq!(t.vertex_texcoord[1..], [NONE; 5]);
    }

    #[test]
    fn vertex_corners_are_positions_in_the_indices() {
        let t = tables(&mesh(Scheme::None, None)).unwrap();
        assert_eq!(t.indices, [0, 1, 2, 0, 2, 3, 1, 4, 2]);
        assert_eq!(t.vertex_offsets, [0, 2, 4, 7, 8]);
        assert_eq!(t.vertex_corners, [0, 3, 1, 6, 2, 4, 8, 5, 7]);
        // Control: a second mesh's corners count from the start of the indices.
        let mut t = t;
        triangulate(
            &mesh(Scheme::None, None),
            Firsts {
                point: 0,
                cage: 0,
                normal: 0,
                texcoord: 0,
                cage_texcoord: 0,
            },
            &mut t,
        )
        .unwrap();
        assert_eq!(t.vertex_offsets[5..], [9, 11, 13, 16, 17]);
        assert_eq!(t.vertex_corners[9..11], [9, 12]);
    }

    #[test]
    fn vertices_weld_over_texcoord_too() {
        let st = |interpolation, values: Vec<[f32; 2]>| {
            Some(Texcoord {
                name: "st".into(),
                interpolation,
                values,
                refined: None,
            })
        };
        let mut m = mesh(Scheme::None, None);
        m.texcoord = Some(st(Interpolation::Vertex, vec![[0.5, 0.5]; 5]));
        let t = tables(&m).unwrap();
        assert_eq!(t.vertices.len() - 1, 5);
        assert_eq!(t.vertex_texcoord[1..], [40, 41, 42, 43, 44]);
        // A face-varying texcoord welds by value.
        let (a, b, c, d, e) = ([0., 0.], [1., 0.], [1., 1.], [0., 1.], [2., 0.]);
        let mut m = mesh(Scheme::None, normals(Interpolation::Vertex, 5));
        m.texcoord = Some(st(Interpolation::FaceVarying, vec![a, b, c, d, b, e, c]));
        let t = tables(&m).unwrap();
        assert_eq!(t.vertices.len() - 1, 5);
        assert_eq!(t.vertex_texcoord[1..], [40, 41, 42, 43, 45]);
        assert_eq!(t.indices, [0, 1, 2, 0, 2, 3, 1, 4, 2]);
        // Control: a seam splits the shared points.
        m.texcoord = Some(st(Interpolation::FaceVarying, vec![a, b, c, d, e, e, e]));
        let t = tables(&m).unwrap();
        assert_eq!(t.vertices.len() - 1, 7);
        assert_eq!(t.vertices[5..], [[11, 21], [14, 24], [12, 22]]);
        assert_eq!(t.vertex_texcoord[5..], [44, 44, 44]);
        // Control: the wrong count is refused.
        let mut m = mesh(Scheme::None, None);
        m.texcoord = Some(st(Interpolation::FaceVarying, vec![[0.5, 0.5]; 5]));
        let error = tables(&m).err().unwrap();
        assert!(
            error.contains("5 st values") && error.contains("needs 7"),
            "{error}"
        );
    }

    #[test]
    fn refined_texcoords_are_read_through_their_refined_corners() {
        fn textured() -> Mesh {
            let mut m = refined();
            m.texcoord = Some(Some(Texcoord {
                name: "st".into(),
                interpolation: Interpolation::FaceVarying,
                values: vec![[0.0, 0.0], [1.0, 1.0]],
                refined: Some(RefinedTexcoord {
                    face_vertex_indices: vec![0, 1, 2, 2, 1, 3],
                    stencils: Stencils {
                        offsets: vec![0, 1, 2, 4, 5],
                        indices: vec![0, 1, 0, 1, 1],
                        weights: vec![1.0, 1.0, 0.5, 0.5, 1.0],
                    },
                }),
            }));
            m
        }
        let t = tables(&textured()).unwrap();
        assert_eq!(t.vertex_texcoord[1..], [40, 41, 42]);
        assert_eq!(t.texcoord_stencils.offsets, [0, 1, 2, 4]);
        assert_eq!(t.texcoord_stencils.indices, [50, 51, 50, 51, 51]);
        assert_eq!(t.texcoord_stencils.weights, [1.0, 1.0, 0.5, 0.5, 1.0]);
        // Controls: unrefined on a refined mesh, refined on a cage, extra corner, missing value.
        fn texcoord(m: &mut Mesh) -> &mut Texcoord {
            m.texcoord.as_mut().unwrap().as_mut().unwrap()
        }
        fn indices(m: &mut Mesh) -> &mut Vec<i32> {
            &mut texcoord(m).refined.as_mut().unwrap().face_vertex_indices
        }
        let with = |edit: fn(&mut Mesh)| {
            let mut m = textured();
            edit(&mut m);
            tables(&m).err().unwrap_or_default()
        };
        assert!(with(|m| texcoord(m).refined = None).contains("not refined with it"));
        assert!(with(|m| m.topology.as_mut().unwrap().refined = None).contains("cage is drawn"));
        assert!(with(|m| indices(m).push(0)).contains("7 refined st indices for 6"));
        assert!(with(|m| indices(m)[0] = 4).contains("index 4 is not one of 4 refined values"));
    }

    #[test]
    fn subdivided_meshes_ignore_authored_normals() {
        let t = tables(&mesh(
            Scheme::CatmullClark,
            normals(Interpolation::Vertex, 5),
        ))
        .unwrap();
        assert_eq!(t.vertices[1..3], [[10, SMOOTH], [11, SMOOTH]]);
        assert_eq!(t.corners.len(), 7);
        // Ignored on subdivided meshes even when the count is wrong.
        assert!(tables(&mesh(Scheme::Loop, normals(Interpolation::Vertex, 4))).is_ok());
    }

    #[test]
    fn corners_follow_hydras_adjacency() {
        let t = tables(&mesh(Scheme::CatmullClark, None)).unwrap();
        assert_eq!(t.corner_offsets, [0, 1, 3, 5, 6]);
        assert_eq!(t.corners[1..5], [[10, 12], [12, 14], [11, 13], [14, 11]]);
        let mut m = mesh(Scheme::CatmullClark, None);
        m.topology.as_mut().unwrap().left_handed = true;
        let t = tables(&m).unwrap();
        assert_eq!(t.corners[1..3], [[12, 10], [14, 12]]);
    }

    #[test]
    fn holes_are_not_drawn_but_shape_normals() {
        let mut m = mesh(Scheme::CatmullClark, None);
        m.topology.as_mut().unwrap().holes = vec![1];
        let t = tables(&m).unwrap();
        assert_eq!(t.indices, [0, 1, 2, 0, 2, 3]);
        // The hole's corners still count, as Hydra counts them.
        assert_eq!(t.corners.len(), 7);
        // Control: without its hole both faces draw.
        assert_eq!(
            tables(&mesh(Scheme::CatmullClark, None))
                .unwrap()
                .indices
                .len(),
            9
        );
    }

    fn refined() -> Mesh {
        let mut m = mesh(Scheme::CatmullClark, None);
        m.topology.as_mut().unwrap().refined = Some(Refined {
            face_vertex_counts: vec![3, 3],
            face_vertex_indices: vec![0, 1, 2, 2, 1, 3],
            holes: vec![1],
            stencils: Stencils {
                offsets: vec![0, 1, 3, 4, 6],
                indices: vec![0, 1, 2, 3, 4, 0],
                weights: vec![1.0, 0.5, 0.5, 1.0, 0.25, 0.75],
            },
        });
        m
    }

    #[test]
    fn refined_meshes_draw_their_refined_faces() {
        let t = tables(&refined()).unwrap();
        assert_eq!(t.indices, [0, 1, 2]);
        assert_eq!(t.vertices[1..], [[10, SMOOTH], [11, SMOOTH], [12, SMOOTH]]);
        assert_eq!(t.corner_offsets, [0, 1, 3, 5]);
        assert_eq!(t.stencils.offsets, [0, 2, 3, 5, 6]);
        assert_eq!(t.stencils.indices[2..], [30, 31, 32, 33, 34, 30]);
        assert_eq!(t.stencils.weights.len(), 2 + 6);
        // The control: the cage alone draws its own three triangles.
        assert_eq!(
            tables(&mesh(Scheme::CatmullClark, None))
                .unwrap()
                .indices
                .len(),
            9
        );
    }

    #[test]
    fn bad_stencils_are_refused() {
        let with = |edit: fn(&mut Refined)| {
            let mut m = refined();
            edit(m.topology.as_mut().unwrap().refined.as_mut().unwrap());
            tables(&m).err().unwrap_or_default()
        };
        assert!(with(|r| r.stencils.indices[5] = 5).contains("index 5 is not one of 5"));
        assert!(with(|r| r.stencils.offsets[4] = 5).contains("do not divide"));
        assert!(with(|r| r.stencils.offsets.swap(1, 2)).contains("do not divide"));
        assert!(with(|r| r.face_vertex_indices[5] = 4).contains("index 4 is not one of 4"));
        assert!(with(|r| r.holes[0] = 2).contains("hole index 2"));
        assert!(
            with(|r| r.stencils.weights[0] = 0.5).contains("stencil 0's weights do not sum to 1")
        );
    }

    fn refined_points(r: &Refined, cage: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let s = &r.stencils;
        (s.offsets.windows(2))
            .map(|w| {
                let entries = w[0] as usize..w[1] as usize;
                std::array::from_fn(|k| {
                    (entries.clone())
                        .map(|e| s.weights[e] * cage[s.indices[e] as usize][k])
                        .sum()
                })
            })
            .collect()
    }

    #[test]
    fn boxes_hold_every_refined_point() {
        let mut m = refined();
        m.points = Some(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 1.0],
            [2.0, 1.0, 0.0],
        ]);
        // Point 3 leans on a negative weight and lands outside the cage's box.
        let r = m.topology.as_mut().unwrap().refined.as_mut().unwrap();
        r.stencils.weights[4..6].copy_from_slice(&[-0.5, 1.5]);
        let r = r.clone();
        let points = refined_points(&r, m.points.as_ref().unwrap());
        assert_eq!(points[3], [-1.0, -0.5, 0.0]);
        let inside =
            |b: [f32; 6]| (points.iter()).all(|p| (0..3).all(|k| b[k] <= p[k] && p[k] <= b[k + 3]));
        assert_eq!(spread(&r), 0.5);
        assert!(inside(mesh_box(&m, spread(&r))));
        // The control: the cage's own box misses it.
        assert!(!inside(mesh_box(&m, 0.0)));
    }

    #[test]
    fn mismatched_data_is_refused() {
        let error = tables(&mesh(Scheme::None, normals(Interpolation::Vertex, 4)))
            .err()
            .unwrap();
        assert!(error.contains("needs 5"), "{error}");
        let mut m = mesh(Scheme::None, None);
        m.topology.as_mut().unwrap().face_vertex_indices[6] = 5;
        assert!(tables(&m).err().unwrap().contains("index 5"));
        let mut m = mesh(Scheme::None, None);
        m.topology.as_mut().unwrap().face_vertex_counts[1] = 4;
        assert!(tables(&m).is_err());
        let mut m = mesh(Scheme::None, None);
        m.topology.as_mut().unwrap().holes = vec![2];
        assert!(tables(&m).err().unwrap().contains("hole index 2"));
    }

    const IDENTITY: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    fn drawn(meshes: &[(&str, &str)]) -> (Store, BTreeMap<u64, Range>) {
        let mut store = Store::new();
        let mut ranges = BTreeMap::new();
        for &(path, tag) in meshes {
            let id = store.insert_mesh(path.into());
            let m = mesh(Scheme::None, None);
            store.set_topology(id, m.topology.unwrap()).unwrap();
            store.set_transform(id, IDENTITY).unwrap();
            store.set_visible(id, true).unwrap();
            store.set_instancing(id, None).unwrap();
            store.set_render_tag(id, tag.into()).unwrap();
            let range = Range {
                first_point: 0,
                point_count: 5,
                first_cage: None,
                first_normal: 0,
                normal_count: 0,
                first_texcoord: 0,
                texcoord_count: 0,
                first_cage_texcoord: None,
                first_vertex: 0,
                vertex_count: 5,
                index_count: 9,
                parts: vec![Part {
                    subset: None,
                    first_index: 0,
                    index_count: 9,
                }],
                spread: 0.0,
                density: None,
            };
            ranges.insert(id, range);
        }
        (store, ranges)
    }

    #[test]
    fn draws_are_of_the_passes_tags() {
        let (store, ranges) = drawn(&[("/final", "render"), ("/draft", "proxy")]);
        let drawn = |tags: &[&str]| {
            let tags: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
            draws(&store, &ranges, &tags).records.len()
        };
        // A retagged mesh keeps its data in the store, so drawing filters.
        assert_eq!(drawn(&["geometry", "render"]), 1);
        assert_eq!(drawn(&["geometry", "proxy"]), 1);
        assert_eq!(drawn(&["guide"]), 0);
        // No tags draws every tag, as Hydra's own filter does.
        assert_eq!(drawn(&[]), 2);
    }

    #[test]
    fn records_are_grouped_by_material() {
        let (mut store, ranges) = drawn(&[("/a", "render"), ("/b", "render"), ("/c", "render")]);
        let red = store.insert_material("/mtl/red".into());
        let ids: Vec<u64> = ranges.keys().copied().collect();
        store.set_material(ids[0], "/mtl/red".into()).unwrap();
        store.set_material(ids[2], "/mtl/red".into()).unwrap();
        store.set_material(ids[1], "/mtl/missing".into()).unwrap();
        let t = draws(&store, &ranges, &[]);
        let group = |material, first, count| Group {
            material,
            first,
            count,
        };
        assert_eq!(t.groups, [group(None, 0, 1), group(Some(red), 1, 2)]);
        let record = |id| t.rows[&id][0].record;
        assert_eq!([record(ids[1]), record(ids[0]), record(ids[2])], [0, 1, 2]);
        assert_ne!(record(ids[0]), 0);
        assert_eq!(t.owners, [0, 1, 2]);
    }

    #[test]
    fn records_start_empty_and_instances_name_theirs() {
        let (store, ranges) = drawn(&[("/a", "render"), ("/b", "render")]);
        let t = draws(&store, &ranges, &[]);
        assert!(t.records.iter().all(|r| r.instance_count == 0));
        assert_eq!(t.owners, [0, 1]);
        assert_eq!(
            (t.records.iter().map(|r| r.first_instance)).collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(t.boxes.len(), 2);
        let rows: Vec<_> = t.rows.values().cloned().collect();
        assert_eq!(
            rows,
            [
                vec![Row {
                    record: 0,
                    instances: 0..1
                }],
                vec![Row {
                    record: 1,
                    instances: 1..2
                }]
            ]
        );
    }

    #[test]
    fn subsets_are_parts_drawn_with_their_materials() {
        let (mut store, mut ranges) = drawn(&[("/a", "render"), ("/b", "render")]);
        let red = store.insert_material("/mtl/red".into());
        let ids: Vec<u64> = ranges.keys().copied().collect();
        let mut topology = mesh(Scheme::None, None).topology.unwrap();
        topology.subsets = vec![Subset {
            material: "/mtl/red".into(),
            faces: vec![1],
        }];
        store.set_topology(ids[0], topology).unwrap();
        let part = |subset, first_index, index_count| Part {
            subset,
            first_index,
            index_count,
        };
        ranges.get_mut(&ids[0]).unwrap().parts = vec![part(None, 0, 6), part(Some(0), 6, 3)];
        let t = draws(&store, &ranges, &[]);
        let group = |material, first, count| Group {
            material,
            first,
            count,
        };
        assert_eq!(t.groups, [group(None, 0, 2), group(Some(red), 2, 1)]);
        let record = |r: &vk::DrawIndexedIndirectCommand| (r.first_index, r.index_count);
        let records: Vec<_> = t.records.iter().map(record).collect();
        assert_eq!(records, [(0, 6), (0, 9), (6, 3)]);
        assert_eq!(t.owners, [0, 1, 2]);
        assert_eq!(t.rows[&ids[0]].len(), 2);
        assert_eq!(t.boxes.len(), 3);
        // Control: a subset binding no material draws with the mesh's.
        store.set_material(ids[0], "/mtl/red".into()).unwrap();
        let mut topology = mesh(Scheme::None, None).topology.unwrap();
        topology.subsets = vec![Subset {
            material: String::new(),
            faces: vec![1],
        }];
        store.set_topology(ids[0], topology).unwrap();
        let t = draws(&store, &ranges, &[]);
        assert_eq!(t.groups, [group(None, 0, 1), group(Some(red), 1, 2)]);
    }

    #[test]
    fn faces_are_ordered_by_subset_and_refined_ones_by_their_cage_face() {
        let mut m = mesh(Scheme::None, None);
        let faces = |m: &Mesh| -> Result<Vec<(Option<usize>, usize)>, String> {
            let t = m.topology.as_ref().unwrap();
            let drawn = match &t.refined {
                Some(r) => Faces {
                    counts: &r.face_vertex_counts,
                    corners: &r.face_vertex_indices,
                    holes: &r.holes,
                    points: 0,
                },
                None => Faces {
                    counts: &t.face_vertex_counts,
                    corners: &t.face_vertex_indices,
                    holes: &t.holes,
                    points: 0,
                },
            };
            let mut holes = vec![false; drawn.counts.len()];
            for &f in drawn.holes {
                holes[f as usize] = true;
            }
            ordered_faces(t, &drawn, &holes)
        };
        assert_eq!(faces(&m).unwrap(), [(None, 0), (None, 1)]);
        let subset = |faces: Vec<i32>| Subset {
            material: "/mtl/red".into(),
            faces,
        };
        m.topology.as_mut().unwrap().subsets = vec![subset(vec![0]), subset(vec![0, 1])];
        assert_eq!(faces(&m).unwrap(), [(Some(0), 0), (Some(1), 1)]);
        let mut r = refined();
        let t = r.topology.as_mut().unwrap();
        t.subsets = vec![subset(vec![1])];
        t.refined.as_mut().unwrap().face_vertex_counts = vec![4; 7];
        t.refined.as_mut().unwrap().holes = vec![5];
        assert_eq!(
            faces(&r).unwrap(),
            [
                (None, 0),
                (None, 1),
                (None, 2),
                (None, 3),
                (Some(0), 4),
                (Some(0), 6)
            ]
        );
        // Controls: a face the cage lacks, and refined faces the rule does not account for.
        r.topology.as_mut().unwrap().subsets = vec![subset(vec![2])];
        assert!(faces(&r).unwrap_err().contains("subset face 2"));
        let t = r.topology.as_mut().unwrap();
        t.subsets = vec![subset(vec![1])];
        t.refined.as_mut().unwrap().face_vertex_counts = vec![4; 6];
        assert!(faces(&r).unwrap_err().contains("refine to 7"));
    }

    #[test]
    fn moved_transforms_patch_their_rows() {
        let (mut store, ranges) = drawn(&[("/a", "render"), ("/b", "render"), ("/c", "render")]);
        let mut tables = draws(&store, &ranges, &[]);
        let rows = tables.rows.clone();
        let ids: Vec<u64> = ranges.keys().copied().collect();
        let mut moved = IDENTITY;
        moved[12] = 2.0;
        store.set_transform(ids[1], moved).unwrap();
        let moved = store.take_moved();
        let rewritten = draws(&store, &ranges, &[]).instances;
        assert_ne!(tables.instances, rewritten);
        let patched = patch(&store, &ranges, &rows, &moved, &[]).unwrap();
        patched.apply(&mut tables);
        assert_eq!(tables.instances, rewritten);
        assert!(patched.boxes.is_empty());

        // Control: a flattening transform drops its instance, shifting later rows.
        store.set_transform(ids[1], [0.0; 16]).unwrap();
        let moved = store.take_moved();
        assert!(patch(&store, &ranges, &rows, &moved, &[]).is_none());
    }
}
