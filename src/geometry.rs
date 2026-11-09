//! The GPU copy of the store's meshes: one device-local buffer per kind of
//! data, with a range in each for every mesh, and the instances and draw
//! records the mesh pass reads.
//!
//! The layout (the vertex table, the triangle indices, the stencils, the
//! smooth normal pass's table and every mesh's ranges) is rebuilt only when a
//! mesh is added, removed or remodelled. A playing frame writes only the
//! points, normals and texcoords that moved, reruns the stencil and smooth
//! normal passes on the meshes whose points moved, the stencil pass on
//! the texcoords that did, the tangent pass on the vertices of meshes
//! whose points, normals or texcoord did, and rewrites the instances of
//! meshes whose transform moved. It rewrites every instance and draw record when a
//! visibility, render tag or instancer changes, or a moved mesh's instance
//! count does. Each mesh is one draw record per part (its faces in no
//! GeomSubset, then each subset's, which the layout orders its triangles
//! by), of the instances of it that the cull pass finds in a view; its box,
//! in its own space, is recomputed when its points move.
//!
//! A subdivision surface the delegate refined is drawn as its refined faces,
//! whose points the stencil pass writes from its cage, and whose texcoord it
//! writes from the cage's values likewise. A mesh's normals are
//! its authored ones only when it is not subdivided, as USD specifies; Hydra
//! hands them over on subdivided meshes too. Otherwise they are its smooth
//! normals, which the smooth normal pass computes from the points drawn.

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

/// Where one mesh's data lives in the shared buffers.
struct Range {
    first_point: u64,
    /// The points drawn: a subdivision surface's refined points.
    point_count: u32,
    /// Where a refined mesh's cage points go; `None` when its points are
    /// drawn as they are.
    first_cage: Option<u64>,
    first_normal: u64,
    normal_count: usize,
    /// The texcoord values drawn: a refined mesh's refined values.
    first_texcoord: u64,
    texcoord_count: usize,
    /// Where a refined mesh's cage texcoord values go; `None` when its
    /// values are drawn as they are.
    first_cage_texcoord: Option<u64>,
    first_vertex: i32,
    vertex_count: u32,
    index_count: u32,
    /// The mesh's triangles by material, in order, none empty.
    parts: Vec<Part>,
    /// How far a refined point can lie outside its cage's box, as a fraction
    /// of the box's size on each axis (see `stencils`); 0 when not refined.
    spread: f32,
    /// Its texcoord's units per unit of its own space (`need::density`),
    /// recomputed when its texcoord moves, not its points.
    density: Option<f32>,
}

/// The triangles of the faces of one mesh bound to one material: those in
/// no GeomSubset, or one subset's.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Part {
    /// The subset's index in the topology; `None` for the faces in none.
    subset: Option<usize>,
    first_index: u32,
    index_count: u32,
}

/// What changes only when a mesh is remodelled, and the buffers sized by it.
struct Layout {
    ranges: BTreeMap<u64, Range>,
    point_count: u32,
    vertex_count: u32,
    /// The refined points, which come first in `points`, so refined point
    /// p's stencil is the pth.
    refined_count: u32,
    /// xyz per point drawn, every mesh's in turn.
    points: Buffer,
    /// xyz per cage point of the refined meshes.
    cage: Buffer,
    stencils: Stencils,
    /// xyz per normal, as authored: per point, face, corner or mesh.
    normals: Buffer,
    /// xyz per point: its smooth normal, or zero when its mesh's normals are
    /// authored.
    smooth: Buffer,
    adjacency: Adjacency,
    /// The refined texcoord values, which come first in `texcoord`, so
    /// refined value t's stencil is the tth.
    refined_texcoord_count: u32,
    /// xy per texcoord value drawn, every mesh's in turn: a refined mesh's
    /// refined values, else its authored ones, per point or corner.
    texcoord: Buffer,
    /// xy per cage value of the refined meshes' texcoords.
    cage_texcoord: Buffer,
    texcoord_stencils: Stencils,
    /// Point and normal index (or `SMOOTH`) per unique triple of them and
    /// the texcoord index, every mesh's in turn.
    vertices: Buffer,
    /// The texcoord index (or `NONE`) per vertex.
    vertex_texcoord: Buffer,
    /// The tangent and handedness per vertex, packed into 4 bytes by the
    /// tangent pass (`shaders/view.slang`).
    tangents: Buffer,
    vertex_adjacency: Adjacency,
    /// Three vertex indices per triangle, counted from the mesh's first vertex.
    indices: Buffer,
}

/// Refined values' stencils, which the stencil pass sums.
struct Stencils {
    /// Where each refined value's stencil starts, and the end of the last.
    offsets: Buffer,
    /// The cage value and weight of each stencil entry.
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

/// Each point's corners, whose faces the smooth normal pass sums, or each
/// vertex's, whose triangles the tangent pass sums.
struct Adjacency {
    /// Where each point's (or vertex's) corners start, and the end of the
    /// last one's.
    offsets: Buffer,
    /// Per corner, grouped by point: the previous and next point of its
    /// face; grouped by vertex: its position among the triangle indices.
    corners: Buffer,
}

/// One of a drawn mesh's draw records and its rows of the instances.
#[derive(Clone, Debug, PartialEq)]
struct Row {
    record: u32,
    instances: std::ops::Range<usize>,
}

/// Each drawn mesh's rows, one per part.
type Rows = BTreeMap<u64, Vec<Row>>;

/// The instances and their draw records.
#[derive(Default)]
struct DrawTables {
    instances: Vec<Instance>,
    /// Each instance's draw record.
    owners: Vec<u32>,
    /// Each record's box: its mesh's low corner, then its high one.
    boxes: Vec<[f32; 6]>,
    /// No record has instances until the cull pass counts a view's.
    records: Vec<vk::DrawIndexedIndirectCommand>,
    groups: Vec<Group>,
    rows: Rows,
}

/// The consecutive draw records of the meshes bound to one material, which
/// the material pass draws with one multi-draw.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Group {
    /// The store's material id; none for meshes with no material bound.
    pub material: Option<u64>,
    pub first: u32,
    pub count: u32,
}

/// The instances and their draw records, sized to fit.
struct Draws {
    instances: Buffer,
    owners: Buffer,
    boxes: Buffer,
    records: Buffer,
    /// What the buffers hold, kept on the host for the need estimate
    /// (`footprints`); a patch writes both.
    tables: DrawTables,
    /// The render tags they were written for.
    tags: Vec<String>,
}

/// The buffers the mesh pass reads, and the passes that write them.
pub struct Geometry {
    layout: Option<Layout>,
    draws: Option<Draws>,
    /// The host buffers the frame's uploads are copied from, kept while
    /// frames keep uploading and dropped by the first that does not.
    staging: Vec<Buffer>,
    subdiv: Subdiv,
    smooth: Smooth,
    tangent: Tangent,
}

/// The addresses and buffers the cull and mesh passes bind.
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
    /// The draw records without instances, which each view copies.
    pub records: vk::Buffer,
    pub count: u32,
    pub instance_count: u32,
}

/// What one frame's `update` wrote.
pub struct Update {
    pub bytes: u64,
    /// The layout was rebuilt.
    pub layout: bool,
    /// The instances and draw records were rewritten.
    pub draws: bool,
}

/// One copy into a device-local buffer.
struct Upload<'a> {
    bytes: &'a [u8],
    buffer: vk::Buffer,
    offset: u64,
}

impl<'a> Upload<'a> {
    /// A whole table into the buffer sized for it.
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

    /// Records into `commands` the uploads of what moved, the refined
    /// points, smooth normals and tangents of meshes whose points moved,
    /// and barriers making them visible to the mesh pass, which draws the
    /// meshes of render `tags` (all of them when there are none), marking
    /// each step every frame, run or not. No GPU work may be pending:
    /// buffers are replaced and the staging buffer rewritten.
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

    /// Lays the store's meshes out afresh and returns the tables to upload.
    fn rebuild(&mut self, device: &gpu::Device, store: &Store) -> Result<Tables, String> {
        // Free the old buffers before allocating their replacements.
        self.layout = None;
        let (layout, tables) = build(device, store)?;
        self.layout = Some(layout);
        Ok(tables)
    }

    /// Records the stencil pass over the refined texcoord values and points
    /// in `rewrite`, the smooth normal pass over its points and the tangent
    /// pass over its vertices, each with the barrier its readers need and
    /// a mark after it.
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

    /// The buffers to draw from, once `update` has run.
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

    /// The draw records by material, once `update` has run.
    pub fn groups(&self) -> &[Group] {
        self.draws.as_ref().map_or(&[], |d| &d.tables.groups)
    }

    /// Where each material's meshes are drawn through `view_projection`
    /// into `size` pixels, by the store's material id, once `update` has
    /// run: an instance outside the view adds nothing, and a material
    /// with none inside has no entry.
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

    /// The bytes of every buffer, the staging buffer on the host side.
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

/// The uploads of a rebuilt layout's tables.
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

/// The upload of a mesh's points: to its cage when it is refined, else to
/// the points drawn.
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

/// The upload of a mesh's normals, when it is drawn with its authored ones.
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

/// The upload of a mesh's texcoord values, when it has one: to the cage
/// values when it is refined, else to the values drawn.
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

/// The largest staging buffer: the least `maxMemoryAllocationSize` Vulkan
/// allows, so every driver grants it. A scene's first upload is its whole
/// layout, 4.86 GB on ALab, past the 4.29 GB NVIDIA reports.
const STAGING: u64 = 1 << 30;

/// Writes `copies` into the staging buffers, more added when they do not
/// fit, and records them into `cb` with a barrier making them visible to
/// every pass. A frame with nothing to copy frees the buffers: a scene's
/// first upload is gigabytes of host memory that playback, which moves a
/// frame's points at a time, must not keep, while reusing them between its
/// own frames saves allocating them each.
fn upload(
    device: &gpu::Device,
    staging: &mut Vec<Buffer>,
    copies: &[Upload],
    cb: vk::CommandBuffer,
) -> Result<u64, String> {
    let total: u64 = copies.iter().map(|c| c.bytes.len() as u64).sum();
    if total == 0 {
        staging.clear();
        return Ok(0);
    }
    let mut room: u64 = staging.iter().map(|s| s.size).sum();
    while room < total {
        let size = (total - room).min(STAGING);
        // Write-combined memory: written once in order, never read.
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        staging.push(Buffer::host(
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            host,
        )?);
        room += size;
    }
    let d = &device.raw;
    let (mut buffer, mut at) = (0, 0);
    for c in copies {
        // A copy runs on into the next buffer where one ends.
        let (mut bytes, mut offset) = (c.bytes, c.offset);
        while !bytes.is_empty() {
            if at == staging[buffer].size {
                (buffer, at) = (buffer + 1, 0);
            }
            let s = &mut staging[buffer];
            let n = (s.size - at).min(bytes.len() as u64);
            s.bytes()[at as usize..(at + n) as usize].copy_from_slice(&bytes[..n as usize]);
            let region = vk::BufferCopy {
                src_offset: at,
                dst_offset: offset,
                size: n,
            };
            // SAFETY: `cb` is recording outside any rendering, both buffers
            // were created on `d`, and the region lies inside each.
            unsafe { d.cmd_copy_buffer(cb, s.raw, c.buffer, &[region]) };
            bytes = &bytes[n as usize..];
            (offset, at) = (offset + n, at + n);
        }
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

/// The vertex table, triangle indices, stencils and the smooth normal and
/// tangent passes' tables of every mesh, before upload.
#[derive(Default)]
struct Tables {
    vertices: Vec<[u32; 2]>,
    vertex_texcoord: Vec<u32>,
    indices: Vec<u32>,
    /// One per point, then the end of the last point's corners.
    corner_offsets: Vec<u32>,
    corners: Vec<[u32; 2]>,
    /// One per vertex, then the end of the last vertex's corners.
    vertex_offsets: Vec<u32>,
    /// Each corner's position in `indices`, grouped by vertex.
    vertex_corners: Vec<u32>,
    /// The refined points' stencils, and the refined texcoord values'.
    stencils: StencilTables,
    texcoord_stencils: StencilTables,
}

/// Stencils before upload, their entries counted from the start of the
/// tables and their cage values from the start of the cage buffer.
#[derive(Default)]
struct StencilTables {
    /// One per refined value, then the end of the last one's stencil.
    offsets: Vec<u32>,
    indices: Vec<u32>,
    weights: Vec<f32>,
}

/// Lays out every mesh with topology and points, the refined ones first. A
/// mesh whose data does not fit together is left out, with a warning.
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
        // Refined points and texcoord values are written by the stencil
        // pass, the rest uploaded.
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

/// What the compute passes rewrite: the refined points to sum from their
/// cages, the points drawn whose smooth normals to recompute, the refined
/// texcoord values to sum, and the vertices whose tangents to recompute.
struct Rewrite {
    refined: Vec<std::ops::Range<u32>>,
    points: Vec<std::ops::Range<u32>>,
    texcoords: Vec<std::ops::Range<u32>>,
    vertices: Vec<std::ops::Range<u32>>,
}

/// The rewrite for the meshes whose points, normals or texcoord moved.
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

/// `0..count` as a list of ranges, empty when there is nothing to rewrite.
fn whole(count: u32) -> Vec<std::ops::Range<u32>> {
    (count > 0).then_some(0..count).into_iter().collect()
}

/// Makes the compute passes' writes visible to `stage`'s reads.
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

/// The normals a mesh is drawn with, when they are authored and it is not
/// subdivided.
fn authored(mesh: &Mesh) -> Option<&Normals> {
    match (&mesh.topology, &mesh.normals) {
        (Some(t), Some(Some(n))) if t.scheme == Scheme::None => Some(n),
        _ => None,
    }
}

/// The texcoord a mesh is drawn with, when it has one.
fn texcoord(mesh: &Mesh) -> Option<&Texcoord> {
    mesh.texcoord.as_ref()?.as_ref()
}

/// Where a mesh's data starts in each buffer.
#[derive(Clone, Copy)]
struct Firsts {
    point: u32,
    cage: u32,
    normal: u32,
    texcoord: u32,
    cage_texcoord: u32,
}

/// The faces a mesh is drawn as: its refined ones, or its cage's.
struct Faces<'a> {
    counts: &'a [i32],
    corners: &'a [i32],
    holes: &'a [i32],
    points: usize,
}

/// Appends a mesh's vertex table and its faces but the holes, part by
/// part, split into triangles by a fan from each face's first corner, its
/// points' corners, its vertices' corners and, when refined, its stencils
/// and its texcoord's. A vertex is a unique triple of point, normal and
/// texcoord, a faceVarying texcoord's corners welded by value; each index
/// counts from the start of its buffer. Returns the points drawn and the
/// parts.
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

    // A faceVarying texcoord's corners agreeing on a value (bitwise, as the
    // delegate welds a subdivided mesh's) are one vertex, so the tangent
    // pass sums both sides of the edge; a seam keeps them apart.
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

/// The faces drawn (not the holes) in the order of their parts: those in no
/// subset, then each subset's, a face in two subsets in the first. A
/// refined face belongs to its cage face's subset: OpenSubdiv's uniform
/// refinement lists a cage face's children together, in the cage's face
/// order, one per corner under catmullClark and bilinear and four under
/// loop.
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

/// Appends each vertex's corners, from the mesh's first vertex and
/// triangle index on: the positions in the indices of the corners reading
/// it, which the tangent pass sums the triangles of.
fn vertex_corners(first_vertex: usize, first_index: usize, tables: &mut Tables) {
    let count = tables.vertices.len() - first_vertex;
    let mut uses = vec![0u32; count];
    for &v in &tables.indices[first_index..] {
        uses[v as usize] += 1;
    }
    // Where each vertex's next corner goes.
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

/// Appends a refined mesh's texcoord stencils, after checking that it was
/// refined with the mesh (a cage's texcoord must not be) and that its
/// `corners` each read one of the refined values.
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

/// Appends stencils over `cage_count` values, their entries counted from the
/// start of the tables' and their cage values from `first_cage`.
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

/// How far outside its cage's box a refined point can lie, as a fraction of
/// the box's size on each axis. A stencil's weights sum to 1, so its point
/// is the box's high corner less a weighted sum of each cage point's
/// distance below it: the positive weights pull it inside, and only the
/// negative ones, which Catmull-Clark gives a vertex of valence 2, push it
/// out, by at most their sum times the box's size.
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

/// The box around a mesh's cage points, grown by `spread` of its size: the
/// low corner, then the high one. It holds every point drawn.
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

/// Appends each point's corners, as Hydra's Hd_VertexAdjacency lists them:
/// the previous and next point of the face, swapped on a left-handed mesh so
/// the normal faces its front. Hole faces are kept, as Hydra keeps them: a
/// hole is not drawn, but the surface around it is shaped as if it were.
fn point_corners(faces: &Faces, left_handed: bool, first_point: u32, tables: &mut Tables) {
    let indices = faces.corners;
    let mut uses = vec![0u32; faces.points];
    for &p in indices {
        uses[p as usize] += 1;
    }
    // Where each point's next corner goes.
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

/// One draw record per part of each visible mesh of render `tags` (or of
/// any, when there are none), with its box and its rows of the instances
/// it is drawn at, grouped by material: the unbound parts first, then each
/// material's.
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

/// The store id of the material a part is drawn with: its subset's, or the
/// mesh's for the faces in no subset and for a subset binding none; `None`
/// when Hydra has not made it.
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

/// Rewrites every instance and draw record of render `tags` into `slot`,
/// its buffers replaced when they are too small, for `draw_uploads`.
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

/// The uploads of rewritten draws.
fn draw_uploads(d: &Draws) -> [Upload<'_>; 4] {
    let t = &d.tables;
    [
        Upload::table(bytes(&t.instances), &d.instances),
        Upload::table(bytes(&t.owners), &d.owners),
        Upload::table(bytes(&t.boxes), &d.boxes),
        Upload::table(bytes(&t.records), &d.records),
    ]
}

/// What a frame writes into draws it keeps: the instances of the meshes
/// whose transform moved, each with its first row, and the box of each
/// record whose mesh's points moved.
struct Patch {
    instances: Vec<(usize, Vec<Instance>)>,
    boxes: Vec<(u32, [f32; 6])>,
}

impl Patch {
    /// Writes it into the tables it was made against.
    fn apply(&self, t: &mut DrawTables) {
        for (first, instances) in &self.instances {
            t.instances[*first..first + instances.len()].copy_from_slice(instances);
        }
        for (record, b) in &self.boxes {
            t.boxes[*record as usize] = *b;
        }
    }
}

/// The patch for the meshes in `moved`. `None` when a mesh's instance count
/// changed, so the rows no longer fit and every record must be rewritten.
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

/// The uploads of a patch.
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

/// The instances `mesh` is drawn at in a pass of render `tags`. None when it
/// is invisible, unsynced, empty or of another tag, or when its instancers do
/// not fit together, with a warning.
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

/// The bytes of a slice of plain values: floats, integers and the `repr(C)`
/// records the shader reads, none of which has padding.
fn bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: every caller's `T` is plain data without padding, so all its
    // bytes are initialized, and u8 has no alignment requirement.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{RefinedTexcoord, Stencils, Subset};

    /// A quad and a triangle sharing an edge, with optional normals.
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

    #[test]
    fn faces_fan_and_vertices_weld() {
        let t = tables(&mesh(Scheme::None, normals(Interpolation::Vertex, 5))).unwrap();
        // Vertex normals weld to one vertex per point; indices count from
        // the mesh's first vertex, after the earlier mesh's one.
        assert_eq!(
            t.vertices[1..],
            [[10, 20], [11, 21], [12, 22], [13, 23], [14, 24]]
        );
        assert_eq!(t.indices, [0, 1, 2, 0, 2, 3, 1, 4, 2]);
        // Authored normals need no smooth normals.
        assert_eq!((t.corner_offsets, t.corners.len()), (vec![0; 5], 0));

        // Face-varying normals keep every corner apart.
        let t = tables(&mesh(Scheme::None, normals(Interpolation::FaceVarying, 7))).unwrap();
        assert_eq!(t.vertices.len() - 1, 7);
        assert_eq!(t.vertices[5..], [[11, 24], [14, 25], [12, 26]]);
        // Uniform normals split the points the two faces share.
        let t = tables(&mesh(Scheme::None, normals(Interpolation::Uniform, 2))).unwrap();
        assert_eq!(t.vertices.len() - 1, 4 + 3);
        // Smooth normals are per point, so vertices are points.
        let t = tables(&mesh(Scheme::None, None)).unwrap();
        assert_eq!(t.vertices[1], [10, SMOOTH]);
        assert_eq!(t.vertices.len() - 1, 5);
        assert_eq!(t.vertex_texcoord[1..], [NONE; 5]);
    }

    #[test]
    fn vertex_corners_are_positions_in_the_indices() {
        // The quad's fan: vertex 0 and 2 are read by both triangles, the
        // triangle's vertex 4 by one.
        let t = tables(&mesh(Scheme::None, None)).unwrap();
        assert_eq!(t.indices, [0, 1, 2, 0, 2, 3, 1, 4, 2]);
        assert_eq!(t.vertex_offsets, [0, 2, 4, 7, 8]);
        assert_eq!(t.vertex_corners, [0, 3, 1, 6, 2, 4, 8, 5, 7]);
        // The control: a second mesh's corners count from the start of the
        // indices, not its own, and its offsets from the earlier corners.
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
        // A face-varying texcoord welds by value: the triangle's corners at
        // points 1 and 2 (corners 4 and 6) read the quad's values there, so
        // they are the quad's vertices; its own corner at point 4 is new.
        let (a, b, c, d, e) = ([0., 0.], [1., 0.], [1., 1.], [0., 1.], [2., 0.]);
        let mut m = mesh(Scheme::None, normals(Interpolation::Vertex, 5));
        m.texcoord = Some(st(Interpolation::FaceVarying, vec![a, b, c, d, b, e, c]));
        let t = tables(&m).unwrap();
        assert_eq!(t.vertices.len() - 1, 5);
        assert_eq!(t.vertex_texcoord[1..], [40, 41, 42, 43, 45]);
        assert_eq!(t.indices, [0, 1, 2, 0, 2, 3, 1, 4, 2]);
        // The control: a seam, where the triangle reads other values at the
        // shared points, splits them, as face-varying normals do.
        m.texcoord = Some(st(Interpolation::FaceVarying, vec![a, b, c, d, e, e, e]));
        let t = tables(&m).unwrap();
        assert_eq!(t.vertices.len() - 1, 7);
        assert_eq!(t.vertices[5..], [[11, 21], [14, 24], [12, 22]]);
        assert_eq!(t.vertex_texcoord[5..], [44, 44, 44]);
        // The wrong count is refused, as for normals.
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
        /// The refined mesh with a texcoord refined with it: two welded
        /// values, refined into four.
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
        // Its stencils are laid out like the points', from the mesh's first
        // cage value.
        assert_eq!(t.texcoord_stencils.offsets, [0, 1, 2, 4]);
        assert_eq!(t.texcoord_stencils.indices, [50, 51, 50, 51, 51]);
        assert_eq!(t.texcoord_stencils.weights, [1.0, 1.0, 0.5, 0.5, 1.0]);
        // The controls: a texcoord not refined with its refined mesh, one
        // refined for a cage, a corner too many and a value too few.
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
        // Even when they would not fit.
        assert!(tables(&mesh(Scheme::Loop, normals(Interpolation::Vertex, 4))).is_ok());
    }

    #[test]
    fn corners_follow_hydras_adjacency() {
        let t = tables(&mesh(Scheme::CatmullClark, None)).unwrap();
        // Points 1 and 2 are on both faces, the others on one.
        assert_eq!(t.corner_offsets, [0, 1, 3, 5, 6]);
        assert_eq!(t.corners[1..5], [[10, 12], [12, 14], [11, 13], [14, 11]]);
        // Left-handed faces wind the other way.
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
        // The control: the same mesh without its hole draws both faces.
        assert_eq!(
            tables(&mesh(Scheme::CatmullClark, None))
                .unwrap()
                .indices
                .len(),
            9
        );
    }

    /// The quad and triangle refined into two triangles over four points,
    /// the second a hole.
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
        // Smooth normals come from the refined faces, the hole's included.
        assert_eq!(t.corner_offsets, [0, 1, 3, 5]);
        // Stencils count their entries after the earlier mesh's two, and
        // their cage points from the mesh's first cage point.
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

    /// Each refined point, as the stencil pass writes it.
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
        // Point 3's stencil leans on a negative weight, as Catmull-Clark's
        // does at a vertex of valence 2, and lands outside the cage's box.
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

    /// A store of drawn meshes, one per path and render tag, and their ranges.
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
        // An unbound mesh, and one bound to a material Hydra has not made.
        store.set_material(ids[1], "/mtl/missing".into()).unwrap();
        let t = draws(&store, &ranges, &[]);
        let group = |material, first, count| Group {
            material,
            first,
            count,
        };
        assert_eq!(t.groups, [group(None, 0, 1), group(Some(red), 1, 2)]);
        // The records follow the groups, not the meshes' ids.
        let record = |id| t.rows[&id][0].record;
        assert_eq!([record(ids[1]), record(ids[0]), record(ids[2])], [0, 1, 2]);
        assert_ne!(record(ids[0]), 0);
        assert_eq!(t.owners, [0, 1, 2]);
    }

    #[test]
    fn records_start_empty_and_instances_name_theirs() {
        let (store, ranges) = drawn(&[("/a", "render"), ("/b", "render")]);
        let t = draws(&store, &ranges, &[]);
        // The cull pass counts each view's instances.
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
        // The subset's record is grouped with its material, after the two
        // unbound ones, and draws its own triangles over the mesh's instance.
        assert_eq!(t.groups, [group(None, 0, 2), group(Some(red), 2, 1)]);
        let record = |r: &vk::DrawIndexedIndirectCommand| (r.first_index, r.index_count);
        let records: Vec<_> = t.records.iter().map(record).collect();
        assert_eq!(records, [(0, 6), (0, 9), (6, 3)]);
        assert_eq!(t.owners, [0, 1, 2]);
        assert_eq!(t.rows[&ids[0]].len(), 2);
        assert_eq!(t.boxes.len(), 3);
        // The control: a subset binding no material draws with the mesh's.
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
        // Refined: the quad's four children, then the triangle's three; the
        // subset of the triangle has the last three, less the hole.
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
        // The controls: a face the cage lacks, and refined faces the rule
        // does not account for.
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

        // The control: a transform that flattens the mesh drops its instance,
        // so the rows after it shift.
        store.set_transform(ids[1], [0.0; 16]).unwrap();
        let moved = store.take_moved();
        assert!(patch(&store, &ranges, &rows, &moved, &[]).is_none());
    }
}
