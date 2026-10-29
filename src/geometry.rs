//! The GPU copy of the store's meshes: one device-local buffer per kind of
//! data, with a range in each for every mesh, and the instances and draw
//! records the mesh pass reads.
//!
//! The layout (the vertex table, the triangle indices, the stencils, the
//! smooth normal pass's table and every mesh's ranges) is rebuilt only when a
//! mesh is added, removed or remodelled. A playing frame writes only the
//! points and normals that moved, reruns the stencil and smooth normal passes
//! on the meshes whose points moved, and rewrites the instances of meshes
//! whose transform moved. It rewrites every instance and draw record when a
//! visibility, render tag or instancer changes, or a moved mesh's instance
//! count does. Each mesh is one draw record, of the instances of it that the
//! cull pass finds in a view; its box, in its own space, is recomputed when
//! its points move.
//!
//! A subdivision surface the delegate refined is drawn as its refined faces,
//! whose points the stencil pass writes from its cage. A mesh's normals are
//! its authored ones only when it is not subdivided, as USD specifies; Hydra
//! hands them over on subdivided meshes too. Otherwise they are its smooth
//! normals, which the smooth normal pass computes from the points drawn.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ash::vk;

use crate::gpu::{self, Buffer};
use crate::smooth::{self, Smooth};
use crate::store::{Instancer, Interpolation, Mesh, Moved, Normals, Refined, Scheme, Store};
use crate::subdiv::{self, Subdiv};
use crate::transform::{Instance, instance_transforms, multiply};

/// A vertex's normal index when its normal is its point's smooth normal.
const SMOOTH: u32 = u32::MAX;

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
    first_vertex: i32,
    first_index: u32,
    index_count: u32,
    /// How far a refined point can lie outside its cage's box, as a fraction
    /// of the box's size on each axis (see `stencils`); 0 when not refined.
    spread: f32,
}

/// What changes only when a mesh is remodelled, and the buffers sized by it.
struct Layout {
    ranges: BTreeMap<u64, Range>,
    point_count: u32,
    /// The refined points, which come first in `points`, so refined point
    /// p's stencil is the pth.
    refined_count: u32,
    /// xyz per point drawn, every mesh's in turn.
    points: Buffer,
    /// xyz per cage point of the refined meshes.
    cage: Buffer,
    /// Where each refined point's stencil starts, and the end of the last.
    stencils: Buffer,
    /// The cage point and weight of each stencil entry.
    stencil_indices: Buffer,
    stencil_weights: Buffer,
    /// xyz per normal, as authored: per point, face, corner or mesh.
    normals: Buffer,
    /// xyz per point: its smooth normal, or zero when its mesh's normals are
    /// authored.
    smooth: Buffer,
    /// Where each point's corners start, and the end of the last point's.
    offsets: Buffer,
    /// The previous and next point of each corner, grouped by point.
    corners: Buffer,
    /// Point and normal index (or `SMOOTH`) per unique pair of them, every
    /// mesh's in turn.
    vertices: Buffer,
    /// Three vertex indices per triangle, counted from the mesh's first vertex.
    indices: Buffer,
}

/// A drawn mesh's draw record and its rows of the instances.
#[derive(Clone, Debug, PartialEq)]
struct Row {
    record: u32,
    instances: std::ops::Range<usize>,
}

type Rows = BTreeMap<u64, Row>;

/// The instances and their draw records, before upload.
struct DrawTables {
    instances: Vec<Instance>,
    /// Each instance's draw record.
    owners: Vec<u32>,
    /// Each record's box: its mesh's low corner, then its high one.
    boxes: Vec<[f32; 6]>,
    /// No record has instances until the cull pass counts a view's.
    records: Vec<vk::DrawIndexedIndirectCommand>,
    rows: Rows,
}

/// The instances and their draw records, sized to fit.
struct Draws {
    instances: Buffer,
    owners: Buffer,
    boxes: Buffer,
    records: Buffer,
    count: u32,
    instance_count: u32,
    rows: Rows,
}

/// The buffers the mesh pass reads.
pub struct Geometry {
    layout: Option<Layout>,
    draws: Option<Draws>,
    /// The render tags the draws were written for.
    tags: Vec<String>,
    /// The host buffer each frame's uploads are copied from, grown as needed.
    staging: Option<Buffer>,
}

/// The addresses and buffers the cull and mesh passes bind.
pub struct Bound {
    pub points: vk::DeviceAddress,
    pub normals: vk::DeviceAddress,
    pub smooth: vk::DeviceAddress,
    pub vertices: vk::DeviceAddress,
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

impl Geometry {
    pub fn new() -> Self {
        Self {
            layout: None,
            draws: None,
            tags: Vec::new(),
            staging: None,
        }
    }

    /// Records into `cb` the uploads of what moved, the refined points and
    /// smooth normals of meshes whose points moved, and barriers making them
    /// visible to the mesh pass, which draws the meshes of render `tags` (all
    /// of them when there are none). No GPU work may be pending: buffers are
    /// replaced and the staging buffer rewritten.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        device: &gpu::Device,
        store: &Store,
        moved: &Moved,
        tags: &[String],
        subdiv: &Subdiv,
        smooth: &Smooth,
        cb: vk::CommandBuffer,
    ) -> Result<Update, String> {
        let mut copies = Vec::new();
        let built;
        let rebuilt = moved.layout || self.layout.is_none();
        if rebuilt {
            self.layout = None;
            let (layout, tables) = build(device, store)?;
            built = tables;
            let layout = self.layout.insert(layout);
            copies.push(Upload {
                bytes: bytes(&built.vertices),
                buffer: layout.vertices.raw,
                offset: 0,
            });
            copies.push(Upload {
                bytes: bytes(&built.indices),
                buffer: layout.indices.raw,
                offset: 0,
            });
            copies.push(Upload {
                bytes: bytes(&built.offsets),
                buffer: layout.offsets.raw,
                offset: 0,
            });
            copies.push(Upload {
                bytes: bytes(&built.corners),
                buffer: layout.corners.raw,
                offset: 0,
            });
            copies.push(Upload {
                bytes: bytes(&built.stencils),
                buffer: layout.stencils.raw,
                offset: 0,
            });
            copies.push(Upload {
                bytes: bytes(&built.stencil_indices),
                buffer: layout.stencil_indices.raw,
                offset: 0,
            });
            copies.push(Upload {
                bytes: bytes(&built.stencil_weights),
                buffer: layout.stencil_weights.raw,
                offset: 0,
            });
        }
        let layout = self.layout.as_ref().expect("built above");
        for (&id, range) in &layout.ranges {
            let mesh = &store.meshes()[&id];
            if rebuilt || moved.points.contains(&id) {
                let points = mesh.points.as_deref().unwrap_or_default();
                let (buffer, first) = match range.first_cage {
                    Some(first) => (layout.cage.raw, first),
                    None => (layout.points.raw, range.first_point),
                };
                copies.push(Upload {
                    bytes: bytes(points),
                    buffer,
                    offset: first * 12,
                });
            }
            if (rebuilt || moved.normals.contains(&id)) && range.normal_count > 0 {
                let Some(normals) = authored(mesh) else {
                    continue;
                };
                copies.push(Upload {
                    bytes: bytes(&normals.values),
                    buffer: layout.normals.raw,
                    offset: range.first_normal * 12,
                });
            }
        }

        let tables;
        let mut patches = Vec::new();
        let mut boxes = Vec::new();
        let mut rewrite = rebuilt || moved.draws || self.draws.is_none() || self.tags != tags;
        if let (false, Some(draws)) = (rewrite, &self.draws) {
            match patch(store, &layout.ranges, &draws.rows, &moved.transforms, tags) {
                Some(p) => patches = p,
                None => rewrite = true,
            }
            for (first, instances) in &patches {
                copies.push(Upload {
                    bytes: bytes(instances),
                    buffer: draws.instances.raw,
                    offset: (first * size_of::<Instance>()) as u64,
                });
            }
            for id in &moved.points {
                let (Some(row), Some(range)) = (draws.rows.get(id), layout.ranges.get(id)) else {
                    continue;
                };
                boxes.push((row.record, mesh_box(&store.meshes()[id], range.spread)));
            }
            for (record, b) in &boxes {
                copies.push(Upload {
                    bytes: bytes(b),
                    buffer: draws.boxes.raw,
                    offset: *record as u64 * 24,
                });
            }
        }
        if rewrite {
            self.tags = tags.to_vec();
            tables = draws(store, &layout.ranges, tags);
            let t = &tables;
            let draws = match self.draws.take() {
                Some(d)
                    if d.instances.size >= size(&t.instances)
                        && d.owners.size >= size(&t.owners)
                        && d.boxes.size >= size(&t.boxes)
                        && d.records.size >= size(&t.records) =>
                {
                    d
                }
                _ => {
                    let storage =
                        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
                    let copied =
                        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
                    Draws {
                        instances: Buffer::device_local(device, size(&t.instances), storage)?,
                        owners: Buffer::device_local(device, size(&t.owners), storage)?,
                        boxes: Buffer::device_local(device, size(&t.boxes), storage)?,
                        records: Buffer::device_local(device, size(&t.records), copied)?,
                        count: 0,
                        instance_count: 0,
                        rows: Rows::new(),
                    }
                }
            };
            let draws = self.draws.insert(Draws {
                count: t.records.len() as u32,
                instance_count: t.instances.len() as u32,
                rows: t.rows.clone(),
                ..draws
            });
            copies.push(Upload {
                bytes: bytes(&t.instances),
                buffer: draws.instances.raw,
                offset: 0,
            });
            copies.push(Upload {
                bytes: bytes(&t.owners),
                buffer: draws.owners.raw,
                offset: 0,
            });
            copies.push(Upload {
                bytes: bytes(&t.boxes),
                buffer: draws.boxes.raw,
                offset: 0,
            });
            copies.push(Upload {
                bytes: bytes(&t.records),
                buffer: draws.records.raw,
                offset: 0,
            });
        }
        let bytes = self.record(device, &copies, cb)?;
        let layout = self.layout.as_ref().expect("built above");
        let (refined, points) = match rebuilt {
            true => (whole(layout.refined_count), whole(layout.point_count)),
            false => moved_points(&layout.ranges, &moved.points),
        };
        if !refined.is_empty() {
            let buffers = subdiv::Buffers {
                cage: layout.cage.address,
                offsets: layout.stencils.address,
                indices: layout.stencil_indices.address,
                weights: layout.stencil_weights.address,
                points: layout.points.address,
            };
            subdiv.record(cb, buffers, &refined);
            barrier(device, cb, vk::PipelineStageFlags2::COMPUTE_SHADER);
        }
        if !points.is_empty() {
            let buffers = smooth::Buffers {
                points: layout.points.address,
                offsets: layout.offsets.address,
                corners: layout.corners.address,
                normals: layout.smooth.address,
            };
            smooth.record(cb, buffers, &points);
            barrier(device, cb, vk::PipelineStageFlags2::VERTEX_SHADER);
        }
        Ok(Update {
            bytes,
            layout: rebuilt,
            draws: rewrite,
        })
    }

    /// The buffers to draw from, once `update` has run.
    pub fn bound(&self) -> Option<Bound> {
        let (layout, draws) = (self.layout.as_ref()?, self.draws.as_ref()?);
        Some(Bound {
            points: layout.points.address,
            normals: layout.normals.address,
            smooth: layout.smooth.address,
            vertices: layout.vertices.address,
            instances: draws.instances.address,
            owners: draws.owners.address,
            boxes: draws.boxes.address,
            indices: layout.indices.raw,
            records: draws.records.raw,
            count: draws.count,
            instance_count: draws.instance_count,
        })
    }

    /// The bytes of every buffer, the staging buffer on the host side.
    pub fn memory(&self) -> gpu::Memory {
        let layout = self.layout.iter().flat_map(|l| {
            [
                &l.points,
                &l.cage,
                &l.stencils,
                &l.stencil_indices,
                &l.stencil_weights,
                &l.normals,
                &l.smooth,
                &l.offsets,
                &l.corners,
                &l.vertices,
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

    /// Writes `copies` into the staging buffer and records them into `cb`.
    fn record(
        &mut self,
        device: &gpu::Device,
        copies: &[Upload],
        cb: vk::CommandBuffer,
    ) -> Result<u64, String> {
        let total: u64 = copies.iter().map(|c| c.bytes.len() as u64).sum();
        if total == 0 {
            return Ok(0);
        }
        let staging = match self.staging.take() {
            Some(s) if s.size >= total => s,
            _ => {
                // Write-combined memory: written once in order, never read.
                let host =
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                Buffer::host(device, total, vk::BufferUsageFlags::TRANSFER_SRC, host)?
            }
        };
        let staging = self.staging.insert(staging);
        let d = &device.raw;
        let mut at = 0;
        for c in copies.iter().filter(|c| !c.bytes.is_empty()) {
            let end = at + c.bytes.len();
            staging.bytes()[at..end].copy_from_slice(c.bytes);
            let region = vk::BufferCopy {
                src_offset: at as u64,
                dst_offset: c.offset,
                size: c.bytes.len() as u64,
            };
            // SAFETY: `cb` is recording outside any rendering, both buffers
            // were created on `d`, and the region lies inside each.
            unsafe { d.cmd_copy_buffer(cb, staging.raw, c.buffer, &[region]) };
            at = end;
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
}

/// The vertex table, triangle indices, stencils and smooth normal table of
/// every mesh, before upload.
struct Tables {
    vertices: Vec<[u32; 2]>,
    indices: Vec<u32>,
    /// One per point, then the end of the last point's corners.
    offsets: Vec<u32>,
    corners: Vec<[u32; 2]>,
    /// One per refined point, then the end of the last one's stencil.
    stencils: Vec<u32>,
    stencil_indices: Vec<u32>,
    stencil_weights: Vec<f32>,
}

/// Lays out every mesh with topology and points, the refined ones first. A
/// mesh whose data does not fit together is left out, with a warning.
fn build(device: &gpu::Device, store: &Store) -> Result<(Layout, Tables), String> {
    let mut ranges = BTreeMap::new();
    let mut tables = Tables {
        vertices: Vec::new(),
        indices: Vec::new(),
        offsets: Vec::new(),
        corners: Vec::new(),
        stencils: Vec::new(),
        stencil_indices: Vec::new(),
        stencil_weights: Vec::new(),
    };
    let (mut points, mut cage, mut normals) = (0u64, 0u64, 0u64);
    let mut refined_count = 0;
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
            let first_entry = tables.stencil_indices.len();
            let normal_count = authored(mesh).map_or(0, |n| n.values.len());
            let first_cage = refined_pass.then_some(cage);
            let first = Firsts {
                point: points as u32,
                cage: cage as u32,
                normal: normals as u32,
            };
            let drawn = match triangulate(mesh, first, &mut tables) {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("pika: warning: {}: {e}; not drawn", mesh.path);
                    tables.vertices.truncate(first_vertex);
                    tables.indices.truncate(first_index);
                    tables.offsets.truncate(points as usize);
                    tables.corners.truncate(first_corner);
                    if refined_pass {
                        tables.stencils.truncate(points as usize);
                    }
                    tables.stencil_indices.truncate(first_entry);
                    tables.stencil_weights.truncate(first_entry);
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
                    first_vertex: first_vertex as i32,
                    first_index: first_index as u32,
                    index_count: (tables.indices.len() - first_index) as u32,
                    spread: t.refined.as_ref().map_or(0.0, spread),
                },
            );
            points += drawn as u64;
            normals += normal_count as u64;
            if refined_pass {
                cage += p.len() as u64;
            }
        }
        if refined_pass {
            refined_count = points as u32;
            tables.stencils.push(tables.stencil_indices.len() as u32);
        }
    }
    tables.offsets.push(tables.corners.len() as u32);
    let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let index = vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let written = vk::BufferUsageFlags::STORAGE_BUFFER;
    let layout = Layout {
        point_count: points as u32,
        refined_count,
        // Refined points are written by the stencil pass, the rest uploaded.
        points: Buffer::device_local(device, points * 12, storage)?,
        cage: Buffer::device_local(device, cage * 12, storage)?,
        stencils: Buffer::device_local(device, size(&tables.stencils), storage)?,
        stencil_indices: Buffer::device_local(device, size(&tables.stencil_indices), storage)?,
        stencil_weights: Buffer::device_local(device, size(&tables.stencil_weights), storage)?,
        normals: Buffer::device_local(device, normals * 12, storage)?,
        smooth: Buffer::device_local(device, points * 12, written)?,
        offsets: Buffer::device_local(device, size(&tables.offsets), storage)?,
        corners: Buffer::device_local(device, size(&tables.corners), storage)?,
        vertices: Buffer::device_local(device, size(&tables.vertices), storage)?,
        indices: Buffer::device_local(device, size(&tables.indices), index)?,
        ranges,
    };
    Ok((layout, tables))
}

/// The ranges of points the stencil pass and the smooth normal pass rewrite
/// when the meshes in `moved` moved: the refined points, and every point
/// drawn.
fn moved_points(
    ranges: &BTreeMap<u64, Range>,
    moved: &BTreeSet<u64>,
) -> (Vec<std::ops::Range<u32>>, Vec<std::ops::Range<u32>>) {
    let (mut refined, mut points) = (Vec::new(), Vec::new());
    for range in moved.iter().filter_map(|id| ranges.get(id)) {
        let first = range.first_point as u32;
        let drawn = first..first + range.point_count;
        if range.first_cage.is_some() {
            refined.push(drawn.clone());
        }
        points.push(drawn);
    }
    (refined, points)
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

/// Where a mesh's data starts in each buffer.
#[derive(Clone, Copy)]
struct Firsts {
    point: u32,
    cage: u32,
    normal: u32,
}

/// The faces a mesh is drawn as: its refined ones, or its cage's.
struct Faces<'a> {
    counts: &'a [i32],
    corners: &'a [i32],
    holes: &'a [i32],
    points: usize,
}

/// Appends a mesh's vertex table and its faces but the holes, split into
/// triangles by a fan from each face's first corner, its points' corners
/// and, when refined, its stencils. A vertex is a unique pair of point and
/// normal; both indices count from the start of their buffers. Returns the
/// points drawn.
fn triangulate(mesh: &Mesh, first: Firsts, tables: &mut Tables) -> Result<usize, String> {
    let (Some(topology), Some(points)) = (&mesh.topology, &mesh.points) else {
        return Ok(0);
    };
    let faces = match &topology.refined {
        Some(r) => {
            stencils(r, points.len(), first.cage, tables)?;
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
    let normals = authored(mesh);
    if let Some(n) = normals {
        let expected = match n.interpolation {
            Interpolation::Vertex | Interpolation::Varying => points.len(),
            Interpolation::FaceVarying => corners.len(),
            Interpolation::Uniform => counts.len(),
            Interpolation::Constant => 1,
        };
        if n.values.len() != expected {
            return Err(format!(
                "{} normals for {:?} interpolation, which needs {expected}",
                n.values.len(),
                n.interpolation
            ));
        }
    }
    match normals {
        Some(_) => (tables.offsets).extend(std::iter::repeat_n(
            tables.corners.len() as u32,
            faces.points,
        )),
        None => point_corners(&faces, topology.left_handed, first.point, tables),
    }

    let first_vertex = tables.vertices.len();
    let mut welded: HashMap<[u32; 2], u32> = HashMap::new();
    let mut vertex = |point: i32, normal: Option<usize>| {
        let pair = [
            first.point + point as u32,
            normal.map_or(SMOOTH, |n| first.normal + n as u32),
        ];
        *welded.entry(pair).or_insert_with(|| {
            tables.vertices.push(pair);
            (tables.vertices.len() - 1 - first_vertex) as u32
        })
    };
    let mut holes = vec![false; counts.len()];
    for &f in faces.holes {
        holes[f as usize] = true;
    }
    let mut face_vertices = Vec::new();
    let mut corner = 0;
    for (face, &count) in counts.iter().enumerate() {
        let start = corner;
        corner += count as usize;
        if holes[face] {
            continue;
        }
        face_vertices.clear();
        for (c, &point) in corners.iter().enumerate().take(corner).skip(start) {
            let normal = normals.map(|n| match n.interpolation {
                Interpolation::Vertex | Interpolation::Varying => point as usize,
                Interpolation::FaceVarying => c,
                Interpolation::Uniform => face,
                Interpolation::Constant => 0,
            });
            face_vertices.push(vertex(point, normal));
        }
        for i in 1..face_vertices.len().saturating_sub(1) {
            tables
                .indices
                .extend([face_vertices[0], face_vertices[i], face_vertices[i + 1]]);
        }
    }
    Ok(faces.points)
}

/// Appends a refined mesh's stencils, their entries counted from the start
/// of the tables' and their cage points from `first_cage`.
fn stencils(
    refined: &Refined,
    cage_count: usize,
    first_cage: u32,
    tables: &mut Tables,
) -> Result<(), String> {
    let (offsets, entries) = (&refined.offsets, refined.indices.len());
    let ordered = offsets.first() == Some(&0) && offsets.is_sorted();
    if !ordered || offsets.last() != Some(&(entries as u32)) || refined.weights.len() != entries {
        return Err(format!(
            "its stencil offsets do not divide its {entries} stencil entries"
        ));
    }
    if let Some(&bad) = (refined.indices.iter()).find(|&&p| p < 0 || p as usize >= cage_count) {
        return Err(format!(
            "stencil index {bad} is not one of {cage_count} points"
        ));
    }
    // The cull pass's boxes rely on every stencil being an affine
    // combination (see `spread`), as subdivision's are.
    if let Some(p) = (offsets.windows(2)).position(|w| {
        let sum: f32 = refined.weights[w[0] as usize..w[1] as usize].iter().sum();
        (sum - 1.0).abs() > 1e-3
    }) {
        return Err(format!("stencil {p}'s weights do not sum to 1"));
    }
    let first_entry = tables.stencil_indices.len() as u32;
    let starts = &offsets[..offsets.len() - 1];
    (tables.stencils).extend(starts.iter().map(|&o| first_entry + o));
    (tables.stencil_indices).extend(refined.indices.iter().map(|&p| first_cage + p as u32));
    (tables.stencil_weights).extend_from_slice(&refined.weights);
    Ok(())
}

/// How far outside its cage's box a refined point can lie, as a fraction of
/// the box's size on each axis. A stencil's weights sum to 1, so its point
/// is the box's high corner less a weighted sum of each cage point's
/// distance below it: the positive weights pull it inside, and only the
/// negative ones, which Catmull-Clark gives a vertex of valence 2, push it
/// out, by at most their sum times the box's size.
fn spread(refined: &Refined) -> f32 {
    (refined.offsets.windows(2))
        .map(|w| {
            let weights = &refined.weights[w[0] as usize..w[1] as usize];
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
        tables.offsets.push(at);
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

/// One draw record per visible mesh of render `tags` (or of any, when there
/// are none), with its box and its rows of the instances it is drawn at.
fn draws(store: &Store, ranges: &BTreeMap<u64, Range>, tags: &[String]) -> DrawTables {
    let instancers = by_path(store);
    let mut t = DrawTables {
        instances: Vec::new(),
        owners: Vec::new(),
        boxes: Vec::new(),
        records: Vec::new(),
        rows: Rows::new(),
    };
    for (&id, range) in ranges {
        let mesh = &store.meshes()[&id];
        let first = t.instances.len();
        t.instances
            .extend(mesh_instances(&instancers, mesh, range, tags));
        if t.instances.len() == first {
            continue;
        }
        let record = t.records.len() as u32;
        t.owners.resize(t.instances.len(), record);
        t.rows.insert(
            id,
            Row {
                record,
                instances: first..t.instances.len(),
            },
        );
        t.boxes.push(mesh_box(mesh, range.spread));
        t.records.push(vk::DrawIndexedIndirectCommand {
            index_count: range.index_count,
            instance_count: 0,
            first_index: range.first_index,
            vertex_offset: range.first_vertex,
            first_instance: first as u32,
        });
    }
    t
}

/// The instances of the meshes whose transform moved, each with its first
/// row. `None` when a mesh's instance count changed, so the rows no longer
/// fit and every record must be rewritten.
fn patch(
    store: &Store,
    ranges: &BTreeMap<u64, Range>,
    rows: &Rows,
    transforms: &BTreeSet<u64>,
    tags: &[String],
) -> Option<Vec<(usize, Vec<Instance>)>> {
    let instancers = by_path(store);
    let mut patches = Vec::new();
    for id in transforms {
        let Some(range) = ranges.get(id) else {
            continue;
        };
        let instances = mesh_instances(&instancers, &store.meshes()[id], range, tags);
        let rows = rows.get(id).map_or(0..0, |r| r.instances.clone());
        if instances.len() != rows.len() {
            return None;
        }
        if !instances.is_empty() {
            patches.push((rows.start, instances));
        }
    }
    Some(patches)
}

/// The instances `mesh` is drawn at in a pass of render `tags`. None when it
/// is invisible, unsynced, empty or of another tag, or when its instancers do
/// not fit together, with a warning.
fn mesh_instances(
    instancers: &HashMap<&str, &Instancer>,
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
    (matrices.iter())
        .filter_map(|m| Instance::new(m, topology.left_handed))
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
    use crate::store::Topology;

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
            transform: None,
            visible: None,
            instancing: None,
            material: None,
            render_tag: None,
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
            indices: vec![],
            offsets: vec![],
            corners: vec![],
            stencils: vec![0],
            stencil_indices: vec![0, 0],
            stencil_weights: vec![0.5, 0.5],
        };
        let first = Firsts {
            point: 10,
            cage: 30,
            normal: 20,
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
        assert_eq!((t.offsets, t.corners.len()), (vec![0; 5], 0));

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
        assert_eq!(t.offsets, [0, 1, 3, 5, 6]);
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
            offsets: vec![0, 1, 3, 4, 6],
            indices: vec![0, 1, 2, 3, 4, 0],
            weights: vec![1.0, 0.5, 0.5, 1.0, 0.25, 0.75],
        });
        m
    }

    #[test]
    fn refined_meshes_draw_their_refined_faces() {
        let t = tables(&refined()).unwrap();
        assert_eq!(t.indices, [0, 1, 2]);
        assert_eq!(t.vertices[1..], [[10, SMOOTH], [11, SMOOTH], [12, SMOOTH]]);
        // Smooth normals come from the refined faces, the hole's included.
        assert_eq!(t.offsets, [0, 1, 3, 5]);
        // Stencils count their entries after the earlier mesh's two, and
        // their cage points from the mesh's first cage point.
        assert_eq!(t.stencils, [0, 2, 3, 5, 6]);
        assert_eq!(t.stencil_indices[2..], [30, 31, 32, 33, 34, 30]);
        assert_eq!(t.stencil_weights.len(), 2 + 6);
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
        assert!(with(|r| r.indices[5] = 5).contains("index 5 is not one of 5"));
        assert!(with(|r| r.offsets[4] = 5).contains("do not divide"));
        assert!(with(|r| r.offsets.swap(1, 2)).contains("do not divide"));
        assert!(with(|r| r.face_vertex_indices[5] = 4).contains("index 4 is not one of 4"));
        assert!(with(|r| r.holes[0] = 2).contains("hole index 2"));
        assert!(with(|r| r.weights[0] = 0.5).contains("stencil 0's weights do not sum to 1"));
    }

    /// Each refined point, as the stencil pass writes it.
    fn refined_points(r: &Refined, cage: &[[f32; 3]]) -> Vec<[f32; 3]> {
        (r.offsets.windows(2))
            .map(|w| {
                let entries = w[0] as usize..w[1] as usize;
                std::array::from_fn(|k| {
                    (entries.clone())
                        .map(|e| r.weights[e] * cage[r.indices[e] as usize][k])
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
        r.weights[4..6].copy_from_slice(&[-0.5, 1.5]);
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
                first_vertex: 0,
                first_index: 0,
                index_count: 9,
                spread: 0.0,
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
                Row {
                    record: 0,
                    instances: 0..1
                },
                Row {
                    record: 1,
                    instances: 1..2
                }
            ]
        );
    }

    #[test]
    fn moved_transforms_patch_their_rows() {
        let (mut store, ranges) = drawn(&[("/a", "render"), ("/b", "render"), ("/c", "render")]);
        let DrawTables {
            mut instances,
            rows,
            ..
        } = draws(&store, &ranges, &[]);
        let ids: Vec<u64> = ranges.keys().copied().collect();
        let mut moved = IDENTITY;
        moved[12] = 2.0;
        store.set_transform(ids[1], moved).unwrap();
        let transforms = store.take_moved().transforms;
        let rewritten = draws(&store, &ranges, &[]).instances;
        assert_ne!(instances, rewritten);
        for (first, patch) in patch(&store, &ranges, &rows, &transforms, &[]).unwrap() {
            instances[first..first + patch.len()].copy_from_slice(&patch);
        }
        assert_eq!(instances, rewritten);

        // The control: a transform that flattens the mesh drops its instance,
        // so the rows after it shift.
        store.set_transform(ids[1], [0.0; 16]).unwrap();
        let transforms = store.take_moved().transforms;
        assert_eq!(patch(&store, &ranges, &rows, &transforms, &[]), None);
    }
}
