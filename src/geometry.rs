//! The GPU copy of the store's meshes: one device-local buffer per kind of
//! data, with a range in each for every mesh, and the instances and draw
//! records the mesh pass reads.
//!
//! The layout (the vertex table, the triangle indices and every mesh's
//! ranges) is rebuilt only when a mesh is added, removed or remodelled. A
//! playing frame writes only the points and normals that moved, and rewrites
//! the instances and draw records when a transform, visibility or instancer
//! changes. Each mesh is one draw record, of all its instances.

use std::collections::{BTreeMap, HashMap};

use ash::vk;

use crate::gpu::{self, Buffer};
use crate::store::{Instancer, Interpolation, Mesh, Moved, Store};
use crate::transform::{Instance, instance_transforms, multiply};

/// A vertex's normal index when the mesh has no normals. Normals are not
/// computed yet, so it shades black.
const NO_NORMAL: u32 = u32::MAX;

/// Where one mesh's data lives in the shared buffers.
struct Range {
    first_point: u64,
    first_normal: u64,
    normal_count: usize,
    first_vertex: i32,
    first_index: u32,
    index_count: u32,
}

/// What changes only when a mesh is remodelled, and the buffers sized by it.
struct Layout {
    ranges: BTreeMap<u64, Range>,
    /// xyz per point, every mesh's in turn.
    points: Buffer,
    /// xyz per normal, as authored: per point, face, corner or mesh.
    normals: Buffer,
    /// Point and normal index per unique pair of them, every mesh's in turn.
    vertices: Buffer,
    /// Three vertex indices per triangle, counted from the mesh's first vertex.
    indices: Buffer,
}

/// The instances and their draw records, sized to fit.
struct Draws {
    instances: Buffer,
    records: Buffer,
    count: u32,
}

/// The buffers the mesh pass reads.
pub struct Geometry {
    layout: Option<Layout>,
    draws: Option<Draws>,
    /// The host buffer each frame's uploads are copied from, grown as needed.
    staging: Option<Buffer>,
}

/// The addresses and buffers the mesh pass binds.
pub struct Bound {
    pub points: vk::DeviceAddress,
    pub normals: vk::DeviceAddress,
    pub vertices: vk::DeviceAddress,
    pub instances: vk::DeviceAddress,
    pub indices: vk::Buffer,
    pub records: vk::Buffer,
    pub count: u32,
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
            staging: None,
        }
    }

    /// Records into `cb` the uploads of what moved, and a barrier making them
    /// visible to the mesh pass. Returns the bytes uploaded. No GPU work may
    /// be pending: buffers are replaced and the staging buffer rewritten.
    pub fn update(
        &mut self,
        device: &gpu::Device,
        store: &Store,
        moved: &Moved,
        cb: vk::CommandBuffer,
    ) -> Result<u64, String> {
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
        }
        let layout = self.layout.as_ref().expect("built above");
        for (&id, range) in &layout.ranges {
            let mesh = &store.meshes()[&id];
            if rebuilt || moved.points.contains(&id) {
                let points = mesh.points.as_deref().unwrap_or_default();
                copies.push(Upload {
                    bytes: bytes(points),
                    buffer: layout.points.raw,
                    offset: range.first_point * 12,
                });
            }
            if (rebuilt || moved.normals.contains(&id)) && range.normal_count > 0 {
                let Some(Some(normals)) = &mesh.normals else {
                    continue;
                };
                copies.push(Upload {
                    bytes: bytes(&normals.values),
                    buffer: layout.normals.raw,
                    offset: range.first_normal * 12,
                });
            }
        }

        let records;
        if rebuilt || moved.draws || self.draws.is_none() {
            let (instances, commands) = draws(store, layout);
            records = (instances, commands);
            let draws = match self.draws.take() {
                Some(d)
                    if d.instances.size >= size(&records.0)
                        && d.records.size >= size(&records.1) =>
                {
                    d
                }
                _ => Draws {
                    instances: Buffer::device_local(
                        device,
                        size(&records.0),
                        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    )?,
                    records: Buffer::device_local(
                        device,
                        size(&records.1),
                        vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    )?,
                    count: 0,
                },
            };
            let draws = self.draws.insert(Draws {
                count: records.1.len() as u32,
                ..draws
            });
            copies.push(Upload {
                bytes: bytes(&records.0),
                buffer: draws.instances.raw,
                offset: 0,
            });
            copies.push(Upload {
                bytes: bytes(&records.1),
                buffer: draws.records.raw,
                offset: 0,
            });
        }
        self.record(device, &copies, cb)
    }

    /// The buffers to draw from, once `update` has run.
    pub fn bound(&self) -> Option<Bound> {
        let (layout, draws) = (self.layout.as_ref()?, self.draws.as_ref()?);
        Some(Bound {
            points: layout.points.address,
            normals: layout.normals.address,
            vertices: layout.vertices.address,
            instances: draws.instances.address,
            indices: layout.indices.raw,
            records: draws.records.raw,
            count: draws.count,
        })
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
                vk::PipelineStageFlags2::VERTEX_SHADER
                    | vk::PipelineStageFlags2::INDEX_INPUT
                    | vk::PipelineStageFlags2::DRAW_INDIRECT,
            )
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::INDEX_READ
                    | vk::AccessFlags2::INDIRECT_COMMAND_READ,
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

/// The vertex table and triangle indices of every mesh, before upload.
struct Tables {
    vertices: Vec<[u32; 2]>,
    indices: Vec<u32>,
}

/// Lays out every mesh with topology and points. A mesh whose data does not
/// fit together is left out, with a warning.
fn build(device: &gpu::Device, store: &Store) -> Result<(Layout, Tables), String> {
    let mut ranges = BTreeMap::new();
    let mut tables = Tables {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    let (mut points, mut normals) = (0u64, 0u64);
    for (&id, mesh) in store.meshes() {
        let (Some(_), Some(p)) = (&mesh.topology, &mesh.points) else {
            continue;
        };
        let first_vertex = tables.vertices.len();
        let first_index = tables.indices.len();
        let normal_count = match &mesh.normals {
            Some(Some(n)) => n.values.len(),
            _ => 0,
        };
        if let Err(e) = triangulate(mesh, points as u32, normals as u32, &mut tables) {
            eprintln!("pika: warning: {}: {e}; not drawn", mesh.path);
            tables.vertices.truncate(first_vertex);
            tables.indices.truncate(first_index);
            continue;
        }
        ranges.insert(
            id,
            Range {
                first_point: points,
                first_normal: normals,
                normal_count,
                first_vertex: first_vertex as i32,
                first_index: first_index as u32,
                index_count: (tables.indices.len() - first_index) as u32,
            },
        );
        points += p.len() as u64;
        normals += normal_count as u64;
    }
    let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let index = vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    let layout = Layout {
        points: Buffer::device_local(device, points * 12, storage)?,
        normals: Buffer::device_local(device, normals * 12, storage)?,
        vertices: Buffer::device_local(device, size(&tables.vertices), storage)?,
        indices: Buffer::device_local(device, size(&tables.indices), index)?,
        ranges,
    };
    Ok((layout, tables))
}

/// Appends a mesh's vertex table and its faces, split into triangles by a
/// fan from each face's first corner. A vertex is a unique pair of point and
/// normal; both indices count from the start of their buffers.
fn triangulate(
    mesh: &Mesh,
    first_point: u32,
    first_normal: u32,
    tables: &mut Tables,
) -> Result<(), String> {
    let (Some(topology), Some(points)) = (&mesh.topology, &mesh.points) else {
        return Ok(());
    };
    let counts = &topology.face_vertex_counts;
    let corners = &topology.face_vertex_indices;
    let total: i64 = counts.iter().map(|&n| n as i64).sum();
    if counts.iter().any(|&n| n < 0) || total != corners.len() as i64 {
        return Err(format!(
            "face vertex counts sum to {total}, but there are {} face vertex indices",
            corners.len()
        ));
    }
    if let Some(&bad) = corners
        .iter()
        .find(|&&p| p < 0 || p as usize >= points.len())
    {
        return Err(format!(
            "face vertex index {bad} is not one of {} points",
            points.len()
        ));
    }
    let normals = match &mesh.normals {
        Some(Some(n)) => Some(n),
        _ => None,
    };
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

    let first_vertex = tables.vertices.len();
    let mut welded: HashMap<[u32; 2], u32> = HashMap::new();
    let mut vertex = |point: i32, normal: Option<usize>| {
        let pair = [
            first_point + point as u32,
            normal.map_or(NO_NORMAL, |n| first_normal + n as u32),
        ];
        *welded.entry(pair).or_insert_with(|| {
            tables.vertices.push(pair);
            (tables.vertices.len() - 1 - first_vertex) as u32
        })
    };
    let mut face_vertices = Vec::new();
    let mut corner = 0;
    for (face, &count) in counts.iter().enumerate() {
        face_vertices.clear();
        let face_corners = corners.iter().enumerate().skip(corner).take(count as usize);
        for (c, &point) in face_corners {
            let normal = normals.map(|n| match n.interpolation {
                Interpolation::Vertex | Interpolation::Varying => point as usize,
                Interpolation::FaceVarying => c,
                Interpolation::Uniform => face,
                Interpolation::Constant => 0,
            });
            face_vertices.push(vertex(point, normal));
        }
        corner += count as usize;
        for i in 1..face_vertices.len().saturating_sub(1) {
            tables
                .indices
                .extend([face_vertices[0], face_vertices[i], face_vertices[i + 1]]);
        }
    }
    Ok(())
}

/// One draw record per visible mesh, drawing it at each of its instances.
/// A mesh whose instancers do not fit together is left out, with a warning.
fn draws(store: &Store, layout: &Layout) -> (Vec<Instance>, Vec<vk::DrawIndexedIndirectCommand>) {
    let instancers: HashMap<&str, &Instancer> = (store.instancers().values())
        .map(|i| (i.path.as_str(), i))
        .collect();
    let mut instances = Vec::new();
    let mut records = Vec::new();
    for (&id, range) in &layout.ranges {
        let mesh = &store.meshes()[&id];
        let (Some(true), Some(instancing), Some(transform), Some(topology)) = (
            mesh.visible,
            &mesh.instancing,
            &mesh.transform,
            &mesh.topology,
        ) else {
            continue;
        };
        if range.index_count == 0 {
            continue;
        }
        let matrices = match instancing {
            None => vec![*transform],
            Some(i) => match instance_transforms(&instancers, i) {
                Ok(m) => m.iter().map(|m| multiply(transform, m)).collect(),
                Err(e) => {
                    eprintln!("pika: warning: {}: {e}; not drawn", mesh.path);
                    continue;
                }
            },
        };
        let first = instances.len();
        let left_handed = topology.left_handed;
        instances.extend(
            matrices
                .iter()
                .filter_map(|m| Instance::new(m, left_handed)),
        );
        if instances.len() == first {
            continue;
        }
        records.push(vk::DrawIndexedIndirectCommand {
            index_count: range.index_count,
            instance_count: (instances.len() - first) as u32,
            first_index: range.first_index,
            vertex_offset: range.first_vertex,
            first_instance: first as u32,
        });
    }
    (instances, records)
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
    use crate::store::{Normals, Topology};

    /// A quad and a triangle sharing an edge, with optional normals.
    fn mesh(normals: Option<Normals>) -> Mesh {
        Mesh {
            path: "/m".into(),
            topology: Some(Topology {
                face_vertex_counts: vec![4, 3],
                face_vertex_indices: vec![0, 1, 2, 3, 1, 4, 2],
                left_handed: false,
                subsets: vec![],
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
        };
        triangulate(mesh, 10, 20, &mut t)?;
        Ok(t)
    }

    #[test]
    fn faces_fan_and_vertices_weld() {
        let t = tables(&mesh(normals(Interpolation::Vertex, 5))).unwrap();
        // Vertex normals weld to one vertex per point; indices count from
        // the mesh's first vertex, after the earlier mesh's one.
        assert_eq!(
            t.vertices[1..],
            [[10, 20], [11, 21], [12, 22], [13, 23], [14, 24]]
        );
        assert_eq!(t.indices, [0, 1, 2, 0, 2, 3, 1, 4, 2]);

        // Face-varying normals keep every corner apart.
        let t = tables(&mesh(normals(Interpolation::FaceVarying, 7))).unwrap();
        assert_eq!(t.vertices.len() - 1, 7);
        assert_eq!(t.vertices[5..], [[11, 24], [14, 25], [12, 26]]);
        // Uniform normals split the points the two faces share.
        let t = tables(&mesh(normals(Interpolation::Uniform, 2))).unwrap();
        assert_eq!(t.vertices.len() - 1, 4 + 3);
        // Without normals, vertices are points.
        let t = tables(&mesh(None)).unwrap();
        assert_eq!(t.vertices[1], [10, NO_NORMAL]);
        assert_eq!(t.vertices.len() - 1, 5);
    }

    #[test]
    fn mismatched_data_is_refused() {
        let error = tables(&mesh(normals(Interpolation::Vertex, 4)))
            .err()
            .unwrap();
        assert!(error.contains("needs 5"), "{error}");
        let mut m = mesh(None);
        m.topology.as_mut().unwrap().face_vertex_indices[6] = 5;
        assert!(tables(&m).err().unwrap().contains("index 5"));
        let mut m = mesh(None);
        m.topology.as_mut().unwrap().face_vertex_counts[1] = 4;
        assert!(tables(&m).is_err());
    }
}
