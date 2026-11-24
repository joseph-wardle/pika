//! The draws of the geometry's meshes: one instance per placement of each mesh, and one record
//! per material part of it. The draws are rewritten when a mesh is added, hidden, retagged or
//! re-instanced, and patched in place when meshes only move.

use std::collections::{BTreeMap, HashMap};

use ash::vk;

use crate::geometry::{Part, Range, Upload, mesh_box, size};
use crate::gpu::{self, Buffer, bytes};
use crate::link::{Links, Mask};
use crate::store::{Instancer, Mesh, Moved, Store};
use crate::transform::{Instance, instances, multiply};

/// One record's instances of a mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub record: u32,
    pub instances: std::ops::Range<usize>,
}

pub type Rows = BTreeMap<u64, Vec<Row>>;

/// The draws on the host: what the buffers in `Draws` hold, and where each mesh's are.
#[derive(Default)]
pub struct Tables {
    pub instances: Vec<Instance>,
    pub owners: Vec<u32>,
    /// Each record's mesh box: the low corner, then the high one, in the mesh's own space.
    pub boxes: Vec<[f32; 6]>,
    /// No record has instances until the cull pass counts a view's.
    pub records: Vec<vk::DrawIndexedIndirectCommand>,
    pub groups: Vec<Group>,
    pub rows: Rows,
}

/// Consecutive records of one material, drawn with one multi-draw.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Group {
    /// `None` for meshes with no material bound.
    pub material: Option<u64>,
    pub first: u32,
    pub count: u32,
}

pub struct Draws {
    pub instances: Buffer,
    /// `instances` as the frame before drew them. Motion is measured from these.
    pub previous_instances: Buffer,
    /// Whether the last patch moved an instance, so `previous_instances` differs from `instances`.
    last_moved: bool,
    pub owners: Buffer,
    pub boxes: Buffer,
    pub records: Buffer,
    /// Kept on the host for the texture needs. A patch writes both copies.
    pub tables: Tables,
    tags: Vec<String>,
}

impl Draws {
    /// Writes the draws from scratch, into `slot`'s buffers when they are large enough. An empty
    /// `tags` draws every tag.
    pub fn rewrite(
        slot: &mut Option<Draws>,
        device: &gpu::Device,
        store: &Store,
        ranges: &BTreeMap<u64, Range>,
        tags: &[String],
    ) -> Result<(), String> {
        let t = tables(store, ranges, tags);
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
                let storage =
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
                let copied =
                    vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
                let instances = size(&t.instances);
                Draws {
                    instances: Buffer::device_local(device, instances, storage | copied)?,
                    previous_instances: Buffer::device_local(device, instances, storage)?,
                    last_moved: false,
                    owners: Buffer::device_local(device, size(&t.owners), storage)?,
                    boxes: Buffer::device_local(device, size(&t.boxes), storage)?,
                    records: Buffer::device_local(device, size(&t.records), storage | copied)?,
                    tables: Tables::default(),
                    tags: Vec::new(),
                }
            }
        };
        *slot = Some(Draws {
            tables: t,
            tags: tags.to_vec(),
            last_moved: false,
            ..draws
        });
        Ok(())
    }

    /// The patch that follows moved transforms and points in place, recording the copy of the
    /// frame before's instances that it needs. `None` when the draws must be rewritten instead,
    /// because a visibility, tag, instancing, link or instance count changed.
    pub fn patch(
        &mut self,
        device: &gpu::Device,
        cb: vk::CommandBuffer,
        store: &Store,
        ranges: &BTreeMap<u64, Range>,
        moved: &Moved,
        tags: &[String],
    ) -> Option<Patch> {
        if moved.draws || self.tags != tags {
            return None;
        }
        let patch = Patch::new(store, ranges, &self.tables.rows, moved, tags)?;
        // An instance that just stopped has still moved since the frame before, and
        // `previous_instances` must hold that until the frame after.
        let moves = !patch.instances.is_empty();
        if self.last_moved || moves {
            self.remember(device, cb);
        }
        self.last_moved = moves;
        Some(patch)
    }

    /// Every table, for rewritten draws. Rewriting renumbers the instances, so each is drawn as if
    /// it hadn't moved, as EEVEE draws an object it can't match.
    pub fn uploads(&self) -> [Upload<'_>; 5] {
        let t = &self.tables;
        [
            Upload::table(bytes(&t.instances), &self.instances),
            Upload::table(bytes(&t.instances), &self.previous_instances),
            Upload::table(bytes(&t.owners), &self.owners),
            Upload::table(bytes(&t.boxes), &self.boxes),
            Upload::table(bytes(&t.records), &self.records),
        ]
    }

    /// Copies the instances the frame before drew into `previous_instances`, before this frame's
    /// patch overwrites them.
    fn remember(&self, device: &gpu::Device, cb: vk::CommandBuffer) {
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: size(&self.tables.instances),
        };
        let copied = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::VERTEX_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
            );
        let (from, to) = (self.instances.raw, self.previous_instances.raw);
        // SAFETY: the command buffer is recording outside a render pass, and both buffers belong to
        // this device and hold the tables, which a patch never grows. Vulkan doesn't allow empty
        // copies, so none are made.
        unsafe {
            if region.size > 0 {
                (device.raw).cmd_copy_buffer(cb, from, to, &[region]);
            }
            (device.raw).cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().memory_barriers(&[copied]),
            );
        }
    }

    /// The indices of the instances whose mesh's transform or points moved, in ascending order.
    pub fn moved(&self, moved: &Moved) -> Vec<u32> {
        let ids = moved.transforms.iter().chain(&moved.points);
        let rows = ids.filter_map(|id| self.tables.rows.get(id)).flatten();
        let mut instances: Vec<u32> = rows
            .flat_map(|r| r.instances.clone())
            .map(|i| i as u32)
            .collect();
        instances.sort_unstable();
        instances.dedup();
        instances
    }
}

/// An empty `tags` draws every tag. Records are grouped by material, with unbound parts first.
fn tables(store: &Store, ranges: &BTreeMap<u64, Range>, tags: &[String]) -> Tables {
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
    let mut t = Tables {
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

/// A subset with no material binding uses the mesh's. `None` when Hydra hasn't made the
/// material yet.
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

pub struct Patch {
    instances: Vec<(usize, Vec<Instance>)>,
    boxes: Vec<(u32, [f32; 6])>,
}

impl Patch {
    /// Writes the host's copy, once the uploads that read the patch are staged.
    pub fn apply(&self, t: &mut Tables) {
        for (first, instances) in &self.instances {
            t.instances[*first..first + instances.len()].copy_from_slice(instances);
        }
        for (record, b) in &self.boxes {
            t.boxes[*record as usize] = *b;
        }
    }

    /// `None` when an instance count changed, so every record must be rewritten.
    fn new(
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
            let placed = mesh_instances(&instancers, &links, &store.meshes()[id], range, tags);
            let rows = rows.get(id).map_or(&[][..], Vec::as_slice);
            if placed.len() != rows.first().map_or(0, |r| r.instances.len()) {
                return None;
            }
            for row in rows {
                instances.push((row.instances.start, placed.clone()));
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

    pub fn uploads<'a>(&'a self, d: &Draws) -> Vec<Upload<'a>> {
        let instances = self.instances.iter().map(|(first, instances)| Upload {
            bytes: bytes(instances),
            buffer: d.instances.raw,
            offset: (first * size_of::<Instance>()) as u64,
        });
        let boxes = self.boxes.iter().map(|(record, b)| Upload {
            bytes: bytes(b),
            buffer: d.boxes.raw,
            offset: *record as u64 * 24,
        });
        instances.chain(boxes).collect()
    }
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
    let placed = match instancing {
        None => vec![(*transform, Mask::default())],
        Some(i) => match instances(instancers, links, i) {
            Ok(p) => (p.iter())
                .map(|(m, l)| (multiply(transform, m), *l))
                .collect(),
            Err(e) => {
                eprintln!("pika: warning: {}: {e}; not drawn", mesh.path);
                return Vec::new();
            }
        },
    };
    let link = links.mesh(mesh);
    (placed.iter())
        .filter_map(|(m, l)| Instance::new(m, topology.left_handed, link | *l))
        .collect()
}

fn by_path(store: &Store) -> HashMap<&str, &Instancer> {
    (store.instancers().values())
        .map(|i| (i.path.as_str(), i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Scheme, Subset, Topology};

    /// A quad and a triangle over five points.
    fn faces() -> Topology {
        Topology {
            scheme: Scheme::None,
            face_vertex_counts: vec![4, 3],
            face_vertex_indices: vec![0, 1, 2, 3, 1, 4, 2],
            left_handed: false,
            holes: vec![],
            subsets: vec![],
            refined: None,
        }
    }

    const IDENTITY: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    fn drawn(meshes: &[(&str, &str)]) -> (Store, BTreeMap<u64, Range>) {
        let mut store = Store::new();
        let mut ranges = BTreeMap::new();
        for &(path, tag) in meshes {
            let id = store.insert_mesh(path.into());
            store.set_topology(id, faces()).unwrap();
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
            tables(&store, &ranges, &tags).records.len()
        };
        // A retagged mesh keeps its data in the store, so drawing must filter.
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
        let t = tables(&store, &ranges, &[]);
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
        let t = tables(&store, &ranges, &[]);
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
        let mut topology = faces();
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
        let t = tables(&store, &ranges, &[]);
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
        // Control: a subset with no material binding draws with the mesh's.
        store.set_material(ids[0], "/mtl/red".into()).unwrap();
        let mut topology = faces();
        topology.subsets = vec![Subset {
            material: String::new(),
            faces: vec![1],
        }];
        store.set_topology(ids[0], topology).unwrap();
        let t = tables(&store, &ranges, &[]);
        assert_eq!(t.groups, [group(None, 0, 1), group(Some(red), 1, 2)]);
    }

    #[test]
    fn moved_transforms_patch_their_rows() {
        let (mut store, ranges) = drawn(&[("/a", "render"), ("/b", "render"), ("/c", "render")]);
        let mut host = tables(&store, &ranges, &[]);
        let rows = host.rows.clone();
        let ids: Vec<u64> = ranges.keys().copied().collect();
        let mut moved = IDENTITY;
        moved[12] = 2.0;
        store.set_transform(ids[1], moved).unwrap();
        let moved = store.take_moved();
        let rewritten = tables(&store, &ranges, &[]).instances;
        assert_ne!(host.instances, rewritten);
        let patched = Patch::new(&store, &ranges, &rows, &moved, &[]).unwrap();
        patched.apply(&mut host);
        assert_eq!(host.instances, rewritten);
        assert!(patched.boxes.is_empty());

        // Control: a flattening transform drops its instance, which shifts later rows.
        store.set_transform(ids[1], [0.0; 16]).unwrap();
        let moved = store.take_moved();
        assert!(Patch::new(&store, &ranges, &rows, &moved, &[]).is_none());
    }
}
