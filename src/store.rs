//! The scene as Hydra last synced it. One store per render delegate: the
//! delegate's prim syncs write it, field by field, and the renderer reads it.
//!
//! Each field is stored in Hydra's representation (world transforms, flattened
//! primvars, untriangulated faces), so a dump of the store can be checked
//! against what Hydra is known to hand over.

use std::collections::BTreeMap;
use std::fmt;

pub struct Store {
    /// Ids are never reused, so a stale id is an error rather than a write to
    /// a newer prim.
    next_id: u64,
    meshes: BTreeMap<u64, Mesh>,
    instancers: BTreeMap<u64, Instancer>,
    curves: BTreeMap<u64, Curves>,
    counted: BTreeMap<u64, Counted>,
    stats: Stats,
}

/// Each field is `None` until Hydra first hands it over.
pub struct Mesh {
    pub path: String,
    pub topology: Option<Topology>,
    pub points: Option<Vec<[f32; 3]>>,
    /// `Some(None)` when the mesh has no normals primvar.
    pub normals: Option<Option<Normals>>,
    /// Hydra's `GfMatrix4d`: row-major, translation in the last row.
    pub transform: Option<[f64; 16]>,
    pub visible: Option<bool>,
    /// `Some(None)` when the mesh is not instanced.
    pub instancing: Option<Option<Instancing>>,
    /// Hydra syncs only the prims whose render tag a render pass draws, and
    /// leaves the rest in the store with stale data: a proxy mesh is inserted
    /// but never synced, and a mesh retagged as proxy keeps its old fields.
    /// Drawing must filter by the pass's tags.
    pub render_tag: Option<String>,
}

/// A prototype mesh's place in an instancer: it is drawn once per index, at
/// that index of the instancer's primvars.
#[derive(Clone, PartialEq, Debug)]
pub struct Instancing {
    pub instancer: String,
    pub indices: Vec<i32>,
}

/// An instancer, as Hydra hands it over. Each instance's transform is the
/// prototype's own, then the instance's primvars, then the instancer's
/// transform, then the parent instancer's, if any.
pub struct Instancer {
    pub path: String,
    /// `Some(None)` when the instancer is not nested in another.
    pub parent: Option<Option<String>>,
    pub transform: Option<[f64; 16]>,
    pub primvars: Option<InstancePrimvars>,
}

/// Hydra's per-instance transform primvars, each `None` when absent: native
/// instances arrive as matrices, point instancers as their separate parts.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct InstancePrimvars {
    pub transforms: Option<Vec<[f64; 16]>>,
    pub translations: Option<Vec<[f32; 3]>>,
    /// Quaternions in `GfQuatf`'s order: i, j, k, then the real part.
    pub rotations: Option<Vec<[f32; 4]>>,
    pub scales: Option<Vec<[f32; 3]>>,
}

/// Basis curves are fetched to measure what they cost, then dropped: Pika
/// does not draw them yet.
pub struct Curves {
    pub path: String,
    /// Every byte pulled since the curves were inserted.
    pub bytes: usize,
}

/// A prim Pika accepts only to count it: Hydra inserts no prim of a type the
/// delegate does not support, so the census could not see it.
pub struct Counted {
    pub prim_type: String,
    pub path: String,
}

/// What changes only when a mesh is remodelled.
#[derive(Clone, PartialEq, Debug)]
pub struct Topology {
    pub face_vertex_counts: Vec<i32>,
    pub face_vertex_indices: Vec<i32>,
    pub left_handed: bool,
    pub subsets: Vec<Subset>,
}

/// A GeomSubset: the faces bound to one material.
#[derive(Clone, PartialEq, Debug)]
pub struct Subset {
    /// Empty when the subset has no material bound.
    pub material: String,
    pub faces: Vec<i32>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Normals {
    pub interpolation: Interpolation,
    pub values: Vec<[f32; 3]>,
}

/// Hydra's primvar interpolations, as USD names them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Interpolation {
    Constant,
    Uniform,
    Varying,
    Vertex,
    FaceVarying,
}

impl Interpolation {
    /// The value `pika.h` passes for this interpolation.
    pub fn from_abi(value: u32) -> Result<Self, String> {
        match value {
            1 => Ok(Self::Constant),
            2 => Ok(Self::Uniform),
            3 => Ok(Self::Varying),
            4 => Ok(Self::Vertex),
            5 => Ok(Self::FaceVarying),
            _ => Err(format!("unknown pika_interpolation {value}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Uniform => "uniform",
            Self::Varying => "varying",
            Self::Vertex => "vertex",
            Self::FaceVarying => "faceVarying",
        }
    }
}

/// What one Hydra sync did to the store.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Stats {
    pub inserted: usize,
    pub removed: usize,
    /// Fields written, changed or not.
    pub pulls: usize,
    pub bytes: usize,
    /// Bytes of basis curves, which are not stored, so are not counted in
    /// `bytes` and are never known to be redundant.
    pub curve_bytes: usize,
    /// Bytes of pulls that matched what the store already held: data Hydra
    /// marked dirty that did not change, such as constants authored as time
    /// samples.
    pub redundant_bytes: usize,
}

impl Store {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            meshes: BTreeMap::new(),
            instancers: BTreeMap::new(),
            curves: BTreeMap::new(),
            counted: BTreeMap::new(),
            stats: Stats::default(),
        }
    }

    fn new_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.stats.inserted += 1;
        id
    }

    /// Adds a mesh with nothing synced yet.
    pub fn insert_mesh(&mut self, path: String) -> u64 {
        let id = self.new_id();
        self.meshes.insert(
            id,
            Mesh {
                path,
                topology: None,
                points: None,
                normals: None,
                transform: None,
                visible: None,
                instancing: None,
                render_tag: None,
            },
        );
        id
    }

    pub fn remove_mesh(&mut self, id: u64) -> Result<(), String> {
        self.meshes.remove(&id).ok_or_else(|| stale("mesh", id))?;
        self.stats.removed += 1;
        Ok(())
    }

    pub fn set_topology(&mut self, id: u64, topology: Topology) -> Result<(), String> {
        let bytes = 4 * (topology.face_vertex_counts.len() + topology.face_vertex_indices.len())
            + 1
            + topology
                .subsets
                .iter()
                .map(|s| s.material.len() + 4 * s.faces.len())
                .sum::<usize>();
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        write(&mut self.stats, &mut mesh.topology, topology, bytes);
        Ok(())
    }

    pub fn set_points(&mut self, id: u64, points: Vec<[f32; 3]>) -> Result<(), String> {
        let bytes = size_of_val(points.as_slice());
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        write(&mut self.stats, &mut mesh.points, points, bytes);
        Ok(())
    }

    /// `None` when the mesh has no normals primvar.
    pub fn set_normals(&mut self, id: u64, normals: Option<Normals>) -> Result<(), String> {
        let bytes = normals
            .as_ref()
            .map_or(0, |n| size_of_val(n.values.as_slice()));
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        write(&mut self.stats, &mut mesh.normals, normals, bytes);
        Ok(())
    }

    pub fn set_transform(&mut self, id: u64, transform: [f64; 16]) -> Result<(), String> {
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        write(&mut self.stats, &mut mesh.transform, transform, 128);
        Ok(())
    }

    pub fn set_visible(&mut self, id: u64, visible: bool) -> Result<(), String> {
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        write(&mut self.stats, &mut mesh.visible, visible, 1);
        Ok(())
    }

    /// `None` when the mesh is not instanced.
    pub fn set_instancing(
        &mut self,
        id: u64,
        instancing: Option<Instancing>,
    ) -> Result<(), String> {
        let bytes = instancing
            .as_ref()
            .map_or(0, |i| i.instancer.len() + 4 * i.indices.len());
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        write(&mut self.stats, &mut mesh.instancing, instancing, bytes);
        Ok(())
    }

    pub fn set_render_tag(&mut self, id: u64, tag: String) -> Result<(), String> {
        let bytes = tag.len();
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        write(&mut self.stats, &mut mesh.render_tag, tag, bytes);
        Ok(())
    }

    /// Adds an instancer with nothing synced yet.
    pub fn insert_instancer(&mut self, path: String) -> u64 {
        let id = self.new_id();
        let instancer = Instancer {
            path,
            parent: None,
            transform: None,
            primvars: None,
        };
        self.instancers.insert(id, instancer);
        id
    }

    pub fn remove_instancer(&mut self, id: u64) -> Result<(), String> {
        self.instancers
            .remove(&id)
            .ok_or_else(|| stale("instancer", id))?;
        self.stats.removed += 1;
        Ok(())
    }

    /// `None` when the instancer is not nested in another.
    pub fn set_instancer_parent(&mut self, id: u64, parent: Option<String>) -> Result<(), String> {
        let bytes = parent.as_ref().map_or(0, String::len);
        let instancer = self
            .instancers
            .get_mut(&id)
            .ok_or_else(|| stale("instancer", id))?;
        write(&mut self.stats, &mut instancer.parent, parent, bytes);
        Ok(())
    }

    pub fn set_instancer_transform(&mut self, id: u64, transform: [f64; 16]) -> Result<(), String> {
        let instancer = self
            .instancers
            .get_mut(&id)
            .ok_or_else(|| stale("instancer", id))?;
        write(&mut self.stats, &mut instancer.transform, transform, 128);
        Ok(())
    }

    pub fn set_instance_primvars(
        &mut self,
        id: u64,
        primvars: InstancePrimvars,
    ) -> Result<(), String> {
        fn size<T>(values: &Option<Vec<T>>) -> usize {
            values.as_deref().map_or(0, size_of_val)
        }
        let bytes = size(&primvars.transforms)
            + size(&primvars.translations)
            + size(&primvars.rotations)
            + size(&primvars.scales);
        let instancer = self
            .instancers
            .get_mut(&id)
            .ok_or_else(|| stale("instancer", id))?;
        write(&mut self.stats, &mut instancer.primvars, primvars, bytes);
        Ok(())
    }

    pub fn insert_curves(&mut self, path: String) -> u64 {
        let id = self.new_id();
        self.curves.insert(id, Curves { path, bytes: 0 });
        id
    }

    pub fn remove_curves(&mut self, id: u64) -> Result<(), String> {
        self.curves.remove(&id).ok_or_else(|| stale("curves", id))?;
        self.stats.removed += 1;
        Ok(())
    }

    /// Records `bytes` the delegate fetched for basis curves and dropped.
    pub fn pulled_curves(&mut self, id: u64, bytes: usize) -> Result<(), String> {
        let curves = self
            .curves
            .get_mut(&id)
            .ok_or_else(|| stale("curves", id))?;
        curves.bytes += bytes;
        self.stats.pulls += 1;
        self.stats.curve_bytes += bytes;
        Ok(())
    }

    /// Adds a prim that is only counted, of Hydra's `prim_type`.
    pub fn insert_counted(&mut self, prim_type: String, path: String) -> u64 {
        let id = self.new_id();
        self.counted.insert(id, Counted { prim_type, path });
        id
    }

    pub fn remove_counted(&mut self, id: u64) -> Result<(), String> {
        self.counted
            .remove(&id)
            .ok_or_else(|| stale("counted prim", id))?;
        self.stats.removed += 1;
        Ok(())
    }

    /// Ends one Hydra sync: returns what it did and starts counting the next.
    pub fn end_sync(&mut self) -> Stats {
        std::mem::take(&mut self.stats)
    }

    /// The whole store as canonical text: meshes sorted by path, floats in
    /// their shortest exact form. Ids are left out because they depend on
    /// the order Hydra inserted prims in.
    pub fn dump(&self) -> String {
        let mut prims: Vec<(&str, String)> = (self.meshes.values())
            .map(|m| (m.path.as_str(), m.to_string()))
            .chain(
                self.instancers
                    .values()
                    .map(|i| (i.path.as_str(), i.to_string())),
            )
            .chain(
                self.curves
                    .values()
                    .map(|c| (c.path.as_str(), c.to_string())),
            )
            .chain(
                self.counted
                    .values()
                    .map(|c| (c.path.as_str(), c.to_string())),
            )
            .collect();
        prims.sort();
        prims.into_iter().map(|(_, text)| text).collect()
    }
}

const UNSYNCED: &str = "not synced";

impl fmt::Display for Mesh {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "mesh {}", self.path)?;
        match &self.render_tag {
            Some(tag) => writeln!(f, "  render_tag {tag}")?,
            None => writeln!(f, "  render_tag {UNSYNCED}")?,
        }
        match self.visible {
            Some(visible) => writeln!(f, "  visible {visible}")?,
            None => writeln!(f, "  visible {UNSYNCED}")?,
        }
        match &self.instancing {
            Some(Some(i)) => {
                writeln!(f, "  instancer {} indices{}", i.instancer, List(&i.indices))?
            }
            Some(None) => writeln!(f, "  instancer none")?,
            None => writeln!(f, "  instancer {UNSYNCED}")?,
        }
        transform(f, self.transform.as_ref())?;
        match &self.topology {
            Some(t) => {
                let orientation = if t.left_handed {
                    "leftHanded"
                } else {
                    "rightHanded"
                };
                writeln!(f, "  topology {orientation}")?;
                writeln!(f, "    face_vertex_counts{}", List(&t.face_vertex_counts))?;
                writeln!(f, "    face_vertex_indices{}", List(&t.face_vertex_indices))?;
                for s in &t.subsets {
                    writeln!(f, "    subset {:?} faces{}", s.material, List(&s.faces))?;
                }
            }
            None => writeln!(f, "  topology {UNSYNCED}")?,
        }
        match &self.points {
            Some(points) => {
                writeln!(f, "  points {}", points.len())?;
                for p in points {
                    writeln!(f, "   {}", List(p))?;
                }
            }
            None => writeln!(f, "  points {UNSYNCED}")?,
        }
        match &self.normals {
            Some(Some(n)) => {
                writeln!(f, "  normals {} {}", n.interpolation.name(), n.values.len())?;
                for v in &n.values {
                    writeln!(f, "   {}", List(v))?;
                }
            }
            Some(None) => writeln!(f, "  normals none")?,
            None => writeln!(f, "  normals {UNSYNCED}")?,
        }
        Ok(())
    }
}

impl fmt::Display for Instancer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "instancer {}", self.path)?;
        match &self.parent {
            Some(Some(parent)) => writeln!(f, "  parent {parent}")?,
            Some(None) => writeln!(f, "  parent none")?,
            None => writeln!(f, "  parent {UNSYNCED}")?,
        }
        transform(f, self.transform.as_ref())?;
        let Some(p) = &self.primvars else {
            return writeln!(f, "  primvars {UNSYNCED}");
        };
        values(f, "instance_transforms", p.transforms.as_deref())?;
        values(f, "instance_translations", p.translations.as_deref())?;
        values(f, "instance_rotations", p.rotations.as_deref())?;
        values(f, "instance_scales", p.scales.as_deref())
    }
}

impl fmt::Display for Curves {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "curves {}", self.path)?;
        writeln!(f, "  bytes pulled {}", self.bytes)
    }
}

impl fmt::Display for Counted {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "counted {} {}", self.prim_type, self.path)
    }
}

/// A `GfMatrix4d`, one row per line.
fn transform(f: &mut fmt::Formatter, transform: Option<&[f64; 16]>) -> fmt::Result {
    let Some(transform) = transform else {
        return writeln!(f, "  transform {UNSYNCED}");
    };
    writeln!(f, "  transform")?;
    for row in transform.chunks(4) {
        writeln!(f, "   {}", List(row))?;
    }
    Ok(())
}

/// A per-instance primvar, one value per line; nothing when it is absent.
fn values<T: fmt::Display, const N: usize>(
    f: &mut fmt::Formatter,
    name: &str,
    values: Option<&[[T; N]]>,
) -> fmt::Result {
    let Some(values) = values else {
        return Ok(());
    };
    writeln!(f, "  {name} {}", values.len())?;
    for v in values {
        writeln!(f, "   {}", List(v))?;
    }
    Ok(())
}

/// Values each preceded by a space, so an empty list leaves no trailing one.
struct List<'a, T>(&'a [T]);

impl<T: fmt::Display> fmt::Display for List<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.iter().try_for_each(|v| write!(f, " {v}"))
    }
}

fn stale(kind: &str, id: u64) -> String {
    format!("no {kind} with id {id} in the store: it was removed, or never inserted")
}

/// Stores one pulled field, counting its bytes as redundant when Hydra handed
/// over the same value before.
fn write<T: PartialEq>(stats: &mut Stats, slot: &mut Option<T>, value: T, bytes: usize) {
    stats.pulls += 1;
    stats.bytes += bytes;
    if slot.as_ref() == Some(&value) {
        stats.redundant_bytes += bytes;
    } else {
        *slot = Some(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_never_reused() {
        let mut store = Store::new();
        let a = store.insert_mesh("/a".into());
        store.remove_mesh(a).unwrap();
        let b = store.insert_mesh("/a".into());
        assert_ne!(a, b);
    }

    #[test]
    fn stale_id_is_an_error() {
        let mut store = Store::new();
        let a = store.insert_mesh("/a".into());
        // The control: the same write succeeds while the mesh exists.
        store.set_visible(a, false).unwrap();
        store.remove_mesh(a).unwrap();
        let error = store.set_visible(a, false).unwrap_err();
        assert!(error.contains(&format!("id {a}")), "{error}");
        assert!(store.remove_mesh(a).is_err());

        // Kinds share one id space, so an instancer's id is no mesh's.
        let i = store.insert_instancer("/i".into());
        assert!(store.set_instancer_parent(i, None).is_ok());
        assert!(store.set_visible(i, false).is_err());
    }

    #[test]
    fn curve_bytes_are_counted_apart() {
        let mut store = Store::new();
        let c = store.insert_curves("/c".into());
        store.end_sync();
        store.pulled_curves(c, 68).unwrap();
        store.pulled_curves(c, 68).unwrap();
        let stats = store.end_sync();
        // Curves are not stored, so their bytes can never be found redundant.
        assert_eq!(
            (stats.curve_bytes, stats.bytes, stats.redundant_bytes),
            (136, 0, 0)
        );
    }

    #[test]
    fn unchanged_pulls_count_as_redundant() {
        let mut store = Store::new();
        let a = store.insert_mesh("/a".into());
        store.end_sync();
        // A first pull is never redundant, even of a value like Hydra's
        // default visibility.
        let points = vec![[0.0, 1.0, 2.0]; 4];
        store.set_points(a, points.clone()).unwrap();
        store.set_visible(a, true).unwrap();
        let first = store.end_sync();
        assert_eq!(
            (first.pulls, first.bytes, first.redundant_bytes),
            (2, 49, 0)
        );

        store.set_points(a, points).unwrap();
        store.set_visible(a, true).unwrap();
        let same = store.end_sync();
        assert_eq!((same.pulls, same.bytes, same.redundant_bytes), (2, 49, 49));

        // A changed value is not redundant, and is stored.
        store.set_points(a, vec![[0.0, 1.0, 3.0]; 4]).unwrap();
        assert_eq!(store.end_sync().redundant_bytes, 0);
        assert_eq!(
            store.meshes[&a].points.as_ref().unwrap()[0],
            [0.0, 1.0, 3.0]
        );
    }

    #[test]
    fn end_sync_counts_insertions_and_removals() {
        let mut store = Store::new();
        let a = store.insert_mesh("/a".into());
        store.insert_mesh("/b".into());
        store.remove_mesh(a).unwrap();
        let stats = store.end_sync();
        assert_eq!((stats.inserted, stats.removed), (2, 1));
        assert_eq!(store.end_sync(), Stats::default());
    }

    #[test]
    fn dump_is_sorted_by_path_and_exact() {
        let mut store = Store::new();
        let b = store.insert_mesh("/world/b".into());
        let a = store.insert_mesh("/world/a".into());
        store
            .set_topology(
                b,
                Topology {
                    face_vertex_counts: vec![3],
                    face_vertex_indices: vec![0, 1, 2],
                    left_handed: true,
                    subsets: vec![Subset {
                        material: "/mtl/red".into(),
                        faces: vec![0],
                    }],
                },
            )
            .unwrap();
        store
            .set_points(b, vec![[0.0, 0.0, 0.0], [0.5, -0.0, 0.0], [0.0, 1e-7, 3.0]])
            .unwrap();
        store
            .set_normals(
                b,
                Some(Normals {
                    interpolation: Interpolation::Uniform,
                    values: vec![[0.0, 0.0, 1.0]],
                }),
            )
            .unwrap();
        let mut moved = [0.0; 16];
        for i in [0, 5, 10, 15] {
            moved[i] = 1.0;
        }
        moved[12] = 2.0;
        store.set_transform(b, moved).unwrap();
        store.set_visible(a, false).unwrap();
        store.set_render_tag(a, "proxy".into()).unwrap();
        let instancing = Instancing {
            instancer: "/world/i".into(),
            indices: vec![1, 0],
        };
        store.set_instancing(a, Some(instancing)).unwrap();
        let i = store.insert_instancer("/world/i".into());
        store.set_instancer_parent(i, None).unwrap();
        let primvars = InstancePrimvars {
            translations: Some(vec![[1.0, 0.0, 0.0]]),
            rotations: Some(vec![[0.0, 0.0, 0.0, 1.0]]),
            ..Default::default()
        };
        store.set_instance_primvars(i, primvars).unwrap();
        let c = store.insert_curves("/world/c".into());
        store.pulled_curves(c, 68).unwrap();
        // Prims of every kind are sorted together.
        store.insert_counted("sphereLight".into(), "/world/aa".into());

        let expected = "\
mesh /world/a
  render_tag proxy
  visible false
  instancer /world/i indices 1 0
  transform not synced
  topology not synced
  points not synced
  normals not synced
counted sphereLight /world/aa
mesh /world/b
  render_tag not synced
  visible not synced
  instancer not synced
  transform
    1 0 0 0
    0 1 0 0
    0 0 1 0
    2 0 0 1
  topology leftHanded
    face_vertex_counts 3
    face_vertex_indices 0 1 2
    subset \"/mtl/red\" faces 0
  points 3
    0 0 0
    0.5 -0 0
    0 0.0000001 3
  normals uniform 1
    0 0 1
curves /world/c
  bytes pulled 68
instancer /world/i
  parent none
  transform not synced
  instance_translations 1
    1 0 0
  instance_rotations 1
    0 0 0 1
";
        assert_eq!(store.dump(), expected);

        // A one-ulp change must show in the dump.
        let next = f32::from_bits(1e-7f32.to_bits() + 1);
        store
            .set_points(b, vec![[0.0, 0.0, 0.0], [0.5, -0.0, 0.0], [0.0, next, 3.0]])
            .unwrap();
        assert_ne!(store.dump(), expected);
    }
}
