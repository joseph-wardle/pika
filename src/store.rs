//! The scene as Hydra last synced it. One store per render delegate: the
//! delegate's prim syncs write it, field by field, and the renderer reads it.
//!
//! Each field is stored in Hydra's representation (world transforms, flattened
//! primvars, untriangulated faces), so a dump of the store can be checked
//! against what Hydra is known to hand over.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use crate::material::{self, Document, Network, Value};

pub struct Store {
    /// Ids are never reused, so a stale id is an error rather than a write to
    /// a newer prim; the exception is a material inserted again at its path
    /// within a frame, the same prim resynced (`freed`).
    next_id: u64,
    meshes: BTreeMap<u64, Mesh>,
    instancers: BTreeMap<u64, Instancer>,
    materials: BTreeMap<u64, Material>,
    /// Material ids by path, which is how a mesh's bindings name them.
    material_ids: BTreeMap<String, u64>,
    /// The materials removed since the last frame, by path: one inserted
    /// again at its path before the frame keeps its id (`insert_material`).
    freed: BTreeMap<String, u64>,
    curves: BTreeMap<u64, Curves>,
    lights: BTreeMap<u64, Light>,
    counted: BTreeMap<u64, Counted>,
    /// Each mesh's triangles, its cage's then as drawn, counted when its
    /// topology is set: the census runs every sync, and counting ALab's
    /// refined faces there cost more than the sync itself.
    triangles: HashMap<u64, [usize; 2]>,
    stats: Stats,
    moved: Moved,
}

/// Each field is `None` until Hydra first hands it over.
pub struct Mesh {
    pub path: String,
    pub topology: Option<Topology>,
    pub points: Option<Vec<[f32; 3]>>,
    /// `Some(None)` when the mesh has no normals primvar.
    pub normals: Option<Option<Normals>>,
    /// The texture coordinates: the float2 primvar the mesh's material
    /// reads, by name. `Some(None)` when the material reads none or the
    /// mesh lacks it.
    pub texcoord: Option<Option<Texcoord>>,
    /// Hydra's `GfMatrix4d`: row-major, translation in the last row.
    pub transform: Option<[f64; 16]>,
    pub visible: Option<bool>,
    /// `Some(None)` when the mesh is not instanced.
    pub instancing: Option<Option<Instancing>>,
    /// The whole mesh's material, for the full binding purpose; empty when
    /// none is bound. Subsets override it for their faces.
    pub material: Option<String>,
    /// Hydra syncs only the prims whose render tag a render pass draws, and
    /// leaves the rest in the store with stale data: a proxy mesh is inserted
    /// but never synced, and a mesh retagged as proxy keeps its old fields.
    /// Drawing must filter by the pass's tags.
    pub render_tag: Option<String>,
    /// The light-link collections the mesh is in, by Hydra's tokens: the
    /// lights linked to one of them, or to everything, light it.
    pub categories: Option<Vec<String>>,
}

/// A prototype's place in an instancer: it is drawn once per index, at that
/// index of the instancer's primvars. The prototype is a mesh, or an
/// instancer nested in another, whose instances are all repeated per index.
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
    pub parent: Option<Option<Instancing>>,
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

/// A material sprim: its shading network as Hydra hands it over, node by
/// node in USD's or MaterialX's vocabulary. Pika turns it into a MaterialX
/// document itself, so the host's view of it is kept whole.
pub struct Material {
    pub path: String,
    pub network: Option<Network>,
    /// The network's document, derived as the network is set: none without
    /// a surface; an error is warned of when its build is tried (`shade`).
    pub document: Result<Option<Document>, String>,
}

impl Material {
    /// The float2 primvar its surface reads, the texcoord its meshes carry:
    /// none without a surface, or when the network cannot be built.
    pub fn texcoord(&self) -> Option<&str> {
        match &self.document {
            Ok(Some(document)) => Some(&document.texcoord),
            _ => None,
        }
    }
}

/// Basis curves are fetched to measure what they cost, then dropped: Pika
/// does not draw them yet.
pub struct Curves {
    pub path: String,
    /// Every byte pulled since the curves were inserted.
    pub bytes: usize,
}

/// A distant, sphere, rect, disk or cylinder light. Each field is `None` until Hydra first hands
/// it over.
#[derive(PartialEq, Debug)]
pub struct Light {
    pub path: String,
    pub params: Option<LightParams>,
    pub transform: Option<[f64; 16]>,
    /// The token of its lightLink collection, the category of the meshes it
    /// lights; empty when it lights every mesh.
    pub link: Option<String>,
}

/// The UsdLux inputs Pika reads, as Hydra hands them over.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LightParams {
    pub intensity: f32,
    pub exposure: f32,
    pub color: [f32; 3],
    pub normalize: bool,
    pub shape: Shape,
    /// Hydra's visibility, which a light's params dirty bit covers too.
    pub visible: bool,
}

/// A light's type and the inputs only it has.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Shape {
    /// The angular diameter in degrees.
    Distant {
        angle: f32,
    },
    /// The sizes are before the light's transform.
    Sphere {
        radius: f32,
    },
    Rect {
        width: f32,
        height: f32,
    },
    Disk {
        radius: f32,
    },
    /// Along its x axis.
    Cylinder {
        length: f32,
        radius: f32,
    },
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
    pub scheme: Scheme,
    pub face_vertex_counts: Vec<i32>,
    pub face_vertex_indices: Vec<i32>,
    pub left_handed: bool,
    /// Faces that are not drawn. They still shape their points' normals.
    pub holes: Vec<i32>,
    pub subsets: Vec<Subset>,
    /// `None` when the scheme is none or OpenSubdiv refused the mesh: its
    /// cage is drawn.
    pub refined: Option<Refined>,
}

/// A subdivision surface as the delegate's OpenSubdiv refined it, one
/// uniform level. The refined faces are drawn in place of the cage's, and
/// each refined point is a stencil over the cage points.
#[derive(Clone, PartialEq, Debug)]
pub struct Refined {
    pub face_vertex_counts: Vec<i32>,
    pub face_vertex_indices: Vec<i32>,
    /// The refined faces of the cage's holes.
    pub holes: Vec<i32>,
    pub stencils: Stencils,
}

impl Refined {
    /// The refined points.
    pub fn point_count(&self) -> usize {
        self.stencils.count()
    }

    fn bytes(&self) -> usize {
        4 * (self.face_vertex_counts.len() + self.face_vertex_indices.len() + self.holes.len())
            + self.stencils.bytes()
    }
}

/// Refined values as weighted sums of the values they are refined from:
/// value i's stencil is entries `offsets[i]` to `offsets[i + 1]` of the
/// indices and weights.
#[derive(Clone, PartialEq, Debug)]
pub struct Stencils {
    pub offsets: Vec<u32>,
    pub indices: Vec<i32>,
    pub weights: Vec<f32>,
}

impl Stencils {
    /// The refined values.
    pub fn count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn bytes(&self) -> usize {
        4 * (self.offsets.len() + self.indices.len() + self.weights.len())
    }
}

/// USD's subdivision schemes. Every scheme but `None` is a subdivision
/// surface, whose normals come from its points, not from a primvar.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scheme {
    CatmullClark,
    Loop,
    Bilinear,
    None,
}

impl Scheme {
    /// The scheme USD names `token`.
    pub fn from_token(token: &str) -> Result<Self, String> {
        match token {
            "catmullClark" => Ok(Self::CatmullClark),
            "loop" => Ok(Self::Loop),
            "bilinear" => Ok(Self::Bilinear),
            "none" => Ok(Self::None),
            _ => Err(format!("unknown subdivision scheme {token:?}")),
        }
    }

    fn token(self) -> &'static str {
        match self {
            Self::CatmullClark => "catmullClark",
            Self::Loop => "loop",
            Self::Bilinear => "bilinear",
            Self::None => "none",
        }
    }
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

#[derive(Clone, PartialEq, Debug)]
pub struct Texcoord {
    pub name: String,
    pub interpolation: Interpolation,
    pub values: Vec<[f32; 2]>,
    /// How a subdivided mesh's refined faces read it; `None` when the mesh
    /// is drawn as its cage.
    pub refined: Option<RefinedTexcoord>,
}

/// A texcoord refined with its mesh by the delegate's OpenSubdiv: per
/// corner of the refined faces, in `Refined`'s order, the refined value it
/// reads, each a stencil over the values handed over. A faceVarying
/// texcoord's values are welded first (see the delegate), so they are the
/// distinct ones.
#[derive(Clone, PartialEq, Debug)]
pub struct RefinedTexcoord {
    pub face_vertex_indices: Vec<i32>,
    pub stencils: Stencils,
}

impl RefinedTexcoord {
    fn bytes(&self) -> usize {
        4 * self.face_vertex_indices.len() + self.stencils.bytes()
    }
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
    /// The interpolation USD names `token`.
    pub fn from_token(token: &str) -> Result<Self, String> {
        match token {
            "constant" => Ok(Self::Constant),
            "uniform" => Ok(Self::Uniform),
            "varying" => Ok(Self::Varying),
            "vertex" => Ok(Self::Vertex),
            "faceVarying" => Ok(Self::FaceVarying),
            _ => Err(format!("unknown primvar interpolation {token:?}")),
        }
    }

    fn token(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Uniform => "uniform",
            Self::Varying => "varying",
            Self::Vertex => "vertex",
            Self::FaceVarying => "faceVarying",
        }
    }
}

/// What changed since the renderer last took it, so a frame uploads only
/// that. Only changed values count: a redundant pull moves nothing.
#[derive(PartialEq, Debug, Default)]
pub struct Moved {
    /// A mesh was added, removed or remodelled, its points, normals or
    /// texcoord changed size or kind, or a cage's faceVarying texcoord its
    /// values: the layout of every mesh's data is rebuilt.
    pub layout: bool,
    /// Meshes whose points changed, keeping their count.
    pub points: BTreeSet<u64>,
    /// Meshes whose normals changed, keeping their interpolation and count.
    pub normals: BTreeSet<u64>,
    /// Meshes whose texcoord changed, keeping its interpolation and count.
    pub texcoord: BTreeSet<u64>,
    /// Meshes whose transform changed: only their instances are rewritten.
    pub transforms: BTreeSet<u64>,
    /// A visibility, render tag, instancing or light link changed: the
    /// instances, which carry their meshes' links, and draw records are
    /// rewritten.
    pub draws: bool,
    /// Materials whose network changed.
    pub materials: BTreeSet<u64>,
    /// A light was added, removed or changed.
    pub lights: bool,
}

/// What one Hydra sync did to the store.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Stats {
    pub inserted: usize,
    pub removed: usize,
    pub mesh: Bytes,
    pub instancer: Bytes,
    pub material: Bytes,
    pub light: Bytes,
    /// Bytes of basis curves, which are not stored, so are never known to be
    /// redundant.
    pub curve_bytes: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Bytes {
    pub pulled: usize,
    /// Bytes of pulls that matched what the store already held: data Hydra
    /// marked dirty that did not change, such as constants authored as time
    /// samples.
    pub redundant: usize,
}

/// What the store holds that a final-look pass will draw: meshes Hydra
/// synced as visible, with the geometry or render tag. An invisible or
/// retagged mesh keeps its old fields.
#[derive(PartialEq, Debug, Default)]
pub struct Census {
    pub meshes: usize,
    /// Each mesh once per instance it is drawn at.
    pub mesh_instances: usize,
    /// Triangles as faces fan into them, before any subdivision.
    pub triangles: usize,
    /// Triangles counting each mesh once, as its data would be stored.
    pub prototype_triangles: usize,
    /// The same two counts as drawn: a refined mesh's refined faces.
    pub refined_triangles: usize,
    pub refined_prototype_triangles: usize,
    /// Points, normals, texcoord and face indices, counting each mesh once.
    pub geometry_bytes: usize,
    /// The refined faces and stencils, counting each mesh once.
    pub refined_bytes: usize,
    /// Materials bound to the meshes or their subsets.
    pub materials: usize,
    pub instancers: usize,
    pub curves: usize,
    pub lights: usize,
    /// Counted prims by Hydra type.
    pub counted: BTreeMap<String, usize>,
}

impl Store {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            meshes: BTreeMap::new(),
            instancers: BTreeMap::new(),
            materials: BTreeMap::new(),
            material_ids: BTreeMap::new(),
            freed: BTreeMap::new(),
            curves: BTreeMap::new(),
            lights: BTreeMap::new(),
            counted: BTreeMap::new(),
            triangles: HashMap::new(),
            stats: Stats::default(),
            moved: Moved::default(),
        }
    }

    pub fn meshes(&self) -> &BTreeMap<u64, Mesh> {
        &self.meshes
    }

    pub fn instancers(&self) -> &BTreeMap<u64, Instancer> {
        &self.instancers
    }

    pub fn materials(&self) -> &BTreeMap<u64, Material> {
        &self.materials
    }

    pub fn lights(&self) -> &BTreeMap<u64, Light> {
        &self.lights
    }

    /// What changed since the last call.
    pub fn take_moved(&mut self) -> Moved {
        self.freed.clear();
        std::mem::take(&mut self.moved)
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
                texcoord: None,
                transform: None,
                visible: None,
                instancing: None,
                material: None,
                render_tag: None,
                categories: None,
            },
        );
        self.moved.layout = true;
        id
    }

    pub fn remove_mesh(&mut self, id: u64) -> Result<(), String> {
        self.meshes.remove(&id).ok_or_else(|| stale("mesh", id))?;
        self.triangles.remove(&id);
        self.stats.removed += 1;
        self.moved.layout = true;
        Ok(())
    }

    pub fn set_topology(&mut self, id: u64, topology: Topology) -> Result<(), String> {
        let bytes = 4
            * (topology.face_vertex_counts.len()
                + topology.face_vertex_indices.len()
                + topology.holes.len())
            + topology.scheme.token().len()
            + 1
            + topology.refined.as_ref().map_or(0, Refined::bytes)
            + topology
                .subsets
                .iter()
                .map(|s| s.material.len() + 4 * s.faces.len())
                .sum::<usize>();
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        let fanned = |counts: &[i32]| -> usize {
            counts.iter().map(|&n| n.saturating_sub(2) as usize).sum()
        };
        let cage = fanned(&topology.face_vertex_counts);
        let drawn = (topology.refined.as_ref()).map_or(cage, |r| fanned(&r.face_vertex_counts));
        self.triangles.insert(id, [cage, drawn]);
        self.moved.layout |= write(&mut self.stats.mesh, &mut mesh.topology, topology, bytes);
        Ok(())
    }

    pub fn set_points(&mut self, id: u64, points: Vec<[f32; 3]>) -> Result<(), String> {
        let bytes = size_of_val(points.as_slice());
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        let count = mesh.points.as_ref().map(Vec::len);
        let resized = count != Some(points.len());
        if write(&mut self.stats.mesh, &mut mesh.points, points, bytes) {
            if resized {
                self.moved.layout = true;
            } else {
                self.moved.points.insert(id);
            }
        }
        Ok(())
    }

    /// `None` when the mesh has no normals primvar.
    pub fn set_normals(&mut self, id: u64, normals: Option<Normals>) -> Result<(), String> {
        let bytes = normals
            .as_ref()
            .map_or(0, |n| size_of_val(n.values.as_slice()));
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        let kind = |n: &Option<Normals>| n.as_ref().map(|n| (n.interpolation, n.values.len()));
        let reshaped = mesh.normals.as_ref().map(kind) != Some(kind(&normals));
        if write(&mut self.stats.mesh, &mut mesh.normals, normals, bytes) {
            if reshaped {
                self.moved.layout = true;
            } else {
                self.moved.normals.insert(id);
            }
        }
        Ok(())
    }

    /// `None` when the mesh's material reads no primvar or the mesh lacks
    /// the one it reads.
    pub fn set_texcoord(&mut self, id: u64, texcoord: Option<Texcoord>) -> Result<(), String> {
        let bytes = (texcoord.as_ref()).map_or(0, |t| {
            size_of_val(t.values.as_slice()) + t.refined.as_ref().map_or(0, RefinedTexcoord::bytes)
        });
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        // Values alone may change in place: the interpolation, the count and
        // the refinement are the layout's, as are a cage's faceVarying
        // values, which its vertices weld by value (`geometry`).
        let reshaped = match (&mesh.texcoord, &texcoord) {
            (Some(Some(old)), Some(new)) => {
                old.interpolation != new.interpolation
                    || old.values.len() != new.values.len()
                    || old.refined != new.refined
                    || (new.refined.is_none() && new.interpolation == Interpolation::FaceVarying)
            }
            (Some(None), None) => false,
            _ => true,
        };
        if write(&mut self.stats.mesh, &mut mesh.texcoord, texcoord, bytes) {
            if reshaped {
                self.moved.layout = true;
            } else {
                self.moved.texcoord.insert(id);
            }
        }
        Ok(())
    }

    /// The float2 primvar mesh `id`'s materials read, the subsets'
    /// included, the texcoord the delegate fetches for it: none when no
    /// material is bound, or none bound is in the store with a network
    /// whose surface reads one. The materials must agree on it, since the
    /// mesh is drawn with one texcoord.
    pub fn texcoord_name(&self, id: u64) -> Result<Option<String>, String> {
        let mesh = self.meshes.get(&id).ok_or_else(|| stale("mesh", id))?;
        let subsets = (mesh.topology.iter()).flat_map(|t| t.subsets.iter().map(|s| &s.material));
        let mut reads: Vec<(&str, &str)> = Vec::new();
        for path in mesh.material.iter().chain(subsets) {
            let material = self.material_ids.get(path).map(|id| &self.materials[id]);
            let Some(name) = material.and_then(Material::texcoord) else {
                continue;
            };
            if !reads.iter().any(|(n, _)| *n == name) {
                reads.push((name, path));
            }
        }
        match reads.as_slice() {
            [] => Ok(None),
            [(name, _)] => Ok(Some((*name).to_owned())),
            [(a, pa), (b, pb), ..] => Err(format!(
                "{pa} reads {a} and {pb} reads {b}; a mesh is drawn with one texcoord, so bind materials that read the same float2 primvar"
            )),
        }
    }

    pub fn set_transform(&mut self, id: u64, transform: [f64; 16]) -> Result<(), String> {
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        if write(&mut self.stats.mesh, &mut mesh.transform, transform, 128) {
            self.moved.transforms.insert(id);
        }
        Ok(())
    }

    pub fn set_visible(&mut self, id: u64, visible: bool) -> Result<(), String> {
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        self.moved.draws |= write(&mut self.stats.mesh, &mut mesh.visible, visible, 1);
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
        self.moved.draws |= write(
            &mut self.stats.mesh,
            &mut mesh.instancing,
            instancing,
            bytes,
        );
        Ok(())
    }

    /// `material` is empty when none is bound. The draw records are grouped
    /// by material, so a binding rewrites them.
    pub fn set_material(&mut self, id: u64, material: String) -> Result<(), String> {
        let bytes = material.len();
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        self.moved.draws |= write(&mut self.stats.mesh, &mut mesh.material, material, bytes);
        Ok(())
    }

    pub fn set_render_tag(&mut self, id: u64, tag: String) -> Result<(), String> {
        let bytes = tag.len();
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        self.moved.draws |= write(&mut self.stats.mesh, &mut mesh.render_tag, tag, bytes);
        Ok(())
    }

    pub fn set_categories(&mut self, id: u64, categories: Vec<String>) -> Result<(), String> {
        let bytes = categories.iter().map(String::len).sum();
        let mesh = self.meshes.get_mut(&id).ok_or_else(|| stale("mesh", id))?;
        self.moved.draws |= write(
            &mut self.stats.mesh,
            &mut mesh.categories,
            categories,
            bytes,
        );
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
        self.moved.draws = true;
        id
    }

    pub fn remove_instancer(&mut self, id: u64) -> Result<(), String> {
        self.instancers
            .remove(&id)
            .ok_or_else(|| stale("instancer", id))?;
        self.stats.removed += 1;
        self.moved.draws = true;
        Ok(())
    }

    /// `None` when the instancer is not nested in another.
    pub fn set_instancer_parent(
        &mut self,
        id: u64,
        parent: Option<Instancing>,
    ) -> Result<(), String> {
        let bytes = parent
            .as_ref()
            .map_or(0, |p| p.instancer.len() + 4 * p.indices.len());
        let instancer = self
            .instancers
            .get_mut(&id)
            .ok_or_else(|| stale("instancer", id))?;
        self.moved.draws |= write(
            &mut self.stats.instancer,
            &mut instancer.parent,
            parent,
            bytes,
        );
        Ok(())
    }

    pub fn set_instancer_transform(&mut self, id: u64, transform: [f64; 16]) -> Result<(), String> {
        let instancer = self
            .instancers
            .get_mut(&id)
            .ok_or_else(|| stale("instancer", id))?;
        self.moved.draws |= write(
            &mut self.stats.instancer,
            &mut instancer.transform,
            transform,
            128,
        );
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
        self.moved.draws |= write(
            &mut self.stats.instancer,
            &mut instancer.primvars,
            primvars,
            bytes,
        );
        Ok(())
    }

    /// A material's meshes may be in the store before it: both rewrite the
    /// draw records.
    pub fn insert_material(&mut self, path: String) -> u64 {
        // Houdini 21's UsdImagingGL resyncs a material whose connections
        // change, removing it and inserting it again at its path before
        // the next frame: it keeps its id, the one exception to `next_id`,
        // so `shade` keeps drawing its look until its new build lands.
        let id = match self.freed.remove(&path) {
            Some(id) => {
                self.stats.inserted += 1;
                id
            }
            None => self.new_id(),
        };
        self.material_ids.insert(path.clone(), id);
        self.materials.insert(
            id,
            Material {
                path,
                network: None,
                document: Ok(None),
            },
        );
        self.moved.draws = true;
        id
    }

    pub fn remove_material(&mut self, id: u64) -> Result<(), String> {
        let material = self
            .materials
            .remove(&id)
            .ok_or_else(|| stale("material", id))?;
        self.material_ids.remove(&material.path);
        self.freed.insert(material.path, id);
        self.stats.removed += 1;
        self.moved.draws = true;
        Ok(())
    }

    /// Replaces the material's network and its document, and says whether
    /// its meshes must sync again: the surface reads a primvar the one before
    /// did not, so their texcoord is stale. A first network is not a change:
    /// its meshes sync after it anyway.
    pub fn set_network(&mut self, id: u64, network: Network) -> Result<bool, String> {
        let bytes = network.bytes();
        let material = self
            .materials
            .get_mut(&id)
            .ok_or_else(|| stale("material", id))?;
        let before = material.texcoord().map(str::to_owned);
        material.document = material::document(&network);
        let changed = material.network.is_some() && material.texcoord() != before.as_deref();
        if write(
            &mut self.stats.material,
            &mut material.network,
            network,
            bytes,
        ) {
            self.moved.materials.insert(id);
        }
        Ok(changed)
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
        self.stats.curve_bytes += bytes;
        Ok(())
    }

    /// Adds a light with nothing synced yet.
    pub fn insert_light(&mut self, path: String) -> u64 {
        let id = self.new_id();
        let light = Light {
            path,
            params: None,
            transform: None,
            link: None,
        };
        self.lights.insert(id, light);
        self.moved.lights = true;
        id
    }

    pub fn remove_light(&mut self, id: u64) -> Result<(), String> {
        let light = self.lights.remove(&id).ok_or_else(|| stale("light", id))?;
        self.stats.removed += 1;
        self.moved.lights = true;
        self.moved.draws |= light.link.is_some_and(|l| !l.is_empty());
        Ok(())
    }

    pub fn set_light_params(&mut self, id: u64, params: LightParams) -> Result<(), String> {
        let light = self.lights.get_mut(&id).ok_or_else(|| stale("light", id))?;
        let bytes = size_of::<LightParams>();
        self.moved.lights |= write(&mut self.stats.light, &mut light.params, params, bytes);
        Ok(())
    }

    pub fn set_light_transform(&mut self, id: u64, transform: [f64; 16]) -> Result<(), String> {
        let light = self.lights.get_mut(&id).ok_or_else(|| stale("light", id))?;
        self.moved.lights |= write(&mut self.stats.light, &mut light.transform, transform, 128);
        Ok(())
    }

    /// A link moves the meshes too: each collection's bit is assigned over
    /// every light's link (`link::Links`).
    pub fn set_light_link(&mut self, id: u64, link: String) -> Result<(), String> {
        let bytes = link.len();
        let light = self.lights.get_mut(&id).ok_or_else(|| stale("light", id))?;
        let moved = write(&mut self.stats.light, &mut light.link, link, bytes);
        self.moved.lights |= moved;
        self.moved.draws |= moved;
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

    /// What drawing the store would cost. Fails when a prototype names an
    /// instancer the store does not hold, which Hydra's sync order rules out.
    pub fn census(&self) -> Result<Census, String> {
        let instancers: HashMap<&str, &Instancer> = (self.instancers.values())
            .map(|i| (i.path.as_str(), i))
            .collect();
        let mut census = Census {
            instancers: self.instancers.len(),
            curves: self.curves.len(),
            lights: self.lights.len(),
            ..Default::default()
        };
        let mut materials = BTreeSet::new();
        for (id, mesh) in &self.meshes {
            let (Some(true), Some(topology)) = (mesh.visible, &mesh.topology) else {
                continue;
            };
            if !matches!(mesh.render_tag.as_deref(), Some("geometry" | "render")) {
                continue;
            }
            let instances = match &mesh.instancing {
                Some(Some(i)) => i.indices.len() * repeats(&instancers, &i.instancer)?,
                _ => 1,
            };
            let [triangles, refined] = self.triangles[id];
            census.meshes += 1;
            census.mesh_instances += instances;
            census.triangles += triangles * instances;
            census.prototype_triangles += triangles;
            census.refined_triangles += refined * instances;
            census.refined_prototype_triangles += refined;
            census.refined_bytes += topology.refined.as_ref().map_or(0, Refined::bytes)
                + match &mesh.texcoord {
                    Some(Some(t)) => t.refined.as_ref().map_or(0, RefinedTexcoord::bytes),
                    _ => 0,
                };
            census.geometry_bytes += 4 * topology.face_vertex_indices.len()
                + mesh
                    .points
                    .as_ref()
                    .map_or(0, |p| size_of_val(p.as_slice()))
                + match &mesh.normals {
                    Some(Some(n)) => size_of_val(n.values.as_slice()),
                    _ => 0,
                }
                + match &mesh.texcoord {
                    Some(Some(t)) => size_of_val(t.values.as_slice()),
                    _ => 0,
                };
            materials.extend(mesh.material.iter().map(String::as_str));
            materials.extend(topology.subsets.iter().map(|s| s.material.as_str()));
        }
        materials.remove("");
        census.materials = materials.len();
        for c in self.counted.values() {
            *census.counted.entry(c.prim_type.clone()).or_default() += 1;
        }
        Ok(census)
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
                self.materials
                    .values()
                    .map(|m| (m.path.as_str(), m.to_string())),
            )
            .chain(
                self.curves
                    .values()
                    .map(|c| (c.path.as_str(), c.to_string())),
            )
            .chain(
                self.lights
                    .values()
                    .map(|l| (l.path.as_str(), l.to_string())),
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
        match &self.material {
            Some(material) => writeln!(f, "  material {material:?}")?,
            None => writeln!(f, "  material {UNSYNCED}")?,
        }
        match &self.categories {
            Some(categories) => writeln!(f, "  categories {categories:?}")?,
            None => writeln!(f, "  categories {UNSYNCED}")?,
        }
        match &self.topology {
            Some(t) => {
                let orientation = if t.left_handed {
                    "leftHanded"
                } else {
                    "rightHanded"
                };
                writeln!(f, "  topology {} {orientation}", t.scheme.token())?;
                writeln!(f, "    face_vertex_counts{}", List(&t.face_vertex_counts))?;
                writeln!(f, "    face_vertex_indices{}", List(&t.face_vertex_indices))?;
                if !t.holes.is_empty() {
                    writeln!(f, "    holes{}", List(&t.holes))?;
                }
                for s in &t.subsets {
                    writeln!(f, "    subset {:?} faces{}", s.material, List(&s.faces))?;
                }
                // The refined faces' order is OpenSubdiv's, and the image
                // check covers their shape, so only their sizes are dumped.
                if let Some(r) = &t.refined {
                    let (faces, points) = (r.face_vertex_counts.len(), r.point_count());
                    write!(f, "    refined faces {faces} points {points}")?;
                    if !r.holes.is_empty() {
                        write!(f, " holes{}", List(&r.holes))?;
                    }
                    writeln!(f)?;
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
                writeln!(
                    f,
                    "  normals {} {}",
                    n.interpolation.token(),
                    n.values.len()
                )?;
                for v in &n.values {
                    writeln!(f, "   {}", List(v))?;
                }
            }
            Some(None) => writeln!(f, "  normals none")?,
            None => writeln!(f, "  normals {UNSYNCED}")?,
        }
        match &self.texcoord {
            Some(Some(t)) => {
                let (name, kind) = (&t.name, t.interpolation.token());
                write!(f, "  texcoord {name} {kind} {}", t.values.len())?;
                if let Some(r) = &t.refined {
                    write!(f, " refined values {}", r.stencils.count())?;
                }
                writeln!(f)?;
                for v in &t.values {
                    writeln!(f, "   {}", List(v))?;
                }
            }
            Some(None) => writeln!(f, "  texcoord none")?,
            None => writeln!(f, "  texcoord {UNSYNCED}")?,
        }
        Ok(())
    }
}

impl fmt::Display for Instancer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "instancer {}", self.path)?;
        match &self.parent {
            Some(Some(p)) => writeln!(f, "  parent {} indices{}", p.instancer, List(&p.indices))?,
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

impl fmt::Display for Material {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "material {}", self.path)?;
        let Some(network) = &self.network else {
            return writeln!(f, "  network {UNSYNCED}");
        };
        match &network.surface {
            Some(surface) => writeln!(f, "  surface {surface}")?,
            None => writeln!(f, "  surface none")?,
        }
        for (path, node) in &network.nodes {
            writeln!(f, "  node {path} {}", node.node_type)?;
            for (name, value) in &node.params {
                match value {
                    Value::Float(v) => writeln!(f, "    {name} float{}", List(v))?,
                    Value::Int(v) => writeln!(f, "    {name} int {v}")?,
                    Value::Bool(v) => writeln!(f, "    {name} bool {v}")?,
                    Value::Text(s) => writeln!(f, "    {name} text {s:?}")?,
                    Value::Asset(s) => writeln!(f, "    {name} asset {s:?}")?,
                }
            }
            for (name, c) in &node.connections {
                writeln!(f, "    {name} from {} {}", c.node, c.output)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for Curves {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "curves {}", self.path)?;
        writeln!(f, "  bytes pulled {}", self.bytes)
    }
}

impl fmt::Display for Light {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "light {}", self.path)?;
        match &self.params {
            Some(p) => {
                writeln!(f, "  visible {}", p.visible)?;
                writeln!(
                    f,
                    "  intensity {} exposure {} color{} normalize {}",
                    p.intensity,
                    p.exposure,
                    List(&p.color),
                    p.normalize,
                )?;
                match p.shape {
                    Shape::Distant { angle } => writeln!(f, "  distant angle {angle}")?,
                    Shape::Sphere { radius } => writeln!(f, "  sphere radius {radius}")?,
                    Shape::Rect { width, height } => {
                        writeln!(f, "  rect width {width} height {height}")?
                    }
                    Shape::Disk { radius } => writeln!(f, "  disk radius {radius}")?,
                    Shape::Cylinder { length, radius } => {
                        writeln!(f, "  cylinder length {length} radius {radius}")?
                    }
                }
            }
            None => writeln!(f, "  params {UNSYNCED}")?,
        }
        match &self.link {
            Some(link) => writeln!(f, "  link {link:?}")?,
            None => writeln!(f, "  link {UNSYNCED}")?,
        }
        transform(f, self.transform.as_ref())
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

/// How many times each instance of the instancer at `path` is drawn: once
/// per index its parent has for it, and so on up.
fn repeats<'a>(
    instancers: &HashMap<&'a str, &'a Instancer>,
    mut path: &'a str,
) -> Result<usize, String> {
    let mut repeats = 1;
    for _ in 0..=instancers.len() {
        let instancer = instancers
            .get(path)
            .ok_or_else(|| format!("no instancer {path} in the store"))?;
        let Some(Some(parent)) = &instancer.parent else {
            return Ok(repeats);
        };
        repeats *= parent.indices.len();
        path = &parent.instancer;
    }
    Err(format!("instancer {path} is nested in itself"))
}

fn stale(kind: &str, id: u64) -> String {
    format!("no {kind} with id {id} in the store: it was removed, or never inserted")
}

/// Stores one pulled field, counting its bytes as redundant when Hydra handed
/// over the same value before. Returns whether the value changed.
fn write<T: PartialEq>(stats: &mut Bytes, slot: &mut Option<T>, value: T, bytes: usize) -> bool {
    stats.pulled += bytes;
    if slot.as_ref() == Some(&value) {
        stats.redundant += bytes;
        return false;
    }
    *slot = Some(value);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Connection;

    #[test]
    fn lights_move_only_when_they_change() {
        let mut store = Store::new();
        let id = store.insert_light("/sun".into());
        assert!(store.take_moved().lights);
        let params = LightParams {
            intensity: 1.0,
            exposure: 0.0,
            color: [1.0; 3],
            normalize: false,
            shape: Shape::Distant { angle: 0.53 },
            visible: true,
        };
        let mut transform = [0.0; 16];
        transform[0] = 1.0;
        store.set_light_params(id, params).unwrap();
        store.set_light_transform(id, transform).unwrap();
        assert!(store.take_moved().lights);
        // Hydra resends a light's fields unchanged, as usdview does its
        // camera light each frame: they move nothing.
        store.end_sync();
        store.set_light_params(id, params).unwrap();
        store.set_light_transform(id, transform).unwrap();
        assert_eq!(store.take_moved(), Moved::default());
        let stats = store.end_sync();
        assert!(stats.light.pulled > 0 && stats.light.redundant == stats.light.pulled);
        let brighter = LightParams {
            intensity: 2.0,
            ..params
        };
        store.set_light_params(id, brighter).unwrap();
        assert!(store.take_moved().lights);
        store.remove_light(id).unwrap();
        assert!(store.take_moved().lights);
        assert!(store.set_light_params(id, params).is_err());
        assert_eq!(store.census().unwrap().lights, 0);
    }

    #[test]
    fn ids_are_never_reused() {
        let mut store = Store::new();
        let a = store.insert_mesh("/a".into());
        store.remove_mesh(a).unwrap();
        let b = store.insert_mesh("/a".into());
        assert_ne!(a, b);
    }

    #[test]
    fn a_material_inserted_again_in_the_frame_keeps_its_id() {
        let mut store = Store::new();
        let id = store.insert_material("/m".into());
        store.remove_material(id).unwrap();
        assert_eq!(
            store.insert_material("/m".into()),
            id,
            "resynced in the frame"
        );
        store.remove_material(id).unwrap();
        assert_ne!(store.insert_material("/n".into()), id, "another path");
        store.take_moved();
        assert_ne!(store.insert_material("/m".into()), id, "a frame later");
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
        assert_eq!((stats.curve_bytes, stats.mesh), (136, Bytes::default()));
    }

    #[test]
    fn network_edits_are_not_redundant() {
        let mut store = Store::new();
        let m = store.insert_material("/m".into());
        store.end_sync();
        let mut network = Network {
            surface: Some("/m/s".into()),
            ..Default::default()
        };
        let node = network.nodes.entry("/m/s".into()).or_default();
        node.node_type = "UsdPreviewSurface".into();
        node.params
            .insert("roughness".into(), Value::Float(vec![0.5]));
        node.params
            .insert("file".into(), Value::Asset("/t.exr".into()));
        node.connections.insert(
            "diffuseColor".into(),
            Connection {
                node: "/m/t".into(),
                output: "rgb".into(),
            },
        );
        store.set_network(m, network.clone()).unwrap();
        store.set_network(m, network.clone()).unwrap();
        let stats = store.end_sync();
        let bytes = network.bytes();
        assert_eq!(
            (stats.material.pulled, stats.material.redundant),
            (2 * bytes, bytes)
        );
        // The control: a value edit is a pull the store had to keep.
        let node = network.nodes.get_mut("/m/s").unwrap();
        node.params
            .insert("roughness".into(), Value::Float(vec![0.25]));
        store.set_network(m, network).unwrap();
        let stats = store.end_sync();
        assert_eq!(
            (stats.material.pulled, stats.material.redundant),
            (bytes, 0)
        );
        assert_eq!(
            store.dump(),
            "material /m\n  surface /m/s\n  node /m/s UsdPreviewSurface\n    file asset \"/t.exr\"\n    roughness float 0.25\n    diffuseColor from /m/t rgb\n"
        );
    }

    #[test]
    fn the_texcoord_name_follows_the_bound_material() {
        let mut store = Store::new();
        let a = store.insert_mesh("/a".into());
        assert_eq!(store.texcoord_name(a).unwrap(), None);
        store.set_material(a, "/mtl/paint".into()).unwrap();
        // Bound before the material is in the store, or has a network.
        assert_eq!(store.texcoord_name(a).unwrap(), None);
        let m = store.insert_material("/mtl/paint".into());
        assert_eq!(store.texcoord_name(a).unwrap(), None);
        let network = material::tests::fixture();
        let st = Some("st".to_owned());
        // A first network is not a change.
        assert!(!store.set_network(m, network.clone()).unwrap());
        assert_eq!(store.texcoord_name(a).unwrap(), st);
        // A value edit keeps what it reads; a new name is a change.
        let mut edited = network.clone();
        (edited.nodes.get_mut("surface").unwrap().params)
            .insert("roughness".into(), Value::Float(vec![0.75]));
        assert!(!store.set_network(m, edited).unwrap());
        assert!(store.set_network(m, reading("perfuv")).unwrap());
        assert_eq!(store.texcoord_name(a).unwrap(), Some("perfuv".to_owned()));
        // A network that cannot be built gives its meshes none, and keeps
        // the error for its build; the control is a stale mesh.
        let mut broken = network.clone();
        broken.nodes.get_mut("surface").unwrap().node_type = "bogus".into();
        assert!(store.set_network(m, broken).unwrap());
        assert_eq!(store.texcoord_name(a).unwrap(), None);
        assert!(store.materials()[&m].document.is_err());
        assert!(store.texcoord_name(a + 1).is_err());
        // A material removed is unbound; one of the same path is bound again.
        store.set_network(m, network.clone()).unwrap();
        store.remove_material(m).unwrap();
        assert_eq!(store.texcoord_name(a).unwrap(), None);
        let m = store.insert_material("/mtl/paint".into());
        store.set_network(m, network).unwrap();
        assert_eq!(store.texcoord_name(a).unwrap(), st);
    }

    /// The fixture network, its texture read at the float2 primvar `name`.
    fn reading(name: &str) -> Network {
        let mut network = material::tests::fixture();
        (network.nodes.get_mut("st").unwrap().params)
            .insert("varname".into(), Value::Text(name.into()));
        network
    }

    #[test]
    fn the_texcoord_name_is_the_one_the_subsets_materials_agree_on() {
        let mut store = Store::new();
        let a = store.insert_mesh("/a".into());
        let paint = store.insert_material("/mtl/paint".into());
        let fur = store.insert_material("/mtl/fur".into());
        (store.set_network(paint, reading("st"))).unwrap();
        (store.set_network(fur, reading("perfuv"))).unwrap();
        let subsets = |materials: &[&str]| Topology {
            scheme: Scheme::None,
            face_vertex_counts: vec![3; materials.len()],
            face_vertex_indices: (0..3 * materials.len() as i32).collect(),
            left_handed: false,
            holes: vec![],
            subsets: (materials.iter().enumerate())
                .map(|(f, m)| Subset {
                    material: m.to_string(),
                    faces: vec![f as i32],
                })
                .collect(),
            refined: None,
        };
        // Unbound as a whole, the mesh reads what its subsets' materials do.
        store.set_topology(a, subsets(&["/mtl/paint", ""])).unwrap();
        assert_eq!(store.texcoord_name(a).unwrap(), Some("st".to_owned()));
        store.set_material(a, "/mtl/paint".into()).unwrap();
        store
            .set_topology(a, subsets(&["/mtl/paint", "/mtl/paint"]))
            .unwrap();
        assert_eq!(store.texcoord_name(a).unwrap(), Some("st".to_owned()));
        // The control: materials reading different primvars name the two.
        store
            .set_topology(a, subsets(&["/mtl/paint", "/mtl/fur"]))
            .unwrap();
        let e = store.texcoord_name(a).unwrap_err();
        assert!(e.contains("/mtl/fur reads perfuv"), "{e}");
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
        assert_eq!((first.mesh.pulled, first.mesh.redundant), (49, 0));

        store.set_points(a, points).unwrap();
        store.set_visible(a, true).unwrap();
        let same = store.end_sync();
        assert_eq!((same.mesh.pulled, same.mesh.redundant), (49, 49));
        assert_eq!(same.instancer, Bytes::default());

        // A changed value is not redundant, and is stored.
        store.set_points(a, vec![[0.0, 1.0, 3.0]; 4]).unwrap();
        assert_eq!(store.end_sync().mesh.redundant, 0);
        assert_eq!(
            store.meshes[&a].points.as_ref().unwrap()[0],
            [0.0, 1.0, 3.0]
        );
    }

    #[test]
    fn moved_holds_only_what_changed() {
        let mut store = Store::new();
        let a = drawn(&mut store, "/a", None);
        assert!(store.take_moved().layout);
        assert_eq!(store.take_moved(), Moved::default());

        // A redundant pull moves nothing; a changed one moves only its kind.
        store.set_points(a, vec![[0.0; 3]; 4]).unwrap();
        store.set_visible(a, true).unwrap();
        store.set_render_tag(a, "geometry".into()).unwrap();
        assert_eq!(store.take_moved(), Moved::default());
        // A retagged mesh keeps its data, but the passes drawing it change.
        store.set_render_tag(a, "proxy".into()).unwrap();
        assert!(store.take_moved().draws);
        store.set_points(a, vec![[1.0; 3]; 4]).unwrap();
        let normals = |values| Normals {
            interpolation: Interpolation::Vertex,
            values,
        };
        store
            .set_normals(a, Some(normals(vec![[0.0; 3]; 4])))
            .unwrap();
        let moved = store.take_moved();
        // Normals arriving for the first time change the layout.
        assert_eq!((moved.layout, &moved.points), (true, &BTreeSet::from([a])));
        store
            .set_normals(a, Some(normals(vec![[1.0; 3]; 4])))
            .unwrap();
        store.set_transform(a, [1.0; 16]).unwrap();
        let moved = store.take_moved();
        assert_eq!((moved.layout, moved.draws), (false, false));
        assert_eq!(moved.normals, BTreeSet::from([a]));
        assert_eq!(moved.transforms, BTreeSet::from([a]));
        let st = |values, refined| Texcoord {
            name: "st".into(),
            interpolation: Interpolation::FaceVarying,
            values,
            refined,
        };
        store
            .set_texcoord(a, Some(st(vec![[0.0; 2]; 7], None)))
            .unwrap();
        assert!(store.take_moved().layout);
        store
            .set_texcoord(a, Some(st(vec![[0.0; 2]; 7], None)))
            .unwrap();
        assert_eq!(store.take_moved(), Moved::default());
        // A cage's faceVarying values are welded into its vertices, so new
        // ones move the layout; vertex values move the texcoord alone.
        store
            .set_texcoord(a, Some(st(vec![[1.0; 2]; 7], None)))
            .unwrap();
        let moved = store.take_moved();
        assert!(moved.layout && moved.texcoord.is_empty());
        let vertex = |values| Texcoord {
            interpolation: Interpolation::Vertex,
            ..st(values, None)
        };
        store
            .set_texcoord(a, Some(vertex(vec![[0.0; 2]; 4])))
            .unwrap();
        assert!(store.take_moved().layout);
        store
            .set_texcoord(a, Some(vertex(vec![[1.0; 2]; 4])))
            .unwrap();
        let moved = store.take_moved();
        assert_eq!(
            (moved.layout, &moved.texcoord),
            (false, &BTreeSet::from([a]))
        );
        // Refining it moves the layout, as does refining it differently;
        // new values under the same refinement move the texcoord.
        let refined = |weight| {
            Some(RefinedTexcoord {
                face_vertex_indices: vec![0; 3],
                stencils: Stencils {
                    offsets: vec![0, 1],
                    indices: vec![0],
                    weights: vec![weight],
                },
            })
        };
        store
            .set_texcoord(a, Some(st(vec![[1.0; 2]; 7], refined(1.0))))
            .unwrap();
        assert!(store.take_moved().layout);
        store
            .set_texcoord(a, Some(st(vec![[2.0; 2]; 7], refined(1.0))))
            .unwrap();
        let moved = store.take_moved();
        assert_eq!(
            (moved.layout, &moved.texcoord),
            (false, &BTreeSet::from([a]))
        );
        store
            .set_texcoord(a, Some(st(vec![[2.0; 2]; 7], refined(0.5))))
            .unwrap();
        assert!(store.take_moved().layout);
        store.set_texcoord(a, None).unwrap();
        assert!(store.take_moved().layout);

        // Changing the point count moves the layout, not the points.
        store.set_points(a, vec![[1.0; 3]; 5]).unwrap();
        let moved = store.take_moved();
        assert!(moved.layout && moved.points.is_empty());
        store.remove_mesh(a).unwrap();
        assert!(store.take_moved().layout);
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

    /// A mesh with one triangle and one quad: 3 triangles.
    fn drawn(store: &mut Store, path: &str, instancing: Option<Instancing>) -> u64 {
        let id = store.insert_mesh(path.into());
        let topology = Topology {
            scheme: Scheme::None,
            face_vertex_counts: vec![3, 4],
            face_vertex_indices: vec![0, 1, 2, 0, 1, 2, 3],
            left_handed: false,
            holes: vec![],
            subsets: vec![Subset {
                material: "/mtl/red".into(),
                faces: vec![1],
            }],
            refined: None,
        };
        store.set_topology(id, topology).unwrap();
        store.set_points(id, vec![[0.0; 3]; 4]).unwrap();
        store.set_visible(id, true).unwrap();
        store.set_render_tag(id, "geometry".into()).unwrap();
        store.set_material(id, "/mtl/blue".into()).unwrap();
        store.set_instancing(id, instancing).unwrap();
        id
    }

    #[test]
    fn census_repeats_nested_instances() {
        let mut store = Store::new();
        let plain = drawn(&mut store, "/plain", None);
        let at = |instancer: &str, indices: Vec<i32>| {
            let instancer = instancer.into();
            Some(Instancing { instancer, indices })
        };
        // Three instances of an outer instancer, two of them its inner one,
        // which draws its prototype at four instances: eight in all.
        let outer = store.insert_instancer("/outer".into());
        store.set_instancer_parent(outer, None).unwrap();
        let inner = store.insert_instancer("/inner".into());
        store
            .set_instancer_parent(inner, at("/outer", vec![0, 2]))
            .unwrap();
        drawn(&mut store, "/leaf", at("/inner", vec![0, 1, 2, 3]));
        store.insert_counted("sphereLight".into(), "/key".into());
        store.insert_counted("sphereLight".into(), "/fill".into());

        let census = store.census().unwrap();
        assert_eq!(
            (census.meshes, census.mesh_instances),
            (2, 1 + 8),
            "{census:?}"
        );
        assert_eq!(
            (census.triangles, census.prototype_triangles),
            (3 * 9, 3 * 2)
        );
        assert_eq!(census.geometry_bytes, 2 * (7 * 4 + 4 * 12));
        // Nothing is refined, so every triangle is drawn as it is.
        assert_eq!(
            (census.refined_triangles, census.refined_prototype_triangles),
            (3 * 9, 3 * 2)
        );
        // Refined into three quads, /plain draws six triangles, not three.
        let mut topology = store.meshes()[&plain].topology.clone().unwrap();
        topology.scheme = Scheme::CatmullClark;
        topology.refined = Some(Refined {
            face_vertex_counts: vec![4; 3],
            face_vertex_indices: vec![0; 12],
            holes: vec![],
            stencils: Stencils {
                offsets: vec![0, 1],
                indices: vec![0],
                weights: vec![1.0],
            },
        });
        store.set_topology(plain, topology).unwrap();
        let census = store.census().unwrap();
        assert_eq!(
            (census.refined_triangles, census.refined_prototype_triangles),
            (3 * 8 + 6, 3 + 6)
        );
        assert_eq!(census.triangles, 3 * 9);
        assert_eq!(census.refined_bytes, 4 * (3 + 12 + 2 + 1 + 1));
        assert_eq!((census.materials, census.instancers), (2, 2));
        assert_eq!(census.counted["sphereLight"], 2);

        // Hidden and proxy meshes are not drawn.
        let hidden = drawn(&mut store, "/hidden", None);
        store.set_visible(hidden, false).unwrap();
        let proxy = drawn(&mut store, "/proxy", None);
        store.set_render_tag(proxy, "proxy".into()).unwrap();
        assert_eq!(store.census().unwrap().mesh_instances, 9);

        // A prototype of an instancer the store does not hold is an error.
        drawn(&mut store, "/lost", at("/missing", vec![0]));
        let error = store.census().unwrap_err();
        assert!(error.contains("/missing"), "{error}");
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
                    scheme: Scheme::Loop,
                    face_vertex_counts: vec![3],
                    face_vertex_indices: vec![0, 1, 2],
                    left_handed: true,
                    holes: vec![0],
                    subsets: vec![Subset {
                        material: "/mtl/red".into(),
                        faces: vec![0],
                    }],
                    // The triangle split in four, its corners kept and its
                    // edges halved.
                    refined: Some(Refined {
                        face_vertex_counts: vec![3; 4],
                        face_vertex_indices: vec![0, 3, 5, 3, 1, 4, 5, 4, 2, 3, 4, 5],
                        holes: vec![0, 1, 2, 3],
                        stencils: Stencils {
                            offsets: vec![0, 1, 2, 3, 5, 7, 9],
                            indices: vec![0, 1, 2, 0, 1, 1, 2, 2, 0],
                            weights: vec![1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        },
                    }),
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
        store
            .set_texcoord(
                b,
                Some(Texcoord {
                    name: "st".into(),
                    interpolation: Interpolation::Vertex,
                    values: vec![[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]],
                    // Refined with the points: the same stencils.
                    refined: Some(RefinedTexcoord {
                        face_vertex_indices: vec![0, 3, 5, 3, 1, 4, 5, 4, 2, 3, 4, 5],
                        stencils: Stencils {
                            offsets: vec![0, 1, 2, 3, 5, 7, 9],
                            indices: vec![0, 1, 2, 0, 1, 1, 2, 2, 0],
                            weights: vec![1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        },
                    }),
                }),
            )
            .unwrap();
        let mut moved = [0.0; 16];
        for i in [0, 5, 10, 15] {
            moved[i] = 1.0;
        }
        moved[12] = 2.0;
        store.set_transform(b, moved).unwrap();
        store.set_material(b, "/mtl/blue".into()).unwrap();
        store
            .set_categories(b, vec!["group_0".into(), "group_2".into()])
            .unwrap();
        store.set_visible(a, false).unwrap();
        store.set_render_tag(a, "proxy".into()).unwrap();
        let instancing = Instancing {
            instancer: "/world/i".into(),
            indices: vec![1, 0],
        };
        store.set_instancing(a, Some(instancing)).unwrap();
        let i = store.insert_instancer("/world/i".into());
        let parent = Instancing {
            instancer: "/world/j".into(),
            indices: vec![2],
        };
        store.set_instancer_parent(i, Some(parent)).unwrap();
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
  material not synced
  categories not synced
  topology not synced
  points not synced
  normals not synced
  texcoord not synced
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
  material \"/mtl/blue\"
  categories [\"group_0\", \"group_2\"]
  topology loop leftHanded
    face_vertex_counts 3
    face_vertex_indices 0 1 2
    holes 0
    subset \"/mtl/red\" faces 0
    refined faces 4 points 6 holes 0 1 2 3
  points 3
    0 0 0
    0.5 -0 0
    0 0.0000001 3
  normals uniform 1
    0 0 1
  texcoord st vertex 3 refined values 6
    0 0
    0.5 0
    0 1
curves /world/c
  bytes pulled 68
instancer /world/i
  parent /world/j indices 2
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
