#include "prims.h"
#include "report.h"

#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/enum.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/pxOsd/refinerFactory.h>
#include <pxr/imaging/pxOsd/tokens.h>
#include <pxr/usd/sdf/assetPath.h>

#include <opensubdiv/far/stencilTableFactory.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

// An empty VtArray's data may be null, so this casts rather than dereferences.
const float *xyz(VtVec3fArray const &values) {
    return reinterpret_cast<const float *>(values.cdata());
}

const float *xy(VtVec2fArray const &values) {
    return reinterpret_cast<const float *>(values.cdata());
}

// The level every subdivision surface is refined to. It is part of Pika's
// look, so the host's complexity setting (HdDisplayStyle's refineLevel) does
// not change it.
constexpr int refine_level = 1;

// The arrays a pika_stencils points to.
struct Stencils {
    std::vector<uint32_t> offsets;
    std::vector<int32_t> indices;
    std::vector<float> weights;

    pika_stencils view() const {
        return {offsets.data(), offsets.size() - 1, indices.data(), weights.data(), weights.size()};
    }
};

// The arrays a pika_refined points to.
struct Refined {
    std::vector<int32_t> counts, indices, holes;
    Stencils stencils;
};

// The arrays a pika_refined_texcoord points to, and the values its stencils
// are over.
struct RefinedTexcoord {
    std::vector<GfVec2f> values;
    std::vector<int32_t> indices;
    Stencils stencils;
};

// Refines `topology` uniformly with OpenSubdiv, through the factory Storm
// uses, so USD's subdivision tags read as they do there, with a face-varying
// channel per topology in `fvar`. Null when the topology is invalid, which
// OpenSubdiv does not check, or OpenSubdiv refuses it (a Loop mesh with a
// quad, say).
PxOsdTopologyRefinerSharedPtr refine(HdMeshTopology const &topology,
                                     std::vector<VtIntArray> const &fvar) {
    using namespace OpenSubdiv;
    if (!topology.GetPxOsdMeshTopology().Validate())
        return nullptr;
    PxOsdTopologyRefinerSharedPtr refiner =
        PxOsdRefinerFactory::Create(topology.GetPxOsdMeshTopology(), fvar);
    if (refiner)
        refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(refine_level));
    return refiner;
}

// The stencils of the refined level's `count` values interpolated in `mode`.
// False when OpenSubdiv's table does not cover them.
bool stencils(OpenSubdiv::Far::TopologyRefiner const &refiner,
              OpenSubdiv::Far::StencilTableFactory::Mode mode, int count, Stencils &out) {
    using namespace OpenSubdiv;
    Far::StencilTableFactory::Options options;
    options.generateIntermediateLevels = false;
    options.interpolationMode = mode;
    std::unique_ptr<Far::StencilTable const> table(
        Far::StencilTableFactory::Create(refiner, options));
    if (!table || table->GetNumStencils() != count)
        return false;
    // The table's weight array can be longer than its index array, so each
    // stencil is read through its own size.
    for (int i = 0; i < count; i++) {
        Far::Stencil stencil = table->GetStencil(i);
        out.offsets.push_back(out.indices.size());
        Far::Index const *indices = stencil.GetVertexIndices();
        float const *weights = stencil.GetWeights();
        out.indices.insert(out.indices.end(), indices, indices + stencil.GetSize());
        out.weights.insert(out.weights.end(), weights, weights + stencil.GetSize());
    }
    out.offsets.push_back(out.indices.size());
    return true;
}

// The refined faces and the refined points' stencils of `topology`. False
// when OpenSubdiv refuses it.
bool refine(HdMeshTopology const &topology, Refined &out) {
    using namespace OpenSubdiv;
    PxOsdTopologyRefinerSharedPtr refiner = refine(topology, {});
    if (!refiner)
        return false;
    Far::TopologyLevel const &level = refiner->GetLevel(refine_level);
    if (!stencils(*refiner, Far::StencilTableFactory::INTERPOLATE_VERTEX, level.GetNumVertices(),
                  out.stencils))
        return false;
    for (int f = 0; f < level.GetNumFaces(); f++) {
        Far::ConstIndexArray face = level.GetFaceVertices(f);
        out.counts.push_back(face.size());
        out.indices.insert(out.indices.end(), face.begin(), face.end());
        if (level.IsFaceHole(f))
            out.holes.push_back(f);
    }
    return true;
}

struct CornerHash {
    size_t operator()(std::array<uint32_t, 3> const &key) const {
        size_t h = 0;
        for (uint32_t k : key)
            h ^= std::hash<uint32_t>()(k) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// Welds a face-varying primvar's values by point and value (bitwise): the
// distinct values, and the one each corner reads, a face-varying topology
// for OpenSubdiv. It makes the primvar continuous across an edge whose
// faces agree on it and a seam where they do not, the reading of Houdini's
// subdivision, Karma and Blender; Storm keeps every corner apart, which
// interpolates each cage face on its own.
void weld(VtIntArray const &corners, VtVec2fArray const &values, std::vector<GfVec2f> &welded,
          VtIntArray &indices) {
    std::unordered_map<std::array<uint32_t, 3>, int32_t, CornerHash> seen;
    for (size_t c = 0; c < corners.size(); c++) {
        std::array<uint32_t, 3> key;
        key[0] = corners[c];
        memcpy(&key[1], values[c].data(), 2 * sizeof(float));
        auto [it, inserted] = seen.try_emplace(key, welded.size());
        if (inserted)
            welded.push_back(values[c]);
        indices.push_back(it->second);
    }
}

// Refines a subdivided mesh's texcoord with its faces, so each refined
// corner reads a refined value: a vertex or varying texcoord through the
// points' stencils, a face-varying one through its welded values' under the
// mesh's faceVaryingLinearInterpolation. False when OpenSubdiv refuses the
// mesh, as it then refused its topology.
bool refine_texcoord(HdMeshTopology const &topology, HdInterpolation interpolation,
                     VtVec2fArray const &values, RefinedTexcoord &out) {
    using namespace OpenSubdiv;
    bool facevarying = interpolation == HdInterpolationFaceVarying;
    std::vector<VtIntArray> fvar;
    if (facevarying) {
        fvar.emplace_back();
        weld(topology.GetFaceVertexIndices(), values, out.values, fvar.back());
    } else {
        out.values.assign(values.begin(), values.end());
    }
    PxOsdTopologyRefinerSharedPtr refiner = refine(topology, fvar);
    if (!refiner)
        return false;
    Far::TopologyLevel const &level = refiner->GetLevel(refine_level);
    Far::StencilTableFactory::Mode mode =
        facevarying                               ? Far::StencilTableFactory::INTERPOLATE_FACE_VARYING
        : interpolation == HdInterpolationVertex ? Far::StencilTableFactory::INTERPOLATE_VERTEX
                                                 : Far::StencilTableFactory::INTERPOLATE_VARYING;
    int count = facevarying ? level.GetNumFVarValues() : level.GetNumVertices();
    if (!stencils(*refiner, mode, count, out.stencils))
        return false;
    for (int f = 0; f < level.GetNumFaces(); f++) {
        Far::ConstIndexArray face = facevarying ? level.GetFaceFVarValues(f) : level.GetFaceVertices(f);
        out.indices.insert(out.indices.end(), face.begin(), face.end());
    }
    return true;
}

} // namespace

PikaMesh::PikaMesh(SdfPath const &id, Pika &pika) : HdMesh(id), _pika(pika) {
    if (pika_renderer *renderer = _pika.renderer())
        _pika.report(pika_mesh_new(renderer, id.GetText(), &_id), "adding mesh " + id.GetString());
}

PikaMesh::~PikaMesh() {
    if (pika_renderer *renderer = _pika.renderer())
        _pika.report(pika_mesh_free(renderer, _id), "removing mesh " + GetId().GetString());
}

HdDirtyBits PikaMesh::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::DirtyTopology | HdChangeTracker::DirtyPoints |
           HdChangeTracker::DirtyNormals | HdChangeTracker::DirtyPrimvar |
           HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility |
           HdChangeTracker::DirtyMaterialId | HdChangeTracker::DirtyInstancer |
           HdChangeTracker::DirtyInstanceIndex | HdChangeTracker::DirtySubdivTags;
}

void PikaMesh::Sync(HdSceneDelegate *delegate, HdRenderParam *,
                    HdDirtyBits *dirtyBits, TfToken const &) {
    // Hydra reads IsVisible() to skip syncing invisible prims; their other
    // dirty bits wait until they are visible again.
    _UpdateVisibility(delegate, dirtyBits);
    _UpdateInstancer(delegate, dirtyBits);
    HdDirtyBits bits = *dirtyBits;
    *dirtyBits = HdChangeTracker::Clean;
    pika_renderer *renderer = _pika.renderer();
    if (!renderer)
        return;
    HdInstancer::_SyncInstancerAndParents(delegate->GetRenderIndex(), GetInstancerId());

    std::string const &path = GetId().GetString();
    if (bits & HdChangeTracker::DirtyVisibility)
        _pika.report(pika_mesh_visible(renderer, _id, IsVisible()), path + " visibility");
    if (bits & (HdChangeTracker::DirtyTopology | HdChangeTracker::DirtySubdivTags))
        _SyncTopology(delegate, renderer);
    if (bits & HdChangeTracker::DirtyPoints)
        _SyncPoints(delegate, renderer);
    // Normals have their own dirty bit, but an edit can also arrive as a
    // general primvar change.
    if (bits & (HdChangeTracker::DirtyNormals | HdChangeTracker::DirtyPrimvar))
        _SyncNormals(delegate, renderer);
    if (bits & HdChangeTracker::DirtyTransform) {
        GfMatrix4d matrix = delegate->GetTransform(GetId());
        _pika.report(pika_mesh_transform(renderer, _id, matrix.data()), path + " transform");
    }
    if (bits & HdChangeTracker::DirtyMaterialId) {
        SdfPath material = delegate->GetMaterialId(GetId());
        // Kept on the rprim for PikaMaterial::Sync to find the meshes bound
        // to it.
        SetMaterialId(material);
        _pika.report(pika_mesh_material(renderer, _id, material.GetText()), path + " material");
    }
    // The material decides which primvar is fetched, so it goes first, and
    // the topology how it is refined.
    if (bits & (HdChangeTracker::DirtyPrimvar | HdChangeTracker::DirtyMaterialId |
                HdChangeTracker::DirtyTopology | HdChangeTracker::DirtySubdivTags))
        SyncTexcoord(delegate, renderer);
    if (bits & (HdChangeTracker::DirtyInstancer | HdChangeTracker::DirtyInstanceIndex))
        _SyncInstancing(delegate, renderer);
}

// Hydra calls this for every rprim with a dirty render tag, before it decides
// which ones to sync.
void PikaMesh::UpdateRenderTag(HdSceneDelegate *delegate, HdRenderParam *renderParam) {
    HdMesh::UpdateRenderTag(delegate, renderParam);
    if (pika_renderer *renderer = _pika.renderer())
        _pika.report(pika_mesh_render_tag(renderer, _id, GetRenderTag().GetText()),
                     GetId().GetString() + " render tag");
}

void PikaMesh::_SyncTopology(HdSceneDelegate *delegate, pika_renderer *renderer) {
    HdMeshTopology topology = GetMeshTopology(delegate);
    std::string const &path = GetId().GetString();
    _texcoord.reset();
    Refined refined;
    pika_refined r = {};
    if (topology.GetScheme() != PxOsdOpenSubdivTokens->none) {
        topology.SetSubdivTags(GetSubdivTags(delegate));
        if (refine(topology, refined))
            r = {
                refined.counts.data(),
                refined.counts.size(),
                refined.indices.data(),
                refined.indices.size(),
                refined.holes.data(),
                refined.holes.size(),
                refined.stencils.view(),
            };
        else
            warn(path + ": OpenSubdiv cannot refine it; its cage is drawn");
    }
    VtIntArray const &counts = topology.GetFaceVertexCounts();
    VtIntArray const &indices = topology.GetFaceVertexIndices();
    VtIntArray const &holes = topology.GetHoleIndices();
    std::vector<pika_subset> subsets;
    _subset_materials.clear();
    for (HdGeomSubset const &s : topology.GetGeomSubsets()) {
        subsets.push_back({s.materialId.GetText(), s.indices.cdata(), s.indices.size()});
        _subset_materials.push_back(s.materialId);
    }
    pika_topology t = {
        topology.GetScheme().GetText(),
        counts.cdata(),
        counts.size(),
        indices.cdata(),
        indices.size(),
        topology.GetOrientation() == PxOsdOpenSubdivTokens->leftHanded,
        holes.cdata(),
        holes.size(),
        subsets.data(),
        subsets.size(),
        refined.stencils.offsets.empty() ? nullptr : &r,
    };
    _pika.report(pika_mesh_topology(renderer, _id, &t), path + " topology");
}

void PikaMesh::_SyncPoints(HdSceneDelegate *delegate, pika_renderer *renderer) {
    VtValue value = GetPoints(delegate);
    if (!value.IsHolding<VtVec3fArray>()) {
        fail(GetId().GetString() + ": points are " + value.GetTypeName() +
             "; Pika reads float3 points");
        return;
    }
    VtVec3fArray const &points = value.UncheckedGet<VtVec3fArray>();
    _pika.report(pika_mesh_points(renderer, _id, xyz(points), points.size()),
           GetId().GetString() + " points");
}

bool PikaMesh::Binds(SdfPath const &material) const {
    return GetMaterialId() == material ||
           std::find(_subset_materials.begin(), _subset_materials.end(), material) !=
               _subset_materials.end();
}

bool PikaMesh::_FindPrimvar(HdSceneDelegate *delegate, TfToken const &name,
                            HdInterpolation &interpolation) const {
    // The interpolations a mesh primvar can have.
    static const HdInterpolation interpolations[] = {
        HdInterpolationConstant, HdInterpolationUniform, HdInterpolationVarying,
        HdInterpolationVertex,   HdInterpolationFaceVarying,
    };
    for (HdInterpolation candidate : interpolations) {
        for (HdPrimvarDescriptor const &primvar : GetPrimvarDescriptors(delegate, candidate)) {
            if (primvar.name == name) {
                interpolation = candidate;
                return true;
            }
        }
    }
    return false;
}

void PikaMesh::_SyncNormals(HdSceneDelegate *delegate, pika_renderer *renderer) {
    std::string const &path = GetId().GetString();
    HdInterpolation interpolation;
    if (!_FindPrimvar(delegate, HdTokens->normals, interpolation)) {
        _pika.report(pika_mesh_no_normals(renderer, _id), path + " normals");
        return;
    }
    VtValue value = GetNormals(delegate);
    if (!value.IsHolding<VtVec3fArray>()) {
        fail(path + ": normals are " + value.GetTypeName() + "; Pika reads float3 normals");
        return;
    }
    VtVec3fArray const &normals = value.UncheckedGet<VtVec3fArray>();
    // Hydra's display name for an interpolation is USD's token.
    std::string token = TfEnum::GetDisplayName(interpolation);
    _pika.report(
        pika_mesh_normals(renderer, _id, token.c_str(), xyz(normals), normals.size()),
        path + " normals");
}

// Materials sync before meshes, so the name is there on the first frame.
void PikaMesh::SyncTexcoord(HdSceneDelegate *delegate, pika_renderer *renderer) {
    std::string const &path = GetId().GetString();
    char name[256];
    if (!_pika.report(pika_mesh_texcoord_name(renderer, _id, name, sizeof name),
                      path + " texcoord"))
        return;
    Texcoord fetched;
    if (name[0] && _FindPrimvar(delegate, TfToken(name), fetched.interpolation)) {
        VtValue value = GetPrimvar(delegate, TfToken(name));
        if (!value.IsHolding<VtVec2fArray>()) {
            fail(path + ": " + name + " is " + value.GetTypeName() +
                 "; its material reads a float2");
            return;
        }
        fetched.name = name;
        fetched.values = value.UncheckedGet<VtVec2fArray>();
    }
    // Hydra's one dirty bit covers every primvar, so this is where an
    // animated primvar the materials do not read is told from the texcoord.
    if (_texcoord && _texcoord->name == fetched.name &&
        _texcoord->interpolation == fetched.interpolation && _texcoord->values == fetched.values)
        return;
    _texcoord = fetched;
    if (fetched.name.empty()) {
        _pika.report(pika_mesh_no_texcoord(renderer, _id), path + " texcoord");
        return;
    }
    HdInterpolation interpolation = fetched.interpolation;
    VtVec2fArray const &values = fetched.values;
    std::string token = TfEnum::GetDisplayName(interpolation);
    HdMeshTopology topology = GetMeshTopology(delegate);
    if (topology.GetScheme() == PxOsdOpenSubdivTokens->none) {
        _pika.report(pika_mesh_texcoord(renderer, _id, name, token.c_str(), xy(values),
                                        values.size(), nullptr),
                     path + " texcoord");
        return;
    }
    size_t expected = interpolation == HdInterpolationFaceVarying
                          ? topology.GetFaceVertexIndices().size()
                          : topology.GetNumPoints();
    if (interpolation != HdInterpolationFaceVarying && interpolation != HdInterpolationVertex &&
        interpolation != HdInterpolationVarying) {
        fail(path + ": " + name + " is " + token +
             "; Pika refines a vertex, varying or faceVarying texcoord with a subdivision surface");
        return;
    }
    if (values.size() != expected) {
        fail(path + ": " + std::to_string(values.size()) + " " + name + " values for " + token +
             " interpolation, which needs " + std::to_string(expected));
        return;
    }
    topology.SetSubdivTags(GetSubdivTags(delegate));
    RefinedTexcoord refined;
    if (!refine_texcoord(topology, interpolation, values, refined)) {
        // Its cage is drawn (_SyncTopology warned), with the texcoord as it is.
        _pika.report(pika_mesh_texcoord(renderer, _id, name, token.c_str(), xy(values),
                                        values.size(), nullptr),
                     path + " texcoord");
        return;
    }
    pika_refined_texcoord r = {refined.indices.data(), refined.indices.size(),
                               refined.stencils.view()};
    _pika.report(pika_mesh_texcoord(renderer, _id, name, token.c_str(),
                                    reinterpret_cast<const float *>(refined.values.data()),
                                    refined.values.size(), &r),
                 path + " texcoord");
}

void PikaMesh::_SyncInstancing(HdSceneDelegate *delegate, pika_renderer *renderer) {
    SdfPath const &instancer = GetInstancerId();
    VtIntArray indices;
    if (!instancer.IsEmpty())
        indices = delegate->GetInstanceIndices(instancer, GetId());
    _pika.report(pika_mesh_instancer(renderer, _id, instancer.GetText(), indices.cdata(),
                                     indices.size()),
                 GetId().GetString() + " instancer");
}

PikaInstancer::PikaInstancer(HdSceneDelegate *delegate, SdfPath const &id, Pika &pika)
    : HdInstancer(delegate, id), _pika(pika) {
    if (pika_renderer *renderer = _pika.renderer())
        _pika.report(pika_instancer_new(renderer, id.GetText(), &_id),
                     "adding instancer " + id.GetString());
}

PikaInstancer::~PikaInstancer() {
    if (pika_renderer *renderer = _pika.renderer())
        _pika.report(pika_instancer_free(renderer, _id),
                     "removing instancer " + GetId().GetString());
}

HdDirtyBits PikaInstancer::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyPrimvar |
           HdChangeTracker::DirtyInstancer | HdChangeTracker::DirtyInstanceIndex;
}

void PikaInstancer::Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirtyBits) {
    _UpdateInstancer(delegate, dirtyBits);
    HdDirtyBits bits = *dirtyBits;
    pika_renderer *renderer = _pika.renderer();
    if (!renderer)
        return;
    std::string const &path = GetId().GetString();
    if (bits & (HdChangeTracker::DirtyInstancer | HdChangeTracker::DirtyInstanceIndex)) {
        SdfPath const &parent = GetParentId();
        VtIntArray indices;
        if (!parent.IsEmpty())
            indices = delegate->GetInstanceIndices(parent, GetId());
        _pika.report(pika_instancer_parent(renderer, _id, parent.GetText(), indices.cdata(),
                                     indices.size()),
               path + " parent");
    }
    if (bits & HdChangeTracker::DirtyTransform) {
        GfMatrix4d matrix = delegate->GetInstancerTransform(GetId());
        _pika.report(pika_instancer_transform(renderer, _id, matrix.data()), path + " transform");
    }
    if (bits & HdChangeTracker::DirtyPrimvar)
        _SyncPrimvars(delegate, renderer);
}

void PikaInstancer::_SyncPrimvars(HdSceneDelegate *delegate, pika_renderer *renderer) {
    std::string const &path = GetId().GetString();
    // Kept alive until the call: the primvars point into them.
    VtMatrix4dArray transforms;
    VtVec3fArray translations, scales;
    VtQuatfArray rotations;
    pika_instance_primvars p = {};
    for (HdPrimvarDescriptor const &primvar :
         delegate->GetPrimvarDescriptors(GetId(), HdInterpolationInstance)) {
        VtValue value = delegate->Get(GetId(), primvar.name);
        bool read = true;
        if (primvar.name == HdInstancerTokens->instanceTransforms) {
            read = value.IsHolding<VtMatrix4dArray>();
            if (read) {
                transforms = value.UncheckedGet<VtMatrix4dArray>();
                p.transforms = reinterpret_cast<const double *>(transforms.cdata());
                p.transform_count = transforms.size();
            }
        } else if (primvar.name == HdInstancerTokens->instanceTranslations) {
            read = value.IsHolding<VtVec3fArray>();
            if (read) {
                translations = value.UncheckedGet<VtVec3fArray>();
                p.translations = xyz(translations);
                p.translation_count = translations.size();
            }
        } else if (primvar.name == HdInstancerTokens->instanceRotations) {
            // USD's point instancer authors half-float orientations.
            if (value.IsHolding<VtQuathArray>()) {
                for (GfQuath const &q : value.UncheckedGet<VtQuathArray>())
                    rotations.push_back(GfQuatf(q));
            } else if (value.IsHolding<VtQuatfArray>()) {
                rotations = value.UncheckedGet<VtQuatfArray>();
            } else {
                read = false;
            }
            p.rotations = reinterpret_cast<const float *>(rotations.cdata());
            p.rotation_count = rotations.size();
        } else if (primvar.name == HdInstancerTokens->instanceScales) {
            read = value.IsHolding<VtVec3fArray>();
            if (read) {
                scales = value.UncheckedGet<VtVec3fArray>();
                p.scales = xyz(scales);
                p.scale_count = scales.size();
            }
        }
        if (!read)
            fail(path + ": " + primvar.name.GetString() + " is " + value.GetTypeName() +
                 "; Pika reads matrix4d transforms and float3 or quaternion parts");
    }
    _pika.report(pika_instancer_primvars(renderer, _id, &p), path + " primvars");
}

PikaMaterial::PikaMaterial(SdfPath const &id, Pika &pika) : HdMaterial(id), _pika(pika) {
    if (id.IsEmpty())
        return;
    if (pika_renderer *renderer = _pika.renderer())
        _pika.report(pika_material_new(renderer, id.GetText(), &_id),
                     "adding material " + id.GetString());
}

PikaMaterial::~PikaMaterial() {
    pika_renderer *renderer = _id ? _pika.renderer() : nullptr;
    if (renderer)
        _pika.report(pika_material_free(renderer, _id), "removing material " + GetId().GetString());
}

// Flattens a parameter's value into `param`; false for a type Pika has no
// use for.
static bool flatten(VtValue const &value, pika_param &param) {
    auto floats = [&](float const *data, size_t count) {
        param.kind = "float";
        param.float_count = count;
        std::copy_n(data, count, param.floats);
        return true;
    };
    if (value.IsHolding<float>())
        return floats(&value.UncheckedGet<float>(), 1);
    if (value.IsHolding<GfVec2f>())
        return floats(value.UncheckedGet<GfVec2f>().data(), 2);
    if (value.IsHolding<GfVec3f>())
        return floats(value.UncheckedGet<GfVec3f>().data(), 3);
    if (value.IsHolding<GfVec4f>())
        return floats(value.UncheckedGet<GfVec4f>().data(), 4);
    if (value.IsHolding<GfMatrix4d>())
        return floats(GfMatrix4f(value.UncheckedGet<GfMatrix4d>()).data(), 16);
    if (value.IsHolding<int>()) {
        param.kind = "int";
        param.integer = value.UncheckedGet<int>();
        return true;
    }
    if (value.IsHolding<bool>()) {
        param.kind = "bool";
        param.integer = value.UncheckedGet<bool>();
        return true;
    }
    if (value.IsHolding<TfToken>()) {
        param.kind = "text";
        param.text = value.UncheckedGet<TfToken>().GetText();
        return true;
    }
    if (value.IsHolding<std::string>()) {
        param.kind = "text";
        param.text = value.UncheckedGet<std::string>().c_str();
        return true;
    }
    if (value.IsHolding<SdfAssetPath>()) {
        param.kind = "asset";
        param.text = value.UncheckedGet<SdfAssetPath>().GetResolvedPath().c_str();
        return true;
    }
    return false;
}

void PikaMaterial::Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirtyBits) {
    HdDirtyBits bits = *dirtyBits;
    *dirtyBits = HdChangeTracker::Clean;
    pika_renderer *renderer = _pika.renderer();
    if (!renderer || !_id || !(bits & (DirtyResource | DirtyParams)))
        return;
    std::string const &path = GetId().GetString();
    VtValue resource = delegate->GetMaterialResource(GetId());
    HdMaterialNetwork2 network;
    if (resource.IsHolding<HdMaterialNetworkMap>())
        network = HdConvertToHdMaterialNetwork2(resource.UncheckedGet<HdMaterialNetworkMap>());

    // The flat arrays borrow every string from `network`, which outlives
    // the call.
    std::vector<pika_node> nodes;
    std::vector<pika_connection> connections;
    std::vector<std::vector<pika_param>> params(network.nodes.size());
    for (auto const &[nodePath, node] : network.nodes) {
        std::vector<pika_param> &ps = params[nodes.size()];
        for (auto const &[name, value] : node.parameters) {
            pika_param param = {};
            param.name = name.GetText();
            param.text = "";
            if (flatten(value, param))
                ps.push_back(param);
            else
                warn(path + ": parameter " + name.GetString() + " of " + nodePath.GetString() +
                     " is a " + value.GetTypeName() + ", which Pika cannot use");
        }
        nodes.push_back({nodePath.GetText(), node.nodeTypeId.GetText(), ps.data(), ps.size()});
        for (auto const &[input, upstream] : node.inputConnections)
            for (HdMaterialConnection2 const &c : upstream)
                connections.push_back({c.upstreamNode.GetText(), c.upstreamOutputName.GetText(),
                                       nodePath.GetText(), input.GetText()});
    }
    auto surface = network.terminals.find(HdMaterialTerminalTokens->surface);
    pika_network flat = {nodes.data(), nodes.size(), connections.data(), connections.size(),
                         surface == network.terminals.end()
                             ? ""
                             : surface->second.upstreamNode.GetText()};
    bool dirty = false;
    if (!_pika.report(pika_material_network(renderer, _id, &flat, &dirty), path + " network"))
        return;
    // Hydra tells no mesh that its material's network changed, so the
    // meshes bound to it, whole or by a subset, fetch their texcoord again
    // here. Marking them dirty instead does not reach them: with scene
    // index emulation on, the bits go through the emulation scene index,
    // which drops them on USD 25 and the custom bits too on USD 26.
    if (!dirty)
        return;
    HdRenderIndex &index = delegate->GetRenderIndex();
    for (SdfPath const &id : index.GetRprimIds()) {
        // The index hands out its rprims const; the mesh's sync is Pika's own.
        auto mesh = const_cast<PikaMesh *>(dynamic_cast<PikaMesh const *>(index.GetRprim(id)));
        if (mesh && mesh->Binds(GetId()))
            mesh->SyncTexcoord(delegate, renderer);
    }
}

PikaLight::PikaLight(SdfPath const &id, Pika &pika) : HdLight(id), _pika(pika) {
    if (id.IsEmpty())
        return;
    if (pika_renderer *renderer = _pika.renderer())
        _pika.report(pika_light_new(renderer, id.GetText(), &_id), "adding light " + id.GetString());
}

PikaLight::~PikaLight() {
    pika_renderer *renderer = _id ? _pika.renderer() : nullptr;
    if (renderer)
        _pika.report(pika_light_free(renderer, _id), "removing light " + GetId().GetString());
}

// A light's input as `T`, or UsdLux's fallback when the host hands over
// none.
template <typename T>
static T light_param(HdSceneDelegate *delegate, SdfPath const &id, TfToken const &name,
                     T fallback) {
    VtValue value = delegate->GetLightParamValue(id, name);
    return value.CanCast<T>() ? value.Cast<T>().template UncheckedGet<T>() : fallback;
}

void PikaLight::Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirtyBits) {
    HdDirtyBits bits = *dirtyBits;
    *dirtyBits = Clean;
    pika_renderer *renderer = _pika.renderer();
    if (!renderer || !_id)
        return;
    SdfPath const &id = GetId();
    if (bits & DirtyTransform) {
        GfMatrix4d matrix = delegate->GetTransform(id);
        _pika.report(pika_light_transform(renderer, _id, matrix.data()),
                     id.GetString() + " transform");
    }
    if (!(bits & DirtyParams))
        return;
    pika_light params = {};
    // DistantLight's schema overrides LightAPI's fallback intensity of 1,
    // approximating sunlight.
    params.intensity = light_param(delegate, id, HdLightTokens->intensity, 50000.0f);
    params.exposure = light_param(delegate, id, HdLightTokens->exposure, 0.0f);
    GfVec3f color = light_param(delegate, id, HdLightTokens->color, GfVec3f(1.0f));
    std::copy_n(color.data(), 3, params.color);
    params.normalize = light_param(delegate, id, HdLightTokens->normalize, false);
    params.angle = light_param(delegate, id, HdLightTokens->angle, 0.53f);
    params.visible = delegate->GetVisible(id);
    _pika.report(pika_light_params(renderer, _id, &params), id.GetString() + " params");
}

PikaCurves::PikaCurves(SdfPath const &id, Pika &pika) : HdBasisCurves(id), _pika(pika) {
    if (pika_renderer *renderer = _pika.renderer())
        _pika.report(pika_curves_new(renderer, id.GetText(), &_id),
                     "adding curves " + id.GetString());
}

PikaCurves::~PikaCurves() {
    if (pika_renderer *renderer = _pika.renderer())
        _pika.report(pika_curves_free(renderer, _id), "removing curves " + GetId().GetString());
}

HdDirtyBits PikaCurves::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::DirtyTopology | HdChangeTracker::DirtyPoints |
           HdChangeTracker::DirtyWidths | HdChangeTracker::DirtyVisibility;
}

void PikaCurves::Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirtyBits,
                      TfToken const &) {
    _UpdateVisibility(delegate, dirtyBits);
    HdDirtyBits bits = *dirtyBits;
    *dirtyBits = HdChangeTracker::Clean;
    pika_renderer *renderer = _pika.renderer();
    if (!renderer)
        return;
    size_t bytes = 0;
    if (bits & HdChangeTracker::DirtyTopology) {
        HdBasisCurvesTopology topology = GetBasisCurvesTopology(delegate);
        bytes += sizeof(int) *
                 (topology.GetCurveVertexCounts().size() + topology.GetCurveIndices().size());
    }
    // Points are float3 and widths float; the sizes are what Hydra handed over.
    if (bits & HdChangeTracker::DirtyPoints)
        bytes += GetPrimvar(delegate, HdTokens->points).GetArraySize() * sizeof(GfVec3f);
    if (bits & HdChangeTracker::DirtyWidths)
        bytes += delegate->Get(GetId(), HdTokens->widths).GetArraySize() * sizeof(float);
    _pika.report(pika_curves_pulled(renderer, _id, bytes), GetId().GetString() + " curves");
}

PikaCount::PikaCount(TfToken const &type, SdfPath const &id, Pika &pika)
    : _pika(pika), _path(id) {
    // Hydra's fallback prims have no path and stand for nothing in the scene.
    if (id.IsEmpty())
        return;
    if (pika_renderer *renderer = _pika.renderer())
        _pika.report(pika_counted_new(renderer, type.GetText(), id.GetText(), &_id),
                     "adding " + type.GetString() + " " + id.GetString());
}

PikaCount::~PikaCount() {
    // Store ids start at 1: a fallback prim was never added.
    pika_renderer *renderer = _id ? _pika.renderer() : nullptr;
    if (renderer)
        _pika.report(pika_counted_free(renderer, _id), "removing " + _path.GetString());
}

PXR_NAMESPACE_CLOSE_SCOPE
