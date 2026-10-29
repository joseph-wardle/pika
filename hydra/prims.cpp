#include "prims.h"
#include "report.h"

#include <pxr/base/gf/quath.h>
#include <pxr/base/tf/enum.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/pxOsd/refinerFactory.h>
#include <pxr/imaging/pxOsd/tokens.h>

#include <opensubdiv/far/stencilTableFactory.h>

#include <memory>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

// An empty VtArray's data may be null, so this casts rather than dereferences.
const float *xyz(VtVec3fArray const &values) {
    return reinterpret_cast<const float *>(values.cdata());
}

// The level every subdivision surface is refined to. It is part of Pika's
// look, so the host's complexity setting (HdDisplayStyle's refineLevel) does
// not change it.
constexpr int refine_level = 1;

// The arrays a pika_refined points to.
struct Refined {
    std::vector<int32_t> counts, indices, holes;
    std::vector<uint32_t> offsets;
    std::vector<int32_t> stencil_indices;
    std::vector<float> weights;
};

// Refines `topology` uniformly with OpenSubdiv, through the factory Storm
// uses, so USD's subdivision tags read as they do there. False when
// the topology is invalid, which OpenSubdiv does not check, or OpenSubdiv
// refuses it (a Loop mesh with a quad, say).
bool refine(HdMeshTopology const &topology, Refined &out) {
    using namespace OpenSubdiv;
    if (!topology.GetPxOsdMeshTopology().Validate())
        return false;
    PxOsdTopologyRefinerSharedPtr refiner =
        PxOsdRefinerFactory::Create(topology.GetPxOsdMeshTopology());
    if (!refiner)
        return false;
    refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(refine_level));
    Far::StencilTableFactory::Options options;
    options.generateIntermediateLevels = false;
    std::unique_ptr<Far::StencilTable const> table(
        Far::StencilTableFactory::Create(*refiner, options));
    Far::TopologyLevel const &level = refiner->GetLevel(refine_level);
    if (!table || table->GetNumStencils() != level.GetNumVertices())
        return false;
    for (int f = 0; f < level.GetNumFaces(); f++) {
        Far::ConstIndexArray face = level.GetFaceVertices(f);
        out.counts.push_back(face.size());
        out.indices.insert(out.indices.end(), face.begin(), face.end());
        if (level.IsFaceHole(f))
            out.holes.push_back(f);
    }
    // The table's weight array can be longer than its index array, so each
    // stencil is read through its own size.
    for (int i = 0; i < table->GetNumStencils(); i++) {
        Far::Stencil stencil = table->GetStencil(i);
        out.offsets.push_back(out.stencil_indices.size());
        Far::Index const *points = stencil.GetVertexIndices();
        float const *weights = stencil.GetWeights();
        out.stencil_indices.insert(out.stencil_indices.end(), points, points + stencil.GetSize());
        out.weights.insert(out.weights.end(), weights, weights + stencil.GetSize());
    }
    out.offsets.push_back(out.stencil_indices.size());
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
    if (bits & HdChangeTracker::DirtyMaterialId)
        _pika.report(pika_mesh_material(renderer, _id, delegate->GetMaterialId(GetId()).GetText()),
                     path + " material");
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
                refined.offsets.data(),
                refined.offsets.size() - 1,
                refined.stencil_indices.data(),
                refined.weights.data(),
                refined.weights.size(),
            };
        else
            warn(path + ": OpenSubdiv cannot refine it; its cage is drawn");
    }
    VtIntArray const &counts = topology.GetFaceVertexCounts();
    VtIntArray const &indices = topology.GetFaceVertexIndices();
    VtIntArray const &holes = topology.GetHoleIndices();
    std::vector<pika_subset> subsets;
    for (HdGeomSubset const &s : topology.GetGeomSubsets())
        subsets.push_back({s.materialId.GetText(), s.indices.cdata(), s.indices.size()});
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
        refined.offsets.empty() ? nullptr : &r,
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

void PikaMesh::_SyncNormals(HdSceneDelegate *delegate, pika_renderer *renderer) {
    // The interpolations a mesh primvar can have.
    static const HdInterpolation interpolations[] = {
        HdInterpolationConstant, HdInterpolationUniform, HdInterpolationVarying,
        HdInterpolationVertex,   HdInterpolationFaceVarying,
    };
    std::string const &path = GetId().GetString();
    for (HdInterpolation interpolation : interpolations) {
        for (HdPrimvarDescriptor const &primvar :
             GetPrimvarDescriptors(delegate, interpolation)) {
            if (primvar.name != HdTokens->normals)
                continue;
            VtValue value = GetNormals(delegate);
            if (!value.IsHolding<VtVec3fArray>()) {
                fail(path + ": normals are " + value.GetTypeName() +
                     "; Pika reads float3 normals");
                return;
            }
            VtVec3fArray const &normals = value.UncheckedGet<VtVec3fArray>();
            // Hydra's display name for an interpolation is USD's token.
            std::string token = TfEnum::GetDisplayName(interpolation);
            _pika.report(pika_mesh_normals(renderer, _id, token.c_str(), xyz(normals),
                                           normals.size()),
                         path + " normals");
            return;
        }
    }
    _pika.report(pika_mesh_no_normals(renderer, _id), path + " normals");
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
