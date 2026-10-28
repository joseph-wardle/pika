#include "prims.h"
#include "report.h"

#include <pxr/base/gf/quath.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/pxOsd/tokens.h>

#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

// An empty VtArray's data may be null, so this casts rather than dereferences.
const float *xyz(VtVec3fArray const &values) {
    return reinterpret_cast<const float *>(values.cdata());
}

} // namespace

PikaMesh::PikaMesh(SdfPath const &id, pika_renderer *renderer) : HdMesh(id) {
    if (!renderer)
        return;
    int status = pika_mesh_new(renderer, id.GetText(), &_id);
    report(status, "adding mesh " + id.GetString());
    if (status == PIKA_OK)
        _renderer = renderer;
}

PikaMesh::~PikaMesh() {
    if (_renderer)
        report(pika_mesh_free(_renderer, _id), "removing mesh " + GetId().GetString());
}

HdDirtyBits PikaMesh::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::DirtyTopology | HdChangeTracker::DirtyPoints |
           HdChangeTracker::DirtyNormals | HdChangeTracker::DirtyPrimvar |
           HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility |
           HdChangeTracker::DirtyInstancer | HdChangeTracker::DirtyInstanceIndex;
}

void PikaMesh::Sync(HdSceneDelegate *delegate, HdRenderParam *,
                    HdDirtyBits *dirtyBits, TfToken const &) {
    // Hydra reads IsVisible() to skip syncing invisible prims; their other
    // dirty bits wait until they are visible again.
    _UpdateVisibility(delegate, dirtyBits);
    _UpdateInstancer(delegate, dirtyBits);
    HdDirtyBits bits = *dirtyBits;
    *dirtyBits = HdChangeTracker::Clean;
    if (!_renderer)
        return;
    HdInstancer::_SyncInstancerAndParents(delegate->GetRenderIndex(), GetInstancerId());

    std::string const &path = GetId().GetString();
    if (bits & HdChangeTracker::DirtyVisibility)
        report(pika_mesh_visible(_renderer, _id, IsVisible()), path + " visibility");
    if (bits & HdChangeTracker::DirtyTopology)
        _SyncTopology(delegate);
    if (bits & HdChangeTracker::DirtyPoints)
        _SyncPoints(delegate);
    // Normals have their own dirty bit, but an edit can also arrive as a
    // general primvar change.
    if (bits & (HdChangeTracker::DirtyNormals | HdChangeTracker::DirtyPrimvar))
        _SyncNormals(delegate);
    if (bits & HdChangeTracker::DirtyTransform) {
        GfMatrix4d matrix = delegate->GetTransform(GetId());
        report(pika_mesh_transform(_renderer, _id, matrix.data()), path + " transform");
    }
    if (bits & (HdChangeTracker::DirtyInstancer | HdChangeTracker::DirtyInstanceIndex))
        _SyncInstancing(delegate);
}

// Hydra calls this for every rprim with a dirty render tag, before it decides
// which ones to sync.
void PikaMesh::UpdateRenderTag(HdSceneDelegate *delegate, HdRenderParam *renderParam) {
    HdMesh::UpdateRenderTag(delegate, renderParam);
    if (_renderer)
        report(pika_mesh_render_tag(_renderer, _id, GetRenderTag().GetText()),
               GetId().GetString() + " render tag");
}

void PikaMesh::_SyncTopology(HdSceneDelegate *delegate) {
    HdMeshTopology topology = GetMeshTopology(delegate);
    VtIntArray const &counts = topology.GetFaceVertexCounts();
    VtIntArray const &indices = topology.GetFaceVertexIndices();
    std::vector<pika_subset> subsets;
    for (HdGeomSubset const &s : topology.GetGeomSubsets())
        subsets.push_back({s.materialId.GetText(), s.indices.cdata(), s.indices.size()});
    pika_topology t = {
        counts.cdata(),
        counts.size(),
        indices.cdata(),
        indices.size(),
        topology.GetOrientation() == PxOsdOpenSubdivTokens->leftHanded,
        subsets.data(),
        subsets.size(),
    };
    report(pika_mesh_topology(_renderer, _id, &t), GetId().GetString() + " topology");
}

void PikaMesh::_SyncPoints(HdSceneDelegate *delegate) {
    VtValue value = GetPoints(delegate);
    if (!value.IsHolding<VtVec3fArray>()) {
        fail(GetId().GetString() + ": points are " + value.GetTypeName() +
             "; Pika reads float3 points");
        return;
    }
    VtVec3fArray const &points = value.UncheckedGet<VtVec3fArray>();
    report(pika_mesh_points(_renderer, _id, xyz(points), points.size()),
           GetId().GetString() + " points");
}

void PikaMesh::_SyncNormals(HdSceneDelegate *delegate) {
    static const std::pair<HdInterpolation, pika_interpolation> interpolations[] = {
        {HdInterpolationConstant, PIKA_CONSTANT},
        {HdInterpolationUniform, PIKA_UNIFORM},
        {HdInterpolationVarying, PIKA_VARYING},
        {HdInterpolationVertex, PIKA_VERTEX},
        {HdInterpolationFaceVarying, PIKA_FACE_VARYING},
    };
    std::string const &path = GetId().GetString();
    for (auto [hd, pika] : interpolations) {
        for (HdPrimvarDescriptor const &primvar : GetPrimvarDescriptors(delegate, hd)) {
            if (primvar.name != HdTokens->normals)
                continue;
            VtValue value = GetNormals(delegate);
            if (!value.IsHolding<VtVec3fArray>()) {
                fail(path + ": normals are " + value.GetTypeName() +
                     "; Pika reads float3 normals");
                return;
            }
            VtVec3fArray const &normals = value.UncheckedGet<VtVec3fArray>();
            report(pika_mesh_normals(_renderer, _id, pika, xyz(normals), normals.size()),
                   path + " normals");
            return;
        }
    }
    report(pika_mesh_no_normals(_renderer, _id), path + " normals");
}

void PikaMesh::_SyncInstancing(HdSceneDelegate *delegate) {
    SdfPath const &instancer = GetInstancerId();
    VtIntArray indices;
    if (!instancer.IsEmpty())
        indices = delegate->GetInstanceIndices(instancer, GetId());
    report(pika_mesh_instancer(_renderer, _id, instancer.GetText(), indices.cdata(),
                               indices.size()),
           GetId().GetString() + " instancer");
}

PikaInstancer::PikaInstancer(HdSceneDelegate *delegate, SdfPath const &id,
                             pika_renderer *renderer)
    : HdInstancer(delegate, id) {
    if (!renderer)
        return;
    int status = pika_instancer_new(renderer, id.GetText(), &_id);
    report(status, "adding instancer " + id.GetString());
    if (status == PIKA_OK)
        _renderer = renderer;
}

PikaInstancer::~PikaInstancer() {
    if (_renderer)
        report(pika_instancer_free(_renderer, _id), "removing instancer " + GetId().GetString());
}

HdDirtyBits PikaInstancer::GetInitialDirtyBitsMask() const {
    return HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyPrimvar |
           HdChangeTracker::DirtyInstancer;
}

void PikaInstancer::Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirtyBits) {
    _UpdateInstancer(delegate, dirtyBits);
    HdDirtyBits bits = *dirtyBits;
    if (!_renderer)
        return;
    std::string const &path = GetId().GetString();
    if (bits & HdChangeTracker::DirtyInstancer)
        report(pika_instancer_parent(_renderer, _id, GetParentId().GetText()), path + " parent");
    if (bits & HdChangeTracker::DirtyTransform) {
        GfMatrix4d matrix = delegate->GetInstancerTransform(GetId());
        report(pika_instancer_transform(_renderer, _id, matrix.data()), path + " transform");
    }
    if (bits & HdChangeTracker::DirtyPrimvar)
        _SyncPrimvars(delegate);
}

void PikaInstancer::_SyncPrimvars(HdSceneDelegate *delegate) {
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
    report(pika_instancer_primvars(_renderer, _id, &p), path + " primvars");
}

PikaCurves::PikaCurves(SdfPath const &id, pika_renderer *renderer) : HdBasisCurves(id) {
    if (!renderer)
        return;
    int status = pika_curves_new(renderer, id.GetText(), &_id);
    report(status, "adding curves " + id.GetString());
    if (status == PIKA_OK)
        _renderer = renderer;
}

PikaCurves::~PikaCurves() {
    if (_renderer)
        report(pika_curves_free(_renderer, _id), "removing curves " + GetId().GetString());
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
    if (!_renderer)
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
    report(pika_curves_pulled(_renderer, _id, bytes), GetId().GetString() + " curves");
}

PikaCount::PikaCount(TfToken const &type, SdfPath const &id, pika_renderer *renderer)
    : _path(id) {
    // Hydra's fallback prims have no path and stand for nothing in the scene.
    if (!renderer || id.IsEmpty())
        return;
    int status = pika_counted_new(renderer, type.GetText(), id.GetText(), &_id);
    report(status, "adding " + type.GetString() + " " + id.GetString());
    if (status == PIKA_OK)
        _renderer = renderer;
}

PikaCount::~PikaCount() {
    if (_renderer)
        report(pika_counted_free(_renderer, _id), "removing " + _path.GetString());
}

PXR_NAMESPACE_CLOSE_SCOPE
