#include "prims.h"
#include "report.h"

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
           HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility;
}

void PikaMesh::Sync(HdSceneDelegate *delegate, HdRenderParam *,
                    HdDirtyBits *dirtyBits, TfToken const &) {
    // Hydra reads IsVisible() to skip syncing invisible prims; their other
    // dirty bits wait until they are visible again.
    _UpdateVisibility(delegate, dirtyBits);
    HdDirtyBits bits = *dirtyBits;
    *dirtyBits = HdChangeTracker::Clean;
    if (!_renderer)
        return;

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

PXR_NAMESPACE_CLOSE_SCOPE
