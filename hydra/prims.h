// The rprims Pika supports. Each copies its dirty fields into the store and
// keeps nothing but its store id.
#pragma once

#include "pika.h"

#include <pxr/imaging/hd/mesh.h>

PXR_NAMESPACE_OPEN_SCOPE

class PikaMesh final : public HdMesh {
public:
    // Without a renderer, the mesh is accepted and never synced.
    PikaMesh(SdfPath const &id, pika_renderer *renderer);
    ~PikaMesh() override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;
    void Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirtyBits,
              TfToken const &) override;
    void UpdateRenderTag(HdSceneDelegate *delegate, HdRenderParam *renderParam) override;

protected:
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override {
        return bits;
    }
    void _InitRepr(TfToken const &, HdDirtyBits *) override {}

private:
    void _SyncTopology(HdSceneDelegate *delegate);
    void _SyncPoints(HdSceneDelegate *delegate);
    void _SyncNormals(HdSceneDelegate *delegate);

    pika_renderer *_renderer = nullptr;
    uint64_t _id = 0;
};

PXR_NAMESPACE_CLOSE_SCOPE
