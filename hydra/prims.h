// The rprims Pika supports. Each copies its dirty fields into the store and
// keeps nothing but its store id.
#pragma once

#include "pika.h"

#include <pxr/imaging/hd/basisCurves.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/mesh.h>
#include <pxr/imaging/hd/sprim.h>

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
    void _SyncInstancing(HdSceneDelegate *delegate);

    pika_renderer *_renderer = nullptr;
    uint64_t _id = 0;
};

// Hydra syncs an instancer only when a prototype it draws asks it to.
class PikaInstancer final : public HdInstancer {
public:
    PikaInstancer(HdSceneDelegate *delegate, SdfPath const &id, pika_renderer *renderer);
    ~PikaInstancer() override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;
    void Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirtyBits) override;

private:
    void _SyncPrimvars(HdSceneDelegate *delegate);

    pika_renderer *_renderer = nullptr;
    uint64_t _id = 0;
};

// Basis curves are fetched and measured, then dropped.
class PikaCurves final : public HdBasisCurves {
public:
    PikaCurves(SdfPath const &id, pika_renderer *renderer);
    ~PikaCurves() override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;
    void Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirtyBits,
              TfToken const &) override;

protected:
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override {
        return bits;
    }
    void _InitRepr(TfToken const &, HdDirtyBits *) override {}

private:
    pika_renderer *_renderer = nullptr;
    uint64_t _id = 0;
};

// An entry in the store's count of a prim type, for as long as it lives.
class PikaCount {
public:
    PikaCount(TfToken const &type, SdfPath const &id, pika_renderer *renderer);
    ~PikaCount();
    PikaCount(PikaCount const &) = delete;
    PikaCount &operator=(PikaCount const &) = delete;

private:
    pika_renderer *_renderer = nullptr;
    uint64_t _id = 0;
    SdfPath _path;
};

// An rprim of a type Pika only counts. It asks Hydra for nothing.
class PikaCountedRprim final : public HdRprim {
public:
    PikaCountedRprim(TfToken const &type, SdfPath const &id, pika_renderer *renderer)
        : HdRprim(id), _count(type, id, renderer) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override { return HdChangeTracker::Clean; }
    void Sync(HdSceneDelegate *, HdRenderParam *, HdDirtyBits *dirtyBits,
              TfToken const &) override {
        *dirtyBits = HdChangeTracker::Clean;
    }
    TfTokenVector const &GetBuiltinPrimvarNames() const override {
        static const TfTokenVector none;
        return none;
    }

protected:
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override {
        return bits;
    }
    void _InitRepr(TfToken const &, HdDirtyBits *) override {}

private:
    PikaCount _count;
};

// An sprim of a type Pika only counts, such as a light or a material.
class PikaCountedSprim final : public HdSprim {
public:
    PikaCountedSprim(TfToken const &type, SdfPath const &id, pika_renderer *renderer)
        : HdSprim(id), _count(type, id, renderer) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override { return HdChangeTracker::Clean; }
    void Sync(HdSceneDelegate *, HdRenderParam *, HdDirtyBits *dirtyBits) override {
        *dirtyBits = HdChangeTracker::Clean;
    }

private:
    PikaCount _count;
};

PXR_NAMESPACE_CLOSE_SCOPE
