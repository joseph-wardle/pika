// Each prim copies its dirty fields into the store and keeps only its store id.
#pragma once

#include "report.h"

#include <pxr/imaging/hd/basisCurves.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/light.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/mesh.h>
#include <pxr/imaging/hd/sprim.h>

#include <optional>

PXR_NAMESPACE_OPEN_SCOPE

class PikaMesh final : public HdMesh {
public:
    PikaMesh(SdfPath const &id, Pika &pika);
    ~PikaMesh() override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;
    void Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirtyBits,
              TfToken const &) override;
    void UpdateRenderTag(HdSceneDelegate *delegate, HdRenderParam *renderParam) override;
    void SyncTexcoord(HdSceneDelegate *delegate, pika_renderer *renderer);
    bool Binds(SdfPath const &material) const;

protected:
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override {
        return bits;
    }
    void _InitRepr(TfToken const &, HdDirtyBits *) override {}

private:
    void _SyncTopology(HdSceneDelegate *delegate, pika_renderer *renderer);
    void _SyncPoints(HdSceneDelegate *delegate, pika_renderer *renderer);
    void _SyncNormals(HdSceneDelegate *delegate, pika_renderer *renderer);
    bool _FindPrimvar(HdSceneDelegate *delegate, TfToken const &name,
                      HdInterpolation &interpolation) const;
    void _SyncInstancing(HdSceneDelegate *delegate, pika_renderer *renderer);

    Pika &_pika;
    std::vector<SdfPath> _subset_materials;
    struct Texcoord {
        std::string name;
        HdInterpolation interpolation = HdInterpolationConstant;
        VtVec2fArray values;
    };
    std::optional<Texcoord> _texcoord;
    uint64_t _id = 0;
};

// Hydra syncs an instancer only when a prototype it draws asks it to.
class PikaInstancer final : public HdInstancer {
public:
    PikaInstancer(HdSceneDelegate *delegate, SdfPath const &id, Pika &pika);
    ~PikaInstancer() override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;
    void Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirtyBits) override;

private:
    void _SyncPrimvars(HdSceneDelegate *delegate, pika_renderer *renderer);

    Pika &_pika;
    uint64_t _id = 0;
};

// Copies its whole network whenever Hydra marks it dirty. Pika finds what changed.
class PikaMaterial final : public HdMaterial {
public:
    PikaMaterial(SdfPath const &id, Pika &pika);
    ~PikaMaterial() override;

    HdDirtyBits GetInitialDirtyBitsMask() const override { return AllDirty; }
    void Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirtyBits) override;

private:
    Pika &_pika;
    uint64_t _id = 0;
};

class PikaLight final : public HdLight {
public:
    PikaLight(TfToken const &type, SdfPath const &id, Pika &pika);
    ~PikaLight() override;

    HdDirtyBits GetInitialDirtyBitsMask() const override {
        return DirtyTransform | DirtyParams | DirtyCollection;
    }
    void Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirtyBits) override;

private:
    TfToken _type;
    Pika &_pika;
    uint64_t _id = 0;
};

class PikaCurves final : public HdBasisCurves {
public:
    PikaCurves(SdfPath const &id, Pika &pika);
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
    Pika &_pika;
    uint64_t _id = 0;
};

class PikaCount {
public:
    PikaCount(TfToken const &type, SdfPath const &id, Pika &pika);
    ~PikaCount();
    PikaCount(PikaCount const &) = delete;
    PikaCount &operator=(PikaCount const &) = delete;

private:
    Pika &_pika;
    uint64_t _id = 0;
    SdfPath _path;
};

class PikaCountedRprim final : public HdRprim {
public:
    PikaCountedRprim(TfToken const &type, SdfPath const &id, Pika &pika)
        : HdRprim(id), _count(type, id, pika) {}

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

class PikaCountedSprim final : public HdSprim {
public:
    PikaCountedSprim(TfToken const &type, SdfPath const &id, Pika &pika)
        : HdSprim(id), _count(type, id, pika) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override { return HdChangeTracker::Clean; }
    void Sync(HdSceneDelegate *, HdRenderParam *, HdDirtyBits *dirtyBits) override {
        *dirtyBits = HdChangeTracker::Clean;
    }

private:
    PikaCount _count;
};

PXR_NAMESPACE_CLOSE_SCOPE
