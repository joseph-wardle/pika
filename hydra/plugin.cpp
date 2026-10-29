// The Pika Hydra delegate. It reads what Hydra hands it and copies bytes into
// the Rust renderer (pika.h); it holds no renderer logic of its own.
//
// Meshes and instancers are synced into Pika's store, basis curves are
// fetched and measured, and other prims are counted. Pika draws the meshes
// into the color, depth and normal AOVs.

#include "buffer.h"
#include "pika.h"
#include "prims.h"
#include "report.h"

#include <pxr/imaging/cameraUtil/framing.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/resourceRegistry.h>
#include <pxr/imaging/hd/tokens.h>

#include <algorithm>
#include <string>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

// The pika_format of a render buffer, or false when Pika writes none.
bool Format(HdFormat format, pika_format *out) {
    switch (format) {
    case HdFormatFloat16Vec4: *out = PIKA_FLOAT16_VEC4; return true;
    case HdFormatFloat32Vec4: *out = PIKA_FLOAT32_VEC4; return true;
    case HdFormatFloat32: *out = PIKA_FLOAT32; return true;
    case HdFormatFloat32Vec3: *out = PIKA_FLOAT32_VEC3; return true;
    default: return false;
    }
}

class PikaRenderPass final : public HdRenderPass {
public:
    PikaRenderPass(HdRenderIndex *index, HdRprimCollection const &collection,
                   pika_renderer *renderer)
        : HdRenderPass(index, collection), _renderer(renderer) {}

protected:
    void _Execute(HdRenderPassStateSharedPtr const &state,
                  TfTokenVector const &) override {
        if (!_renderer)
            return;
        if (!state->GetCamera()) {
            fail("the render pass has no camera");
            return;
        }
        pika_frame frame = {};
        std::copy_n(state->GetWorldToViewMatrix().data(), 16, frame.world_to_view);
        std::copy_n(state->GetProjectionMatrix().data(), 16, frame.projection);
        GfVec4f clear(0.0f);
        PikaRenderBuffer *first = nullptr;
        for (HdRenderPassAovBinding const &aov : state->GetAovBindings()) {
            pika_aov *slot = aov.aovName == HdAovTokens->color    ? &frame.color
                             : aov.aovName == HdAovTokens->depth  ? &frame.depth
                             : aov.aovName == HdAovTokens->normal ? &frame.normal
                                                                  : nullptr;
            if (!slot)
                continue;
            HdRenderBuffer *bound = aov.renderBuffer;
            if (!bound)
                bound = static_cast<HdRenderBuffer *>(GetRenderIndex()->GetBprim(
                    HdPrimTypeTokens->renderBuffer, aov.renderBufferId));
            // CreateBprim makes every render buffer in this index.
            auto *buffer = static_cast<PikaRenderBuffer *>(bound);
            if (!buffer)
                continue;
            if (!Format(buffer->GetFormat(), &slot->format)) {
                fail("the " + aov.aovName.GetString() + " AOV is " +
                     TfEnum::GetName(buffer->GetFormat()) + ", which Pika does not write");
                continue;
            }
            if (first && (buffer->GetWidth() != first->GetWidth() ||
                          buffer->GetHeight() != first->GetHeight())) {
                fail("the " + aov.aovName.GetString() + " render buffer's size differs from " +
                     "the other AOVs'");
                continue;
            }
            first = buffer;
            slot->pixels = buffer->Data().data();
            slot->len = buffer->Data().size();
            if (aov.aovName == HdAovTokens->color && aov.clearValue.IsHolding<GfVec4f>())
                clear = aov.clearValue.UncheckedGet<GfVec4f>();
        }
        if (!first)
            return;
        frame.width = first->GetWidth();
        frame.height = first->GetHeight();
        std::copy_n(clear.data(), 4, frame.clear_color);

        CameraUtilFraming const &framing = state->GetFraming();
        if (framing.IsValid()) {
            GfRect2i const &window = framing.dataWindow;
            frame.window[0] = window.GetMinX();
            frame.window[1] = window.GetMinY();
            frame.window[2] = window.GetWidth();
            frame.window[3] = window.GetHeight();
        } else {
            // The deprecated viewport counts y up from the bottom row.
            GfVec4f const &viewport = state->GetViewport();
            // husk renders a dataWindowNDC by sizing the buffers to the data
            // window, placing the viewport in the full image and conforming
            // the projection to the window's own aspect. The full image's
            // size is not in the render pass, so the window cannot be framed.
            if (viewport[0] < 0 || viewport[1] < 0 || viewport[0] + viewport[2] > frame.width ||
                viewport[1] + viewport[3] > frame.height) {
                fail("the viewport lies outside the render buffers, as husk sets it for a "
                     "dataWindowNDC, which Pika does not support yet");
                return;
            }
            frame.window[0] = int(viewport[0]);
            frame.window[1] = int(frame.height - viewport[1] - viewport[3]);
            frame.window[2] = int(viewport[2]);
            frame.window[3] = int(viewport[3]);
        }
        report(pika_render(_renderer, &frame), "render");
    }

private:
    pika_renderer *_renderer;
};

class PikaRenderDelegate final : public HdRenderDelegate {
public:
    PikaRenderDelegate() { _Open(); }
    explicit PikaRenderDelegate(HdRenderSettingsMap const &settings)
        : HdRenderDelegate(settings) {
        _Open();
    }
    ~PikaRenderDelegate() override {
        if (_renderer)
            report(pika_renderer_free(_renderer), "shutting down");
    }

    // Supporting a type changes what Hydra hands over, so the counted types
    // leave out the implicit surfaces (cube, sphere...), which Hydra
    // otherwise turns into meshes, and ext computations, which would move
    // skinned points out of the meshes' primvars.
    TfTokenVector const &GetSupportedRprimTypes() const override {
        static const TfTokenVector types = {
            HdPrimTypeTokens->mesh,        HdPrimTypeTokens->basisCurves,
            HdPrimTypeTokens->points,      HdPrimTypeTokens->nurbsCurves,
            HdPrimTypeTokens->nurbsPatch,  HdPrimTypeTokens->volume,
            HdPrimTypeTokens->tetMesh,
        };
        return types;
    }
    TfTokenVector const &GetSupportedSprimTypes() const override {
        static const TfTokenVector types = [] {
            TfTokenVector t = HdLightPrimTypeTokens();
            t.push_back(HdPrimTypeTokens->lightFilter);
            t.push_back(HdPrimTypeTokens->material);
            t.push_back(HdPrimTypeTokens->camera);
            return t;
        }();
        return types;
    }
    TfTokenVector const &GetSupportedBprimTypes() const override {
        static const TfTokenVector types = {HdPrimTypeTokens->renderBuffer};
        return types;
    }

    // Hydra's default is the preview binding; Pika's look target is the
    // final one, which ALab binds separately.
    TfToken GetMaterialBindingPurpose() const override { return HdTokens->full; }

    HdResourceRegistrySharedPtr GetResourceRegistry() const override {
        return _resources;
    }

    HdAovDescriptor GetDefaultAovDescriptor(TfToken const &name) const override {
        if (name == HdAovTokens->color)
            return HdAovDescriptor(HdFormatFloat16Vec4, false,
                                   VtValue(GfVec4f(0.0f)));
        if (name == HdAovTokens->depth)
            return HdAovDescriptor(HdFormatFloat32, false, VtValue(1.0f));
        if (name == HdAovTokens->normal)
            return HdAovDescriptor(HdFormatFloat32Vec3, false,
                                   VtValue(GfVec3f(0.0f)));
        return HdAovDescriptor();
    }

    HdRenderPassSharedPtr CreateRenderPass(
        HdRenderIndex *index, HdRprimCollection const &collection) override {
        return std::make_shared<PikaRenderPass>(index, collection, _renderer);
    }

    HdInstancer *CreateInstancer(HdSceneDelegate *delegate,
                                 SdfPath const &id) override {
        return new PikaInstancer(delegate, id, _renderer);
    }
    void DestroyInstancer(HdInstancer *instancer) override { delete instancer; }

    HdRprim *CreateRprim(TfToken const &typeId, SdfPath const &id) override {
        if (typeId == HdPrimTypeTokens->mesh)
            return new PikaMesh(id, _renderer);
        if (typeId == HdPrimTypeTokens->basisCurves)
            return new PikaCurves(id, _renderer);
        return new PikaCountedRprim(typeId, id, _renderer);
    }
    void DestroyRprim(HdRprim *rprim) override { delete rprim; }

    HdSprim *CreateSprim(TfToken const &typeId, SdfPath const &id) override {
        if (typeId == HdPrimTypeTokens->camera)
            return new HdCamera(id);
        return new PikaCountedSprim(typeId, id, _renderer);
    }
    HdSprim *CreateFallbackSprim(TfToken const &typeId) override {
        return CreateSprim(typeId, SdfPath::EmptyPath());
    }
    void DestroySprim(HdSprim *sprim) override { delete sprim; }

    HdBprim *CreateBprim(TfToken const &typeId, SdfPath const &id) override {
        if (typeId == HdPrimTypeTokens->renderBuffer)
            return new PikaRenderBuffer(id);
        return nullptr;
    }
    HdBprim *CreateFallbackBprim(TfToken const &typeId) override {
        return CreateBprim(typeId, SdfPath::EmptyPath());
    }
    void DestroyBprim(HdBprim *bprim) override { delete bprim; }

    // Hydra calls Update before its prim syncs and CommitResources after.
    void Update() override {
        if (_renderer)
            report(pika_sync_begin(_renderer), "beginning a sync");
    }
    void CommitResources(HdChangeTracker *) override {
        if (_renderer)
            report(pika_sync_end(_renderer), "ending a sync");
    }

private:
    // Without a renderer the delegate still loads, and renders nothing.
    void _Open() { report(pika_renderer_new(&_renderer), "opening a GPU"); }

    pika_renderer *_renderer = nullptr;
    HdResourceRegistrySharedPtr _resources =
        std::make_shared<HdResourceRegistry>();
};

} // namespace

class PikaRendererPlugin final : public HdRendererPlugin {
public:
    HdRenderDelegate *CreateRenderDelegate() override {
        return new PikaRenderDelegate();
    }
    HdRenderDelegate *CreateRenderDelegate(
        HdRenderSettingsMap const &settings) override {
        return new PikaRenderDelegate(settings);
    }
    void DeleteRenderDelegate(HdRenderDelegate *delegate) override {
        delete delegate;
    }
    // Pika brings its own Vulkan device, so the host's GPU setting is irrelevant.
    bool IsSupported(bool) const override { return true; }
};

TF_REGISTRY_FUNCTION(TfType) {
    HdRendererPluginRegistry::Define<PikaRendererPlugin>();
}

PXR_NAMESPACE_CLOSE_SCOPE
