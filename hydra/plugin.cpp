// The Pika Hydra delegate. It reads what Hydra hands it and copies bytes into
// the Rust renderer (pika.h); it holds no renderer logic of its own.
//
// The skeleton: meshes are accepted but not synced, and the color AOV
// receives Pika's cleared frame.

#include "pika.h"

#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/mesh.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/resourceRegistry.h>
#include <pxr/imaging/hd/tokens.h>

#include <atomic>
#include <cstdio>
#include <string>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

namespace {

// husk drops every Tf diagnostic, so errors also go to stderr.
void fail(std::string const &message) {
    fprintf(stderr, "pika: error: %s\n", message.c_str());
    TF_RUNTIME_ERROR("Pika: %s", message.c_str());
}

void report(int status, const char *what) {
    if (status != PIKA_OK)
        fail(std::string(what) + ": " + pika_last_error());
}

class PikaMesh final : public HdMesh {
public:
    explicit PikaMesh(SdfPath const &id) : HdMesh(id) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override {
        return HdChangeTracker::Clean;
    }

    void Sync(HdSceneDelegate *, HdRenderParam *, HdDirtyBits *dirtyBits,
              TfToken const &) override {
        *dirtyBits = HdChangeTracker::Clean;
    }

protected:
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override {
        return bits;
    }
    void _InitRepr(TfToken const &, HdDirtyBits *) override {}
};

// The pixels live here, in host memory; Pika copies each frame in.
class PikaRenderBuffer final : public HdRenderBuffer {
public:
    explicit PikaRenderBuffer(SdfPath const &id) : HdRenderBuffer(id) {}

    bool Allocate(GfVec3i const &dimensions, HdFormat format,
                  bool multiSampled) override {
        _Deallocate();
        if (dimensions[0] <= 0 || dimensions[1] <= 0 || dimensions[2] != 1)
            return false;
        _width = dimensions[0];
        _height = dimensions[1];
        _format = format;
        _multiSampled = multiSampled;
        _data.resize(size_t(_width) * _height * HdDataSizeOfFormat(format));
        return true;
    }

    unsigned int GetWidth() const override { return _width; }
    unsigned int GetHeight() const override { return _height; }
    unsigned int GetDepth() const override { return 1; }
    HdFormat GetFormat() const override { return _format; }
    bool IsMultiSampled() const override { return _multiSampled; }

    void *Map() override {
        _mappers++;
        return _data.data();
    }
    void Unmap() override { _mappers--; }
    bool IsMapped() const override { return _mappers.load() != 0; }
    void Resolve() override {}
    bool IsConverged() const override { return true; }

    std::vector<uint8_t> &Data() { return _data; }

protected:
    void _Deallocate() override {
        _width = _height = 0;
        _format = HdFormatInvalid;
        _data = {};
    }

private:
    unsigned int _width = 0, _height = 0;
    HdFormat _format = HdFormatInvalid;
    bool _multiSampled = false;
    std::vector<uint8_t> _data;
    std::atomic<int> _mappers{0};
};

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
        for (HdRenderPassAovBinding const &aov : state->GetAovBindings()) {
            if (aov.aovName != HdAovTokens->color)
                continue;
            HdRenderBuffer *bound = aov.renderBuffer;
            if (!bound)
                bound = static_cast<HdRenderBuffer *>(GetRenderIndex()->GetBprim(
                    HdPrimTypeTokens->renderBuffer, aov.renderBufferId));
            // CreateBprim makes every render buffer in this index.
            auto *buffer = static_cast<PikaRenderBuffer *>(bound);
            if (!buffer)
                continue;
            pika_format format;
            if (buffer->GetFormat() == HdFormatFloat16Vec4)
                format = PIKA_FLOAT16_VEC4;
            else if (buffer->GetFormat() == HdFormatFloat32Vec4)
                format = PIKA_FLOAT32_VEC4;
            else {
                fail("the color AOV is " + TfEnum::GetName(buffer->GetFormat()) +
                     "; Pika renders HdFormatFloat16Vec4 and HdFormatFloat32Vec4");
                continue;
            }
            std::vector<uint8_t> &pixels = buffer->Data();
            report(pika_render(_renderer, format, buffer->GetWidth(),
                               buffer->GetHeight(), pixels.data(), pixels.size()),
                   "render");
        }
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

    TfTokenVector const &GetSupportedRprimTypes() const override {
        static const TfTokenVector types = {HdPrimTypeTokens->mesh};
        return types;
    }
    TfTokenVector const &GetSupportedSprimTypes() const override {
        static const TfTokenVector types = {HdPrimTypeTokens->camera};
        return types;
    }
    TfTokenVector const &GetSupportedBprimTypes() const override {
        static const TfTokenVector types = {HdPrimTypeTokens->renderBuffer};
        return types;
    }

    HdResourceRegistrySharedPtr GetResourceRegistry() const override {
        return _resources;
    }

    HdAovDescriptor GetDefaultAovDescriptor(TfToken const &name) const override {
        if (name == HdAovTokens->color)
            return HdAovDescriptor(HdFormatFloat16Vec4, false,
                                   VtValue(GfVec4f(0.0f)));
        return HdAovDescriptor();
    }

    HdRenderPassSharedPtr CreateRenderPass(
        HdRenderIndex *index, HdRprimCollection const &collection) override {
        return std::make_shared<PikaRenderPass>(index, collection, _renderer);
    }

    HdInstancer *CreateInstancer(HdSceneDelegate *delegate,
                                 SdfPath const &id) override {
        return new HdInstancer(delegate, id);
    }
    void DestroyInstancer(HdInstancer *instancer) override { delete instancer; }

    HdRprim *CreateRprim(TfToken const &typeId, SdfPath const &id) override {
        if (typeId == HdPrimTypeTokens->mesh)
            return new PikaMesh(id);
        return nullptr;
    }
    void DestroyRprim(HdRprim *rprim) override { delete rprim; }

    HdSprim *CreateSprim(TfToken const &typeId, SdfPath const &id) override {
        if (typeId == HdPrimTypeTokens->camera)
            return new HdCamera(id);
        return nullptr;
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

    void CommitResources(HdChangeTracker *) override {}

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
