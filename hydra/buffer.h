// The render buffer Pika writes each frame into.
#pragma once

#include <pxr/base/gf/vec3i.h>
#include <pxr/base/tf/enum.h>
#include <pxr/imaging/hd/renderBuffer.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

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
        _name = TfEnum::GetName(format);
        _multiSampled = multiSampled;
        _data.resize(size_t(_width) * _height * HdDataSizeOfFormat(format));
        return true;
    }

    unsigned int GetWidth() const override { return _width; }
    unsigned int GetHeight() const override { return _height; }
    unsigned int GetDepth() const override { return 1; }
    HdFormat GetFormat() const override { return _format; }
    // Hydra's name for the format, as pika_aov takes it.
    const char *FormatName() const { return _name.c_str(); }
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
        _name = {};
        _data = {};
    }

private:
    unsigned int _width = 0, _height = 0;
    HdFormat _format = HdFormatInvalid;
    std::string _name;
    bool _multiSampled = false;
    std::vector<uint8_t> _data;
    std::atomic<int> _mappers{0};
};

PXR_NAMESPACE_CLOSE_SCOPE
