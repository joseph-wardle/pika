// The texture job: an image as authored in, the one file the renderer
// uploads out. OpenImageIO decodes the source; its stored mip levels at or
// below `cap` texels on a side are kept and the missing ones are built by
// box filtering; OpenColorIO converts the declared colour space to
// lin_rec709, Pika's working space; then the levels are encoded:
//
//   float    32-bit floats, one, two or four channels as the source has them
//   color    BC7, stored sRGB-encoded so that sampling decodes to linear
//   data     BC7 at linear 8 bits (normals and other vectors)
//   single   BC4, the first channel at linear 8 bits
//
// The file is a 24-byte header of little-endian u32: "pika", width, height,
// layers, levels, the VkFormat; then each level in order, every layer within
// it. src/texture.rs reads it.

#include "bake.h"
#include "bc7enc/bc7enc.h"

#include <OpenImageIO/color.h>
#include <OpenImageIO/imageio.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

// MaterialX's document colour space, under the name every ACES config
// carries. Its full name is "Linear Rec.709 (sRGB)"; under an ACES config
// "linear" alone can mean ACEScg.
const char *const WORKING = "lin_rec709";

// VkFormat values, as the renderer reads them back.
enum : uint32_t {
    R32_SFLOAT = 100,
    R32G32_SFLOAT = 103,
    R32G32B32A32_SFLOAT = 109,
    BC4_UNORM_BLOCK = 139,
    BC7_UNORM_BLOCK = 145,
    BC7_SRGB_BLOCK = 146,
};

// A mip level as four floats per texel, (r, g, 0, 1) and so on where the
// source has fewer channels, a grey source repeated across r, g and b.
struct Level {
    unsigned width = 0, height = 0;
    std::vector<float> rgba;
    float const *at(unsigned x, unsigned y) const { return &rgba[(size_t(y) * width + x) * 4]; }
};

unsigned half(unsigned n) { return std::max(1u, n / 2); }

bool fits(Level const &level, unsigned cap) { return level.width <= cap && level.height <= cap; }

Level read(OIIO::ImageInput &in, std::string const &path) {
    OIIO::ImageSpec const &spec = in.spec();
    int channels = std::min(spec.nchannels, 4);
    Level level;
    level.width = spec.width;
    level.height = spec.height;
    std::vector<float> data(size_t(spec.width) * spec.height * channels);
    if (!in.read_image(in.current_subimage(), in.current_miplevel(), 0, channels,
                       OIIO::TypeDesc::FLOAT, data.data()))
        throw std::runtime_error("cannot read " + path + ": " + in.geterror());
    level.rgba.resize(size_t(spec.width) * spec.height * 4);
    for (size_t i = 0; i < level.rgba.size() / 4; ++i) {
        float const *p = &data[i * channels];
        float *q = &level.rgba[i * 4];
        q[0] = p[0];
        q[1] = channels > 1 ? p[1] : p[0];
        q[2] = channels > 2 ? p[2] : channels == 1 ? p[0] : 0.0f;
        q[3] = channels > 3 ? p[3] : 1.0f;
    }
    return level;
}

// The source's stored levels from the first at or below the cap, or from
// the smallest when none is; later levels that do not halve the one before
// (a ripmap's) are left to the filter.
std::vector<Level> stored(std::string const &path, unsigned cap, int &channels,
                          OIIO::TypeDesc &type) {
    std::unique_ptr<OIIO::ImageInput> in = OIIO::ImageInput::open(path);
    if (!in)
        throw std::runtime_error("cannot open " + path + ": " + OIIO::geterror());
    int first = -1, count = 0;
    for (; in->seek_subimage(0, count); ++count) {
        if (first < 0 && in->spec().width <= int(cap) && in->spec().height <= int(cap))
            first = count;
    }
    if (first < 0)
        first = count - 1;
    std::vector<Level> levels;
    for (int m = first; m < count; ++m) {
        in->seek_subimage(0, m);
        if (m == first) {
            channels = std::min(in->spec().nchannels, 4);
            type = in->spec().format;
        }
        Level level = read(*in, path);
        if (!levels.empty() && (level.width != half(levels.back().width) ||
                                level.height != half(levels.back().height)))
            break;
        levels.push_back(std::move(level));
    }
    in->close();
    return levels;
}

void convert(std::vector<Level> &levels, std::string const &space, std::string const &config) {
    std::string name = config.empty() ? "OpenImageIO's default OCIO config" : config;
    OIIO::ColorConfig ocio(config);
    if (ocio.has_error())
        throw std::runtime_error("cannot read " + name + ": " + ocio.geterror());
    OIIO::ColorProcessorHandle processor = ocio.createColorProcessor(space, WORKING);
    if (!processor)
        throw std::runtime_error("no transform from the colour space " + space + " to " +
                                 WORKING + " in " + name + ": " + ocio.geterror());
    if (processor->isNoOp())
        return;
    for (Level &level : levels)
        processor->apply(level.rgba.data(), level.width, level.height, 4, sizeof(float),
                         4 * sizeof(float), level.width * 4 * sizeof(float));
}

// A 2×2 box, the last column or row repeated where the size is odd.
Level halve(Level const &src) {
    Level dst;
    dst.width = half(src.width);
    dst.height = half(src.height);
    dst.rgba.resize(size_t(dst.width) * dst.height * 4);
    for (unsigned y = 0; y < dst.height; ++y) {
        for (unsigned x = 0; x < dst.width; ++x) {
            unsigned x0 = std::min(2 * x, src.width - 1), x1 = std::min(2 * x + 1, src.width - 1);
            unsigned y0 = std::min(2 * y, src.height - 1), y1 = std::min(2 * y + 1, src.height - 1);
            float *q = &dst.rgba[(size_t(y) * dst.width + x) * 4];
            for (int c = 0; c < 4; ++c)
                q[c] = (src.at(x0, y0)[c] + src.at(x1, y0)[c] + src.at(x0, y1)[c] +
                        src.at(x1, y1)[c]) *
                       0.25f;
        }
    }
    return dst;
}

// sRGB-encodes the colour channels, so that 8 bits of a BC7_SRGB level
// decode to the linear value.
void encode_srgb(Level &level) {
    for (size_t i = 0; i < level.rgba.size(); ++i) {
        if (i % 4 == 3)
            continue;
        float v = std::clamp(level.rgba[i], 0.0f, 1.0f);
        level.rgba[i] = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    }
}

uint8_t unorm(float v) { return uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }

// The 4×4 block at (bx, by) as RGBA8, the image's edge repeated past it.
void block(Level const &level, unsigned bx, unsigned by, uint8_t out[64]) {
    for (unsigned i = 0; i < 16; ++i) {
        float const *p = level.at(std::min(bx * 4 + i % 4, level.width - 1),
                                  std::min(by * 4 + i / 4, level.height - 1));
        for (int c = 0; c < 4; ++c)
            out[i * 4 + c] = unorm(p[c]);
    }
}

// Encodes the level's 4×4 blocks, `size` bytes each, a row of blocks per
// thread at a time: a 4K level is a million blocks, seven seconds on one
// core.
template <typename Encode>
std::vector<uint8_t> blocks(Level const &level, size_t size, Encode encode) {
    unsigned bw = (level.width + 3) / 4, bh = (level.height + 3) / 4;
    std::vector<uint8_t> out(size_t(bw) * bh * size);
    std::atomic<unsigned> next(0);
    auto rows = [&] {
        for (unsigned by = next++; by < bh; by = next++)
            for (unsigned bx = 0; bx < bw; ++bx)
                encode(bx, by, &out[(size_t(by) * bw + bx) * size]);
    };
    std::vector<std::thread> threads;
    for (unsigned i = 1; i < std::min(bh, std::thread::hardware_concurrency()); ++i)
        threads.emplace_back(rows);
    rows();
    for (std::thread &thread : threads)
        thread.join();
    return out;
}

std::vector<uint8_t> bc7(Level const &level, bc7enc_compress_block_params const &params) {
    return blocks(level, 16, [&](unsigned bx, unsigned by, uint8_t *out) {
        uint8_t texels[64];
        block(level, bx, by, texels);
        bc7enc_compress_block(out, texels, &params);
    });
}

// The plain BC4 encoder: the block's extremes as the endpoints, each texel
// the nearest of the eight values between them.
std::vector<uint8_t> bc4(Level const &level) {
    return blocks(level, 8, [&](unsigned bx, unsigned by, uint8_t *out) {
        uint8_t texels[64];
        block(level, bx, by, texels);
        uint8_t lo = 255, hi = 0;
        for (unsigned i = 0; i < 16; ++i) {
            lo = std::min(lo, texels[i * 4]);
            hi = std::max(hi, texels[i * 4]);
        }
        // With the first endpoint the larger, index 0 is hi, 1 is lo and 2
        // to 7 step from hi to lo in sevenths.
        int palette[8] = {hi, lo};
        for (int k = 2; k < 8; ++k)
            palette[k] = ((8 - k) * hi + (k - 1) * lo) / 7;
        uint64_t bits = 0;
        for (unsigned i = 0; i < 16; ++i) {
            int best = 0;
            for (int k = 1; k < 8 && hi != lo; ++k) {
                if (std::abs(palette[k] - texels[i * 4]) < std::abs(palette[best] - texels[i * 4]))
                    best = k;
            }
            bits |= uint64_t(best) << (3 * i);
        }
        out[0] = hi;
        out[1] = lo;
        for (int i = 0; i < 6; ++i)
            out[2 + i] = uint8_t(bits >> (8 * i));
    });
}

std::vector<uint8_t> floats(Level const &level, int channels) {
    std::vector<uint8_t> out(size_t(level.width) * level.height * channels * 4);
    for (size_t i = 0; i < size_t(level.width) * level.height; ++i)
        memcpy(&out[i * channels * 4], &level.rgba[i * 4], channels * 4);
    return out;
}

void put(std::ofstream &file, uint32_t v) {
    char b[4] = {char(v), char(v >> 8), char(v >> 16), char(v >> 24)};
    file.write(b, 4);
}

} // namespace

void texture(std::string const &source, std::string const &space, std::string const &encoding,
             unsigned cap, std::string const &config, std::string const &out) {
    if (encoding != "float" && encoding != "color" && encoding != "data" && encoding != "single")
        throw std::runtime_error("the encoding " + encoding + " is not float, color, data or single");
    int channels = 0;
    OIIO::TypeDesc type;
    std::vector<Level> levels = stored(source, cap, channels, type);
    // USD's "auto": sRGB for 8-bit images, raw for the rest. The file's own
    // colour space metadata is not read: OpenImageIO calls a float image
    // "linear", which under an ACES config means ACEScg.
    std::string resolved = space;
    if (space == "auto")
        resolved = type == OIIO::TypeDesc::UINT8 ? "srgb_texture" : "raw";
    convert(levels, resolved, config);
    while (!fits(levels.front(), cap))
        levels.front() = halve(levels.front());
    while (levels.back().width > 1 || levels.back().height > 1)
        levels.push_back(halve(levels.back()));

    uint32_t format;
    std::vector<std::vector<uint8_t>> bytes;
    if (encoding == "float") {
        channels = channels == 3 ? 4 : channels;
        format = channels == 1 ? R32_SFLOAT : channels == 2 ? R32G32_SFLOAT : R32G32B32A32_SFLOAT;
        for (Level const &level : levels)
            bytes.push_back(floats(level, channels));
    } else if (encoding == "single") {
        format = BC4_UNORM_BLOCK;
        for (Level const &level : levels)
            bytes.push_back(bc4(level));
    } else {
        bc7enc_compress_block_init();
        bc7enc_compress_block_params params;
        bc7enc_compress_block_params_init(&params);
        if (encoding == "color") {
            format = BC7_SRGB_BLOCK;
            bc7enc_compress_block_params_init_perceptual_weights(&params);
            for (Level &level : levels)
                encode_srgb(level);
        } else {
            format = BC7_UNORM_BLOCK;
            bc7enc_compress_block_params_init_linear_weights(&params);
        }
        for (Level const &level : levels)
            bytes.push_back(bc7(level, params));
    }

    std::ofstream file(out, std::ios::binary);
    file.write("pika", 4);
    put(file, levels.front().width);
    put(file, levels.front().height);
    put(file, 1);
    put(file, uint32_t(levels.size()));
    put(file, format);
    for (std::vector<uint8_t> const &level : bytes)
        file.write(reinterpret_cast<char const *>(level.data()), level.size());
    if (!file)
        throw std::runtime_error("cannot write " + out);
}
