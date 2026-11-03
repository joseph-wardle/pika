// Pika's textures are 2D arrays: one layer for a plain image, and for a
// UDIM set one layer per tile the set has, found through `tiles`, which
// maps tile 1001 + i to its layer, or past the layers for a tile the set
// lacks (bake/texture.cpp writes the table into the baked file). This file
// shadows the MaterialX library's, which samples a Texture2D, under the
// same struct and function names so that the library's image nodes need no
// change.
struct SamplerTexture2D
{
    Texture2DArray tex;
    SamplerState sampler;
    StructuredBuffer<uint> tiles;

    int get_width(int mipLevel = 0)
    {
        uint width, height, layers, levels;
        tex.GetDimensions(mipLevel, width, height, layers, levels);
        return width;
    }

    int get_height(int mipLevel = 0)
    {
        uint width, height, layers, levels;
        tex.GetDimensions(mipLevel, width, height, layers, levels);
        return height;
    }

    int get_num_mip_levels()
    {
        uint width, height, layers, levels;
        tex.GetDimensions(0, width, height, layers, levels);
        return levels;
    }

    // The layer and the coordinate within it, the layer past the texture's
    // for a tile the set lacks. A plain image, whose table has the one
    // entry, leaves the wrapping to the sampler; a UDIM set holds tile
    // 1001 + u + 10 v at its own layer, the rows ten tiles wide, as USD
    // numbers them. USD and MaterialX put st (0, 0) at an image's
    // bottom-left corner, and a baked texture keeps the image's rows top
    // first, so v is flipped here, inside the tile
    // (hydra/images/textured.usda, hydra/images/udim.usda).
    float3 locate(float2 uv)
    {
        uint count, stride;
        tiles.GetDimensions(count, stride);
        if (count == 1)
            return float3(uv.x, 1.0 - uv.y, 0.0);
        float2 tile = floor(uv);
        uint layer = 0xFFFFFFFF;
        if (all(tile >= 0.0) && all(tile < 10.0))
            layer = tiles[uint(tile.x) + 10 * uint(tile.y)];
        return float3(uv.x - tile.x, 1.0 - (uv.y - tile.y), layer);
    }

    bool missing(float3 at)
    {
        uint width, height, layers, levels;
        tex.GetDimensions(0, width, height, layers, levels);
        return at.z >= layers;
    }
}

float4 textureLod(SamplerTexture2D tex, float2 uv, float lod)
{
    float3 at = tex.locate(uv);
    float4 texel = tex.tex.SampleLevel(tex.sampler, at, lod);
    return tex.missing(at) ? float4(0.0) : texel;
}

// The coordinate within a tile jumps at the tile's edge, so the mip level
// is chosen from the gradients of the uv itself, which are continuous
// across the set, not from those of the coordinate sampled.
float4 texture(SamplerTexture2D tex, float2 uv)
{
    return textureGrad(tex, uv, ddx(uv), ddy(uv));
}

float4 textureGrad(SamplerTexture2D tex, float2 uv, float2 ddx_, float2 ddy_)
{
    float3 at = tex.locate(uv);
    float4 texel = tex.tex.SampleGrad(tex.sampler, at, ddx_, ddy_);
    return tex.missing(at) ? float4(0.0) : texel;
}

int2 textureSize(SamplerTexture2D tex, int mipLevel)
{
    return int2(tex.get_width(mipLevel), tex.get_height(mipLevel));
}
