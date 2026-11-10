// Modified from MaterialX 1.39.5's libraries/stdlib/genslang/lib/mx_texture.slang
// (Academy Software Foundation, Apache License 2.0: bake/libraries/NOTICE).
//
// Samples a 2D array (one layer per UDIM tile) under the library's own names, so its
// image nodes need no change.
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

    // A UDIM set holds tile 1001 + u + 10 v at layer `tiles[u + 10 v]`, past the layers if
    // missing. v is flipped inside the tile: USD's st (0, 0) is bottom-left, baked rows top first.
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

// The mip level comes from the uv's gradients, continuous across tiles.
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
