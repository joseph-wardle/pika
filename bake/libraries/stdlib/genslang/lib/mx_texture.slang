// Pika's textures are 2D arrays: one layer for a plain image, and for a
// UDIM set one layer per tile in a dense grid from 1001. This file shadows
// the MaterialX library's, which samples a Texture2D, under the same struct
// and function names so that the library's image nodes need no change.
struct SamplerTexture2D
{
    Texture2DArray tex;
    SamplerState sampler;

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

    // The layer and the coordinate within it. A one-layer texture leaves
    // the wrapping to the sampler; a UDIM set holds tile 1001 + u + 10 v.
    // USD and MaterialX put st (0, 0) at an image's bottom-left corner,
    // and a baked texture keeps the image's rows top first, so v is
    // flipped here, inside the tile (hydra/images/textured.usda).
    float3 locate(float2 uv)
    {
        uint width, height, layers, levels;
        tex.GetDimensions(0, width, height, layers, levels);
        if (layers == 1)
            return float3(uv.x, 1.0 - uv.y, 0.0);
        float2 tile = floor(uv);
        return float3(uv.x - tile.x, 1.0 - (uv.y - tile.y), tile.x + 10.0 * tile.y);
    }
}

float4 textureLod(SamplerTexture2D tex, float2 uv, float lod)
{
    return tex.tex.SampleLevel(tex.sampler, tex.locate(uv), lod);
}

float4 texture(SamplerTexture2D tex, float2 uv)
{
    return tex.tex.Sample(tex.sampler, tex.locate(uv));
}

float4 textureGrad(SamplerTexture2D tex, float2 uv, float2 ddx_, float2 ddy_)
{
    return tex.tex.SampleGrad(tex.sampler, tex.locate(uv), ddx_, ddy_);
}

int2 textureSize(SamplerTexture2D tex, int mipLevel)
{
    return int2(tex.get_width(mipLevel), tex.get_height(mipLevel));
}
