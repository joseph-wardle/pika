// The lit dome's sky, drawn behind the scene where the prepass left the far depth.

#include "view.slang"

struct Push
{
    // Rows taking (pixel x, pixel y, 1) to the direction through that pixel, in the dome's axes.
    float4 rays[3];
    // Rows taking it to (x, y, w), where x / w and y / w are the pixel the direction was at in
    // the frame before, less this frame's jitter. w is 0 or less behind that camera.
    float4 previous[3];
    float3 radiance;
    uint levels;
    // RGB9E5 levels, the finest first, each from its top row.
    uint *texels;
    // The finest level's.
    uint2 size;
};

[[vk::push_constant]]
ConstantBuffer<Push> push;

[shader("vertex")]
float4 vs(uint vertex : SV_VulkanVertexID) : SV_Position
{
    // One triangle over the whole target, at the depth the prepass cleared to.
    return float4(float2(vertex & 1, vertex >> 1) * 4 - 1, 0, 1);
}

float3 texel(uint *level, int2 size, int2 t)
{
    // Round the sphere in longitude, clamped at the poles.
    t = int2((t.x + size.x) % size.x, clamp(t.y, 0, size.y - 1));
    uint v = level[t.y * size.x + t.x];
    return float3(v & 0x1ff, v >> 9 & 0x1ff, v >> 18 & 0x1ff) * exp2(float(v >> 27) - 24);
}

float3 bilinear(uint level, float2 uv)
{
    uint first = 0;
    for (uint l = 0; l < level; l++)
    {
        uint2 size = max(push.size >> l, 1);
        first += size.x * size.y;
    }
    uint *texels = push.texels + first;
    int2 size = int2(max(push.size >> level, 1));
    float2 at = uv * size - 0.5;
    int2 t = int2(floor(at));
    float2 f = at - floor(at);
    float3 a = lerp(texel(texels, size, t), texel(texels, size, t + int2(1, 0)), f.x);
    float3 b = lerp(texel(texels, size, t + int2(0, 1)), texel(texels, size, t + int2(1, 1)), f.x);
    return lerp(a, b, f.y);
}

float3 rows(float4 m[3], float3 p)
{
    return float3(dot(m[0].xyz, p), dot(m[1].xyz, p), dot(m[2].xyz, p));
}

// mesh.slang's targets, of which this writes colour and motion.
struct Targets
{
    float4 color : SV_Target0;
    float depth : SV_Target1;
    float4 normal : SV_Target2;
    float2 motion : SV_Target3;
};

[shader("fragment")]
Targets fs(float4 position : SV_Position)
{
    float3 p = float3(position.xy, 1);
    float3 d = rows(push.rays, p);
    float3 n = normalize(d);
    // The angle a pixel spans, from its ray, so the latlong's seam has no derivative to break.
    float3 across = float3(push.rays[0].x, push.rays[1].x, push.rays[2].x);
    float3 up = float3(push.rays[0].y, push.rays[1].y, push.rays[2].y);
    float spread = max(length(across - n * dot(n, across)), length(up - n * dot(n, up)));
    spread /= length(d);
    // A finest texel spans π / height.
    float level = clamp(log2(spread * push.size.y / 3.1415927), 0, push.levels - 1);
    uint coarse = min(uint(level) + 1, push.levels - 1);
    float2 uv = latlong(n);
    float3 sky = lerp(bilinear(uint(level), uv), bilinear(coarse, uv), level - uint(level));

    Targets t;
    // Opaque, as Storm's and EEVEE's skies are.
    t.color = float4(sky * push.radiance, 1);
    t.depth = 0;
    t.normal = float4(0);
    float3 was = rows(push.previous, p);
    // As in mesh.slang, a direction behind the camera before takes the history off the image.
    t.motion = was.z > 0 ? was.xy / was.z - position.xy : float2(65504);
    return t;
}
