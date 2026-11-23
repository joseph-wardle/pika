// A dome light's sky, worked when the sky changes: its irradiance image, and its reflection's
// folded squares.

#include "view.slang"

struct Push
{
    // A latlong's texels from the top row, 3 floats each.
    float *sky;
    // The irradiance image, or the reflection.
    float *into;
    uint2 extent;
    uint first;
    uint count;
    uint stride;
    // The blurred level or plain halving written.
    uint level;
    // The blurred level's GGX alpha.
    float alpha;
    uint samples;
};

[[vk::push_constant]]
ConstantBuffer<Push> push;

void write(uint texel, float3 value)
{
    push.into[3 * texel] = value.x;
    push.into[3 * texel + 1] = value.y;
    push.into[3 * texel + 2] = value.z;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sum(uint3 thread : SV_DispatchThreadID)
{
    for (uint i = thread.x; i < push.count; i += push.stride)
    {
        uint t = push.first + i;
        float3 n = from_latlong((float2(t % IRRADIANCE.x, t / IRRADIANCE.x) + 0.5) / IRRADIANCE);
        float3 sum = 0;
        for (uint y = 0; y < push.extent.y; y++)
        {
            // A row's band of the sphere, shared evenly by its texels.
            float2 edge = cos(3.1415927 * float2(y, y + 1) / push.extent.y);
            float solid_angle = (edge.x - edge.y) * 6.2831853 / push.extent.x;
            for (uint x = 0; x < push.extent.x; x++)
            {
                float *texel = push.sky + 3 * (y * push.extent.x + x);
                float3 toward = from_latlong((float2(x, y) + 0.5) / push.extent);
                sum += float3(texel[0], texel[1], texel[2]) * (max(dot(n, toward), 0) * solid_angle);
            }
        }
        write(t, sum / 3.1415927);
    }
}

float3 sky(int2 t)
{
    int2 extent = int2(push.extent);
    float *texel = push.sky + 3 * (clamp(t.y, 0, extent.y - 1) * extent.x + (t.x + extent.x) % extent.x);
    return float3(texel[0], texel[1], texel[2]);
}

// The sharp square, each texel its direction's four texels of the sky, a row wrapping.
[shader("compute")]
[numthreads(64, 1, 1)]
void fold(uint3 thread : SV_DispatchThreadID)
{
    for (uint i = thread.x; i < push.count; i += push.stride)
    {
        float2 uv = (float2(i % REFLECTION, i / REFLECTION) + 0.5) / REFLECTION;
        float2 at = latlong(from_folded(uv)) * push.extent - 0.5;
        int2 t = int2(floor(at));
        float2 f = at - floor(at);
        float3 a = lerp(sky(t), sky(t + int2(1, 0)), f.x);
        float3 b = lerp(sky(t + int2(0, 1)), sky(t + int2(1, 1)), f.x);
        write(i, lerp(a, b, f.y));
    }
}

// A plain halving, each texel the mean of the four it covers in the one before.
[shader("compute")]
[numthreads(64, 1, 1)]
void halve(uint3 thread : SV_DispatchThreadID)
{
    int side = int(REFLECTION >> push.level);
    float *before = push.into + 3 * reflection_halving(push.level - 1);
    for (uint i = thread.x; i < push.count; i += push.stride)
    {
        int2 t = 2 * int2(i % side, i / side);
        float3 sum = 0;
        for (int k = 0; k < 4; k++)
            sum += reflection_texel(before, 2 * side, t + int2(k & 1, k >> 1));
        write(reflection_halving(push.level) + i, sum / 4);
    }
}

// The sharp square about uv, averaged over about 4^halving of its texels.
float3 halved(float2 uv, float halving)
{
    uint last = firstbithigh(REFLECTION);
    uint h = min(uint(halving), last - 1);
    float3 a = reflection_square(push.into + 3 * reflection_halving(h), int(REFLECTION >> h), uv);
    float3 b = reflection_square(push.into + 3 * reflection_halving(h + 1), int(REFLECTION >> (h + 1)), uv);
    return lerp(a, b, saturate(halving - h));
}

// A blurred level: what a surface of the level's roughness reflects of the sky seen along its
// normal, the GGX lobe D(h) G(l). Half vectors are drawn by D, each read from the halving whose
// texels are as wide as its share of the lobe (filtered importance sampling, GPU Gems 3, 20).
[shader("compute")]
[numthreads(64, 1, 1)]
void blur(uint3 thread : SV_DispatchThreadID)
{
    uint side = reflection_side(push.level);
    float a2 = push.alpha * push.alpha;
    for (uint i = thread.x; i < push.count; i += push.stride)
    {
        float3 n = from_folded((float2(i % side, i / side) + 0.5) / side);
        float3 u = normalize(cross(abs(n.y) < 0.9 ? float3(0, 1, 0) : float3(1, 0, 0), n));
        float3 v = cross(n, u);
        float3 sum = 0;
        float weights = 0;
        for (uint s = 0; s < push.samples; s++)
        {
            // Hammersley's points.
            float2 xi = float2((s + 0.5) / push.samples, reversebits(s) * 2.3283064e-10);
            float cos2 = (1 - xi.x) / (1 + (a2 - 1) * xi.x);
            float sine = sqrt(1 - cos2);
            float turn = 6.2831853 * xi.y;
            float3 h = sqrt(cos2) * n + sine * (cos(turn) * u + sin(turn) * v);
            float3 l = 2 * dot(n, h) * h - n;
            float nl = dot(n, l);
            if (nl <= 0)
                continue;
            float weight = 2 * nl / (nl + sqrt(a2 + (1 - a2) * nl * nl));
            // Seen along the normal, a direction's density is D / 4.
            float d = a2 / (3.1415927 * pow(cos2 * (a2 - 1) + 1, 2));
            float texels = 4 / (d * push.samples) * REFLECTION * REFLECTION / 12.566371;
            sum += weight * halved(folded(l), max(0.5 * log2(texels) + 1, 0));
            weights += weight;
        }
        write(reflection_level(push.level) + i, sum / weights);
    }
}
