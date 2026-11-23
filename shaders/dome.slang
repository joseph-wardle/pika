// A dome light's irradiance image, summed from its sky when the sky changes.

#include "view.slang"

struct Push
{
    // A latlong's texels from the top row, 3 floats each.
    float *sky;
    float *irradiance;
    uint2 extent;
    uint first;
    uint count;
    uint stride;
};

[[vk::push_constant]]
ConstantBuffer<Push> push;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs(uint3 thread : SV_DispatchThreadID)
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
        sum /= 3.1415927;
        push.irradiance[3 * t] = sum.x;
        push.irradiance[3 * t + 1] = sum.y;
        push.irradiance[3 * t + 2] = sum.z;
    }
}
