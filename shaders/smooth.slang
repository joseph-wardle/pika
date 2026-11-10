// Smooth normals as Hydra's Hd_SmoothNormals computes them.

struct Push
{
    float *points;
    uint *offsets;
    // Per corner, the face's previous and next points, swapped if left-handed.
    uint2 *corners;
    float *normals;
    uint first;
    uint count;
    uint stride;
};

[[vk::push_constant]]
ConstantBuffer<Push> push;

float3 load3(float *values, uint i)
{
    return float3(values[3 * i], values[3 * i + 1], values[3 * i + 2]);
}

// A grid-stride loop: Vulkan guarantees only 65535 groups per dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void cs(uint3 thread : SV_DispatchThreadID)
{
    for (uint i = thread.x; i < push.count; i += push.stride)
    {
        uint p = push.first + i;
        float3 curr = load3(push.points, p);
        float3 n = float3(0);
        for (uint e = push.offsets[p]; e < push.offsets[p + 1]; e++)
        {
            uint2 c = push.corners[e];
            n += cross(load3(push.points, c.y) - curr, load3(push.points, c.x) - curr);
        }
        n = dot(n, n) > 0 ? normalize(n) : float3(0);
        push.normals[3 * p] = n.x;
        push.normals[3 * p + 1] = n.y;
        push.normals[3 * p + 2] = n.z;
    }
}
