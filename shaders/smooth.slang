// The smooth normal pass: each point's normal from the faces around it, as
// Hydra's Hd_SmoothNormals computes it. src/geometry.rs builds the table of
// each point's corners.

struct Push
{
    float *points;
    // Point p's corners are entries offsets[p] to offsets[p + 1].
    uint *offsets;
    // Per corner, the face's previous and next points, swapped on
    // left-handed meshes.
    uint2 *corners;
    float *normals;
    uint count;
    // The threads dispatched.
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
    for (uint p = thread.x; p < push.count; p += push.stride)
    {
        float3 curr = load3(push.points, p);
        float3 n = float3(0);
        for (uint e = push.offsets[p]; e < push.offsets[p + 1]; e++)
        {
            uint2 c = push.corners[e];
            n += cross(load3(push.points, c.y) - curr, load3(push.points, c.x) - curr);
        }
        // A point no face uses, or whose faces are degenerate, gets none.
        n = dot(n, n) > 0 ? normalize(n) : float3(0);
        push.normals[3 * p] = n.x;
        push.normals[3 * p + 1] = n.y;
        push.normals[3 * p + 2] = n.z;
    }
}
