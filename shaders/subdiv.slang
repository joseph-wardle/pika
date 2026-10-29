// The stencil pass: each refined point of a subdivision surface as the
// weighted sum of the cage points its stencil names. The delegate's
// OpenSubdiv built the stencils; src/geometry.rs lays them out, with refined
// point p's stencil at offsets[p].

struct Push
{
    float *cage;
    // Point p's stencil is entries offsets[p] to offsets[p + 1].
    uint *offsets;
    uint *indices;
    float *weights;
    float *points;
    // The points this dispatch writes.
    uint first;
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
    for (uint i = thread.x; i < push.count; i += push.stride)
    {
        uint p = push.first + i;
        float3 sum = float3(0);
        for (uint e = push.offsets[p]; e < push.offsets[p + 1]; e++)
        {
            sum += push.weights[e] * load3(push.cage, push.indices[e]);
        }
        push.points[3 * p] = sum.x;
        push.points[3 * p + 1] = sum.y;
        push.points[3 * p + 2] = sum.z;
    }
}
