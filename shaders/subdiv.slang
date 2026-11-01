// The stencil pass: each refined value of a subdivision surface, a point's
// xyz or a texcoord's xy, as the weighted sum of the cage values its stencil
// names. The delegate's OpenSubdiv built the stencils; src/geometry.rs lays
// them out, with refined value p's stencil at offsets[p].

struct Push
{
    float *cage;
    // Value p's stencil is entries offsets[p] to offsets[p + 1].
    uint *offsets;
    uint *indices;
    float *weights;
    float *refined;
    // The values this dispatch writes.
    uint first;
    uint count;
    // The threads dispatched.
    uint stride;
    // The floats per value.
    uint width;
};

[[vk::push_constant]]
ConstantBuffer<Push> push;

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
            uint base = push.width * push.indices[e];
            for (uint c = 0; c < push.width; c++)
            {
                sum[c] += push.weights[e] * push.cage[base + c];
            }
        }
        for (uint c = 0; c < push.width; c++)
        {
            push.refined[push.width * p + c] = sum[c];
        }
    }
}
