struct Push
{
    float *cage;
    uint *offsets;
    uint *indices;
    float *weights;
    float *refined;
    uint first;
    uint count;
    uint stride;
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
