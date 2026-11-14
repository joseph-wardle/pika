// Adds a still view's frame to the sum of its frames so far and writes their mean in its place.

struct Push
{
    float4 *sum;
    // The color attachment's pixels as copied out: 2 words a pixel of half floats, or 4 of floats.
    uint *color;
    uint first;
    uint count;
    uint stride;
    // The frame's sample index; 0 starts the sum.
    uint sample;
    uint words;
    uint unused;
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
        uint *c = push.color + p * push.words;
        float4 color;
        if (push.words == 2)
            color = float4(f16tof32(c[0]), f16tof32(c[0] >> 16), f16tof32(c[1]), f16tof32(c[1] >> 16));
        else
            color = asfloat(uint4(c[0], c[1], c[2], c[3]));
        float4 sum = push.sample == 0 ? color : push.sum[p] + color;
        push.sum[p] = sum;
        float4 mean = sum / float(push.sample + 1);
        if (push.words == 2)
        {
            c[0] = f32tof16(mean.x) | f32tof16(mean.y) << 16;
            c[1] = f32tof16(mean.z) | f32tof16(mean.w) << 16;
            continue;
        }
        uint4 bits = asuint(mean);
        c[0] = bits.x;
        c[1] = bits.y;
        c[2] = bits.z;
        c[3] = bits.w;
    }
}
