// Adds a still view's frame to the sum of its frames so far, filtered, and writes their mean
// for it to be copied back: each pixel gathers its 3×3 neighbours' samples by the frame's weights
// (src/accumulation.rs), the edge's pixels standing in past the image's edge, as EEVEE's film
// extends them. Half floats are packed in pairs: half types would need the shaderFloat16 feature.

struct Push
{
    float4 *sum;
    // The color attachment's pixels as copied out: 2 words a pixel of half floats, or 4 of floats.
    uint *frame;
    // The mean, laid out as frame is.
    uint *mean;
    // Row by row from the neighbour at (-1, -1).
    float weights[9];
    // The weights of every sample the sum holds, this frame's included.
    float total;
    uint2 extent;
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

float4 load(uint p)
{
    uint *c = push.frame + p * push.words;
    if (push.words == 2)
        return float4(unpackHalf2x16ToFloat(c[0]), unpackHalf2x16ToFloat(c[1]));
    return asfloat(uint4(c[0], c[1], c[2], c[3]));
}

void store(uint p, float4 color)
{
    uint *c = push.mean + p * push.words;
    if (push.words == 2)
    {
        c[0] = packHalf2x16(color.xy);
        c[1] = packHalf2x16(color.zw);
        return;
    }
    uint4 bits = asuint(color);
    c[0] = bits.x;
    c[1] = bits.y;
    c[2] = bits.z;
    c[3] = bits.w;
}

// A grid-stride loop: Vulkan guarantees only 65535 groups per dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void cs(uint3 thread : SV_DispatchThreadID)
{
    int2 last = int2(push.extent) - 1;
    for (uint i = thread.x; i < push.count; i += push.stride)
    {
        uint p = push.first + i;
        int2 at = int2(p % push.extent.x, p / push.extent.x);
        float4 sum = push.sample == 0 ? float4(0) : push.sum[p];
        for (uint k = 0; k < 9; k++)
        {
            int2 q = clamp(at + int2(k % 3, k / 3) - 1, 0, last);
            sum += push.weights[k] * load(q.y * push.extent.x + q.x);
        }
        push.sum[p] = sum;
        store(p, sum / push.total);
    }
}
