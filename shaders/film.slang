// The film's two passes (src/film.rs), each leaving the image a frame shows for it to be copied
// back. Each pixel gathers its 3×3 neighbours' samples by the frame's weights, the edge's pixels
// standing in past the image's edge, as EEVEE's film extends them. accumulate adds a still frame
// to the sum of still samples; resolve blends a frame drawn while anything changes into the
// history, as EEVEE's film does: reprojected by the nearest motion, filtered by Catmull-Rom,
// clipped to the box of the colors around the pixel and blended by luma-weighted factors.
// Images are the color attachment's pixels: half floats packed in pairs, as half types would
// need the shaderFloat16 feature, or floats.

struct Push
{
    float4 *sum;
    // The color attachment's pixels as copied out: 2 words a pixel of half floats, or 4 of floats.
    uint *frame;
    // The depth AOV's pixels, 0 at the near plane.
    float *depth;
    // The motion attachment's: per pixel two half floats, the offset in pixels to where it was.
    uint *motion;
    // The image the history is: the one the view stopped at for accumulate, the one shown last
    // for resolve. Laid out as frame is, as image is.
    uint *history;
    uint *image;
    // Row by row from the neighbour at (-1, -1).
    float weights[9];
    // What the colors summed weigh: every still sample's weights for accumulate, this frame's
    // for resolve.
    float total;
    uint2 extent;
    // How much the history counts: against the still mean for accumulate; for resolve, 1, or 0
    // with no history.
    float prior;
    uint first;
    uint count;
    uint stride;
    // The frame's sample index; for accumulate, 0 starts the sum.
    uint sample;
    uint words;
};

[[vk::push_constant]]
ConstantBuffer<Push> push;

float4 load(uint *pixels, uint p)
{
    uint *c = pixels + p * push.words;
    if (push.words == 2)
        return float4(unpackHalf2x16ToFloat(c[0]), unpackHalf2x16ToFloat(c[1]));
    return asfloat(uint4(c[0], c[1], c[2], c[3]));
}

void store(uint *pixels, uint p, float4 color)
{
    uint *c = pixels + p * push.words;
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

static const int2 PLUS[5] = { int2(0, 0), int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1) };

uint at(int2 q)
{
    q = clamp(q, int2(0), int2(push.extent) - 1);
    return q.y * push.extent.x + q.x;
}

// The frame's samples around q, filtered: the sum of their weighted colors.
float4 gather(int2 q)
{
    float4 sum = float4(0);
    for (uint k = 0; k < 9; k++)
        sum += push.weights[k] * load(push.frame, at(q + int2(k % 3, k / 3) - 1));
    return sum;
}

// A grid-stride loop: Vulkan guarantees only 65535 groups per dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void accumulate(uint3 thread : SV_DispatchThreadID)
{
    for (uint i = thread.x; i < push.count; i += push.stride)
    {
        uint p = push.first + i;
        int2 q = int2(p % push.extent.x, p / push.extent.x);
        float4 sum = (push.sample == 0 ? float4(0) : push.sum[p]) + gather(q);
        push.sum[p] = sum;
        float4 mean = sum / push.total;
        // At the last sample the prior is 0, and the history is not read.
        if (push.prior > 0)
            mean = (mean + push.prior * load(push.history, p)) / (1 + push.prior);
        store(push.image, p, mean);
    }
}

float3 ycocg(float3 c)
{
    return float3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * (c.r - c.b),
                  0.5 * c.g - 0.25 * (c.r + c.b));
}

float3 rgb(float3 c)
{
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

float4 ycocg_alpha(float4 c)
{
    return float4(ycocg(c.rgb), c.a);
}

// Karis's luma weight, which keeps a bright sample from flickering through the blend; exposure
// is the host's, after Pika, so it is 1 here.
float luma_weight(float luma)
{
    return 1 / (4 + luma);
}

// The history at h, pixel centres at whole numbers: Catmull-Rom over its 4×4 pixels, which
// keeps detail that bilinear filtering would blur away frame after frame.
float4 catmull_rom(float2 h)
{
    float2 base = floor(h);
    float2 t = h - base;
    float2 w[4] = {
        t * (-0.5 + t * (1 - 0.5 * t)),
        1 + t * t * (-2.5 + 1.5 * t),
        t * (0.5 + t * (2 - 1.5 * t)),
        t * t * (-0.5 + 0.5 * t),
    };
    float4 color = float4(0);
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            color += w[x].x * w[y].y * load(push.history, at(int2(base) + int2(x, y) - 1));
    return color;
}

// How far along d from o the box [lo, hi] begins, per channel the plane o is beyond: 0 or less
// when o is inside it. Clipping rather than clamping keeps the hue a corner would shift.
float entry(float3 o, float3 d, float3 lo, float3 hi)
{
    float t = -1e30;
    for (int i = 0; i < 3; i++)
        if (abs(d[i]) > 1e-5)
            t = max(t, ((d[i] > 0 ? lo[i] : hi[i]) - o[i]) / d[i]);
    return t;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void resolve(uint3 thread : SV_DispatchThreadID)
{
    for (uint i = thread.x; i < push.count; i += push.stride)
    {
        uint p = push.first + i;
        int2 q = int2(p % push.extent.x, p / push.extent.x);
        float4 current = ycocg_alpha(gather(q) / push.total);

        // The box of the frame's own samples around q, the plus's and the 3×3's averaged, so its
        // corners do not show as squares (Karis).
        float4 lo = 1e30, hi = -1e30;
        for (uint k = 0; k < 5; k++)
        {
            float4 c = ycocg_alpha(load(push.frame, at(q + PLUS[k])));
            lo = min(lo, c);
            hi = max(hi, c);
        }
        float4 lo9 = lo, hi9 = hi;
        for (uint k = 0; k < 4; k++)
        {
            float4 c = ycocg_alpha(load(push.frame, at(q + int2(k & 1, k >> 1) * 2 - 1)));
            lo9 = min(lo9, c);
            hi9 = max(hi9, c);
        }
        lo = 0.5 * (lo + lo9);
        hi = 0.5 * (hi + hi9);

        // Motion from the nearest surface among q and the corners 2 pixels off, so an edge's
        // history follows the surface in front (Karis).
        uint nearest = at(q);
        for (uint k = 0; k < 4; k++)
        {
            uint c = at(q + int2(k & 1, k >> 1) * 4 - 2);
            if (push.depth[c] < push.depth[nearest])
                nearest = c;
        }
        float2 motion = unpackHalf2x16ToFloat(push.motion[nearest]);
        float2 h = float2(q) + motion;

        float blend = 1;
        float4 history = current;
        if (push.prior > 0 && all(h >= 0) && all(h < float2(push.extent)))
        {
            history = ycocg_alpha(catmull_rom(h));
            // Mostly history; more of the frame as it moves faster, when the history is blurrier.
            blend = lerp(0.05, 0.2, saturate(length(motion) * 0.02));
            // Less of the frame when the history is near the box's luma bounds, where the clip
            // would otherwise flicker with each sample.
            float range = hi.x - lo.x;
            float near = range > 0 ? 2 * min(history.x - lo.x, hi.x - history.x) / range : 0;
            blend *= saturate(near * 4 + 0.1);
            float4 d = current - history;
            history.rgb += d.rgb * saturate(entry(history.rgb, d.rgb, lo.rgb, hi.rgb));
            // Alpha is clipped on its own, as it is not a color.
            if (abs(d.a) > 1e-5)
                history.a += d.a * saturate(((d.a > 0 ? lo.a : hi.a) - history.a) / d.a);
        }
        float kept = luma_weight(history.x) * (1 - blend);
        float added = luma_weight(current.x) * blend;
        float4 color = (history * kept + current * added) / (kept + added);
        color.rgb = rgb(color.rgb);
        // Catmull-Rom overshoots below 0; a NaN would stay in the history for good.
        if (any(isnan(color)))
            color = float4(0);
        store(push.image, p, max(color, 0));
    }
}
