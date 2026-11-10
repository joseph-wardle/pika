// The first-frame check: fills `rect` (pixels, x0 y0 x1 y1, end exclusive) of a `size` target.

struct Push
{
    int4 rect;
    int2 size;
};

[[vk::push_constant]]
ConstantBuffer<Push> push;

// SV_VulkanVertexID, since SV_VertexID subtracts the base vertex (needs shaderDrawParameters).
static const int2 corners[6] = { int2(0, 0), int2(1, 0), int2(0, 1),
                                 int2(0, 1), int2(1, 0), int2(1, 1) };

[shader("vertex")]
float4 vs(uint id: SV_VulkanVertexID) : SV_Position
{
    int2 c = corners[id];
    int2 px = int2(c.x == 1 ? push.rect.z : push.rect.x, c.y == 1 ? push.rect.w : push.rect.y);
    // Vulkan's NDC has y down; exact in float for a power-of-two size, keeping pixel edges.
    return float4(float2(px) * 2.0 / float2(push.size) - 1.0, 0.0, 1.0);
}

[shader("fragment")]
uint4 fs(float4 pos: SV_Position) : SV_Target
{
    return uint4(uint2(pos.xy), 1, 255);
}
