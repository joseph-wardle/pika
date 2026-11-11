// Casters into a light's plain map: the least light depth per texel by atomic min, so the
// map needs no depth attachment.

#include "vertex.slang"

struct Varyings
{
    float4 position : SV_Position;
};

[shader("vertex")]
Varyings vs(uint vertex : SV_VulkanVertexID, uint slot : SV_VulkanInstanceID)
{
    Varyings o;
    o.position = pull(vertex, slot).position;
    return o;
}

[shader("fragment")]
void fs(Varyings v)
{
    Shadow *s = view->shadows;
    uint2 t = uint2(v.position.xy);
    // Depth is in [0, 1], where float bits order as integers do.
    s->map[t.y * s->size + t.x].min(asuint(v.position.z));
}
