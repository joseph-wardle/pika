// The prepass: every mesh instance into the depth buffer and the depth and
// normal AOVs, pulled through vertex.slang. The material pass draws the
// same instances again with depth EQUAL, so its vertex stage must run the
// same pull() and project() code: slangc 2026.9 ignores `precise` and has
// no invariant attribute, and hydra/images shows holes if they diverge.

#include "vertex.slang"

struct Varyings
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
    nointerpolation uint flip : FLIP;
};

[shader("vertex")]
Varyings vs(uint vertex : SV_VulkanVertexID, uint slot : SV_VulkanInstanceID)
{
    Vertex p = pull(vertex, slot);
    Varyings o;
    o.position = p.position;
    o.normal = p.normal;
    o.flip = p.flip;
    return o;
}

// slangc 2026.9 numbers the outputs from 0 in declaration order, whatever
// SV_Target says, so the color AOV keeps its slot here although the
// pipeline's write mask leaves it to the material pass.
struct Targets
{
    float4 color : SV_Target0;
    float depth : SV_Target1;
    float4 normal : SV_Target2;
};

[shader("fragment")]
Targets fs(Varyings v, bool front : SV_IsFrontFace)
{
    float3 n = dot(v.normal, v.normal) > 0 ? normalize(v.normal) : float3(0);
    // Everything is two-sided: a normal seen from behind is turned toward
    // the viewer. Which side is behind follows the winding, which a
    // left-handed mesh or a mirroring transform reverses.
    if (front == (v.flip != 0))
        n = -n;

    Targets t;
    t.color = float4(0);
    // Hydra's depth is OpenGL's window depth, 0 at the near plane.
    t.depth = 1 - v.position.z;
    t.normal = float4(n, 0);
    return t;
}
