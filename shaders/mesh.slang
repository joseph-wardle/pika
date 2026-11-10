// The prepass. The material pass redraws with depth EQUAL, so both must run the
// same pull() and project(): slangc 2026.9 ignores `precise`, has no invariant.

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

// slangc 2026.9 numbers outputs in declaration order, ignoring SV_Target, so
// color keeps its slot although the write mask leaves it to the material pass.
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
    // Two-sided; a left-handed mesh or mirroring transform reverses the winding.
    if (front == (v.flip != 0))
        n = -n;

    Targets t;
    t.color = float4(0);
    // Hydra's depth is OpenGL's window depth, 0 at the near plane.
    t.depth = 1 - v.position.z;
    t.normal = float4(n, 0);
    return t;
}
