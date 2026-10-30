// The mesh pass: every mesh instance into the color, depth and normal AOVs,
// pulled through vertex.slang.

#include "vertex.slang"

struct Varyings
{
    float4 position : SV_Position;
    float3 world : WORLD;
    float3 normal : NORMAL;
    nointerpolation uint flip : FLIP;
};

[shader("vertex")]
Varyings vs(uint vertex : SV_VulkanVertexID, uint slot : SV_VulkanInstanceID)
{
    Vertex p = pull(vertex, slot);
    Varyings o;
    o.position = p.position;
    o.world = p.world;
    o.normal = p.normal;
    o.flip = p.flip;
    return o;
}

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
    float3 toward = normalize(view->eye.xyz - v.world * view->eye.w);
    float grey = saturate(dot(n, toward));

    Targets t;
    t.color = float4(grey, grey, grey, 1);
    // Hydra's depth is OpenGL's window depth, 0 at the near plane.
    t.depth = 1 - v.position.z;
    t.normal = float4(n, 0);
    return t;
}
