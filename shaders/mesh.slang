// The prepass. The material pass redraws with depth EQUAL, so both must run the same pull() and
// project(). slangc 2026.9 ignores `precise` and has no invariant.

#include "vertex.slang"

struct Varyings
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float4 previous_position : PREVIOUS;
    nointerpolation uint flip : FLIP;
};

float3 previous_world(uint vertex, uint slot)
{
    float4 *m = view->previous_instances + 7 * view->visible[slot];
    float4 p = float4(load3(view->previous_points, view->vertices[vertex].x), 1);
    return float3(dot(m[0], p), dot(m[1], p), dot(m[2], p));
}

[shader("vertex")]
Varyings vs(uint vertex : SV_VulkanVertexID, uint slot : SV_VulkanInstanceID)
{
    Vertex p = pull(vertex, slot);
    Varyings o;
    o.position = p.position;
    o.normal = p.normal;
    o.previous_position = transform(view->previous_view_projection, float4(previous_world(vertex, slot), 1));
    o.flip = p.flip;
    return o;
}

// slangc 2026.9 numbers outputs in declaration order, ignoring SV_Target, so color keeps its slot
// although the write mask leaves it to the material pass.
struct Targets
{
    float4 color : SV_Target0;
    float depth : SV_Target1;
    float4 normal : SV_Target2;
    float2 motion : SV_Target3;
};

[shader("fragment")]
Targets fs(Varyings v, bool front : SV_IsFrontFace)
{
    float3 n = dot(v.normal, v.normal) > 0 ? normalize(v.normal) : float3(0);
    // Two-sided. A left-handed mesh or mirroring transform reverses the winding.
    if (front == (v.flip != 0))
        n = -n;

    Targets t;
    t.color = float4(0);
    // Hydra's depth is OpenGL's window depth, 0 at the near plane.
    t.depth = 1 - v.position.z;
    t.normal = float4(n, 0);
    float2 now = v.position.xy + view->jitter;
    float2 was = (v.previous_position.xy / v.previous_position.w * 0.5 + 0.5) * float2(view->extent);
    // A point behind the camera before gets the greatest half float, which takes resolve off the
    // image, where it drops the history.
    t.motion = v.previous_position.w > 0 ? was - now : float2(65504);
    return t;
}
