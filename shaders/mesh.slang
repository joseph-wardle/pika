// The mesh pass: every mesh instance into the color, depth and normal AOVs.
// Vertices are pulled from src/geometry.rs's buffers by address, and each
// instance through the view's visible list (shaders/cull.slang); only the
// index buffer and the view's draw records are bound.

struct Push
{
    // World to clip space (src/transform.rs Camera): rows of the
    // column-vector matrix, depth reversed.
    float4 view_projection[4];
    // The camera position (w = 1), or the direction toward it (w = 0).
    float4 eye;
    float *points;
    float *normals;
    // One per point (src/geometry.rs Layout).
    float *smooth;
    uint2 *vertices;
    // 7 per instance (src/transform.rs Instance).
    float4 *instances;
    // The instances the cull pass kept, at each record's first instance.
    uint *visible;
};

[[vk::push_constant]]
ConstantBuffer<Push> push;

// A vertex's normal index when its normal is its point's smooth normal.
static const uint SMOOTH = 0xffffffff;

struct Varyings
{
    float4 position : SV_Position;
    float3 world : WORLD;
    float3 normal : NORMAL;
    nointerpolation uint flip : FLIP;
};

float3 load3(float *values, uint i)
{
    return float3(values[3 * i], values[3 * i + 1], values[3 * i + 2]);
}

// SV_VulkanVertexID and SV_VulkanInstanceID include the draw's vertex offset
// and first instance, which select the mesh's vertices and its visible slot.
[shader("vertex")]
Varyings vs(uint vertex : SV_VulkanVertexID, uint slot : SV_VulkanInstanceID)
{
    uint2 v = push.vertices[vertex];
    float4 *m = push.instances + 7 * push.visible[slot];
    float4 p = float4(load3(push.points, v.x), 1);
    float4 world = float4(dot(m[0], p), dot(m[1], p), dot(m[2], p), 1);
    float3 n = v.y == SMOOTH ? load3(push.smooth, v.x) : load3(push.normals, v.y);

    Varyings o;
    o.position = float4(dot(push.view_projection[0], world), dot(push.view_projection[1], world),
                        dot(push.view_projection[2], world), dot(push.view_projection[3], world));
    o.world = world.xyz;
    o.normal = float3(dot(m[3].xyz, n), dot(m[4].xyz, n), dot(m[5].xyz, n));
    o.flip = asuint(m[6].x) & 1;
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
    float3 toward = normalize(push.eye.xyz - v.world * push.eye.w);
    float grey = saturate(dot(n, toward));

    Targets t;
    t.color = float4(grey, grey, grey, 1);
    // Hydra's depth is OpenGL's window depth, 0 at the near plane.
    t.depth = 1 - v.position.z;
    t.normal = float4(n, 0);
    return t;
}
