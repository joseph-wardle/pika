// Vertex pulling for every pass that draws mesh instances: each vertex from
// src/geometry.rs's buffers by address, each instance through the view's
// visible list (shaders/cull.slang), so only the index buffer and the
// view's draw records are bound. mesh.slang uses it, and so does the vertex
// stage src/surface.rs writes for each material.

#include "view.slang"

[[vk::push_constant]]
View *view;

// A vertex's normal index when its normal is its point's smooth normal.
static const uint SMOOTH = 0xffffffff;

struct Vertex
{
    float4 position;
    float3 world;
    // In world space, not normalized; zero for a degenerate normal.
    float3 normal;
    // Whether the mesh's winding is reversed (src/transform.rs Instance).
    uint flip;
};

float3 load3(float *values, uint i)
{
    return float3(values[3 * i], values[3 * i + 1], values[3 * i + 2]);
}

// SV_VulkanVertexID and SV_VulkanInstanceID include the draw's vertex offset
// and first instance, which select the mesh's vertices and its visible slot.
Vertex pull(uint vertex, uint slot)
{
    uint2 v = view->vertices[vertex];
    float4 *m = view->instances + 7 * view->visible[slot];
    float4 p = float4(load3(view->points, v.x), 1);
    float4 world = float4(dot(m[0], p), dot(m[1], p), dot(m[2], p), 1);
    float3 n = v.y == SMOOTH ? load3(view->smooth, v.x) : load3(view->normals, v.y);

    Vertex o;
    o.position = project(view, world);
    o.world = world.xyz;
    o.normal = float3(dot(m[3].xyz, n), dot(m[4].xyz, n), dot(m[5].xyz, n));
    o.flip = asuint(m[6].x) & 1;
    return o;
}

// A unit tangent orthogonal to `n`, which stands in until the meshes'
// own tangents reach the shader with normal maps.
float3 tangent(float3 n)
{
    float3 axis = abs(n.y) < 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t = cross(axis, n);
    return dot(t, t) > 0 ? normalize(t) : float3(1, 0, 0);
}
