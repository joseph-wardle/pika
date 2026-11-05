// Vertex pulling for every pass that draws mesh instances: each vertex from
// src/geometry.rs's buffers by address, each instance through the view's
// visible list (shaders/cull.slang), so only the index buffer and the
// view's draw records are bound. mesh.slang uses it, and so does the vertex
// stage the bake tool writes for each material (bake/generate.cpp), whose
// pixel stage reads the view through it (frame.slang).

#pragma once

#include "view.slang"

[[vk::push_constant]]
View *view;

struct Vertex
{
    float4 position;
    float3 world;
    // In world space, not normalized; zero for a degenerate normal.
    float3 normal;
    // The float2 primvar the mesh's material reads; zero where the mesh has none.
    float2 texcoord;
    // In world space, not normalized: the directions the texcoord's u and v
    // increase in (shaders/tangent.slang), for normal maps.
    float3 tangent;
    float3 bitangent;
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
    uint t = view->vertex_texcoord[vertex];
    float4 tangent = unpack_tangent(view->tangents[vertex]);
    float3 bitangent = tangent.w * cross(n, tangent.xyz);

    Vertex o;
    o.position = project(view, world);
    o.world = world.xyz;
    o.normal = float3(dot(m[3].xyz, n), dot(m[4].xyz, n), dot(m[5].xyz, n));
    o.texcoord = t == NONE ? float2(0) : float2(view->texcoord[2 * t], view->texcoord[2 * t + 1]);
    // Surface directions move with the mesh, not its normals' inverse transpose.
    o.tangent = float3(dot(m[0].xyz, tangent.xyz), dot(m[1].xyz, tangent.xyz), dot(m[2].xyz, tangent.xyz));
    o.bitangent = float3(dot(m[0].xyz, bitangent), dot(m[1].xyz, bitangent), dot(m[2].xyz, bitangent));
    o.flip = asuint(m[6].x) & 1;
    return o;
}
