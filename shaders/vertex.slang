// Vertex pulling for every pass that draws mesh instances.

#pragma once

#include "view.slang"

[[vk::push_constant]]
View *view;

struct Vertex
{
    float4 position;
    float3 world;
    // World space, not normalized; zero for a degenerate normal.
    float3 normal;
    // Zero where the mesh has none.
    float2 texcoord;
    // World space, not normalized.
    float3 tangent;
    float3 bitangent;
    uint flip;
    // One bit per light-link collection the mesh is in.
    uint link;
};

float3 load3(float *values, uint i)
{
    return float3(values[3 * i], values[3 * i + 1], values[3 * i + 2]);
}

// The Vulkan IDs include the draw's vertex offset and first instance.
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
    o.link = asuint(m[6].y);
    return o;
}
