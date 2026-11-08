// One view, as its passes read it: the camera, the lights, src/geometry.rs's
// buffers and the view's own. src/view.rs writes it each frame into a
// buffer whose address is a pass's one push constant: push constants are
// 128 bytes at minimum, which the camera and the addresses alone filled,
// so what lights and materials add goes here.

#pragma once

// A vertex's normal index when its normal is its point's smooth normal.
static const uint SMOOTH = 0xffffffff;
// A vertex's texcoord index when its mesh has none.
static const uint NONE = 0xffffffff;

// A vertex's tangent in 4 bytes: x, y and z as signed 10-bit fractions of
// 511 from the low bits up, and its handedness, 1 or -1, in the top two
// (Vulkan's A2B10G10R10_SNORM). 10 bits turn a unit vector by at most 0.1
// degree, under what a normal map's 8-bit texels resolve.
uint pack_tangent(float4 t)
{
    uint3 xyz = uint3(int3(round(clamp(t.xyz, -1, 1) * 511))) & 0x3ff;
    uint w = t.w < 0 ? 3 : 1;
    return xyz.x | xyz.y << 10 | xyz.z << 20 | w << 30;
}

float4 unpack_tangent(uint packed)
{
    int3 xyz = (int3(packed) << int3(22, 12, 2)) >> 22;
    return float4(float3(xyz) / 511, float(int(packed) >> 30));
}

// A light as the light loop reads it (shaders/frame.slang): the unit
// direction toward it and the illuminance it gives a surface facing it.
struct Light
{
    float3 direction;
    float3 illuminance;
};

struct View
{
    // World to clip space (src/transform.rs Camera): rows of the
    // column-vector matrix, depth reversed.
    float4 view_projection[4];
    // The camera position (w = 1), or the direction toward it (w = 0).
    float4 eye;
    // The unit direction the camera looks along.
    float4 forward;
    // light_count of them (src/view.rs Light).
    Light *lights;
    uint light_count;
    // Keeps the addresses after it 8-byte aligned, as src/view.rs lays them out.
    uint pad;
    float *points;
    float *normals;
    // One per point (src/geometry.rs Layout).
    float *smooth;
    uint2 *vertices;
    // Per texcoord value, and the texcoord index (or NONE) per vertex.
    float *texcoord;
    uint *vertex_texcoord;
    // Per vertex: its tangent and handedness, packed (shaders/tangent.slang).
    uint *tangents;
    // 7 per instance (src/transform.rs Instance).
    float4 *instances;
    // Each instance's draw record.
    uint *owners;
    // Per record: the low corner of its mesh's box, then the high one.
    float *boxes;
    // vk::DrawIndexedIndirectCommand, 5 per record: index_count,
    // instance_count, first_index, vertex_offset, first_instance. The cull
    // pass counts the instances it keeps.
    Atomic<uint> *records;
    // The instances the cull pass kept, at each record's first instance.
    uint *visible;
    // The instances and triangles drawn.
    Atomic<uint> *drawn;
};

// A world-space position in the view's clip space.
float4 project(View *view, float4 world)
{
    return float4(dot(view->view_projection[0], world), dot(view->view_projection[1], world),
                  dot(view->view_projection[2], world), dot(view->view_projection[3], world));
}
