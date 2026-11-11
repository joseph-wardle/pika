// One view as its passes read it, at the one push constant's address: push constants
// guarantee only 128 bytes, which the camera and addresses alone filled.

#pragma once

// A vertex's normal index when its normal is its point's smooth normal.
static const uint SMOOTH = 0xffffffff;
// A vertex's texcoord index when its mesh has none.
static const uint NONE = 0xffffffff;

// x, y, z as signed 10-bit fractions of 511 from the low bits, handedness in the top two
// (A2B10G10R10_SNORM); 10 bits turn a unit vector by at most 0.1°, under 8-bit normal maps.
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

// A light's shape (src/light.rs).
static const uint DISTANT = 0;
static const uint SPHERE = 1;
static const uint RECT = 2;
static const uint DISK = 3;
static const uint CYLINDER = 4;

// A light as src/light.rs writes it, field for field.
struct Light
{
    float3 position;
    float radius;
    float3 illuminance;
    float reach;
    float3 u;
    uint shape;
    float3 v;
    float fade;
    uint link;
    float far;
    // Its map in the view's shadows, or NO_SHADOW.
    uint shadow;
    uint pad;
};

static const uint NO_SHADOW = 0xffffffff;

// A light's plain shadow map (src/shadow.rs Shadow): light depth, 0 toward the light,
// as float bits, so the nearest caster is the least.
struct Shadow
{
    // World to the map's clip space, orthographic: rows of the column-vector matrix.
    float4 view_projection[4];
    Atomic<uint> *map;
    // Texels a side, and a texel's width in world units.
    uint size;
    float texel;
};

// The side of a light tile in pixels (src/light.rs TILE).
static const uint TILE = 16;

struct View
{
    // World to clip space (src/transform.rs Camera): rows of the
    // column-vector matrix, depth reversed.
    float4 view_projection[4];
    // The camera position (w = 1), or the direction toward it (w = 0).
    float4 eye;
    Light *lights;
    // Per tile, row by row: the first of its tile_lights and how many.
    uint2 *tiles;
    uint *tile_lights;
    uint2 extent;
    float *points;
    float *normals;
    float *smooth;
    uint2 *vertices;
    // Per texcoord value, and the texcoord index (or NONE) per vertex.
    float *texcoord;
    uint *vertex_texcoord;
    uint *tangents;
    // 7 per instance (src/transform.rs Instance).
    float4 *instances;
    uint *owners;
    // Per record: the low corner of its mesh's box, then the high one.
    float *boxes;
    // vk::DrawIndexedIndirectCommand, 5 per record; the cull pass counts instance_count.
    Atomic<uint> *records;
    // The instances the cull pass kept, at each record's first instance.
    uint *visible;
    Atomic<uint> *drawn;
    // The camera's view reads every light's; a light's view draws into its own, the first.
    Shadow *shadows;
};

float4 project(View *view, float4 world)
{
    return float4(dot(view->view_projection[0], world), dot(view->view_projection[1], world),
                  dot(view->view_projection[2], world), dot(view->view_projection[3], world));
}
