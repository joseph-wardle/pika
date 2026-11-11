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

// A light's virtual shadow map (src/shadow.rs Shadow), of tiles of PAGE² texels; a tile drawn
// this frame holds a page of the pool: depth from the light as float bits, so the nearest
// caster is the least. A distant light's is a clipmap of up to LEVELS square levels around
// the camera, each TILES² tiles, level i's side 2^(level + i) world units, its depth in
// [0, 1] over the casters. A local light's is a cube of faces around its centre, each face's
// LOD l (TILES >> l)² tiles, its depth the distance along the face's axis.
static const uint LEVELS = 16;
static const uint TILES = 32;
static const uint PAGE = 256;
static const uint TILEMAP = TILES * TILES;
static const uint LOCAL_LEVELS = 6;
// A tile with no page; before allocation, 0 is unmarked.
static const uint NO_PAGE = 0xffffffff;

struct Shadow
{
    // World to light space, rows of the column-vector matrix. A distant light's: x and y in
    // world units across the light, z its depth in [0, 1] over the casters. A local light's:
    // its frame about its centre, z out of a rect or disk light's lit side.
    float4 light_space[3];
    // Per level, its first tile counted from the light-space origin, in that level's tiles;
    // 0 for a local light.
    int2 origin[LEVELS];
    // A distant light's finest level's exponent, and how many levels follow it.
    int level;
    uint levels;
    // A pixel's world width per unit of camera clip w.
    float pixel;
    // The shadow-link bit casters must carry, 0 when every mesh casts.
    uint link;
    // 0 for a distant light, else 6, or 5 for a rect or disk light, whose -z face sees
    // nothing it lights.
    uint faces;
    // A local light's clipping distances along a face's axis.
    float near;
    float far;
    uint pad;
    // LEVELS × TILEMAP per shadow (start()): a page or NO_PAGE.
    uint *tiles;
    Atomic<uint> *pool;
};

float3 to_light(Shadow *s, float3 P)
{
    float4 p = float4(P, 1);
    return float3(dot(s->light_space[0], p), dot(s->light_space[1], p), dot(s->light_space[2], p));
}

// Tiles across level i, or a local light's LOD i.
uint tiles_across(Shadow *s, uint i)
{
    return s->faces == 0 ? TILES : TILES >> i;
}

// Where level i's tiles start in a shadow's tiles: a distant light's level by level, a local
// light's LOD by LOD, six faces each, face by face, then row by row.
uint start(Shadow *s, uint i)
{
    if (s->faces == 0)
        return i * TILEMAP;
    // 6 × the tiles of LODs 0 to i - 1: 4096 (1 - 4^-i) / 3 × 6.
    return 8192 - (8192 >> (2 * i));
}

uint tile_index(Shadow *s, uint i, uint face, int2 t)
{
    uint across = tiles_across(s, i);
    return start(s, i) + (face * across + t.y) * across + t.x;
}

// A face's axes in a local light's space: x and y across it, z along it.
float3 on_face(uint face, float3 q)
{
    switch (face)
    {
    case 0: return q;
    case 1: return float3(q.y, q.z, q.x);
    case 2: return float3(q.y, q.z, -q.x);
    case 3: return float3(q.z, q.x, q.y);
    case 4: return float3(q.z, q.x, -q.y);
    default: return float3(q.x, q.y, -q.z);
    }
}

// The face a direction from the light's centre meets, 0 for +z (on_face's order).
uint face_of(float3 q)
{
    float3 a = abs(q);
    if (a.z >= a.x && a.z >= a.y)
        return q.z > 0 ? 0 : 5;
    if (a.x >= a.y)
        return q.x > 0 ? 1 : 2;
    return q.y > 0 ? 3 : 4;
}

// A point across a face, in [-1, 1], in its tiles at LOD i.
float2 face_tiles(float2 uv, uint across)
{
    return (uv * 0.5 + 0.5) * float(across);
}

// A tile's world width at level i: a power of two, so dividing by it is exact.
float tile_width(Shadow *s, uint i)
{
    return exp2(float(s->level + int(i))) / TILES;
}

// Outside the level, a tile index just past its edge.
int2 tile_at(Shadow *s, float2 q, uint i)
{
    float2 t = floor(q / tile_width(s, i)) - float2(s->origin[i]);
    return int2(clamp(t, -1, float(TILES)));
}

bool inside(int2 t)
{
    return all(t >= 0) && all(t < int(TILES));
}

// The finest level whose texels are no wider than a pixel at P, w its camera clip w, and
// whose tiles hold P; the coarsest holds every mesh (src/shadow.rs).
uint level_at(Shadow *s, float3 q, float w)
{
    float want = floor(log2(w * s->pixel * TILES * PAGE)) - float(s->level);
    uint i = uint(clamp(want, 0, float(s->levels - 1)));
    while (i + 1 < s->levels && !inside(tile_at(s, q.xy, i)))
        i++;
    return i;
}

// The finest LOD whose texels are no wider than a pixel at a point depth along its face from
// a local light; a face is 2 depth wide.
uint lod_at(Shadow *s, float depth, float w)
{
    float want = floor(log2(w * s->pixel * TILES * PAGE / (2 * depth)));
    return uint(clamp(want, 0, float(s->levels - 1)));
}

// A texel's world width at level i; for a local light, at a point depth along its face.
float texel_width(Shadow *s, uint i, float depth)
{
    return s->faces == 0 ? tile_width(s, i) / PAGE : 2 * depth * exp2(float(i)) / (TILES * PAGE);
}

// Past its reach it has faded to 0; a rect or disk lights only the cross(u, v) side.
bool lit_by(Light light, float3 P, uint link)
{
    if ((light.link & link) != light.link)
        return false;
    if (light.shape == DISTANT)
        return true;
    float3 d = P - light.position;
    if (dot(d, d) >= light.reach * light.reach)
        return false;
    return (light.shape != RECT && light.shape != DISK) || dot(d, cross(light.u, light.v)) > 0;
}

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
    // Each shadowed light's, at its Light.shadow.
    Shadow *shadows;
};

float4 project(View *view, float4 world)
{
    return float4(dot(view->view_projection[0], world), dot(view->view_projection[1], world),
                  dot(view->view_projection[2], world), dot(view->view_projection[3], world));
}
