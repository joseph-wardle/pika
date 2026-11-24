// One view, read at the push constant's address. Push constants guarantee only 128 bytes.

#pragma once

static const uint SMOOTH = 0xffffffff;
static const uint NONE = 0xffffffff;

// A2B10G10R10_SNORM, handedness in the top two bits.
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

static const uint DISTANT = 0;
static const uint SPHERE = 1;
static const uint RECT = 2;
static const uint DISK = 3;
static const uint CYLINDER = 4;

// src/light.rs mirrors it.
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
    uint shadow;
    uint pad;
};

static const uint NO_SHADOW = 0xffffffff;

// Depth as float bits, nearest least. A distant light's level i spans 2^(finest + i) world units,
// its tiles wrapping modulo TILES so a level moving with the camera keeps those it still covers.
static const uint LEVELS = 16;
// At most 32, since shadow.slang keeps a row of tiles in a uint and a tile's place in 8 bits.
static const uint TILES = 32;
static const uint PAGE = 256;
static const uint TILEMAP = TILES * TILES;
// Before allocation, 0 means unmarked.
static const uint NO_PAGE = 0xffffffff;

struct Shadow
{
    // Rows of the column-vector matrix.
    float4 light_space[3];
    int2 origin[LEVELS];
    int finest;
    uint levels;
    // A pixel's world width per unit of clip w.
    float pixel;
    uint link;
    // 0 for distant, 5 for a rect or disk, whose -z face sees nothing it lights, else 6.
    uint faces;
    float near;
    float far;
    uint reset;
    uint *tiles;
    Atomic<uint> *pool;
};

float3 to_light(Shadow *s, float3 P)
{
    float4 p = float4(P, 1);
    return float3(dot(s->light_space[0], p), dot(s->light_space[1], p), dot(s->light_space[2], p));
}

bool distant(Shadow *s)
{
    return s->faces == 0;
}

uint face_count(Shadow *s)
{
    return max(s->faces, 1);
}

uint tiles_across(Shadow *s, uint i)
{
    return distant(s) ? TILES : TILES >> i;
}

// Distant levels in order. Local LODs in order, six faces each. Then row by row.
uint start(Shadow *s, uint i)
{
    if (distant(s))
        return i * TILEMAP;
    // 6 × the tiles of LODs 0 to i - 1, 4096 (1 - 4^-i) / 3 × 6
    return 8192 - (8192 >> (2 * i));
}

uint tile_index(Shadow *s, uint i, uint face, int2 t)
{
    uint across = tiles_across(s, i);
    if (distant(s))
        t = (t + s->origin[i]) & int(TILES - 1);
    return start(s, i) + (face * across + t.y) * across + t.x;
}

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

uint face_of(float3 q)
{
    float3 a = abs(q);
    if (a.z >= a.x && a.z >= a.y)
        return q.z > 0 ? 0 : 5;
    if (a.x >= a.y)
        return q.x > 0 ? 1 : 2;
    return q.y > 0 ? 3 : 4;
}

float2 face_tiles(float2 uv, uint across)
{
    return (uv * 0.5 + 0.5) * float(across);
}

// A power of two, so dividing by it is exact.
float tile_width(Shadow *s, uint i)
{
    return exp2(float(s->finest + int(i))) / TILES;
}

int2 tile_at(Shadow *s, float2 q, uint i)
{
    float2 t = floor(q / tile_width(s, i)) - float2(s->origin[i]);
    return int2(clamp(t, -1, float(TILES)));
}

bool inside(int2 t)
{
    return all(t >= 0) && all(t < int(TILES));
}

// The finest level whose texels are no wider than a pixel at P and whose tiles hold P.
uint level_at(Shadow *s, float3 q, float w)
{
    float want = floor(log2(w * s->pixel * TILES * PAGE)) - float(s->finest);
    uint i = uint(clamp(want, 0, float(s->levels - 1)));
    while (i + 1 < s->levels && !inside(tile_at(s, q.xy, i)))
        i++;
    return i;
}

uint lod_at(Shadow *s, float depth, float w)
{
    float want = floor(log2(w * s->pixel * TILES * PAGE / (2 * depth)));
    return uint(clamp(want, 0, float(s->levels - 1)));
}

float texel_width(Shadow *s, uint i, float depth)
{
    return distant(s) ? tile_width(s, i) / PAGE : 2 * depth * exp2(float(i)) / (TILES * PAGE);
}

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

static const uint2 IRRADIANCE = uint2(64, 32);

struct Dome
{
    float3 axes[3];
    float3 radiance;
    // 3 floats per texel from the top row, irradiance over π.
    float *irradiance;
    // Folded squares, 3 floats a texel: the sharp one, then each blurred level at half the width
    // down to REFLECTION_LEAST, then the sharp one's plain halvings.
    float *reflection;
};

static const uint REFLECTION = 1024;
static const uint REFLECTION_LEVELS = 8;
// Coarser levels crease along the fold's edges, where four texels blend across a bend.
static const uint REFLECTION_LEAST = 128;
static const float REFLECTION_ROUGHEST = 0.6;
static const float REFLECTION_MATTE = 0.9;
static const float REFLECTION_SPACING = 0.75;

// Octahedral, +y at the centre, -y at the corners.
float2 folded(float3 d)
{
    d /= abs(d.x) + abs(d.y) + abs(d.z);
    float2 p = d.y >= 0 ? d.xz : (1 - abs(d.zx)) * float2(d.x >= 0 ? 1 : -1, d.z >= 0 ? 1 : -1);
    return p * 0.5 + 0.5;
}

float3 from_folded(float2 uv)
{
    float2 p = uv * 2 - 1;
    float y = 1 - abs(p.x) - abs(p.y);
    float2 below = (1 - abs(p.yx)) * float2(p.x >= 0 ? 1 : -1, p.y >= 0 ? 1 : -1);
    return normalize(y >= 0 ? float3(p.x, y, p.y) : float3(below.x, y, below.y));
}

uint reflection_side(uint level)
{
    return max(REFLECTION >> level, REFLECTION_LEAST);
}

uint reflection_level(uint level)
{
    uint first = 0;
    for (uint l = 0; l < level; l++)
        first += reflection_side(l) * reflection_side(l);
    return first;
}

uint reflection_halving(uint halving)
{
    uint first = halving == 0 ? 0 : reflection_level(REFLECTION_LEVELS);
    for (uint h = 1; h < halving; h++)
        first += (REFLECTION >> h) * (REFLECTION >> h);
    return first;
}

float3 reflection_texel(float *square, int side, int2 t)
{
    if (t.x < 0 || t.x >= side)
        t = int2(clamp(t.x, 0, side - 1), side - 1 - t.y);
    if (t.y < 0 || t.y >= side)
        t = int2(side - 1 - t.x, clamp(t.y, 0, side - 1));
    float *texel = square + 3 * (t.y * side + t.x);
    return float3(texel[0], texel[1], texel[2]);
}

float3 reflection_square(float *square, int side, float2 uv)
{
    float2 at = uv * side - 0.5;
    int2 t = int2(floor(at));
    float2 f = at - floor(at);
    float3 a = lerp(reflection_texel(square, side, t), reflection_texel(square, side, t + int2(1, 0)), f.x);
    float3 b = lerp(reflection_texel(square, side, t + int2(0, 1)), reflection_texel(square, side, t + int2(1, 1)), f.x);
    return lerp(a, b, f.y);
}

// As OpenEXR and UsdLux lay it out: +y along the top row, +z at the centre, +x a quarter in.
float2 latlong(float3 d)
{
    // atan2 is undefined at a pole, where any column serves.
    float longitude = d.x == 0 && d.z == 0 ? 0 : atan2(d.x, d.z);
    return float2(0.5 - longitude / 6.2831853, acos(clamp(d.y, -1, 1)) / 3.1415927);
}

float3 from_latlong(float2 uv)
{
    float2 angle = (0.5 - uv) * float2(6.2831853, 3.1415927);
    return float3(cos(angle.y) * sin(angle.x), sin(angle.y), cos(angle.y) * cos(angle.x));
}

static const uint LIGHT_TILE = 16;

struct View
{
    // Rows of the column-vector matrix, depth reversed.
    float4 view_projection[4];
    // The camera position (w = 1), or the direction to it (w = 0).
    float4 eye;
    float4 previous_view_projection[4];
    Light *lights;
    uint2 *light_tiles;
    uint *tile_lights;
    uint2 extent;
    // In pixels from the pixel's centre.
    float2 jitter;
    uint pattern;
    uint unused;
    float *points;
    float *normals;
    float *smooth;
    uint2 *vertices;
    float *texcoord;
    uint *vertex_texcoord;
    uint *tangents;
    float4 *instances;
    float4 *previous_instances;
    float *previous_points;
    uint *owners;
    float *boxes;
    // The cull pass counts instance_count.
    Atomic<uint> *records;
    uint *visible;
    Atomic<uint> *drawn;
    Shadow *shadows;
    Dome dome;
};

float4 transform(float4 rows[4], float4 v)
{
    return float4(dot(rows[0], v), dot(rows[1], v), dot(rows[2], v), dot(rows[3], v));
}

float4 project(View *view, float4 world)
{
    return transform(view->view_projection, world);
}
