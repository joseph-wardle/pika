// The frame data a generated pixel stage reads, the eye and the lights, written once a
// frame in the view rather than in each material's block.

#pragma once

#include "vertex.slang"

// An orthographic eye goes far along its direction, so the view vector is that everywhere.
float3 eye()
{
    return view->eye.w == 0.0 ? view->eye.xyz * 1e7 : view->eye.xyz;
}

#define u_viewPosition eye()

struct Incident
{
    float3 direction;
    float3 illuminance;
};

// Found from P rather than the pixel, which the loop is not given.
float2 pixel_at(float3 P)
{
    float4 clip = project(view, float4(P, 1));
    return (clip.xy / clip.w * 0.5 + 0.5) * float2(view->extent);
}

uint2 tile(float3 P)
{
    float2 pixel = pixel_at(P);
    uint2 grid = (view->extent + LIGHT_TILE - 1) / LIGHT_TILE;
    uint2 xy = min(uint2(max(pixel, 0) / LIGHT_TILE), grid - 1);
    return view->light_tiles[xy.y * grid.x + xy.x];
}

// Lambert's formula over the light's outline as P sees it, a polygon of up to 8 corners;
// dot(N, it) is exact while the light is wholly above the horizon.
float3 illuminance_vector(Light light, float3 c)
{
    float3 corners[8];
    uint n = 8;
    if (light.shape == CYLINDER)
    {
        float3 axis = normalize(light.u);
        float3 out = dot(c, axis) * axis - c;
        float rho = length(out);
        // Inside it, or past an end within its radius, P sees no side.
        if (rho <= light.radius)
            return 0;
        out /= rho;
        float3 side = cross(axis, out);
        // Each end arc gets four points, the middle two pushed out so chords touch it midway.
        float phi = acos(light.radius / rho);
        for (uint k = 0; k < 4; k++)
        {
            float t = phi * (1 - 2 * float(k) / 3);
            float r = k == 0 || k == 3 ? light.radius : light.radius / cos(phi / 3);
            float3 arc = r * (cos(t) * out + sin(t) * side);
            corners[k] = c + light.u + arc;
            corners[7 - k] = c - light.u + arc;
        }
    }
    else
    {
        if (dot(c, cross(light.u, light.v)) >= 0)
            return 0;
        float3 u = light.u;
        float3 v = light.v;
        if (light.shape == RECT)
        {
            n = 4;
            corners = { c + u + v, c - u + v, c - u - v, c + u - v, 0, 0, 0, 0 };
        }
        else
        {
            float3 a = (u + v) * sqrt(0.5);
            float3 b = (v - u) * sqrt(0.5);
            corners = { c + u, c + a, c + v, c + b, c - u, c - a, c - v, c - b };
        }
    }
    float3 sum = 0;
    for (uint i = 0; i < n; i++)
    {
        float3 a = corners[i];
        float3 b = corners[(i + 1) % n];
        float3 k = cross(a, b);
        float s = length(k);
        // atan2 keeps the small angles of a far light that acos loses.
        if (s > 0)
            sum += atan2(s, dot(a, b)) / s * k;
    }
    // Wound either way round, it points toward the light.
    return 0.5 * (dot(sum, c) < 0 ? -sum : sum);
}

// Past its far: a point showing its projected area, 4 |u × v| for a rect, 2√2 |u × v|
// for the octagon, 2 radius × 2 |u| side on for a cylinder.
float3 far_illuminance_vector(Light light, float3 c)
{
    float d2 = dot(c, c);
    float3 toward = c * rsqrt(d2);
    float area;
    if (light.shape == CYLINDER)
        area = 4 * light.radius * length(cross(light.u, toward));
    else
    {
        float3 n = cross(light.u, light.v);
        float flat = light.shape == RECT ? 4 : 2 * sqrt(2.0);
        area = flat * max(-dot(toward, n), 0);
    }
    return toward * (area / d2);
}

// Lights skipped here, outside P's tile or link mask, would give exactly 0.
struct Lights
{
    float3 P;
    uint link;
    uint next_slot;
    uint end_slot;

    [mutating]
    bool next(out uint light_index)
    {
        while (next_slot < end_slot)
        {
            light_index = view->tile_lights[next_slot++];
            if (lit_by(view->lights[light_index], P, link))
                return true;
        }
        light_index = 0;
        return false;
    }
}

// The geometric normal, which the shadow lookup offsets along. Derivatives need the uniform
// control flow lights_at() runs in, not the light loop's; the loop gives visibility() only P,
// and changing the call bake/generate.cpp emits re-bakes every texture.
static float3 shadow_normal;

Lights lights_at(float3 P, int link)
{
    float3 g = cross(ddx(P), ddy(P));
    shadow_normal = dot(g, g) > 0 ? normalize(g) : float3(0);
    uint2 t = tile(P);
    Lights lights;
    lights.P = P;
    lights.link = asuint(link);
    lights.next_slot = t.x;
    lights.end_slot = t.x + t.y;
    return lights;
}

Incident incident_light(uint light_index, float3 P, float3 N)
{
    Light light = view->lights[light_index];
    Incident incident;
    if (light.shape == DISTANT)
    {
        incident.direction = light.position;
        incident.illuminance = light.illuminance;
        return incident;
    }
    if (light.shape == SPHERE)
    {
        float3 toward = light.position - P;
        float d2 = dot(toward, toward);
        incident.direction = toward * rsqrt(d2);
        // Exact as a point while above the horizon. Inside, no more than at its surface,
        // less what it gives at its reach, so it fades to 0 there.
        float r = light.radius;
        incident.illuminance = light.illuminance * max(1 / max(d2, r * r) - light.fade, 0);
        return incident;
    }
    // Diffuse takes the illuminance vector, as the BSDF's cosine wants; specular a point's.
    float3 c = light.position - P;
    float3 e = dot(c, c) >= light.far * light.far ? far_illuminance_vector(light, c)
                                                  : illuminance_vector(light, c);
    float magnitude = length(e);
    incident.direction = magnitude > 0 ? e / magnitude : N;
    incident.illuminance = light.illuminance * max(magnitude - light.fade, 0);
    return incident;
}

// Where light-space q falls in level i's tiles: the tile, q's place across it in [0, 1] and
// its depth from the light. False off the level, or off a local light's faces.
bool locate(Shadow *s, float3 q, uint i, out uint tile, out float2 across, out float depth)
{
    tile = 0;
    across = 0;
    depth = 0;
    if (distant(s))
    {
        int2 t = tile_at(s, q.xy, i);
        if (!inside(t))
            return false;
        float2 u = q.xy / tile_width(s, i);
        tile = tile_index(s, i, 0, t);
        across = u - floor(u);
        // Negative depth, nearer the light than any caster, would read as a huge integer.
        depth = max(q.z, 0.0);
        return true;
    }
    uint face = face_of(q);
    if (face >= s->faces)
        return false;
    float3 f = on_face(face, q);
    uint n = tiles_across(s, i);
    float2 u = face_tiles(f.xy / f.z, n);
    int2 t = clamp(int2(floor(u)), 0, int(n) - 1);
    tile = tile_index(s, i, face, t);
    across = saturate(u - float2(t));
    depth = f.z;
    return true;
}

// A texel with no caster (shaders/shadow.slang clear).
static const uint EMPTY = 0xffffffff;

// What level i stores at light-space q moved a texel of that level along nq, off the surface
// toward the light so the surface does not shadow itself: the caster nearest the light, as
// float bits, and the moved q's depth. False where the level holds no page. along is q's depth
// along its face of a local light, which sets the texel's width.
bool read(Shadow *s, float3 q, float3 nq, uint i, float along, out uint caster, out float depth,
          out float texel)
{
    caster = EMPTY;
    texel = texel_width(s, i, along);
    uint tile;
    float2 across;
    if (!locate(s, q + nq * texel, i, tile, across, depth))
        return false;
    uint page = s->tiles[tile];
    if (page == NO_PAGE)
        return false;
    uint2 x = min(uint2(across * PAGE), PAGE - 1);
    caster = s->pool[(page * PAGE + x.y) * PAGE + x.x].load();
    return true;
}

// What the level a pixel at camera clip w wants at q reads (read()). Past the pool a level may
// have no page there, so coarser ones are tried, then the finer one a pixel on a level's edge
// may have marked instead.
bool lookup(Shadow *s, float3 q, float3 nq, float w, out uint caster, out float depth,
            out float texel)
{
    caster = EMPTY;
    depth = 0;
    texel = 0;
    // A local light's texels widen with the depth along q's own face.
    float along = on_face(face_of(q), q).z;
    uint first = distant(s) ? level_at(s, q, w) : lod_at(s, along, w);
    for (uint i = first; i < s->levels; i++)
    {
        if (read(s, q, nq, i, along, caster, depth, texel))
            return true;
    }
    return first > 0 && read(s, q, nq, first - 1, along, caster, depth, texel);
}

// A light with no size: q, P in light space, is shadowed where its map holds a caster nearer.
float hard(Shadow *s, float3 q, float3 nq, float w)
{
    uint caster;
    float depth;
    float texel;
    if (!lookup(s, q, nq, w, caster, depth, texel))
        return 1.0;
    return caster < asuint(depth) ? 0.0 : 1.0;
}

// World distance from the light to q, at depth, and to the caster its map holds on the line
// from the light through q.
float2 from_light(Shadow *s, float3 q, float depth, float caster)
{
    if (distant(s))
        return float2(depth, caster) / length(s->light_space[2].xyz);
    float d = length(q);
    return float2(d, d * caster / max(depth, 1e-30));
}

// Steps along a shadow ray, and the last at P.
static const uint STEPS = 6;

// Whether the ray from P, start in light space, to a point across the light, end, meets a
// caster: 0 if so. The map is drawn from the light's centre, so a ray toward another point
// may lie behind a caster's surface without meeting it, having passed under its edge. Marched
// from the light toward P, the ray meets the surface it lies behind if at the step before it
// lay nearer the light than that surface, or lay before another surface that came nearer the
// light no faster than the ray went away (a side the light sees). Otherwise it came in under
// the edge, so P inside the hard shadow of the light's centre is still lit by the part of the
// light past the edge. A step too long to tell counts as meeting it.
float soft(Shadow *s, float3 start, float3 end, float3 nq, float w, float jitter)
{
    // The ray's distance from the light at the step before, and the surface's there.
    float2 before = float2(-1e30, 0);
    for (uint k = 0; k <= STEPS; k++)
    {
        // Steps close in toward P, where a near caster's penumbra is narrow.
        float a = 1 - (k + jitter) / STEPS;
        float3 q = lerp(start, end, k == STEPS ? 0.0 : a * a);
        uint caster;
        float depth;
        float texel;
        if (!lookup(s, q, nq, w, caster, depth, texel) || caster == EMPTY)
            continue;
        float2 d = from_light(s, q, depth, asfloat(caster));
        bool crossed = before.x <= d.y + texel ||
                       (before.x <= before.y && before.y - d.y <= d.x - before.x + texel);
        if (d.x > d.y && crossed)
            return 0.0;
        before = d;
    }
    return 1.0;
}

// Jimenez's interleaved gradient noise: a fixed pattern over pixels, in [0, 1).
float noise(float2 pixel)
{
    return frac(52.9829189 * frac(dot(floor(pixel), float2(0.06711056, 0.00583715))));
}

// The pattern's three values at a pixel for the view's sample, each shifted per sample along
// Roberts' R3 sequence, so a pixel's samples spread evenly; sample 0's are the pattern itself.
// While anything changes it stays at sample 0: a pattern moving each frame would shimmer until
// temporal history averages it.
float3 pattern(float2 pixel)
{
    float3 shift = float(view->sample) * float3(0.8191725, 0.6710436, 0.5497005);
    return frac(float3(noise(pixel), noise(pixel + float2(113, 47)), noise(pixel + float2(59, 193))) + shift);
}

// A point across the light's shape from P, at u in [0, 1)², or for a distant light a point
// along the direction toward one across its disk: a rect's or cylinder's uniform over its
// side, a sphere's or disk's over its disk. A distant light's disk is kept within a tile of
// the level a pixel at camera clip w wants, so its ray stays over pages P's tile marked.
float3 across_light(Light light, Shadow *s, float3 P, float3 q, float w, float2 u)
{
    float2 disk = sqrt(u.x) * float2(cos(6.2831853 * u.y), sin(6.2831853 * u.y));
    float2 side = 2 * u - 1;
    float3 c = light.position;
    switch (light.shape)
    {
    case DISTANT:
    {
        float span = rcp(length(s->light_space[2].xyz));
        float wide = tile_width(s, level_at(s, q, w)) / max(q.z * span, 1e-30);
        float r = min(light.radius, wide);
        return P + light.position + r * (disk.x * s->light_space[0].xyz + disk.y * s->light_space[1].xyz);
    }
    case SPHERE:
    {
        float3 z = normalize(c - P);
        float3 x = normalize(cross(z, abs(z.x) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0)));
        return c + light.radius * (disk.x * x + disk.y * cross(z, x));
    }
    case RECT:
        return c + side.x * light.u + side.y * light.v;
    case DISK:
        // src/light.rs widens the disk's axes to its octagon's.
        return c + (disk.x * light.u + disk.y * light.v) / 1.0539074;
    default:
    {
        float3 out = cross(light.u, P - c);
        float l = length(out);
        return c + side.x * light.u + (l > 0 ? side.y * light.radius / l * out : 0);
    }
    }
}

// How much of light light_index reaches P: none where its map holds a caster nearer the
// light, for a light with a size along a ray toward one point across it, its pattern's.
float visibility(uint light_index, float3 P)
{
    Light light = view->lights[light_index];
    if (light.shadow == NO_SHADOW)
        return 1.0;
    Shadow *s = view->shadows + light.shadow;
    float3 toward = light.shape == DISTANT ? light.position : light.position - P;
    float3 n = dot(shadow_normal, toward) < 0 ? -shadow_normal : shadow_normal;
    float w = dot(view->view_projection[3], float4(P, 1));
    float3 q = to_light(s, P);
    float3 nq = float3(dot(s->light_space[0].xyz, n), dot(s->light_space[1].xyz, n),
                       dot(s->light_space[2].xyz, n));
    if (light.radius == 0 && light.shape != RECT && light.shape != DISK)
        return hard(s, q, nq, w);
    float3 u = pattern(pixel_at(P));
    float3 e = across_light(light, s, P, q, w, u.xy);
    // A point below P's horizon would have P shadow itself; the light's own evaluation leaves
    // it out already.
    float3 v = e - P;
    float below = dot(v, n);
    if (below < 0.01 * length(v))
        e += n * (0.01 * length(v) - below);
    float3 end;
    if (light.shape == DISTANT)
    {
        // To depth 0, nearer the light than any caster.
        float3 r = to_light(s, e) - q;
        end = q + r * (q.z / max(-r.z, 1e-30));
    }
    else
    {
        // To its near, inside which nothing is drawn.
        float l = length(e - P);
        end = to_light(s, P + (e - P) * max(l - s->near, 0) / l);
    }
    return soft(s, q, end, nq, w, u.z);
}
