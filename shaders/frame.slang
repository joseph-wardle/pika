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
uint2 tile(float3 P)
{
    float4 clip = project(view, float4(P, 1));
    float2 pixel = (clip.xy / clip.w * 0.5 + 0.5) * float2(view->extent);
    uint2 grid = (view->extent + TILE - 1) / TILE;
    uint2 xy = min(uint2(max(pixel, 0) / TILE), grid - 1);
    return view->tiles[xy.y * grid.x + xy.x];
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

// Where P falls in level i's tiles: the tile, P's place across it in [0, 1] and its depth
// from the light. False off the level, or off a local light's faces.
bool locate(Shadow *s, float3 P, uint i, out uint tile, out float2 across, out float depth)
{
    float3 q = to_light(s, P);
    tile = 0;
    across = 0;
    depth = 0;
    if (s->faces == 0)
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

// Whether level i holds a page at P moved a texel of that level off the surface toward the
// light, so the surface does not shadow itself; if so, lit is how much of the light reaches.
// depth is P's along its face of a local light.
bool lookup(Shadow *s, float3 P, float3 n, uint i, float depth, out float lit)
{
    lit = 1.0;
    uint tile;
    float2 across;
    float moved;
    if (!locate(s, P + n * texel_width(s, i, depth), i, tile, across, moved))
        return false;
    uint page = s->tiles[tile];
    if (page == NO_PAGE)
        return false;
    uint2 x = min(uint2(across * PAGE), PAGE - 1);
    uint caster = s->pool[(page * PAGE + x.y) * PAGE + x.x].load();
    lit = caster < asuint(moved) ? 0.0 : 1.0;
    return true;
}

// How much of light light_index reaches P: none where its map holds a caster nearer the
// light. Past the pool a level may have no page there, so coarser ones are tried, then the
// finer one a pixel on a level's edge may have marked instead.
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
    // A local light's texels widen with the depth along P's own face.
    float depth = on_face(face_of(q), q).z;
    uint first = s->faces == 0 ? level_at(s, q, w) : lod_at(s, depth, w);
    float lit;
    for (uint i = first; i < s->levels; i++)
    {
        if (lookup(s, P, n, i, depth, lit))
            return lit;
    }
    if (first > 0 && lookup(s, P, n, first - 1, depth, lit))
        return lit;
    return 1.0;
}
