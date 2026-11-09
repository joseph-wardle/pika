// What the pixel stage the generate tool writes for a material reads of the
// frame, from the view every pass reads, so it is written once a frame
// rather than once a material: the eye, under the name MaterialX's
// generator gives the uniform it would put in the material's own block
// (bake/generate.cpp leaves it out), and the lights, through the functions
// Pika's light loop calls (bake/generate.cpp Surface).

#pragma once

// vertex.slang declares the view, the one push constant.
#include "vertex.slang"

// The generated code takes the eye as a position. An orthographic camera's
// eye is the direction toward the viewer, so it is put far along it, where
// the view vector is that direction from everywhere drawn.
float3 eye()
{
    return view->eye.w == 0.0 ? view->eye.xyz * 1e7 : view->eye.xyz;
}

#define u_viewPosition eye()

// What a light gives a point: the unit direction toward the light and the
// illuminance it gives a surface there facing it.
struct Incident
{
    float3 direction;
    float3 illuminance;
};

// The tile of the lights that may reach P: the one its pixel is in, found
// from P rather than the pixel, which the loop is not given.
uint2 tile(float3 P)
{
    float4 clip = project(view, float4(P, 1));
    float2 pixel = (clip.xy / clip.w * 0.5 + 0.5) * float2(view->extent);
    uint2 grid = (view->extent + TILE - 1) / TILE;
    uint2 xy = min(uint2(max(pixel, 0) / TILE), grid - 1);
    return view->tiles[xy.y * grid.x + xy.x];
}


// The illuminance vector at P of a rect, disk or cylinder of radiance 1: a
// surface there with normal N gets dot(N, it), exactly while the light is
// wholly above the surface's horizon. It is Lambert's formula over the
// light's outline as P sees it, a polygon of up to 8 corners: a disk's
// matches its area, and a cylinder's follows its silhouette (src/light.rs
// area).
float3 illuminance_vector(Light light, float3 P)
{
    float3 c = light.position - P;
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
        // The silhouette's ends are arcs between the tangents from P, four
        // points each, the middle two pushed out so the chords touch the arc
        // midway.
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
        // Behind its plane, it gives nothing.
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

// illuminance_vector of a light at c from P, past its far: a point at its
// centre showing its projected area, of 4 |u × v| for a rect, the octagon's
// 2√2 |u × v| for a disk and 2 radius × 2 |u| side on for a cylinder, whose
// ends give nothing.
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

// The lights that light P, on a mesh in the light-link collections of mask
// `link`, one by one: the loop's whole view of which lights there are, so
// that how they are found changes this file alone. Its tile lists those that
// may reach P's pixel anywhere in depth; the rest are skipped here, before
// the BSDF runs, where they give exactly 0.
struct Lights
{
    float3 P;
    uint link;
    // The tile's slots in tile_lights still to walk.
    uint next_slot;
    uint end_slot;

    // Writes the next light's index in view->lights, or returns false.
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

Lights lights_at(float3 P, int link)
{
    uint2 t = tile(P);
    Lights lights;
    lights.P = P;
    lights.link = uint(link);
    lights.next_slot = t.x;
    lights.end_slot = t.x + t.y;
    return lights;
}

// Whether P, on a mesh in the collections of mask `link`, is lit by `light`.
// It must be linked to everything or to one of them (src/link.rs). Past its
// reach its illuminance has faded to 0, and a rect or disk lights only the
// side cross(u, v) points to (illuminance_vector gives 0 behind it).
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

// What light light_index gives P, on a surface with normal N.
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
        // A sphere gives what a point at its centre does, exactly while it
        // is above the surface's horizon (src/light.rs sphere). Inside, it
        // gives no more than at its surface, less what it gives at its
        // reach, so it fades to 0 there.
        float r = light.radius;
        incident.illuminance = light.illuminance * max(1 / max(d2, r * r) - light.fade, 0);
        return incident;
    }
    // An area light shines along its illuminance vector, as the BSDF's
    // cosine wants for diffuse light; specular takes it as a point's.
    float3 c = light.position - P;
    float3 e = dot(c, c) >= light.far * light.far ? far_illuminance_vector(light, c)
                                                  : illuminance_vector(light, P);
    float magnitude = length(e);
    incident.direction = magnitude > 0 ? e / magnitude : N;
    incident.illuminance = light.illuminance * max(magnitude - light.fade, 0);
    return incident;
}

// How much of light light_index reaches P: all of it, until shadows.
float visibility(uint light_index, float3 P)
{
    return 1.0;
}
