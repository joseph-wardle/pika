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

// The lights that may reach P.
uint light_count(float3 P)
{
    return tile(P).y;
}

// What light light_index of P's tile gives P, on a surface with normal N.
Incident incident_light(uint light_index, float3 P, float3 N)
{
    Light light = view->lights[view->tile_lights[tile(P).x + light_index]];
    Incident incident;
    if (isinf(light.reach))
    {
        incident.direction = light.position;
        incident.illuminance = light.illuminance;
        return incident;
    }
    float3 toward = light.position - P;
    float d2 = dot(toward, toward);
    incident.direction = toward * rsqrt(d2);
    // A sphere gives what a point at its centre does, exactly while it is
    // above the surface's horizon (src/light.rs sphere). Inside, it gives
    // no more than at its surface, less what it gives at its reach, so it
    // fades to 0 there.
    float r = light.radius;
    float reach = light.reach;
    incident.illuminance = light.illuminance * max(1 / max(d2, r * r) - 1 / (reach * reach), 0);
    return incident;
}

// How much of light light_index reaches P: all of it, until shadows.
float visibility(uint light_index, float3 P)
{
    return 1.0;
}
