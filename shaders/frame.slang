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

// The lights that may reach P.
uint light_count(float3 P)
{
    return view->light_count;
}

// What light light_index gives P, on a surface with normal N.
Incident incident_light(uint light_index, float3 P, float3 N)
{
    Light light = view->lights[light_index];
    Incident incident;
    incident.direction = light.direction;
    incident.illuminance = light.illuminance;
    return incident;
}

// How much of light light_index reaches P: all of it, until shadows.
float visibility(uint light_index, float3 P)
{
    return 1.0;
}
