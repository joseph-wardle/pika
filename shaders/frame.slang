// What the pixel stage the bake tool generates for a material reads of the
// frame: the eye and the lights, from the view every pass reads, under the
// names MaterialX's generator gives the uniforms it would put in the
// material's own block (bake/generate.cpp leaves them out of it), so they
// are written once a frame rather than once a material.

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
#define u_numActiveLightSources LIGHTS
#define u_lightData view->lights
