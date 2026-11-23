// Modified from MaterialX 1.39.5's libraries/pbrlib/genglsl/lib/mx_environment_none.glsl
// (Academy Software Foundation, Apache License 2.0: bake/libraries/NOTICE).
//
// mx_environment_irradiance is shaders/frame.slang's, which reads the view's dome light. The
// include is named from the search path, as its neighbours are not beside this file.
#include "libraries/pbrlib/genglsl/lib/mx_microfacet_specular.glsl"

vec3 mx_environment_radiance(vec3 N, vec3 V, vec3 X, vec2 roughness, int distribution, FresnelData fd)
{
    return vec3(0.0);
}
