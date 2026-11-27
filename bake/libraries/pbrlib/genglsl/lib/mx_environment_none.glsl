// Modified from MaterialX 1.39.5's libraries/pbrlib/genglsl/lib/mx_environment_none.glsl
// (Academy Software Foundation, Apache License 2.0: bake/libraries/NOTICE).
//
// mx_environment_irradiance is frame.slang's, and mx_environment_radiance reads its dome_radiance.
// The include is named from the search path, as its neighbours aren't beside this file.
#include "libraries/pbrlib/genglsl/lib/mx_microfacet_specular.glsl"

vec3 mx_environment_radiance(vec3 N, vec3 V, vec3 X, vec2 alpha, int distribution, FresnelData fd)
{
    N = mx_forward_facing_normal(N, V);
    vec3 L = fd.refraction ? mx_refraction_solid_sphere(-V, N, fd.ior.x) : -reflect(V, N);
    // Total internal reflection in the sphere leaves no exit, and refract() returns zero.
    if (dot(L, L) == 0.0)
        return vec3(0.0);

    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);

    float avgAlpha = mx_average_alpha(alpha);
    vec3 FG = mx_ggx_dir_albedo(NdotV, avgAlpha, fd);
    FG = fd.refraction ? vec3(1.0) - FG : FG;

    return dome_radiance(N, L, avgAlpha) * FG;
}
