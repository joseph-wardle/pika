// Modified from MaterialX 1.39.5's libraries/stdlib/genglsl/mx_normalmap.glsl
// (Academy Software Foundation, Apache License 2.0: bake/libraries/NOTICE).
//
// A map that decodes to the zero vector leaves the normal as it is, where the library's
// normalize makes a NaN of it. A UsdPreviewSurface hands its normal over encoded, so the zero
// its texture reads off a missing UDIM tile or a missing file's fallback passes the test for
// an unset map on the first line.
void mx_normalmap_vector2(vec3 value, vec2 normal_scale, vec3 N, vec3 T, vec3 B, out vec3 result)
{
    value = (dot(value, value) == 0.0) ? vec3(0.0, 0.0, 1.0) : value * 2.0 - 1.0;
    value = T * value.x * normal_scale.x +
            B * value.y * normal_scale.y +
            N * value.z;
    result = dot(value, value) > 0.0 ? normalize(value) : N;
}

void mx_normalmap_float(vec3 value, float normal_scale, vec3 N, vec3 T, vec3 B, out vec3 result)
{
    mx_normalmap_vector2(value, vec2(normal_scale), N, T, B, result);
}
