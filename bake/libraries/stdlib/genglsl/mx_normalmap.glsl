// Modified from MaterialX 1.39.5's libraries/stdlib/genglsl/mx_normalmap.glsl
// (Academy Software Foundation, Apache License 2.0: bake/libraries/NOTICE).
//
// A map decoding to the zero vector, as a missing UDIM tile or file reads, leaves the normal as
// it is. The library's normalize would make a NaN of it.
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
