// Lengyel's tangents. w is -1 on a mirrored mapping. A vertex with no texcoord
// area still gets a unit tangent orthogonal to its normal, so a frame exists.

#include "view.slang"

struct Push
{
    float *points;
    float *normals;
    float *smooth;
    uint2 *vertices;
    float *texcoord;
    uint *vertex_texcoord;
    // Vertex v's corners, as positions in `indices`, are offsets[v]..offsets[v + 1].
    uint *offsets;
    uint *corners;
    // Counted from the mesh's first vertex.
    uint *indices;
    uint *tangents;
    uint first;
    uint count;
    uint stride;
};

[[vk::push_constant]]
ConstantBuffer<Push> push;

float3 load3(float *values, uint i)
{
    return float3(values[3 * i], values[3 * i + 1], values[3 * i + 2]);
}

float3 point(uint vertex)
{
    return load3(push.points, push.vertices[vertex].x);
}

float2 texcoord(uint vertex)
{
    uint t = push.vertex_texcoord[vertex];
    return float2(push.texcoord[2 * t], push.texcoord[2 * t + 1]);
}

float3 orthogonal(float3 n)
{
    float3 axis = abs(n.y) < 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t = cross(axis, n);
    return dot(t, t) > 0 ? normalize(t) : float3(1, 0, 0);
}

// A grid-stride loop: Vulkan guarantees only 65535 groups per dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void cs(uint3 thread : SV_DispatchThreadID)
{
    for (uint i = thread.x; i < push.count; i += push.stride)
    {
        uint v = push.first + i;
        uint2 pn = push.vertices[v];
        float3 p = load3(push.points, pn.x);
        float3 n = pn.y == SMOOTH ? load3(push.smooth, pn.x) : load3(push.normals, pn.y);
        n = dot(n, n) > 0 ? normalize(n) : float3(0);
        float3 u = float3(0), w = float3(0);
        if (push.vertex_texcoord[v] != NONE)
        {
            float2 uv = texcoord(v);
            for (uint e = push.offsets[v]; e < push.offsets[v + 1]; e++)
            {
                uint c = push.corners[e];
                uint k = c % 3;
                uint tri = c - k;
                uint base = v - push.indices[c];
                uint a = base + push.indices[tri + (k + 1) % 3];
                uint b = base + push.indices[tri + (k + 2) % 3];
                float3 e1 = point(a) - p, e2 = point(b) - p;
                float2 d1 = texcoord(a) - uv, d2 = texcoord(b) - uv;
                float det = d1.x * d2.y - d2.x * d1.y;
                if (det == 0)
                    continue;
                u += (e1 * d2.y - e2 * d1.y) / det;
                w += (e2 * d1.x - e1 * d2.x) / det;
            }
        }
        u -= n * dot(n, u);
        float4 t = float4(orthogonal(n), 1);
        if (dot(u, u) > 0)
        {
            t.xyz = normalize(u);
            t.w = dot(cross(n, t.xyz), w) < 0 ? -1 : 1;
        }
        push.tangents[v] = pack_tangent(t);
    }
}
