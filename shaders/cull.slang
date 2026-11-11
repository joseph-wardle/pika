#include "view.slang"

struct Push
{
    View *view;
    uint first;
    uint count;
    uint stride;
    // A light's shadow-link bit, which its casters must carry; 0 for the camera.
    uint shadow;
};

[[vk::push_constant]]
ConstantBuffer<Push> push;

float4 clip(float4 *m, float3 p)
{
    float4 world = float4(dot(m[0], float4(p, 1)), dot(m[1], float4(p, 1)), dot(m[2], float4(p, 1)), 1);
    return project(push.view, world);
}

// False only when all 8 corners are outside one clip plane; this stays right for
// boxes crossing the eye plane (w < 0), unlike testing the centre.
bool seen(float4 *m, float3 lo, float3 hi)
{
    uint inside = 0;
    for (uint c = 0; c < 8; c++)
    {
        float3 p = select((c & uint3(1, 2, 4)) != 0, hi, lo);
        float4 q = clip(m, p);
        inside |= uint(q.x >= -q.w) | uint(q.x <= q.w) << 1 | uint(q.y >= -q.w) << 2 |
                  uint(q.y <= q.w) << 3 | uint(q.z >= 0) << 4 | uint(q.z <= q.w) << 5;
    }
    return inside == 0x3f;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cs(uint3 thread : SV_DispatchThreadID)
{
    View *view = push.view;
    for (uint i = thread.x; i < push.count; i += push.stride)
    {
        uint instance = push.first + i;
        uint r = view->owners[instance];
        float *box = view->boxes + 6 * r;
        if ((asuint(view->instances[7 * instance + 6].z) & push.shadow) != push.shadow)
            continue;
        if (!seen(view->instances + 7 * instance, float3(box[0], box[1], box[2]),
                  float3(box[3], box[4], box[5])))
            continue;
        uint slot = view->records[5 * r + 1].increment();
        view->visible[view->records[5 * r + 4].load() + slot] = instance;
        view->drawn[0].increment();
        view->drawn[1].add(view->records[5 * r].load() / 3);
    }
}
