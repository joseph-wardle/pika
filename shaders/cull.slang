// The cull pass: for one view, each mesh instance whose box is inside the
// view's frustum is added to its draw record's visible instances.
// src/geometry.rs writes the records with no instances; this pass counts
// them, and shaders/mesh.slang reads each instance through the visible list.

struct Push
{
    // World to clip space (src/transform.rs Camera), depth reversed.
    float4 view_projection[4];
    // 7 per instance (src/transform.rs Instance).
    float4 *instances;
    // Each instance's draw record.
    uint *owners;
    // Per record: the low corner of its mesh's box, then the high one.
    float *boxes;
    // vk::DrawIndexedIndirectCommand, 5 per record: index_count,
    // instance_count, first_index, vertex_offset, first_instance.
    Atomic<uint> *records;
    // Indexed by first_instance and a record's slots.
    uint *visible;
    // The instances and triangles drawn.
    Atomic<uint> *drawn;
    uint first;
    uint count;
    uint stride;
    uint unused;
};

[[vk::push_constant]]
ConstantBuffer<Push> push;

float4 clip(float4 *m, float3 p)
{
    float4 world = float4(dot(m[0], float4(p, 1)), dot(m[1], float4(p, 1)), dot(m[2], float4(p, 1)), 1);
    return float4(dot(push.view_projection[0], world), dot(push.view_projection[1], world),
                  dot(push.view_projection[2], world), dot(push.view_projection[3], world));
}

// Whether the box may be seen: false only when all 8 corners are outside
// one plane of -w <= x, y <= w and 0 <= z <= w. A box crossing the eye
// plane has corners with w < 0, for which the planes still hold, so it is
// kept unless a plane has every corner outside. Testing the centre, or
// corners against different planes, would drop boxes that are partly seen.
bool seen(float4 *m, float3 lo, float3 hi)
{
    // Bit k: some corner is inside plane k.
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
    for (uint i = thread.x; i < push.count; i += push.stride)
    {
        uint instance = push.first + i;
        uint r = push.owners[instance];
        float *box = push.boxes + 6 * r;
        if (!seen(push.instances + 7 * instance, float3(box[0], box[1], box[2]),
                  float3(box[3], box[4], box[5])))
            continue;
        uint slot = push.records[5 * r + 1].increment();
        push.visible[push.records[5 * r + 4].load() + slot] = instance;
        push.drawn[0].increment();
        push.drawn[1].add(push.records[5 * r].load() / 3);
    }
}
