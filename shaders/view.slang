// One view, as its passes read it: the camera, src/geometry.rs's buffers and
// the view's own. src/view.rs writes it each frame into a buffer whose
// address is a pass's one push constant: push constants are 128 bytes at
// minimum, which the camera and the addresses alone filled, so what
// materials and lights add goes here.

struct View
{
    // World to clip space (src/transform.rs Camera): rows of the
    // column-vector matrix, depth reversed.
    float4 view_projection[4];
    // The camera position (w = 1), or the direction toward it (w = 0).
    float4 eye;
    float *points;
    float *normals;
    // One per point (src/geometry.rs Layout).
    float *smooth;
    uint2 *vertices;
    // 7 per instance (src/transform.rs Instance).
    float4 *instances;
    // Each instance's draw record.
    uint *owners;
    // Per record: the low corner of its mesh's box, then the high one.
    float *boxes;
    // vk::DrawIndexedIndirectCommand, 5 per record: index_count,
    // instance_count, first_index, vertex_offset, first_instance. The cull
    // pass counts the instances it keeps.
    Atomic<uint> *records;
    // The instances the cull pass kept, at each record's first instance.
    uint *visible;
    // The instances and triangles drawn.
    Atomic<uint> *drawn;
};

// A world-space position in the view's clip space.
float4 project(View *view, float4 world)
{
    return float4(dot(view->view_projection[0], world), dot(view->view_projection[1], world),
                  dot(view->view_projection[2], world), dot(view->view_projection[3], world));
}
