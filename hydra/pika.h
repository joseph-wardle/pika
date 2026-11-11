// The C ABI of the Pika renderer. The delegate calls Pika, never the reverse.
// On a status other than PIKA_OK, pika_last_error() says why. Pointers are
// borrowed for the call only; a "" path means none. Tokens are USD's where it
// has one, else Hydra's name for the value (TfEnum::GetName). Matrices are
// row-major GfMatrix4d.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A panic poisons the renderer: every later call fails until it is freed.
enum { PIKA_OK = 0, PIKA_ERROR = 1, PIKA_PANIC = 2 };

typedef struct pika_subset {
    const char *material;
    const int32_t *faces;
    size_t face_count;
} pika_subset;

// Refined value i weighs entries offsets[i] to offsets[i + 1] (count + 1
// offsets) of the indices and weights.
typedef struct pika_stencils {
    const uint32_t *offsets;
    size_t count;
    const int32_t *indices;
    const float *weights;
    size_t entry_count;
} pika_stencils;

// The delegate's OpenSubdiv refinement: untriangulated faces, stencils over
// the cage points.
typedef struct pika_refined {
    const int32_t *face_vertex_counts;
    size_t face_count;
    const int32_t *face_vertex_indices;
    size_t index_count;
    const int32_t *holes;
    size_t hole_count;
    pika_stencils stencils;
} pika_refined;

// Per refined face corner, in pika_refined's order, the refined value it
// reads; stencils over the values handed over.
typedef struct pika_refined_texcoord {
    const int32_t *face_vertex_indices;
    size_t index_count;
    pika_stencils stencils;
} pika_refined_texcoord;

// Untriangulated. `refined` is null when the scheme is "none" or OpenSubdiv
// refused the mesh.
typedef struct pika_topology {
    const char *scheme;
    const int32_t *face_vertex_counts;
    size_t face_count;
    const int32_t *face_vertex_indices;
    size_t index_count;
    bool left_handed;
    const int32_t *holes;
    size_t hole_count;
    const pika_subset *subsets;
    size_t subset_count;
    const pika_refined *refined;
} pika_topology;

// Null when absent. Rotations are quaternions (i, j, k, real).
typedef struct pika_instance_primvars {
    const double *transforms;
    size_t transform_count;
    const float *translations;
    size_t translation_count;
    const float *rotations;
    size_t rotation_count;
    const float *scales;
    size_t scale_count;
} pika_instance_primvars;

// A VtValue by `kind`: "float" in `floats`; "int" and "bool" (0 or 1) in
// `integer`; "text" and "asset" (its resolved path) in `text`.
typedef struct pika_param {
    const char *name;
    const char *kind;
    float floats[16];
    size_t float_count;
    int32_t integer;
    const char *text;
} pika_param;

// husk names `path` relative to the material. `type` is nodeTypeId. `params`
// are the unconnected inputs' authored values, each with a "typeName:" entry
// for its USD type and, when authored, a "colorSpace:" one.
typedef struct pika_node {
    const char *path;
    const char *type;
    const pika_param *params;
    size_t param_count;
} pika_node;

typedef struct pika_connection {
    const char *from;
    const char *output;
    const char *to;
    const char *input;
} pika_connection;

// `surface` is the node the surface terminal connects to.
typedef struct pika_network {
    const pika_node *nodes;
    size_t node_count;
    const pika_connection *connections;
    size_t connection_count;
    const char *surface;
} pika_network;

// UsdLux inputs: intensity in nits, exposure in stops, `angle` a distant
// light's angular diameter in degrees.
typedef struct pika_light {
    const char *type;
    float intensity;
    float exposure;
    float color[3];
    bool normalize;
    float angle;
    float radius;
    float width;
    float height;
    float length;
    bool visible;
    // UsdLux's shadow:enable.
    bool shadow;
} pika_light;

// Bottom row first; `pixels` is null when the AOV is not bound.
typedef struct pika_aov {
    const char *format;
    uint8_t *pixels;
    size_t len;
} pika_aov;

typedef struct pika_frame {
    // The projection is conformed to the framing: it maps the data window to
    // OpenGL's clip space.
    double world_to_view[16];
    double projection[16];
    // Every bound buffer's size.
    uint32_t width, height;
    // The data window Pika writes: x, y, width, height, y down from the top.
    int32_t window[4];
    float clear_color[4];
    pika_aov color, depth, normal;
    // None means every tag.
    const char *const *render_tags;
    size_t render_tag_count;
    // Texels a texture's longest side is capped at (pika:textureCap); 0 is none.
    uint32_t texture_cap;
    // The shadow page pool's megabytes (pika:shadowPool), in quarter-megabyte pages; 0 is an
    // error.
    uint32_t shadow_pool;
} pika_frame;

// A key ending in _ms is milliseconds; any other value is a count.
typedef struct pika_stat {
    char key[48];
    double value;
} pika_stat;

typedef struct pika_renderer pika_renderer;

// Valid until this thread's next failed call.
const char *pika_last_error(void);

int pika_renderer_new(pika_renderer **out);
// Frees even a poisoned renderer.
int pika_renderer_free(pika_renderer *renderer);

// Bracket each Hydra sync. The end appends its stats to $PIKA_STATS and the
// store to $PIKA_STORE_DUMP.
int pika_sync_begin(const pika_renderer *renderer);
int pika_sync_end(const pika_renderer *renderer);

// Fails when the stats outnumber `capacity`. gpuMemoryUsed is for usdview's HUD.
int pika_stats(const pika_renderer *renderer, pika_stat *out, size_t capacity, size_t *count);

// Each *_new writes an id, never reused, to `*out`.
int pika_mesh_new(const pika_renderer *renderer, const char *path, uint64_t *out);
int pika_mesh_free(const pika_renderer *renderer, uint64_t id);
int pika_mesh_topology(const pika_renderer *renderer, uint64_t id,
                       const pika_topology *topology);
// Arrays hold `count` tuples: xyz triples, texcoord pairs.
int pika_mesh_points(const pika_renderer *renderer, uint64_t id,
                     const float *points, size_t count);
int pika_mesh_normals(const pika_renderer *renderer, uint64_t id, const char *interpolation,
                      const float *normals, size_t count);
int pika_mesh_no_normals(const pika_renderer *renderer, uint64_t id);
// The float2 primvar the mesh's material reads, or ""; set the material first.
int pika_mesh_texcoord_name(const pika_renderer *renderer, uint64_t id, char *out,
                            size_t capacity);
// `refined` is null when the cage is drawn.
int pika_mesh_texcoord(const pika_renderer *renderer, uint64_t id, const char *name,
                       const char *interpolation, const float *values, size_t count,
                       const pika_refined_texcoord *refined);
int pika_mesh_no_texcoord(const pika_renderer *renderer, uint64_t id);
int pika_mesh_transform(const pika_renderer *renderer, uint64_t id,
                        const double matrix[16]);
int pika_mesh_visible(const pika_renderer *renderer, uint64_t id, bool visible);
int pika_mesh_render_tag(const pika_renderer *renderer, uint64_t id, const char *tag);
int pika_mesh_material(const pika_renderer *renderer, uint64_t id, const char *material);

// Hydra's light-link categories, a token per collection.
int pika_mesh_categories(const pika_renderer *renderer, uint64_t id,
                         const char *const *categories, size_t count);

// The instance indices the prototype is drawn at.
int pika_mesh_instancer(const pika_renderer *renderer, uint64_t id, const char *instancer,
                        const int32_t *indices, size_t count);

int pika_instancer_new(const pika_renderer *renderer, const char *path, uint64_t *out);
int pika_instancer_free(const pika_renderer *renderer, uint64_t id);
// The parent's instance indices this instancer repeats at.
int pika_instancer_parent(const pika_renderer *renderer, uint64_t id, const char *parent,
                          const int32_t *indices, size_t count);
int pika_instancer_transform(const pika_renderer *renderer, uint64_t id,
                             const double matrix[16]);
int pika_instancer_primvars(const pika_renderer *renderer, uint64_t id,
                            const pika_instance_primvars *primvars);

int pika_material_new(const pika_renderer *renderer, const char *path, uint64_t *out);
int pika_material_free(const pika_renderer *renderer, uint64_t id);
// `*dirty`: its meshes must sync again, as it reads a primvar it did not.
int pika_material_network(const pika_renderer *renderer, uint64_t id,
                          const pika_network *network, bool *dirty);

// Curves are only measured: the delegate reports the bytes pulled and drops them.
int pika_curves_new(const pika_renderer *renderer, const char *path, uint64_t *out);
int pika_curves_free(const pika_renderer *renderer, uint64_t id);
int pika_curves_pulled(const pika_renderer *renderer, uint64_t id, size_t bytes);

int pika_light_new(const pika_renderer *renderer, const char *path, uint64_t *out);
int pika_light_free(const pika_renderer *renderer, uint64_t id);
int pika_light_params(const pika_renderer *renderer, uint64_t id,
                      const pika_light *params);
// The lightLink collection's token, the category of the meshes it lights, and the
// shadowLink collection's, of the meshes that cast its shadows; "" reaches every mesh.
int pika_light_link(const pika_renderer *renderer, uint64_t id, const char *light,
                    const char *shadow);
// A distant light shines down its -z axis.
int pika_light_transform(const pika_renderer *renderer, uint64_t id,
                         const double matrix[16]);

// Any other supported prim is only counted, by its Hydra type.
int pika_counted_new(const pika_renderer *renderer, const char *type, const char *path,
                     uint64_t *out);
int pika_counted_free(const pika_renderer *renderer, uint64_t id);

// Pixels outside the data window are left alone. Unlit is black, unbound
// 18 % grey clay; depth is OpenGL window depth cleared to 1; normals are world
// space, facing the viewer. While frames change the buffers get the previous
// frame, so its GPU work overlaps the host's. `*converged` is false until a
// call with nothing changed delivers it, and while a material builds (drawn
// clay before its first build, else as it was).
int pika_render(const pika_renderer *renderer, const pika_frame *frame, bool *converged);

#ifdef __cplusplus
}
#endif
