// The renderer's C ABI. On a status other than PIKA_OK, pika_last_error() says why. Pointers are
// borrowed for the call, and "" means no path. Matrices are row-major GfMatrix4d.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// After a panic every call fails until the renderer is freed.
enum { PIKA_OK = 0, PIKA_ERROR = 1, PIKA_PANIC = 2 };

typedef struct pika_subset {
    const char *material;
    const int32_t *faces;
    size_t face_count;
} pika_subset;

// Refined value i weighs entries offsets[i] to offsets[i + 1] (count + 1 offsets).
typedef struct pika_stencils {
    const uint32_t *offsets;
    size_t count;
    const int32_t *indices;
    const float *weights;
    size_t entry_count;
} pika_stencils;

typedef struct pika_refined {
    const int32_t *face_vertex_counts;
    size_t face_count;
    const int32_t *face_vertex_indices;
    size_t index_count;
    const int32_t *holes;
    size_t hole_count;
    pika_stencils stencils;
} pika_refined;

typedef struct pika_refined_texcoord {
    const int32_t *face_vertex_indices;
    size_t index_count;
    pika_stencils stencils;
} pika_refined_texcoord;

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

// "float" in `floats`, "int" and "bool" in `integer`, "text" and "asset" (resolved) in `text`.
typedef struct pika_param {
    const char *name;
    const char *kind;
    float floats[16];
    size_t float_count;
    int32_t integer;
    const char *text;
} pika_param;

// husk names `path` relative to the material. `params` are the unconnected inputs' authored
// values, each with a "typeName:" entry and, when authored, a "colorSpace:" one.
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

typedef struct pika_network {
    const pika_node *nodes;
    size_t node_count;
    const pika_connection *connections;
    size_t connection_count;
    const char *surface;
} pika_network;

// intensity in nits, exposure in stops, `angle` a distant light's diameter in degrees.
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
    bool shadow;
} pika_light;

// Bottom row first. `pixels` is null when the AOV isn't bound.
typedef struct pika_aov {
    const char *format;
    uint8_t *pixels;
    size_t len;
} pika_aov;

typedef struct pika_frame {
    double world_to_view[16];
    double projection[16];
    uint32_t width, height;
    // x, y, width, height, y down from the top.
    int32_t window[4];
    float clear_color[4];
    pika_aov color, depth, normal;
    // Null means every tag.
    const char *const *render_tags;
    size_t render_tag_count;
    // pika:textureCap, texels on a texture's longest side. 0 is none.
    uint32_t texture_cap;
    // pika:shadowPool, in MB of quarter-MB pages. 0 is an error.
    uint32_t shadow_pool;
    float sun_threshold;
} pika_frame;

// Keys ending in _ms are milliseconds, the rest counts.
typedef struct pika_stat {
    char key[48];
    double value;
} pika_stat;

typedef struct pika_renderer pika_renderer;

// Valid until this thread's next failed call.
const char *pika_last_error(void);

int pika_renderer_new(pika_renderer **out);
int pika_renderer_free(pika_renderer *renderer);

// Bracket each sync. The end appends stats to $PIKA_STATS and the store to $PIKA_STORE_DUMP.
int pika_sync_begin(const pika_renderer *renderer);
int pika_sync_end(const pika_renderer *renderer);

// Fails when the stats outnumber `capacity`.
int pika_stats(const pika_renderer *renderer, pika_stat *out, size_t capacity, size_t *count);

// Each *_new writes a never-reused id to `*out`.
int pika_mesh_new(const pika_renderer *renderer, const char *path, uint64_t *out);
int pika_mesh_free(const pika_renderer *renderer, uint64_t id);
int pika_mesh_topology(const pika_renderer *renderer, uint64_t id,
                       const pika_topology *topology);
int pika_mesh_points(const pika_renderer *renderer, uint64_t id,
                     const float *points, size_t count);
int pika_mesh_normals(const pika_renderer *renderer, uint64_t id, const char *interpolation,
                      const float *normals, size_t count);
int pika_mesh_no_normals(const pika_renderer *renderer, uint64_t id);
// The float2 primvar the mesh's material reads, or "". Set the material first.
int pika_mesh_texcoord_name(const pika_renderer *renderer, uint64_t id, char *out,
                            size_t capacity);
int pika_mesh_texcoord(const pika_renderer *renderer, uint64_t id, const char *name,
                       const char *interpolation, const float *values, size_t count,
                       const pika_refined_texcoord *refined);
int pika_mesh_no_texcoord(const pika_renderer *renderer, uint64_t id);
int pika_mesh_transform(const pika_renderer *renderer, uint64_t id,
                        const double matrix[16]);
int pika_mesh_visible(const pika_renderer *renderer, uint64_t id, bool visible);
int pika_mesh_render_tag(const pika_renderer *renderer, uint64_t id, const char *tag);
int pika_mesh_material(const pika_renderer *renderer, uint64_t id, const char *material);

int pika_mesh_categories(const pika_renderer *renderer, uint64_t id,
                         const char *const *categories, size_t count);

int pika_mesh_instancer(const pika_renderer *renderer, uint64_t id, const char *instancer,
                        const int32_t *indices, size_t count);

int pika_instancer_new(const pika_renderer *renderer, const char *path, uint64_t *out);
int pika_instancer_free(const pika_renderer *renderer, uint64_t id);
int pika_instancer_parent(const pika_renderer *renderer, uint64_t id, const char *parent,
                          const int32_t *indices, size_t count);
// `counts[i]` tokens per instance, in order, all in `categories`.
int pika_instancer_categories(const pika_renderer *renderer, uint64_t id, const size_t *counts,
                              size_t count, const char *const *categories);
int pika_instancer_transform(const pika_renderer *renderer, uint64_t id,
                             const double matrix[16]);
int pika_instancer_primvars(const pika_renderer *renderer, uint64_t id,
                            const pika_instance_primvars *primvars);

int pika_material_new(const pika_renderer *renderer, const char *path, uint64_t *out);
int pika_material_free(const pika_renderer *renderer, uint64_t id);
// `*dirty` when its meshes must sync again, since it reads a new primvar.
int pika_material_network(const pika_renderer *renderer, uint64_t id,
                          const pika_network *network, bool *dirty);

int pika_curves_new(const pika_renderer *renderer, const char *path, uint64_t *out);
int pika_curves_free(const pika_renderer *renderer, uint64_t id);
int pika_curves_pulled(const pika_renderer *renderer, uint64_t id, size_t bytes);

int pika_light_new(const pika_renderer *renderer, const char *path, uint64_t *out);
int pika_light_free(const pika_renderer *renderer, uint64_t id);
int pika_light_params(const pika_renderer *renderer, uint64_t id,
                      const pika_light *params);
// Tokens of the lightLink and shadowLink collections. "" reaches every mesh.
int pika_light_link(const pika_renderer *renderer, uint64_t id, const char *light,
                    const char *shadow);
int pika_light_texture(const pika_renderer *renderer, uint64_t id, const char *file,
                       const char *format);
// A distant light shines down its -z axis.
int pika_light_transform(const pika_renderer *renderer, uint64_t id,
                         const double matrix[16]);

int pika_counted_new(const pika_renderer *renderer, const char *type, const char *path,
                     uint64_t *out);
int pika_counted_free(const pika_renderer *renderer, uint64_t id);

// Unlit is black, unbound 18% grey clay, depth OpenGL window depth cleared to 1, normals world
// space facing the viewer. While frames change the buffers get the previous frame, and
// `*converged` is false until a call with nothing changed delivers it.
int pika_render(const pika_renderer *renderer, const pika_frame *frame, bool *converged);

#ifdef __cplusplus
}
#endif
