// The C ABI of the Pika renderer, implemented in src/abi.rs.
//
// Calls go one way: the delegate calls Pika, and Pika never calls back.
// Every call returns a status; on anything but PIKA_OK, pika_last_error()
// says why. Pointers passed in are borrowed only for the call.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { PIKA_OK = 0, PIKA_ERROR = 1, PIKA_PANIC = 2 };

// Render buffer formats, named after Hydra's HdFormat values they match.
// Color is half or full float RGBA, depth float and normals float xyz.
enum pika_format {
    PIKA_FLOAT16_VEC4 = 1,
    PIKA_FLOAT32_VEC4 = 2,
    PIKA_FLOAT32 = 3,
    PIKA_FLOAT32_VEC3 = 4,
};

// Primvar interpolations, named after Hydra's HdInterpolation values.
enum pika_interpolation {
    PIKA_CONSTANT = 1,
    PIKA_UNIFORM = 2,
    PIKA_VARYING = 3,
    PIKA_VERTEX = 4,
    PIKA_FACE_VARYING = 5,
};

// A GeomSubset: the faces bound to one material ("" when none is bound).
typedef struct pika_subset {
    const char *material;
    const int32_t *faces;
    size_t face_count;
} pika_subset;

// A subdivision surface refined by the delegate's OpenSubdiv: the refined
// faces, untriangulated, with those refined from holes, and per refined point
// a stencil, the weighted sum of cage points it is. Stencil i is entries
// stencil_offsets[i] to stencil_offsets[i + 1] of the indices and weights.
typedef struct pika_refined {
    const int32_t *face_vertex_counts;
    size_t face_count;
    const int32_t *face_vertex_indices;
    size_t index_count;
    const int32_t *holes;
    size_t hole_count;
    // point_count + 1 offsets.
    const uint32_t *stencil_offsets;
    size_t point_count;
    const int32_t *stencil_indices;
    const float *stencil_weights;
    size_t entry_count;
} pika_refined;

// HdMeshTopology, untriangulated. `scheme` is USD's subdivision scheme
// token: "catmullClark", "loop", "bilinear" or "none". `refined` is null
// when the scheme is none or OpenSubdiv refused the mesh.
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

// Hydra's per-instance transform primvars. Each pointer is null when the
// primvar is absent, and holds `count` matrices (row-major GfMatrix4d),
// xyz triples or quaternions (i, j, k, real) when present.
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

// A render buffer bound to an AOV: `len` bytes of `format`, bottom row
// first. `pixels` is null when the AOV is not bound.
typedef struct pika_aov {
    enum pika_format format;
    uint8_t *pixels;
    size_t len;
} pika_aov;

// One frame of a render pass, from Hydra's render pass state.
typedef struct pika_frame {
    // Row-major GfMatrix4ds. The projection is conformed to the framing, so
    // it maps the data window to OpenGL's clip space.
    double world_to_view[16];
    double projection[16];
    // Every bound render buffer's size.
    uint32_t width, height;
    // The data window, the pixels Pika writes: x, y, width, height, with y
    // counted down from the top row, as CameraUtilFraming counts it.
    int32_t window[4];
    float clear_color[4];
    pika_aov color, depth, normal;
    // The render tags the pass draws; none means every tag.
    const char *const *render_tags;
    size_t render_tag_count;
} pika_frame;

// One stat for the host: a key as PIKA_STATS writes it and its value. A key
// ending in _ms is milliseconds; any other value is a count.
typedef struct pika_stat {
    char key[48];
    double value;
} pika_stat;

typedef struct pika_renderer pika_renderer;

// The message for this thread's last failed call, valid until its next one.
const char *pika_last_error(void);

// One per render delegate: opens a Vulkan device.
int pika_renderer_new(pika_renderer **out);
// Prints the stats summary and destroys the renderer, even after a panic.
int pika_renderer_free(pika_renderer *renderer);

// Bracket each Hydra sync. The end records the sync's stats and appends them
// to the file PIKA_STATS names, and the whole store to PIKA_STORE_DUMP's.
int pika_sync_begin(const pika_renderer *renderer);
int pika_sync_end(const pika_renderer *renderer);

// Writes up to `capacity` stats to `out` and how many to `*count`: the
// counts of syncs and frames, the last sync's pairs (keys prefixed sync_),
// the last frame's, and gpuMemoryUsed, which usdview's HUD shows.
int pika_stats(const pika_renderer *renderer, pika_stat *out, size_t capacity, size_t *count);

// Adds a mesh and writes its id, never reused, to `*out`.
int pika_mesh_new(const pika_renderer *renderer, const char *path, uint64_t *out);
int pika_mesh_free(const pika_renderer *renderer, uint64_t id);
// Each call replaces one field of the mesh, in Hydra's representation.
int pika_mesh_topology(const pika_renderer *renderer, uint64_t id,
                       const pika_topology *topology);
// `points` and `normals` hold `count` xyz triples.
int pika_mesh_points(const pika_renderer *renderer, uint64_t id,
                     const float *points, size_t count);
int pika_mesh_normals(const pika_renderer *renderer, uint64_t id,
                      enum pika_interpolation interpolation,
                      const float *normals, size_t count);
// For a mesh without a normals primvar.
int pika_mesh_no_normals(const pika_renderer *renderer, uint64_t id);
// The world transform: Hydra's row-major GfMatrix4d.
int pika_mesh_transform(const pika_renderer *renderer, uint64_t id,
                        const double matrix[16]);
int pika_mesh_visible(const pika_renderer *renderer, uint64_t id, bool visible);
// The purpose Hydra filters syncs by: "geometry", "render", "proxy" or "guide".
int pika_mesh_render_tag(const pika_renderer *renderer, uint64_t id, const char *tag);
// The material bound to the whole mesh, or "".
int pika_mesh_material(const pika_renderer *renderer, uint64_t id, const char *material);

// The instancer a prototype mesh belongs to ("" when the mesh is not
// instanced), and the instance indices it is drawn at.
int pika_mesh_instancer(const pika_renderer *renderer, uint64_t id, const char *instancer,
                        const int32_t *indices, size_t count);

// Adds an instancer and writes its id, never reused, to `*out`.
int pika_instancer_new(const pika_renderer *renderer, const char *path, uint64_t *out);
int pika_instancer_free(const pika_renderer *renderer, uint64_t id);
// The instancer this one is nested in ("" when none), and the parent's
// instance indices it is repeated at.
int pika_instancer_parent(const pika_renderer *renderer, uint64_t id, const char *parent,
                          const int32_t *indices, size_t count);
int pika_instancer_transform(const pika_renderer *renderer, uint64_t id,
                             const double matrix[16]);
int pika_instancer_primvars(const pika_renderer *renderer, uint64_t id,
                            const pika_instance_primvars *primvars);

// Basis curves are fetched only to measure them: the delegate reports the
// bytes it pulled and drops them.
int pika_curves_new(const pika_renderer *renderer, const char *path, uint64_t *out);
int pika_curves_free(const pika_renderer *renderer, uint64_t id);
int pika_curves_pulled(const pika_renderer *renderer, uint64_t id, size_t bytes);

// A prim of any other supported type is only counted, by its Hydra type.
int pika_counted_new(const pika_renderer *renderer, const char *type, const char *path,
                     uint64_t *out);
int pika_counted_free(const pika_renderer *renderer, uint64_t id);

// Draws the store's meshes of the frame's render tags into its data window:
// color is grey, shaded by the normal against the view direction; depth is
// OpenGL's window depth, cleared to 1; normals are in world space, turned
// toward the viewer.
// Pixels outside the data window are left as they are.
// While frames change, the buffers get the frame before this one, which
// leaves this one's GPU work running while the host works; `*converged` is
// false until a call with nothing changed delivers it.
int pika_render(const pika_renderer *renderer, const pika_frame *frame, bool *converged);

#ifdef __cplusplus
}
#endif
