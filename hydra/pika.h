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
enum pika_format { PIKA_FLOAT16_VEC4 = 1, PIKA_FLOAT32_VEC4 = 2 };

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

// HdMeshTopology, untriangulated.
typedef struct pika_topology {
    const int32_t *face_vertex_counts;
    size_t face_count;
    const int32_t *face_vertex_indices;
    size_t index_count;
    bool left_handed;
    const pika_subset *subsets;
    size_t subset_count;
} pika_topology;

typedef struct pika_renderer pika_renderer;

// The message for this thread's last failed call, valid until its next one.
const char *pika_last_error(void);

// One per render delegate: opens a Vulkan device.
int pika_renderer_new(pika_renderer **out);
// Prints sync and frame timings and destroys the renderer, even after a panic.
int pika_renderer_free(pika_renderer *renderer);

// Bracket each Hydra sync. The end records the sync's stats and, when
// PIKA_STORE_DUMP names a file, appends the whole store to it.
int pika_sync_begin(const pika_renderer *renderer);
int pika_sync_end(const pika_renderer *renderer);

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

// Renders a frame into `pixels`, a render buffer of `format` and `len` bytes.
int pika_render(const pika_renderer *renderer, enum pika_format format,
                uint32_t width, uint32_t height, uint8_t *pixels, size_t len);

#ifdef __cplusplus
}
#endif
