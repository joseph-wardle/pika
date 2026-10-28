// The C ABI of the Pika renderer, implemented in src/abi.rs.
//
// Calls go one way: the delegate calls Pika, and Pika never calls back.
// Every call returns a status; on anything but PIKA_OK, pika_last_error()
// says why. Pointers passed in are borrowed only for the call.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { PIKA_OK = 0, PIKA_ERROR = 1, PIKA_PANIC = 2 };

// Render buffer formats, named after Hydra's HdFormat values they match.
enum pika_format { PIKA_FLOAT16_VEC4 = 1, PIKA_FLOAT32_VEC4 = 2 };

typedef struct pika_renderer pika_renderer;

// The message for this thread's last failed call, valid until its next one.
const char *pika_last_error(void);

// One per render delegate: opens a Vulkan device.
int pika_renderer_new(pika_renderer **out);
// Prints frame timings and destroys the renderer, even after a panic.
int pika_renderer_free(pika_renderer *renderer);

// Renders a frame into `pixels`, a render buffer of `format` and `len` bytes.
int pika_render(const pika_renderer *renderer, enum pika_format format,
                uint32_t width, uint32_t height, uint8_t *pixels, size_t len);

#ifdef __cplusplus
}
#endif
