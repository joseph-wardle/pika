// The bake tool: the one Pika process that links MaterialX and OpenImageIO.
// The renderer spawns it, so neither library loads into the host beside the
// host's own.
//
//   bake generate <document.mtlx> <out>
//   bake texture <source> <space> <encoding> <cap> <config> <out>
//
// generate turns a material's document into Slang (generate.cpp); texture
// turns an image into the one file the renderer uploads (texture.cpp). The
// renderer keys its cache on every argument, so the tool decides nothing.

#include "bake.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    std::string job = argc > 1 ? argv[1] : "";
    try {
        if (job == "generate" && argc == 4) {
            generate(argv[2], argv[3]);
        } else if (job == "texture" && argc == 8) {
            char *end = nullptr;
            unsigned long cap = strtoul(argv[5], &end, 10);
            if (*argv[5] == 0 || *end != 0 || cap == 0)
                throw std::runtime_error(std::string("the cap ") + argv[5] +
                                         " is not a positive number of texels");
            texture(argv[2], argv[3], argv[4], cap, argv[6], argv[7]);
        } else {
            fprintf(stderr, "usage: bake generate <document.mtlx> <out>\n"
                            "       bake texture <source> <space> <encoding> <cap> <config> <out>\n");
            return 2;
        }
    } catch (std::exception const &e) {
        fprintf(stderr, "bake: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
