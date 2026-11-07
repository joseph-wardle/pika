// The bake tool: the one Pika process that links MaterialX and OpenImageIO.
// The renderer spawns it, so neither library loads into the host beside the
// host's own. generate turns a material's document into Slang
// (generate.cpp), and libraries lists what it reads beyond the document;
// texture turns an image, or the UDIM tiles listed after the output, into
// the one file the renderer uploads (texture.cpp), every level of it. The
// renderer keys its cache on every argument and on that listing, so the
// tool decides nothing.

#include "bake.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    std::string job = argc > 1 ? argv[1] : "";
    try {
        if (job == "generate" && argc == 4) {
            generate(argv[2], argv[3]);
        } else if (job == "libraries" && argc == 2) {
            libraries();
        } else if (job == "texture" && argc >= 8 && argc % 2 == 0) {
            char *end = nullptr;
            std::vector<Tile> set;
            for (int i = 6; i < argc; i += 2) {
                unsigned long number = strtoul(argv[i], &end, 10);
                if (*argv[i] == 0 || *end != 0)
                    throw std::runtime_error(std::string("the tile ") + argv[i] + " is not a number");
                set.push_back({unsigned(number), argv[i + 1]});
            }
            texture(set, argv[2], argv[3], argv[4], argv[5]);
        } else {
            fprintf(stderr,
                    "usage: bake generate <document.mtlx> <out>\n"
                    "       bake libraries\n"
                    "       bake texture <space> <encoding> <config> <out> (<tile> <file>)...\n");
            return 2;
        }
    } catch (std::exception const &e) {
        fprintf(stderr, "bake: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
