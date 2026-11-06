// The bake tool: the one Pika process that links MaterialX and OpenImageIO.
// The renderer spawns it, so neither library loads into the host beside the
// host's own. generate turns a material's document into Slang
// (generate.cpp), and libraries lists what it reads beyond the document;
// texture turns an image, or the UDIM tiles listed after the output, into
// the one file the renderer uploads (texture.cpp), its levels kept under
// the cap, texels a side, 0 for none. The renderer keys its cache on every
// argument and on that listing, so the tool decides nothing.

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
        } else if (job == "texture" && argc >= 9 && argc % 2 == 1) {
            char *end = nullptr;
            unsigned long cap = strtoul(argv[4], &end, 10);
            if (*argv[4] == 0 || *end != 0 || cap > 0xffffffffu)
                throw std::runtime_error(std::string("the cap ") + argv[4] +
                                         " is not a number of texels, 0 for none");
            std::vector<Tile> set;
            for (int i = 7; i < argc; i += 2) {
                unsigned long number = strtoul(argv[i], &end, 10);
                if (*argv[i] == 0 || *end != 0)
                    throw std::runtime_error(std::string("the tile ") + argv[i] + " is not a number");
                set.push_back({unsigned(number), argv[i + 1]});
            }
            texture(set, argv[2], argv[3], cap, argv[5], argv[6]);
        } else {
            fprintf(stderr,
                    "usage: bake generate <document.mtlx> <out>\n"
                    "       bake libraries\n"
                    "       bake texture <space> <encoding> <cap> <config> <out> (<tile> <file>)...\n");
            return 2;
        }
    } catch (std::exception const &e) {
        fprintf(stderr, "bake: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
