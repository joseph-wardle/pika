// The bake tool's jobs, one per file; bake.cpp dispatches to them.
#pragma once

#include <string>
#include <vector>

// Generates `document`'s pixel stage into the directory `out`.
void generate(std::string const &document, std::string const &out);

// Prints the MaterialX version and library files generate reads.
void libraries();

// One file of a texture: an image on its own, numbered 0, or a UDIM tile
// numbered 1001 to 1100, as the renderer lists them (src/bake.rs).
struct Tile {
    unsigned number;
    std::string path;
};

// Bakes the image, or UDIM set, `set` into the texture file `out`.
void texture(std::vector<Tile> const &set, std::string const &space, std::string const &encoding,
             std::string const &config, std::string const &out);
