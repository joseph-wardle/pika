// The bake tool's jobs, one per file; bake.cpp dispatches to them.
#pragma once

#include <string>

// Generates `document`'s pixel stage into the directory `out`.
void generate(std::string const &document, std::string const &out);

// Bakes the image `source` into the texture file `out`.
void texture(std::string const &source, std::string const &space, std::string const &encoding,
             unsigned cap, std::string const &config, std::string const &out);
