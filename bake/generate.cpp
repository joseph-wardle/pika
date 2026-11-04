// The generate job: a material's document (src/material.rs) in, the pixel
// stage from Houdini's MaterialX Slang generator out, as <out>/pixel.slang,
// with <out>/manifest saying what the stage expects of the renderer: its
// public uniforms with their types and initial values, its textures, and the
// varyings the vertex stage must deliver. The libraries job lists what
// generate reads beyond the document, for the renderer's cache key.

#include "bake.h"

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXGenHw/HwShaderGenerator.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenSlang/SlangShaderGenerator.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>

namespace mx = MaterialX;

namespace {

// The Makefile names the Houdini and this directory.
const std::string materialx = HFS "/houdini/materialx";
const std::string bake = BAKE;

void write(std::string const &path, std::string const &text) {
    std::ofstream file(path);
    file << text;
    if (!file)
        throw std::runtime_error("cannot write " + path);
}

std::string lines(mx::VariableBlock const &block, std::string const &kind) {
    std::string text;
    for (size_t i = 0; i < block.size(); ++i) {
        mx::ShaderPort const *port = block[i];
        if (port->getType() == mx::Type::FILENAME) {
            text += "texture " + port->getVariable() + "\n";
            continue;
        }
        text += kind + " " + port->getType().getName() + " " + port->getVariable();
        if (port->getValue())
            text += " " + port->getValue()->getValueString();
        text += "\n";
    }
    return text;
}

} // namespace

void generate(std::string const &path, std::string const &out) {
    mx::DocumentPtr libraries = mx::createDocument();
    mx::loadLibraries({"libraries"}, mx::FileSearchPath(materialx), libraries);
    mx::DocumentPtr doc = mx::createDocument();
    mx::readFromXmlFile(doc, path);
    doc->importLibrary(libraries);
    for (mx::NodePtr const &node : doc->getNodes()) {
        if (!node->getNodeDef())
            throw std::runtime_error("node " + node->getName() + " names the nodedef " +
                                     node->getNodeDefString() +
                                     ", which Houdini's MaterialX libraries do not define");
    }
    mx::NodePtr material = doc->getNode("material");
    if (!material)
        throw std::runtime_error("the document has no node named material");

    mx::ShaderGeneratorPtr generator = mx::SlangShaderGenerator::create();
    mx::GenContext context(generator);
    // A library file under bake/ shadows Houdini's of the same path.
    context.registerSourceCodeSearchPath(mx::FilePath(bake));
    context.registerSourceCodeSearchPath(mx::FilePath(materialx));
    mx::GenOptions &options = context.getOptions();
    options.hwSpecularEnvironmentMethod = mx::SPECULAR_ENVIRONMENT_NONE;
    options.hwMaxActiveLightSources = 1;
    options.hwImplicitBitangents = true;
    // The headlight is the one light; the generated loop reads it from
    // u_lightData[0] with u_numActiveLightSources set to 1.
    mx::HwShaderGenerator::bindLightShader(*libraries->getNodeDef("ND_directional_light"), 1,
                                           context);
    mx::ShaderPtr shader = generator->generate("pika", material, context);

    // Without environment lighting the generated code reads this without
    // declaring it.
    std::string pixel = "static const bool u_refractionTwoSided = false;\n" +
                        shader->getSourceCode(mx::Stage::PIXEL);
    mx::ShaderStage const &stage = shader->getStage(mx::Stage::PIXEL);
    std::string manifest = "materialx " + mx::getVersionString() + "\n" +
                           lines(stage.getUniformBlock(mx::HW::PUBLIC_UNIFORMS), "uniform") +
                           lines(stage.getInputBlock(mx::HW::VERTEX_DATA), "varying");
    write(out + "/pixel.slang", pixel);
    write(out + "/manifest", manifest);
}

// Each library file with its size and modification time rather than its
// bytes: a Houdini build stamps every file it installs, so the listing
// changes with the generator's shared objects too, which no library file
// shows.
void libraries() {
    std::filesystem::path root = materialx + "/libraries";
    std::vector<std::string> lines;
    for (auto const &entry : std::filesystem::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file())
            continue;
        // The time as Unix seconds: std::filesystem's has an epoch of its
        // own before C++20.
        struct stat status;
        if (stat(entry.path().c_str(), &status) != 0)
            throw std::runtime_error("cannot stat " + entry.path().string());
        lines.push_back(entry.path().lexically_relative(root).string() + " " +
                        std::to_string(status.st_size) + " " + std::to_string(status.st_mtime));
    }
    std::sort(lines.begin(), lines.end());
    std::string text = "materialx " + mx::getVersionString() + "\n";
    for (std::string const &line : lines)
        text += line + "\n";
    fputs(text.c_str(), stdout);
}
