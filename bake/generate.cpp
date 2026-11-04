// The generate job: a material's document (src/material.rs) in, the pixel
// stage from Houdini's MaterialX Slang generator out, as <out>/pixel.slang,
// with <out>/surface.slang, the vertex stage that fills what it reads from
// the renderer's vertices, and <out>/manifest saying what the stage expects
// of the renderer: its public uniforms with their types and initial values
// and its textures. The libraries job lists what generate reads beyond the
// document, for the renderer's cache key.

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
#include <set>
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

// The vertex stage: pixel.slang's VertexData, filled from the vertex
// shaders/vertex.slang pulls. Every float2 primvar the surface reads is the
// mesh's one texcoord: a geomprop it names, or UV0, the texcoord MaterialX's
// nodes read by default, which is the primvar st, as Storm and Karma read
// it. The renderer fetches st for a network naming no primvar and refuses
// one naming two (src/material.rs); only the nodedefs say which inputs read
// UV0, so a surface reading it beside another name is refused here.
std::string vertex(mx::ShaderStage const &stage) {
    std::string text = "// Written by the bake tool: the vertex stage for the pixel stage beside it.\n\n"
                       "#include \"pixel.slang\"\n#include \"vertex.slang\"\n\n"
                       "[shader(\"vertex\")]\n"
                       "VertexData vs(uint vertex : SV_VulkanVertexID, uint slot : SV_VulkanInstanceID)\n"
                       "{\n    Vertex p = pull(vertex, slot);\n    VertexData o;\n"
                       "    o.SV_Position = p.position;\n";
    mx::VariableBlock const &block = stage.getInputBlock(mx::HW::VERTEX_DATA);
    std::set<std::string> primvars;
    for (size_t i = 0; i < block.size(); ++i) {
        mx::ShaderPort const *port = block[i];
        std::string const &name = port->getVariable();
        std::string value;
        if (name == "positionWorld")
            value = "p.world";
        else if (name == "normalWorld")
            value = "p.normal";
        else if (name == "tangentWorld")
            value = "p.tangent";
        else if (name == "bitangentWorld")
            value = "p.bitangent";
        else if (name == "texcoord_0" && port->getType() == mx::Type::VECTOR2) {
            value = "p.texcoord";
            primvars.insert("st");
        } else if (name.rfind("i_geomprop_", 0) == 0 && port->getType() == mx::Type::VECTOR2) {
            value = "p.texcoord";
            primvars.insert(name.substr(std::string("i_geomprop_").size()));
        } else
            throw std::runtime_error("the surface reads " + name + " as a " +
                                     port->getType().getName() +
                                     ", which Pika's vertex stage does not provide (of the "
                                     "primvars, it provides one float2: the texcoord, UV0 or "
                                     "a float2 geomprop)");
        text += "    o." + name + " = " + value + ";\n";
    }
    if (primvars.size() > 1) {
        std::string names;
        for (std::string const &primvar : primvars)
            names += (names.empty() ? "" : " and ") + primvar;
        throw std::runtime_error("the surface reads the primvars " + names +
                                 " (UV0, the texcoord its nodes read by default, is st), and a "
                                 "mesh is drawn with one texcoord");
    }
    return text + "    return o;\n}\n";
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
                           lines(stage.getUniformBlock(mx::HW::PUBLIC_UNIFORMS), "uniform");
    write(out + "/pixel.slang", pixel);
    write(out + "/surface.slang", vertex(stage));
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
