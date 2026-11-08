// The generate tool, the one Pika program that links MaterialX; the
// renderer runs it (src/bake.rs), so MaterialX never loads into the host
// beside the host's own.
//
//   generate <document.mtlx> <out>
//
// A material's document (src/material.rs) in, the pixel stage from
// Houdini's MaterialX Slang generator out, as <out>/pixel.slang, with
// <out>/surface.slang, the vertex stage that fills what it reads from the
// renderer's vertices, and <out>/manifest saying what the stage expects of
// the renderer: its public uniforms with their types and initial values
// and its textures.
//
//   generate libraries
//
// Lists what generate reads beyond the document, for the renderer's cache
// key. The eye and the lights the pixel
// stage reads come from the renderer's view (shaders/frame.slang), not from
// the material's block, which holds the public uniforms alone, and the
// light loop is Pika's: MaterialX's generator supplies the BSDFs alone.

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXGenHw/HwConstants.h>
#include <MaterialXGenHw/HwShaderGenerator.h>
#include <MaterialXGenHw/Nodes/HwSurfaceNode.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenSlang/SlangShaderGenerator.h>

#include <algorithm>
#include <cstdio>
#include <exception>
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
// it. The renderer fetches the first primvar a network names, else st
// (src/material.rs); only the nodedefs say which inputs read UV0, so a
// surface reading two is refused here.
std::string vertex(mx::ShaderStage const &stage) {
    std::string text = "// Written by bake/generate.cpp: the vertex stage for the pixel stage beside it.\n\n"
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

// MaterialX's surface with Pika's light loop in place of its own. The
// generator is told of no lights (hwMaxActiveLightSources 0), so the pixel
// stage holds no LightData, light shaders or sampleLightSource; the loop
// asks shaders/frame.slang for the lights instead, and a skipped light
// costs no BSDF. The loop calls functions rather than reading the view so
// that the lights' kinds and lists change frame.slang alone: a change to
// the text emitted here changes the tool, which re-bakes every texture
// (src/bake.rs keys both on the tool's bytes).
class Surface : public mx::HwSurfaceNode {
  public:
    static mx::ShaderNodeImplPtr create() { return std::make_shared<Surface>(); }

    void emitLightLoop(mx::ShaderNode const &node, mx::GenContext &context, mx::ShaderStage &stage,
                       std::string const &color) const override {
        mx::ShaderInput const *input = node.getInput(getBsdfInputName());
        mx::ShaderNode const *bsdf = input ? input->getConnectedSibling() : nullptr;
        if (!bsdf)
            return;
        auto const &g = static_cast<mx::HwShaderGenerator const &>(context.getShaderGenerator());
        // N, V, P, L and occlusion are HwSurfaceNode's, declared before the loop.
        g.emitComment("Pika's light loop (bake/generate.cpp)", stage);
        g.emitLine("for (uint light_index = 0; light_index < light_count(P); ++light_index)", stage,
                   false);
        g.emitScopeBegin(stage);
        g.emitLine("Incident incident = incident_light(light_index, P, N)", stage);
        g.emitLine("L = incident.direction", stage);
        g.emitLine("occlusion = visibility(light_index, P)", stage);
        if (bsdf->hasClassification(mx::ShaderNode::Classification::BSDF_R)) {
            g.emitLine("ClosureData closureData = makeClosureData(CLOSURE_TYPE_REFLECTION, L, V, "
                       "N, P, occlusion)",
                       stage);
            g.emitFunctionCall(*bsdf, context, stage);
        } else {
            g.emitLineBegin(stage);
            g.emitOutput(bsdf->getOutput(), true, true, context, stage);
            g.emitLineEnd(stage);
        }
        g.emitLine(color + " += incident.illuminance * " + bsdf->getOutput()->getVariable() +
                       ".response",
                   stage);
        g.emitScopeEnd(stage);
        // The environment and emission after the loop read it unshadowed.
        g.emitLine("occlusion = 1.0", stage);
        g.emitLineBreak(stage);
    }
};

// The generator with Pika's surface, and the pixel stage's block holding
// the public uniforms alone: the private ones, which are the eye alone, are
// read from the view (shaders/frame.slang), which is written once a frame
// rather than once a material. A surface reading a private uniform the view
// lacks (a time node's u_time) is refused.
class Generator : public mx::SlangShaderGenerator {
  public:
    Generator() : mx::SlangShaderGenerator(mx::TypeSystem::create()) {
        registerImplementation("IM_surface_" + mx::SlangShaderGenerator::TARGET, Surface::create);
    }

  protected:
    void emitUniforms(mx::GenContext &context, mx::ShaderStage &stage,
                      bool lighting) const override {
        if (stage.getName() != mx::Stage::PIXEL)
            return mx::SlangShaderGenerator::emitUniforms(context, stage, lighting);
        // A private uniform is still its token here ($time), substituted
        // with its name (u_time) once the stage is complete.
        mx::VariableBlock const &priv = stage.getUniformBlock(mx::HW::PRIVATE_UNIFORMS);
        for (size_t i = 0; i < priv.size(); ++i) {
            std::string const &token = priv[i]->getVariable();
            if (token == mx::HW::T_VIEW_POSITION)
                continue;
            auto const named = _tokenSubstitutions.find(token);
            std::string const name = named == _tokenSubstitutions.end() ? token : named->second;
            throw std::runtime_error("the surface reads " + name +
                                     ", which Pika's view does not provide");
        }
        emitLine("#include \"frame.slang\"", stage, false);
        emitLineBreak(stage);
        emitLine("cbuffer " + stage.getName() + "CB", stage, false);
        emitScopeBegin(stage);
        emitVariableDeclarations(stage.getUniformBlock(mx::HW::PUBLIC_UNIFORMS),
                                 _syntax->getUniformQualifier(), mx::Syntax::SEMICOLON, context,
                                 stage);
        emitScopeEnd(stage);
        emitLineBreak(stage);
    }
};

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

    mx::ShaderGeneratorPtr generator = std::make_shared<Generator>();
    mx::GenContext context(generator);
    // A library file under bake/ shadows Houdini's of the same path.
    context.registerSourceCodeSearchPath(mx::FilePath(bake));
    context.registerSourceCodeSearchPath(mx::FilePath(materialx));
    mx::GenOptions &options = context.getOptions();
    options.hwSpecularEnvironmentMethod = mx::SPECULAR_ENVIRONMENT_NONE;
    // Surface's loop reads the lights from the view.
    options.hwMaxActiveLightSources = 0;
    options.hwImplicitBitangents = true;
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

} // namespace

int main(int argc, char **argv) {
    std::string job = argc > 1 ? argv[1] : "";
    try {
        if (job == "libraries" && argc == 2) {
            libraries();
        } else if (argc == 3 && job != "libraries") {
            generate(argv[1], argv[2]);
        } else {
            fprintf(stderr, "usage: generate <document.mtlx> <out>\n"
                            "       generate libraries\n");
            return 2;
        }
    } catch (std::exception const &e) {
        fprintf(stderr, "generate: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
