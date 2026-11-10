// The generate tool, run apart from the host so MaterialX never loads beside the host's own.
// MaterialX supplies the BSDFs; the eye, the lights and the light loop are Pika's.

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

// HFS and BAKE come from the Makefile.
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

// Every float2 primvar the surface reads is the mesh's one texcoord: a geomprop, or
// UV0, which is st as Storm and Karma read it. Only the nodedefs say which inputs
// read UV0, so a surface reading two is refused here.
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
        else if (name == "link")
            value = "asint(p.link)";
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

// MaterialX's surface with Pika's light loop. The generator is told of no lights;
// the loop asks frame.slang for them, so lights change frame.slang alone. Editing
// the text emitted here changes the tool, which re-bakes every texture.
class Surface : public mx::HwSurfaceNode {
  public:
    static mx::ShaderNodeImplPtr create() { return std::make_shared<Surface>(); }

    void createVariables(mx::ShaderNode const &node, mx::GenContext &context,
                         mx::Shader &shader) const override {
        mx::HwSurfaceNode::createVariables(node, context, shader);
        // An int, as MaterialX has no uint (and Slang does not interpolate ints). Only the
        // loop reads it; Slang drops an unread input, unmatching the vertex stage's output.
        if (bsdf(node))
            mx::addStageInput(mx::HW::VERTEX_DATA, mx::Type::INTEGER, "link",
                              shader.getStage(mx::Stage::PIXEL));
    }

    void emitLightLoop(mx::ShaderNode const &node, mx::GenContext &context, mx::ShaderStage &stage,
                       std::string const &color) const override {
        mx::ShaderNode const *bsdf = this->bsdf(node);
        if (!bsdf)
            return;
        auto const &g = static_cast<mx::HwShaderGenerator const &>(context.getShaderGenerator());
        // N, V, P, L and occlusion are HwSurfaceNode's, declared before the loop.
        g.emitComment("Pika's light loop (bake/generate.cpp)", stage);
        std::string const &vd = stage.getInputBlock(mx::HW::VERTEX_DATA).getInstance();
        g.emitLine("Lights lights = lights_at(P, " + vd + ".link)", stage);
        g.emitLine("uint light_index", stage);
        g.emitLine("while (lights.next(light_index))", stage, false);
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

  private:
    mx::ShaderNode const *bsdf(mx::ShaderNode const &node) const {
        mx::ShaderInput const *input = node.getInput(getBsdfInputName());
        return input ? input->getConnectedSibling() : nullptr;
    }
};

// The pixel block holds the public uniforms alone; the eye comes from the view.
// A surface reading another private uniform (a time node's u_time) is refused.
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
        // A private uniform is still its token here ($time), not its name (u_time).
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

// Size and modification time rather than bytes: a Houdini build stamps every file,
// so the listing also changes with the generator's shared objects.
void libraries() {
    std::filesystem::path root = materialx + "/libraries";
    std::vector<std::string> lines;
    for (auto const &entry : std::filesystem::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file())
            continue;
        // stat, as std::filesystem's clock has its own epoch before C++20.
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
