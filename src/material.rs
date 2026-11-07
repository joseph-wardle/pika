//! A material's shading network, as Hydra hands it over and as the MaterialX
//! document Pika compiles, and the key that says which materials share a shader.
//!
//! The document holds only what decides the generated code: the nodes
//! upstream of the surface, named by their MaterialX nodedefs, their
//! connections, and their text inputs (a geomprop name is compiled in).
//! Numbers, assets and address modes are left out: they reach the shader
//! as uniforms, textures and samplers, so a material whose values change
//! keeps its shader, and materials of one structure share it.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::cache;

/// `HdMaterialNetwork2`: shader nodes by path, and the node whose output is
/// the material's surface, when it has one.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Network {
    pub nodes: BTreeMap<String, Node>,
    pub surface: Option<String>,
}

/// One shader node: its identifier ("UsdPreviewSurface",
/// "ND_open_pbr_surface_surfaceshader"), its authored values by input name,
/// and the upstream outputs connected to its inputs. An input may have both.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Node {
    pub node_type: String,
    pub params: BTreeMap<String, Value>,
    pub connections: BTreeMap<String, Connection>,
}

/// An upstream node's output.
#[derive(Clone, PartialEq, Debug)]
pub struct Connection {
    pub node: String,
    pub output: String,
}

/// A parameter value as the `VtValue` held it. Numbers keep their width (a
/// `GfVec3f` is three floats) but not whether they are a colour or a
/// vector, which the node's definition says.
#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    Float(Vec<f32>),
    Int(i32),
    Bool(bool),
    /// A token or string.
    Text(String),
    /// An asset's resolved path.
    Asset(String),
}

impl Network {
    /// The bytes Hydra handed over for the network, as counted for a pull.
    pub fn bytes(&self) -> usize {
        let mut bytes = self.surface.as_ref().map_or(0, String::len);
        for (path, node) in &self.nodes {
            bytes += path.len() + node.node_type.len();
            for (name, value) in &node.params {
                bytes += name.len()
                    + match value {
                        Value::Float(v) => size_of_val(v.as_slice()),
                        Value::Int(_) | Value::Bool(_) => 4,
                        Value::Text(s) | Value::Asset(s) => s.len(),
                    };
            }
            for (name, c) in &node.connections {
                bytes += name.len() + c.node.len() + c.output.len();
            }
        }
        bytes
    }
}

/// A texture as a material asks the bake tool for it: the file, the colour
/// space the file is in, the encoding it is stored with, and, when
/// the file is missing, the value the node draws instead.
#[derive(Debug, PartialEq)]
pub struct Source {
    /// The node reading it, by path.
    pub node: String,
    /// The input it is read through, as the store names it.
    pub input: String,
    pub file: String,
    pub space: String,
    pub encoding: &'static str,
    /// A UsdUVTexture's `fallback` or an image node's `default`, its
    /// nodedef's when unauthored, for a file that is empty; empty itself
    /// for a file that resolves, so an edit to a drawn fallback, and no
    /// other, makes a new source.
    pub default: Vec<f32>,
}

/// The texture `node`, at `path`, reads through its input `input`. The
/// encoding follows the node's output type: colour compresses as BC7
/// sRGB, or as plain BC7 when its space is raw; a float as BC4; a vector3
/// or vector4 as BC7; a vector2 stays float, so a coordinate map stays
/// exact (the oracles' uv ramp). The space is the asset's `colorSpace` metadata, else
/// UsdUVTexture's `sourceColorSpace`, in OpenColorIO's names, else `auto`
/// for a colour and raw for the rest.
pub fn source(path: &str, node: &Node, input: &str) -> Result<Source, String> {
    let file = match node.params.get(input) {
        Some(Value::Asset(file)) => file.clone(),
        Some(_) => return Err(format!("{path}'s {input} is not an asset")),
        None => return Err(format!("{path}'s {input} has no asset")),
    };
    let kind = match node.node_type.as_str() {
        "UsdUVTexture" => "color4",
        name => (name.strip_prefix("ND_image_"))
            .or_else(|| name.strip_prefix("ND_tiledimage_"))
            .ok_or_else(|| {
                format!("{path} is {name}, which reads a texture Pika does not know how to bake")
            })?,
    };
    let declared = (node.params.get(&format!("colorSpace:{input}")))
        .or_else(|| node.params.get("sourceColorSpace"));
    let space = match declared {
        Some(Value::Text(space)) if space == "sRGB" => "srgb_texture".to_owned(),
        Some(Value::Text(space)) => space.clone(),
        // USD's auto, sRGB at 8 bits, is a rule for colour: a normal or
        // roughness map at 8 bits is read as it is, as Karma reads it.
        _ if kind == "color3" || kind == "color4" => "auto".to_owned(),
        _ => "raw".to_owned(),
    };
    let encoding = match kind {
        "color3" | "color4" if space == "raw" => "data",
        "color3" | "color4" => "color",
        "float" => "single",
        "vector2" => "float",
        "vector3" | "vector4" => "data",
        _ => {
            return Err(format!(
                "{path} is {}, which reads a texture of type {kind}",
                node.node_type
            ));
        }
    };
    // The delegate hands over the path as it resolved, so a file that does
    // not exist arrives empty, as one never authored does, and the node's
    // fallback is drawn, as Storm draws it.
    let usd = node.node_type == "UsdUVTexture";
    let default = match node.params.get(if usd { "fallback" } else { "default" }) {
        _ if !file.is_empty() => Vec::new(),
        Some(Value::Float(v)) => v.clone(),
        _ if usd => vec![0.0, 0.0, 0.0, 1.0],
        _ => vec![0.0; 4],
    };
    Ok(Source {
        node: path.to_owned(),
        input: input.to_owned(),
        file,
        space,
        encoding,
        default,
    })
}

/// A network's MaterialX document, its key, and the texcoord its meshes
/// carry.
#[derive(Clone, Debug)]
pub struct Document {
    pub text: String,
    /// FNV-1a of the text, which keys the cache.
    pub hash: u64,
    /// The store's node paths in document order: node `n2` is `nodes[2]`.
    pub nodes: Vec<String>,
    /// The float2 primvar the surface reads: the first its reader nodes
    /// name, else `st`, which is UV0 (bake/generate.cpp says which inputs
    /// read UV0 and refuses a surface reading two).
    pub texcoord: String,
}

impl Document {
    /// The node of `network`, by path, and its input as the store names
    /// it, that the generated variable `name` stands for.
    pub fn reader<'a>(
        &'a self,
        network: &'a Network,
        name: &'a str,
    ) -> Result<(&'a str, &'a Node, &'a str), String> {
        let (index, input) = node_input(name)?;
        let path = (self.nodes.get(index))
            .ok_or_else(|| format!("{name} names a node the document lacks"))?;
        let node = &network.nodes[path];
        Ok((path, node, store_input(node, input)))
    }
}

/// The node and input a generated variable stands for: `n1_file` is node
/// 1's `file`.
fn node_input(name: &str) -> Result<(usize, &str), String> {
    (name.strip_prefix('n'))
        .and_then(|rest| rest.split_once('_'))
        .and_then(|(i, input)| Some((i.parse::<usize>().ok()?, input)))
        .ok_or_else(|| format!("{name} is not a node's input"))
}

/// The inputs that choose a texture's sampler, u then v: a UsdUVTexture's
/// and an image node's. The generator compiles a text input in, but these
/// land as integer uniforms the generated code never reads, since the
/// sampler addresses the texture, so they are values: left out of the
/// document and read into the sampler (`address_modes`), and an edit to
/// one compiles nothing.
pub const ADDRESS_MODES: [&str; 4] = ["wrapS", "wrapT", "uaddressmode", "vaddressmode"];

/// The address modes `node` samples its texture through, u then v, in the
/// image nodedefs' enumeration: its UsdUVTexture `wrapS`/`wrapT` or image
/// `uaddressmode`/`vaddressmode` tokens. A UsdUVTexture's `useMetadata`,
/// USD's default when none is authored, asks for the `file`'s own, as the
/// bake read them (`texture::Texture::wrap`); an image node authoring none
/// is periodic, its nodedef's initial value.
pub fn address_modes(node: &Node, file: [u32; 2]) -> Result<[u32; 2], String> {
    let usd = node.node_type == "UsdUVTexture";
    let mode = |names: [&str; 2], axis: usize| -> Result<u32, String> {
        let authored = names.iter().find_map(|name| match node.params.get(*name) {
            Some(Value::Text(token)) => Some((name, token)),
            _ => None,
        });
        match authored {
            Some((name, token)) => {
                let mode = address_mode(token).map_err(|e| format!("{name} {e}"))?;
                Ok(mode.unwrap_or(file[axis]))
            }
            None if usd => Ok(file[axis]),
            None => Ok(2),
        }
    };
    Ok([
        mode(["wrapS", "uaddressmode"], 0)?,
        mode(["wrapT", "vaddressmode"], 1)?,
    ])
}

/// The integer the image nodedefs' enumeration gives the address mode
/// `token`, USD's names included (black is constant, repeat periodic), or
/// none for `useMetadata`, which asks for the file's own.
fn address_mode(token: &str) -> Result<Option<u32>, String> {
    Ok(Some(match token {
        "black" | "constant" => 0,
        "clamp" => 1,
        "repeat" | "periodic" => 2,
        "mirror" => 3,
        "useMetadata" => return Ok(None),
        _ => {
            return Err(format!(
                "is {token}, not black (constant), clamp, repeat (periodic), mirror or useMetadata"
            ));
        }
    }))
}

/// A USD shader id and the MaterialX nodedef that implements it, with the
/// inputs and outputs whose names differ and the inputs the nodedef lacks.
struct Rename {
    usd: &'static str,
    nodedef: &'static str,
    inputs: &'static [(&'static str, &'static str)],
    outputs: &'static [(&'static str, &'static str)],
    dropped: &'static [&'static str],
}

const RENAMES: &[Rename] = &[
    Rename {
        usd: "UsdPreviewSurface",
        nodedef: "ND_UsdPreviewSurface_surfaceshader",
        inputs: &[],
        outputs: &[("surface", "out")],
        dropped: &[],
    },
    Rename {
        usd: "UsdUVTexture",
        nodedef: "ND_UsdUVTexture_23",
        inputs: &[],
        outputs: &[],
        // The texture's colour space is applied when it is loaded.
        dropped: &["sourceColorSpace"],
    },
    Rename {
        usd: "UsdPrimvarReader_float2",
        nodedef: "ND_geompropvalue_vector2",
        inputs: &[("varname", "geomprop"), ("fallback", "default")],
        outputs: &[("result", "out")],
        dropped: &[],
    },
    Rename {
        usd: "UsdTransform2d",
        nodedef: "ND_UsdTransform2d",
        inputs: &[],
        outputs: &[("result", "out")],
        dropped: &[],
    },
];

/// A MaterialX node: nothing to rename.
const NONE: &Rename = &Rename {
    usd: "",
    nodedef: "",
    inputs: &[],
    outputs: &[],
    dropped: &[],
};

/// The network of meshes with no material bound: a clay surface, Lambert
/// at 18 % grey, through the same path as any other material.
pub fn clay() -> Network {
    let mut surface = Node {
        node_type: "ND_surface".into(),
        ..Node::default()
    };
    let bsdf = Connection {
        node: "/clay/bsdf".into(),
        output: "out".into(),
    };
    surface.connections.insert("bsdf".into(), bsdf);
    let mut diffuse = Node {
        node_type: "ND_oren_nayar_diffuse_bsdf".into(),
        ..Node::default()
    };
    diffuse
        .params
        .insert("color".into(), Value::Float(vec![0.18; 3]));
    Network {
        nodes: BTreeMap::from([
            ("/clay/surface".to_owned(), surface),
            ("/clay/bsdf".to_owned(), diffuse),
        ]),
        surface: Some("/clay/surface".into()),
    }
}

/// The network of a material Pika could not build: an unlit magenta
/// emission, an error under any lighting.
pub fn pink() -> Network {
    let mut surface = Node {
        node_type: "ND_surface".into(),
        ..Node::default()
    };
    let edf = Connection {
        node: "/pink/edf".into(),
        output: "out".into(),
    };
    surface.connections.insert("edf".into(), edf);
    let mut emission = Node {
        node_type: "ND_uniform_edf".into(),
        ..Node::default()
    };
    emission
        .params
        .insert("color".into(), Value::Float(vec![1.0, 0.0, 1.0]));
    Network {
        nodes: BTreeMap::from([
            ("/pink/surface".to_owned(), surface),
            ("/pink/edf".to_owned(), emission),
        ]),
        surface: Some("/pink/surface".into()),
    }
}

/// The document for a network, or none when the network has no surface.
/// A mesh is drawn with one texcoord, the primvar its material reads:
/// ALab's meshes carry `st` and `perfuv` both and each material reads one
/// of them, so carrying every float2 primvar would double the texture
/// coordinates for nothing.
pub fn document(network: &Network) -> Result<Option<Document>, String> {
    let Some(surface) = &network.surface else {
        return Ok(None);
    };
    let mut order = Vec::new();
    visit(network, surface, &mut order)?;
    let index: BTreeMap<&str, usize> = order.iter().enumerate().map(|(i, p)| (*p, i)).collect();

    // The library version the generator links. MaterialX upgrades older
    // documents on reading.
    let mut text = String::from("<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n");
    let mut primvars = BTreeSet::new();
    for (i, path) in order.iter().enumerate() {
        let node = &network.nodes[*path];
        let (nodedef, rename) = resolve(path, node)?;
        if nodedef == "ND_geompropvalue_vector2"
            && let Some(Value::Text(name)) = node.params.get(store_input(node, "geomprop"))
        {
            primvars.insert(name.clone());
        }
        // The element's category and type are the nodedef's to say and Pika
        // has no MaterialX library; the generator resolves the `nodedef`
        // attribute without them.
        writeln!(text, "  <node name=\"n{i}\" nodedef=\"{nodedef}\">").unwrap();
        let mut inputs = BTreeMap::new();
        for (name, value) in &node.params {
            let Value::Text(value) = value else {
                continue;
            };
            // Hydra adds `typeName:` and `colorSpace:` entries beside the
            // inputs; they describe the values this document leaves out.
            if name.contains(':')
                || rename.dropped.contains(&name.as_str())
                || ADDRESS_MODES.contains(&name.as_str())
            {
                continue;
            }
            let attributes = format!("type=\"string\" value=\"{}\"", escape(value));
            inputs.insert(renamed(rename.inputs, name), attributes);
        }
        for (name, connection) in &node.connections {
            let upstream = &network.nodes[&connection.node];
            let (_, upstream_rename) = resolve(&connection.node, upstream)?;
            let mut attributes = format!("nodename=\"n{}\"", index[connection.node.as_str()]);
            // Only a multi-output node names its outputs; a single one is `out`.
            let output = renamed(upstream_rename.outputs, &connection.output);
            if output != "out" {
                write!(attributes, " output=\"{output}\"").unwrap();
            }
            inputs.insert(renamed(rename.inputs, name), attributes);
        }
        for (name, attributes) in inputs {
            writeln!(text, "    <input name=\"{name}\" {attributes} />").unwrap();
        }
        text.push_str("  </node>\n");
    }
    text.push_str(
        "  <surfacematerial name=\"material\" type=\"material\">\n    <input \
         name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"n0\" />\n  \
         </surfacematerial>\n</materialx>\n",
    );
    // A surface reading two is the bake tool's to refuse, naming both.
    let texcoord = primvars
        .into_iter()
        .next()
        .unwrap_or_else(|| "st".to_owned());
    Ok(Some(Document {
        hash: cache::fnv1a(&[text.as_bytes()]),
        nodes: order.iter().map(|p| (*p).to_owned()).collect(),
        text,
        texcoord,
    }))
}

/// Lists the nodes upstream of `path` in document order: each node before
/// what feeds it, inputs by name.
fn visit<'a>(network: &'a Network, path: &'a str, order: &mut Vec<&'a str>) -> Result<(), String> {
    if order.contains(&path) {
        return Ok(());
    }
    let node = network
        .nodes
        .get(path)
        .ok_or_else(|| format!("node {path} is connected but not in the network"))?;
    order.push(path);
    for connection in node.connections.values() {
        visit(network, &connection.node, order)?;
    }
    Ok(())
}

fn resolve<'a>(path: &str, node: &'a Node) -> Result<(&'a str, &'static Rename), String> {
    if let Some(rename) = RENAMES.iter().find(|r| r.usd == node.node_type) {
        return Ok((rename.nodedef, rename));
    }
    if node.node_type.starts_with("ND_") {
        return Ok((&node.node_type, NONE));
    }
    Err(format!(
        "node {path} has type {}, which is neither a UsdPreviewSurface shader nor a MaterialX \
         nodedef",
        node.node_type
    ))
}

/// The store's name for `node`'s MaterialX input: the USD name the rename
/// table maps to it, else the same name.
pub fn store_input<'a>(node: &Node, input: &'a str) -> &'a str {
    let Some(rename) = RENAMES.iter().find(|r| r.usd == node.node_type) else {
        return input;
    };
    (rename.inputs.iter())
        .find(|(_, to)| *to == input)
        .map_or(input, |(from, _)| from)
}

fn renamed<'a>(table: &[(&'static str, &'static str)], name: &'a str) -> &'a str {
    table
        .iter()
        .find(|(from, _)| *from == name)
        .map_or(name, |(_, to)| to)
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// The `network` fixture as husk syncs it: a UsdPreviewSurface reading
    /// a UsdUVTexture through a UsdPrimvarReader_float2.
    pub fn fixture() -> Network {
        let mut surface = Node {
            node_type: "UsdPreviewSurface".into(),
            ..Default::default()
        };
        surface
            .params
            .insert("roughness".into(), Value::Float(vec![0.25]));
        surface
            .params
            .insert("typeName:roughness".into(), Value::Text("float".into()));
        surface
            .params
            .insert("useSpecularWorkflow".into(), Value::Int(1));
        surface
            .connections
            .insert("diffuseColor".into(), connection("tex", "rgb"));
        let mut tex = Node {
            node_type: "UsdUVTexture".into(),
            ..Default::default()
        };
        tex.params.insert(
            "file".into(),
            Value::Asset("bake/fixtures/swatch.png".into()),
        );
        tex.params
            .insert("colorSpace:file".into(), Value::Text("raw".into()));
        tex.params
            .insert("sourceColorSpace".into(), Value::Text("auto".into()));
        tex.params
            .insert("wrapS".into(), Value::Text("clamp".into()));
        tex.params
            .insert("typeName:wrapS".into(), Value::Text("token".into()));
        tex.connections
            .insert("st".into(), connection("st", "result"));
        let mut st = Node {
            node_type: "UsdPrimvarReader_float2".into(),
            ..Default::default()
        };
        st.params.insert("varname".into(), Value::Text("st".into()));
        let mut network = Network {
            surface: Some("surface".into()),
            ..Default::default()
        };
        network.nodes.insert("surface".into(), surface);
        network.nodes.insert("tex".into(), tex);
        network.nodes.insert("st".into(), st);
        network
    }

    fn connection(node: &str, output: &str) -> Connection {
        Connection {
            node: node.into(),
            output: output.into(),
        }
    }

    fn text(network: &Network) -> String {
        document(network).unwrap().unwrap().text
    }

    fn hash(network: &Network) -> u64 {
        document(network).unwrap().unwrap().hash
    }

    fn texcoord(network: &Network) -> Result<Option<String>, String> {
        document(network).map(|d| d.map(|d| d.texcoord))
    }

    // Houdini 22's MaterialX 1.39.5 generates this document; bake/check.sh
    // proves it against the manifest beside it.
    const FIXTURE: &str = include_str!("../bake/fixtures/network.mtlx");

    #[test]
    fn usd_nodes_become_nodedefs() {
        let network = fixture();
        let doc = document(&network).unwrap().unwrap();
        assert_eq!(doc.text, FIXTURE);
        assert_eq!(doc.nodes, ["surface", "tex", "st"]);

        let mut mtlx = fixture();
        mtlx.nodes.get_mut("st").unwrap().node_type = "Foo".into();
        let error = document(&mtlx).unwrap_err();
        assert!(error.starts_with("node st has type "), "{error}");
        mtlx.nodes.get_mut("st").unwrap().node_type = "ND_texcoord_vector2".into();
        assert!(text(&mtlx).contains("nodedef=\"ND_texcoord_vector2\""));
    }

    #[test]
    fn values_are_not_structure() {
        let mut edited = fixture();
        let surface = edited.nodes.get_mut("surface").unwrap();
        surface
            .params
            .insert("roughness".into(), Value::Float(vec![0.75]));
        surface
            .params
            .insert("useSpecularWorkflow".into(), Value::Int(0));
        let tex = edited.nodes.get_mut("tex").unwrap();
        tex.params
            .insert("file".into(), Value::Asset("/other.exr".into()));
        tex.params
            .insert("sourceColorSpace".into(), Value::Text("sRGB".into()));
        assert_eq!(hash(&edited), hash(&fixture()));

        // A text input is compiled in.
        let mut edited = fixture();
        edited
            .nodes
            .get_mut("st")
            .unwrap()
            .params
            .insert("varname".into(), Value::Text("uv".into()));
        assert_ne!(hash(&edited), hash(&fixture()));
        assert!(text(&edited).contains("value=\"uv\""));
    }

    #[test]
    fn the_texcoord_is_the_one_primvar_read() {
        assert_eq!(texcoord(&fixture()).unwrap(), Some("st".to_owned()));

        // A network naming no primvar reads UV0, which is st; one with no
        // surface reads none.
        let mut uv0 = fixture();
        (uv0.nodes.get_mut("tex").unwrap().connections).remove("st");
        assert_eq!(texcoord(&uv0).unwrap(), Some("st".to_owned()));
        assert_eq!(texcoord(&clay()).unwrap(), Some("st".to_owned()));
        assert_eq!(texcoord(&Network::default()).unwrap(), None);

        // A second reader of the same name is one texcoord; of another,
        // the first by name, which the bake tool then refuses.
        let mut network = fixture();
        let mut reader = network.nodes["st"].clone();
        network.nodes.insert("uv".into(), reader.clone());
        (network.nodes.get_mut("tex").unwrap().connections)
            .insert("bias".into(), connection("uv", "result"));
        assert_eq!(texcoord(&network).unwrap(), Some("st".to_owned()));
        reader
            .params
            .insert("varname".into(), Value::Text("perfuv".into()));
        network.nodes.insert("uv".into(), reader);
        assert_eq!(texcoord(&network).unwrap(), Some("perfuv".to_owned()));
        // The control: a reader nothing reads through does not count.
        (network.nodes.get_mut("tex").unwrap().connections).remove("bias");
        assert_eq!(texcoord(&network).unwrap(), Some("st".to_owned()));
    }

    #[test]
    fn address_modes_are_values() {
        let wrapped = |token: &str| {
            let mut network = fixture();
            let tex = network.nodes.get_mut("tex").unwrap();
            tex.params.insert("wrapS".into(), Value::Text(token.into()));
            tex.params.insert("wrapT".into(), Value::Text(token.into()));
            network
        };
        for token in ["repeat", "mirror", "useMetadata"] {
            assert_eq!(hash(&wrapped(token)), hash(&fixture()), "{token}");
        }
        let mut image = fixture();
        let tex = image.nodes.get_mut("tex").unwrap();
        tex.node_type = "ND_image_color3".into();
        (tex.params).insert("uaddressmode".into(), Value::Text("constant".into()));
        (tex.params).insert("vaddressmode".into(), Value::Text("mirror".into()));
        assert!(!text(&image).contains("addressmode"));
        // The control: a text input that is not an address mode is
        // compiled in, whatever the token.
        let mut reader = fixture();
        let st = reader.nodes.get_mut("st").unwrap();
        st.params
            .insert("varname".into(), Value::Text("mirror".into()));
        assert!(text(&reader).contains("name=\"geomprop\" type=\"string\" value=\"mirror\""));
    }

    #[test]
    fn connections_are_structure() {
        let mut moved = fixture();
        let surface = moved.nodes.get_mut("surface").unwrap();
        let connection = surface.connections.remove("diffuseColor").unwrap();
        surface
            .connections
            .insert("emissiveColor".into(), connection);
        assert_ne!(hash(&moved), hash(&fixture()));

        let mut cut = fixture();
        cut.nodes.get_mut("tex").unwrap().connections.clear();
        assert_ne!(hash(&cut), hash(&fixture()));
        assert_eq!(cut.nodes.len(), 3, "the cut-off node stays in the network");
        assert_eq!(document(&cut).unwrap().unwrap().nodes, ["surface", "tex"]);
    }

    #[test]
    fn paths_are_not_structure() {
        let mut renamed = Network {
            surface: Some("a".into()),
            ..Default::default()
        };
        let fixture = fixture();
        let mut a = fixture.nodes["surface"].clone();
        a.connections
            .insert("diffuseColor".into(), connection("b", "rgb"));
        let mut b = fixture.nodes["tex"].clone();
        b.connections.insert("st".into(), connection("c", "result"));
        renamed.nodes.insert("a".into(), a);
        renamed.nodes.insert("b".into(), b);
        renamed
            .nodes
            .insert("c".into(), fixture.nodes["st"].clone());
        let doc = document(&renamed).unwrap().unwrap();
        assert_eq!(doc.hash, hash(&fixture));
        assert_eq!(doc.nodes, ["a", "b", "c"]);

        let mut spare = fixture.clone();
        spare
            .nodes
            .insert("spare".into(), fixture.nodes["tex"].clone());
        assert_eq!(hash(&spare), hash(&fixture));
        let surface = spare.nodes.get_mut("surface").unwrap();
        surface
            .connections
            .insert("emissiveColor".into(), connection("spare", "rgb"));
        assert_eq!(
            document(&spare).unwrap().unwrap().nodes,
            ["surface", "tex", "st", "spare"]
        );
    }

    #[test]
    fn clay_is_a_lambert_surface() {
        let doc = document(&clay()).unwrap().unwrap();
        assert!(
            doc.text.contains("nodedef=\"ND_oren_nayar_diffuse_bsdf\""),
            "{}",
            doc.text
        );
        assert_eq!(doc.nodes, ["/clay/surface", "/clay/bsdf"]);
        // The grey is a value, packed as a uniform, not structure.
        let mut white = clay();
        let bsdf = white.nodes.get_mut("/clay/bsdf").unwrap();
        bsdf.params
            .insert("color".into(), Value::Float(vec![1.0; 3]));
        assert_eq!(document(&white).unwrap().unwrap().hash, doc.hash);
        let mut rough = clay();
        let bsdf = rough.nodes.get_mut("/clay/bsdf").unwrap();
        bsdf.params
            .insert("roughness".into(), Value::Float(vec![0.5]));
        assert_eq!(document(&rough).unwrap().unwrap().hash, doc.hash);
        assert_ne!(document(&fixture()).unwrap().unwrap().hash, doc.hash);
    }

    #[test]
    fn pink_is_an_unlit_emission() {
        let doc = document(&pink()).unwrap().unwrap();
        assert!(
            doc.text.contains("nodedef=\"ND_uniform_edf\""),
            "{}",
            doc.text
        );
        assert_eq!(doc.nodes, ["/pink/surface", "/pink/edf"]);
        assert_ne!(document(&clay()).unwrap().unwrap().hash, doc.hash);
    }

    #[test]
    fn no_surface_no_document() {
        assert!(document(&Network::default()).unwrap().is_none());
        let mut unterminated = fixture();
        unterminated.surface = None;
        assert!(document(&unterminated).unwrap().is_none());
        unterminated.surface = Some("missing".into());
        assert!(
            document(&unterminated)
                .unwrap_err()
                .starts_with("node missing is connected")
        );
    }

    #[test]
    fn sources_follow_the_node_and_its_colour_space() {
        let tex = &fixture().nodes["tex"];
        assert_eq!(
            source("tex", tex, "file").unwrap(),
            Source {
                node: "tex".into(),
                input: "file".into(),
                file: "bake/fixtures/swatch.png".into(),
                space: "raw".into(),
                encoding: "data",
                default: vec![],
            }
        );
        // A missing file carries the fallback the node draws instead.
        let mut missing = tex.clone();
        missing
            .params
            .insert("file".into(), Value::Asset(String::new()));
        assert_eq!(
            source("tex", &missing, "file").unwrap().default,
            [0.0, 0.0, 0.0, 1.0]
        );
        (missing.params).insert("fallback".into(), Value::Float(vec![0.2, 0.4, 0.6, 1.0]));
        assert_eq!(
            source("tex", &missing, "file").unwrap().default,
            [0.2, 0.4, 0.6, 1.0]
        );
        missing.node_type = "ND_image_float".into();
        assert_eq!(source("tex", &missing, "file").unwrap().default, [0.0; 4]);
        missing
            .params
            .insert("default".into(), Value::Float(vec![0.5]));
        assert_eq!(source("tex", &missing, "file").unwrap().default, [0.5]);
        let mut srgb = tex.clone();
        srgb.params.remove("colorSpace:file");
        srgb.params
            .insert("sourceColorSpace".into(), Value::Text("sRGB".into()));
        let s = source("tex", &srgb, "file").unwrap();
        assert_eq!((s.space.as_str(), s.encoding), ("srgb_texture", "color"));
        srgb.params.remove("sourceColorSpace");
        assert_eq!(source("tex", &srgb, "file").unwrap().space, "auto");

        // Declaring nothing, a colour is sRGB when 8-bit, the rest raw.
        let image = |kind: &str| {
            let mut node = tex.clone();
            node.node_type = format!("ND_image_{kind}");
            node.params.remove("colorSpace:file");
            node.params.remove("sourceColorSpace");
            source("tex", &node, "file").map(|s| (s.space, s.encoding))
        };
        assert_eq!(image("vector2"), Ok(("raw".into(), "float")));
        assert_eq!(image("float"), Ok(("raw".into(), "single")));
        assert_eq!(image("vector3"), Ok(("raw".into(), "data")));
        assert_eq!(image("color3"), Ok(("auto".into(), "color")));
        assert!(image("matrix33").is_err());
        let mut other = tex.clone();
        other.node_type = "ND_triplanarprojection_color3".into();
        assert!(source("tex", &other, "file").is_err());
        assert!(source("tex", tex, "st").is_err());
    }

    #[test]
    fn text_is_escaped() {
        let mut quoted = fixture();
        let st = quoted.nodes.get_mut("st").unwrap();
        st.params
            .insert("varname".into(), Value::Text("a<\"b\">&c".into()));
        assert!(text(&quoted).contains("value=\"a&lt;&quot;b&quot;&gt;&amp;c\""));
    }

    #[test]
    fn address_modes_follow_the_node_then_the_file() {
        let network = fixture();
        let tex = &network.nodes["tex"];
        // wrapT, unauthored, is useMetadata: the file's, as is a wrapS
        // saying so.
        assert_eq!(address_modes(tex, [3, 0]).unwrap(), [1, 0]);
        let mut metadata = tex.clone();
        (metadata.params).insert("wrapS".into(), Value::Text("useMetadata".into()));
        assert_eq!(address_modes(&metadata, [3, 0]).unwrap(), [3, 0]);
        // An image node authoring none is periodic; its tokens are the
        // enumeration's own.
        let mut image = tex.clone();
        image.node_type = "ND_image_color3".into();
        image.params.remove("wrapS");
        assert_eq!(address_modes(&image, [3, 0]).unwrap(), [2, 2]);
        (image.params).insert("uaddressmode".into(), Value::Text("constant".into()));
        assert_eq!(address_modes(&image, [3, 0]).unwrap(), [0, 2]);
        (image.params).insert("vaddressmode".into(), Value::Text("bogus".into()));
        let error = address_modes(&image, [3, 0]).unwrap_err();
        assert!(error.starts_with("vaddressmode is bogus"), "{error}");
        let surface = &network.nodes["surface"];
        assert_eq!(
            address_modes(surface, [3, 0]).unwrap(),
            [2, 2],
            "no texture: periodic"
        );
    }

    #[test]
    fn reader_reads_the_generated_name() {
        let network = fixture();
        let doc = document(&network).unwrap().unwrap();
        let (path, node, input) = doc.reader(&network, "n1_file").unwrap();
        assert_eq!(
            (path, node.node_type.as_str(), input),
            ("tex", "UsdUVTexture", "file")
        );
        assert!(
            doc.reader(&network, "n9_file")
                .unwrap_err()
                .contains("lacks")
        );
        assert!(doc.reader(&network, "u_viewPosition").is_err());
    }
}
