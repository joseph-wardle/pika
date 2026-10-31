//! A material's shading network as the MaterialX document Pika compiles, and
//! the key that says which materials share a shader.
//!
//! The document holds only what decides the generated code: the nodes
//! upstream of the surface, named by their MaterialX nodedefs, their
//! connections, and their text inputs (a geomprop name or a wrap mode is
//! compiled in). Numbers and assets are left out: they reach the shader as
//! uniforms and textures, so a material whose values change keeps its
//! shader, and materials of one structure share it.

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::cache;
use crate::store::{Connection, Network, Node, Value};

/// A texture as a node's input declares it: the file, the colour space the
/// file is in, and the encoding the bake tool stores it with.
#[derive(Debug, PartialEq)]
pub struct Source {
    pub file: String,
    pub space: String,
    pub encoding: &'static str,
}

/// The texture `node`'s input `input` reads. The encoding follows the
/// node's output type: colour compresses as BC7 sRGB, or as plain BC7 when
/// its space is raw; a float as BC4; a vector3 or vector4 as BC7; a
/// vector2 stays float, so a coordinate map stays exact (the oracles' uv
/// ramp). The space is the asset's `colorSpace` metadata, else
/// UsdUVTexture's `sourceColorSpace`, else `auto`, in OpenColorIO's names.
pub fn source(node: &Node, input: &str) -> Result<Source, String> {
    let file = match node.params.get(input) {
        Some(Value::Asset(path)) => path.clone(),
        Some(_) => return Err(format!("{input} is not an asset")),
        None => return Err(format!("{input} has no asset")),
    };
    let declared = (node.params.get(&format!("colorSpace:{input}")))
        .or_else(|| node.params.get("sourceColorSpace"));
    let space = match declared {
        Some(Value::Text(space)) if space == "sRGB" => "srgb_texture".to_owned(),
        Some(Value::Text(space)) => space.clone(),
        _ => "auto".to_owned(),
    };
    let kind = match node.node_type.as_str() {
        "UsdUVTexture" => "color4",
        name => (name.strip_prefix("ND_image_"))
            .or_else(|| name.strip_prefix("ND_tiledimage_"))
            .ok_or_else(|| format!("{name} reads a texture Pika does not know how to bake"))?,
    };
    let encoding = match kind {
        "color3" | "color4" if space == "raw" => "data",
        "color3" | "color4" => "color",
        "float" => "single",
        "vector2" => "float",
        "vector3" | "vector4" => "data",
        _ => return Err(format!("{} reads a texture of type {kind}", node.node_type)),
    };
    Ok(Source {
        file,
        space,
        encoding,
    })
}

/// A network's MaterialX document and its key.
#[derive(Debug)]
pub struct Document {
    pub text: String,
    /// FNV-1a of the text, which keys the cache.
    pub hash: u64,
    /// The store's node paths in document order: node `n2` is `nodes[2]`.
    pub nodes: Vec<String>,
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

/// The network of meshes with no material bound, or one Pika cannot show:
/// a clay surface, Lambert at 18 % grey, through the same path as any
/// other material.
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

/// The document for a network, or none when the network has no surface.
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
    for (i, path) in order.iter().enumerate() {
        let node = &network.nodes[*path];
        let (nodedef, rename) = resolve(path, node)?;
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
            if name.contains(':') || rename.dropped.contains(&name.as_str()) {
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
    Ok(Some(Document {
        hash: cache::fnv1a(&[text.as_bytes()]),
        nodes: order.iter().map(|p| (*p).to_owned()).collect(),
        text,
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
        assert_eq!(text(&edited), FIXTURE);

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
        assert_eq!(doc.text, FIXTURE);
        assert_eq!(doc.nodes, ["a", "b", "c"]);

        let mut spare = fixture.clone();
        spare
            .nodes
            .insert("spare".into(), fixture.nodes["tex"].clone());
        assert_eq!(text(&spare), FIXTURE);
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
        assert_eq!(
            doc.text,
            "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n  <node name=\"n0\" \
             nodedef=\"ND_surface\">\n    <input name=\"bsdf\" nodename=\"n1\" />\n  </node>\n  \
             <node name=\"n1\" nodedef=\"ND_oren_nayar_diffuse_bsdf\">\n  </node>\n  \
             <surfacematerial name=\"material\" type=\"material\">\n    <input \
             name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"n0\" />\n  \
             </surfacematerial>\n</materialx>\n"
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
            source(tex, "file").unwrap(),
            Source {
                file: "bake/fixtures/swatch.png".into(),
                space: "raw".into(),
                encoding: "data",
            }
        );
        let mut srgb = tex.clone();
        srgb.params.remove("colorSpace:file");
        srgb.params
            .insert("sourceColorSpace".into(), Value::Text("sRGB".into()));
        let s = source(&srgb, "file").unwrap();
        assert_eq!((s.space.as_str(), s.encoding), ("srgb_texture", "color"));
        srgb.params.remove("sourceColorSpace");
        assert_eq!(source(&srgb, "file").unwrap().space, "auto");

        let image = |kind: &str| {
            let mut node = tex.clone();
            node.node_type = format!("ND_image_{kind}");
            node.params.remove("colorSpace:file");
            source(&node, "file").map(|s| s.encoding)
        };
        assert_eq!(image("vector2"), Ok("float"));
        assert_eq!(image("float"), Ok("single"));
        assert_eq!(image("vector3"), Ok("data"));
        assert_eq!(image("color3"), Ok("color"));
        assert!(image("matrix33").is_err());
        let mut other = tex.clone();
        other.node_type = "ND_triplanarprojection_color3".into();
        assert!(source(&other, "file").is_err());
        assert!(source(tex, "st").is_err());
    }

    #[test]
    fn text_is_escaped() {
        let mut quoted = fixture();
        let st = quoted.nodes.get_mut("st").unwrap();
        st.params
            .insert("varname".into(), Value::Text("a<\"b\">&c".into()));
        assert!(text(&quoted).contains("value=\"a&lt;&quot;b&quot;&gt;&amp;c\""));
    }
}
