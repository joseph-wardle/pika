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

use crate::store::{Network, Node, Value};

/// A network's MaterialX document and its key.
#[derive(Debug)]
pub struct Document {
    pub text: String,
    /// FNV-1a of the text. It keys a disk cache, so it must not change with
    /// the Rust version as `DefaultHasher` may.
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
        hash: fnv1a(text.as_bytes()),
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

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Connection;

    /// The `network` fixture as husk syncs it: a UsdPreviewSurface reading
    /// a UsdUVTexture through a UsdPrimvarReader_float2.
    fn fixture() -> Network {
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
            Value::Asset("hydra/fixtures/network.usda".into()),
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

    // Houdini 22's MaterialX 1.39.5 generates this document; the generator
    // check proves it on every build.
    const FIXTURE: &str = "<?xml version=\"1.0\"?>
<materialx version=\"1.39\">
  <node name=\"n0\" nodedef=\"ND_UsdPreviewSurface_surfaceshader\">
    <input name=\"diffuseColor\" nodename=\"n1\" output=\"rgb\" />
  </node>
  <node name=\"n1\" nodedef=\"ND_UsdUVTexture_23\">
    <input name=\"st\" nodename=\"n2\" />
    <input name=\"wrapS\" type=\"string\" value=\"clamp\" />
  </node>
  <node name=\"n2\" nodedef=\"ND_geompropvalue_vector2\">
    <input name=\"geomprop\" type=\"string\" value=\"st\" />
  </node>
  <surfacematerial name=\"material\" type=\"material\">
    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"n0\" />
  </surfacematerial>
</materialx>
";

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
    fn text_is_escaped() {
        let mut quoted = fixture();
        let st = quoted.nodes.get_mut("st").unwrap();
        st.params
            .insert("varname".into(), Value::Text("a<\"b\">&c".into()));
        assert!(text(&quoted).contains("value=\"a&lt;&quot;b&quot;&gt;&amp;c\""));
    }
}
