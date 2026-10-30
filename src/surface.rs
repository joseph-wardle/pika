//! What a material's surface shader asks of the renderer, read from the
//! bake tool's manifest and slangc's reflection: the vertex stage that
//! fills its varyings, the byte layout of its uniform block, and each
//! material's values packed into that block.

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::bake::Manifest;
use crate::json::{self, Json};
use crate::material::{self, Document};
use crate::store::{Network, Value};

/// The Slang source of a surface's vertex stage: `pixel.slang`'s
/// `VertexData`, filled from `shaders/vertex.slang`'s pulling.
pub fn vertex_stage(manifest: &Manifest) -> Result<String, String> {
    let mut text = String::from(
        "// Written by src/surface.rs: the vertex stage for the pixel stage beside it.\n\n\
         #include \"pixel.slang\"\n#include \"vertex.slang\"\n\n\
         [shader(\"vertex\")]\n\
         VertexData vs(uint vertex : SV_VulkanVertexID, uint slot : SV_VulkanInstanceID)\n\
         {\n    Vertex p = pull(vertex, slot);\n    VertexData o;\n    \
         o.SV_Position = p.position;\n",
    );
    for v in &manifest.varyings {
        let value = match v.name.as_str() {
            "positionWorld" => "p.world".to_owned(),
            "normalWorld" => "p.normal".to_owned(),
            "tangentWorld" => "tangent(p.normal)".to_owned(),
            "bitangentWorld" => "cross(p.normal, tangent(p.normal))".to_owned(),
            // Primvars reach the vertex table with textures.
            name if name.starts_with("i_geomprop_") => format!("{}(0)", slang_type(&v.kind)?),
            name => {
                return Err(format!(
                    "the surface reads {name}, which Pika's vertex stage does not provide"
                ));
            }
        };
        writeln!(text, "    o.{} = {value};", v.name).unwrap();
    }
    text.push_str("    return o;\n}\n");
    Ok(text)
}

fn slang_type(kind: &str) -> Result<&'static str, String> {
    Ok(match kind {
        "float" => "float",
        "vector2" => "float2",
        "vector3" | "color3" => "float3",
        "vector4" | "color4" => "float4",
        "integer" => "int",
        _ => {
            return Err(format!(
                "a varying of type {kind} has no Slang type in Pika"
            ));
        }
    })
}

/// The uniform block's byte layout and its textures' bindings, in
/// descriptor set 0, as slangc laid them out.
#[derive(Debug, Default, PartialEq)]
pub struct Layout {
    pub size: usize,
    /// The block's binding.
    pub block: u32,
    /// Each uniform's byte offset, nested ones as `u_lightData[0].direction`.
    pub offsets: BTreeMap<String, usize>,
    pub textures: Vec<Texture>,
}

#[derive(Debug, PartialEq)]
pub struct Texture {
    pub name: String,
    pub image: u32,
    pub sampler: u32,
}

/// Reads `slangc -reflection-json`'s output for a surface.
pub fn layout(reflection: &str) -> Result<Layout, String> {
    let root = json::parse(reflection)?;
    let block = (root.get("parameters").map_or(&[][..], Json::list).iter())
        .find(|p| p.get("name").and_then(Json::text) == Some("pixelCB"))
        .ok_or("the reflection has no pixelCB block")?;
    let mut l = Layout {
        block: index(block, "binding", "index")?,
        ..Default::default()
    };
    let fields = (block.get("type").and_then(|t| t.get("elementType")))
        .and_then(|t| t.get("fields"))
        .map_or(&[][..], Json::list);
    for field in fields {
        let name = field.get("name").and_then(Json::text).unwrap_or("");
        match field
            .get("binding")
            .and_then(|b| b.get("kind"))
            .and_then(Json::text)
        {
            Some("uniform") => {
                let offset = index(field, "binding", "offset")? as usize;
                l.size = l
                    .size
                    .max(offset + index(field, "binding", "size")? as usize);
                walk(name, field.get("type"), offset, &mut l.offsets)?;
            }
            Some("descriptorTableSlot") => {
                let base = l.block + index(field, "binding", "index")?;
                let slot = |member: &str| -> Result<u32, String> {
                    let fields = field.get("type").and_then(|t| t.get("fields"));
                    let found = (fields.map_or(&[][..], Json::list).iter())
                        .find(|f| f.get("name").and_then(Json::text) == Some(member))
                        .ok_or(format!("texture {name} has no {member}"))?;
                    Ok(base + index(found, "binding", "index")?)
                };
                l.textures.push(Texture {
                    name: name.to_owned(),
                    image: slot("tex")?,
                    sampler: slot("sampler")?,
                });
            }
            _ => return Err(format!("uniform {name} has a binding Pika does not know")),
        }
    }
    l.size = l.size.div_ceil(16) * 16;
    Ok(l)
}

fn index(value: &Json, member: &str, key: &str) -> Result<u32, String> {
    (value.get(member).and_then(|b| b.get(key)))
        .and_then(Json::number)
        .map(|n| n as u32)
        .ok_or_else(|| {
            let name = value.get("name").and_then(Json::text).unwrap_or("?");
            format!("the reflection of {name} has no {member}.{key}")
        })
}

/// Records the offset of `name` and, for a struct or an array, of each
/// member inside it.
fn walk(
    name: &str,
    kind: Option<&Json>,
    offset: usize,
    offsets: &mut BTreeMap<String, usize>,
) -> Result<(), String> {
    let kind = kind.ok_or(format!("uniform {name} has no type"))?;
    match kind.get("kind").and_then(Json::text) {
        Some("struct") => {
            for field in kind.get("fields").map_or(&[][..], Json::list) {
                let member = field.get("name").and_then(Json::text).unwrap_or("");
                let at = offset + index(field, "binding", "offset")? as usize;
                walk(&format!("{name}.{member}"), field.get("type"), at, offsets)?;
            }
        }
        Some("array") => {
            let count = kind
                .get("elementCount")
                .and_then(Json::number)
                .unwrap_or(0.0) as usize;
            let stride = kind
                .get("uniformStride")
                .and_then(Json::number)
                .unwrap_or(0.0) as usize;
            for i in 0..count {
                let at = offset + i * stride;
                walk(
                    &format!("{name}[{i}]"),
                    kind.get("elementType"),
                    at,
                    offsets,
                )?;
            }
        }
        _ => {
            offsets.insert(name.to_owned(), offset);
        }
    }
    Ok(())
}

/// Writes `network`'s values for `manifest`'s uniforms into `bytes`, a
/// block of `layout`: each from the store's value of the input, by
/// `document`'s node order, else from the generator's initial value.
pub fn pack(
    manifest: &Manifest,
    layout: &Layout,
    document: &Document,
    network: &Network,
    bytes: &mut [u8],
) -> Result<(), String> {
    for uniform in &manifest.uniforms {
        // Structs and strings have no value to pack; their bytes stay zero.
        let Some(shape) = Shape::of(&uniform.kind) else {
            continue;
        };
        let offset = *(layout.offsets.get(&uniform.name))
            .ok_or_else(|| format!("the reflection has no uniform {}", uniform.name))?;
        let (index, input) = (uniform.name.strip_prefix('n'))
            .and_then(|rest| rest.split_once('_'))
            .and_then(|(i, input)| Some((i.parse::<usize>().ok()?, input)))
            .ok_or_else(|| format!("uniform {} is not a node's input", uniform.name))?;
        let path = (document.nodes.get(index))
            .ok_or_else(|| format!("uniform {} names a node the document lacks", uniform.name))?;
        let node = &network.nodes[path];
        let words = match node.params.get(material::store_input(node, input)) {
            // A text value (a wrap mode) went into the document, and the
            // generator resolved it into the initial value.
            Some(Value::Text(_)) | None => shape.initial(&uniform.initial)?,
            Some(value) => shape
                .words(value)
                .map_err(|e| format!("{path}'s {input} {e}"))?,
        };
        for (i, word) in words.iter().enumerate() {
            bytes[offset + 4 * i..offset + 4 * i + 4].copy_from_slice(&word.to_ne_bytes());
        }
    }
    Ok(())
}

/// Writes what the generated light loop reads: the eye, and the headlight,
/// the directional light bake.cpp bound as type 1, along `forward`.
pub fn frame(layout: &Layout, bytes: &mut [u8], eye: [f32; 3], forward: [f32; 3]) {
    let mut put = |name: &str, words: &[u32]| {
        if let Some(&offset) = layout.offsets.get(name) {
            for (i, word) in words.iter().enumerate() {
                bytes[offset + 4 * i..offset + 4 * i + 4].copy_from_slice(&word.to_ne_bytes());
            }
        }
    };
    put("u_viewPosition", &eye.map(f32::to_bits));
    put("u_numActiveLightSources", &[1]);
    put("u_lightData[0].type", &[1]);
    put("u_lightData[0].direction", &forward.map(f32::to_bits));
    put("u_lightData[0].color", &[1.0f32; 3].map(f32::to_bits));
    put("u_lightData[0].intensity", &[1.0f32.to_bits()]);
}

/// A packable MaterialX type: how many 4-byte words, of what.
enum Shape {
    Floats(usize),
    Int,
    Bool,
}

impl Shape {
    fn of(kind: &str) -> Option<Self> {
        Some(match kind {
            "float" => Self::Floats(1),
            "vector2" => Self::Floats(2),
            "vector3" | "color3" => Self::Floats(3),
            "vector4" | "color4" => Self::Floats(4),
            "integer" => Self::Int,
            "boolean" => Self::Bool,
            _ => return None,
        })
    }

    fn words(&self, value: &Value) -> Result<Vec<u32>, String> {
        match (self, value) {
            (Self::Floats(n), Value::Float(v)) if v.len() == *n => {
                Ok(v.iter().map(|f| f.to_bits()).collect())
            }
            (Self::Floats(n), Value::Float(v)) => {
                Err(format!("has {} floats; the shader takes {n}", v.len()))
            }
            (Self::Int, Value::Int(i)) => Ok(vec![*i as u32]),
            (Self::Int, Value::Bool(b)) | (Self::Bool, Value::Bool(b)) => Ok(vec![u32::from(*b)]),
            (Self::Bool, Value::Int(i)) => Ok(vec![u32::from(*i != 0)]),
            (Self::Floats(_), _) => Err("is not a float value".into()),
            (Self::Int | Self::Bool, _) => Err("is not an integer or boolean value".into()),
        }
    }

    fn initial(&self, text: &str) -> Result<Vec<u32>, String> {
        let invalid = || format!("the initial value {text:?} does not parse");
        match self {
            Self::Floats(n) => {
                let words: Vec<u32> = (text.split(','))
                    .filter_map(|v| v.trim().parse::<f32>().ok().map(f32::to_bits))
                    .collect();
                if words.len() != *n {
                    return Err(invalid());
                }
                Ok(words)
            }
            Self::Int => Ok(vec![
                text.trim().parse::<i32>().map_err(|_| invalid())? as u32
            ]),
            Self::Bool => match text.trim() {
                "true" => Ok(vec![1]),
                "false" => Ok(vec![0]),
                _ => Err(invalid()),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::{self, Varying};
    use crate::material::tests::fixture;

    // bake/check.sh compiles this stage and proves the reflection beside it.
    const STAGE: &str = include_str!("../bake/fixtures/network.surface.slang");
    const REFLECTION: &str = include_str!("../bake/fixtures/network.reflection.json");

    fn manifest() -> Manifest {
        bake::manifest(bake::tests::FIXTURE).unwrap()
    }

    #[test]
    fn vertex_stage_fills_the_varyings() {
        assert_eq!(vertex_stage(&manifest()).unwrap(), STAGE);

        let mut unknown = manifest();
        unknown.varyings.push(Varying {
            name: "texcoord_0".into(),
            kind: "vector2".into(),
        });
        let error = vertex_stage(&unknown).unwrap_err();
        assert!(error.contains("texcoord_0"), "{error}");
    }

    #[test]
    fn layout_follows_the_reflection() {
        let l = layout(REFLECTION).unwrap();
        assert_eq!((l.block, l.size), (0, 272));
        assert_eq!(l.offsets["n0_roughness"], 160);
        assert_eq!(l.offsets["n1_fallback"], 80);
        assert_eq!(l.offsets["u_lightData[0].direction"], 240);
        assert_eq!(l.offsets["backsurfaceshader.transparency"], 32);
        assert_eq!(
            l.textures,
            [Texture {
                name: "n1_file".into(),
                image: 1,
                sampler: 2
            }]
        );

        let error = layout(&REFLECTION.replace("pixelCB", "other")).unwrap_err();
        assert!(error.contains("pixelCB"), "{error}");
    }

    fn f32_at(bytes: &[u8], offset: usize) -> f32 {
        f32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn pack_takes_the_store_values_over_the_initial_ones() {
        let (manifest, l, network) = (manifest(), layout(REFLECTION).unwrap(), fixture());
        let document = material::document(&network).unwrap().unwrap();
        let mut bytes = vec![0; l.size];
        pack(&manifest, &l, &document, &network, &mut bytes).unwrap();
        // Authored in the store.
        assert_eq!(f32_at(&bytes, l.offsets["n0_roughness"]), 0.25);
        assert_eq!(bytes[l.offsets["n0_useSpecularWorkflow"]], 1);
        // Not authored: the generator's initial values.
        assert_eq!(f32_at(&bytes, l.offsets["n0_ior"]), 1.5);
        let fallback = l.offsets["n1_fallback"];
        assert_eq!(
            (0..4)
                .map(|i| f32_at(&bytes, fallback + 4 * i))
                .collect::<Vec<_>>(),
            [0.0, 0.0, 0.0, 1.0]
        );
        // The wrap token was compiled in as the nodedef's integer.
        assert_eq!(bytes[l.offsets["n1_wrapS"]], 1);

        frame(&l, &mut bytes, [1.0, 2.0, 3.0], [0.0, 0.0, -1.0]);
        assert_eq!(
            f32_at(&bytes, l.offsets["u_lightData[0].direction"] + 8),
            -1.0
        );
        assert_eq!(bytes[l.offsets["u_numActiveLightSources"]], 1);

        let mut wide = fixture();
        let roughness = |network: &mut Network, value| {
            let surface = network.nodes.get_mut("surface").unwrap();
            surface.params.insert("roughness".into(), value);
        };
        roughness(&mut wide, Value::Float(vec![0.5, 0.5]));
        let error = pack(&manifest, &l, &document, &wide, &mut bytes).unwrap_err();
        assert!(
            error.starts_with("surface's roughness has 2 floats"),
            "{error}"
        );
        roughness(&mut wide, Value::Int(1));
        assert!(pack(&manifest, &l, &document, &wide, &mut bytes).is_err());
    }
}
