//! A material's uniform block, what its surface shader asks of the
//! renderer: its byte layout and its textures' bindings, read from the bake
//! tool's manifest and slangc's reflection, and each material's values
//! packed into the block.

use std::collections::BTreeMap;

use crate::json::{self, Json};
use crate::material::{Document, Network, Value};

/// What the pixel stage expects of the renderer, as the tool lists it:
/// its public uniforms and textures.
#[derive(Debug, Default, PartialEq)]
pub struct Manifest {
    pub materialx: String,
    pub uniforms: Vec<Uniform>,
    pub textures: Vec<String>,
}

/// A public uniform: its variable, its MaterialX type and the initial
/// value the generator gave it, as MaterialX writes values ("0, 0, 1").
#[derive(Debug, PartialEq)]
pub struct Uniform {
    pub name: String,
    pub kind: String,
    pub initial: String,
}

pub fn manifest(text: &str) -> Result<Manifest, String> {
    let mut m = Manifest::default();
    for line in text.lines() {
        let mut words = line.splitn(3, ' ');
        let (Some(kind), Some(second)) = (words.next(), words.next()) else {
            return Err(format!("the manifest line {line:?} has no value"));
        };
        let rest = words.next().unwrap_or("");
        match kind {
            "materialx" => m.materialx = second.to_owned(),
            "texture" => m.textures.push(second.to_owned()),
            "uniform" => {
                let mut rest = rest.splitn(2, ' ');
                let name = rest.next().unwrap_or("").to_owned();
                if name.is_empty() {
                    return Err(format!("the manifest line {line:?} names no variable"));
                }
                m.uniforms.push(Uniform {
                    name,
                    kind: second.to_owned(),
                    initial: rest.next().unwrap_or("").to_owned(),
                });
            }
            _ => {
                return Err(format!(
                    "the manifest line {line:?} is not one the renderer knows"
                ));
            }
        }
    }
    Ok(m)
}

/// The uniform block's byte layout and its textures' bindings, in
/// descriptor set 0, as slangc laid them out.
#[derive(Debug, Default, PartialEq)]
pub struct Layout {
    pub size: usize,
    /// The block's binding.
    pub block: u32,
    /// Each uniform's byte offset, nested ones as `backsurfaceshader.color`.
    pub offsets: BTreeMap<String, usize>,
    pub textures: Vec<Bindings>,
}

/// A texture's bindings: its image, sampler and tile table.
#[derive(Debug, PartialEq)]
pub struct Bindings {
    pub name: String,
    pub image: u32,
    pub sampler: u32,
    pub tiles: u32,
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
                l.textures.push(Bindings {
                    name: name.to_owned(),
                    image: slot("tex")?,
                    sampler: slot("sampler")?,
                    tiles: slot("tiles")?,
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
        // The generator's own node for a geomprop read by default (UV0's
        // texcoord index) is in no document: its value is the initial one.
        let words = if uniform.name.starts_with("geomprop_") {
            shape.initial(&uniform.initial)?
        } else {
            let (path, node, input) = document.reader(network, &uniform.name)?;
            match node.params.get(input) {
                // A text value went into the document, and the generator
                // resolved it into the initial (a geomprop name), or is an
                // address mode, which only the sampler reads
                // (`material::address_modes`).
                Some(Value::Text(_)) | None => shape.initial(&uniform.initial)?,
                Some(value) => shape
                    .words(value)
                    .map_err(|e| format!("{path}'s {input} {e}"))?,
            }
        };
        for (i, word) in words.iter().enumerate() {
            bytes[offset + 4 * i..offset + 4 * i + 4].copy_from_slice(&word.to_ne_bytes());
        }
    }
    Ok(())
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
pub mod tests {
    use super::*;
    use crate::material::{self, tests::fixture};

    /// The bake tool wrote this for bake/fixtures/network.mtlx.
    pub const MANIFEST: &str = include_str!("../bake/fixtures/network.manifest");

    // slangc's reflection of the tool's output for bake/fixtures/network.mtlx,
    // copied from bake/check.sh's run when slangc or the libraries change.
    const REFLECTION: &str = include_str!("../bake/fixtures/network.reflection.json");

    pub fn fixture_manifest() -> Manifest {
        manifest(MANIFEST).unwrap()
    }

    #[test]
    fn reads_the_fixture_manifest() {
        let m = fixture_manifest();
        assert_eq!(m.materialx, "1.39.5");
        assert_eq!(m.textures, ["n1_file"]);
        assert_eq!(m.uniforms.len(), 21);
        assert_eq!(
            m.uniforms[4],
            Uniform {
                name: "n1_fallback".into(),
                kind: "color4".into(),
                initial: "0, 0, 0, 1".into(),
            }
        );
        assert_eq!(m.uniforms[0].initial, "", "a struct uniform has no value");

        assert!(manifest("uniform float").is_err());
        assert!(manifest("varying vector2 i_geomprop_st").is_err());
        assert!(manifest("sampler n1_file").is_err());
    }

    #[test]
    fn layout_follows_the_reflection() {
        let l = layout(REFLECTION).unwrap();
        assert_eq!((l.block, l.size), (0, 208));
        assert_eq!(l.offsets["n0_roughness"], 144);
        assert_eq!(l.offsets["n1_fallback"], 64);
        assert_eq!(l.offsets["backsurfaceshader.transparency"], 16);
        assert!(
            !l.offsets.contains_key("u_viewPosition"),
            "the eye is the view's"
        );
        assert_eq!(
            l.textures,
            [Bindings {
                name: "n1_file".into(),
                image: 1,
                sampler: 2,
                tiles: 3,
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
        let (manifest, l, network) = (fixture_manifest(), layout(REFLECTION).unwrap(), fixture());
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
        // An address mode is the sampler's alone: its uniform, which the
        // generated code never reads, keeps the initial value.
        assert_eq!(bytes[l.offsets["n1_wrapS"]], 2);

        // The generator's node for a default geomprop keeps its initial
        // value; any other uniform is a node's input.
        let mut extra = fixture_manifest();
        let mut extended = layout(REFLECTION).unwrap();
        extended
            .offsets
            .insert("geomprop_UV0_index".into(), extended.size);
        extended
            .offsets
            .insert("u_viewPosition".into(), extended.size + 4);
        extended.size += 8;
        let mut bytes = vec![0; extended.size];
        extra.uniforms.push(Uniform {
            name: "geomprop_UV0_index".into(),
            kind: "integer".into(),
            initial: "7".into(),
        });
        pack(&extra, &extended, &document, &network, &mut bytes).unwrap();
        assert_eq!(bytes[l.size], 7);
        extra.uniforms.push(Uniform {
            name: "u_viewPosition".into(),
            kind: "vector3".into(),
            initial: "0, 0, 0".into(),
        });
        let error = pack(&extra, &extended, &document, &network, &mut bytes).unwrap_err();
        assert!(error.contains("not a node's input"), "{error}");

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
