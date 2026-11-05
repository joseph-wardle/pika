//! A material's uniform block, what its surface shader asks of the
//! renderer: its byte layout and its textures' bindings, read from the bake
//! tool's manifest and slangc's reflection, and each material's values
//! packed into the block.

use std::collections::BTreeMap;

use crate::bake::Manifest;
use crate::json::{self, Json};
use crate::material::{self, Document};
use crate::store::{Network, Node, Value};

/// The uniform block's byte layout and its textures' bindings, in
/// descriptor set 0, as slangc laid them out.
#[derive(Debug, Default, PartialEq)]
pub struct Layout {
    pub size: usize,
    /// The block's binding.
    pub block: u32,
    /// Each uniform's byte offset, nested ones as `u_lightData[0].direction`.
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
            let (index, input) = node_input(&uniform.name)?;
            let path = (document.nodes.get(index)).ok_or_else(|| {
                format!("uniform {} names a node the document lacks", uniform.name)
            })?;
            let node = &network.nodes[path];
            match node.params.get(material::store_input(node, input)) {
                // An address mode's token is its integer; useMetadata, the
                // file's own, leaves the nodedef's (the sampler reads the
                // file: `address_modes`).
                Some(Value::Text(token)) if material::ADDRESS_MODES.contains(&input) => {
                    match address_mode(token).map_err(|e| format!("{path}'s {input} {e}"))? {
                        Some(mode) => vec![mode],
                        None => shape.initial(&uniform.initial)?,
                    }
                }
                // Any other text value (a geomprop name) went into the
                // document, and the generator resolved it into the initial.
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

/// The node and input a generated variable stands for: `n1_file` is node
/// 1's `file`.
pub fn node_input(name: &str) -> Result<(usize, &str), String> {
    (name.strip_prefix('n'))
        .and_then(|rest| rest.split_once('_'))
        .and_then(|(i, input)| Some((i.parse::<usize>().ok()?, input)))
        .ok_or_else(|| format!("uniform {name} is not a node's input"))
}

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
pub fn address_mode(token: &str) -> Result<Option<u32>, String> {
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
    // A Lambert surface reflects albedo / π of a light's intensity, so a
    // headlight of π shows a facing surface's albedo.
    put(
        "u_lightData[0].intensity",
        &[std::f32::consts::PI.to_bits()],
    );
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
    use crate::bake;
    use crate::material::tests::fixture;

    // slangc's reflection of the tool's output for bake/fixtures/network.mtlx,
    // copied from bake/check.sh's run when slangc or the libraries change.
    const REFLECTION: &str = include_str!("../bake/fixtures/network.reflection.json");

    fn manifest() -> Manifest {
        bake::manifest(bake::tests::FIXTURE).unwrap()
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
        // The wrap token is packed as the nodedef's integer, and read into
        // the sampler; wrapT, unauthored, is useMetadata: the file's, as
        // is a wrapS saying so, the block keeping the initial value.
        assert_eq!(bytes[l.offsets["n1_wrapS"]], 1);
        assert_eq!(bytes[l.offsets["n1_wrapT"]], 2);
        let tex = &network.nodes["tex"];
        assert_eq!(address_modes(tex, [3, 0]).unwrap(), [1, 0]);
        let mut metadata = tex.clone();
        (metadata.params).insert("wrapS".into(), Value::Text("useMetadata".into()));
        assert_eq!(address_modes(&metadata, [3, 0]).unwrap(), [3, 0]);
        let mut edited = network.clone();
        let wrap_s = |network: &mut Network, token: &str| {
            let tex = network.nodes.get_mut("tex").unwrap();
            tex.params.insert("wrapS".into(), Value::Text(token.into()));
        };
        wrap_s(&mut edited, "mirror");
        pack(&manifest, &l, &document, &edited, &mut bytes).unwrap();
        assert_eq!(bytes[l.offsets["n1_wrapS"]], 3);
        wrap_s(&mut edited, "bogus");
        let error = pack(&manifest, &l, &document, &edited, &mut bytes).unwrap_err();
        assert!(error.starts_with("tex's wrapS is bogus"), "{error}");
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
        assert_eq!(node_input("n12_file").unwrap(), (12, "file"));
        assert!(node_input("u_viewPosition").is_err());

        // The generator's node for a default geomprop keeps its initial
        // value; any other uniform is a node's input.
        let mut extra = bake::manifest(bake::tests::FIXTURE).unwrap();
        let mut extended = layout(REFLECTION).unwrap();
        extended
            .offsets
            .insert("geomprop_UV0_index".into(), extended.size);
        extended
            .offsets
            .insert("u_viewPosition".into(), extended.size + 4);
        extended.size += 8;
        let mut bytes = vec![0; extended.size];
        extra.uniforms.push(bake::Uniform {
            name: "geomprop_UV0_index".into(),
            kind: "integer".into(),
            initial: "7".into(),
        });
        pack(&extra, &extended, &document, &network, &mut bytes).unwrap();
        assert_eq!(bytes[l.size], 7);
        extra.uniforms.push(bake::Uniform {
            name: "u_viewPosition".into(),
            kind: "vector3".into(),
            initial: "0, 0, 0".into(),
        });
        let error = pack(&extra, &extended, &document, &network, &mut bytes).unwrap_err();
        assert!(error.contains("not a node's input"), "{error}");

        frame(&l, &mut bytes, [1.0, 2.0, 3.0], [0.0, 0.0, -1.0]);
        assert_eq!(
            f32_at(&bytes, l.offsets["u_lightData[0].direction"] + 8),
            -1.0
        );
        assert_eq!(bytes[l.offsets["u_numActiveLightSources"]], 1);
        assert_eq!(
            f32_at(&bytes, l.offsets["u_lightData[0].intensity"]),
            std::f32::consts::PI
        );

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
