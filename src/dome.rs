//! The dome light a view is lit by, and its sky's irradiance toward every direction.

use ash::vk;

use crate::bake::Bake;
use crate::gpu::{self, Buffer};
use crate::light;
use crate::material::Source;
use crate::store::{self, Shape, Store};

/// `shaders/view.slang`'s `IRRADIANCE`, as texels.
const IRRADIANCE: u32 = 64 * 32;
/// The widest level of an image summed: irradiance is too smooth to tell a finer one.
const SUMMED: u32 = 256;
/// A sky of one value, with texels enough that their sum is as near as an image's.
const FLAT: [u32; 2] = [SUMMED, SUMMED / 2];

/// `shaders/view.slang`'s `Dome`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Constants {
    axes: [[f32; 3]; 3],
    radiance: [f32; 3],
    irradiance: vk::DeviceAddress,
}

/// `shaders/dome.slang`'s `Push`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    sky: vk::DeviceAddress,
    irradiance: vk::DeviceAddress,
    extent: [u32; 2],
    first: u32,
    count: u32,
    stride: u32,
    unused: u32,
}

/// What a dome shows before its colour and intensity.
#[derive(Clone, PartialEq, Debug)]
enum Sky {
    /// No dome lights.
    Black,
    /// A dome with no image.
    White,
    Image(store::Texture),
}

pub struct Dome {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    compute: gpu::Compute,
    bake: Bake,
    /// The sky's texels, 3 floats each, which the pass reads the frame the sky changes.
    sky: Buffer,
    irradiance: Buffer,
    summed: Option<Sky>,
    /// The domes last warned of as giving no light.
    warned: Vec<String>,
}

impl Dome {
    pub fn new(device: &gpu::Device, spirv: &[u32], bake: Bake) -> Result<Self, String> {
        Ok(Self {
            device: device.raw.clone(),
            compute: gpu::Compute::new(device, spirv, c"cs", size_of::<Push>())?,
            bake,
            sky: sky_buffer(device, 12)?,
            irradiance: Buffer::device_local(
                device,
                IRRADIANCE as u64 * 12,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?,
            summed: None,
            warned: Vec::new(),
        })
    }

    /// Records the irradiance pass into `cb`, outside rendering, when the sky changed. No GPU
    /// work may be pending.
    pub fn update(
        &mut self,
        device: &gpu::Device,
        store: &Store,
        cb: vk::CommandBuffer,
    ) -> Result<Constants, String> {
        let (lit, rest) = choose(store);
        if rest != self.warned {
            if !rest.is_empty() {
                eprintln!(
                    "pika: warning: one dome light lights a view; {} give no light",
                    rest.join(", ")
                );
            }
            self.warned = rest;
        }
        let (sky, axes, radiance) = match lit {
            Some((light, params, transform)) => {
                let sky = match &light.texture {
                    Some(texture) if !texture.file.is_empty() => Sky::Image(texture.clone()),
                    _ => Sky::White,
                };
                (sky, axes(transform), light::radiance(params))
            }
            None => (Sky::Black, [[0.0; 3]; 3], [0.0; 3]),
        };
        if self.summed.as_ref() != Some(&sky) {
            // A failed image is not tried again until the sky changes.
            let (extent, texels) = self.texels(&sky).unwrap_or_else(|e| {
                eprintln!("pika: warning: a dome light's image: {e}; it gives no light");
                flat(0.0)
            });
            self.sky = sky_buffer(device, texels.len() as u64 * 4)?;
            self.sky.bytes().copy_from_slice(gpu::bytes(&texels));
            self.record(cb, extent);
            self.summed = Some(sky);
        }
        Ok(Constants {
            axes,
            radiance,
            irradiance: self.irradiance.address,
        })
    }

    /// A sky's extent and texels from the top row, 3 floats each.
    fn texels(&self, sky: &Sky) -> Result<([u32; 2], Vec<f32>), String> {
        let image = match sky {
            Sky::Black => return Ok(flat(0.0)),
            Sky::White => return Ok(flat(1.0)),
            Sky::Image(image) => image,
        };
        if image.format != "automatic" && image.format != "latlong" {
            eprintln!(
                "pika: warning: {} is read as a latlong, not as {}",
                image.file, image.format
            );
        }
        let source = Source {
            node: String::new(),
            input: String::new(),
            file: image.file.clone(),
            space: "auto".into(),
            encoding: "float",
            default: Vec::new(),
            transformed: false,
        };
        let texture = self.bake.texture(&source)?.texture;
        let channels = match texture.format {
            vk::Format::R32_SFLOAT => 1,
            vk::Format::R32G32B32A32_SFLOAT => 4,
            _ => return Err(format!("{} has two channels", image.file)),
        };
        let level = texture.first(SUMMED);
        let (width, height) = texture.extent(level);
        // A set of tiles' first.
        let mut bytes = vec![0; (width * height) as usize * channels * 4];
        (texture.read_at(texture.levels[level].start, &mut bytes))
            .map_err(|e| format!("cannot read the bake of {}: {e}", image.file))?;
        let texels = (bytes.chunks_exact(channels * 4))
            .flat_map(|texel| {
                [0, 1, 2].map(|c| {
                    let at = 4 * c.min(channels - 1);
                    f32::from_le_bytes(texel[at..at + 4].try_into().unwrap())
                })
            })
            .collect();
        Ok(([width, height], texels))
    }

    fn record(&self, cb: vk::CommandBuffer, extent: [u32; 2]) {
        let texels = std::slice::from_ref(&(0..IRRADIANCE));
        (self.compute).record(cb, texels, |first, count, stride| Push {
            sky: self.sky.address,
            irradiance: self.irradiance.address,
            extent,
            first,
            count,
            stride,
            unused: 0,
        });
        let summed = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ);
        // SAFETY: `cb` is recording outside any rendering.
        unsafe {
            (self.device).cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().memory_barriers(&[summed]),
            );
        }
    }

    pub fn memory(&self) -> gpu::Memory {
        let mut memory = self.sky.memory();
        memory += self.irradiance.memory();
        memory
    }
}

fn flat(value: f32) -> ([u32; 2], Vec<f32>) {
    (FLAT, vec![value; (FLAT[0] * FLAT[1] * 3) as usize])
}

fn sky_buffer(device: &gpu::Device, size: u64) -> Result<Buffer, String> {
    Buffer::host(
        device,
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
}

type Lit<'a> = (&'a store::Light, &'a store::LightParams, &'a [f64; 16]);

/// The dome that lights, the first shown with an image or else the first shown, as Storm
/// chooses; and the paths of the other domes shown.
fn choose(store: &Store) -> (Option<Lit<'_>>, Vec<String>) {
    let shown: Vec<Lit> = (store.lights().values())
        .filter_map(|l| match (&l.params, &l.transform) {
            (Some(p), Some(m)) if p.shape == Shape::Dome && p.visible => Some((l, p, m)),
            _ => None,
        })
        .collect();
    let imaged = |l: &store::Light| l.texture.as_ref().is_some_and(|t| !t.file.is_empty());
    let lit = shown.iter().position(|(l, ..)| imaged(l)).unwrap_or(0);
    let rest = (shown.iter().enumerate())
        .filter(|(i, _)| *i != lit)
        .map(|(_, (l, ..))| l.path.clone())
        .collect();
    (shown.get(lit).copied(), rest)
}

/// World to a dome's axes: where its transform sends +x, +y and +z, each made unit, so a
/// scaled dome turns the same.
fn axes(m: &[f64; 16]) -> [[f32; 3]; 3] {
    [0, 1, 2].map(|r| {
        let row = [m[4 * r], m[4 * r + 1], m[4 * r + 2]];
        let length = row.iter().map(|c| c * c).sum::<f64>().sqrt();
        row.map(|c| {
            if length > 0.0 {
                (c / length) as f32
            } else {
                0.0
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{LightParams, Texture};

    const IDENTITY: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    fn light(store: &mut Store, path: &str, shape: Shape, visible: bool, file: &str) {
        let id = store.insert_light(path.into());
        let params = LightParams {
            intensity: 1.0,
            exposure: 0.0,
            color: [1.0; 3],
            normalize: false,
            shape,
            visible,
            shadow: true,
        };
        store.set_light_params(id, params).unwrap();
        store.set_light_transform(id, IDENTITY).unwrap();
        let texture = Texture {
            file: file.into(),
            format: "automatic".into(),
        };
        store.set_light_texture(id, texture).unwrap();
    }

    fn chosen(store: &Store) -> (Option<&str>, Vec<String>) {
        let (lit, rest) = choose(store);
        (lit.map(|(l, ..)| l.path.as_str()), rest)
    }

    #[test]
    fn the_first_dome_shown_with_an_image_lights() {
        let mut store = Store::new();
        assert_eq!(chosen(&store), (None, vec![]));
        light(&mut store, "/sun", Shape::Distant { angle: 0.5 }, true, "");
        light(&mut store, "/hidden", Shape::Dome, false, "sky.exr");
        assert_eq!(chosen(&store), (None, vec![]));
        light(&mut store, "/debug", Shape::Dome, true, "");
        assert_eq!(chosen(&store), (Some("/debug"), vec![]));
        light(&mut store, "/sky", Shape::Dome, true, "sky.exr");
        light(&mut store, "/other", Shape::Dome, true, "other.exr");
        // Negative control: the first dome shown has no image, so it is not the one.
        let rest = vec!["/debug".to_string(), "/other".to_string()];
        assert_eq!(chosen(&store), (Some("/sky"), rest));
    }

    #[test]
    fn axes_take_a_world_direction_to_the_domes() {
        // A quarter turn about y, scaled by 2: the dome's +x lies along world -z.
        let mut m = IDENTITY;
        (m[0], m[2]) = (0.0, -2.0);
        (m[8], m[10]) = (2.0, 0.0);
        let a = axes(&m);
        let along = |d: [f32; 3]| a.map(|row| row[0] * d[0] + row[1] * d[1] + row[2] * d[2]);
        assert_eq!(along([0.0, 0.0, -1.0]), [1.0, 0.0, 0.0]);
        assert_eq!(along([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0]);
        // Negative control: unturned, world -z is not the dome's +x.
        assert_ne!(axes(&IDENTITY).map(|row| -row[2]), [1.0, 0.0, 0.0]);
    }
}
