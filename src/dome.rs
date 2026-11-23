//! The dome light that lights a view. From its sky this makes the irradiance toward every
//! direction, a blurred copy of the sky for each roughness of reflection, and a sun lifted out
//! of the sky.

use std::f64::consts::{PI, TAU};
use std::ops::Range;

use ash::vk;

use crate::bake::Bake;
use crate::gpu::{self, Buffer};
use crate::light;
use crate::material::Source;
use crate::store::{self, Shape, Store};

/// `IRRADIANCE` in `shaders/view.slang`, counted in texels.
const IRRADIANCE: u32 = 64 * 32;
/// The widest level of an image that is summed. Irradiance is too smooth to show anything
/// finer.
const SUMMED: u32 = 256;
/// The widest level of an image that is reflected, enough for a `REFLECTION` square's detail at
/// the horizon.
const REFLECTED: u32 = 2 * REFLECTION;
/// These must match `REFLECTION`, `REFLECTION_LEVELS`, `REFLECTION_LEAST`, `REFLECTION_ROUGHEST`
/// and `REFLECTION_SPACING` in `shaders/view.slang`.
const REFLECTION: u32 = 1024;
const LEVELS: u32 = 8;
const LEAST: u32 = 128;
const ROUGHEST: f32 = 0.6;
const SPACING: f32 = 0.75;
/// How many directions of the sky each blurred texel reads.
const SAMPLES: u32 = 4096;
/// The size of a sky with one value, with enough texels that its sum is as close as an image's.
const FLAT: [u32; 2] = [SUMMED, SUMMED / 2];
/// The smallest share of a sky's light worth lifting into a sun.
const SHARE: f64 = 0.01;
const BAND: u32 = 64;

/// Must match `Dome` in `shaders/view.slang`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Constants {
    axes: [[f32; 3]; 3],
    radiance: [f32; 3],
    irradiance: vk::DeviceAddress,
    reflection: vk::DeviceAddress,
}

/// Must match `Push` in `shaders/dome.slang`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    sky: vk::DeviceAddress,
    into: vk::DeviceAddress,
    extent: [u32; 2],
    first: u32,
    count: u32,
    stride: u32,
    level: u32,
    alpha: f32,
    samples: u32,
}

/// What a dome shows, before its colour and intensity.
#[derive(Clone, PartialEq, Debug)]
enum Sky {
    Black,
    /// A dome with no image.
    White,
    Image(store::Texture),
}

/// The part of a sky's light over the threshold, lifted out as EEVEE does.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Lifted {
    toward: [f32; 3],
    /// Its texels summed over their solid angles, before the dome's colour and intensity.
    light: [f32; 3],
}

/// A sky's extent within `REFLECTED`, its texels from the top row with 3 floats each, and its
/// sun.
type Texels = ([u32; 2], Vec<f32>, Option<Lifted>);

pub struct Dome {
    // A copy of the owner's device table. The owner's device outlives this, by field order.
    device: ash::Device,
    sum: gpu::Compute,
    fold: gpu::Compute,
    halve: gpu::Compute,
    blur: gpu::Compute,
    bake: Bake,
    /// The sky's texels within `SUMMED` and within `REFLECTED`, 3 floats each. The passes read them
    /// in the frame the sky changes.
    summed_sky: Buffer,
    reflected_sky: Buffer,
    irradiance: Buffer,
    reflection: Buffer,
    /// The sky that was summed, and the scale that takes its texels to 1 at the sun's threshold.
    summed: Option<(Sky, [f32; 3])>,
    sun: Option<Lifted>,
    /// The domes last warned about for giving no light.
    warned: Vec<String>,
}

impl Dome {
    pub fn new(device: &gpu::Device, spirv: &[u32], bake: Bake) -> Result<Self, String> {
        let compute = |name| gpu::Compute::new(device, spirv, name, size_of::<Push>());
        let local = |texels: u32| {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
            Buffer::device_local(device, texels as u64 * 12, usage)
        };
        Ok(Self {
            device: device.raw.clone(),
            sum: compute(c"sum")?,
            fold: compute(c"fold")?,
            halve: compute(c"halve")?,
            blur: compute(c"blur")?,
            bake,
            summed_sky: sky_buffer(device, 12)?,
            reflected_sky: sky_buffer(device, 12)?,
            irradiance: local(IRRADIANCE)?,
            // The blurred levels, then the plain halvings below the sharp square.
            reflection: local(
                level(LEVELS) + (1..=HALVINGS).map(|h| halving(h).pow(2)).sum::<u32>(),
            )?,
            summed: None,
            sun: None,
            warned: Vec::new(),
        })
    }

    /// Records the irradiance pass into `cb`, outside a render pass, when the sky or what
    /// `threshold` lifts from it has changed. No GPU work may be pending.
    pub fn update(
        &mut self,
        device: &gpu::Device,
        store: &Store,
        cb: vk::CommandBuffer,
        threshold: f32,
    ) -> Result<(Constants, Option<light::Sun>), String> {
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
        let (id, sky, axes, radiance) = match lit {
            Some((id, light, params, transform)) => {
                let sky = match &light.texture {
                    Some(texture) if !texture.file.is_empty() => Sky::Image(texture.clone()),
                    _ => Sky::White,
                };
                (id, sky, axes(transform), light::radiance(params))
            }
            // Unturned rather than zero, because a zero direction has no place in a folded square.
            None => (0, Sky::Black, axes(&IDENTITY), [0.0; 3]),
        };
        // The threshold applies to the light a view sees, as EEVEE's does.
        let scale = if threshold > 0.0 && matches!(sky, Sky::Image(_)) {
            radiance.map(|l| l / threshold)
        } else {
            [0.0; 3]
        };
        let summed = (sky, scale);
        if self.summed.as_ref() != Some(&summed) {
            // A failed image isn't tried again until the sky changes.
            let (extent, texels, sun) = self.texels(&summed.0, scale).unwrap_or_else(|e| {
                eprintln!("pika: warning: a dome light's image: {e}; it gives no light");
                flat(0.0)
            });
            let coarse = summed_extent(extent);
            let summed_texels = average(extent, &texels, coarse);
            self.summed_sky = sky_buffer(device, summed_texels.len() as u64 * 4)?;
            (self.summed_sky.bytes()).copy_from_slice(gpu::bytes(&summed_texels));
            self.reflected_sky = sky_buffer(device, texels.len() as u64 * 4)?;
            (self.reflected_sky.bytes()).copy_from_slice(gpu::bytes(&texels));
            self.record(cb, coarse, extent);
            self.sun = sun;
            self.summed = Some(summed);
        }
        let constants = Constants {
            axes,
            radiance,
            irradiance: self.irradiance.address,
            reflection: self.reflection.address,
        };
        let sun = self.sun.and_then(|sun| {
            // The transpose of the axes turns back, which is exact for an unsheared dome.
            let d = [0, 1, 2].map(|c| (0..3).map(|r| axes[r][c] * sun.toward[r]).sum::<f32>());
            let length = d.iter().map(|c| c * c).sum::<f32>().sqrt();
            (length > 0.0).then(|| light::Sun {
                dome: id,
                direction: d.map(|c| c / length),
                illuminance: [0, 1, 2].map(|c| sun.light[c] * radiance[c]),
            })
        });
        Ok((constants, sun))
    }

    /// A sky's texels. For an image, whatever `scale` takes over 1 is lifted out into the sun.
    fn texels(&self, sky: &Sky, scale: [f32; 3]) -> Result<Texels, String> {
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
        // Rows of one level of the first tile of a set, 3 floats a texel.
        let rows = |level: usize, rows: Range<u32>| -> Result<Vec<f32>, String> {
            let row = texture.extent(level).0 as usize * channels * 4;
            let mut bytes = vec![0; rows.len() * row];
            let from = texture.levels[level].start + rows.start as usize * row;
            (texture.read_at(from, &mut bytes))
                .map_err(|e| format!("cannot read the bake of {}: {e}", image.file))?;
            let texels = bytes.chunks_exact(channels * 4).flat_map(|texel| {
                [0, 1, 2].map(|c| {
                    let at = 4 * c.min(channels - 1);
                    f32::from_le_bytes(texel[at..at + 4].try_into().unwrap())
                })
            });
            Ok(texels.collect())
        };
        let level = texture.first(REFLECTED);
        let (width, height) = texture.extent(level);
        if scale != [0.0; 3] {
            // The whole image, because a level's average would hide a sun.
            let (w, h) = texture.extent(0);
            let mut lift = Lift::new(scale, [w, h], [width, height]);
            for y in (0..h).step_by(BAND as usize) {
                let band = rows(0, y..(y + BAND).min(h))?;
                for (row, texels) in (y..).zip(band.chunks_exact(3 * w as usize)) {
                    lift.row(row, texels);
                }
            }
            if let Some((texels, sun)) = lift.finish() {
                return Ok(([width, height], texels, Some(sun)));
            }
        }
        Ok(([width, height], rows(level, 0..height)?, None))
    }

    /// Records the irradiance image, then the sharp square and its halvings, each made from the
    /// one before, then the blurred levels made from those.
    fn record(&self, cb: vk::CommandBuffer, summed: [u32; 2], reflected: [u32; 2]) {
        let pass = |compute: &gpu::Compute, texels: u32, push: Push| {
            let texels = 0..texels;
            compute.record(cb, std::slice::from_ref(&texels), |first, count, stride| {
                Push {
                    first,
                    count,
                    stride,
                    ..push
                }
            });
        };
        let written = |reader| {
            let barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(reader)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ);
            // SAFETY: the command buffer is recording outside a render pass.
            unsafe {
                (self.device).cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().memory_barriers(&[barrier]),
                );
            }
        };
        let push = Push {
            sky: self.summed_sky.address,
            into: self.irradiance.address,
            extent: summed,
            first: 0,
            count: 0,
            stride: 0,
            level: 0,
            alpha: 0.0,
            samples: SAMPLES,
        };
        pass(&self.sum, IRRADIANCE, push);
        let push = Push {
            sky: self.reflected_sky.address,
            into: self.reflection.address,
            extent: reflected,
            ..push
        };
        pass(&self.fold, side(0).pow(2), push);
        for level in 1..=HALVINGS {
            written(vk::PipelineStageFlags2::COMPUTE_SHADER);
            pass(&self.halve, halving(level).pow(2), Push { level, ..push });
        }
        written(vk::PipelineStageFlags2::COMPUTE_SHADER);
        for level in 1..LEVELS {
            let alpha = roughness(level).powi(2);
            let push = Push {
                level,
                alpha,
                ..push
            };
            pass(&self.blur, side(level).pow(2), push);
        }
        written(vk::PipelineStageFlags2::FRAGMENT_SHADER);
    }

    pub fn memory(&self) -> gpu::Memory {
        let mut memory = self.summed_sky.memory();
        memory += self.reflected_sky.memory();
        memory += self.irradiance.memory();
        memory += self.reflection.memory();
        memory
    }
}

const HALVINGS: u32 = REFLECTION.ilog2();

/// A copy of `reflection_side` in `shaders/view.slang`.
fn side(level: u32) -> u32 {
    (REFLECTION >> level).max(LEAST)
}

fn halving(halving: u32) -> u32 {
    REFLECTION >> halving
}

/// A copy of `reflection_level` in `shaders/view.slang`.
fn level(level: u32) -> u32 {
    (0..level).map(|l| side(l).pow(2)).sum()
}

/// The roughness a blurred level holds, as `dome_radiance` in `shaders/frame.slang` reads it.
fn roughness(level: u32) -> f32 {
    ROUGHEST * (level as f32 / (LEVELS - 1) as f32).powf(1.0 / SPACING)
}

fn summed_extent(extent: [u32; 2]) -> [u32; 2] {
    let over = (0..).take_while(|l| (extent[0].max(extent[1]) >> l) > SUMMED);
    extent.map(|side| (side >> over.clone().count()).max(1))
}

/// A latlong's texels averaged down to `to`, which is no wider or taller than `from`.
fn average(from: [u32; 2], texels: &[f32], to: [u32; 2]) -> Vec<f32> {
    let [w, h] = from.map(|v| v as usize);
    let [to_w, to_h] = to.map(|v| v as usize);
    let mut sums = vec![0.0; 3 * to_w * to_h];
    let mut counts = vec![0u32; to_w * to_h];
    for (i, texel) in texels.as_chunks::<3>().0.iter().enumerate() {
        let bin = (i / w * to_h / h) * to_w + (i % w) * to_w / w;
        counts[bin] += 1;
        for c in 0..3 {
            sums[3 * bin + c] += texel[c];
        }
    }
    (sums.as_chunks::<3>().0.iter().zip(&counts))
        .flat_map(|(sum, &count)| [0, 1, 2].map(|c| sum[c] / count.max(1) as f32))
        .collect()
}

fn flat(value: f32) -> Texels {
    (FLAT, vec![value; (FLAT[0] * FLAT[1] * 3) as usize], None)
}

/// Sums a latlong's texels over a threshold into a sun, and averages the rest down.
struct Lift {
    /// The scale that takes a texel's channels to 1 at the threshold.
    scale: [f32; 3],
    from: [u32; 2],
    to: [u32; 2],
    sums: Vec<f32>,
    counts: Vec<u32>,
    light: [f64; 3],
    toward: [f64; 3],
    lifted: f64,
    total: f64,
}

impl Lift {
    /// `to` must be no wider or taller than `from`.
    fn new(scale: [f32; 3], from: [u32; 2], to: [u32; 2]) -> Self {
        let texels = (to[0] * to[1]) as usize;
        Self {
            scale,
            from,
            to,
            sums: vec![0.0; 3 * texels],
            counts: vec![0; texels],
            light: [0.0; 3],
            toward: [0.0; 3],
            lifted: 0.0,
            total: 0.0,
        }
    }

    /// Adds row `y`, counted from the top, with 3 floats a texel.
    fn row(&mut self, y: u32, texels: &[f32]) {
        let [w, h] = self.from.map(|v| v as usize);
        let [to_w, to_h] = self.to.map(|v| v as usize);
        // The row's band of the sphere, shared evenly by its texels, as in `shaders/dome.slang`.
        let edge = |y: u32| (PI * y as f64 / h as f64).cos();
        let solid_angle = (edge(y) - edge(y + 1)) * TAU / w as f64;
        let latitude = (0.5 - (y as f64 + 0.5) / h as f64) * PI;
        let first = (y as usize * to_h / h) * to_w;
        for (x, texel) in texels.as_chunks::<3>().0.iter().enumerate() {
            let scaled = [0, 1, 2].map(|c| texel[c] * self.scale[c]);
            let over = scaled.into_iter().fold(0.0, f32::max);
            // The colour kept, with its largest channel at the threshold.
            let kept = if over > 1.0 { 1.0 / over } else { 1.0 };
            let bin = first + x * to_w / w;
            self.counts[bin] += 1;
            for c in 0..3 {
                self.sums[3 * bin + c] += texel[c] * kept;
            }
            let power = scaled.iter().sum::<f32>() as f64 * solid_angle;
            self.total += power;
            if over > 1.0 {
                let gone = (1.0 - kept) as f64;
                // As `from_latlong` in `shaders/view.slang`.
                let longitude = (0.5 - (x as f64 + 0.5) / w as f64) * TAU;
                let toward = [
                    latitude.cos() * longitude.sin(),
                    latitude.sin(),
                    latitude.cos() * longitude.cos(),
                ];
                for c in 0..3 {
                    self.light[c] += texel[c] as f64 * gone * solid_angle;
                    self.toward[c] += toward[c] * power * gone;
                }
                self.lifted += power * gone;
            }
        }
    }

    /// The texels left and the sun, or `None` when too little was lifted to make a sun.
    fn finish(self) -> Option<(Vec<f32>, Lifted)> {
        let length = self.toward.iter().map(|c| c * c).sum::<f64>().sqrt();
        if self.lifted < SHARE * self.total || length == 0.0 {
            return None;
        }
        let texels = (self.sums.as_chunks::<3>().0.iter().zip(&self.counts))
            .flat_map(|(sum, &count)| [0, 1, 2].map(|c| sum[c] / count.max(1) as f32))
            .collect();
        let sun = Lifted {
            toward: self.toward.map(|c| (c / length) as f32),
            light: self.light.map(|c| c as f32),
        };
        Some((texels, sun))
    }
}

fn sky_buffer(device: &gpu::Device, size: u64) -> Result<Buffer, String> {
    Buffer::host(
        device,
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
}

const IDENTITY: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

type Lit<'a> = (u64, &'a store::Light, &'a store::LightParams, &'a [f64; 16]);

/// The dome that lights the scene, which is the first shown with an image, or else the first
/// shown, as Storm chooses. Also returns the paths of the other domes shown.
fn choose(store: &Store) -> (Option<Lit<'_>>, Vec<String>) {
    let shown: Vec<Lit> = (store.lights().iter())
        .filter_map(|(&id, l)| match (&l.params, &l.transform) {
            (Some(p), Some(m)) if p.shape == Shape::Dome && p.visible => Some((id, l, p, m)),
            _ => None,
        })
        .collect();
    let imaged = |l: &store::Light| l.texture.as_ref().is_some_and(|t| !t.file.is_empty());
    let lit = shown.iter().position(|(_, l, ..)| imaged(l)).unwrap_or(0);
    let rest = (shown.iter().enumerate())
        .filter(|(i, _)| *i != lit)
        .map(|(_, (_, l, ..))| l.path.clone())
        .collect();
    (shown.get(lit).copied(), rest)
}

/// World to a dome's axes: where its transform sends +x, +y and +z, each made unit length, so a
/// scaled dome turns the same way.
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
        (lit.map(|(_, l, ..)| l.path.as_str()), rest)
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
        // Negative control: the first dome shown has no image, so it isn't the one chosen.
        let rest = vec!["/debug".to_string(), "/other".to_string()];
        assert_eq!(chosen(&store), (Some("/sky"), rest));
    }

    #[test]
    fn a_sky_is_averaged_down_within_the_summed_width() {
        assert_eq!(summed_extent([2048, 1024]), [SUMMED, SUMMED / 2]);
        assert_eq!(summed_extent([64, 32]), [64, 32]);
        // Two texels a bin, grey 1 and 3 on the left and 5 and 7 on the right.
        let texels: Vec<f32> = [1.0, 3.0, 5.0, 7.0]
            .into_iter()
            .flat_map(|v| [v; 3])
            .collect();
        assert_eq!(
            average([4, 1], &texels, [2, 1]),
            [2.0, 2.0, 2.0, 6.0, 6.0, 6.0]
        );
        // Negative control: a bin takes nothing from its neighbour.
        assert_ne!(average([4, 1], &texels, [2, 1])[0], 4.0);
    }

    #[test]
    fn the_blurred_levels_span_mirror_to_the_roughest() {
        assert_eq!(roughness(0), 0.0);
        assert!((roughness(LEVELS - 1) - ROUGHEST).abs() < 1e-6);
        assert!((1..LEVELS).all(|l| roughness(l) > roughness(l - 1)));
        // Negative control: a level past the last is rougher than any that is read.
        assert!(roughness(LEVELS) > ROUGHEST);
        assert_eq!(level(2), REFLECTION.pow(2) + (REFLECTION / 2).pow(2));
        assert_eq!(side(LEVELS - 1), LEAST);
        // Negative control: a plain halving goes on below the smallest blurred side.
        assert!(halving(LEVELS - 1) < LEAST);
    }

    #[test]
    fn a_sun_is_lifted_out_of_a_sky() {
        // A grey sky of 1 with one texel of 1001, an eighth of a turn up and round from +z.
        let lifted = |scale: f32| {
            let mut lift = Lift::new([scale; 3], [8, 4], [4, 2]);
            for y in 0..4 {
                let mut row = [1.0; 24];
                if y == 1 {
                    row[9..12].fill(1001.0);
                }
                lift.row(y, &row);
            }
            lift.finish()
        };
        // With a threshold of 10, the texel keeps 10 and gives 991 over its solid angle.
        let (texels, sun) = lifted(0.1).unwrap();
        let solid_angle = (PI / 4.0).cos() * TAU / 8.0;
        let close = |a: f32, b: f64| (a as f64 - b).abs() < 1e-4 * b.abs().max(1.0);
        assert!(sun.light.iter().all(|&l| close(l, 991.0 * solid_angle)));
        let toward = [(PI / 8.0).cos() * (PI / 8.0).sin(), (PI / 8.0).sin()];
        assert!(close(sun.toward[0], toward[0]) && close(sun.toward[1], toward[1]));
        // Its bin averages 10 with three 1s, and the others are untouched.
        assert_eq!(texels.len(), 4 * 2 * 3);
        assert!(close(texels[3], 13.0 / 4.0) && close(texels[0], 1.0));
        // Negative control: nothing is over a threshold of 2000, so there is no sun.
        assert!(lifted(0.0005).is_none());
        // Under a hundredth of the sky's light is no sun either, here 1 over 4π.
        let mut faint = Lift::new([1.0; 3], [8, 4], [4, 2]);
        for y in 0..4 {
            let mut row = [1.0; 24];
            row[9] = if y == 1 { 1.1 } else { 1.0 };
            faint.row(y, &row);
        }
        assert!(faint.finish().is_none());
    }

    #[test]
    fn axes_take_a_world_direction_to_the_domes() {
        // A quarter turn about y, scaled by 2, so the dome's +x lies along world -z.
        let mut m = IDENTITY;
        (m[0], m[2]) = (0.0, -2.0);
        (m[8], m[10]) = (2.0, 0.0);
        let a = axes(&m);
        let along = |d: [f32; 3]| a.map(|row| row[0] * d[0] + row[1] * d[1] + row[2] * d[2]);
        assert_eq!(along([0.0, 0.0, -1.0]), [1.0, 0.0, 0.0]);
        assert_eq!(along([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0]);
        // Negative control: unturned, world -z isn't the dome's +x.
        assert_ne!(axes(&IDENTITY).map(|row| -row[2]), [1.0, 0.0, 0.0]);
    }
}
