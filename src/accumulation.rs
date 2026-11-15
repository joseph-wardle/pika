//! Still accumulation: while nothing changes, each frame draws the next sample, its pixels'
//! positions jittered (`jitter`) and its soft shadows' pattern turned, and shows the filtered
//! mean of the frames so far, until `SAMPLES`. As EEVEE's film does, each pixel gathers its 3×3
//! neighbours' samples, each weighted by the filter at its distance from the pixel's centre.

use ash::vk;

use crate::gpu::{self, Buffer, Memory};
use crate::target::Target;

/// Samples a still view converges at, as EEVEE renders take: shading is linear in each
/// light's visibility, so the mean of 64 frames is each penumbra's mean of 64 rays.
pub const SAMPLES: u32 = 64;

/// The pixel filter's radius in pixels, EEVEE's default Filter Size. Past it a sample weighs
/// nothing, so with jitter within half a pixel only a pixel's 3×3 neighbours reach it.
pub const FILTER_RADIUS: f64 = 1.5;

pub struct Accumulation {
    compute: gpu::Compute,
    /// Allocated by the first still frame.
    sum: Option<Sum>,
    /// Frames the sum holds, drawn while nothing changed; the next frame's sample.
    pub drawn: u32,
    /// The weights of all the samples the sum holds, which every pixel shares.
    total: f32,
}

struct Sum {
    /// Per pixel, four floats: the weighted colors of each frame since the view was last still.
    colors: Buffer,
    /// The color attachment's pixels, copied out for the sum.
    frame: Buffer,
    /// The mean, as the attachment's pixels, copied back in.
    mean: Buffer,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    sum: vk::DeviceAddress,
    frame: vk::DeviceAddress,
    mean: vk::DeviceAddress,
    weights: [f32; 9],
    total: f32,
    extent: [u32; 2],
    first: u32,
    count: u32,
    stride: u32,
    sample: u32,
    words: u32,
    unused: u32,
}

/// Where sample `n` falls in its pixel, in pixels from its centre along the attachment's x and
/// y: Halton's bases 2 and 3, shifted half a pixel so that sample 0, from which a still view's
/// depth and normal come, is the centre. Each base stratifies its axis whatever the shift.
pub fn jitter(n: u32) -> [f64; 2] {
    [2, 3].map(|base| (radical_inverse(base, n) + 0.5).fract() - 0.5)
}

fn radical_inverse(base: u32, mut n: u32) -> f64 {
    let (mut value, mut scale) = (0.0, 1.0);
    while n > 0 {
        scale /= f64::from(base);
        value += f64::from(n % base) * scale;
        n /= base;
    }
    value
}

/// The weight of a sample `d` pixels from a pixel's centre, for a filter of `radius` pixels:
/// EEVEE's Gaussian fitted to Blackman-Harris, its standard deviation 0.284 of the radius.
pub fn weight(radius: f64, d: [f64; 2]) -> f64 {
    let r = (d[0] * d[0] + d[1] * d[1]) / (radius * radius);
    if r >= 1.0 {
        return 0.0;
    }
    const SIGMA: f64 = 0.284;
    (-0.5 * r / (SIGMA * SIGMA)).exp()
}

/// The weights a pixel gives its 3×3 neighbours' samples at `jitter`, row by row from the
/// neighbour at (-1, -1), as `shaders/accumulation.slang` reads them.
pub fn weights(jitter: [f64; 2]) -> [f32; 9] {
    std::array::from_fn(|i| {
        let offset = [(i % 3) as f64 - 1.0, (i / 3) as f64 - 1.0];
        weight(
            FILTER_RADIUS,
            [offset[0] + jitter[0], offset[1] + jitter[1]],
        ) as f32
    })
}

impl Accumulation {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
        let compute = gpu::Compute::new(device, spirv, c"cs", size_of::<Push>())?;
        Ok(Self {
            compute,
            sum: None,
            drawn: 0,
            total: 0.0,
        })
    }

    pub fn converged(&self) -> bool {
        self.drawn >= SAMPLES
    }

    /// Records outside rendering, after a still frame's shading, drawn at `jitter(self.drawn)`:
    /// adds the color attachment's filtered samples to the sum, which sample 0 starts, and
    /// leaves the mean in the attachment.
    pub fn record(
        &mut self,
        device: &gpu::Device,
        cb: vk::CommandBuffer,
        target: &Target,
    ) -> Result<(), String> {
        let pixels = u64::from(target.width * target.height);
        let bytes = pixels * target.color_bytes;
        let sum = match &mut self.sum {
            Some(s) if s.frame.size == bytes && s.colors.size == pixels * 16 => s,
            slot => {
                // Free the old sum before allocating its replacement.
                *slot = None;
                let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
                slot.insert(Sum {
                    colors: Buffer::device_local(device, pixels * 16, storage)?,
                    frame: Buffer::device_local(
                        device,
                        bytes,
                        storage | vk::BufferUsageFlags::TRANSFER_DST,
                    )?,
                    mean: Buffer::device_local(
                        device,
                        bytes,
                        storage | vk::BufferUsageFlags::TRANSFER_SRC,
                    )?,
                })
            }
        };
        let sample = self.drawn;
        let weights = weights(jitter(sample));
        let earlier = if sample == 0 { 0.0 } else { self.total };
        self.total = earlier + weights.iter().sum::<f32>();
        target.copy_out(cb, 0, sum.frame.raw);
        let words = (target.color_bytes / 4) as u32;
        let all = 0..target.width * target.height;
        (self.compute).record(cb, &[all], |first, count, stride| Push {
            sum: sum.colors.address,
            frame: sum.frame.address,
            mean: sum.mean.address,
            weights,
            total: self.total,
            extent: [target.width, target.height],
            first,
            count,
            stride,
            sample,
            words,
            unused: 0,
        });
        target.copy_in(cb, 0, sum.mean.raw);
        self.drawn += 1;
        Ok(())
    }

    pub fn memory(&self) -> Memory {
        let mut memory = Memory::default();
        if let Some(s) = &self.sum {
            memory += s.colors.memory();
            memory += s.frame.memory();
            memory += s.mean.memory();
        }
        memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_0_is_the_centre_and_the_rest_stay_in_the_pixel() {
        assert_eq!(jitter(0), [0.0, 0.0]);
        for n in 0..SAMPLES {
            assert!(jitter(n).iter().all(|v| (-0.5..0.5).contains(v)), "{n}");
        }
    }

    /// The samples cover the pixel evenly: each of a 4×4 grid of its cells holds 4 ± 1. The
    /// control, one base for both axes, lays them along the diagonal.
    #[test]
    fn samples_cover_the_pixel_evenly() {
        let even = |points: &[[f64; 2]]| {
            let mut cells = [0usize; 16];
            for p in points {
                let [x, y] = p.map(|v| ((v + 0.5) * 4.0) as usize);
                cells[y * 4 + x] += 1;
            }
            cells.iter().all(|&c| c.abs_diff(points.len() / 16) <= 1)
        };
        let samples: Vec<_> = (0..SAMPLES).map(jitter).collect();
        assert!(even(&samples));
        let diagonal: Vec<_> = (0..SAMPLES)
            .map(|n| [2, 2].map(|base| (radical_inverse(base, n) + 0.5).fract() - 0.5))
            .collect();
        assert!(!even(&diagonal));
    }

    /// Weight a 5×5 gather finds past the 3×3 one, at `jitter`.
    fn past_3x3(radius: f64, jitter: [f64; 2]) -> f64 {
        let mut past = 0.0;
        for y in -2i32..=2 {
            for x in -2i32..=2 {
                if x.abs() == 2 || y.abs() == 2 {
                    let d = [f64::from(x) + jitter[0], f64::from(y) + jitter[1]];
                    past += weight(radius, d);
                }
            }
        }
        past
    }

    /// The 3×3 gather holds the whole filter for every sample, so its weights are all of each
    /// frame's; the control, a filter of 2 pixels, reaches past it.
    #[test]
    fn the_filter_fits_the_3x3_gather() {
        for n in 0..SAMPLES {
            assert_eq!(past_3x3(FILTER_RADIUS, jitter(n)), 0.0, "{n}");
        }
        assert!((0..SAMPLES).any(|n| past_3x3(2.0, jitter(n)) > 0.0));
    }

    /// At no jitter, the gather's weights are the filter's at each neighbour's distance, the
    /// centre's 1, symmetric, and their sum the Gaussian's closed form there.
    #[test]
    fn centred_weights_are_the_filters() {
        let w = weights([0.0, 0.0]);
        let edge = (-0.5 / (FILTER_RADIUS * 0.284).powi(2)).exp();
        let corner = (-1.0 / (FILTER_RADIUS * 0.284).powi(2)).exp();
        let expected = 1.0 + 4.0 * edge + 4.0 * corner;
        assert!((w.iter().sum::<f32>() as f64 - expected).abs() < 1e-6);
        assert_eq!(w[4], 1.0);
        assert_eq!([w[0], w[2], w[6]], [w[8]; 3]);
        assert_eq!([w[1], w[3], w[5]], [w[7]; 3]);
        // Samples half a pixel right: the left neighbour's lands half a pixel from the centre,
        // the right one's on the radius, which it does not reach.
        let shifted = weights([0.5, 0.0]);
        assert_eq!(shifted[4], shifted[3]);
        assert_eq!(shifted[5], 0.0);
    }
}
