//! The film: what each frame's samples become, as EEVEE's film does.
//!
//! While nothing changes, each frame draws the next still sample, its pixels' positions jittered
//! (`jitter`) and its soft shadows' pattern turned, and shows the filtered mean of the still
//! samples so far until `SAMPLES`, blended with the image the view stopped at so that stopping
//! does not jump to one sample's noise. While anything changes, each frame's filtered samples are
//! blended into the image before, reprojected by the prepass's motion and clipped to the colors
//! around each pixel, which is what keeps moving edges from ghosting. Either way each pixel
//! gathers its 3×3 neighbours' samples, each weighted by the filter at its distance from the
//! pixel's centre.

use ash::vk;

use crate::gpu::{self, Buffer, Memory};
use crate::raster;
use crate::target::Target;
use crate::transform::Camera;

/// Samples a still view converges at, as EEVEE renders take: shading is linear in each
/// light's visibility, so the mean of 64 frames is each penumbra's mean of 64 rays.
pub const SAMPLES: u32 = 64;

/// The pixel filter's radius in pixels, EEVEE's default Filter Size. Past it a sample weighs
/// nothing, so with jitter within half a pixel only a pixel's 3×3 neighbours reach it.
pub const FILTER_RADIUS: f64 = 1.5;

/// The jitter's samples while anything changes, cycled as EEVEE cycles them: history's blend
/// keeps about twenty frames, so each pixel's filter sees all eight positions.
const PLAYING_SAMPLES: u32 = 8;

/// The shadow pattern's samples while anything changes, cycled as EEVEE cycles its ray-traced
/// ones: more than history's blend keeps, so each pixel's history averages a penumbra's rays.
const PLAYING_PATTERNS: u32 = 32;

/// How many still samples the image the view stopped at first counts as. Stopping ALab's
/// trailer camera chose it: at 1 the first still frame jumps nearly twice as far from the image
/// stopped at for little less error, and at 16 the still keeps that image's error half again
/// as long.
const PRIOR: f64 = 4.0;

pub struct Film {
    accumulate: gpu::Compute,
    resolve: gpu::Compute,
    /// Allocated by the first frame, again when the target's size or format changes.
    buffers: Option<Buffers>,
    still: bool,
    /// Still samples the sum holds; the next still frame's sample.
    drawn: u32,
    /// The weights of all the still samples the sum holds, which every pixel shares.
    total: f32,
    /// Frames drawn while anything changed, which cycle the jitter.
    played: u32,
    /// The camera, unjittered, of the frame `images[shown]` holds; none without one.
    previous: Option<[[f32; 4]; 4]>,
    /// The image last shown, the history the next frame drawn while anything changes reads.
    shown: usize,
    /// The image the last frame drawn while anything changed left, which still frames blend in.
    stopped: usize,
}

struct Buffers {
    /// Per pixel, four floats: the weighted colors of each still sample.
    sum: Buffer,
    /// The color, depth AOV and motion attachments' pixels, copied out for the passes.
    frame: Buffer,
    depth: Buffer,
    motion: Buffer,
    /// Shown images, as the color attachment's pixels: one is copied back in each frame, and
    /// the next reads it. A still frame keeps `stopped`'s and writes the other.
    images: [Buffer; 2],
}

/// `shaders/film.slang`'s, for both of its passes.
#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    sum: vk::DeviceAddress,
    frame: vk::DeviceAddress,
    depth: vk::DeviceAddress,
    motion: vk::DeviceAddress,
    history: vk::DeviceAddress,
    image: vk::DeviceAddress,
    weights: [f32; 9],
    total: f32,
    extent: [u32; 2],
    prior: f32,
    first: u32,
    count: u32,
    stride: u32,
    sample: u32,
    words: u32,
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
/// neighbour at (-1, -1), as `shaders/film.slang` reads them.
pub fn weights(jitter: [f64; 2]) -> [f32; 9] {
    std::array::from_fn(|i| {
        let offset = [(i % 3) as f64 - 1.0, (i / 3) as f64 - 1.0];
        weight(
            FILTER_RADIUS,
            [offset[0] + jitter[0], offset[1] + jitter[1]],
        ) as f32
    })
}

/// The weight the image the view stopped at keeps against the mean of `n` still samples: it
/// fades to none by the last, so a converged still is the same whatever came before it.
fn prior(n: u32) -> f64 {
    PRIOR * (1.0 - f64::from(n) / f64::from(SAMPLES)).max(0.0) / f64::from(n)
}

impl Film {
    pub fn new(device: &gpu::Device, spirv: &[u32]) -> Result<Self, String> {
        let push = size_of::<Push>();
        Ok(Self {
            accumulate: gpu::Compute::new(device, spirv, c"accumulate", push)?,
            resolve: gpu::Compute::new(device, spirv, c"resolve", push)?,
            buffers: None,
            still: false,
            drawn: 0,
            total: 0.0,
            played: 0,
            previous: None,
            shown: 0,
            stopped: 0,
        })
    }

    /// Starts a frame, `still` if nothing changed since the last one drawn, and returns its still
    /// sample, 0 while anything changes.
    pub fn begin(&mut self, still: bool) -> u32 {
        self.still = still;
        if !still {
            self.drawn = 0;
        }
        self.drawn
    }

    pub fn converged(&self) -> bool {
        self.drawn >= SAMPLES
    }

    /// The frame's jitter, as `begin` left it.
    pub fn jitter(&self) -> [f64; 2] {
        jitter(self.sample())
    }

    /// The frame's index into the shadow pattern's sequence (`shaders/frame.slang`).
    pub fn pattern(&self) -> u32 {
        match self.still {
            true => self.drawn,
            false => self.played % PLAYING_PATTERNS,
        }
    }

    /// The frame's index into the jitter's sequence.
    fn sample(&self) -> u32 {
        match self.still {
            true => self.drawn,
            false => self.played % PLAYING_SAMPLES,
        }
    }

    /// The camera, unjittered, of the image the frame's history reads; none without one.
    pub fn previous(&self) -> Option<[[f32; 4]; 4]> {
        self.previous
    }

    /// Records outside rendering, after the frame's shading, drawn by `camera` (unjittered)
    /// jittered by `jitter()`: leaves in the color attachment the image the frame shows.
    pub fn record(
        &mut self,
        device: &gpu::Device,
        cb: vk::CommandBuffer,
        target: &Target,
        camera: &Camera,
    ) -> Result<(), String> {
        let sample = self.sample();
        let pixels = u64::from(target.width * target.height);
        let bytes = pixels * target.color_bytes;
        let buffers = match &mut self.buffers {
            Some(b) if b.frame.size == bytes && b.sum.size == pixels * 16 => b,
            slot => {
                // Free the old buffers before allocating their replacements. Their history is
                // gone with them.
                *slot = None;
                self.previous = None;
                let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
                let copied = storage | vk::BufferUsageFlags::TRANSFER_DST;
                let shown = storage | vk::BufferUsageFlags::TRANSFER_SRC;
                slot.insert(Buffers {
                    sum: Buffer::device_local(device, pixels * 16, storage)?,
                    frame: Buffer::device_local(device, bytes, copied)?,
                    depth: Buffer::device_local(device, pixels * 4, copied)?,
                    motion: Buffer::device_local(device, pixels * 4, copied)?,
                    images: [
                        Buffer::device_local(device, bytes, shown)?,
                        Buffer::device_local(device, bytes, shown)?,
                    ],
                })
            }
        };
        let weights = weights(jitter(sample));
        let frame_total = weights.iter().sum::<f32>();
        let history = self.previous.is_some();
        target.copy_out(cb, 0, buffers.frame.raw);
        let (pass, total, prior, read, write) = if self.still {
            let earlier = if sample == 0 { 0.0 } else { self.total };
            self.total = earlier + frame_total;
            let prior = if history { prior(sample + 1) } else { 0.0 };
            let read = self.stopped;
            (&self.accumulate, self.total, prior as f32, read, 1 - read)
        } else {
            target.copy_out(cb, 1, buffers.depth.raw);
            target.copy_out(cb, raster::MOTION_ATTACHMENT, buffers.motion.raw);
            let read = self.shown;
            let prior = if history { 1.0 } else { 0.0 };
            (&self.resolve, frame_total, prior, read, 1 - read)
        };
        let words = (target.color_bytes / 4) as u32;
        let all = 0..target.width * target.height;
        pass.record(cb, &[all], |first, count, stride| Push {
            sum: buffers.sum.address,
            frame: buffers.frame.address,
            depth: buffers.depth.address,
            motion: buffers.motion.address,
            history: buffers.images[read].address,
            image: buffers.images[write].address,
            weights,
            total,
            extent: [target.width, target.height],
            prior,
            first,
            count,
            stride,
            sample,
            words,
        });
        target.copy_in(cb, 0, buffers.images[write].raw);
        self.shown = write;
        match self.still {
            true => self.drawn += 1,
            false => {
                self.stopped = write;
                self.played = self.played.wrapping_add(1);
            }
        }
        self.previous = Some(camera.view_projection);
        Ok(())
    }

    pub fn memory(&self) -> Memory {
        let mut memory = Memory::default();
        if let Some(b) = &self.buffers {
            for buffer in [
                &b.sum,
                &b.frame,
                &b.depth,
                &b.motion,
                &b.images[0],
                &b.images[1],
            ] {
                memory += buffer.memory();
            }
        }
        memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The prior fades with each still sample to none at the last, so a converged still is the
    /// same whatever image the view stopped at; the control, a prior that does not fade, still
    /// counts at the last.
    #[test]
    fn the_prior_is_gone_by_the_last_sample() {
        let fades = |prior: &dyn Fn(u32) -> f64| {
            (1..SAMPLES).all(|n| prior(n + 1) < prior(n)) && prior(SAMPLES) == 0.0
        };
        assert!(prior(1) > 0.0);
        assert!(fades(&prior));
        assert!(!fades(&|n| PRIOR / f64::from(n)));
    }

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
