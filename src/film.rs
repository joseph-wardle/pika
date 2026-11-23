//! The film turns each frame's samples into the image shown, as EEVEE's film does.
//!
//! A still view adds one jittered sample a frame, up to `SAMPLES`, and shows their filtered
//! mean. While anything changes, each frame is blended into the reprojected image before,
//! clipped to the colours around each pixel so moving edges don't ghost.

use ash::vk;

use crate::gpu::{self, Buffer, Memory};
use crate::raster;
use crate::target::Target;
use crate::transform::Camera;

/// The samples a still view converges at, as in EEVEE renders. Shading is linear in each
/// light's visibility, so the mean of 64 frames is each penumbra's mean of 64 rays.
pub const SAMPLES: u32 = 64;

/// The pixel filter's radius in pixels, EEVEE's default Filter Size. A sample past it weighs
/// nothing, so with jitter inside half a pixel only a pixel's 3×3 neighbours reach it.
pub const FILTER_RADIUS: f64 = 1.5;

/// The jitter positions cycled while anything changes, as EEVEE cycles them. The history blend
/// keeps about twenty frames, so each pixel's filter sees all eight positions.
const PLAYING_SAMPLES: u32 = 8;

/// Shadow patterns cycled while anything changes. There are more than the history blend keeps,
/// so each pixel's history averages a penumbra's rays.
const PLAYING_PATTERNS: u32 = 32;

/// How many still samples the image the view stopped at counts as at first.
const PRIOR: f64 = 4.0;

pub struct Film {
    accumulate: gpu::Compute,
    resolve: gpu::Compute,
    /// Allocated by the first frame, and again when the target's size or format changes.
    buffers: Option<Buffers>,
    still: bool,
    /// The still samples the sum holds, which is also the next still frame's sample.
    drawn: u32,
    /// The total weight of the still samples in the sum, which every pixel shares.
    total: f32,
    /// Frames drawn while anything changed, which cycle the jitter.
    played: u32,
    /// The unjittered camera of the frame in `images[shown]`, or `None` if there was no camera.
    previous_view_projection: Option<[[f32; 4]; 4]>,
    /// The image last shown. It is the history the next moving frame reads.
    shown: usize,
    /// The image the last moving frame left behind, which still frames blend in.
    stopped: usize,
}

struct Buffers {
    /// Four floats per pixel: the weighted colour of every still sample, summed.
    sum: Buffer,
    frame: Buffer,
    depth: Buffer,
    motion: Buffer,
    /// The shown images, stored as the colour attachment's pixels. Each frame one is copied back
    /// in and the next frame reads it. A still frame keeps `stopped`'s image and writes the other.
    images: [Buffer; 2],
}

impl Buffers {
    fn push(&self, target: &Target, read: usize, write: usize, weights: [f32; 9]) -> Push {
        Push {
            sum: self.sum.address,
            frame: self.frame.address,
            depth: self.depth.address,
            motion: self.motion.address,
            history: self.images[read].address,
            image: self.images[write].address,
            weights,
            total: 0.0,
            extent: [target.width, target.height],
            prior: 0.0,
            first: 0,
            count: 0,
            stride: 0,
            sample: 0,
            words: (target.color_bytes / 4) as u32,
        }
    }
}

/// Must match `Push` in `shaders/film.slang`, which both of its passes use.
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

/// Where sample `n` falls in its pixel, from Halton bases 2 and 3. They are shifted so sample 0
/// lands on the centre, since a still view's depth and normal come from it.
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

/// The weight of a sample `d` pixels from a pixel's centre. It is EEVEE's Gaussian fitted to
/// Blackman-Harris, with a standard deviation of 0.284 times the radius.
pub fn weight(radius: f64, d: [f64; 2]) -> f64 {
    let r = (d[0] * d[0] + d[1] * d[1]) / (radius * radius);
    if r >= 1.0 {
        return 0.0;
    }
    const SIGMA: f64 = 0.284;
    (-0.5 * r / (SIGMA * SIGMA)).exp()
}

/// The weights a pixel gives its 3×3 neighbours' samples at `jitter`, row by row from the
/// neighbour at (-1, -1), in the order `shaders/film.slang` reads them.
pub fn weights(jitter: [f64; 2]) -> [f32; 9] {
    std::array::from_fn(|i| {
        let offset = [(i % 3) as f64 - 1.0, (i / 3) as f64 - 1.0];
        weight(
            FILTER_RADIUS,
            [offset[0] + jitter[0], offset[1] + jitter[1]],
        ) as f32
    })
}

/// The weight the image the view stopped at keeps against the mean of `n` still samples. It
/// fades to nothing by the last sample, so a converged still is the same whatever came before.
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
            previous_view_projection: None,
            shown: 0,
            stopped: 0,
        })
    }

    /// Starts a frame, which is `still` if nothing changed since the last one drawn. Returns its
    /// still sample, or 0 while anything changes.
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

    pub fn jitter(&self) -> [f64; 2] {
        jitter(self.sample())
    }

    /// The frame's index into the shadow pattern sequence (see `shaders/frame.slang`).
    pub fn pattern(&self) -> u32 {
        match self.still {
            true => self.drawn,
            false => self.played % PLAYING_PATTERNS,
        }
    }

    fn sample(&self) -> u32 {
        match self.still {
            true => self.drawn,
            false => self.played % PLAYING_SAMPLES,
        }
    }

    /// The unjittered camera of the image the frame's history reads, or `None` if there was none.
    pub fn previous_view_projection(&self) -> Option<[[f32; 4]; 4]> {
        self.previous_view_projection
    }

    /// Records the film after the frame's shading, outside a render pass. Afterwards the colour
    /// attachment holds the image the frame shows.
    pub fn record(
        &mut self,
        device: &gpu::Device,
        cb: vk::CommandBuffer,
        target: &Target,
        camera: &Camera,
    ) -> Result<(), String> {
        self.allocate(device, target)?;
        let buffers = self.buffers.as_ref().expect("allocated above");
        target.copy_out(cb, 0, buffers.frame.raw);
        let written = match self.still {
            true => self.record_still(cb, target),
            false => self.record_playing(cb, target),
        };
        let buffers = self.buffers.as_ref().expect("allocated above");
        target.copy_in(cb, 0, buffers.images[written].raw);
        self.shown = written;
        self.previous_view_projection = Some(camera.view_projection);
        Ok(())
    }

    fn allocate(&mut self, device: &gpu::Device, target: &Target) -> Result<(), String> {
        let pixels = u64::from(target.width * target.height);
        let bytes = pixels * target.color_bytes;
        if (self.buffers.as_ref())
            .is_some_and(|b| b.frame.size == bytes && b.sum.size == pixels * 16)
        {
            return Ok(());
        }
        // Free the old buffers before allocating the new ones. Their history goes with them.
        self.buffers = None;
        self.previous_view_projection = None;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let copied = storage | vk::BufferUsageFlags::TRANSFER_DST;
        let shown = storage | vk::BufferUsageFlags::TRANSFER_SRC;
        self.buffers = Some(Buffers {
            sum: Buffer::device_local(device, pixels * 16, storage)?,
            frame: Buffer::device_local(device, bytes, copied)?,
            depth: Buffer::device_local(device, pixels * 4, copied)?,
            motion: Buffer::device_local(device, pixels * 4, copied)?,
            images: [
                Buffer::device_local(device, bytes, shown)?,
                Buffer::device_local(device, bytes, shown)?,
            ],
        });
        Ok(())
    }

    /// Adds the frame to the sum of still samples and writes their mean, blended with the image the
    /// view stopped at by a fading weight. Returns the image written.
    fn record_still(&mut self, cb: vk::CommandBuffer, target: &Target) -> usize {
        let buffers = self.buffers.as_ref().expect("allocated by record");
        let sample = self.drawn;
        let weights = weights(jitter(sample));
        let earlier = if sample == 0 { 0.0 } else { self.total };
        self.total = earlier + weights.iter().sum::<f32>();
        let prior = match self.previous_view_projection {
            Some(_) => prior(sample + 1) as f32,
            None => 0.0,
        };
        let (read, write) = (self.stopped, 1 - self.stopped);
        let push = Push {
            total: self.total,
            prior,
            sample,
            ..buffers.push(target, read, write, weights)
        };
        let all = 0..target.width * target.height;
        (self.accumulate).record(cb, &[all], |first, count, stride| Push {
            first,
            count,
            stride,
            ..push
        });
        self.drawn += 1;
        write
    }

    /// Blends the frame into the image shown last, moved to where each pixel was. Returns the image
    /// written, which is also the one a still view stops at.
    fn record_playing(&mut self, cb: vk::CommandBuffer, target: &Target) -> usize {
        let buffers = self.buffers.as_ref().expect("allocated by record");
        target.copy_out(cb, 1, buffers.depth.raw);
        target.copy_out(cb, raster::MOTION_ATTACHMENT, buffers.motion.raw);
        let sample = self.sample();
        let weights = weights(jitter(sample));
        let (read, write) = (self.shown, 1 - self.shown);
        let push = Push {
            total: weights.iter().sum(),
            // The resolve reads this as whether there is any history.
            prior: if self.previous_view_projection.is_some() {
                1.0
            } else {
                0.0
            },
            sample,
            ..buffers.push(target, read, write, weights)
        };
        let all = 0..target.width * target.height;
        (self.resolve).record(cb, &[all], |first, count, stride| Push {
            first,
            count,
            stride,
            ..push
        });
        self.stopped = write;
        self.played = self.played.wrapping_add(1);
        write
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

    /// The prior fades with each still sample and is gone by the last, so a converged still is the
    /// same whatever image the view stopped at. The control, a prior that doesn't fade, still counts
    /// at the last sample.
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

    /// The samples cover the pixel evenly, with 4 ± 1 in each cell of a 4×4 grid. The control, one
    /// base for both axes, lines them up along the diagonal.
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

    /// The weight a 5×5 gather finds beyond the 3×3 one, at `jitter`.
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
    /// frame's weight. The control, a filter of 2 pixels, reaches past it.
    #[test]
    fn the_filter_fits_the_3x3_gather() {
        for n in 0..SAMPLES {
            assert_eq!(past_3x3(FILTER_RADIUS, jitter(n)), 0.0, "{n}");
        }
        assert!((0..SAMPLES).any(|n| past_3x3(2.0, jitter(n)) > 0.0));
    }

    /// With no jitter, the gather's weights are the filter's at each neighbour's distance. The
    /// centre's is 1, they are symmetric, and they sum to the Gaussian's closed form there.
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
        // Samples half a pixel to the right. The left neighbour's sample lands half a pixel from the
        // centre, and the right one's lands on the radius, which it doesn't reach.
        let shifted = weights([0.5, 0.0]);
        assert_eq!(shifted[4], shifted[3]);
        assert_eq!(shifted[5], 0.0);
    }
}
