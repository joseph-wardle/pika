//! The lights a view draws, as `shaders/frame.slang` reads them, and the
//! screen tiles each may reach. A light reaches as far as what it gives
//! falls to `THRESHOLD`, and each `TILE`-pixel tile lists the lights that
//! reach into it, so a pixel's light loop runs over its tile's lights
//! alone. The lists are built on the CPU each frame from the lights' boxes,
//! which costs little at tens of lights.

use std::ops::Range;

use crate::transform::Camera;

/// The side of a tile in pixels.
pub const TILE: u32 = 16;

/// The illuminance at which a light's reach ends. A facing white Lambert
/// surface shows illuminance / π, so a pixel loses at most this / π of
/// each light (`Light::point`).
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Pika makes no point light until it syncs Hydra's lights"
    )
)]
pub const THRESHOLD: f32 = 0.01;

/// `shaders/view.slang`'s `Light`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Light {
    /// The unit direction toward a distant light (w = 0), or a point
    /// light's position (w = 1).
    pub position: [f32; 4],
    /// The illuminance it gives a surface facing it: for a point light, at
    /// unit distance.
    pub illuminance: [f32; 3],
    /// How far it reaches: infinite for a distant light.
    pub radius: f32,
}

impl Light {
    pub fn distant(direction: [f32; 3], illuminance: [f32; 3]) -> Self {
        let [x, y, z] = direction;
        Self {
            position: [x, y, z, 0.0],
            illuminance,
            radius: f32::INFINITY,
        }
    }

    /// A point light at `position`. Its reach is where its brightest
    /// channel falls to `THRESHOLD`: the light loop gives illuminance ×
    /// (1/d² - 1/radius²), which loses that much everywhere and so reaches
    /// 0 at the radius without a step.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Pika makes no point light until it syncs Hydra's lights"
        )
    )]
    pub fn point(position: [f32; 3], illuminance: [f32; 3]) -> Self {
        let [x, y, z] = position;
        let brightest = illuminance.into_iter().fold(0.0, f32::max);
        Self {
            position: [x, y, z, 1.0],
            illuminance,
            radius: (brightest / THRESHOLD).sqrt(),
        }
    }
}

/// Each tile's lights, in light order, so a tile sums them as a loop over
/// every light would. Tiles run row by row from pixel (0, 0), where clip
/// (-1, -1) lands, as Vulkan's viewport puts it.
pub struct Tiles {
    /// Per tile: the first of its lights in `lights`, and how many.
    pub tiles: Vec<[u32; 2]>,
    /// Indices into the view's lights.
    pub lights: Vec<u32>,
}

impl Tiles {
    /// The tiles of a target of `extent` pixels that each light may reach.
    pub fn new(lights: &[Light], camera: &Camera, extent: [u32; 2]) -> Self {
        let [columns, rows] = grid(extent);
        let reach: Vec<_> = (lights.iter())
            .map(|l| reach(l, &camera.view_projection, extent))
            .collect();
        let mut counts = vec![0u32; (columns * rows) as usize];
        for [x, y] in reach.iter().flatten() {
            for row in y.clone() {
                let start = (row * columns) as usize;
                for c in &mut counts[start + x.start as usize..start + x.end as usize] {
                    *c += 1;
                }
            }
        }
        // Each tile's first light, and how many are placed so far.
        let mut first = 0;
        let mut tiles: Vec<[u32; 2]> = (counts.iter())
            .map(|&count| {
                first += count;
                [first - count, 0]
            })
            .collect();
        let mut indices = vec![0; first as usize];
        for (i, r) in reach.iter().enumerate() {
            let Some([x, y]) = r else { continue };
            for row in y.clone() {
                for column in x.clone() {
                    let tile = &mut tiles[(row * columns + column) as usize];
                    indices[(tile[0] + tile[1]) as usize] = i as u32;
                    tile[1] += 1;
                }
            }
        }
        Self {
            tiles,
            lights: indices,
        }
    }
}

/// The columns and rows of tiles covering `extent`.
pub fn grid(extent: [u32; 2]) -> [u32; 2] {
    extent.map(|e| e.div_ceil(TILE))
}

/// The most bytes a view's lights and tiles take at `extent`: every light
/// in every tile.
pub fn capacity(lights: usize, extent: [u32; 2]) -> u64 {
    let [columns, rows] = grid(extent);
    let tiles = (columns * rows) as u64;
    (lights * size_of::<Light>()) as u64 + tiles * 8 + tiles * lights as u64 * 4
}

/// The columns and rows of the tiles `light` may reach, from the screen
/// box of its box's corners. A point inside the light's sphere is inside
/// its box, and while every corner is in front of the eye, its pixel is
/// inside the corners' screen box; a pixel of margin covers the shader's
/// rounding, which differs from this.
fn reach(
    light: &Light,
    view_projection: &[[f32; 4]; 4],
    extent: [u32; 2],
) -> Option<[Range<u32>; 2]> {
    let [columns, rows] = grid(extent);
    let every = Some([0..columns, 0..rows]);
    if light.radius == f32::INFINITY {
        return every;
    }
    if light.radius <= 0.0 {
        return None;
    }
    let mut low = [f32::INFINITY; 2];
    let mut high = [f32::NEG_INFINITY; 2];
    let mut behind = 0;
    for corner in 0..8 {
        let p: [f32; 3] = std::array::from_fn(|i| {
            let side = if corner >> i & 1 == 1 { 1.0 } else { -1.0 };
            light.position[i] + side * light.radius
        });
        let clip = project(view_projection, p);
        if clip[3] <= 0.0 {
            behind += 1;
            continue;
        }
        for i in 0..2 {
            let pixel = (clip[i] / clip[3] * 0.5 + 0.5) * extent[i] as f32;
            low[i] = low[i].min(pixel);
            high[i] = high[i].max(pixel);
        }
    }
    // Every point drawn is in front of the eye.
    match behind {
        8 => return None,
        0 => {}
        _ => return every,
    }
    let [x, y] = [0, 1].map(|i| {
        let count = [columns, rows][i];
        let tile = |pixel: f32| (pixel / TILE as f32).floor().clamp(0.0, count as f32) as u32;
        tile(low[i] - 1.0)..(tile(high[i] + 1.0) + 1).min(count)
    });
    (!x.is_empty() && !y.is_empty()).then_some([x, y])
}

/// A world-space position in clip space, as `shaders/view.slang` projects it.
fn project(view_projection: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 4] {
    view_projection.map(|row| row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + row[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera at (0, 0, 5) looking down -z, perspective or orthographic.
    fn camera(perspective: bool) -> Camera {
        #[rustfmt::skip]
        let world_to_view = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, -5.0, 1.0,
        ];
        let (n, f) = (0.1, 100.0);
        // OpenGL's, as Hydra gives them, in row-vector form.
        #[rustfmt::skip]
        let projection = match perspective {
            true => [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.5, 0.0, 0.0,
                0.0, 0.0, -(f + n) / (f - n), -1.0,
                0.0, 0.0, -2.0 * f * n / (f - n), 0.0,
            ],
            false => [
                0.25, 0.0, 0.0, 0.0,
                0.0, 0.4, 0.0, 0.0,
                0.0, 0.0, -2.0 / (f - n), 0.0,
                0.0, 0.0, -(f + n) / (f - n), 1.0,
            ],
        };
        Camera::new(&world_to_view, &projection).unwrap()
    }

    /// xorshift: the same numbers on every run.
    struct Random(u64);

    impl Random {
        fn next(&mut self, low: f32, high: f32) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            low + (high - low) * (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// The tile the shader finds a point in (`shaders/frame.slang` tile).
    fn tile(camera: &Camera, extent: [u32; 2], p: [f32; 3]) -> Option<usize> {
        let clip = project(&camera.view_projection, p);
        let [columns, rows] = grid(extent);
        let inside = (0..2).all(|i| clip[i].abs() <= clip[3]) && clip[2] >= 0.0 && clip[3] > 0.0;
        if !inside {
            return None;
        }
        let [x, y] = [0, 1].map(|i| {
            let pixel = (clip[i] / clip[3] * 0.5 + 0.5) * extent[i] as f32;
            ((pixel.max(0.0) / TILE as f32) as u32).min([columns, rows][i] - 1)
        });
        Some((y * columns + x) as usize)
    }

    /// Point lights across the view, with a distant light among them.
    fn lights(random: &mut Random) -> Vec<Light> {
        let mut lights: Vec<_> = (0..40)
            .map(|_| {
                let p = [-4.0, -4.0, -6.0].map(|v| random.next(v, -v));
                Light::point(p, [random.next(0.0, 0.02), 0.005, 0.001])
            })
            .collect();
        lights.insert(7, Light::distant([0.0, 0.0, 1.0], [1.0; 3]));
        lights
    }

    /// Over random visible points, the lights reaching a point that its
    /// tile does not list, and those reaching one, with the tiles built
    /// from the lights' radii times `shrink`; and the share of the tiles'
    /// slots, every light in every tile, that they fill.
    fn missed(perspective: bool, shrink: f32) -> (usize, usize, f32) {
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        let camera = camera(perspective);
        let extent = [333, 200];
        let lights = lights(&mut random);
        let binned: Vec<_> = (lights.iter())
            .map(|l| Light {
                radius: l.radius * shrink,
                ..*l
            })
            .collect();
        let tiles = Tiles::new(&binned, &camera, extent);
        let mut missed = 0;
        let mut reached = 0;
        let mut seen = 0;
        while seen < 20_000 {
            let p = [[-6.0, 6.0], [-6.0, 6.0], [-8.0, 4.0]].map(|[a, b]| random.next(a, b));
            let Some(t) = tile(&camera, extent, p) else {
                continue;
            };
            seen += 1;
            let [first, count] = tiles.tiles[t];
            let listed = &tiles.lights[first as usize..(first + count) as usize];
            for (i, l) in lights.iter().enumerate() {
                let d: f32 = (0..3).map(|k| (p[k] - l.position[k]).powi(2)).sum();
                let reaches = l.position[3] == 0.0 || d < l.radius * l.radius;
                reached += (reaches && l.position[3] == 1.0) as usize;
                missed += (reaches && !listed.contains(&(i as u32))) as usize;
            }
        }
        let filled = tiles.lights.len() as f32 / (tiles.tiles.len() * lights.len()) as f32;
        (missed, reached, filled)
    }

    #[test]
    fn tiles_list_every_light_reaching_their_points() {
        for perspective in [true, false] {
            let (missed, reached, filled) = missed(perspective, 1.0);
            assert_eq!(missed, 0);
            // Not vacuous: point lights reach points, and tiles leave most
            // out. Lights straddling the eye plane are in every tile.
            assert!(reached > 1000 && filled < 0.5);
        }
    }

    #[test]
    fn tiles_of_undersized_radii_miss_lights() {
        assert!(missed(true, 0.5).0 > 0);
        assert!(missed(false, 0.5).0 > 0);
    }

    #[test]
    fn tiles_list_lights_in_order() {
        let lights = lights(&mut Random(7));
        let tiles = Tiles::new(&lights, &camera(true), [64, 48]);
        for [first, count] in &tiles.tiles {
            let listed = &tiles.lights[*first as usize..(first + count) as usize];
            assert!(listed.is_sorted() && listed.contains(&7));
        }
        let listed = tiles.tiles.iter().map(|t| t[1] as usize).sum::<usize>();
        assert_eq!(listed, tiles.lights.len());
    }

    #[test]
    fn a_point_light_reaches_where_it_gives_the_threshold() {
        let light = Light::point([0.0; 3], [0.5, 2.0, 1.0]);
        assert!((2.0 / (light.radius * light.radius) - THRESHOLD).abs() < 1e-7);
        // Reach from the dimmest channel would leave the brightest unfaded.
        assert!(0.5 / (light.radius * light.radius) < THRESHOLD);
    }

    #[test]
    fn lights_behind_the_eye_reach_no_tile() {
        let behind = Light::point([0.0, 0.0, 8.0], [0.01; 3]);
        assert!(behind.radius < 3.0);
        let tiles = Tiles::new(&[behind], &camera(true), [64, 48]);
        assert!(tiles.lights.is_empty());
    }
}
