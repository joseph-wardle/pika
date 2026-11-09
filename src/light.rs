//! The lights a view draws, as `shaders/frame.slang` reads them, and the
//! screen tiles each may reach. A light reaches as far as what it gives
//! falls to `THRESHOLD`, and each `TILE`-pixel tile lists the lights that
//! reach into it, so a pixel's light loop runs over its tile's lights
//! alone. The lists are built on the CPU each frame from the lights' boxes,
//! which costs little at tens of lights.

use std::f32::consts::PI;
use std::ops::Range;

use crate::store::{self, LightParams, Shape};
use crate::transform::Camera;

/// The side of a tile in pixels.
pub const TILE: u32 = 16;

/// The illuminance at which a light's reach ends. A facing white Lambert
/// surface shows illuminance / π, so a pixel loses at most this / π of
/// each light (`Light::sphere`).
pub const THRESHOLD: f32 = 0.01;

/// `shaders/view.slang`'s `Light`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Light {
    /// The unit direction toward a distant light, or a sphere's centre.
    pub position: [f32; 3],
    /// A sphere's radius: 0 for a distant light.
    pub radius: f32,
    /// The illuminance it gives a surface facing it: for a sphere, at unit
    /// distance from its centre.
    pub illuminance: [f32; 3],
    /// How far it reaches: infinite for a distant light, and only for one.
    pub reach: f32,
}

impl Light {
    pub fn distant(direction: [f32; 3], illuminance: [f32; 3]) -> Self {
        Self {
            position: direction,
            radius: 0.0,
            illuminance,
            reach: f32::INFINITY,
        }
    }

    /// A sphere at `centre`; of radius 0, a point. Its reach is where its
    /// brightest channel falls to `THRESHOLD`: the light loop gives
    /// illuminance × (1/d² - 1/reach²), which loses that much everywhere
    /// and so reaches 0 at the reach without a step.
    pub fn sphere(centre: [f32; 3], radius: f32, illuminance: [f32; 3]) -> Self {
        let brightest = illuminance.into_iter().fold(0.0, f32::max);
        Self {
            position: centre,
            radius,
            illuminance,
            reach: (brightest / THRESHOLD).sqrt(),
        }
    }
}

/// A UsdLux light as the loop draws it: `None` until Hydra has handed over
/// its params and transform, while it is hidden and when it gives nothing.
pub fn resolve(light: &store::Light) -> Option<Light> {
    let (Some(params), Some(m)) = (&light.params, &light.transform) else {
        return None;
    };
    // The row-vector matrix's rows are where +x, +y and +z go, then where
    // the origin goes.
    let row = |r: usize| [m[4 * r], m[4 * r + 1], m[4 * r + 2]].map(|v| v as f32);
    let length = |v: [f32; 3]| v.iter().map(|c| c * c).sum::<f32>().sqrt();
    let light = match params.shape {
        // It shines down its -z axis.
        Shape::Distant { angle } => {
            let z = row(2);
            let length = length(z);
            (length > 0.0).then(|| Light::distant(z.map(|v| v / length), distant(params, angle)))
        }
        // Scaled by the geometric mean of its axes' lengths, which is exact
        // under a uniform scale.
        Shape::Sphere { radius } => {
            let scale = (length(row(0)) * length(row(1)) * length(row(2))).cbrt();
            let radius = radius.abs() * scale;
            Some(Light::sphere(row(3), radius, sphere(params, radius)))
        }
    }?;
    (params.visible && light.illuminance != [0.0; 3]).then_some(light)
}

/// UsdLux's radiance in nits: the intensity scaled by 2^exposure and the
/// color, before normalizing divides it by the light's size factor
/// (`usdLux/lightAPI.h` on normalize).
fn radiance(p: &LightParams) -> [f32; 3] {
    p.color.map(|c| c * p.intensity * p.exposure.exp2())
}

/// The illuminance a distant light gives a surface facing it. A disk of
/// angular radius θ gives π sin²θ times its radiance; a perfectly parallel
/// light (angle 0) gives its radiance, as lux, whose size factor is 1.
fn distant(p: &LightParams, angle: f32) -> [f32; 3] {
    let theta = (angle.to_radians() / 2.0).clamp(0.0, PI);
    let sin2 = theta.sin().powi(2);
    // The light past a surface's horizon does not reach it.
    let (received, size) = if theta == 0.0 {
        (1.0, 1.0)
    } else if theta <= PI / 2.0 {
        (PI * sin2, PI * sin2)
    } else {
        (PI, (2.0 - sin2) * PI)
    };
    let size = if p.normalize { size } else { 1.0 };
    radiance(p).map(|l| l * received / size)
}

/// The illuminance a sphere of `radius`, world-space, gives a surface facing
/// it at unit distance from its centre. Seen from d away, it subtends an
/// angle α with sin α = radius / d and gives π sin²α times its radiance:
/// π radius² / d², a point's falloff. While the whole sphere is above the
/// surface's horizon, the surface receives that times the cosine toward
/// its centre, as from a point there, so the loop's point is exact. Its
/// size factor is its area, 4π radius², so normalized it gives a quarter of
/// its intensity at any radius, a point's too.
fn sphere(p: &LightParams, radius: f32) -> [f32; 3] {
    let received = if p.normalize {
        0.25
    } else {
        PI * radius * radius
    };
    radiance(p).map(|l| l * received)
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
/// box of its box's corners. A point inside the light's reach is inside
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
    if light.reach == f32::INFINITY {
        return every;
    }
    if light.reach <= 0.0 {
        return None;
    }
    let mut low = [f32::INFINITY; 2];
    let mut high = [f32::NEG_INFINITY; 2];
    let mut behind = 0;
    for corner in 0..8 {
        let p: [f32; 3] = std::array::from_fn(|i| {
            let side = if corner >> i & 1 == 1 { 1.0 } else { -1.0 };
            light.position[i] + side * light.reach
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

    /// Points across the view, with a distant light among them.
    fn lights(random: &mut Random) -> Vec<Light> {
        let mut lights: Vec<_> = (0..40)
            .map(|_| {
                let p = [-4.0, -4.0, -6.0].map(|v| random.next(v, -v));
                Light::sphere(p, 0.0, [random.next(0.0, 0.02), 0.005, 0.001])
            })
            .collect();
        lights.insert(7, Light::distant([0.0, 0.0, 1.0], [1.0; 3]));
        lights
    }

    /// Over random visible points, the lights reaching a point that its
    /// tile does not list, and those reaching one, with the tiles built
    /// from the lights' reach times `shrink`; and the share of the tiles'
    /// slots, every light in every tile, that they fill.
    fn missed(perspective: bool, shrink: f32) -> (usize, usize, f32) {
        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        let camera = camera(perspective);
        let extent = [333, 200];
        let lights = lights(&mut random);
        let binned: Vec<_> = (lights.iter())
            .map(|l| Light {
                reach: l.reach * shrink,
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
                let reaches = l.reach == f32::INFINITY || d < l.reach * l.reach;
                reached += (reaches && l.reach.is_finite()) as usize;
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
    fn tiles_of_undersized_reach_miss_lights() {
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
        let light = Light::sphere([0.0; 3], 0.0, [0.5, 2.0, 1.0]);
        assert!((2.0 / (light.reach * light.reach) - THRESHOLD).abs() < 1e-7);
        // Reach from the dimmest channel would leave the brightest unfaded.
        assert!(0.5 / (light.reach * light.reach) < THRESHOLD);
    }

    fn params(intensity: f32, angle: f32, normalize: bool) -> LightParams {
        LightParams {
            intensity,
            exposure: 0.0,
            color: [1.0; 3],
            normalize,
            shape: Shape::Distant { angle },
            visible: true,
        }
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= 1e-5 * b[i].abs().max(1e-6))
    }

    #[test]
    fn distant_lights_give_usdlux_illuminance() {
        // Hydra's camera light (HdxTaskController): 15000 nits from a disk
        // the sun's size, unnormalized, by ∫cos dω = π sin²(0.265°).
        let camera = distant(&params(15000.0, 0.53, false), 0.53)[0];
        assert!((camera - 1.008).abs() < 5e-4, "{camera}");
        // Reading the intensity as lux is four orders of magnitude off.
        assert!(15000.0 / camera > 1e4);
        // Normalized, the intensity is lux at any angle; at angle 0 the
        // size factor is 1 either way.
        for (angle, normalize) in [(0.53, true), (0.0, false), (0.0, true), (90.0, true)] {
            assert!(close(
                distant(&params(2.5, angle, normalize), angle),
                [2.5; 3]
            ));
        }
        // Exposure doubles per stop, and the color scales each channel.
        let p = LightParams {
            exposure: 1.5,
            color: [1.0, 0.5, 0.25],
            ..params(2.0, 0.0, false)
        };
        let lux = 2.0 * 1.5f32.exp2();
        assert!(close(distant(&p, 0.0), [lux, lux / 2.0, lux / 4.0]));
        // Past the horizon, a wider light lights no more of a facing
        // surface, but normalizes by its whole size.
        let wide = distant(&params(1.0, 270.0, false), 270.0)[0];
        assert!((wide - PI).abs() < 1e-5);
        let sin2 = (135f32.to_radians()).sin().powi(2);
        let normalized = distant(&params(1.0, 270.0, true), 270.0)[0];
        assert!((normalized - 1.0 / (2.0 - sin2)).abs() < 1e-6);
    }

    #[test]
    fn distant_lights_shine_down_their_z_axis() {
        // Turned 90° about x, scaled by 2 and moved: +z goes to -y.
        #[rustfmt::skip]
        let transform = [
            2.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 0.0,
            0.0, -2.0, 0.0, 0.0,
            5.0, 6.0, 7.0, 1.0,
        ];
        let mut light = store::Light {
            path: "/sun".into(),
            params: Some(params(1.0, 0.0, false)),
            transform: Some(transform),
        };
        let resolved = resolve(&light).unwrap();
        assert_eq!(resolved.position, [0.0, -1.0, 0.0]);
        assert_eq!(resolved.illuminance, [1.0; 3]);
        // Hidden, black or not yet synced, it gives nothing.
        light.params.as_mut().unwrap().visible = false;
        assert_eq!(resolve(&light), None);
        light.params = Some(params(0.0, 0.0, false));
        assert_eq!(resolve(&light), None);
        light.params = None;
        assert_eq!(resolve(&light), None);
    }

    /// What `shaders/frame.slang`'s loop gives a point `p` with normal `n`
    /// from a sphere.
    fn incident(light: &Light, p: [f32; 3], n: [f32; 3]) -> f32 {
        let toward: [f32; 3] = std::array::from_fn(|i| light.position[i] - p[i]);
        let d2: f32 = toward.iter().map(|v| v * v).sum();
        let cos = (0..3).map(|i| toward[i] * n[i]).sum::<f32>() / d2.sqrt();
        let falloff = 1.0 / d2.max(light.radius * light.radius) - 1.0 / light.reach.powi(2);
        light.illuminance[0] * falloff.max(0.0) * cos.max(0.0)
    }

    /// The illuminance a sphere of radiance 1 and `radius`, `distance` away
    /// along +z, gives a surface with unit normal `n`: the cosine to `n`
    /// summed over the cone of directions the sphere fills.
    fn integrated(radius: f32, distance: f32, n: [f32; 3]) -> f32 {
        let alpha = (radius / distance).asin();
        let steps = 600;
        let (dt, dp) = (alpha / steps as f32, 2.0 * PI / steps as f32);
        let mut sum = 0.0;
        for i in 0..steps {
            let t = (i as f32 + 0.5) * dt;
            for j in 0..steps {
                let p = (j as f32 + 0.5) * dp;
                let d = [t.sin() * p.cos(), t.sin() * p.sin(), t.cos()];
                let cos = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
                sum += cos.max(0.0) * t.sin() * dt * dp;
            }
        }
        sum
    }

    /// A sphere light of radius 0.5, turned 90° about z, scaled by `scale`
    /// and moved to `centre`.
    fn sphere_light(intensity: f32, normalize: bool, scale: f64, centre: [f64; 3]) -> store::Light {
        let [x, y, z] = centre;
        #[rustfmt::skip]
        let transform = [
            0.0, scale, 0.0, 0.0,
            -scale, 0.0, 0.0, 0.0,
            0.0, 0.0, scale, 0.0,
            x, y, z, 1.0,
        ];
        store::Light {
            path: "/bulb".into(),
            params: Some(LightParams {
                shape: Shape::Sphere { radius: 0.5 },
                ..params(intensity, 0.0, normalize)
            }),
            transform: Some(transform),
        }
    }

    #[test]
    fn spheres_give_what_their_cone_does_above_the_horizon() {
        // A surface at the origin, tilted 30° from facing the sphere, which
        // is wholly above its horizon at each distance.
        let n = [30f32.to_radians().sin(), 0.0, 30f32.to_radians().cos()];
        for distance in [1.5, 4.0, 12.0] {
            let reference = 100.0 * integrated(1.0, distance as f32, n);
            let light = resolve(&sphere_light(100.0, false, 2.0, [0.0, 0.0, distance])).unwrap();
            let pika = incident(&light, [0.0; 3], n);
            assert!(
                (pika - reference).abs() < 2e-3 * reference + THRESHOLD,
                "{pika} {reference}"
            );
            // Ignoring the transform's scale reads a radius of 0.5, a
            // quarter of the light.
            let unscaled = resolve(&sphere_light(100.0, false, 1.0, [0.0, 0.0, distance])).unwrap();
            assert!(incident(&unscaled, [0.0; 3], n) < 0.3 * reference);
            // Normalized, it is divided by its area, 4π.
            let normalized =
                resolve(&sphere_light(100.0, true, 2.0, [0.0, 0.0, distance])).unwrap();
            let pika = incident(&normalized, [0.0; 3], n);
            let reference = reference / (4.0 * PI);
            assert!(
                (pika - reference).abs() < 2e-3 * reference + THRESHOLD,
                "{pika} {reference}"
            );
        }
    }

    #[test]
    fn spheres_give_no_more_than_their_radiance_inside() {
        // Inside, the sphere fills every direction: π times its radiance.
        let light = resolve(&sphere_light(1.0, false, 2.0, [0.0, 0.0, 0.5])).unwrap();
        let facing = incident(&light, [0.0; 3], [0.0, 0.0, 1.0]);
        assert!((facing - PI).abs() < THRESHOLD, "{facing}");
        // Without the bound, a point there gives 4π.
        assert!(light.illuminance[0] / 0.25 > 3.0 * PI);
    }

    #[test]
    fn lights_behind_the_eye_reach_no_tile() {
        let behind = Light::sphere([0.0, 0.0, 8.0], 0.0, [0.01; 3]);
        assert!(behind.reach < 3.0);
        let tiles = Tiles::new(&[behind], &camera(true), [64, 48]);
        assert!(tiles.lights.is_empty());
    }
}
