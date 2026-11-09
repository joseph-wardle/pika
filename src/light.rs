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
/// each light (`Light::sphere`, `Light::area`).
pub const THRESHOLD: f32 = 0.01;

/// A light's shape, `shaders/view.slang`'s constants.
pub const DISTANT: u32 = 0;
pub const SPHERE: u32 = 1;
pub const RECT: u32 = 2;
pub const DISK: u32 = 3;
pub const CYLINDER: u32 = 4;

/// The scale of a disk's polygon's axes that gives the regular octagon its
/// disk's area, so it shows as much from far away.
const OCTAGON: f32 = 1.053_907_4;

/// `shaders/view.slang`'s `Light`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Light {
    /// The unit direction toward a distant light, or the centre of another.
    pub position: [f32; 3],
    /// A sphere's or cylinder's radius.
    pub radius: f32,
    /// The illuminance a distant light gives a surface facing it, or a
    /// sphere at unit distance from its centre; the radiance of an area
    /// light.
    pub illuminance: [f32; 3],
    /// How far it reaches: infinite for a distant light, and only for one.
    pub reach: f32,
    /// A rect's half sides or the axes of a disk's polygon, which light the
    /// side u × v points to; half a cylinder's axis.
    pub u: [f32; 3],
    pub shape: u32,
    pub v: [f32; 3],
    /// What a sphere's 1/d² or an area light's illuminance vector loses so
    /// that it ends at its reach.
    pub fade: f32,
}

impl Light {
    pub fn distant(direction: [f32; 3], illuminance: [f32; 3]) -> Self {
        Self {
            position: direction,
            radius: 0.0,
            illuminance,
            reach: f32::INFINITY,
            u: [0.0; 3],
            shape: DISTANT,
            v: [0.0; 3],
            fade: 0.0,
        }
    }

    /// A sphere at `centre`; of radius 0, a point. Its reach is where its
    /// brightest channel falls to `THRESHOLD`: the light loop gives
    /// illuminance × (1/d² - 1/reach²), which loses that much everywhere
    /// and so reaches 0 at the reach without a step.
    pub fn sphere(centre: [f32; 3], radius: f32, illuminance: [f32; 3]) -> Self {
        let reach = (brightest(illuminance) / THRESHOLD).sqrt();
        Self {
            position: centre,
            radius,
            illuminance,
            reach,
            u: [0.0; 3],
            shape: SPHERE,
            v: [0.0; 3],
            fade: 1.0 / (reach * reach),
        }
    }

    /// A rect, disk or cylinder of `radiance` (`shape` and the axes as in
    /// `Light`), of `area` and within `bound` of its centre. A light shows
    /// no more of itself than its area, nearer than `bound` less than its
    /// centre's distance d, so its illuminance vector is under
    /// area / (d - bound)². The loop takes `THRESHOLD` / radiance off it,
    /// which loses that much and ends the light where the bound falls to
    /// it.
    #[allow(clippy::too_many_arguments)]
    fn area(
        shape: u32,
        centre: [f32; 3],
        u: [f32; 3],
        v: [f32; 3],
        radius: f32,
        radiance: [f32; 3],
        area: f32,
        bound: f32,
    ) -> Self {
        let brightest = brightest(radiance);
        Self {
            position: centre,
            radius,
            illuminance: radiance,
            reach: bound + (brightest * area / THRESHOLD).sqrt(),
            u,
            shape,
            v,
            fade: THRESHOLD / brightest,
        }
    }
}

fn brightest(c: [f32; 3]) -> f32 {
    c.into_iter().fold(0.0, f32::max)
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
    let scaled = |v: [f32; 3], s: f32| v.map(|c| c * s);
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
        // Any transform keeps a rect a parallelogram and a disk an ellipse,
        // whose outlines move with it.
        Shape::Rect { width, height } => {
            let [u, v] = lit_side(
                scaled(row(0), width / 2.0),
                scaled(row(1), height / 2.0),
                row(2),
            );
            let area = 4.0 * length(cross(u, v));
            let bound = length(u) + length(v);
            let radiance = area_radiance(params, area)?;
            Some(Light::area(RECT, row(3), u, v, 0.0, radiance, area, bound))
        }
        Shape::Disk { radius } => {
            let [u, v] = lit_side(scaled(row(0), radius), scaled(row(1), radius), row(2));
            let area = PI * length(cross(u, v));
            let [u, v] = [u, v].map(|a| scaled(a, OCTAGON));
            let bound = length(u) + length(v);
            let radiance = area_radiance(params, area)?;
            Some(Light::area(DISK, row(3), u, v, 0.0, radiance, area, bound))
        }
        // Its radius scaled by the geometric mean of y's and z's lengths,
        // exact while they scale evenly. Its silhouette's points lie within
        // 1 / cos 30° of its radius (shaders/frame.slang).
        Shape::Cylinder { length: l, radius } => {
            let u = scaled(row(0), l / 2.0);
            let radius = radius.abs() * (length(row(1)) * length(row(2))).sqrt();
            let area = 4.0 * PI * radius * length(u);
            let bound = length(u) + radius / (PI / 6.0).cos();
            let radiance = area_radiance(params, area)?;
            Some(Light::area(
                CYLINDER,
                row(3),
                u,
                [0.0; 3],
                radius,
                radiance,
                area,
                bound,
            ))
        }
    }?;
    (params.visible && light.illuminance != [0.0; 3]).then_some(light)
}

/// `u` and `v` in the order whose cross product points to the side the
/// light emits toward, the one `z` points away from.
fn lit_side(u: [f32; 3], v: [f32; 3], z: [f32; 3]) -> [[f32; 3]; 2] {
    let n = cross(u, v);
    match n[0] * z[0] + n[1] * z[1] + n[2] * z[2] > 0.0 {
        true => [v, u],
        false => [u, v],
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// UsdLux's radiance in nits: the intensity scaled by 2^exposure and the
/// color, before normalizing divides it by the light's size factor
/// (`usdLux/lightAPI.h` on normalize).
fn radiance(p: &LightParams) -> [f32; 3] {
    p.color.map(|c| c * p.intensity * p.exposure.exp2())
}

/// An area light's radiance, normalized by its world-space `area`: `None`
/// for a light of no area, which gives nothing.
fn area_radiance(p: &LightParams, area: f32) -> Option<[f32; 3]> {
    let size = if p.normalize { area } else { 1.0 };
    (area > 0.0).then(|| radiance(p).map(|l| l / size))
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

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        (0..3).map(|i| a[i] * b[i]).sum()
    }

    fn add(a: [f32; 3], b: [f32; 3], s: f32) -> [f32; 3] {
        std::array::from_fn(|i| a[i] + b[i] * s)
    }

    /// `shaders/frame.slang`'s illuminance_vector.
    fn illuminance_vector(light: &Light, p: [f32; 3]) -> [f32; 3] {
        let c = add(light.position, p, -1.0);
        let (u, v) = (light.u, light.v);
        let corners: Vec<[f32; 3]> = match light.shape {
            CYLINDER => {
                let axis = light.u.map(|a| a / dot(light.u, light.u).sqrt());
                let out = add(axis.map(|a| a * dot(c, axis)), c, -1.0);
                let rho = dot(out, out).sqrt();
                if rho <= light.radius {
                    return [0.0; 3];
                }
                let out = out.map(|o| o / rho);
                let side = cross(axis, out);
                let phi = (light.radius / rho).acos();
                let arc: Vec<_> = (0..4)
                    .map(|k| {
                        let t = phi * (1.0 - 2.0 * k as f32 / 3.0);
                        let r = match k {
                            0 | 3 => light.radius,
                            _ => light.radius / (phi / 3.0).cos(),
                        };
                        add(out.map(|o| o * r * t.cos()), side, r * t.sin())
                    })
                    .collect();
                let ends = |e: f32| arc.iter().map(move |a| add(add(c, u, e), *a, 1.0));
                ends(1.0).chain(ends(-1.0).rev()).collect()
            }
            _ if dot(c, cross(u, v)) >= 0.0 => return [0.0; 3],
            RECT => [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
                .map(|[a, b]| add(add(c, u, a), v, b))
                .to_vec(),
            _ => (0..8)
                .map(|k| {
                    let t = k as f32 * PI / 4.0;
                    add(add(c, u, t.cos()), v, t.sin())
                })
                .collect(),
        };
        let mut sum = [0.0; 3];
        for (i, a) in corners.iter().enumerate() {
            let b = corners[(i + 1) % corners.len()];
            let k = cross(*a, b);
            let s = dot(k, k).sqrt();
            if s > 0.0 {
                sum = add(sum, k, s.atan2(dot(*a, b)) / s);
            }
        }
        let sign = if dot(sum, c) < 0.0 { -0.5 } else { 0.5 };
        sum.map(|v| v * sign)
    }

    /// What `shaders/frame.slang`'s loop gives a point `p` with normal `n`
    /// from a sphere or an area light, in its first channel.
    fn incident(light: &Light, p: [f32; 3], n: [f32; 3]) -> f32 {
        if light.shape != SPHERE {
            let e = illuminance_vector(light, p);
            let magnitude = dot(e, e).sqrt();
            if magnitude == 0.0 {
                return 0.0;
            }
            let cos = dot(e, n) / magnitude;
            return light.illuminance[0] * (magnitude - light.fade).max(0.0) * cos.max(0.0);
        }
        let toward = add(light.position, p, -1.0);
        let d2 = dot(toward, toward);
        let cos = dot(toward, n) / d2.sqrt();
        let falloff = 1.0 / d2.max(light.radius * light.radius) - light.fade;
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

    /// A transform: scaled along x, y and z, turned `turn` degrees about y
    /// and moved to `centre`.
    fn transform(scale: [f64; 3], turn: f64, centre: [f64; 3]) -> [f64; 16] {
        let (sin, cos) = turn.to_radians().sin_cos();
        let [x, y, z] = scale;
        let [a, b, c] = centre;
        #[rustfmt::skip]
        let m = [
            x * cos, 0.0, -x * sin, 0.0,
            0.0, y, 0.0, 0.0,
            z * sin, 0.0, z * cos, 0.0,
            a, b, c, 1.0,
        ];
        m
    }

    fn area_light(shape: Shape, intensity: f32, normalize: bool, m: [f64; 16]) -> store::Light {
        store::Light {
            path: "/panel".into(),
            params: Some(LightParams {
                shape,
                ..params(intensity, 0.0, normalize)
            }),
            transform: Some(m),
        }
    }

    type V = [f64; 3];

    fn add64(a: V, b: V, s: f64) -> V {
        std::array::from_fn(|i| a[i] + b[i] * s)
    }

    fn dot64(a: V, b: V) -> f64 {
        (0..3).map(|i| a[i] * b[i]).sum()
    }

    /// The illuminance a light of radiance 1 and transform `m` gives a
    /// point `p` with unit normal `n`, summed over its surface, where only
    /// what faces `p` is seen; and its area. `surface` maps s and t in
    /// [0, 1) and the matrix's rows to a point, its unit normal and the
    /// area per unit of s and t.
    fn over_surface(
        m: &[f64; 16],
        surface: impl Fn(f64, f64, [V; 4]) -> (V, V, f64),
        p: V,
        n: V,
    ) -> (f64, f64) {
        let rows = std::array::from_fn(|r| [m[4 * r], m[4 * r + 1], m[4 * r + 2]]);
        let steps = 500;
        let (mut sum, mut area) = (0.0, 0.0);
        for i in 0..steps {
            for j in 0..steps {
                let (s, t) = (
                    (i as f64 + 0.5) / steps as f64,
                    (j as f64 + 0.5) / steps as f64,
                );
                let (x, normal, da) = surface(s, t, rows);
                let da = da / (steps * steps) as f64;
                let toward = add64(x, p, -1.0);
                let d2 = dot64(toward, toward);
                let lit = (-dot64(toward, normal)).max(0.0) * dot64(toward, n).max(0.0);
                sum += lit / (d2 * d2) * da;
                area += da;
            }
        }
        (sum, area)
    }

    fn length64(v: V) -> f64 {
        dot64(v, v).sqrt()
    }

    fn cross64(a: V, b: V) -> V {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    /// A rect's, disk's or cylinder's surface, as `over_surface` takes it.
    fn surface(shape: Shape) -> impl Fn(f64, f64, [V; 4]) -> (V, V, f64) {
        move |s, t, [x, y, z, o]| {
            let lit = cross64(x, y);
            let lit = lit.map(|v| v / length64(lit) * -dot64(lit, z).signum());
            match shape {
                Shape::Rect { width, height } => {
                    let (w, h) = (width as f64, height as f64);
                    let p = add64(add64(o, x, w * (s - 0.5)), y, h * (t - 0.5));
                    (p, lit, w * h * length64(cross64(x, y)))
                }
                Shape::Disk { radius } => {
                    let (r, a) = (radius as f64 * s, 2.0 * std::f64::consts::PI * t);
                    let p = add64(add64(o, x, r * a.cos()), y, r * a.sin());
                    let da = 2.0 * std::f64::consts::PI * radius as f64 * r;
                    (p, lit, da * length64(cross64(x, y)))
                }
                Shape::Cylinder { length, radius } => {
                    let (l, r) = (length as f64, radius as f64);
                    let a = 2.0 * std::f64::consts::PI * t;
                    let out = add64(y.map(|v| v * a.cos()), z, a.sin());
                    let p = add64(add64(o, x, l * (s - 0.5)), out, r);
                    let da = l * length64(x) * 2.0 * std::f64::consts::PI * r * length64(out);
                    (p, out.map(|v| v / length64(out)), da)
                }
                _ => unreachable!(),
            }
        }
    }

    /// A rect a fifth of its size from a point, then further, a disk and a
    /// cylinder near and far, each wholly above the point's horizon and
    /// seen square on or obliquely: the shape, its transform, the point
    /// and its normal, and what Pika may lose of the reference.
    fn area_cases() -> Vec<(Shape, [f64; 16], V, V, f64)> {
        let rect = Shape::Rect {
            width: 2.0,
            height: 1.0,
        };
        let disk = Shape::Disk { radius: 0.5 };
        let tube = Shape::Cylinder {
            length: 2.0,
            radius: 0.25,
        };
        let (facing, tilted) = ([0.0, 0.0, 1.0], [0.3, 0.0, -0.954]);
        // The shape, its scale, turn and centre, the normal and the loss.
        #[rustfmt::skip]
        let cases = [
            (rect, [1.0; 3], 0.0, [0.0, 0.0, 0.4], facing, 2e-3),
            (rect, [0.6, 0.5, 2.0], 210.0, [0.5, 0.2, -0.6], tilted, 2e-3),
            (rect, [1.0; 3], 20.0, [1.0, -0.5, 6.0], facing, 2e-3),
            (disk, [1.0; 3], 0.0, [0.0, 0.0, 0.2], facing, 6e-3),
            (disk, [2.0, 1.0, 1.0], 150.0, [0.3, 0.4, -0.7], tilted, 6e-3),
            (disk, [1.0; 3], 10.0, [-1.0, 0.0, 5.0], facing, 6e-3),
            (tube, [1.0; 3], 0.0, [0.0, 0.0, 0.6], facing, 1.5e-2),
            (tube, [1.5, 2.0, 2.0], 40.0, [0.8, 0.3, -1.8], tilted, 1.5e-2),
            (tube, [1.0; 3], 60.0, [1.5, 0.0, 6.0], facing, 1.5e-2),
        ];
        (cases.into_iter())
            .map(|(shape, scale, turn, centre, n, loss)| {
                (shape, transform(scale, turn, centre), [0.0; 3], n, loss)
            })
            .collect()
    }

    /// Bright enough that the fade is lost in the tolerances.
    const BRIGHT: f32 = 1000.0;

    fn f32s(v: V) -> [f32; 3] {
        v.map(|c| c as f32)
    }

    #[test]
    fn area_lights_give_what_their_surface_does() {
        for (shape, m, p, n, tolerance) in area_cases() {
            let (reference, area) = over_surface(&m, surface(shape), p, n);
            assert!(reference > 0.01, "{shape:?}: {reference}");
            for normalize in [false, true] {
                let light = resolve(&area_light(shape, BRIGHT, normalize, m)).unwrap();
                let pika = incident(&light, f32s(p), f32s(n)) as f64;
                let size = if normalize { area } else { 1.0 };
                let reference = BRIGHT as f64 * reference / size;
                // What the fade loses, besides.
                let allowed =
                    tolerance * reference + light.fade as f64 * light.illuminance[0] as f64;
                assert!(
                    (pika - reference).abs() < allowed,
                    "{shape:?} {normalize}: {pika} {reference}"
                );
            }
        }
    }

    #[test]
    fn area_lights_turned_or_read_as_points_miss_their_surface() {
        let mut near = 0;
        for (shape, m, p, n, tolerance) in area_cases() {
            let o = [m[12], m[13], m[14]];
            let toward = add64(o, p, -1.0);
            let d2 = dot64(toward, toward);
            // From further, either can come close.
            if d2 >= 1.0 {
                continue;
            }
            near += 1;
            let (reference, area) = over_surface(&m, surface(shape), p, n);
            let reference = BRIGHT as f64 * reference;
            let missed = |pika: f64| (pika - reference).abs() > 3.0 * tolerance * reference;
            // Turned 90° about y.
            let [x, _, z] = [0, 4, 8].map(|r| [m[r], m[r + 1], m[r + 2]]);
            let mut turned = m;
            turned[..3].copy_from_slice(&z.map(|v| -v));
            turned[8..11].copy_from_slice(&x);
            let light = resolve(&area_light(shape, BRIGHT, false, turned)).unwrap();
            let pika = incident(&light, f32s(p), f32s(n)) as f64;
            assert!(missed(pika), "{shape:?} turned");
            // A point at its centre showing the point its area, or a
            // cylinder its side: the naive inverse square.
            let shown = match shape {
                Shape::Cylinder { .. } => area / std::f64::consts::PI,
                _ => area,
            };
            let point = BRIGHT as f64 * shown * dot64(toward, n).max(0.0) / (d2 * d2.sqrt());
            assert!(missed(point), "{shape:?} point");
        }
        // Two rects, two disks and a cylinder.
        assert_eq!(near, 5);
    }

    #[test]
    fn rects_and_disks_light_one_side() {
        for shape in [
            Shape::Rect {
                width: 1.0,
                height: 1.0,
            },
            Shape::Disk { radius: 0.5 },
        ] {
            // It shines down -z, so a point below it facing up is lit, and
            // one above it facing down is not. Mirrored in x, it still
            // shines down -z; mirrored in z, its -z is up.
            let mirrors = [
                ([1.0; 3], -1.0),
                ([-1.0, 1.0, 1.0], -1.0),
                ([1.0, 1.0, -1.0], 1.0),
            ];
            for (scale, lit) in mirrors {
                let m = transform(scale, 0.0, [0.0; 3]);
                let light = resolve(&area_light(shape, 1.0, false, m)).unwrap();
                assert!(incident(&light, [0.0, 0.0, lit], [0.0, 0.0, -lit]) > 0.1);
                assert_eq!(incident(&light, [0.0, 0.0, -lit], [0.0, 0.0, lit]), 0.0);
            }
            // Turned over, it lights the other side.
            let over = resolve(&area_light(
                shape,
                1.0,
                false,
                transform([1.0; 3], 180.0, [0.0; 3]),
            ))
            .unwrap();
            assert!(incident(&over, [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]) > 0.1);
        }
    }

    #[test]
    fn area_lights_end_at_their_reach() {
        let mut random = Random(0x2545_f491_4f6c_dd1d);
        for (shape, m, ..) in area_cases() {
            let light = resolve(&area_light(shape, 30.0, false, m)).unwrap();
            let unfaded = Light { fade: 0.0, ..light };
            let mut lit_past = 0;
            for _ in 0..2000 {
                let d = [0; 3].map(|_| random.next(-1.0, 1.0));
                let length = dot(d, d).sqrt();
                if length > 1.0 || length < 0.1 {
                    continue;
                }
                let d = d.map(|v| v / length);
                for scale in [0.3, 0.9, 1.0, 1.5] {
                    let p = add(light.position, d, light.reach * scale);
                    let n = d.map(|v| -v);
                    let (faded, full) = (incident(&light, p, n), incident(&unfaded, p, n));
                    assert!(full - faded <= THRESHOLD * 1.001 && faded <= full);
                    if scale >= 1.0 {
                        assert_eq!(faded, 0.0, "{shape:?}");
                        lit_past += (full > 0.0) as usize;
                    }
                }
            }
            // Not vacuous: without the fade, it lights past its reach.
            assert!(lit_past > 0, "{shape:?}");
        }
    }

    #[test]
    fn lights_behind_the_eye_reach_no_tile() {
        let behind = Light::sphere([0.0, 0.0, 8.0], 0.0, [0.01; 3]);
        assert!(behind.reach < 3.0);
        let tiles = Tiles::new(&[behind], &camera(true), [64, 48]);
        assert!(tiles.lights.is_empty());
    }
}
