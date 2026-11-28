//! Hydra's transforms, converted for `shaders/mesh.slang`. `GfMatrix4d` multiplies row vectors
//! and the shader multiplies column vectors, so each matrix is sent transposed, after the maths
//! is done in f64.

use std::collections::HashMap;

use crate::link::{Links, Mask};
use crate::store::{Instancer, Instancing};

/// Laid out as the shader's 7 `float4`s per instance.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Instance {
    world: [[f32; 4]; 3],
    normal: [[f32; 4]; 3],
    /// Bit 0 of [0] is set when front faces wind clockwise. [1] and [2] hold the light-link and
    /// shadow-link collection bits.
    flags: [u32; 4],
}

impl Instance {
    /// The rows of the column-vector matrix.
    pub fn world(&self) -> &[[f32; 4]; 3] {
        &self.world
    }

    pub fn shadow_link(&self) -> u32 {
        self.flags[2]
    }

    /// `None` when the transform flattens the mesh, since it would then cover no pixels.
    pub fn new(matrix: &[f64; 16], left_handed: bool, link: Mask) -> Option<Self> {
        let linear = linear(matrix);
        let det = det3(&linear);
        let inverse = inverse3(&linear, det)?;
        let world = std::array::from_fn(|j| column(matrix, j).map(|v| v as f32));
        // Normals take the inverse transpose, which in column-vector form is the inverse read by rows.
        let normal = inverse.map(|[a, b, c]| [a as f32, b as f32, c as f32, 0.0]);
        // A mirroring transform reverses the winding, as a left-handed mesh does.
        let flip = left_handed != (det < 0.0);
        Some(Self {
            world,
            normal,
            flags: [flip as u32, link.light, link.shadow, 0],
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Camera {
    /// The rows of the column-vector matrix.
    pub view_projection: [[f32; 4]; 4],
    /// The camera's position (w = 1), or the direction toward an orthographic one (w = 0).
    pub eye: [f32; 4],
}

impl Camera {
    /// Hydra's clip z/w runs from -1 at the near plane to 1 at the far plane. Pika's runs from 1 to
    /// 0, reversed for precision far away. y isn't flipped, so the first image row is the bottom
    /// one, as in Hydra's buffers.
    pub fn new(world_to_view: &[f64; 16], projection: &[f64; 16]) -> Result<Self, String> {
        let m = multiply(world_to_view, projection);
        let [x, y, z, w] = std::array::from_fn(|j| column(&m, j));
        // z' = (w - z) / 2 is built into the matrix, so no vertex subtracts nearly equal values.
        let z = std::array::from_fn(|i| 0.5 * (w[i] - z[i]));
        let view_projection = [x, y, z, w].map(|row| row.map(|v| v as f32));

        let linear = linear(world_to_view);
        let inverse = inverse3(&linear, det3(&linear))
            .ok_or("the camera's world-to-view matrix is singular")?;
        // In row-vector form, the view-to-world matrix's last row is the eye, and its z row is the
        // direction toward the viewer.
        let t = [world_to_view[12], world_to_view[13], world_to_view[14]];
        let [a, b, c] = inverse[2];
        let length = (a * a + b * b + c * c).sqrt();
        let backward = [a / length, b / length, c / length];
        let eye = if projection[15] == 0.0 {
            let p: [f64; 3] =
                std::array::from_fn(|c| -(0..3).map(|r| t[r] * inverse[r][c]).sum::<f64>());
            [p[0], p[1], p[2], 1.0]
        } else {
            [backward[0], backward[1], backward[2], 0.0]
        };
        Ok(Self {
            view_projection,
            eye: eye.map(|v| v as f32),
        })
    }

    /// The camera whose pixels, over `extent`, sample `jitter` pixels from their centres along the
    /// attachment's x and y. The view moves by -`jitter`, and depth is unchanged.
    pub fn jittered(&self, jitter: [f64; 2], extent: [u32; 2]) -> Self {
        let mut c = *self;
        let w = self.view_projection[3].map(f64::from);
        for axis in 0..2 {
            // Clip x / w spans 2 across the extent.
            let shift = -2.0 * jitter[axis] / f64::from(extent[axis]);
            let row = self.view_projection[axis].map(f64::from);
            c.view_projection[axis] = std::array::from_fn(|k| (row[k] + shift * w[k]) as f32);
        }
        c
    }

    /// Rows taking clip (x, y, 1) to the world direction through that point, away from the camera.
    pub fn rays(&self) -> [[f64; 3]; 3] {
        let [x, y, _, w] = self
            .view_projection
            .map(|row| [0, 1, 2].map(|i| f64::from(row[i])));
        // A direction d at (x, y) has (x_row - x w_row)·d = (y_row - y w_row)·d = 0, so d is the
        // cross product, whose xy term cancels.
        let columns = [cross(y, w), cross(w, x), cross(x, y)];
        // w_row·d is the same everywhere, as is an orthographic camera's d.
        let ahead = match self.eye[3] {
            0.0 => -(0..3)
                .map(|i| f64::from(self.eye[i]) * columns[2][i])
                .sum::<f64>(),
            _ => (0..3).map(|i| w[i] * columns[2][i]).sum(),
        };
        std::array::from_fn(|r| columns.map(|c| c[r] * ahead.signum()))
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| a[(i + 1) % 3] * b[(i + 2) % 3] - a[(i + 2) % 3] * b[(i + 1) % 3])
}

/// The instances a prototype is drawn at, composed as Hydra does. At each instancer level the
/// primvars apply, then its transform. Each instance gets the link collections of every level.
pub fn instances(
    instancers: &HashMap<&str, &Instancer>,
    links: &Links,
    instancing: &Instancing,
) -> Result<Vec<([f64; 16], Mask)>, String> {
    let mut instances = vec![(IDENTITY, Mask::default())];
    let mut at = instancing;
    for _ in 0..=instancers.len() {
        let instancer = instancers
            .get(at.instancer.as_str())
            .ok_or_else(|| format!("no instancer {} in the store", at.instancer))?;
        let level = level(instancer, links, &at.indices)?;
        // Outer instances come first, as Hydra orders them.
        instances = (level.iter())
            .flat_map(|(outer, o)| {
                (instances.iter()).map(move |(inner, i)| (multiply(inner, outer), *i | *o))
            })
            .collect();
        let Some(Some(parent)) = &instancer.parent else {
            return Ok(instances);
        };
        at = parent;
    }
    Err(format!("instancer {} is nested in itself", at.instancer))
}

fn level(
    instancer: &Instancer,
    links: &Links,
    indices: &[i32],
) -> Result<Vec<([f64; 16], Mask)>, String> {
    let (Some(transform), Some(primvars), Some(categories)) = (
        &instancer.transform,
        &instancer.primvars,
        &instancer.categories,
    ) else {
        return Err(format!("instancer {} is not synced", instancer.path));
    };
    let masks: Vec<Mask> = (categories.sets.iter())
        .map(|c| links.categories(c))
        .collect();
    let get = |name: &str, count: usize, index: i32| {
        usize::try_from(index)
            .ok()
            .filter(|&i| i < count)
            .ok_or_else(|| {
                let path = &instancer.path;
                format!("instance index {index} is not one of {path}'s {count} {name}")
            })
    };
    let p = primvars;
    indices
        .iter()
        .map(|&index| {
            // With row vectors, the leftmost matrix applies first.
            let mut m = IDENTITY;
            if let Some(t) = &p.transforms {
                m = t[get("transforms", t.len(), index)?];
            }
            if let Some(s) = &p.scales {
                let [x, y, z] = s[get("scales", s.len(), index)?].map(f64::from);
                m = multiply(
                    &m,
                    &[x, 0., 0., 0., 0., y, 0., 0., 0., 0., z, 0., 0., 0., 0., 1.],
                );
            }
            if let Some(r) = &p.rotations {
                m = multiply(&m, &rotate(r[get("rotations", r.len(), index)?]));
            }
            if let Some(t) = &p.translations {
                let [x, y, z] = t[get("translations", t.len(), index)?].map(f64::from);
                m = multiply(
                    &m,
                    &[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., x, y, z, 1.],
                );
            }
            let s = &categories.set;
            let mask = if s.is_empty() {
                Mask::default()
            } else {
                masks[s[get("category sets", s.len(), index)?] as usize]
            };
            Ok((multiply(&m, transform), mask))
        })
        .collect()
}

/// Builds a rotation from (i, j, k, real), as Gf does. Point instancers author half-float
/// quaternions that aren't quite unit length, and dividing by the squared length keeps them
/// rigid.
fn rotate(q: [f32; 4]) -> [f64; 16] {
    let [i, j, k, r] = q.map(f64::from);
    let s = 2.0 / (i * i + j * j + k * k + r * r);
    [
        1.0 - s * (j * j + k * k),
        s * (i * j + k * r),
        s * (k * i - j * r),
        0.0,
        s * (i * j - k * r),
        1.0 - s * (k * k + i * i),
        s * (j * k + i * r),
        0.0,
        s * (k * i + j * r),
        s * (j * k - i * r),
        1.0 - s * (j * j + i * i),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

const IDENTITY: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

fn column(m: &[f64; 16], j: usize) -> [f64; 4] {
    [m[j], m[4 + j], m[8 + j], m[12 + j]]
}

fn linear(m: &[f64; 16]) -> [[f64; 3]; 3] {
    std::array::from_fn(|r| std::array::from_fn(|c| m[r * 4 + c]))
}

/// Row-major `a b`, where `a` applies first to a row vector.
pub fn multiply(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    std::array::from_fn(|i| (0..4).map(|k| a[i / 4 * 4 + k] * b[k * 4 + i % 4]).sum())
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn inverse3(m: &[[f64; 3]; 3], det: f64) -> Option<[[f64; 3]; 3]> {
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    // The adjugate's entry (r, c) is the cofactor of (c, r).
    Some(std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            let (r1, r2) = ((c + 1) % 3, (c + 2) % 3);
            let (c1, c2) = ((r + 1) % 3, (r + 2) % 3);
            (m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1]) / det
        })
    }))
}

/// Gauss–Jordan elimination with partial pivoting. Returns `None` when `m` is singular.
pub fn invert(m: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let mut a = *m;
    let mut inverse: [[f64; 4]; 4] =
        std::array::from_fn(|i| std::array::from_fn(|j| f64::from(u8::from(i == j))));
    for col in 0..4 {
        let pivot = (col..4).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        if a[pivot][col] == 0.0 {
            return None;
        }
        a.swap(col, pivot);
        inverse.swap(col, pivot);
        let scale = 1.0 / a[col][col];
        for j in 0..4 {
            a[col][j] *= scale;
            inverse[col][j] *= scale;
        }
        for row in (0..4).filter(|&r| r != col) {
            let f = a[row][col];
            for j in 0..4 {
                a[row][j] -= f * a[col][j];
                inverse[row][j] -= f * inverse[col][j];
            }
        }
    }
    Some(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_undoes_the_matrix() {
        let m = [
            [2.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, -1.0, 2.0],
            [0.5, 4.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ];
        let inverse = invert(&m).unwrap();
        for (i, row) in m.iter().enumerate() {
            for j in 0..4 {
                let v: f64 = row.iter().zip(&inverse).map(|(a, r)| a * r[j]).sum();
                assert!(
                    (v - f64::from(u8::from(i == j))).abs() < 1e-12,
                    "{i} {j}: {v}"
                );
            }
        }
        // Negative control: a repeated row has no inverse.
        assert!(invert(&[m[0], m[0], m[2], m[3]]).is_none());
    }
    use crate::store::{InstanceCategories, InstancePrimvars, Link, Store};

    fn unlinked() -> Links {
        Links::new(&Store::new())
    }

    fn dot(a: [f32; 4], b: [f64; 4]) -> f64 {
        (0..4).map(|i| a[i] as f64 * b[i]).sum()
    }

    fn translate(x: f64, y: f64, z: f64) -> [f64; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0,
        ]
    }

    /// As GfFrustum computes it, with a 90° field of view, near 1 and far 100.
    fn perspective() -> [f64; 16] {
        let (n, f) = (1.0, 100.0);
        let mut p = [0.0; 16];
        p[0] = 1.0;
        p[5] = 1.0;
        p[10] = -(f + n) / (f - n);
        p[11] = -1.0;
        p[14] = -2.0 * f * n / (f - n);
        p
    }

    fn clip(camera: &Camera, p: [f64; 3]) -> ([f64; 4], f64) {
        let p = [p[0], p[1], p[2], 1.0];
        let c: [f64; 4] = std::array::from_fn(|i| dot(camera.view_projection[i], p));
        (c, c[2] / c[3])
    }

    #[test]
    fn depth_is_reversed_and_converts_to_hydras() {
        // The camera sits at z = 5 looking down -z.
        let camera = Camera::new(&translate(0.0, 0.0, -5.0), &perspective()).unwrap();
        let near = clip(&camera, [0.0, 0.0, 4.0]).1;
        let far = clip(&camera, [0.0, 0.0, -95.0]).1;
        assert!(
            (near - 1.0).abs() < 1e-6 && far.abs() < 1e-6,
            "{near} {far}"
        );

        // The depth AOV, 1 - z/w, equals OpenGL's window depth (z_ndc + 1) / 2.
        let p = [0.3, -0.2, -2.0];
        let (_, depth) = clip(&camera, p);
        let m = multiply(&translate(0.0, 0.0, -5.0), &perspective());
        let gl: [f64; 4] =
            std::array::from_fn(|j| (0..3).map(|k| p[k] * m[k * 4 + j]).sum::<f64>() + m[12 + j]);
        let window = (gl[2] / gl[3] + 1.0) / 2.0;
        assert!((1.0 - depth - window).abs() < 1e-6, "{depth} {window}");
        // The control: the unconverted depth is far from Hydra's.
        assert!((depth - window).abs() > 0.01);
    }

    #[test]
    fn rows_are_not_flipped() {
        let camera = Camera::new(&translate(0.0, 0.0, -5.0), &perspective()).unwrap();
        // A point above the view axis has positive clip y, so the top row comes last.
        let (c, _) = clip(&camera, [0.0, 1.0, 0.0]);
        assert!(c[1] / c[3] > 0.0);
    }

    /// Where `p` lands in an image of `extent`, in pixels along the attachment's x and y.
    fn pixel(camera: &Camera, p: [f64; 3], extent: [u32; 2]) -> [f64; 2] {
        let (c, _) = clip(camera, p);
        [0, 1].map(|i| (c[i] / c[3] * 0.5 + 0.5) * f64::from(extent[i]))
    }

    /// A jittered pixel's centre sees what lies `jitter` away from it unjittered, at any depth.
    /// The control, the opposite jitter, misses by twice the jitter.
    #[test]
    fn jitter_moves_where_pixels_sample() {
        let camera = Camera::new(&translate(0.0, 0.0, -5.0), &perspective()).unwrap();
        let extent = [160, 90];
        let jitter = [0.375, -0.25];
        let jittered = camera.jittered(jitter, extent);
        let opposite = camera.jittered(jitter.map(|v| -v), extent);
        for p in [[0.3, -0.2, -2.0], [-4.0, 1.5, -40.0]] {
            let [x, y] = pixel(&camera, p, extent);
            let seen = [x - jitter[0], y - jitter[1]];
            let error = |c: &Camera| {
                let [a, b] = pixel(c, p, extent);
                ((a - seen[0]).powi(2) + (b - seen[1]).powi(2)).sqrt()
            };
            assert!(error(&jittered) < 1e-3, "{}", error(&jittered));
            assert!(error(&opposite) > 0.5);
            assert_eq!(clip(&jittered, p).1 as f32, clip(&camera, p).1 as f32);
        }
    }

    #[test]
    fn eye_is_the_camera_position_or_direction() {
        // A quarter turn about y, then placed at (2, 0, 0). World-to-view is the inverse.
        let world_to_view = [
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 1.0,
        ];
        let camera = Camera::new(&world_to_view, &perspective()).unwrap();
        assert_eq!(camera.eye, [2.0, 0.0, 0.0, 1.0]);
        let mut ortho = [0.0; 16];
        (ortho[0], ortho[5], ortho[10], ortho[15]) = (1.0, 1.0, -0.02, 1.0);
        let camera = Camera::new(&world_to_view, &ortho).unwrap();
        assert_eq!(camera.eye, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn rays_point_from_the_eye_through_each_point() {
        // A quarter turn about y, then placed at (2, 0, 0), so it looks down -x.
        let world_to_view = [
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 1.0,
        ];
        let camera = Camera::new(&world_to_view, &perspective()).unwrap();
        let jittered = camera.jittered([0.375, -0.25], [16, 9]);
        let along = |c: &Camera, at: [f64; 3]| {
            let rays = c.rays();
            let (clip, _) = clip(&camera, at);
            let [x, y] = [0, 1].map(|i| clip[i] / clip[3]);
            let d = rays.map(|row| row[0] * x + row[1] * y + row[2]);
            let to = [at[0] - 2.0, at[1], at[2]];
            let length = |v: [f64; 3]| v.iter().map(|c| c * c).sum::<f64>().sqrt();
            let cos = (0..3).map(|i| d[i] * to[i]).sum::<f64>() / (length(d) * length(to));
            1.0 - cos
        };
        for at in [[0.0, 0.0, 0.0], [-3.0, 1.0, 2.0], [-40.0, -9.0, -30.0]] {
            assert!(along(&camera, at) < 1e-9, "{at:?}");
            // Negative control: the jittered camera's ray through the same point misses it.
            assert!(along(&jittered, at) > 1e-6, "{at:?}");
        }
        let mut ortho = [0.0; 16];
        (ortho[0], ortho[5], ortho[10], ortho[15]) = (1.0, 1.0, -0.02, 1.0);
        let rays = Camera::new(&world_to_view, &ortho).unwrap().rays();
        let d = |x: f64, y: f64| rays.map(|row| row[0] * x + row[1] * y + row[2]);
        assert_eq!(d(0.0, 0.0), d(0.5, -1.0));
        assert!(d(0.0, 0.0)[0] < 0.0 && d(0.0, 0.0)[1] == 0.0 && d(0.0, 0.0)[2] == 0.0);
    }

    #[test]
    fn instance_moves_points_and_normals() {
        // Scale x by 2, rotate a quarter turn about z, then translate by (1, 2, 3).
        let m = [
            0.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0,
        ];
        let i = Instance::new(&m, false, Mask::default()).unwrap();
        let p = [1.0, 1.0, 0.0, 1.0];
        let world: [f64; 3] = std::array::from_fn(|r| dot(i.world[r], p));
        // As a row vector, p·M takes (1, 1, 0) to (0 - 1, 2 + 0, 0) + (1, 2, 3).
        assert_eq!(world, [0.0, 4.0, 3.0]);
        // The plane x + y = 0 has tangent (1, -1, 0) and normal (1, 1, 0).
        let n: [f64; 3] = std::array::from_fn(|r| dot(i.normal[r], [1.0, 1.0, 0.0, 0.0]));
        let t: [f64; 3] =
            std::array::from_fn(|c| (0..3).map(|k| [1.0, -1.0, 0.0][k] * m[k * 4 + c]).sum());
        assert!((0..3).map(|k| n[k] * t[k]).sum::<f64>().abs() < 1e-6);
        // The control: transforming the normal like a point breaks that.
        let wrong: [f64; 3] = std::array::from_fn(|r| dot(i.world[r], [1.0, 1.0, 0.0, 0.0]));
        assert!((0..3).map(|k| wrong[k] * t[k]).sum::<f64>().abs() > 0.1);
    }

    fn moved(m: &[f64; 16], p: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|c| (0..3).map(|k| p[k] * m[k * 4 + c]).sum::<f64>() + m[12 + c])
    }

    fn nested() -> (Instancer, Instancer) {
        let to = translate(10.0, 0.0, 0.0);
        // A quarter turn about x.
        let ti = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let outer = Instancer {
            path: "/outer".into(),
            parent: Some(None),
            transform: Some(to),
            primvars: Some(InstancePrimvars {
                translations: Some(vec![[0.0; 3], [0.0, 100.0, 0.0]]),
                ..Default::default()
            }),
            categories: Some(InstanceCategories::default()),
        };
        let inner = Instancer {
            path: "/inner".into(),
            parent: Some(Some(Instancing {
                instancer: "/outer".into(),
                indices: vec![0, 1],
            })),
            transform: Some(ti),
            primvars: Some(InstancePrimvars {
                transforms: Some(vec![translate(0.0, 0.0, 1.0)]),
                translations: Some(vec![[4.0, 5.0, 6.0]]),
                // Not unit length: (0.1, 0.5, -0.3, 0.8) / 0.99.
                rotations: Some(vec![[0.1, 0.5, -0.3, 0.8]]),
                scales: Some(vec![[1.0, 2.0, -3.0]]),
            }),
            categories: Some(InstanceCategories::default()),
        };
        (outer, inner)
    }

    #[test]
    fn instances_compose_as_gf_does() {
        let (outer, inner) = nested();
        let instancers = HashMap::from([("/outer", &outer), ("/inner", &inner)]);
        let at = Instancing {
            instancer: "/inner".into(),
            indices: vec![0],
        };
        let m: Vec<_> = (instances(&instancers, &unlinked(), &at)
            .unwrap()
            .into_iter())
        .map(|(m, _)| m)
        .collect();
        // (1, 1, 1) as Gf moves it through M S R T Ti, then O To for each outer instance O.
        let expected = [[11.0, -2.0, 9.0], [11.0, 98.0, 9.0]];
        assert_eq!(m.len(), 2);
        for (m, e) in m.iter().zip(expected) {
            let p = moved(m, [1.0, 1.0, 1.0]);
            assert!((0..3).all(|k| (p[k] - e[k]).abs() < 1e-5), "{p:?} {e:?}");
        }
        let x = moved(&rotate([0.1, 0.5, -0.3, 0.8]), [1.0, 0.0, 0.0]);
        let e = [31.0 / 99.0, -38.0 / 99.0, -86.0 / 99.0];
        assert!((0..3).all(|k| (x[k] - e[k]).abs() < 1e-6), "{x:?}");
        // The control: the real part read first, as (real, i, j, k).
        let wrong = moved(&rotate([0.8, 0.1, 0.5, -0.3]), [1.0, 0.0, 0.0]);
        assert!((0..3).any(|k| (wrong[k] - e[k]).abs() > 0.1));
    }

    /// Modelled on ALab's walls, which are native instances whose shadow-link collections are their
    /// own rather than their prototype's.
    #[test]
    fn instances_are_in_their_own_and_their_holders_collections() {
        let (mut outer, mut inner) = nested();
        let mut store = Store::new();
        for (path, light, shadow) in [("/sun", "", "sun_casters"), ("/key", "key_lit", "")] {
            let id = store.insert_light(path.into());
            let link = Link {
                light: light.into(),
                shadow: shadow.into(),
            };
            store.set_light_link(id, link).unwrap();
        }
        let links = Links::new(&store);
        let sun = links.categories(&["sun_casters".into()]);
        let key = links.categories(&["key_lit".into()]);
        let tokens = |t: &[&str]| t.iter().map(|&t| t.to_owned()).collect();
        outer.categories = Some(InstanceCategories::new(vec![
            tokens(&["sun_casters"]),
            tokens(&[]),
        ]));
        inner.categories = Some(InstanceCategories::new(vec![tokens(&["key_lit"])]));
        let at = Instancing {
            instancer: "/inner".into(),
            indices: vec![0],
        };
        let masks = |outer: &Instancer, inner: &Instancer| {
            let instancers = HashMap::from([("/outer", outer), ("/inner", inner)]);
            (instances(&instancers, &links, &at).unwrap().into_iter())
                .map(|(_, mask)| mask)
                .collect::<Vec<_>>()
        };
        assert_eq!(masks(&outer, &inner), [key | sun, key]);
        // Negative control: instances without categories of their own are in no collection.
        outer.categories = Some(InstanceCategories::default());
        inner.categories = Some(InstanceCategories::default());
        assert_eq!(masks(&outer, &inner), [Mask::default(); 2]);
    }

    #[test]
    fn broken_instancers_are_refused() {
        let (outer, mut inner) = nested();
        let at = |instancer: &str, index| Instancing {
            instancer: instancer.into(),
            indices: vec![index],
        };
        let instancers = HashMap::from([("/outer", &outer), ("/inner", &inner)]);
        let error = instances(&instancers, &unlinked(), &at("/inner", 1)).unwrap_err();
        assert!(
            error.contains("index 1 is not one of /inner's 1 transforms"),
            "{error}"
        );
        let error = instances(&instancers, &unlinked(), &at("/lost", 0)).unwrap_err();
        assert!(error.contains("no instancer /lost"), "{error}");
        inner.parent = Some(Some(at("/inner", 0)));
        let instancers = HashMap::from([("/inner", &inner)]);
        let error = instances(&instancers, &unlinked(), &at("/inner", 0)).unwrap_err();
        assert!(error.contains("nested in itself"), "{error}");
        let (outer, mut inner) = nested();
        let p = inner.primvars.as_mut().unwrap();
        (p.translations, p.rotations, p.scales) = (None, None, None);
        p.transforms = Some(vec![IDENTITY; 2]);
        inner.categories = Some(InstanceCategories::new(vec![vec!["a".into()]]));
        let instancers = HashMap::from([("/outer", &outer), ("/inner", &inner)]);
        let error = instances(&instancers, &unlinked(), &at("/inner", 1)).unwrap_err();
        assert!(
            error.contains("index 1 is not one of /inner's 1 category sets"),
            "{error}"
        );
    }

    #[test]
    fn mirroring_and_left_handed_reverse_the_winding() {
        let mut mirror = translate(0.0, 0.0, 0.0);
        mirror[0] = -1.0;
        let flip = |m: &[f64; 16], left| Instance::new(m, left, Mask::default()).unwrap().flags[0];
        assert_eq!(flip(&translate(0.0, 0.0, 0.0), false), 0);
        assert_eq!(flip(&mirror, false), 1);
        assert_eq!(flip(&translate(0.0, 0.0, 0.0), true), 1);
        assert_eq!(flip(&mirror, true), 0);
        // A transform that flattens the mesh draws nothing.
        let mut flat = mirror;
        flat[0] = 0.0;
        assert_eq!(Instance::new(&flat, false, Mask::default()), None);
    }
}
