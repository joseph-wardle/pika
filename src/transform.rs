//! Hydra's transforms, converted for `shaders/mesh.slang`. `GfMatrix4d` multiplies row vectors
//! and the shader column vectors, so each matrix is sent transposed, converted in f64.

use std::collections::HashMap;

use crate::link::{Links, Mask};
use crate::store::{Instancer, Instancing};

/// Laid out as the shader's 7 `float4`s per instance.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Instance {
    world: [[f32; 4]; 3],
    normal: [[f32; 4]; 3],
    /// Bit 0 of [0]: front faces wind clockwise. [1] and [2]: the light- and shadow-link
    /// collection bits.
    flags: [u32; 4],
}

impl Instance {
    /// Rows of the column-vector matrix.
    pub fn world(&self) -> &[[f32; 4]; 3] {
        &self.world
    }

    /// The shadow-link collections the mesh is in.
    pub fn shadow_link(&self) -> u32 {
        self.flags[2]
    }

    /// `None` when the transform flattens the mesh, which then covers no pixels.
    pub fn new(matrix: &[f64; 16], left_handed: bool, link: Mask) -> Option<Self> {
        let linear = linear(matrix);
        let det = det3(&linear);
        let inverse = inverse3(&linear, det)?;
        let world = std::array::from_fn(|j| column(matrix, j).map(|v| v as f32));
        // Normals take the inverse transpose: in column-vector form, the inverse read by rows.
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
    /// Rows of the column-vector matrix.
    pub view_projection: [[f32; 4]; 4],
    /// The camera position (w = 1), or for an orthographic camera the
    /// direction toward it (w = 0).
    pub eye: [f32; 4],
}

impl Camera {
    /// Hydra's clip z/w runs -1 (near) to 1 (far); Pika's runs 1 to 0, reversed for precision far
    /// away. y is not flipped, so the first image row is the bottom one, as in Hydra's buffers.
    pub fn new(world_to_view: &[f64; 16], projection: &[f64; 16]) -> Result<Self, String> {
        let m = multiply(world_to_view, projection);
        let [x, y, z, w] = std::array::from_fn(|j| column(&m, j));
        // z' = (w - z) / 2, built into the matrix so no vertex subtracts nearly equal values.
        let z = std::array::from_fn(|i| 0.5 * (w[i] - z[i]));
        let view_projection = [x, y, z, w].map(|row| row.map(|v| v as f32));

        let linear = linear(world_to_view);
        let inverse = inverse3(&linear, det3(&linear))
            .ok_or("the camera's world-to-view matrix is singular")?;
        // Row-vector algebra: the view-to-world matrix's last row is the eye,
        // and its z row the direction toward the viewer.
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
}

/// The instances a prototype is drawn at: each one's transform, composed as Hydra does (per
/// instancer level, each instance's primvars then the instancer's transform, outer levels
/// repeating per instance), and the link collections its instances at every level are in.
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
        // Outer instances first, as Hydra orders them.
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
            // Row vectors: the leftmost matrix applies first.
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

/// From (i, j, k, real), as Gf builds it. Point instancers author half-float quaternions, not
/// quite unit length; dividing by the squared length keeps them rigid.
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

/// Row-major `a b`: `a` applies first to a row vector.
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

/// Gauss–Jordan with partial pivoting; `None` when `m` is singular.
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

    /// As GfFrustum computes it: a 90° field of view, near 1 and far 100.
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

        // The depth AOV, 1 - z/w, is OpenGL's window depth (z_ndc + 1) / 2.
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
        // A point above the view axis has positive clip y: the top row is last.
        let (c, _) = clip(&camera, [0.0, 1.0, 0.0]);
        assert!(c[1] / c[3] > 0.0);
    }

    #[test]
    fn eye_is_the_camera_position_or_direction() {
        // A quarter turn about y, then placed at (2, 0, 0); world-to-view is the inverse.
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
    fn instance_moves_points_and_normals() {
        // Scale x by 2, rotate a quarter turn about z, translate by (1, 2, 3).
        let m = [
            0.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0,
        ];
        let i = Instance::new(&m, false, Mask::default()).unwrap();
        let p = [1.0, 1.0, 0.0, 1.0];
        let world: [f64; 3] = std::array::from_fn(|r| dot(i.world[r], p));
        // Row vector p·M: (1, 1, 0) goes to (0 - 1, 2 + 0, 0) + (1, 2, 3).
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

    /// ALab's walls: native instances whose shadow-link collections are their own, not their
    /// prototype's.
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
