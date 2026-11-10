//! What detail of each texture the view needs. The estimate is a mean over a mesh's faces and
//! its box's nearest point, so uneven UVs or a mesh reaching past the camera read finer.

use crate::store::{Interpolation, Mesh, Stencils};

/// Rendered frames a texture keeps its finest needed level: more than a looping shot, so a
/// loop's second pass needs nothing its first did not. A paused view ages nothing.
pub const WINDOW: u64 = 240;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Footprint {
    /// Infinite when none could be estimated.
    pub uv_per_pixel: f32,
    /// One of them has no texcoord that varies over its faces.
    pub unestimated: bool,
}

impl Footprint {
    pub const NONE: Footprint = Footprint {
        uv_per_pixel: f32::INFINITY,
        unestimated: false,
    };

    pub fn add(&mut self, other: Footprint) {
        self.uv_per_pixel = self.uv_per_pixel.min(other.uv_per_pixel);
        self.unestimated |= other.unestimated;
    }
}

/// Texture units per unit of the mesh's own space, over its drawn (refined) faces. `None`
/// when it has no texcoord, one constant over a face, or no area.
pub fn density(mesh: &Mesh) -> Option<f32> {
    let (Some(topology), Some(points), Some(Some(texcoord))) =
        (&mesh.topology, &mesh.points, &mesh.texcoord)
    else {
        return None;
    };
    let (area, uv_area) = match (&topology.refined, &texcoord.refined) {
        (Some(r), Some(t)) => {
            let points = refine(&r.stencils, points)?;
            let values = refine(&t.stencils, &texcoord.values)?;
            let uv_index = |corner: usize, _| t.face_vertex_indices.get(corner).copied();
            let faces = (&r.face_vertex_counts, &r.face_vertex_indices);
            areas(faces, &points, &values, uv_index)?
        }
        (None, None) => {
            let faces = (&topology.face_vertex_counts, &topology.face_vertex_indices);
            let values = &texcoord.values;
            match texcoord.interpolation {
                Interpolation::Vertex | Interpolation::Varying => {
                    areas(faces, points, values, |_, point| Some(point))?
                }
                Interpolation::FaceVarying => areas(faces, points, values, |corner, _| {
                    i32::try_from(corner).ok()
                })?,
                Interpolation::Constant | Interpolation::Uniform => return None,
            }
        }
        _ => return None,
    };
    let density = (uv_area / area).sqrt();
    (density.is_finite() && density > 0.0).then_some(density as f32)
}

/// Summed area of `faces` in space and in texture space, triangulated as fans.
fn areas<const N: usize>(
    (counts, indices): (&Vec<i32>, &Vec<i32>),
    points: &[[f32; 3]],
    values: &[[f32; N]],
    uv_index: impl Fn(usize, i32) -> Option<i32>,
) -> Option<(f64, f64)> {
    let (mut area, mut uv_area) = (0.0f64, 0.0f64);
    let mut corner = 0;
    for &count in counts {
        let count = usize::try_from(count).ok()?;
        let at = |k: usize| -> Option<([f64; 3], [f64; 2])> {
            let point = *indices.get(corner + k)?;
            let p = *points.get(usize::try_from(point).ok()?)?;
            let v = *values.get(usize::try_from(uv_index(corner + k, point)?).ok()?)?;
            Some((p.map(f64::from), [f64::from(v[0]), f64::from(v[1])]))
        };
        for k in 1..count.saturating_sub(1) {
            let ((p0, t0), (p1, t1), (p2, t2)) = (at(0)?, at(k)?, at(k + 1)?);
            area += cross_length(sub(p1, p0), sub(p2, p0)) / 2.0;
            let (a, b) = (
                [t1[0] - t0[0], t1[1] - t0[1]],
                [t2[0] - t0[0], t2[1] - t0[1]],
            );
            uv_area += (a[0] * b[1] - a[1] * b[0]).abs() / 2.0;
        }
        corner += count;
    }
    Some((area, uv_area))
}

fn refine<const N: usize>(stencils: &Stencils, cage: &[[f32; N]]) -> Option<Vec<[f32; N]>> {
    (stencils.offsets.windows(2))
        .map(|w| {
            let mut value = [0.0; N];
            for e in w[0] as usize..w[1] as usize {
                let from = cage.get(usize::try_from(*stencils.indices.get(e)?).ok()?)?;
                let weight = *stencils.weights.get(e)?;
                for (v, f) in value.iter_mut().zip(from) {
                    *v += weight * f;
                }
            }
            Some(value)
        })
        .collect()
}

/// `world` and `view_projection` are rows of column-vector matrices, clip space Vulkan's with
/// reversed z. `None` when the box is outside the view.
pub fn uv_per_pixel(
    view_projection: &[[f32; 4]; 4],
    size: [u32; 2],
    world: &[[f32; 4]; 3],
    bounds: &[f32; 6],
    density: f32,
) -> Option<f32> {
    let mut outside = [true; 7];
    let mut nearest = f32::INFINITY;
    for i in 0..8 {
        let p: [f32; 3] = std::array::from_fn(|k| bounds[k + 3 * ((i >> k) & 1)]);
        let w: [f32; 3] = std::array::from_fn(|r| dot3(world[r], p) + world[r][3]);
        let [x, y, z, d] = view_projection.map(|row| dot3(row, w) + row[3]);
        // Past left, right, bottom, top, nearer than near, past far, behind the camera.
        let past = [x < -d, x > d, y < -d, y > d, z > d, z < 0.0, d <= 0.0];
        for (o, p) in outside.iter_mut().zip(past) {
            *o &= p;
        }
        nearest = nearest.min(d);
    }
    if outside.iter().any(|&o| o) {
        return None;
    }
    // A box reaching past the camera's plane is as near as can be.
    let depth = nearest.max(f32::MIN_POSITIVE);
    let length = |r: [f32; 4]| dot3(r, [r[0], r[1], r[2]]).sqrt();
    let [vx, vy] = [length(view_projection[0]), length(view_projection[1])];
    let pixels_per_unit = (size[0] as f32 * vx).max(size[1] as f32 * vy) / (2.0 * depth);
    let scale = det3(world).abs().cbrt();
    Some(density / scale / pixels_per_unit)
}

/// The floor of a trilinear sampler's level of detail; 0 when magnified.
pub fn level(side: u32, uv_per_pixel: f32) -> usize {
    let texels_per_pixel = side as f32 * uv_per_pixel;
    match texels_per_pixel > 1.0 {
        true => texels_per_pixel.log2().floor() as usize,
        false => 0,
    }
}

/// `seen` holds the last frame each level was the finest needed, 0 for never.
pub fn finest(seen: &[u64], frame: u64) -> Option<usize> {
    seen.iter()
        .position(|&f| f > 0 && frame.saturating_sub(f) < WINDOW)
}

fn dot3(a: [f32; 4], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn det3(m: &[[f32; 4]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross_length(a: [f64; 3], b: [f64; 3]) -> f64 {
    let c = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Refined, RefinedTexcoord, Scheme, Texcoord, Topology};
    use crate::transform::Camera;

    /// At the origin looking down -z, 90° each way, near 0.1 and far 100.
    fn camera() -> Camera {
        let (n, f) = (0.1, 100.0);
        #[rustfmt::skip]
        let projection = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, (f + n) / (n - f), -1.0,
            0.0, 0.0, 2.0 * f * n / (n - f), 0.0,
        ];
        #[rustfmt::skip]
        let identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        Camera::new(&identity, &projection).unwrap()
    }

    fn quad(uv: f32, interpolation: Interpolation) -> Mesh {
        let values = vec![[0.0, 0.0], [uv, 0.0], [uv, uv], [0.0, uv]];
        Mesh {
            path: "/quad".into(),
            topology: Some(Topology {
                scheme: Scheme::None,
                face_vertex_counts: vec![4],
                face_vertex_indices: vec![0, 1, 2, 3],
                left_handed: false,
                holes: vec![],
                subsets: vec![],
                refined: None,
            }),
            points: Some(vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            normals: Some(None),
            texcoord: Some(Some(Texcoord {
                name: "st".into(),
                interpolation,
                values,
                refined: None,
            })),
            transform: None,
            visible: Some(true),
            instancing: Some(None),
            material: None,
            render_tag: None,
            categories: None,
        }
    }

    fn placed(scale: f32, at: [f32; 3]) -> [[f32; 4]; 3] {
        [
            [scale, 0.0, 0.0, at[0]],
            [0.0, scale, 0.0, at[1]],
            [0.0, 0.0, scale, at[2]],
        ]
    }

    const QUAD: [f32; 6] = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0];

    #[test]
    fn density_is_texture_units_per_unit_of_surface() {
        assert_eq!(density(&quad(1.0, Interpolation::Vertex)), Some(1.0));
        assert_eq!(density(&quad(4.0, Interpolation::FaceVarying)), Some(4.0));
        // A texcoord that does not vary over a face cannot be estimated.
        assert_eq!(density(&quad(1.0, Interpolation::Uniform)), None);
        let mut flat = quad(1.0, Interpolation::Vertex);
        flat.texcoord = Some(Some(Texcoord {
            values: vec![[0.5, 0.5]; 4],
            ..quad(1.0, Interpolation::Vertex).texcoord.unwrap().unwrap()
        }));
        assert_eq!(density(&flat), None);
    }

    #[test]
    fn a_refined_mesh_is_measured_over_its_refined_faces() {
        let stencils = |indices: Vec<i32>, weights: Vec<f32>| Stencils {
            offsets: (0..=4).map(|i| 2 * i).collect(),
            indices,
            weights,
        };
        let mut mesh = quad(1.0, Interpolation::FaceVarying);
        // One refined face, a quarter of the cage's area over the same texture area.
        mesh.topology.as_mut().unwrap().refined = Some(Refined {
            face_vertex_counts: vec![4],
            face_vertex_indices: vec![0, 1, 2, 3],
            holes: vec![],
            stencils: stencils(vec![0, 0, 0, 1, 0, 2, 0, 3], vec![0.5; 8]),
        });
        let texcoord = mesh.texcoord.as_mut().unwrap().as_mut().unwrap();
        texcoord.refined = Some(RefinedTexcoord {
            face_vertex_indices: vec![0, 1, 2, 3],
            stencils: stencils(vec![0, 0, 1, 1, 2, 2, 3, 3], vec![0.5; 8]),
        });
        assert_eq!(density(&mesh), Some(2.0));
        // The control: a texcoord not refined with its mesh is not drawn, nor estimated.
        let texcoord = mesh.texcoord.as_mut().unwrap().as_mut().unwrap();
        texcoord.refined = None;
        assert_eq!(density(&mesh), None);
    }

    #[test]
    fn near_and_far_quads_need_the_levels_worked_by_hand() {
        let vp = camera().view_projection;
        let size = [1000, 1000];
        // At depth d, 90° across 1000 pixels: 500/d pixels per unit; a
        // 1024-texel texture over one unit is 1024·d/500 texels a pixel.
        let need = |scale: f32, depth: f32| {
            let uv = uv_per_pixel(&vp, size, &placed(scale, [0.0, 0.0, -depth]), &QUAD, 1.0);
            uv.map(|uv| level(1024, uv))
        };
        assert_eq!(need(1.0, 0.5), Some(0));
        // 20.48: between 16 and 32, level 4.
        assert_eq!(need(1.0, 10.0), Some(4));
        // Scaled twice as large, a unit of texture spans two: 10.24, level 3.
        assert_eq!(need(2.0, 10.0), Some(3));
        assert_eq!(need(1.0, 0.2), Some(0));
        // Behind the camera, and off to its side, nothing is needed.
        assert_eq!(need(1.0, -5.0), None);
        let aside = uv_per_pixel(&vp, size, &placed(1.0, [50.0, 0.0, -5.0]), &QUAD, 1.0);
        assert_eq!(aside, None);
    }

    #[test]
    fn a_shared_texture_needs_its_finest_reader() {
        let vp = camera().view_projection;
        let mut footprint = Footprint::NONE;
        for depth in [10.0, 0.5] {
            let at = placed(1.0, [0.0, 0.0, -depth]);
            let uv = uv_per_pixel(&vp, [1000, 1000], &at, &QUAD, 1.0).unwrap();
            footprint.add(Footprint {
                uv_per_pixel: uv,
                unestimated: false,
            });
        }
        assert_eq!(level(1024, footprint.uv_per_pixel), 0);
    }

    #[test]
    fn a_need_is_kept_for_the_window() {
        // Level 1 was needed at frame 1, level 3 every frame since.
        let mut seen = vec![0, 1, 0, 0];
        for frame in 2..=WINDOW + 1 {
            seen[3] = frame;
            let expected = if frame < 1 + WINDOW { 1 } else { 3 };
            assert_eq!(finest(&seen, frame), Some(expected), "frame {frame}");
        }
        assert_eq!(finest(&[0, 0, 0], 5), None);
    }
}
