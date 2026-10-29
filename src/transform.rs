//! Hydra's transforms, converted into what `shaders/mesh.slang` reads.
//!
//! Hydra's `GfMatrix4d` is row-major and multiplies row vectors, with the
//! translation in the last row. The shader multiplies column vectors, so each
//! matrix it reads is the transpose, sent as rows of f32. The conversions are
//! done in f64, so the f32 rounding happens once.

/// One mesh instance: its world matrix and normal matrix, and whether its
/// winding is reversed. Laid out as the shader's 7 `float4`s per instance.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Instance {
    world: [[f32; 4]; 3],
    normal: [[f32; 4]; 3],
    /// Bit 0: front faces wind clockwise, as seen from in front.
    flags: [u32; 4],
}

impl Instance {
    /// The instance at world transform `matrix` of a mesh with the given
    /// orientation. `None` when the transform flattens the mesh, which then
    /// covers no pixels.
    pub fn new(matrix: &[f64; 16], left_handed: bool) -> Option<Self> {
        let linear = linear(matrix);
        let det = det3(&linear);
        let inverse = inverse3(&linear, det)?;
        let world = std::array::from_fn(|j| column(matrix, j).map(|v| v as f32));
        // Normals transform by the inverse transpose of the linear part. In
        // column-vector form that is the inverse itself, read by rows.
        let normal = inverse.map(|[a, b, c]| [a as f32, b as f32, c as f32, 0.0]);
        // A mirroring transform reverses the winding, as a left-handed mesh does.
        let flip = left_handed != (det < 0.0);
        Some(Self {
            world,
            normal,
            flags: [flip as u32, 0, 0, 0],
        })
    }
}

/// The camera: world to Vulkan clip space, and where the viewer is.
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
    /// From the render pass's world-to-view and projection matrices.
    /// Hydra's projection is OpenGL's: clip z/w runs from -1 at the near
    /// plane to 1 at the far plane. Pika's runs from 1 to 0 (reversed), which
    /// keeps float depth precise far away. Its y is not flipped, so the first
    /// image row is the bottom one, as in Hydra's render buffers.
    pub fn new(world_to_view: &[f64; 16], projection: &[f64; 16]) -> Result<Self, String> {
        let m = multiply(world_to_view, projection);
        let [x, y, z, w] = std::array::from_fn(|j| column(&m, j));
        // z' = (w - z) / 2, built into the matrix so no vertex subtracts
        // nearly equal values.
        let z = std::array::from_fn(|i| 0.5 * (w[i] - z[i]));
        let view_projection = [x, y, z, w].map(|row| row.map(|v| v as f32));

        let linear = linear(world_to_view);
        let inverse = inverse3(&linear, det3(&linear))
            .ok_or("the camera's world-to-view matrix is singular")?;
        // Row-vector algebra: the view-to-world matrix's last row is the eye,
        // and its z row the direction toward the viewer.
        let t = [world_to_view[12], world_to_view[13], world_to_view[14]];
        let eye = if projection[15] == 0.0 {
            let p: [f64; 3] =
                std::array::from_fn(|c| -(0..3).map(|r| t[r] * inverse[r][c]).sum::<f64>());
            [p[0], p[1], p[2], 1.0]
        } else {
            let [a, b, c] = inverse[2];
            let length = (a * a + b * b + c * c).sqrt();
            [a / length, b / length, c / length, 0.0]
        };
        Ok(Self {
            view_projection,
            eye: eye.map(|v| v as f32),
        })
    }
}

/// Column `j` of a row-major 4×4 matrix: row `j` of its transpose.
fn column(m: &[f64; 16], j: usize) -> [f64; 4] {
    [m[j], m[4 + j], m[8 + j], m[12 + j]]
}

/// The upper-left 3×3 of a row-major 4×4 matrix.
fn linear(m: &[f64; 16]) -> [[f64; 3]; 3] {
    std::array::from_fn(|r| std::array::from_fn(|c| m[r * 4 + c]))
}

fn multiply(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    std::array::from_fn(|i| (0..4).map(|k| a[i / 4 * 4 + k] * b[k * 4 + i % 4]).sum())
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// The inverse of `m`, whose determinant is `det`; `None` when it is singular.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: [f32; 4], b: [f64; 4]) -> f64 {
        (0..4).map(|i| a[i] as f64 * b[i]).sum()
    }

    /// A row-major translation.
    fn translate(x: f64, y: f64, z: f64) -> [f64; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0,
        ]
    }

    /// OpenGL's perspective projection, as GfFrustum computes it: a 90°
    /// field of view, near 1 and far 100.
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

    /// Clip-space position of a world point, and its depth z/w.
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
        // Rotated a quarter turn about y, then placed at (2, 0, 0): in
        // row-vector form, view-to-world is R then T, and world-to-view the inverse.
        let world_to_view = [
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 1.0,
        ];
        let camera = Camera::new(&world_to_view, &perspective()).unwrap();
        assert_eq!(camera.eye, [2.0, 0.0, 0.0, 1.0]);
        let mut ortho = [0.0; 16];
        (ortho[0], ortho[5], ortho[10], ortho[15]) = (1.0, 1.0, -0.02, 1.0);
        let camera = Camera::new(&world_to_view, &ortho).unwrap();
        // The camera's +z, toward the viewer, points along world +x.
        assert_eq!(camera.eye, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn instance_moves_points_and_normals() {
        // Scale x by 2, rotate a quarter turn about z, translate by (1, 2, 3).
        let m = [
            0.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0,
        ];
        let i = Instance::new(&m, false).unwrap();
        let p = [1.0, 1.0, 0.0, 1.0];
        let world: [f64; 3] = std::array::from_fn(|r| dot(i.world[r], p));
        // Row vector p·M: (1, 1, 0) goes to (0 - 1, 2 + 0, 0) + (1, 2, 3).
        assert_eq!(world, [0.0, 4.0, 3.0]);
        // A normal stays perpendicular to the surface: the plane x + y = 0
        // has tangent (1, -1, 0) and normal (1, 1, 0).
        let n: [f64; 3] = std::array::from_fn(|r| dot(i.normal[r], [1.0, 1.0, 0.0, 0.0]));
        let t: [f64; 3] =
            std::array::from_fn(|c| (0..3).map(|k| [1.0, -1.0, 0.0][k] * m[k * 4 + c]).sum());
        assert!((0..3).map(|k| n[k] * t[k]).sum::<f64>().abs() < 1e-6);
        // The control: transforming the normal like a point breaks that.
        let wrong: [f64; 3] = std::array::from_fn(|r| dot(i.world[r], [1.0, 1.0, 0.0, 0.0]));
        assert!((0..3).map(|k| wrong[k] * t[k]).sum::<f64>().abs() > 0.1);
    }

    #[test]
    fn mirroring_and_left_handed_reverse_the_winding() {
        let mut mirror = translate(0.0, 0.0, 0.0);
        mirror[0] = -1.0;
        let flip = |m: &[f64; 16], left| Instance::new(m, left).unwrap().flags[0];
        assert_eq!(flip(&translate(0.0, 0.0, 0.0), false), 0);
        assert_eq!(flip(&mirror, false), 1);
        assert_eq!(flip(&translate(0.0, 0.0, 0.0), true), 1);
        assert_eq!(flip(&mirror, true), 0);
        // A transform that flattens the mesh draws nothing.
        let mut flat = mirror;
        flat[0] = 0.0;
        assert_eq!(Instance::new(&flat, false), None);
    }
}
