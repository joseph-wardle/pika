//! The C ABI the Hydra delegate calls, declared in `hydra/pika.h`.
//!
//! Calls go one way: C++ calls Rust and Rust never calls back. Every entry
//! returns a status and catches panics, so a Rust bug cannot take down the
//! host. A panic while the renderer is locked poisons it, and every later
//! call on it fails, so the delegate stops calling and renders nothing rather
//! than a frame from half-updated state.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard};

use crate::renderer::{Aov, Format, Frame, Renderer};
use crate::store::{
    InstancePrimvars, Instancing, Interpolation, Normals, Refined, Scheme, Subset, Topology,
};

pub const OK: i32 = 0;
pub const ERROR: i32 = 1;
pub const PANIC: i32 = 2;

/// The opaque `pika_renderer` C++ holds. The mutex serializes calls from
/// Hydra's parallel sync and is what a panic poisons.
pub type Handle = Mutex<Renderer>;

/// `pika_subset`.
#[repr(C)]
pub struct CSubset {
    material: *const c_char,
    faces: *const i32,
    face_count: usize,
}

/// `pika_topology`.
#[repr(C)]
pub struct CTopology {
    scheme: *const c_char,
    face_vertex_counts: *const i32,
    face_count: usize,
    face_vertex_indices: *const i32,
    index_count: usize,
    left_handed: bool,
    holes: *const i32,
    hole_count: usize,
    subsets: *const CSubset,
    subset_count: usize,
    refined: *const CRefined,
}

/// `pika_refined`.
#[repr(C)]
pub struct CRefined {
    face_vertex_counts: *const i32,
    face_count: usize,
    face_vertex_indices: *const i32,
    index_count: usize,
    holes: *const i32,
    hole_count: usize,
    stencil_offsets: *const u32,
    point_count: usize,
    stencil_indices: *const i32,
    stencil_weights: *const f32,
    entry_count: usize,
}

/// `pika_instance_primvars`.
#[repr(C)]
pub struct CInstancePrimvars {
    transforms: *const [f64; 16],
    transform_count: usize,
    translations: *const [f32; 3],
    translation_count: usize,
    rotations: *const [f32; 4],
    rotation_count: usize,
    scales: *const [f32; 3],
    scale_count: usize,
}

/// `pika_aov`.
#[repr(C)]
pub struct CAov {
    format: *const c_char,
    pixels: *mut u8,
    len: usize,
}

/// `pika_stat`.
#[repr(C)]
pub struct CStat {
    key: [c_char; 48],
    value: f64,
}

/// `pika_frame`.
#[repr(C)]
pub struct CFrame {
    world_to_view: [f64; 16],
    projection: [f64; 16],
    width: u32,
    height: u32,
    window: [i32; 4],
    clear_color: [f32; 4],
    color: CAov,
    depth: CAov,
    normal: CAov,
    render_tags: *const *const c_char,
    render_tag_count: usize,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Runs `f` as one ABI call: its error or panic becomes a status and this
/// thread's last error.
fn call(f: impl FnOnce() -> Result<(), String>) -> i32 {
    let (status, message) = match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => return OK,
        Ok(Err(message)) => (ERROR, message),
        Err(payload) => {
            let what = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown payload".into());
            (
                PANIC,
                format!(
                    "Pika panicked ({what}); it renders nothing until the delegate is recreated"
                ),
            )
        }
    };
    // Messages are Rust strings, which may hold a NUL; C would cut them there.
    let message = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = message);
    status
}

/// Borrows `len` values at `ptr`. Empty Hydra arrays may pass a null pointer,
/// which `slice::from_raw_parts` does not allow.
///
/// # Safety
/// Unless `len` is 0, `ptr` must be valid for `len` reads for `'a`.
unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 {
        return &[];
    }
    // SAFETY: the caller guarantees `len` readable values at `ptr`.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// A primvar the delegate passes as null when it is absent.
///
/// # Safety
/// As for `slice`.
unsafe fn primvar<T: Clone>(ptr: *const T, len: usize) -> Option<Vec<T>> {
    // SAFETY: the caller guarantees `len` readable values at `ptr`.
    (!ptr.is_null()).then(|| unsafe { slice(ptr, len) }.to_vec())
}

/// A path the delegate passes as "" when there is none.
///
/// # Safety
/// As for `string`.
unsafe fn path(ptr: *const c_char) -> Result<Option<String>, String> {
    // SAFETY: the caller guarantees a NUL-terminated string.
    let path = unsafe { string(ptr)? };
    Ok((!path.is_empty()).then_some(path))
}

/// # Safety
/// `ptr` must be a NUL-terminated string.
unsafe fn string(ptr: *const c_char) -> Result<String, String> {
    // SAFETY: the caller guarantees a NUL-terminated string.
    let s = unsafe { CStr::from_ptr(ptr) };
    s.to_str()
        .map(str::to_owned)
        .map_err(|_| format!("{s:?} is not UTF-8"))
}

fn lock(handle: &Handle) -> Result<MutexGuard<'_, Renderer>, String> {
    handle
        .lock()
        .map_err(|_| "the renderer was poisoned by an earlier panic (reported above)".into())
}

/// The message for the last call on this thread that did not return `OK`.
/// Valid until the next failing call on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn pika_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Opens a Vulkan device and writes the new renderer to `*out`.
///
/// # Safety
/// `out` must be valid for one pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_renderer_new(out: *mut *mut Handle) -> i32 {
    call(|| {
        let renderer = Renderer::new()?;
        eprintln!("pika: rendering on {}", renderer.device_name);
        let handle = Box::into_raw(Box::new(Mutex::new(renderer)));
        // SAFETY: the caller guarantees `out` is writable.
        unsafe { out.write(handle) };
        Ok(())
    })
}

/// Prints the renderer's frame timings and destroys it, poisoned or not.
///
/// # Safety
/// `renderer` must come from `pika_renderer_new` and not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_renderer_free(renderer: *mut Handle) -> i32 {
    call(|| {
        // SAFETY: the caller passes ownership of a pointer from `Box::into_raw`.
        let handle = unsafe { Box::from_raw(renderer) };
        let renderer = handle.into_inner().unwrap_or_else(|e| e.into_inner());
        eprintln!("pika: {}", renderer.stats.summary());
        Ok(())
    })
}

/// Renders one frame into the render buffers it binds, and writes to
/// `*converged` whether they now hold this frame rather than the one before.
///
/// # Safety
/// `renderer` must be live, `frame` valid for reads, each bound AOV's
/// `pixels` valid for `len` byte writes for the duration of the call, and
/// `converged` valid for one write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_render(
    renderer: *const Handle,
    frame: *const CFrame,
    converged: *mut bool,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer and the frame.
        let (handle, f) = unsafe { (&*renderer, &*frame) };
        let aov = |a: &CAov, name: &str| -> Result<Option<Aov>, String> {
            if a.pixels.is_null() {
                return Ok(None);
            }
            // SAFETY: the caller guarantees the format string.
            let format = unsafe { string(a.format)? };
            Ok(Some(Aov {
                format: Format::from_name(&format).map_err(|e| format!("the {name} AOV: {e}"))?,
                // SAFETY: the caller guarantees `len` writable bytes.
                pixels: unsafe { std::slice::from_raw_parts_mut(a.pixels, a.len) },
            }))
        };
        let window = f.window.map(|v| u32::try_from(v).unwrap_or(u32::MAX));
        // SAFETY: the caller guarantees the frame's arrays and strings.
        let render_tags = unsafe { slice(f.render_tags, f.render_tag_count) }
            .iter()
            // SAFETY: as above.
            .map(|&t| unsafe { string(t) })
            .collect::<Result<_, _>>()?;
        let frame = Frame {
            world_to_view: f.world_to_view,
            projection: f.projection,
            width: f.width,
            height: f.height,
            window,
            clear_color: f.clear_color,
            color: aov(&f.color, "color")?,
            depth: aov(&f.depth, "depth")?,
            normal: aov(&f.normal, "normal")?,
            render_tags,
        };
        let done = lock(handle)?.render(frame)?;
        // SAFETY: the caller guarantees one write.
        unsafe { *converged = done };
        Ok(())
    })
}

/// Adds a mesh to the store and writes its id, never reused, to `*out`.
///
/// # Safety
/// `renderer` must be live, `path` a NUL-terminated string and `out` valid
/// for one write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_mesh_new(
    renderer: *const Handle,
    path: *const c_char,
    out: *mut u64,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer and the string.
        let (handle, path) = unsafe { (&*renderer, string(path)?) };
        let id = lock(handle)?.store.insert_mesh(path);
        // SAFETY: the caller guarantees `out` is writable.
        unsafe { out.write(id) };
        Ok(())
    })
}

/// # Safety
/// `renderer` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_mesh_free(renderer: *const Handle, id: u64) -> i32 {
    // SAFETY: the caller guarantees the renderer.
    call(|| lock(unsafe { &*renderer })?.store.remove_mesh(id))
}

/// # Safety
/// `renderer` must be live, and `topology` and the arrays it points to valid
/// for reads for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_mesh_topology(
    renderer: *const Handle,
    id: u64,
    topology: *const CTopology,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer and every array.
        let (handle, t) = unsafe { (&*renderer, &*topology) };
        let mut subsets = Vec::new();
        // SAFETY: as above.
        for s in unsafe { slice(t.subsets, t.subset_count) } {
            subsets.push(Subset {
                // SAFETY: as above.
                material: unsafe { string(s.material)? },
                faces: unsafe { slice(s.faces, s.face_count) }.to_vec(),
            });
        }
        let topology = Topology {
            // SAFETY: as above.
            scheme: Scheme::from_token(&unsafe { string(t.scheme)? })?,
            // SAFETY: as above.
            face_vertex_counts: unsafe { slice(t.face_vertex_counts, t.face_count) }.to_vec(),
            face_vertex_indices: unsafe { slice(t.face_vertex_indices, t.index_count) }.to_vec(),
            left_handed: t.left_handed,
            // SAFETY: as above.
            holes: unsafe { slice(t.holes, t.hole_count) }.to_vec(),
            subsets,
            // SAFETY: as above.
            refined: unsafe { t.refined.as_ref() }.map(|r| Refined {
                // SAFETY: as above.
                face_vertex_counts: unsafe { slice(r.face_vertex_counts, r.face_count) }.to_vec(),
                face_vertex_indices: unsafe { slice(r.face_vertex_indices, r.index_count) }
                    .to_vec(),
                holes: unsafe { slice(r.holes, r.hole_count) }.to_vec(),
                offsets: unsafe { slice(r.stencil_offsets, r.point_count + 1) }.to_vec(),
                indices: unsafe { slice(r.stencil_indices, r.entry_count) }.to_vec(),
                weights: unsafe { slice(r.stencil_weights, r.entry_count) }.to_vec(),
            }),
        };
        lock(handle)?.store.set_topology(id, topology)
    })
}

/// # Safety
/// `renderer` must be live, and `points` valid for `count` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_mesh_points(
    renderer: *const Handle,
    id: u64,
    points: *const [f32; 3],
    count: usize,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer and the array.
        let (handle, points) = unsafe { (&*renderer, slice(points, count).to_vec()) };
        lock(handle)?.store.set_points(id, points)
    })
}

/// # Safety
/// `renderer` must be live, `interpolation` a NUL-terminated string and
/// `normals` valid for `count` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_mesh_normals(
    renderer: *const Handle,
    id: u64,
    interpolation: *const c_char,
    normals: *const [f32; 3],
    count: usize,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer, the string and the array.
        let (handle, token) = unsafe { (&*renderer, string(interpolation)?) };
        // SAFETY: as above.
        let values = unsafe { slice(normals, count) }.to_vec();
        let normals = Normals {
            interpolation: Interpolation::from_token(&token)?,
            values,
        };
        lock(handle)?.store.set_normals(id, Some(normals))
    })
}

/// # Safety
/// `renderer` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_mesh_no_normals(renderer: *const Handle, id: u64) -> i32 {
    // SAFETY: the caller guarantees the renderer.
    call(|| lock(unsafe { &*renderer })?.store.set_normals(id, None))
}

/// # Safety
/// `renderer` must be live, and `matrix` valid for 16 reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_mesh_transform(
    renderer: *const Handle,
    id: u64,
    matrix: *const [f64; 16],
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer and the matrix.
        let (handle, matrix) = unsafe { (&*renderer, *matrix) };
        lock(handle)?.store.set_transform(id, matrix)
    })
}

/// # Safety
/// `renderer` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_mesh_visible(renderer: *const Handle, id: u64, visible: bool) -> i32 {
    // SAFETY: the caller guarantees the renderer.
    call(|| lock(unsafe { &*renderer })?.store.set_visible(id, visible))
}

/// # Safety
/// `renderer` must be live, and `tag` a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_mesh_render_tag(
    renderer: *const Handle,
    id: u64,
    tag: *const c_char,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer and the string.
        let (handle, tag) = unsafe { (&*renderer, string(tag)?) };
        lock(handle)?.store.set_render_tag(id, tag)
    })
}

/// Sets the material bound to the whole mesh ("" when none).
///
/// # Safety
/// `renderer` must be live, and `material` a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_mesh_material(
    renderer: *const Handle,
    id: u64,
    material: *const c_char,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer and the string.
        let (handle, material) = unsafe { (&*renderer, string(material)?) };
        lock(handle)?.store.set_material(id, material)
    })
}

/// Sets the instancer a mesh is a prototype of ("" when none) and the
/// instance indices it is drawn at.
///
/// # Safety
/// `renderer` must be live, `instancer` a NUL-terminated string and
/// `indices` valid for `count` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_mesh_instancer(
    renderer: *const Handle,
    id: u64,
    instancer: *const c_char,
    indices: *const i32,
    count: usize,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer, the string and the array.
        let (handle, instancer) = unsafe { (&*renderer, path(instancer)?) };
        let instancing = instancer.map(|instancer| Instancing {
            instancer,
            // SAFETY: as above.
            indices: unsafe { slice(indices, count) }.to_vec(),
        });
        lock(handle)?.store.set_instancing(id, instancing)
    })
}

/// Adds an instancer to the store and writes its id, never reused, to `*out`.
///
/// # Safety
/// `renderer` must be live, `path` a NUL-terminated string and `out` valid
/// for one write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_instancer_new(
    renderer: *const Handle,
    path: *const c_char,
    out: *mut u64,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer and the string.
        let (handle, path) = unsafe { (&*renderer, string(path)?) };
        let id = lock(handle)?.store.insert_instancer(path);
        // SAFETY: the caller guarantees `out` is writable.
        unsafe { out.write(id) };
        Ok(())
    })
}

/// # Safety
/// `renderer` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_instancer_free(renderer: *const Handle, id: u64) -> i32 {
    // SAFETY: the caller guarantees the renderer.
    call(|| lock(unsafe { &*renderer })?.store.remove_instancer(id))
}

/// Sets the instancer this one is nested in ("" when none) and the
/// instance indices of the parent it is repeated at.
///
/// # Safety
/// `renderer` must be live, `parent` a NUL-terminated string and `indices`
/// valid for `count` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_instancer_parent(
    renderer: *const Handle,
    id: u64,
    parent: *const c_char,
    indices: *const i32,
    count: usize,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer, the string and the array.
        let (handle, parent) = unsafe { (&*renderer, path(parent)?) };
        let parent = parent.map(|instancer| Instancing {
            instancer,
            // SAFETY: as above.
            indices: unsafe { slice(indices, count) }.to_vec(),
        });
        lock(handle)?.store.set_instancer_parent(id, parent)
    })
}

/// # Safety
/// `renderer` must be live, and `matrix` valid for 16 reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_instancer_transform(
    renderer: *const Handle,
    id: u64,
    matrix: *const [f64; 16],
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer and the matrix.
        let (handle, matrix) = unsafe { (&*renderer, *matrix) };
        lock(handle)?.store.set_instancer_transform(id, matrix)
    })
}

/// # Safety
/// `renderer` must be live, and `primvars` and the arrays it points to valid
/// for reads for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_instancer_primvars(
    renderer: *const Handle,
    id: u64,
    primvars: *const CInstancePrimvars,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer and every array.
        let (handle, p) = unsafe { (&*renderer, &*primvars) };
        // SAFETY: as above.
        let primvars = unsafe {
            InstancePrimvars {
                transforms: primvar(p.transforms, p.transform_count),
                translations: primvar(p.translations, p.translation_count),
                rotations: primvar(p.rotations, p.rotation_count),
                scales: primvar(p.scales, p.scale_count),
            }
        };
        lock(handle)?.store.set_instance_primvars(id, primvars)
    })
}

/// Adds basis curves to the store and writes their id, never reused, to `*out`.
///
/// # Safety
/// `renderer` must be live, `path` a NUL-terminated string and `out` valid
/// for one write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_curves_new(
    renderer: *const Handle,
    path: *const c_char,
    out: *mut u64,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer and the string.
        let (handle, path) = unsafe { (&*renderer, string(path)?) };
        let id = lock(handle)?.store.insert_curves(path);
        // SAFETY: the caller guarantees `out` is writable.
        unsafe { out.write(id) };
        Ok(())
    })
}

/// # Safety
/// `renderer` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_curves_free(renderer: *const Handle, id: u64) -> i32 {
    // SAFETY: the caller guarantees the renderer.
    call(|| lock(unsafe { &*renderer })?.store.remove_curves(id))
}

/// # Safety
/// `renderer` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_curves_pulled(renderer: *const Handle, id: u64, bytes: usize) -> i32 {
    // SAFETY: the caller guarantees the renderer.
    call(|| lock(unsafe { &*renderer })?.store.pulled_curves(id, bytes))
}

/// Adds a prim that is only counted and writes its id, never reused, to
/// `*out`.
///
/// # Safety
/// `renderer` must be live, `prim_type` and `path` NUL-terminated strings and
/// `out` valid for one write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_counted_new(
    renderer: *const Handle,
    prim_type: *const c_char,
    path: *const c_char,
    out: *mut u64,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer and the strings.
        let (handle, prim_type, path) = unsafe { (&*renderer, string(prim_type)?, string(path)?) };
        let id = lock(handle)?.store.insert_counted(prim_type, path);
        // SAFETY: the caller guarantees `out` is writable.
        unsafe { out.write(id) };
        Ok(())
    })
}

/// # Safety
/// `renderer` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_counted_free(renderer: *const Handle, id: u64) -> i32 {
    // SAFETY: the caller guarantees the renderer.
    call(|| lock(unsafe { &*renderer })?.store.remove_counted(id))
}

/// Marks the start of a Hydra sync, for its timing.
///
/// # Safety
/// `renderer` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_sync_begin(renderer: *const Handle) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer.
        lock(unsafe { &*renderer })?.begin_sync();
        Ok(())
    })
}

/// Ends a Hydra sync: records its stats and writes `PIKA_STORE_DUMP`.
///
/// # Safety
/// `renderer` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_sync_end(renderer: *const Handle) -> i32 {
    // SAFETY: the caller guarantees the renderer.
    call(|| lock(unsafe { &*renderer })?.end_sync())
}

/// Writes the stats the host shows, up to `capacity` of them, to `out`, and
/// how many to `*count`.
///
/// # Safety
/// `renderer` must be live, `out` valid for `capacity` writes and `count`
/// for one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_stats(
    renderer: *const Handle,
    out: *mut CStat,
    capacity: usize,
    count: *mut usize,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer.
        let pairs = lock(unsafe { &*renderer })?.stats.host();
        if pairs.len() > capacity {
            return Err(format!("{} stats do not fit in {capacity}", pairs.len()));
        }
        for (i, (key, value)) in pairs.iter().enumerate() {
            let mut stat = CStat {
                key: [0; 48],
                value: *value,
            };
            if key.len() >= stat.key.len() {
                return Err(format!("the stat key {key} is too long for pika_stat"));
            }
            for (c, b) in stat.key.iter_mut().zip(key.bytes()) {
                *c = b as c_char;
            }
            // SAFETY: `i` is below `capacity`, which the caller guarantees.
            unsafe { out.add(i).write(stat) };
        }
        // SAFETY: the caller guarantees one write.
        unsafe { count.write(pairs.len()) };
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    fn last_error() -> String {
        // SAFETY: the pointer is valid until the next failing call on this thread.
        unsafe { CStr::from_ptr(pika_last_error()) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn ok_leaves_no_error() {
        assert_eq!(call(|| Ok(())), OK);
    }

    #[test]
    fn error_becomes_status_and_message() {
        assert_eq!(call(|| Err("no device".into())), ERROR);
        assert_eq!(last_error(), "no device");
    }

    #[test]
    fn panic_becomes_status_and_message() {
        assert_eq!(call(|| panic!("index out of range")), PANIC);
        assert!(
            last_error().contains("index out of range"),
            "{}",
            last_error()
        );
    }

    #[test]
    fn panic_while_locked_poisons_later_calls() {
        let handle = Mutex::new(());
        let use_lock = || {
            let _guard = handle.lock().map_err(|_| "poisoned".to_string())?;
            Ok(())
        };
        assert_eq!(call(use_lock), OK);
        let panic_while_locked = || {
            let _guard = handle.lock().unwrap();
            panic!("bug");
        };
        assert_eq!(call(panic_while_locked), PANIC);
        // The control above passed with the same lock, so this fails only
        // because of the panic.
        assert_eq!(call(use_lock), ERROR);
        assert_eq!(last_error(), "poisoned");
    }
}
