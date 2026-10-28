//! The C ABI the Hydra delegate calls, declared in `hydra/pika.h`.
//!
//! Calls go one way: C++ calls Rust and Rust never calls back. Every entry
//! returns a status and catches panics, so a Rust bug cannot take down the
//! host. A panic while the renderer is locked poisons it, and every later
//! call on it fails, so the delegate renders nothing rather than a frame from
//! half-updated state.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard};

use crate::renderer::{Format, Renderer};
use crate::store::{Interpolation, Normals, Subset, Topology};

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
    face_vertex_counts: *const i32,
    face_count: usize,
    face_vertex_indices: *const i32,
    index_count: usize,
    left_handed: bool,
    subsets: *const CSubset,
    subset_count: usize,
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
        eprintln!("pika: {}", renderer.report());
        Ok(())
    })
}

/// Renders a `width` × `height` frame into `pixels`, a render buffer of
/// `format` (a `pika_format`) and `len` bytes.
///
/// # Safety
/// `renderer` must be live, and `pixels` valid for `len` byte writes for the
/// duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_render(
    renderer: *const Handle,
    format: u32,
    width: u32,
    height: u32,
    pixels: *mut u8,
    len: usize,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees both pointers for this call.
        let (handle, pixels) = unsafe { (&*renderer, std::slice::from_raw_parts_mut(pixels, len)) };
        lock(handle)?.render(Format::from_abi(format)?, width, height, pixels)
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
            face_vertex_counts: unsafe { slice(t.face_vertex_counts, t.face_count) }.to_vec(),
            face_vertex_indices: unsafe { slice(t.face_vertex_indices, t.index_count) }.to_vec(),
            left_handed: t.left_handed,
            subsets,
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
/// `renderer` must be live, and `normals` valid for `count` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pika_mesh_normals(
    renderer: *const Handle,
    id: u64,
    interpolation: u32,
    normals: *const [f32; 3],
    count: usize,
) -> i32 {
    call(|| {
        // SAFETY: the caller guarantees the renderer and the array.
        let (handle, values) = unsafe { (&*renderer, slice(normals, count).to_vec()) };
        let normals = Normals {
            interpolation: Interpolation::from_abi(interpolation)?,
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
