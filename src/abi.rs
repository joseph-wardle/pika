//! The C ABI the Hydra delegate calls, declared in `hydra/pika.h`.
//!
//! Calls go one way: C++ calls Rust and Rust never calls back. Every entry
//! returns a status and catches panics, so a Rust bug cannot take down the
//! host. A panic while the renderer is locked poisons it, and every later
//! call on it fails, so the delegate renders nothing rather than a frame from
//! half-updated state.

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard};

use crate::renderer::{Format, Renderer};

pub const OK: i32 = 0;
pub const ERROR: i32 = 1;
pub const PANIC: i32 = 2;

/// The opaque `pika_renderer` C++ holds. The mutex serializes calls from
/// Hydra's parallel sync and is what a panic poisons.
pub type Handle = Mutex<Renderer>;

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
