//! Pika's cache directory, where generated shaders and their SPIR-V live
//! between runs: `PIKA_CACHE`, else `$XDG_CACHE_HOME/pika`, else
//! `~/.cache/pika`. Each entry is keyed by a hash of everything that
//! determines it, so nothing in it is ever stale, only unused; one a crash
//! left unreadable is rebuilt as if it were absent (`discard`).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub fn root() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("PIKA_CACHE") {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(Path::new(&path).join("pika"));
    }
    let home = std::env::var_os("HOME").ok_or("neither PIKA_CACHE nor HOME is set")?;
    Ok(Path::new(&home).join(".cache/pika"))
}

/// Whether `path`, which exists, lies on NFS by `mounts`, the contents of
/// `/proc/self/mounts`: the render thread reads textures' levels from the
/// cache as they upload, which a network file system stalls.
pub fn on_nfs(mounts: &str, path: &Path) -> bool {
    let Ok(path) = std::fs::canonicalize(path) else {
        return false;
    };
    // The deepest mount holding `path`, the last of equals: an automounter
    // lists its trigger before the file system mounted over it.
    let mut kind = "";
    let mut depth = 0;
    for line in mounts.lines() {
        let mut fields = line.split(' ').skip(1);
        let (Some(dir), Some(fs)) = (fields.next(), fields.next()) else {
            continue;
        };
        if path.starts_with(dir) && dir.len() >= depth {
            (kind, depth) = (fs, dir.len());
        }
    }
    kind.starts_with("nfs")
}

/// FNV-1a over `parts` in order. It keys the cache, so it must not change
/// with the Rust version as `DefaultHasher` may.
pub fn fnv1a(parts: &[&[u8]]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for &byte in parts.iter().copied().flatten() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// A temporary name's suffix no other process or thread writing the cache
/// uses, so an entry is written whole under it and renamed into place.
pub fn unique() -> String {
    static COUNT: AtomicU64 = AtomicU64::new(0);
    format!(
        "tmp{}.{}",
        std::process::id(),
        COUNT.fetch_add(1, Ordering::Relaxed)
    )
}

/// Removes the entry `path`, a file or a directory, that cannot be read
/// (`why`), so its owner rebuilds it as if it were absent.
pub fn discard(path: &Path, why: &str) -> Result<(), String> {
    eprintln!(
        "pika: warning: the cache entry {} is unreadable ({why}); rebuilding it",
        path.display()
    );
    let removed = if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    };
    match removed {
        // Another process discarded it meanwhile.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        other => other.map_err(|e| format!("cannot remove {}: {e}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfs_is_the_deepest_mount() {
        let mounts = "/dev/root / xfs rw 0 0\n\
                      auto /proc autofs rw 0 0\n\
                      server:/proc /proc nfs4 rw 0 0\n\
                      tmpfs /proc/sys tmpfs rw 0 0\n";
        let proc = Path::new("/proc/self");
        assert!(on_nfs(mounts, proc), "the file system over the automounter");
        assert!(!on_nfs(mounts, Path::new("/")), "the root above it");
        assert!(!on_nfs(mounts, Path::new("/proc/sys")), "a mount within it");
        assert!(!on_nfs(mounts, Path::new("/proc/nowhere")), "a path absent");
        let local = mounts.replace("nfs4", "xfs");
        assert!(!on_nfs(&local, proc), "local under the automounter");
    }

    #[test]
    fn fnv1a_matches_the_reference_vectors() {
        assert_eq!(fnv1a(&[b""]), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(&[b"a"]), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(&[b"foo", b"bar"]), fnv1a(&[b"foobar"]));
        assert_ne!(fnv1a(&[b"foobar"]), fnv1a(&[b"foobaz"]));
    }
}
