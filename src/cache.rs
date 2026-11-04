//! Pika's cache directory, where generated shaders and their SPIR-V live
//! between runs: `PIKA_CACHE`, else `$XDG_CACHE_HOME/pika`, else
//! `~/.cache/pika`. Each entry is keyed by a hash of everything that
//! determines it, so nothing in it is ever stale, only unused.

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

/// Writes `bytes` to `path` through a temporary name beside it, so another
/// process reads the whole file or none.
pub fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = path.with_file_name(format!("{name}.{}", unique()));
    std::fs::write(&tmp, bytes)
        .and_then(|()| std::fs::rename(&tmp, path))
        .map_err(|e| format!("cannot write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_matches_the_reference_vectors() {
        assert_eq!(fnv1a(&[b""]), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(&[b"a"]), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(&[b"foo", b"bar"]), fnv1a(&[b"foobar"]));
        assert_ne!(fnv1a(&[b"foobar"]), fnv1a(&[b"foobaz"]));
    }
}
