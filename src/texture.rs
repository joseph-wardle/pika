//! Baked textures: the file `bake texture` writes for one image under the
//! cache (`bake/texture.cpp`), as the renderer reads it. A 24-byte header,
//! `pika` and five little-endian u32 (width, height, layers, levels, the
//! VkFormat), then every mip level in order, each layer within it, so an
//! upload is one copy per level.

use std::ops::Range;
use std::path::Path;

use ash::vk;

pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub format: vk::Format,
    /// Each level's bytes within `file`, all its layers, finest first.
    pub levels: Vec<Range<usize>>,
    pub file: Vec<u8>,
}

pub fn read(path: &Path) -> Result<Texture, String> {
    let file = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse(file).map_err(|e| format!("{} {e}", path.display()))
}

pub fn parse(file: Vec<u8>) -> Result<Texture, String> {
    if file.len() < 24 || &file[..4] != b"pika" {
        return Err("is not a baked texture".into());
    }
    let word = |i: usize| u32::from_le_bytes(file[4 + 4 * i..8 + 4 * i].try_into().unwrap());
    let (width, height, layers, levels) = (word(0), word(1), word(2), word(3));
    let format = vk::Format::from_raw(word(4) as i32);
    if level_size(format, 1, 1).is_none() {
        return Err(format!(
            "has the format {format:?}, which the bake tool does not write"
        ));
    }
    if width == 0 || height == 0 || layers == 0 {
        return Err(format!("is {width}×{height} with {layers} layers"));
    }
    let chain = 32 - width.max(height).leading_zeros();
    if levels != chain {
        return Err(format!(
            "is {width}×{height}, which has {chain} mip levels, not {levels}"
        ));
    }
    let mut ranges = Vec::new();
    let (mut offset, mut w, mut h) = (24, width, height);
    for _ in 0..levels {
        let size = level_size(format, w, h).unwrap() * layers as usize;
        ranges.push(offset..offset + size);
        offset += size;
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    if offset != file.len() {
        return Err(format!(
            "holds {} bytes of levels where {} are expected",
            file.len() - 24,
            offset - 24
        ));
    }
    Ok(Texture {
        width,
        height,
        layers,
        format,
        levels: ranges,
        file,
    })
}

/// One layer of a `width`×`height` level in the formats the tool writes:
/// block formats store 4×4 blocks, the edges padded to whole blocks.
pub fn level_size(format: vk::Format, width: u32, height: u32) -> Option<usize> {
    let texels = width as usize * height as usize;
    let blocks = width.div_ceil(4) as usize * height.div_ceil(4) as usize;
    Some(match format {
        vk::Format::R32_SFLOAT => texels * 4,
        vk::Format::R32G32_SFLOAT => texels * 8,
        vk::Format::R32G32B32A32_SFLOAT => texels * 16,
        vk::Format::BC4_UNORM_BLOCK => blocks * 8,
        vk::Format::BC7_UNORM_BLOCK | vk::Format::BC7_SRGB_BLOCK => blocks * 16,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bake tool wrote this for bake/fixtures/ramp.exr, whose texels
    /// hold their own centres.
    const RAMP: &[u8] = include_bytes!("../bake/fixtures/ramp.texture");

    #[test]
    fn reads_the_ramp_fixture() {
        let t = parse(RAMP.to_vec()).unwrap();
        assert_eq!((t.width, t.height, t.layers), (8, 8, 1));
        assert_eq!(t.format, vk::Format::R32G32_SFLOAT);
        assert_eq!(t.levels.len(), 4);
        let texel = |level: usize, x: usize, y: usize| {
            let at = t.levels[level].start + (y * (8 >> level) + x) * 8;
            let f = |i: usize| f32::from_le_bytes(t.file[at + i..at + i + 4].try_into().unwrap());
            [f(0), f(4)]
        };
        assert_eq!(texel(0, 3, 5), [3.5 / 8.0, 5.5 / 8.0]);
        assert_eq!(texel(1, 1, 2), [1.5 / 4.0, 2.5 / 4.0]);
        assert_eq!(texel(3, 0, 0), [0.5, 0.5]);

        assert!(parse(RAMP[..RAMP.len() - 1].to_vec()).is_err(), "truncated");
        let mut bad = RAMP.to_vec();
        bad[0] = b'q';
        assert!(parse(bad).is_err(), "another file");
        let mut bad = RAMP.to_vec();
        bad[16] = 3;
        assert!(parse(bad).is_err(), "a level missing");
        let mut bad = RAMP.to_vec();
        bad[20] = 0;
        assert!(parse(bad).is_err(), "an unknown format");
    }

    #[test]
    fn block_formats_pad_to_whole_blocks() {
        assert_eq!(level_size(vk::Format::BC7_SRGB_BLOCK, 5, 1), Some(32));
        assert_eq!(level_size(vk::Format::BC4_UNORM_BLOCK, 4, 4), Some(8));
        assert_eq!(level_size(vk::Format::R32G32_SFLOAT, 5, 1), Some(40));
        assert_eq!(level_size(vk::Format::R8G8B8A8_UNORM, 4, 4), None);
    }
}
