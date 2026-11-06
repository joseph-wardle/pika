//! Baked textures: the file `bake texture` writes for one image or UDIM
//! set under the cache, laid out as bake/texture.cpp says, as the renderer
//! reads it, the image it is uploaded to, and the set of them loaded for
//! the materials drawn, with the samplers they are read through.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::{Path, PathBuf};

use ash::vk;

use crate::gpu::{self, Buffer};

pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub format: vk::Format,
    /// The source's width and height before the cap, a set's largest tile's.
    pub source: [u32; 2],
    /// The wrap modes the source's metadata states, u then v, in the image
    /// nodedefs' enumeration (black, clamp, periodic, mirror), black where
    /// it states none: what a UsdUVTexture's useMetadata reads
    /// (`material::address_modes`).
    pub wrap: [u32; 2],
    /// The layer of each tile, which the shader reads (`mx_texture.slang`).
    pub tiles: Vec<u32>,
    /// Each level's bytes within `file`, all its layers, finest first.
    pub levels: Vec<Range<usize>>,
    pub file: Vec<u8>,
}

impl Texture {
    /// Its levels' bytes: what loading it uploads.
    pub fn bytes(&self) -> u64 {
        self.levels.iter().map(|l| l.len() as u64).sum()
    }

    /// Whether the cap shrank it: its source was larger.
    pub fn capped(&self) -> bool {
        self.source != [self.width, self.height]
    }
}

/// A 1×1 texture of `texel`: what an image whose file is missing is drawn
/// as (`bake::build`). Periodic, so a node reading the file's wrap modes
/// samples the texel everywhere.
pub fn solid(texel: [f32; 4]) -> Texture {
    Texture {
        width: 1,
        height: 1,
        layers: 1,
        format: vk::Format::R32G32B32A32_SFLOAT,
        source: [1, 1],
        wrap: [2, 2],
        tiles: vec![0],
        levels: std::iter::once(0..16).collect(),
        file: texel.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

pub fn read(path: &Path) -> Result<Texture, String> {
    let file = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse(file).map_err(|e| format!("{} {e}", path.display()))
}

pub fn parse(file: Vec<u8>) -> Result<Texture, String> {
    if file.len() < 44 || &file[..4] != b"pika" {
        return Err("is not a baked texture".into());
    }
    let word = |i: usize| u32::from_le_bytes(file[4 + 4 * i..8 + 4 * i].try_into().unwrap());
    let (width, height, layers, levels) = (word(0), word(1), word(2), word(3));
    let format = vk::Format::from_raw(word(4) as i32);
    let source = [word(5), word(6)];
    let wrap = [word(7), word(8)];
    let entries = word(9) as usize;
    if entries != 1 && entries != 100 || file.len() < 44 + 4 * entries {
        return Err(format!(
            "has a tile table of {entries} entries, not 1 or 100"
        ));
    }
    let tiles: Vec<u32> = (0..entries).map(|i| word(10 + i)).collect();
    if tiles
        .iter()
        .any(|&layer| layer != u32::MAX && layer >= layers)
    {
        return Err(format!("maps a tile past its {layers} layers"));
    }
    if level_size(format, 1, 1).is_none() {
        return Err(format!(
            "has the format {format:?}, which the bake tool does not write"
        ));
    }
    if width == 0 || height == 0 || layers == 0 {
        return Err(format!("is {width}×{height} with {layers} layers"));
    }
    if source[0] < width || source[1] < height {
        return Err(format!(
            "is {width}×{height} from a source of {}×{}",
            source[0], source[1]
        ));
    }
    if let Some(mode) = wrap.iter().find(|&&m| m > 3) {
        return Err(format!(
            "states the wrap mode {mode}, not black, clamp, periodic or mirror"
        ));
    }
    let chain = 32 - width.max(height).leading_zeros();
    if levels != chain {
        return Err(format!(
            "is {width}×{height}, which has {chain} mip levels, not {levels}"
        ));
    }
    let mut ranges = Vec::new();
    let (mut offset, mut w, mut h) = (44 + 4 * entries, width, height);
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
            file.len() - ranges[0].start,
            offset - ranges[0].start
        ));
    }
    Ok(Texture {
        width,
        height,
        layers,
        format,
        source,
        wrap,
        tiles,
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

/// A texture on the GPU: a 2D array image with every level, in the layout
/// the material pass samples once its upload has run, and its tile table.
struct Image {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    bytes: u64,
    tiles: Buffer,
    wrap: [u32; 2],
}

/// What a surface binds for one texture: its view and its tile table,
/// whose range is the table's own length, which the shader reads to tell a
/// plain image from a set (a buffer itself is at least 16 bytes), and the
/// wrap modes its file states (`Texture::wrap`), which choose its sampler
/// with its node's (`material::address_modes`).
#[derive(Clone, Copy)]
pub struct Binding {
    pub view: vk::ImageView,
    pub tiles: vk::DescriptorBufferInfo,
    pub wrap: [u32; 2],
}

impl Image {
    /// Creates the image for `texture` and the staging buffer holding its
    /// levels, which `record` copies from.
    fn new(device: &gpu::Device, texture: &Texture) -> Result<(Self, Buffer), String> {
        let d = &device.raw;
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(texture.format)
            .extent(vk::Extent3D {
                width: texture.width,
                height: texture.height,
                depth: 1,
            })
            .mip_levels(texture.levels.len() as u32)
            .array_layers(texture.layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST);
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let mut tiles = Buffer::host(device, 4 * texture.tiles.len() as u64, usage, host)?;
        for (i, layer) in texture.tiles.iter().enumerate() {
            tiles.bytes()[4 * i..4 * i + 4].copy_from_slice(&layer.to_ne_bytes());
        }
        let mut i = Self {
            device: d.clone(),
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            bytes: 0,
            tiles,
            wrap: texture.wrap,
        };
        // SAFETY (this and every call below): `d` is live, each info struct
        // outlives its call, and each handle passed in was created on `d`
        // just above. `drop` destroys what was made before an error.
        i.image = unsafe { d.create_image(&info, None) }.map_err(|e| {
            format!(
                "vkCreateImage ({}×{} {:?}) failed: {e}",
                texture.width, texture.height, texture.format
            )
        })?;
        let needs = unsafe { d.get_image_memory_requirements(i.image) };
        i.bytes = needs.size;
        let flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        i.memory = gpu::allocate(device, needs, flags, vk::MemoryAllocateFlags::empty())?;
        unsafe { d.bind_image_memory(i.image, i.memory, 0) }
            .map_err(|e| format!("vkBindImageMemory failed: {e}"))?;
        let view_info = vk::ImageViewCreateInfo::default()
            .image(i.image)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(texture.format)
            .subresource_range(range(texture));
        i.view = unsafe { d.create_image_view(&view_info, None) }
            .map_err(|e| format!("vkCreateImageView failed: {e}"))?;
        // The levels alone: each starts at a multiple of its format's block
        // size then, which a copy's buffer offset must be; the header is
        // not one.
        let usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let levels = &texture.file[texture.levels[0].start..];
        let mut staging = Buffer::host(device, levels.len() as u64, usage, host)?;
        staging.bytes()[..levels.len()].copy_from_slice(levels);
        Ok((i, staging))
    }

    /// Records into `cb`, outside rendering, the copies of `texture`'s
    /// levels from `staging` and the move into the layout the material
    /// pass samples. `staging` must live until the GPU has run `cb`.
    fn record(&self, cb: vk::CommandBuffer, staging: &Buffer, texture: &Texture) {
        let to_copy = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(self.image)
            .subresource_range(range(texture));
        let to_sample = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.image)
            .subresource_range(range(texture));
        let copies: Vec<_> = (texture.levels.iter().enumerate())
            .map(|(level, bytes)| {
                vk::BufferImageCopy::default()
                    .buffer_offset((bytes.start - texture.levels[0].start) as u64)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: level as u32,
                        base_array_layer: 0,
                        layer_count: texture.layers,
                    })
                    .image_extent(vk::Extent3D {
                        width: (texture.width >> level).max(1),
                        height: (texture.height >> level).max(1),
                        depth: 1,
                    })
            })
            .collect();
        let d = &self.device;
        // SAFETY: `cb` is recording outside rendering, the image was created
        // on `d` with TRANSFER_DST usage for `texture`, and `staging` holds
        // its levels at the offsets copied from.
        unsafe {
            let barriers = [to_copy];
            let barrier = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            d.cmd_pipeline_barrier2(cb, &barrier);
            let layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            d.cmd_copy_buffer_to_image(cb, staging.raw, self.image, layout, &copies);
            let barriers = [to_sample];
            let barrier = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            d.cmd_pipeline_barrier2(cb, &barrier);
        }
    }
}

impl Image {
    fn binding(&self) -> Binding {
        Binding {
            view: self.view,
            tiles: vk::DescriptorBufferInfo::default()
                .buffer(self.tiles.raw)
                .range(self.tiles.size),
            wrap: self.wrap,
        }
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device` or is null, and the
        // owner waited for idle before this drops.
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

fn range(texture: &Texture) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: texture.levels.len() as u32,
        base_array_layer: 0,
        layer_count: texture.layers,
    }
}

/// The textures loaded for the materials drawn, by baked file, and the
/// samplers they are read through. Loads are recorded into a frame's
/// commands, outside rendering, and their staging buffers kept until the
/// next frame begins, when the GPU has read them.
pub struct Textures {
    device: ash::Device,
    images: BTreeMap<PathBuf, Image>,
    /// By address modes, u then v (`material::address_modes`).
    samplers: BTreeMap<[u32; 2], vk::Sampler>,
    staging: Vec<Buffer>,
}

impl Textures {
    pub fn new(device: &gpu::Device) -> Self {
        Self {
            device: device.raw.clone(),
            images: BTreeMap::new(),
            samplers: BTreeMap::new(),
            staging: Vec::new(),
        }
    }

    /// Begins a frame's loads. No GPU work may be pending.
    pub fn begin(&mut self) {
        self.staging.clear();
    }

    /// The baked texture at `path`, read as `texture`, loaded now unless it
    /// is, and the bytes that uploaded: 0 when it was.
    pub fn load(
        &mut self,
        device: &gpu::Device,
        cb: vk::CommandBuffer,
        path: &Path,
        texture: &Texture,
    ) -> Result<u64, String> {
        if self.images.contains_key(path) {
            return Ok(0);
        }
        let (image, staging) = Image::new(device, texture)?;
        image.record(cb, &staging, texture);
        self.staging.push(staging);
        self.images.insert(path.to_owned(), image);
        Ok(texture.bytes())
    }

    /// What a surface binds for the texture at `path`, loaded and kept
    /// while a material lists it (`retain`).
    pub fn binding(&self, path: &Path) -> Result<Binding, String> {
        (self.images.get(path).map(Image::binding))
            .ok_or_else(|| format!("texture {} is not loaded", path.display()))
    }

    /// Whether the texture baked at `path` is loaded.
    pub fn holds(&self, path: &Path) -> bool {
        self.images.contains_key(path)
    }

    /// Drops the textures not in `used`. No GPU work may be pending.
    pub fn retain(&mut self, used: &BTreeSet<PathBuf>) {
        self.images.retain(|path, _| used.contains(path));
    }

    /// A trilinear sampler with the address modes `modes`, u then v, in
    /// the image nodedefs' enumeration: black (a transparent border),
    /// clamp, periodic, mirror.
    pub fn sampler(&mut self, modes: [u32; 2]) -> Result<vk::Sampler, String> {
        if let Some(&sampler) = self.samplers.get(&modes) {
            return Ok(sampler);
        }
        let mode = |m: u32| match m {
            0 => Ok(vk::SamplerAddressMode::CLAMP_TO_BORDER),
            1 => Ok(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            2 => Ok(vk::SamplerAddressMode::REPEAT),
            3 => Ok(vk::SamplerAddressMode::MIRRORED_REPEAT),
            _ => Err(format!(
                "the address mode {m} is not black, clamp, periodic or mirror"
            )),
        };
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(mode(modes[0])?)
            .address_mode_v(mode(modes[1])?)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
            .max_lod(vk::LOD_CLAMP_NONE);
        // SAFETY: `device` is live and `info` outlives the call.
        let sampler = unsafe { self.device.create_sampler(&info, None) }
            .map_err(|e| format!("vkCreateSampler failed: {e}"))?;
        self.samplers.insert(modes, sampler);
        Ok(sampler)
    }

    /// The images' bytes, and the tile tables' and staging buffers' on the
    /// host side.
    pub fn memory(&self) -> gpu::Memory {
        let mut memory = gpu::Memory {
            device: self.images.values().map(|i| i.bytes).sum(),
            host: 0,
        };
        for b in self
            .staging
            .iter()
            .chain(self.images.values().map(|i| &i.tiles))
        {
            memory += b.memory();
        }
        memory
    }
}

impl Drop for Textures {
    fn drop(&mut self) {
        // The images and buffers drop after this, before the device.
        // SAFETY: the samplers were created on `device`, and the owner
        // waited for idle before this drops.
        unsafe {
            for &sampler in self.samplers.values() {
                self.device.destroy_sampler(sampler, None);
            }
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn a_solid_texture_is_one_float_texel() {
        let t = solid([0.2, 0.4, 0.6, 1.0]);
        assert_eq!((t.width, t.height, t.layers, t.levels.len()), (1, 1, 1, 1));
        assert_eq!(t.bytes(), 16);
        assert!(!t.capped());
        assert_eq!(level_size(t.format, 1, 1), Some(16));
        assert_eq!(f32::from_le_bytes(t.file[4..8].try_into().unwrap()), 0.4);
    }

    /// The bake tool wrote this for bake/fixtures/ramp.exr, whose texels
    /// hold their own centres.
    pub const RAMP: &[u8] = include_bytes!("../bake/fixtures/ramp.texture");

    #[test]
    fn reads_the_ramp_fixture() {
        let t = parse(RAMP.to_vec()).unwrap();
        assert_eq!((t.width, t.height, t.layers), (8, 8, 1));
        assert_eq!(t.format, vk::Format::R32G32_SFLOAT);
        assert_eq!(t.source, [8, 8]);
        assert_eq!(t.wrap, [0, 0], "the ramp states no wrap mode: black");
        assert_eq!(t.tiles, [0]);
        assert_eq!(t.levels.len(), 4);
        assert_eq!(t.levels[0].start, 48);
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
        let mut bad = RAMP.to_vec();
        bad[24] = 4;
        assert!(parse(bad).is_err(), "a source smaller than the bake");
        let mut bad = RAMP.to_vec();
        bad[36] = 4;
        assert!(parse(bad).is_err(), "a wrap mode past mirror");
        let mut bad = RAMP.to_vec();
        bad[40] = 2;
        assert!(parse(bad).is_err(), "a table of two entries");
        let mut bad = RAMP.to_vec();
        bad[44] = 1;
        assert!(parse(bad).is_err(), "a tile past the layers");
    }

    #[test]
    fn reads_a_set_of_tiles() {
        // Two 1×1 R32_SFLOAT layers, periodic in u, clamped in v, tiles
        // 1001 and 1012.
        let mut file = b"pika".to_vec();
        for word in [1u32, 1, 2, 1, 100, 1, 1, 2, 1, 100] {
            file.extend_from_slice(&word.to_le_bytes());
        }
        let mut table = [u32::MAX; 100];
        (table[0], table[11]) = (0, 1);
        for entry in table {
            file.extend_from_slice(&entry.to_le_bytes());
        }
        file.extend_from_slice(&0.25f32.to_le_bytes());
        file.extend_from_slice(&0.75f32.to_le_bytes());
        let t = parse(file.clone()).unwrap();
        assert_eq!((t.layers, t.tiles.len()), (2, 100));
        assert_eq!(t.wrap, [2, 1]);
        assert_eq!((t.tiles[0], t.tiles[11], t.tiles[1]), (0, 1, u32::MAX));
        assert_eq!(t.levels, vec![444..452]);
        file.truncate(file.len() - 4);
        assert!(parse(file).is_err(), "a layer missing");
    }

    #[test]
    fn block_formats_pad_to_whole_blocks() {
        assert_eq!(level_size(vk::Format::BC7_SRGB_BLOCK, 5, 1), Some(32));
        assert_eq!(level_size(vk::Format::BC4_UNORM_BLOCK, 4, 4), Some(8));
        assert_eq!(level_size(vk::Format::R32G32_SFLOAT, 5, 1), Some(40));
        assert_eq!(level_size(vk::Format::R8G8B8A8_UNORM, 4, 4), None);
    }
}
