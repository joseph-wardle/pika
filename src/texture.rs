//! Baked textures: the file `bake texture` writes for one image or UDIM
//! set under the cache, laid out as bake/texture.cpp says, as the renderer
//! reads it, the image it is uploaded to, and the set of them loaded for
//! the materials drawn, with the samplers they are read through.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::{Duration, Instant};

use ash::vk;

use crate::gpu::{self, Buffer};

pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub format: vk::Format,
    /// Its baked width and height before the cap (`cap`), a set's largest
    /// tile's.
    pub source: [u32; 2],
    /// The wrap modes the source's metadata states, u then v, in the image
    /// nodedefs' enumeration (black, clamp, periodic, mirror), black where
    /// it states none: what a UsdUVTexture's useMetadata reads
    /// (`material::address_modes`).
    pub wrap: [u32; 2],
    /// The layer of each tile, which the shader reads (`mx_texture.slang`).
    pub tiles: Vec<u32>,
    /// Each level's bytes within the file, all its layers, finest first.
    pub levels: Vec<Range<usize>>,
    pub contents: Contents,
}

/// Where a texture's levels are read from when it uploads (`read_levels`).
pub enum Contents {
    /// The baked file, open from its header's read, so a build holds no
    /// level bytes in memory however large its textures.
    Open(File),
    /// The whole file in memory.
    Memory(Vec<u8>),
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

    /// Drops its levels with a side over `cap` texels, 0 for none: the file
    /// holds every level, so a cap reads fewer of it and bakes nothing.
    pub fn cap(&mut self, cap: u32) {
        while cap > 0 && self.levels.len() > 1 && self.width.max(self.height) > cap {
            self.levels.remove(0);
            self.width = (self.width / 2).max(1);
            self.height = (self.height / 2).max(1);
        }
    }

    /// Adds into `needs`, bytes by heap index, what loading it allocates as
    /// the driver sizes it: its image with alignment and tiling, its tile
    /// table and its staging buffer. Nothing is created.
    pub fn needs(&self, device: &gpu::Device, needs: &mut [u64]) -> Result<(), String> {
        let info = image_info(self);
        let ask = vk::DeviceImageMemoryRequirements::default().create_info(&info);
        let mut image = vk::MemoryRequirements2::default();
        // SAFETY: `device.raw` is a live Vulkan 1.3 device, and `info`
        // outlives the call.
        unsafe {
            device
                .raw
                .get_device_image_memory_requirements(&ask, &mut image)
        };
        let image = image.memory_requirements;
        let tiles = Buffer::host_needs(device, 4 * self.tiles.len() as u64, TILES);
        let staging = Buffer::host_needs(device, self.bytes(), STAGING);
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        needs[device.heap(image.memory_type_bits, local)?] += image.size;
        needs[device.heap(tiles.memory_type_bits, HOST)?] += tiles.size;
        needs[device.heap(staging.memory_type_bits, HOST)?] += staging.size;
        Ok(())
    }

    /// Reads its levels, finest first, into `into`, which is `bytes()` long.
    pub fn read_levels(&self, into: &mut [u8]) -> std::io::Result<()> {
        let start = self.levels[0].start;
        match &self.contents {
            Contents::Open(file) => file.read_exact_at(into, start as u64),
            Contents::Memory(file) => {
                into.copy_from_slice(&file[start..]);
                Ok(())
            }
        }
    }
}

/// The memory and usages of a texture's host buffers: its tile table, which
/// the shader reads, and the staging buffer its levels upload from.
const HOST: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
        | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
);
const TILES: vk::BufferUsageFlags = vk::BufferUsageFlags::STORAGE_BUFFER;
const STAGING: vk::BufferUsageFlags = vk::BufferUsageFlags::TRANSFER_SRC;

/// The longest header: the fixed words and a set's tile table.
const HEADER: usize = FIXED + 4 * 100;
/// "pika" and eight words before the tile table.
const FIXED: usize = 36;

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
        contents: Contents::Memory(texel.iter().flat_map(|v| v.to_le_bytes()).collect()),
    }
}

/// Reads the baked file `path`'s header, checked against the file's
/// length, and keeps the file open for its levels.
pub fn read(path: &Path) -> Result<Texture, String> {
    let cannot = |e: std::io::Error| format!("cannot read {}: {e}", path.display());
    let file = File::open(path).map_err(cannot)?;
    let len = file.metadata().map_err(cannot)?.len() as usize;
    let mut head = vec![0; len.min(HEADER)];
    file.read_exact_at(&mut head, 0).map_err(cannot)?;
    header(&head, len, Contents::Open(file)).map_err(|e| format!("{} {e}", path.display()))
}

/// The texture whose whole file is `file`, as the tests build them.
#[cfg(test)]
pub fn parse(file: Vec<u8>) -> Result<Texture, String> {
    let head = file[..file.len().min(HEADER)].to_vec();
    header(&head, file.len(), Contents::Memory(file))
}

/// The texture whose file of `len` bytes begins with `head`, the file
/// itself or at least its `HEADER` bytes, its levels in `contents`.
fn header(head: &[u8], len: usize, contents: Contents) -> Result<Texture, String> {
    if head.len() < FIXED || &head[..4] != b"pika" {
        return Err("is not a baked texture".into());
    }
    let word = |i: usize| u32::from_le_bytes(head[4 + 4 * i..8 + 4 * i].try_into().unwrap());
    let (width, height, layers, levels) = (word(0), word(1), word(2), word(3));
    let format = vk::Format::from_raw(word(4) as i32);
    let wrap = [word(5), word(6)];
    let entries = word(7) as usize;
    if entries != 1 && entries != 100 || head.len() < FIXED + 4 * entries {
        return Err(format!(
            "has a tile table of {entries} entries, not 1 or 100"
        ));
    }
    let tiles: Vec<u32> = (0..entries).map(|i| word(8 + i)).collect();
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
    let (mut offset, mut w, mut h) = (FIXED + 4 * entries, width, height);
    for _ in 0..levels {
        let size = level_size(format, w, h).unwrap() * layers as usize;
        ranges.push(offset..offset + size);
        offset += size;
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    if offset != len {
        return Err(format!(
            "holds {} bytes of levels where {} are expected",
            len.saturating_sub(ranges[0].start),
            offset - ranges[0].start
        ));
    }
    Ok(Texture {
        width,
        height,
        layers,
        format,
        source: [width, height],
        wrap,
        tiles,
        levels: ranges,
        contents,
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

/// The image `texture` uploads to, which `Texture::needs` sizes.
fn image_info(texture: &Texture) -> vk::ImageCreateInfo<'static> {
    vk::ImageCreateInfo::default()
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
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
}

impl Image {
    /// Creates the image for `texture`.
    fn new(device: &gpu::Device, texture: &Texture) -> Result<Self, String> {
        let d = &device.raw;
        let info = image_info(texture);
        let mut tiles = Buffer::host(device, 4 * texture.tiles.len() as u64, TILES, HOST)?;
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
        Ok(i)
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

/// The textures loaded for the materials drawn, by `bake::Baked::key`, and the
/// samplers they are read through. Loads are recorded into a frame's
/// commands, outside rendering, and their staging buffers kept until the
/// next frame begins, when the GPU has read them.
pub struct Textures {
    device: ash::Device,
    images: BTreeMap<String, Image>,
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

    /// The baked texture keyed `key`, read as `texture`, loaded now unless
    /// it is, its levels read into a staging buffer; the bytes that
    /// uploaded, 0 when it was loaded, and the time their read took.
    pub fn load(
        &mut self,
        device: &gpu::Device,
        cb: vk::CommandBuffer,
        key: &str,
        texture: &Texture,
    ) -> Result<(u64, Duration), String> {
        if self.images.contains_key(key) {
            return Ok((0, Duration::ZERO));
        }
        let image = Image::new(device, texture).map_err(|e| format!("{key}: {e}"))?;
        // The levels alone: each starts at a multiple of its format's block
        // size then, which a copy's buffer offset must be; the header is
        // not one.
        let size = texture.bytes();
        let mut staging = Buffer::host(device, size, STAGING, HOST)?;
        let start = Instant::now();
        (texture.read_levels(&mut staging.bytes()[..size as usize]))
            .map_err(|e| format!("cannot read the levels of {key}: {e}"))?;
        let read = start.elapsed();
        image.record(cb, &staging, texture);
        self.staging.push(staging);
        self.images.insert(key.to_owned(), image);
        Ok((size, read))
    }

    /// What a surface binds for the texture keyed `key`, loaded and kept
    /// while a material lists it (`retain`).
    pub fn binding(&self, key: &str) -> Result<Binding, String> {
        (self.images.get(key).map(Image::binding))
            .ok_or_else(|| format!("texture {key} is not loaded"))
    }

    /// Whether the texture keyed `key` is loaded.
    pub fn holds(&self, key: &str) -> bool {
        self.images.contains_key(key)
    }

    /// Drops the textures not in `used`. No GPU work may be pending.
    pub fn retain(&mut self, used: &BTreeSet<String>) {
        self.images.retain(|key, _| used.contains(key));
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
        let mut texel = [0; 16];
        t.read_levels(&mut texel).unwrap();
        assert_eq!(f32::from_le_bytes(texel[4..8].try_into().unwrap()), 0.4);
    }

    /// `t`'s levels as an upload reads them.
    fn levels(t: &Texture) -> Vec<u8> {
        let mut bytes = vec![0; t.bytes() as usize];
        t.read_levels(&mut bytes).unwrap();
        bytes
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
        assert_eq!(t.levels[0].start, 40);
        let bytes = levels(&t);
        let texel = |level: usize, x: usize, y: usize| {
            let at = t.levels[level].start - 40 + (y * (8 >> level) + x) * 8;
            let f = |i: usize| f32::from_le_bytes(bytes[at + i..at + i + 4].try_into().unwrap());
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
        bad[28] = 4;
        assert!(parse(bad).is_err(), "a wrap mode past mirror");
        let mut bad = RAMP.to_vec();
        bad[32] = 2;
        assert!(parse(bad).is_err(), "a table of two entries");
        let mut bad = RAMP.to_vec();
        bad[36] = 1;
        assert!(parse(bad).is_err(), "a tile past the layers");
    }

    #[test]
    fn reads_the_levels_from_the_open_file() {
        let path = std::env::temp_dir().join(format!("pika_texture_{}", std::process::id()));
        std::fs::write(&path, RAMP).unwrap();
        let t = read(&path).unwrap();
        assert!(matches!(t.contents, Contents::Open(_)));
        assert_eq!(levels(&t), RAMP[40..], "the levels, not the header");

        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(RAMP.len() as u64 - 8).unwrap();
        let mut bytes = vec![0; t.bytes() as usize];
        assert!(
            t.read_levels(&mut bytes).is_err(),
            "cut short after its header was read"
        );
        file.set_len(40).unwrap();
        let why = read(&path).err().unwrap();
        assert!(
            why.contains("holds 0 bytes of levels where 680 are expected"),
            "a header and no levels: {why}"
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn levels_follow_the_header_in_each_format() {
        // A 5×3 image of two layers: three levels, 5×3, 2×1 and 1×1.
        for (format, sizes) in [
            (vk::Format::R32_SFLOAT, [60, 8, 4]),
            (vk::Format::R32G32_SFLOAT, [120, 16, 8]),
            (vk::Format::R32G32B32A32_SFLOAT, [240, 32, 16]),
            (vk::Format::BC4_UNORM_BLOCK, [16, 8, 8]),
            (vk::Format::BC7_UNORM_BLOCK, [32, 16, 16]),
            (vk::Format::BC7_SRGB_BLOCK, [32, 16, 16]),
        ] {
            let mut file = b"pika".to_vec();
            let words = [5, 3, 2, 3, format.as_raw() as u32, 0, 0, 1, 0];
            for word in words {
                file.extend_from_slice(&word.to_le_bytes());
            }
            let size: usize = sizes.iter().map(|s| 2 * s).sum();
            file.resize(40 + size, 7);
            let t = parse(file.clone()).unwrap();
            let mut start = 40;
            for (level, s) in t.levels.iter().zip(sizes) {
                assert_eq!(*level, start..start + 2 * s, "{format:?}");
                start = level.end;
            }
            assert_eq!(t.bytes() as usize, size);
            file.pop();
            assert!(parse(file).is_err(), "{format:?} a byte short");
        }
    }

    #[test]
    fn reads_a_set_of_tiles() {
        // Two 1×1 R32_SFLOAT layers, periodic in u, clamped in v, tiles
        // 1001 and 1012.
        let mut file = b"pika".to_vec();
        for word in [1u32, 1, 2, 1, 100, 2, 1, 100] {
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
        assert_eq!(t.levels, vec![436..444]);
        file.truncate(file.len() - 4);
        assert!(parse(file).is_err(), "a layer missing");
    }

    #[test]
    fn a_cap_drops_the_levels_over_it() {
        let ramp = || parse(RAMP.to_vec()).unwrap();
        let mut t = ramp();
        t.cap(4);
        assert_eq!((t.width, t.height, t.levels.len()), (4, 4, 3));
        assert_eq!(t.source, [8, 8]);
        assert!(t.capped());
        // The ramp's level 1, which holds its own texel centres at 4×4.
        let bytes = levels(&t);
        assert_eq!(bytes, RAMP[40 + 8 * 8 * 8..], "the levels from 4×4 on");
        let texel = |x: usize, y: usize| {
            let at = (y * 4 + x) * 8;
            let f = |i: usize| f32::from_le_bytes(bytes[at + i..at + i + 4].try_into().unwrap());
            [f(0), f(4)]
        };
        assert_eq!(texel(1, 2), [1.5 / 4.0, 2.5 / 4.0]);
        t.cap(1);
        assert_eq!((t.width, t.height, t.levels.len()), (1, 1, 1));

        // The controls: a cap of none, or one the source fits, keeps all.
        for cap in [0, 8, 4096] {
            let mut t = ramp();
            t.cap(cap);
            assert_eq!(
                (t.width, t.levels.len(), t.bytes()),
                (8, 4, 680),
                "cap {cap}"
            );
            assert!(!t.capped(), "cap {cap}");
        }

        // A 5×3 image: its levels are 5×3, 2×1 and 1×1, so a cap of 4 or
        // 3 keeps 2×1, as the bake tool's halving once did.
        let mut file = b"pika".to_vec();
        for word in [5u32, 3, 1, 3, 100, 0, 0, 1, 0] {
            file.extend_from_slice(&word.to_le_bytes());
        }
        file.resize(40 + 4 * (15 + 2 + 1), 0);
        for cap in [4, 3, 2] {
            let mut t = parse(file.clone()).unwrap();
            t.cap(cap);
            assert_eq!((t.width, t.height, t.levels.len()), (2, 1, 2), "cap {cap}");
            assert_eq!(t.levels[0], 100..108);
        }
    }

    #[test]
    fn block_formats_pad_to_whole_blocks() {
        assert_eq!(level_size(vk::Format::BC7_SRGB_BLOCK, 5, 1), Some(32));
        assert_eq!(level_size(vk::Format::BC4_UNORM_BLOCK, 4, 4), Some(8));
        assert_eq!(level_size(vk::Format::R32G32_SFLOAT, 5, 1), Some(40));
        assert_eq!(level_size(vk::Format::R8G8B8A8_UNORM, 4, 4), None);
    }
}
