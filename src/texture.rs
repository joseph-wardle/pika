//! Baked textures: the file `bake texture` writes for one image or UDIM
//! set under the cache, laid out as bake/texture.cpp says, as the renderer
//! reads it, the image it is uploaded to, holding the levels within the
//! texture cap, and the set of them loaded for the materials drawn, with
//! the samplers they are read through. A changed cap swaps each image for
//! one of its new levels, read from the file, a frame's worth of bytes at
//! a time; nothing is baked or built again.

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
    /// The first of its levels within `cap` texels a side, 0 for none: the
    /// file holds every level, so a cap reads fewer of it and bakes nothing.
    /// Its last level is within any cap.
    pub fn first(&self, cap: u32) -> usize {
        let over = |&level: &usize| {
            let (w, h) = self.extent(level);
            cap > 0 && w.max(h) > cap
        };
        (0..self.levels.len() - 1).take_while(over).count()
    }

    /// The width and height of its level `level`.
    pub fn extent(&self, level: usize) -> (u32, u32) {
        ((self.width >> level).max(1), (self.height >> level).max(1))
    }

    /// The bytes of its levels from `first` on: what loading them uploads.
    pub fn bytes(&self, first: usize) -> u64 {
        self.levels[first..].iter().map(|l| l.len() as u64).sum()
    }

    /// Adds into `needs`, bytes by heap index, what loading its levels
    /// from `first` on allocates as the driver sizes it: their image with
    /// alignment and tiling, its tile table and at most a staging buffer of
    /// its own. Nothing is created.
    pub fn needs(
        &self,
        device: &gpu::Device,
        first: usize,
        needs: &mut [u64],
    ) -> Result<(), String> {
        let info = image_info(self, first);
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
        let staging = Buffer::host_needs(device, self.bytes(first), STAGING);
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        needs[device.heap(image.memory_type_bits, local)?] += image.size;
        needs[device.heap(tiles.memory_type_bits, HOST)?] += tiles.size;
        needs[device.heap(staging.memory_type_bits, HOST)?] += staging.size;
        Ok(())
    }

    /// Reads its levels from `first` on, finest first, into `into`, which
    /// is `bytes(first)` long.
    pub fn read_levels(&self, first: usize, into: &mut [u8]) -> std::io::Result<()> {
        let start = self.levels[first].start;
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

/// The bytes of levels a frame uploads, loads and swaps together, past
/// which the rest wait for the next: reading and copying a texture's levels
/// costs the render thread about 0.8 ms a MiB on the RTX 4070 Ti SUPER. A
/// frame's first upload goes however large, so a texture larger than this
/// still loads, alone in its frame.
pub const UPLOAD: u64 = 32 << 20;

/// A texture's levels on the GPU from `first` on: a 2D array image, in the
/// layout the material pass samples once its upload has run.
struct Image {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    bytes: u64,
    first: usize,
}

/// A texture loaded: its file, its tile table, which a swap keeps, and the
/// image of its levels now.
struct Loaded {
    texture: Texture,
    tiles: Buffer,
    image: Image,
    /// The first level its image is to hold: the cap's, or the image's own
    /// once a swap to the cap's failed, until the cap changes.
    target: usize,
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

/// The image of `texture`'s levels from `first` on, which `Texture::needs`
/// sizes.
fn image_info(texture: &Texture, first: usize) -> vk::ImageCreateInfo<'static> {
    let (width, height) = texture.extent(first);
    vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(texture.format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels((texture.levels.len() - first) as u32)
        .array_layers(texture.layers)
        .samples(vk::SampleCountFlags::TYPE_1)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
}

impl Image {
    /// Creates the image of `texture`'s levels from `first` on.
    fn new(device: &gpu::Device, texture: &Texture, first: usize) -> Result<Self, String> {
        let d = &device.raw;
        let info = image_info(texture, first);
        let mut i = Self {
            device: d.clone(),
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            bytes: 0,
            first,
        };
        // SAFETY (this and every call below): `d` is live, each info struct
        // outlives its call, and each handle passed in was created on `d`
        // just above. `drop` destroys what was made before an error.
        i.image = unsafe { d.create_image(&info, None) }.map_err(|e| {
            let (width, height) = texture.extent(first);
            format!(
                "vkCreateImage ({width}×{height} {:?}) failed: {e}",
                texture.format
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
            .subresource_range(i.range(texture));
        i.view = unsafe { d.create_image_view(&view_info, None) }
            .map_err(|e| format!("vkCreateImageView failed: {e}"))?;
        Ok(i)
    }

    /// Records into `cb`, outside rendering, the copies of its levels of
    /// `texture` from `staging`, where they start at `at`, and the move into
    /// the layout the material pass samples. `staging` must live until the
    /// GPU has run `cb`.
    fn record(&self, cb: vk::CommandBuffer, staging: &Buffer, at: u64, texture: &Texture) {
        let to_copy = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(self.image)
            .subresource_range(self.range(texture));
        let to_sample = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.image)
            .subresource_range(self.range(texture));
        let start = texture.levels[self.first].start;
        let copies: Vec<_> = (self.first..texture.levels.len())
            .map(|level| {
                let (width, height) = texture.extent(level);
                vk::BufferImageCopy::default()
                    .buffer_offset(at + (texture.levels[level].start - start) as u64)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: (level - self.first) as u32,
                        base_array_layer: 0,
                        layer_count: texture.layers,
                    })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
            })
            .collect();
        let d = &self.device;
        // SAFETY: `cb` is recording outside rendering, the image was created
        // on `d` with TRANSFER_DST usage for these levels of `texture`, and
        // `staging` holds them at the offsets copied from.
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

    fn range(&self, texture: &Texture) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: (texture.levels.len() - self.first) as u32,
            base_array_layer: 0,
            layer_count: texture.layers,
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

/// The first level of `texture` a frame with `left` bytes still to upload
/// loads for a target of `target`: the target, unless its levels are more
/// than are left and another upload went first, then the finest whose
/// levels fit, its last at least, which swaps raise in later frames.
fn fit(texture: &Texture, target: usize, left: u64) -> usize {
    let last = texture.levels.len() - 1;
    if left == UPLOAD {
        return target;
    }
    (target..last)
        .find(|&first| texture.bytes(first) <= left)
        .unwrap_or(last)
}

/// The host memory a frame's level reads land in, which its commands copy
/// into the images: one buffer of `UPLOAD` bytes kept from frame to frame,
/// since on ALab a buffer of a texture's own cost seven times its read to
/// create and free, and one of its own for an upload past what is left.
struct Staging {
    buffer: Buffer,
    /// The bytes of `buffer` this frame's uploads hold.
    used: u64,
    own: Vec<Buffer>,
}

impl Staging {
    fn new(device: &gpu::Device) -> Result<Self, String> {
        Ok(Self {
            buffer: Buffer::host(device, UPLOAD, STAGING, HOST)?,
            used: 0,
            own: Vec::new(),
        })
    }

    /// Room for an upload of `size` bytes: the buffer and where in it.
    fn room(&mut self, device: &gpu::Device, size: u64) -> Result<(&mut Buffer, u64), String> {
        // A copy's buffer offset is a multiple of its format's block size,
        // at most 16 bytes.
        let at = self.used.next_multiple_of(16);
        if at + size <= self.buffer.size {
            self.used = at + size;
            return Ok((&mut self.buffer, at));
        }
        self.own.push(Buffer::host(device, size, STAGING, HOST)?);
        Ok((self.own.last_mut().unwrap(), 0))
    }
}

/// The image of `texture`'s levels from `first` on, created now, and their
/// upload recorded into `cb` from `staging`; the time reading them took.
fn upload(
    device: &gpu::Device,
    cb: vk::CommandBuffer,
    texture: &Texture,
    first: usize,
    staging: &mut Staging,
) -> Result<(Image, Duration), String> {
    let image = Image::new(device, texture, first)?;
    // The levels alone: each starts at a multiple of its format's block
    // size then, which a copy's buffer offset must be; the header is not
    // one.
    let size = texture.bytes(first);
    let (buffer, at) = staging.room(device, size)?;
    let start = Instant::now();
    let into = &mut buffer.bytes()[at as usize..(at + size) as usize];
    (texture.read_levels(first, into)).map_err(|e| format!("cannot read its levels: {e}"))?;
    let read = start.elapsed();
    image.record(cb, buffer, at, texture);
    Ok((image, read))
}

/// The textures loaded for the materials drawn, by `bake::Baked::key`, each
/// holding the levels within the texture cap, and the samplers they are
/// read through. Loads and swaps are recorded into a frame's commands,
/// outside rendering, `UPLOAD` bytes a frame, and their staging is kept
/// until the next frame begins, when the GPU has read it.
pub struct Textures {
    device: ash::Device,
    loaded: BTreeMap<String, Loaded>,
    /// By address modes, u then v (`material::address_modes`).
    samplers: BTreeMap<[u32; 2], vk::Sampler>,
    staging: Staging,
    /// The images swaps replaced, kept until the next frame begins, by
    /// when every material binding them has been rebound.
    retired: Vec<Image>,
    /// The texels a side of the cap, 0 for none.
    cap: u32,
    /// The bytes this frame may still upload.
    left: u64,
}

impl Textures {
    pub fn new(device: &gpu::Device) -> Result<Self, String> {
        Ok(Self {
            device: device.raw.clone(),
            loaded: BTreeMap::new(),
            samplers: BTreeMap::new(),
            staging: Staging::new(device)?,
            retired: Vec::new(),
            cap: 0,
            left: UPLOAD,
        })
    }

    /// Begins a frame's loads and swaps. No GPU work may be pending.
    pub fn begin(&mut self) {
        (self.staging.used, self.staging.own) = (0, Vec::new());
        self.retired.clear();
        self.left = UPLOAD;
    }

    pub fn cap(&self) -> u32 {
        self.cap
    }

    /// Sets the cap to `cap` texels a side, 0 for none, and so each loaded
    /// texture's target, which `swap` moves its image to; whether it changed.
    pub fn set_cap(&mut self, cap: u32) -> bool {
        if cap == self.cap {
            return false;
        }
        self.cap = cap;
        for loaded in self.loaded.values_mut() {
            loaded.target = loaded.texture.first(cap);
        }
        true
    }

    /// The baked texture keyed `key`, read as `texture`, loaded now unless
    /// it is: its levels within the cap, or as many of them as the frame's
    /// bytes allow (`fit`), read into a staging buffer; the bytes that
    /// uploaded, 0 when it was loaded, and the time their read took.
    pub fn load(
        &mut self,
        device: &gpu::Device,
        cb: vk::CommandBuffer,
        key: &str,
        texture: Texture,
    ) -> Result<(u64, Duration), String> {
        if self.loaded.contains_key(key) {
            return Ok((0, Duration::ZERO));
        }
        let mut tiles = Buffer::host(device, 4 * texture.tiles.len() as u64, TILES, HOST)?;
        for (i, layer) in texture.tiles.iter().enumerate() {
            tiles.bytes()[4 * i..4 * i + 4].copy_from_slice(&layer.to_ne_bytes());
        }
        let target = texture.first(self.cap);
        let first = fit(&texture, target, self.left);
        let (image, read) = upload(device, cb, &texture, first, &mut self.staging)
            .map_err(|e| format!("{key}: {e}"))?;
        let size = texture.bytes(first);
        self.left = self.left.saturating_sub(size);
        let loaded = Loaded {
            texture,
            tiles,
            image,
            target,
        };
        self.loaded.insert(key.to_owned(), loaded);
        Ok((size, read))
    }

    /// Swaps each image not holding its target's levels for one that does,
    /// read from its file, while the frame's bytes last (`UPLOAD`), in key
    /// order: the new image is made and uploaded first, and the old one
    /// retired once the materials binding it are rebound, so a swap that
    /// fails keeps the old. A raise goes only if its image fits the
    /// device's memory budget beside the old. A swap that fails is warned
    /// of, and its texture kept as it is until the cap changes. Records
    /// the uploads into `cb`; the keys swapped, which the caller rebinds
    /// before the next frame begins, the bytes uploaded and the time
    /// their read took.
    pub fn swap(
        &mut self,
        device: &gpu::Device,
        cb: vk::CommandBuffer,
    ) -> (BTreeSet<String>, u64, Duration) {
        let (mut keys, mut bytes, mut read) = (BTreeSet::new(), 0, Duration::ZERO);
        let heaps = device.heaps();
        let mut raised = vec![0; heaps.len()];
        for (key, loaded) in &mut self.loaded {
            let (texture, target) = (&loaded.texture, loaded.target);
            if loaded.image.first == target {
                continue;
            }
            let size = texture.bytes(target);
            if self.left < UPLOAD && size > self.left {
                continue;
            }
            let swapped = (|| {
                if target < loaded.image.first {
                    let mut needs = raised.clone();
                    texture.needs(device, target, &mut needs)?;
                    if gpu::overflow(&heaps, &needs).is_some() {
                        let (w, h) = texture.extent(target);
                        return Err(format!(
                            "its {w}×{h} levels do not fit the device's memory budget"
                        ));
                    }
                    raised = needs;
                }
                upload(device, cb, texture, target, &mut self.staging)
            })();
            match swapped {
                Ok((image, time)) => {
                    let old = std::mem::replace(&mut loaded.image, image);
                    self.retired.push(old);
                    self.left = self.left.saturating_sub(size);
                    (bytes, read) = (bytes + size, read + time);
                    keys.insert(key.clone());
                }
                Err(e) => {
                    let (w, h) = texture.extent(loaded.image.first);
                    eprintln!("pika: warning: texture {key}: {e}; kept at {w}×{h}");
                    loaded.target = loaded.image.first;
                }
            }
        }
        (keys, bytes, read)
    }

    /// The textures whose image does not hold its target's levels yet.
    pub fn pending(&self) -> usize {
        (self.loaded.values())
            .filter(|l| l.image.first != l.target)
            .count()
    }

    /// The textures the cap shrinks.
    pub fn capped(&self) -> usize {
        (self.loaded.values())
            .filter(|l| l.texture.first(self.cap) > 0)
            .count()
    }

    /// What a surface binds for the texture keyed `key`, loaded and kept
    /// while a material lists it (`retain`).
    pub fn binding(&self, key: &str) -> Result<Binding, String> {
        let loaded =
            (self.loaded.get(key)).ok_or_else(|| format!("texture {key} is not loaded"))?;
        Ok(Binding {
            view: loaded.image.view,
            tiles: vk::DescriptorBufferInfo::default()
                .buffer(loaded.tiles.raw)
                .range(loaded.tiles.size),
            wrap: loaded.texture.wrap,
        })
    }

    /// Whether the texture keyed `key` is loaded.
    pub fn holds(&self, key: &str) -> bool {
        self.loaded.contains_key(key)
    }

    /// Drops the textures not in `used`. No GPU work may be pending.
    pub fn retain(&mut self, used: &BTreeSet<String>) {
        self.loaded.retain(|key, _| used.contains(key));
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

    /// The images' bytes, retired ones included, and the tile tables' and
    /// staging's on the host side.
    pub fn memory(&self) -> gpu::Memory {
        let images = self.loaded.values().map(|l| &l.image).chain(&self.retired);
        let mut memory = gpu::Memory {
            device: images.map(|i| i.bytes).sum(),
            host: 0,
        };
        let tiles = self.loaded.values().map(|l| &l.tiles);
        let staging = std::iter::once(&self.staging.buffer).chain(&self.staging.own);
        for b in staging.chain(tiles) {
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
        assert_eq!(t.bytes(0), 16);
        assert_eq!(t.first(1), 0);
        assert_eq!(level_size(t.format, 1, 1), Some(16));
        let mut texel = [0; 16];
        t.read_levels(0, &mut texel).unwrap();
        assert_eq!(f32::from_le_bytes(texel[4..8].try_into().unwrap()), 0.4);
    }

    /// `t`'s levels from `first` on as an upload reads them.
    fn levels(t: &Texture, first: usize) -> Vec<u8> {
        let mut bytes = vec![0; t.bytes(first) as usize];
        t.read_levels(first, &mut bytes).unwrap();
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
        assert_eq!(t.wrap, [0, 0], "the ramp states no wrap mode: black");
        assert_eq!(t.tiles, [0]);
        assert_eq!(t.levels.len(), 4);
        assert_eq!(t.levels[0].start, 40);
        let bytes = levels(&t, 0);
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
        assert_eq!(levels(&t, 0), RAMP[40..], "the levels, not the header");

        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(RAMP.len() as u64 - 8).unwrap();
        let mut bytes = vec![0; t.bytes(0) as usize];
        assert!(
            t.read_levels(0, &mut bytes).is_err(),
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
            assert_eq!(t.bytes(0) as usize, size);
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
    fn a_cap_starts_at_the_first_level_within_it() {
        let t = parse(RAMP.to_vec()).unwrap();
        assert_eq!(t.first(4), 1);
        assert_eq!(t.extent(1), (4, 4));
        assert_eq!(t.bytes(1), 680 - 8 * 8 * 8);
        // The ramp's level 1, which holds its own texel centres at 4×4.
        let bytes = levels(&t, 1);
        assert_eq!(bytes, RAMP[40 + 8 * 8 * 8..], "the levels from 4×4 on");
        let texel = |x: usize, y: usize| {
            let at = (y * 4 + x) * 8;
            let f = |i: usize| f32::from_le_bytes(bytes[at + i..at + i + 4].try_into().unwrap());
            [f(0), f(4)]
        };
        assert_eq!(texel(1, 2), [1.5 / 4.0, 2.5 / 4.0]);
        assert_eq!((t.first(3), t.first(2), t.first(1)), (2, 2, 3));

        // The controls: a cap of none, or one the source fits, keeps all.
        for cap in [0, 8, 4096] {
            assert_eq!(t.first(cap), 0, "cap {cap}");
        }

        // A 5×3 image: its levels are 5×3, 2×1 and 1×1, so a cap of 4 or
        // 3 starts at 2×1, as the bake tool's halving once did.
        let mut file = b"pika".to_vec();
        for word in [5u32, 3, 1, 3, 100, 0, 0, 1, 0] {
            file.extend_from_slice(&word.to_le_bytes());
        }
        file.resize(40 + 4 * (15 + 2 + 1), 0);
        let t = parse(file).unwrap();
        for cap in [4, 3, 2] {
            assert_eq!(t.first(cap), 1, "cap {cap}");
            assert_eq!(t.extent(1), (2, 1));
            assert_eq!(t.levels[1], 100..108);
        }
        assert_eq!(t.first(1), 2);
    }

    #[test]
    fn a_load_past_the_frames_bytes_starts_coarser() {
        let t = parse(RAMP.to_vec()).unwrap();
        // The ramp's levels from 8×8, 4×4, 2×2 and 1×1 on.
        assert_eq!([0, 1, 2, 3].map(|first| t.bytes(first)), [680, 168, 40, 8]);
        assert_eq!(fit(&t, 0, UPLOAD), 0, "the frame's first upload");
        assert_eq!(fit(&t, 0, 680), 0, "the control: every level fits");
        assert_eq!(fit(&t, 0, 679), 1);
        assert_eq!(fit(&t, 0, 40), 2);
        assert_eq!(fit(&t, 0, 0), 3, "the last level, whatever is left");
        assert_eq!(fit(&t, 2, 168), 2, "never finer than the target");
        assert_eq!(fit(&t, 1, UPLOAD - 1), 1);
    }

    #[test]
    fn block_formats_pad_to_whole_blocks() {
        assert_eq!(level_size(vk::Format::BC7_SRGB_BLOCK, 5, 1), Some(32));
        assert_eq!(level_size(vk::Format::BC4_UNORM_BLOCK, 4, 4), Some(8));
        assert_eq!(level_size(vk::Format::R32G32_SFLOAT, 5, 1), Some(40));
        assert_eq!(level_size(vk::Format::R8G8B8A8_UNORM, 4, 4), None);
    }
}
