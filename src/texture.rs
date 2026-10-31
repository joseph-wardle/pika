//! Baked textures: the file `bake texture` writes for one image under the
//! cache (`bake/texture.cpp`), as the renderer reads it, the image it is
//! uploaded to, and the set of them loaded for the materials drawn, with
//! the samplers they are read through. The file is a 24-byte header, `pika`
//! and five little-endian u32 (width, height, layers, levels, the VkFormat),
//! then every mip level in order, each layer within it, so an upload is one
//! copy per level.

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

/// A texture on the GPU: a 2D array image with every level, in the layout
/// the material pass samples once its upload has run.
struct Image {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    bytes: u64,
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
        let mut i = Self {
            device: d.clone(),
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            bytes: 0,
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
        // size then, which a copy's buffer offset must be; the header's 24
        // bytes are not one.
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let levels = &texture.file[HEADER..];
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
                    .buffer_offset((bytes.start - HEADER) as u64)
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

const HEADER: usize = 24;

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
    /// By address modes, u then v (`surface::address_modes`).
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

    /// The view of the baked texture at `path`, loaded now unless it is.
    pub fn load(
        &mut self,
        device: &gpu::Device,
        cb: vk::CommandBuffer,
        path: &Path,
    ) -> Result<vk::ImageView, String> {
        if let Some(image) = self.images.get(path) {
            return Ok(image.view);
        }
        let texture = read(path)?;
        let (image, staging) = Image::new(device, &texture)?;
        image.record(cb, &staging, &texture);
        self.staging.push(staging);
        let view = image.view;
        self.images.insert(path.to_owned(), image);
        Ok(view)
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

    /// The images' bytes, and the staging buffers' on the host side.
    pub fn memory(&self) -> gpu::Memory {
        let mut memory = gpu::Memory {
            device: self.images.values().map(|i| i.bytes).sum(),
            host: 0,
        };
        for b in &self.staging {
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
