//! Baked texture files and the GPU images holding each one's levels within the
//! cap, lowered to fit the device's memory budget.
//!
//! A level is an index into a texture's mips, 0 the finest, so a greater level is coarser and
//! lowering a texture raises its first level. Each frame, in `shade`'s order: `begin` drops
//! what the frame before retired; `need` and `weigh` note the finest level each drawn texture
//! needs; `refit` sets every texture's target level so that all fit; `swap` moves images toward
//! their targets, a raise's bytes arriving from the reader threads over several frames while
//! its old image is still drawn; the materials of the keys swapped are then rebound.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use ash::vk;

use crate::gpu::{self, Buffer};
use crate::need;

pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub format: vk::Format,
    /// u then v: 0 black, 1 clamp, 2 periodic, 3 mirror; black when unstated.
    pub wrap: [u32; 2],
    pub tiles: Vec<u32>,
    /// File byte ranges, all layers, finest first.
    pub levels: Vec<Range<usize>>,
    pub contents: Contents,
}

pub enum Contents {
    /// Opened for each read, so a build holds no level bytes in memory. Not kept open:
    /// a scene holds thousands of textures, past hosts' usual limit of 1024 open files.
    File(PathBuf),
    Memory(Vec<u8>),
}

impl Texture {
    /// The first level within `cap` texels a side (0 is no cap); the last always is.
    pub fn first(&self, cap: u32) -> usize {
        let over = |&level: &usize| {
            let (w, h) = self.extent(level);
            cap > 0 && w.max(h) > cap
        };
        (0..self.levels.len() - 1).take_while(over).count()
    }

    pub fn extent(&self, level: usize) -> (u32, u32) {
        ((self.width >> level).max(1), (self.height >> level).max(1))
    }

    pub fn bytes(&self, first: usize) -> u64 {
        self.levels[first..].iter().map(|l| l.len() as u64).sum()
    }

    pub fn read_at(&self, from: usize, into: &mut [u8]) -> std::io::Result<()> {
        match &self.contents {
            Contents::File(path) => read_file(path, from as u64, into),
            Contents::Memory(file) => {
                into.copy_from_slice(&file[from..from + into.len()]);
                Ok(())
            }
        }
    }
}

fn read_file(path: &Path, from: u64, into: &mut [u8]) -> std::io::Result<()> {
    File::open(path)?.read_exact_at(into, from)
}

fn unread(e: std::io::Error) -> String {
    format!("cannot read its levels: {e}")
}

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

/// Drawn for a missing file; periodic, so a node reading the file's wrap modes
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

pub fn read(path: &Path) -> Result<Texture, String> {
    let cannot = |e: std::io::Error| format!("cannot read {}: {e}", path.display());
    let file = File::open(path).map_err(cannot)?;
    let len = file.metadata().map_err(cannot)?.len() as usize;
    let mut head = vec![0; len.min(HEADER)];
    file.read_exact_at(&mut head, 0).map_err(cannot)?;
    header(&head, len, Contents::File(path.into())).map_err(|e| format!("{} {e}", path.display()))
}

#[cfg(test)]
pub fn parse(file: Vec<u8>) -> Result<Texture, String> {
    let head = file[..file.len().min(HEADER)].to_vec();
    header(&head, file.len(), Contents::Memory(file))
}

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
            "has the format {format:?}, which the texture tool does not write"
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

/// One layer's bytes; block formats pad the edges to whole 4×4 blocks.
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

fn block_rows(format: vk::Format) -> u32 {
    match format {
        vk::Format::BC4_UNORM_BLOCK | vk::Format::BC7_UNORM_BLOCK | vk::Format::BC7_SRGB_BLOCK => 4,
        _ => 1,
    }
}

/// Copies of whole block rows of one layer, `budget` bytes of the levels from
/// `first` on starting at byte `done`; buffer offsets count from `done`.
fn copies(
    texture: &Texture,
    first: usize,
    done: u64,
    budget: u64,
) -> (Vec<vk::BufferImageCopy>, u64) {
    let base = texture.levels[first].start as u64;
    let rows = block_rows(texture.format);
    let (mut copies, mut at) = (Vec::new(), done);
    let holds = |at: u64, l: usize| texture.levels[l].end as u64 - base > at;
    while let Some(level) = (first..texture.levels.len()).find(|&l| holds(at, l)) {
        let (width, height) = texture.extent(level);
        let range = &texture.levels[level];
        let layer_bytes = range.len() as u64 / u64::from(texture.layers);
        let bands = u64::from(height.div_ceil(rows));
        let band_bytes = layer_bytes / bands;
        let within = at - (range.start as u64 - base);
        let (layer, band) = (within / layer_bytes, within % layer_bytes / band_bytes);
        let count = (budget.saturating_sub(at - done) / band_bytes).min(bands - band);
        if count == 0 {
            break;
        }
        let y = band as u32 * rows;
        copies.push(
            vk::BufferImageCopy::default()
                .buffer_offset(at - done)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: (level - first) as u32,
                    base_array_layer: layer as u32,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D {
                    x: 0,
                    y: y as i32,
                    z: 0,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height: (count as u32 * rows).min(height - y),
                    depth: 1,
                }),
        );
        at += count * band_bytes;
    }
    (copies, at)
}

/// Level bytes a frame uploads: reading and copying costs the render thread about
/// 0.8 ms a MiB (RTX 4070 Ti SUPER). Larger raises span frames.
pub const UPLOAD: u64 = 32 << 20;

/// Bytes a swap's read asks for, so a frame's `UPLOAD` takes several, read from
/// different files at once.
const CHUNK: u64 = 8 << 20;

/// Bytes asked of the readers and not yet uploaded: the most their reads hold in
/// host memory.
const AHEAD: u64 = 2 * UPLOAD;

/// The file server answers parallel reads faster than one (hypothesis: four are
/// enough on NFS).
const READERS: usize = 4;

/// Longest side of the coarsest level kept: one 4×4 block, since each coarser
/// level takes a block too.
pub const TAIL: u32 = 4;

/// Budget fitting leaves free, for a raise's old image beside its new one and for
/// the budget to drift without moving targets. Hypothesis: enough on ALab.
pub const MARGIN: u64 = 256 << 20;

/// Budget a swap leaves free: the driver can refuse an allocation the budget
/// allows (RTX 4070 Ti SUPER, with 1 MB left). Hypothesis: enough elsewhere.
pub const RESERVE: u64 = 64 << 20;

struct Image {
    // The owner's device outlives this (field order).
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    bytes: u64,
    heap: usize,
    first: usize,
}

struct Loaded {
    texture: Texture,
    tiles: Buffer,
    image: Image,
    /// Replaces `image` once whole.
    incoming: Option<Incoming>,
    chain: Chain,
    /// The first level `fit` gives its image.
    target: usize,
    /// Per level, the last frame it was the finest a drawn reader needed; 0 never.
    seen: Vec<u64>,
    /// The last frame a reader with no estimated need was drawn; 0 never.
    unestimated: u64,
    /// A lower failed: retried each frame, warned of once.
    failing: bool,
}

impl Loaded {
    /// Whether its image is coarser than the finest level a view needed, within `cap`.
    fn blurred(&self, cap: u32) -> bool {
        let need = self
            .chain
            .need
            .map(|level| level.max(self.texture.first(cap)));
        need.is_some_and(|level| level < self.image.first)
    }
}

/// Level bytes before `done` are copied.
struct Incoming {
    image: Image,
    done: u64,
}

/// Counts of loaded textures, within `need::WINDOW`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Needs {
    pub unseen: usize,
    pub unestimated: usize,
    pub blurred: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct Chain {
    heap: usize,
    side: u32,
    /// Image bytes from each first level, as the driver sizes them.
    sizes: Vec<u64>,
    /// The finest level allowed: the cap's, or the image's own while a failed
    /// raise is held.
    ceiling: usize,
    /// The coarsest: the first within `TAIL`, or the cap's past it.
    floor: usize,
    /// Within `need::WINDOW`; 0 when a reader's need was not estimated, `None`
    /// when no view drew it.
    need: Option<usize>,
    /// The heap's room when a raise failed; holds it until `released`.
    held: Option<u64>,
}

impl Chain {
    fn new(device: &gpu::Device, texture: &Texture, cap: u32) -> Result<Self, String> {
        let mut chain = Chain {
            heap: 0,
            side: texture.width.max(texture.height),
            sizes: Vec::new(),
            ceiling: 0,
            floor: 0,
            need: None,
            held: None,
        };
        for first in 0..texture.levels.len() {
            let info = image_info(texture, first);
            let ask = vk::DeviceImageMemoryRequirements::default().create_info(&info);
            let mut requirements = vk::MemoryRequirements2::default();
            // SAFETY: `device.raw` is a live Vulkan 1.3 device, and `info`
            // outlives the call.
            unsafe {
                device
                    .raw
                    .get_device_image_memory_requirements(&ask, &mut requirements)
            };
            let requirements = requirements.memory_requirements;
            let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            chain.heap = device.heap(requirements.memory_type_bits, local)?;
            chain.sizes.push(requirements.size);
        }
        chain.limit(texture, cap);
        Ok(chain)
    }

    fn limit(&mut self, texture: &Texture, cap: u32) {
        self.ceiling = texture.first(cap);
        self.floor = texture.first(TAIL).max(self.ceiling);
        self.held = None;
    }

    /// A failed lower keeps its target, as other lowers free memory; a failed
    /// raise is held at `first`, else it would fail every frame.
    fn fail(&mut self, first: usize, target: usize, room: u64) -> usize {
        if target > first {
            return target;
        }
        (self.ceiling, self.held) = (first, Some(room));
        first
    }

    /// Only room grown past `held` by more than `MARGIN` releases it, so budget
    /// drift retries nothing.
    fn released(&self, room: &[u64], changed: bool) -> bool {
        (self.held).is_some_and(|held| changed || room[self.heap] > held.saturating_add(MARGIN))
    }

    fn extent(&self, level: usize) -> u32 {
        (self.side >> level).max(1)
    }
}

/// Lowering priority, greatest first: undrawn chains, then levels finer than the
/// need, then the chain least far below its need (so blur spreads evenly), then
/// the farthest; ties go to the longest side, then the most bytes.
fn order(chain: &Chain, level: usize) -> impl Ord {
    let (seen, below, need) = match chain.need {
        None => (0, 0, 0),
        Some(need) if level < need => (1, 0, need),
        Some(need) => (2, level - need, need),
    };
    let tier = (Reverse(seen), Reverse(below), need);
    (tier, chain.extent(level), chain.sizes[level])
}

/// A chain's place in `order` hangs on its own level alone, so more capacity
/// stops earlier in the same sequence and never lowers a chain.
fn fit(chains: &[&Chain], capacity: &[u64]) -> Vec<usize> {
    let mut targets: Vec<usize> = chains.iter().map(|c| c.ceiling).collect();
    let mut totals = totals(chains, &targets, capacity.len());
    let key = |i: usize, level: usize| (order(chains[i], level), Reverse(i));
    let mut queue: BinaryHeap<_> = (0..chains.len())
        .filter(|&i| chains[i].ceiling < chains[i].floor)
        .map(|i| key(i, chains[i].ceiling))
        .collect();
    while let Some((_, Reverse(i))) = queue.pop() {
        let (c, target) = (chains[i], &mut targets[i]);
        if totals[c.heap] <= capacity[c.heap] {
            continue;
        }
        totals[c.heap] -= c.sizes[*target];
        *target += 1;
        totals[c.heap] += c.sizes[*target];
        if *target < c.floor {
            queue.push(key(i, *target));
        }
    }
    targets
}

/// Refits under `room` less `MARGIN` when `afresh` or when raising `current` would
/// not fit; else only raises, so room wobbling under the margin lowers nothing.
fn settle(chains: &[&Chain], current: &[usize], room: &[u64], afresh: bool) -> (Vec<usize>, bool) {
    let capacity: Vec<u64> = room.iter().map(|r| r.saturating_sub(MARGIN)).collect();
    let fitted = fit(chains, &capacity);
    let kept: Vec<usize> = current
        .iter()
        .zip(&fitted)
        .map(|(&c, &f)| c.min(f))
        .collect();
    let taken = totals(chains, &kept, room.len());
    if afresh || taken.iter().zip(room).any(|(t, r)| t > r) {
        return (fitted, true);
    }
    (kept, false)
}

/// Lowers first, as they free memory; then begun raises, so they finish before
/// others take memory; then the chain `fit` would lower last.
fn queue(chains: &[&Chain], firsts: &[usize], targets: &[usize], begun: &[bool]) -> Vec<usize> {
    let mut queue: Vec<usize> = (0..chains.len())
        .filter(|&i| begun[i] || targets[i] != firsts[i])
        .collect();
    queue.sort_by_key(|&i| {
        let lower = targets[i] > firsts[i];
        (
            Reverse(lower),
            Reverse(begun[i]),
            order(chains[i], firsts[i]),
        )
    });
    queue
}

fn totals(chains: &[&Chain], targets: &[usize], heaps: usize) -> Vec<u64> {
    let mut totals = vec![0; heaps];
    for (c, &target) in chains.iter().zip(targets) {
        totals[c.heap] += c.sizes[target];
    }
    totals
}

/// `tiles`' range is the table's own length, by which the shader tells a plain
/// image from a set (a buffer is at least 16 bytes).
#[derive(Clone, Copy)]
pub struct Binding {
    pub view: vk::ImageView,
    pub tiles: vk::DescriptorBufferInfo,
    pub wrap: [u32; 2],
}

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
        .usage(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
}

impl Image {
    fn new(device: &gpu::Device, texture: &Texture, first: usize) -> Result<Self, String> {
        let d = &device.raw;
        let info = image_info(texture, first);
        let mut i = Self {
            device: d.clone(),
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            bytes: 0,
            heap: 0,
            first,
        };
        // SAFETY (all calls below): `d` is live, infos outlive their calls, handles
        // come from `d`; `drop` destroys what was made before an error.
        i.image = unsafe { d.create_image(&info, None) }.map_err(|e| {
            let (width, height) = texture.extent(first);
            format!(
                "vkCreateImage ({width}×{height} {:?}) failed: {e}",
                texture.format
            )
        })?;
        let requirements = unsafe { d.get_image_memory_requirements(i.image) };
        i.bytes = requirements.size;
        let flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        i.heap = device.heap(requirements.memory_type_bits, flags)?;
        i.memory = gpu::allocate(
            device,
            requirements,
            flags,
            vk::MemoryAllocateFlags::empty(),
        )?;
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

    fn to_copy(&self, cb: vk::CommandBuffer, texture: &Texture) {
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(self.image)
            .subresource_range(self.range(texture));
        self.barrier(cb, barrier);
    }

    fn to_sample(&self, cb: vk::CommandBuffer, texture: &Texture) {
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.image)
            .subresource_range(self.range(texture));
        self.barrier(cb, barrier);
    }

    fn barrier(&self, cb: vk::CommandBuffer, barrier: vk::ImageMemoryBarrier2) {
        let barriers = [barrier];
        let info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        // SAFETY: `cb` is recording outside rendering on `device`, and the
        // barrier names this image's own levels.
        unsafe { self.device.cmd_pipeline_barrier2(cb, &info) };
    }

    /// `staging` must live until the GPU has run `cb`.
    fn copy(&self, cb: vk::CommandBuffer, staging: &Buffer, copies: &[vk::BufferImageCopy]) {
        let layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        // SAFETY: `cb` records outside rendering, the image has TRANSFER_DST usage,
        // and `staging` holds the levels at the offsets copied from.
        unsafe {
            (self.device).cmd_copy_buffer_to_image(cb, staging.raw, self.image, layout, copies)
        };
    }

    fn copy_from(&self, cb: vk::CommandBuffer, old: &Image, texture: &Texture) {
        let layers = |level: usize| vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: level as u32,
            base_array_layer: 0,
            layer_count: texture.layers,
        };
        let regions: Vec<vk::ImageCopy> = (self.first..texture.levels.len())
            .map(|level| {
                let (width, height) = texture.extent(level);
                vk::ImageCopy::default()
                    .src_subresource(layers(level - old.first))
                    .dst_subresource(layers(level - self.first))
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
            })
            .collect();
        let (sampled, source) = (
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        let out = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(sampled)
            .new_layout(source)
            .image(old.image)
            .subresource_range(old.range(texture));
        old.barrier(cb, out);
        self.to_copy(cb, texture);
        let layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        // SAFETY: `cb` records outside rendering, both images have the transfer
        // usages, each region exists in both, and the barriers set the layouts.
        unsafe {
            (self.device).cmd_copy_image(cb, old.image, source, self.image, layout, &regions)
        };
        let back = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(source)
            .new_layout(sampled)
            .image(old.image)
            .subresource_range(old.range(texture));
        old.barrier(cb, back);
        self.to_sample(cb, texture);
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

/// One `UPLOAD`-byte buffer kept across frames, since a buffer per texture cost
/// several times its read to create; a load past it gets its own.
struct Staging {
    buffer: Buffer,
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

    fn free(&self) -> u64 {
        (self.buffer.size).saturating_sub(self.used.next_multiple_of(16))
    }

    fn reserve(&mut self, device: &gpu::Device, size: u64) -> Result<(&mut Buffer, u64), String> {
        // A copy's buffer offset is a multiple of the block size, at most 16 bytes.
        let at = self.used.next_multiple_of(16);
        if at + size <= self.buffer.size {
            self.used = at + size;
            return Ok((&mut self.buffer, at));
        }
        self.own.push(Buffer::host(device, size, STAGING, HOST)?);
        Ok((self.own.last_mut().unwrap(), 0))
    }
}

/// Stages the level bytes `part`, as `copies` cuts them, from `fill`; the time it took.
fn upload(
    device: &gpu::Device,
    cb: vk::CommandBuffer,
    (texture, image): (&Texture, &Image),
    part: Range<u64>,
    staging: &mut Staging,
    fill: impl FnOnce(&mut [u8]) -> Result<(), String>,
) -> Result<Duration, String> {
    let size = part.end - part.start;
    let (mut copies, end) = copies(texture, image.first, part.start, size);
    debug_assert_eq!(end, part.end, "a part not cut by `copies`");
    // Stage the levels alone: the header would misalign the block-size offsets.
    let (buffer, at) = staging.reserve(device, size)?;
    let start = Instant::now();
    fill(&mut buffer.bytes()[at as usize..(at + size) as usize])?;
    let fill = start.elapsed();
    for copy in &mut copies {
        copy.buffer_offset += at;
    }
    if part.start == 0 {
        image.to_copy(cb, texture);
    }
    image.copy(cb, buffer, &copies);
    if end == texture.bytes(image.first) {
        image.to_sample(cb, texture);
    }
    Ok(fill)
}

/// A part of a swap's levels, `part` of them from file offset `from`.
struct Read {
    key: String,
    first: usize,
    part: Range<u64>,
    from: u64,
}

impl Read {
    fn size(&self) -> u64 {
        self.part.end - self.part.start
    }
}

type Arrived = (Read, Result<Vec<u8>, String>);

/// A part by texture, first level and start.
type Part = (String, usize, u64);

/// Reads swaps' levels on worker threads, so a frame never waits on a file;
/// the render thread copies what arrived into staging.
struct Reads {
    jobs: Sender<(Read, PathBuf)>,
    done: Receiver<Arrived>,
    /// The parts asked for and not yet taken or dropped.
    asked: BTreeSet<Part>,
    /// A stale read must not take a current one's place.
    arrived: BTreeMap<Part, Arrived>,
    /// Bytes asked for and not yet taken or dropped.
    ahead: u64,
}

impl Reads {
    fn new() -> Self {
        let (jobs, queue) = mpsc::channel::<(Read, PathBuf)>();
        let queue = Arc::new(Mutex::new(queue));
        let (finished, done) = mpsc::channel();
        for _ in 0..READERS {
            let (queue, finished) = (queue.clone(), finished.clone());
            // Exits once `jobs` drops with its owner.
            std::thread::spawn(move || {
                loop {
                    let job = queue.lock().unwrap_or_else(PoisonError::into_inner).recv();
                    let Ok((read, path)) = job else {
                        return;
                    };
                    let mut bytes = vec![0; read.size() as usize];
                    let result = (read_file(&path, read.from, &mut bytes))
                        .map(|()| bytes)
                        .map_err(unread);
                    if finished.send((read, result)).is_err() {
                        return;
                    }
                }
            });
        }
        Self {
            jobs,
            done,
            asked: BTreeSet::new(),
            arrived: BTreeMap::new(),
            ahead: 0,
        }
    }

    /// Asks for the part of the levels from `first` that starts at byte `done`, unless
    /// it was asked for or `AHEAD` bytes are out; whether it is asked for now.
    fn ask(&mut self, key: &str, texture: &Texture, (first, done): (usize, u64)) -> bool {
        let part = (key.to_owned(), first, done);
        if self.asked.contains(&part) {
            return true;
        }
        if self.ahead >= AHEAD {
            return false;
        }
        self.asked.insert(part);
        let (_, end) = copies(texture, first, done, CHUNK);
        let read = Read {
            key: key.to_owned(),
            first,
            part: done..end,
            from: texture.levels[first].start as u64 + done,
        };
        self.ahead += read.size();
        let path = match &texture.contents {
            Contents::File(path) => path.clone(),
            Contents::Memory(_) => {
                let mut bytes = vec![0; read.size() as usize];
                let result = (texture.read_at(read.from as usize, &mut bytes))
                    .map(|()| bytes)
                    .map_err(unread);
                self.land(read, result);
                return true;
            }
        };
        if let Err(mpsc::SendError((read, _))) = self.jobs.send((read, path)) {
            self.land(read, Err("the texture readers have exited".into()));
        }
        true
    }

    fn land(&mut self, read: Read, bytes: Result<Vec<u8>, String>) {
        let at = (read.key.clone(), read.first, read.part.start);
        if let Some((old, _)) = self.arrived.insert(at, (read, bytes)) {
            self.ahead -= old.size();
        }
    }

    /// Lands the finished reads and drops those no swap waits for.
    fn gather(&mut self, wanted: impl Fn(&Read) -> bool) {
        while let Ok((read, bytes)) = self.done.try_recv() {
            self.land(read, bytes);
        }
        let mut dropped = 0;
        self.arrived.retain(|part, (read, _)| {
            let keep = wanted(read);
            if !keep {
                dropped += read.size();
                self.asked.remove(part);
            }
            keep
        });
        self.ahead -= dropped;
    }

    /// The part asked for at `(first, done)`, once it arrived, if it holds at most `room`
    /// bytes.
    fn take(&mut self, key: &str, (first, done): (usize, u64), room: u64) -> Option<Arrived> {
        let part = (key.to_owned(), first, done);
        if self.arrived.get(&part)?.0.size() > room {
            return None;
        }
        let arrived = self.arrived.remove(&part).expect("got above");
        self.asked.remove(&part);
        self.ahead -= arrived.0.size();
        Some(arrived)
    }
}

fn fail(key: &str, loaded: &mut Loaded, room: u64, e: &str) {
    let first = loaded.image.first;
    let (w, h) = loaded.texture.extent(first);
    loaded.target = loaded.chain.fail(first, loaded.target, room);
    if loaded.target == first {
        eprintln!(
            "pika: warning: texture {key}: {e}; held at {w}×{h} until the memory budget grows \
             or the textures change"
        );
    } else if !loaded.failing {
        eprintln!("pika: warning: texture {key}: {e}; kept at {w}×{h}, lowered again each frame");
    }
    loaded.failing = loaded.target != first;
}

/// What a frame's swaps did.
#[derive(Default)]
pub struct Swapped {
    /// The textures drawn from a new image, to rebind.
    pub keys: BTreeSet<String>,
    /// Bytes copied into raises.
    pub bytes: u64,
    /// The render thread's time copying them into staging.
    pub read: Duration,
}

/// What a swap not yet begun does this frame.
#[derive(PartialEq, Debug)]
enum Start {
    Make,
    /// Until the images this frame retired are freed.
    Wait,
    /// Until a refit leaves `MARGIN`.
    Crowded,
    Refuse,
}

/// A swap whose new image `fits` beside the old is made. A lower frees memory once its old
/// image goes, so it is made anyway unless the frame already `retired` an image, whose memory
/// it then waits for. A raise waits for that too, then for a `fresh` fit; after one, no more
/// room can be made.
fn start(fits: bool, lowers: bool, retired: bool, fresh: bool) -> Start {
    if fits || (lowers && !retired) {
        Start::Make
    } else if retired {
        Start::Wait
    } else if !fresh {
        Start::Crowded
    } else {
        Start::Refuse
    }
}

/// Copies into a raise's incoming image the next part the readers have read, and asks them
/// for the part after. Gives the bytes copied and the time filling staging; `None` when the
/// part has not arrived or staging has no room for it this frame.
fn advance(
    device: &gpu::Device,
    cb: vk::CommandBuffer,
    key: &str,
    loaded: &mut Loaded,
    reads: &mut Reads,
    staging: &mut Staging,
) -> Result<Option<(u64, Duration)>, String> {
    let incoming = loaded.incoming.as_mut().expect("a raise begun");
    let first = incoming.image.first;
    let at = (first, incoming.done);
    reads.ask(key, &loaded.texture, at);
    let Some((asked, arrived)) = reads.take(key, at, staging.free()) else {
        return Ok(None);
    };
    let end = asked.part.end;
    let fill = |into: &mut [u8]| {
        into.copy_from_slice(&arrived?);
        Ok(())
    };
    let both = (&loaded.texture, &incoming.image);
    let time = upload(device, cb, both, asked.part.clone(), staging, fill)?;
    incoming.done = end;
    if end != loaded.texture.bytes(first) {
        reads.ask(key, &loaded.texture, (first, end));
    }
    Ok(Some((asked.size(), time)))
}

/// Staging is kept until the next frame begins, when the GPU has read it.
pub struct Textures {
    device: ash::Device,
    loaded: BTreeMap<String, Loaded>,
    samplers: BTreeMap<[u32; 2], vk::Sampler>,
    staging: Staging,
    reads: Reads,
    /// Kept until the next frame begins, when every material binding them is rebound.
    retired: Vec<Image>,
    /// Texels a side; 0 is no cap.
    cap: u32,
    heaps: Vec<gpu::Heap>,
    /// Each heap's room for textures when the targets were last set.
    room: Vec<u64>,
    // Four flags carry the fit from frame to frame. `refit` runs only when `dirty`, `crowded`
    // or `moved` is set or the heaps changed, and clears the three; `fresh` says whether it
    // then fitted afresh, and `begin` clears it. In `swap`, a raise that does not fit sets
    // `crowded` for the next frame's refit, unless this frame's was `fresh`: then no more
    // room can be made, and the raise fails.
    /// The textures or limits changed: fit afresh and release held raises.
    dirty: bool,
    /// A raise did not fit beside its old image: fit afresh, leaving `MARGIN`.
    crowded: bool,
    /// This frame's targets were fitted afresh, leaving `MARGIN` for its raises.
    fresh: bool,
    short: u64,
    frame: u64,
    /// A need changed since the targets were last set.
    moved: bool,
}

impl Textures {
    pub fn new(device: &gpu::Device) -> Result<Self, String> {
        Ok(Self {
            device: device.raw.clone(),
            loaded: BTreeMap::new(),
            samplers: BTreeMap::new(),
            staging: Staging::new(device)?,
            reads: Reads::new(),
            retired: Vec::new(),
            cap: 0,
            heaps: Vec::new(),
            room: Vec::new(),
            dirty: false,
            crowded: false,
            fresh: false,
            short: 0,
            frame: 0,
            moved: false,
        })
    }

    /// No GPU work may be pending.
    pub fn begin(&mut self) {
        (self.staging.used, self.staging.own) = (0, Vec::new());
        self.retired.clear();
        self.fresh = false;
        self.frame += 1;
    }

    pub fn set_cap(&mut self, cap: u32) -> bool {
        if cap == self.cap {
            return false;
        }
        self.cap = cap;
        for loaded in self.loaded.values_mut() {
            loaded.chain.limit(&loaded.texture, cap);
        }
        self.dirty = true;
        true
    }

    /// Loads at its floor unless loaded; the bytes uploaded and the read time.
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
        let chain = Chain::new(device, &texture, self.cap).map_err(|e| format!("{key}: {e}"))?;
        let first = chain.floor;
        let image = Image::new(device, &texture, first).map_err(|e| format!("{key}: {e}"))?;
        // A load is a few blocks, read here: its image is drawn this frame.
        let (size, from) = (texture.bytes(first), texture.levels[first].start);
        let read = |into: &mut [u8]| texture.read_at(from, into).map_err(unread);
        let both = (&texture, &image);
        let read = upload(device, cb, both, 0..size, &mut self.staging, read)
            .map_err(|e| format!("{key}: {e}"))?;
        let loaded = Loaded {
            seen: vec![0; texture.levels.len()],
            unestimated: 0,
            texture,
            tiles,
            image,
            incoming: None,
            chain,
            target: first,
            failing: false,
        };
        self.loaded.insert(key.to_owned(), loaded);
        self.dirty = true;
        Ok((size, read))
    }

    /// A heap's room for textures is its budget less what the process holds besides them.
    pub fn refit(&mut self, heaps: &[gpu::Heap]) {
        if !self.dirty && !self.crowded && !self.moved && self.heaps == heaps {
            return;
        }
        self.moved = false;
        self.heaps = heaps.to_vec();
        let mut ours = vec![0; heaps.len()];
        for image in self.images() {
            ours[image.heap] += image.bytes;
        }
        let room: Vec<u64> = (heaps.iter().zip(&ours))
            .map(|(h, &ours)| match h.allowed {
                0 => u64::MAX,
                allowed => allowed.saturating_sub(h.used.saturating_sub(ours)),
            })
            .collect();
        for loaded in self.loaded.values_mut() {
            if loaded.chain.released(&room, self.dirty) {
                loaded.chain.limit(&loaded.texture, self.cap);
            }
        }
        let chains: Vec<&Chain> = self.loaded.values().map(|l| &l.chain).collect();
        let current: Vec<usize> = self.loaded.values().map(|l| l.target).collect();
        let targets;
        let afresh = self.dirty || self.crowded;
        (targets, self.fresh) = settle(&chains, &current, &room, afresh);
        let totals = totals(&chains, &targets, heaps.len());
        self.short = totals
            .iter()
            .zip(&room)
            .map(|(t, r)| t.saturating_sub(*r))
            .sum();
        for (loaded, target) in self.loaded.values_mut().zip(targets) {
            loaded.target = target;
        }
        self.room = room;
        (self.dirty, self.crowded) = (false, false);
    }

    /// Moves each image toward its target: begins the swaps that fit, and copies in the parts
    /// of the raises begun that the readers have read. The old image is drawn until the new one
    /// is whole, so a failed swap keeps it. The keys swapped must be rebound before the next
    /// frame begins.
    pub fn swap(
        &mut self,
        device: &gpu::Device,
        cb: vk::CommandBuffer,
        heaps: &[gpu::Heap],
    ) -> Swapped {
        let mut swapped = Swapped::default();
        // Per heap, the bytes of the images this frame made.
        let mut added = vec![0; heaps.len()];
        let loaded = &self.loaded;
        self.reads.gather(|read| {
            let incoming = loaded.get(&read.key).and_then(|l| l.incoming.as_ref());
            (incoming).is_some_and(|i| i.image.first == read.first && i.done == read.part.start)
        });
        let order: Vec<String> = {
            let loaded: Vec<(&String, &Loaded)> = self.loaded.iter().collect();
            let chains: Vec<&Chain> = loaded.iter().map(|(_, l)| &l.chain).collect();
            let firsts: Vec<usize> = loaded.iter().map(|(_, l)| l.image.first).collect();
            let targets: Vec<usize> = loaded.iter().map(|(_, l)| l.target).collect();
            let begun: Vec<bool> = loaded.iter().map(|(_, l)| l.incoming.is_some()).collect();
            let queue = queue(&chains, &firsts, &targets, &begun);
            queue.into_iter().map(|i| loaded[i].0.clone()).collect()
        };
        for key in order {
            let loaded = self.loaded.get_mut(&key).expect("queued from the loaded");
            let (first, target) = (loaded.image.first, loaded.target);
            let room = self.room.get(loaded.chain.heap).copied().unwrap_or(0);
            // No commands recorded this frame read its image yet.
            if (loaded.incoming.as_ref()).is_some_and(|i| i.image.first != target) {
                loaded.incoming = None;
            }
            if target == first {
                continue;
            }
            if loaded.incoming.is_none() {
                let (texture, chain) = (&loaded.texture, &loaded.chain);
                let size = chain.sizes[target];
                let mut with = added.clone();
                with[chain.heap] += size + RESERVE;
                let fits = gpu::overflow(heaps, &with).is_none();
                let lowers = target > first;
                let retired = !swapped.keys.is_empty();
                let made = match start(fits, lowers, retired, self.fresh) {
                    Start::Make => Image::new(device, texture, target),
                    Start::Wait => continue,
                    Start::Crowded => {
                        self.crowded = true;
                        continue;
                    }
                    Start::Refuse => {
                        let (w, h) = texture.extent(target);
                        Err(format!(
                            "its {w}×{h} levels do not fit the device's memory budget beside \
                             its image now"
                        ))
                    }
                };
                let image = match made {
                    Ok(image) => image,
                    Err(e) => {
                        fail(&key, loaded, room, &e);
                        self.moved = true;
                        continue;
                    }
                };
                added[chain.heap] += size;
                if lowers {
                    image.copy_from(cb, &loaded.image, &loaded.texture);
                    self.retired
                        .push(std::mem::replace(&mut loaded.image, image));
                    loaded.failing = false;
                    swapped.keys.insert(key);
                    continue;
                }
                loaded.incoming = Some(Incoming { image, done: 0 });
            }
            match advance(device, cb, &key, loaded, &mut self.reads, &mut self.staging) {
                Ok(None) => {}
                Ok(Some((bytes, time))) => {
                    (swapped.bytes, swapped.read) = (swapped.bytes + bytes, swapped.read + time);
                    let incoming = loaded.incoming.as_ref().expect("advanced above");
                    if incoming.done == loaded.texture.bytes(incoming.image.first) {
                        let new = loaded.incoming.take().expect("advanced above").image;
                        self.retired.push(std::mem::replace(&mut loaded.image, new));
                        swapped.keys.insert(key);
                    }
                }
                Err(e) => {
                    loaded.incoming = None;
                    fail(&key, loaded, room, &e);
                    self.moved = true;
                }
            }
        }
        swapped
    }

    pub fn pending(&self) -> usize {
        (self.loaded.values())
            .filter(|l| l.image.first != l.target)
            .count()
    }

    /// `None` when the reader's need is not estimated.
    pub fn need(&mut self, key: &str, uv_per_pixel: Option<f32>) {
        let Some(loaded) = self.loaded.get_mut(key) else {
            return;
        };
        match uv_per_pixel {
            Some(uv) => {
                let level = need::level(loaded.chain.side, uv);
                let last = loaded.seen.len() - 1;
                loaded.seen[level.min(last)] = self.frame;
            }
            None => loaded.unestimated = self.frame,
        }
    }

    pub fn weigh(&mut self) {
        let frame = self.frame;
        for loaded in self.loaded.values_mut() {
            let unestimated = loaded.unestimated > 0 && frame - loaded.unestimated < need::WINDOW;
            let need = match unestimated {
                true => Some(0),
                false => need::finest(&loaded.seen, frame),
            };
            self.moved |= need != loaded.chain.need;
            loaded.chain.need = need;
        }
    }

    pub fn needs(&self) -> Needs {
        let mut needs = Needs::default();
        for loaded in self.loaded.values() {
            let unestimated =
                loaded.unestimated > 0 && self.frame - loaded.unestimated < need::WINDOW;
            needs.unestimated += usize::from(unestimated);
            needs.unseen += usize::from(loaded.chain.need.is_none());
            needs.blurred += usize::from(loaded.blurred(self.cap));
        }
        needs
    }

    pub fn lowered(&self) -> usize {
        (self.loaded.values())
            .filter(|l| l.target > l.texture.first(self.cap))
            .count()
    }

    /// 0 when none is blurred.
    pub fn blurred_side(&self) -> u32 {
        (self.loaded.values())
            .filter(|l| l.blurred(self.cap))
            .map(|l| l.chain.extent(l.image.first))
            .max()
            .unwrap_or(0)
    }

    /// The bytes the targets take past the heaps' room, at their floors.
    pub fn short(&self) -> u64 {
        self.short
    }

    pub fn capped(&self) -> usize {
        (self.loaded.values())
            .filter(|l| l.texture.first(self.cap) > 0)
            .count()
    }

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

    /// No GPU work may be pending.
    pub fn retain(&mut self, used: &BTreeSet<String>) {
        let before = self.loaded.len();
        self.loaded.retain(|key, _| used.contains(key));
        self.dirty |= self.loaded.len() != before;
    }

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

    fn images(&self) -> impl Iterator<Item = &Image> {
        (self.loaded.values())
            .flat_map(|l| std::iter::once(&l.image).chain(l.incoming.as_ref().map(|i| &i.image)))
            .chain(&self.retired)
    }

    pub fn memory(&self) -> gpu::Memory {
        let images = self.images();
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
        t.read_at(t.levels[0].start, &mut texel).unwrap();
        assert_eq!(f32::from_le_bytes(texel[4..8].try_into().unwrap()), 0.4);
    }

    fn levels(t: &Texture, first: usize) -> Vec<u8> {
        let mut bytes = vec![0; t.bytes(first) as usize];
        t.read_at(t.levels[first].start, &mut bytes).unwrap();
        bytes
    }

    /// Baked from bake/fixtures/ramp.exr, whose texels hold their own centres.
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
    fn reads_the_levels_from_the_file_without_keeping_it_open() {
        let path = std::env::temp_dir().join(format!("pika_texture_{}", std::process::id()));
        std::fs::write(&path, RAMP).unwrap();
        let t = read(&path).unwrap();
        assert_eq!(levels(&t, 0), RAMP[40..], "the levels, not the header");
        let open = |path: &Path| {
            (std::fs::read_dir("/proc/self/fd").unwrap())
                .filter_map(|fd| std::fs::read_link(fd.unwrap().path()).ok())
                .filter(|target| *target == path.canonicalize().unwrap())
                .count()
        };
        assert_eq!(open(&path), 0, "the texture keeps its file open");
        // Negative control: the count sees a file held open.
        let held = File::open(&path).unwrap();
        assert_eq!(open(&path), 1);
        drop(held);

        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(RAMP.len() as u64 - 8).unwrap();
        let mut bytes = vec![0; t.bytes(0) as usize];
        assert!(
            t.read_at(t.levels[0].start, &mut bytes).is_err(),
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

    /// The readers' parts carry the file's levels, each asked for once; a part no swap waits
    /// for is dropped, and one larger than the room left stays.
    #[test]
    fn readers_read_parts_of_the_levels() {
        let path = std::env::temp_dir().join(format!("pika_reads_{}", std::process::id()));
        std::fs::write(&path, RAMP).unwrap();
        let t = read(&path).unwrap();
        let mut reads = Reads::new();
        let arrive = |reads: &mut Reads, (first, done)| {
            for _ in 0..2000 {
                reads.gather(|_| true);
                if reads
                    .arrived
                    .contains_key(&("ramp".to_owned(), first, done))
                {
                    return;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            panic!("the part at {:?} never arrived", (first, done));
        };
        assert!(reads.ask("ramp", &t, (0, 0)));
        assert!(reads.ask("ramp", &t, (0, 0)), "asked for already");
        assert_eq!(
            reads.ahead,
            t.bytes(0),
            "the ramp is one part, asked for once"
        );
        arrive(&mut reads, (0, 0));
        assert!(
            reads.take("ramp", (0, 0), t.bytes(0) - 1).is_none(),
            "past the room"
        );
        let (part, bytes) = reads.take("ramp", (0, 0), t.bytes(0)).unwrap();
        assert_eq!(part.part.end, t.bytes(0));
        assert_eq!(bytes.unwrap(), RAMP[40..], "the levels, not the header");
        assert_eq!(reads.ahead, 0);
        assert!(reads.asked.is_empty(), "taken");

        assert!(reads.ask("ramp", &t, (1, 0)));
        arrive(&mut reads, (1, 0));
        let room = u64::MAX;
        assert!(
            reads.take("ramp", (0, 0), room).is_none(),
            "a part of other levels"
        );
        reads.gather(|read| read.first == 0);
        assert!(
            reads.take("ramp", (1, 0), room).is_none(),
            "a part no swap waits for"
        );
        assert_eq!(reads.ahead, 0);
        assert!(reads.asked.is_empty(), "dropped");

        reads.ahead = AHEAD;
        assert!(!reads.ask("ramp", &t, (0, 0)), "past the bytes ahead");
        reads.ahead = 0;

        // Negative control: a file gone since its header was read.
        std::fs::remove_file(&path).unwrap();
        assert!(reads.ask("ramp", &t, (0, 0)));
        arrive(&mut reads, (0, 0));
        let why = reads.take("ramp", (0, 0), room).unwrap().1.unwrap_err();
        assert!(why.contains("cannot read its levels"), "{why}");
    }

    #[test]
    fn levels_follow_the_header_in_each_format() {
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
        // Two 1×1 layers, periodic in u, clamped in v, tiles 1001 and 1012.
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
        let bytes = levels(&t, 1);
        assert_eq!(bytes, RAMP[40 + 8 * 8 * 8..], "the levels from 4×4 on");
        let texel = |x: usize, y: usize| {
            let at = (y * 4 + x) * 8;
            let f = |i: usize| f32::from_le_bytes(bytes[at + i..at + i + 4].try_into().unwrap());
            [f(0), f(4)]
        };
        assert_eq!(texel(1, 2), [1.5 / 4.0, 2.5 / 4.0]);
        assert_eq!((t.first(3), t.first(2), t.first(1)), (2, 2, 3));

        // Controls: no cap, or one the source fits, keeps all.
        for cap in [0, 8, 4096] {
            assert_eq!(t.first(cap), 0, "cap {cap}");
        }

        // 5×3: levels 5×3, 2×1 and 1×1.
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

    /// A square texture of one byte a texel, with no cap.
    fn chain(heap: usize, side: u32) -> Chain {
        let levels = 32 - side.leading_zeros() as usize;
        let sizes = (0..levels)
            .map(|first| {
                (first..levels)
                    .map(|l| u64::from((side >> l) * (side >> l)))
                    .sum()
            })
            .collect();
        let floor = (0..levels).find(|&l| side >> l <= TAIL).unwrap();
        Chain {
            heap,
            side,
            sizes,
            ceiling: 0,
            floor,
            need: None,
            held: None,
        }
    }

    #[test]
    fn a_fit_lowers_the_longest_sides_first() {
        let (big, small) = (chain(0, 64), chain(0, 16));
        assert_eq!(
            (big.sizes[0], big.sizes[1], small.sizes[0]),
            (5461, 1365, 341)
        );
        let chains = [&big, &small];
        assert_eq!(fit(&chains, &[u64::MAX]), [0, 0], "the control: both fit");
        assert_eq!(fit(&chains, &[5461 + 341]), [0, 0], "exactly");
        assert_eq!(fit(&chains, &[5461 + 340]), [1, 0], "the longer side first");
        assert_eq!(fit(&chains, &[1365 + 340]), [2, 0]);
        assert_eq!((big.floor, small.floor), (4, 2));
        assert_eq!(fit(&chains, &[0]), [4, 2], "no further than the floors");
        // Equal sides: the most bytes first, then the lowest index.
        let mut pair = chain(0, 16);
        pair.sizes.iter_mut().for_each(|s| *s *= 2);
        assert_eq!(fit(&[&small, &pair], &[341 + 681]), [0, 1]);
        assert_eq!(fit(&[&pair, &small], &[341 + 681]), [1, 0]);
        assert_eq!(fit(&[&small, &small], &[681]), [1, 0]);

        for capacity in [0, 100, 341 + 340, 1365 + 340, 5000, 5461 + 340] {
            let targets = fit(&chains, &[capacity]);
            let taken = totals(&chains, &targets, 1)[0];
            assert_eq!(fit(&chains, &[taken]), targets, "under {capacity}");
        }
        // More capacity never lowers a chain.
        let mut last = fit(&chains, &[0]);
        for capacity in 0..6000 {
            let targets = fit(&chains, &[capacity]);
            assert!(
                targets.iter().zip(&last).all(|(t, l)| t <= l),
                "at {capacity}"
            );
            last = targets;
        }
    }

    #[test]
    fn a_fit_lowers_what_no_view_needs_first() {
        let (mut near, mut far, unseen) = (chain(0, 64), chain(0, 64), chain(0, 64));
        (near.need, far.need) = (Some(0), Some(2));
        let chains = [&near, &far, &unseen];
        assert_eq!(fit(&chains, &[u64::MAX]), [0, 0, 0], "the control: all fit");
        assert_eq!(fit(&chains, &[3 * 5461]), [0, 0, 0]);
        assert_eq!(fit(&chains, &[3 * 5461 - 1]), [0, 0, 1]);
        assert_eq!(fit(&chains, &[2 * 5461 + 21]), [0, 0, 4]);
        assert_eq!(fit(&chains, &[2 * 5461 + 20]), [0, 1, 4]);
        assert_eq!(fit(&chains, &[5461 + 341 + 21]), [0, 2, 4]);
        assert_eq!(fit(&chains, &[5461 + 341 + 20]), [0, 3, 4]);
        assert_eq!(fit(&chains, &[5461 + 85 + 20]), [1, 3, 4]);
        assert_eq!(fit(&chains, &[1365 + 85 + 20]), [1, 4, 4]);
        assert_eq!(fit(&chains, &[0]), [4, 4, 4], "no further than the floors");
        // Reversed control: the order follows the needs, not the index.
        (near.need, far.need) = (Some(2), Some(0));
        assert_eq!(fit(&[&near, &far, &unseen], &[5461 + 341 + 20]), [3, 0, 4]);

        // A level no view sees goes before one it does, however coarse.
        let (mut capped, mut fine) = (chain(0, 64), chain(0, 64));
        (capped.ceiling, capped.need, fine.need) = (3, Some(3), Some(1));
        assert_eq!(fit(&[&capped, &fine], &[85 + 5461 - 1]), [3, 1]);

        (near.need, far.need) = (Some(0), Some(2));
        let chains = [&near, &far, &unseen];
        let mut last = fit(&chains, &[0]);
        for capacity in (0..17_000).step_by(7) {
            let targets = fit(&chains, &[capacity]);
            let taken = totals(&chains, &targets, 1)[0];
            assert_eq!(fit(&chains, &[taken]), targets, "under {capacity}");
            assert!(
                targets.iter().zip(&last).all(|(t, l)| t <= l),
                "at {capacity}"
            );
            last = targets;
        }
    }

    #[test]
    fn budget_noise_under_the_margin_moves_nothing() {
        let chains: Vec<Chain> = [256, 128, 64, 32].map(|side| chain(0, side)).into();
        let chains: Vec<&Chain> = chains.iter().collect();
        let room = MARGIN + 20_000;
        let (start, fresh) = settle(&chains, &[0; 4], &[room], true);
        assert!(fresh);
        assert_eq!(start, [2, 1, 0, 0]);
        let mut targets = start.clone();
        let mut moves = 0;
        for step in 0..200u64 {
            let wobble = (step * 0x9e37_79b9) % (MARGIN / 2);
            let (next, fresh) = settle(&chains, &targets, &[room + wobble - MARGIN / 4], false);
            assert!(!fresh, "step {step}");
            moves += usize::from(next != targets);
            assert!(
                next.iter().zip(&targets).all(|(n, t)| n <= t),
                "never lowered"
            );
            targets = next;
        }
        assert_eq!(moves, 1, "one raise, to every ceiling");
        assert_eq!(targets, [0; 4]);
        // Control: room falling past the targets fits afresh, lower.
        let (lower, fresh) = settle(&chains, &targets, &[30_000], false);
        assert!(fresh);
        assert!(lower.iter().zip(&targets).any(|(l, t)| l > t));
        assert_eq!(settle(&chains, &targets, &[room], true), (start, true));
    }

    #[test]
    fn a_fit_keeps_to_each_heap_and_the_limits() {
        let mut capped = chain(0, 64);
        capped.ceiling = 2;
        let other = chain(1, 64);
        let chains = [&capped, &other];
        assert_eq!(
            fit(&chains, &[u64::MAX, u64::MAX]),
            [2, 0],
            "the cap's level"
        );
        assert_eq!(
            fit(&chains, &[u64::MAX, 0]),
            [2, 4],
            "the other heap's budget"
        );
        assert_eq!(fit(&chains, &[0, u64::MAX]), [4, 0]);
        // A ceiling past the floor (a cap under `TAIL`) stays put.
        let mut kept = chain(0, 64);
        (kept.ceiling, kept.floor) = (5, 5);
        assert_eq!(fit(&[&kept], &[0]), [5]);
        // A longer side on another heap is not lowered for this one's.
        let small = chain(0, 16);
        assert_eq!(fit(&[&other, &small], &[0, u64::MAX]), [0, 2]);
    }

    /// Level bytes differ from their neighbours, to see where copies put them.
    fn numbered(format: vk::Format, (width, height, layers): (u32, u32, u32)) -> Texture {
        let levels = 32 - width.max(height).leading_zeros();
        let mut file = b"pika".to_vec();
        for word in [
            width,
            height,
            layers,
            levels,
            format.as_raw() as u32,
            0,
            0,
            1,
            0,
        ] {
            file.extend_from_slice(&word.to_le_bytes());
        }
        let (mut w, mut h, mut size) = (width, height, 0);
        for _ in 0..levels {
            size += level_size(format, w, h).unwrap() * layers as usize;
            (w, h) = ((w / 2).max(1), (h / 2).max(1));
        }
        file.extend((0..size).map(|i| (i * 7 % 251) as u8));
        parse(file).unwrap()
    }

    /// Applies `copies` as `vkCmdCopyBufferToImage` does, counting each byte's writes.
    fn copied(
        t: &Texture,
        first: usize,
        copies: &[vk::BufferImageCopy],
        staging: &[u8],
        (image, writes): (&mut [u8], &mut [u32]),
    ) {
        let base = t.levels[first].start;
        let rows = block_rows(t.format);
        for c in copies {
            let level = first + c.image_subresource.mip_level as usize;
            let (w, h) = t.extent(level);
            assert_eq!(
                (c.image_offset.x, c.image_extent.width),
                (0, w),
                "whole rows"
            );
            assert_eq!(c.image_subresource.layer_count, 1);
            let row = level_size(t.format, w, 1).unwrap();
            let layer = level_size(t.format, w, h).unwrap();
            let at = t.levels[level].start - base
                + c.image_subresource.base_array_layer as usize * layer
                + c.image_offset.y as usize / rows as usize * row;
            let len = c.image_extent.height.div_ceil(rows) as usize * row;
            let from = c.buffer_offset as usize;
            image[at..at + len].copy_from_slice(&staging[from..from + len]);
            writes[at..at + len].iter_mut().for_each(|w| *w += 1);
        }
    }

    #[test]
    fn an_upload_in_parts_copies_every_byte_once() {
        for (format, shape) in [
            (vk::Format::BC7_SRGB_BLOCK, (16, 8, 2)),
            (vk::Format::BC4_UNORM_BLOCK, (13, 6, 3)),
            (vk::Format::R32_SFLOAT, (5, 3, 2)),
        ] {
            let t = numbered(format, shape);
            for first in [0, 1] {
                let (total, file) = (t.bytes(first), levels(&t, first));
                for budget in [64, 100, 333, u64::MAX] {
                    let mut image = vec![0; total as usize];
                    let mut writes = vec![0; total as usize];
                    let mut done = 0;
                    while done < total {
                        let (copies, end) = copies(&t, first, done, budget);
                        assert!(done < end && end - done <= budget, "{format:?} at {done}");
                        let staging = &file[done as usize..end as usize];
                        copied(&t, first, &copies, staging, (&mut image, &mut writes));
                        done = end;
                    }
                    let what = format!("{format:?} from level {first} in parts of {budget}");
                    assert_eq!(image, file, "{what}");
                    assert!(writes.iter().all(|&w| w == 1), "{what}: each byte once");
                }
            }
        }
        let t = numbered(vk::Format::BC7_SRGB_BLOCK, (16, 8, 2));
        let (whole, end) = copies(&t, 0, 0, u64::MAX);
        assert_eq!((whole.len(), end), (5 * 2, t.bytes(0)));
        // Controls: a budget under one block row uploads nothing; a missing copy shows.
        let (none, end) = copies(&t, 0, 0, 63);
        assert_eq!((none.len(), end), (0, 0));
        let (file, len) = (levels(&t, 0), t.bytes(0) as usize);
        let (mut image, mut writes) = (vec![0; len], vec![0; len]);
        copied(&t, 0, &whole[1..], &file, (&mut image, &mut writes));
        assert_ne!(image, file, "a layer missing");
    }

    #[test]
    fn a_swap_without_room_waits_before_it_is_refused() {
        assert_eq!(start(true, false, true, false), Start::Make);
        // A lower is made without room, since its old image's is freed.
        assert_eq!(start(false, true, false, false), Start::Make);
        assert_eq!(start(false, true, true, false), Start::Wait);
        assert_eq!(start(false, false, true, true), Start::Wait);
        assert_eq!(start(false, false, false, false), Start::Crowded);
        // Control: a raise is refused only once a fresh fit has left no more room to make.
        assert_eq!(start(false, false, false, true), Start::Refuse);
    }

    #[test]
    fn swaps_go_lowers_then_begun_then_the_most_blurred_raise() {
        let (mut near, mut far, unseen, mut lower) =
            (chain(0, 64), chain(0, 64), chain(0, 64), chain(0, 64));
        (near.need, far.need, lower.need) = (Some(0), Some(2), Some(3));
        let (firsts, targets) = ([3; 4], [0, 1, 2, 4]);
        let none = [false; 4];
        let chains = [&near, &far, &unseen, &lower];
        assert_eq!(queue(&chains, &firsts, &targets, &none), [3, 0, 1, 2]);
        // Reversed control: the order follows the needs, not the index.
        (near.need, far.need) = (Some(2), Some(0));
        let chains = [&near, &far, &unseen, &lower];
        assert_eq!(queue(&chains, &firsts, &targets, &none), [3, 1, 0, 2]);
        // One holding its target and not begun is not queued.
        let begun = [false, false, true, false];
        assert_eq!(queue(&chains, &firsts, &targets, &begun), [3, 2, 1, 0]);
        assert_eq!(queue(&chains, &firsts, &[0, 1, 3, 4], &begun), [3, 2, 1, 0]);
        assert_eq!(queue(&chains, &firsts, &[3, 1, 3, 4], &none), [3, 1]);
    }

    #[test]
    fn a_failed_raise_is_held_until_the_room_grows_and_a_failed_lower_is_not() {
        let (mut held, other) = (chain(0, 64), chain(0, 64));
        let capacity = held.sizes[3] + other.sizes[0];
        let room = 10_000;
        assert_eq!(held.fail(3, 0, room), 3, "held where it is");
        assert_eq!((held.ceiling, held.held), (3, Some(room)));
        assert_eq!(fit(&[&held, &other], &[capacity]), [3, 0]);
        assert!(!held.released(&[room], false));
        assert!(!held.released(&[room + MARGIN], false));
        // One never held is never released.
        assert!(held.released(&[room + MARGIN + 1], false));
        assert!(held.released(&[room], true));
        assert!(!other.released(&[u64::MAX], true));
        // Control: released, it competes for the room again.
        held.limit(&numbered(vk::Format::R32_SFLOAT, (64, 64, 1)), 0);
        assert_eq!((held.ceiling, held.held), (0, None));
        assert_eq!(fit(&[&held, &other], &[capacity]), [1, 1]);
        let mut lowered = chain(0, 64);
        assert_eq!(lowered.fail(1, 3, room), 3);
        assert_eq!((lowered.ceiling, lowered.held), (0, None));
    }

    #[test]
    fn block_formats_pad_to_whole_blocks() {
        assert_eq!(level_size(vk::Format::BC7_SRGB_BLOCK, 5, 1), Some(32));
        assert_eq!(level_size(vk::Format::BC4_UNORM_BLOCK, 4, 4), Some(8));
        assert_eq!(level_size(vk::Format::R32G32_SFLOAT, 5, 1), Some(40));
        assert_eq!(level_size(vk::Format::R8G8B8A8_UNORM, 4, 4), None);
    }
}
