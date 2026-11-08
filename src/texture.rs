//! Baked textures: the file `bake texture` writes for one image or UDIM
//! set under the cache, laid out as bake/texture.cpp says, as the renderer
//! reads it, the image it is uploaded to, holding the levels within the
//! texture cap, and the set of them loaded for the materials drawn, with
//! the samplers they are read through. Each texture holds the levels within
//! the cap unless they do not all fit the device's memory, when the textures
//! no view drew are lowered first, then levels finer than the view needs,
//! then the rest by how far below their need they fall (`fit`, `need`). A
//! texture whose levels
//! should change swaps its image for one of its new levels, read from the
//! file, a frame's worth of bytes at a time; nothing is baked or built again.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::{Duration, Instant};

use ash::vk;

use crate::gpu::{self, Buffer};
use crate::need;

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

/// Where a texture's levels are read from when it uploads (`Texture::read_at`).
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

    /// Reads its file's bytes from `from` on into `into`, which they fill.
    pub fn read_at(&self, from: usize, into: &mut [u8]) -> std::io::Result<()> {
        match &self.contents {
            Contents::Open(file) => file.read_exact_at(into, from as u64),
            Contents::Memory(file) => {
                into.copy_from_slice(&file[from..from + into.len()]);
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

/// The rows of texels a row of `format`'s blocks spans: 4 for the block
/// formats, 1 for the rest.
fn block_rows(format: vk::Format) -> u32 {
    match format {
        vk::Format::BC4_UNORM_BLOCK | vk::Format::BC7_UNORM_BLOCK | vk::Format::BC7_SRGB_BLOCK => 4,
        _ => 1,
    }
}

/// The copies that upload `texture`'s levels from `first` on, from byte
/// `done` of them, in whole rows of blocks of one layer, as many as
/// `budget` bytes hold, none when not a row does: each copy's buffer
/// offset counts from `done`. The copies, and the byte after the last.
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

/// The bytes of levels a frame uploads, loads and swaps together, past
/// which the rest wait for the next: reading and copying a texture's levels
/// costs the render thread about 0.8 ms a MiB on the RTX 4070 Ti SUPER. A
/// raise uploads whole rows of blocks of one layer (`copies`), so a texture
/// larger than this raises over frames into its new image; a load lands
/// whole, at its floor of a few blocks. A lower uploads nothing: it copies
/// its levels from the old image on the GPU.
pub const UPLOAD: u64 = 32 << 20;

/// The longest side, in texels, of the coarsest level a texture is lowered
/// to, and loads at: a 4×4 block, the smallest level the block formats
/// store whole. Each coarser level takes a block too, so lowering past it
/// saves next to nothing.
pub const TAIL: u32 = 4;

/// The bytes of a heap's budget that fitting leaves free: room for a
/// raise's old image beside its new one, and for the budget to move, by
/// other processes or this one's geometry, without a target moving.
/// Hypothesis: enough on ALab beside usdview and Houdini.
pub const MARGIN: u64 = 256 << 20;

/// The bytes of a heap's budget that the images swaps begin leave free,
/// since the driver can refuse an allocation the budget allows: on the RTX
/// 4070 Ti SUPER, in ALab's cut run, which lowers hundreds of textures at
/// each cut, raises of 3 and 4 MB that fit the frame's budget failed with
/// the budget filled to 1 MB; with this left free, none did. Hypothesis:
/// enough elsewhere.
pub const RESERVE: u64 = 64 << 20;

/// A texture's levels on the GPU from `first` on: a 2D array image, in the
/// layout the material pass samples once its upload has run.
struct Image {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    bytes: u64,
    /// The memory heap `memory` is from.
    heap: usize,
    first: usize,
}

/// A texture loaded: its file, its tile table, which a swap keeps, the
/// image of its levels now, and what fitting it weighs.
struct Loaded {
    texture: Texture,
    tiles: Buffer,
    image: Image,
    /// The image a raise is uploading for its target, which replaces
    /// `image` once whole.
    incoming: Option<Incoming>,
    chain: Chain,
    /// The first level its image is to hold, `fit`'s.
    target: usize,
    /// For each level, the last frame it was the finest a reader drawn
    /// needed, 0 for never (`need::finest`).
    seen: Vec<u64>,
    /// The last frame a reader was drawn whose need was not estimated, 0
    /// for never.
    unestimated: u64,
    /// Whether its lower failed and is tried again each frame, warned of
    /// once.
    failing: bool,
}

impl Loaded {
    /// Whether its image holds coarser levels than the finest a view
    /// needed, within `cap`: what fitting it to the memory blurs.
    fn blurred(&self, cap: u32) -> bool {
        let need = self
            .chain
            .need
            .map(|level| level.max(self.texture.first(cap)));
        need.is_some_and(|level| level < self.image.first)
    }
}

/// A raise's new image, uploaded over frames: its levels' bytes before
/// `done` are copied.
struct Incoming {
    image: Image,
    done: u64,
}

/// What the view needs of the loaded textures, within `need::WINDOW`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Needs {
    /// Textures no reader was drawn of.
    pub unseen: usize,
    /// Textures a reader was drawn of whose need was not estimated.
    pub unestimated: usize,
    /// Textures whose image holds coarser levels than the finest needed,
    /// within the cap: what fitting them to the memory blurs.
    pub blurred: usize,
}

/// What `fit` weighs of a loaded texture.
#[derive(Clone, Debug, PartialEq)]
struct Chain {
    /// The memory heap its images are allocated from.
    heap: usize,
    /// The longest side of its level 0.
    side: u32,
    /// The bytes of its image holding the levels from each on, as the
    /// driver sizes it with alignment and tiling.
    sizes: Vec<u64>,
    /// The finest level its image may hold: the cap's, or its image's own
    /// while a raise that failed is held.
    ceiling: usize,
    /// The coarsest: the first within `TAIL`, or the cap's past it.
    floor: usize,
    /// The finest level a view needed within `need::WINDOW`, the finest
    /// of all when a reader's need was not estimated; `None` when no view
    /// drew it.
    need: Option<usize>,
    /// The room its heap had for textures when a raise of it failed, which
    /// holds it at its image's levels (`fail`) until `released`.
    held: Option<u64>,
}

impl Chain {
    /// Sizes each image of `texture`'s levels without creating any, and
    /// limits them to `cap`.
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
            let mut needs = vk::MemoryRequirements2::default();
            // SAFETY: `device.raw` is a live Vulkan 1.3 device, and `info`
            // outlives the call.
            unsafe {
                device
                    .raw
                    .get_device_image_memory_requirements(&ask, &mut needs)
            };
            let needs = needs.memory_requirements;
            let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            chain.heap = device.heap(needs.memory_type_bits, local)?;
            chain.sizes.push(needs.size);
        }
        chain.limit(texture, cap);
        Ok(chain)
    }

    /// Sets its ceiling and floor for the cap `cap`, 0 for none, releasing
    /// it if held.
    fn limit(&mut self, texture: &Texture, cap: u32) {
        self.ceiling = texture.first(cap);
        self.floor = texture.first(TAIL).max(self.ceiling);
        self.held = None;
    }

    /// The level to target after its swap from `first` toward `target`
    /// failed while its heap had `room` bytes for textures. A lower keeps
    /// its target, tried again next frame, as the others' lowers free
    /// memory. A raise is held at `first`, its ceiling, until `released`:
    /// tried each frame, it would fail each frame and never settle.
    fn fail(&mut self, first: usize, target: usize, room: u64) -> usize {
        if target > first {
            return target;
        }
        (self.ceiling, self.held) = (first, Some(room));
        first
    }

    /// Whether it is held and to be released, by heap index the rooms
    /// `room`: once the textures or their limits `changed`, or its heap's
    /// room grew past the room it was held at by more than `MARGIN`, so the
    /// budget wandering retries nothing.
    fn released(&self, room: &[u64], changed: bool) -> bool {
        (self.held).is_some_and(|held| changed || room[self.heap] > held.saturating_add(MARGIN))
    }

    /// The longest side of its level `level`.
    fn extent(&self, level: usize) -> u32 {
        (self.side >> level).max(1)
    }
}

/// Where lowering `chain` from `level` falls in a fit, the greatest first.
/// First the chains no view drew, then levels finer than a chain's need,
/// which no view sees go; then a level below its need, the chain left
/// least far below its own first, so the blur spreads evenly, then the one
/// needing the coarsest levels, the farthest from the camera. Within each,
/// the longest side goes first, then the most bytes.
fn order(chain: &Chain, level: usize) -> impl Ord {
    let (seen, below, need) = match chain.need {
        None => (0, 0, 0),
        Some(need) if level < need => (1, 0, need),
        Some(need) => (2, level - need, need),
    };
    let tier = (Reverse(seen), Reverse(below), need);
    (tier, chain.extent(level), chain.sizes[level])
}

/// Each chain's first level, by index: its ceiling, lowered a level at a
/// time while its heap's images take more than `capacity`, bytes by heap
/// index, none past its floor, in `order`, then the lowest index first. A
/// chain's place in the order hangs on its own level alone, so more
/// capacity stops earlier in the same sequence: a chain is never raised by
/// less capacity, nor lowered by more.
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

/// The chains' targets under `room`, bytes by heap index, from `current`,
/// and whether they were fitted afresh: `fit`'s under the room less
/// `MARGIN` when `afresh`, or when `current` raised toward them would not
/// fit the room, else `current` only raised toward them. So the room
/// moving back and forth by less than the margin raises a chain at most
/// once, and lowers none.
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

/// The order a frame's swaps go in, by index, of the chains whose image
/// holds the levels from `firsts` and is to hold those from `targets`:
/// lowers first, which free memory and upload nothing, then raises, those
/// `begun` first, so their images finish before others take memory, then
/// first the chain the fit would lower last from its level now (`order`):
/// the furthest below its need, then the nearest.
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

/// The bytes the chains' images from `targets` take, by heap index.
fn totals(chains: &[&Chain], targets: &[usize], heaps: usize) -> Vec<u64> {
    let mut totals = vec![0; heaps];
    for (c, &target) in chains.iter().zip(targets) {
        totals[c.heap] += c.sizes[target];
    }
    totals
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
        .usage(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
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
            heap: 0,
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
        i.heap = device.heap(needs.memory_type_bits, flags)?;
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

    /// Records into `cb`, outside rendering, the move of its levels of
    /// `texture`, whose contents are discarded, into the layout copies write.
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

    /// Records into `cb`, outside rendering, the move of its levels of
    /// `texture` from the layout copies write into the one the material
    /// pass samples, after every copy into them submitted before.
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

    /// Records into `cb`, outside rendering, `copies` into its levels, in
    /// the layout copies write, from `staging`, which must live until the
    /// GPU has run `cb`.
    fn copy(&self, cb: vk::CommandBuffer, staging: &Buffer, copies: &[vk::BufferImageCopy]) {
        let layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        // SAFETY: `cb` is recording outside rendering, the image was created
        // on `device` with TRANSFER_DST usage for the levels copied
        // (`copies`), and `staging` holds them at the offsets copied from.
        unsafe {
            (self.device).cmd_copy_buffer_to_image(cb, staging.raw, self.image, layout, copies)
        };
    }

    /// Records into `cb`, outside rendering, the copy into its levels of
    /// `texture` of `old`'s, which holds them and finer ones, sampled: `old`
    /// moves into the layout copies read from and back, after the material
    /// pass of every frame submitted before, and before this frame's.
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
        // SAFETY: `cb` is recording outside rendering, both images were
        // created on `device` with the transfer usages, each region's
        // levels exist in both at the same extent, and the barriers above
        // put each in the layout named.
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

/// The host memory a frame's level reads land in, which its commands copy
/// into the images: one buffer of `UPLOAD` bytes kept from frame to frame,
/// since on ALab a buffer of a texture's own cost seven times its read to
/// create and free, and one of its own for a load past what is left.
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

    /// The bytes left in its buffer this frame.
    fn free(&self) -> u64 {
        (self.buffer.size).saturating_sub(self.used.next_multiple_of(16))
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

/// Uploads `image`'s levels of `texture` from byte `done` of them on, as
/// many rows as `budget` bytes hold (`copies`), read now into `staging` and
/// copied by commands recorded into `cb`: the first part moves the image
/// into the layout copies write, the last into the one the material pass
/// samples. The byte after the last uploaded, `done` when not a row fits,
/// and the time the read took.
fn upload(
    device: &gpu::Device,
    cb: vk::CommandBuffer,
    texture: &Texture,
    image: &Image,
    (done, budget): (u64, u64),
    staging: &mut Staging,
) -> Result<(u64, Duration), String> {
    let (mut copies, end) = copies(texture, image.first, done, budget);
    if end == done {
        return Ok((done, Duration::ZERO));
    }
    // The levels alone, whole rows of blocks: each starts at a multiple of
    // its format's block size then, which a copy's buffer offset must be;
    // the header is not one.
    let size = end - done;
    let (buffer, at) = staging.room(device, size)?;
    let start = Instant::now();
    let into = &mut buffer.bytes()[at as usize..(at + size) as usize];
    let from = texture.levels[image.first].start + done as usize;
    (texture.read_at(from, into)).map_err(|e| format!("cannot read its levels: {e}"))?;
    let read = start.elapsed();
    for copy in &mut copies {
        copy.buffer_offset += at;
    }
    if done == 0 {
        image.to_copy(cb, texture);
    }
    image.copy(cb, buffer, &copies);
    if end == texture.bytes(image.first) {
        image.to_sample(cb, texture);
    }
    Ok((end, read))
}

/// Leaves the texture keyed `key` to `Chain::fail` after a swap that failed
/// with `e` while its heap had `room` bytes for textures, warned of: a
/// raise as it is held, a lower once until it succeeds.
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

/// The textures loaded for the materials drawn, by `bake::Baked::key`, each
/// holding the levels `fit` gives it, and the samplers they are read
/// through. Loads and swaps are recorded into a frame's commands, outside
/// rendering, `UPLOAD` bytes a frame, and their staging is kept until the
/// next frame begins, when the GPU has read it.
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
    /// The heaps' use and budgets when the targets were last set.
    heaps: Vec<gpu::Heap>,
    /// The room each heap then had for textures, by heap index.
    room: Vec<u64>,
    /// Whether the textures or their limits changed since then, so the
    /// targets are fitted afresh and held raises released.
    dirty: bool,
    /// Whether a raise did not fit beside its old image under targets only
    /// raised, so the next are fitted afresh, leaving `MARGIN`.
    crowded: bool,
    /// Whether this frame's targets were fitted afresh, leaving `MARGIN`
    /// free for its raises.
    fresh: bool,
    /// The bytes the targets take past the heaps' room, at their floors.
    short: u64,
    /// The frames begun, which the textures' needs are stamped with.
    frame: u64,
    /// Whether a texture's need changed since the targets were last set,
    /// so they are settled again (`weigh`).
    moved: bool,
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

    /// Begins a frame's loads and swaps. No GPU work may be pending.
    pub fn begin(&mut self) {
        (self.staging.used, self.staging.own) = (0, Vec::new());
        self.retired.clear();
        self.fresh = false;
        self.frame += 1;
    }

    /// Sets the cap to `cap` texels a side, 0 for none, and so each loaded
    /// texture's ceiling, which the next `refit` fits under; whether it
    /// changed. A raise held after it failed is tried again.
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

    /// The baked texture keyed `key`, read as `texture`, loaded now unless
    /// it is: its levels from its floor on, read into a staging buffer,
    /// which `refit` and `swap` then raise; the bytes that uploaded, 0 when
    /// it was loaded, and the time their read took.
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
        let all = (0, u64::MAX);
        let (size, read) = upload(device, cb, &texture, &image, all, &mut self.staging)
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

    /// Sets each loaded texture's target from the memory heaps' use and
    /// budgets, `heaps`: `fit`'s under the room its heap has for textures,
    /// its budget less what the process holds besides them, less `MARGIN`.
    /// Targets that still fit the room are only raised, so the budget
    /// moving by less than the margin moves none; a load, a drop, a cap
    /// change or targets past the room fit them afresh. A need that moved
    /// settles them as the budget moving does. A held raise is released
    /// first (`Chain::released`).
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

    /// Swaps each image not holding its target's levels for one that does,
    /// in `queue`'s order: a lower copies its levels from the old image on
    /// the GPU, a raise reads them from its file into a new image while the
    /// frame's bytes last (`UPLOAD`), over as many frames as it takes. The
    /// old image is drawn until the new one is whole, then retired once the
    /// materials binding it are rebound, so a swap that fails keeps the
    /// old. A raise begun toward a target that has since moved starts
    /// again. A swap begins only if its image fits its heap's budget in
    /// `heaps` less `RESERVE`, beside the old and those begun this frame,
    /// else waits for the images this frame retires to be freed; with none,
    /// a lower goes ahead, and a raise waits for a refit to leave the
    /// margin. A swap that fails, or a
    /// raise that still does not fit, is warned of and left to `fail`.
    /// Records the copies and uploads into `cb`; the keys swapped, which
    /// the caller rebinds before the next frame begins, the bytes uploaded
    /// and the time their read took.
    pub fn swap(
        &mut self,
        device: &gpu::Device,
        cb: vk::CommandBuffer,
        heaps: &[gpu::Heap],
    ) -> (BTreeSet<String>, u64, Duration) {
        let (mut keys, mut bytes, mut read) = (BTreeSet::new(), 0, Duration::ZERO);
        let mut added = vec![0; heaps.len()];
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
                let mut needs = added.clone();
                needs[chain.heap] += size + RESERVE;
                // A lower frees more than its image takes once its old one
                // is freed, so past its budget it waits only for the
                // images this frame retires; a raise also for a refit to
                // leave the margin.
                let lowers = target > first;
                let fits = gpu::overflow(heaps, &needs).is_none();
                let made = if fits || (lowers && keys.is_empty()) {
                    Image::new(device, texture, target)
                } else if !keys.is_empty() {
                    continue;
                } else if !self.fresh {
                    self.crowded = true;
                    continue;
                } else {
                    let (w, h) = texture.extent(target);
                    Err(format!(
                        "its {w}×{h} levels do not fit the device's memory budget beside its \
                         image now"
                    ))
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
                    keys.insert(key);
                    continue;
                }
                loaded.incoming = Some(Incoming { image, done: 0 });
            }
            let incoming = loaded.incoming.as_mut().expect("begun above");
            let part = (incoming.done, self.staging.free());
            let image = &incoming.image;
            match upload(device, cb, &loaded.texture, image, part, &mut self.staging) {
                Ok((end, _)) if end == incoming.done => break,
                Ok((end, time)) => {
                    (bytes, read) = (bytes + end - incoming.done, read + time);
                    incoming.done = end;
                    if end == loaded.texture.bytes(incoming.image.first) {
                        let new = loaded.incoming.take().expect("begun above").image;
                        self.retired.push(std::mem::replace(&mut loaded.image, new));
                        keys.insert(key);
                    }
                }
                Err(e) => {
                    loaded.incoming = None;
                    fail(&key, loaded, room, &e);
                    self.moved = true;
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

    /// Notes that a reader of the texture keyed `key` is drawn this frame
    /// where a pixel covers `uv_per_pixel` texture coordinate units, or,
    /// for `None`, where its need is not estimated.
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

    /// Sets each texture's need from what the views needed within the
    /// window, which the next `refit` weighs.
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

    /// What the views needed of the loaded textures within the window, as
    /// `weigh` last set it.
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

    /// The textures lowered past the cap to fit, a held raise included.
    pub fn lowered(&self) -> usize {
        (self.loaded.values())
            .filter(|l| l.target > l.texture.first(self.cap))
            .count()
    }

    /// The longest side of the finest level any blurred texture's image
    /// holds (`Needs::blurred`), 0 when none is.
    pub fn blurred_side(&self) -> u32 {
        (self.loaded.values())
            .filter(|l| l.blurred(self.cap))
            .map(|l| l.chain.extent(l.image.first))
            .max()
            .unwrap_or(0)
    }

    /// The bytes the textures take past the room their heaps have, at the
    /// floors they are not lowered past.
    pub fn short(&self) -> u64 {
        self.short
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

    /// Drops the textures not in `used`. No GPU work may be pending.
    pub fn retain(&mut self, used: &BTreeSet<String>) {
        let before = self.loaded.len();
        self.loaded.retain(|key, _| used.contains(key));
        self.dirty |= self.loaded.len() != before;
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

    /// Every image it holds: those drawn, those raises are uploading, and
    /// those retired.
    fn images(&self) -> impl Iterator<Item = &Image> {
        (self.loaded.values())
            .flat_map(|l| std::iter::once(&l.image).chain(l.incoming.as_ref().map(|i| &i.image)))
            .chain(&self.retired)
    }

    /// The images' bytes, those swaps are uploading and retired ones
    /// included, and the tile tables' and staging's on the host side.
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

    /// `t`'s levels from `first` on as an upload reads them.
    fn levels(t: &Texture, first: usize) -> Vec<u8> {
        let mut bytes = vec![0; t.bytes(first) as usize];
        t.read_at(t.levels[first].start, &mut bytes).unwrap();
        bytes
    }

    /// The texture tool wrote this for bake/fixtures/ramp.exr, whose texels
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
        // 3 starts at 2×1, as the texture tool's halving once did.
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

    /// A chain on heap `heap` of a square texture `side` texels a side of
    /// one byte a texel, with no cap.
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
        // From level 0: 5461 and 341 bytes; from 1, 1365 and 85.
        assert_eq!(
            (big.sizes[0], big.sizes[1], small.sizes[0]),
            (5461, 1365, 341)
        );
        let chains = [&big, &small];
        assert_eq!(fit(&chains, &[u64::MAX]), [0, 0], "the control: both fit");
        assert_eq!(fit(&chains, &[5461 + 341]), [0, 0], "exactly");
        assert_eq!(fit(&chains, &[5461 + 340]), [1, 0], "the longer side first");
        // The big one at 32 texels, 1365 + 341, then at 16, as long as the
        // small one's side.
        assert_eq!(fit(&chains, &[1365 + 340]), [2, 0]);
        // Floors: 4×4 from level 4 and level 2, 21 bytes each; none fits.
        assert_eq!((big.floor, small.floor), (4, 2));
        assert_eq!(fit(&chains, &[0]), [4, 2], "no further than the floors");
        // Equal sides: the most bytes first, as a set's two layers have,
        // then the lowest index.
        let mut pair = chain(0, 16);
        pair.sizes.iter_mut().for_each(|s| *s *= 2);
        assert_eq!(fit(&[&small, &pair], &[341 + 681]), [0, 1]);
        assert_eq!(fit(&[&pair, &small], &[341 + 681]), [1, 0]);
        assert_eq!(fit(&[&small, &small], &[681]), [1, 0]);

        // A fixed point: fitting under what a fit takes gives it again.
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
        // Three 64-texel chains: from level 0 they take 5461 bytes, from 1
        // 1365, 2 341, 3 85 and 4, their floor, 21. The near one needs
        // level 0, the far one level 2, and no view drew the third.
        let (mut near, mut far, unseen) = (chain(0, 64), chain(0, 64), chain(0, 64));
        (near.need, far.need) = (Some(0), Some(2));
        let chains = [&near, &far, &unseen];
        assert_eq!(fit(&chains, &[u64::MAX]), [0, 0, 0], "the control: all fit");
        assert_eq!(fit(&chains, &[3 * 5461]), [0, 0, 0]);
        // The unseen one first, to its floor.
        assert_eq!(fit(&chains, &[3 * 5461 - 1]), [0, 0, 1]);
        assert_eq!(fit(&chains, &[2 * 5461 + 21]), [0, 0, 4]);
        // Then the far one's levels finer than it needs, which no view sees.
        assert_eq!(fit(&chains, &[2 * 5461 + 20]), [0, 1, 4]);
        assert_eq!(fit(&chains, &[5461 + 341 + 21]), [0, 2, 4]);
        // Then a level below a need: the far one's first, then the near
        // one's, each one below before either goes two.
        assert_eq!(fit(&chains, &[5461 + 341 + 20]), [0, 3, 4]);
        assert_eq!(fit(&chains, &[5461 + 85 + 20]), [1, 3, 4]);
        assert_eq!(fit(&chains, &[1365 + 85 + 20]), [1, 4, 4]);
        assert_eq!(fit(&chains, &[0]), [4, 4, 4], "no further than the floors");
        // The reversed control: with the needs swapped, the other chain
        // stays sharp, so the order follows the needs, not the index.
        (near.need, far.need) = (Some(2), Some(0));
        assert_eq!(fit(&[&near, &far, &unseen], &[5461 + 341 + 20]), [3, 0, 4]);

        // A level no view sees goes before one it does, however coarse: a
        // chain capped to level 3 that needs it keeps it while another
        // drops a level finer than its need of 1.
        let (mut capped, mut fine) = (chain(0, 64), chain(0, 64));
        (capped.ceiling, capped.need, fine.need) = (3, Some(3), Some(1));
        assert_eq!(fit(&[&capped, &fine], &[85 + 5461 - 1]), [3, 1]);

        // The fixed point and the monotony hold as they do without needs.
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
        // A room that lowers some, 20 kB past the margin: the 256 to 64
        // and the 128 to 64, 17748 bytes in all.
        let room = MARGIN + 20_000;
        let (start, fresh) = settle(&chains, &[0; 4], &[room], true);
        assert!(fresh);
        assert_eq!(start, [2, 1, 0, 0]);
        // The room wanders over less than the margin around it: one raise,
        // when it first rises past the next level, then nothing.
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
        // The control: the room falling past the targets fits afresh, lower.
        let (lower, fresh) = settle(&chains, &targets, &[30_000], false);
        assert!(fresh);
        assert!(lower.iter().zip(&targets).any(|(l, t)| l > t));
        // A load, a drop or a cap change fits afresh at the room.
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
        // A ceiling past the floor, as a cap under `TAIL` makes it: the
        // chain stays where it is.
        let mut kept = chain(0, 64);
        (kept.ceiling, kept.floor) = (5, 5);
        assert_eq!(fit(&[&kept], &[0]), [5]);
        // A longer side on another heap is not lowered for this one's.
        let small = chain(0, 16);
        assert_eq!(fit(&[&other, &small], &[0, u64::MAX]), [0, 2]);
    }

    /// A texture of `format`, `width`×`height` with `layers` layers, whose
    /// level bytes differ from their neighbours, to see where copies put
    /// them.
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

    /// Writes `staging` where `copies` put it in the levels of `t` from
    /// `first` on, as `vkCmdCopyBufferToImage` reads tightly packed rows of
    /// blocks, into `image`, laid out as the file lays them, counting each
    /// byte's writes into `writes`.
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
        // Uploaded whole, a copy per layer of each level.
        let t = numbered(vk::Format::BC7_SRGB_BLOCK, (16, 8, 2));
        let (whole, end) = copies(&t, 0, 0, u64::MAX);
        assert_eq!((whole.len(), end), (5 * 2, t.bytes(0)));
        // The controls: a budget under a row of 4 blocks of 16 bytes
        // uploads nothing, and a copy left out is seen.
        let (none, end) = copies(&t, 0, 0, 63);
        assert_eq!((none.len(), end), (0, 0));
        let (file, len) = (levels(&t, 0), t.bytes(0) as usize);
        let (mut image, mut writes) = (vec![0; len], vec![0; len]);
        copied(&t, 0, &whole[1..], &file, (&mut image, &mut writes));
        assert_ne!(image, file, "a layer missing");
    }

    #[test]
    fn swaps_go_begun_then_lowers_then_the_most_blurred_raise() {
        // Four 64-texel chains holding level 3: one needing level 0, one
        // level 2, one no view drew, all to be raised, and one lowered.
        let (mut near, mut far, unseen, mut lower) =
            (chain(0, 64), chain(0, 64), chain(0, 64), chain(0, 64));
        (near.need, far.need, lower.need) = (Some(0), Some(2), Some(3));
        let (firsts, targets) = ([3; 4], [0, 1, 2, 4]);
        let none = [false; 4];
        let chains = [&near, &far, &unseen, &lower];
        assert_eq!(queue(&chains, &firsts, &targets, &none), [3, 0, 1, 2]);
        // The reversed control: with the needs swapped, the other goes
        // first, so the order follows the needs, not the index.
        (near.need, far.need) = (Some(2), Some(0));
        let chains = [&near, &far, &unseen, &lower];
        assert_eq!(queue(&chains, &firsts, &targets, &none), [3, 1, 0, 2]);
        // Lowers go first, then a raise begun, even toward the level it
        // holds; one holding its target and not begun is not queued.
        let begun = [false, false, true, false];
        assert_eq!(queue(&chains, &firsts, &targets, &begun), [3, 2, 1, 0]);
        assert_eq!(queue(&chains, &firsts, &[0, 1, 3, 4], &begun), [3, 2, 1, 0]);
        assert_eq!(queue(&chains, &firsts, &[3, 1, 3, 4], &none), [3, 1]);
    }

    #[test]
    fn a_failed_raise_is_held_until_the_room_grows_and_a_failed_lower_is_not() {
        // Two 64-texel chains on one heap, room for one whole and the
        // other at level 3: the first's raise from level 3 failed.
        let (mut held, other) = (chain(0, 64), chain(0, 64));
        let capacity = held.sizes[3] + other.sizes[0];
        let room = 10_000;
        assert_eq!(held.fail(3, 0, room), 3, "held where it is");
        assert_eq!((held.ceiling, held.held), (3, Some(room)));
        // The fit leaves it there and gives the other the room it freed.
        assert_eq!(fit(&[&held, &other], &[capacity]), [3, 0]);
        // The room the same, or grown by the margin alone, holds it.
        assert!(!held.released(&[room], false));
        assert!(!held.released(&[room + MARGIN], false));
        // Grown past that, or the textures changed, releases it; one never
        // held is never released.
        assert!(held.released(&[room + MARGIN + 1], false));
        assert!(held.released(&[room], true));
        assert!(!other.released(&[u64::MAX], true));
        // The control: released, it competes for the room again, both
        // lowered a level, so the fit differs.
        held.limit(&numbered(vk::Format::R32_SFLOAT, (64, 64, 1)), 0);
        assert_eq!((held.ceiling, held.held), (0, None));
        assert_eq!(fit(&[&held, &other], &[capacity]), [1, 1]);
        // A failed lower keeps its target, to be tried again, and holds
        // nothing.
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
