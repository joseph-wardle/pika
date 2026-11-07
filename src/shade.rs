//! The material pass: each surface's pipelines, built from the bake tool's
//! pixel stage and the vertex stage written for it, each material's
//! uniform block, textures and descriptor set, bound to its surface, and
//! the draw of each material's meshes after the prepass, lit by the view's
//! headlight (`shaders/frame.slang`). A material is built off the render
//! thread (`bake`) and drawn clay until its first build lands, and as it
//! was until an edit's does; a value edit writes its block in place and
//! rebinds its samplers, and a texture swapped for the levels that fit
//! (`Textures::swap`) rebinds the views of the materials reading it.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use ash::vk;

use crate::bake::{self, Bake, Builder, Built};
use crate::geometry::{Bound, Group};
use crate::gpu::{self, Buffer};
use crate::material::{self, Document, Network, Source};
use crate::need::Footprint;
use crate::raster::{COLOR_FORMATS, DEPTH_AOV, NORMAL_AOV};
use crate::stats;
use crate::store::{Moved, Store};
use crate::target;
use crate::texture::{Binding, Needs, Textures};
use crate::uniforms::{self, Layout, Manifest};
use crate::view::View;

pub struct Shade {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    pub builder: Builder,
    textures: Textures,
    /// By document hash.
    surfaces: BTreeMap<u64, Surface>,
    /// By the store's material id; a material whose first build is out,
    /// or that has no surface, has no entry and is drawn clay, and one
    /// whose edit's build is out keeps the look it had until the build
    /// lands.
    looks: BTreeMap<u64, Look>,
    /// Whether a look changed since the textures no look binds were last
    /// dropped.
    prune: bool,
    /// Whether the textures the cap shrinks, those fitting the memory
    /// blurs, and the textures past the memory at their floors are still to
    /// be warned of since the cap last changed: in a line each, the first
    /// time every build and swap is done with any to count, a scene's worth
    /// at once rather than a line per texture, thousands on a production
    /// asset. Once, since every camera cut swaps hundreds of textures on
    /// ALab.
    warn_cap: bool,
    warn_blur: bool,
    warn_short: bool,
    /// `material::clay`, drawn where `looks` has no entry.
    clay: Material,
    /// `material::pink`, drawn for `Look::Failed`.
    pink: Material,
}

/// How a material is drawn.
enum Look {
    Built(Box<Material>),
    /// Its build failed, warned of once when it did; `update` tries it
    /// again only when its network moves.
    Failed,
}

impl Look {
    fn built(&self) -> Option<&Material> {
        match self {
            Look::Built(m) => Some(m),
            Look::Failed => None,
        }
    }
}

/// One shader structure: the pipelines of every material sharing a document.
pub struct Surface {
    device: ash::Device,
    pub manifest: Manifest,
    pub layout: Layout,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    /// One per `COLOR_FORMATS` entry.
    pipelines: Vec<vk::Pipeline>,
}

/// One material as drawn: its uniform block, written in place when its
/// values change, and the descriptor set binding it with its textures to
/// its surface.
struct Material {
    device: ash::Device,
    hash: u64,
    buffer: Buffer,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// What the textures were baked from, in the surface's order: a
    /// material edit that keeps them and the document only rewrites the
    /// block.
    sources: Vec<Source>,
    /// The baked textures bound, by key, in the surface's order.
    textures: Vec<String>,
}

/// What one frame's `update` did.
#[derive(Default)]
pub struct Update {
    /// Surfaces slangc compiled; the cache answered for the rest.
    pub compiled: usize,
    /// Textures the bake tool baked, likewise.
    pub baked: usize,
    /// The baked bytes the textures loaded and swapped uploaded.
    pub loaded: u64,
    /// Reading those textures' levels from their files.
    pub read: Duration,
    /// Materials whose build is out, drawn clay or as they were.
    pub pending: usize,
    /// Textures whose image does not hold the levels that fit yet.
    pub pending_textures: usize,
    /// Textures lowered past the cap's levels to fit the device's memory.
    pub lowered_textures: usize,
    pub needs: Needs,
}

impl Shade {
    pub fn new(device: &gpu::Device) -> Result<Self, String> {
        let bake = Bake::new()?;
        let mut textures = Textures::new(device)?;
        let mut surfaces = BTreeMap::new();
        let mut builtin = |network: Network| -> Result<Material, String> {
            let document = material::document(&network)?.expect("Pika's networks have a surface");
            let built = Built::now(&bake, document, network);
            // No textures, so nothing is recorded or counted.
            let (cb, mut update) = (vk::CommandBuffer::null(), Update::default());
            Shade::install(device, &mut surfaces, &mut textures, built, cb, &mut update)
        };
        let clay = builtin(material::clay()).map_err(|e| format!("the clay material: {e}"))?;
        let pink = builtin(material::pink()).map_err(|e| format!("the pink material: {e}"))?;
        Ok(Self {
            device: device.raw.clone(),
            builder: Builder::new(bake),
            textures,
            surfaces,
            looks: BTreeMap::new(),
            prune: false,
            warn_cap: true,
            warn_blur: true,
            warn_short: true,
            clay,
            pink,
        })
    }

    /// Rewrites the block of each material whose network changed its
    /// values alone, and submits a build of each whose structure or
    /// textures did, drawn as it was until it lands; installs the builds
    /// that landed, and swaps textures for the levels within the texture
    /// cap `cap` (0 for none) that fit the device's memory
    /// (`Textures::refit`), loads and swaps uploading `texture::UPLOAD`
    /// bytes a frame. A material that cannot be built or installed is
    /// warned of and drawn pink, and a texture whose file is missing is
    /// warned of and drawn as its node's fallback; the frame goes on.
    /// Records into `cb`, outside rendering, the uploads. No GPU work may
    /// be pending.
    pub fn update(
        &mut self,
        device: &gpu::Device,
        store: &Store,
        moved: &Moved,
        cap: u32,
        footprints: &BTreeMap<u64, Footprint>,
        cb: vk::CommandBuffer,
    ) -> Result<Update, String> {
        self.textures.begin();
        let before = self.looks.len();
        self.looks
            .retain(|id, _| store.materials().contains_key(id));
        self.prune |= self.looks.len() != before;
        (self.builder).retain(|id| store.materials().contains_key(id));
        // Before any load: a material whose install fails this frame
        // keeps the textures it loaded until the next, when the uploads
        // recorded into `cb` have run. Collecting the used paths costs
        // milliseconds on a large scene, so only a look changing asks for
        // it.
        if self.prune {
            let used: BTreeSet<String> = (self.looks.values().filter_map(Look::built))
                .flat_map(|m| m.textures.iter().cloned())
                .collect();
            self.textures.retain(&used);
            self.prune = false;
        }
        // Before the builds land, so they load under the new cap.
        if self.textures.set_cap(cap) {
            (self.warn_cap, self.warn_blur, self.warn_short) = (true, true, true);
        }
        let mut update = Update::default();
        for &id in &moved.materials {
            let Some(material) = store.materials().get(&id) else {
                continue;
            };
            let Some(network) = &material.network else {
                continue;
            };
            // A failed material is tried again, drawn clay meanwhile.
            if let Some(Look::Failed) = self.looks.get(&id) {
                self.looks.remove(&id);
            }
            if let Err(e) = self.material(id, network, &material.document) {
                eprintln!("pika: warning: material {}: {e}; drawn pink", material.path);
                self.looks.insert(id, Look::Failed);
                self.prune = true;
            }
        }
        for built in self.builder.drain() {
            let Some(material) = store.materials().get(&built.id) else {
                continue;
            };
            let id = built.id;
            // Even a failed install leaves textures it loaded unbound.
            self.prune = true;
            let missing = built.missing().to_vec();
            let (surfaces, textures) = (&mut self.surfaces, &mut self.textures);
            match Shade::install(device, surfaces, textures, built, cb, &mut update) {
                Ok(drawn) => {
                    for input in &missing {
                        eprintln!(
                            "pika: warning: material {}: {input}; drawn with its fallback",
                            material.path
                        );
                    }
                    self.looks.insert(id, Look::Built(Box::new(drawn)));
                }
                Err(e) => {
                    eprintln!("pika: warning: material {}: {e}; drawn pink", material.path);
                    self.looks.insert(id, Look::Failed);
                }
            }
        }
        for (id, footprint) in footprints {
            let Some(drawn) = self.looks.get(id).and_then(Look::built) else {
                continue;
            };
            for (key, source) in drawn.textures.iter().zip(&drawn.sources) {
                if footprint.uv_per_pixel.is_finite() && !source.transformed {
                    self.textures.need(key, Some(footprint.uv_per_pixel));
                }
                if footprint.unestimated || source.transformed {
                    self.textures.need(key, None);
                }
            }
        }
        self.textures.weigh();
        // Once the scene's builds have landed, so its first textures are
        // not raised past what the whole fits, then lowered again: a
        // 34-tile set at 4K uploads 750 MB in a frame. Loads meanwhile draw
        // their floors.
        let heaps = device.heaps();
        if self.builder.is_empty() {
            self.textures.refit(&heaps);
        }
        let (swapped, bytes, read) = self.textures.swap(device, cb, &heaps);
        (update.loaded, update.read) = (update.loaded + bytes, update.read + read);
        if !swapped.is_empty() {
            for drawn in self.looks.values().filter_map(Look::built) {
                if drawn.textures.iter().any(|key| swapped.contains(key)) {
                    drawn.bind_views(&self.textures, &self.surfaces[&drawn.hash])?;
                }
            }
        }
        update.needs = self.textures.needs();
        update.pending = self.builder.len();
        update.pending_textures = self.textures.pending();
        update.lowered_textures = self.textures.lowered();
        if update.pending > 0 || update.pending_textures > 0 {
            return Ok(update);
        }
        let counted = |n: usize| match n {
            1 => "1 texture was".to_string(),
            _ => format!("{n} textures were"),
        };
        let capped = self.textures.capped();
        if self.warn_cap && capped > 0 {
            eprintln!(
                "pika: warning: {} capped to {cap} texels a side by pika:textureCap",
                counted(capped)
            );
            self.warn_cap = false;
        }
        let (blurred, short) = (update.needs.blurred, self.textures.short());
        let past = format!("textures still take {} past it", stats::size(short));
        if self.warn_blur && blurred > 0 {
            eprintln!(
                "pika: warning: {} drawn coarser than the view needs, at {} texels a side or \
                 less, to fit the device's memory budget{}",
                counted(blurred),
                self.textures.blurred_side(),
                if short > 0 {
                    format!(", and {past}")
                } else {
                    String::new()
                }
            );
            (self.warn_blur, self.warn_short) = (false, self.warn_short && short == 0);
        } else if self.warn_short && short > 0 {
            eprintln!(
                "pika: warning: none of the textures can be lowered to fit the device's \
                 memory budget, and {past}"
            );
            self.warn_short = false;
        }
        Ok(update)
    }

    /// Whether every build has landed and every texture holds the levels
    /// that fit, which a host rendering until converged waits for.
    pub fn settled(&self) -> bool {
        self.builder.is_empty() && self.textures.pending() == 0
    }

    /// Rewrites material `id`'s block from `network` when it keeps its
    /// document and its textures, else submits its build,
    /// leaving the material as it was, clay before its first, until the
    /// build lands. A network with no surface leaves the material drawn clay.
    fn material(
        &mut self,
        id: u64,
        network: &Network,
        document: &Result<Option<Document>, String>,
    ) -> Result<(), String> {
        let Some(document) = document.as_ref().map_err(String::clone)? else {
            self.builder.forget(id);
            self.prune |= self.looks.remove(&id).is_some();
            return Ok(());
        };
        if let Some(Look::Built(drawn)) = self.looks.get_mut(&id)
            && drawn.hash == document.hash
        {
            let surface = &self.surfaces[&document.hash];
            if bake::sources(&surface.manifest, document, network)? == drawn.sources {
                self.builder.forget(id);
                uniforms::pack(
                    &surface.manifest,
                    &surface.layout,
                    document,
                    network,
                    drawn.buffer.bytes(),
                )?;
                // An address mode is a value too: the samplers follow it.
                return drawn.bind(&mut self.textures, surface, document, network);
            }
        }
        (self.builder).submit(id, document.clone(), network.clone())
    }

    /// Installs a build that landed: its surface into `surfaces` unless
    /// another material's build put it there, its uniforms, and its
    /// textures, loaded now into `textures` at their floors with their
    /// uploads recorded into `cb`; counts into `update` what the build did.
    fn install(
        device: &gpu::Device,
        surfaces: &mut BTreeMap<u64, Surface>,
        textures: &mut Textures,
        built: Built,
        cb: vk::CommandBuffer,
        update: &mut Update,
    ) -> Result<Material, String> {
        let output = built.result?;
        let hash = built.document.hash;
        let surface = match surfaces.entry(hash) {
            Entry::Occupied(existing) => existing.into_mut(),
            Entry::Vacant(entry) => {
                let compiled = output.compiled;
                update.compiled += usize::from(compiled.fresh);
                let layout = uniforms::layout(&compiled.reflection)?;
                let spirv = &compiled.shader.spirv;
                entry.insert(Surface::new(device, spirv, output.manifest, layout)?)
            }
        };
        update.baked += output.textures.iter().filter(|b| b.fresh).count();
        let (manifest, layout) = (&surface.manifest, &surface.layout);
        let mut drawn = Material::new(device, surface, hash)?;
        uniforms::pack(
            manifest,
            layout,
            &built.document,
            &built.network,
            drawn.buffer.bytes(),
        )?;
        for baked in output.textures {
            let (loaded, read) = textures.load(device, cb, &baked.key, baked.texture)?;
            update.loaded += loaded;
            update.read += read;
            drawn.textures.push(baked.key);
        }
        drawn.bind(textures, surface, &built.document, &built.network)?;
        drawn.sources = output.sources;
        Ok(drawn)
    }

    /// Records the draws of `view`'s `groups`, after the prepass, into `cb`,
    /// inside the same rendering: each group with its material's surface,
    /// pink where the material failed and clay where it has no surface.
    pub fn draw(
        &self,
        cb: vk::CommandBuffer,
        color: vk::Format,
        area: vk::Rect2D,
        geometry: &Bound,
        groups: &[Group],
        view: &View,
    ) {
        let index = COLOR_FORMATS.iter().position(|&f| f == color);
        let index = index.expect("the renderer chooses a color format");
        let push = view.constants().to_ne_bytes();
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: area.extent.width as f32,
            height: area.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let stride = size_of::<vk::DrawIndexedIndirectCommand>() as u32;
        let d = &self.device;
        // SAFETY: `cb` is recording inside rendering of the formats the
        // pipelines were built for, and every handle was created on `d`.
        // The view's constants address live buffers, as large as the draws
        // read, and each group's records lie within the view's.
        unsafe {
            d.cmd_set_viewport(cb, 0, &[viewport]);
            d.cmd_set_scissor(cb, 0, &[area]);
            d.cmd_bind_index_buffer(cb, geometry.indices, 0, vk::IndexType::UINT32);
            for group in groups {
                let drawn = match group.material.and_then(|id| self.looks.get(&id)) {
                    Some(Look::Built(drawn)) => drawn,
                    Some(Look::Failed) => &self.pink,
                    None => &self.clay,
                };
                let surface = &self.surfaces[&drawn.hash];
                let point = vk::PipelineBindPoint::GRAPHICS;
                let layout = surface.pipeline_layout;
                d.cmd_bind_pipeline(cb, point, surface.pipelines[index]);
                let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
                d.cmd_push_constants(cb, layout, stages, 0, &push);
                d.cmd_bind_descriptor_sets(cb, point, layout, 0, &[drawn.set], &[]);
                let offset = u64::from(group.first * stride);
                d.cmd_draw_indexed_indirect(cb, view.records(), offset, group.count, stride);
            }
        }
    }

    pub fn memory(&self) -> gpu::Memory {
        let mut memory = self.textures.memory();
        let built = self.looks.values().filter_map(Look::built);
        for m in built.chain([&self.clay, &self.pink]) {
            memory += m.buffer.memory();
        }
        memory
    }
}

impl Surface {
    fn new(
        device: &gpu::Device,
        spirv: &[u32],
        manifest: Manifest,
        layout: Layout,
    ) -> Result<Self, String> {
        let d = &device.raw;
        let fragment = vk::ShaderStageFlags::FRAGMENT;
        let mut bindings = vec![
            vk::DescriptorSetLayoutBinding::default()
                .binding(layout.block)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(fragment),
        ];
        for texture in &layout.textures {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(texture.image)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(fragment),
            );
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(texture.sampler)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(fragment),
            );
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(texture.tiles)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(fragment),
            );
        }
        let mut s = Self {
            device: d.clone(),
            manifest,
            layout,
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipelines: Vec::new(),
        };
        let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY (this and every call below): `d` is live, each info struct
        // outlives its call, and each handle passed in was created on `d`
        // just above. `drop` destroys what was made before an error.
        s.set_layout = unsafe { d.create_descriptor_set_layout(&set_info, None) }
            .map_err(|e| format!("vkCreateDescriptorSetLayout failed: {e}"))?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(size_of::<vk::DeviceAddress>() as u32)];
        let set_layouts = [s.set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push);
        s.pipeline_layout = unsafe { d.create_pipeline_layout(&layout_info, None) }
            .map_err(|e| format!("vkCreatePipelineLayout failed: {e}"))?;
        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        let module = unsafe { d.create_shader_module(&module_info, None) }
            .map_err(|e| format!("vkCreateShaderModule failed: {e}"))?;
        // The prepass writes depth; the material pass shades the surface
        // it left at each pixel, into the color AOV only, inside the same
        // rendering, so it declares the prepass's attachments.
        let depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::EQUAL);
        let built: Vec<_> = (COLOR_FORMATS.iter())
            .map(|&color| {
                let entries = [c"vs", c"fragmentMain"];
                let layout = s.pipeline_layout;
                let colors = [
                    (color, vk::ColorComponentFlags::RGBA),
                    (DEPTH_AOV.0, vk::ColorComponentFlags::empty()),
                    (NORMAL_AOV.0, vk::ColorComponentFlags::empty()),
                ];
                gpu::graphics(
                    d,
                    layout,
                    module,
                    entries,
                    &colors,
                    target::DEPTH_FORMAT,
                    depth,
                )
            })
            .collect();
        // SAFETY: pipeline creation is complete, so the module is no longer used.
        unsafe { d.destroy_shader_module(module, None) };
        s.pipelines = built.iter().filter_map(|p| p.clone().ok()).collect();
        built.into_iter().collect::<Result<Vec<_>, _>>()?;
        Ok(s)
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device` or is null, and the
        // owner's target waited for idle before this drops (field order).
        unsafe {
            for &p in &self.pipelines {
                self.device.destroy_pipeline(p, None);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

impl Material {
    /// A block and a set for `surface`, with nothing bound yet.
    fn new(device: &gpu::Device, surface: &Surface, hash: u64) -> Result<Self, String> {
        let d = &device.raw;
        let layout = &surface.layout;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let usage = vk::BufferUsageFlags::UNIFORM_BUFFER;
        let buffer = Buffer::host(device, layout.size as u64, usage, host)?;
        let mut sizes = vec![
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
        ];
        // Vulkan has no empty pool sizes.
        let textures = layout.textures.len() as u32;
        if textures > 0 {
            sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(textures),
            );
            sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .descriptor_count(textures),
            );
            sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(textures),
            );
        }
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY (this and every call below): `d` is live, each info struct
        // outlives its call, and each handle passed in was created on `d`.
        let pool = unsafe { d.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| format!("vkCreateDescriptorPool failed: {e}"))?;
        let mut u = Self {
            device: d.clone(),
            hash,
            buffer,
            pool,
            set: vk::DescriptorSet::null(),
            sources: Vec::new(),
            textures: Vec::new(),
        };
        let set_layouts = [surface.set_layout];
        let set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&set_layouts);
        u.set = unsafe { d.allocate_descriptor_sets(&set_info) }
            .map_err(|e| format!("vkAllocateDescriptorSets failed: {e}"))?[0];
        Ok(u)
    }

    /// Writes the set: the block, and each of `self.textures`, in the
    /// surface's order, through a sampler of the address modes its node
    /// in `network`, by `document`'s node order, and its file say
    /// (`material::address_modes`). The GPU must not be reading the set.
    fn bind(
        &self,
        textures: &mut Textures,
        surface: &Surface,
        document: &Document,
        network: &Network,
    ) -> Result<(), String> {
        let layout = &surface.layout;
        assert_eq!(self.textures.len(), layout.textures.len());
        let mut bindings: Vec<(Binding, vk::Sampler)> = Vec::new();
        for (name, key) in surface.manifest.textures.iter().zip(&self.textures) {
            let (path, node, _) = document.reader(network, name)?;
            let binding = textures.binding(key)?;
            let modes =
                material::address_modes(node, binding.wrap).map_err(|e| format!("{path}'s {e}"))?;
            bindings.push((binding, textures.sampler(modes)?));
        }
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.buffer.raw)
            .range(vk::WHOLE_SIZE)];
        let images: Vec<_> = (bindings.iter())
            .map(|&(binding, sampler)| {
                [
                    vk::DescriptorImageInfo::default()
                        .image_view(binding.view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    vk::DescriptorImageInfo::default().sampler(sampler),
                ]
            })
            .collect();
        let tiles: Vec<_> = bindings.iter().map(|&(b, _)| [b.tiles]).collect();
        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(layout.block)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info),
        ];
        for ((texture, info), table) in layout.textures.iter().zip(&images).zip(&tiles) {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(texture.image)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&info[..1]),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(texture.sampler)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&info[1..]),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(texture.tiles)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(table),
            );
        }
        // SAFETY: the set, buffers, views and samplers were created on
        // `device`, which is live, and the infos outlive the call.
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        Ok(())
    }
}

impl Material {
    /// Rewrites the set's views, each of `self.textures` as it is loaded
    /// now, after a swap replaced its image; the samplers and tile tables
    /// stay. The GPU must not be reading the set.
    fn bind_views(&self, textures: &Textures, surface: &Surface) -> Result<(), String> {
        let mut views = Vec::new();
        for key in &self.textures {
            views.push([vk::DescriptorImageInfo::default()
                .image_view(textures.binding(key)?.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]);
        }
        let writes: Vec<_> = (surface.layout.textures.iter().zip(&views))
            .map(|(texture, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(texture.image)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(info)
            })
            .collect();
        // SAFETY: the set and views were created on `device`, which is
        // live, and the infos outlive the call.
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        Ok(())
    }
}

impl Drop for Material {
    fn drop(&mut self) {
        // Destroying the pool frees its set; the buffer drops after.
        // SAFETY: the pool was created on `device`, and the owner's target
        // waited for idle before this drops (field order).
        unsafe { self.device.destroy_descriptor_pool(self.pool, None) };
    }
}
