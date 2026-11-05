//! The material pass: each surface's pipelines, built from the bake tool's
//! pixel stage and the vertex stage written for it, each material's
//! uniform block, textures and descriptor set, bound to its surface, and
//! the draw of each material's meshes after the prepass, lit by a
//! headlight. A material is built off the render thread (`build`) and drawn
//! clay until its build lands; a value edit writes its block in place.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use ash::vk;

use crate::bake::{Bake, Manifest};
use crate::build::{self, Builder, Built};
use crate::geometry::{Bound, Group};
use crate::gpu::{self, Buffer};
use crate::material::{self, Source};
use crate::raster::{COLOR_FORMATS, DEPTH_AOV, NORMAL_AOV};
use crate::store::{Moved, Network, Store};
use crate::surface::{self, Layout};
use crate::target;
use crate::texture::{Binding, Textures};
use crate::transform::Camera;
use crate::view::View;

pub struct Shade {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    pub builder: Builder,
    textures: Textures,
    /// By document hash.
    surfaces: BTreeMap<u64, Surface>,
    /// By the store's material id; a material whose build is out has no
    /// entry, and is drawn clay.
    materials: BTreeMap<u64, Uniforms>,
    /// The materials that failed to build, each warned of once when it
    /// did; `update` tries one again only when its network moves or the
    /// cap changes.
    failed: BTreeSet<u64>,
    /// Whether a material came or went since the textures no material
    /// uses were last dropped.
    stale: bool,
    /// The texture cap of the frame; 0 before the first.
    cap: u32,
    /// `material::clay`, drawn where `materials` has no entry.
    clay: Uniforms,
    /// `material::pink`, drawn for `failed`.
    pink: Uniforms,
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

/// One material's uniform block, written in place when its values change,
/// and the descriptor set binding it with its textures to its surface.
struct Uniforms {
    device: ash::Device,
    hash: u64,
    buffer: Buffer,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// What the textures were baked from, in the surface's order, and under
    /// which cap: a material edit that keeps them and the document only
    /// rewrites the block.
    sources: Vec<Source>,
    cap: u32,
    /// The baked textures bound, in the surface's order.
    textures: Vec<PathBuf>,
}

/// What one frame's `update` did.
#[derive(Default)]
pub struct Update {
    /// Surfaces slangc compiled; the cache answered for the rest.
    pub compiled: usize,
    /// Textures the bake tool baked, likewise.
    pub baked: usize,
    /// Materials whose build is out, drawn clay.
    pub pending: usize,
}

impl Shade {
    pub fn new(device: &gpu::Device) -> Result<Self, String> {
        let bake = Bake::new()?;
        let mut textures = Textures::new(device);
        let mut surfaces = BTreeMap::new();
        let mut builtin = |network: Network| -> Result<Uniforms, String> {
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
            materials: BTreeMap::new(),
            failed: BTreeSet::new(),
            stale: false,
            cap: 0,
            clay,
            pink,
        })
    }

    /// Rewrites the block of each material whose network changed its
    /// values alone, and submits a build of each whose structure, textures
    /// or, when the texture cap `cap` changed, cap did, to be drawn clay
    /// until it lands; installs the builds that landed, loading their
    /// textures; then writes every material's eye and headlight from
    /// `camera`. A material that cannot be built is warned of and drawn
    /// pink; the frame goes on. Records into `cb`, outside rendering, the
    /// uploads. No GPU work may be pending.
    pub fn update(
        &mut self,
        device: &gpu::Device,
        store: &Store,
        moved: &Moved,
        cap: u32,
        camera: &Camera,
        cb: vk::CommandBuffer,
    ) -> Result<Update, String> {
        if cap == 0 {
            return Err(
                "the texture cap is 0: set pika:textureCap to the texels a texture's \
                        longest side is capped at"
                    .into(),
            );
        }
        self.textures.begin();
        let before = self.materials.len();
        self.materials
            .retain(|id, _| store.materials().contains_key(id));
        self.stale |= self.materials.len() != before;
        self.failed.retain(|id| store.materials().contains_key(id));
        (self.builder).retain(|id| store.materials().contains_key(id));
        // Before any load: a material whose install fails this frame
        // keeps the textures it loaded until the next, when the uploads
        // recorded into `cb` have run. Collecting the used paths costs
        // milliseconds on a large scene, so only a material coming or
        // going asks for it.
        if self.stale {
            let used: BTreeSet<PathBuf> = (self.materials.values())
                .flat_map(|u| u.textures.iter().cloned())
                .collect();
            self.textures.retain(&used);
            self.stale = false;
        }
        let mut update = Update::default();
        let ids: Vec<u64> = if cap == self.cap {
            moved.materials.iter().copied().collect()
        } else {
            store.materials().keys().copied().collect()
        };
        self.cap = cap;
        for id in ids {
            let Some(material) = store.materials().get(&id) else {
                continue;
            };
            let Some(network) = &material.network else {
                continue;
            };
            self.failed.remove(&id);
            self.stale = true;
            if let Err(e) = self.material(id, network) {
                eprintln!("pika: warning: material {}: {e}; drawn pink", material.path);
                self.materials.remove(&id);
                self.failed.insert(id);
            }
        }
        for built in self.builder.drain() {
            let Some(material) = store.materials().get(&built.id) else {
                continue;
            };
            let id = built.id;
            self.stale = true;
            let (surfaces, textures) = (&mut self.surfaces, &mut self.textures);
            match Shade::install(device, surfaces, textures, built, cb, &mut update) {
                Ok(uniforms) => {
                    self.materials.insert(id, uniforms);
                }
                Err(e) => {
                    eprintln!("pika: warning: material {}: {e}; drawn pink", material.path);
                    self.materials.remove(&id);
                    self.failed.insert(id);
                }
            }
        }
        update.pending = self.builder.len();

        let [x, y, z, w] = camera.eye;
        // An orthographic camera's eye is the direction toward the viewer:
        // the generated code takes a position, so it is put far along it,
        // where the view vector is that direction from everywhere drawn.
        let far = if w == 0.0 { 1e7 } else { 1.0 };
        let eye = [x * far, y * far, z * far];
        let [a, b, c, _] = camera.forward;
        for u in (self.materials.values_mut()).chain([&mut self.clay, &mut self.pink]) {
            let layout = &self.surfaces[&u.hash].layout;
            surface::frame(layout, u.buffer.bytes(), eye, [a, b, c]);
        }
        Ok(update)
    }

    /// Rewrites material `id`'s block from `network` when it keeps its
    /// document, textures and cap, else submits its build and drops it,
    /// drawn clay until the build lands. A network with no surface leaves
    /// the material drawn clay.
    fn material(&mut self, id: u64, network: &Network) -> Result<(), String> {
        let Some(document) = material::document(network)? else {
            self.builder.forget(id);
            self.materials.remove(&id);
            return Ok(());
        };
        if let Some(uniforms) = self.materials.get_mut(&id)
            && uniforms.hash == document.hash
            && uniforms.cap == self.cap
        {
            let surface = &self.surfaces[&document.hash];
            if build::sources(&surface.manifest, &document, network)? == uniforms.sources {
                self.builder.forget(id);
                return surface::pack(
                    &surface.manifest,
                    &surface.layout,
                    &document,
                    network,
                    uniforms.buffer.bytes(),
                );
            }
        }
        self.materials.remove(&id);
        let held = self.surfaces.contains_key(&document.hash);
        (self.builder).submit(id, document, network.clone(), self.cap, held)
    }

    /// Installs a build that landed: its surface into `surfaces` unless
    /// held, its uniforms, and its textures, loaded now into `textures`
    /// with their uploads recorded into `cb`; counts into `update` what the
    /// build did.
    fn install(
        device: &gpu::Device,
        surfaces: &mut BTreeMap<u64, Surface>,
        textures: &mut Textures,
        built: Built,
        cb: vk::CommandBuffer,
        update: &mut Update,
    ) -> Result<Uniforms, String> {
        let output = built.result?;
        let hash = built.document.hash;
        if let Some(compiled) = output.surface
            && !surfaces.contains_key(&hash)
        {
            update.compiled += usize::from(compiled.compiled);
            let layout = surface::layout(&compiled.reflection)?;
            let surface = Surface::new(device, &compiled.shader.spirv, output.manifest, layout)?;
            surfaces.insert(hash, surface);
        }
        let surface = &surfaces[&hash];
        let (manifest, layout) = (&surface.manifest, &surface.layout);
        let mut uniforms = Uniforms::new(device, surface, hash)?;
        surface::pack(
            manifest,
            layout,
            &built.document,
            &built.network,
            uniforms.buffer.bytes(),
        )?;
        let mut bindings = Vec::new();
        for (name, baked) in manifest.textures.iter().zip(&output.textures) {
            let (index, _) = surface::node_input(name)?;
            update.baked += usize::from(baked.fresh);
            let binding = textures.load(device, cb, &baked.path, &baked.texture, &baked.source)?;
            let node = &built.network.nodes[&built.document.nodes[index]];
            let modes = surface::address_modes(
                layout,
                uniforms.buffer.bytes(),
                node,
                index,
                baked.texture.wrap,
            );
            let sampler = textures.sampler(modes)?;
            bindings.push((binding, sampler));
            uniforms.textures.push(baked.path.clone());
        }
        uniforms.bind(layout, &bindings);
        uniforms.sources = output.sources;
        uniforms.cap = built.cap;
        Ok(uniforms)
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
                let uniforms = match group.material {
                    Some(id) if self.failed.contains(&id) => &self.pink,
                    Some(id) => self.materials.get(&id).unwrap_or(&self.clay),
                    None => &self.clay,
                };
                let surface = &self.surfaces[&uniforms.hash];
                let point = vk::PipelineBindPoint::GRAPHICS;
                let layout = surface.pipeline_layout;
                d.cmd_bind_pipeline(cb, point, surface.pipelines[index]);
                let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
                d.cmd_push_constants(cb, layout, stages, 0, &push);
                d.cmd_bind_descriptor_sets(cb, point, layout, 0, &[uniforms.set], &[]);
                let offset = u64::from(group.first * stride);
                d.cmd_draw_indexed_indirect(cb, view.records(), offset, group.count, stride);
            }
        }
    }

    pub fn memory(&self) -> gpu::Memory {
        let mut memory = self.textures.memory();
        for u in self.materials.values().chain([&self.clay, &self.pink]) {
            memory += u.buffer.memory();
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

impl Uniforms {
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
            cap: 0,
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

    /// Writes the set: the block, and each of `layout`'s textures from
    /// `bindings` in order. The GPU must not be reading the set.
    fn bind(&self, layout: &Layout, bindings: &[(Binding, vk::Sampler)]) {
        assert_eq!(bindings.len(), layout.textures.len());
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
    }
}

impl Drop for Uniforms {
    fn drop(&mut self) {
        // Destroying the pool frees its set; the buffer drops after.
        // SAFETY: the pool was created on `device`, and the owner's target
        // waited for idle before this drops (field order).
        unsafe { self.device.destroy_descriptor_pool(self.pool, None) };
    }
}
