//! The material pass: each surface's pipelines, built from the bake tool's
//! pixel stage and the vertex stage written for it, each material's
//! uniform block, textures and descriptor set, bound to its surface, and
//! the draw of each material's meshes after the prepass, lit by a
//! headlight.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use ash::vk;

use crate::bake::{Bake, Manifest};
use crate::cache;
use crate::geometry::{Bound, Group};
use crate::gpu::{self, Buffer};
use crate::material::{self, Document};
use crate::raster::{COLOR_FORMATS, DEPTH_AOV, NORMAL_AOV};
use crate::shader;
use crate::store::{Moved, Store};
use crate::surface::{self, Layout};
use crate::target;
use crate::texture::Textures;
use crate::transform::Camera;
use crate::view::View;

/// The texels a texture's longest side is capped at, until the cap is a
/// render setting.
const CAP: u32 = 4096;

pub struct Shade {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    bake: Bake,
    textures: Textures,
    /// By document hash.
    surfaces: BTreeMap<u64, Surface>,
    /// By the store's material id.
    materials: BTreeMap<u64, Uniforms>,
    /// `material::clay`, drawn where `materials` has no entry.
    clay: Uniforms,
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
    /// The baked textures bound, in the surface's order.
    textures: Vec<PathBuf>,
}

/// What one frame's `update` did.
pub struct Update {
    /// Surfaces slangc compiled now; the cache answered for the rest.
    pub compiled: usize,
    /// Textures the bake tool baked now, likewise.
    pub baked: usize,
}

impl Shade {
    pub fn new(device: &gpu::Device) -> Result<Self, String> {
        let bake = Bake::new()?;
        let textures = Textures::new(device);
        let clay = material::clay();
        let document = material::document(&clay)
            .map_err(|e| format!("the clay material: {e}"))?
            .expect("clay has a surface");
        let (surface, _) = Shade::build(&bake, device, &document)
            .map_err(|e| format!("the clay material: {e}"))?;
        let mut uniforms = Uniforms::new(device, &surface, document.hash)
            .map_err(|e| format!("the clay material: {e}"))?;
        let (manifest, layout) = (&surface.manifest, &surface.layout);
        surface::pack(manifest, layout, &document, &clay, uniforms.buffer.bytes())
            .map_err(|e| format!("the clay material: {e}"))?;
        uniforms.bind(layout, &[]);
        Ok(Self {
            device: device.raw.clone(),
            bake,
            textures,
            surfaces: BTreeMap::from([(document.hash, surface)]),
            materials: BTreeMap::new(),
            clay: uniforms,
        })
    }

    /// Builds the surface of each material whose network changed, bakes
    /// and loads its textures and writes its uniforms, then every
    /// material's eye and headlight from `camera`. Records into `cb`,
    /// outside rendering, the uploads. No GPU work may be pending.
    pub fn update(
        &mut self,
        device: &gpu::Device,
        store: &Store,
        moved: &Moved,
        camera: &Camera,
        cb: vk::CommandBuffer,
    ) -> Result<Update, String> {
        self.textures.begin();
        self.materials
            .retain(|id, _| store.materials().contains_key(id));
        let (mut compiled, mut baked) = (0, 0);
        for id in &moved.materials {
            let Some(material) = store.materials().get(id) else {
                continue;
            };
            let Some(network) = &material.network else {
                continue;
            };
            let at = |e: String| format!("material {}: {e}", material.path);
            let Some(document) = material::document(network).map_err(at)? else {
                self.materials.remove(id);
                continue;
            };
            if !self.surfaces.contains_key(&document.hash) {
                let (surface, fresh) = Shade::build(&self.bake, device, &document).map_err(at)?;
                compiled += usize::from(fresh);
                self.surfaces.insert(document.hash, surface);
            }
            let surface = &self.surfaces[&document.hash];
            let uniforms = match self.materials.get_mut(id) {
                Some(u) if u.hash == document.hash => u,
                _ => {
                    let u = Uniforms::new(device, surface, document.hash)?;
                    self.materials.insert(*id, u);
                    self.materials.get_mut(id).unwrap()
                }
            };
            let (manifest, layout) = (&surface.manifest, &surface.layout);
            surface::pack(
                manifest,
                layout,
                &document,
                network,
                uniforms.buffer.bytes(),
            )
            .map_err(at)?;
            let mut bindings = Vec::new();
            uniforms.textures.clear();
            for name in &manifest.textures {
                let (index, input) = surface::node_input(name).map_err(at)?;
                let path = (document.nodes.get(index))
                    .ok_or_else(|| at(format!("{name} names a node the document lacks")))?;
                let node = &network.nodes[path];
                let source = material::source(node, material::store_input(node, input))
                    .map_err(|e| at(format!("{path}'s {input} {e}")))?;
                let file = Path::new(&source.file);
                let (texture, fresh) = (self.bake)
                    .texture(file, &source.space, source.encoding, CAP)
                    .map_err(at)?;
                baked += usize::from(fresh);
                let view = self.textures.load(device, cb, &texture).map_err(at)?;
                let modes = surface::address_modes(layout, uniforms.buffer.bytes(), index);
                let sampler = self.textures.sampler(modes).map_err(at)?;
                bindings.push((view, sampler));
                uniforms.textures.push(texture);
            }
            uniforms.bind(layout, &bindings);
        }
        let used: BTreeSet<PathBuf> = (self.materials.values())
            .flat_map(|u| u.textures.iter().cloned())
            .collect();
        self.textures.retain(&used);

        let [x, y, z, w] = camera.eye;
        // An orthographic camera's eye is the direction toward the viewer:
        // the generated code takes a position, so it is put far along it,
        // where the view vector is that direction from everywhere drawn.
        let far = if w == 0.0 { 1e7 } else { 1.0 };
        let eye = [x * far, y * far, z * far];
        let [a, b, c, _] = camera.forward;
        for u in self.materials.values_mut().chain([&mut self.clay]) {
            let layout = &self.surfaces[&u.hash].layout;
            surface::frame(layout, u.buffer.bytes(), eye, [a, b, c]);
        }
        Ok(Update { compiled, baked })
    }

    /// Records the draws of `view`'s `groups`, after the prepass, into `cb`,
    /// inside the same rendering: each group with its material's surface,
    /// and with clay where the material has none.
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
                let bound = group.material.and_then(|id| self.materials.get(&id));
                let uniforms = bound.unwrap_or(&self.clay);
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

    /// Generates, compiles and builds `document`'s surface, and whether
    /// slangc ran for it.
    fn build(
        bake: &Bake,
        device: &gpu::Device,
        document: &Document,
    ) -> Result<(Surface, bool), String> {
        let generated = bake.generate(document)?;
        let stage = surface::vertex_stage(&generated.manifest)?;
        let path = generated.dir.join("surface.slang");
        if std::fs::read_to_string(&path).ok().as_deref() != Some(&stage) {
            cache::write(&path, stage.as_bytes())?;
        }
        let compiled = shader::surface(&generated.dir)?;
        let layout = surface::layout(&compiled.reflection)?;
        let surface = Surface::new(device, &compiled.shader.spirv, generated.manifest, layout)?;
        Ok((surface, compiled.compiled))
    }

    pub fn memory(&self) -> gpu::Memory {
        let mut memory = self.textures.memory();
        for u in self.materials.values().chain([&self.clay]) {
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
    fn bind(&self, layout: &Layout, bindings: &[(vk::ImageView, vk::Sampler)]) {
        assert_eq!(bindings.len(), layout.textures.len());
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.buffer.raw)
            .range(vk::WHOLE_SIZE)];
        let images: Vec<_> = (bindings.iter())
            .map(|&(view, sampler)| {
                [
                    vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    vk::DescriptorImageInfo::default().sampler(sampler),
                ]
            })
            .collect();
        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(layout.block)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info),
        ];
        for (texture, info) in layout.textures.iter().zip(&images) {
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
        }
        // SAFETY: the set, buffer, views and samplers were created on
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
