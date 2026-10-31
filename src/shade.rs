//! The material pass: each surface's pipelines, built from the bake tool's
//! pixel stage and the vertex stage written for it, each material's
//! uniform block and descriptor set, bound to its surface, and the draw of
//! each material's meshes after the prepass, lit by a headlight.

use std::collections::BTreeMap;

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
use crate::transform::Camera;
use crate::view::View;

pub struct Shade {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    bake: Bake,
    sampler: vk::Sampler,
    /// A 1×1 white array texture, bound where no texture is loaded.
    white: White,
    /// Whether `white` has been cleared and laid out for sampling.
    ready: bool,
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
}

struct White {
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

/// What one frame's `update` did.
pub struct Update {
    /// Surfaces slangc compiled now; the cache answered for the rest.
    pub compiled: usize,
}

const WHITE_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

impl Shade {
    pub fn new(device: &gpu::Device) -> Result<Self, String> {
        let d = &device.raw;
        let bake = Bake::new()?;
        let white = White::new(device)?;
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .max_lod(vk::LOD_CLAMP_NONE);
        // SAFETY: `d` is live and `sampler_info` outlives the call.
        let sampler = unsafe { d.create_sampler(&sampler_info, None) }
            .map_err(|e| format!("vkCreateSampler failed: {e}"))?;
        let clay = material::clay();
        let document = material::document(&clay)
            .map_err(|e| format!("the clay material: {e}"))?
            .expect("clay has a surface");
        let (surface, _) = Shade::build(&bake, device, &document)
            .map_err(|e| format!("the clay material: {e}"))?;
        let mut uniforms = Uniforms::new(device, &surface, document.hash, &white, sampler);
        if let Ok(u) = &mut uniforms {
            let (manifest, layout) = (&surface.manifest, &surface.layout);
            surface::pack(manifest, layout, &document, &clay, u.buffer.bytes())
                .map_err(|e| format!("the clay material: {e}"))?;
        }
        Ok(Self {
            device: d.clone(),
            bake,
            sampler,
            white,
            ready: false,
            surfaces: BTreeMap::from([(document.hash, surface)]),
            materials: BTreeMap::new(),
            clay: uniforms?,
        })
    }

    /// Builds the surface of each material whose network changed, and
    /// writes its uniforms, then every material's eye and headlight from
    /// `camera`. Records into `cb`, outside rendering, what the first call
    /// needs on the GPU. No GPU work may be pending.
    pub fn update(
        &mut self,
        device: &gpu::Device,
        store: &Store,
        moved: &Moved,
        camera: &Camera,
        cb: vk::CommandBuffer,
    ) -> Result<Update, String> {
        if !self.ready {
            self.white.clear(cb);
            self.ready = true;
        }
        self.materials
            .retain(|id, _| store.materials().contains_key(id));
        let mut compiled = 0;
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
                    let u =
                        Uniforms::new(device, surface, document.hash, &self.white, self.sampler)?;
                    self.materials.insert(*id, u);
                    self.materials.get_mut(id).unwrap()
                }
            };
            let bytes = uniforms.buffer.bytes();
            surface::pack(
                &surface.manifest,
                &surface.layout,
                &document,
                network,
                bytes,
            )
            .map_err(at)?;
        }
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
        Ok(Update { compiled })
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
        let mut memory = gpu::Memory {
            device: self.white.bytes(),
            host: 0,
        };
        for u in self.materials.values().chain([&self.clay]) {
            memory += u.buffer.memory();
        }
        memory
    }
}

impl Drop for Shade {
    fn drop(&mut self) {
        // The surfaces and materials drop after this, before the device.
        // SAFETY: the sampler was created on `device`, and the owner's
        // target waited for idle before this drops (field order).
        unsafe { self.device.destroy_sampler(self.sampler, None) };
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
    fn new(
        device: &gpu::Device,
        surface: &Surface,
        hash: u64,
        white: &White,
        sampler: vk::Sampler,
    ) -> Result<Self, String> {
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
        };
        let set_layouts = [surface.set_layout];
        let set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&set_layouts);
        u.set = unsafe { d.allocate_descriptor_sets(&set_info) }
            .map_err(|e| format!("vkAllocateDescriptorSets failed: {e}"))?[0];

        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(u.buffer.raw)
            .range(vk::WHOLE_SIZE)];
        let image_info = [vk::DescriptorImageInfo::default()
            .image_view(white.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let sampler_info = [vk::DescriptorImageInfo::default().sampler(sampler)];
        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(u.set)
                .dst_binding(layout.block)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info),
        ];
        for texture in &layout.textures {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(u.set)
                    .dst_binding(texture.image)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&image_info),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(u.set)
                    .dst_binding(texture.sampler)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&sampler_info),
            );
        }
        unsafe { d.update_descriptor_sets(&writes, &[]) };
        Ok(u)
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

impl White {
    fn new(device: &gpu::Device) -> Result<Self, String> {
        let d = &device.raw;
        let format = vk::Format::R8G8B8A8_UNORM;
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST);
        let mut w = Self {
            device: d.clone(),
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
        };
        // SAFETY (this and every call below): `d` is live, each info struct
        // outlives its call, and each handle passed in was created on `d`
        // just above. `drop` destroys what was made before an error.
        w.image = unsafe { d.create_image(&info, None) }
            .map_err(|e| format!("vkCreateImage (1×1 {format:?}) failed: {e}"))?;
        let needs = unsafe { d.get_image_memory_requirements(w.image) };
        let flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        w.memory = gpu::allocate(device, needs, flags, vk::MemoryAllocateFlags::empty())?;
        unsafe { d.bind_image_memory(w.image, w.memory, 0) }
            .map_err(|e| format!("vkBindImageMemory failed: {e}"))?;
        // Every texture is an array: a UDIM set is one, with a layer per tile.
        let view_info = vk::ImageViewCreateInfo::default()
            .image(w.image)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(format)
            .subresource_range(WHITE_RANGE);
        w.view = unsafe { d.create_image_view(&view_info, None) }
            .map_err(|e| format!("vkCreateImageView failed: {e}"))?;
        Ok(w)
    }

    /// Records into `cb`, outside rendering, the clear to white and the
    /// move into the layout the material pass samples.
    fn clear(&self, cb: vk::CommandBuffer) {
        let to_clear = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(self.image)
            .subresource_range(WHITE_RANGE);
        let to_sample = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.image)
            .subresource_range(WHITE_RANGE);
        let white = vk::ClearColorValue { float32: [1.0; 4] };
        let d = &self.device;
        // SAFETY: `cb` is recording outside rendering, and the image was
        // created on `d` with TRANSFER_DST usage.
        unsafe {
            let barriers = [to_clear];
            let barrier = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            d.cmd_pipeline_barrier2(cb, &barrier);
            let layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            d.cmd_clear_color_image(cb, self.image, layout, &white, &[WHITE_RANGE]);
            let barriers = [to_sample];
            let barrier = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            d.cmd_pipeline_barrier2(cb, &barrier);
        }
    }

    fn bytes(&self) -> u64 {
        // SAFETY: the image was created on `device`, which is live.
        unsafe { self.device.get_image_memory_requirements(self.image) }.size
    }
}

impl Drop for White {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device` or is null, and the
        // owner's target waited for idle before this drops (field order).
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
