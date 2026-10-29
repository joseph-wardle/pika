//! Vulkan instance, validation, device choice and logical device, and the
//! buffers and compute pipelines created on it.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use ash::vk;

/// Pika needs Vulkan 1.3 for dynamic rendering and synchronization2.
pub const REQUIRED_API: u32 = vk::API_VERSION_1_3;

const VALIDATION_LAYER: &std::ffi::CStr = c"VK_LAYER_KHRONOS_validation";

/// What Pika needs to know about a physical device to choose one.
pub struct DeviceInfo {
    pub handle: vk::PhysicalDevice,
    pub name: String,
    pub kind: vk::PhysicalDeviceType,
    pub api_version: u32,
    pub vram_bytes: u64,
    pub ray_query: bool,
    /// `VK_EXT_memory_budget`, which says how much of the VRAM this process
    /// uses and may use.
    pub memory_budget: bool,
    /// First queue family that supports graphics, compute and timestamps.
    pub queue_family: Option<u32>,
    /// That family's valid timestamp bits.
    pub timestamp_bits: u32,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period: f32,
}

impl DeviceInfo {
    /// Why Pika cannot use this device, or `None` if it can.
    pub fn unsuitable_reason(&self) -> Option<&'static str> {
        if self.kind == vk::PhysicalDeviceType::CPU {
            Some("CPU device")
        } else if self.api_version < REQUIRED_API {
            Some("needs Vulkan 1.3")
        } else if self.queue_family.is_none() {
            Some("no graphics+compute queue with timestamps")
        } else {
            None
        }
    }
}

/// Picks the suitable device most likely to be the artist's GPU:
/// discrete before integrated before virtual, then the most VRAM.
pub fn choose(devices: &[DeviceInfo]) -> Option<usize> {
    let rank = |kind| match kind {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        _ => 3,
    };
    devices
        .iter()
        .enumerate()
        .filter(|(_, d)| d.unsuitable_reason().is_none())
        .min_by_key(|(_, d)| (rank(d.kind), std::cmp::Reverse(d.vram_bytes)))
        .map(|(i, _)| i)
}

/// The first memory type allowed by `type_bits` (from `vkGet*MemoryRequirements`)
/// that has every flag in `flags`.
pub fn memory_type(
    types: &[vk::MemoryType],
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32, String> {
    (0..types.len() as u32)
        .find(|&i| type_bits & (1 << i) != 0 && types[i as usize].property_flags.contains(flags))
        .ok_or_else(|| format!("no memory type has {flags:?} (allowed types {type_bits:#b})"))
}

pub fn version_string(v: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}

/// A Vulkan instance with every physical device it reports.
pub struct Instance {
    // `entry` keeps libvulkan loaded; it must outlive `raw`.
    _entry: ash::Entry,
    raw: ash::Instance,
    // The messenger's callback writes through a pointer to this counter, so it
    // is boxed and outlives the messenger.
    messages: Box<AtomicU32>,
    messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub loader_version: u32,
    pub devices: Vec<DeviceInfo>,
}

impl Instance {
    pub fn new() -> Result<Self, String> {
        // SAFETY: loading libvulkan runs its initialisers; the loader is trusted.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| {
            format!(
                "could not load the Vulkan loader (libvulkan.so.1): {e}. \
                 Install it (RHEL: `dnf install vulkan-loader`, Ubuntu: `apt install libvulkan1`) \
                 and a GPU driver."
            )
        })?;

        // SAFETY: `entry` is a loaded Vulkan library.
        let loader_version = unsafe { entry.try_enumerate_instance_version() }
            .map_err(|e| format!("vkEnumerateInstanceVersion failed: {e}"))?
            .unwrap_or(vk::API_VERSION_1_0);
        if loader_version < REQUIRED_API {
            return Err(format!(
                "the Vulkan loader is {}; Pika needs 1.3 or newer. Update the Vulkan loader.",
                version_string(loader_version)
            ));
        }

        let validation = validation_found(&entry)?;
        let messages = Box::new(AtomicU32::new(0));

        // Sync validation is off in the layer's defaults.
        let sync_on = vk::TRUE;
        let mut sync = vk::LayerSettingEXT::default()
            .layer_name(VALIDATION_LAYER)
            .setting_name(c"validate_sync")
            .ty(vk::LayerSettingTypeEXT::BOOL32);
        // ash's `values` takes bytes and would count four values, not one.
        sync.value_count = 1;
        sync.p_values = (&raw const sync_on).cast();
        let settings = [sync];
        let mut layer_settings = vk::LayerSettingsCreateInfoEXT::default().settings(&settings);
        let mut messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(count_message))
            .user_data((&raw const *messages).cast_mut().cast());

        let app = vk::ApplicationInfo::default()
            .application_name(c"pika")
            .api_version(REQUIRED_API);
        let layers = [VALIDATION_LAYER.as_ptr()];
        let extensions = [
            vk::EXT_DEBUG_UTILS_NAME.as_ptr(),
            vk::EXT_LAYER_SETTINGS_NAME.as_ptr(),
        ];
        let mut create_info = vk::InstanceCreateInfo::default().application_info(&app);
        if validation {
            // The chained messenger also reports on vkCreateInstance itself.
            create_info = create_info
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions)
                .push_next(&mut layer_settings)
                .push_next(&mut messenger_info);
        }
        // SAFETY: `create_info` and what it points to outlive the call.
        let raw = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| format!("vkCreateInstance failed: {e}"))?;

        // From here `raw` must be destroyed on error, so build `Self` first.
        let mut instance = Self {
            messenger: None,
            messages,
            raw,
            loader_version,
            devices: Vec::new(),
            _entry: entry,
        };
        if validation {
            let loader = ash::ext::debug_utils::Instance::new(&instance._entry, &instance.raw);
            // SAFETY: the instance enabled VK_EXT_debug_utils, and `user_data`
            // points to `instance.messages`, which outlives the messenger.
            let messenger = unsafe { loader.create_debug_utils_messenger(&messenger_info, None) }
                .map_err(|e| format!("vkCreateDebugUtilsMessengerEXT failed: {e}"))?;
            instance.messenger = Some((loader, messenger));
        }
        // SAFETY: `instance.raw` is a live instance.
        let physical = unsafe { instance.raw.enumerate_physical_devices() }
            .map_err(|e| format!("vkEnumeratePhysicalDevices failed: {e}"))?;
        instance.devices = physical
            .into_iter()
            .map(|pd| instance.describe(pd))
            .collect::<Result<_, _>>()?;
        Ok(instance)
    }

    /// Warnings and errors the validation layer has reported so far, or `None`
    /// if the layer was not found.
    pub fn validation_messages(&self) -> Option<u32> {
        self.messenger
            .as_ref()
            .map(|_| self.messages.load(Ordering::Relaxed))
    }

    fn describe(&self, pd: vk::PhysicalDevice) -> Result<DeviceInfo, String> {
        // SAFETY: `pd` came from this instance, which is live.
        let (props, memory, families, extensions) = unsafe {
            (
                self.raw.get_physical_device_properties(pd),
                self.raw.get_physical_device_memory_properties(pd),
                self.raw.get_physical_device_queue_family_properties(pd),
                self.raw.enumerate_device_extension_properties(pd),
            )
        };
        let extensions =
            extensions.map_err(|e| format!("vkEnumerateDeviceExtensionProperties failed: {e}"))?;

        let name = props
            .device_name_as_c_str()
            .map_or_else(|_| "<unnamed>".into(), |n| n.to_string_lossy().into_owned());
        let vram_bytes = memory
            .memory_heaps_as_slice()
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .sum();
        let has = |name| {
            extensions
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let ray_query = has(vk::KHR_RAY_QUERY_NAME);
        let memory_budget = has(vk::EXT_MEMORY_BUDGET_NAME);
        let wanted = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        let queue_family = families
            .iter()
            .position(|f| f.queue_flags.contains(wanted) && f.timestamp_valid_bits > 0);

        Ok(DeviceInfo {
            handle: pd,
            name,
            kind: props.device_type,
            api_version: props.api_version,
            vram_bytes,
            ray_query,
            memory_budget,
            queue_family: queue_family.map(|i| i as u32),
            timestamp_bits: queue_family.map_or(0, |i| families[i].timestamp_valid_bits),
            timestamp_period: props.limits.timestamp_period,
        })
    }

    /// Opens a logical device with one graphics+compute queue on `devices[index]`,
    /// with dynamic rendering and synchronization2 enabled.
    pub fn open(&self, index: usize) -> Result<Device<'_>, String> {
        let info = &self.devices[index];
        let queue_family = info.queue_family.ok_or_else(|| {
            format!(
                "{} has no graphics+compute queue with timestamps",
                info.name
            )
        })?;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        // Dynamic rendering, synchronization2 and buffer device addresses are
        // mandatory in Vulkan 1.3. The rest are optional, but every desktop
        // GPU has them.
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let mut vulkan12 =
            vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
        let features = vk::PhysicalDeviceFeatures::default()
            .multi_draw_indirect(true)
            .draw_indirect_first_instance(true);
        let extensions = [vk::EXT_MEMORY_BUDGET_NAME.as_ptr()];
        let extensions = &extensions[..usize::from(info.memory_budget)];
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_features(&features)
            .enabled_extension_names(extensions)
            .push_next(&mut vulkan12)
            .push_next(&mut vulkan13);
        // SAFETY: the physical device belongs to this live instance and
        // `create_info` outlives the call.
        let raw =
            unsafe { self.raw.create_device(info.handle, &create_info, None) }.map_err(|e| {
                format!(
                    "vkCreateDevice on {} failed: {e}. Pika needs multiDrawIndirect and \
                 drawIndirectFirstInstance.",
                    info.name
                )
            })?;
        // SAFETY: `raw` was created with one queue in `queue_family`.
        let queue = unsafe { raw.get_device_queue(queue_family, 0) };
        // SAFETY: the physical device belongs to this live instance.
        let memory = unsafe { self.raw.get_physical_device_memory_properties(info.handle) };
        Ok(Device {
            instance: self,
            physical: info.handle,
            raw,
            queue_family,
            queue,
            timestamp_bits: info.timestamp_bits,
            timestamp_period: info.timestamp_period,
            memory_types: memory.memory_types_as_slice().to_vec(),
            memory_budget: info.memory_budget,
        })
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        if let Some((loader, messenger)) = &self.messenger {
            // SAFETY: the messenger was created on this instance and is destroyed once.
            unsafe { loader.destroy_debug_utils_messenger(*messenger, None) };
        }
        // SAFETY: every `Device` borrows this instance, so none outlives it.
        unsafe { self.raw.destroy_instance(None) };
    }
}

/// Whether the validation layer is installed. It must also provide
/// VK_EXT_layer_settings, which Pika uses to turn on sync validation.
fn validation_found(entry: &ash::Entry) -> Result<bool, String> {
    // SAFETY: `entry` is a loaded Vulkan library.
    let layers = unsafe { entry.enumerate_instance_layer_properties() }
        .map_err(|e| format!("vkEnumerateInstanceLayerProperties failed: {e}"))?;
    if !layers
        .iter()
        .any(|l| l.layer_name_as_c_str() == Ok(VALIDATION_LAYER))
    {
        return Ok(false);
    }
    // SAFETY: `entry` is a loaded Vulkan library.
    let extensions =
        unsafe { entry.enumerate_instance_extension_properties(Some(VALIDATION_LAYER)) }
            .map_err(|e| format!("vkEnumerateInstanceExtensionProperties failed: {e}"))?;
    if !extensions
        .iter()
        .any(|e| e.extension_name_as_c_str() == Ok(vk::EXT_LAYER_SETTINGS_NAME))
    {
        return Err(
            "the Vulkan validation layer is too old to provide VK_EXT_layer_settings. \
             Point VK_ADD_LAYER_PATH at Vulkan SDK 1.3.268 or newer."
                .into(),
        );
    }
    Ok(true)
}

/// Counts and prints each message the validation layer reports.
unsafe extern "system" fn count_message(
    _severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    messages: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the layer passes valid callback data, and `messages` is the
    // instance's boxed counter, which outlives the messenger.
    let (text, messages) = unsafe { ((*data).message_as_c_str(), &*messages.cast::<AtomicU32>()) };
    messages.fetch_add(1, Ordering::Relaxed);
    let text = text.map_or("<no message>".into(), |t| t.to_string_lossy());
    eprintln!("validation: {text}");
    vk::FALSE
}

/// A logical device. It borrows its `Instance` so it cannot outlive it.
pub struct Device<'a> {
    instance: &'a Instance,
    physical: vk::PhysicalDevice,
    pub raw: ash::Device,
    pub queue_family: u32,
    pub queue: vk::Queue,
    pub timestamp_bits: u32,
    pub timestamp_period: f32,
    pub memory_types: Vec<vk::MemoryType>,
    memory_budget: bool,
}

impl Device<'_> {
    /// The bytes of device-local memory this process uses, and the bytes
    /// the driver lets it use before allocations fail or spill to host
    /// memory, summed over the device-local heaps. Both are 0 on a driver
    /// without `VK_EXT_memory_budget`.
    pub fn budget(&self) -> (u64, u64) {
        if !self.memory_budget {
            return (0, 0);
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: the physical device belongs to the live instance, and the
        // extension that fills `budget` is enabled.
        unsafe {
            (self.instance.raw)
                .get_physical_device_memory_properties2(self.physical, &mut properties)
        };
        let heaps = properties.memory_properties.memory_heaps_as_slice();
        (0..heaps.len())
            .filter(|&i| heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .fold((0, 0), |(used, allowed), i| {
                (used + budget.heap_usage[i], allowed + budget.heap_budget[i])
            })
    }
}

impl Drop for Device<'_> {
    fn drop(&mut self) {
        // SAFETY: every object created on this device borrows it, so all have
        // been destroyed, and their owners waited for idle first.
        unsafe { self.raw.destroy_device(None) };
    }
}

/// Allocates memory of the first type allowed by `needs` that has `flags`.
pub fn allocate(
    device: &Device,
    needs: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
    allocate_flags: vk::MemoryAllocateFlags,
) -> Result<vk::DeviceMemory, String> {
    let mut flags_info = vk::MemoryAllocateFlagsInfo::default().flags(allocate_flags);
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(needs.size)
        .memory_type_index(memory_type(
            &device.memory_types,
            needs.memory_type_bits,
            flags,
        )?)
        .push_next(&mut flags_info);
    // SAFETY: `device.raw` is live and `info` outlives the call.
    unsafe { device.raw.allocate_memory(&info, None) }.map_err(|e| {
        format!(
            "vkAllocateMemory ({} bytes, {flags:?}) failed: {e}",
            needs.size
        )
    })
}

/// Bytes allocated in device-local and in host-visible memory, for the stats.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Memory {
    pub device: u64,
    pub host: u64,
}

impl std::ops::AddAssign for Memory {
    fn add_assign(&mut self, other: Self) {
        self.device += other.device;
        self.host += other.host;
    }
}

/// A buffer and the memory only it uses. Device-local buffers have a device
/// address, through which shaders read them; host buffers stay mapped, and
/// have one when their usage asks for it.
/// Every handle starts null, so a failed `new` can drop a partly built value.
pub struct Buffer {
    // A copy of the owner's device table. Every owner keeps the `Device`
    // alive longer than its buffers (field order), which `drop` relies on.
    device: ash::Device,
    pub raw: vk::Buffer,
    memory: vk::DeviceMemory,
    pub size: u64,
    /// 0 for a host buffer without SHADER_DEVICE_ADDRESS usage.
    pub address: vk::DeviceAddress,
    mapped: *mut u8,
}

// SAFETY: `mapped` points into memory this buffer owns, and it is only
// dereferenced through `&mut self`, so moving the buffer to another thread is sound.
unsafe impl Send for Buffer {}

impl Buffer {
    /// A device-local buffer of `size` bytes, with a device address.
    pub fn device_local(
        device: &Device,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self, String> {
        let usage = usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let mut b = Self::create(device, size, usage)?;
        // SAFETY: `b.raw` was created on `device.raw`, which is live.
        let needs = unsafe { device.raw.get_buffer_memory_requirements(b.raw) };
        b.memory = allocate(
            device,
            needs,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryAllocateFlags::DEVICE_ADDRESS,
        )?;
        // SAFETY: as above, and the memory was just allocated for this buffer.
        unsafe { device.raw.bind_buffer_memory(b.raw, b.memory, 0) }
            .map_err(|e| format!("vkBindBufferMemory failed: {e}"))?;
        let info = vk::BufferDeviceAddressInfo::default().buffer(b.raw);
        // SAFETY: the buffer was created with SHADER_DEVICE_ADDRESS and is bound.
        b.address = unsafe { device.raw.get_buffer_device_address(&info) };
        Ok(b)
    }

    /// A mapped host buffer of `size` bytes in memory with `flags`, which
    /// must include HOST_VISIBLE and HOST_COHERENT.
    pub fn host(
        device: &Device,
        size: u64,
        usage: vk::BufferUsageFlags,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<Self, String> {
        let mut b = Self::create(device, size, usage)?;
        // SAFETY: `b.raw` was created on `device.raw`, which is live.
        let needs = unsafe { device.raw.get_buffer_memory_requirements(b.raw) };
        let addressed = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let allocate_flags = match addressed {
            true => vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            false => vk::MemoryAllocateFlags::empty(),
        };
        b.memory = allocate(device, needs, flags, allocate_flags)?;
        // SAFETY: as above, and the memory was just allocated for this buffer.
        unsafe { device.raw.bind_buffer_memory(b.raw, b.memory, 0) }
            .map_err(|e| format!("vkBindBufferMemory failed: {e}"))?;
        if addressed {
            let info = vk::BufferDeviceAddressInfo::default().buffer(b.raw);
            // SAFETY: the buffer was created with SHADER_DEVICE_ADDRESS and is bound.
            b.address = unsafe { device.raw.get_buffer_device_address(&info) };
        }
        // Freeing the memory in `drop` also unmaps it.
        // SAFETY: the memory is host visible and not yet mapped.
        let mapped = unsafe {
            device
                .raw
                .map_memory(b.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        };
        b.mapped = mapped
            .map_err(|e| format!("vkMapMemory failed: {e}"))?
            .cast();
        Ok(b)
    }

    fn create(device: &Device, size: u64, usage: vk::BufferUsageFlags) -> Result<Self, String> {
        let mut b = Self {
            device: device.raw.clone(),
            raw: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size,
            address: 0,
            mapped: std::ptr::null_mut(),
        };
        // Vulkan has no empty buffers.
        let info = vk::BufferCreateInfo::default()
            .size(size.max(16))
            .usage(usage);
        // SAFETY: `device.raw` is live and `info` outlives the call.
        b.raw = unsafe { device.raw.create_buffer(&info, None) }
            .map_err(|e| format!("vkCreateBuffer ({size} bytes, {usage:?}) failed: {e}"))?;
        Ok(b)
    }

    /// A host buffer's bytes. The owner must have waited for any GPU work
    /// that uses the buffer.
    pub fn bytes(&mut self) -> &mut [u8] {
        assert!(!self.mapped.is_null(), "a device-local buffer has no bytes");
        // SAFETY: `mapped` is the whole of the buffer's memory, at least
        // `size` bytes long.
        unsafe { std::slice::from_raw_parts_mut(self.mapped, self.size as usize) }
    }

    /// The bytes this buffer holds, on the side it was allocated on.
    pub fn memory(&self) -> Memory {
        match self.mapped.is_null() {
            true => Memory {
                device: self.size,
                host: 0,
            },
            false => Memory {
                device: 0,
                host: self.size,
            },
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: both handles were created on `device` or are null, which
        // destroy calls accept, and the owner waited for any work using them.
        unsafe {
            self.device.destroy_buffer(self.raw, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// A compute pipeline whose only input is its push constants: buffer
/// addresses and a range of items, which its shader walks in a grid-stride
/// loop of `numthreads(64, 1, 1)` groups.
pub struct Compute {
    // A copy of the owner's device table, which outlives this (field order).
    device: ash::Device,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Compute {
    /// The threads in a group, as the shaders declare.
    const GROUP_SIZE: u64 = 64;
    /// Vulkan's guaranteed minimum of groups in one dispatch dimension.
    const MAX_GROUPS: u64 = 65535;

    /// The pipeline of `spirv`'s entry point `cs`, with `push_size` bytes of
    /// push constants.
    pub fn new(device: &Device, spirv: &[u32], push_size: usize) -> Result<Self, String> {
        let d = &device.raw;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_size as u32)];
        let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push);
        // SAFETY: `d` is live and `layout_info` outlives the call.
        let layout = unsafe { d.create_pipeline_layout(&layout_info, None) }
            .map_err(|e| format!("vkCreatePipelineLayout failed: {e}"))?;
        let mut c = Self {
            device: d.clone(),
            layout,
            pipeline: vk::Pipeline::null(),
        };
        let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: `d` is live and `module_info` outlives the call.
        let module = unsafe { d.create_shader_module(&module_info, None) }
            .map_err(|e| format!("vkCreateShaderModule failed: {e}"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"cs");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(c.layout);
        // SAFETY: `d` is live, `module` and the layout were created on it,
        // and `info` outlives the call.
        let built = unsafe { d.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
        // SAFETY: pipeline creation is complete, so the module is no longer used.
        unsafe { d.destroy_shader_module(module, None) };
        c.pipeline = built
            .map(|p| p[0])
            .map_err(|(_, e)| format!("vkCreateComputePipelines failed: {e}"))?;
        Ok(c)
    }

    /// Records into `cb`, outside rendering, one dispatch per range of
    /// items. `push` is called with each range's first item, count and
    /// stride, the threads dispatched, and returns the push constants: a
    /// plain `repr(C)` struct without padding.
    pub fn record<P: Copy>(
        &self,
        cb: vk::CommandBuffer,
        ranges: &[std::ops::Range<u32>],
        push: impl Fn(u32, u32, u32) -> P,
    ) {
        let d = &self.device;
        // SAFETY: `cb` is recording outside rendering, and the pipeline was
        // created on `d`.
        unsafe { d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline) };
        for r in ranges.iter().filter(|r| !r.is_empty()) {
            let count = r.end - r.start;
            let groups = (count as u64)
                .div_ceil(Self::GROUP_SIZE)
                .min(Self::MAX_GROUPS);
            let constants = push(r.start, count, (groups * Self::GROUP_SIZE) as u32);
            // SAFETY: `P` is plain data without padding, so all its bytes are
            // initialized.
            let bytes = unsafe {
                std::slice::from_raw_parts((&raw const constants).cast::<u8>(), size_of::<P>())
            };
            // SAFETY: as above, and the layout was created on `d`.
            unsafe {
                d.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
                d.cmd_dispatch(cb, groups as u32, 1, 1);
            }
        }
    }
}

impl Drop for Compute {
    fn drop(&mut self) {
        // SAFETY: both handles were created on `device` or are null, and the
        // owner's target waited for idle before this drops (field order).
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(kind: vk::PhysicalDeviceType, api_version: u32, vram_gib: u64) -> DeviceInfo {
        DeviceInfo {
            handle: vk::PhysicalDevice::null(),
            name: format!("{kind:?}"),
            kind,
            api_version,
            vram_bytes: vram_gib << 30,
            ray_query: false,
            memory_budget: false,
            queue_family: Some(0),
            timestamp_bits: 64,
            timestamp_period: 1.0,
        }
    }

    fn memory(flags: vk::MemoryPropertyFlags) -> vk::MemoryType {
        vk::MemoryType {
            property_flags: flags,
            heap_index: 0,
        }
    }

    #[test]
    fn memory_type_needs_all_flags_and_an_allowed_type() {
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let types = [
            memory(vk::MemoryPropertyFlags::DEVICE_LOCAL),
            memory(vk::MemoryPropertyFlags::HOST_VISIBLE),
            memory(host),
            memory(host | vk::MemoryPropertyFlags::HOST_CACHED),
        ];
        assert_eq!(memory_type(&types, 0b1111, host), Ok(2));
        // Type 2 matches but is not allowed, so the next match is chosen.
        assert_eq!(memory_type(&types, 0b1011, host), Ok(3));
        assert!(memory_type(&types, 0b0011, host).is_err());
    }

    #[test]
    fn prefers_discrete_over_larger_integrated() {
        let devices = [
            device(vk::PhysicalDeviceType::INTEGRATED_GPU, REQUIRED_API, 32),
            device(vk::PhysicalDeviceType::DISCRETE_GPU, REQUIRED_API, 16),
        ];
        assert_eq!(choose(&devices), Some(1));
    }

    #[test]
    fn prefers_more_vram_between_discrete() {
        let devices = [
            device(vk::PhysicalDeviceType::DISCRETE_GPU, REQUIRED_API, 12),
            device(vk::PhysicalDeviceType::DISCRETE_GPU, REQUIRED_API, 16),
        ];
        assert_eq!(choose(&devices), Some(1));
    }

    #[test]
    fn rejects_cpu_old_api_and_missing_queue() {
        let mut no_queue = device(vk::PhysicalDeviceType::DISCRETE_GPU, REQUIRED_API, 16);
        no_queue.queue_family = None;
        let devices = [
            // A software rasteriser such as llvmpipe is never chosen.
            device(vk::PhysicalDeviceType::CPU, REQUIRED_API, 64),
            device(
                vk::PhysicalDeviceType::DISCRETE_GPU,
                vk::API_VERSION_1_2,
                16,
            ),
            no_queue,
        ];
        assert_eq!(choose(&devices), None);
        assert_eq!(devices[0].unsuitable_reason(), Some("CPU device"));
        assert_eq!(devices[1].unsuitable_reason(), Some("needs Vulkan 1.3"));
        assert_eq!(
            devices[2].unsuitable_reason(),
            Some("no graphics+compute queue with timestamps")
        );
    }
}
