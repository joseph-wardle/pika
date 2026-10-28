//! Vulkan instance, validation, device choice and logical device.

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
    /// First queue family that supports both graphics and compute.
    pub queue_family: Option<u32>,
}

impl DeviceInfo {
    /// Why Pika cannot use this device, or `None` if it can.
    pub fn unsuitable_reason(&self) -> Option<&'static str> {
        if self.kind == vk::PhysicalDeviceType::CPU {
            Some("CPU device")
        } else if self.api_version < REQUIRED_API {
            Some("needs Vulkan 1.3")
        } else if self.queue_family.is_none() {
            Some("no graphics+compute queue")
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
#[cfg_attr(not(test), expect(dead_code, reason = "check.rs will use it"))]
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
        let ray_query = extensions
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(vk::KHR_RAY_QUERY_NAME));
        let wanted = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        let queue_family = families
            .iter()
            .position(|f| f.queue_flags.contains(wanted))
            .map(|i| i as u32);

        Ok(DeviceInfo {
            handle: pd,
            name,
            kind: props.device_type,
            api_version: props.api_version,
            vram_bytes,
            ray_query,
            queue_family,
        })
    }

    /// Opens a logical device with one graphics+compute queue on `devices[index]`,
    /// with dynamic rendering and synchronization2 enabled.
    pub fn open(&self, index: usize) -> Result<Device<'_>, String> {
        let info = &self.devices[index];
        let queue_family = info
            .queue_family
            .ok_or_else(|| format!("{} has no graphics+compute queue", info.name))?;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        // Both features are mandatory in Vulkan 1.3, so support needs no check.
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .push_next(&mut vulkan13);
        // SAFETY: the physical device belongs to this live instance and
        // `create_info` outlives the call.
        let raw = unsafe { self.raw.create_device(info.handle, &create_info, None) }
            .map_err(|e| format!("vkCreateDevice on {} failed: {e}", info.name))?;
        // SAFETY: `raw` was created with one queue in `queue_family`.
        let queue = unsafe { raw.get_device_queue(queue_family, 0) };
        // SAFETY: the physical device belongs to this live instance.
        let memory = unsafe { self.raw.get_physical_device_memory_properties(info.handle) };
        Ok(Device {
            _instance: self,
            raw,
            queue_family,
            queue,
            memory_types: memory.memory_types_as_slice().to_vec(),
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
    _instance: &'a Instance,
    raw: ash::Device,
    pub queue_family: u32,
    #[expect(dead_code, reason = "check.rs will use it")]
    pub queue: vk::Queue,
    #[expect(dead_code, reason = "check.rs will use it")]
    pub memory_types: Vec<vk::MemoryType>,
}

impl Drop for Device<'_> {
    fn drop(&mut self) {
        // SAFETY: every object created on this device borrows it, so all have
        // been destroyed, and their owners waited for idle first.
        unsafe { self.raw.destroy_device(None) };
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
            queue_family: Some(0),
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
            Some("no graphics+compute queue")
        );
    }
}
