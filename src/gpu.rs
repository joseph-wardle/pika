//! Vulkan instance, device choice and logical device.

use ash::vk;

/// Pika targets Vulkan 1.3 drivers; it uses no 1.3 features yet.
pub const REQUIRED_API: u32 = vk::API_VERSION_1_3;

/// What Pika needs to know about a physical device to choose one.
pub struct DeviceInfo {
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
    pub loader_version: u32,
    pub physical: Vec<vk::PhysicalDevice>,
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

        let app = vk::ApplicationInfo::default()
            .application_name(c"pika")
            .api_version(REQUIRED_API);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: `create_info` and what it points to outlive the call.
        let raw = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| format!("vkCreateInstance failed: {e}"))?;

        // From here `raw` must be destroyed on error, so build `Self` first.
        let mut instance = Self {
            _entry: entry,
            raw,
            loader_version,
            physical: Vec::new(),
            devices: Vec::new(),
        };
        // SAFETY: `instance.raw` is a live instance.
        instance.physical = unsafe { instance.raw.enumerate_physical_devices() }
            .map_err(|e| format!("vkEnumeratePhysicalDevices failed: {e}"))?;
        instance.devices = instance
            .physical
            .iter()
            .map(|&pd| instance.describe(pd))
            .collect::<Result<_, _>>()?;
        Ok(instance)
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
            name,
            kind: props.device_type,
            api_version: props.api_version,
            vram_bytes,
            ray_query,
            queue_family,
        })
    }

    /// Opens a logical device with one graphics+compute queue on `devices[index]`.
    pub fn open(&self, index: usize) -> Result<Device<'_>, String> {
        let info = &self.devices[index];
        let queue_family = info
            .queue_family
            .ok_or_else(|| format!("{} has no graphics+compute queue", info.name))?;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let create_info = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
        // SAFETY: the physical device belongs to this live instance and
        // `create_info` outlives the call.
        let raw = unsafe {
            self.raw
                .create_device(self.physical[index], &create_info, None)
        }
        .map_err(|e| format!("vkCreateDevice on {} failed: {e}", info.name))?;
        Ok(Device {
            _instance: self,
            raw,
            queue_family,
        })
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: every `Device` borrows this instance, so none outlives it.
        unsafe { self.raw.destroy_instance(None) };
    }
}

/// A logical device. It borrows its `Instance` so it cannot outlive it.
pub struct Device<'a> {
    _instance: &'a Instance,
    raw: ash::Device,
    pub queue_family: u32,
}

impl Drop for Device<'_> {
    fn drop(&mut self) {
        // SAFETY: Pika has submitted no work, so nothing is in flight.
        unsafe { self.raw.destroy_device(None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(kind: vk::PhysicalDeviceType, api_version: u32, vram_gib: u64) -> DeviceInfo {
        DeviceInfo {
            name: format!("{kind:?}"),
            kind,
            api_version,
            vram_bytes: vram_gib << 30,
            ray_query: false,
            queue_family: Some(0),
        }
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
