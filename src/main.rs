//! `pika`: checks that this machine can run the renderer.
//!
//! Loads Vulkan, lists every device, chooses one and opens it.
//! Exits 0 only if a suitable device opened.

mod gpu;

use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let instance = gpu::Instance::new()?;
    println!(
        "Vulkan loader {}",
        gpu::version_string(instance.loader_version)
    );
    for (i, d) in instance.devices.iter().enumerate() {
        let status = d.unsuitable_reason().unwrap_or("ok");
        println!(
            "  [{i}] {}  {:?}  Vulkan {}  {:.1} GiB  ray query: {}  {status}",
            d.name,
            d.kind,
            gpu::version_string(d.api_version),
            d.vram_bytes as f64 / (1u64 << 30) as f64,
            if d.ray_query { "yes" } else { "no" },
        );
    }

    let index = gpu::choose(&instance.devices).ok_or(
        "no suitable Vulkan device: Pika needs a GPU with Vulkan 1.3 and a graphics+compute \
         queue. Check the driver with `vulkaninfo --summary`.",
    )?;
    let device = instance.open(index)?;
    println!(
        "opened [{index}] {}, queue family {}",
        instance.devices[index].name, device.queue_family
    );
    Ok(())
}
