//! `pika`: checks that this machine can run the renderer.
//!
//! Loads Vulkan, lists every device, chooses one and opens it, then draws the
//! first-frame check and its shifted negative control.
//! Exits 0 only if the check matched exactly, the control failed by exactly
//! its predicted count, and the validation layer, when found, reported nothing.

mod check;
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
    match instance.validation_messages() {
        Some(_) => println!("validation: on (sync)"),
        None => println!("validation: not found"),
    }
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
    let shader = check::compile()?;
    println!(
        "slangc {}: check.slang is {} SPIR-V words",
        shader.slangc_version,
        shader.spirv.len()
    );

    let mut offscreen = check::Offscreen::new(&device, &shader)?;
    let reference = check::expected(check::RECT);
    let drawn = check::mismatches(&reference, &offscreen.draw(check::RECT)?);
    let shifted = check::mismatches(&reference, &offscreen.draw(check::SHIFTED)?);
    println!("first frame: {drawn} mismatched pixels (must be 0)");
    println!(
        "shifted control: {shifted} mismatched pixels (must be {})",
        check::SHIFTED_MISMATCHES
    );

    // Read the count after the device is destroyed: vkDestroyDevice reports leaks.
    drop(offscreen);
    drop(device);
    match instance.validation_messages() {
        Some(0) => println!("validation: 0 messages"),
        Some(n) => return Err(format!("validation reported {n} messages (printed above)")),
        None => {}
    }
    if drawn != 0 || shifted != check::SHIFTED_MISMATCHES {
        return Err("the first-frame check failed (counts above)".into());
    }
    Ok(())
}
