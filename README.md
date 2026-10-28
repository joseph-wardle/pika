# Pika

Pika is a real-time raster renderer for animation production, in the style of Blender's EEVEE. It is designed around a USD native pipe using Hydra. The goal is final-look playback of cached shots at 1920×1080 and 24 fps on a 16 GB GPU.

- It runs on Vulkan through [ash](https://crates.io/crates/ash), with shaders in [Slang](https://shader-slang.org).
- The native boundaries stay thin. A small C++ Hydra render delegate copies scene data into the Rust renderer. Materials come from the upstream MaterialX shader generator rather than a custom compiler.
- An earlier feasibility study drew a full production shot offscreen within the frame budget. Interactive use inside a host has not been shown yet. Features such as indirect light, ray tracing and streaming go in only when a measurement on a real shot calls for them.
- Correctness comes before speed. Every check has a reference result and an input that must fail. A fast but wrong frame counts as a failure.

## Status

This is the bootstrap. The only program is `pika`, which checks that this machine can run the renderer. It loads Vulkan, lists every device, chooses a GPU with Vulkan 1.3 and a graphics+compute queue, and opens it. It exits non-zero, with the reason, if it can't.

There is no rendering, Hydra delegate or MaterialX code yet.

## Requirements

| | Needed | Notes |
| --- | --- | --- |
| Build | [rustup](https://rustup.rs) and a C linker (`cc`) | Pika follows the latest stable Rust, with rustfmt and clippy (included in rustup's default install). Run `rustup update` after each release. No Vulkan SDK is needed: libvulkan is loaded at run time. |
| Run | The Vulkan loader (`libvulkan.so.1`) and a Vulkan 1.3 GPU driver | Tested on Linux with NVIDIA driver 580.142 (RTX 4070 Ti SUPER). |

The Slang compiler, MaterialX, the Vulkan validation layer and a Houdini/OpenUSD SDK will be added along with the code that needs them.

## Develop

```sh
rustup update stable
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo run --locked --release   # device check; needs a GPU
```

CI (`.github/workflows/ci.yml`) runs the first four commands on a GitHub-hosted Linux runner. That runner has no GPU, so CI never runs `cargo run`. Run the device check on a GPU machine before merging changes to `src/gpu.rs`.
