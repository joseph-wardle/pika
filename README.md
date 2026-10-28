# Pika

Pika is a real-time raster renderer for animation production, in the style of Blender's EEVEE. It is designed around a USD native pipe using Hydra. The goal is to deliver the visual capabilities and creative flexibility of a production renderer at the speed of a game engine, without actually being a game engine.

- It runs on Vulkan through [ash](https://crates.io/crates/ash), with shaders in [Slang](https://shader-slang.org).
- The native boundaries stay thin. A small C++ Hydra render delegate copies scene data into the Rust renderer. Materials come from the upstream MaterialX shader generator rather than a custom compiler.
- An earlier feasibility study drew a full production shot offscreen within the frame budget. Interactive use inside a host has not been shown yet. Features such as indirect light, ray tracing and streaming go in only when a measurement on a real shot calls for them.
- Correctness comes before speed. Every check has a reference result and an input that must fail. A fast but wrong frame counts as a failure.

## Status

This is the bootstrap. The only program is `pika`, which checks that this machine can run the renderer. It loads Vulkan, lists every device, chooses a GPU with Vulkan 1.3 and a graphics+compute queue, and opens it. Then it draws the first offscreen frame: `shaders/check.slang`, compiled by `slangc` at run time, fills a rectangle in which each pixel writes its own position, so the read-back image must match exactly. A second draw, shifted 1 px, must fail by exactly its predicted count. When the Vulkan validation layer is found, `pika` turns on sync validation and fails on any message. It exits non-zero, with the reason, if anything fails.

There is no scene rendering, Hydra delegate or MaterialX code yet.

## Requirements

| | Needed | Notes |
| --- | --- | --- |
| Build | [rustup](https://rustup.rs) and a C linker (`cc`) | Pika follows the latest stable Rust, with rustfmt and clippy (included in rustup's default install). Run `rustup update` after each release. No Vulkan SDK is needed: libvulkan is loaded at run time. |
| Run | The Vulkan loader (`libvulkan.so.1`) and a Vulkan 1.3 GPU driver | Tested on Linux with NVIDIA driver 580.142 (RTX 4070 Ti SUPER). |
| Run | [`slangc`](https://github.com/shader-slang/slang/releases) on `PATH` | Tested with 2026.9.2. `pika` reads `shaders/` from the checkout, so run it from there. |
| Validate | The Vulkan validation layer from a [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) 1.3.268 or newer | Optional; see below. Tested with SDK 1.4.328.1. |

MaterialX and a Houdini/OpenUSD SDK will be added along with the code that needs them.

## Develop

```sh
rustup update stable
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo run --locked --release   # device check; needs a GPU
```

CI (`.github/workflows/ci.yml`) runs the first four commands on a GitHub-hosted Linux runner. That runner has no GPU, so CI never runs `cargo run`. Run the device check on a GPU machine, with validation, before merging changes to `src/gpu.rs`, `src/check.rs` or `shaders/`.

To validate, point the loader at the SDK's layer. Its manifest names the layer library without a path, so `LD_LIBRARY_PATH` is needed too:

```sh
S=/path/to/vulkan_sdk/1.4.328.1/x86_64
LD_LIBRARY_PATH=$S/lib VK_ADD_LAYER_PATH=$S/share/vulkan/explicit_layer.d cargo run --locked --release
```

The run should print `validation: on (sync)` and end with `validation: 0 messages`.
