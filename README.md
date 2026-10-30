# Pika

Pika is a real-time raster renderer for animation production, in the style of Blender's EEVEE. It is built for a USD-native pipeline: it loads into any Hydra host, from usdview to husk, and aims to play cached shots back at their final look, 1080p at 24 fps on a 16 GB GPU. The goal is the visual range and creative flexibility of a production renderer at the speed of a game engine, without being a game engine.

It is built in Rust. It runs on Vulkan through [ash](https://crates.io/crates/ash), with shaders in [Slang](https://shader-slang.org). A small C++ Hydra render delegate copies scene data in, and materials will come from the upstream MaterialX shader generator rather than a custom compiler.

Today Pika draws meshes. It syncs them from the host with their subdivision, instancing, holes, normals, visibility and purpose, and renders color, depth and normal AOVs for the host's camera, in grey lit by the view direction. Materials, lights and shadows come next.

## Requirements

| | Needed | Notes |
| --- | --- | --- |
| Build | [rustup](https://rustup.rs) and a C linker (`cc`) | Pika follows the latest stable Rust, with rustfmt and clippy (included in rustup's default install). Run `rustup update` after each release. No Vulkan SDK is needed: libvulkan is loaded at run time. |
| Run | The Vulkan loader (`libvulkan.so.1`) and a Vulkan 1.3 GPU driver | Tested on Linux with NVIDIA driver 580.142 (RTX 4070 Ti SUPER). |
| Run | [`slangc`](https://github.com/shader-slang/slang/releases) on `PATH` | Tested with 2026.9.2. Pika reads `shaders/` from the checkout, so run it from there. |
| Validate | The Vulkan validation layer from a [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) 1.3.268 or newer | Optional. Tested with SDK 1.4.328.1. |
| Delegate | Houdini 21.0 (its HDK headers and USD libraries), `g++` and `make` | Tested with Houdini 21.0.596 (USD 0.25.5) and g++ 11.5. |

## Build and check

```sh
rustup update stable
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo run --locked --release   # device check; needs a GPU
```

CI runs all but the device check, on a runner without a GPU. Run the device check, and the delegate's check below, on a GPU machine with the validation layer on. The layer's manifest names its library without a path, so the loader needs both of these:

```sh
S=/path/to/vulkan_sdk/1.4.328.1/x86_64
export LD_LIBRARY_PATH=$S/lib VK_ADD_LAYER_PATH=$S/share/vulkan/explicit_layer.d
```

`make -C hydra` builds the delegate into `target/hydra`. To render with it, source Houdini's `houdini_setup_bash` and point USD at it:

```sh
export PXR_PLUGINPATH_NAME=$PWD/target/hydra
husk -R Pika -c /world/camera -o out.exr hydra/scenes/cube.usda
```

`hydra/check.sh` is the delegate's check, `hydra/gate.sh` the timing run on ALab, and `PIKA_STATS` with `hydra/stats.py` measure any run. Each script's header says what it does and how to run it.
