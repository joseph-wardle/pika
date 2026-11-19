# Pika

Pika is a real-time raster renderer for animation production, in the style of Blender's EEVEE. It is built for a USD-native pipeline: it loads into any Hydra host, and aims to play cached shots back at their final look, 1080p at 24 fps on a 16 GB GPU. The goal is the visual range and creative flexibility of a production renderer at the speed of a game engine, without being a game engine.

It is built in Rust. It runs on Vulkan through [ash](https://crates.io/crates/ash), with shaders in [Slang](https://shader-slang.org). A small C++ Hydra render delegate copies scene data in, and materials come from the upstream MaterialX shader generator rather than a custom compiler, so OpenPBR, Standard Surface, UsdPreviewSurface and any other surface the host's MaterialX libraries define draw as those libraries say.

Today Pika draws meshes, synced from the host with their subdivision, instancing, holes, normals, visibility and purpose, and renders color, depth and normal AOVs for the host's camera. Each mesh is shaded by its material's MaterialX surface under the scene's distant, sphere, rect, disk and cylinder lights, those linked to it, the host's camera light among them, its GeomSubsets by theirs; textures are baked once into a cache, UDIM sets as one array, those the camera needs least are lowered until they fit the card's memory, and the render setting `pika:textureCap` shrinks them all. A mesh with no material, or one still being built, draws as grey clay; one Pika cannot build draws pink, and a missing texture file draws its node's fallback, each with a warning. Every light casts shadows by its shadow links, and a light with a size casts soft ones: while the view is still, each frame adds a sample until 64 have been drawn, as EEVEE renders take, so a still image converges and playback shows one sample a frame. Penumbrae follow EEVEE's method, and read somewhat narrower than a ray tracer's. Shadows are cached from frame to frame and redrawn only where a light or a moving mesh changes them, within a pool sized by the render setting `pika:shadowPool` (256 MB by default, and as much again for moving meshes); a view that needs more draws its farthest shadows coarser, with a warning. Dome lights come next.

## Requirements

| | Needed | Notes |
| --- | --- | --- |
| Build | Rust and a C linker | No Vulkan SDK is needed: libvulkan is loaded at run time. |
| Run | The Vulkan loader and a Vulkan 1.3 GPU driver | Tested with NVIDIA driver 595.71 (RTX 4070 Ti SUPER). |
| Run | [`slangc`](https://github.com/shader-slang/slang/releases) on `PATH` | Tested with 2026.21. Pika reads `shaders/` and the bake tools from the checkout, so run it from there. Generated shaders and baked textures are kept in `$XDG_CACHE_HOME/pika` (`PIKA_CACHE` overrides it); ALab's fill 145 GB. |
| Validate | The Vulkan validation layer | Optional; the shell turns it on. Pika enables it when `VK_ADD_LAYER_PATH` names it. |
| Delegate | OpenUSD with its Python, OpenSubdiv, `g++` and `make` | Tested with OpenUSD 26.11 and g++ 16.2. |
| Bake | MaterialX 1.39.5 or newer (the first with a Slang generator), OpenImageIO, `g++`, `make` and [`ispc`](https://github.com/ispc/ispc/releases) | Tested with MaterialX 1.39.5, OpenImageIO 3.1 and ispc 1.32. `MATERIALX` names the directory holding MaterialX's `libraries/`. Two tools the renderer runs, so build them before rendering: `generate` turns material documents into shaders, and `texture` bakes textures. They compile in the paths of MaterialX and of `bake/`, so rebuild them after moving either. Textures are converted through the OCIO config named in `OCIO`, else OpenImageIO's default. |
| Check | A display, or `Xvfb`; `curl` and `unzip` to fetch ALab | `hydra/check.sh` starts its own Xvfb. |

Houdini is a second host: with `HFS` naming a Houdini (22.0 for the bake tools, as 21.0 has no Slang generator), the Makefiles and the checks build against its USD and MaterialX and render through husk, which also checks depth, normals and the render settings. That path was last tested with Houdini 21.0.596 and 22.0.466. `BAKE_HFS` names the bake tools' Houdini when `HFS` is a 21.0. On NixOS, Houdini runs inside an FHS environment, so a check is `HFS=~/sidefx/hfs22.0.466 nix develop --command houdini-fhs hydra/check.sh`.

## Build and check

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo run --locked --release   # device check; needs a GPU
make -C hydra                  # the delegate, into target/hydra
make -C bake                   # the bake tools, into target/bake
bake/check.sh
hydra/check.sh                 # needs a GPU
```

CI runs the first three cargo commands, on a runner without a GPU. To render with the delegate, point USD at it:

```sh
export PXR_PLUGINPATH_NAME=$PWD/target/hydra
usdview --renderer Pika hydra/scenes/cube.usda
```

`corpus/fetch.sh` downloads ALab 2.3.0 (108 GB) into `corpus/`, which git ignores, and `hydra/gate.sh` times its playback. `PIKA_STATS` with `hydra/stats.py` measure any run. Each script's header says what it does and how to run it. `NOTICE` lists the third-party code Pika ships and its licences.
