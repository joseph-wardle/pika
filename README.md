# Pika

Pika is a real-time raster renderer for animation production, in the style of Blender's EEVEE. It is designed around a USD native pipe using Hydra. The goal is to deliver the visual capabilities and creative flexibility of a production renderer at the speed of a game engine, without actually being a game engine.

- It runs on Vulkan through [ash](https://crates.io/crates/ash), with shaders in [Slang](https://shader-slang.org).
- The native boundaries stay thin. A small C++ Hydra render delegate copies scene data into the Rust renderer. Materials come from the upstream MaterialX shader generator rather than a custom compiler.
- An earlier feasibility study drew a full production shot offscreen within the frame budget. Interactive use inside a host has not been shown yet. Features such as indirect light, ray tracing and streaming go in only when a measurement on a real shot calls for them.
- Correctness comes before speed. Every check has a reference result and an input that must fail. A fast but wrong frame counts as a failure.

## Status

There are two programs:

- **`pika`** checks that this machine can run the renderer. It loads Vulkan, lists every device, chooses a GPU with Vulkan 1.3 and a graphics+compute queue with timestamps, and opens it. Then it draws the first offscreen frame: `shaders/check.slang`, compiled by `slangc` at run time, fills a rectangle in which each pixel writes its own position, so the read-back image must match exactly. A second draw, shifted 1 px, must fail by exactly its predicted count. When the Vulkan validation layer is found, `pika` turns on sync validation and fails on any message. It exits non-zero, with the reason, if anything fails.
- **The Hydra delegate** (`hydra/`) loads into any Hydra host and syncs meshes into Pika's store: points, topology with its subdivision scheme, holes and GeomSubsets, the material bound to each mesh and subset (for the full, final-look purpose), normals, world transform, visibility and render tag. It also stores instancers, fetches basis curves only to measure their bytes, and counts lights, materials and other prim types. It draws meshes into the color, depth and normal AOVs, with the host's camera and framing. Instanced meshes draw once per instance, through native, point and nested instancers, and only meshes of the render pass's tags are drawn. Every mesh draws as its cage, without its hole faces; subdivision is not handled yet. Authored normals are used on meshes whose scheme is `none`; every other mesh gets smooth normals, computed on the GPU from its points as Hydra computes them. Color is grey, shaded by the normal against the view direction. The CPU reads the frame back into the host's render buffers.

There is no MaterialX code yet.

## Requirements

| | Needed | Notes |
| --- | --- | --- |
| Build | [rustup](https://rustup.rs) and a C linker (`cc`) | Pika follows the latest stable Rust, with rustfmt and clippy (included in rustup's default install). Run `rustup update` after each release. No Vulkan SDK is needed: libvulkan is loaded at run time. |
| Run | The Vulkan loader (`libvulkan.so.1`) and a Vulkan 1.3 GPU driver | Tested on Linux with NVIDIA driver 580.142 (RTX 4070 Ti SUPER). |
| Run | [`slangc`](https://github.com/shader-slang/slang/releases) on `PATH` | Tested with 2026.9.2. `pika` reads `shaders/` from the checkout, so run it from there. |
| Validate | The Vulkan validation layer from a [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) 1.3.268 or newer | Optional; see below. Tested with SDK 1.4.328.1. |
| Delegate | Houdini 21.0 (its HDK headers and USD libraries), `g++` and `make` | Tested with Houdini 21.0.596 (USD 0.25.5) and g++ 11.5. |

MaterialX will be added along with the code that needs it.

## Develop

```sh
rustup update stable
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo run --locked --release   # device check; needs a GPU
```

CI (`.github/workflows/ci.yml`) runs the first four commands on a GitHub-hosted Linux runner. That runner has no GPU, so CI never runs `cargo run`. Run the device check on a GPU machine, with validation, before merging changes to `src/gpu.rs`, `src/target.rs`, `src/check.rs` or `shaders/`.

To validate, point the loader at the SDK's layer. Its manifest names the layer library without a path, so `LD_LIBRARY_PATH` is needed too:

```sh
S=/path/to/vulkan_sdk/1.4.328.1/x86_64
LD_LIBRARY_PATH=$S/lib VK_ADD_LAYER_PATH=$S/share/vulkan/explicit_layer.d cargo run --locked --release
```

The run should print `validation: on (sync)` and end with `validation: 0 messages`.

## Hydra delegate

`make -C hydra` builds the Rust library and links it into `target/hydra/pika.so`, next to its `plugInfo.json`. `HFS` defaults to `/opt/hfs21.0`. To use the delegate, point USD at it from a shell where `houdini_setup_bash` has been sourced:

```sh
export PXR_PLUGINPATH_NAME=$PWD/target/hydra
husk -R Pika -c /world/camera -o out.exr hydra/scenes/cube.usda
```

The delegate's errors go to stderr as `pika: error: ...`, as well as to Hydra, because husk does not print Hydra's diagnostics. When the host exits, Pika prints each Hydra sync's time and the bytes it pulled, with how many of them were unchanged, the bytes of basis curves it fetched and dropped, and the median and maximum of each frame stage.

To measure a sync, set `PIKA_STATS` to a file. Each Hydra sync appends one line of `key value` pairs to it: the sync's time, the time since the last sync began, the bytes pulled by prim type with how many were unchanged, and a census of the store (meshes drawn, instances, triangles, geometry bytes, materials bound, and counted prims by type). Each rendered frame appends a `frame` line: the CPU's recording, its wait for the GPU and its copy into the host's render buffers; the GPU's timestamps around the uploads with the smooth normal pass, the draw and the readback; and what the frame wrote: bytes uploaded and read back, meshes whose points or transform moved, and whether the layout and the draw records were rewritten. A moved transform rewrites only that mesh's instances.

To see what Pika received, set `PIKA_STORE_DUMP` to a file. At the end of every Hydra sync, the whole store is appended to it as text, in Hydra's representation: world transforms, flattened primvars, untriangulated faces. It changes nothing about rendering.

`hydra/check.sh` is the delegate's check. Run it on a GPU host before merging changes to `hydra/` or `src/`. It builds the delegate and runs `hydra/image.py`, which renders the scenes in `hydra/images/` for two frames in husk (full float color, depth and normal) and in usdrecord (half float color, on a private Xvnc display that the script starts). Each image is compared with a ray cast from USD, one ray per pixel centre, with instances placed by USD's instance proxies and point instancer, and smooth normals computed as Hydra computes them: coverage away from face edges, and color, depth and normal values within a tolerance. Controls must fail: the camera's focal length 0.5 % long, rays half a pixel right, the aperture cropped instead of expanded, depth not converted to Hydra's, normals not flipped toward the viewer, flat normals, the point instancer's own transform dropped, native instances collapsed into their prototype, point instancer orientations read in the wrong component order, authored normals used on subdivided meshes, hole faces drawn, smooth normals averaged from unit face normals, left-handed faces' corners not swapped, frame 2's smooth normals left at frame 1's, meshes of every purpose drawn, and frame 2 against frame 1's reference. husk draws the render purpose and usdrecord the proxy purpose, so each host's reference has its own purposes. Validation must report nothing from Pika.

husk renders a `dataWindowNDC` in a way the render pass cannot describe, so Pika refuses it with an error; usdview's framing works.

Then it checks the fixtures in `hydra/fixtures/`, one small scene per rule of what Hydra hands a delegate: composed transforms, flattened indexed primvars, normal interpolation, orientation, subdivision scheme and holes, GeomSubsets, the material binding purpose, inherited visibility, purpose, removal, time samples, native, point and nested instancing, basis curves' bytes and the counted prim types. Each scene is synced for two frames. Its `.py` script, if any, edits husk's stage after frame 1. Each frame's store dump must match the hand-written `.dump`, and frame 2 must differ from frame 1's expectation, so a store left stale by the edit fails. If Hydra disagrees with a fixture, find out why before changing the expectation.

```sh
VULKAN_SDK=/path/to/vulkan_sdk/1.4.328.1/x86_64 hydra/check.sh
```

`hydra/oracle.py` checks the same rules on a real stage. It samples a stage's non-instanced meshes, half static and half animated, works out from USD what Hydra should hand over at one frame, syncs only the sample in husk and compares the store value by value. Two controls must fail: each mesh against the next mesh's expectation, and each animated mesh against the next frame's. Run it on ALab, from a shell set up as `check.sh` sets one up:

```sh
hython hydra/oracle.py /path/to/ALab/entry.usda 1004 /tmp/oracle
```

`hydra/census.py` measures a shot through the delegate, uncached, and gives a verdict on each threshold that orders the renderer work: sync time with clip files cached and cold, unchanged bytes, mesh instances and triangles. It syncs the frame range in husk three times: twice without the stage's basis curves (fur is outside the gate), first with the stage's value clip files evicted from this machine's page cache and then cached, and once with the curves. It reports warm p50/p95/max sync times, bytes per sync by prim type, the census and husk's peak memory. Run it on a quiet machine, because other load inflates the sync times. On ALab the three runs take about 20 s, with husk under 2 GiB resident:

```sh
hython hydra/census.py /path/to/ALab/entry.usda /root/camera01/GEO/renderCam_hrc/renderCam_buffer/renderCam_srt/renderCam 1004 1057 /tmp/census
```

A Rust panic inside the delegate is caught at the C ABI. The delegate reports the panic and renders nothing until it is recreated. Some failures cannot be contained and take down the host with them: a panic during unwinding, running out of memory, a stack overflow, and a GPU driver crash.
