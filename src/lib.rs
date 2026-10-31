//! The Pika renderer. The Hydra delegate links it as a static library and
//! calls it through `abi`; the `pika` binary uses `gpu`, `shader`, `commands`
//! and `target` directly.

mod abi;
pub mod bake;
mod cache;
pub mod commands;
mod cull;
mod geometry;
pub mod gpu;
mod json;
pub mod material;
mod raster;
mod renderer;
mod shade;
pub mod shader;
mod smooth;
mod stats;
mod store;
mod subdiv;
mod surface;
pub mod target;
pub mod texture;
mod transform;
mod view;
