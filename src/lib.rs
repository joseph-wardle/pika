//! The Pika renderer. The Hydra delegate links it as a static library and
//! calls it through `abi`; the `pika` binary uses `gpu`, `shader`, `commands`
//! and `target` directly.

mod abi;
pub mod commands;
mod cull;
mod geometry;
pub mod gpu;
mod raster;
mod renderer;
pub mod shader;
mod smooth;
mod stats;
mod store;
mod subdiv;
pub mod target;
mod transform;
mod view;
