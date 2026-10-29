//! The Pika renderer. The Hydra delegate links it as a static library and
//! calls it through `abi`; the `pika` binary uses `gpu`, `shader` and `target`
//! directly.

mod abi;
mod geometry;
pub mod gpu;
mod raster;
mod renderer;
pub mod shader;
mod smooth;
mod store;
mod subdiv;
pub mod target;
mod transform;
