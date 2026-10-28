//! The Pika renderer. The Hydra delegate links it as a static library and
//! calls it through `abi`; the `pika` binary uses `gpu` and `target` directly.

mod abi;
pub mod gpu;
mod renderer;
pub mod target;
