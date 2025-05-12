//! Quantum circuit simulators for the quantrs framework.
//!
//! This crate provides various simulation backends for quantum circuits,
//! including state vector simulation on CPU and optionally GPU.

pub mod statevector;
pub mod tensor;
pub mod utils;

#[cfg(feature = "gpu")]
pub mod gpu;

/// Re-exports of commonly used types and traits
pub mod prelude {
    pub use crate::statevector::*;
    pub use crate::tensor::*;
    pub use crate::utils::*;
    
    #[cfg(feature = "gpu")]
    pub use crate::gpu::*;
}