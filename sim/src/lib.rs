//! Quantum circuit simulators for the quantrs framework.
//!
//! This crate provides various simulation backends for quantum circuits,
//! including state vector simulation on CPU and optionally GPU.
//!
//! It includes both standard and optimized implementations, with the optimized
//! versions leveraging SIMD, memory-efficient algorithms, and parallel processing
//! to enable simulation of larger qubit counts (30+).

pub mod statevector;
pub mod tensor;
pub mod utils;
// pub mod optimized;  // Temporarily disabled due to implementation issues
// pub mod optimized_simulator;  // Temporarily disabled due to implementation issues
pub mod optimized_simple;
pub mod optimized_simulator_simple;
pub mod optimized_chunked;
pub mod optimized_simulator_chunked;
pub mod optimized_simd;
pub mod optimized_simulator;
pub mod benchmark;
pub mod tests;
pub mod tests_simple;
pub mod tests_optimized;

#[cfg(feature = "gpu")]
pub mod gpu;

/// Re-exports of commonly used types and traits
pub mod prelude {
    pub use crate::statevector::*;
    pub use crate::tensor::*;
    pub use crate::utils::*;
    // pub use crate::optimized::*;  // Temporarily disabled
    // pub use crate::optimized_simulator::*;  // Temporarily disabled
    pub use crate::optimized_simple::*;
    pub use crate::optimized_simulator_simple::*;
    pub use crate::optimized_chunked::*;
    pub use crate::optimized_simulator_chunked::*;
    pub use crate::optimized_simd::*;
    pub use crate::optimized_simulator::*;
    pub use crate::benchmark::*;

    #[cfg(feature = "gpu")]
    pub use crate::gpu::*;
}