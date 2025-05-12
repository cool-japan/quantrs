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
pub mod tensor_network;
pub mod utils;
// pub mod optimized;  // Temporarily disabled due to implementation issues
// pub mod optimized_simulator;  // Temporarily disabled due to implementation issues
pub mod benchmark;
pub mod optimized_chunked;
pub mod optimized_simd;
pub mod optimized_simple;
pub mod optimized_simulator;
pub mod optimized_simulator_chunked;
pub mod optimized_simulator_simple;
#[cfg(test)]
pub mod tests;
#[cfg(test)]
pub mod tests_optimized;
#[cfg(test)]
pub mod tests_simple;
#[cfg(test)]
pub mod tests_tensor_network;

/// Noise models for quantum simulation
pub mod noise;

/// Quantum error correction codes and utilities (placeholder for future implementation)
#[allow(clippy::module_inception)]
pub mod error_correction {
    //! Quantum error correction codes and utilities
    //!
    //! This module will provide error correction codes like the Steane code,
    //! Surface code, and related utilities. For now, it's a placeholder.

    /// A placeholder for future error correction code implementations
    #[derive(Debug, Clone)]
    pub struct ErrorCorrection;
}

#[cfg(feature = "gpu")]
pub mod gpu;

/// Re-exports of commonly used types and traits
pub mod prelude {
    pub use crate::error_correction::*;
    pub use crate::noise::*;
    pub use crate::statevector::*;
    pub use crate::tensor::*;
    pub use crate::tensor_network::*;
    pub use crate::utils::*;
    // pub use crate::optimized::*;  // Temporarily disabled
    // pub use crate::optimized_simulator::*;  // Temporarily disabled
    pub use crate::benchmark::*;
    pub use crate::optimized_chunked::*;
    pub use crate::optimized_simd::*;
    pub use crate::optimized_simple::*;
    pub use crate::optimized_simulator::*;
    pub use crate::optimized_simulator_chunked::*;
    pub use crate::optimized_simulator_simple::*;

    #[cfg(feature = "gpu")]
    pub use crate::gpu::*;
}
