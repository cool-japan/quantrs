//! Quantum circuit simulators for the QuantRS2 framework.
//!
//! This crate provides various simulation backends for quantum circuits,
//! including state vector simulation on CPU and optionally GPU.
//!
//! It includes both standard and optimized implementations, with the optimized
//! versions leveraging SIMD, memory-efficient algorithms, and parallel processing
//! to enable simulation of larger qubit counts (30+).

pub mod dynamic;
pub mod simulator;
pub mod statevector;
pub mod tensor;

#[cfg(feature = "advanced_math")]
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

/// Advanced noise models for realistic device simulation
pub mod noise_advanced;

/// Quantum error correction codes and utilities (placeholder for future implementation)
#[allow(clippy::module_inception)]
pub mod error_correction {
    //! Quantum error correction codes and utilities
    //!
    //! This module will provide error correction codes like the Steane code,
    //! Surface code, and related utilities. For now, it's a placeholder.
}

/// Prelude module that re-exports common types and traits
pub mod prelude {
    //! Common types and traits for quantum simulation
    pub use crate::dynamic::*;
    pub use crate::error_correction::*;
    pub use crate::noise::*;
    pub use crate::noise::{NoiseChannel, NoiseModel};
    pub use crate::noise_advanced::*;
    pub use crate::noise_advanced::{AdvancedNoiseModel, RealisticNoiseModelBuilder};
    pub use crate::simulator::*;
    pub use crate::simulator::{Simulator, SimulatorResult};
    pub use crate::statevector::StateVectorSimulator;
    pub use crate::statevector::*;
    pub use crate::tensor::*;
    pub use crate::utils::*;
    pub use num_complex::Complex64;
}

/// A placeholder for future error correction code implementations
#[derive(Debug, Clone)]
pub struct ErrorCorrection;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "advanced_math")]
pub use crate::tensor_network::*;

// Temporarily disabled features
// pub use crate::optimized::*;
// pub use crate::optimized_simulator::*;
