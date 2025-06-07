//! Quantum circuit simulators for the QuantRS2 framework.
//!
//! This crate provides various simulation backends for quantum circuits,
//! including state vector simulation on CPU and optionally GPU.
//!
//! It includes both standard and optimized implementations, with the optimized
//! versions leveraging SIMD, memory-efficient algorithms, and parallel processing
//! to enable simulation of larger qubit counts (30+).

pub mod dynamic;
pub mod enhanced_statevector;
pub mod linalg_ops;
pub mod simulator;
pub mod specialized_gates;
pub mod specialized_simulator;
pub mod stabilizer;
pub mod statevector;
pub mod tensor;

#[cfg(feature = "advanced_math")]
pub mod tensor_network;
pub mod utils;
// pub mod optimized;  // Temporarily disabled due to implementation issues
// pub mod optimized_simulator;  // Temporarily disabled due to implementation issues
pub mod benchmark;
pub mod clifford_sparse;
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

#[allow(clippy::module_inception)]
pub mod error_correction {
    //! Quantum error correction codes and utilities
    //!
    //! This module will provide error correction codes like the Steane code,
    //! Surface code, and related utilities. For now, it's a placeholder.
}

/// Prelude module that re-exports common types and traits
pub mod prelude {
    pub use crate::clifford_sparse::{CliffordGate, SparseCliffordSimulator};
    pub use crate::dynamic::*;
    pub use crate::enhanced_statevector::EnhancedStateVectorSimulator;
    #[allow(unused_imports)]
    pub use crate::error_correction::*;
    pub use crate::noise::*;
    pub use crate::noise::{NoiseChannel, NoiseModel};
    pub use crate::noise_advanced::*;
    pub use crate::noise_advanced::{AdvancedNoiseModel, RealisticNoiseModelBuilder};
    #[allow(unused_imports)]
    pub use crate::simulator::*;
    pub use crate::simulator::{Simulator, SimulatorResult};
    pub use crate::stabilizer::{is_clifford_circuit, StabilizerGate, StabilizerSimulator};
    pub use crate::statevector::StateVectorSimulator;
    pub use crate::specialized_gates::{
        SpecializedGate, specialize_gate,
        HadamardSpecialized, PauliXSpecialized, PauliYSpecialized, PauliZSpecialized,
        PhaseSpecialized, SGateSpecialized, TGateSpecialized,
        RXSpecialized, RYSpecialized, RZSpecialized,
        CNOTSpecialized, CZSpecialized, SWAPSpecialized, CPhaseSpecialized,
        ToffoliSpecialized, FredkinSpecialized,
    };
    pub use crate::specialized_simulator::{
        SpecializedStateVectorSimulator, SpecializedSimulatorConfig,
        SpecializationStats, benchmark_specialization,
    };

    #[cfg(feature = "gpu")]
    pub use crate::gpu_linalg::{benchmark_gpu_linalg, GpuLinearAlgebra};
    #[allow(unused_imports)]
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

#[cfg(feature = "gpu")]
pub mod gpu_linalg;

#[cfg(feature = "advanced_math")]
pub use crate::tensor_network::*;

// Temporarily disabled features
// pub use crate::optimized::*;
// pub use crate::optimized_simulator::*;
