//! Organized prelude modules for simulation functionality

pub mod essentials {
    //! Essential types for basic quantum simulation
    //!
    //! Core types needed for quantum circuit simulation

    // Basic simulators
    pub use crate::api::simulation::{Result, SimulatorError};
    pub use crate::api::simulation::{Simulator, SimulatorResult, StateVectorSimulator};

    // Core optimization
    pub use crate::api::optimization::{optimize_circuit, CircuitOptimizer};

    // Basic noise models
    pub use crate::api::noise::{NoiseChannel, NoiseModel};
}

pub mod simulation {
    //! Complete toolkit for simulation development
    //!
    //! Everything needed for advanced quantum simulation

    pub use super::essentials::*;

    // All simulation backends
    pub use crate::api::simulation::*;
    pub use crate::api::specialized::*;

    // Performance optimization
    pub use crate::api::optimization::*;
    pub use crate::api::profiling::*;

    // Memory management
    pub use crate::api::memory::*;
}

pub mod gpu {
    //! GPU and high-performance computing
    //!
    //! GPU and accelerated simulation backends

    pub use super::essentials::*;

    // GPU backends
    pub use crate::api::gpu::*;

    // SIMD operations
    pub use crate::api::simd::*;

    // High-performance optimizations
    pub use crate::api::optimization::*;
}

pub mod distributed {
    //! Large-scale and distributed simulation
    //!
    //! Tools for large-scale distributed quantum simulation

    pub use super::simulation::*;

    // Distributed simulation
    pub use crate::api::distributed::*;

    // Tensor networks
    pub use crate::api::tensor_networks::*;
}

pub mod algorithms {
    //! Algorithm development and research
    //!
    //! Advanced quantum algorithms and applications

    pub use super::simulation::*;

    // Quantum algorithms
    pub use crate::api::algorithms::*;

    // Machine learning
    pub use crate::api::quantum_ml::*;

    // Specialized methods
    pub use crate::api::specialized::*;
}

pub mod noise_modeling {
    //! Noise modeling and error correction
    //!
    //! Comprehensive noise modeling and error correction

    pub use super::essentials::*;

    // Noise models
    pub use crate::api::noise::*;

    // Error correction
    pub use crate::api::error_correction::*;
}

pub mod dev_tools {
    //! Developer tools and debugging
    //!
    //! Tools for simulation debugging and development

    pub use super::essentials::*;

    // Debugging tools
    pub use crate::api::dev_tools::*;

    // Profiling and analysis
    pub use crate::api::profiling::*;

    // SciRS2 enhanced tools
    pub use crate::api::scirs2::*;
}

#[deprecated(
    since = "1.0.0",
    note = "Use organized modules like `essentials`, `simulation`, etc."
)]
pub mod legacy {
    //! Legacy compatibility - provides the old flat API
    //!
    //! Backward compatibility exports

    pub use crate::api::algorithms::*;
    pub use crate::api::dev_tools::*;
    pub use crate::api::distributed::*;
    pub use crate::api::dynamic::*;
    pub use crate::api::error_correction::*;
    pub use crate::api::gates::*;
    pub use crate::api::gpu::*;
    pub use crate::api::measurement::*;
    pub use crate::api::memory::*;
    pub use crate::api::noise::*;
    pub use crate::api::optimization::*;
    pub use crate::api::precision::*;
    pub use crate::api::profiling::*;
    pub use crate::api::quantum_ml::*;
    pub use crate::api::scirs2::*;
    pub use crate::api::simd::*;
    pub use crate::api::simulation::*;
    pub use crate::api::specialized::*;
    pub use crate::api::tensor_networks::*;
    pub use crate::api::utils::*;
}

pub mod full {
    //! Full API re-export (non-deprecated flat access)
    //!
    //! Complete API access with new naming conventions

    pub use crate::api::algorithms::*;
    pub use crate::api::dev_tools::*;
    pub use crate::api::distributed::*;
    pub use crate::api::dynamic::*;
    pub use crate::api::error_correction::*;
    pub use crate::api::gates::*;
    pub use crate::api::gpu::*;
    pub use crate::api::measurement::*;
    pub use crate::api::memory::*;
    pub use crate::api::noise::*;
    pub use crate::api::optimization::*;
    pub use crate::api::precision::*;
    pub use crate::api::profiling::*;
    pub use crate::api::quantum_ml::*;
    pub use crate::api::scirs2::*;
    pub use crate::api::simd::*;
    pub use crate::api::simulation::*;
    pub use crate::api::specialized::*;
    pub use crate::api::tensor_networks::*;
    pub use crate::api::utils::*;
}
