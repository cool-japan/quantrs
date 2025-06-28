//! Organized prelude modules for simulation functionality

/// Essential types for basic quantum simulation
pub mod essentials {
    //! Core types needed for quantum circuit simulation
    
    // Basic simulators
    pub use crate::api::simulation::{StateVectorSimulator, Simulator, SimulatorResult};
    pub use crate::api::simulation::{Result, SimulatorError};
    
    // Core optimization
    pub use crate::api::optimization::{CircuitOptimizer, optimize_circuit};
    
    // Basic noise models
    pub use crate::api::noise::{NoiseModel, NoiseChannel};
}

/// Complete toolkit for simulation development
pub mod simulation {
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

/// GPU and high-performance computing
pub mod gpu {
    //! GPU and accelerated simulation backends
    
    pub use super::essentials::*;
    
    // GPU backends
    pub use crate::api::gpu::*;
    
    // SIMD operations
    pub use crate::api::simd::*;
    
    // High-performance optimizations
    pub use crate::api::optimization::*;
}

/// Large-scale and distributed simulation
pub mod distributed {
    //! Tools for large-scale distributed quantum simulation
    
    pub use super::simulation::*;
    
    // Distributed simulation
    pub use crate::api::distributed::*;
    
    // Tensor networks
    pub use crate::api::tensor_networks::*;
}

/// Algorithm development and research
pub mod algorithms {
    //! Advanced quantum algorithms and applications
    
    pub use super::simulation::*;
    
    // Quantum algorithms
    pub use crate::api::algorithms::*;
    
    // Machine learning
    pub use crate::api::quantum_ml::*;
    
    // Specialized methods
    pub use crate::api::specialized::*;
}

/// Noise modeling and error correction
pub mod noise_modeling {
    //! Comprehensive noise modeling and error correction
    
    pub use super::essentials::*;
    
    // Noise models
    pub use crate::api::noise::*;
    
    // Error correction
    pub use crate::api::error_correction::*;
}

/// Developer tools and debugging
pub mod dev_tools {
    //! Tools for simulation debugging and development
    
    pub use super::essentials::*;
    
    // Debugging tools
    pub use crate::api::dev_tools::*;
    
    // Profiling and analysis
    pub use crate::api::profiling::*;
    
    // SciRS2 enhanced tools
    pub use crate::api::scirs2::*;
}

/// Legacy compatibility - provides the old flat API
#[deprecated(since = "1.0.0", note = "Use organized modules like `essentials`, `simulation`, etc.")]
pub mod legacy {
    //! Backward compatibility exports
    
    pub use crate::api::simulation::*;
    pub use crate::api::gpu::*;
    pub use crate::api::distributed::*;
    pub use crate::api::optimization::*;
    pub use crate::api::profiling::*;
    pub use crate::api::dev_tools::*;
    pub use crate::api::noise::*;
    pub use crate::api::error_correction::*;
    pub use crate::api::algorithms::*;
    pub use crate::api::quantum_ml::*;
    pub use crate::api::specialized::*;
    pub use crate::api::tensor_networks::*;
    pub use crate::api::memory::*;
    pub use crate::api::simd::*;
    pub use crate::api::gates::*;
    pub use crate::api::measurement::*;
    pub use crate::api::scirs2::*;
    pub use crate::api::utils::*;
    pub use crate::api::dynamic::*;
    pub use crate::api::precision::*;
}

/// Full API re-export (non-deprecated flat access)
pub mod full {
    //! Complete API access with new naming conventions
    
    pub use crate::api::simulation::*;
    pub use crate::api::gpu::*;
    pub use crate::api::distributed::*;
    pub use crate::api::optimization::*;
    pub use crate::api::profiling::*;
    pub use crate::api::dev_tools::*;
    pub use crate::api::noise::*;
    pub use crate::api::error_correction::*;
    pub use crate::api::algorithms::*;
    pub use crate::api::quantum_ml::*;
    pub use crate::api::specialized::*;
    pub use crate::api::tensor_networks::*;
    pub use crate::api::memory::*;
    pub use crate::api::simd::*;
    pub use crate::api::gates::*;
    pub use crate::api::measurement::*;
    pub use crate::api::scirs2::*;
    pub use crate::api::utils::*;
    pub use crate::api::dynamic::*;
    pub use crate::api::precision::*;
}