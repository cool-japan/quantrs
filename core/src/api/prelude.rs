//! Organized prelude modules for different use cases

pub mod essentials {
    //! Essential types for basic quantum programming
    //!
    //! This is the recommended starting point for most users.
    //! Provides core types needed for basic quantum circuit programming.

    // Fundamental quantum types
    pub use crate::api::quantum::{GateOp, QubitId, Register};
    pub use crate::api::quantum::{MeasurementOutcome, QuantumOperation};
    pub use crate::api::quantum::{QuantRS2Error, QuantRS2Result};

    // Basic mathematical operations
    pub use crate::api::math::{tensor_product_many, DenseMatrix, QuantumMatrix};

    // Essential synthesis tools
    pub use crate::api::synthesis::{synthesize_unitary, SingleQubitDecomposition};

    // Re-export num_complex for convenience
    pub use scirs2_core::Complex64;
}

pub mod algorithms {
    //! Complete API for quantum algorithm development
    //!
    //! Everything needed for developing quantum algorithms

    pub use super::essentials::*;

    // Variational algorithms
    pub use crate::api::variational::*;

    // Advanced algorithms
    pub use crate::api::algorithms::*;

    // Optimization tools
    pub use crate::api::optimization::{OptimizationChain, OptimizationPass};

    // ML integration
    pub use crate::api::quantum_ml::*;

    // Symbolic computation
    pub use crate::api::symbolic::*;
}

pub mod hardware {
    //! Hardware programming and device interfaces
    //!
    //! Types for programming quantum hardware

    pub use super::essentials::*;

    // Hardware interfaces
    pub use crate::api::hardware::*;

    // Backends and GPU support
    pub use crate::api::backends::*;

    // Error correction
    pub use crate::api::error_correction::*;
}

pub mod research {
    //! Advanced simulation and research tools
    //!
    //! Advanced features for quantum computing research

    pub use super::algorithms::*;

    // Tensor networks
    pub use crate::api::tensor_networks::*;

    // Topological computing
    pub use crate::api::topological::*;

    // Networking and distributed computing
    pub use crate::api::networking::*;

    // ZX-calculus
    pub use crate::api::zx_calculus::*;

    // Batch processing
    pub use crate::api::batch::*;
}

pub mod dev_tools {
    //! Developer tools and debugging utilities
    //!
    //! Tools for debugging and development

    pub use super::essentials::*;

    // Debugging and profiling
    pub use crate::api::dev_tools::*;

    // SciRS2 enhanced tools
    pub use crate::api::scirs2::*;
}

#[cfg(feature = "python")]
pub mod python {
    //! Python integration (when feature enabled)
    //!
    //! Python bindings and Jupyter notebook integration

    pub use crate::api::python::*;
}

#[deprecated(
    since = "1.0.0",
    note = "Use organized modules like `essentials`, `algorithms`, etc."
)]
pub mod legacy {
    //! Legacy compatibility - provides the old flat API
    //!
    //! This module re-exports all types in the old flat structure
    //! for backward compatibility. Use is discouraged for new code.
    //!
    //! This module provides the old flat API structure for compatibility.
    //! New code should use the organized modules instead.

    pub use crate::api::algorithms::*;
    pub use crate::api::backends::*;
    pub use crate::api::batch::*;
    pub use crate::api::dev_tools::*;
    pub use crate::api::error_correction::*;
    pub use crate::api::hardware::*;
    pub use crate::api::math::*;
    pub use crate::api::networking::*;
    pub use crate::api::optimization::*;
    pub use crate::api::quantum::*;
    pub use crate::api::quantum_ml::*;
    pub use crate::api::scirs2::*;
    pub use crate::api::symbolic::*;
    pub use crate::api::synthesis::*;
    pub use crate::api::tensor_networks::*;
    pub use crate::api::topological::*;
    pub use crate::api::variational::*;
    pub use crate::api::zx_calculus::*;

    #[cfg(feature = "python")]
    pub use crate::api::python::*;
}

pub mod full {
    //! Full API re-export (non-deprecated flat access)
    //!
    //! This provides access to all functionality in a flat namespace
    //! while maintaining the new naming conventions.
    //!
    //! Complete API access with new naming conventions

    pub use crate::api::algorithms::*;
    pub use crate::api::backends::*;
    pub use crate::api::batch::*;
    pub use crate::api::dev_tools::*;
    pub use crate::api::error_correction::*;
    pub use crate::api::hardware::*;
    pub use crate::api::math::*;
    pub use crate::api::networking::*;
    pub use crate::api::optimization::*;
    pub use crate::api::quantum::*;
    pub use crate::api::quantum_ml::*;
    pub use crate::api::scirs2::*;
    pub use crate::api::symbolic::*;
    pub use crate::api::synthesis::*;
    pub use crate::api::tensor_networks::*;
    pub use crate::api::topological::*;
    pub use crate::api::variational::*;
    pub use crate::api::zx_calculus::*;

    #[cfg(feature = "python")]
    pub use crate::api::python::*;
}
