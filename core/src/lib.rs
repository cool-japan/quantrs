//! Core types and traits for the QuantRS2 quantum computing framework.
//!
//! This crate provides the foundational types and traits used throughout
//! the QuantRS2 ecosystem, including qubit identifiers, quantum gates,
//! and register representations.

pub mod complex_ext;
pub mod decomposition;
pub mod error;
pub mod gate;
pub mod hhl;
pub mod memory_efficient;
pub mod parametric;
pub mod qaoa;
pub mod qubit;
pub mod register;
pub mod simd_ops;

/// Re-exports of commonly used types and traits
pub mod prelude {
    // Import specific items from each module to avoid ambiguous glob re-exports
    pub use crate::complex_ext::{quantum_states, QuantumComplexExt};
    pub use crate::decomposition::decompose_u_gate;
    pub use crate::decomposition::utils::{
        clone_gate, decompose_circuit, optimize_gate_sequence, GateSequence,
    };
    pub use crate::error::*;
    pub use crate::gate::*;
    pub use crate::memory_efficient::{EfficientStateVector, StateMemoryStats};
    pub use crate::parametric::{Parameter, ParametricGate, SymbolicParameter};
    pub use crate::qubit::*;
    pub use crate::register::*;
    pub use crate::simd_ops::{
        apply_phase_simd, controlled_phase_simd, expectation_z_simd, inner_product, normalize_simd,
    };
    pub use crate::qaoa::{
        QAOAParams, QAOACircuit, QAOAOptimizer, CostHamiltonian, MixerHamiltonian,
    };
    pub use crate::hhl::{HHLParams, HHLAlgorithm, hhl_example};
}
