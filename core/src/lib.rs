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
pub mod qpca;
pub mod quantum_counting;
pub mod quantum_walk;
pub mod qubit;
pub mod register;
pub mod simd_ops;
pub mod testing;

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
    pub use crate::hhl::{hhl_example, HHLAlgorithm, HHLParams};
    pub use crate::memory_efficient::{EfficientStateVector, StateMemoryStats};
    pub use crate::parametric::{Parameter, ParametricGate, SymbolicParameter};
    pub use crate::qaoa::{
        CostHamiltonian, MixerHamiltonian, QAOACircuit, QAOAOptimizer, QAOAParams,
    };
    pub use crate::qpca::{DensityMatrixPCA, QPCAParams, QuantumPCA};
    pub use crate::quantum_counting::{
        amplitude_estimation_example, quantum_counting_example, QuantumAmplitudeEstimation,
        QuantumCounting, QuantumPhaseEstimation,
    };
    pub use crate::quantum_walk::{
        CoinOperator, ContinuousQuantumWalk, DiscreteQuantumWalk, Graph, GraphType,
        QuantumWalkSearch, SearchOracle,
    };
    pub use crate::qubit::*;
    pub use crate::register::*;
    pub use crate::simd_ops::{
        apply_phase_simd, controlled_phase_simd, expectation_z_simd, inner_product, normalize_simd,
    };
    pub use crate::testing::{
        QuantumAssert, QuantumTest, QuantumTestSuite, TestResult, TestSuiteResult,
        DEFAULT_TOLERANCE,
    };
}
