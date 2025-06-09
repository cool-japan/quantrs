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
pub mod error;
pub mod linalg_ops;
pub mod mps_simulator;
pub mod mps_basic;
#[cfg(feature = "mps")]
pub mod mps_enhanced;
pub mod simulator;
pub mod sparse;
pub mod specialized_gates;
pub mod specialized_simulator;
pub mod stabilizer;
pub mod statevector;
pub mod tensor;
pub mod trotter;
pub mod qmc;
pub mod precision;
pub mod fusion;
pub mod pauli;
pub mod debugger;
pub mod autodiff_vqe;
pub mod scirs2_integration;
pub mod shot_sampling;
pub mod open_quantum_systems;
pub mod decision_diagram;
pub mod quantum_supremacy;
pub mod fermionic_simulation;
pub mod quantum_volume;
pub mod noise_extrapolation;
pub mod scirs2_qft;
pub mod scirs2_sparse;
pub mod scirs2_eigensolvers;
pub mod path_integral;
pub mod photonic;

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
    pub use crate::error::{SimulatorError, Result};
    #[allow(unused_imports)]
    pub use crate::error_correction::*;
    pub use crate::noise::*;
    pub use crate::noise::{NoiseChannel, NoiseModel};
    pub use crate::noise_advanced::*;
    pub use crate::noise_advanced::{AdvancedNoiseModel, RealisticNoiseModelBuilder};
    #[allow(unused_imports)]
    pub use crate::simulator::*;
    pub use crate::simulator::{Simulator, SimulatorResult};
    pub use crate::sparse::{CSRMatrix, SparseGates, SparseMatrixBuilder, apply_sparse_gate};
    pub use crate::stabilizer::{is_clifford_circuit, StabilizerGate, StabilizerSimulator};
    pub use crate::statevector::StateVectorSimulator;
    pub use crate::mps_simulator::{MPS, MPSSimulator};
    pub use crate::mps_basic::{BasicMPS, BasicMPSSimulator, BasicMPSConfig};
    #[cfg(feature = "mps")]
    pub use crate::mps_enhanced::{
        EnhancedMPS, EnhancedMPSSimulator, MPSConfig, utils::*
    };
    pub use crate::trotter::{
        Hamiltonian, HamiltonianTerm, TrotterDecomposer, TrotterMethod, HamiltonianLibrary,
    };
    pub use crate::qmc::{
        VMC, DMC, PIMC, WaveFunction, Walker, VMCResult, DMCResult, PIMCResult,
    };
    pub use crate::precision::{
        Precision, AdaptiveStateVector, AdaptivePrecisionConfig, PrecisionTracker,
        PrecisionStats, ComplexAmplitude, ComplexF16, benchmark_precisions,
    };
    pub use crate::fusion::{
        GateFusion, FusionStrategy, GateGroup, FusedGate, OptimizedGate,
        OptimizedCircuit, FusionStats, benchmark_fusion_strategies,
    };
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
    pub use crate::pauli::{
        PauliOperator, PauliString, PauliOperatorSum, PauliUtils,
    };
    pub use crate::debugger::{
        QuantumDebugger, DebugConfig, BreakCondition, Watchpoint, WatchProperty, 
        WatchFrequency, DebugReport, StepResult, PerformanceMetrics,
    };
    pub use crate::autodiff_vqe::{
        VQEWithAutodiff, ParametricCircuit, ParametricGate, AutoDiffContext,
        GradientMethod, VQEResult, VQEIteration, ConvergenceCriteria,
        ParametricRX, ParametricRY, ParametricRZ, ansatze,
    };
    pub use crate::scirs2_integration::{
        SciRS2Backend, BackendStats, MemoryStats, OptimizationMethod,
        OptimizationResult, BenchmarkResults, get_backend, benchmark_scirs2_ops,
    };
    pub use crate::shot_sampling::{
        QuantumSampler, ShotResult, BitString, SamplingConfig, MeasurementStatistics,
        ExpectationResult, ConvergenceResult, NoiseModel as SamplingNoiseModel, SimpleReadoutNoise,
        ComparisonResult, analysis,
    };
    pub use crate::open_quantum_systems::{
        LindladSimulator, LindladOperator, IntegrationMethod, EvolutionResult,
        QuantumChannel, ProcessTomography, NoiseModelBuilder, CompositeNoiseModel,
        quantum_fidelity,
    };
    pub use crate::decision_diagram::{
        DecisionDiagram, DDNode, Edge, DDSimulator, DDStats, DDOptimizer,
        benchmark_dd_simulator,
    };
    pub use crate::quantum_supremacy::{
        QuantumSupremacyVerifier, CrossEntropyResult, PorterThomasResult, HOGAnalysis,
        CostComparison, VerificationParams, GateSet, RandomCircuit, CircuitLayer,
        QuantumGate, benchmark_quantum_supremacy, verify_supremacy_claim,
    };
    pub use crate::fermionic_simulation::{
        FermionicOperator, FermionicString, FermionicHamiltonian, JordanWignerTransform,
        FermionicSimulator, FermionicStats, benchmark_fermionic_simulation,
    };
    pub use crate::quantum_volume::{
        QuantumVolumeResult, QVStats, QVCircuit, QVGate, QuantumVolumeCalculator,
        QVParams, benchmark_quantum_volume, calculate_quantum_volume_with_params,
    };
    pub use crate::noise_extrapolation::{
        ZeroNoiseExtrapolator, ZNEResult, VirtualDistillation, VirtualDistillationResult,
        SymmetryVerification, SymmetryVerificationResult, SymmetryOperation,
        ExtrapolationMethod, NoiseScalingMethod, FitStatistics, DistillationProtocol,
        benchmark_noise_extrapolation,
    };
    pub use crate::scirs2_qft::{
        SciRS2QFT, QFTMethod, QFTConfig, QFTStats, QFTUtils,
        benchmark_qft_methods, compare_qft_accuracy,
    };
    pub use crate::scirs2_sparse::{
        SciRS2SparseSolver, SparseMatrix, SparseFormat, SparseSolverMethod, 
        Preconditioner, SparseSolverConfig, SparseSolverStats, SparseEigenResult,
        SparseMatrixUtils, benchmark_sparse_solvers, compare_sparse_solver_accuracy,
    };
    pub use crate::scirs2_eigensolvers::{
        SciRS2SpectralAnalyzer, SpectralAnalysisResult, PhaseTransitionResult,
        SpectralDensityResult, EntanglementSpectrumResult, BandStructureResult,
        SpectralConfig, SpectralStatistics, QuantumHamiltonianLibrary,
        benchmark_spectral_analysis,
    };
    pub use crate::path_integral::{
        PathIntegralSimulator, PathIntegralMethod, PathIntegralConfig, PathIntegralResult,
        QuantumPath, ConvergenceStats, PathIntegralStats, PathIntegralUtils,
        benchmark_path_integral_methods,
    };
    pub use crate::photonic::{
        PhotonicSimulator, PhotonicMethod, PhotonicConfig, PhotonicState, FockState,
        PhotonicOperator, PhotonicResult, PhotonicStats, PhotonicUtils,
        benchmark_photonic_methods,
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
