//! Core types and traits for the QuantRS2 quantum computing framework.
//!
//! This crate provides the foundational types and traits used throughout
//! the QuantRS2 ecosystem, including qubit identifiers, quantum gates,
//! and register representations.

pub mod batch;
pub mod bosonic;
pub mod cartan;
pub mod characterization;
pub mod complex_ext;
pub mod controlled;
pub mod decomposition;
pub mod eigensolve;
pub mod error;
pub mod error_correction;
pub mod fermionic;
pub mod gate;
pub mod gpu;
pub mod hhl;
pub mod kak_multiqubit;
pub mod matrix_ops;
pub mod mbqc;
pub mod memory_efficient;
pub mod operations;
pub mod optimization;
pub mod parametric;
pub mod qaoa;
pub mod qml;
pub mod qpca;
pub mod quantum_channels;
pub mod quantum_counting;
pub mod quantum_walk;
pub mod qubit;
pub mod register;
pub mod shannon;
pub mod simd_ops;
pub mod synthesis;
pub mod tensor_network;
pub mod testing;
pub mod topological;
pub mod variational;
pub mod variational_optimization;
pub mod zx_calculus;
pub mod zx_extraction;

/// Re-exports of commonly used types and traits
pub mod prelude {
    // Import specific items from each module to avoid ambiguous glob re-exports
    pub use crate::batch::execution::{
        create_optimized_executor, BatchCircuit, BatchCircuitExecutor,
    };
    pub use crate::batch::measurement::{
        measure_batch, measure_batch_with_statistics, measure_expectation_batch,
        measure_tomography_batch, BatchMeasurementStatistics, BatchTomographyResult,
        MeasurementConfig, TomographyBasis,
    };
    pub use crate::batch::operations::{
        apply_gate_sequence_batch, apply_single_qubit_gate_batch, apply_two_qubit_gate_batch,
        compute_expectation_values_batch,
    };
    pub use crate::batch::optimization::{
        BatchParameterOptimizer, BatchQAOA, BatchVQE,
        OptimizationConfig as BatchOptimizationConfig, QAOAResult, VQEResult,
    };
    pub use crate::batch::{
        create_batch, merge_batches, split_batch, BatchConfig, BatchExecutionResult, BatchGateOp,
        BatchMeasurementResult, BatchStateVector,
    };
    pub use crate::bosonic::{
        boson_to_qubit_encoding, BosonHamiltonian, BosonOperator, BosonOperatorType, BosonTerm,
        GaussianState,
    };
    pub use crate::cartan::{
        cartan_decompose, CartanCoefficients, CartanDecomposer, CartanDecomposition,
        OptimizedCartanDecomposer,
    };
    pub use crate::characterization::{GateCharacterizer, GateEigenstructure, GateType};
    pub use crate::complex_ext::{quantum_states, QuantumComplexExt};
    pub use crate::controlled::{
        make_controlled, make_multi_controlled, ControlledGate, FredkinGate, MultiControlledGate,
        ToffoliGate,
    };
    pub use crate::decomposition::clifford_t::{
        count_t_gates_in_sequence, optimize_gate_sequence as optimize_clifford_t_sequence,
        CliffordGate, CliffordTDecomposer, CliffordTGate, CliffordTSequence,
    };
    pub use crate::decomposition::decompose_u_gate;
    pub use crate::decomposition::solovay_kitaev::{
        count_t_gates, BaseGateSet, SolovayKitaev, SolovayKitaevConfig,
    };
    pub use crate::decomposition::utils::{
        clone_gate, decompose_circuit, optimize_gate_sequence, GateSequence,
    };
    pub use crate::error::*;
    pub use crate::error_correction::{
        ColorCode, LookupDecoder, MWPMDecoder, Pauli, PauliString, StabilizerCode, SurfaceCode,
        SyndromeDecoder,
    };
    pub use crate::fermionic::{
        qubit_operator_to_gates, BravyiKitaev, FermionHamiltonian, FermionOperator,
        FermionOperatorType, FermionTerm, JordanWigner, PauliOperator, QubitOperator, QubitTerm,
    };
    pub use crate::gate::*;
    pub use crate::gpu::{
        cpu_backend::CpuBackend, GpuBackend, GpuBackendFactory, GpuBuffer, GpuConfig, GpuKernel,
        GpuStateVector,
    };
    pub use crate::hhl::{hhl_example, HHLAlgorithm, HHLParams};
    pub use crate::kak_multiqubit::{
        kak_decompose_multiqubit, DecompositionMethod, DecompositionStats, DecompositionTree,
        KAKTreeAnalyzer, MultiQubitKAK, MultiQubitKAKDecomposer,
    };
    pub use crate::matrix_ops::{
        matrices_approx_equal, partial_trace, tensor_product_many, DenseMatrix, QuantumMatrix,
        SparseMatrix,
    };
    pub use crate::mbqc::{
        CircuitToMBQC, ClusterState, Graph as MBQCGraph, MBQCComputation, MeasurementBasis,
        MeasurementPattern,
    };
    pub use crate::memory_efficient::{EfficientStateVector, StateMemoryStats};
    pub use crate::operations::{
        apply_and_sample, sample_outcome, MeasurementOutcome, OperationResult, POVMMeasurement,
        ProjectiveMeasurement, QuantumOperation, Reset,
    };
    pub use crate::optimization::compression::{
        CompressedGate, CompressionConfig, CompressionStats, GateSequenceCompressor,
    };
    pub use crate::optimization::fusion::{CliffordFusion, GateFusion};
    pub use crate::optimization::peephole::{PeepholeOptimizer, TCountOptimizer};
    pub use crate::optimization::zx_optimizer::ZXOptimizationPass;
    pub use crate::optimization::{
        gates_are_disjoint, gates_can_commute, OptimizationChain, OptimizationPass,
    };
    pub use crate::parametric::{Parameter, ParametricGate, SymbolicParameter};
    pub use crate::qaoa::{
        CostHamiltonian, MixerHamiltonian, QAOACircuit, QAOAOptimizer, QAOAParams,
    };
    pub use crate::qml::encoding::{DataEncoder, DataReuploader, FeatureMap, FeatureMapType};
    pub use crate::qml::layers::{
        EntanglingLayer, HardwareEfficientLayer, PoolingStrategy, QuantumPoolingLayer,
        RotationLayer, StronglyEntanglingLayer,
    };
    pub use crate::qml::training::{
        HPOStrategy, HyperparameterOptimizer, LossFunction, Optimizer, QMLTrainer, TrainingConfig,
        TrainingMetrics,
    };
    pub use crate::qml::{
        create_entangling_gates, natural_gradient, quantum_fisher_information, EncodingStrategy,
        EntanglementPattern, QMLCircuit, QMLConfig, QMLLayer,
    };
    pub use crate::qpca::{DensityMatrixPCA, QPCAParams, QuantumPCA};
    pub use crate::quantum_channels::{
        ChoiRepresentation, KrausRepresentation, ProcessTomography, QuantumChannel,
        QuantumChannels, StinespringRepresentation,
    };
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
    pub use crate::shannon::{shannon_decompose, OptimizedShannonDecomposer, ShannonDecomposer};
    pub use crate::simd_ops::{
        apply_phase_simd, controlled_phase_simd, expectation_z_simd, inner_product, normalize_simd,
    };
    pub use crate::synthesis::{
        decompose_single_qubit_xyx, decompose_single_qubit_zyz, decompose_two_qubit_kak,
        identify_gate, synthesize_unitary, KAKDecomposition, SingleQubitDecomposition,
    };
    pub use crate::tensor_network::{
        contraction_optimization::DynamicProgrammingOptimizer, Tensor, TensorEdge, TensorNetwork,
        TensorNetworkBuilder, TensorNetworkSimulator,
    };
    pub use crate::testing::{
        QuantumAssert, QuantumTest, QuantumTestSuite, TestResult, TestSuiteResult,
        DEFAULT_TOLERANCE,
    };
    pub use crate::topological::{
        AnyonModel, AnyonType, AnyonWorldline, BraidingOperation, FibonacciModel, FusionTree,
        IsingModel, TopologicalGate, TopologicalQC, ToricCode,
    };
    pub use crate::variational::{
        ComputationGraph, DiffMode, Dual, Node, Operation, VariationalCircuit, VariationalGate,
        VariationalOptimizer,
    };
    pub use crate::variational_optimization::{
        create_natural_gradient_optimizer, create_qaoa_optimizer, create_spsa_optimizer,
        create_vqe_optimizer, ConstrainedVariationalOptimizer,
        HyperparameterOptimizer as VariationalHyperparameterOptimizer,
        OptimizationConfig as VariationalOptimizationConfig, OptimizationHistory,
        OptimizationMethod, OptimizationResult, VariationalQuantumOptimizer,
    };
    pub use crate::zx_calculus::{
        CircuitToZX, Edge, EdgeType, Spider, SpiderType, ZXDiagram, ZXOptimizer,
    };
    pub use crate::zx_extraction::{ZXExtractor, ZXPipeline};
}
