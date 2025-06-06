//! Core types and traits for the QuantRS2 quantum computing framework.
//!
//! This crate provides the foundational types and traits used throughout
//! the QuantRS2 ecosystem, including qubit identifiers, quantum gates,
//! and register representations.

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
pub mod batch;
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
    pub use crate::complex_ext::{quantum_states, QuantumComplexExt};
    pub use crate::controlled::{
        make_controlled, make_multi_controlled, ControlledGate, MultiControlledGate,
        ToffoliGate, FredkinGate,
    };
    pub use crate::decomposition::decompose_u_gate;
    pub use crate::decomposition::solovay_kitaev::{
        SolovayKitaev, SolovayKitaevConfig, BaseGateSet, count_t_gates,
    };
    pub use crate::decomposition::clifford_t::{
        CliffordTDecomposer, CliffordTSequence, CliffordTGate, CliffordGate,
        count_t_gates_in_sequence, optimize_gate_sequence as optimize_clifford_t_sequence,
    };
    pub use crate::decomposition::utils::{
        clone_gate, decompose_circuit, optimize_gate_sequence, GateSequence,
    };
    pub use crate::error::*;
    pub use crate::gate::*;
    pub use crate::matrix_ops::{
        DenseMatrix, SparseMatrix, QuantumMatrix, partial_trace, tensor_product_many,
        matrices_approx_equal,
    };
    pub use crate::operations::{
        QuantumOperation, OperationResult, MeasurementOutcome,
        ProjectiveMeasurement, POVMMeasurement, Reset,
        sample_outcome, apply_and_sample,
    };
    pub use crate::synthesis::{
        synthesize_unitary, decompose_single_qubit_zyz, decompose_single_qubit_xyx,
        decompose_two_qubit_kak, identify_gate, SingleQubitDecomposition, KAKDecomposition,
    };
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
    pub use crate::optimization::{
        OptimizationPass, OptimizationChain, gates_are_disjoint, gates_can_commute,
    };
    pub use crate::optimization::compression::{
        GateSequenceCompressor, CompressionConfig, CompressedGate, CompressionStats,
    };
    pub use crate::optimization::fusion::{
        GateFusion, CliffordFusion,
    };
    pub use crate::optimization::peephole::{
        PeepholeOptimizer, TCountOptimizer,
    };
    pub use crate::testing::{
        QuantumAssert, QuantumTest, QuantumTestSuite, TestResult, TestSuiteResult,
        DEFAULT_TOLERANCE,
    };
    pub use crate::characterization::{
        GateCharacterizer, GateEigenstructure, GateType,
    };
    pub use crate::zx_calculus::{
        ZXDiagram, Spider, SpiderType, Edge, EdgeType,
        CircuitToZX, ZXOptimizer,
    };
    pub use crate::zx_extraction::{
        ZXExtractor, ZXPipeline,
    };
    pub use crate::optimization::zx_optimizer::ZXOptimizationPass;
    pub use crate::shannon::{ShannonDecomposer, OptimizedShannonDecomposer, shannon_decompose};
    pub use crate::cartan::{CartanDecomposer, OptimizedCartanDecomposer, CartanDecomposition, CartanCoefficients, cartan_decompose};
    pub use crate::kak_multiqubit::{
        MultiQubitKAKDecomposer, MultiQubitKAK, DecompositionTree, DecompositionMethod,
        KAKTreeAnalyzer, DecompositionStats, kak_decompose_multiqubit,
    };
    pub use crate::quantum_channels::{
        QuantumChannel, KrausRepresentation, ChoiRepresentation, StinespringRepresentation,
        QuantumChannels, ProcessTomography,
    };
    pub use crate::variational::{
        VariationalGate, VariationalCircuit, VariationalOptimizer,
        DiffMode, Dual, ComputationGraph, Node, Operation,
    };
    pub use crate::variational_optimization::{
        VariationalQuantumOptimizer, OptimizationMethod, 
        OptimizationConfig as VariationalOptimizationConfig,
        OptimizationResult, OptimizationHistory, ConstrainedVariationalOptimizer,
        HyperparameterOptimizer as VariationalHyperparameterOptimizer, 
        create_vqe_optimizer, create_qaoa_optimizer,
        create_natural_gradient_optimizer, create_spsa_optimizer,
    };
    pub use crate::tensor_network::{
        Tensor, TensorNetwork, TensorNetworkBuilder, TensorNetworkSimulator,
        TensorEdge, contraction_optimization::DynamicProgrammingOptimizer,
    };
    pub use crate::fermionic::{
        FermionOperator, FermionOperatorType, FermionTerm, FermionHamiltonian,
        JordanWigner, BravyiKitaev, PauliOperator, QubitOperator, QubitTerm,
        qubit_operator_to_gates,
    };
    pub use crate::bosonic::{
        BosonOperator, BosonOperatorType, BosonTerm, BosonHamiltonian,
        GaussianState, boson_to_qubit_encoding,
    };
    pub use crate::error_correction::{
        Pauli, PauliString, StabilizerCode, SurfaceCode, ColorCode,
        SyndromeDecoder, LookupDecoder, MWPMDecoder,
    };
    pub use crate::topological::{
        AnyonType, AnyonModel, FibonacciModel, IsingModel,
        AnyonWorldline, BraidingOperation, FusionTree,
        TopologicalQC, TopologicalGate, ToricCode,
    };
    pub use crate::mbqc::{
        MeasurementBasis, Graph as MBQCGraph, MeasurementPattern, ClusterState,
        MBQCComputation, CircuitToMBQC,
    };
    pub use crate::gpu::{
        GpuBackend, GpuBuffer, GpuKernel, GpuStateVector, GpuBackendFactory,
        GpuConfig, cpu_backend::CpuBackend,
    };
    pub use crate::qml::{
        QMLLayer, QMLCircuit, QMLConfig, EncodingStrategy, EntanglementPattern,
        create_entangling_gates, quantum_fisher_information, natural_gradient,
    };
    pub use crate::qml::layers::{
        RotationLayer, EntanglingLayer, StronglyEntanglingLayer, HardwareEfficientLayer,
        QuantumPoolingLayer, PoolingStrategy,
    };
    pub use crate::qml::encoding::{
        DataEncoder, FeatureMap, FeatureMapType, DataReuploader,
    };
    pub use crate::qml::training::{
        QMLTrainer, TrainingConfig, TrainingMetrics, LossFunction, Optimizer,
        HyperparameterOptimizer, HPOStrategy,
    };
    pub use crate::batch::{
        BatchStateVector, BatchConfig, BatchExecutionResult, BatchMeasurementResult,
        BatchGateOp, create_batch, split_batch, merge_batches,
    };
    pub use crate::batch::operations::{
        apply_single_qubit_gate_batch, apply_two_qubit_gate_batch,
        apply_gate_sequence_batch, compute_expectation_values_batch,
    };
    pub use crate::batch::execution::{
        BatchCircuitExecutor, BatchCircuit, create_optimized_executor,
    };
    pub use crate::batch::measurement::{
        measure_batch, measure_batch_with_statistics, measure_expectation_batch,
        measure_tomography_batch, MeasurementConfig, BatchMeasurementStatistics,
        TomographyBasis, BatchTomographyResult,
    };
    pub use crate::batch::optimization::{
        BatchParameterOptimizer, OptimizationConfig as BatchOptimizationConfig, 
        BatchVQE, BatchQAOA, VQEResult, QAOAResult,
    };
}
