//! Core types and traits for the QuantRS2 quantum computing framework.
//!
//! This crate provides the foundational types and traits used throughout
//! the QuantRS2 ecosystem, including qubit identifiers, quantum gates,
//! and register representations.

#![allow(dead_code)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::new_without_default)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::module_inception)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::op_ref)]
#![allow(clippy::manual_flatten)]
#![allow(clippy::map_clone)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::default_constructed_unit_structs)]
#![allow(clippy::useless_vec)]
#![allow(clippy::identity_op)]
#![allow(clippy::single_match)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::legacy_numeric_constants)]
#![allow(clippy::unnecessary_min_or_max)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::borrowed_box)]
#![allow(clippy::explicit_auto_deref)]
#![allow(clippy::await_holding_lock)]
#![allow(clippy::unused_enumerate_index)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::needless_bool)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::needless_question_mark)]

pub mod batch;
pub mod bosonic;
pub mod cartan;
pub mod characterization;
pub mod compilation_cache;
pub mod complex_ext;
pub mod controlled;
pub mod decomposition;
pub mod distributed_quantum_networks;
pub mod eigensolve;
pub mod error;
pub mod error_correction;
pub mod fermionic;
pub mod gate;
pub mod gpu;
pub mod hhl;
pub mod holonomic;
pub mod kak_multiqubit;
pub mod matrix_ops;
pub mod mbqc;
pub mod memory_efficient;
pub mod operations;
pub mod optimization;
pub mod parametric;
pub mod post_quantum_crypto;
pub mod qaoa;
pub mod qml;
pub mod qpca;
pub mod quantum_algorithm_profiling;
pub mod quantum_aware_interpreter;
pub mod quantum_channels;
pub mod quantum_counting;
pub mod quantum_debugging_profiling;
pub mod quantum_garbage_collection;
pub mod quantum_hardware_abstraction;
pub mod quantum_internet;
pub mod quantum_memory_hierarchy;
pub mod quantum_memory_integration;
pub mod quantum_ml_accelerators;
pub mod quantum_operating_system;
pub mod quantum_process_isolation;
pub mod quantum_resource_management;
pub mod quantum_sensor_networks;
pub mod quantum_supremacy_algorithms;
pub mod quantum_universal_framework;
pub mod quantum_walk;
pub mod qubit;
pub mod real_time_compilation;
pub mod register;
pub mod shannon;
pub mod simd_ops;
pub mod symbolic;
pub mod symbolic_hamiltonian;
pub mod symbolic_optimization;
pub mod synthesis;
pub mod tensor_network;
pub mod testing;
pub mod topological;
pub mod ultra_high_fidelity_synthesis;
pub mod ultrathink_core;
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
    pub use crate::compilation_cache::{
        get_compilation_cache, initialize_compilation_cache, CacheConfig, CacheStatistics,
        CompilationCache, CompiledGate,
    };
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
    pub use crate::distributed_quantum_networks::{
        DistributedGateType, DistributedQuantumGate, EntanglementManager, EntanglementProtocol,
        NetworkScheduler, QuantumNetwork, QuantumNode,
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
        cpu_backend::CpuBackend, initialize_adaptive_simd, GpuBackend, GpuBackendFactory,
        GpuBuffer, GpuConfig, GpuKernel, GpuStateVector, OptimizationConfig, SpecializedGpuKernels,
    };
    pub use crate::hhl::{hhl_example, HHLAlgorithm, HHLParams};
    pub use crate::holonomic::{
        // GeometricErrorCorrection, HolonomicGate, HolonomicGateSynthesis, HolonomicPath,
        // HolonomicQuantumComputer, PathOptimizationConfig,
        WilsonLoop,
    };
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
        CompressedGate, CompressionConfig, CompressionStats, CompressionType, GateMetadata,
        GateSequenceCompressor,
    };
    pub use crate::optimization::fusion::{CliffordFusion, GateFusion};
    pub use crate::optimization::lazy_evaluation::{
        LazyEvaluationConfig, LazyEvaluationStats, LazyGateContext, LazyOptimizationPipeline,
        OptimizationResult as LazyOptimizationResult, OptimizationStats,
    };
    pub use crate::optimization::peephole::{PeepholeOptimizer, TCountOptimizer};
    pub use crate::optimization::zx_optimizer::ZXOptimizationPass;
    pub use crate::optimization::{
        gates_are_disjoint, gates_can_commute, OptimizationChain, OptimizationPass,
    };
    pub use crate::parametric::{Parameter, ParametricGate, SymbolicParameter};
    pub use crate::post_quantum_crypto::{
        CompressionFunction, QKDProtocol, QKDResult, QuantumDigitalSignature, QuantumHashFunction,
        QuantumKeyDistribution, QuantumSignature,
    };
    pub use crate::qaoa::{
        CostHamiltonian, MixerHamiltonian, QAOACircuit, QAOAOptimizer, QAOAParams,
    };
    pub use crate::qml::encoding::{DataEncoder, DataReuploader, FeatureMap, FeatureMapType};
    pub use crate::qml::generative_adversarial::{
        NoiseType, QGANConfig, QGANIterationMetrics, QGANTrainingStats, QuantumDiscriminator,
        QuantumGenerator, QGAN,
    };
    pub use crate::qml::layers::{
        EntanglingLayer, HardwareEfficientLayer, PoolingStrategy, QuantumPoolingLayer,
        RotationLayer, StronglyEntanglingLayer,
    };
    pub use crate::qml::reinforcement_learning::{
        Experience, QLearningStats, QuantumActorCritic, QuantumDQN, QuantumPolicyNetwork,
        QuantumRLConfig, QuantumValueNetwork, ReplayBuffer, TrainingMetrics as RLTrainingMetrics,
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
    pub use crate::quantum_algorithm_profiling::{
        AlgorithmType, ComplexityClass, ProfilingLevel, QuantumAdvantageCalculator,
        QuantumAlgorithmProfiler, QuantumBenchmarkResult, QuantumBottleneckDetector,
        QuantumComplexityAnalyzer, QuantumOptimizationAdvisor, QuantumPerformanceAnalyzer,
        QuantumProfilingAdvantageReport, QuantumProfilingReport, QuantumResourceMonitor,
    };
    pub use crate::quantum_aware_interpreter::{
        ExecutionStrategy, OperationResult as InterpreterOperationResult, QuantumAwareInterpreter,
        QuantumJITCompiler, QuantumStateTracker, RuntimeOptimizationEngine,
    };
    pub use crate::quantum_channels::{
        ChoiRepresentation, KrausRepresentation, ProcessTomography, QuantumChannel,
        QuantumChannels, StinespringRepresentation,
    };
    pub use crate::quantum_counting::{
        amplitude_estimation_example, quantum_counting_example, QuantumAmplitudeEstimation,
        QuantumCounting, QuantumPhaseEstimation,
    };
    pub use crate::quantum_debugging_profiling::{
        CircuitAnalysisReport, ProfilingReport, QuantumCircuitAnalyzer, QuantumDebugProfiling,
        QuantumDebugProfilingReport, QuantumDebugger, QuantumErrorTracker,
        QuantumPerformanceProfiler, QuantumStateInspector, StateInspectionReport,
    };
    pub use crate::quantum_garbage_collection::{
        CoherenceBasedGC, GCCollectionMode, GCCollectionResult, QuantumAllocationRequest,
        QuantumAllocationResult, QuantumGCAdvantageReport, QuantumGarbageCollector,
        QuantumLifecycleManager, QuantumReferenceCounter,
    };
    pub use crate::quantum_hardware_abstraction::{
        AdaptiveMiddleware, CalibrationEngine, ErrorMitigationLayer, ExecutionRequirements,
        HardwareCapabilities, HardwareResourceManager, HardwareType, QuantumHardwareAbstraction,
        QuantumHardwareBackend,
    };
    pub use crate::quantum_internet::{
        DistributedQuantumComputing, GlobalQuantumKeyDistribution, QuantumInternet,
        QuantumInternetAdvantageReport, QuantumInternetNode, QuantumInternetSecurity,
        QuantumNetworkInfrastructure, QuantumRouting,
    };
    pub use crate::quantum_memory_hierarchy::{
        CacheReplacementPolicy, L1QuantumCache, L2QuantumCache, L3QuantumCache,
        MemoryOperationType, OptimizationResult as MemoryOptimizationResult, QuantumMainMemory,
        QuantumMemoryAdvantageReport, QuantumMemoryHierarchy, QuantumMemoryOperation,
        QuantumMemoryResult,
    };
    pub use crate::quantum_memory_integration::{
        CoherenceManager, MemoryAccessController, QuantumMemory, QuantumMemoryErrorCorrection,
        QuantumState, QuantumStorageLayer,
    };
    pub use crate::quantum_ml_accelerators::{
        HardwareEfficientMLLayer, ParameterShiftOptimizer, QuantumFeatureMap,
        QuantumKernelOptimizer, QuantumNaturalGradient, TensorNetworkMLAccelerator,
    };
    pub use crate::quantum_operating_system::{
        QuantumMemoryManager, QuantumOSAdvantageReport, QuantumOperatingSystem,
        QuantumProcessManager, QuantumScheduler, QuantumSecurityManager,
    };
    pub use crate::quantum_process_isolation::{
        IsolatedProcessResult, IsolatedQuantumProcess, IsolationLevel, QuantumAccessController,
        QuantumProcessIsolation, QuantumSandbox, QuantumSecurityAdvantageReport,
        QuantumStateIsolator, SecureQuantumOperation, SecurityDomain, VirtualQuantumMachine,
    };
    pub use crate::quantum_resource_management::{
        AdvancedQuantumScheduler, AdvancedSchedulingResult, CoherenceAwareManager,
        OptimizationLevel, QuantumProcess, QuantumResourceAdvantageReport,
        QuantumResourceAllocator, QuantumResourceManager, QuantumWorkloadOptimizer,
        SchedulingPolicy,
    };
    pub use crate::quantum_sensor_networks::{
        DistributedSensingResult, EntanglementDistribution, EnvironmentalMonitoringResult,
        QuantumMetrologyEngine, QuantumSensor, QuantumSensorAdvantageReport, QuantumSensorNetwork,
        QuantumSensorType,
    };
    pub use crate::quantum_supremacy_algorithms::{
        BosonSampling, BosonSamplingSupremacyResult, IQPSampling, QuantumSimulationAdvantageResult,
        QuantumSupremacyBenchmarkReport, QuantumSupremacyEngine, RandomCircuitSampling,
        RandomCircuitSupremacyResult,
    };
    pub use crate::quantum_universal_framework::{
        AdaptiveExecutionResult, AdaptiveQuantumRuntime, ArchitectureType, CrossPlatformOptimizer,
        QuantumHardwareRegistry, QuantumPortabilityEngine, UniversalCompilationResult,
        UniversalFrameworkAdvantageReport, UniversalQuantumCircuit, UniversalQuantumCompiler,
        UniversalQuantumFramework,
    };
    pub use crate::quantum_walk::{
        CoinOperator, ContinuousQuantumWalk, DiscreteQuantumWalk, Graph, GraphType,
        QuantumWalkSearch, SearchOracle,
    };
    pub use crate::qubit::*;
    pub use crate::real_time_compilation::{
        CompilationContext, HardwareTarget, OptimizationPipeline, PerformanceMonitor,
        RealTimeQuantumCompiler,
    };
    pub use crate::register::*;
    pub use crate::shannon::{shannon_decompose, OptimizedShannonDecomposer, ShannonDecomposer};
    pub use crate::simd_ops::{
        apply_phase_simd, controlled_phase_simd, expectation_z_simd, inner_product, normalize_simd,
    };
    pub use crate::symbolic::{
        SymbolicExpression,
        matrix::SymbolicMatrix,
    };
    #[cfg(feature = "symbolic")]
    pub use crate::symbolic::{
        calculus::{diff, integrate, limit, expand, simplify},
    };
    pub use crate::symbolic_hamiltonian::{
        PauliOperator as SymbolicPauliOperator, PauliString as SymbolicPauliString, 
        SymbolicHamiltonian, SymbolicHamiltonianTerm,
        hamiltonians::{transverse_field_ising, heisenberg, maxcut, number_partitioning, molecular_h2},
    };
    pub use crate::symbolic_optimization::{
        SymbolicOptimizationConfig, SymbolicOptimizer, OptimizationResult, 
        SymbolicObjective, HamiltonianExpectation, QAOACostFunction,
        circuit_optimization::{optimize_parametric_circuit, extract_circuit_parameters},
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
    pub use crate::ultra_high_fidelity_synthesis::{
        ErrorSuppressedSequence, ErrorSuppressionSynthesis, GateOperation, GrapeOptimizer,
        GrapeResult, NoiseModel, QuantumGateRL, RLResult, SynthesisConfig, SynthesisMethod,
        UltraFidelityResult, UltraHighFidelitySynthesis,
    };
    pub use crate::ultrathink_core::{
        DistributedQuantumNetwork, HolonomicProcessor, QuantumAdvantageReport,
        QuantumMLAccelerator, QuantumMemoryCore, RealTimeCompiler, UltraThinkQuantumComputer,
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
        OptimizationMethod, OptimizationResult as VariationalOptimizationResult,
        VariationalQuantumOptimizer,
    };
    pub use crate::zx_calculus::{
        CircuitToZX, Edge, EdgeType, Spider, SpiderType, ZXDiagram, ZXOptimizer,
    };
    pub use crate::zx_extraction::{ZXExtractor, ZXPipeline};
}
