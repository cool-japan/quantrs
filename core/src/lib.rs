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
pub mod holonomic;
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
pub mod post_quantum_crypto;
pub mod quantum_channels;
pub mod quantum_counting;
pub mod quantum_ml_accelerators;
pub mod quantum_walk;
pub mod qubit;
pub mod register;
pub mod shannon;
pub mod simd_ops;
pub mod synthesis;
pub mod tensor_network;
pub mod testing;
pub mod topological;
pub mod ultra_high_fidelity_synthesis;
pub mod variational;
pub mod variational_optimization;
pub mod quantum_hardware_abstraction;
pub mod distributed_quantum_networks;
pub mod quantum_memory_integration;
pub mod real_time_compilation;
pub mod quantum_aware_interpreter;
pub mod ultrathink_core;
pub mod quantum_operating_system;
pub mod quantum_internet;
pub mod quantum_sensor_networks;
pub mod quantum_supremacy_algorithms;
pub mod quantum_debugging_profiling;
pub mod quantum_resource_management;
pub mod quantum_memory_hierarchy;
pub mod quantum_process_isolation;
pub mod quantum_garbage_collection;
pub mod quantum_universal_framework;
pub mod quantum_algorithm_profiling;
pub mod zx_calculus;
pub mod zx_extraction;
pub mod compilation_cache;

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
        GpuStateVector, initialize_adaptive_simd, SpecializedGpuKernels, OptimizationConfig,
    };
    pub use crate::compilation_cache::{
        CompilationCache, CacheConfig, CompiledGate, CacheStatistics,
        initialize_compilation_cache, get_compilation_cache,
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
    pub use crate::quantum_ml_accelerators::{
        HardwareEfficientMLLayer, ParameterShiftOptimizer,
        QuantumFeatureMap, QuantumKernelOptimizer, QuantumNaturalGradient,
        TensorNetworkMLAccelerator,
    };
    pub use crate::post_quantum_crypto::{
        CompressionFunction, QKDProtocol, QKDResult, QuantumDigitalSignature,
        QuantumHashFunction, QuantumKeyDistribution, QuantumSignature,
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
    pub use crate::ultra_high_fidelity_synthesis::{
        ErrorSuppressionSynthesis, ErrorSuppressedSequence, GateOperation, GrapeOptimizer,
        GrapeResult, NoiseModel, QuantumGateRL, RLResult, SynthesisConfig, SynthesisMethod,
        UltraFidelityResult, UltraHighFidelitySynthesis,
    };
    pub use crate::quantum_hardware_abstraction::{
        AdaptiveMiddleware, CalibrationEngine, ErrorMitigationLayer, ExecutionRequirements,
        HardwareCapabilities, HardwareResourceManager, HardwareType,
        QuantumHardwareAbstraction, QuantumHardwareBackend,
    };
    pub use crate::distributed_quantum_networks::{
        DistributedQuantumGate, DistributedGateType, EntanglementManager, EntanglementProtocol,
        NetworkScheduler, QuantumNetwork, QuantumNode,
    };
    pub use crate::quantum_memory_integration::{
        QuantumMemory, QuantumState, QuantumStorageLayer, CoherenceManager, 
        MemoryAccessController, QuantumMemoryErrorCorrection,
    };
    pub use crate::real_time_compilation::{
        RealTimeQuantumCompiler, CompilationContext, HardwareTarget,
        OptimizationPipeline, PerformanceMonitor,
    };
    pub use crate::quantum_aware_interpreter::{
        QuantumAwareInterpreter, QuantumStateTracker, QuantumJITCompiler,
        RuntimeOptimizationEngine, ExecutionStrategy, 
        OperationResult as InterpreterOperationResult,
    };
    pub use crate::ultrathink_core::{
        UltraThinkQuantumComputer, HolonomicProcessor, QuantumMLAccelerator,
        QuantumMemoryCore, RealTimeCompiler, DistributedQuantumNetwork,
        QuantumAdvantageReport,
    };
    pub use crate::quantum_operating_system::{
        QuantumOperatingSystem, QuantumScheduler, QuantumMemoryManager,
        QuantumProcessManager, QuantumSecurityManager,
        QuantumOSAdvantageReport,
    };
    pub use crate::quantum_internet::{
        QuantumInternet, QuantumNetworkInfrastructure, QuantumInternetNode, QuantumRouting,
        QuantumInternetSecurity, GlobalQuantumKeyDistribution, DistributedQuantumComputing,
        QuantumInternetAdvantageReport,
    };
    pub use crate::quantum_sensor_networks::{
        QuantumSensorNetwork, QuantumSensor, QuantumSensorType, EntanglementDistribution,
        QuantumMetrologyEngine, DistributedSensingResult, QuantumSensorAdvantageReport,
        EnvironmentalMonitoringResult,
    };
    pub use crate::quantum_supremacy_algorithms::{
        QuantumSupremacyEngine, RandomCircuitSampling, BosonSampling, IQPSampling,
        QuantumSupremacyBenchmarkReport, RandomCircuitSupremacyResult, BosonSamplingSupremacyResult,
        QuantumSimulationAdvantageResult,
    };
    pub use crate::quantum_debugging_profiling::{
        QuantumDebugProfiling, QuantumDebugger, QuantumPerformanceProfiler, QuantumCircuitAnalyzer,
        QuantumStateInspector, QuantumErrorTracker, ProfilingReport, CircuitAnalysisReport,
        StateInspectionReport, QuantumDebugProfilingReport,
    };
    pub use crate::quantum_resource_management::{
        QuantumResourceManager, AdvancedQuantumScheduler, QuantumResourceAllocator, CoherenceAwareManager,
        QuantumWorkloadOptimizer, QuantumProcess, QuantumResourceAdvantageReport, AdvancedSchedulingResult,
        OptimizationLevel, SchedulingPolicy,
    };
    pub use crate::quantum_memory_hierarchy::{
        QuantumMemoryHierarchy, L1QuantumCache, L2QuantumCache, L3QuantumCache, QuantumMainMemory,
        QuantumMemoryOperation, QuantumMemoryResult, QuantumMemoryAdvantageReport, CacheReplacementPolicy,
        MemoryOperationType, OptimizationResult,
    };
    pub use crate::quantum_process_isolation::{
        QuantumProcessIsolation, QuantumSandbox, QuantumAccessController, QuantumStateIsolator,
        IsolatedQuantumProcess, SecurityDomain, VirtualQuantumMachine, IsolationLevel,
        SecureQuantumOperation, QuantumSecurityAdvantageReport, IsolatedProcessResult,
    };
    pub use crate::quantum_garbage_collection::{
        QuantumGarbageCollector, QuantumReferenceCounter, QuantumLifecycleManager, QuantumGCAdvantageReport,
        QuantumAllocationRequest, QuantumAllocationResult, GCCollectionMode, GCCollectionResult,
        CoherenceBasedGC,
    };
    pub use crate::quantum_universal_framework::{
        UniversalQuantumFramework, QuantumHardwareRegistry, UniversalQuantumCompiler,
        CrossPlatformOptimizer, AdaptiveQuantumRuntime, QuantumPortabilityEngine,
        ArchitectureType, UniversalQuantumCircuit, UniversalCompilationResult,
        AdaptiveExecutionResult, UniversalFrameworkAdvantageReport,
    };
    pub use crate::quantum_algorithm_profiling::{
        QuantumAlgorithmProfiler, QuantumPerformanceAnalyzer, QuantumComplexityAnalyzer,
        QuantumBottleneckDetector, QuantumOptimizationAdvisor, QuantumAdvantageCalculator,
        QuantumResourceMonitor, QuantumProfilingReport, QuantumBenchmarkResult,
        QuantumProfilingAdvantageReport, ComplexityClass, AlgorithmType, ProfilingLevel,
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
        OptimizationMethod, OptimizationResult as VariationalOptimizationResult, VariationalQuantumOptimizer,
    };
    pub use crate::zx_calculus::{
        CircuitToZX, Edge, EdgeType, Spider, SpiderType, ZXDiagram, ZXOptimizer,
    };
    pub use crate::zx_extraction::{ZXExtractor, ZXPipeline};
}
