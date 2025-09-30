//! Core types and traits for the QuantRS2 quantum computing framework.
//!
//! This crate provides the foundational types and traits used throughout
//! the QuantRS2 ecosystem, including qubit identifiers, quantum gates,
//! and register representations.
//!
//! ## Key Features
//!
//! - **Platform-Aware Optimization**: Automatic detection of CPU/GPU capabilities for optimal performance
//! - **SIMD Acceleration**: Fully migrated to `scirs2_core::simd_ops` for vectorized quantum operations
//! - **GPU Support**: CUDA, OpenCL, and Metal (macOS) backends with forward-compatible SciRS2 integration
//! - **Adaptive Algorithms**: Runtime selection of optimal implementations based on hardware capabilities
//!
//! ## Recent Updates (v0.1.0-beta.2)
//!
//! - Refined SciRS2 v0.1.0-beta.3 integration with unified import patterns
//! - Comprehensive policy documentation (SCIRS2_INTEGRATION_POLICY.md)
//! - Enhanced random number generation with `UnifiedNormal`, `UnifiedBeta`
//! - Consistent SciRS2 usage: `scirs2_core::ndarray::*`, `scirs2_core::random::prelude::*`
//! - Improved developer experience with CLAUDE.md development guidelines

#![allow(clippy::fn_address_comparisons)]
#![allow(clippy::elided_lifetimes_in_paths)]
#![allow(clippy::ptr_eq)]
#![warn(clippy::all)]
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

pub mod adaptive_precision;
pub mod adiabatic;
pub mod batch;
pub mod bosonic;
pub mod buffer_pool;
pub mod cartan;
pub mod characterization;
pub mod circuit_synthesis;
pub mod compilation_cache;
pub mod complex_ext;
pub mod controlled;
pub mod decomposition;
pub mod distributed_quantum_networks;
pub mod eigensolve;
pub mod equivalence_checker;
pub mod error;
pub mod error_correction;
pub mod fermionic;
pub mod gate;
pub mod gate_translation;
pub mod gpu;
mod gpu_stubs;
pub mod hardware_compilation;
pub mod hhl;
pub mod holonomic;
pub mod hybrid_learning;
#[cfg(feature = "python")]
pub mod jupyter_visualization;
pub mod kak_multiqubit;
pub mod linalg_stubs;
pub mod matrix_ops;
pub mod mbqc;
pub mod memory_efficient;
pub mod neutral_atom;
pub mod operations;
pub mod optimization;
pub mod optimization_stubs;
pub mod optimizations_stable;
// Temporarily disabled until scirs2-core cache/profiling features are stable
// pub mod optimizations;
pub mod parallel_ops_stubs;
pub mod parametric;
pub mod photonic;
pub mod platform;
pub mod post_quantum_crypto;
pub mod pulse;
#[cfg(feature = "python")]
pub mod python_bindings;
pub mod qaoa;
pub mod qml;
pub mod qpca;
pub mod quantum_algorithm_profiling;
pub mod quantum_autodiff;
pub mod quantum_aware_interpreter;
pub mod quantum_cellular_automata;
pub mod quantum_channels;
#[cfg(feature = "python")]
pub mod quantum_complexity_analysis;
pub mod quantum_counting;
pub mod quantum_debugger;
pub mod quantum_debugging_profiling;
pub mod quantum_game_theory;
pub mod quantum_garbage_collection;
pub mod quantum_hardware_abstraction;
pub mod quantum_internet;
pub mod scirs2_equivalence_checker;
// pub mod quantum_internet_enhancements;  // Temporarily disabled due to compilation issues
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
pub mod realtime_monitoring;
pub mod register;
pub mod resource_estimator;
pub mod scirs2_auto_optimizer;
pub mod scirs2_circuit_verifier;
pub mod scirs2_circuit_verifier_enhanced;
pub mod scirs2_quantum_formatter;
pub mod scirs2_quantum_formatter_enhanced;
pub mod scirs2_quantum_linter;
pub mod scirs2_quantum_linter_enhanced;
pub mod scirs2_quantum_profiler;
pub mod scirs2_quantum_profiler_enhanced;
pub mod scirs2_resource_estimator_enhanced;
pub mod shannon;
pub mod silicon_quantum_dots;
pub mod simd_ops;
pub mod simd_ops_stubs;
pub mod symbolic;
pub mod symbolic_hamiltonian;
pub mod symbolic_optimization;
pub mod synthesis;
pub mod tensor_network;
pub mod testing;
pub mod topological;
pub mod trapped_ion;
pub mod ultra_high_fidelity_synthesis;
pub mod ultrathink_core;
pub mod variational;
pub mod variational_optimization;
pub mod zx_calculus;
pub mod zx_extraction;

/// New organized API for QuantRS2 1.0
///
/// This module provides a hierarchical organization of the core API
/// with clear naming conventions and logical grouping.
pub mod api;

/// Re-exports of commonly used types and traits
///
/// For new code, consider using the organized API modules in `api::prelude` instead.
/// This module is maintained for backward compatibility.
pub mod prelude {
    // Import specific items from each module to avoid ambiguous glob re-exports
    pub use crate::adiabatic::{
        AdiabaticQuantumComputer, AnnealingSchedule, IsingProblem, ProblemGenerator, ProblemType,
        QUBOProblem, QuantumAnnealer, QuantumAnnealingSnapshot,
    };
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
    pub use crate::neutral_atom::{
        AtomSpecies, AtomState, LaserSystem, NeutralAtom, NeutralAtomErrorModel, NeutralAtomGates,
        NeutralAtomQC, OpticalTweezer, Position3D,
    };
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
    pub use crate::photonic::{
        OpticalMode, PhotonicCircuit, PhotonicEncoding, PhotonicErrorCorrection, PhotonicGate,
        PhotonicGateType, PhotonicSystem,
    };
    pub use crate::post_quantum_crypto::{
        CompressionFunction, QKDProtocol, QKDResult, QuantumDigitalSignature, QuantumHashFunction,
        QuantumKeyDistribution, QuantumSignature,
    };
    pub use crate::pulse::{
        CouplingParams, HardwareCalibration, Pulse, PulseCompiler, PulseEnvelope, PulseNoiseModel,
        PulseSequence, QubitControlParams, TimingConstraints,
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
    pub use crate::quantum_cellular_automata::{
        BoundaryCondition, QCARule, QCAType, QuantumCellularAutomaton1D,
        QuantumCellularAutomaton2D, UnitaryRule,
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
    pub use crate::quantum_game_theory::{
        GameOutcome, GameType, QuantumGame, QuantumMechanism, QuantumPlayer, QuantumStrategy,
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
        CoinOperator, ContinuousQuantumWalk, DecoherentQuantumWalk, DiscreteQuantumWalk, Graph,
        GraphType, MultiWalkerQuantumWalk, QuantumWalkSearch, SearchOracle, SzegedyQuantumWalk,
    };
    pub use crate::qubit::*;
    pub use crate::real_time_compilation::{
        CompilationContext, HardwareTarget, OptimizationPipeline, PerformanceMonitor,
        RealTimeQuantumCompiler,
    };
    pub use crate::register::*;
    pub use crate::scirs2_auto_optimizer::{
        AutoOptimizer, AutoOptimizerConfig, BackendConfiguration, BackendSelection, BackendType,
        CommunicationBackend, ComplexityClass as AutoOptimizerComplexityClass, ComplexityEstimate,
        DistributedConfiguration, EntanglementAnalysis, FloatPrecision, GPUConfiguration,
        GPUMemoryStrategy, GateComposition, LoadBalancingStrategy, MemoryPattern, MemoryStrategy,
        OptimizationRecommendation as AutoOptimizerOptimizationRecommendation,
        ParallelizationPotential, PerformanceMetrics as AutoOptimizerPerformanceMetrics,
        PerformanceProfile, PrecisionSettings, ProblemAnalysis, ProblemSizeLimits,
        ProblemType as AutoOptimizerProblemType,
        RecommendationType as AutoOptimizerRecommendationType, ResourceCost, ResourceMonitor,
        ResourceRequirements as AutoOptimizerResourceRequirements, ResourceUtilization,
    };
    pub use crate::scirs2_circuit_verifier::{
        AlgorithmSpecification, AlgorithmVerificationResult, CircuitVerificationResult,
        EquivalenceType, EquivalenceVerificationResult, NumericalStabilityAnalysis,
        SciRS2CircuitVerifier, SciRS2VerificationEnhancements, VerificationConfig,
        VerificationVerdict,
    };
    pub use crate::scirs2_circuit_verifier_enhanced::{
        CertificateFormat, CircuitProperty, ConfidenceStatistics, Counterexample,
        EnhancedCircuitVerifier, EnhancedVerificationConfig, FormalProof, FormalVerificationResult,
        ProofStep, ProofStepType, ProofType, QCTLSpecification, QHLSpecification,
        QLTLSpecification, SpecificationLanguage, VerificationReport, VerificationSummary,
        VerificationTechnique, ZXSpecification,
    };
    pub use crate::scirs2_quantum_formatter::{
        AnnotationLocation, AnnotationType, CodeAnnotation, CommentStyle, FormattedCode,
        FormattingConfig, FormattingStatistics, FormattingStyle, IndentationStyle, OutputFormat,
        ProgrammingLanguage, SciRS2QuantumFormatter,
    };
    pub use crate::scirs2_quantum_formatter_enhanced::{
        AlgorithmPhase, BeautificationSuggestions, ChangeType, CircuitChange, ColorScheme,
        CustomFormattingRule, EnhancedFormattedCode, EnhancedFormattingConfig,
        EnhancedQuantumFormatter, FormattingOptions, FormattingSuggestion, HardwareFormattingInfo,
        IncrementalUpdate, InteractiveSuggestion, PlatformOptimization, QuantumBackend,
        QuantumPattern, SemanticInfo, SuggestionLocation, SuggestionType, SyntaxMetadata,
        SyntaxScope, SyntaxToken, TemplatedCode, TokenType, UpdatedSection, VisualFormat,
    };
    pub use crate::scirs2_quantum_linter::{
        AutomaticFix, LintFinding, LintFindingType, LintSeverity, LintingConfig, LintingReport,
        OptimizationSuggestion, SciRS2Enhancement, SciRS2QuantumLinter,
    };
    pub use crate::scirs2_quantum_linter_enhanced::{
        ChangeOperation, CircuitLocation, CircuitMetadata, CodeChange, Compatibility,
        CustomLintRule, EnhancedLintFinding, EnhancedLintingConfig, EnhancedLintingReport,
        EnhancedQuantumLinter, FixSuggestion, GatePatternMatcher, HardwareArchitecture,
        ImpactAnalysis, LintPattern, LintingSummary, PerformanceImpact, QualityMetrics,
        ReportFormat, ResourceImpact, ResourceMatcher, RiskLevel, StructuralMatcher,
    };
    pub use crate::scirs2_quantum_profiler::{
        CircuitProfilingResult, GateProfilingResult, MemoryAnalysis, OptimizationRecommendation,
        ProfilingPrecision, ProfilingSessionReport, SciRS2EnhancementSummary,
        SciRS2ProfilingConfig, SciRS2QuantumProfiler, SimdAnalysis,
    };
    pub use crate::scirs2_quantum_profiler_enhanced::{
        AnomalyEvent, AnomalySeverity, Bottleneck, BottleneckAnalysis, BottleneckType, Difficulty,
        EnhancedGateProfilingResult, EnhancedOptimizationRecommendation, EnhancedProfilingConfig,
        EnhancedProfilingReport, EnhancedQuantumProfiler, ExportFormat, HardwareCharacteristics,
        HardwareOptimizationStrategy, HardwarePerformanceModel, MetricStatistics, MetricType,
        OpportunityType, OptimizationOpportunity, PerformanceMetrics, PerformancePredictions,
        PredictedPerformance, Priority, ProfilingSummary, RecommendationType, ScalingAnalysis,
        ScalingModel, ScalingType,
    };
    pub use crate::scirs2_resource_estimator_enhanced::{
        AnalysisDepth, BasicResourceAnalysis, CircuitTopology, CloudPlatform, ComparativeAnalysis,
        ComplexityMetrics, ConstraintPriority, ConstraintType, CostAnalysisResult, CostBreakdown,
        CostOptimization, Effort, EnhancedResourceConfig, EnhancedResourceEstimate,
        EnhancedResourceEstimator, ErrorBudget, EstimationOptions, GatePattern, GateStatistics,
        HardwareRecommendation, Impact, MLPredictions, MemoryRequirements, MonitoringReport,
        OptimizationLevel as ResourceOptimizationLevel, OptimizationObjective,
        OptimizationStrategy, PlatformCost, Priority as ResourcePriority, ReadinessLevel,
        Recommendation, RecommendationCategory, ResourceAnomaly, ResourceConstraint,
        ResourceImprovement, ResourceRequirements, ResourceScores, RiskAssessment,
        ScalingPredictions, TopologyType,
    };
    pub use crate::shannon::{shannon_decompose, OptimizedShannonDecomposer, ShannonDecomposer};
    pub use crate::silicon_quantum_dots::{
        DeviceParams, QuantumDotParams, QuantumDotType, SiliconQuantumDot, SiliconQuantumDotGates,
        SiliconQuantumDotSystem,
    };
    pub use crate::simd_ops::{
        apply_phase_simd, controlled_phase_simd, expectation_z_simd, inner_product, normalize_simd,
    };
    #[cfg(feature = "symbolic")]
    pub use crate::symbolic::calculus::{diff, expand, integrate, limit, simplify};
    pub use crate::symbolic::{matrix::SymbolicMatrix, SymbolicExpression};
    pub use crate::symbolic_hamiltonian::{
        hamiltonians::{
            heisenberg, maxcut, molecular_h2, number_partitioning, transverse_field_ising,
        },
        PauliOperator as SymbolicPauliOperator, PauliString as SymbolicPauliString,
        SymbolicHamiltonian, SymbolicHamiltonianTerm,
    };
    pub use crate::symbolic_optimization::{
        circuit_optimization::{extract_circuit_parameters, optimize_parametric_circuit},
        HamiltonianExpectation, OptimizationResult, QAOACostFunction, SymbolicObjective,
        SymbolicOptimizationConfig, SymbolicOptimizer,
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
    pub use crate::trapped_ion::{
        IonLevel, IonSpecies, LaserPulse, MotionalMode, MotionalModeType, TrappedIon,
        TrappedIonGates, TrappedIonSystem,
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

    #[cfg(feature = "python")]
    pub use crate::python_bindings::{
        PyCartanDecomposition, PyNumRS2Array, PyQuantumGate, PyQuantumInternet,
        PyQuantumSensorNetwork, PyQubitId, PySingleQubitDecomposition, PyVariationalCircuit,
    };

    #[cfg(feature = "python")]
    pub use crate::jupyter_visualization::{
        PyQuantumCircuitVisualizer, PyQuantumPerformanceMonitor, PyQuantumStateVisualizer,
    };

    #[cfg(feature = "python")]
    pub use crate::quantum_complexity_analysis::PyQuantumComplexityAnalyzer;

    #[cfg(feature = "python")]
    pub use crate::python_bindings::{
        PyAggregatedStats, PyAlert, PyMetricMeasurement, PyMonitoringConfig, PyMonitoringStatus,
        PyOptimizationRecommendation, PyRealtimeMonitor,
    };
}

// For backward compatibility, also re-export the prelude at the top level
#[deprecated(since = "1.0.0", note = "Use api::prelude modules for new code")]
pub use prelude::*;

/// Convenient access to the new organized API
///
/// # Examples
///
/// ```rust
/// // For basic quantum programming
/// use quantrs2_core::v1::essentials::*;
///
/// // For algorithm development
/// use quantrs2_core::v1::algorithms::*;
///
/// // For hardware programming
/// use quantrs2_core::v1::hardware::*;
/// ```
pub mod v1 {
    pub use crate::api::prelude::*;
}
