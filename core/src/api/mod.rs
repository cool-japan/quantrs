//! Organized public API for QuantRS2 Core
//!
//! This module provides a hierarchical organization of the core crate's public API,
//! designed for the 1.0 release with clear naming conventions and logical grouping.

pub mod prelude;

/// Core quantum computing primitives and fundamental types
pub mod quantum {
    pub use crate::gate::*;
    /// Basic quantum types
    pub use crate::qubit::*;
    pub use crate::register::*;

    /// Quantum operations and measurements
    pub use crate::operations::{
        apply_and_sample, sample_outcome, MeasurementOutcome, OperationResult, POVMMeasurement,
        ProjectiveMeasurement, QuantumOperation, Reset,
    };

    /// Error handling
    pub use crate::error::*;
}

/// Circuit decomposition and synthesis tools
pub mod synthesis {
    pub use crate::cartan::{
        cartan_decompose, CartanCoefficients, CartanDecomposer, CartanDecomposition,
        OptimizedCartanDecomposer,
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
    pub use crate::kak_multiqubit::{
        kak_decompose_multiqubit, DecompositionMethod, DecompositionStats, DecompositionTree,
        KAKTreeAnalyzer, MultiQubitKAK, MultiQubitKAKDecomposer,
    };
    pub use crate::shannon::{shannon_decompose, OptimizedShannonDecomposer, ShannonDecomposer};
    pub use crate::synthesis::{
        decompose_single_qubit_xyx, decompose_single_qubit_zyz, decompose_two_qubit_kak,
        identify_gate, synthesize_unitary, KAKDecomposition, SingleQubitDecomposition,
    };
}

/// Mathematical operations and linear algebra
pub mod math {
    pub use crate::complex_ext::{quantum_states, QuantumComplexExt};
    pub use crate::matrix_ops::{
        matrices_approx_equal, partial_trace, tensor_product_many, DenseMatrix, QuantumMatrix,
        SparseMatrix,
    };
    pub use crate::simd_ops::{
        apply_phase_simd, controlled_phase_simd, expectation_z_simd, inner_product, normalize_simd,
    };
}

/// Performance computing backends
pub mod backends {
    pub use crate::gpu::{
        cpu_backend::CpuBackend, initialize_adaptive_simd, GpuBackend, GpuBackendFactory,
        GpuBuffer, GpuConfig, GpuKernel, GpuStateVector, OptimizationConfig, SpecializedGpuKernels,
    };
    pub use crate::platform::*;
}

/// Optimization and compilation tools
pub mod optimization {
    pub use crate::compilation_cache::{
        get_compilation_cache, initialize_compilation_cache, CacheConfig, CacheStatistics,
        CompilationCache, CompiledGate,
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
    pub use crate::real_time_compilation::{
        CompilationContext, HardwareTarget, OptimizationPipeline, PerformanceMonitor,
        RealTimeQuantumCompiler,
    };
}

/// Developer tools and utilities
pub mod dev_tools {
    pub use crate::quantum_debugger::*;
    pub use crate::quantum_debugging_profiling::{
        CircuitAnalysisReport, ProfilingReport, QuantumCircuitAnalyzer, QuantumDebugProfiling,
        QuantumDebugProfilingReport, QuantumDebugger, QuantumErrorTracker,
        QuantumPerformanceProfiler, QuantumStateInspector, StateInspectionReport,
    };
    pub use crate::testing::{
        QuantumAssert, QuantumTest, QuantumTestSuite, TestResult, TestSuiteResult,
        DEFAULT_TOLERANCE,
    };
}

/// Error correction and noise models
pub mod error_correction {
    pub use crate::error_correction::{
        ColorCode, LookupDecoder, MWPMDecoder, Pauli, PauliString, StabilizerCode, SurfaceCode,
        SyndromeDecoder,
    };
}

/// Quantum machine learning primitives
pub mod quantum_ml {
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
    pub use crate::quantum_ml_accelerators::{
        HardwareEfficientMLLayer, ParameterShiftOptimizer, QuantumFeatureMap,
        QuantumKernelOptimizer, QuantumNaturalGradient, TensorNetworkMLAccelerator,
    };
}

/// Variational quantum algorithms
pub mod variational {
    pub use crate::qaoa::{
        CostHamiltonian, MixerHamiltonian, QAOACircuit, QAOAOptimizer, QAOAParams,
    };
    pub use crate::qpca::{DensityMatrixPCA, QPCAParams, QuantumPCA};
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
}

/// Hardware and device interfaces
pub mod hardware {
    pub use crate::neutral_atom::{
        AtomSpecies, AtomState, LaserSystem, NeutralAtom, NeutralAtomErrorModel, NeutralAtomGates,
        NeutralAtomQC, OpticalTweezer, Position3D,
    };
    pub use crate::photonic::{
        OpticalMode, PhotonicCircuit, PhotonicEncoding, PhotonicErrorCorrection, PhotonicGate,
        PhotonicGateType, PhotonicSystem,
    };
    pub use crate::pulse::{
        CouplingParams, HardwareCalibration, Pulse, PulseCompiler, PulseEnvelope, PulseNoiseModel,
        PulseSequence, QubitControlParams, TimingConstraints,
    };
    pub use crate::quantum_hardware_abstraction::{
        AdaptiveMiddleware, CalibrationEngine, ErrorMitigationLayer, ExecutionRequirements,
        HardwareCapabilities, HardwareResourceManager, HardwareType, QuantumHardwareAbstraction,
        QuantumHardwareBackend,
    };
    pub use crate::silicon_quantum_dots::{
        DeviceParams, QuantumDotParams, QuantumDotType, SiliconQuantumDot, SiliconQuantumDotGates,
        SiliconQuantumDotSystem,
    };
    pub use crate::trapped_ion::{
        IonLevel, IonSpecies, LaserPulse, MotionalMode, MotionalModeType, TrappedIon,
        TrappedIonGates, TrappedIonSystem,
    };
}

/// Advanced quantum algorithms
pub mod algorithms {
    pub use crate::adiabatic::{
        AdiabaticQuantumComputer, AnnealingSchedule, IsingProblem, ProblemGenerator, ProblemType,
        QUBOProblem, QuantumAnnealer, QuantumAnnealingSnapshot,
    };
    pub use crate::hhl::{hhl_example, HHLAlgorithm, HHLParams};
    pub use crate::quantum_counting::{
        amplitude_estimation_example, quantum_counting_example, QuantumAmplitudeEstimation,
        QuantumCounting, QuantumPhaseEstimation,
    };
    pub use crate::quantum_walk::{
        CoinOperator, ContinuousQuantumWalk, DecoherentQuantumWalk, DiscreteQuantumWalk, Graph,
        GraphType, MultiWalkerQuantumWalk, QuantumWalkSearch, SearchOracle, SzegedyQuantumWalk,
    };
}

/// Tensor networks and simulation methods
pub mod tensor_networks {
    pub use crate::memory_efficient::{EfficientStateVector, StateMemoryStats};
    pub use crate::tensor_network::{
        contraction_optimization::DynamicProgrammingOptimizer, Tensor, TensorEdge, TensorNetwork,
        TensorNetworkBuilder, TensorNetworkSimulator,
    };
}

/// Symbolic computation and mathematical modeling
pub mod symbolic {
    pub use crate::parametric::{Parameter, ParametricGate, SymbolicParameter};
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
}

/// ZX-calculus and graphical reasoning
pub mod zx_calculus {
    pub use crate::zx_calculus::{
        CircuitToZX, Edge, EdgeType, Spider, SpiderType, ZXDiagram, ZXOptimizer,
    };
    pub use crate::zx_extraction::{ZXExtractor, ZXPipeline};
}

/// Topological quantum computing
pub mod topological {
    pub use crate::holonomic::{
        // GeometricErrorCorrection, HolonomicGate, HolonomicGateSynthesis, HolonomicPath,
        // HolonomicQuantumComputer, PathOptimizationConfig,
        WilsonLoop,
    };
    pub use crate::topological::{
        AnyonModel, AnyonType, AnyonWorldline, BraidingOperation, FibonacciModel, FusionTree,
        IsingModel, TopologicalGate, TopologicalQC, ToricCode,
    };
}

/// Quantum networking and distributed computing
pub mod networking {
    pub use crate::distributed_quantum_networks::{
        DistributedGateType, DistributedQuantumGate, EntanglementManager, EntanglementProtocol,
        NetworkScheduler, QuantumNetwork, QuantumNode,
    };
    pub use crate::post_quantum_crypto::{
        CompressionFunction, QKDProtocol, QKDResult, QuantumDigitalSignature, QuantumHashFunction,
        QuantumKeyDistribution, QuantumSignature,
    };
    pub use crate::quantum_internet::{
        DistributedQuantumComputing, GlobalQuantumKeyDistribution, QuantumInternet,
        QuantumInternetAdvantageReport, QuantumInternetNode, QuantumInternetSecurity,
        QuantumNetworkInfrastructure, QuantumRouting,
    };
}

/// SciRS2 integration and enhanced tools
pub mod scirs2 {
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
}

/// Batch processing and parallel execution
pub mod batch {
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
}

/// Python bindings (when enabled)
#[cfg(feature = "python")]
pub mod python {
    pub use crate::jupyter_visualization::{
        PyQuantumCircuitVisualizer, PyQuantumPerformanceMonitor, PyQuantumStateVisualizer,
    };
    pub use crate::python_bindings::{
        PyAggregatedStats, PyAlert, PyMetricMeasurement, PyMonitoringConfig, PyMonitoringStatus,
        PyOptimizationRecommendation, PyRealtimeMonitor,
    };
    pub use crate::python_bindings::{
        PyCartanDecomposition, PyNumRS2Array, PyQuantumGate, PyQuantumInternet,
        PyQuantumSensorNetwork, PyQubitId, PySingleQubitDecomposition, PyVariationalCircuit,
    };
    pub use crate::quantum_complexity_analysis::PyQuantumComplexityAnalyzer;
}
