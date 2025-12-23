// Allow specific lints that are intentional design decisions or would require breaking changes
#![allow(clippy::too_many_arguments)] // Quantum operations naturally have many parameters
#![allow(clippy::module_inception)] // Module organization matches quantum circuit hierarchy
#![allow(clippy::large_enum_variant)] // Quantum state representations require large variants
#![allow(unexpected_cfgs)] // Allow cfg feature gates for future SciRS2 features

//! # QuantRS2-Circuit
//!
//! Quantum circuit representation and DSL for the `QuantRS2` framework.
//!
//! This crate provides a comprehensive toolkit for creating, manipulating,
//! and analyzing quantum circuits with a fluent API and modern Rust patterns.
//!
//! ## Recent Updates (v0.1.0-beta.3)
//!
//! - Enhanced code quality with clippy warning fixes
//! - Refined `SciRS2` v0.1.0-rc.2 integration with unified patterns
//! - Enhanced graph-based circuit optimization algorithms (60+ strategies)
//! - Improved hardware-aware compilation with `SciRS2` graph algorithms
//! - Comprehensive policy documentation for quantum circuit development
//! - Added refactoring recommendations for large modules
//!
//! ## Features
//!
//! - **50,055 lines** of production Rust code
//! - **237 test functions** providing comprehensive coverage
//! - **16 example programs** demonstrating key features
//! - Type-safe circuit construction with const generics
//! - Fluent builder API for intuitive circuit creation
//! - QASM 2.0 / 3.0 import and export
//! - Circuit optimization passes (60+ strategies)
//! - Hardware-aware transpilation for IBM, Google, Rigetti
//! - SciRS2-powered performance analysis and profiling
//! - Advanced debugging and verification tools
//!
//! ## Module Organization
//!
//! - **Core**: `builder`, `classical`, `measurement`
//! - **Optimization**: `optimization`, `optimizer`, `graph_optimizer`
//! - **Hardware**: `transpiler`, `routing`, `pulse`, `crosstalk`
//! - **Analysis**: `profiler`, `debugger`, `equivalence`, `validation`
//! - **Advanced**: `synthesis`, `zx_calculus`, `tensor_network`, `topological`
pub mod buffer_manager;
pub mod builder;
pub mod circuit_cache;
pub mod classical;
pub mod commutation;
pub mod crosstalk;
pub mod dag;
pub mod debugger;
pub mod distributed;
pub mod equivalence;
pub mod fault_tolerant;
pub mod formatter;
pub mod graph_optimizer;
pub mod linter;
pub mod measurement;
pub mod ml_optimization;
pub mod noise_models;
pub mod optimization;
pub mod optimizer;
pub mod photonic;
pub mod profiler;
pub mod pulse;
pub mod qasm;
pub mod qc_co_optimization;
pub mod resource_estimator;
pub mod routing;
pub mod scirs2_benchmarking;
pub mod scirs2_cross_compilation_enhanced;
pub mod scirs2_integration;
pub mod scirs2_ir_tools;
pub mod scirs2_matrices;
pub mod scirs2_optimization;
pub mod scirs2_pulse_control_enhanced;
pub mod scirs2_qasm_compiler_enhanced;
pub mod scirs2_similarity;
pub mod scirs2_transpiler_enhanced;
pub mod simulator_interface;
pub mod slicing;
pub mod synthesis;
pub mod tensor_network;
pub mod topological;
pub mod topology;
pub mod transpiler;
pub mod validation;
pub mod verifier;
pub mod vqe;
pub mod zx_calculus;

// Re-exports of commonly used types and traits
pub mod prelude {
    pub use crate::builder::{CircuitStats, *};
    // Convenience re-export
    pub use crate::circuit_cache::{
        CacheConfig, CacheEntry, CacheManager, CacheStats, CircuitCache, CircuitSignature,
        CompiledCircuitCache, EvictionPolicy, ExecutionResultCache, SignatureGenerator,
        TranspilationCache,
    };
    pub use crate::classical::{
        CircuitOp, ClassicalBit, ClassicalCircuit, ClassicalCircuitBuilder, ClassicalCondition,
        ClassicalOp, ClassicalRegister, ClassicalValue, ComparisonOp, ConditionalGate, MeasureOp,
    };
    pub use crate::commutation::{
        CommutationAnalyzer, CommutationOptimization, CommutationResult, CommutationRules, GateType,
    };
    pub use crate::crosstalk::{
        CrosstalkAnalysis, CrosstalkAnalyzer, CrosstalkModel, CrosstalkSchedule,
        CrosstalkScheduler, SchedulingStrategy, TimeSlice,
    };
    pub use crate::dag::{circuit_to_dag, CircuitDag, DagEdge, DagNode, EdgeType};
    pub use crate::debugger::{
        AnalysisDepth, BlochVector, BottleneckType as DebuggerBottleneckType, BreakpointAction,
        BreakpointCondition, BreakpointManager, ConditionalBreakpoint, ConnectionType,
        ConnectionVisualization, CorrelationType, DebugError, DebuggerConfig, Difficulty,
        ErrorAnalysisResults, ErrorCorrelation, ErrorDetectionConfig, ErrorDetector, ErrorPattern,
        ErrorSeverity, ErrorStatistics, ErrorType as DebuggerErrorType, ExecutionHistory,
        ExecutionState, ExecutionStatus as DebuggerExecutionStatus, ExecutionSummary,
        ExportFormat as DebuggerExportFormat, ExportOptions as DebuggerExportOptions,
        ExpressionResult, ExpressionType, ExpressionValue, GateAttributes, GateExecutionMetrics,
        GateExecutionResult, GateProperties as DebuggerGateProperties, GateSnapshot,
        GateType as DebuggerGateType, GateVisualization, HistoryEntry, HistoryStatistics,
        ImpactAssessment, MemorySnapshot, MemoryStatistics, MemoryUsage, MetricSnapshot,
        OptimizationSuggestion as DebuggerOptimizationSuggestion, PatternType, PerformanceAnalysis,
        PerformanceBottleneck, PerformanceProfiler, PerformanceSample, PredictionResult,
        Priority as DebuggerPriority, ProfilerConfig, ProfilingStatistics, QuantumDebugger,
        RenderingQuality, RenderingStatistics, RootCause, Solution, StateBreakpoint, StatePattern,
        StateSnapshot, StepResult, SuggestionType as DebuggerSuggestionType, TimingInfo,
        TimingStatistics, TrendAnalysis, TrendDirection, Visualization, VisualizationConfig,
        VisualizationData, VisualizationEngine, VisualizationMetadata, VisualizationSnapshot,
        VisualizationType, WatchConfig, WatchExpression, WatchManager, WatchedGate, WatchedMetric,
        WatchedState,
    };
    pub use crate::distributed::{
        BackendType, DistributedExecutor, DistributedJob, DistributedResult, ExecutionBackend,
        ExecutionParameters, ExecutionStatus, LoadBalancingStrategy, Priority, SystemHealthStatus,
    };
    pub use crate::equivalence::{
        circuits_equivalent, circuits_structurally_equal, EquivalenceChecker, EquivalenceOptions,
        EquivalenceResult, EquivalenceType,
    };
    pub use crate::fault_tolerant::{
        FaultTolerantCircuit, FaultTolerantCompiler, LogicalQubit, MagicState, QECCode,
        ResourceOverhead, SyndromeMeasurement, SyndromeType,
    };
    pub use crate::formatter::{
        AlignmentConfig, AlignmentEngine, AlignmentGroup, AlignmentItem, AlignmentOptimization,
        AlignmentState, AlignmentStatistics, AutoCorrectionConfig, ChangeType, CodeOrganizer,
        CodeSection, CodeStructure, CommentAlignment, CommentConfig, CommentFormatter,
        CommentFormatterState, CommentRule, ComplianceLevel as FormatterComplianceLevel,
        ConsistencyMetrics, FormattedCircuit, FormatterConfig, FormattingChange, FormattingResult,
        FormattingStatistics, GroupingStrategy, IndentationConfig, IndentationStyle,
        LayoutOptimizer, OptimizationConfig as FormatterOptimizationConfig, OrganizationConfig,
        Position, QualityMetrics, QuantumFormatter, SciRS2AnalysisConfig,
        SeverityLevel as FormatterSeverityLevel, SpacingConfig, SpacingStyle, StyleCompliance,
        StyleEnforcementConfig, StyleEnforcer, StyleInformation, StyleIssue, WhitespaceManager,
        WhitespaceOptimization, WhitespaceState, WhitespaceStatistics,
    };
    pub use crate::graph_optimizer::{CircuitDAG, GraphGate, GraphOptimizer, OptimizationStats};
    pub use crate::linter::{
        AnalysisScope, AntiPatternDetectionResult, AntiPatternDetector, AutoFix, AutoFixType,
        BarrierUsageStyle, BestPracticeResult, BestPracticeRule, BestPracticeViolation,
        BestPracticesChecker, BestPracticesCompliance, CircuitLocation, ComplexityAnalysisResult,
        ComplexityAnalyzer, ComplexityClassification, ComplexityMetric,
        ComplexityMetrics as LinterComplexityMetrics, ComplexityTrend, ComplianceLevel,
        ConnectivityPattern, ConstraintType as LinterConstraintType, CustomGuideline,
        Difficulty as LinterDifficulty, GateGroupingStyle, Importance,
        IndentationStyle as LinterIndentationStyle, InteractionType, IssueType, LintIssue,
        LinterConfig, LintingMetadata, LintingResult, LintingStatistics, MeasurementPlacementStyle,
        NamingConvention, OptimizationAnalysisResult, OptimizationAnalyzer,
        OptimizationImprovement as LinterOptimizationImprovement, OptimizationRule,
        OptimizationSuggestion as LinterOptimizationSuggestion, OptimizationType,
        ParameterConstraint, PatternAnalysisResult, PatternDetectionResult, PatternDetector,
        PatternFlexibility, PatternInteraction, PatternMatcher, PatternPerformanceProfile,
        PatternStatistics, PerformanceImpact, PerformanceMetrics as LinterPerformanceMetrics,
        PerformanceProjection, PracticeGuidelines, QuantumAntiPattern, QuantumLinter,
        QuantumPattern, QubitOrderingStyle, Risk, SafetyLevel,
        ScalingBehavior as LinterScalingBehavior, Severity, SimplificationSuggestion,
        SimplificationType, StyleAnalysisResult, StyleCheckResult, StyleChecker, StyleConfig,
        StyleRule, StyleStrictness, StyleViolation, TrendDirection as LinterTrendDirection,
    };
    pub use crate::measurement::{
        CircuitOp as MeasurementCircuitOp, FeedForward, Measurement, MeasurementCircuit,
        MeasurementCircuitBuilder, MeasurementDependencies,
    };
    pub use crate::ml_optimization::{
        AcquisitionFunction, FeatureExtractor, ImprovementMetrics, MLCircuitOptimizer,
        MLCircuitRepresentation, MLOptimizationResult, MLStrategy, TrainingExample,
    };
    pub use crate::noise_models::{
        DecoherenceParams, ErrorSource, LeakageError, NoiseAnalysisResult, NoiseAnalyzer,
        ReadoutError, SingleQubitError, ThermalNoise, TwoQubitError,
    };
    pub use crate::optimization::{
        AbstractCostModel, CircuitAnalyzer, CircuitOptimizer2, CircuitRewriting,
        CoherenceOptimization, CommutationTable, CostBasedOptimization, CostModel,
        DecompositionOptimization, DecouplingSequence, DynamicalDecoupling, GateCancellation,
        GateCommutation, GateCost, GateError, GateMerging, GateProperties, HardwareCostModel,
        NoiseAwareCostModel, NoiseAwareMapping, NoiseAwareOptimizer, NoiseModel, OptimizationLevel,
        OptimizationPass, OptimizationReport, PassConfig, PassManager, RotationMerging,
        TemplateMatching, TwoQubitOptimization,
    };
    pub use crate::optimizer::{
        CircuitOptimizer, HardwareOptimizer, OptimizationPassType, OptimizationResult,
        RedundantGateElimination, SingleQubitGateFusion,
    };
    pub use crate::photonic::{
        CVCircuit, CVGate, CVMeasurement, PhotonicCircuit, PhotonicCircuitBuilder,
        PhotonicConverter, PhotonicGate, PhotonicMeasurement, PhotonicMode, Polarization,
        PolarizationBasis,
    };
    pub use crate::profiler::{
        AccuracyMeasurement, AccuracyTracking, AggregationFunction, AggregationRule, Alert,
        AlertChannel, AlertChannelType, AlertCondition, AlertLevel, AlertRule, AlertSystem,
        AllocationEvent, AllocationEventType, AllocationInfo, AllocationStatistics,
        AllocationTracker, AllocationType, AnalysisConfig, AnalysisDepth as ProfilerAnalysisDepth,
        AnalyticsConfig, AnomalyAlgorithmType, AnomalyDetectionAlgorithm, AnomalyDetectionConfig,
        AnomalyDetector, AnomalyType, AnomySeverity, ArchivalPolicy, BackupConfig, BaselineManager,
        BaselineUpdatePolicy, BaselineValidationResults,
        BenchmarkConfig as ProfilerBenchmarkConfig, BenchmarkEngine, BenchmarkResult,
        BenchmarkSuite, BenchmarkSuiteConfig, BenchmarkTest, BenchmarkTestType,
        BottleneckAnalysis as ProfilerBottleneckAnalysis, BottleneckImpactAnalysis,
        BottleneckSeverity, CacheMissRates, CascadingEffect, ChecksumAlgorithm, ComparisonOperator,
        ComparisonResults, ComparisonSummary, CompressionAlgorithm, CompressionSettings,
        ConnectionStatistics, CostBenefitAnalysis, CpuOptimization, CpuOptimizationType,
        CpuProfilingData, CpuState, DataRetentionPolicy, DetailedAnalysis, EnvironmentInfo,
        ErrorAnalysis, ErrorCharacteristics, ErrorDistribution,
        ErrorPattern as ProfilerErrorPattern, ErrorSeverity as ProfilerErrorSeverity,
        ExportFormat as ProfilerExportFormat, GateProfile, GateProfiler, GpuInfo, GpuOptimization,
        GpuOptimizationType, GpuProfilingData, HardwareConfig, HistoricalPerformanceData,
        InsightType, IntegrityChecks, IoOptimization, IoOptimizationType, IoProfilingData, IoState,
        LatencyDistribution, LeakAnalysisResults, LeakDetector, LeakSeverity, LogicalOperator,
        MemoryLeak, MemoryOptimization as ProfilerMemoryOptimization, MemoryOptimizationType,
        MemoryPattern, MemoryProfiler, MemorySnapshot as ProfilerMemorySnapshot, MemoryState,
        MemoryTransferTimes, MetricCategory, MetricStream, MetricsCollector,
        MitigationStrategy as ProfilerMitigationStrategy, MitigationStrategyType, MlModel,
        NetworkOptimization, NetworkOptimizationType, NetworkProfilingData, NetworkState,
        PerformanceAnalyzer, PerformanceAnomaly, PerformanceBaseline,
        PerformanceInsight as ProfilerPerformanceInsight, PerformanceMetric, PerformanceRegression,
        PerformanceSnapshot, PerformanceSummary, PrecisionLevel, PredictionConfig,
        PredictionEngine, PredictionModel as ProfilerPredictionModel, PredictionModelType,
        PredictionResult as ProfilerPredictionResult, ProfilerConfig as ProfilerConfiguration,
        ProfilingReport, ProfilingSession, QuantumProfiler, RealtimeMetrics, RecoveryStatistics,
        RegressionAlgorithmType, RegressionAnalysisResults, RegressionDetectionAlgorithm,
        RegressionDetectionConfig, RegressionDetector, RegressionSeverity, ResourceBottleneck,
        ResourceBottleneckType, ResourceProfiler,
        ResourceRequirements as ProfilerResourceRequirements, ResourceUtilization,
        SerializationConfig, SerializationFormat, SessionAnalytics, SessionConfig, SessionData,
        SessionManager, SessionStatistics, SessionStatus, SessionStorage, SessionTrendAnalysis,
        SeverityLevel, StatisticalMethod, StorageBackend, StorageConfig, StorageInfo, StorageType,
        StreamStatistics, SuppressionCondition, SuppressionRule, SystemState, TestResult,
        ThroughputStatistics, TimingStatistics as ProfilerTimingStatistics, TrainingStatus,
        TrendAnalysisResults, ValidationStatus,
    };
    pub use crate::pulse::{
        Channel, DeviceConfig, PulseCalibration, PulseCompiler, PulseInstruction, PulseOptimizer,
        PulseSchedule, Waveform,
    };
    pub use crate::qasm::exporter::ExportError;
    pub use crate::qasm::{
        export_qasm3, parse_qasm3, validate_qasm3, ExportOptions, ParseError, QasmExporter,
        QasmGate, QasmParser, QasmProgram, QasmRegister, QasmStatement, ValidationError,
    };
    pub use crate::qc_co_optimization::{
        ClassicalProcessingStep, ClassicalStepType, DataFlowGraph, DataType,
        HybridOptimizationAlgorithm, HybridOptimizationProblem, HybridOptimizationResult,
        HybridOptimizer, LearningRateSchedule, ObjectiveFunction as HybridObjectiveFunction,
        ObjectiveFunctionType, ParameterizedQuantumComponent, RegularizationType,
    };
    pub use crate::resource_estimator::{
        AlgorithmClass, CircuitMetrics as ResourceCircuitMetrics, ComplexityAnalysis,
        ComplexityClass, ExecutionTimeEstimate, HardwareRequirements, MemoryRequirements,
        OptimizationSuggestion as ResourceOptimizationSuggestion, PlatformRecommendation,
        ResourceEstimate, ScalabilityAnalysis, ScalingBehavior, ScalingFunction,
    };
    pub use crate::routing::{
        CircuitRouter, CouplingMap, Distance, LookaheadConfig, LookaheadRouter, RoutedCircuit,
        RoutingPassType, RoutingResult, RoutingStatistics, RoutingStrategy, SabreConfig,
        SabreRouter, SwapLayer, SwapNetwork,
    };
    pub use crate::scirs2_benchmarking::{
        BaselineComparison, BenchmarkConfig, BenchmarkReport, BenchmarkRun, CircuitBenchmark,
        CircuitMetrics as BenchmarkCircuitMetrics, DescriptiveStats, Distribution, DistributionFit,
        HypothesisTestResult, InsightCategory, OutlierAnalysis, OutlierDetectionMethod,
        PerformanceInsight, PracticalSignificance, RegressionAnalysis, StatisticalAnalyzer,
        StatisticalTest,
    };
    pub use crate::scirs2_cross_compilation_enhanced::{
        AppliedOptimization, BatchCompilationReport, BatchCompilationResult, BatchPerformanceStats,
        CircuitMetrics, CircuitSize, CircuitVisualization, ClassicalOpType, ClassicalOperation,
        CodeFormat, ComparisonVisualization, CompilationComplexity, CompilationRecommendation,
        CompilationReport, CompilationStage, CompilationStrategy, CompilationSummary,
        CrossCompilationConfig, CrossCompilationResult, DataFlow, Difference, DifferenceType,
        EnhancedCrossCompilationConfig, EnhancedCrossCompiler, FailedCompilation, FlowEdge,
        FlowNode, GraphLayout, IRClassicalOp, IRClassicalOpType, IREdge, IRGate, IRGraph, IRNode,
        IROperation, IROperationType, IRVisualization, NodeType, OperationType, OptimizationImpact,
        OptimizationImprovement, OptimizationTimeline, OptimizationVisualization, ParsedCircuit,
        QuantumFramework, QuantumIR, QuantumOperation,
        RecommendationCategory as CompilationRecommendationCategory, ResourceUsage, SourceCircuit,
        StageAnalysis, StageImpact, StagePerformance, TargetCode, TargetPlatform, TimelineEvent,
        ValidationError as CrossValidationError, ValidationErrorType,
        ValidationResult as CrossValidationResult, ValidationWarning as CrossValidationWarning,
        ValidationWarningType, VisualCompilationFlow,
    };
    pub use crate::scirs2_integration::{
        AnalysisResult, AnalyzerConfig, GraphMetrics, GraphMotif, OptimizationSuggestion,
        SciRS2CircuitAnalyzer, SciRS2CircuitGraph, SciRS2Edge, SciRS2Node, SciRS2NodeType,
    };
    pub use crate::scirs2_matrices::{
        CircuitToSparseMatrix, Complex64, SparseFormat, SparseGate, SparseGateLibrary,
        SparseMatrix, SparseOptimizer,
    };
    pub use crate::scirs2_optimization::{
        CircuitTemplate, EarlyStoppingCriteria, KernelType, ObjectiveFunction,
        OptimizationAlgorithm, OptimizationConfig, OptimizationHistory, Parameter,
        ParameterizedGate, QAOAObjective, QuantumCircuitOptimizer, VQEObjective,
    };
    pub use crate::scirs2_pulse_control_enhanced::{
        AWGSpecifications, AmplitudeNoiseSpec, CosinePulse, CustomPulseShape, DRAGPulse,
        EnhancedPulseConfig, EnhancedPulseController, ErfPulse, FilterState, FilterType,
        GateAnalysis, GaussianPulse, HardwareConstraints, IQMixerSpecifications,
        MitigationStrategy, OptimizationFeedback, OptimizationStep, PhaseNoiseSpec,
        PredistortionModel, PulseChannel, PulseConstraints, PulseControlConfig, PulseExportFormat,
        PulseLibrary, PulseMetadata, PulseOptimizationModel, PulseOptimizationObjective,
        PulseSequence, SechPulse, SignalProcessingConfig, SignalProcessor, SignalProcessorConfig,
        Waveform as EnhancedWaveform, WindowType,
    };
    pub use crate::scirs2_qasm_compiler_enhanced::{
        ASTNode, ASTStatistics, AnalysisOptions, BinaryOp, CompilationResult,
        CompilationStatistics, CompilationTarget, CompilationVisualizations, CompilationWarning,
        EnhancedQASMCompiler, EnhancedQASMConfig, ErrorType, ExportFormat, Expression,
        GateDefinition, GeneratedCode, Location, OptimizedQASM, ParsedQASM, QASMCompilerConfig,
        QASMVersion, Token, TokenType, TypeCheckingLevel, UnaryOp, ValidationResult,
        ValidationWarning, WarningType, AST,
    };
    pub use crate::scirs2_similarity::{
        BatchSimilarityComputer, CircuitDistanceMetrics, CircuitFeatures,
        CircuitSimilarityAnalyzer, CircuitSimilarityMetrics, EntanglementStructure,
        GraphKernelType, GraphSimilarityAlgorithm, MLModelType, SciRS2Graph, SimilarityAlgorithm,
        SimilarityConfig, SimilarityWeights,
    };
    pub use crate::scirs2_transpiler_enhanced::{
        AdvancedHardwareFeatures, Bottleneck, BottleneckType, CircuitAnalysis,
        CircuitFeatures as EnhancedCircuitFeatures, CompatibilityReport, DecomposedGate,
        DecompositionResult, DecompositionStrategy, EnhancedTranspiler, EnhancedTranspilerConfig,
        ErrorMitigationSupport, GateDecomposition, GateStatistics, HardwareBackend, ImpactLevel,
        MitigationResult, NativeGateSet as EnhancedNativeGateSet, ParallelismAnalysis, PassResult,
        PerformanceConstraints, PerformanceMetrics, PerformancePrediction, PredictionModel,
        RoutingFeedback, RoutingModel, SuggestionType, SwapGate, TopologyAnalysis, TopologyType,
        TranspilationPass, TranspilationResult as EnhancedTranspilationResult,
        VisualRepresentation,
    };
    pub use crate::simulator_interface::{
        CircuitCompiler, CompiledCircuit, ContractionStrategy, ExecutionResult, InstructionSet,
        MemoryOptimization, OptimizationLevel as SimulatorOptimizationLevel, ResourceRequirements,
        SimulatorBackend, SimulatorExecutor,
    };
    pub use crate::slicing::{CircuitSlice, CircuitSlicer, SlicingResult, SlicingStrategy};
    pub use crate::synthesis::{
        GateSet, MultiQubitSynthesizer, SingleQubitSynthesizer, SynthesisConfig,
        TwoQubitSynthesizer, UnitaryOperation, UnitarySynthesizer,
    };
    pub use crate::tensor_network::{
        CircuitToTensorNetwork, CompressedCircuit, CompressionMethod, MatrixProductState, Tensor,
        TensorNetwork, TensorNetworkCompressor,
    };
    pub use crate::topological::{
        Anyon, AnyonModel, AnyonType, BraidingOperation, BraidingOptimizer, OptimizationStrategy,
        TopologicalCircuit, TopologicalCompiler, TopologicalGate,
    };
    pub use crate::topology::{TopologicalAnalysis, TopologicalAnalyzer, TopologicalStrategy};
    pub use crate::transpiler::{
        DeviceTranspiler, HardwareSpec, NativeGateSet, TranspilationOptions, TranspilationResult,
        TranspilationStats, TranspilationStrategy,
    };
    pub use crate::validation::{
        CircuitValidator, ClassicalConstraints, ConnectivityConstraints, DepthLimits,
        GateRestrictions, MeasurementConstraints, ResourceLimits, ValidationRules, ValidationStats,
        ValidationSuggestion,
    };
    pub use crate::verifier::{
        BinaryOperator, CircuitInvariant, CircuitLocation as VerifierCircuitLocation,
        ComplexityClass as VerifierComplexityClass, ConfidenceStatistics,
        ConstraintSatisfactionResult, ConstraintType, CorrectnessChecker, CorrectnessCriterion,
        CorrectnessResult, Counterexample, CustomInvariantChecker, CustomPredicate,
        EntanglementType, ErrorBounds, ErrorModel, EvidenceType, ExecutionTrace, ExpectedOutput,
        FormalProof, InvariantCheckResult, InvariantChecker, IssueSeverity,
        IssueType as VerifierIssueType, ModelCheckResult, ModelChecker, NumericalEvidence,
        ProofComplexityMetrics, ProofNode, ProofObligation, ProofStatus, ProofStep, ProofStrategy,
        ProofTree, PropertyChecker, PropertyVerificationResult, QuantumProperty, QuantumState,
        QuantumTheorem, QuantumVerifier, StateSpace, StateSpaceStatistics, StateTransition,
        SuperpositionType, SymbolicConstraint, SymbolicExecutionConfig, SymbolicExecutionResult,
        SymbolicExecutionStatus, SymbolicExecutor, SymbolicExpression, SymbolicState, SymbolicType,
        SymbolicVariable, TemporalProperty, TestCase, TestOutcome, TheoremProver, TheoremResult,
        UnaryOperator, VerificationIssue, VerificationMetadata, VerificationOutcome,
        VerificationResult, VerificationStatistics, VerificationStatus, VerifierConfig,
        VerifierTestResult, ViolationSeverity,
    };
    pub use crate::vqe::{
        PauliOperator, VQEAnsatz, VQECircuit, VQEObservable, VQEOptimizer, VQEOptimizerType,
        VQEResult,
    };
    pub use crate::zx_calculus::{
        OptimizedZXResult, ZXDiagram, ZXEdge, ZXNode, ZXOptimizationResult, ZXOptimizer,
    };
    pub use quantrs2_core::qubit::QubitId as Qubit;
}

// The following should be proc macros, but we'll implement them later
// for now they're just stubs

/// Creates a qubit set for quantum operations
///
/// # Example
///
/// ```
/// use quantrs2_circuit::qubits;
/// let qs = qubits![0, 1, 2];
/// ```
#[macro_export]
macro_rules! qubits {
    ($($id:expr),* $(,)?) => {
        {
            use quantrs2_core::qubit::QubitSet;

            let mut qs = QubitSet::new();
            $(qs.add($id);)*
            qs
        }
    };
}

/// Constructs a quantum circuit with a fixed number of qubits
///
/// # Example
///
/// ```
/// use quantrs2_circuit::circuit;
/// let circuit = circuit![4];
/// ```
#[macro_export]
macro_rules! circuit {
    ($n:expr) => {
        quantrs2_circuit::builder::Circuit::<$n>::new()
    };
}

/// Provides a DSL for constructing quantum circuits
///
/// # Example
///
/// ```
/// use quantrs2_circuit::quantum;
///
/// let my_circuit = quantum! {
///     let qc = circuit(4);  // 4 qubits
///     qc.h(0);
///     qc.cnot(0, 1);
///     qc.measure_all();
/// };
/// ```
#[macro_export]
macro_rules! quantum {
    (
        let $var:ident = circuit($n:expr);
        $( $stmt_var:ident . $method:ident ( $( $args:expr ),* $(,)? ) ; )*
    ) => {
        {
            let mut $var = quantrs2_circuit::builder::Circuit::<$n>::new();
            $(
                $stmt_var.$method($($args),*).unwrap();
            )*
            $var
        }
    };
}
