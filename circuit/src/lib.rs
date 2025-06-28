#![allow(dead_code)]
#![allow(clippy::all)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unexpected_cfgs)]
#![allow(deprecated)]

extern crate proc_macro;

/// Quantum circuit representation and DSL for the QuantRS2 framework.
///
/// This crate provides types for constructing and manipulating
/// quantum circuits with a fluent API.
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
pub mod graph_optimizer;
pub mod measurement;
pub mod ml_optimization;
pub mod noise_models;
pub mod optimization;
pub mod optimizer;
pub mod photonic;
pub mod profiler;
pub mod pulse;
pub mod resource_estimator;
pub mod scirs2_pulse_control_enhanced;
pub mod qasm;
pub mod scirs2_qasm_compiler_enhanced;
pub mod scirs2_cross_compilation_enhanced;
pub mod qc_co_optimization;
pub mod routing;
pub mod scirs2_benchmarking;
pub mod scirs2_integration;
pub mod scirs2_matrices;
pub mod scirs2_optimization;
pub mod scirs2_similarity;
pub mod simulator_interface;
pub mod slicing;
pub mod synthesis;
pub mod tensor_network;
pub mod topological;
pub mod topology;
pub mod transpiler;
pub mod scirs2_transpiler_enhanced;
pub mod formatter;
pub mod linter;
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
        QuantumDebugger, DebuggerConfig, ExecutionState, ExecutionStatus as DebuggerExecutionStatus, BreakpointManager,
        BreakpointCondition, StateBreakpoint, ConditionalBreakpoint, StatePattern, BreakpointAction,
        WatchManager, WatchedState, WatchedGate, WatchedMetric, WatchExpression, ExpressionType,
        ExpressionResult, ExpressionValue, WatchConfig, ExecutionHistory, HistoryEntry,
        HistoryStatistics, MemoryUsage, MemoryStatistics, TimingInfo, TimingStatistics,
        PerformanceProfiler, ProfilerConfig, AnalysisDepth, PerformanceSample, PerformanceAnalysis,
        TrendAnalysis, TrendDirection, PerformanceBottleneck, BottleneckType as DebuggerBottleneckType, ImpactAssessment,
        OptimizationSuggestion as DebuggerOptimizationSuggestion, SuggestionType as DebuggerSuggestionType,
        Priority as DebuggerPriority, Difficulty, PredictionResult, ProfilingStatistics, VisualizationEngine,
        VisualizationConfig, VisualizationType, RenderingQuality, ExportOptions as DebuggerExportOptions, 
        ExportFormat as DebuggerExportFormat, Visualization, VisualizationData, GateVisualization, 
        ConnectionVisualization, GateType as DebuggerGateType, GateAttributes, ConnectionType, BlochVector, 
        VisualizationMetadata, VisualizationSnapshot, RenderingStatistics, ErrorDetector, ErrorDetectionConfig, 
        ErrorType as DebuggerErrorType, DebugError, ErrorSeverity, ErrorStatistics, ErrorAnalysisResults, 
        ErrorPattern, PatternType, RootCause, Solution, ErrorCorrelation, CorrelationType, StepResult, 
        GateExecutionResult, ExecutionSummary, GateProperties as DebuggerGateProperties, GateExecutionMetrics, 
        MetricSnapshot, StateSnapshot, GateSnapshot, MemorySnapshot,
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
        QuantumFormatter, FormatterConfig, IndentationConfig, IndentationStyle, SpacingConfig,
        SpacingStyle, AlignmentConfig, CommentConfig, CommentAlignment, OrganizationConfig,
        GroupingStrategy, OptimizationConfig as FormatterOptimizationConfig, OptimizationLevel as FormatterOptimizationLevel, StyleEnforcementConfig,
        StyleStrictness as FormatterStyleStrictness, CustomStyleRule, RulePriority, 
        SciRS2AnalysisConfig, AutoCorrectionConfig, FormattingResult, FormattedCircuit,
        CodeStructure, CodeSection, SectionType, ImportStatement, ImportType, VariableDeclaration,
        FunctionDefinition, Parameter as FormatterParameter, CircuitDefinition, GateOperation, MeasurementOperation,
        LayoutInformation, AlignmentColumn, ColumnType, AlignedElement, WrappingPoint, WrappingType,
        StyleInformation, AppliedStyleRule, StyleViolationFix, ConsistencyMetrics, FormattingStatistics,
        FormattingChange, ChangeType, StyleCompliance, ComplianceLevel as FormatterComplianceLevel,
        OptimizationResults, LayoutOptimization, ReadabilityImprovement, PerformanceOptimization,
        FormattingWarning, WarningType as FormatterWarningType, WarningSeverity, FormattingMetadata, SciRS2FormattingAnalysis,
        GraphAnalysisResults, PatternAnalysisResults, DetectedPattern, PatternFormattingSuggestion,
        DependencyAnalysisResults, GateDependency, DependencyType, DataFlowEdge, OrderingConstraint,
        ParallelizationOpportunity, SciRS2OptimizationSuggestion, LayoutSuggestion, InputStatistics,
    };
    pub use crate::graph_optimizer::{CircuitDAG, GraphGate, GraphOptimizer, OptimizationStats};
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
        QuantumProfiler, ProfilerConfig as ProfilerConfiguration, PrecisionLevel, MetricsCollector, PerformanceMetric,
        MetricCategory, AggregationRule, AggregationFunction, MetricStream, StreamStatistics,
        GateProfiler, GateProfile, MemoryPattern, ErrorCharacteristics, ErrorDistribution,
        TimingStatistics as ProfilerTimingStatistics, ErrorAnalysis, ErrorSeverity as ProfilerErrorSeverity, 
        ErrorPattern as ProfilerErrorPattern, RecoveryStatistics, MemoryProfiler, 
        MemorySnapshot as ProfilerMemorySnapshot, LeakDetector, MemoryLeak,
        LeakAnalysisResults, LeakSeverity, MemoryOptimization as ProfilerMemoryOptimization, MemoryOptimizationType,
        AllocationTracker, AllocationInfo, AllocationType, AllocationEvent, AllocationEventType,
        AllocationStatistics, ResourceProfiler, CpuProfilingData, CacheMissRates, CpuOptimization,
        CpuOptimizationType, GpuProfilingData, MemoryTransferTimes, GpuOptimization, GpuOptimizationType,
        IoProfilingData, LatencyDistribution, IoOptimization, IoOptimizationType, NetworkProfilingData,
        ConnectionStatistics, ThroughputStatistics, NetworkOptimization, NetworkOptimizationType,
        BottleneckAnalysis as ProfilerBottleneckAnalysis, ResourceBottleneck, ResourceBottleneckType,
        BottleneckSeverity, SeverityLevel, BottleneckImpactAnalysis, CascadingEffect, CostBenefitAnalysis,
        MitigationStrategy as ProfilerMitigationStrategy, MitigationStrategyType, ResourceRequirements as ProfilerResourceRequirements, PerformanceAnalyzer,
        AnalysisConfig, AnalysisDepth as ProfilerAnalysisDepth, StatisticalMethod, MlModel,
        HistoricalPerformanceData, PerformanceSnapshot, SystemState, CpuState, MemoryState,
        IoState, NetworkState, EnvironmentInfo, HardwareConfig, GpuInfo, StorageInfo, StorageType,
        DataRetentionPolicy, ArchivalPolicy, CompressionSettings, CompressionAlgorithm,
        IntegrityChecks, ChecksumAlgorithm, AnomalyDetector, AnomalyDetectionAlgorithm,
        AnomalyAlgorithmType, PerformanceAnomaly, AnomalyType, AnomySeverity, AnomalyDetectionConfig,
        AlertSystem, AlertChannel, AlertChannelType, Alert, AlertLevel, AlertRule, AlertCondition,
        ComparisonOperator, LogicalOperator, SuppressionRule, SuppressionCondition, PredictionEngine,
        PredictionModel as ProfilerPredictionModel, PredictionModelType, PredictionResult as ProfilerPredictionResult,
        TrainingStatus, PredictionConfig, AccuracyTracking, AccuracyMeasurement, BenchmarkEngine,
        BenchmarkSuite, BenchmarkTest, BenchmarkTestType, BenchmarkSuiteConfig, BenchmarkResult,
        TestResult, ComparisonResults, ComparisonSummary, RegressionAnalysisResults,
        PerformanceRegression, RegressionSeverity, TrendAnalysisResults, BenchmarkConfig as ProfilerBenchmarkConfig,
        RegressionDetector, RegressionDetectionAlgorithm, RegressionAlgorithmType,
        RegressionDetectionConfig, BaselineManager, PerformanceBaseline, BaselineUpdatePolicy,
        BaselineValidationResults, ValidationStatus, SessionManager, ProfilingSession,
        SessionStatus, SessionData, SessionConfig, SessionStorage, StorageBackend, StorageConfig,
        BackupConfig, SerializationConfig, SerializationFormat, SessionAnalytics, AnalyticsConfig,
        SessionStatistics, PerformanceInsight as ProfilerPerformanceInsight, InsightType, SessionTrendAnalysis, ExportFormat as ProfilerExportFormat,
        RealtimeMetrics, ResourceUtilization, ProfilingReport, PerformanceSummary, DetailedAnalysis,
    };
    pub use crate::pulse::{
        Channel, DeviceConfig, PulseCalibration, PulseCompiler, PulseInstruction, PulseOptimizer,
        PulseSchedule, Waveform,
    };
    pub use crate::resource_estimator::{
        ResourceEstimate, CircuitMetrics as ResourceCircuitMetrics, ComplexityAnalysis, ComplexityClass, MemoryRequirements,
        ExecutionTimeEstimate, HardwareRequirements, ScalabilityAnalysis, OptimizationSuggestion as ResourceOptimizationSuggestion,
        AlgorithmClass, ScalingBehavior, PlatformRecommendation,
        ScalingFunction,
    };
    pub use crate::scirs2_pulse_control_enhanced::{
        EnhancedPulseController, EnhancedPulseConfig, PulseControlConfig, HardwareConstraints,
        AWGSpecifications, IQMixerSpecifications, PhaseNoiseSpec, AmplitudeNoiseSpec,
        PulseLibrary, GaussianPulse, DRAGPulse, CosinePulse, ErfPulse, SechPulse,
        CustomPulseShape, PulseOptimizationObjective, PulseConstraints, SignalProcessingConfig,
        FilterType, WindowType, PulseExportFormat, PulseSequence, PulseChannel,
        Waveform as EnhancedWaveform, PulseMetadata, GateAnalysis, GateType as PulseGateType,
        ControlRequirements, ControlType, PerformanceTargets, PulseShape, CalibrationData,
        CalibrationMeasurement, ErrorMetrics, EnvironmentalData, CalibrationResult,
        CalibrationAnalysis, QualityMetrics, DriftAnalysis, CalibrationParameters,
        ParameterUpdate, PulseVisualization, ChannelPlot, FrequencyPlot, PhasePlot,
        MitigationStrategy, PulseOptimizationModel, OptimizationFeedback,
    };
    pub use crate::qasm::exporter::ExportError;
    pub use crate::qasm::{
        export_qasm3, parse_qasm3, validate_qasm3, ExportOptions, ParseError, QasmExporter,
        QasmGate, QasmParser, QasmProgram, QasmRegister, QasmStatement, ValidationError,
    };
    pub use crate::scirs2_qasm_compiler_enhanced::{
        EnhancedQASMCompiler, EnhancedQASMConfig, QASMCompilerConfig, QASMVersion,
        CompilationTarget, AnalysisOptions, TypeCheckingLevel,
        ExportFormat, GateDefinition, CompilationResult,
        GeneratedCode, CompilationVisualizations, CompilationStatistics,
        CompilationWarning, WarningType, ParsedQASM, ValidationResult,
        ErrorType, ValidationWarning, OptimizedQASM, ASTStatistics,
        Token, TokenType, AST, ASTNode, Expression, BinaryOp, UnaryOp, Location,
    };
    pub use crate::scirs2_cross_compilation_enhanced::{
        EnhancedCrossCompiler, EnhancedCrossCompilationConfig, CrossCompilationConfig,
        QuantumFramework, TargetPlatform, CompilationStrategy,
        SourceCircuit, CrossCompilationResult, CompilationStage, ParsedCircuit,
        QuantumOperation, OperationType, ClassicalOperation, ClassicalOpType,
        QuantumIR, IROperation, IROperationType, IRGate, IRClassicalOp,
        IRClassicalOpType, TargetCode, CodeFormat, ValidationResult as CrossValidationResult,
        ValidationError as CrossValidationError, ValidationErrorType, ValidationWarning as CrossValidationWarning,
        ValidationWarningType, CompilationReport, CompilationSummary, CircuitSize,
        StageAnalysis, StagePerformance, StageImpact,
        AppliedOptimization, OptimizationImpact, OptimizationImprovement,
        ResourceUsage, CompilationComplexity, CompilationRecommendation,
        RecommendationCategory as CompilationRecommendationCategory,
        VisualCompilationFlow, FlowNode, NodeType, FlowEdge,
        DataFlow, IRVisualization, IRGraph, IRNode, IREdge, GraphLayout,
        OptimizationVisualization, ComparisonVisualization, CircuitVisualization,
        CircuitMetrics, Difference, DifferenceType, OptimizationTimeline,
        TimelineEvent, BatchCompilationResult, FailedCompilation,
        BatchCompilationReport, BatchPerformanceStats,
    };
    pub use crate::qc_co_optimization::{
        ClassicalProcessingStep, ClassicalStepType, DataFlowGraph, DataType,
        HybridOptimizationAlgorithm, HybridOptimizationProblem, HybridOptimizationResult,
        HybridOptimizer, LearningRateSchedule, ObjectiveFunction as HybridObjectiveFunction,
        ObjectiveFunctionType, ParameterizedQuantumComponent, RegularizationType,
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
    pub use crate::scirs2_integration::{
        AnalysisResult, AnalyzerConfig, GraphMetrics, GraphMotif, OptimizationSuggestion,
        SciRS2CircuitAnalyzer, SciRS2CircuitGraph, SciRS2Edge, SciRS2Node, SciRS2NodeType,
    };
    pub use crate::scirs2_matrices::{
        CircuitToSparseMatrix, SparseFormat, SparseGate, SparseGateLibrary,
        SparseMatrix, SparseOptimizer, Complex64,
    };
    pub use crate::linter::{
        QuantumLinter, LinterConfig, StyleStrictness, LintingResult, LintIssue,
        IssueType, Severity, CircuitLocation, PerformanceImpact, PatternDetector,
        QuantumPattern, PatternMatcher, ConnectivityPattern, ParameterConstraint,
        ConstraintType as LinterConstraintType, PatternFlexibility, PatternDetectionResult, PatternStatistics,
        PatternPerformanceProfile, PatternAnalysisResult, PatternInteraction, InteractionType,
        AntiPatternDetector, QuantumAntiPattern, AntiPatternDetectionResult, StyleChecker,
        StyleRule, NamingConvention, QubitOrderingStyle, IndentationStyle as LinterIndentationStyle, GateGroupingStyle,
        MeasurementPlacementStyle, BarrierUsageStyle, StyleConfig, StyleCheckResult,
        StyleViolation, StyleAnalysisResult, OptimizationAnalyzer, OptimizationRule,
        OptimizationSuggestion as LinterOptimizationSuggestion, OptimizationType, OptimizationImprovement as LinterOptimizationImprovement, Difficulty as LinterDifficulty,
        OptimizationAnalysisResult, PerformanceProjection, PerformanceMetrics as LinterPerformanceMetrics,
        ComplexityAnalyzer, ComplexityMetric, ComplexityMetrics as LinterComplexityMetrics,
        ComplexityClassification, ScalingBehavior as LinterScalingBehavior, ComplexityAnalysisResult, ComplexityTrend,
        TrendDirection as LinterTrendDirection, SimplificationSuggestion, SimplificationType, Risk, BestPracticesChecker,
        BestPracticeRule, PracticeGuidelines, CustomGuideline, Importance, BestPracticeResult,
        BestPracticeViolation, BestPracticesCompliance, ComplianceLevel, AutoFix, AutoFixType,
        SafetyLevel, LintingStatistics, LintingMetadata, AnalysisScope,
    };
    pub use crate::scirs2_optimization::{
        CircuitTemplate, EarlyStoppingCriteria, KernelType, ObjectiveFunction,
        OptimizationAlgorithm, OptimizationConfig, OptimizationHistory, Parameter,
        ParameterizedGate, QAOAObjective, QuantumCircuitOptimizer, VQEObjective,
    };
    pub use crate::scirs2_similarity::{
        BatchSimilarityComputer, CircuitDistanceMetrics, CircuitFeatures,
        CircuitSimilarityAnalyzer, CircuitSimilarityMetrics, EntanglementStructure,
        GraphKernelType, GraphSimilarityAlgorithm, MLModelType, SciRS2Graph, SimilarityAlgorithm,
        SimilarityConfig, SimilarityWeights,
    };
    pub use crate::simulator_interface::{
        CircuitCompiler, CompiledCircuit, ContractionStrategy, ExecutionResult,
        InstructionSet, MemoryOptimization, OptimizationLevel as SimulatorOptimizationLevel,
        ResourceRequirements, SimulatorBackend, SimulatorExecutor,
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
    pub use crate::scirs2_transpiler_enhanced::{
        EnhancedTranspiler, EnhancedTranspilerConfig, HardwareBackend,
        PerformanceConstraints, TranspilationPass, DecompositionStrategy,
        AdvancedHardwareFeatures,
        ErrorMitigationSupport, NativeGateSet as EnhancedNativeGateSet,
        GateDecomposition, DecomposedGate, TranspilationResult as EnhancedTranspilationResult,
        CircuitAnalysis, ParallelismAnalysis, GateStatistics, TopologyAnalysis, TopologyType,
        PassResult, DecompositionResult,
        MitigationResult, PerformancePrediction, Bottleneck, BottleneckType,
        VisualRepresentation, CompatibilityReport,
        SuggestionType, ImpactLevel, RoutingModel, PredictionModel, SwapGate, RoutingFeedback,
        CircuitFeatures as EnhancedCircuitFeatures, PerformanceMetrics,
    };
    pub use crate::validation::{
        CircuitValidator, ClassicalConstraints, ConnectivityConstraints, DepthLimits,
        GateRestrictions, MeasurementConstraints, ResourceLimits,
        ValidationRules, ValidationStats, ValidationSuggestion,
    };
    pub use crate::verifier::{
        QuantumVerifier, VerifierConfig, VerificationResult, VerificationStatus,
        PropertyChecker, QuantumProperty, EntanglementType, SuperpositionType, CustomPredicate,
        PropertyVerificationResult, VerificationOutcome, NumericalEvidence, EvidenceType, ErrorBounds,
        InvariantChecker, CircuitInvariant, CustomInvariantChecker, InvariantCheckResult, ViolationSeverity,
        TheoremProver, QuantumTheorem, ErrorModel, ExpectedOutput, ProofObligation, ProofStep,
        TheoremResult, ProofStatus, FormalProof, ProofTree, ProofNode, Counterexample,
        ProofComplexityMetrics, ProofStrategy, ModelChecker, TemporalProperty, ModelCheckResult,
        ExecutionTrace, QuantumState, StateTransition, StateSpace, StateSpaceStatistics,
        CorrectnessChecker, CorrectnessCriterion, TestCase, ComplexityClass as VerifierComplexityClass,
        CorrectnessResult, VerifierTestResult, TestOutcome, SymbolicExecutor, SymbolicExecutionConfig,
        SymbolicState, SymbolicVariable, SymbolicType, SymbolicExpression, BinaryOperator,
        UnaryOperator, SymbolicConstraint, ConstraintType, SymbolicExecutionResult,
        SymbolicExecutionStatus, ConstraintSatisfactionResult, VerificationStatistics,
        ConfidenceStatistics, VerificationIssue, IssueType as VerifierIssueType, IssueSeverity, CircuitLocation as VerifierCircuitLocation,
        VerificationMetadata,
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
