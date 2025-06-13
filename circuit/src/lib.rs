extern crate proc_macro;

/// Quantum circuit representation and DSL for the QuantRS2 framework.
///
/// This crate provides types for constructing and manipulating
/// quantum circuits with a fluent API.
pub mod builder;
pub mod classical;
pub mod commutation;
pub mod crosstalk;
pub mod dag;
pub mod equivalence;
pub mod fault_tolerant;
pub mod graph_optimizer;
pub mod measurement;
pub mod ml_optimization;
pub mod optimization;
pub mod optimizer;
pub mod photonic;
pub mod pulse;
pub mod qasm;
pub mod routing;
pub mod scirs2_integration;
pub mod simulator_interface;
pub mod slicing;
pub mod synthesis;
pub mod tensor_network;
pub mod topological;
pub mod topology;
pub mod zx_calculus;
pub mod transpiler;
pub mod noise_models;
pub mod validation;
pub mod circuit_cache;
pub mod scirs2_matrices;
pub mod scirs2_optimization;
pub mod scirs2_benchmarking;
pub mod scirs2_similarity;

// Re-exports of commonly used types and traits
pub mod prelude {
    pub use crate::builder::*;
    // Convenience re-export
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
    pub use crate::equivalence::{
        circuits_equivalent, circuits_structurally_equal, EquivalenceChecker, EquivalenceOptions,
        EquivalenceResult, EquivalenceType,
    };
    pub use crate::fault_tolerant::{
        FaultTolerantCircuit, FaultTolerantCompiler, LogicalQubit, MagicState, QECCode,
        ResourceOverhead, SyndromeMeasurement, SyndromeType,
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
    pub use crate::pulse::{
        Channel, DeviceConfig, PulseCalibration, PulseCompiler, PulseInstruction, PulseOptimizer,
        PulseSchedule, Waveform,
    };
    pub use crate::qasm::exporter::ExportError;
    pub use crate::qasm::{
        export_qasm3, parse_qasm3, validate_qasm3, ExportOptions, ParseError, QasmExporter,
        QasmGate, QasmParser, QasmProgram, QasmRegister, QasmStatement, ValidationError,
    };
    pub use crate::routing::{
        CircuitRouter, CouplingMap, Distance, LookaheadConfig, LookaheadRouter, RoutedCircuit,
        RoutingPassType, RoutingResult, RoutingStatistics, RoutingStrategy, SabreConfig,
        SabreRouter, SwapLayer, SwapNetwork,
    };
    pub use crate::scirs2_integration::{
        AnalysisResult, AnalyzerConfig, GraphMetrics, GraphMotif, OptimizationSuggestion,
        SciRS2CircuitAnalyzer, SciRS2CircuitGraph, SciRS2Edge, SciRS2Node, SciRS2NodeType,
    };
    pub use crate::simulator_interface::{
        CircuitCompiler, CompilationTarget, CompiledCircuit, ContractionStrategy, ExecutionResult,
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
    pub use crate::zx_calculus::{
        OptimizedZXResult, ZXDiagram, ZXEdge, ZXNode, ZXOptimizationResult, ZXOptimizer,
    };
    pub use crate::transpiler::{
        DeviceTranspiler, HardwareSpec, NativeGateSet, TranspilationOptions, TranspilationResult,
        TranspilationStats, TranspilationStrategy,
    };
    pub use crate::noise_models::{
        DecoherenceParams, ErrorSource, LeakageError, NoiseAnalysisResult,
        NoiseAnalyzer, ReadoutError, SingleQubitError, ThermalNoise, TwoQubitError,
    };
    pub use crate::validation::{
        CircuitValidator, ClassicalConstraints, ConnectivityConstraints, DepthLimits,
        GateRestrictions, MeasurementConstraints, ResourceLimits,
        ValidationResult, ValidationRules, ValidationStats, ValidationSuggestion,
        ValidationWarning,
    };
    pub use crate::circuit_cache::{
        CacheConfig, CacheEntry, CacheManager, CacheStats, CircuitCache, CircuitSignature,
        CompiledCircuitCache, EvictionPolicy, ExecutionResultCache, SignatureGenerator,
        TranspilationCache,
    };
    pub use crate::scirs2_matrices::{
        CircuitToSparseMatrix, Complex64, SparseFormat, SparseGate,
        SparseGateLibrary, SparseMatrix, SparseOptimizer,
    };
    pub use crate::scirs2_optimization::{
        CircuitTemplate, EarlyStoppingCriteria, KernelType, ObjectiveFunction,
        OptimizationAlgorithm, OptimizationConfig, OptimizationHistory,
        Parameter, ParameterizedGate, QAOAObjective, QuantumCircuitOptimizer, VQEObjective,
    };
    pub use crate::scirs2_benchmarking::{
        BaselineComparison, BenchmarkConfig, BenchmarkReport, BenchmarkRun, CircuitBenchmark,
        CircuitMetrics as BenchmarkCircuitMetrics, DescriptiveStats, Distribution, DistributionFit, HypothesisTestResult,
        InsightCategory, OutlierAnalysis, OutlierDetectionMethod, PerformanceInsight,
        PracticalSignificance, RegressionAnalysis, StatisticalAnalyzer, StatisticalTest,
    };
    pub use crate::scirs2_similarity::{
        BatchSimilarityComputer, CircuitDistanceMetrics, CircuitFeatures, CircuitSimilarityAnalyzer,
        CircuitSimilarityMetrics, EntanglementStructure, GraphKernelType, GraphSimilarityAlgorithm,
        MLModelType, SciRS2Graph, SimilarityAlgorithm, SimilarityConfig, SimilarityWeights,
    };
    pub use quantrs2_core::qubit::QubitId as Qubit;
}

// The following should be proc macros, but we'll implement them later
// for now they're just stubs

/// Creates a qubit set for quantum operations
///
/// # Example
///
/// ```ignore
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
/// ```ignore
/// let circuit = circuit![4; // 4 qubits
///     h(0),
///     cnot(0, 1),
///     h(2),
///     cnot(2, 3)
/// ];
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
/// ```ignore
/// use quantrs2_circuit::quantum;
///
/// quantum! {
///     let qc = circuit(4);  // 4 qubits
///     qc.h(0);
///     qc.cnot(0, 1);
///     qc.measure_all();
/// }
/// ```
#[macro_export]
macro_rules! quantum {
    ($($tokens:tt)*) => {
        compile_error!("quantum! macro not fully implemented yet");
    };
}
