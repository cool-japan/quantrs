//! Result types and data structures for cross-compilation
//!
//! This module contains all the data structures used to represent
//! cross-compilation results, IR operations, and visualizations.

use super::config::TargetPlatform;
use super::QuantumFramework;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Source circuit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceCircuit {
    /// Framework
    pub framework: QuantumFramework,

    /// Circuit code
    pub code: String,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Cross-compilation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCompilationResult {
    /// Compilation stages
    pub stages: Vec<CompilationStage>,

    /// Intermediate representation
    pub intermediate_representation: Option<QuantumIR>,

    /// Optimized representation
    pub optimized_representation: Option<QuantumIR>,

    /// ML optimization applied
    pub ml_optimization_applied: bool,

    /// Target code
    pub target_code: TargetCode,

    /// Validation result
    pub validation_result: Option<ValidationResult>,

    /// Compilation report
    pub compilation_report: Option<CompilationReport>,

    /// Visual flow
    pub visual_flow: Option<VisualCompilationFlow>,
}

impl CrossCompilationResult {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            intermediate_representation: None,
            optimized_representation: None,
            ml_optimization_applied: false,
            target_code: TargetCode::new(TargetPlatform::Simulator),
            validation_result: None,
            compilation_report: None,
            visual_flow: None,
        }
    }
}

impl Default for CrossCompilationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Compilation stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationStage {
    /// Stage name
    pub name: String,

    /// Duration
    pub duration: std::time::Duration,

    /// Metrics
    pub metrics: HashMap<String, f64>,
}

/// Parsed circuit
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParsedCircuit {
    /// Number of qubits
    pub num_qubits: usize,

    /// Number of classical bits
    pub num_classical_bits: usize,

    /// Quantum operations
    pub operations: Vec<QuantumOperation>,

    /// Classical operations
    pub classical_operations: Vec<ClassicalOperation>,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Quantum operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumOperation {
    /// Operation type
    pub op_type: OperationType,

    /// Target qubits
    pub qubits: Vec<usize>,

    /// Parameters
    pub parameters: Vec<f64>,
}

/// Operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    Gate(String),
    Measurement,
    Reset,
    Barrier,
}

/// Classical operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalOperation {
    /// Operation type
    pub op_type: ClassicalOpType,

    /// Operands
    pub operands: Vec<usize>,
}

/// Classical operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClassicalOpType {
    Assignment,
    Arithmetic,
    Conditional,
}

/// Quantum IR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumIR {
    /// Number of qubits
    pub num_qubits: usize,

    /// Number of classical bits
    pub num_classical_bits: usize,

    /// IR operations
    pub operations: Vec<IROperation>,

    /// Classical operations
    pub classical_operations: Vec<IRClassicalOp>,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl QuantumIR {
    pub fn new() -> Self {
        Self {
            num_qubits: 0,
            num_classical_bits: 0,
            operations: Vec::new(),
            classical_operations: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_operation(&mut self, op: IROperation) {
        self.operations.push(op);
    }

    pub fn add_classical_operation(&mut self, op: IRClassicalOp) {
        self.classical_operations.push(op);
    }
}

impl Default for QuantumIR {
    fn default() -> Self {
        Self::new()
    }
}

/// IR operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IROperation {
    /// Operation type
    pub operation_type: IROperationType,

    /// Target qubits
    pub qubits: Vec<usize>,

    /// Control qubits
    pub controls: Vec<usize>,

    /// Parameters
    pub parameters: Vec<f64>,
}

/// IR operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IROperationType {
    Gate(IRGate),
    Measurement(Vec<usize>, Vec<usize>), // (qubits, classical_bits)
    Reset(Vec<usize>),
    Barrier(Vec<usize>),
}

/// IR gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IRGate {
    // Single-qubit gates
    H,
    X,
    Y,
    Z,
    S,
    T,
    RX(f64),
    RY(f64),
    RZ(f64),

    // Two-qubit gates
    CNOT,
    CZ,
    SWAP,
    ISWAp,
    SqrtISWAP,

    // Three-qubit gates
    Toffoli,
    Fredkin,

    // Parametric gates
    U1(f64),
    U2(f64, f64),
    U3(f64, f64, f64),

    // Custom gates
    Custom(String, Vec<f64>),
}

/// IR classical operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRClassicalOp {
    /// Operation type
    pub op_type: IRClassicalOpType,

    /// Operands
    pub operands: Vec<usize>,

    /// Result
    pub result: Option<usize>,
}

/// IR classical operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IRClassicalOpType {
    Move,
    Add,
    And,
    Or,
    Xor,
    Not,
}

/// Target code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetCode {
    /// Target platform
    pub platform: TargetPlatform,

    /// Generated code
    pub code: String,

    /// Code format
    pub format: CodeFormat,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl TargetCode {
    pub fn new(platform: TargetPlatform) -> Self {
        Self {
            platform,
            code: String::new(),
            format: CodeFormat::Text,
            metadata: HashMap::new(),
        }
    }
}

/// Code format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeFormat {
    Text,
    QASM,
    Quil,
    Cirq,
    IonQJSON,
    Binary,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Is valid
    pub is_valid: bool,

    /// Errors
    pub errors: Vec<ValidationError>,

    /// Warnings
    pub warnings: Vec<ValidationWarning>,

    /// Semantic validation
    pub semantic_validation: Option<bool>,

    /// Resource validation
    pub resource_validation: Option<bool>,

    /// Fidelity estimate
    pub fidelity_estimate: Option<f64>,
}

impl ValidationResult {
    pub const fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            semantic_validation: None,
            resource_validation: None,
            fidelity_estimate: None,
        }
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error type
    pub error_type: ValidationErrorType,

    /// Description
    pub description: String,

    /// Location
    pub location: Option<String>,
}

/// Validation error type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationErrorType {
    SemanticMismatch,
    ResourceExceeded,
    UnsupportedOperation,
    InvalidConfiguration,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning type
    pub warning_type: ValidationWarningType,

    /// Description
    pub description: String,
}

/// Validation warning type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationWarningType {
    SuboptimalCompilation,
    PotentialError,
    DeprecatedFeature,
}

/// Compilation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationReport {
    /// Summary
    pub summary: CompilationSummary,

    /// Stage analyses
    pub stage_analyses: Vec<StageAnalysis>,

    /// Optimization report
    pub optimization_report: Option<OptimizationReport>,

    /// Resource usage
    pub resource_usage: ResourceUsage,

    /// Recommendations
    pub recommendations: Vec<CompilationRecommendation>,
}

impl CompilationReport {
    pub fn new() -> Self {
        Self {
            summary: CompilationSummary::default(),
            stage_analyses: Vec::new(),
            optimization_report: None,
            resource_usage: ResourceUsage::default(),
            recommendations: Vec::new(),
        }
    }
}

impl Default for CompilationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Compilation summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompilationSummary {
    /// Total compilation time
    pub total_time: std::time::Duration,

    /// Original circuit size
    pub original_size: CircuitSize,

    /// Compiled circuit size
    pub compiled_size: CircuitSize,

    /// Size reduction
    pub size_reduction: f64,

    /// Fidelity estimate
    pub fidelity_estimate: f64,
}

/// Circuit size
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CircuitSize {
    /// Number of gates
    pub gate_count: usize,

    /// Circuit depth
    pub depth: usize,

    /// Two-qubit gate count
    pub two_qubit_gates: usize,
}

/// Stage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageAnalysis {
    /// Stage name
    pub stage_name: String,

    /// Performance metrics
    pub performance: StagePerformance,

    /// Transformations applied
    pub transformations: Vec<String>,

    /// Impact analysis
    pub impact: StageImpact,
}

/// Stage performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagePerformance {
    /// Execution time
    pub execution_time: std::time::Duration,

    /// Memory usage
    pub memory_usage: usize,

    /// CPU usage
    pub cpu_usage: f64,
}

/// Stage impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageImpact {
    /// Gate count change
    pub gate_count_change: i32,

    /// Depth change
    pub depth_change: i32,

    /// Fidelity impact
    pub fidelity_impact: f64,
}

/// Optimization report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    /// Applied optimizations
    pub applied_optimizations: Vec<AppliedOptimization>,

    /// Total improvement
    pub total_improvement: OptimizationImprovement,

    /// Optimization breakdown
    pub breakdown: HashMap<String, f64>,
}

/// Applied optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedOptimization {
    /// Optimization name
    pub name: String,

    /// Number of applications
    pub applications: usize,

    /// Impact
    pub impact: OptimizationImpact,
}

/// Optimization impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationImpact {
    /// Gate reduction
    pub gate_reduction: usize,

    /// Depth reduction
    pub depth_reduction: usize,

    /// Estimated speedup
    pub speedup: f64,
}

/// Optimization improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationImprovement {
    /// Gate count improvement
    pub gate_count_improvement: f64,

    /// Depth improvement
    pub depth_improvement: f64,

    /// Execution time improvement
    pub execution_time_improvement: f64,
}

/// Resource usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Peak memory usage
    pub peak_memory: usize,

    /// Total CPU time
    pub cpu_time: std::time::Duration,

    /// Compilation complexity
    pub complexity: CompilationComplexity,
}

/// Compilation complexity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompilationComplexity {
    /// Time complexity
    pub time_complexity: String,

    /// Space complexity
    pub space_complexity: String,
}

/// Compilation recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationRecommendation {
    /// Category
    pub category: RecommendationCategory,

    /// Description
    pub description: String,

    /// Expected benefit
    pub expected_benefit: String,
}

/// Recommendation category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Performance,
    Quality,
    Compatibility,
    BestPractice,
}

/// Visual compilation flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualCompilationFlow {
    /// Flow nodes
    pub nodes: Vec<FlowNode>,

    /// Flow edges
    pub edges: Vec<FlowEdge>,

    /// IR visualization
    pub ir_visualization: Option<IRVisualization>,

    /// Optimization visualization
    pub optimization_visualization: Option<OptimizationVisualization>,
}

impl VisualCompilationFlow {
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            ir_visualization: None,
            optimization_visualization: None,
        }
    }

    pub fn add_node(&mut self, node: FlowNode) {
        self.nodes.push(node);
    }

    pub fn add_edge(&mut self, edge: FlowEdge) {
        self.edges.push(edge);
    }
}

impl Default for VisualCompilationFlow {
    fn default() -> Self {
        Self::new()
    }
}

/// Flow node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowNode {
    /// Node ID
    pub id: usize,

    /// Node name
    pub name: String,

    /// Node type
    pub node_type: NodeType,

    /// Metrics
    pub metrics: HashMap<String, f64>,
}

/// Node type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    CompilationStage,
    OptimizationPass,
    ValidationStep,
}

/// Flow edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowEdge {
    /// From node
    pub from: usize,

    /// To node
    pub to: usize,

    /// Edge type
    pub edge_type: EdgeType,

    /// Data flow
    pub data_flow: DataFlow,
}

/// Edge type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    Sequential,
    Conditional,
    Parallel,
}

/// Data flow
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataFlow {
    /// Data size
    pub data_size: usize,

    /// Data type
    pub data_type: String,
}

/// IR visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRVisualization {
    /// Graph representation
    pub graph: IRGraph,

    /// Layout
    pub layout: GraphLayout,
}

/// IR graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRGraph {
    /// Nodes
    pub nodes: Vec<IRNode>,

    /// Edges
    pub edges: Vec<IREdge>,
}

/// IR node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRNode {
    /// Node ID
    pub id: usize,

    /// Operation
    pub operation: String,

    /// Properties
    pub properties: HashMap<String, String>,
}

/// IR edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IREdge {
    /// From node
    pub from: usize,

    /// To node
    pub to: usize,

    /// Edge label
    pub label: String,
}

/// Graph layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphLayout {
    /// Node positions
    pub positions: HashMap<usize, (f64, f64)>,

    /// Layout algorithm
    pub algorithm: String,
}

/// Optimization visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationVisualization {
    /// Before/after comparison
    pub comparison: ComparisonVisualization,

    /// Optimization timeline
    pub timeline: OptimizationTimeline,
}

/// Comparison visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonVisualization {
    /// Before state
    pub before: CircuitVisualization,

    /// After state
    pub after: CircuitVisualization,

    /// Differences
    pub differences: Vec<Difference>,
}

/// Circuit visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitVisualization {
    /// Circuit diagram
    pub diagram: String,

    /// Metrics
    pub metrics: CircuitMetrics,
}

/// Circuit metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitMetrics {
    /// Gate count
    pub gate_count: usize,

    /// Depth
    pub depth: usize,

    /// Width
    pub width: usize,
}

/// Difference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Difference {
    /// Difference type
    pub diff_type: DifferenceType,

    /// Location
    pub location: String,

    /// Description
    pub description: String,
}

/// Difference type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifferenceType {
    GateRemoved,
    GateAdded,
    GateReplaced,
    GateMoved,
}

/// Optimization timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTimeline {
    /// Timeline events
    pub events: Vec<TimelineEvent>,

    /// Total duration
    pub total_duration: std::time::Duration,
}

/// Timeline event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    /// Timestamp
    pub timestamp: std::time::Duration,

    /// Event type
    pub event_type: String,

    /// Description
    pub description: String,

    /// Impact
    pub impact: Option<f64>,
}

/// Batch compilation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCompilationResult {
    /// Successful compilations
    pub successful_compilations: Vec<CrossCompilationResult>,

    /// Failed compilations
    pub failed_compilations: Vec<FailedCompilation>,

    /// Batch report
    pub batch_report: Option<BatchCompilationReport>,
}

impl BatchCompilationResult {
    pub const fn new() -> Self {
        Self {
            successful_compilations: Vec::new(),
            failed_compilations: Vec::new(),
            batch_report: None,
        }
    }
}

impl Default for BatchCompilationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Failed compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedCompilation {
    /// Source circuit
    pub source: SourceCircuit,

    /// Error message
    pub error: String,
}

/// Batch compilation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCompilationReport {
    /// Success rate
    pub success_rate: f64,

    /// Average compilation time
    pub avg_compilation_time: std::time::Duration,

    /// Common errors
    pub common_errors: Vec<(String, usize)>,

    /// Performance statistics
    pub performance_stats: BatchPerformanceStats,
}

/// Batch performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPerformanceStats {
    /// Total time
    pub total_time: std::time::Duration,

    /// Throughput (circuits/second)
    pub throughput: f64,

    /// Resource efficiency
    pub resource_efficiency: f64,
}
