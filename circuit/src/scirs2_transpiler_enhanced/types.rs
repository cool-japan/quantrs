//! Result types and data structures for transpilation

use super::hardware::HardwareBackend;
use super::passes::{ExportFormat, MitigationStrategy};
use crate::builder::Circuit;
use crate::routing::RoutingResult;
use petgraph::graph::Graph;
use quantrs2_core::gate::GateOp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

/// Complete transpilation result
#[derive(Debug, Clone)]
pub struct TranspilationResult<const N: usize = 100> {
    pub transpiled_circuit: Circuit<N>,
    pub original_analysis: CircuitAnalysis,
    pub pass_results: Vec<PassResult>,
    pub performance_prediction: Option<PerformancePrediction>,
    pub visual_representation: Option<VisualRepresentation>,
    pub exports: HashMap<ExportFormat, String>,
    pub transpilation_time: Duration,
    pub quality_metrics: QualityMetrics,
    pub hardware_compatibility: CompatibilityReport,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}

/// Circuit analysis results
#[derive(Debug, Clone)]
pub struct CircuitAnalysis {
    pub dependency_graph: DependencyGraph,
    pub critical_path: Vec<usize>,
    pub parallelism: ParallelismAnalysis,
    pub gate_statistics: GateStatistics,
    pub topology: TopologyAnalysis,
    pub resource_requirements: ResourceRequirements,
    pub complexity_score: f64,
}

/// Dependency graph
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    pub graph: Graph<GateNode, f64>,
}

#[derive(Debug, Clone)]
pub struct GateNode {
    pub index: usize,
    pub gate: Box<dyn GateOp>,
    pub depth: usize,
}

/// Parallelism analysis
#[derive(Debug, Clone)]
pub struct ParallelismAnalysis {
    pub max_parallelism: usize,
    pub average_parallelism: f64,
    pub parallelizable_gates: usize,
    pub parallel_blocks: Vec<Vec<usize>>,
}

/// Gate statistics
#[derive(Debug, Clone)]
pub struct GateStatistics {
    pub total_gates: usize,
    pub single_qubit_gates: usize,
    pub two_qubit_gates: usize,
    pub multi_qubit_gates: usize,
    pub gate_types: HashMap<String, usize>,
}

/// Topology analysis
#[derive(Debug, Clone)]
pub struct TopologyAnalysis {
    pub connectivity_required: HashMap<(usize, usize), usize>,
    pub max_distance: usize,
    pub average_distance: f64,
    pub topology_type: TopologyType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyType {
    Linear,
    Grid,
    HeavyHex,
    AllToAll,
    Custom,
}

/// Resource requirements
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub qubits: usize,
    pub depth: usize,
    pub gates: usize,
    pub execution_time: f64,
    pub memory_required: usize,
}

/// Pass results
#[derive(Debug, Clone)]
pub enum PassResult {
    Decomposition(DecompositionResult),
    Routing(RoutingResult),
    Optimization(OptimizationResult),
    ErrorMitigation(MitigationResult),
}

#[derive(Debug, Clone)]
pub struct DecompositionResult {
    pub decomposed_gates: usize,
    pub gate_count_before: usize,
    pub gate_count_after: usize,
    pub depth_before: usize,
    pub depth_after: usize,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub gates_removed: usize,
    pub gates_fused: usize,
    pub depth_reduction: usize,
    pub patterns_matched: usize,
    pub layout: HashMap<usize, usize>,
    pub improvement_score: f64,
    pub iterations: usize,
}

#[derive(Debug, Clone)]
pub struct MitigationResult {
    pub strategies_applied: Vec<MitigationStrategy>,
    pub overhead_factor: f64,
    pub expected_improvement: f64,
}

/// Performance prediction
#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    pub execution_time: f64,
    pub success_probability: f64,
    pub resource_usage: ResourceUsage,
    pub bottlenecks: Vec<Bottleneck>,
}

#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    pub cpu_usage: f64,
    pub memory_usage: usize,
    pub network_usage: f64,
}

#[derive(Debug, Clone)]
pub struct Bottleneck {
    pub bottleneck_type: BottleneckType,
    pub location: Vec<usize>,
    pub severity: f64,
    pub mitigation: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckType {
    GateSequence,
    Connectivity,
    Coherence,
    Calibration,
}

/// Visual representation
#[derive(Debug, Clone)]
pub struct VisualRepresentation {
    pub ascii_art: String,
    pub latex_code: String,
    pub svg_data: String,
    pub interactive_html: String,
}

/// Quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub estimated_fidelity: f64,
    pub gate_overhead: f64,
    pub depth_overhead: f64,
    pub connectivity_overhead: f64,
    pub resource_efficiency: f64,
}

/// Hardware compatibility report
#[derive(Debug, Clone)]
pub struct CompatibilityReport {
    pub is_compatible: bool,
    pub incompatible_gates: Vec<String>,
    pub missing_connections: Vec<(usize, usize)>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Optimization suggestion
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    pub suggestion_type: SuggestionType,
    pub description: String,
    pub impact: ImpactLevel,
    pub implementation_hint: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionType {
    DepthReduction,
    GateReduction,
    ErrorMitigation,
    RoutingOptimization,
    Parallelization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpactLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Layout optimization result
#[derive(Debug, Clone)]
pub struct LayoutOptimization {
    pub layout_map: HashMap<usize, usize>,
    pub improvement_score: f64,
}

/// Optimization metrics
#[derive(Debug, Clone)]
pub struct OptimizationMetrics {
    pub optimization_time: Duration,
    pub improvement_ratio: f64,
    pub memory_usage: usize,
    pub convergence_iterations: usize,
}

/// Routing model trait for ML-based routing
pub trait RoutingModel: Send + Sync {
    fn update(&mut self, feedback: &RoutingFeedback);
}

/// Prediction model trait for performance prediction
pub trait PredictionModel: Send + Sync {
    fn predict(&self, features: &CircuitFeatures) -> PerformancePrediction;
    fn update(&mut self, actual: &PerformanceMetrics);
}

#[derive(Debug, Clone)]
pub struct SwapGate {
    pub qubit1: usize,
    pub qubit2: usize,
    pub position: usize,
}

#[derive(Debug, Clone)]
pub struct RoutingFeedback {
    pub success: bool,
    pub actual_swaps: usize,
    pub execution_time: f64,
}

#[derive(Debug, Clone)]
pub struct CircuitFeatures {
    pub gate_count: usize,
    pub depth: usize,
    pub two_qubit_ratio: f64,
    pub connectivity_score: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub actual_time: f64,
    pub actual_fidelity: f64,
    pub resource_usage: ResourceUsage,
}

impl<const N: usize> fmt::Display for TranspilationResult<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Transpilation Result:")?;
        writeln!(
            f,
            "  Original gates: {} → Transpiled gates: {}",
            self.original_analysis.gate_statistics.total_gates,
            self.transpiled_circuit.gates().len()
        )?;
        writeln!(f, "  Transpilation time: {:?}", self.transpilation_time)?;
        writeln!(
            f,
            "  Estimated fidelity: {:.3}%",
            self.quality_metrics.estimated_fidelity * 100.0
        )?;
        if let Some(ref pred) = self.performance_prediction {
            writeln!(
                f,
                "  Predicted execution time: {:.3}ms",
                pred.execution_time * 1000.0
            )?;
        }
        Ok(())
    }
}
