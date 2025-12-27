//! Result types for hybrid algorithm execution

use super::data_types::BinaryString;
use super::history::{OptimizationHistory, TrainingHistory};
use super::quantum_types::QuantumState;
use scirs2_core::ndarray::{Array1, Array2};
use std::fmt;
use std::time::{Duration, Instant};

/// VQE result
#[derive(Debug, Clone)]
pub struct VQEResult {
    pub ground_state_energy: f64,
    pub optimal_parameters: Array1<f64>,
    pub ground_state: QuantumState,
    pub excited_states: Vec<ExcitedState>,
    pub optimization_history: OptimizationHistory,
    pub visualizations: Option<VQEVisualizations>,
    pub execution_time: Duration,
    pub convergence_achieved: bool,
    pub performance_metrics: PerformanceMetrics,
}

impl fmt::Display for VQEResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VQE Result:\n")?;
        write!(
            f,
            "  Ground state energy: {:.6}\n",
            self.ground_state_energy
        )?;
        write!(f, "  Convergence achieved: {}\n", self.convergence_achieved)?;
        write!(
            f,
            "  Total iterations: {}\n",
            self.optimization_history.iterations.len()
        )?;
        write!(f, "  Execution time: {:?}\n", self.execution_time)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ExcitedState {
    pub energy: f64,
    pub state: QuantumState,
}

/// QAOA result
#[derive(Debug, Clone)]
pub struct QAOAResult {
    pub optimal_cost: f64,
    pub optimal_parameters: Array1<f64>,
    pub best_solution: BinaryString,
    pub solution_analysis: SolutionAnalysis,
    pub optimization_history: OptimizationHistory,
    pub visualizations: Option<QAOAVisualizations>,
    pub execution_time: Duration,
    pub approximation_ratio: f64,
    pub num_layers_used: usize,
}

#[derive(Debug, Clone)]
pub struct SolutionAnalysis {
    pub cost_value: f64,
    pub constraint_violations: Vec<String>,
    pub solution_quality: f64,
}

/// VQC result
#[derive(Debug, Clone)]
pub struct VQCResult {
    pub optimal_parameters: Array1<f64>,
    pub best_accuracy: f64,
    pub test_metrics: ClassificationMetrics,
    pub training_history: TrainingHistory,
    pub visualizations: Option<VQCVisualizations>,
    pub execution_time: Duration,
    pub model_complexity: f64,
    pub feature_importance: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub confusion_matrix: Array2<usize>,
    pub precision: Array1<f64>,
    pub recall: Array1<f64>,
    pub f1_score: Array1<f64>,
}

/// Generic VQA result
#[derive(Debug, Clone)]
pub struct VQAResult {
    pub optimal_cost: f64,
    pub optimal_parameters: Array1<f64>,
    pub optimization_history: OptimizationHistory,
    pub landscape_analysis: Option<LandscapeAnalysis>,
    pub execution_time: Duration,
    pub convergence_achieved: bool,
    pub benchmark_results: Option<BenchmarkReport>,
}

#[derive(Debug, Clone)]
pub struct LandscapeAnalysis {
    pub local_minima: Vec<LocalMinimum>,
    pub landscape_roughness: f64,
    pub gradient_variance: f64,
    pub barren_plateau_indicator: bool,
}

#[derive(Debug, Clone)]
pub struct LocalMinimum {
    pub parameters: Array1<f64>,
    pub cost: f64,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_iterations: usize,
    pub convergence_rate: f64,
    pub wall_time: Duration,
    pub circuit_evaluations: usize,
}

/// Benchmark report
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub total_iterations: usize,
    pub average_iteration_time: Duration,
    pub convergence_profile: Vec<(usize, f64)>,
}

#[derive(Debug)]
pub(crate) struct BenchmarkMeasurement {
    pub iteration: usize,
    pub wall_time: Instant,
    pub cost: f64,
}

// Visualization types

#[derive(Debug, Clone)]
pub struct VQEVisualizations {
    pub energy_convergence: String,
    pub parameter_evolution: String,
    pub gradient_norms: String,
    pub landscape_heatmap: String,
}

#[derive(Debug, Clone)]
pub struct QAOAVisualizations {
    pub cost_evolution: String,
    pub parameter_landscape: String,
    pub solution_distribution: String,
    pub approximation_ratio: String,
}

#[derive(Debug, Clone)]
pub struct VQCVisualizations {
    pub loss_curves: String,
    pub accuracy_evolution: String,
    pub confusion_matrix: String,
    pub feature_importance: String,
}
