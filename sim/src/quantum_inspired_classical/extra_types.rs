//! Extra utilities for quantum-inspired classical algorithms

use scirs2_core::ndarray::Array1;

use super::types::*;

/// Utility functions for quantum-inspired algorithms
pub struct QuantumInspiredUtils;
impl QuantumInspiredUtils {
    /// Generate synthetic optimization problems
    #[must_use]
    pub fn generate_optimization_problem(
        problem_type: ObjectiveFunction,
        dimension: usize,
        bounds: (f64, f64),
    ) -> (ObjectiveFunction, Vec<(f64, f64)>, Array1<f64>) {
        let bounds_vec = vec![bounds; dimension];
        let optimal_solution = Array1::zeros(dimension);
        (problem_type, bounds_vec, optimal_solution)
    }
    /// Analyze convergence behavior
    #[must_use]
    pub fn analyze_convergence(convergence_history: &[f64]) -> ConvergenceAnalysis {
        if convergence_history.len() < 2 {
            return ConvergenceAnalysis::default();
        }
        let final_value = *convergence_history
            .last()
            .expect("convergence_history has at least 2 elements");
        let initial_value = convergence_history[0];
        let improvement = initial_value - final_value;
        let convergence_rate = if improvement > 0.0 {
            improvement / convergence_history.len() as f64
        } else {
            0.0
        };
        let mut convergence_iteration = convergence_history.len();
        if convergence_history.len() >= 5 {
            for (i, window) in convergence_history.windows(5).enumerate() {
                let mean = window.iter().sum::<f64>() / window.len() as f64;
                let variance =
                    window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
                let adaptive_tolerance = (mean.abs() * 0.1).max(0.1);
                if variance < adaptive_tolerance {
                    convergence_iteration = i + 5;
                    break;
                }
            }
        }
        ConvergenceAnalysis {
            convergence_rate,
            iterations_to_convergence: convergence_iteration,
            final_gradient_norm: 0.0,
            converged: convergence_iteration < convergence_history.len(),
            convergence_criterion: "variance".to_string(),
        }
    }
    /// Compare algorithm performances
    #[must_use]
    pub fn compare_algorithms(
        results1: &[OptimizationResult],
        results2: &[OptimizationResult],
    ) -> ComparisonStats {
        let perf1 = results1
            .iter()
            .map(|r| r.objective_value)
            .collect::<Vec<_>>();
        let perf2 = results2
            .iter()
            .map(|r| r.objective_value)
            .collect::<Vec<_>>();
        let mean1 = perf1.iter().sum::<f64>() / perf1.len() as f64;
        let mean2 = perf2.iter().sum::<f64>() / perf2.len() as f64;
        let speedup = if mean2 > 0.0 { mean2 / mean1 } else { 1.0 };
        ComparisonStats {
            quantum_inspired_performance: mean1,
            classical_performance: mean2,
            speedup_factor: speedup,
            solution_quality_ratio: mean1 / mean2,
            convergence_speed_ratio: 1.0,
        }
    }
    /// Estimate quantum advantage
    #[must_use]
    pub fn estimate_quantum_advantage(
        problem_size: usize,
        algorithm_type: OptimizationAlgorithm,
    ) -> QuantumAdvantageMetrics {
        let theoretical_speedup = match algorithm_type {
            OptimizationAlgorithm::QuantumGeneticAlgorithm => (problem_size as f64).sqrt(),
            OptimizationAlgorithm::QuantumParticleSwarm => (problem_size as f64).log2(),
            OptimizationAlgorithm::ClassicalQAOA => (problem_size as f64 / 2.0).exp2(),
            _ => 1.0,
        };
        QuantumAdvantageMetrics {
            theoretical_speedup,
            practical_advantage: theoretical_speedup * 0.5,
            complexity_class: "BQP".to_string(),
            quantum_resource_requirements: problem_size * 10,
            classical_resource_requirements: problem_size * problem_size,
        }
    }
}
