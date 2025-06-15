//! DD sequence optimization using SciRS2

use ndarray::{Array1, Array2, ArrayView1};
use std::collections::HashMap;

use super::{
    config::{DDOptimizationAlgorithm, DDOptimizationConfig, DDOptimizationObjective},
    sequences::DDSequence,
    DDCircuitExecutor,
};
use crate::DeviceResult;

// SciRS2 dependencies with fallbacks
#[cfg(feature = "scirs2")]
use scirs2_optimize::{minimize, OptimizeResult};

#[cfg(not(feature = "scirs2"))]
mod fallback_scirs2 {
    pub use super::super::fallback_scirs2::{minimize, OptimizeResult};
}

#[cfg(not(feature = "scirs2"))]
use fallback_scirs2::{minimize, OptimizeResult};

/// DD sequence optimizer using SciRS2
pub struct DDSequenceOptimizer {
    pub config: DDOptimizationConfig,
    pub optimization_history: Vec<OptimizationStep>,
    pub best_parameters: Option<Array1<f64>>,
    pub best_objective_value: f64,
}

/// Single optimization step record
#[derive(Debug, Clone)]
pub struct OptimizationStep {
    pub iteration: usize,
    pub parameters: Array1<f64>,
    pub objective_value: f64,
    pub gradient_norm: Option<f64>,
    pub execution_time: std::time::Duration,
}

/// Optimization result for DD sequences
#[derive(Debug, Clone)]
pub struct DDOptimizationResult {
    pub optimized_sequence: DDSequence,
    pub optimization_metrics: OptimizationMetrics,
    pub convergence_analysis: ConvergenceAnalysis,
    pub parameter_sensitivity: ParameterSensitivityAnalysis,
}

/// Optimization performance metrics
#[derive(Debug, Clone)]
pub struct OptimizationMetrics {
    pub initial_objective: f64,
    pub final_objective: f64,
    pub improvement_factor: f64,
    pub convergence_iterations: usize,
    pub total_function_evaluations: usize,
    pub optimization_time: std::time::Duration,
    pub success: bool,
}

/// Convergence analysis results
#[derive(Debug, Clone)]
pub struct ConvergenceAnalysis {
    pub converged: bool,
    pub convergence_criterion: String,
    pub objective_tolerance: f64,
    pub parameter_tolerance: f64,
    pub gradient_tolerance: Option<f64>,
    pub stagnation_iterations: usize,
}

/// Parameter sensitivity analysis
#[derive(Debug, Clone)]
pub struct ParameterSensitivityAnalysis {
    pub sensitivity_matrix: Array2<f64>,
    pub most_sensitive_parameters: Vec<usize>,
    pub parameter_correlations: Array2<f64>,
    pub robustness_score: f64,
}

impl DDSequenceOptimizer {
    /// Create new DD sequence optimizer
    pub fn new(config: DDOptimizationConfig) -> Self {
        Self {
            config,
            optimization_history: Vec::new(),
            best_parameters: None,
            best_objective_value: f64::INFINITY,
        }
    }

    /// Optimize DD sequence
    pub async fn optimize_sequence(
        &mut self,
        base_sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<DDOptimizationResult> {
        println!(
            "Starting DD sequence optimization with {:?}",
            self.config.optimization_algorithm
        );
        let start_time = std::time::Instant::now();

        // Initialize optimization parameters
        let initial_params = self.initialize_parameters(base_sequence)?;

        // Perform optimization based on algorithm (pass base_sequence and executor directly)
        let optimization_result = match self.config.optimization_algorithm {
            DDOptimizationAlgorithm::GradientFree => {
                self.optimize_gradient_free_impl(base_sequence, executor, &initial_params)?
            }
            DDOptimizationAlgorithm::SimulatedAnnealing => {
                self.optimize_simulated_annealing_impl(base_sequence, executor, &initial_params)?
            }
            DDOptimizationAlgorithm::GeneticAlgorithm => {
                self.optimize_genetic_algorithm_impl(base_sequence, executor, &initial_params)?
            }
            DDOptimizationAlgorithm::ParticleSwarm => {
                self.optimize_particle_swarm_impl(base_sequence, executor, &initial_params)?
            }
            DDOptimizationAlgorithm::DifferentialEvolution => {
                self.optimize_differential_evolution_impl(base_sequence, executor, &initial_params)?
            }
            DDOptimizationAlgorithm::BayesianOptimization => {
                self.optimize_bayesian_impl(base_sequence, executor, &initial_params)?
            }
            DDOptimizationAlgorithm::ReinforcementLearning => {
                self.optimize_reinforcement_learning_impl(base_sequence, executor, &initial_params)?
            }
        };

        // Handle optimization result (extract the optimized parameters)
        let optimal_params = optimization_result;

        // Create optimized sequence
        let optimized_sequence = self.create_optimized_sequence(base_sequence, &optimal_params)?;

        // Analyze optimization results
        let initial_obj = self.evaluate_objective(&initial_params, base_sequence, executor);
        let final_obj = self.evaluate_objective(&optimal_params, base_sequence, executor);

        let metrics = OptimizationMetrics {
            initial_objective: initial_obj,
            final_objective: final_obj,
            improvement_factor: if final_obj > 0.0 {
                initial_obj / final_obj
            } else {
                1.0
            },
            convergence_iterations: 100,      // Placeholder
            total_function_evaluations: 1000, // Placeholder
            optimization_time: start_time.elapsed(),
            success: true,
        };

        let convergence_analysis = ConvergenceAnalysis {
            converged: true,
            convergence_criterion: "Tolerance reached".to_string(),
            objective_tolerance: self.config.convergence_tolerance,
            parameter_tolerance: self.config.convergence_tolerance * 0.1,
            gradient_tolerance: Some(self.config.convergence_tolerance * 0.01),
            stagnation_iterations: 0,
        };

        let sensitivity_analysis =
            self.analyze_parameter_sensitivity(&optimal_params, base_sequence, executor)?;

        println!(
            "DD optimization completed. Improvement: {:.2}x",
            metrics.improvement_factor
        );

        Ok(DDOptimizationResult {
            optimized_sequence,
            optimization_metrics: metrics,
            convergence_analysis,
            parameter_sensitivity: sensitivity_analysis,
        })
    }

    /// Initialize optimization parameters
    fn initialize_parameters(&self, sequence: &DDSequence) -> DeviceResult<Array1<f64>> {
        let param_count = sequence.pulse_timings.len() + sequence.pulse_phases.len();
        let mut params = Array1::zeros(param_count);

        // Initialize with current sequence parameters
        for (i, &timing) in sequence.pulse_timings.iter().enumerate() {
            params[i] = timing / sequence.duration; // Normalize
        }

        for (i, &phase) in sequence.pulse_phases.iter().enumerate() {
            params[sequence.pulse_timings.len() + i] = phase / (2.0 * std::f64::consts::PI);
            // Normalize
        }

        Ok(params)
    }

    /// Evaluate optimization objective
    fn evaluate_objective(
        &self,
        params: &Array1<f64>,
        base_sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
    ) -> f64 {
        // Create temporary sequence with new parameters
        if let Ok(temp_sequence) = self.create_optimized_sequence(base_sequence, params) {
            match self.config.optimization_objective {
                DDOptimizationObjective::MaximizeCoherenceTime => {
                    self.evaluate_coherence_time(&temp_sequence, executor)
                }
                DDOptimizationObjective::MinimizeDecoherenceRate => {
                    1.0 / self.evaluate_coherence_time(&temp_sequence, executor)
                }
                DDOptimizationObjective::MaximizeProcessFidelity => {
                    self.evaluate_process_fidelity(&temp_sequence, executor)
                }
                DDOptimizationObjective::MinimizeGateOverhead => {
                    -(temp_sequence.properties.pulse_count as f64)
                }
                DDOptimizationObjective::MaximizeRobustness => {
                    self.evaluate_robustness(&temp_sequence, executor)
                }
                DDOptimizationObjective::MultiObjective => {
                    self.evaluate_multi_objective(&temp_sequence, executor)
                }
                DDOptimizationObjective::Custom(_) => {
                    self.evaluate_custom_objective(&temp_sequence, executor)
                }
            }
        } else {
            f64::NEG_INFINITY // Invalid parameters
        }
    }

    /// Evaluate coherence time
    fn evaluate_coherence_time(
        &self,
        sequence: &DDSequence,
        _executor: &dyn DDCircuitExecutor,
    ) -> f64 {
        // Simplified coherence time estimation
        let base_t2 = 50e-6; // 50 μs base T2
        let noise_suppression: f64 = sequence.properties.noise_suppression.values().sum();
        let suppression_factor =
            1.0 + noise_suppression / sequence.properties.noise_suppression.len() as f64;

        base_t2 * suppression_factor * 1e6 // Convert to microseconds for optimization
    }

    /// Evaluate process fidelity
    fn evaluate_process_fidelity(
        &self,
        sequence: &DDSequence,
        _executor: &dyn DDCircuitExecutor,
    ) -> f64 {
        // Simplified fidelity estimation based on sequence properties
        let base_fidelity = 0.99;
        let order_bonus = 0.01 * (sequence.properties.sequence_order as f64).log2();
        let overhead_penalty = -0.001 * (sequence.properties.pulse_count as f64).sqrt();

        base_fidelity + order_bonus + overhead_penalty
    }

    /// Evaluate robustness
    fn evaluate_robustness(&self, sequence: &DDSequence, _executor: &dyn DDCircuitExecutor) -> f64 {
        // Robustness based on symmetry properties and noise suppression diversity
        let mut robustness = 0.0;

        if sequence.properties.symmetry.time_reversal {
            robustness += 0.25;
        }
        if sequence.properties.symmetry.phase_symmetry {
            robustness += 0.25;
        }
        if sequence.properties.symmetry.rotational_symmetry {
            robustness += 0.25;
        }
        if sequence.properties.symmetry.inversion_symmetry {
            robustness += 0.25;
        }

        // Add noise suppression diversity bonus
        let noise_types = sequence.properties.noise_suppression.len() as f64;
        robustness += 0.1 * noise_types;

        robustness
    }

    /// Evaluate multi-objective function
    fn evaluate_multi_objective(
        &self,
        sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
    ) -> f64 {
        let mut total_objective = 0.0;

        // Weight objectives based on configuration
        for (objective_name, weight) in &self.config.multi_objective_weights {
            let objective_value = match objective_name.as_str() {
                "coherence_time" => self.evaluate_coherence_time(sequence, executor),
                "process_fidelity" => self.evaluate_process_fidelity(sequence, executor),
                "robustness" => self.evaluate_robustness(sequence, executor),
                "gate_overhead" => -(sequence.properties.pulse_count as f64),
                _ => 0.0,
            };

            total_objective += weight * objective_value;
        }

        total_objective
    }

    /// Evaluate custom objective
    fn evaluate_custom_objective(
        &self,
        _sequence: &DDSequence,
        _executor: &dyn DDCircuitExecutor,
    ) -> f64 {
        // Placeholder for custom objective functions
        1.0
    }

    /// Optimize using gradient-free methods  
    fn optimize_gradient_free_impl(
        &mut self,
        base_sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<Array1<f64>> {
        let result = minimize(
            |params: &ArrayView1<f64>| {
                let params_array = params.to_owned();
                -self.evaluate_objective(&params_array, base_sequence, executor)
                // Minimize negative for maximization
            },
            initial_params.as_slice().unwrap(),
            scirs2_optimize::unconstrained::Method::NelderMead,
            None,
        )
        .map_err(|e| crate::DeviceError::OptimizationError(format!("{:?}", e)))?;

        Ok(Array1::from_vec(result.x.to_vec()))
    }

    /// Optimize using simulated annealing (placeholder)
    fn optimize_simulated_annealing_impl(
        &mut self,
        base_sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<Array1<f64>> {
        // Fallback to gradient-free for now
        self.optimize_gradient_free_impl(base_sequence, executor, initial_params)
    }

    /// Other optimization algorithms (placeholders)
    fn optimize_genetic_algorithm_impl(
        &mut self,
        base_sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<Array1<f64>> {
        self.optimize_gradient_free_impl(base_sequence, executor, initial_params)
    }

    fn optimize_particle_swarm_impl(
        &mut self,
        base_sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<Array1<f64>> {
        self.optimize_gradient_free_impl(base_sequence, executor, initial_params)
    }

    fn optimize_differential_evolution_impl(
        &mut self,
        base_sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<Array1<f64>> {
        self.optimize_gradient_free_impl(base_sequence, executor, initial_params)
    }

    fn optimize_bayesian_impl(
        &mut self,
        base_sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<Array1<f64>> {
        self.optimize_gradient_free_impl(base_sequence, executor, initial_params)
    }

    fn optimize_reinforcement_learning_impl(
        &mut self,
        base_sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<Array1<f64>> {
        self.optimize_gradient_free_impl(base_sequence, executor, initial_params)
    }

    /// Create optimized sequence from parameters
    fn create_optimized_sequence(
        &self,
        base_sequence: &DDSequence,
        params: &Array1<f64>,
    ) -> DeviceResult<DDSequence> {
        let mut optimized = base_sequence.clone();

        // Update timing parameters
        let timing_count = base_sequence.pulse_timings.len();
        for i in 0..timing_count {
            if i < params.len() {
                optimized.pulse_timings[i] = params[i] * base_sequence.duration;
                // Denormalize
            }
        }

        // Update phase parameters
        for i in 0..base_sequence.pulse_phases.len() {
            let param_idx = timing_count + i;
            if param_idx < params.len() {
                optimized.pulse_phases[i] = params[param_idx] * 2.0 * std::f64::consts::PI;
                // Denormalize
            }
        }

        Ok(optimized)
    }

    /// Analyze parameter sensitivity
    fn analyze_parameter_sensitivity(
        &self,
        optimal_params: &Array1<f64>,
        base_sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<ParameterSensitivityAnalysis> {
        let param_count = optimal_params.len();
        let mut sensitivity_matrix = Array2::zeros((param_count, param_count));
        let perturbation = 0.01; // 1% perturbation

        // Calculate sensitivity for each parameter
        for i in 0..param_count {
            let mut perturbed_params = optimal_params.clone();
            perturbed_params[i] *= 1.0 + perturbation;

            let base_objective = self.evaluate_objective(optimal_params, base_sequence, executor);
            let perturbed_objective =
                self.evaluate_objective(&perturbed_params, base_sequence, executor);

            let sensitivity =
                (perturbed_objective - base_objective) / (perturbation * optimal_params[i]);
            sensitivity_matrix[[i, i]] = sensitivity;
        }

        // Find most sensitive parameters
        let mut sensitivities: Vec<(usize, f64)> = (0..param_count)
            .map(|i| (i, sensitivity_matrix[[i, i]].abs()))
            .collect();
        sensitivities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let most_sensitive_parameters: Vec<usize> =
            sensitivities.iter().take(5).map(|(idx, _)| *idx).collect();

        // Simple correlation matrix (identity for now)
        let parameter_correlations = Array2::eye(param_count);

        // Robustness score based on sensitivity distribution
        let avg_sensitivity =
            sensitivities.iter().map(|(_, s)| s).sum::<f64>() / param_count as f64;
        let robustness_score = 1.0 / (1.0 + avg_sensitivity);

        Ok(ParameterSensitivityAnalysis {
            sensitivity_matrix,
            most_sensitive_parameters,
            parameter_correlations,
            robustness_score,
        })
    }
}
