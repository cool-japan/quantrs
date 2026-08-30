//! `SciRS2` optimization integration for parameter tuning
//!
//! This module integrates `SciRS2`'s advanced optimization capabilities for quantum circuit
//! parameter optimization, variational algorithms, and machine learning-enhanced optimization.

use crate::builder::Circuit;
use crate::scirs2_matrices::SparseMatrix;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Placeholder types representing SciRS2 optimization interface
// In the real implementation, these would be imported from SciRS2

/// Optimization objective function
pub trait ObjectiveFunction: Send + Sync {
    /// Evaluate the objective at given parameters
    fn evaluate(&self, parameters: &[f64]) -> f64;

    /// Compute gradient if available
    fn gradient(&self, parameters: &[f64]) -> Option<Vec<f64>> {
        None
    }

    /// Compute Hessian if available
    fn hessian(&self, parameters: &[f64]) -> Option<Vec<Vec<f64>>> {
        None
    }

    /// Get parameter bounds
    fn bounds(&self) -> Vec<(f64, f64)>;

    /// Get objective name
    fn name(&self) -> &str;
}

/// `SciRS2` optimization algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationAlgorithm {
    /// Gradient descent variants
    GradientDescent { learning_rate: f64, momentum: f64 },
    /// Adam optimizer
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    /// L-BFGS-B
    LBFGSB {
        max_iterations: usize,
        tolerance: f64,
    },
    /// Nelder-Mead simplex
    NelderMead {
        max_iterations: usize,
        tolerance: f64,
    },
    /// Simulated annealing
    SimulatedAnnealing {
        initial_temperature: f64,
        cooling_rate: f64,
        min_temperature: f64,
    },
    /// Genetic algorithm
    GeneticAlgorithm {
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    },
    /// Particle swarm optimization
    ParticleSwarm {
        num_particles: usize,
        inertia_weight: f64,
        cognitive_weight: f64,
        social_weight: f64,
    },
    /// Bayesian optimization
    BayesianOptimization {
        acquisition_function: AcquisitionFunction,
        kernel: KernelType,
        num_initial_samples: usize,
    },
    /// Quantum approximate optimization algorithm (QAOA)
    QAOA {
        num_layers: usize,
        classical_optimizer: Box<Self>,
    },
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone, PartialEq)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound { kappa: f64 },
    Thompson,
}

/// Kernel types for Gaussian processes
#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    RBF { length_scale: f64 },
    Matern { nu: f64, length_scale: f64 },
    Linear { variance: f64 },
    Periodic { period: f64, length_scale: f64 },
}

/// Optimization configuration
pub struct OptimizationConfig {
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
    /// Maximum number of function evaluations
    pub max_evaluations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Parallel evaluation of objective
    pub parallel: bool,
    /// Number of threads for parallel evaluation
    pub num_threads: Option<usize>,
    /// Progress callback
    pub progress_callback: Option<Box<dyn Fn(usize, f64) + Send + Sync>>,
    /// Early stopping criteria
    pub early_stopping: Option<EarlyStoppingCriteria>,
}

impl std::fmt::Debug for OptimizationConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OptimizationConfig")
            .field("algorithm", &self.algorithm)
            .field("max_evaluations", &self.max_evaluations)
            .field("tolerance", &self.tolerance)
            .field("seed", &self.seed)
            .field("parallel", &self.parallel)
            .field("num_threads", &self.num_threads)
            .field(
                "progress_callback",
                &self.progress_callback.as_ref().map(|_| "Some(callback)"),
            )
            .field("early_stopping", &self.early_stopping)
            .finish()
    }
}

impl Clone for OptimizationConfig {
    fn clone(&self) -> Self {
        Self {
            algorithm: self.algorithm.clone(),
            max_evaluations: self.max_evaluations,
            tolerance: self.tolerance,
            seed: self.seed,
            parallel: self.parallel,
            num_threads: self.num_threads,
            progress_callback: None, // Function pointers can't be cloned
            early_stopping: self.early_stopping.clone(),
        }
    }
}

/// Early stopping criteria
#[derive(Debug, Clone)]
pub struct EarlyStoppingCriteria {
    /// Patience (number of iterations without improvement)
    pub patience: usize,
    /// Minimum change to be considered an improvement
    pub min_delta: f64,
    /// Monitor best value or last value
    pub monitor_best: bool,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimal parameters
    pub optimal_parameters: Vec<f64>,
    /// Optimal objective value
    pub optimal_value: f64,
    /// Number of function evaluations
    pub num_evaluations: usize,
    /// Convergence status
    pub converged: bool,
    /// Optimization history
    pub history: OptimizationHistory,
    /// Additional algorithm-specific information
    pub algorithm_info: HashMap<String, String>,
    /// Total optimization time
    pub optimization_time: std::time::Duration,
}

/// Optimization history tracking
#[derive(Debug, Clone)]
pub struct OptimizationHistory {
    /// Parameter values at each iteration
    pub parameters: Vec<Vec<f64>>,
    /// Objective values at each iteration
    pub objective_values: Vec<f64>,
    /// Gradient norms (if available)
    pub gradient_norms: Vec<f64>,
    /// Step sizes
    pub step_sizes: Vec<f64>,
    /// Timestamps
    pub timestamps: Vec<std::time::Instant>,
}

/// Quantum circuit parameter optimizer using `SciRS2`
pub struct QuantumCircuitOptimizer {
    /// Current circuit template
    circuit_template: CircuitTemplate,
    /// Optimization configuration
    config: OptimizationConfig,
    /// Parameter history
    history: Arc<Mutex<OptimizationHistory>>,
    /// Best parameters found so far
    best_parameters: Arc<Mutex<Option<Vec<f64>>>>,
    /// Best objective value
    best_value: Arc<Mutex<f64>>,
}

/// Parameterized circuit template
#[derive(Debug, Clone)]
pub struct CircuitTemplate {
    /// Circuit structure with parameter placeholders
    pub structure: Vec<ParameterizedGate>,
    /// Parameter names and bounds
    pub parameters: Vec<Parameter>,
    /// Number of qubits
    pub num_qubits: usize,
}

/// Parameterized gate in circuit template
#[derive(Debug, Clone)]
pub struct ParameterizedGate {
    /// Gate name
    pub gate_name: String,
    /// Qubits the gate acts on
    pub qubits: Vec<usize>,
    /// Parameter indices
    pub parameter_indices: Vec<usize>,
    /// Fixed parameters (if any)
    pub fixed_parameters: Vec<f64>,
}

/// Parameter definition
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Lower bound
    pub lower_bound: f64,
    /// Upper bound
    pub upper_bound: f64,
    /// Initial value
    pub initial_value: f64,
    /// Whether parameter is discrete
    pub discrete: bool,
}

impl QuantumCircuitOptimizer {
    /// Create a new quantum circuit optimizer
    #[must_use]
    pub fn new(template: CircuitTemplate, config: OptimizationConfig) -> Self {
        Self {
            circuit_template: template,
            config,
            history: Arc::new(Mutex::new(OptimizationHistory {
                parameters: Vec::new(),
                objective_values: Vec::new(),
                gradient_norms: Vec::new(),
                step_sizes: Vec::new(),
                timestamps: Vec::new(),
            })),
            best_parameters: Arc::new(Mutex::new(None)),
            best_value: Arc::new(Mutex::new(f64::INFINITY)),
        }
    }

    /// Optimize circuit parameters
    pub fn optimize(
        &mut self,
        objective: Arc<dyn ObjectiveFunction>,
    ) -> QuantRS2Result<OptimizationResult> {
        let start_time = std::time::Instant::now();

        // Get initial parameters
        let initial_params: Vec<f64> = self
            .circuit_template
            .parameters
            .iter()
            .map(|p| p.initial_value)
            .collect();

        // Validate parameter bounds
        let bounds = objective.bounds();
        if bounds.len() != initial_params.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Parameter count mismatch with bounds".to_string(),
            ));
        }

        // Run optimization based on algorithm
        let result = match &self.config.algorithm {
            OptimizationAlgorithm::GradientDescent {
                learning_rate,
                momentum,
            } => self.optimize_gradient_descent(
                objective,
                &initial_params,
                *learning_rate,
                *momentum,
            ),
            OptimizationAlgorithm::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            } => self.optimize_adam(
                objective,
                &initial_params,
                *learning_rate,
                *beta1,
                *beta2,
                *epsilon,
            ),
            OptimizationAlgorithm::LBFGSB {
                max_iterations,
                tolerance,
            } => self.optimize_lbfgs(objective, &initial_params, *max_iterations, *tolerance),
            OptimizationAlgorithm::NelderMead {
                max_iterations,
                tolerance,
            } => self.optimize_nelder_mead(objective, &initial_params, *max_iterations, *tolerance),
            OptimizationAlgorithm::SimulatedAnnealing {
                initial_temperature,
                cooling_rate,
                min_temperature,
            } => self.optimize_simulated_annealing(
                objective,
                &initial_params,
                *initial_temperature,
                *cooling_rate,
                *min_temperature,
            ),
            OptimizationAlgorithm::BayesianOptimization {
                acquisition_function,
                kernel,
                num_initial_samples,
            } => self.optimize_bayesian(
                objective,
                &initial_params,
                acquisition_function,
                kernel,
                *num_initial_samples,
            ),
            _ => Err(QuantRS2Error::InvalidInput(
                "Algorithm not yet implemented".to_string(),
            )),
        }?;

        let history = self
            .history
            .lock()
            .map_err(|e| QuantRS2Error::RuntimeError(format!("Failed to lock history: {}", e)))?
            .clone();

        Ok(OptimizationResult {
            optimal_parameters: result.0,
            optimal_value: result.1,
            num_evaluations: result.2,
            converged: result.3,
            history,
            algorithm_info: HashMap::new(),
            optimization_time: start_time.elapsed(),
        })
    }

    /// Gradient descent optimization
    fn optimize_gradient_descent(
        &self,
        objective: Arc<dyn ObjectiveFunction>,
        initial_params: &[f64],
        learning_rate: f64,
        momentum: f64,
    ) -> QuantRS2Result<(Vec<f64>, f64, usize, bool)> {
        let mut params = initial_params.to_vec();
        let mut velocity = vec![0.0; params.len()];
        let mut evaluations = 0;
        let mut best_value = f64::INFINITY;
        // The Euclidean norm of the previous iteration's parameter update;
        // this is what `record_iteration` reports as this iteration's
        // "step size" (there is no step yet before the first update).
        let mut last_step_size = 0.0;

        for iteration in 0..self.config.max_evaluations {
            // Evaluate objective
            let value = objective.evaluate(&params);
            evaluations += 1;

            // Update best
            if value < best_value {
                best_value = value;
                if let Ok(mut guard) = self.best_parameters.lock() {
                    *guard = Some(params.clone());
                }
                if let Ok(mut guard) = self.best_value.lock() {
                    *guard = best_value;
                }
            }

            // Compute gradient (numerical if not available) *before* recording
            // history, so `gradient_norms` holds the real gradient driving this
            // iteration rather than a placeholder.
            let gradient = if let Some(grad) = objective.gradient(&params) {
                grad
            } else {
                self.numerical_gradient(&*objective, &params)?
            };
            let gradient_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            // Record history
            self.record_iteration(&params, value, iteration, gradient_norm, last_step_size);

            // Check convergence
            if iteration > 0 {
                let prev_value = self
                    .history
                    .lock()
                    .ok()
                    .and_then(|h| h.objective_values.get(iteration - 1).copied())
                    .unwrap_or(value);
                if (prev_value - value).abs() < self.config.tolerance {
                    return Ok((params, best_value, evaluations, true));
                }
            }

            // Update parameters with momentum
            for i in 0..params.len() {
                velocity[i] = momentum.mul_add(velocity[i], -(learning_rate * gradient[i]));
                params[i] += velocity[i];

                // Apply bounds
                let bounds = objective.bounds();
                params[i] = params[i].max(bounds[i].0).min(bounds[i].1);
            }
            last_step_size = velocity.iter().map(|v| v * v).sum::<f64>().sqrt();

            // Progress callback
            if let Some(callback) = &self.config.progress_callback {
                callback(iteration, value);
            }
        }

        Ok((params, best_value, evaluations, false))
    }

    /// Adam optimization algorithm
    fn optimize_adam(
        &self,
        objective: Arc<dyn ObjectiveFunction>,
        initial_params: &[f64],
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> QuantRS2Result<(Vec<f64>, f64, usize, bool)> {
        let mut params = initial_params.to_vec();
        let mut m = vec![0.0; params.len()]; // First moment
        let mut v = vec![0.0; params.len()]; // Second moment
        let mut evaluations = 0;
        let mut best_value = f64::INFINITY;
        let mut last_step_size = 0.0;

        for iteration in 0..self.config.max_evaluations {
            let t = iteration + 1;

            // Evaluate objective
            let value = objective.evaluate(&params);
            evaluations += 1;

            // Update best
            if value < best_value {
                best_value = value;
                if let Ok(mut guard) = self.best_parameters.lock() {
                    *guard = Some(params.clone());
                }
                if let Ok(mut guard) = self.best_value.lock() {
                    *guard = best_value;
                }
            }

            // Compute gradient before recording so `gradient_norms` holds the
            // real value driving this iteration's update.
            let gradient = if let Some(grad) = objective.gradient(&params) {
                grad
            } else {
                self.numerical_gradient(&*objective, &params)?
            };
            let gradient_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            // Record history
            self.record_iteration(&params, value, iteration, gradient_norm, last_step_size);

            // Check convergence
            if iteration > 0 {
                let prev_value = self
                    .history
                    .lock()
                    .ok()
                    .and_then(|h| h.objective_values.get(iteration - 1).copied())
                    .unwrap_or(value);
                if (prev_value - value).abs() < self.config.tolerance {
                    return Ok((params, best_value, evaluations, true));
                }
            }

            // Update biased first and second moment estimates
            let mut step_sq_norm = 0.0;
            for i in 0..params.len() {
                m[i] = beta1.mul_add(m[i], (1.0 - beta1) * gradient[i]);
                v[i] = beta2.mul_add(v[i], (1.0 - beta2) * gradient[i] * gradient[i]);

                // Bias correction
                let m_hat = m[i] / (1.0 - beta1.powi(t as i32));
                let v_hat = v[i] / (1.0 - beta2.powi(t as i32));

                // Update parameters
                let step = learning_rate * m_hat / (v_hat.sqrt() + epsilon);
                params[i] -= step;
                step_sq_norm += step * step;

                // Apply bounds
                let bounds = objective.bounds();
                params[i] = params[i].max(bounds[i].0).min(bounds[i].1);
            }
            last_step_size = step_sq_norm.sqrt();

            // Progress callback
            if let Some(callback) = &self.config.progress_callback {
                callback(iteration, value);
            }
        }

        Ok((params, best_value, evaluations, false))
    }

    /// L-BFGS-B: limited-memory BFGS quasi-Newton optimization with box
    /// (bound) constraints.
    ///
    /// This is a genuine L-BFGS implementation, not an alias for gradient
    /// descent: it maintains the last `memory` `(s, y)` curvature pairs and
    /// uses the standard *two-loop recursion* (Nocedal & Wright, Algorithm
    /// 7.4) to form the quasi-Newton search direction `d = -H_k ∇f(x_k)`
    /// without ever materializing the dense inverse-Hessian approximation
    /// `H_k`. A backtracking Armijo line search picks the step length, and
    /// bounds are enforced by elementwise clamping of the trial point
    /// ("gradient projection"), a standard practical technique for bound
    /// handling. This is not the full generalized Cauchy-point/active-set
    /// method of the original L-BFGS-B paper, but it is a real quasi-Newton
    /// method with genuine curvature memory, superlinear convergence on
    /// smooth problems, and honest bound support -- not the `learning_rate =
    /// 0.01, momentum = 0.9` gradient descent this used to silently alias.
    fn optimize_lbfgs(
        &self,
        objective: Arc<dyn ObjectiveFunction>,
        initial_params: &[f64],
        max_iterations: usize,
        tolerance: f64,
    ) -> QuantRS2Result<(Vec<f64>, f64, usize, bool)> {
        let bounds = objective.bounds();
        let n = initial_params.len();
        // Limited memory: how many (s, y) curvature pairs to retain.
        let memory = 10.min(max_iterations.max(1));

        let clamp = |v: &mut [f64]| {
            for (vi, &(lo, hi)) in v.iter_mut().zip(bounds.iter()) {
                *vi = vi.max(lo).min(hi);
            }
        };

        let gradient_at = |x: &[f64]| -> QuantRS2Result<Vec<f64>> {
            if let Some(g) = objective.gradient(x) {
                Ok(g)
            } else {
                self.numerical_gradient(&*objective, x)
            }
        };

        let mut x = initial_params.to_vec();
        clamp(&mut x);
        let mut evaluations = 0usize;
        let mut value = objective.evaluate(&x);
        evaluations += 1;
        let mut grad = gradient_at(&x)?;

        let mut best_params = x.clone();
        let mut best_value = value;

        let mut s_history: Vec<Vec<f64>> = Vec::with_capacity(memory);
        let mut y_history: Vec<Vec<f64>> = Vec::with_capacity(memory);
        let mut rho_history: Vec<f64> = Vec::with_capacity(memory);

        for iteration in 0..max_iterations {
            let grad_norm = dot(&grad, &grad).sqrt();

            if grad_norm < tolerance {
                self.record_iteration(&x, value, iteration, grad_norm, 0.0);
                return Ok((best_params, best_value, evaluations, true));
            }

            // Two-loop recursion: d = -H_k * grad, using only the stored
            // (s, y, rho) curvature triples (Nocedal & Wright, Alg. 7.4).
            let k = s_history.len();
            let mut q = grad.clone();
            let mut alpha_i = vec![0.0; k];
            for i in (0..k).rev() {
                let a = rho_history[i] * dot(&s_history[i], &q);
                alpha_i[i] = a;
                for j in 0..n {
                    q[j] -= a * y_history[i][j];
                }
            }
            let gamma = if k > 0 {
                let s = &s_history[k - 1];
                let y = &y_history[k - 1];
                dot(s, y) / dot(y, y).max(1e-12)
            } else {
                1.0
            };
            for qj in &mut q {
                *qj *= gamma;
            }
            for i in 0..k {
                let beta = rho_history[i] * dot(&y_history[i], &q);
                for j in 0..n {
                    q[j] += s_history[i][j] * (alpha_i[i] - beta);
                }
            }
            let direction: Vec<f64> = q.iter().map(|v| -v).collect();

            // Backtracking Armijo line search with elementwise bound clamping.
            let directional_derivative = dot(&grad, &direction);
            let c1 = 1e-4;
            let mut step_length = 1.0;
            let mut new_x = x.clone();
            let mut new_value = value;
            let mut accepted = false;
            for _ in 0..20 {
                for j in 0..n {
                    new_x[j] = x[j] + step_length * direction[j];
                }
                clamp(&mut new_x);
                new_value = objective.evaluate(&new_x);
                evaluations += 1;
                if new_value <= step_length.mul_add(c1 * directional_derivative, value) {
                    accepted = true;
                    break;
                }
                step_length *= 0.5;
            }

            if !accepted {
                // No descent direction respecting the bounds: a local
                // (possibly boundary) minimum under the current curvature
                // model, which is a legitimate convergence condition.
                self.record_iteration(&x, value, iteration, grad_norm, 0.0);
                return Ok((best_params, best_value, evaluations, true));
            }

            let step_vec: Vec<f64> = (0..n).map(|j| new_x[j] - x[j]).collect();
            let step_size = dot(&step_vec, &step_vec).sqrt();
            self.record_iteration(&x, value, iteration, grad_norm, step_size);

            let new_grad = gradient_at(&new_x)?;
            let y_vec: Vec<f64> = (0..n).map(|j| new_grad[j] - grad[j]).collect();
            let sy = dot(&step_vec, &y_vec);
            // Skip the curvature update if the curvature condition `s^T y >
            // 0` fails (a standard L-BFGS safeguard against indefinite
            // updates near the bounds or on non-convex regions).
            if sy > 1e-10 {
                if s_history.len() == memory {
                    s_history.remove(0);
                    y_history.remove(0);
                    rho_history.remove(0);
                }
                s_history.push(step_vec);
                y_history.push(y_vec);
                rho_history.push(1.0 / sy);
            }

            x = new_x;
            value = new_value;
            grad = new_grad;

            if value < best_value {
                best_value = value;
                best_params.clone_from(&x);
                if let Ok(mut guard) = self.best_parameters.lock() {
                    *guard = Some(best_params.clone());
                }
                if let Ok(mut guard) = self.best_value.lock() {
                    *guard = best_value;
                }
            }

            if let Some(callback) = &self.config.progress_callback {
                callback(iteration, value);
            }
        }

        Ok((best_params, best_value, evaluations, false))
    }

    /// Nelder-Mead simplex optimization
    fn optimize_nelder_mead(
        &self,
        objective: Arc<dyn ObjectiveFunction>,
        initial_params: &[f64],
        max_iterations: usize,
        tolerance: f64,
    ) -> QuantRS2Result<(Vec<f64>, f64, usize, bool)> {
        let n = initial_params.len();
        let mut simplex = Vec::new();
        let mut evaluations = 0;

        // Initialize simplex
        simplex.push(initial_params.to_vec());
        for i in 0..n {
            let mut vertex = initial_params.to_vec();
            vertex[i] += if vertex[i] == 0.0 {
                0.00025
            } else {
                vertex[i] * 0.05
            };
            simplex.push(vertex);
        }

        // Evaluate initial simplex
        let mut values: Vec<f64> = simplex
            .iter()
            .map(|params| {
                evaluations += 1;
                objective.evaluate(params)
            })
            .collect();

        // Nelder-Mead is derivative-free, so `gradient_norms` is honestly
        // `0.0`; `step_sizes` tracks the real distance the best vertex moves
        // between iterations, seeded from the initial simplex's best vertex.
        let mut previous_best_point = simplex[0].clone();

        for iteration in 0..max_iterations {
            // Sort simplex by objective values
            let mut indices: Vec<usize> = (0..simplex.len()).collect();
            indices.sort_by(|&i, &j| {
                values[i]
                    .partial_cmp(&values[j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let best_value = values[indices[0]];
            let worst_idx = indices[n];
            let second_worst_idx = indices[n - 1];

            let best_step_size = dot_diff(&simplex[indices[0]], &previous_best_point).sqrt();
            previous_best_point.clone_from(&simplex[indices[0]]);

            // Record best iteration
            self.record_iteration(
                &simplex[indices[0]],
                best_value,
                iteration,
                0.0,
                best_step_size,
            );

            // Check convergence
            let range = values[worst_idx] - values[indices[0]];
            if range < tolerance {
                return Ok((simplex[indices[0]].clone(), best_value, evaluations, true));
            }

            // Compute centroid (excluding worst point)
            let mut centroid = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    centroid[j] += simplex[indices[i]][j];
                }
            }
            for j in 0..n {
                centroid[j] /= n as f64;
            }

            // Reflection
            let alpha = 1.0;
            let mut reflected = vec![0.0; n];
            for j in 0..n {
                reflected[j] = centroid[j] + alpha * (centroid[j] - simplex[worst_idx][j]);
            }

            // Apply bounds
            let bounds = objective.bounds();
            for j in 0..n {
                reflected[j] = reflected[j].max(bounds[j].0).min(bounds[j].1);
            }

            let reflected_value = objective.evaluate(&reflected);
            evaluations += 1;

            if values[indices[0]] <= reflected_value && reflected_value < values[second_worst_idx] {
                // Accept reflection
                simplex[worst_idx] = reflected;
                values[worst_idx] = reflected_value;
            } else if reflected_value < values[indices[0]] {
                // Expansion
                let gamma = 2.0;
                let mut expanded = vec![0.0; n];
                for j in 0..n {
                    expanded[j] = centroid[j] + gamma * (reflected[j] - centroid[j]);
                    expanded[j] = expanded[j].max(bounds[j].0).min(bounds[j].1);
                }

                let expanded_value = objective.evaluate(&expanded);
                evaluations += 1;

                if expanded_value < reflected_value {
                    simplex[worst_idx] = expanded;
                    values[worst_idx] = expanded_value;
                } else {
                    simplex[worst_idx] = reflected;
                    values[worst_idx] = reflected_value;
                }
            } else {
                // Contraction
                let rho = 0.5;
                let mut contracted = vec![0.0; n];
                for j in 0..n {
                    contracted[j] = centroid[j] + rho * (simplex[worst_idx][j] - centroid[j]);
                    contracted[j] = contracted[j].max(bounds[j].0).min(bounds[j].1);
                }

                let contracted_value = objective.evaluate(&contracted);
                evaluations += 1;

                if contracted_value < values[worst_idx] {
                    simplex[worst_idx] = contracted;
                    values[worst_idx] = contracted_value;
                } else {
                    // Shrink
                    let sigma = 0.5;
                    for i in 1..=n {
                        for j in 0..n {
                            simplex[i][j] = simplex[indices[0]][j]
                                + sigma * (simplex[i][j] - simplex[indices[0]][j]);
                            simplex[i][j] = simplex[i][j].max(bounds[j].0).min(bounds[j].1);
                        }
                        values[i] = objective.evaluate(&simplex[i]);
                        evaluations += 1;
                    }
                }
            }

            // Progress callback
            if let Some(callback) = &self.config.progress_callback {
                callback(iteration, best_value);
            }
        }

        // Find best point
        let mut best_idx = 0;
        let mut best_value = values[0];
        for i in 1..values.len() {
            if values[i] < best_value {
                best_value = values[i];
                best_idx = i;
            }
        }

        Ok((simplex[best_idx].clone(), best_value, evaluations, false))
    }

    /// Simulated annealing optimization
    fn optimize_simulated_annealing(
        &self,
        objective: Arc<dyn ObjectiveFunction>,
        initial_params: &[f64],
        initial_temperature: f64,
        cooling_rate: f64,
        min_temperature: f64,
    ) -> QuantRS2Result<(Vec<f64>, f64, usize, bool)> {
        use scirs2_core::random::prelude::*;
        let mut rng = thread_rng();

        let mut current_params = initial_params.to_vec();
        let mut current_value = objective.evaluate(&current_params);
        let mut best_params = current_params.clone();
        let mut best_value = current_value;
        let mut temperature = initial_temperature;
        let mut evaluations = 1;

        let bounds = objective.bounds();

        for iteration in 0..self.config.max_evaluations {
            if temperature < min_temperature {
                break;
            }

            // Generate neighbor solution
            let mut neighbor_params = current_params.clone();
            let mut proposed_step_sq_norm = 0.0;
            for i in 0..neighbor_params.len() {
                let range = bounds[i].1 - bounds[i].0;
                let step = rng.random_range(-0.1..0.1) * range * temperature / initial_temperature;
                neighbor_params[i] = (neighbor_params[i] + step)
                    .max(bounds[i].0)
                    .min(bounds[i].1);
                let applied_step = neighbor_params[i] - current_params[i];
                proposed_step_sq_norm += applied_step * applied_step;
            }
            let metropolis_step_size = proposed_step_sq_norm.sqrt();

            let neighbor_value = objective.evaluate(&neighbor_params);
            evaluations += 1;

            // Accept or reject based on Metropolis criterion
            let delta = neighbor_value - current_value;
            if delta < 0.0 || rng.random::<f64>() < (-delta / temperature).exp() {
                current_params = neighbor_params;
                current_value = neighbor_value;

                if current_value < best_value {
                    best_params.clone_from(&current_params);
                    best_value = current_value;
                }
            }

            // Record iteration. Simulated annealing has no gradient, so
            // `gradient_norms` is honestly `0.0`; `step_sizes` is the real
            // Metropolis proposal size explored this iteration (after
            // clamping to bounds), whether or not it was accepted.
            self.record_iteration(
                &current_params,
                current_value,
                iteration,
                0.0,
                metropolis_step_size,
            );

            // Cool down
            temperature *= cooling_rate;

            // Progress callback
            if let Some(callback) = &self.config.progress_callback {
                callback(iteration, best_value);
            }
        }

        Ok((
            best_params,
            best_value,
            evaluations,
            temperature < min_temperature,
        ))
    }

    /// Bayesian optimization (simplified implementation)
    /// Bayesian optimization with a genuine Gaussian-process (GP) surrogate.
    ///
    /// Unlike the former alias to Nelder-Mead, this actually uses
    /// `acquisition_function`, `kernel`, and `num_initial_samples`:
    ///
    /// 1. Draw `num_initial_samples` design points uniformly within the
    ///    bounds (plus the given `initial_params`) and evaluate the true
    ///    objective at each -- the GP's training data.
    /// 2. Fit an exact GP posterior: build the `kernel`-induced covariance
    ///    matrix over every observed point, Cholesky-factorize it, and solve
    ///    for the (constant-mean) GP regression weights.
    /// 3. Repeatedly maximize `acquisition_function` (Expected Improvement /
    ///    Probability of Improvement / Upper-Confidence-Bound / Thompson
    ///    sampling) over a candidate pool (uniform samples plus local
    ///    perturbations around the current best) using the GP posterior
    ///    mean/std at each candidate, evaluate the true objective at the
    ///    winner, fold it into the training set, and refit the GP.
    ///
    /// The loop runs until `self.config.max_evaluations` true objective
    /// evaluations have been spent.
    fn optimize_bayesian(
        &self,
        objective: Arc<dyn ObjectiveFunction>,
        initial_params: &[f64],
        acquisition_function: &AcquisitionFunction,
        kernel: &KernelType,
        num_initial_samples: usize,
    ) -> QuantRS2Result<(Vec<f64>, f64, usize, bool)> {
        use scirs2_core::random::prelude::*;
        let mut rng = thread_rng();

        let bounds = objective.bounds();
        let n_dims = initial_params.len();
        let max_evaluations = self.config.max_evaluations.max(1);

        let mut points: Vec<Vec<f64>> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        let mut evaluations = 0usize;

        let initial_value = objective.evaluate(initial_params);
        points.push(initial_params.to_vec());
        values.push(initial_value);
        evaluations += 1;

        let num_random_samples = num_initial_samples
            .saturating_sub(1)
            .min(max_evaluations.saturating_sub(evaluations));
        for _ in 0..num_random_samples {
            let candidate: Vec<f64> = bounds
                .iter()
                .map(|&(lo, hi)| lo + rng.random::<f64>() * (hi - lo))
                .collect();
            let value = objective.evaluate(&candidate);
            points.push(candidate);
            values.push(value);
            evaluations += 1;
        }

        let mut best_idx = 0;
        for i in 1..values.len() {
            if values[i] < values[best_idx] {
                best_idx = i;
            }
        }
        let mut best_params = points[best_idx].clone();
        let mut best_value = values[best_idx];
        if let Ok(mut guard) = self.best_parameters.lock() {
            *guard = Some(best_params.clone());
        }
        if let Ok(mut guard) = self.best_value.lock() {
            *guard = best_value;
        }

        const JITTER: f64 = 1e-6;
        let num_candidates = (4 * n_dims).max(32);
        let mut iteration = 0usize;
        let mut converged = false;

        while evaluations < max_evaluations {
            let covariance = build_covariance(kernel, &points, JITTER);
            let Ok(chol) = cholesky(&covariance) else {
                // The observed points became numerically degenerate for this
                // kernel (e.g. near-duplicate samples); stop honestly with
                // whatever has been found so far rather than fabricating a
                // posterior from an unfactorizable covariance.
                break;
            };
            let y_mean = values.iter().sum::<f64>() / values.len() as f64;
            let y_centered: Vec<f64> = values.iter().map(|v| v - y_mean).collect();
            let z = forward_substitution(&chol, &y_centered);
            let alpha = backward_substitution_transpose(&chol, &z);

            let mut best_candidate: Option<Vec<f64>> = None;
            let mut best_acquisition = f64::NEG_INFINITY;
            for c in 0..num_candidates {
                let candidate: Vec<f64> = if c < num_candidates / 2 {
                    bounds
                        .iter()
                        .map(|&(lo, hi)| lo + rng.random::<f64>() * (hi - lo))
                        .collect()
                } else {
                    best_params
                        .iter()
                        .zip(bounds.iter())
                        .map(|(&v, &(lo, hi))| {
                            let scale = (hi - lo) * 0.1;
                            (v + (rng.random::<f64>() - 0.5) * 2.0 * scale).clamp(lo, hi)
                        })
                        .collect()
                };

                let (mean_c, std_c) =
                    gp_posterior(kernel, &points, &alpha, &chol, &candidate, y_mean, JITTER);

                let acquisition_score = match acquisition_function {
                    AcquisitionFunction::Thompson => {
                        // Independent Thompson sampling: draw one posterior
                        // sample per candidate via Box-Muller.
                        let u1 = rng.random::<f64>().max(1e-12);
                        let u2 = rng.random::<f64>();
                        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                        let sampled_value = mean_c + std_c * z;
                        -sampled_value // maximize acquisition == minimize sampled objective
                    }
                    other => acquisition_value(other, mean_c, std_c, best_value),
                };

                if acquisition_score > best_acquisition {
                    best_acquisition = acquisition_score;
                    best_candidate = Some(candidate);
                }
            }

            let Some(next_point) = best_candidate else {
                break;
            };
            let next_value = objective.evaluate(&next_point);
            evaluations += 1;

            let step_size = dot_diff(&next_point, &best_params).sqrt();

            if next_value < best_value {
                best_value = next_value;
                best_params = next_point.clone();
                if let Ok(mut guard) = self.best_parameters.lock() {
                    *guard = Some(best_params.clone());
                }
                if let Ok(mut guard) = self.best_value.lock() {
                    *guard = best_value;
                }
            }

            points.push(next_point);
            values.push(next_value);

            // Bayesian optimization has no parameter-space gradient, so
            // `gradient_norms` is honestly `0.0`; `step_sizes` is the real
            // distance from the previous best point to the newly sampled one.
            self.record_iteration(&best_params, best_value, iteration, 0.0, step_size);

            if let Some(callback) = &self.config.progress_callback {
                callback(iteration, best_value);
            }

            // Converge once the acquisition function itself reports
            // negligible expected benefit and the sampled step is tiny --
            // there is nothing more to gain from further sampling.
            if best_acquisition.abs() < self.config.tolerance && step_size < self.config.tolerance {
                converged = true;
                iteration += 1;
                break;
            }
            iteration += 1;
        }

        Ok((best_params, best_value, evaluations, converged))
    }

    /// Compute numerical gradient
    fn numerical_gradient(
        &self,
        objective: &dyn ObjectiveFunction,
        params: &[f64],
    ) -> QuantRS2Result<Vec<f64>> {
        let epsilon = 1e-8;
        let mut gradient = vec![0.0; params.len()];

        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();

            params_plus[i] += epsilon;
            params_minus[i] -= epsilon;

            let f_plus = objective.evaluate(&params_plus);
            let f_minus = objective.evaluate(&params_minus);

            gradient[i] = (f_plus - f_minus) / (2.0 * epsilon);
        }

        Ok(gradient)
    }

    /// Record one optimization iteration.
    ///
    /// `gradient_norm` and `step_size` are the *real* diagnostics the calling
    /// algorithm already computed for this iteration (the parameter-space
    /// gradient's Euclidean norm and the Euclidean norm of the update /
    /// perturbation actually explored), not placeholders: every call site
    /// passes `0.0` only where the algorithm genuinely has no such quantity
    /// (e.g. no gradient exists for the derivative-free Nelder-Mead simplex
    /// or simulated annealing methods).
    fn record_iteration(
        &self,
        params: &[f64],
        value: f64,
        _iteration: usize,
        gradient_norm: f64,
        step_size: f64,
    ) {
        if let Ok(mut history) = self.history.lock() {
            history.parameters.push(params.to_vec());
            history.objective_values.push(value);
            history.gradient_norms.push(gradient_norm);
            history.step_sizes.push(step_size);
            history.timestamps.push(std::time::Instant::now());
        }
    }

    /// Get current best parameters
    #[must_use]
    pub fn get_best_parameters(&self) -> Option<Vec<f64>> {
        self.best_parameters.lock().ok().and_then(|g| g.clone())
    }

    /// Get current best value
    #[must_use]
    pub fn get_best_value(&self) -> f64 {
        self.best_value.lock().ok().map_or(f64::INFINITY, |g| *g)
    }

    /// Build a concrete circuit from the template, binding the current
    /// `parameters` into each parameterized gate.
    ///
    /// Each [`ParameterizedGate`] names a gate (`"H"`, `"RX"`, `"CNOT"`, …),
    /// the qubits it acts on, and the indices into `parameters` supplying its
    /// angles.  The gate is materialized via the circuit builder so the result
    /// is a fully-formed circuit (previously this returned an empty circuit,
    /// silently discarding the entire template).
    pub fn build_circuit(&self, parameters: &[f64]) -> QuantRS2Result<Circuit<32>> {
        if parameters.len() != self.circuit_template.parameters.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Parameter count mismatch".to_string(),
            ));
        }

        let mut circuit = Circuit::<32>::new();
        for gate_template in &self.circuit_template.structure {
            apply_template_gate(&mut circuit, gate_template, parameters)?;
        }

        Ok(circuit)
    }
}

/// Euclidean inner product `a · b`.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Squared Euclidean distance `‖a − b‖²`.
fn dot_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Gaussian-process kernel `k(a, b)` for the requested [`KernelType`].
fn kernel_value(kernel: &KernelType, a: &[f64], b: &[f64]) -> f64 {
    match kernel {
        KernelType::RBF { length_scale } => {
            let denom = (2.0 * length_scale * length_scale).max(1e-12);
            (-dot_diff(a, b) / denom).exp()
        }
        KernelType::Matern { nu, length_scale } => {
            matern_kernel(*nu, *length_scale, dot_diff(a, b).sqrt())
        }
        KernelType::Linear { variance } => variance * dot(a, b),
        KernelType::Periodic {
            period,
            length_scale,
        } => {
            let dist = dot_diff(a, b).sqrt();
            let s = (std::f64::consts::PI * dist / period.max(1e-12)).sin();
            (-2.0 * s * s / (length_scale * length_scale).max(1e-12)).exp()
        }
    }
}

/// Matérn kernel value for separation `dist` (already divided is done inside).
///
/// Implements the three half-integer orders used in practice in closed form
/// (`ν = 1/2, 3/2, 5/2`; Rasmussen & Williams, *Gaussian Processes for Machine
/// Learning*, eq. 4.16); any other requested `ν` snaps to the nearest of
/// these, since the fully general case needs a modified Bessel function of
/// the second kind that this crate does not implement.
fn matern_kernel(nu: f64, length_scale: f64, dist: f64) -> f64 {
    let length_scale = length_scale.max(1e-12);
    let r = dist / length_scale;
    if nu <= 1.0 {
        (-r).exp()
    } else if nu <= 2.0 {
        let root3 = 3f64.sqrt();
        (1.0 + root3 * r) * (-root3 * r).exp()
    } else {
        let root5 = 5f64.sqrt();
        (1.0 + root5 * r + 5.0 * r * r / 3.0) * (-root5 * r).exp()
    }
}

/// Build the `n×n` covariance matrix `K_ij = k(points_i, points_j) + jitter·δ_ij`.
///
/// The jitter term is a small ridge added to the diagonal for numerical
/// stability of the subsequent Cholesky factorization, standard practice for
/// exact GP regression.
fn build_covariance(kernel: &KernelType, points: &[Vec<f64>], jitter: f64) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut cov = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let value =
                kernel_value(kernel, &points[i], &points[j]) + if i == j { jitter } else { 0.0 };
            cov[i][j] = value;
            cov[j][i] = value;
        }
    }
    cov
}

/// Cholesky factorization `L` (lower-triangular) of a symmetric positive
/// definite matrix, such that `L·Lᵀ = matrix`.
fn cholesky(matrix: &[Vec<f64>]) -> QuantRS2Result<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(QuantRS2Error::ComputationError(
                        "GP covariance matrix is not positive definite".to_string(),
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Ok(l)
}

/// Solve `L·y = b` for lower-triangular `L` (forward substitution).
fn forward_substitution(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    y
}

/// Solve `Lᵀ·x = y` for lower-triangular `L` (back substitution against its
/// transpose).
fn backward_substitution_transpose(l: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    x
}

/// GP posterior mean and standard deviation at `x`, given the training
/// `points`, the precomputed regression weights `alpha = K⁻¹(y − y_mean)`,
/// and the Cholesky factor `l` of the training covariance.
fn gp_posterior(
    kernel: &KernelType,
    points: &[Vec<f64>],
    alpha: &[f64],
    l: &[Vec<f64>],
    x: &[f64],
    y_mean: f64,
    jitter: f64,
) -> (f64, f64) {
    let k_star: Vec<f64> = points.iter().map(|p| kernel_value(kernel, p, x)).collect();
    let mean = y_mean + dot(&k_star, alpha);
    let v = forward_substitution(l, &k_star);
    let k_xx = kernel_value(kernel, x, x) + jitter;
    let variance = (k_xx - dot(&v, &v)).max(0.0);
    (mean, variance.sqrt())
}

/// Standard normal CDF via the Abramowitz & Stegun 7.1.26 `erf` approximation
/// (max absolute error ≈ 1.5×10⁻⁷).
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Standard normal PDF.
fn normal_pdf(z: f64) -> f64 {
    (-(z * z) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Error function approximation (Abramowitz & Stegun, formula 7.1.26).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p: f64 = 0.327_591_1;
    let t = 1.0 / p.mul_add(x, 1.0);
    let y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Value of `acquisition` (for minimization) given the GP posterior
/// `(mean, std)` at a candidate point and the best objective value observed
/// so far. Larger is better; the caller picks the candidate that maximizes
/// this. [`AcquisitionFunction::Thompson`] is handled by the caller via
/// direct posterior sampling and never reaches this function.
fn acquisition_value(
    acquisition: &AcquisitionFunction,
    mean: f64,
    std: f64,
    best_so_far: f64,
) -> f64 {
    let std = std.max(1e-9);
    match acquisition {
        AcquisitionFunction::ExpectedImprovement => {
            let improvement = best_so_far - mean;
            let z = improvement / std;
            improvement * normal_cdf(z) + std * normal_pdf(z)
        }
        AcquisitionFunction::ProbabilityOfImprovement => {
            let z = (best_so_far - mean) / std;
            normal_cdf(z)
        }
        AcquisitionFunction::UpperConfidenceBound { kappa } => {
            // Minimization convention: prefer low mean, high uncertainty.
            kappa * std - mean
        }
        AcquisitionFunction::Thompson => {
            unreachable!("Thompson sampling is handled by the caller, not acquisition_value")
        }
    }
}

/// Resolve the angle for the `k`-th parameter slot of a template gate.
///
/// `parameter_indices` give the positions in the optimizer's parameter vector;
/// any leftover slots fall back to `fixed_parameters`.
fn template_gate_angle(
    gate: &ParameterizedGate,
    parameters: &[f64],
    slot: usize,
) -> QuantRS2Result<f64> {
    if let Some(&idx) = gate.parameter_indices.get(slot) {
        parameters.get(idx).copied().ok_or_else(|| {
            QuantRS2Error::InvalidInput(format!(
                "Gate '{}' references parameter index {idx} which is out of range",
                gate.gate_name
            ))
        })
    } else if let Some(&fixed) = gate
        .fixed_parameters
        .get(slot - gate.parameter_indices.len())
    {
        Ok(fixed)
    } else {
        Err(QuantRS2Error::InvalidInput(format!(
            "Gate '{}' is missing angle #{slot}",
            gate.gate_name
        )))
    }
}

/// Materialize a single template gate onto `circuit`.
///
/// Returns an honest error for an unknown gate name or a malformed qubit list
/// rather than silently skipping it.
fn apply_template_gate<const N: usize>(
    circuit: &mut Circuit<N>,
    gate: &ParameterizedGate,
    parameters: &[f64],
) -> QuantRS2Result<()> {
    let q = |i: usize| -> QuantRS2Result<QubitId> {
        gate.qubits
            .get(i)
            .map(|&qb| QubitId(qb as u32))
            .ok_or_else(|| {
                QuantRS2Error::InvalidInput(format!(
                    "Gate '{}' expects at least {} qubit(s)",
                    gate.gate_name,
                    i + 1
                ))
            })
    };

    match gate.gate_name.as_str() {
        "H" | "h" => {
            circuit.h(q(0)?)?;
        }
        "X" | "x" => {
            circuit.x(q(0)?)?;
        }
        "Y" | "y" => {
            circuit.y(q(0)?)?;
        }
        "Z" | "z" => {
            circuit.z(q(0)?)?;
        }
        "S" | "s" => {
            circuit.s(q(0)?)?;
        }
        "T" | "t" => {
            circuit.t(q(0)?)?;
        }
        "RX" | "rx" => {
            circuit.rx(q(0)?, template_gate_angle(gate, parameters, 0)?)?;
        }
        "RY" | "ry" => {
            circuit.ry(q(0)?, template_gate_angle(gate, parameters, 0)?)?;
        }
        "RZ" | "rz" => {
            circuit.rz(q(0)?, template_gate_angle(gate, parameters, 0)?)?;
        }
        "P" | "p" | "PHASE" => {
            circuit.p(q(0)?, template_gate_angle(gate, parameters, 0)?)?;
        }
        "CNOT" | "cnot" | "CX" | "cx" => {
            circuit.cnot(q(0)?, q(1)?)?;
        }
        "CZ" | "cz" => {
            circuit.cz(q(0)?, q(1)?)?;
        }
        other => {
            return Err(QuantRS2Error::UnsupportedOperation(format!(
                "Template gate '{other}' is not supported by the circuit builder"
            )));
        }
    }
    Ok(())
}

/// Variational quantum eigensolver (VQE) objective
pub struct VQEObjective {
    /// Hamiltonian matrix
    hamiltonian: SparseMatrix,
    /// Circuit template
    circuit_template: CircuitTemplate,
    /// Parameter bounds
    bounds: Vec<(f64, f64)>,
}

impl VQEObjective {
    /// Create new VQE objective
    #[must_use]
    pub fn new(hamiltonian: SparseMatrix, circuit_template: CircuitTemplate) -> Self {
        let bounds = circuit_template
            .parameters
            .iter()
            .map(|p| (p.lower_bound, p.upper_bound))
            .collect();

        Self {
            hamiltonian,
            circuit_template,
            bounds,
        }
    }
}

impl VQEObjective {
    /// Fallible core of [`ObjectiveFunction::evaluate`]: build the ansatz state
    /// `|ψ(θ)⟩`, then compute `⟨ψ|H|ψ⟩` directly from the sparse Hamiltonian's
    /// COO triplets.
    fn try_evaluate(&self, parameters: &[f64]) -> QuantRS2Result<f64> {
        let num_qubits = self.circuit_template.num_qubits;
        let state = statevector::simulate_template(
            num_qubits,
            &self.circuit_template.structure,
            parameters,
        )?;

        let dim = 1usize << num_qubits;
        if self.hamiltonian.shape.0 != dim || self.hamiltonian.shape.1 != dim {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Hamiltonian is {}x{} but a {}-qubit circuit needs {dim}x{dim}",
                self.hamiltonian.shape.0, self.hamiltonian.shape.1, num_qubits
            )));
        }

        let expectation = statevector::sparse_expectation(self.hamiltonian.triplets(), &state);
        if expectation.im.abs() > 1e-6 {
            return Err(QuantRS2Error::ComputationError(format!(
                "VQE energy has non-negligible imaginary part ({:.3e}); Hamiltonian is not Hermitian",
                expectation.im
            )));
        }
        Ok(expectation.re)
    }
}

impl ObjectiveFunction for VQEObjective {
    /// Energy `⟨ψ(θ)|H|ψ(θ)⟩` of the ansatz state produced by the circuit
    /// template at the given parameters.
    ///
    /// On a malformed configuration (dimension mismatch, unsupported gate, …)
    /// this returns `f64::INFINITY` — the worst possible value for the
    /// minimizer, so a broken term is rejected rather than silently accepted as
    /// a plausible energy.
    fn evaluate(&self, parameters: &[f64]) -> f64 {
        self.try_evaluate(parameters).unwrap_or(f64::INFINITY)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        self.bounds.clone()
    }

    fn name(&self) -> &'static str {
        "VQE"
    }
}

/// Quantum Approximate Optimization Algorithm (QAOA) objective
pub struct QAOAObjective {
    /// Problem Hamiltonian
    problem_hamiltonian: SparseMatrix,
    /// Mixer Hamiltonian
    mixer_hamiltonian: SparseMatrix,
    /// Number of QAOA layers
    num_layers: usize,
    /// Parameter bounds
    bounds: Vec<(f64, f64)>,
}

impl QAOAObjective {
    /// Create new QAOA objective
    #[must_use]
    pub fn new(
        problem_hamiltonian: SparseMatrix,
        mixer_hamiltonian: SparseMatrix,
        num_layers: usize,
    ) -> Self {
        // Beta and gamma parameters for each layer
        let bounds = vec![(0.0, 2.0 * std::f64::consts::PI); 2 * num_layers];

        Self {
            problem_hamiltonian,
            mixer_hamiltonian,
            num_layers,
            bounds,
        }
    }
}

impl QAOAObjective {
    /// Fallible core of [`ObjectiveFunction::evaluate`].
    ///
    /// Prepares `|+⟩^⊗n`, applies the alternating QAOA layers
    /// `U(β,γ) = ∏_p exp(-iβ_p H_mixer) exp(-iγ_p H_problem)` and returns the
    /// problem-Hamiltonian energy `⟨ψ|H_problem|ψ⟩`.  Each layer's parameters
    /// are taken as `[γ_0, β_0, γ_1, β_1, …]`.
    fn try_evaluate(&self, parameters: &[f64]) -> QuantRS2Result<f64> {
        if parameters.len() != 2 * self.num_layers {
            return Err(QuantRS2Error::InvalidInput(format!(
                "QAOA with {} layers needs {} parameters, got {}",
                self.num_layers,
                2 * self.num_layers,
                parameters.len()
            )));
        }

        let dim = self.problem_hamiltonian.shape.0;
        if !dim.is_power_of_two() || self.problem_hamiltonian.shape.1 != dim {
            return Err(QuantRS2Error::InvalidInput(
                "QAOA problem Hamiltonian must be a square 2^n matrix".to_string(),
            ));
        }
        if self.mixer_hamiltonian.shape != self.problem_hamiltonian.shape {
            return Err(QuantRS2Error::InvalidInput(
                "QAOA mixer and problem Hamiltonians must have the same shape".to_string(),
            ));
        }
        let num_qubits = dim.trailing_zeros() as usize;

        // Dense complex operators for exact exponentiation (QAOA Hamiltonians
        // are small).  exp(-iθH) is applied via scaling-and-squaring Taylor expm.
        let problem_dense =
            statevector::dense_from_triplets(self.problem_hamiltonian.triplets(), dim);
        let mixer_dense = statevector::dense_from_triplets(self.mixer_hamiltonian.triplets(), dim);

        // |+⟩^⊗n = uniform superposition.
        let amp = Complex64::new(1.0 / (dim as f64).sqrt(), 0.0);
        let mut state = vec![amp; dim];

        for layer in 0..self.num_layers {
            let gamma = parameters[2 * layer];
            let beta = parameters[2 * layer + 1];
            // exp(-iγ H_problem)
            let u_problem =
                statevector::expm_scaled(&problem_dense, dim, Complex64::new(0.0, -gamma))?;
            state = statevector::matvec(&u_problem, dim, &state);
            // exp(-iβ H_mixer)
            let u_mixer = statevector::expm_scaled(&mixer_dense, dim, Complex64::new(0.0, -beta))?;
            state = statevector::matvec(&u_mixer, dim, &state);
        }

        let _ = num_qubits; // documented above; retained for clarity of intent
        let expectation =
            statevector::sparse_expectation(self.problem_hamiltonian.triplets(), &state);
        if expectation.im.abs() > 1e-6 {
            return Err(QuantRS2Error::ComputationError(format!(
                "QAOA energy has non-negligible imaginary part ({:.3e}); problem Hamiltonian is not Hermitian",
                expectation.im
            )));
        }
        Ok(expectation.re)
    }
}

impl ObjectiveFunction for QAOAObjective {
    /// Problem-Hamiltonian energy of the QAOA state at the given `[γ,β,…]`.
    ///
    /// On a malformed configuration this returns `f64::INFINITY` (worst value
    /// for a minimizer) rather than a fabricated plausible energy.
    fn evaluate(&self, parameters: &[f64]) -> f64 {
        self.try_evaluate(parameters).unwrap_or(f64::INFINITY)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        self.bounds.clone()
    }

    fn name(&self) -> &'static str {
        "QAOA"
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            algorithm: OptimizationAlgorithm::Adam {
                learning_rate: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            max_evaluations: 1000,
            tolerance: 1e-6,
            seed: None,
            parallel: false,
            num_threads: None,
            progress_callback: None,
            early_stopping: None,
        }
    }
}

/// Dense state-vector simulation helpers for the variational objectives.
///
/// `quantrs2-circuit` is a dependency of `quantrs2-sim`, so it cannot use the
/// simulator crate (dependency cycle).  These routines provide a small,
/// self-contained exact engine: gate application via the generic
/// [`GateOp::matrix`] interface, sparse `⟨ψ|H|ψ⟩`, and a dense complex
/// matrix exponential (scaling-and-squaring + Taylor) used by QAOA's
/// `exp(-iθH)` layers.
mod statevector {
    use super::{ParameterizedGate, QuantRS2Error, QuantRS2Result};
    use quantrs2_core::gate::{
        single::{Phase, RotationX, RotationY, RotationZ, T},
        GateOp,
    };
    use quantrs2_core::qubit::QubitId;
    use scirs2_core::Complex64;

    /// Simulate a circuit template on `2^num_qubits` amplitudes from `|0…0⟩`.
    pub fn simulate_template(
        num_qubits: usize,
        structure: &[ParameterizedGate],
        parameters: &[f64],
    ) -> QuantRS2Result<Vec<Complex64>> {
        let dim = 1usize << num_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);

        for gate in structure {
            let boxed = super::statevector::boxed_template_gate(gate, parameters)?;
            apply_gate(&mut state, num_qubits, boxed.as_ref())?;
        }
        Ok(state)
    }

    /// Build the concrete `GateOp` for a template gate, binding its parameters.
    ///
    /// Only parameterized rotations need a boxed `core` gate here; the
    /// non-parameterized Clifford+T gates are produced via their builder-less
    /// constructors.  Multi-qubit gates (CNOT/CZ) are handled in
    /// [`apply_gate`] through the matrix interface after construction, so they
    /// are constructed here too.
    fn boxed_template_gate(
        gate: &ParameterizedGate,
        parameters: &[f64],
    ) -> QuantRS2Result<Box<dyn GateOp>> {
        use quantrs2_core::gate::multi::{CNOT, CZ};
        use quantrs2_core::gate::single::{Hadamard, PauliX, PauliY, PauliZ};

        let target = |i: usize| -> QuantRS2Result<QubitId> {
            gate.qubits
                .get(i)
                .map(|&q| QubitId(q as u32))
                .ok_or_else(|| {
                    QuantRS2Error::InvalidInput(format!(
                        "Gate '{}' expects at least {} qubit(s)",
                        gate.gate_name,
                        i + 1
                    ))
                })
        };
        let angle = |slot: usize| super::template_gate_angle(gate, parameters, slot);

        let boxed: Box<dyn GateOp> = match gate.gate_name.as_str() {
            "H" | "h" => Box::new(Hadamard { target: target(0)? }),
            "X" | "x" => Box::new(PauliX { target: target(0)? }),
            "Y" | "y" => Box::new(PauliY { target: target(0)? }),
            "Z" | "z" => Box::new(PauliZ { target: target(0)? }),
            "S" | "s" => Box::new(Phase { target: target(0)? }),
            "T" | "t" => Box::new(T { target: target(0)? }),
            "RX" | "rx" => Box::new(RotationX {
                target: target(0)?,
                theta: angle(0)?,
            }),
            "RY" | "ry" => Box::new(RotationY {
                target: target(0)?,
                theta: angle(0)?,
            }),
            "RZ" | "rz" => Box::new(RotationZ {
                target: target(0)?,
                theta: angle(0)?,
            }),
            "CNOT" | "cnot" | "CX" | "cx" => Box::new(CNOT {
                control: target(0)?,
                target: target(1)?,
            }),
            "CZ" | "cz" => Box::new(CZ {
                control: target(0)?,
                target: target(1)?,
            }),
            other => {
                return Err(QuantRS2Error::UnsupportedOperation(format!(
                    "Template gate '{other}' is not supported by the state-vector engine"
                )))
            }
        };
        Ok(boxed)
    }

    /// Apply a (possibly multi-qubit) gate in place, MSB-first convention
    /// (the first qubit of `qubits()` is the high bit of the gate block).
    pub fn apply_gate(
        state: &mut [Complex64],
        num_qubits: usize,
        gate: &dyn GateOp,
    ) -> QuantRS2Result<()> {
        let targets: Vec<usize> = gate.qubits().iter().map(|q| q.id() as usize).collect();
        let k = targets.len();
        if k == 0 {
            return Ok(());
        }
        for &t in &targets {
            if t >= num_qubits {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Gate '{}' acts on qubit {t} but only {num_qubits} qubits are available",
                    gate.name()
                )));
            }
        }
        let matrix = gate.matrix()?;
        let side = 1usize << k;
        if matrix.len() != side * side {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Gate '{}' returned {} matrix elements, expected {}",
                gate.name(),
                matrix.len(),
                side * side
            )));
        }

        // local bit p (LSB=0) -> targets[k-1-p]  (first qubit = MSB)
        let bit_masks: Vec<usize> = (0..k).map(|p| 1usize << targets[k - 1 - p]).collect();
        let mut fixed_mask = 0usize;
        for &m in &bit_masks {
            fixed_mask |= m;
        }
        let dim = state.len();
        let mut visited = vec![false; dim];
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); side];
        let mut indices = vec![0usize; side];

        for base in 0..dim {
            if visited[base] || (base & fixed_mask) != 0 {
                continue;
            }
            for (local, slot) in indices.iter_mut().enumerate() {
                let mut idx = base;
                for (bit, &mask) in bit_masks.iter().enumerate() {
                    if (local >> bit) & 1 == 1 {
                        idx |= mask;
                    }
                }
                *slot = idx;
                amplitudes[local] = state[idx];
                visited[idx] = true;
            }
            for r in 0..side {
                let mut acc = Complex64::new(0.0, 0.0);
                let row = r * side;
                for (c, amp) in amplitudes.iter().enumerate() {
                    acc += matrix[row + c] * amp;
                }
                state[indices[r]] = acc;
            }
        }
        Ok(())
    }

    /// `⟨ψ|H|ψ⟩` from a sparse Hamiltonian given as `(row, col, value)` triplets.
    pub fn sparse_expectation(
        triplets: &[(usize, usize, Complex64)],
        state: &[Complex64],
    ) -> Complex64 {
        let mut acc = Complex64::new(0.0, 0.0);
        for &(row, col, value) in triplets {
            if row < state.len() && col < state.len() {
                acc += state[row].conj() * value * state[col];
            }
        }
        acc
    }

    /// Materialize a dense `dim × dim` (row-major) complex matrix from triplets.
    pub fn dense_from_triplets(
        triplets: &[(usize, usize, Complex64)],
        dim: usize,
    ) -> Vec<Complex64> {
        let mut dense = vec![Complex64::new(0.0, 0.0); dim * dim];
        for &(row, col, value) in triplets {
            if row < dim && col < dim {
                dense[row * dim + col] += value;
            }
        }
        dense
    }

    /// Dense matrix-vector product `y = M x` (M row-major `dim × dim`).
    pub fn matvec(matrix: &[Complex64], dim: usize, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); dim];
        for (row, y_row) in y.iter_mut().enumerate() {
            let base = row * dim;
            let mut acc = Complex64::new(0.0, 0.0);
            for (col, x_col) in x.iter().enumerate() {
                acc += matrix[base + col] * x_col;
            }
            *y_row = acc;
        }
        y
    }

    /// Compute `exp(scale · H)` for a dense `dim × dim` complex matrix `H` using
    /// scaling-and-squaring with a truncated Taylor series.
    ///
    /// This is the standard `expm` algorithm and is exact to machine precision
    /// for the small dimensions used by the variational objectives.  For
    /// `exp(-iθH)` pass `scale = Complex64::new(0.0, -θ)`.
    pub fn expm_scaled(
        h: &[Complex64],
        dim: usize,
        scale: Complex64,
    ) -> QuantRS2Result<Vec<Complex64>> {
        // A = scale * H
        let a: Vec<Complex64> = h.iter().map(|&v| v * scale).collect();

        // Choose squaring count s so that ||A/2^s|| is small (≤ 0.5).
        let norm = matrix_inf_norm(&a, dim);
        let s = if norm <= 0.5 {
            0u32
        } else {
            (norm.log2().ceil().max(0.0) as u32) + 1
        };
        let scaling = Complex64::new(f64::from(1u32 << s).recip(), 0.0);
        let a_scaled: Vec<Complex64> = a.iter().map(|&v| v * scaling).collect();

        // Taylor series: exp(B) = Σ B^k / k!, B = a_scaled.
        let mut result = identity(dim);
        let mut term = identity(dim);
        for k in 1..=18u32 {
            term = matmul(&term, &a_scaled, dim);
            let inv_factorial = Complex64::new(1.0_f64 / factorial(k), 0.0);
            for (r, t) in result.iter_mut().zip(term.iter()) {
                *r += *t * inv_factorial;
            }
        }

        // Square s times.
        for _ in 0..s {
            result = matmul(&result, &result, dim);
        }
        Ok(result)
    }

    fn factorial(n: u32) -> f64 {
        (1..=n).fold(1.0_f64, |acc, x| acc * f64::from(x))
    }

    fn identity(dim: usize) -> Vec<Complex64> {
        let mut m = vec![Complex64::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            m[i * dim + i] = Complex64::new(1.0, 0.0);
        }
        m
    }

    fn matmul(a: &[Complex64], b: &[Complex64], dim: usize) -> Vec<Complex64> {
        let mut c = vec![Complex64::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            for k in 0..dim {
                let a_ik = a[i * dim + k];
                if a_ik.norm_sqr() == 0.0 {
                    continue;
                }
                let brow = k * dim;
                let crow = i * dim;
                for j in 0..dim {
                    c[crow + j] += a_ik * b[brow + j];
                }
            }
        }
        c
    }

    fn matrix_inf_norm(m: &[Complex64], dim: usize) -> f64 {
        let mut max_row = 0.0_f64;
        for i in 0..dim {
            let base = i * dim;
            let row_sum: f64 = (0..dim).map(|j| m[base + j].norm()).sum();
            if row_sum > max_row {
                max_row = row_sum;
            }
        }
        max_row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_config_creation() {
        let config = OptimizationConfig::default();
        assert_eq!(config.max_evaluations, 1000);
        assert_eq!(config.tolerance, 1e-6);
    }

    #[test]
    fn test_vqe_objective() {
        // ⟨ψ|I|ψ⟩ = 1 for any normalized state, regardless of parameters.
        let hamiltonian = SparseMatrix::identity(4);
        let template = CircuitTemplate {
            structure: vec![ParameterizedGate {
                gate_name: "RY".to_string(),
                qubits: vec![0],
                parameter_indices: vec![0],
                fixed_parameters: Vec::new(),
            }],
            parameters: vec![Parameter {
                name: "theta".to_string(),
                lower_bound: 0.0,
                upper_bound: 2.0 * std::f64::consts::PI,
                initial_value: 0.5,
                discrete: false,
            }],
            num_qubits: 2,
        };

        let objective = VQEObjective::new(hamiltonian, template);
        let value = objective.evaluate(&[0.5]);
        assert!((value - 1.0).abs() < 1e-9, "⟨I⟩ must equal 1, got {value}");
    }

    /// `⟨0|RY(θ)† Z RY(θ)|0⟩ = cos θ` — pins the VQE objective to an analytic
    /// value, killing the former `Σ x²` fabrication.
    #[test]
    fn test_vqe_objective_matches_cos() {
        use std::f64::consts::PI;
        // H = Z on a single qubit: diagonal(+1, -1).
        let mut hamiltonian = SparseMatrix::zeros(2, 2);
        hamiltonian.insert(0, 0, Complex64::new(1.0, 0.0));
        hamiltonian.insert(1, 1, Complex64::new(-1.0, 0.0));

        let template = CircuitTemplate {
            structure: vec![ParameterizedGate {
                gate_name: "RY".to_string(),
                qubits: vec![0],
                parameter_indices: vec![0],
                fixed_parameters: Vec::new(),
            }],
            parameters: vec![Parameter {
                name: "theta".to_string(),
                lower_bound: 0.0,
                upper_bound: 2.0 * PI,
                initial_value: 0.0,
                discrete: false,
            }],
            num_qubits: 1,
        };
        let objective = VQEObjective::new(hamiltonian, template);

        for &theta in &[0.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0] {
            let value = objective.evaluate(&[theta]);
            assert!(
                (value - theta.cos()).abs() < 1e-9,
                "θ={theta}: got {value}, expected {}",
                theta.cos()
            );
        }
    }

    #[test]
    fn test_qaoa_objective() {
        // For H_problem = I, ⟨ψ|I|ψ⟩ = 1 for the normalized QAOA state.
        let problem_h = SparseMatrix::identity(4);
        let mixer_h = SparseMatrix::identity(4);

        let objective = QAOAObjective::new(problem_h, mixer_h, 2);
        assert_eq!(objective.bounds().len(), 4); // 2 parameters per layer

        let value = objective.evaluate(&[0.5, 1.0, 1.5, 2.0]);
        assert!(
            (value - 1.0).abs() < 1e-9,
            "⟨I⟩ for QAOA state must equal 1, got {value}"
        );
    }

    /// QAOA with a diagonal problem Hamiltonian H = diag(0,1,1,2) (a 2-qubit
    /// "number operator" Z-cost): the |+⟩^2 start has expectation = mean of the
    /// diagonal = 1.  With γ=β=0 the layers are identity, so the energy must be
    /// exactly that mean — would fail for the old `Σ sin² x` fabrication.
    #[test]
    fn test_qaoa_objective_zero_angles_is_diagonal_mean() {
        let mut problem_h = SparseMatrix::zeros(4, 4);
        problem_h.insert(0, 0, Complex64::new(0.0, 0.0));
        problem_h.insert(1, 1, Complex64::new(1.0, 0.0));
        problem_h.insert(2, 2, Complex64::new(1.0, 0.0));
        problem_h.insert(3, 3, Complex64::new(2.0, 0.0));
        let mixer_h = SparseMatrix::identity(4);

        let objective = QAOAObjective::new(problem_h, mixer_h, 1);
        // γ=0, β=0 → identity layers → ⟨+⊗+|H|+⊗+⟩ = mean(diag) = 1.
        let value = objective.evaluate(&[0.0, 0.0]);
        assert!(
            (value - 1.0).abs() < 1e-9,
            "QAOA zero-angle energy should be diagonal mean 1, got {value}"
        );
    }

    /// `exp(-iθH)` for a Pauli-X generator must reproduce an `RX(2θ)` rotation,
    /// validating the dense matrix-exponential used by QAOA.
    #[test]
    fn test_expm_matches_rx_rotation() {
        use std::f64::consts::PI;
        // H = X = [[0,1],[1,0]].
        let x = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let theta = PI / 3.0;
        // exp(-iθX) = cos θ I - i sin θ X.
        let u = statevector::expm_scaled(&x, 2, Complex64::new(0.0, -theta)).expect("expm");
        let c = theta.cos();
        let s = theta.sin();
        let expected = [
            Complex64::new(c, 0.0),
            Complex64::new(0.0, -s),
            Complex64::new(0.0, -s),
            Complex64::new(c, 0.0),
        ];
        for (got, want) in u.iter().zip(expected.iter()) {
            assert!((got - want).norm() < 1e-10, "expm element {got} vs {want}");
        }
    }

    #[test]
    fn test_build_circuit_emits_gates() {
        let template = CircuitTemplate {
            structure: vec![
                ParameterizedGate {
                    gate_name: "RY".to_string(),
                    qubits: vec![0],
                    parameter_indices: vec![0],
                    fixed_parameters: Vec::new(),
                },
                ParameterizedGate {
                    gate_name: "CNOT".to_string(),
                    qubits: vec![0, 1],
                    parameter_indices: Vec::new(),
                    fixed_parameters: Vec::new(),
                },
            ],
            parameters: vec![Parameter {
                name: "theta".to_string(),
                lower_bound: 0.0,
                upper_bound: 2.0 * std::f64::consts::PI,
                initial_value: 0.3,
                discrete: false,
            }],
            num_qubits: 2,
        };
        let config = OptimizationConfig::default();
        let optimizer = QuantumCircuitOptimizer::new(template, config);
        let circuit = optimizer.build_circuit(&[0.3]).expect("build");
        // Previously returned an empty circuit; must now contain both gates.
        assert_eq!(circuit.gates().len(), 2);
    }

    #[test]
    fn test_circuit_template() {
        let template = CircuitTemplate {
            structure: vec![ParameterizedGate {
                gate_name: "RY".to_string(),
                qubits: vec![0],
                parameter_indices: vec![0],
                fixed_parameters: Vec::new(),
            }],
            parameters: vec![Parameter {
                name: "theta".to_string(),
                lower_bound: 0.0,
                upper_bound: 2.0 * std::f64::consts::PI,
                initial_value: 0.0,
                discrete: false,
            }],
            num_qubits: 1,
        };

        assert_eq!(template.parameters.len(), 1);
        assert_eq!(template.structure.len(), 1);
    }

    struct TestObjective;

    impl ObjectiveFunction for TestObjective {
        fn evaluate(&self, parameters: &[f64]) -> f64 {
            parameters.iter().map(|x| (x - 1.0).powi(2)).sum()
        }

        fn bounds(&self) -> Vec<(f64, f64)> {
            vec![(-5.0, 5.0); 2]
        }

        fn name(&self) -> &'static str {
            "test"
        }
    }

    #[test]
    fn test_optimizer_creation() {
        let template = CircuitTemplate {
            structure: Vec::new(),
            parameters: vec![
                Parameter {
                    name: "x1".to_string(),
                    lower_bound: -5.0,
                    upper_bound: 5.0,
                    initial_value: 0.0,
                    discrete: false,
                },
                Parameter {
                    name: "x2".to_string(),
                    lower_bound: -5.0,
                    upper_bound: 5.0,
                    initial_value: 0.0,
                    discrete: false,
                },
            ],
            num_qubits: 1,
        };

        let config = OptimizationConfig::default();
        let optimizer = QuantumCircuitOptimizer::new(template, config);

        assert_eq!(optimizer.circuit_template.parameters.len(), 2);
    }
}
