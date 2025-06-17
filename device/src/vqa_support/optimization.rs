//! Optimization algorithms and strategies for VQA

use super::config::*;
use crate::DeviceResult;
use ndarray::{Array1, Array2, ArrayView1};
use std::collections::HashMap;
use std::time::Duration;

// SciRS2 imports with fallback
#[cfg(not(feature = "scirs2"))]
use super::fallback_scirs2;
#[cfg(feature = "scirs2")]
use super::scirs2_optimize;

/// Optimization problem definition
pub struct OptimizationProblem {
    pub ansatz: super::circuits::ParametricCircuit,
    pub objective_function: Box<dyn super::objectives::ObjectiveFunction>,
    pub bounds: Option<Vec<(f64, f64)>>,
    pub constraints: Vec<OptimizationConstraint>,
}

impl OptimizationProblem {
    /// Create new optimization problem
    pub fn new(
        ansatz: super::circuits::ParametricCircuit,
        objective_function: Box<dyn super::objectives::ObjectiveFunction>,
    ) -> Self {
        Self {
            ansatz,
            objective_function,
            bounds: None,
            constraints: Vec::new(),
        }
    }

    /// Add parameter bounds
    pub fn with_bounds(mut self, bounds: Vec<(f64, f64)>) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Add optimization constraint
    pub fn with_constraint(mut self, constraint: OptimizationConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Evaluate objective function at given parameters
    pub fn evaluate_objective(&self, parameters: &Array1<f64>) -> DeviceResult<f64> {
        // For now, return a simple quadratic function as placeholder
        // In practice, this would involve circuit execution and measurement
        Ok(parameters.iter().map(|&x| x.powi(2)).sum::<f64>())
    }

    /// Compute gradient using parameter shift rule
    pub fn compute_gradient(&self, parameters: &Array1<f64>) -> DeviceResult<Array1<f64>> {
        let n_params = parameters.len();
        let mut gradient = Array1::zeros(n_params);
        let shift = std::f64::consts::PI / 2.0;

        for i in 0..n_params {
            let mut params_plus = parameters.clone();
            let mut params_minus = parameters.clone();

            params_plus[i] += shift;
            params_minus[i] -= shift;

            let f_plus = self.evaluate_objective(&params_plus)?;
            let f_minus = self.evaluate_objective(&params_minus)?;

            gradient[i] = (f_plus - f_minus) / 2.0;
        }

        Ok(gradient)
    }

    /// Check if parameters satisfy constraints
    pub fn check_constraints(&self, parameters: &Array1<f64>) -> bool {
        for constraint in &self.constraints {
            if !constraint.is_satisfied(parameters) {
                return false;
            }
        }
        true
    }
}

/// Optimization constraint definition
#[derive(Debug, Clone)]
pub struct OptimizationConstraint {
    pub constraint_type: ConstraintType,
    pub bounds: Vec<f64>,
    pub tolerance: f64,
}

impl OptimizationConstraint {
    /// Create equality constraint
    pub fn equality(target: f64, tolerance: f64) -> Self {
        Self {
            constraint_type: ConstraintType::Equality,
            bounds: vec![target],
            tolerance,
        }
    }

    /// Create inequality constraint
    pub fn inequality(upper_bound: f64) -> Self {
        Self {
            constraint_type: ConstraintType::Inequality,
            bounds: vec![upper_bound],
            tolerance: 0.0,
        }
    }

    /// Create bounds constraint
    pub fn bounds(lower: f64, upper: f64) -> Self {
        Self {
            constraint_type: ConstraintType::Bounds,
            bounds: vec![lower, upper],
            tolerance: 0.0,
        }
    }

    /// Check if parameters satisfy this constraint
    pub fn is_satisfied(&self, parameters: &Array1<f64>) -> bool {
        match self.constraint_type {
            ConstraintType::Equality => {
                if self.bounds.is_empty() {
                    return true;
                }
                let target = self.bounds[0];
                let value = parameters.sum(); // Simplified constraint evaluation
                (value - target).abs() <= self.tolerance
            }
            ConstraintType::Inequality => {
                if self.bounds.is_empty() {
                    return true;
                }
                let upper = self.bounds[0];
                let value = parameters.sum(); // Simplified constraint evaluation
                value <= upper + self.tolerance
            }
            ConstraintType::Bounds => {
                if self.bounds.len() < 2 {
                    return true;
                }
                let lower = self.bounds[0];
                let upper = self.bounds[1];
                parameters.iter().all(|&x| x >= lower && x <= upper)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    Equality,
    Inequality,
    Bounds,
}

/// Internal optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub optimal_parameters: Array1<f64>,
    pub optimal_value: f64,
    pub success: bool,
    pub num_iterations: usize,
    pub num_function_evaluations: usize,
    pub message: String,
    pub optimization_time: Duration,
    pub trajectory: OptimizationTrajectory,
    pub num_restarts: usize,
    pub optimizer_comparison: OptimizerComparison,
}

impl OptimizationResult {
    /// Create new optimization result
    pub fn new(optimal_parameters: Array1<f64>, optimal_value: f64, success: bool) -> Self {
        Self {
            optimal_parameters,
            optimal_value,
            success,
            num_iterations: 0,
            num_function_evaluations: 0,
            message: String::new(),
            optimization_time: Duration::from_secs(0),
            trajectory: OptimizationTrajectory::new(),
            num_restarts: 0,
            optimizer_comparison: OptimizerComparison::new(),
        }
    }

    /// Add iteration information
    pub fn with_iterations(mut self, iterations: usize, evaluations: usize) -> Self {
        self.num_iterations = iterations;
        self.num_function_evaluations = evaluations;
        self
    }

    /// Add timing information
    pub fn with_timing(mut self, duration: Duration) -> Self {
        self.optimization_time = duration;
        self
    }

    /// Add trajectory information
    pub fn with_trajectory(mut self, trajectory: OptimizationTrajectory) -> Self {
        self.trajectory = trajectory;
        self
    }

    /// Check if optimization was successful and converged
    pub fn is_converged(&self) -> bool {
        self.success && self.trajectory.convergence_indicators.objective_convergence
    }

    /// Get convergence rate
    pub fn convergence_rate(&self) -> f64 {
        if self.trajectory.objective_history.len() < 2 {
            return 0.0;
        }

        let initial = self.trajectory.objective_history[0];
        let final_val = self.optimal_value;

        if initial != 0.0 {
            (initial - final_val).abs() / initial.abs()
        } else {
            0.0
        }
    }
}

/// Optimizer comparison results
#[derive(Debug, Clone)]
pub struct OptimizerComparison {
    pub optimizer_results: HashMap<String, OptimizerPerformance>,
    pub best_optimizer: Option<String>,
    pub ranking: Vec<String>,
}

impl OptimizerComparison {
    pub fn new() -> Self {
        Self {
            optimizer_results: HashMap::new(),
            best_optimizer: None,
            ranking: Vec::new(),
        }
    }

    /// Add optimizer performance result
    pub fn add_result(&mut self, optimizer_name: String, performance: OptimizerPerformance) {
        let best_value = performance.best_value;
        self.optimizer_results
            .insert(optimizer_name.clone(), performance);

        // Update best optimizer
        if let Some(ref current_best) = self.best_optimizer {
            let current_best_value = self.optimizer_results[current_best].best_value;
            if best_value < current_best_value {
                self.best_optimizer = Some(optimizer_name);
            }
        } else {
            self.best_optimizer = Some(optimizer_name);
        }

        // Update ranking
        self.update_ranking();
    }

    /// Update optimizer ranking
    fn update_ranking(&mut self) {
        let mut optimizers: Vec<(String, f64)> = self
            .optimizer_results
            .iter()
            .map(|(name, perf)| (name.clone(), perf.best_value))
            .collect();

        optimizers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        self.ranking = optimizers.into_iter().map(|(name, _)| name).collect();
    }
}

/// Performance metrics for individual optimizers
#[derive(Debug, Clone)]
pub struct OptimizerPerformance {
    pub best_value: f64,
    pub convergence_iterations: usize,
    pub total_evaluations: usize,
    pub execution_time: Duration,
    pub success_rate: f64,
    pub robustness_score: f64,
}

impl OptimizerPerformance {
    pub fn new(best_value: f64) -> Self {
        Self {
            best_value,
            convergence_iterations: 0,
            total_evaluations: 0,
            execution_time: Duration::from_secs(0),
            success_rate: 0.0,
            robustness_score: 0.0,
        }
    }
}

/// Optimization strategies and utilities
pub struct OptimizationStrategy {
    pub config: VQAOptimizationConfig,
}

impl OptimizationStrategy {
    /// Create new optimization strategy
    pub fn new(config: VQAOptimizationConfig) -> Self {
        Self { config }
    }

    /// Generate initial parameters using specified strategy
    pub fn generate_initial_parameters(&self, num_params: usize) -> DeviceResult<Array1<f64>> {
        match self.config.multi_start_config.initial_point_strategy {
            InitialPointStrategy::Random => self.generate_random_initial(num_params),
            InitialPointStrategy::LatinHypercube => self.generate_latin_hypercube(num_params),
            InitialPointStrategy::Sobol => self.generate_sobol_sequence(num_params),
            InitialPointStrategy::Grid => self.generate_grid_points(num_params),
            InitialPointStrategy::PreviousBest => self.generate_from_previous_best(num_params),
            InitialPointStrategy::AdaptiveSampling => self.generate_adaptive_sample(num_params),
        }
    }

    /// Generate random initial parameters
    fn generate_random_initial(&self, num_params: usize) -> DeviceResult<Array1<f64>> {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        let params = Array1::from_shape_fn(num_params, |_| {
            rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI)
        });

        Ok(params)
    }

    /// Generate Latin Hypercube sampling initial parameters
    fn generate_latin_hypercube(&self, num_params: usize) -> DeviceResult<Array1<f64>> {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        // Simplified Latin Hypercube sampling
        let mut indices: Vec<usize> = (0..num_params).collect();
        indices.shuffle(&mut rng);

        let params = Array1::from_shape_fn(num_params, |i| {
            let segment = indices[i] as f64 / num_params as f64;
            let offset = rng.gen::<f64>() / num_params as f64;
            let uniform_sample = segment + offset;

            // Scale to parameter range
            -std::f64::consts::PI + 2.0 * std::f64::consts::PI * uniform_sample
        });

        Ok(params)
    }

    /// Generate Sobol sequence initial parameters
    fn generate_sobol_sequence(&self, num_params: usize) -> DeviceResult<Array1<f64>> {
        // Simplified Sobol sequence (in practice would use proper implementation)
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        let params = Array1::from_shape_fn(num_params, |i| {
            let sobol_val = (i as f64 + 0.5) / num_params as f64;
            let jittered = sobol_val + rng.gen::<f64>() * 0.1 - 0.05;
            -std::f64::consts::PI + 2.0 * std::f64::consts::PI * jittered.clamp(0.0, 1.0)
        });

        Ok(params)
    }

    /// Generate grid points for initial parameters
    fn generate_grid_points(&self, num_params: usize) -> DeviceResult<Array1<f64>> {
        let grid_size = (num_params as f64).powf(1.0 / num_params as f64).ceil() as usize;

        let params = Array1::from_shape_fn(num_params, |i| {
            let grid_pos = i % grid_size;
            let grid_val = grid_pos as f64 / (grid_size - 1).max(1) as f64;
            -std::f64::consts::PI + 2.0 * std::f64::consts::PI * grid_val
        });

        Ok(params)
    }

    /// Generate initial parameters from previous best results
    fn generate_from_previous_best(&self, num_params: usize) -> DeviceResult<Array1<f64>> {
        // Placeholder: would use stored best parameters with perturbation
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        let params = Array1::from_shape_fn(num_params, |_| {
            rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI)
        });

        Ok(params)
    }

    /// Generate adaptive sampling initial parameters
    fn generate_adaptive_sample(&self, num_params: usize) -> DeviceResult<Array1<f64>> {
        // Placeholder: would use adaptive sampling based on problem characteristics
        self.generate_latin_hypercube(num_params)
    }

    /// Execute optimization with fallback strategy
    pub fn execute_optimization(
        &self,
        problem: &OptimizationProblem,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<OptimizationResult> {
        let mut best_result = None;
        let mut comparison = OptimizerComparison::new();

        // Try primary optimizer
        let primary_result =
            self.run_single_optimizer(&self.config.primary_optimizer, problem, initial_params)?;

        let primary_performance = OptimizerPerformance::new(primary_result.optimal_value);
        comparison.add_result(
            format!("{}", self.config.primary_optimizer),
            primary_performance,
        );

        best_result = Some(primary_result);

        // Try fallback optimizers if primary fails or performance is poor
        for fallback_optimizer in &self.config.fallback_optimizers {
            let fallback_result =
                self.run_single_optimizer(fallback_optimizer, problem, initial_params)?;

            let fallback_performance = OptimizerPerformance::new(fallback_result.optimal_value);
            comparison.add_result(format!("{}", fallback_optimizer), fallback_performance);

            if let Some(ref current_best) = best_result {
                if fallback_result.optimal_value < current_best.optimal_value {
                    best_result = Some(fallback_result);
                }
            }
        }

        let mut final_result = best_result.ok_or_else(|| {
            crate::DeviceError::OptimizationError("No optimizer succeeded".to_string())
        })?;

        final_result.optimizer_comparison = comparison;
        Ok(final_result)
    }

    /// Run single optimizer on the problem
    fn run_single_optimizer(
        &self,
        optimizer: &VQAOptimizer,
        problem: &OptimizationProblem,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<OptimizationResult> {
        let start_time = std::time::Instant::now();

        // Simplified optimization logic - in practice would call SciRS2 optimizers
        let result = match optimizer {
            VQAOptimizer::LBFGSB => self.run_lbfgsb(problem, initial_params),
            VQAOptimizer::COBYLA => self.run_cobyla(problem, initial_params),
            VQAOptimizer::NelderMead => self.run_nelder_mead(problem, initial_params),
            VQAOptimizer::DifferentialEvolution => {
                self.run_differential_evolution(problem, initial_params)
            }
            _ => self.run_fallback_optimizer(problem, initial_params),
        }?;

        let duration = start_time.elapsed();
        Ok(result.with_timing(duration))
    }

    /// Run L-BFGS-B optimizer
    fn run_lbfgsb(
        &self,
        problem: &OptimizationProblem,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<OptimizationResult> {
        #[cfg(feature = "scirs2")]
        {
            use super::scirs2_optimize::{minimize, unconstrained::Method};

            // Define objective function closure
            let mut objective = |x: &ndarray::ArrayView1<f64>| -> f64 {
                let owned_array = Array1::from_vec(x.to_vec());
                problem
                    .evaluate_objective(&owned_array)
                    .unwrap_or(f64::INFINITY)
            };

            // Set up options with bounds if available
            let options = if let Some(ref bounds) = problem.bounds {
                use super::scirs2_optimize::unconstrained::{Bounds, Options};
                let bound_pairs: Vec<(Option<f64>, Option<f64>)> = bounds
                    .iter()
                    .map(|(low, high)| (Some(*low), Some(*high)))
                    .collect();
                let scirs2_bounds = Bounds::new(&bound_pairs);
                Some(scirs2_optimize::prelude::Options {
                    bounds: Some(scirs2_bounds),
                    max_iter: self.config.max_iterations,
                    ftol: self.config.convergence_tolerance,
                    ..Default::default()
                })
            } else {
                Some(scirs2_optimize::prelude::Options {
                    max_iter: self.config.max_iterations,
                    ftol: self.config.convergence_tolerance,
                    ..Default::default()
                })
            };

            // Configure L-BFGS-B optimization
            let result = minimize(
                objective,
                &initial_params.as_slice().unwrap(),
                Method::LBFGSB,
                options,
            )
            .map_err(|e| {
                crate::DeviceError::OptimizationError(format!("L-BFGS-B failed: {}", e))
            })?;

            let mut trajectory = OptimizationTrajectory::new();
            trajectory.objective_history = Array1::from(vec![result.fun]);
            trajectory.convergence_indicators.objective_convergence = result.success;

            Ok(
                OptimizationResult::new(result.x, result.fun, result.success)
                    .with_iterations(result.nit, result.nfev)
                    .with_trajectory(trajectory),
            )
        }

        #[cfg(not(feature = "scirs2"))]
        {
            // Fallback implementation with simple gradient descent
            let mut params = initial_params.clone();
            let mut value = problem.evaluate_objective(&params)?;
            let learning_rate = 0.01;
            let max_iterations = self.config.max_iterations;
            let mut history = Vec::new();

            for iteration in 0..max_iterations {
                history.push(value);

                let gradient = problem.compute_gradient(&params)?;
                let gradient_norm = gradient.iter().map(|&x| x * x).sum::<f64>().sqrt();

                if gradient_norm < self.config.convergence_tolerance {
                    break;
                }

                // Simple line search
                let step_size = learning_rate / (1.0 + iteration as f64 * 0.01);
                params = &params - &(&gradient * step_size);
                value = problem.evaluate_objective(&params)?;

                if value < self.config.convergence_tolerance {
                    break;
                }
            }

            let mut trajectory = OptimizationTrajectory::new();
            trajectory.objective_history = Array1::from(history);
            trajectory.convergence_indicators.objective_convergence =
                value < self.config.convergence_tolerance;

            Ok(OptimizationResult::new(params, value, true)
                .with_iterations(history.len(), history.len())
                .with_trajectory(trajectory))
        }
    }

    /// Run COBYLA optimizer
    fn run_cobyla(
        &self,
        problem: &OptimizationProblem,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<OptimizationResult> {
        #[cfg(feature = "scirs2")]
        {
            use super::scirs2_optimize::{minimize, unconstrained::Method};

            // Define objective function closure
            let mut objective = |x: &ndarray::ArrayView1<f64>| -> f64 {
                let owned_array = Array1::from_vec(x.to_vec());
                problem
                    .evaluate_objective(&owned_array)
                    .unwrap_or(f64::INFINITY)
            };

            let result = minimize(
                objective,
                &initial_params.as_slice().unwrap(),
                Method::NelderMead,
                None,
            )
            .map_err(|e| {
                crate::DeviceError::OptimizationError(format!("Optimization failed: {}", e))
            })?;

            let mut trajectory = OptimizationTrajectory::new();
            trajectory.objective_history = Array1::from(vec![result.fun]);
            trajectory.convergence_indicators.objective_convergence = result.success;

            Ok(
                OptimizationResult::new(result.x, result.fun, result.success)
                    .with_iterations(result.nit, result.nfev)
                    .with_trajectory(trajectory),
            )
        }

        #[cfg(not(feature = "scirs2"))]
        {
            // Fallback implementation with constrained optimization
            let mut params = initial_params.clone();
            let mut value = problem.evaluate_objective(&params)?;
            let mut history = Vec::new();
            let penalty_weight = 1000.0;

            for iteration in 0..self.config.max_iterations {
                history.push(value);

                // Compute penalty for constraint violations
                let mut penalty = 0.0;
                for constraint in &problem.constraints {
                    if !constraint.is_satisfied(&params) {
                        penalty += penalty_weight;
                    }
                }

                let penalized_value = value + penalty;

                if penalized_value < self.config.convergence_tolerance {
                    break;
                }

                // Simple random perturbation with constraint projection
                use rand::prelude::*;
                let mut rng = rand::thread_rng();

                for param in params.iter_mut() {
                    *param += rng.gen_range(-0.1..0.1);
                }

                // Project back to feasible region
                if let Some(ref bounds) = problem.bounds {
                    for (i, param) in params.iter_mut().enumerate() {
                        if i < bounds.len() {
                            *param = param.clamp(bounds[i].0, bounds[i].1);
                        }
                    }
                }

                value = problem.evaluate_objective(&params)?;
            }

            let mut trajectory = OptimizationTrajectory::new();
            trajectory.objective_history = Array1::from(history);
            trajectory.convergence_indicators.objective_convergence =
                value < self.config.convergence_tolerance;

            Ok(OptimizationResult::new(params, value, true)
                .with_iterations(history.len(), history.len())
                .with_trajectory(trajectory))
        }
    }

    /// Run Nelder-Mead optimizer
    fn run_nelder_mead(
        &self,
        problem: &OptimizationProblem,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<OptimizationResult> {
        // Placeholder implementation
        let optimal_value = problem.evaluate_objective(initial_params)?;
        Ok(OptimizationResult::new(
            initial_params.clone(),
            optimal_value,
            true,
        ))
    }

    /// Run Differential Evolution optimizer
    fn run_differential_evolution(
        &self,
        problem: &OptimizationProblem,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<OptimizationResult> {
        #[cfg(feature = "scirs2")]
        {
            use super::scirs2_optimize::{
                differential_evolution, global::DifferentialEvolutionOptions,
            };

            // Define objective function closure
            let objective = |x: &ArrayView1<f64>| -> f64 {
                let array1 = x.to_owned();
                problem.evaluate_objective(&array1).unwrap_or(f64::INFINITY)
            };

            // Set up bounds - DE requires bounds
            let bounds = if let Some(ref bounds) = problem.bounds {
                bounds.clone()
            } else {
                // Default parameter bounds for quantum circuits
                vec![(-std::f64::consts::PI, std::f64::consts::PI); initial_params.len()]
            };

            // Configure DE parameters
            let de_params = DifferentialEvolutionOptions {
                popsize: (initial_params.len() * 15).max(30), // Population size heuristic
                mutation: (0.5, 1.0),                         // Mutation factor range
                recombination: 0.7,                           // Crossover probability
                seed: None,
                atol: self.config.convergence_tolerance,
                tol: self.config.convergence_tolerance,
                maxiter: self.config.max_iterations,
                polish: true, // Polish the best result
                init: "latinhypercube".to_string(),
                updating: "immediate".to_string(),
                x0: Some(initial_params.clone()),
                parallel: None,
            };

            let result = differential_evolution(
                objective,
                bounds,
                Some(de_params),
                None, // No label
            )
            .map_err(|e| {
                crate::DeviceError::OptimizationError(format!(
                    "Differential Evolution failed: {}",
                    e
                ))
            })?;

            let mut trajectory = OptimizationTrajectory::new();
            trajectory.objective_history = Array1::from(vec![result.fun]);
            trajectory.convergence_indicators.objective_convergence = result.success;

            Ok(
                OptimizationResult::new(result.x, result.fun, result.success)
                    .with_iterations(result.nit, result.nfev)
                    .with_trajectory(trajectory),
            )
        }

        #[cfg(not(feature = "scirs2"))]
        {
            // Fallback implementation with simplified differential evolution
            let num_params = initial_params.len();
            let population_size = (num_params * 10).max(20);
            let mutation_factor = 0.8;
            let crossover_prob = 0.7;

            // Initialize population
            use rand::prelude::*;
            let mut rng = rand::thread_rng();
            let mut population = Vec::new();
            let mut fitness = Vec::new();

            // Get bounds
            let bounds = if let Some(ref bounds) = problem.bounds {
                bounds.clone()
            } else {
                vec![(-std::f64::consts::PI, std::f64::consts::PI); num_params]
            };

            // Initialize population randomly within bounds
            for _ in 0..population_size {
                let individual =
                    Array1::from_shape_fn(num_params, |i| rng.gen_range(bounds[i].0..bounds[i].1));
                let fit = problem.evaluate_objective(&individual)?;
                population.push(individual);
                fitness.push(fit);
            }

            let mut best_idx = 0;
            let mut best_fitness = fitness[0];
            for (i, &fit) in fitness.iter().enumerate() {
                if fit < best_fitness {
                    best_fitness = fit;
                    best_idx = i;
                }
            }

            let mut history = vec![best_fitness];

            // Evolution loop
            for _generation in 0..self.config.max_iterations {
                for i in 0..population_size {
                    // Select three random individuals different from current
                    let mut indices: Vec<usize> =
                        (0..population_size).filter(|&x| x != i).collect();
                    indices.shuffle(&mut rng);
                    let (a, b, c) = (indices[0], indices[1], indices[2]);

                    // Mutation: v = a + F * (b - c)
                    let mut mutant =
                        &population[a] + &((&population[b] - &population[c]) * mutation_factor);

                    // Apply bounds
                    for (j, param) in mutant.iter_mut().enumerate() {
                        *param = param.clamp(bounds[j].0, bounds[j].1);
                    }

                    // Crossover
                    let mut trial = population[i].clone();
                    let crossover_point = rng.gen_range(0..num_params);
                    for j in 0..num_params {
                        if rng.gen::<f64>() < crossover_prob || j == crossover_point {
                            trial[j] = mutant[j];
                        }
                    }

                    // Selection
                    let trial_fitness = problem.evaluate_objective(&trial)?;
                    if trial_fitness < fitness[i] {
                        population[i] = trial;
                        fitness[i] = trial_fitness;

                        if trial_fitness < best_fitness {
                            best_fitness = trial_fitness;
                            best_idx = i;
                        }
                    }
                }

                history.push(best_fitness);

                if best_fitness < self.config.convergence_tolerance {
                    break;
                }
            }

            let mut trajectory = OptimizationTrajectory::new();
            trajectory.objective_history = Array1::from(history);
            trajectory.convergence_indicators.objective_convergence =
                best_fitness < self.config.convergence_tolerance;

            Ok(
                OptimizationResult::new(population[best_idx].clone(), best_fitness, true)
                    .with_iterations(
                        trajectory.objective_history.len(),
                        trajectory.objective_history.len() * population_size,
                    )
                    .with_trajectory(trajectory),
            )
        }
    }

    /// Run fallback optimizer
    fn run_fallback_optimizer(
        &self,
        problem: &OptimizationProblem,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<OptimizationResult> {
        // Simple gradient descent fallback
        let mut params = initial_params.clone();
        let mut value = problem.evaluate_objective(&params)?;
        let learning_rate = 0.01;
        let max_iterations = 100;

        for _ in 0..max_iterations {
            let gradient = problem.compute_gradient(&params)?;
            params = &params - &(&gradient * learning_rate);
            let new_value = problem.evaluate_objective(&params)?;

            if (value - new_value).abs() < self.config.convergence_tolerance {
                break;
            }
            value = new_value;
        }

        Ok(OptimizationResult::new(params, value, true))
    }
}
