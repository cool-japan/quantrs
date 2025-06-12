//! Bayesian Optimization Configuration Types

use crate::ising::IsingError;
use thiserror::Error;
use super::gaussian_process::{GaussianProcessSurrogate, KernelFunction};
use rand_chacha::ChaCha8Rng;

/// Errors that can occur in Bayesian optimization
#[derive(Error, Debug)]
pub enum BayesianOptError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Optimization error
    #[error("Optimization error: {0}")]
    OptimizationError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Gaussian process error
    #[error("Gaussian process error: {0}")]
    GaussianProcessError(String),

    /// Acquisition function error
    #[error("Acquisition function error: {0}")]
    AcquisitionFunctionError(String),

    /// Constraint handling error
    #[error("Constraint handling error: {0}")]
    ConstraintError(String),

    /// Transfer learning error
    #[error("Transfer learning error: {0}")]
    TransferLearningError(String),

    /// Convergence error
    #[error("Convergence error: {0}")]
    ConvergenceError(String),
}

/// Result type for Bayesian optimization operations
pub type BayesianOptResult<T> = Result<T, BayesianOptError>;

/// Parameter types
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterType {
    Continuous,
    Discrete,
    Categorical,
}

/// Parameter value
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterValue {
    Continuous(f64),
    Discrete(i64),
    Categorical(String),
}

/// Parameter definition
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub param_type: ParameterType,
    pub bounds: ParameterBounds,
}

/// Parameter bounds
#[derive(Debug, Clone)]
pub enum ParameterBounds {
    Continuous { min: f64, max: f64 },
    Discrete { min: i64, max: i64 },
    Categorical { values: Vec<String> },
}

/// Parameter space definition
#[derive(Debug, Clone)]
pub struct ParameterSpace {
    pub parameters: Vec<Parameter>,
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self {
            parameters: Vec::new(),
        }
    }
}

/// Constraint handling methods (re-exported from constraints module)
pub use super::constraints::ConstraintHandlingMethod;

/// Scalarization methods (re-exported from multi_objective module)
pub use super::multi_objective::ScalarizationMethod;

/// Optimization history
#[derive(Debug, Clone)]
pub struct OptimizationHistory {
    pub evaluations: Vec<(Vec<f64>, f64)>,
    pub best_values: Vec<f64>,
    pub iteration_times: Vec<f64>,
}

impl Default for OptimizationHistory {
    fn default() -> Self {
        Self {
            evaluations: Vec::new(),
            best_values: Vec::new(),
            iteration_times: Vec::new(),
        }
    }
}

/// Objective function trait
pub trait ObjectiveFunction {
    fn evaluate(&self, parameters: &[f64]) -> f64;
    fn get_bounds(&self) -> Vec<(f64, f64)>;
}

/// Bayesian optimization metrics
#[derive(Debug, Clone)]
pub struct BayesianOptMetrics {
    pub convergence_rate: f64,
    pub regret: Vec<f64>,
    pub acquisition_time: f64,
    pub gp_training_time: f64,
}

impl Default for BayesianOptMetrics {
    fn default() -> Self {
        Self {
            convergence_rate: 0.0,
            regret: Vec::new(),
            acquisition_time: 0.0,
            gp_training_time: 0.0,
        }
    }
}

/// Main Bayesian hyperoptimizer
#[derive(Debug)]
pub struct BayesianHyperoptimizer {
    pub config: BayesianOptConfig,
    pub parameter_space: ParameterSpace,
    pub history: OptimizationHistory,
    pub gp_model: Option<GaussianProcessModel>,
    pub current_best_value: f64,
    pub metrics: BayesianOptMetrics,
}

impl Default for BayesianHyperoptimizer {
    fn default() -> Self {
        Self {
            config: BayesianOptConfig::default(),
            parameter_space: ParameterSpace::default(),
            history: OptimizationHistory::default(),
            gp_model: None,
            current_best_value: f64::INFINITY,
            metrics: BayesianOptMetrics::default(),
        }
    }
}

impl BayesianHyperoptimizer {
    /// Create new Bayesian hyperoptimizer with configuration
    pub fn new(config: BayesianOptConfig, parameter_space: ParameterSpace) -> Self {
        Self {
            config,
            parameter_space,
            history: OptimizationHistory::default(),
            gp_model: None,
            current_best_value: f64::INFINITY,
            metrics: BayesianOptMetrics::default(),
        }
    }
    
    /// Run Bayesian optimization
    pub fn optimize<F>(&mut self, objective_function: F) -> BayesianOptResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        
        let mut rng = if let Some(seed) = self.config.seed {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            ChaCha8Rng::from_entropy()
        };
        
        let start_time = std::time::Instant::now();
        
        // Step 1: Generate initial random samples
        self.generate_initial_samples(&mut rng, &objective_function)?;
        
        // Step 2: Main optimization loop
        for iteration in 0..self.config.max_iterations {
            let iter_start = std::time::Instant::now();
            
            // Update Gaussian Process model
            self.update_gp_model()?;
            
            // Find next point to evaluate using acquisition function
            let next_point = self.suggest_next_point(&mut rng)?;
            
            // Evaluate objective function
            let value = objective_function(&next_point);
            
            // Update history
            self.history.evaluations.push((next_point, value));
            
            // Update best value
            if value < self.current_best_value {
                self.current_best_value = value;
            }
            self.history.best_values.push(self.current_best_value);
            
            // Record iteration time
            let iter_time = iter_start.elapsed().as_secs_f64();
            self.history.iteration_times.push(iter_time);
            
            // Check convergence
            if self.check_convergence()? {
                println!("Bayesian optimization converged after {} iterations", iteration + 1);
                break;
            }
        }
        
        // Update final metrics
        self.metrics.convergence_rate = self.calculate_convergence_rate();
        self.metrics.regret = self.calculate_regret();
        
        // Return best parameter values found
        self.get_best_parameters()
    }
    
    /// Generate initial random samples
    fn generate_initial_samples<F>(&mut self, rng: &mut ChaCha8Rng, objective_function: &F) -> BayesianOptResult<()>
    where
        F: Fn(&[f64]) -> f64,
    {
        for _ in 0..self.config.initial_samples {
            let sample = self.sample_random_point(rng)?;
            let value = objective_function(&sample);
            
            self.history.evaluations.push((sample, value));
            
            if value < self.current_best_value {
                self.current_best_value = value;
            }
            self.history.best_values.push(self.current_best_value);
        }
        
        Ok(())
    }
    
    /// Sample a random point from the parameter space
    fn sample_random_point(&self, rng: &mut ChaCha8Rng) -> BayesianOptResult<Vec<f64>> {
        let mut point = Vec::new();
        
        for param in &self.parameter_space.parameters {
            match &param.bounds {
                ParameterBounds::Continuous { min, max } => {
                    let value = rng.gen_range(*min..*max);
                    point.push(value);
                }
                ParameterBounds::Discrete { min, max } => {
                    let value = rng.gen_range(*min..*max) as f64;
                    point.push(value);
                }
                ParameterBounds::Categorical { values } => {
                    let index = rng.gen_range(0..values.len()) as f64;
                    point.push(index);
                }
            }
        }
        
        Ok(point)
    }
    
    /// Update Gaussian Process model with current data
    fn update_gp_model(&mut self) -> BayesianOptResult<()> {
        if self.history.evaluations.is_empty() {
            return Err(BayesianOptError::GaussianProcessError(
                "No data available for GP model".to_string(),
            ));
        }
        
        let gp_start = std::time::Instant::now();
        
        // Extract training data
        let x_data: Vec<Vec<f64>> = self.history.evaluations.iter().map(|(x, _)| x.clone()).collect();
        let y_data: Vec<f64> = self.history.evaluations.iter().map(|(_, y)| *y).collect();
        
        // Create or update GP model
        let model = GaussianProcessModel::new(
            x_data,
            y_data,
            self.config.gp_config.clone(),
        )?;
        
        self.gp_model = Some(model);
        self.metrics.gp_training_time = gp_start.elapsed().as_secs_f64();
        
        Ok(())
    }
    
    /// Suggest next point to evaluate using acquisition function
    fn suggest_next_point(&self, rng: &mut ChaCha8Rng) -> BayesianOptResult<Vec<f64>> {
        let acq_start = std::time::Instant::now();
        
        let gp_model = self.gp_model.as_ref().ok_or_else(|| {
            BayesianOptError::GaussianProcessError("GP model not initialized".to_string())
        })?;
        
        let mut best_point = self.sample_random_point(rng)?;
        let mut best_acquisition_value = f64::NEG_INFINITY;
        
        // Random search for acquisition function optimization
        // In practice, would use more sophisticated optimization
        for _ in 0..self.config.acquisition_config.num_restarts * 10 {
            let candidate = self.sample_random_point(rng)?;
            let acquisition_value = self.evaluate_acquisition_function(&candidate, gp_model)?;
            
            if acquisition_value > best_acquisition_value {
                best_acquisition_value = acquisition_value;
                best_point = candidate;
            }
        }
        
        // Update acquisition time metric
        let acq_time = acq_start.elapsed().as_secs_f64();
        // Note: This overwrites the previous value. In practice, you might want to accumulate or average
        self.metrics.acquisition_time = acq_time;
        
        Ok(best_point)
    }
    
    /// Evaluate acquisition function at given point
    fn evaluate_acquisition_function(&self, point: &[f64], gp_model: &GaussianProcessModel) -> BayesianOptResult<f64> {
        let (mean, variance) = gp_model.predict(point)?;
        let std_dev = variance.sqrt();
        
        match self.config.acquisition_config.function_type {
            super::AcquisitionFunctionType::ExpectedImprovement => {
                self.expected_improvement(mean, std_dev)
            }
            super::AcquisitionFunctionType::UpperConfidenceBound => {
                self.upper_confidence_bound(mean, std_dev)
            }
            super::AcquisitionFunctionType::ProbabilityOfImprovement => {
                self.probability_of_improvement(mean, std_dev)
            }
            _ => {
                // Fallback to Expected Improvement for unimplemented functions
                self.expected_improvement(mean, std_dev)
            }
        }
    }
    
    /// Expected Improvement acquisition function
    fn expected_improvement(&self, mean: f64, std_dev: f64) -> BayesianOptResult<f64> {
        if std_dev <= 1e-10 {
            return Ok(0.0);
        }
        
        let improvement = self.current_best_value - mean;
        let z = improvement / std_dev;
        
        // Approximation of normal CDF and PDF
        let phi = 0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2));
        let pdf = (1.0 / (std::f64::consts::TAU.sqrt())) * (-0.5 * z * z).exp();
        
        let ei = improvement * phi + std_dev * pdf;
        Ok(ei.max(0.0))
    }
    
    /// Upper Confidence Bound acquisition function
    fn upper_confidence_bound(&self, mean: f64, std_dev: f64) -> BayesianOptResult<f64> {
        let beta = self.config.acquisition_config.exploration_factor;
        Ok(-mean + beta * std_dev) // Negative because we're minimizing
    }
    
    /// Probability of Improvement acquisition function
    fn probability_of_improvement(&self, mean: f64, std_dev: f64) -> BayesianOptResult<f64> {
        if std_dev <= 1e-10 {
            return Ok(0.0);
        }
        
        let z = (self.current_best_value - mean) / std_dev;
        let pi = 0.5 * (1.0 + libm::erf(z / std::f64::consts::SQRT_2));
        Ok(pi)
    }
    
    /// Check convergence criteria
    fn check_convergence(&self) -> BayesianOptResult<bool> {
        if self.history.best_values.len() < 2 {
            return Ok(false);
        }
        
        // Simple convergence check: improvement in last few iterations
        let recent_window = 5.min(self.history.best_values.len());
        let recent_best = self.history.best_values[self.history.best_values.len() - recent_window..].to_vec();
        
        let improvement = recent_best.first().unwrap() - recent_best.last().unwrap();
        let relative_improvement = improvement.abs() / (recent_best.first().unwrap().abs() + 1e-10);
        
        Ok(relative_improvement < 1e-6)
    }
    
    /// Calculate convergence rate
    fn calculate_convergence_rate(&self) -> f64 {
        if self.history.best_values.len() < 2 {
            return 0.0;
        }
        
        let initial = self.history.best_values[0];
        let final_val = *self.history.best_values.last().unwrap();
        
        if initial.abs() < 1e-10 {
            return 0.0;
        }
        
        (initial - final_val) / initial.abs()
    }
    
    /// Calculate regret over time
    fn calculate_regret(&self) -> Vec<f64> {
        if self.history.best_values.is_empty() {
            return Vec::new();
        }
        
        let global_best = *self.history.best_values.last().unwrap();
        self.history.best_values.iter().map(|&v| v - global_best).collect()
    }
    
    /// Get best parameters found during optimization
    fn get_best_parameters(&self) -> BayesianOptResult<Vec<f64>> {
        self.history.evaluations
            .iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(params, _)| params.clone())
            .ok_or_else(|| BayesianOptError::OptimizationError("No evaluations found".to_string()))
    }
    
    /// Get optimization metrics
    pub fn get_metrics(&self) -> &BayesianOptMetrics {
        &self.metrics
    }
    
    /// Get optimization history
    pub fn get_history(&self) -> &OptimizationHistory {
        &self.history
    }
}

/// Configuration for Bayesian optimization
#[derive(Debug, Clone)]
pub struct BayesianOptConfig {
    /// Number of optimization iterations
    pub max_iterations: usize,
    /// Number of initial random samples
    pub initial_samples: usize,
    /// Acquisition function configuration
    pub acquisition_config: AcquisitionConfig,
    /// Gaussian process configuration
    pub gp_config: GaussianProcessConfig,
    /// Multi-objective configuration
    pub multi_objective_config: MultiObjectiveConfig,
    /// Constraint handling configuration
    pub constraint_config: ConstraintConfig,
    /// Convergence criteria
    pub convergence_config: ConvergenceConfig,
    /// Parallel optimization settings
    pub parallel_config: ParallelConfig,
    /// Transfer learning settings
    pub transfer_config: TransferConfig,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for BayesianOptConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            initial_samples: 10,
            acquisition_config: AcquisitionConfig::default(),
            gp_config: GaussianProcessConfig::default(),
            multi_objective_config: MultiObjectiveConfig::default(),
            constraint_config: ConstraintConfig::default(),
            convergence_config: ConvergenceConfig::default(),
            parallel_config: ParallelConfig::default(),
            transfer_config: TransferConfig::default(),
            seed: None,
        }
    }
}

// Forward declarations for types that will be defined in other modules
use super::{
    acquisition::AcquisitionConfig, constraints::ConstraintConfig, convergence::ConvergenceConfig,
    gaussian_process::GaussianProcessConfig, multi_objective::MultiObjectiveConfig,
    parallel::ParallelConfig, transfer::TransferConfig,
};
