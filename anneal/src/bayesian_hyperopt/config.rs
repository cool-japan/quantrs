//! Bayesian Optimization Configuration Types

use crate::ising::IsingError;
use thiserror::Error;

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
}

impl Default for BayesianHyperoptimizer {
    fn default() -> Self {
        Self {
            config: BayesianOptConfig::default(),
            parameter_space: ParameterSpace::default(),
            history: OptimizationHistory::default(),
        }
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
