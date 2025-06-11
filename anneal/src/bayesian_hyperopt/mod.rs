//! Bayesian Optimization for Hyperparameter Tuning
//!
//! This module implements advanced Bayesian optimization techniques for automatically
//! tuning hyperparameters in quantum annealing systems. It uses Gaussian processes
//! as surrogate models and sophisticated acquisition functions to efficiently explore
//! the hyperparameter space.
//!
//! Key features:
//! - Multi-objective Bayesian optimization
//! - Mixed parameter types (continuous, discrete, categorical)
//! - Advanced acquisition functions (EI, UCB, PI, Entropy Search)
//! - Gaussian process surrogate models with different kernels
//! - Constraint handling and feasibility modeling
//! - Transfer learning across related optimization problems
//! - Parallel and batch optimization
//! - Uncertainty quantification and confidence intervals

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::embedding::{Embedding, HardwareTopology};
use crate::hardware_compilation::{CompilerConfig, HardwareCompiler};
use crate::ising::IsingModel;
use crate::simulator::{AnnealingParams, AnnealingResult, ClassicalAnnealingSimulator};

// Module declarations
pub mod acquisition;
pub mod config;
pub mod constraints;
pub mod convergence;
pub mod gaussian_process;
pub mod multi_objective;
pub mod parallel;
pub mod transfer;

// Re-export main types for backward compatibility
pub use acquisition::*;
pub use config::*;
pub use constraints::*;
pub use convergence::*;
pub use gaussian_process::*;
pub use multi_objective::*;
pub use parallel::*;
pub use transfer::*;

/// Create parameter space for annealing hyperparameters
pub fn create_annealing_parameter_space() -> ParameterSpace {
    let mut parameters = Vec::new();
    
    // Common annealing parameters
    parameters.push(Parameter {
        name: "temperature".to_string(),
        param_type: ParameterType::Continuous,
        bounds: ParameterBounds::Continuous { min: 0.1, max: 10.0 },
    });
    
    parameters.push(Parameter {
        name: "num_sweeps".to_string(),
        param_type: ParameterType::Discrete,
        bounds: ParameterBounds::Discrete { min: 100, max: 10000 },
    });
    
    parameters.push(Parameter {
        name: "schedule_type".to_string(),
        param_type: ParameterType::Categorical,
        bounds: ParameterBounds::Categorical { 
            values: vec!["linear".to_string(), "exponential".to_string(), "polynomial".to_string()]
        },
    });
    
    ParameterSpace { parameters }
}

/// Create Bayesian optimizer with default configuration
pub fn create_bayesian_optimizer() -> BayesianHyperoptimizer {
    let config = BayesianOptConfig {
        max_iterations: 50,
        initial_samples: 5,
        acquisition_config: AcquisitionConfig {
            function_type: AcquisitionFunctionType::ExpectedImprovement,
            exploration_factor: 0.1,
            num_restarts: 10,
            batch_strategy: BatchAcquisitionStrategy::LocalPenalization,
            optimization_method: AcquisitionOptimizationMethod::RandomSearch,
        },
        gp_config: GaussianProcessSurrogate {
            kernel: KernelFunction::RBF,
            noise_variance: 1e-6,
            mean_function: MeanFunction::Zero,
        },
        multi_objective_config: MultiObjectiveConfig::default(),
        constraint_config: ConstraintConfig::default(),
        convergence_config: ConvergenceConfig::default(),
        parallel_config: ParallelConfig::default(),
        transfer_config: TransferConfig::default(),
        seed: Some(42),
    };
    
    let parameter_space = create_annealing_parameter_space();
    BayesianHyperoptimizer::new(config, parameter_space)
}

/// Create custom Bayesian optimizer with specified parameters
pub fn create_custom_bayesian_optimizer(
    max_iterations: usize,
    acquisition_function: AcquisitionFunctionType,
    kernel: KernelFunction,
) -> BayesianHyperoptimizer {
    let config = BayesianOptConfig {
        max_iterations,
        initial_samples: (max_iterations / 10).max(3),
        acquisition_config: AcquisitionConfig {
            function_type: acquisition_function,
            exploration_factor: 0.1,
            num_restarts: 10,
            batch_strategy: BatchAcquisitionStrategy::LocalPenalization,
            optimization_method: AcquisitionOptimizationMethod::RandomSearch,
        },
        gp_config: GaussianProcessSurrogate {
            kernel,
            noise_variance: 1e-6,
            mean_function: MeanFunction::Zero,
        },
        multi_objective_config: MultiObjectiveConfig::default(),
        constraint_config: ConstraintConfig::default(),
        convergence_config: ConvergenceConfig::default(),
        parallel_config: ParallelConfig::default(),
        transfer_config: TransferConfig::default(),
        seed: Some(42),
    };
    
    let parameter_space = create_annealing_parameter_space();
    BayesianHyperoptimizer::new(config, parameter_space)
}

/// Optimize annealing parameters for a given problem
pub fn optimize_annealing_parameters<F>(
    objective_function: F,
    max_iterations: Option<usize>,
) -> BayesianOptResult<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let mut optimizer = if let Some(max_iter) = max_iterations {
        create_custom_bayesian_optimizer(
            max_iter,
            AcquisitionFunctionType::ExpectedImprovement,
            KernelFunction::RBF,
        )
    } else {
        create_bayesian_optimizer()
    };
    
    optimizer.optimize(objective_function)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parameter_space_creation() {
        let param_space = create_annealing_parameter_space();
        assert_eq!(param_space.parameters.len(), 3);
        
        // Check parameter types
        assert_eq!(param_space.parameters[0].param_type, ParameterType::Continuous);
        assert_eq!(param_space.parameters[1].param_type, ParameterType::Discrete);
        assert_eq!(param_space.parameters[2].param_type, ParameterType::Categorical);
    }
    
    #[test]
    fn test_optimizer_creation() {
        let optimizer = create_bayesian_optimizer();
        assert_eq!(optimizer.config.max_iterations, 50);
        assert_eq!(optimizer.config.initial_samples, 5);
        assert_eq!(optimizer.parameter_space.parameters.len(), 3);
    }
    
    #[test]
    fn test_custom_optimizer_creation() {
        let optimizer = create_custom_bayesian_optimizer(
            100,
            AcquisitionFunctionType::UpperConfidenceBound,
            KernelFunction::Matern,
        );
        
        assert_eq!(optimizer.config.max_iterations, 100);
        assert_eq!(optimizer.config.initial_samples, 10);
        assert_eq!(
            optimizer.config.acquisition_config.function_type,
            AcquisitionFunctionType::UpperConfidenceBound
        );
        assert_eq!(optimizer.config.gp_config.kernel, KernelFunction::Matern);
    }
    
    #[test]
    fn test_simple_optimization() {
        // Simple quadratic function to minimize
        let objective = |x: &[f64]| {
            x.iter().map(|&xi| (xi - 1.0).powi(2)).sum::<f64>()
        };
        
        let result = optimize_annealing_parameters(objective, Some(10));
        assert!(result.is_ok());
        
        let best_params = result.unwrap();
        assert_eq!(best_params.len(), 3);
        
        // Check that parameters are within bounds
        assert!(best_params[0] >= 0.1 && best_params[0] <= 10.0); // temperature
        assert!(best_params[1] >= 100.0 && best_params[1] <= 10000.0); // num_sweeps
        assert!(best_params[2] >= 0.0 && best_params[2] <= 2.0); // schedule_type (categorical index)
    }
}
