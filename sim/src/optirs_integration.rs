//! `OptiRS` Integration for Quantum Variational Algorithms
//!
//! This module provides integration between `OptiRS` optimizers and `QuantRS2` variational
//! quantum algorithms (VQE, QAOA, etc.). It bridges `OptiRS`'s state-of-the-art ML
//! optimization algorithms with quantum circuit parameter optimization.
//!
//! # Features
//! - Production-ready optimizers from `OptiRS` (Adam, SGD, `RMSprop`, Adagrad)
//! - Gradient-based optimization for VQE/QAOA
//! - Learning rate scheduling
//! - Gradient clipping and regularization
//! - Hardware-aware parameter optimization
//! - Performance metrics and monitoring

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// Import OptiRS optimizers with proper type bounds
use optirs_core::optimizers::{Adagrad, Adam, Optimizer, RMSprop, SGD};

/// `OptiRS` optimizer types available for quantum optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptiRSOptimizerType {
    /// Stochastic Gradient Descent with momentum
    SGD { momentum: bool },
    /// Adam optimizer (Adaptive Moment Estimation)
    Adam,
    /// `RMSprop` optimizer
    RMSprop,
    /// Adagrad optimizer
    Adagrad,
}

/// `OptiRS` optimizer configuration for quantum algorithms
#[derive(Debug, Clone)]
pub struct OptiRSConfig {
    /// Optimizer type
    pub optimizer_type: OptiRSOptimizerType,
    /// Initial learning rate
    pub learning_rate: f64,
    /// Gradient clipping threshold
    pub gradient_clip_norm: Option<f64>,
    /// L2 regularization strength
    pub l2_regularization: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Enable parameter bounds
    pub parameter_bounds: Option<(f64, f64)>,
    /// Momentum (for SGD)
    pub momentum: f64,
}

impl Default for OptiRSConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptiRSOptimizerType::Adam,
            learning_rate: 0.01,
            gradient_clip_norm: Some(1.0),
            l2_regularization: 0.0,
            max_iterations: 1000,
            convergence_tolerance: 1e-6,
            parameter_bounds: Some((-std::f64::consts::PI, std::f64::consts::PI)),
            momentum: 0.9,
        }
    }
}

/// `OptiRS` quantum optimizer wrapper
pub struct OptiRSQuantumOptimizer {
    /// Configuration
    config: OptiRSConfig,
    /// Underlying `OptiRS` optimizer
    optimizer: OptiRSOptimizerImpl,
    /// Current iteration
    iteration: usize,
    /// Best parameters seen
    best_parameters: Option<Vec<f64>>,
    /// Best cost seen
    best_cost: f64,
    /// Optimization history
    cost_history: Vec<f64>,
    /// Gradient norms history
    gradient_norms: Vec<f64>,
}

/// Internal optimizer implementation enum
enum OptiRSOptimizerImpl {
    SGD(SGD<f64>),
    Adam(Adam<f64>),
    RMSprop(RMSprop<f64>),
    Adagrad(Adagrad<f64>),
}

impl OptiRSQuantumOptimizer {
    /// Create a new `OptiRS` quantum optimizer
    pub fn new(config: OptiRSConfig) -> Result<Self> {
        let optimizer = Self::create_optimizer(&config)?;

        Ok(Self {
            config,
            optimizer,
            iteration: 0,
            best_parameters: None,
            best_cost: f64::INFINITY,
            cost_history: Vec::new(),
            gradient_norms: Vec::new(),
        })
    }

    /// Create the underlying `OptiRS` optimizer
    fn create_optimizer(config: &OptiRSConfig) -> Result<OptiRSOptimizerImpl> {
        let optimizer = match config.optimizer_type {
            OptiRSOptimizerType::SGD { momentum } => {
                let sgd = SGD::new(config.learning_rate);
                if momentum {
                    OptiRSOptimizerImpl::SGD(sgd.with_momentum(config.momentum))
                } else {
                    OptiRSOptimizerImpl::SGD(sgd)
                }
            }
            OptiRSOptimizerType::Adam => OptiRSOptimizerImpl::Adam(Adam::new(config.learning_rate)),
            OptiRSOptimizerType::RMSprop => {
                OptiRSOptimizerImpl::RMSprop(RMSprop::new(config.learning_rate))
            }
            OptiRSOptimizerType::Adagrad => {
                OptiRSOptimizerImpl::Adagrad(Adagrad::new(config.learning_rate))
            }
        };

        Ok(optimizer)
    }

    /// Perform one optimization step
    pub fn optimize_step(
        &mut self,
        parameters: &[f64],
        gradients: &[f64],
        cost: f64,
    ) -> Result<Vec<f64>> {
        // Convert to ndarray
        let params_array = Array1::from_vec(parameters.to_vec());
        let mut grads_array = Array1::from_vec(gradients.to_vec());

        // Apply gradient clipping
        if let Some(clip_norm) = self.config.gradient_clip_norm {
            let grad_norm = grads_array.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm > clip_norm {
                grads_array = &grads_array * (clip_norm / grad_norm);
            }
        }

        // Apply L2 regularization
        if self.config.l2_regularization > 0.0 {
            grads_array = &grads_array + &(&params_array * self.config.l2_regularization);
        }

        // Compute gradient norm for tracking
        let grad_norm = grads_array.iter().map(|g| g * g).sum::<f64>().sqrt();
        self.gradient_norms.push(grad_norm);

        // Perform optimization step based on optimizer type
        let new_params = match &mut self.optimizer {
            OptiRSOptimizerImpl::SGD(opt) => opt
                .step(&params_array, &grads_array)
                .map_err(|e| SimulatorError::ComputationError(format!("SGD step failed: {e}")))?,
            OptiRSOptimizerImpl::Adam(opt) => opt
                .step(&params_array, &grads_array)
                .map_err(|e| SimulatorError::ComputationError(format!("Adam step failed: {e}")))?,
            OptiRSOptimizerImpl::RMSprop(opt) => {
                opt.step(&params_array, &grads_array).map_err(|e| {
                    SimulatorError::ComputationError(format!("RMSprop step failed: {e}"))
                })?
            }
            OptiRSOptimizerImpl::Adagrad(opt) => {
                opt.step(&params_array, &grads_array).map_err(|e| {
                    SimulatorError::ComputationError(format!("Adagrad step failed: {e}"))
                })?
            }
        };

        // Apply parameter bounds if specified
        let bounded_params = if let Some((min_val, max_val)) = self.config.parameter_bounds {
            new_params.mapv(|p| p.clamp(min_val, max_val))
        } else {
            new_params
        };

        // Update best parameters
        if cost < self.best_cost {
            self.best_cost = cost;
            self.best_parameters = Some(bounded_params.to_vec());
        }

        // Update history
        self.cost_history.push(cost);
        self.iteration += 1;

        Ok(bounded_params.to_vec())
    }

    /// Check if optimization has converged
    #[must_use]
    pub fn has_converged(&self) -> bool {
        if self.cost_history.len() < 2 {
            return false;
        }

        let recent_costs = &self.cost_history[self.cost_history.len().saturating_sub(10)..];
        if recent_costs.len() < 2 {
            return false;
        }

        let cost_variance = {
            let mean = recent_costs.iter().sum::<f64>() / recent_costs.len() as f64;
            recent_costs
                .iter()
                .map(|&c| (c - mean).powi(2))
                .sum::<f64>()
                / recent_costs.len() as f64
        };

        cost_variance < self.config.convergence_tolerance
    }

    /// Get the best parameters found
    #[must_use]
    pub fn best_parameters(&self) -> Option<&[f64]> {
        self.best_parameters.as_deref()
    }

    /// Get the best cost found
    #[must_use]
    pub const fn best_cost(&self) -> f64 {
        self.best_cost
    }

    /// Get the cost history
    #[must_use]
    pub fn cost_history(&self) -> &[f64] {
        &self.cost_history
    }

    /// Get the gradient norms history
    #[must_use]
    pub fn gradient_norms(&self) -> &[f64] {
        &self.gradient_norms
    }

    /// Get the current iteration
    #[must_use]
    pub const fn iteration(&self) -> usize {
        self.iteration
    }

    /// Reset the optimizer state by recreating it
    pub fn reset(&mut self) -> Result<()> {
        // Recreate the optimizer to reset its state
        self.optimizer = Self::create_optimizer(&self.config)?;
        self.iteration = 0;
        self.best_parameters = None;
        self.best_cost = f64::INFINITY;
        self.cost_history.clear();
        self.gradient_norms.clear();
        Ok(())
    }
}

/// `OptiRS` optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptiRSOptimizationResult {
    /// Optimal parameters found
    pub optimal_parameters: Vec<f64>,
    /// Final cost function value
    pub optimal_cost: f64,
    /// Optimization history
    pub cost_history: Vec<f64>,
    /// Gradient norms history
    pub gradient_norms: Vec<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Convergence flag
    pub converged: bool,
    /// Total optimization time
    pub optimization_time: Duration,
}

impl OptiRSOptimizationResult {
    /// Create a new result from the optimizer
    #[must_use]
    pub fn from_optimizer(
        optimizer: &OptiRSQuantumOptimizer,
        converged: bool,
        optimization_time: Duration,
    ) -> Self {
        Self {
            optimal_parameters: optimizer.best_parameters().unwrap_or(&[]).to_vec(),
            optimal_cost: optimizer.best_cost(),
            cost_history: optimizer.cost_history().to_vec(),
            gradient_norms: optimizer.gradient_norms().to_vec(),
            iterations: optimizer.iteration(),
            converged,
            optimization_time,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optirs_optimizer_creation() {
        let config = OptiRSConfig::default();
        let optimizer = OptiRSQuantumOptimizer::new(config);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_optirs_sgd_optimizer() {
        let config = OptiRSConfig {
            optimizer_type: OptiRSOptimizerType::SGD { momentum: true },
            ..Default::default()
        };
        let mut optimizer =
            OptiRSQuantumOptimizer::new(config).expect("Failed to create SGD optimizer");

        let params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.15];
        let cost = 1.5;

        let new_params = optimizer
            .optimize_step(&params, &grads, cost)
            .expect("Failed to perform optimization step");
        assert_eq!(new_params.len(), params.len());
    }

    #[test]
    fn test_optirs_adam_optimizer() {
        let config = OptiRSConfig {
            optimizer_type: OptiRSOptimizerType::Adam,
            learning_rate: 0.001,
            ..Default::default()
        };
        let mut optimizer =
            OptiRSQuantumOptimizer::new(config).expect("Failed to create Adam optimizer");

        let params = vec![0.5, 1.5, 2.5];
        let grads = vec![0.05, 0.15, 0.1];
        let cost = 2.3;

        let new_params = optimizer
            .optimize_step(&params, &grads, cost)
            .expect("Failed to perform optimization step");
        assert_eq!(new_params.len(), params.len());
        assert_eq!(optimizer.iteration(), 1);
    }

    #[test]
    fn test_optirs_convergence_check() {
        let config = OptiRSConfig {
            convergence_tolerance: 1e-6,
            ..Default::default()
        };
        let mut optimizer =
            OptiRSQuantumOptimizer::new(config).expect("Failed to create optimizer");

        // Should not converge initially
        assert!(!optimizer.has_converged());

        // Add stable cost history
        for _ in 0..15 {
            let params = vec![1.0];
            let grads = vec![0.001];
            optimizer
                .optimize_step(&params, &grads, 1.0)
                .expect("Failed to perform optimization step");
        }

        // Should converge with stable costs
        assert!(optimizer.has_converged());
    }

    #[test]
    fn test_optirs_parameter_bounds() {
        let config = OptiRSConfig {
            parameter_bounds: Some((-1.0, 1.0)),
            learning_rate: 10.0, // Large LR to test bounds
            ..Default::default()
        };
        let mut optimizer =
            OptiRSQuantumOptimizer::new(config).expect("Failed to create optimizer");

        let params = vec![0.9];
        let grads = vec![-1.0]; // Would push parameter > 1.0 without bounds
        let cost = 1.0;

        let new_params = optimizer
            .optimize_step(&params, &grads, cost)
            .expect("Failed to perform optimization step");
        assert!(new_params[0] <= 1.0);
        assert!(new_params[0] >= -1.0);
    }

    #[test]
    fn test_optirs_gradient_clipping() {
        let config = OptiRSConfig {
            gradient_clip_norm: Some(0.5),
            ..Default::default()
        };
        let mut optimizer =
            OptiRSQuantumOptimizer::new(config).expect("Failed to create optimizer");

        let params = vec![1.0, 1.0];
        let large_grads = vec![10.0, 10.0]; // Norm = sqrt(200) >> 0.5
        let cost = 1.0;

        // Should not crash with large gradients
        let new_params = optimizer
            .optimize_step(&params, &large_grads, cost)
            .expect("Failed to perform optimization step");
        assert_eq!(new_params.len(), params.len());
    }

    #[test]
    fn test_optirs_reset() {
        let config = OptiRSConfig::default();
        let mut optimizer =
            OptiRSQuantumOptimizer::new(config).expect("Failed to create optimizer");

        // Perform some steps
        for _ in 0..5 {
            let params = vec![1.0];
            let grads = vec![0.1];
            optimizer
                .optimize_step(&params, &grads, 1.0)
                .expect("Failed to perform optimization step");
        }

        assert_eq!(optimizer.iteration(), 5);

        // Reset
        optimizer.reset().expect("Failed to reset optimizer");

        assert_eq!(optimizer.iteration(), 0);
        assert_eq!(optimizer.cost_history().len(), 0);
    }

    #[test]
    fn test_all_optimizer_types() {
        let optimizers = vec![
            OptiRSOptimizerType::SGD { momentum: false },
            OptiRSOptimizerType::SGD { momentum: true },
            OptiRSOptimizerType::Adam,
            OptiRSOptimizerType::RMSprop,
            OptiRSOptimizerType::Adagrad,
        ];

        for opt_type in optimizers {
            let config = OptiRSConfig {
                optimizer_type: opt_type,
                ..Default::default()
            };
            let mut optimizer =
                OptiRSQuantumOptimizer::new(config).expect("Failed to create optimizer");

            let params = vec![1.0, 2.0];
            let grads = vec![0.1, 0.2];
            let cost = 1.0;

            let result = optimizer.optimize_step(&params, &grads, cost);
            assert!(result.is_ok(), "Failed for optimizer {opt_type:?}");
        }
    }
}
