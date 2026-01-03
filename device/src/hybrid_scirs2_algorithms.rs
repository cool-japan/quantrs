//! Hybrid Quantum-Classical Algorithms with SciRS2 Optimization
//!
//! This module provides comprehensive hybrid algorithm implementations that leverage
//! SciRS2's advanced optimization capabilities for variational quantum algorithms,
//! adaptive optimization, and hardware-efficient hybrid execution.

use crate::{DeviceError, DeviceResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use scirs2_optimize::unconstrained::{minimize, Method, Options};

/// Hybrid algorithm configuration using SciRS2 optimization
#[derive(Debug, Clone)]
pub struct HybridAlgorithmConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Optimization method
    pub optimization_method: OptimizationMethod,
    /// Learning rate (for custom implementations)
    pub learning_rate: f64,
}

impl Default for HybridAlgorithmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            optimization_method: OptimizationMethod::BFGS,
            learning_rate: 0.01,
        }
    }
}

/// Optimization methods leveraging SciRS2
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationMethod {
    /// BFGS quasi-Newton method
    BFGS,
    /// Nelder-Mead simplex method
    NelderMead,
    /// Conjugate gradient
    ConjugateGradient,
    /// Limited-memory BFGS
    LBFGS,
    /// Powell's method
    Powell,
}

/// Result of hybrid algorithm optimization
#[derive(Debug, Clone)]
pub struct HybridOptimizationResult {
    /// Optimal parameters found
    pub optimal_parameters: Array1<f64>,
    /// Final objective function value
    pub optimal_value: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Convergence achieved
    pub converged: bool,
}

/// Hybrid quantum-classical algorithm executor with SciRS2 optimization
pub struct HybridAlgorithmExecutor {
    config: HybridAlgorithmConfig,
    /// Random number generator for stochastic methods
    rng: StdRng,
}

impl HybridAlgorithmExecutor {
    /// Create a new hybrid algorithm executor
    pub fn new(config: HybridAlgorithmConfig) -> Self {
        Self {
            config,
            rng: StdRng::seed_from_u64(42), // Reproducible RNG
        }
    }

    /// Create executor with default configuration
    pub fn default() -> Self {
        Self::new(HybridAlgorithmConfig::default())
    }

    /// Optimize a quantum objective function using SciRS2 optimization
    ///
    /// # Arguments
    /// * `objective` - Objective function to minimize (parameters -> value)
    /// * `initial_params` - Starting point for optimization
    ///
    /// # Returns
    /// Optimization result with optimal parameters and convergence information
    pub fn optimize<F>(
        &mut self,
        mut objective: F,
        initial_params: &Array1<f64>,
    ) -> DeviceResult<HybridOptimizationResult>
    where
        F: FnMut(&ArrayView1<f64>) -> f64 + Clone,
    {
        let method = match self.config.optimization_method {
            OptimizationMethod::BFGS => Method::BFGS,
            OptimizationMethod::NelderMead => Method::NelderMead,
            OptimizationMethod::ConjugateGradient => Method::CG,
            OptimizationMethod::LBFGS => Method::LBFGS,
            OptimizationMethod::Powell => Method::Powell,
        };

        let options = Options {
            max_iter: self.config.max_iterations,
            ftol: self.config.tolerance,
            ..Default::default()
        };

        let x0_slice: Vec<f64> = initial_params.to_vec();

        let result = minimize(objective, &x0_slice, method, Some(options))
            .map_err(|e| DeviceError::OptimizationError(format!("Optimization failed: {}", e)))?;

        Ok(HybridOptimizationResult {
            optimal_parameters: result.x,
            optimal_value: result.fun,
            iterations: result.nit,
            converged: result.success,
        })
    }

    /// Generate initial parameters using SciRS2 random number generation
    pub fn generate_initial_parameters(
        &mut self,
        dimension: usize,
        range: (f64, f64),
    ) -> Array1<f64> {
        let (low, high) = range;
        Array1::from_shape_fn(dimension, |_| low + (high - low) * self.rng.gen::<f64>())
    }
}

/// Variational Quantum Eigensolver (VQE) implementation with SciRS2 optimization
pub struct VQEWithSciRS2 {
    executor: HybridAlgorithmExecutor,
    /// Number of parameters in the ansatz
    num_parameters: usize,
}

impl VQEWithSciRS2 {
    /// Create a new VQE instance
    pub fn new(config: HybridAlgorithmConfig, num_parameters: usize) -> Self {
        Self {
            executor: HybridAlgorithmExecutor::new(config),
            num_parameters,
        }
    }

    /// Run VQE optimization
    ///
    /// # Arguments
    /// * `energy_function` - Function that computes energy given parameters
    /// * `initial_params` - Starting parameters (if None, random initialization)
    pub fn run<F>(
        &mut self,
        energy_function: F,
        initial_params: Option<&Array1<f64>>,
    ) -> DeviceResult<HybridOptimizationResult>
    where
        F: FnMut(&ArrayView1<f64>) -> f64 + Clone,
    {
        let params = if let Some(p) = initial_params {
            p.clone()
        } else {
            self.executor.generate_initial_parameters(
                self.num_parameters,
                (-std::f64::consts::PI, std::f64::consts::PI),
            )
        };

        self.executor.optimize(energy_function, &params)
    }
}

/// Quantum Approximate Optimization Algorithm (QAOA) with SciRS2
pub struct QAOAWithSciRS2 {
    executor: HybridAlgorithmExecutor,
    /// Number of QAOA layers (p)
    num_layers: usize,
}

impl QAOAWithSciRS2 {
    /// Create a new QAOA instance
    pub fn new(config: HybridAlgorithmConfig, num_layers: usize) -> Self {
        Self {
            executor: HybridAlgorithmExecutor::new(config),
            num_layers,
        }
    }

    /// Run QAOA optimization
    ///
    /// # Arguments
    /// * `cost_function` - Function that computes cost given parameters (2*p parameters: gamma and beta)
    pub fn run<F>(&mut self, cost_function: F) -> DeviceResult<HybridOptimizationResult>
    where
        F: FnMut(&ArrayView1<f64>) -> f64 + Clone,
    {
        // QAOA has 2*p parameters (p gamma parameters, p beta parameters)
        let num_params = 2 * self.num_layers;
        let initial_params = self
            .executor
            .generate_initial_parameters(num_params, (0.0, 2.0 * std::f64::consts::PI));

        self.executor.optimize(cost_function, &initial_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_executor_creation() {
        let config = HybridAlgorithmConfig::default();
        let executor = HybridAlgorithmExecutor::new(config);
        assert_eq!(executor.config.max_iterations, 1000);
    }

    #[test]
    fn test_bfgs_optimization() {
        let config = HybridAlgorithmConfig {
            optimization_method: OptimizationMethod::BFGS,
            max_iterations: 100,
            tolerance: 1e-6,
            ..Default::default()
        };

        let mut executor = HybridAlgorithmExecutor::new(config);

        // Minimize f(x) = (x - 2)^2
        let objective = |x: &ArrayView1<f64>| (x[0] - 2.0).powi(2);

        let initial = Array1::from(vec![0.0]);
        let result = executor.optimize(objective, &initial);

        assert!(result.is_ok());
        let result = result.expect("Optimization failed");
        assert!(result.converged);
        assert!((result.optimal_parameters[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_nelder_mead_optimization() {
        let config = HybridAlgorithmConfig {
            optimization_method: OptimizationMethod::NelderMead,
            max_iterations: 500,
            tolerance: 1e-6,
            ..Default::default()
        };

        let mut executor = HybridAlgorithmExecutor::new(config);

        // Minimize Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        let objective = |params: &ArrayView1<f64>| {
            let x = params[0];
            let y = params[1];
            (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2)
        };

        let initial = Array1::from(vec![0.0, 0.0]);
        let result = executor.optimize(objective, &initial);

        assert!(result.is_ok());
        let result = result.expect("Optimization failed");
        // Nelder-Mead may not converge as tightly as gradient methods
        assert!((result.optimal_parameters[0] - 1.0).abs() < 0.2);
        assert!((result.optimal_parameters[1] - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_vqe_creation() {
        let config = HybridAlgorithmConfig::default();
        let vqe = VQEWithSciRS2::new(config, 4);
        assert_eq!(vqe.num_parameters, 4);
    }

    #[test]
    fn test_qaoa_creation() {
        let config = HybridAlgorithmConfig::default();
        let qaoa = QAOAWithSciRS2::new(config, 3);
        assert_eq!(qaoa.num_layers, 3);
    }

    #[test]
    fn test_random_parameter_generation() {
        let config = HybridAlgorithmConfig::default();
        let mut executor = HybridAlgorithmExecutor::new(config);

        let params = executor.generate_initial_parameters(5, (-1.0, 1.0));
        assert_eq!(params.len(), 5);

        // All parameters should be in range [-1, 1]
        for &p in params.iter() {
            assert!((-1.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn test_conjugate_gradient_optimization() {
        let config = HybridAlgorithmConfig {
            optimization_method: OptimizationMethod::ConjugateGradient,
            max_iterations: 100,
            tolerance: 1e-6,
            ..Default::default()
        };

        let mut executor = HybridAlgorithmExecutor::new(config);

        // Minimize f(x) = x^2 + y^2
        let objective = |params: &ArrayView1<f64>| params[0].powi(2) + params[1].powi(2);

        let initial = Array1::from(vec![5.0, 5.0]);
        let result = executor.optimize(objective, &initial);

        assert!(result.is_ok());
        let result = result.expect("Optimization failed");
        assert!(result.optimal_parameters[0].abs() < 0.1);
        assert!(result.optimal_parameters[1].abs() < 0.1);
    }

    #[test]
    fn test_vqe_run() {
        let config = HybridAlgorithmConfig {
            optimization_method: OptimizationMethod::BFGS,
            max_iterations: 50,
            ..Default::default()
        };

        let mut vqe = VQEWithSciRS2::new(config, 2);

        // Simple energy function: E(theta) = (theta[0] - 1)^2 + (theta[1] - 2)^2
        let energy =
            |params: &ArrayView1<f64>| (params[0] - 1.0).powi(2) + (params[1] - 2.0).powi(2);

        let result = vqe.run(energy, None);
        assert!(result.is_ok());

        let result = result.expect("VQE failed");
        assert!((result.optimal_parameters[0] - 1.0).abs() < 0.1);
        assert!((result.optimal_parameters[1] - 2.0).abs() < 0.1);
    }
}
