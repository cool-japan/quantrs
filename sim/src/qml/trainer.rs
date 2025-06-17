//! Quantum machine learning trainer implementation.
//!
//! This module provides the main trainer class for quantum machine learning
//! algorithms with hardware-aware optimization and adaptive training strategies.

use crate::prelude::HardwareOptimizations;
use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::circuit_interfaces::{CircuitInterface, InterfaceCircuit};
use crate::device_noise_models::DeviceNoiseModel;
use crate::error::Result;

use super::circuit::ParameterizedQuantumCircuit;
use super::config::{GradientMethod, HardwareArchitecture, OptimizerType, QMLConfig};

/// Quantum machine learning trainer
pub struct QuantumMLTrainer {
    /// Configuration
    config: QMLConfig,
    /// Parameterized quantum circuit
    pqc: ParameterizedQuantumCircuit,
    /// Optimizer state
    optimizer_state: OptimizerState,
    /// Training history
    training_history: TrainingHistory,
    /// Device noise model
    noise_model: Option<Box<dyn DeviceNoiseModel>>,
    /// Circuit interface
    circuit_interface: CircuitInterface,
    /// Hardware-aware compiler
    hardware_compiler: HardwareAwareCompiler,
}

/// Optimizer state
#[derive(Debug, Clone)]
pub struct OptimizerState {
    /// Current parameter values
    pub parameters: Array1<f64>,
    /// Gradient estimate
    pub gradient: Array1<f64>,
    /// Momentum terms (for Adam, etc.)
    pub momentum: Array1<f64>,
    /// Velocity terms (for Adam, etc.)
    pub velocity: Array1<f64>,
    /// Learning rate schedule
    pub learning_rate: f64,
    /// Iteration counter
    pub iteration: usize,
}

/// Training history
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Loss values over epochs
    pub loss_history: Vec<f64>,
    /// Gradient norms
    pub gradient_norms: Vec<f64>,
    /// Parameter norms
    pub parameter_norms: Vec<f64>,
    /// Training times per epoch
    pub epoch_times: Vec<f64>,
    /// Hardware utilization metrics
    pub hardware_metrics: Vec<HardwareMetrics>,
}

/// Hardware utilization metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardwareMetrics {
    /// Circuit depth after compilation
    pub compiled_depth: usize,
    /// Number of two-qubit gates
    pub two_qubit_gates: usize,
    /// Total execution time
    pub execution_time: f64,
    /// Estimated fidelity
    pub estimated_fidelity: f64,
    /// Shot overhead
    pub shot_overhead: f64,
}

/// Hardware-aware compiler
#[derive(Debug, Clone)]
pub struct HardwareAwareCompiler {
    /// Target hardware architecture
    hardware_arch: HardwareArchitecture,
    /// Hardware optimizations
    hardware_opts: HardwareOptimizations,
    /// Compilation statistics
    compilation_stats: CompilationStats,
}

/// Compilation statistics
#[derive(Debug, Clone, Default)]
pub struct CompilationStats {
    /// Original circuit depth
    pub original_depth: usize,
    /// Compiled circuit depth
    pub compiled_depth: usize,
    /// Number of SWAP gates added
    pub swap_gates_added: usize,
    /// Compilation time
    pub compilation_time: f64,
    /// Estimated execution time
    pub estimated_execution_time: f64,
}

/// Training result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Final parameter values
    pub final_parameters: Array1<f64>,
    /// Final loss value
    pub final_loss: f64,
    /// Number of epochs completed
    pub epochs_completed: usize,
    /// Training history
    pub training_history: TrainingHistory,
    /// Convergence achieved
    pub converged: bool,
}

impl QuantumMLTrainer {
    /// Create a new quantum ML trainer
    pub fn new(
        config: QMLConfig,
        pqc: ParameterizedQuantumCircuit,
        noise_model: Option<Box<dyn DeviceNoiseModel>>,
    ) -> Result<Self> {
        let num_params = pqc.num_parameters();

        let optimizer_state = OptimizerState {
            parameters: pqc.parameters.clone(),
            gradient: Array1::zeros(num_params),
            momentum: Array1::zeros(num_params),
            velocity: Array1::zeros(num_params),
            learning_rate: config.learning_rate,
            iteration: 0,
        };

        let training_history = TrainingHistory::default();
        let circuit_interface = CircuitInterface::new(Default::default())?;
        let hardware_compiler = HardwareAwareCompiler::new(
            config.hardware_architecture,
            pqc.hardware_optimizations.clone(),
        );

        Ok(Self {
            config,
            pqc,
            optimizer_state,
            training_history,
            noise_model,
            circuit_interface,
            hardware_compiler,
        })
    }

    /// Train the quantum ML model
    pub fn train<F>(&mut self, loss_function: F) -> Result<TrainingResult>
    where
        F: Fn(&Array1<f64>) -> Result<f64> + Send + Sync,
    {
        let start_time = std::time::Instant::now();

        for epoch in 0..self.config.max_epochs {
            let epoch_start = std::time::Instant::now();

            // Compute gradient
            let gradient = self.compute_gradient(&loss_function)?;
            self.optimizer_state.gradient = gradient;

            // Update parameters
            self.update_parameters()?;

            // Evaluate loss
            let current_loss = loss_function(&self.optimizer_state.parameters)?;

            // Update training history
            let epoch_time = epoch_start.elapsed().as_secs_f64();
            self.training_history.loss_history.push(current_loss);
            self.training_history.gradient_norms.push(
                self.optimizer_state
                    .gradient
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>()
                    .sqrt(),
            );
            self.training_history.parameter_norms.push(
                self.optimizer_state
                    .parameters
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>()
                    .sqrt(),
            );
            self.training_history.epoch_times.push(epoch_time);

            // Check convergence
            if self.check_convergence(current_loss)? {
                return Ok(TrainingResult {
                    final_parameters: self.optimizer_state.parameters.clone(),
                    final_loss: current_loss,
                    epochs_completed: epoch + 1,
                    training_history: self.training_history.clone(),
                    converged: true,
                });
            }

            self.optimizer_state.iteration += 1;
        }

        // Training completed without convergence
        let final_loss = loss_function(&self.optimizer_state.parameters)?;
        Ok(TrainingResult {
            final_parameters: self.optimizer_state.parameters.clone(),
            final_loss,
            epochs_completed: self.config.max_epochs,
            training_history: self.training_history.clone(),
            converged: false,
        })
    }

    /// Compute gradient using the specified method
    fn compute_gradient<F>(&mut self, loss_function: &F) -> Result<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> Result<f64> + Send + Sync,
    {
        match self.config.gradient_method {
            GradientMethod::ParameterShift => self.compute_parameter_shift_gradient(loss_function),
            GradientMethod::FiniteDifferences => {
                self.compute_finite_difference_gradient(loss_function)
            }
            GradientMethod::AutomaticDifferentiation => {
                self.compute_autodiff_gradient(loss_function)
            }
            GradientMethod::NaturalGradients => self.compute_natural_gradient(loss_function),
            GradientMethod::StochasticParameterShift => {
                self.compute_stochastic_parameter_shift_gradient(loss_function)
            }
        }
    }

    /// Compute gradient using parameter shift rule
    fn compute_parameter_shift_gradient<F>(&self, loss_function: &F) -> Result<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> Result<f64> + Send + Sync,
    {
        let num_params = self.optimizer_state.parameters.len();
        let mut gradient = Array1::zeros(num_params);
        let shift = std::f64::consts::PI / 2.0;

        for i in 0..num_params {
            let mut params_plus = self.optimizer_state.parameters.clone();
            let mut params_minus = self.optimizer_state.parameters.clone();

            params_plus[i] += shift;
            params_minus[i] -= shift;

            let loss_plus = loss_function(&params_plus)?;
            let loss_minus = loss_function(&params_minus)?;

            gradient[i] = (loss_plus - loss_minus) / 2.0;
        }

        Ok(gradient)
    }

    /// Compute gradient using finite differences
    fn compute_finite_difference_gradient<F>(&self, loss_function: &F) -> Result<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> Result<f64> + Send + Sync,
    {
        let num_params = self.optimizer_state.parameters.len();
        let mut gradient = Array1::zeros(num_params);
        let eps = 1e-8;

        for i in 0..num_params {
            let mut params_plus = self.optimizer_state.parameters.clone();
            params_plus[i] += eps;

            let loss_plus = loss_function(&params_plus)?;
            let loss_current = loss_function(&self.optimizer_state.parameters)?;

            gradient[i] = (loss_plus - loss_current) / eps;
        }

        Ok(gradient)
    }

    /// Compute gradient using automatic differentiation
    fn compute_autodiff_gradient<F>(&self, loss_function: &F) -> Result<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> Result<f64> + Send + Sync,
    {
        // Simplified automatic differentiation implementation
        // In practice, this would use a proper autodiff library
        self.compute_parameter_shift_gradient(loss_function)
    }

    /// Compute natural gradient
    fn compute_natural_gradient<F>(&self, loss_function: &F) -> Result<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> Result<f64> + Send + Sync,
    {
        // Simplified natural gradient implementation
        let gradient = self.compute_parameter_shift_gradient(loss_function)?;

        // For simplicity, return regular gradient
        // In practice, this would compute the Fisher information matrix
        Ok(gradient)
    }

    /// Compute stochastic parameter shift gradient
    fn compute_stochastic_parameter_shift_gradient<F>(
        &self,
        loss_function: &F,
    ) -> Result<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> Result<f64> + Send + Sync,
    {
        // Simplified stochastic version
        self.compute_parameter_shift_gradient(loss_function)
    }

    /// Update parameters using the optimizer
    fn update_parameters(&mut self) -> Result<()> {
        match self.config.optimizer_type {
            OptimizerType::Adam => self.update_parameters_adam(),
            OptimizerType::SGD => self.update_parameters_sgd(),
            OptimizerType::RMSprop => self.update_parameters_rmsprop(),
            OptimizerType::LBFGS => self.update_parameters_lbfgs(),
            OptimizerType::QuantumNaturalGradient => self.update_parameters_qng(),
            OptimizerType::SPSA => self.update_parameters_spsa(),
        }
    }

    /// Update parameters using Adam optimizer
    fn update_parameters_adam(&mut self) -> Result<()> {
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;

        // Update momentum and velocity
        for i in 0..self.optimizer_state.parameters.len() {
            self.optimizer_state.momentum[i] = beta1 * self.optimizer_state.momentum[i]
                + (1.0 - beta1) * self.optimizer_state.gradient[i];
            self.optimizer_state.velocity[i] = beta2 * self.optimizer_state.velocity[i]
                + (1.0 - beta2) * self.optimizer_state.gradient[i].powi(2);

            // Bias correction
            let m_hat = self.optimizer_state.momentum[i]
                / (1.0 - beta1.powi(self.optimizer_state.iteration as i32 + 1));
            let v_hat = self.optimizer_state.velocity[i]
                / (1.0 - beta2.powi(self.optimizer_state.iteration as i32 + 1));

            // Update parameter
            self.optimizer_state.parameters[i] -=
                self.optimizer_state.learning_rate * m_hat / (v_hat.sqrt() + eps);
        }

        Ok(())
    }

    /// Update parameters using SGD
    fn update_parameters_sgd(&mut self) -> Result<()> {
        for i in 0..self.optimizer_state.parameters.len() {
            self.optimizer_state.parameters[i] -=
                self.optimizer_state.learning_rate * self.optimizer_state.gradient[i];
        }
        Ok(())
    }

    /// Update parameters using RMSprop
    fn update_parameters_rmsprop(&mut self) -> Result<()> {
        let alpha = 0.99;
        let eps = 1e-8;

        for i in 0..self.optimizer_state.parameters.len() {
            self.optimizer_state.velocity[i] = alpha * self.optimizer_state.velocity[i]
                + (1.0 - alpha) * self.optimizer_state.gradient[i].powi(2);
            self.optimizer_state.parameters[i] -= self.optimizer_state.learning_rate
                * self.optimizer_state.gradient[i]
                / (self.optimizer_state.velocity[i].sqrt() + eps);
        }

        Ok(())
    }

    /// Update parameters using L-BFGS (simplified)
    fn update_parameters_lbfgs(&mut self) -> Result<()> {
        // Simplified L-BFGS - in practice would maintain history
        self.update_parameters_sgd()
    }

    /// Update parameters using Quantum Natural Gradient
    fn update_parameters_qng(&mut self) -> Result<()> {
        // Simplified QNG - in practice would compute metric tensor
        self.update_parameters_sgd()
    }

    /// Update parameters using SPSA
    fn update_parameters_spsa(&mut self) -> Result<()> {
        // Simplified SPSA
        self.update_parameters_sgd()
    }

    /// Check convergence criteria
    fn check_convergence(&self, current_loss: f64) -> Result<bool> {
        if self.training_history.loss_history.len() < 2 {
            return Ok(false);
        }

        let prev_loss =
            self.training_history.loss_history[self.training_history.loss_history.len() - 1];
        let loss_change = (current_loss - prev_loss).abs();

        Ok(loss_change < self.config.convergence_tolerance)
    }

    /// Get current parameters
    pub fn get_parameters(&self) -> &Array1<f64> {
        &self.optimizer_state.parameters
    }

    /// Get training history
    pub fn get_training_history(&self) -> &TrainingHistory {
        &self.training_history
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.optimizer_state.learning_rate = lr;
    }

    /// Reset optimizer state
    pub fn reset_optimizer(&mut self) {
        let num_params = self.optimizer_state.parameters.len();
        self.optimizer_state.gradient = Array1::zeros(num_params);
        self.optimizer_state.momentum = Array1::zeros(num_params);
        self.optimizer_state.velocity = Array1::zeros(num_params);
        self.optimizer_state.iteration = 0;
        self.training_history = TrainingHistory::default();
    }
}

impl HardwareAwareCompiler {
    /// Create a new hardware-aware compiler
    pub fn new(hardware_arch: HardwareArchitecture, hardware_opts: HardwareOptimizations) -> Self {
        Self {
            hardware_arch,
            hardware_opts,
            compilation_stats: CompilationStats::default(),
        }
    }

    /// Compile circuit for target hardware
    pub fn compile_circuit(&mut self, circuit: &InterfaceCircuit) -> Result<InterfaceCircuit> {
        let start_time = std::time::Instant::now();
        self.compilation_stats.original_depth = circuit.gates.len();

        // For now, return the same circuit
        // In practice, this would perform hardware-specific optimizations
        let compiled_circuit = circuit.clone();

        self.compilation_stats.compiled_depth = compiled_circuit.gates.len();
        self.compilation_stats.compilation_time = start_time.elapsed().as_secs_f64();

        Ok(compiled_circuit)
    }

    /// Get compilation statistics
    pub fn get_stats(&self) -> &CompilationStats {
        &self.compilation_stats
    }
}

impl OptimizerState {
    /// Create new optimizer state
    pub fn new(num_parameters: usize, learning_rate: f64) -> Self {
        Self {
            parameters: Array1::zeros(num_parameters),
            gradient: Array1::zeros(num_parameters),
            momentum: Array1::zeros(num_parameters),
            velocity: Array1::zeros(num_parameters),
            learning_rate,
            iteration: 0,
        }
    }
}

impl TrainingHistory {
    /// Get the latest loss value
    pub fn latest_loss(&self) -> Option<f64> {
        self.loss_history.last().copied()
    }

    /// Get the best (minimum) loss value
    pub fn best_loss(&self) -> Option<f64> {
        self.loss_history
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
    }

    /// Get average epoch time
    pub fn average_epoch_time(&self) -> f64 {
        if self.epoch_times.is_empty() {
            0.0
        } else {
            self.epoch_times.iter().sum::<f64>() / self.epoch_times.len() as f64
        }
    }
}
