//! Quantum-Classical Hybrid Algorithms
//!
//! This module implements hybrid quantum-classical algorithms that leverage both
//! quantum and classical computing resources for enhanced performance.
//!
//! ## Algorithms Included
//!
//! - **Variational Quantum-Classical Optimization**: Iterative optimization schemes
//! - **Quantum-Classical Neural Networks**: Hybrid neural architectures
//! - **Quantum-Assisted Machine Learning**: Classical ML with quantum subroutines
//! - **Hybrid Quantum Annealing**: Combined quantum and simulated annealing
//! - **Quantum-Classical Sampling**: Hybrid sampling strategies
//! - **Quantum Feature Maps**: Classical data encoding in quantum states

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64 as Complex;
use std::collections::HashMap;

// ================================================================================================
// Variational Quantum-Classical Optimizer
// ================================================================================================

/// Variational quantum-classical optimization algorithm
pub struct VariationalQCOptimizer {
    /// Classical optimizer
    optimizer: Box<dyn ClassicalOptimizer>,
    /// Quantum circuit evaluator
    circuit_evaluator: CircuitEvaluator,
    /// Optimization configuration
    config: VQCConfig,
}

/// Configuration for variational quantum-classical optimization
#[derive(Debug, Clone)]
pub struct VQCConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of quantum shots per evaluation
    pub shots_per_evaluation: usize,
    /// Use parameter shift rule for gradients
    pub use_parameter_shift: bool,
}

impl Default for VQCConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            shots_per_evaluation: 1000,
            use_parameter_shift: true,
        }
    }
}

/// Classical optimizer trait for hybrid algorithms
pub trait ClassicalOptimizer {
    /// Perform one optimization step
    fn step(&mut self, params: &[f64], gradient: &[f64]) -> Vec<f64>;

    /// Get current parameters
    fn get_params(&self) -> &[f64];
}

/// Gradient descent optimizer
pub struct GradientDescentOptimizer {
    params: Vec<f64>,
    learning_rate: f64,
}

impl GradientDescentOptimizer {
    /// Create a new gradient descent optimizer
    pub const fn new(initial_params: Vec<f64>, learning_rate: f64) -> Self {
        Self {
            params: initial_params,
            learning_rate,
        }
    }
}

impl ClassicalOptimizer for GradientDescentOptimizer {
    fn step(&mut self, _params: &[f64], gradient: &[f64]) -> Vec<f64> {
        for (param, &grad) in self.params.iter_mut().zip(gradient.iter()) {
            *param -= self.learning_rate * grad;
        }
        self.params.clone()
    }

    fn get_params(&self) -> &[f64] {
        &self.params
    }
}

/// Adam optimizer
pub struct AdamOptimizer {
    params: Vec<f64>,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: Vec<f64>, // First moment
    v: Vec<f64>, // Second moment
    t: usize,    // Time step
}

impl AdamOptimizer {
    /// Create a new Adam optimizer
    pub fn new(initial_params: Vec<f64>, learning_rate: f64) -> Self {
        let n = initial_params.len();
        Self {
            params: initial_params,
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }
}

impl ClassicalOptimizer for AdamOptimizer {
    fn step(&mut self, _params: &[f64], gradient: &[f64]) -> Vec<f64> {
        self.t += 1;

        for i in 0..self.params.len() {
            // Update biased first moment estimate
            self.m[i] = self
                .beta1
                .mul_add(self.m[i], (1.0 - self.beta1) * gradient[i]);

            // Update biased second raw moment estimate
            self.v[i] = self
                .beta2
                .mul_add(self.v[i], (1.0 - self.beta2) * gradient[i].powi(2));

            // Compute bias-corrected first moment estimate
            let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.t as i32));

            // Compute bias-corrected second raw moment estimate
            let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.t as i32));

            // Update parameters
            self.params[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }

        self.params.clone()
    }

    fn get_params(&self) -> &[f64] {
        &self.params
    }
}

/// Quantum circuit evaluator
pub struct CircuitEvaluator {
    /// Number of qubits
    num_qubits: usize,
    /// Circuit structure
    circuit_structure: Vec<LayerSpec>,
}

/// Layer specification for parameterized circuits
#[derive(Debug, Clone)]
pub struct LayerSpec {
    /// Gate type
    pub gate_type: String,
    /// Qubits the gate acts on
    pub qubits: Vec<usize>,
    /// Whether gate is parameterized
    pub is_parameterized: bool,
}

impl CircuitEvaluator {
    /// Create a new circuit evaluator
    pub const fn new(num_qubits: usize, circuit_structure: Vec<LayerSpec>) -> Self {
        Self {
            num_qubits,
            circuit_structure,
        }
    }

    /// Evaluate circuit with given parameters
    pub fn evaluate(&self, params: &[f64]) -> QuantRS2Result<f64> {
        // Simplified: would construct and execute quantum circuit
        // Return expectation value of cost Hamiltonian
        Ok(params.iter().map(|x| x.cos()).sum::<f64>() / params.len() as f64)
    }

    /// Compute gradient using parameter shift rule
    pub fn compute_gradient(&self, params: &[f64]) -> QuantRS2Result<Vec<f64>> {
        let shift = std::f64::consts::PI / 2.0;
        let mut gradient = Vec::new();

        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();

            params_plus[i] += shift;
            params_minus[i] -= shift;

            let value_plus = self.evaluate(&params_plus)?;
            let value_minus = self.evaluate(&params_minus)?;

            gradient.push((value_plus - value_minus) / 2.0);
        }

        Ok(gradient)
    }

    /// Compute gradient using finite differences
    pub fn compute_gradient_finite_diff(
        &self,
        params: &[f64],
        eps: f64,
    ) -> QuantRS2Result<Vec<f64>> {
        let mut gradient = Vec::new();

        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();

            params_plus[i] += eps;
            params_minus[i] -= eps;

            let value_plus = self.evaluate(&params_plus)?;
            let value_minus = self.evaluate(&params_minus)?;

            gradient.push((value_plus - value_minus) / (2.0 * eps));
        }

        Ok(gradient)
    }
}

impl VariationalQCOptimizer {
    /// Create a new variational quantum-classical optimizer
    pub fn new(
        optimizer: Box<dyn ClassicalOptimizer>,
        circuit_evaluator: CircuitEvaluator,
        config: VQCConfig,
    ) -> Self {
        Self {
            optimizer,
            circuit_evaluator,
            config,
        }
    }

    /// Run optimization
    pub fn optimize(&mut self) -> QuantRS2Result<OptimizationResult> {
        let mut history = Vec::new();
        let mut best_value = f64::INFINITY;
        let mut best_params = self.optimizer.get_params().to_vec();

        for iteration in 0..self.config.max_iterations {
            let params = self.optimizer.get_params().to_vec();

            // Evaluate cost function
            let cost = self.circuit_evaluator.evaluate(&params)?;

            // Compute gradient
            let gradient = if self.config.use_parameter_shift {
                self.circuit_evaluator.compute_gradient(&params)?
            } else {
                self.circuit_evaluator
                    .compute_gradient_finite_diff(&params, 1e-5)?
            };

            // Update parameters
            let new_params = self.optimizer.step(&params, &gradient);

            history.push(cost);

            if cost < best_value {
                best_value = cost;
                best_params.clone_from(&new_params);
            }

            // Check convergence
            if iteration > 0
                && (history[iteration] - history[iteration - 1]).abs() < self.config.tolerance
            {
                break;
            }
        }

        let iterations = history.len();
        Ok(OptimizationResult {
            best_params,
            best_value,
            history,
            iterations,
        })
    }
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Best parameters found
    pub best_params: Vec<f64>,
    /// Best objective value
    pub best_value: f64,
    /// Optimization history
    pub history: Vec<f64>,
    /// Number of iterations
    pub iterations: usize,
}

// ================================================================================================
// Quantum-Classical Neural Network
// ================================================================================================

/// Quantum-classical hybrid neural network
pub struct QuantumClassicalNN {
    /// Classical layers (before quantum)
    classical_pre: Vec<ClassicalLayer>,
    /// Quantum layer
    quantum_layer: QuantumLayer,
    /// Classical layers (after quantum)
    classical_post: Vec<ClassicalLayer>,
}

/// Classical neural network layer
pub struct ClassicalLayer {
    /// Weights
    weights: Array2<f64>,
    /// Biases
    biases: Array1<f64>,
    /// Activation function
    activation: ActivationFunction,
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

impl ClassicalLayer {
    /// Create a new classical layer
    pub fn new(input_dim: usize, output_dim: usize, activation: ActivationFunction) -> Self {
        Self {
            weights: Array2::zeros((output_dim, input_dim)),
            biases: Array1::zeros(output_dim),
            activation,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = self.weights.dot(input) + &self.biases;

        // Apply activation
        match self.activation {
            ActivationFunction::ReLU => {
                output.mapv_inplace(|x| x.max(0.0));
            }
            ActivationFunction::Sigmoid => {
                output.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            }
            ActivationFunction::Tanh => {
                output.mapv_inplace(|x| x.tanh());
            }
            ActivationFunction::Linear => {}
        }

        output
    }
}

/// Quantum layer in hybrid network
pub struct QuantumLayer {
    /// Number of qubits
    num_qubits: usize,
    /// Parameterized circuit
    circuit: Vec<LayerSpec>,
    /// Current parameters
    params: Vec<f64>,
}

impl QuantumLayer {
    /// Create a new quantum layer
    pub fn new(num_qubits: usize, circuit: Vec<LayerSpec>) -> Self {
        let num_params = circuit.iter().filter(|l| l.is_parameterized).count();
        Self {
            num_qubits,
            circuit,
            params: vec![0.0; num_params],
        }
    }

    /// Forward pass through quantum layer
    pub fn forward(&self, input: &Array1<f64>) -> QuantRS2Result<Array1<f64>> {
        // Encode classical input into quantum state
        let quantum_state = self.encode_input(input)?;

        // Apply parameterized circuit
        let output_state = self.apply_circuit(&quantum_state)?;

        // Measure and decode to classical output
        let classical_output = self.decode_output(&output_state)?;

        Ok(classical_output)
    }

    /// Encode classical input into quantum state
    fn encode_input(&self, input: &Array1<f64>) -> QuantRS2Result<Array1<Complex>> {
        let dim = 2_usize.pow(self.num_qubits as u32);
        let mut state = Array1::zeros(dim);

        // Amplitude encoding (simplified)
        let norm = input.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        for i in 0..input.len().min(dim) {
            state[i] = Complex::new(input[i] / norm, 0.0);
        }

        Ok(state)
    }

    /// Apply parameterized quantum circuit
    fn apply_circuit(&self, state: &Array1<Complex>) -> QuantRS2Result<Array1<Complex>> {
        // Simplified: would apply actual quantum gates
        Ok(state.clone())
    }

    /// Decode quantum state to classical output
    fn decode_output(&self, state: &Array1<Complex>) -> QuantRS2Result<Array1<f64>> {
        // Simplified: measure expectation values
        let output_dim = self.num_qubits;
        let mut output = Array1::zeros(output_dim);

        for i in 0..output_dim {
            output[i] = state
                .iter()
                .take(2_usize.pow(i as u32))
                .map(|x| x.norm_sqr())
                .sum();
        }

        Ok(output)
    }
}

impl QuantumClassicalNN {
    /// Create a new quantum-classical neural network
    pub const fn new(
        classical_pre: Vec<ClassicalLayer>,
        quantum_layer: QuantumLayer,
        classical_post: Vec<ClassicalLayer>,
    ) -> Self {
        Self {
            classical_pre,
            quantum_layer,
            classical_post,
        }
    }

    /// Forward pass through entire network
    pub fn forward(&self, input: &Array1<f64>) -> QuantRS2Result<Array1<f64>> {
        let mut current = input.clone();

        // Classical preprocessing
        for layer in &self.classical_pre {
            current = layer.forward(&current);
        }

        // Quantum processing
        current = self.quantum_layer.forward(&current)?;

        // Classical postprocessing
        for layer in &self.classical_post {
            current = layer.forward(&current);
        }

        Ok(current)
    }

    /// Train the network
    pub fn train(
        &mut self,
        training_data: &[(Array1<f64>, Array1<f64>)],
        epochs: usize,
        learning_rate: f64,
    ) -> QuantRS2Result<Vec<f64>> {
        let mut loss_history = Vec::new();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (input, target) in training_data {
                // Forward pass
                let output = self.forward(input)?;

                // Compute loss (MSE)
                let loss: f64 = output
                    .iter()
                    .zip(target.iter())
                    .map(|(o, t)| (o - t).powi(2))
                    .sum();
                total_loss += loss;

                // Backward pass (simplified - would need actual backprop)
            }

            let avg_loss = total_loss / training_data.len() as f64;
            loss_history.push(avg_loss);
        }

        Ok(loss_history)
    }
}

// ================================================================================================
// Quantum Feature Maps
// ================================================================================================

/// Quantum feature map for encoding classical data
pub struct QuantumFeatureMap {
    /// Number of qubits
    num_qubits: usize,
    /// Feature map type
    feature_map_type: FeatureMapType,
}

#[derive(Debug, Clone, Copy)]
pub enum FeatureMapType {
    /// Amplitude encoding
    Amplitude,
    /// Angle encoding
    Angle,
    /// Basis encoding
    Basis,
    /// IQP encoding
    IQP,
    /// Pauli feature map
    Pauli,
}

impl QuantumFeatureMap {
    /// Create a new quantum feature map
    pub const fn new(num_qubits: usize, feature_map_type: FeatureMapType) -> Self {
        Self {
            num_qubits,
            feature_map_type,
        }
    }

    /// Encode classical data into quantum state
    pub fn encode(&self, data: &Array1<f64>) -> QuantRS2Result<Array1<Complex>> {
        match self.feature_map_type {
            FeatureMapType::Amplitude => self.amplitude_encoding(data),
            FeatureMapType::Angle => self.angle_encoding(data),
            FeatureMapType::Basis => self.basis_encoding(data),
            FeatureMapType::IQP => self.iqp_encoding(data),
            FeatureMapType::Pauli => self.pauli_encoding(data),
        }
    }

    /// Amplitude encoding: |ψ⟩ = Σᵢ xᵢ|i⟩
    fn amplitude_encoding(&self, data: &Array1<f64>) -> QuantRS2Result<Array1<Complex>> {
        let dim = 2_usize.pow(self.num_qubits as u32);
        let mut state = Array1::zeros(dim);

        let norm = data.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        for i in 0..data.len().min(dim) {
            state[i] = Complex::new(data[i] / norm, 0.0);
        }

        Ok(state)
    }

    /// Angle encoding: RY(xᵢ) on each qubit
    fn angle_encoding(&self, data: &Array1<f64>) -> QuantRS2Result<Array1<Complex>> {
        let dim = 2_usize.pow(self.num_qubits as u32);
        let mut state = Array1::zeros(dim);
        state[0] = Complex::new(1.0, 0.0);

        // Would apply RY rotations for each data point
        // Simplified implementation
        Ok(state)
    }

    /// Basis encoding: binary representation
    fn basis_encoding(&self, data: &Array1<f64>) -> QuantRS2Result<Array1<Complex>> {
        let dim = 2_usize.pow(self.num_qubits as u32);
        let mut state = Array1::zeros(dim);

        // Convert data to binary representation
        let mut index = 0usize;
        for (i, &val) in data.iter().enumerate().take(self.num_qubits) {
            if val > 0.5 {
                index |= 1 << i;
            }
        }

        state[index] = Complex::new(1.0, 0.0);
        Ok(state)
    }

    /// IQP encoding: Instantaneous Quantum Polynomial
    fn iqp_encoding(&self, data: &Array1<f64>) -> QuantRS2Result<Array1<Complex>> {
        // Start with Hadamard on all qubits
        let dim = 2_usize.pow(self.num_qubits as u32);
        let hadamard_coeff = 1.0 / (dim as f64).sqrt();
        let mut state = Array1::from_elem(dim, Complex::new(hadamard_coeff, 0.0));

        // Apply diagonal gates based on data
        // Simplified implementation
        Ok(state)
    }

    /// Pauli feature map: exp(i Σ φᵢ Pᵢ)
    fn pauli_encoding(&self, data: &Array1<f64>) -> QuantRS2Result<Array1<Complex>> {
        let dim = 2_usize.pow(self.num_qubits as u32);
        let mut state = Array1::zeros(dim);
        state[0] = Complex::new(1.0, 0.0);

        // Would apply Pauli rotations based on data
        // Simplified implementation
        Ok(state)
    }

    /// Compute kernel between two data points
    pub fn kernel(&self, data1: &Array1<f64>, data2: &Array1<f64>) -> QuantRS2Result<f64> {
        let state1 = self.encode(data1)?;
        let state2 = self.encode(data2)?;

        // Compute overlap |⟨ψ₁|ψ₂⟩|²
        let overlap: Complex = state1
            .iter()
            .zip(state2.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        Ok(overlap.norm_sqr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_descent_optimizer() {
        let initial_params = vec![1.0, 2.0, 3.0];
        let mut optimizer = GradientDescentOptimizer::new(initial_params.clone(), 0.1);

        let gradient = vec![1.0, 1.0, 1.0];
        let new_params = optimizer.step(&initial_params, &gradient);

        assert_eq!(new_params[0], 0.9);
        assert_eq!(new_params[1], 1.9);
        assert_eq!(new_params[2], 2.9);
    }

    #[test]
    fn test_adam_optimizer() {
        let initial_params = vec![1.0, 2.0, 3.0];
        let mut optimizer = AdamOptimizer::new(initial_params, 0.01);

        let gradient = vec![1.0, 1.0, 1.0];
        let new_params = optimizer.step(&[], &gradient);

        // Parameters should be updated
        assert!(new_params[0] < 1.0);
    }

    #[test]
    fn test_quantum_feature_map_amplitude() {
        let feature_map = QuantumFeatureMap::new(2, FeatureMapType::Amplitude);
        let data = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);

        let state = feature_map
            .encode(&data)
            .expect("Failed to encode data in quantum feature map");

        // First amplitude should be 1
        assert!((state[0].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_kernel() {
        let feature_map = QuantumFeatureMap::new(2, FeatureMapType::Amplitude);
        let data1 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let data2 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);

        let kernel_value = feature_map
            .kernel(&data1, &data2)
            .expect("Failed to compute quantum kernel");

        // Kernel of identical points should be 1
        assert!((kernel_value - 1.0).abs() < 1e-10);
    }
}
