//! Quantum Meta-Learning
//!
//! This module implements meta-learning algorithms for quantum machine learning,
//! enabling rapid adaptation to new tasks with minimal training data.
//!
//! # Theoretical Background
//!
//! Quantum meta-learning extends classical meta-learning (MAML, Reptile) to
//! quantum neural networks. The goal is to learn an initialization of quantum
//! circuit parameters that can quickly adapt to new tasks through fine-tuning.
//!
//! # Key Algorithms
//!
//! - **Quantum MAML**: Model-Agnostic Meta-Learning for quantum circuits
//! - **Quantum Reptile**: First-order approximation of MAML
//! - **Quantum ProtoNets**: Prototype networks using quantum metric learning
//! - **Quantum Matching Networks**: Attention-based few-shot learning
//!
//! # Applications
//!
//! - Few-shot quantum classification
//! - Fast quantum state tomography
//! - Adaptive quantum control
//! - Quantum drug discovery with limited data
//!
//! # References
//!
//! - "Meta-Learning for Quantum Neural Networks"
//! - "Few-Shot Learning with Quantum Classifiers"
//! - "Quantum Model-Agnostic Meta-Learning"

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::f64::consts::PI;

/// Configuration for quantum meta-learning
#[derive(Debug, Clone)]
pub struct QuantumMetaLearningConfig {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Inner loop learning rate
    pub inner_lr: f64,
    /// Outer loop learning rate (meta-learning)
    pub outer_lr: f64,
    /// Number of inner loop steps
    pub inner_steps: usize,
    /// Number of support examples per class
    pub n_support: usize,
    /// Number of query examples per class
    pub n_query: usize,
    /// Number of classes per task
    pub n_way: usize,
    /// Meta-batch size (number of tasks)
    pub meta_batch_size: usize,
}

impl Default for QuantumMetaLearningConfig {
    fn default() -> Self {
        Self {
            num_qubits: 4,
            circuit_depth: 4,
            inner_lr: 0.01,
            outer_lr: 0.001,
            inner_steps: 5,
            n_support: 5,
            n_query: 15,
            n_way: 2,
            meta_batch_size: 4,
        }
    }
}

/// Quantum task for meta-learning
#[derive(Debug, Clone)]
pub struct QuantumTask {
    /// Support set states
    pub support_states: Vec<Array1<Complex64>>,
    /// Support set labels
    pub support_labels: Vec<usize>,
    /// Query set states
    pub query_states: Vec<Array1<Complex64>>,
    /// Query set labels
    pub query_labels: Vec<usize>,
}

impl QuantumTask {
    /// Create new quantum task
    pub const fn new(
        support_states: Vec<Array1<Complex64>>,
        support_labels: Vec<usize>,
        query_states: Vec<Array1<Complex64>>,
        query_labels: Vec<usize>,
    ) -> Self {
        Self {
            support_states,
            support_labels,
            query_states,
            query_labels,
        }
    }

    /// Generate random task for testing
    pub fn random(num_qubits: usize, n_way: usize, n_support: usize, n_query: usize) -> Self {
        let mut rng = thread_rng();
        let dim = 1 << num_qubits;

        let mut support_states = Vec::new();
        let mut support_labels = Vec::new();
        let mut query_states = Vec::new();
        let mut query_labels = Vec::new();

        for class in 0..n_way {
            // Generate class prototype
            let mut prototype = Array1::from_shape_fn(dim, |_| {
                Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
            });
            let norm: f64 = prototype.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            for i in 0..dim {
                prototype[i] = prototype[i] / norm;
            }

            // Generate support examples
            for _ in 0..n_support {
                let mut state = prototype.clone();
                // Add small noise
                for i in 0..dim {
                    state[i] = state[i]
                        + Complex64::new(rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1));
                }
                let norm: f64 = state.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                for i in 0..dim {
                    state[i] = state[i] / norm;
                }
                support_states.push(state);
                support_labels.push(class);
            }

            // Generate query examples
            for _ in 0..n_query {
                let mut state = prototype.clone();
                // Add small noise
                for i in 0..dim {
                    state[i] = state[i]
                        + Complex64::new(rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1));
                }
                let norm: f64 = state.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                for i in 0..dim {
                    state[i] = state[i] / norm;
                }
                query_states.push(state);
                query_labels.push(class);
            }
        }

        Self {
            support_states,
            support_labels,
            query_states,
            query_labels,
        }
    }
}

/// Quantum circuit for meta-learning
#[derive(Debug, Clone)]
pub struct QuantumMetaCircuit {
    /// Number of qubits
    num_qubits: usize,
    /// Circuit depth
    depth: usize,
    /// Number of output classes
    num_classes: usize,
    /// Circuit parameters
    params: Array2<f64>,
    /// Readout weights
    readout_weights: Array2<f64>,
}

impl QuantumMetaCircuit {
    /// Create new quantum meta circuit
    pub fn new(num_qubits: usize, depth: usize, num_classes: usize) -> Self {
        let mut rng = thread_rng();

        let params = Array2::from_shape_fn((depth, num_qubits * 3), |_| rng.gen_range(-PI..PI));

        let scale = (2.0 / num_qubits as f64).sqrt();
        let readout_weights =
            Array2::from_shape_fn((num_classes, num_qubits), |_| rng.gen_range(-scale..scale));

        Self {
            num_qubits,
            depth,
            num_classes,
            params,
            readout_weights,
        }
    }

    /// Forward pass
    pub fn forward(&self, state: &Array1<Complex64>) -> QuantRS2Result<Array1<f64>> {
        // Apply quantum circuit
        let mut encoded = state.clone();

        for layer in 0..self.depth {
            // Rotation gates
            for q in 0..self.num_qubits {
                let rx = self.params[[layer, q * 3]];
                let ry = self.params[[layer, q * 3 + 1]];
                let rz = self.params[[layer, q * 3 + 2]];

                encoded = self.apply_rotation(&encoded, q, rx, ry, rz)?;
            }

            // Entangling gates
            for q in 0..self.num_qubits - 1 {
                encoded = self.apply_cnot(&encoded, q, q + 1)?;
            }
        }

        // Measure expectations and classify
        let mut expectations = Array1::zeros(self.num_qubits);
        for q in 0..self.num_qubits {
            expectations[q] = self.pauli_z_expectation(&encoded, q)?;
        }

        // Linear readout
        let mut logits = Array1::zeros(self.num_classes);
        for i in 0..self.num_classes {
            for j in 0..self.num_qubits {
                logits[i] += self.readout_weights[[i, j]] * expectations[j];
            }
        }

        // Softmax
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut probs = Array1::zeros(self.num_classes);
        let mut sum_exp = 0.0;

        for i in 0..self.num_classes {
            probs[i] = (logits[i] - max_logit).exp();
            sum_exp += probs[i];
        }

        for i in 0..self.num_classes {
            probs[i] /= sum_exp;
        }

        Ok(probs)
    }

    /// Compute loss
    pub fn compute_loss(
        &self,
        states: &[Array1<Complex64>],
        labels: &[usize],
    ) -> QuantRS2Result<f64> {
        let mut total_loss = 0.0;

        for (state, &label) in states.iter().zip(labels.iter()) {
            let probs = self.forward(state)?;
            // Cross-entropy loss
            total_loss -= probs[label].ln();
        }

        Ok(total_loss / states.len() as f64)
    }

    /// Compute gradients (simplified with finite differences)
    pub fn compute_gradients(
        &self,
        states: &[Array1<Complex64>],
        labels: &[usize],
    ) -> QuantRS2Result<(Array2<f64>, Array2<f64>)> {
        let epsilon = 1e-4;

        // Gradients for circuit parameters
        let mut param_grads = Array2::zeros(self.params.dim());

        for i in 0..self.params.shape()[0] {
            for j in 0..self.params.shape()[1] {
                let mut circuit_plus = self.clone();
                circuit_plus.params[[i, j]] += epsilon;
                let loss_plus = circuit_plus.compute_loss(states, labels)?;

                let mut circuit_minus = self.clone();
                circuit_minus.params[[i, j]] -= epsilon;
                let loss_minus = circuit_minus.compute_loss(states, labels)?;

                param_grads[[i, j]] = (loss_plus - loss_minus) / (2.0 * epsilon);
            }
        }

        // Gradients for readout weights
        let mut readout_grads = Array2::zeros(self.readout_weights.dim());

        for i in 0..self.readout_weights.shape()[0] {
            for j in 0..self.readout_weights.shape()[1] {
                let mut circuit_plus = self.clone();
                circuit_plus.readout_weights[[i, j]] += epsilon;
                let loss_plus = circuit_plus.compute_loss(states, labels)?;

                let mut circuit_minus = self.clone();
                circuit_minus.readout_weights[[i, j]] -= epsilon;
                let loss_minus = circuit_minus.compute_loss(states, labels)?;

                readout_grads[[i, j]] = (loss_plus - loss_minus) / (2.0 * epsilon);
            }
        }

        Ok((param_grads, readout_grads))
    }

    /// Update parameters
    pub fn update_params(
        &mut self,
        param_grads: &Array2<f64>,
        readout_grads: &Array2<f64>,
        lr: f64,
    ) {
        self.params = &self.params - &(param_grads * lr);
        self.readout_weights = &self.readout_weights - &(readout_grads * lr);
    }

    /// Helper methods
    fn apply_rotation(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        rx: f64,
        ry: f64,
        rz: f64,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let mut result = state.clone();
        result = self.apply_rz_gate(&result, qubit, rz)?;
        result = self.apply_ry_gate(&result, qubit, ry)?;
        result = self.apply_rx_gate(&result, qubit, rx)?;
        Ok(result)
    }

    fn apply_rx_gate(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        angle: f64,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = Array1::zeros(dim);
        let cos_half = Complex64::new((angle / 2.0).cos(), 0.0);
        let sin_half = Complex64::new(0.0, -(angle / 2.0).sin());

        for i in 0..dim {
            let j = i ^ (1 << qubit);
            new_state[i] = state[i] * cos_half + state[j] * sin_half;
        }

        Ok(new_state)
    }

    fn apply_ry_gate(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        angle: f64,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = Array1::zeros(dim);
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            let j = i ^ (1 << qubit);
            if bit == 0 {
                new_state[i] = state[i] * cos_half - state[j] * sin_half;
            } else {
                new_state[i] = state[i] * cos_half + state[j] * sin_half;
            }
        }

        Ok(new_state)
    }

    fn apply_rz_gate(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        angle: f64,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = state.clone();
        let phase = Complex64::new((angle / 2.0).cos(), -(angle / 2.0).sin());

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            new_state[i] = if bit == 1 {
                new_state[i] * phase
            } else {
                new_state[i] * phase.conj()
            };
        }

        Ok(new_state)
    }

    fn apply_cnot(
        &self,
        state: &Array1<Complex64>,
        control: usize,
        target: usize,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = state.clone();

        for i in 0..dim {
            let control_bit = (i >> control) & 1;
            if control_bit == 1 {
                let j = i ^ (1 << target);
                if i < j {
                    let temp = new_state[i];
                    new_state[i] = new_state[j];
                    new_state[j] = temp;
                }
            }
        }

        Ok(new_state)
    }

    fn pauli_z_expectation(&self, state: &Array1<Complex64>, qubit: usize) -> QuantRS2Result<f64> {
        let dim = state.len();
        let mut expectation = 0.0;

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            let sign = if bit == 0 { 1.0 } else { -1.0 };
            expectation += sign * state[i].norm_sqr();
        }

        Ok(expectation)
    }
}

/// Quantum MAML (Model-Agnostic Meta-Learning)
#[derive(Debug, Clone)]
pub struct QuantumMAML {
    /// Configuration
    config: QuantumMetaLearningConfig,
    /// Meta-model (initialization point)
    meta_model: QuantumMetaCircuit,
}

impl QuantumMAML {
    /// Create new Quantum MAML
    pub fn new(config: QuantumMetaLearningConfig) -> Self {
        let meta_model =
            QuantumMetaCircuit::new(config.num_qubits, config.circuit_depth, config.n_way);

        Self { config, meta_model }
    }

    /// Meta-training step
    pub fn meta_train_step(&mut self, tasks: &[QuantumTask]) -> QuantRS2Result<f64> {
        let mut meta_param_grads = Array2::zeros(self.meta_model.params.dim());
        let mut meta_readout_grads = Array2::zeros(self.meta_model.readout_weights.dim());
        let mut total_loss = 0.0;

        for task in tasks {
            // Inner loop: adapt to task
            let mut adapted_model = self.meta_model.clone();

            for _ in 0..self.config.inner_steps {
                let (param_grads, readout_grads) =
                    adapted_model.compute_gradients(&task.support_states, &task.support_labels)?;

                adapted_model.update_params(&param_grads, &readout_grads, self.config.inner_lr);
            }

            // Compute loss on query set
            let query_loss = adapted_model.compute_loss(&task.query_states, &task.query_labels)?;
            total_loss += query_loss;

            // Compute meta-gradients
            let (param_grads, readout_grads) =
                adapted_model.compute_gradients(&task.query_states, &task.query_labels)?;

            meta_param_grads = meta_param_grads + param_grads;
            meta_readout_grads = meta_readout_grads + readout_grads;
        }

        // Average gradients
        meta_param_grads = meta_param_grads / (tasks.len() as f64);
        meta_readout_grads = meta_readout_grads / (tasks.len() as f64);

        // Update meta-model
        self.meta_model
            .update_params(&meta_param_grads, &meta_readout_grads, self.config.outer_lr);

        Ok(total_loss / tasks.len() as f64)
    }

    /// Adapt to new task
    pub fn adapt(&self, task: &QuantumTask) -> QuantRS2Result<QuantumMetaCircuit> {
        let mut adapted_model = self.meta_model.clone();

        for _ in 0..self.config.inner_steps {
            let (param_grads, readout_grads) =
                adapted_model.compute_gradients(&task.support_states, &task.support_labels)?;

            adapted_model.update_params(&param_grads, &readout_grads, self.config.inner_lr);
        }

        Ok(adapted_model)
    }

    /// Evaluate on new task
    pub fn evaluate(&self, task: &QuantumTask) -> QuantRS2Result<f64> {
        let adapted_model = self.adapt(task)?;

        let mut correct = 0;
        for (state, &label) in task.query_states.iter().zip(task.query_labels.iter()) {
            let probs = adapted_model.forward(state)?;
            let mut max_prob = f64::NEG_INFINITY;
            let mut predicted = 0;

            for (i, &prob) in probs.iter().enumerate() {
                if prob > max_prob {
                    max_prob = prob;
                    predicted = i;
                }
            }

            if predicted == label {
                correct += 1;
            }
        }

        Ok(correct as f64 / task.query_states.len() as f64)
    }

    /// Get meta-model
    pub const fn meta_model(&self) -> &QuantumMetaCircuit {
        &self.meta_model
    }
}

/// Quantum Reptile (simpler first-order MAML)
#[derive(Debug, Clone)]
pub struct QuantumReptile {
    /// Configuration
    config: QuantumMetaLearningConfig,
    /// Meta-model
    meta_model: QuantumMetaCircuit,
}

impl QuantumReptile {
    /// Create new Quantum Reptile
    pub fn new(config: QuantumMetaLearningConfig) -> Self {
        let meta_model =
            QuantumMetaCircuit::new(config.num_qubits, config.circuit_depth, config.n_way);

        Self { config, meta_model }
    }

    /// Meta-training step
    pub fn meta_train_step(&mut self, task: &QuantumTask) -> QuantRS2Result<f64> {
        // Adapt to task
        let mut adapted_model = self.meta_model.clone();

        for _ in 0..self.config.inner_steps {
            let (param_grads, readout_grads) =
                adapted_model.compute_gradients(&task.support_states, &task.support_labels)?;

            adapted_model.update_params(&param_grads, &readout_grads, self.config.inner_lr);
        }

        // Compute loss
        let loss = adapted_model.compute_loss(&task.query_states, &task.query_labels)?;

        // Update meta-model towards adapted model
        let param_diff = &adapted_model.params - &self.meta_model.params;
        let readout_diff = &adapted_model.readout_weights - &self.meta_model.readout_weights;

        self.meta_model.params = &self.meta_model.params + &(param_diff * self.config.outer_lr);
        self.meta_model.readout_weights =
            &self.meta_model.readout_weights + &(readout_diff * self.config.outer_lr);

        Ok(loss)
    }

    /// Adapt to new task (same as MAML)
    pub fn adapt(&self, task: &QuantumTask) -> QuantRS2Result<QuantumMetaCircuit> {
        let mut adapted_model = self.meta_model.clone();

        for _ in 0..self.config.inner_steps {
            let (param_grads, readout_grads) =
                adapted_model.compute_gradients(&task.support_states, &task.support_labels)?;

            adapted_model.update_params(&param_grads, &readout_grads, self.config.inner_lr);
        }

        Ok(adapted_model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_meta_circuit() {
        let circuit = QuantumMetaCircuit::new(3, 2, 2);

        let state = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);

        let probs = circuit
            .forward(&state)
            .expect("forward pass should succeed");
        assert_eq!(probs.len(), 2);

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantum_maml() {
        let config = QuantumMetaLearningConfig {
            num_qubits: 2,
            circuit_depth: 2,
            inner_lr: 0.01,
            outer_lr: 0.001,
            inner_steps: 3,
            n_support: 2,
            n_query: 5,
            n_way: 2,
            meta_batch_size: 2,
        };

        let maml = QuantumMAML::new(config.clone());

        let task = QuantumTask::random(
            config.num_qubits,
            config.n_way,
            config.n_support,
            config.n_query,
        );

        let adapted_model = maml.adapt(&task).expect("MAML adaptation should succeed");
        let probs = adapted_model
            .forward(&task.query_states[0])
            .expect("adapted model forward pass should succeed");

        assert_eq!(probs.len(), config.n_way);
    }
}
