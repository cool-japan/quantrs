//! Quantum Boltzmann Machines
//!
//! This module implements quantum versions of Boltzmann machines for
//! probabilistic modeling and generative learning with quantum advantages.
//!
//! # Theoretical Background
//!
//! Quantum Boltzmann Machines (QBMs) extend classical Boltzmann machines
//! to leverage quantum superposition and entanglement for enhanced
//! representational power and potentially faster sampling via quantum annealing.
//!
//! # Key Features
//!
//! - **Quantum Restricted Boltzmann Machines (QRBM)**: Bipartite quantum architecture
//! - **Quantum Sampling**: Quantum annealing for Boltzmann sampling
//! - **Contrastive Divergence**: Quantum CD-k training algorithm
//! - **Energy-Based Learning**: Quantum energy function optimization
//! - **Generative Modeling**: Quantum state generation and sampling
//!
//! # References
//!
//! - "Quantum Boltzmann Machine" (Amin et al., 2018)
//! - "Training Quantum Boltzmann Machines" (Kieferová & Wiebe, 2017)
//! - "Quantum-Enhanced Machine Learning" (2024)

use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::f64::consts::PI;

/// Configuration for Quantum Restricted Boltzmann Machine
#[derive(Debug, Clone)]
pub struct QRBMConfig {
    /// Number of visible qubits
    pub num_visible: usize,
    /// Number of hidden qubits
    pub num_hidden: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of Gibbs sampling steps (CD-k)
    pub k_steps: usize,
    /// Temperature for Boltzmann sampling
    pub temperature: f64,
    /// Regularization strength
    pub l2_reg: f64,
}

impl Default for QRBMConfig {
    fn default() -> Self {
        Self {
            num_visible: 4,
            num_hidden: 2,
            learning_rate: 0.01,
            k_steps: 1,
            temperature: 1.0,
            l2_reg: 0.001,
        }
    }
}

/// Quantum Restricted Boltzmann Machine
#[derive(Debug, Clone)]
pub struct QuantumRBM {
    /// Configuration
    config: QRBMConfig,
    /// Weight matrix (visible × hidden)
    weights: Array2<f64>,
    /// Visible bias
    visible_bias: Array1<f64>,
    /// Hidden bias
    hidden_bias: Array1<f64>,
    /// Training history
    history: Vec<f64>,
}

impl QuantumRBM {
    /// Create new Quantum RBM
    pub fn new(config: QRBMConfig) -> Self {
        let mut rng = thread_rng();
        let scale = 0.01;

        let weights = Array2::from_shape_fn((config.num_visible, config.num_hidden), |_| {
            rng.gen_range(-scale..scale)
        });

        let visible_bias = Array1::zeros(config.num_visible);
        let hidden_bias = Array1::zeros(config.num_hidden);

        Self {
            config,
            weights,
            visible_bias,
            hidden_bias,
            history: Vec::new(),
        }
    }

    /// Train on batch of quantum states
    pub fn train_batch(&mut self, data: &[Array1<Complex64>]) -> QuantRS2Result<f64> {
        let mut total_error = 0.0;

        for state in data {
            // Convert quantum state to classical visible units
            let visible = self.quantum_to_classical(state)?;

            // Positive phase: compute hidden probabilities given data
            let hidden_probs = self.hidden_given_visible(&visible)?;
            let hidden_sample = self.sample_binary(&hidden_probs)?;

            // Negative phase: run k steps of Gibbs sampling
            let mut v_neg = visible.clone();
            let mut h_neg = hidden_sample.clone();

            for _ in 0..self.config.k_steps {
                v_neg = self.visible_given_hidden(&h_neg)?;
                h_neg = self.hidden_given_visible(&v_neg)?;
            }

            // Compute contrastive divergence gradient
            let pos_grad = self.outer_product(&visible, &hidden_probs);
            let neg_grad = self.outer_product(&v_neg, &h_neg);

            // Update weights and biases
            let grad = (pos_grad - neg_grad) / data.len() as f64;
            self.weights = &self.weights + &(grad * self.config.learning_rate)
                - &(&self.weights * self.config.l2_reg * self.config.learning_rate);

            let visible_grad = &visible - &v_neg;
            let hidden_grad = &hidden_probs - &h_neg;

            self.visible_bias = &self.visible_bias + &(visible_grad * self.config.learning_rate);
            self.hidden_bias = &self.hidden_bias + &(hidden_grad * self.config.learning_rate);

            // Reconstruction error
            let error = (&visible - &v_neg)
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            total_error += error;
        }

        let avg_error = total_error / data.len() as f64;
        self.history.push(avg_error);

        Ok(avg_error)
    }

    /// Convert quantum state to classical probabilities
    fn quantum_to_classical(&self, state: &Array1<Complex64>) -> QuantRS2Result<Array1<f64>> {
        let dim = 1 << self.config.num_visible;

        if state.len() != dim {
            return Err(QuantRS2Error::InvalidInput(format!(
                "State dimension {} doesn't match visible units 2^{}",
                state.len(),
                self.config.num_visible
            )));
        }

        // Extract marginal probabilities for each qubit
        let mut probs = Array1::zeros(self.config.num_visible);

        for q in 0..self.config.num_visible {
            let mut prob_one = 0.0;

            for i in 0..dim {
                let bit = (i >> q) & 1;
                if bit == 1 {
                    prob_one += state[i].norm_sqr();
                }
            }

            probs[q] = prob_one;
        }

        Ok(probs)
    }

    /// Compute hidden probabilities given visible
    fn hidden_given_visible(&self, visible: &Array1<f64>) -> QuantRS2Result<Array1<f64>> {
        let mut hidden_probs = self.hidden_bias.clone();

        for j in 0..self.config.num_hidden {
            for i in 0..self.config.num_visible {
                hidden_probs[j] += self.weights[[i, j]] * visible[i];
            }
            // Sigmoid activation
            hidden_probs[j] = 1.0 / (1.0 + (-hidden_probs[j] / self.config.temperature).exp());
        }

        Ok(hidden_probs)
    }

    /// Compute visible probabilities given hidden
    fn visible_given_hidden(&self, hidden: &Array1<f64>) -> QuantRS2Result<Array1<f64>> {
        let mut visible_probs = self.visible_bias.clone();

        for i in 0..self.config.num_visible {
            for j in 0..self.config.num_hidden {
                visible_probs[i] += self.weights[[i, j]] * hidden[j];
            }
            // Sigmoid activation
            visible_probs[i] = 1.0 / (1.0 + (-visible_probs[i] / self.config.temperature).exp());
        }

        Ok(visible_probs)
    }

    /// Sample binary units from probabilities
    fn sample_binary(&self, probs: &Array1<f64>) -> QuantRS2Result<Array1<f64>> {
        let mut rng = thread_rng();
        let mut samples = Array1::zeros(probs.len());

        for i in 0..probs.len() {
            samples[i] = if rng.gen::<f64>() < probs[i] {
                1.0
            } else {
                0.0
            };
        }

        Ok(samples)
    }

    /// Outer product of two vectors
    fn outer_product(&self, a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
        let mut result = Array2::zeros((a.len(), b.len()));

        for i in 0..a.len() {
            for j in 0..b.len() {
                result[[i, j]] = a[i] * b[j];
            }
        }

        result
    }

    /// Generate quantum sample from learned distribution
    pub fn generate_sample(&self) -> QuantRS2Result<Array1<Complex64>> {
        let mut rng = thread_rng();

        // Start with random hidden state
        let mut hidden = Array1::from_shape_fn(self.config.num_hidden, |_| {
            if rng.gen::<f64>() < 0.5 {
                0.0
            } else {
                1.0
            }
        });

        // Run Gibbs sampling to equilibrate
        for _ in 0..100 {
            let visible = self.visible_given_hidden(&hidden)?;
            hidden = self.hidden_given_visible(&visible)?;
        }

        // Final visible probabilities
        let visible_probs = self.visible_given_hidden(&hidden)?;

        // Convert to quantum state
        self.classical_to_quantum(&visible_probs)
    }

    /// Convert classical probabilities to quantum state
    fn classical_to_quantum(&self, probs: &Array1<f64>) -> QuantRS2Result<Array1<Complex64>> {
        let dim = 1 << self.config.num_visible;
        let mut state = Array1::zeros(dim);

        // Create product state from marginals
        for i in 0..dim {
            let mut amplitude = 1.0;

            for q in 0..self.config.num_visible {
                let bit = (i >> q) & 1;
                amplitude *= if bit == 1 {
                    probs[q].sqrt()
                } else {
                    (1.0 - probs[q]).sqrt()
                };
            }

            state[i] = Complex64::new(amplitude, 0.0);
        }

        // Normalize
        let norm: f64 = state
            .iter()
            .map(|x: &Complex64| x.norm_sqr())
            .sum::<f64>()
            .sqrt();
        for i in 0..dim {
            state[i] = state[i] / norm;
        }

        Ok(state)
    }

    /// Compute free energy of visible configuration
    pub fn free_energy(&self, visible: &Array1<f64>) -> QuantRS2Result<f64> {
        let mut energy = 0.0;

        // Visible bias term
        for i in 0..self.config.num_visible {
            energy -= self.visible_bias[i] * visible[i];
        }

        // Hidden layer contribution
        for j in 0..self.config.num_hidden {
            let mut h_input = self.hidden_bias[j];

            for i in 0..self.config.num_visible {
                h_input += self.weights[[i, j]] * visible[i];
            }

            energy -= h_input.exp().ln_1p();
        }

        Ok(energy)
    }

    /// Get training history
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Get weights
    pub const fn weights(&self) -> &Array2<f64> {
        &self.weights
    }
}

/// Deep Quantum Boltzmann Machine (stacked RBMs)
#[derive(Debug)]
pub struct DeepQuantumBoltzmannMachine {
    /// Layers of RBMs
    layers: Vec<QuantumRBM>,
    /// Layer configurations
    layer_configs: Vec<QRBMConfig>,
}

impl DeepQuantumBoltzmannMachine {
    /// Create new deep QBM
    pub fn new(layer_configs: Vec<QRBMConfig>) -> Self {
        let layers = layer_configs
            .iter()
            .map(|config| QuantumRBM::new(config.clone()))
            .collect();

        Self {
            layers,
            layer_configs,
        }
    }

    /// Pretrain layers greedily
    pub fn pretrain(
        &mut self,
        data: &[Array1<Complex64>],
        epochs_per_layer: usize,
    ) -> QuantRS2Result<Vec<Vec<f64>>> {
        let mut all_histories = Vec::new();
        let mut current_data = data.to_vec();
        let num_layers = self.layers.len();

        for layer_idx in 0..num_layers {
            println!("Pretraining layer {layer_idx}...");
            let mut layer_history = Vec::new();

            for epoch in 0..epochs_per_layer {
                let error = self.layers[layer_idx].train_batch(&current_data)?;
                layer_history.push(error);

                if epoch % 10 == 0 {
                    println!("  Epoch {epoch}: Error = {error:.6}");
                }
            }

            all_histories.push(layer_history);

            // Transform data for next layer
            if layer_idx < num_layers - 1 {
                current_data =
                    self.transform_to_next_layer(&current_data, &self.layers[layer_idx])?;
            }
        }

        Ok(all_histories)
    }

    /// Transform data through a layer
    fn transform_to_next_layer(
        &self,
        data: &[Array1<Complex64>],
        layer: &QuantumRBM,
    ) -> QuantRS2Result<Vec<Array1<Complex64>>> {
        let mut transformed = Vec::new();

        for state in data {
            let visible = layer.quantum_to_classical(state)?;
            let hidden_probs = layer.hidden_given_visible(&visible)?;

            // Convert hidden probabilities to quantum state
            transformed.push(layer.classical_to_quantum(&hidden_probs)?);
        }

        Ok(transformed)
    }

    /// Generate sample from deep model
    pub fn generate(&self) -> QuantRS2Result<Array1<Complex64>> {
        // Start from top layer
        let mut sample = self
            .layers
            .last()
            .ok_or_else(|| {
                QuantRS2Error::RuntimeError(
                    "No layers in deep quantum Boltzmann machine".to_string(),
                )
            })?
            .generate_sample()?;

        // Propagate down through layers
        for layer in self.layers.iter().rev().skip(1) {
            let hidden = layer.quantum_to_classical(&sample)?;
            let visible_probs = layer.visible_given_hidden(&hidden)?;
            sample = layer.classical_to_quantum(&visible_probs)?;
        }

        Ok(sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qrbm() {
        let config = QRBMConfig {
            num_visible: 2,
            num_hidden: 2,
            learning_rate: 0.01,
            k_steps: 1,
            temperature: 1.0,
            l2_reg: 0.001,
        };

        let mut rbm = QuantumRBM::new(config);

        // Create simple training data
        let state = Array1::from_vec(vec![
            Complex64::new(0.7, 0.0),
            Complex64::new(0.3, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.6, 0.0),
        ]);

        let error = rbm
            .train_batch(&[state])
            .expect("Failed to train quantum RBM on batch");
        assert!(error >= 0.0);
    }

    #[test]
    fn test_deep_qbm() {
        let layer1 = QRBMConfig {
            num_visible: 2,
            num_hidden: 2,
            ..Default::default()
        };

        let layer2 = QRBMConfig {
            num_visible: 2,
            num_hidden: 1,
            ..Default::default()
        };

        let dbm = DeepQuantumBoltzmannMachine::new(vec![layer1, layer2]);
        assert_eq!(dbm.layers.len(), 2);
    }
}
