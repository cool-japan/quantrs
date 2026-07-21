//! # QuantumContinuousFlow - update_flow_parameters_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{FlowForwardOutput, FlowTrainingConfig};

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

/// Clip a 1-D gradient in place to a maximum L2 norm.
fn clip_gradient_norm_1d(gradient: &mut Array1<f64>, max_norm: f64) {
    let norm = gradient.mapv(|g| g * g).sum().sqrt();
    if norm > max_norm && norm > 0.0 {
        gradient.mapv_inplace(|g| g * (max_norm / norm));
    }
}

/// Clip a 2-D gradient in place to a maximum L2 (Frobenius) norm.
fn clip_gradient_norm_2d(gradient: &mut Array2<f64>, max_norm: f64) {
    let norm = gradient.mapv(|g| g * g).sum().sqrt();
    if norm > max_norm && norm > 0.0 {
        gradient.mapv_inplace(|g| g * (max_norm / norm));
    }
}

impl QuantumContinuousFlow {
    /// Update flow parameters from a single training sample's forward pass.
    ///
    /// `forward_output` does not carry the raw input sample, only the latent
    /// code it was mapped to, so this reconstructs the input exactly via the
    /// flow's own [`Self::inverse`] and then estimates a real gradient of the
    /// negative log-likelihood objective with respect to every flow layer's
    /// quantum and classical parameters using SPSA (simultaneous perturbation
    /// stochastic approximation): all parameters are perturbed together along
    /// one random +/-1 direction, the loss is re-evaluated at the plus and
    /// minus perturbations (two extra real forward passes), and the resulting
    /// finite-difference estimate is used to take a genuine gradient-descent
    /// step, in addition to the existing learning-rate decay.
    pub(super) fn update_flow_parameters(
        &mut self,
        forward_output: &FlowForwardOutput,
        config: &FlowTrainingConfig,
    ) -> Result<()> {
        // Reconstruct the training sample from the latent code via the exact
        // inverse flow so the loss can be re-evaluated at perturbed parameters.
        let reconstructed_input = self.inverse(&forward_output.latent_sample)?.data_sample;

        if self.flow_layers.is_empty() {
            self.optimization_state.learning_rate *= config.learning_rate_decay;
            return Ok(());
        }

        let perturbation_scale = (config.learning_rate * 0.1).max(1e-4);
        let mut rng = thread_rng();

        // One Rademacher (+/-1) direction per parameter, for every layer.
        let mut quantum_directions: Vec<Array1<f64>> = Vec::with_capacity(self.flow_layers.len());
        let mut classical_directions: Vec<Array2<f64>> = Vec::with_capacity(self.flow_layers.len());
        for layer in &self.flow_layers {
            quantum_directions.push(layer.quantum_parameters.mapv(|_| {
                if rng.random::<f64>() < 0.5 {
                    -1.0
                } else {
                    1.0
                }
            }));
            classical_directions.push(layer.classical_parameters.mapv(|_| {
                if rng.random::<f64>() < 0.5 {
                    -1.0
                } else {
                    1.0
                }
            }));
        }

        // Evaluate the loss at the "+" perturbation.
        for (idx, layer) in self.flow_layers.iter_mut().enumerate() {
            layer.quantum_parameters =
                &layer.quantum_parameters + &(&quantum_directions[idx] * perturbation_scale);
            layer.classical_parameters =
                &layer.classical_parameters + &(&classical_directions[idx] * perturbation_scale);
        }
        let loss_plus = -self.forward(&reconstructed_input)?.quantum_log_probability;

        // Move from "+" to "-" (subtract twice the perturbation) and re-evaluate.
        for (idx, layer) in self.flow_layers.iter_mut().enumerate() {
            layer.quantum_parameters = &layer.quantum_parameters
                - &(&quantum_directions[idx] * (2.0 * perturbation_scale));
            layer.classical_parameters = &layer.classical_parameters
                - &(&classical_directions[idx] * (2.0 * perturbation_scale));
        }
        let loss_minus = -self.forward(&reconstructed_input)?.quantum_log_probability;
        let loss_delta = loss_plus - loss_minus;

        // Restore original parameters ("-" + one perturbation back to center),
        // then apply a genuine gradient-descent step using the SPSA estimate
        //   d(loss)/d(theta_j) ~= (loss_plus - loss_minus) / (2 * scale * direction_j)
        // Since direction_j in {-1, +1}, 1 / direction_j == direction_j.
        let gradient_clip = config.gradient_clipping_norm.max(1e-12);
        for (idx, layer) in self.flow_layers.iter_mut().enumerate() {
            layer.quantum_parameters =
                &layer.quantum_parameters + &(&quantum_directions[idx] * perturbation_scale);
            layer.classical_parameters =
                &layer.classical_parameters + &(&classical_directions[idx] * perturbation_scale);

            let mut quantum_gradient = quantum_directions[idx]
                .mapv(|direction| (loss_delta / (2.0 * perturbation_scale)) * direction);
            let mut classical_gradient = classical_directions[idx]
                .mapv(|direction| (loss_delta / (2.0 * perturbation_scale)) * direction);

            clip_gradient_norm_1d(&mut quantum_gradient, gradient_clip);
            clip_gradient_norm_2d(&mut classical_gradient, gradient_clip);

            layer.quantum_parameters = &layer.quantum_parameters
                - &(&quantum_gradient * self.optimization_state.learning_rate);
            layer.classical_parameters = &layer.classical_parameters
                - &(&classical_gradient * self.optimization_state.learning_rate);
        }

        self.optimization_state.learning_rate *= config.learning_rate_decay;
        Ok(())
    }
}
