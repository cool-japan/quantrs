//! Quantum Contrastive Learning
//!
//! This module implements contrastive learning methods for quantum machine learning,
//! enabling self-supervised representation learning without labeled data.
//!
//! # Theoretical Background
//!
//! Quantum contrastive learning extends classical contrastive methods (SimCLR, MoCo)
//! to the quantum domain. The key idea is to learn quantum representations that
//! maximize agreement between differently augmented views of the same quantum state
//! while minimizing agreement with other states.
//!
//! # Key Components
//!
//! - **Quantum Data Augmentation**: Noise channels, rotations, and decoherence
//! - **Quantum Encoder**: Parameterized quantum circuit for representation
//! - **Contrastive Loss**: Quantum fidelity-based NT-Xent loss
//! - **Momentum Encoder**: Slow-moving encoder for stable learning
//!
//! # References
//!
//! - "Quantum Contrastive Learning for Noisy Datasets" (2023)
//! - "Self-Supervised Quantum Machine Learning" (2024)
//! - "Contrastive Learning with Quantum Neural Networks" (2024)

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::f64::consts::PI;

/// Configuration for quantum contrastive learning
#[derive(Debug, Clone)]
pub struct QuantumContrastiveConfig {
    /// Number of qubits
    pub num_qubits: usize,
    /// Encoder depth
    pub encoder_depth: usize,
    /// Temperature parameter for contrastive loss
    pub temperature: f64,
    /// Momentum coefficient for momentum encoder
    pub momentum: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of augmentation views
    pub num_views: usize,
}

impl Default for QuantumContrastiveConfig {
    fn default() -> Self {
        Self {
            num_qubits: 4,
            encoder_depth: 4,
            temperature: 0.5,
            momentum: 0.999,
            batch_size: 32,
            num_views: 2,
        }
    }
}

/// Quantum data augmentation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumAugmentation {
    /// Random unitary rotations
    RandomRotation,
    /// Depolarizing noise
    DepolarizingNoise,
    /// Amplitude damping
    AmplitudeDamping,
    /// Phase damping
    PhaseDamping,
    /// Random Pauli gates
    RandomPauli,
    /// Circuit cutting and rejoining
    CircuitCutting,
}

/// Quantum augmenter for creating multiple views of quantum states
#[derive(Debug, Clone)]
pub struct QuantumAugmenter {
    /// Number of qubits
    num_qubits: usize,
    /// Augmentation strategies
    strategies: Vec<QuantumAugmentation>,
    /// Noise strength
    noise_strength: f64,
}

impl QuantumAugmenter {
    /// Create new quantum augmenter
    pub const fn new(
        num_qubits: usize,
        strategies: Vec<QuantumAugmentation>,
        noise_strength: f64,
    ) -> Self {
        Self {
            num_qubits,
            strategies,
            noise_strength,
        }
    }

    /// Augment a quantum state
    pub fn augment(
        &self,
        state: &Array1<Complex64>,
        strategy: QuantumAugmentation,
    ) -> QuantRS2Result<Array1<Complex64>> {
        match strategy {
            QuantumAugmentation::RandomRotation => self.random_rotation(state),
            QuantumAugmentation::DepolarizingNoise => self.depolarizing_noise(state),
            QuantumAugmentation::AmplitudeDamping => self.amplitude_damping(state),
            QuantumAugmentation::PhaseDamping => self.phase_damping(state),
            QuantumAugmentation::RandomPauli => self.random_pauli(state),
            QuantumAugmentation::CircuitCutting => self.circuit_cutting(state),
        }
    }

    /// Apply random unitary rotation
    fn random_rotation(&self, state: &Array1<Complex64>) -> QuantRS2Result<Array1<Complex64>> {
        let mut rng = thread_rng();
        let mut new_state = state.clone();

        // Apply random single-qubit rotations
        for q in 0..self.num_qubits {
            let angle = rng.gen_range(-PI..PI) * self.noise_strength;
            let axis = rng.gen_range(0..3); // X, Y, or Z

            new_state = match axis {
                0 => self.apply_rx(&new_state, q, angle)?,
                1 => self.apply_ry(&new_state, q, angle)?,
                _ => self.apply_rz(&new_state, q, angle)?,
            };
        }

        Ok(new_state)
    }

    /// Apply depolarizing noise
    fn depolarizing_noise(&self, state: &Array1<Complex64>) -> QuantRS2Result<Array1<Complex64>> {
        let mut rng = thread_rng();
        let p = self.noise_strength;
        let dim = state.len();
        let mut new_state = state.clone();

        // With probability p, replace with maximally mixed state
        if rng.gen::<f64>() < p {
            let uniform_val = Complex64::new(1.0 / (dim as f64).sqrt(), 0.0);
            new_state = Array1::from_elem(dim, uniform_val);
        }

        Ok(new_state)
    }

    /// Apply amplitude damping (simulates energy relaxation)
    fn amplitude_damping(&self, state: &Array1<Complex64>) -> QuantRS2Result<Array1<Complex64>> {
        let gamma = self.noise_strength;
        let mut new_state = state.clone();

        for q in 0..self.num_qubits {
            new_state = self.apply_amplitude_damping_qubit(&new_state, q, gamma)?;
        }

        Ok(new_state)
    }

    /// Apply amplitude damping to single qubit
    fn apply_amplitude_damping_qubit(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        gamma: f64,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = state.clone();

        let k0_coeff = 1.0;
        let k1_coeff = gamma.sqrt();

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            if bit == 1 {
                let j = i ^ (1 << qubit);
                // |1⟩ → √γ |0⟩
                new_state[j] = new_state[j] + state[i] * k1_coeff;
                new_state[i] = state[i] * ((1.0 - gamma).sqrt());
            }
        }

        Ok(new_state)
    }

    /// Apply phase damping (simulates dephasing)
    fn phase_damping(&self, state: &Array1<Complex64>) -> QuantRS2Result<Array1<Complex64>> {
        let lambda = self.noise_strength;
        let mut new_state = state.clone();

        for q in 0..self.num_qubits {
            new_state = self.apply_phase_damping_qubit(&new_state, q, lambda)?;
        }

        Ok(new_state)
    }

    /// Apply phase damping to single qubit
    fn apply_phase_damping_qubit(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        lambda: f64,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = state.clone();

        let damp_factor = (1.0 - lambda).sqrt();

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            if bit == 1 {
                new_state[i] = state[i] * damp_factor;
            }
        }

        Ok(new_state)
    }

    /// Apply random Pauli gates
    fn random_pauli(&self, state: &Array1<Complex64>) -> QuantRS2Result<Array1<Complex64>> {
        let mut rng = thread_rng();
        let mut new_state = state.clone();

        for q in 0..self.num_qubits {
            if rng.gen::<f64>() < self.noise_strength {
                let pauli = rng.gen_range(0..4); // I, X, Y, Z
                new_state = match pauli {
                    1 => self.apply_pauli_x(&new_state, q)?,
                    2 => self.apply_pauli_y(&new_state, q)?,
                    3 => self.apply_pauli_z(&new_state, q)?,
                    _ => new_state,
                };
            }
        }

        Ok(new_state)
    }

    /// Circuit cutting augmentation
    fn circuit_cutting(&self, state: &Array1<Complex64>) -> QuantRS2Result<Array1<Complex64>> {
        // For now, just apply a combination of other augmentations
        let mut new_state = state.clone();
        new_state = self.random_rotation(&new_state)?;
        new_state = self.phase_damping(&new_state)?;
        Ok(new_state)
    }

    /// Helper: Apply Pauli-X
    fn apply_pauli_x(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = state.clone();

        for i in 0..dim {
            let j = i ^ (1 << qubit);
            if i < j {
                let temp = new_state[i];
                new_state[i] = new_state[j];
                new_state[j] = temp;
            }
        }

        Ok(new_state)
    }

    /// Helper: Apply Pauli-Y
    fn apply_pauli_y(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = state.clone();

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            let j = i ^ (1 << qubit);
            if i < j {
                let factor = if bit == 0 {
                    Complex64::new(0.0, 1.0)
                } else {
                    Complex64::new(0.0, -1.0)
                };
                let temp = new_state[i];
                new_state[i] = new_state[j] * factor;
                new_state[j] = temp * (-factor);
            }
        }

        Ok(new_state)
    }

    /// Helper: Apply Pauli-Z
    fn apply_pauli_z(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = state.clone();

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            if bit == 1 {
                new_state[i] = -new_state[i];
            }
        }

        Ok(new_state)
    }

    /// Helper rotation functions
    fn apply_rx(
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

    fn apply_ry(
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

    fn apply_rz(
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
}

/// Quantum encoder network
#[derive(Debug, Clone)]
pub struct QuantumEncoder {
    /// Number of qubits
    num_qubits: usize,
    /// Circuit depth
    depth: usize,
    /// Parameters
    params: Array2<f64>,
}

impl QuantumEncoder {
    /// Create new quantum encoder
    pub fn new(num_qubits: usize, depth: usize) -> Self {
        let mut rng = thread_rng();
        let num_params = num_qubits * depth * 3; // 3 rotations per qubit per layer

        let params = Array2::from_shape_fn((depth, num_qubits * 3), |_| rng.gen_range(-PI..PI));

        Self {
            num_qubits,
            depth,
            params,
        }
    }

    /// Encode quantum state
    pub fn encode(&self, state: &Array1<Complex64>) -> QuantRS2Result<Array1<Complex64>> {
        let mut encoded = state.clone();

        // Apply parameterized layers
        for layer in 0..self.depth {
            // Single-qubit rotations
            for q in 0..self.num_qubits {
                let rx_angle = self.params[[layer, q * 3]];
                let ry_angle = self.params[[layer, q * 3 + 1]];
                let rz_angle = self.params[[layer, q * 3 + 2]];

                encoded = self.apply_rotation(&encoded, q, rx_angle, ry_angle, rz_angle)?;
            }

            // Entangling layer (CNOT ladder)
            for q in 0..self.num_qubits - 1 {
                encoded = self.apply_cnot(&encoded, q, q + 1)?;
            }
        }

        Ok(encoded)
    }

    /// Apply rotation gates
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

    /// Apply CNOT gate
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

    /// Update parameters
    pub fn update_params(&mut self, gradients: &Array2<f64>, learning_rate: f64) {
        self.params = &self.params - &(gradients * learning_rate);
    }

    /// Get parameters
    pub const fn params(&self) -> &Array2<f64> {
        &self.params
    }
}

/// Quantum contrastive learner
#[derive(Debug, Clone)]
pub struct QuantumContrastiveLearner {
    /// Configuration
    config: QuantumContrastiveConfig,
    /// Main encoder
    encoder: QuantumEncoder,
    /// Momentum encoder
    momentum_encoder: QuantumEncoder,
    /// Augmenter
    augmenter: QuantumAugmenter,
}

impl QuantumContrastiveLearner {
    /// Create new quantum contrastive learner
    pub fn new(config: QuantumContrastiveConfig) -> Self {
        let encoder = QuantumEncoder::new(config.num_qubits, config.encoder_depth);
        let momentum_encoder = encoder.clone();

        let augmenter = QuantumAugmenter::new(
            config.num_qubits,
            vec![
                QuantumAugmentation::RandomRotation,
                QuantumAugmentation::PhaseDamping,
            ],
            0.1,
        );

        Self {
            config,
            encoder,
            momentum_encoder,
            augmenter,
        }
    }

    /// Compute contrastive loss
    pub fn contrastive_loss(
        &self,
        states1: &[Array1<Complex64>],
        states2: &[Array1<Complex64>],
    ) -> QuantRS2Result<f64> {
        let n = states1.len();
        if n != states2.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Batch size mismatch".to_string(),
            ));
        }

        // Encode all states
        let mut z1 = Vec::with_capacity(n);
        let mut z2 = Vec::with_capacity(n);

        for i in 0..n {
            z1.push(self.encoder.encode(&states1[i])?);
            z2.push(self.momentum_encoder.encode(&states2[i])?);
        }

        // Compute NT-Xent loss
        let mut total_loss = 0.0;

        for i in 0..n {
            let mut numerator = 0.0;
            let mut denominator = 0.0;

            // Positive pair fidelity
            let pos_fidelity = self.quantum_fidelity(&z1[i], &z2[i]);
            numerator = (pos_fidelity / self.config.temperature).exp();

            // Negative pairs
            for j in 0..n {
                if i != j {
                    let neg_fidelity1 = self.quantum_fidelity(&z1[i], &z2[j]);
                    let neg_fidelity2 = self.quantum_fidelity(&z1[i], &z1[j]);

                    denominator += (neg_fidelity1 / self.config.temperature).exp();
                    denominator += (neg_fidelity2 / self.config.temperature).exp();
                }
            }

            denominator += numerator;

            total_loss -= (numerator / denominator).ln();
        }

        Ok(total_loss / n as f64)
    }

    /// Compute quantum fidelity between two states
    fn quantum_fidelity(&self, state1: &Array1<Complex64>, state2: &Array1<Complex64>) -> f64 {
        let mut fidelity = 0.0;
        for (a, b) in state1.iter().zip(state2.iter()) {
            fidelity += (a.conj() * b).norm_sqr();
        }
        fidelity
    }

    /// Update momentum encoder
    pub fn update_momentum_encoder(&mut self) {
        let main_params = self.encoder.params();
        let mut momentum_params = self.momentum_encoder.params().clone();

        // Exponential moving average
        momentum_params =
            &momentum_params * self.config.momentum + main_params * (1.0 - self.config.momentum);

        self.momentum_encoder.params = momentum_params;
    }

    /// Train on a batch
    pub fn train_step(
        &mut self,
        batch: &[Array1<Complex64>],
        learning_rate: f64,
    ) -> QuantRS2Result<f64> {
        // Generate augmented views
        let mut view1 = Vec::with_capacity(batch.len());
        let mut view2 = Vec::with_capacity(batch.len());

        for state in batch {
            view1.push(
                self.augmenter
                    .augment(state, QuantumAugmentation::RandomRotation)?,
            );
            view2.push(
                self.augmenter
                    .augment(state, QuantumAugmentation::PhaseDamping)?,
            );
        }

        // Compute loss
        let loss = self.contrastive_loss(&view1, &view2)?;

        // Compute gradients (simplified - in practice use parameter-shift rule)
        let epsilon = 1e-4;
        let mut gradients = Array2::zeros(self.encoder.params().dim());

        for i in 0..gradients.shape()[0] {
            for j in 0..gradients.shape()[1] {
                // Finite difference approximation
                let mut params_plus = self.encoder.params().clone();
                params_plus[[i, j]] += epsilon;
                self.encoder.params = params_plus;
                let loss_plus = self.contrastive_loss(&view1, &view2)?;

                let mut params_minus = self.encoder.params().clone();
                params_minus[[i, j]] -= 2.0 * epsilon;
                self.encoder.params = params_minus;
                let loss_minus = self.contrastive_loss(&view1, &view2)?;

                gradients[[i, j]] = (loss_plus - loss_minus) / (2.0 * epsilon);

                // Restore params
                let mut params_restore = self.encoder.params().clone();
                params_restore[[i, j]] += epsilon;
                self.encoder.params = params_restore;
            }
        }

        // Update encoder parameters
        self.encoder.update_params(&gradients, learning_rate);

        // Update momentum encoder
        self.update_momentum_encoder();

        Ok(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_augmenter() {
        let augmenter = QuantumAugmenter::new(2, vec![QuantumAugmentation::RandomRotation], 0.1);

        let state = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);

        let augmented = augmenter
            .augment(&state, QuantumAugmentation::RandomRotation)
            .expect("Failed to augment quantum state with random rotation");
        assert_eq!(augmented.len(), 4);
    }

    #[test]
    fn test_quantum_contrastive_learner() {
        let config = QuantumContrastiveConfig {
            num_qubits: 2,
            encoder_depth: 2,
            temperature: 0.5,
            momentum: 0.999,
            batch_size: 4,
            num_views: 2,
        };

        let learner = QuantumContrastiveLearner::new(config);

        let state = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);

        let encoded = learner
            .encoder
            .encode(&state)
            .expect("Failed to encode quantum state with encoder");
        assert_eq!(encoded.len(), 4);
    }
}
