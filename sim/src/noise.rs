//! Noise models for quantum simulation
//!
//! This module provides a framework for adding noise to quantum simulations,
//! including common noise models such as depolarizing, amplitude damping,
//! phase damping, and bit flip/phase flip channels.

#![allow(clippy::needless_range_loop)]

use scirs2_core::Complex64;
use std::fmt::Debug;

use quantrs2_core::error::QuantRS2Result;
use quantrs2_core::qubit::QubitId;

/// An enum that represents all possible noise channel types
#[derive(Debug, Clone)]
pub enum NoiseChannelType {
    BitFlip(BitFlipChannel),
    PhaseFlip(PhaseFlipChannel),
    Depolarizing(DepolarizingChannel),
    AmplitudeDamping(AmplitudeDampingChannel),
    PhaseDamping(PhaseDampingChannel),
}

impl NoiseChannelType {
    /// Get the name of the noise channel
    pub fn name(&self) -> &'static str {
        match self {
            Self::BitFlip(ch) => ch.name(),
            Self::PhaseFlip(ch) => ch.name(),
            Self::Depolarizing(ch) => ch.name(),
            Self::AmplitudeDamping(ch) => ch.name(),
            Self::PhaseDamping(ch) => ch.name(),
        }
    }

    /// Normalize a state vector to ensure it has unit norm
    pub fn normalize_state(state: &mut [Complex64]) {
        // Calculate current norm
        let mut norm_squared = 0.0;
        for amp in state.iter() {
            norm_squared += amp.norm_sqr();
        }

        // Apply normalization if needed
        if (norm_squared - 1.0).abs() > 1e-10 {
            let norm = norm_squared.sqrt();
            for amp in state.iter_mut() {
                *amp /= Complex64::new(norm, 0.0);
            }
        }
    }

    /// Get the qubits this channel affects
    pub fn qubits(&self) -> Vec<QubitId> {
        match self {
            Self::BitFlip(ch) => ch.qubits(),
            Self::PhaseFlip(ch) => ch.qubits(),
            Self::Depolarizing(ch) => ch.qubits(),
            Self::AmplitudeDamping(ch) => ch.qubits(),
            Self::PhaseDamping(ch) => ch.qubits(),
        }
    }

    /// Apply the noise channel to a state vector
    pub fn apply_to_statevector(&self, state: &mut [Complex64]) -> QuantRS2Result<()> {
        match self {
            Self::BitFlip(ch) => ch.apply_to_statevector(state),
            Self::PhaseFlip(ch) => ch.apply_to_statevector(state),
            Self::Depolarizing(ch) => ch.apply_to_statevector(state),
            Self::AmplitudeDamping(ch) => ch.apply_to_statevector(state),
            Self::PhaseDamping(ch) => ch.apply_to_statevector(state),
        }
    }

    /// Get the probability of the noise occurring
    pub fn probability(&self) -> f64 {
        match self {
            Self::BitFlip(ch) => ch.probability(),
            Self::PhaseFlip(ch) => ch.probability(),
            Self::Depolarizing(ch) => ch.probability(),
            Self::AmplitudeDamping(ch) => ch.probability(),
            Self::PhaseDamping(ch) => ch.probability(),
        }
    }
}

/// Trait for quantum noise channels
pub trait NoiseChannel: Debug + Clone {
    /// Return the name of the channel
    fn name(&self) -> &'static str;

    /// Return the qubits this channel affects
    fn qubits(&self) -> Vec<QubitId>;

    /// Apply the noise channel to a state vector
    fn apply_to_statevector(&self, state: &mut [Complex64]) -> QuantRS2Result<()>;

    /// Return the Kraus operators for this channel
    fn kraus_operators(&self) -> Vec<Vec<Complex64>>;

    /// Probability of the noise occurring
    fn probability(&self) -> f64;
}

/// Bitflip noise channel (X errors)
#[derive(Debug, Clone)]
pub struct BitFlipChannel {
    /// Target qubit
    pub target: QubitId,

    /// Probability of bit flip
    pub probability: f64,
}

impl NoiseChannel for BitFlipChannel {
    fn name(&self) -> &'static str {
        "BitFlip"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.target]
    }

    fn apply_to_statevector(&self, state: &mut [Complex64]) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        let dim = state.len();

        // Apply bit flip with probability p
        if fastrand::f64() < self.probability {
            // Create a copy of the state to read from
            let state_copy = state.to_vec();

            // Apply bit flip to each amplitude
            for i in 0..dim {
                let flipped_i = i ^ (1 << target_idx);
                state[i] = state_copy[flipped_i];
            }
        }

        Ok(())
    }

    fn kraus_operators(&self) -> Vec<Vec<Complex64>> {
        // Kraus operators for bit flip:
        // K0 = sqrt(1-p) * I, K1 = sqrt(p) * X
        let p = self.probability;
        let sqrt_1_minus_p = (1.0 - p).sqrt();
        let sqrt_p = p.sqrt();

        // I operator
        let k0 = vec![
            Complex64::new(sqrt_1_minus_p, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(sqrt_1_minus_p, 0.0),
        ];

        // X operator
        let k1 = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(sqrt_p, 0.0),
            Complex64::new(sqrt_p, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        vec![k0, k1]
    }

    fn probability(&self) -> f64 {
        self.probability
    }
}

/// Phase flip noise channel (Z errors)
#[derive(Debug, Clone)]
pub struct PhaseFlipChannel {
    /// Target qubit
    pub target: QubitId,

    /// Probability of phase flip
    pub probability: f64,
}

impl NoiseChannel for PhaseFlipChannel {
    fn name(&self) -> &'static str {
        "PhaseFlip"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.target]
    }

    fn apply_to_statevector(&self, state: &mut [Complex64]) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        let dim = state.len();

        // Apply phase flip with probability p
        if fastrand::f64() < self.probability {
            // Apply phase flip to each amplitude
            for i in 0..dim {
                if (i >> target_idx) & 1 == 1 {
                    // Apply phase flip to |1⟩ component
                    state[i] = -state[i];
                }
            }
        }

        Ok(())
    }

    fn kraus_operators(&self) -> Vec<Vec<Complex64>> {
        // Kraus operators for phase flip:
        // K0 = sqrt(1-p) * I, K1 = sqrt(p) * Z
        let p = self.probability;
        let sqrt_1_minus_p = (1.0 - p).sqrt();
        let sqrt_p = p.sqrt();

        // I operator
        let k0 = vec![
            Complex64::new(sqrt_1_minus_p, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(sqrt_1_minus_p, 0.0),
        ];

        // Z operator
        let k1 = vec![
            Complex64::new(sqrt_p, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-sqrt_p, 0.0),
        ];

        vec![k0, k1]
    }

    fn probability(&self) -> f64 {
        self.probability
    }
}

/// Depolarizing noise channel (equal probability of X, Y, or Z errors)
#[derive(Debug, Clone)]
pub struct DepolarizingChannel {
    /// Target qubit
    pub target: QubitId,

    /// Probability of depolarizing
    pub probability: f64,
}

impl NoiseChannel for DepolarizingChannel {
    fn name(&self) -> &'static str {
        "Depolarizing"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.target]
    }

    fn apply_to_statevector(&self, state: &mut [Complex64]) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        let dim = state.len();

        // Apply depolarizing noise with probability p
        if fastrand::f64() < self.probability {
            // Choose randomly between X, Y, and Z errors
            let error_type = fastrand::u32(..) % 3;

            // Create a copy of the state to read from
            let state_copy = state.to_vec();

            match error_type {
                0 => {
                    // X error (bit flip)
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << target_idx);
                        state[i] = state_copy[flipped_i];
                    }
                }
                1 => {
                    // Y error (bit and phase flip)
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << target_idx);
                        let phase = if (i >> target_idx) & 1 == 1 {
                            -1.0
                        } else {
                            1.0
                        };
                        state[i] = state_copy[flipped_i] * Complex64::new(0.0, phase);
                    }
                }
                2 => {
                    // Z error (phase flip)
                    for i in 0..dim {
                        if (i >> target_idx) & 1 == 1 {
                            state[i] = -state_copy[i];
                        } else {
                            state[i] = state_copy[i];
                        }
                    }
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    fn kraus_operators(&self) -> Vec<Vec<Complex64>> {
        // Kraus operators for depolarizing:
        // K0 = sqrt(1-3p/4) * I
        // K1 = sqrt(p/4) * X
        // K2 = sqrt(p/4) * Y
        // K3 = sqrt(p/4) * Z
        let p = self.probability;
        let sqrt_1_minus_3p_4 = (1.0 - 3.0 * p / 4.0).sqrt();
        let sqrt_p_4 = (p / 4.0).sqrt();

        // I operator
        let k0 = vec![
            Complex64::new(sqrt_1_minus_3p_4, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(sqrt_1_minus_3p_4, 0.0),
        ];

        // X operator
        let k1 = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(sqrt_p_4, 0.0),
            Complex64::new(sqrt_p_4, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        // Y operator
        let k2 = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -sqrt_p_4),
            Complex64::new(0.0, sqrt_p_4),
            Complex64::new(0.0, 0.0),
        ];

        // Z operator
        let k3 = vec![
            Complex64::new(sqrt_p_4, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-sqrt_p_4, 0.0),
        ];

        vec![k0, k1, k2, k3]
    }

    fn probability(&self) -> f64 {
        self.probability
    }
}

/// Amplitude damping noise channel (energy dissipation, T1 decay)
#[derive(Debug, Clone)]
pub struct AmplitudeDampingChannel {
    /// Target qubit
    pub target: QubitId,

    /// Damping probability
    pub gamma: f64,
}

impl NoiseChannel for AmplitudeDampingChannel {
    fn name(&self) -> &'static str {
        "AmplitudeDamping"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.target]
    }

    fn apply_to_statevector(&self, state: &mut [Complex64]) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        let dim = state.len();

        // Create a copy of the state to read from
        let state_copy = state.to_vec();

        // Apply amplitude damping to each basis state
        for i in 0..dim {
            if (i >> target_idx) & 1 == 1 {
                // This basis state has the target qubit in |1⟩
                let base_idx = i & !(1 << target_idx); // Flip the target bit to 0

                // Damping from |1⟩ to |0⟩ with probability gamma
                if fastrand::f64() < self.gamma {
                    // Collapse to |0⟩ state
                    state[base_idx] += state_copy[i];
                    state[i] = Complex64::new(0.0, 0.0);
                } else {
                    // Renormalize the |1⟩ state
                    state[i] = state_copy[i] * Complex64::new((1.0 - self.gamma).sqrt(), 0.0);
                }
            }
        }

        Ok(())
    }

    fn kraus_operators(&self) -> Vec<Vec<Complex64>> {
        // Kraus operators for amplitude damping:
        // K0 = [[1, 0], [0, sqrt(1-gamma)]]
        // K1 = [[0, sqrt(gamma)], [0, 0]]
        let gamma = self.gamma;
        let sqrt_1_minus_gamma = (1.0 - gamma).sqrt();
        let sqrt_gamma = gamma.sqrt();

        // K0 operator
        let k0 = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(sqrt_1_minus_gamma, 0.0),
        ];

        // K1 operator
        let k1 = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(sqrt_gamma, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        vec![k0, k1]
    }

    fn probability(&self) -> f64 {
        self.gamma
    }
}

/// Phase damping noise channel (pure dephasing, T2 decay)
#[derive(Debug, Clone)]
pub struct PhaseDampingChannel {
    /// Target qubit
    pub target: QubitId,

    /// Damping probability
    pub lambda: f64,
}

impl NoiseChannel for PhaseDampingChannel {
    fn name(&self) -> &'static str {
        "PhaseDamping"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.target]
    }

    fn apply_to_statevector(&self, state: &mut [Complex64]) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        let dim = state.len();

        // Apply phase damping to each basis state
        for i in 0..dim {
            if (i >> target_idx) & 1 == 1 {
                // This basis state has the target qubit in |1⟩
                // Apply phase damping
                if fastrand::f64() < self.lambda {
                    // Random phase
                    let phase = 2.0 * std::f64::consts::PI * fastrand::f64();
                    state[i] *= Complex64::new(phase.cos(), phase.sin());
                }
            }
        }

        Ok(())
    }

    fn kraus_operators(&self) -> Vec<Vec<Complex64>> {
        // Kraus operators for phase damping:
        // K0 = [[1, 0], [0, sqrt(1-lambda)]]
        // K1 = [[0, 0], [0, sqrt(lambda)]]
        let lambda = self.lambda;
        let sqrt_1_minus_lambda = (1.0 - lambda).sqrt();
        let sqrt_lambda = lambda.sqrt();

        // K0 operator
        let k0 = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(sqrt_1_minus_lambda, 0.0),
        ];

        // K1 operator
        let k1 = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(sqrt_lambda, 0.0),
        ];

        vec![k0, k1]
    }

    fn probability(&self) -> f64 {
        self.lambda
    }
}

/// Noise model that combines multiple noise channels
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// List of noise channels
    pub channels: Vec<NoiseChannelType>,

    /// Whether the noise is applied after each gate
    pub per_gate: bool,
}

impl NoiseModel {
    /// Create a new empty noise model
    pub fn new(per_gate: bool) -> Self {
        Self {
            channels: Vec::new(),
            per_gate,
        }
    }

    /// Add a bit flip noise channel to the model
    pub fn add_bit_flip(&mut self, channel: BitFlipChannel) -> &mut Self {
        self.channels.push(NoiseChannelType::BitFlip(channel));
        self
    }

    /// Add a phase flip noise channel to the model
    pub fn add_phase_flip(&mut self, channel: PhaseFlipChannel) -> &mut Self {
        self.channels.push(NoiseChannelType::PhaseFlip(channel));
        self
    }

    /// Add a depolarizing noise channel to the model
    pub fn add_depolarizing(&mut self, channel: DepolarizingChannel) -> &mut Self {
        self.channels.push(NoiseChannelType::Depolarizing(channel));
        self
    }

    /// Add an amplitude damping noise channel to the model
    pub fn add_amplitude_damping(&mut self, channel: AmplitudeDampingChannel) -> &mut Self {
        self.channels
            .push(NoiseChannelType::AmplitudeDamping(channel));
        self
    }

    /// Add a phase damping noise channel to the model
    pub fn add_phase_damping(&mut self, channel: PhaseDampingChannel) -> &mut Self {
        self.channels.push(NoiseChannelType::PhaseDamping(channel));
        self
    }

    /// Apply all noise channels to a state vector
    pub fn apply_to_statevector(&self, state: &mut [Complex64]) -> QuantRS2Result<()> {
        for channel in &self.channels {
            channel.apply_to_statevector(state)?;
        }

        // Normalize the state vector after applying all noise channels
        NoiseChannelType::normalize_state(state);

        Ok(())
    }

    /// Get the total number of channels
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }
}

impl Default for NoiseModel {
    fn default() -> Self {
        Self::new(true)
    }
}

/// Builder for common noise models
pub struct NoiseModelBuilder {
    model: NoiseModel,
}

impl NoiseModelBuilder {
    /// Create a new noise model builder
    pub fn new(per_gate: bool) -> Self {
        Self {
            model: NoiseModel::new(per_gate),
        }
    }

    /// Add depolarizing noise to all qubits
    pub fn with_depolarizing_noise(mut self, qubits: &[QubitId], probability: f64) -> Self {
        for &qubit in qubits {
            self.model.add_depolarizing(DepolarizingChannel {
                target: qubit,
                probability,
            });
        }
        self
    }

    /// Add bit flip noise to all qubits
    pub fn with_bit_flip_noise(mut self, qubits: &[QubitId], probability: f64) -> Self {
        for &qubit in qubits {
            self.model.add_bit_flip(BitFlipChannel {
                target: qubit,
                probability,
            });
        }
        self
    }

    /// Add phase flip noise to all qubits
    pub fn with_phase_flip_noise(mut self, qubits: &[QubitId], probability: f64) -> Self {
        for &qubit in qubits {
            self.model.add_phase_flip(PhaseFlipChannel {
                target: qubit,
                probability,
            });
        }
        self
    }

    /// Add amplitude damping to all qubits
    pub fn with_amplitude_damping(mut self, qubits: &[QubitId], gamma: f64) -> Self {
        for &qubit in qubits {
            self.model.add_amplitude_damping(AmplitudeDampingChannel {
                target: qubit,
                gamma,
            });
        }
        self
    }

    /// Add phase damping to all qubits
    pub fn with_phase_damping(mut self, qubits: &[QubitId], lambda: f64) -> Self {
        for &qubit in qubits {
            self.model.add_phase_damping(PhaseDampingChannel {
                target: qubit,
                lambda,
            });
        }
        self
    }

    /// Build the noise model
    pub fn build(self) -> NoiseModel {
        self.model
    }
}
