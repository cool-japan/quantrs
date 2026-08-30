//! Advanced noise models beyond simple single-qubit depolarising channels.
//!
//! Provides two-qubit correlated noise, crosstalk, leakage to higher energy
//! levels, and time-dependent amplitude / phase damping noise channels for
//! realistic device simulation.

#![allow(clippy::needless_range_loop)]

use scirs2_core::Complex64;
use std::f64::consts::PI;
use std::time::Duration;

use quantrs2_core::error::QuantRS2Result;
use quantrs2_core::qubit::QubitId;

use crate::noise::{NoiseChannel, NoiseChannelType, NoiseModel};

/// Multiply two single-qubit (2x2) operators stored as flattened row-major
/// `Complex64` slices and return the product `a * b` (also flattened 2x2).
///
/// Index layout: element `[i][j]` lives at `2 * i + j`.
fn matmul_2x2(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); 4];
    for i in 0..2 {
        for j in 0..2 {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..2 {
                acc += a[2 * i + k] * b[2 * k + j];
            }
            out[2 * i + j] = acc;
        }
    }
    out
}

/// Tensor (Kronecker) product of two single-qubit (2x2) operators stored as
/// flattened row-major `Complex64` slices, returning the resulting 4x4 operator
/// as a flattened row-major slice (16 elements).
///
/// For `A ⊗ B`, the element at `[2*r1 + r2][2*c1 + c2]` equals
/// `A[r1][c1] * B[r2][c2]`. In the flattened 4x4 layout the destination index is
/// `4 * (2*r1 + r2) + (2*c1 + c2)`.
fn tensor_2x2(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); 16];
    for r1 in 0..2 {
        for c1 in 0..2 {
            let a_val = a[2 * r1 + c1];
            for r2 in 0..2 {
                for c2 in 0..2 {
                    let row = 2 * r1 + r2;
                    let col = 2 * c1 + c2;
                    out[4 * row + col] = a_val * b[2 * r2 + c2];
                }
            }
        }
    }
    out
}

/// The four single-qubit Pauli matrices `{I, X, Y, Z}` as flattened row-major
/// 2x2 `Complex64` slices.
fn single_qubit_paulis() -> [Vec<Complex64>; 4] {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let i_unit = Complex64::new(0.0, 1.0);
    [
        // I
        vec![one, zero, zero, one],
        // X
        vec![zero, one, one, zero],
        // Y
        vec![zero, -i_unit, i_unit, zero],
        // Z
        vec![one, zero, zero, -one],
    ]
}

/// Two-qubit depolarizing noise channel
#[derive(Debug, Clone)]
pub struct TwoQubitDepolarizingChannel {
    /// First qubit
    pub qubit1: QubitId,

    /// Second qubit
    pub qubit2: QubitId,

    /// Probability of error
    pub probability: f64,
}

impl NoiseChannel for TwoQubitDepolarizingChannel {
    fn name(&self) -> &'static str {
        "TwoQubitDepolarizing"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.qubit1, self.qubit2]
    }

    fn apply_to_statevector(&self, state: &mut [Complex64]) -> QuantRS2Result<()> {
        let q1_idx = self.qubit1.id() as usize;
        #[allow(clippy::needless_range_loop)]
        let q2_idx = self.qubit2.id() as usize;
        let dim = state.len();

        // Apply two-qubit depolarizing noise with probability p
        if fastrand::f64() < self.probability {
            // Choose randomly between 15 possible Pauli errors (excluding I⊗I)
            let error_type = fastrand::u32(..) % 15;

            // Create a copy of the state to read from
            let state_copy = state.to_vec();

            match error_type {
                0 => {
                    // X⊗I
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << q1_idx);
                        state[i] = state_copy[flipped_i];
                    }
                }
                1 => {
                    // I⊗X
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << q2_idx);
                        state[i] = state_copy[flipped_i];
                    }
                }
                2 => {
                    // X⊗X
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << q1_idx) ^ (1 << q2_idx);
                        state[i] = state_copy[flipped_i];
                    }
                }
                3 => {
                    // Y⊗I
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << q1_idx);
                        let phase = if (i >> q1_idx) & 1 == 1 { 1.0 } else { -1.0 };
                        state[i] = state_copy[flipped_i] * Complex64::new(0.0, phase);
                    }
                }
                4 => {
                    // I⊗Y
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << q2_idx);
                        let phase = if (i >> q2_idx) & 1 == 1 { 1.0 } else { -1.0 };
                        state[i] = state_copy[flipped_i] * Complex64::new(0.0, phase);
                    }
                }
                5 => {
                    // Y⊗Y
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << q1_idx) ^ (1 << q2_idx);
                        let phase1 = if (i >> q1_idx) & 1 == 1 { 1.0 } else { -1.0 };
                        let phase2 = if (i >> q2_idx) & 1 == 1 { 1.0 } else { -1.0 };
                        state[i] = state_copy[flipped_i] * Complex64::new(0.0, phase1 * phase2);
                    }
                }
                6 => {
                    // Z⊗I
                    for i in 0..dim {
                        if (i >> q1_idx) & 1 == 1 {
                            state[i] = -state_copy[i];
                        }
                    }
                }
                7 => {
                    // I⊗Z
                    for i in 0..dim {
                        if (i >> q2_idx) & 1 == 1 {
                            state[i] = -state_copy[i];
                        }
                    }
                }
                8 => {
                    // Z⊗Z
                    for i in 0..dim {
                        let parity = ((i >> q1_idx) & 1) ^ ((i >> q2_idx) & 1);
                        if parity == 1 {
                            state[i] = -state_copy[i];
                        }
                    }
                }
                9 => {
                    // X⊗Y
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << q1_idx) ^ (1 << q2_idx);
                        let phase = if (i >> q2_idx) & 1 == 1 { 1.0 } else { -1.0 };
                        state[i] = state_copy[flipped_i] * Complex64::new(0.0, phase);
                    }
                }
                10 => {
                    // X⊗Z
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << q1_idx);
                        if (flipped_i >> q2_idx) & 1 == 1 {
                            state[i] = -state_copy[flipped_i];
                        } else {
                            state[i] = state_copy[flipped_i];
                        }
                    }
                }
                11 => {
                    // Y⊗X
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << q1_idx) ^ (1 << q2_idx);
                        let phase = if (i >> q1_idx) & 1 == 1 { 1.0 } else { -1.0 };
                        state[i] = state_copy[flipped_i] * Complex64::new(0.0, phase);
                    }
                }
                12 => {
                    // Y⊗Z
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << q1_idx);
                        let phase = if ((i >> q1_idx) & 1 == 1) ^ ((i >> q2_idx) & 1 == 1) {
                            Complex64::new(0.0, -1.0)
                        } else {
                            Complex64::new(0.0, 1.0)
                        };
                        state[i] = state_copy[flipped_i] * phase;
                    }
                }
                13 => {
                    // Z⊗X
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << q2_idx);
                        if (i >> q1_idx) & 1 == 1 {
                            state[i] = -state_copy[flipped_i];
                        } else {
                            state[i] = state_copy[flipped_i];
                        }
                    }
                }
                14 => {
                    // Z⊗Y
                    for i in 0..dim {
                        let flipped_i = i ^ (1 << q2_idx);
                        let phase = if ((i >> q1_idx) & 1 == 1) ^ ((i >> q2_idx) & 1 == 1) {
                            Complex64::new(0.0, -1.0)
                        } else {
                            Complex64::new(0.0, 1.0)
                        };
                        state[i] = state_copy[flipped_i] * phase;
                    }
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    fn kraus_operators(&self) -> Vec<Vec<Complex64>> {
        // Two-qubit depolarizing channel:
        //   rho -> (1 - p) rho + (p / 15) * sum_{(a,b) != (I,I)} (P_a x P_b) rho (P_a x P_b)^dagger
        // This yields 16 Kraus operators, each a flattened 4x4 matrix (16 elems):
        //   K_0    = sqrt(1 - p)  * (I x I)
        //   K_1..15 = sqrt(p / 15) * (P_a x P_b)   for the 15 non-identity Pauli pairs.
        // Trace preservation: sum_k K_k^dagger K_k
        //   = (1 - p) I_4 + (p / 15) * 15 * I_4 = I_4   (each Pauli is unitary).
        let p = self.probability;
        let sqrt_1_minus_p = Complex64::new((1.0 - p).sqrt(), 0.0);
        let sqrt_p_15 = Complex64::new((p / 15.0).sqrt(), 0.0);

        let paulis = single_qubit_paulis();

        let mut operators = Vec::with_capacity(16);
        for (a_idx, pauli_a) in paulis.iter().enumerate() {
            for (b_idx, pauli_b) in paulis.iter().enumerate() {
                let mut op = tensor_2x2(pauli_a, pauli_b);
                // (a_idx, b_idx) == (0, 0) is the identity pair I x I.
                let scale = if a_idx == 0 && b_idx == 0 {
                    sqrt_1_minus_p
                } else {
                    sqrt_p_15
                };
                for value in &mut op {
                    *value *= scale;
                }
                operators.push(op);
            }
        }

        operators
    }

    fn probability(&self) -> f64 {
        self.probability
    }
}

/// Thermal relaxation noise channel (combination of T1 and T2 effects)
#[derive(Debug, Clone)]
pub struct ThermalRelaxationChannel {
    /// Target qubit
    pub target: QubitId,

    /// T1 relaxation time (seconds)
    pub t1: f64,

    /// T2 pure dephasing time (seconds)
    pub t2: f64,

    /// Gate time (seconds)
    pub gate_time: f64,

    /// Excited state population at thermal equilibrium (0.0 to 1.0)
    pub excited_state_population: f64,
}

impl NoiseChannel for ThermalRelaxationChannel {
    fn name(&self) -> &'static str {
        "ThermalRelaxation"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.target]
    }

    fn apply_to_statevector(&self, state: &mut [Complex64]) -> QuantRS2Result<()> {
        let target_idx = self.target.id() as usize;
        let dim = state.len();

        // Calculate relaxation and dephasing probabilities
        let p_reset = 1.0 - (-self.gate_time / self.t1).exp();
        let p_phase = 0.5 * (1.0 - (-self.gate_time / self.t2).exp());

        // Create a copy of the state for reading
        let state_copy = state.to_vec();

        // Apply thermal relaxation
        // First apply amplitude damping (relaxation)
        for i in 0..dim {
            if (i >> target_idx) & 1 == 1 {
                // This basis state has the target qubit in |1⟩
                let base_idx = i & !(1 << target_idx); // Flip the target bit to 0

                // Apply relaxation with probability p_reset
                if fastrand::f64() < p_reset {
                    // With probability (1-p_eq), collapse to |0⟩ state
                    // With probability p_eq, collapse to |1⟩ state (thermal equilibrium)
                    if fastrand::f64() < self.excited_state_population {
                        // Stay in |1⟩ due to thermal excitation
                        state[i] = state_copy[i];
                    } else {
                        // Collapse to |0⟩
                        state[base_idx] += state_copy[i];
                        state[i] = Complex64::new(0.0, 0.0);
                    }
                } else {
                    // No relaxation occurs, but apply sqrt(1-p) factor
                    state[i] = state_copy[i] * Complex64::new((1.0 - p_reset).sqrt(), 0.0);
                }
            }
        }

        // Then apply phase damping (dephasing on top of amplitude damping)
        for i in 0..dim {
            if (i >> target_idx) & 1 == 1 {
                // Apply additional pure dephasing
                if fastrand::f64() < p_phase {
                    // Random phase
                    state[i] *= Complex64::new(-1.0, 0.0); // Apply phase flip
                }
            }
        }

        // Normalize the state
        NoiseChannelType::normalize_state(state);

        Ok(())
    }

    fn kraus_operators(&self) -> Vec<Vec<Complex64>> {
        // Thermal relaxation = generalized amplitude damping (T1) composed with
        // pure phase damping (the residual T2 dephasing). Both are CPTP, so their
        // composition is CPTP and trace-preserving.
        //
        // Generalized amplitude damping with gamma = p_reset and excited-state
        // population p_e (all 2x2, flattened row-major as [a00, a01, a10, a11]):
        //   A0 = sqrt(p_e)     * [[1, 0], [0, sqrt(1 - gamma)]]
        //   A1 = sqrt(p_e)     * [[0, sqrt(gamma)], [0, 0]]
        //   A2 = sqrt(1 - p_e) * [[sqrt(1 - gamma), 0], [0, 1]]
        //   A3 = sqrt(1 - p_e) * [[0, 0], [sqrt(gamma), 0]]
        // Phase damping with gamma_phi = 2 * p_phase (clamped to [0, 1]):
        //   P0 = [[1, 0], [0, sqrt(1 - gamma_phi)]]
        //   P1 = [[0, 0], [0, sqrt(gamma_phi)]]
        // Final Kraus set = { P_i * A_j } (8 operators).
        // Trace check: sum_{i,j} (P_i A_j)^dagger (P_i A_j)
        //   = sum_j A_j^dagger (sum_i P_i^dagger P_i) A_j
        //   = sum_j A_j^dagger A_j = I_2.
        let p_reset = 1.0 - (-self.gate_time / self.t1).exp();
        let p_phase = 0.5 * (1.0 - (-self.gate_time / self.t2).exp());

        let gamma = p_reset.clamp(0.0, 1.0);
        let gamma_phi = (2.0 * p_phase).clamp(0.0, 1.0);
        let p_e = self.excited_state_population.clamp(0.0, 1.0);

        let zero = Complex64::new(0.0, 0.0);
        let sqrt_p_e = Complex64::new(p_e.sqrt(), 0.0);
        let sqrt_1_minus_p_e = Complex64::new((1.0 - p_e).sqrt(), 0.0);
        let sqrt_gamma = Complex64::new(gamma.sqrt(), 0.0);
        let sqrt_1_minus_gamma = Complex64::new((1.0 - gamma).sqrt(), 0.0);
        let sqrt_gamma_phi = Complex64::new(gamma_phi.sqrt(), 0.0);
        let sqrt_1_minus_gamma_phi = Complex64::new((1.0 - gamma_phi).sqrt(), 0.0);
        let one = Complex64::new(1.0, 0.0);

        // Generalized amplitude damping operators.
        let a0 = vec![sqrt_p_e, zero, zero, sqrt_p_e * sqrt_1_minus_gamma];
        let a1 = vec![zero, sqrt_p_e * sqrt_gamma, zero, zero];
        let a2 = vec![
            sqrt_1_minus_p_e * sqrt_1_minus_gamma,
            zero,
            zero,
            sqrt_1_minus_p_e,
        ];
        let a3 = vec![zero, zero, sqrt_1_minus_p_e * sqrt_gamma, zero];
        let amplitude_ops = [a0, a1, a2, a3];

        // Pure phase-damping operators.
        let p0 = vec![one, zero, zero, sqrt_1_minus_gamma_phi];
        let p1 = vec![zero, zero, zero, sqrt_gamma_phi];
        let phase_ops = [p0, p1];

        let mut operators = Vec::with_capacity(amplitude_ops.len() * phase_ops.len());
        for phase_op in &phase_ops {
            for amplitude_op in &amplitude_ops {
                operators.push(matmul_2x2(phase_op, amplitude_op));
            }
        }

        operators
    }

    fn probability(&self) -> f64 {
        // Return the combined probability of an error occurring
        let p_reset = 1.0 - (-self.gate_time / self.t1).exp();
        let p_phase = 0.5 * (1.0 - (-self.gate_time / self.t2).exp());
        p_reset + p_phase - p_reset * p_phase // Combined probability
    }
}

/// Crosstalk noise channel for adjacent qubits
#[derive(Debug, Clone)]
pub struct CrosstalkChannel {
    /// Primary qubit
    pub primary: QubitId,

    /// Neighbor qubit
    pub neighbor: QubitId,

    /// Crosstalk strength (0.0 to 1.0)
    pub strength: f64,
}

impl NoiseChannel for CrosstalkChannel {
    fn name(&self) -> &'static str {
        "Crosstalk"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.primary, self.neighbor]
    }

    fn apply_to_statevector(&self, state: &mut [Complex64]) -> QuantRS2Result<()> {
        let primary_idx = self.primary.id() as usize;
        let neighbor_idx = self.neighbor.id() as usize;
        let dim = state.len();

        // Apply crosstalk with probability based on strength
        if fastrand::f64() < self.strength {
            // Create a copy of the state for reading
            let state_copy = state.to_vec();

            // Randomly select an effect (simplified model):
            // 1. ZZ interaction
            // 2. Neighbor rotation
            let effect = fastrand::u32(..) % 2;

            match effect {
                0 => {
                    // ZZ interaction
                    for i in 0..dim {
                        let parity = ((i >> primary_idx) & 1) ^ ((i >> neighbor_idx) & 1);
                        if parity == 1 {
                            // Apply phase shift if qubits have different parity
                            let phase = fastrand::f64() * PI;
                            state[i] *= Complex64::new(phase.cos(), phase.sin());
                        }
                    }
                }
                1 => {
                    // Small rotation on neighbor when primary qubit is |1⟩
                    for i in 0..dim {
                        if (i >> primary_idx) & 1 == 1 {
                            // Primary qubit is |1⟩, apply partial X rotation to neighbor
                            let neighbor_bit = (i >> neighbor_idx) & 1;
                            let flipped_i = i ^ (1 << neighbor_idx);

                            // Small, random amplitude swap
                            let theta: f64 = fastrand::f64() * 0.2; // Small angle
                            let cos_theta = theta.cos();
                            let sin_theta = theta.sin();

                            let amp_original = state_copy[i];
                            let amp_flipped = state_copy[flipped_i];

                            if neighbor_bit == 0 {
                                state[i] = amp_original * Complex64::new(cos_theta, 0.0)
                                    + amp_flipped * Complex64::new(sin_theta, 0.0);
                                state[flipped_i] = amp_original * Complex64::new(-sin_theta, 0.0)
                                    + amp_flipped * Complex64::new(cos_theta, 0.0);
                            } else {
                                state[i] = amp_original * Complex64::new(cos_theta, 0.0)
                                    - amp_flipped * Complex64::new(sin_theta, 0.0);
                                state[flipped_i] = amp_original * Complex64::new(sin_theta, 0.0)
                                    + amp_flipped * Complex64::new(cos_theta, 0.0);
                            }
                        }
                    }
                }
                _ => unreachable!(),
            }
        }

        // Normalize the state
        NoiseChannelType::normalize_state(state);

        Ok(())
    }

    fn kraus_operators(&self) -> Vec<Vec<Complex64>> {
        // The dominant, always-on crosstalk between two coupled qubits is the
        // coherent ZZ interaction U_ZZ(phi) = exp(-i * phi/2 * (Z x Z)), with the
        // coupling angle proportional to the crosstalk strength (phi = strength * PI).
        // Because this is a coherent (unitary) process, the channel is described by
        // a SINGLE Kraus operator K = U_ZZ, which is trivially trace-preserving
        // (K^dagger K = I_4 since U_ZZ is unitary).
        //
        // Z x Z is diagonal in the computational basis (|00>, |01>, |10>, |11>)
        // with eigenvalues (+1, -1, -1, +1), so
        //   U_ZZ = diag(e^{-i phi/2}, e^{+i phi/2}, e^{+i phi/2}, e^{-i phi/2}).
        // The result is returned as a flattened row-major 4x4 matrix (16 elems).
        let phi = self.strength * PI;
        let half = phi / 2.0;
        let zero = Complex64::new(0.0, 0.0);
        let phase_neg = Complex64::new(half.cos(), -half.sin()); // e^{-i phi/2}
        let phase_pos = Complex64::new(half.cos(), half.sin()); // e^{+i phi/2}

        // Diagonal 4x4 with the ZZ phases on the diagonal.
        let mut op = vec![zero; 16];
        op[0] = phase_neg; // |00>
        op[5] = phase_pos; // |01>
        op[10] = phase_pos; // |10>
        op[15] = phase_neg; // |11>

        vec![op]
    }

    fn probability(&self) -> f64 {
        self.strength
    }
}

/// Extension to `NoiseChannelType` to include advanced noise channels
#[derive(Debug, Clone)]
pub enum AdvancedNoiseChannelType {
    /// Base noise channel types
    Base(NoiseChannelType),

    /// Two-qubit depolarizing channel
    TwoQubitDepolarizing(TwoQubitDepolarizingChannel),

    /// Thermal relaxation channel
    ThermalRelaxation(ThermalRelaxationChannel),

    /// Crosstalk channel
    Crosstalk(CrosstalkChannel),
}

impl AdvancedNoiseChannelType {
    /// Get the name of the noise channel
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Base(ch) => ch.name(),
            Self::TwoQubitDepolarizing(ch) => ch.name(),
            Self::ThermalRelaxation(ch) => ch.name(),
            Self::Crosstalk(ch) => ch.name(),
        }
    }

    /// Get the qubits this channel affects
    #[must_use]
    pub fn qubits(&self) -> Vec<QubitId> {
        match self {
            Self::Base(ch) => ch.qubits(),
            Self::TwoQubitDepolarizing(ch) => ch.qubits(),
            Self::ThermalRelaxation(ch) => ch.qubits(),
            Self::Crosstalk(ch) => ch.qubits(),
        }
    }

    /// Apply the noise channel to a state vector
    pub fn apply_to_statevector(&self, state: &mut [Complex64]) -> QuantRS2Result<()> {
        match self {
            Self::Base(ch) => ch.apply_to_statevector(state),
            Self::TwoQubitDepolarizing(ch) => ch.apply_to_statevector(state),
            Self::ThermalRelaxation(ch) => ch.apply_to_statevector(state),
            Self::Crosstalk(ch) => ch.apply_to_statevector(state),
        }
    }

    /// Get the probability of the noise occurring
    #[must_use]
    pub fn probability(&self) -> f64 {
        match self {
            Self::Base(ch) => ch.probability(),
            Self::TwoQubitDepolarizing(ch) => ch.probability(),
            Self::ThermalRelaxation(ch) => ch.probability(),
            Self::Crosstalk(ch) => ch.probability(),
        }
    }
}

/// Advanced noise model that supports the new noise channel types
#[derive(Debug, Clone)]
pub struct AdvancedNoiseModel {
    /// List of noise channels
    pub channels: Vec<AdvancedNoiseChannelType>,

    /// Whether the noise is applied after each gate
    pub per_gate: bool,
}

impl AdvancedNoiseModel {
    /// Create a new empty noise model
    #[must_use]
    pub const fn new(per_gate: bool) -> Self {
        Self {
            channels: Vec::new(),
            per_gate,
        }
    }

    /// Add a basic noise channel to the model
    pub fn add_base_channel(&mut self, channel: NoiseChannelType) -> &mut Self {
        self.channels.push(AdvancedNoiseChannelType::Base(channel));
        self
    }

    /// Add a two-qubit depolarizing noise channel to the model
    pub fn add_two_qubit_depolarizing(
        &mut self,
        channel: TwoQubitDepolarizingChannel,
    ) -> &mut Self {
        self.channels
            .push(AdvancedNoiseChannelType::TwoQubitDepolarizing(channel));
        self
    }

    /// Add a thermal relaxation noise channel to the model
    pub fn add_thermal_relaxation(&mut self, channel: ThermalRelaxationChannel) -> &mut Self {
        self.channels
            .push(AdvancedNoiseChannelType::ThermalRelaxation(channel));
        self
    }

    /// Add a crosstalk noise channel to the model
    pub fn add_crosstalk(&mut self, channel: CrosstalkChannel) -> &mut Self {
        self.channels
            .push(AdvancedNoiseChannelType::Crosstalk(channel));
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
    #[must_use]
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Convert to basic noise model (for backward compatibility)
    #[must_use]
    pub fn to_basic_model(&self) -> NoiseModel {
        let mut model = NoiseModel::new(self.per_gate);

        for channel in &self.channels {
            if let AdvancedNoiseChannelType::Base(ch) = channel {
                model.channels.push(ch.clone());
            }
        }

        model
    }
}

impl Default for AdvancedNoiseModel {
    fn default() -> Self {
        Self::new(true)
    }
}

/// Builder for realistic device noise models
pub struct RealisticNoiseModelBuilder {
    model: AdvancedNoiseModel,
}

impl RealisticNoiseModelBuilder {
    /// Create a new noise model builder
    #[must_use]
    pub const fn new(per_gate: bool) -> Self {
        Self {
            model: AdvancedNoiseModel::new(per_gate),
        }
    }

    /// Add realistic IBM Quantum device noise parameters
    #[must_use]
    pub fn with_ibm_device_noise(mut self, qubits: &[QubitId], device_name: &str) -> Self {
        match device_name {
            "ibmq_lima" | "ibmq_belem" | "ibmq_quito" => {
                // 5-qubit IBM Quantum Falcon processors
                // Parameters are approximate and based on typical values

                // Relaxation and dephasing times
                let t1_values = [115e-6, 100e-6, 120e-6, 105e-6, 110e-6]; // ~100 microseconds
                let t2_values = [95e-6, 80e-6, 100e-6, 90e-6, 85e-6]; // ~90 microseconds

                // Single-qubit gates
                let gate_time_1q = 35e-9; // 35 nanoseconds
                let gate_error_1q = 0.001; // 0.1% error rate

                // Two-qubit gates (CNOT)
                // let _gate_time_2q = 300e-9; // 300 nanoseconds
                let gate_error_2q = 0.01; // 1% error rate

                // Readout errors
                let readout_error = 0.025; // 2.5% error

                // Add individual qubit noise
                for (i, &qubit) in qubits.iter().enumerate().take(5) {
                    let t1 = t1_values[i % 5];
                    let t2 = t2_values[i % 5];

                    // Add thermal relaxation
                    self.model.add_thermal_relaxation(ThermalRelaxationChannel {
                        target: qubit,
                        t1,
                        t2,
                        gate_time: gate_time_1q,
                        excited_state_population: 0.01, // ~1% thermal excitation
                    });

                    // Add depolarizing noise for single-qubit gates
                    self.model.add_base_channel(NoiseChannelType::Depolarizing(
                        crate::noise::DepolarizingChannel {
                            target: qubit,
                            probability: gate_error_1q,
                        },
                    ));

                    // Add readout error as a bit flip channel
                    self.model.add_base_channel(NoiseChannelType::BitFlip(
                        crate::noise::BitFlipChannel {
                            target: qubit,
                            probability: readout_error,
                        },
                    ));
                }

                // Add two-qubit gate noise (for nearest-neighbor connectivity)
                for i in 0..qubits.len().saturating_sub(1) {
                    let q1 = qubits[i];
                    let q2 = qubits[i + 1];

                    // Add two-qubit depolarizing noise
                    self.model
                        .add_two_qubit_depolarizing(TwoQubitDepolarizingChannel {
                            qubit1: q1,
                            qubit2: q2,
                            probability: gate_error_2q,
                        });

                    // Add crosstalk between adjacent qubits
                    self.model.add_crosstalk(CrosstalkChannel {
                        primary: q1,
                        neighbor: q2,
                        strength: 0.003, // 0.3% crosstalk
                    });
                }
            }
            "ibmq_bogota" | "ibmq_santiago" | "ibmq_casablanca" => {
                // 5-qubit IBM Quantum Falcon processors (newer)
                // Parameters are approximate and based on typical values

                // Relaxation and dephasing times
                let t1_values = [140e-6, 130e-6, 145e-6, 135e-6, 150e-6]; // ~140 microseconds
                let t2_values = [120e-6, 110e-6, 125e-6, 115e-6, 130e-6]; // ~120 microseconds

                // Single-qubit gates
                let gate_time_1q = 30e-9; // 30 nanoseconds
                let gate_error_1q = 0.0005; // 0.05% error rate

                // Two-qubit gates (CNOT)
                // let _gate_time_2q = 250e-9; // 250 nanoseconds
                let gate_error_2q = 0.008; // 0.8% error rate

                // Readout errors
                let readout_error = 0.02; // 2% error

                // Add individual qubit noise
                for (i, &qubit) in qubits.iter().enumerate().take(5) {
                    let t1 = t1_values[i % 5];
                    let t2 = t2_values[i % 5];

                    // Add thermal relaxation
                    self.model.add_thermal_relaxation(ThermalRelaxationChannel {
                        target: qubit,
                        t1,
                        t2,
                        gate_time: gate_time_1q,
                        excited_state_population: 0.008, // ~0.8% thermal excitation
                    });

                    // Add depolarizing noise for single-qubit gates
                    self.model.add_base_channel(NoiseChannelType::Depolarizing(
                        crate::noise::DepolarizingChannel {
                            target: qubit,
                            probability: gate_error_1q,
                        },
                    ));

                    // Add readout error as a bit flip channel
                    self.model.add_base_channel(NoiseChannelType::BitFlip(
                        crate::noise::BitFlipChannel {
                            target: qubit,
                            probability: readout_error,
                        },
                    ));
                }

                // Add two-qubit gate noise (for nearest-neighbor connectivity)
                for i in 0..qubits.len().saturating_sub(1) {
                    let q1 = qubits[i];
                    let q2 = qubits[i + 1];

                    // Add two-qubit depolarizing noise
                    self.model
                        .add_two_qubit_depolarizing(TwoQubitDepolarizingChannel {
                            qubit1: q1,
                            qubit2: q2,
                            probability: gate_error_2q,
                        });

                    // Add crosstalk between adjacent qubits
                    self.model.add_crosstalk(CrosstalkChannel {
                        primary: q1,
                        neighbor: q2,
                        strength: 0.002, // 0.2% crosstalk
                    });
                }
            }
            "ibm_cairo" | "ibm_hanoi" | "ibm_auckland" => {
                // 27-qubit IBM Quantum Falcon processors
                // Parameters are approximate and based on typical values

                // Relaxation and dephasing times (average values)
                let t1 = 130e-6; // 130 microseconds
                let t2 = 100e-6; // 100 microseconds

                // Single-qubit gates
                let gate_time_1q = 35e-9; // 35 nanoseconds
                let gate_error_1q = 0.0004; // 0.04% error rate

                // Two-qubit gates (CNOT)
                // let _gate_time_2q = 275e-9; // 275 nanoseconds
                let gate_error_2q = 0.007; // 0.7% error rate

                // Readout errors
                let readout_error = 0.018; // 1.8% error

                // Add individual qubit noise
                for &qubit in qubits {
                    // Add thermal relaxation
                    self.model.add_thermal_relaxation(ThermalRelaxationChannel {
                        target: qubit,
                        t1,
                        t2,
                        gate_time: gate_time_1q,
                        excited_state_population: 0.007, // ~0.7% thermal excitation
                    });

                    // Add depolarizing noise for single-qubit gates
                    self.model.add_base_channel(NoiseChannelType::Depolarizing(
                        crate::noise::DepolarizingChannel {
                            target: qubit,
                            probability: gate_error_1q,
                        },
                    ));

                    // Add readout error as a bit flip channel
                    self.model.add_base_channel(NoiseChannelType::BitFlip(
                        crate::noise::BitFlipChannel {
                            target: qubit,
                            probability: readout_error,
                        },
                    ));
                }

                // Add two-qubit gate noise (for nearest-neighbor connectivity)
                for i in 0..qubits.len().saturating_sub(1) {
                    let q1 = qubits[i];
                    let q2 = qubits[i + 1];

                    // Add two-qubit depolarizing noise
                    self.model
                        .add_two_qubit_depolarizing(TwoQubitDepolarizingChannel {
                            qubit1: q1,
                            qubit2: q2,
                            probability: gate_error_2q,
                        });

                    // Add crosstalk between adjacent qubits
                    self.model.add_crosstalk(CrosstalkChannel {
                        primary: q1,
                        neighbor: q2,
                        strength: 0.0015, // 0.15% crosstalk
                    });
                }
            }
            "ibm_washington" | "ibm_eagle" => {
                // 127-qubit IBM Quantum Eagle processors
                // Parameters are approximate and based on typical values

                // Relaxation and dephasing times (average values)
                let t1 = 150e-6; // 150 microseconds
                let t2 = 120e-6; // 120 microseconds

                // Single-qubit gates
                let gate_time_1q = 30e-9; // 30 nanoseconds
                let gate_error_1q = 0.0003; // 0.03% error rate

                // Two-qubit gates (CNOT)
                // let _gate_time_2q = 220e-9; // 220 nanoseconds
                let gate_error_2q = 0.006; // 0.6% error rate

                // Readout errors
                let readout_error = 0.015; // 1.5% error

                // Add individual qubit noise
                for &qubit in qubits {
                    // Add thermal relaxation
                    self.model.add_thermal_relaxation(ThermalRelaxationChannel {
                        target: qubit,
                        t1,
                        t2,
                        gate_time: gate_time_1q,
                        excited_state_population: 0.006, // ~0.6% thermal excitation
                    });

                    // Add depolarizing noise for single-qubit gates
                    self.model.add_base_channel(NoiseChannelType::Depolarizing(
                        crate::noise::DepolarizingChannel {
                            target: qubit,
                            probability: gate_error_1q,
                        },
                    ));

                    // Add readout error as a bit flip channel
                    self.model.add_base_channel(NoiseChannelType::BitFlip(
                        crate::noise::BitFlipChannel {
                            target: qubit,
                            probability: readout_error,
                        },
                    ));
                }

                // Add two-qubit gate noise (for nearest-neighbor connectivity)
                for i in 0..qubits.len().saturating_sub(1) {
                    let q1 = qubits[i];
                    let q2 = qubits[i + 1];

                    // Add two-qubit depolarizing noise
                    self.model
                        .add_two_qubit_depolarizing(TwoQubitDepolarizingChannel {
                            qubit1: q1,
                            qubit2: q2,
                            probability: gate_error_2q,
                        });

                    // Add crosstalk between adjacent qubits
                    self.model.add_crosstalk(CrosstalkChannel {
                        primary: q1,
                        neighbor: q2,
                        strength: 0.001, // 0.1% crosstalk
                    });
                }
            }
            _ => {
                // Generic IBM Quantum device (conservative estimates)
                // Parameters are approximate and based on typical values

                // Relaxation and dephasing times (average values)
                let t1 = 100e-6; // 100 microseconds
                let t2 = 80e-6; // 80 microseconds

                // Single-qubit gates
                let gate_time_1q = 40e-9; // 40 nanoseconds
                let gate_error_1q = 0.001; // 0.1% error rate

                // Two-qubit gates (CNOT)
                // let _gate_time_2q = 300e-9; // 300 nanoseconds
                let gate_error_2q = 0.01; // 1% error rate

                // Readout errors
                let readout_error = 0.025; // 2.5% error

                // Add individual qubit noise
                for &qubit in qubits {
                    // Add thermal relaxation
                    self.model.add_thermal_relaxation(ThermalRelaxationChannel {
                        target: qubit,
                        t1,
                        t2,
                        gate_time: gate_time_1q,
                        excited_state_population: 0.01, // ~1% thermal excitation
                    });

                    // Add depolarizing noise for single-qubit gates
                    self.model.add_base_channel(NoiseChannelType::Depolarizing(
                        crate::noise::DepolarizingChannel {
                            target: qubit,
                            probability: gate_error_1q,
                        },
                    ));

                    // Add readout error as a bit flip channel
                    self.model.add_base_channel(NoiseChannelType::BitFlip(
                        crate::noise::BitFlipChannel {
                            target: qubit,
                            probability: readout_error,
                        },
                    ));
                }

                // Add two-qubit gate noise (for nearest-neighbor connectivity)
                for i in 0..qubits.len().saturating_sub(1) {
                    let q1 = qubits[i];
                    let q2 = qubits[i + 1];

                    // Add two-qubit depolarizing noise
                    self.model
                        .add_two_qubit_depolarizing(TwoQubitDepolarizingChannel {
                            qubit1: q1,
                            qubit2: q2,
                            probability: gate_error_2q,
                        });

                    // Add crosstalk between adjacent qubits
                    self.model.add_crosstalk(CrosstalkChannel {
                        primary: q1,
                        neighbor: q2,
                        strength: 0.003, // 0.3% crosstalk
                    });
                }
            }
        }

        self
    }

    /// Add realistic Rigetti device noise parameters
    #[must_use]
    pub fn with_rigetti_device_noise(mut self, qubits: &[QubitId], device_name: &str) -> Self {
        match device_name {
            "Aspen-M-3" | "Aspen-M-2" => {
                // Rigetti Aspen-M series processors
                // Parameters are approximate and based on typical values

                // Relaxation and dephasing times (average values)
                let t1 = 20e-6; // 20 microseconds
                let t2 = 15e-6; // 15 microseconds

                // Single-qubit gates
                let gate_time_1q = 50e-9; // 50 nanoseconds
                let gate_error_1q = 0.0015; // 0.15% error rate

                // Two-qubit gates (CZ)
                // let _gate_time_2q = 220e-9; // 220 nanoseconds
                let gate_error_2q = 0.02; // 2% error rate

                // Readout errors
                let readout_error = 0.03; // 3% error

                // Add individual qubit noise
                for &qubit in qubits {
                    // Add thermal relaxation
                    self.model.add_thermal_relaxation(ThermalRelaxationChannel {
                        target: qubit,
                        t1,
                        t2,
                        gate_time: gate_time_1q,
                        excited_state_population: 0.02, // ~2% thermal excitation
                    });

                    // Add depolarizing noise for single-qubit gates
                    self.model.add_base_channel(NoiseChannelType::Depolarizing(
                        crate::noise::DepolarizingChannel {
                            target: qubit,
                            probability: gate_error_1q,
                        },
                    ));

                    // Add readout error as a bit flip channel
                    self.model.add_base_channel(NoiseChannelType::BitFlip(
                        crate::noise::BitFlipChannel {
                            target: qubit,
                            probability: readout_error,
                        },
                    ));
                }

                // Add two-qubit gate noise (for nearest-neighbor connectivity)
                for i in 0..qubits.len().saturating_sub(1) {
                    let q1 = qubits[i];
                    let q2 = qubits[i + 1];

                    // Add two-qubit depolarizing noise
                    self.model
                        .add_two_qubit_depolarizing(TwoQubitDepolarizingChannel {
                            qubit1: q1,
                            qubit2: q2,
                            probability: gate_error_2q,
                        });

                    // Add crosstalk between adjacent qubits
                    self.model.add_crosstalk(CrosstalkChannel {
                        primary: q1,
                        neighbor: q2,
                        strength: 0.004, // 0.4% crosstalk
                    });
                }
            }
            _ => {
                // Generic Rigetti device (conservative estimates)
                // Parameters are approximate and based on typical values

                // Relaxation and dephasing times (average values)
                let t1 = 15e-6; // 15 microseconds
                let t2 = 12e-6; // 12 microseconds

                // Single-qubit gates
                let gate_time_1q = 60e-9; // 60 nanoseconds
                let gate_error_1q = 0.002; // 0.2% error rate

                // Two-qubit gates (CZ)
                // let _gate_time_2q = 250e-9; // 250 nanoseconds
                let gate_error_2q = 0.025; // 2.5% error rate

                // Readout errors
                let readout_error = 0.035; // 3.5% error

                // Add individual qubit noise
                for &qubit in qubits {
                    // Add thermal relaxation
                    self.model.add_thermal_relaxation(ThermalRelaxationChannel {
                        target: qubit,
                        t1,
                        t2,
                        gate_time: gate_time_1q,
                        excited_state_population: 0.025, // ~2.5% thermal excitation
                    });

                    // Add depolarizing noise for single-qubit gates
                    self.model.add_base_channel(NoiseChannelType::Depolarizing(
                        crate::noise::DepolarizingChannel {
                            target: qubit,
                            probability: gate_error_1q,
                        },
                    ));

                    // Add readout error as a bit flip channel
                    self.model.add_base_channel(NoiseChannelType::BitFlip(
                        crate::noise::BitFlipChannel {
                            target: qubit,
                            probability: readout_error,
                        },
                    ));
                }

                // Add two-qubit gate noise (for nearest-neighbor connectivity)
                for i in 0..qubits.len().saturating_sub(1) {
                    let q1 = qubits[i];
                    let q2 = qubits[i + 1];

                    // Add two-qubit depolarizing noise
                    self.model
                        .add_two_qubit_depolarizing(TwoQubitDepolarizingChannel {
                            qubit1: q1,
                            qubit2: q2,
                            probability: gate_error_2q,
                        });

                    // Add crosstalk between adjacent qubits
                    self.model.add_crosstalk(CrosstalkChannel {
                        primary: q1,
                        neighbor: q2,
                        strength: 0.005, // 0.5% crosstalk
                    });
                }
            }
        }

        self
    }

    /// Add custom thermal relaxation parameters
    #[must_use]
    pub fn with_custom_thermal_relaxation(
        mut self,
        qubits: &[QubitId],
        t1: Duration,
        t2: Duration,
        gate_time: Duration,
    ) -> Self {
        let t1_seconds = t1.as_secs_f64();
        let t2_seconds = t2.as_secs_f64();
        let gate_time_seconds = gate_time.as_secs_f64();

        for &qubit in qubits {
            self.model.add_thermal_relaxation(ThermalRelaxationChannel {
                target: qubit,
                t1: t1_seconds,
                t2: t2_seconds,
                gate_time: gate_time_seconds,
                excited_state_population: 0.01, // Default 1% thermal excitation
            });
        }

        self
    }

    /// Add custom two-qubit depolarizing noise
    #[must_use]
    pub fn with_custom_two_qubit_noise(
        mut self,
        qubit_pairs: &[(QubitId, QubitId)],
        probability: f64,
    ) -> Self {
        for &(q1, q2) in qubit_pairs {
            self.model
                .add_two_qubit_depolarizing(TwoQubitDepolarizingChannel {
                    qubit1: q1,
                    qubit2: q2,
                    probability,
                });
        }

        self
    }

    /// Add custom crosstalk noise between pairs of qubits
    #[must_use]
    pub fn with_custom_crosstalk(
        mut self,
        qubit_pairs: &[(QubitId, QubitId)],
        strength: f64,
    ) -> Self {
        for &(q1, q2) in qubit_pairs {
            self.model.add_crosstalk(CrosstalkChannel {
                primary: q1,
                neighbor: q2,
                strength,
            });
        }

        self
    }

    /// Build the noise model
    #[must_use]
    pub fn build(self) -> AdvancedNoiseModel {
        self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Accumulate `sum_k K_k^dagger K_k` for a Kraus set whose operators are
    /// flattened row-major square matrices, returning the resulting `dim x dim`
    /// matrix (flattened row-major). The dimension is inferred from the length of
    /// each inner vector (a `dim x dim` matrix has `dim * dim` entries).
    fn accumulate_kraus_completeness(kraus: &[Vec<Complex64>]) -> Vec<Complex64> {
        assert!(!kraus.is_empty(), "Kraus set must be non-empty");

        let len = kraus[0].len();
        let dim = (len as f64).sqrt().round() as usize;
        assert_eq!(
            dim * dim,
            len,
            "each Kraus operator must be a square matrix"
        );

        let mut acc = vec![Complex64::new(0.0, 0.0); len];
        for op in kraus {
            assert_eq!(op.len(), len, "all Kraus operators must share a dimension");
            // (K^dagger K)[i][j] = sum_k conj(K[k][i]) * K[k][j]
            for i in 0..dim {
                for j in 0..dim {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for k in 0..dim {
                        sum += op[k * dim + i].conj() * op[k * dim + j];
                    }
                    acc[i * dim + j] += sum;
                }
            }
        }

        acc
    }

    /// Assert that a flattened `dim x dim` matrix equals the identity within
    /// `tol`.
    fn assert_is_identity(matrix: &[Complex64], tol: f64) {
        let len = matrix.len();
        let dim = (len as f64).sqrt().round() as usize;
        assert_eq!(dim * dim, len, "matrix must be square");

        for i in 0..dim {
            for j in 0..dim {
                let value = matrix[i * dim + j];
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (value.re - expected).abs() < tol && value.im.abs() < tol,
                    "element [{i}][{j}] = {value:?} is not {expected} within {tol}"
                );
            }
        }
    }

    #[test]
    fn test_two_qubit_depolarizing_kraus_trace_preserving() {
        let channel = TwoQubitDepolarizingChannel {
            qubit1: QubitId::new(0),
            qubit2: QubitId::new(1),
            probability: 0.137,
        };

        let kraus = channel.kraus_operators();

        // Two-qubit depolarizing has exactly 16 Kraus operators (15 Pauli pairs
        // plus identity), not the old fabricated 2 one-element vectors.
        assert_eq!(kraus.len(), 16, "expected 16 Kraus operators");
        for op in &kraus {
            assert_eq!(op.len(), 16, "each operator must be a flattened 4x4 matrix");
        }

        let completeness = accumulate_kraus_completeness(&kraus);
        assert_is_identity(&completeness, 1e-10);
    }

    #[test]
    fn test_two_qubit_depolarizing_kraus_edge_probabilities() {
        for &probability in &[0.0_f64, 1.0_f64] {
            let channel = TwoQubitDepolarizingChannel {
                qubit1: QubitId::new(0),
                qubit2: QubitId::new(1),
                probability,
            };
            let kraus = channel.kraus_operators();
            assert_eq!(kraus.len(), 16);
            let completeness = accumulate_kraus_completeness(&kraus);
            assert_is_identity(&completeness, 1e-10);
        }
    }

    #[test]
    fn test_thermal_relaxation_kraus_trace_preserving() {
        let channel = ThermalRelaxationChannel {
            target: QubitId::new(0),
            t1: 100e-6,
            t2: 80e-6,
            gate_time: 35e-9,
            excited_state_population: 0.03,
        };

        let kraus = channel.kraus_operators();

        // Composition of generalized amplitude damping (4 ops) and phase damping
        // (2 ops) yields 8 single-qubit operators, well beyond the old single
        // fabricated one-element vector.
        assert!(
            kraus.len() >= 4,
            "expected at least 4 Kraus operators, got {}",
            kraus.len()
        );
        assert_eq!(kraus.len(), 8, "expected 8 composed Kraus operators");
        for op in &kraus {
            assert_eq!(op.len(), 4, "each operator must be a flattened 2x2 matrix");
        }

        let completeness = accumulate_kraus_completeness(&kraus);
        assert_is_identity(&completeness, 1e-10);
    }

    #[test]
    fn test_thermal_relaxation_kraus_extreme_parameters() {
        // Very short T1/T2 relative to gate time drive gamma and gamma_phi toward
        // 1; the decomposition must remain trace-preserving and clamped.
        let channel = ThermalRelaxationChannel {
            target: QubitId::new(0),
            t1: 1e-9,
            t2: 1e-9,
            gate_time: 1e-6,
            excited_state_population: 0.5,
        };

        let kraus = channel.kraus_operators();
        assert_eq!(kraus.len(), 8);
        let completeness = accumulate_kraus_completeness(&kraus);
        assert_is_identity(&completeness, 1e-10);
    }

    #[test]
    fn test_crosstalk_kraus_trace_preserving() {
        let channel = CrosstalkChannel {
            primary: QubitId::new(0),
            neighbor: QubitId::new(1),
            strength: 0.3,
        };

        let kraus = channel.kraus_operators();

        // Coherent ZZ crosstalk is a single unitary Kraus operator on the
        // two-qubit (4x4) space, not the old fabricated 1-element vector.
        assert_eq!(kraus.len(), 1, "coherent crosstalk has a single Kraus op");
        assert_eq!(
            kraus[0].len(),
            16,
            "the operator must be a flattened 4x4 unitary"
        );

        let completeness = accumulate_kraus_completeness(&kraus);
        assert_is_identity(&completeness, 1e-10);
    }

    #[test]
    fn test_tensor_2x2_matches_known_pauli_product() {
        let paulis = single_qubit_paulis();
        // Z x X should map |0>|+> style basis correctly; verify the explicit
        // 4x4 layout against a hand-computed reference.
        let z = &paulis[3];
        let x = &paulis[1];
        let zx = tensor_2x2(z, x);

        // Z x X = [[0,1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,-1,0]]
        let one = Complex64::new(1.0, 0.0);
        let neg_one = Complex64::new(-1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let expected = vec![
            zero, one, zero, zero, one, zero, zero, zero, zero, zero, zero, neg_one, zero, zero,
            neg_one, zero,
        ];
        for (got, want) in zx.iter().zip(expected.iter()) {
            assert!((got.re - want.re).abs() < 1e-12 && (got.im - want.im).abs() < 1e-12);
        }
    }

    #[test]
    fn test_matmul_2x2_identity_and_pauli() {
        let paulis = single_qubit_paulis();
        let identity = &paulis[0];
        let x = &paulis[1];

        // I * X == X
        let product = matmul_2x2(identity, x);
        for (got, want) in product.iter().zip(x.iter()) {
            assert!((got.re - want.re).abs() < 1e-12 && (got.im - want.im).abs() < 1e-12);
        }

        // X * X == I
        let xx = matmul_2x2(x, x);
        for (got, want) in xx.iter().zip(identity.iter()) {
            assert!((got.re - want.re).abs() < 1e-12 && (got.im - want.im).abs() < 1e-12);
        }
    }
}
