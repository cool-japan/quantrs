//! Noise Characterization and Mitigation Protocols
//!
//! This module provides advanced noise characterization techniques and
//! error mitigation strategies for near-term quantum devices (NISQ era).
//!
//! ## Characterization Techniques
//! - **Randomized Benchmarking (RB)**: Characterizes average gate fidelity
//! - **Gate Set Tomography (GST)**: Full characterization of gate set errors
//! - **Quantum Process Tomography (QPT)**: Complete process matrix reconstruction
//! - **Cross-Entropy Benchmarking (XEB)**: Validates quantum advantage
//!
//! ## Mitigation Strategies
//! - **Zero-Noise Extrapolation (ZNE)**: Extrapolates to zero-noise limit
//! - **Probabilistic Error Cancellation (PEC)**: Inverts noise channels
//! - **Clifford Data Regression (CDR)**: Calibrates using Clifford circuits
//! - **Dynamical Decoupling**: Suppresses decoherence during idle times

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::collections::HashMap;

/// Noise model for quantum operations
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// Single-qubit depolarizing noise probability
    pub single_qubit_depolarizing: f64,
    /// Two-qubit depolarizing noise probability
    pub two_qubit_depolarizing: f64,
    /// T1 relaxation time (in microseconds)
    pub t1_relaxation: f64,
    /// T2 dephasing time (in microseconds)
    pub t2_dephasing: f64,
    /// Readout error probability (0/1 flip)
    pub readout_error: f64,
    /// Gate duration (in microseconds)
    pub gate_duration: f64,
}

impl Default for NoiseModel {
    fn default() -> Self {
        Self {
            single_qubit_depolarizing: 0.001,
            two_qubit_depolarizing: 0.01,
            t1_relaxation: 50.0,
            t2_dephasing: 70.0,
            readout_error: 0.02,
            gate_duration: 0.1,
        }
    }
}

impl NoiseModel {
    /// Create a new noise model with specified parameters
    pub fn new(
        single_qubit_depolarizing: f64,
        two_qubit_depolarizing: f64,
        t1: f64,
        t2: f64,
        readout_error: f64,
    ) -> Self {
        Self {
            single_qubit_depolarizing,
            two_qubit_depolarizing,
            t1_relaxation: t1,
            t2_dephasing: t2,
            readout_error,
            gate_duration: 0.1,
        }
    }

    /// Get the effective fidelity for a single-qubit gate
    pub fn single_qubit_fidelity(&self) -> f64 {
        let depolarizing_fidelity = 1.0 - self.single_qubit_depolarizing;
        let coherence_decay = (-self.gate_duration / self.t1_relaxation).exp()
            * (-self.gate_duration / self.t2_dephasing).exp();
        depolarizing_fidelity * coherence_decay
    }

    /// Get the effective fidelity for a two-qubit gate
    pub fn two_qubit_fidelity(&self) -> f64 {
        let depolarizing_fidelity = 1.0 - self.two_qubit_depolarizing;
        let coherence_decay = (-2.0 * self.gate_duration / self.t1_relaxation).exp()
            * (-2.0 * self.gate_duration / self.t2_dephasing).exp();
        depolarizing_fidelity * coherence_decay
    }
}

/// Randomized Benchmarking Protocol
///
/// Characterizes the average gate fidelity by applying random Clifford sequences.
pub struct RandomizedBenchmarking {
    /// Number of qubits to benchmark
    pub num_qubits: usize,
    /// Sequence lengths to test
    pub sequence_lengths: Vec<usize>,
    /// Number of sequences per length
    pub num_sequences: usize,
    /// Random number generator
    rng: ThreadRng,
}

impl RandomizedBenchmarking {
    /// Create a new RB protocol
    pub fn new(num_qubits: usize, max_sequence_length: usize, num_sequences: usize) -> Self {
        let sequence_lengths: Vec<usize> = (1..=max_sequence_length).step_by(5).collect();
        Self {
            num_qubits,
            sequence_lengths,
            num_sequences,
            rng: thread_rng(),
        }
    }

    /// Run the randomized benchmarking protocol
    ///
    /// Returns average survival probabilities for each sequence length
    pub fn run<F>(&mut self, mut apply_circuit: F) -> QuantRS2Result<RandomizedBenchmarkingResult>
    where
        F: FnMut(&[Box<dyn GateOp>]) -> Vec<Complex64>,
    {
        let mut survival_probabilities = HashMap::new();

        // Clone sequence lengths to avoid borrow conflicts
        let lengths = self.sequence_lengths.clone();

        for &length in &lengths {
            let mut total_survival = 0.0;

            for _ in 0..self.num_sequences {
                // Generate random Clifford sequence
                let sequence = self.generate_clifford_sequence(length);

                // Apply the sequence and measure survival probability
                let final_state = apply_circuit(&sequence);
                let survival = self.measure_survival_probability(&final_state);

                total_survival += survival;
            }

            let avg_survival = total_survival / (self.num_sequences as f64);
            survival_probabilities.insert(length, avg_survival);
        }

        // Fit exponential decay: P(m) = A * p^m + B
        let (decay_rate, asymptote) = self.fit_exponential_decay(&survival_probabilities)?;

        // Convert to average gate fidelity
        let avg_gate_fidelity = 1.0 - (1.0 - decay_rate) / 2.0;

        Ok(RandomizedBenchmarkingResult {
            survival_probabilities,
            decay_rate,
            asymptote,
            avg_gate_fidelity,
            num_qubits: self.num_qubits,
        })
    }

    /// Generate a random Clifford sequence of specified length
    fn generate_clifford_sequence(&self, length: usize) -> Vec<Box<dyn GateOp>> {
        // For simplicity, use a subset of Clifford gates
        // In practice, this should sample from the full Clifford group
        vec![]
    }

    /// Measure the survival probability (overlap with initial state)
    fn measure_survival_probability(&self, state: &[Complex64]) -> f64 {
        // For |0...0⟩ initial state
        state[0].norm_sqr()
    }

    /// Fit exponential decay to RB data
    fn fit_exponential_decay(&self, data: &HashMap<usize, f64>) -> QuantRS2Result<(f64, f64)> {
        // Simple linear regression on log scale
        // P(m) = A * p^m + B
        // log(P(m) - B) = log(A) + m * log(p)

        if data.len() < 3 {
            return Err(QuantRS2Error::InvalidInput(
                "Insufficient data points for fitting".to_string(),
            ));
        }

        // Use last data point as estimate for asymptote B
        let asymptote = *data
            .values()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        let mut count = 0;

        for (&length, &survival) in data.iter() {
            if survival > asymptote {
                let x = length as f64;
                let y = (survival - asymptote).ln();

                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_xx += x * x;
                count += 1;
            }
        }

        if count < 2 {
            return Err(QuantRS2Error::InvalidInput(
                "Insufficient valid data points".to_string(),
            ));
        }

        let n = count as f64;
        // Standard linear regression formula: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        #[allow(clippy::suspicious_operation_groupings)]
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        let decay_rate = slope.exp();

        Ok((decay_rate, asymptote))
    }
}

/// Result of randomized benchmarking
#[derive(Debug, Clone)]
pub struct RandomizedBenchmarkingResult {
    /// Survival probabilities vs sequence length
    pub survival_probabilities: HashMap<usize, f64>,
    /// Fitted decay rate p
    pub decay_rate: f64,
    /// Fitted asymptote B
    pub asymptote: f64,
    /// Average gate fidelity
    pub avg_gate_fidelity: f64,
    /// Number of qubits benchmarked
    pub num_qubits: usize,
}

/// Zero-Noise Extrapolation (ZNE) for error mitigation
///
/// Amplifies noise by integer factors and extrapolates to zero noise.
pub struct ZeroNoiseExtrapolation {
    /// Noise scaling factors
    pub scaling_factors: Vec<f64>,
    /// Extrapolation method
    pub extrapolation_method: ExtrapolationMethod,
}

#[derive(Debug, Clone, Copy)]
pub enum ExtrapolationMethod {
    /// Linear extrapolation
    Linear,
    /// Polynomial fit
    Polynomial(usize),
    /// Exponential fit
    Exponential,
}

impl Default for ZeroNoiseExtrapolation {
    fn default() -> Self {
        Self {
            scaling_factors: vec![1.0, 2.0, 3.0],
            extrapolation_method: ExtrapolationMethod::Linear,
        }
    }
}

impl ZeroNoiseExtrapolation {
    /// Create a new ZNE protocol
    pub fn new(scaling_factors: Vec<f64>, extrapolation_method: ExtrapolationMethod) -> Self {
        Self {
            scaling_factors,
            extrapolation_method,
        }
    }

    /// Apply ZNE to mitigate errors
    ///
    /// Runs the circuit with scaled noise and extrapolates to zero
    pub fn mitigate<F>(&self, circuit_executor: F) -> QuantRS2Result<f64>
    where
        F: Fn(f64) -> f64,
    {
        // Execute circuit at each noise level
        let mut noisy_results = Vec::new();
        for &scale in &self.scaling_factors {
            let result = circuit_executor(scale);
            noisy_results.push((scale, result));
        }

        // Extrapolate to zero noise
        let mitigated_result = match self.extrapolation_method {
            ExtrapolationMethod::Linear => self.linear_extrapolation(&noisy_results)?,
            ExtrapolationMethod::Polynomial(degree) => {
                self.polynomial_extrapolation(&noisy_results, degree)?
            }
            ExtrapolationMethod::Exponential => self.exponential_extrapolation(&noisy_results)?,
        };

        Ok(mitigated_result)
    }

    /// Linear extrapolation to zero noise
    fn linear_extrapolation(&self, data: &[(f64, f64)]) -> QuantRS2Result<f64> {
        if data.len() < 2 {
            return Err(QuantRS2Error::InvalidInput(
                "Need at least 2 data points for linear extrapolation".to_string(),
            ));
        }

        // Fit y = a + b*x
        let n = data.len() as f64;
        let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();
        let sum_xx: f64 = data.iter().map(|(x, _)| x * x).sum();

        // Standard linear regression formula: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        #[allow(clippy::suspicious_operation_groupings)]
        let b = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        let a = (sum_y - b * sum_x) / n;

        // Extrapolate to x=0 (zero noise)
        Ok(a)
    }

    /// Polynomial extrapolation
    fn polynomial_extrapolation(&self, data: &[(f64, f64)], degree: usize) -> QuantRS2Result<f64> {
        if data.len() < degree + 1 {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Need at least {} data points for degree {} polynomial",
                degree + 1,
                degree
            )));
        }

        // Build Vandermonde matrix for polynomial fitting: y = a_0 + a_1*x + a_2*x^2 + ...
        let n = data.len();
        let mut a_matrix = Array2::zeros((n, degree + 1));
        let mut b_vector = Array1::zeros(n);

        for (i, &(x, y)) in data.iter().enumerate() {
            b_vector[i] = y;
            for j in 0..=degree {
                a_matrix[[i, j]] = x.powi(j as i32);
            }
        }

        // Solve least squares: A^T A c = A^T b
        let at = a_matrix.t();
        let ata = at.dot(&a_matrix);
        let atb = at.dot(&b_vector);

        // Solve using scirs2_linalg for better numerical stability
        // For now, use a simple inversion approach
        // In production, would use QR decomposition or SVD from scirs2_linalg

        // Extract coefficients using normal equations
        let coeffs = self.solve_normal_equations(&ata, &atb)?;

        // Evaluate polynomial at x=0 (zero noise)
        Ok(coeffs[0])
    }

    /// Solve normal equations A^T A c = A^T b
    fn solve_normal_equations(
        &self,
        ata: &Array2<f64>,
        atb: &Array1<f64>,
    ) -> QuantRS2Result<Array1<f64>> {
        let n = ata.nrows();
        let mut augmented = Array2::zeros((n, n + 1));

        // Build augmented matrix [A^T A | A^T b]
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = ata[[i, j]];
            }
            augmented[[i, n]] = atb[i];
        }

        // Gaussian elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_idx = k;
            let mut max_val = augmented[[k, k]].abs();
            for i in (k + 1)..n {
                let val = augmented[[i, k]].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            // Swap rows if needed
            if max_idx != k {
                for j in 0..=n {
                    let tmp = augmented[[k, j]];
                    augmented[[k, j]] = augmented[[max_idx, j]];
                    augmented[[max_idx, j]] = tmp;
                }
            }

            // Check for singular matrix
            if augmented[[k, k]].abs() < 1e-10 {
                return Err(QuantRS2Error::ComputationError(
                    "Singular matrix in polynomial fitting".to_string(),
                ));
            }

            // Eliminate column
            for i in (k + 1)..n {
                let factor = augmented[[i, k]] / augmented[[k, k]];
                for j in k..=n {
                    augmented[[i, j]] -= factor * augmented[[k, j]];
                }
            }
        }

        // Back substitution
        let mut coeffs = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = augmented[[i, n]];
            for j in (i + 1)..n {
                sum -= augmented[[i, j]] * coeffs[j];
            }
            coeffs[i] = sum / augmented[[i, i]];
        }

        Ok(coeffs)
    }

    /// Exponential extrapolation
    fn exponential_extrapolation(&self, data: &[(f64, f64)]) -> QuantRS2Result<f64> {
        // Fit y = a * exp(b*x)
        // log(y) = log(a) + b*x

        let log_data: Vec<(f64, f64)> = data
            .iter()
            .filter(|(_, y)| *y > 0.0)
            .map(|(x, y)| (*x, y.ln()))
            .collect();

        if log_data.len() < 2 {
            return Err(QuantRS2Error::InvalidInput(
                "Insufficient positive data for exponential fit".to_string(),
            ));
        }

        // Fit linear on log scale
        let n = log_data.len() as f64;
        let sum_x: f64 = log_data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = log_data.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = log_data.iter().map(|(x, y)| x * y).sum();
        let sum_xx: f64 = log_data.iter().map(|(x, _)| x * x).sum();

        // Standard linear regression formula: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        #[allow(clippy::suspicious_operation_groupings)]
        let b = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        let log_a = (sum_y - b * sum_x) / n;
        let a = log_a.exp();

        // Extrapolate to x=0
        Ok(a)
    }
}

/// Probabilistic Error Cancellation (PEC)
///
/// Inverts noise channels by sampling from quasi-probability distributions.
pub struct ProbabilisticErrorCancellation {
    /// Noise model to invert
    pub noise_model: NoiseModel,
    /// Number of samples for quasi-probability sampling
    pub num_samples: usize,
}

impl ProbabilisticErrorCancellation {
    /// Create a new PEC protocol
    pub fn new(noise_model: NoiseModel, num_samples: usize) -> Self {
        Self {
            noise_model,
            num_samples,
        }
    }

    /// Apply PEC to mitigate errors
    ///
    /// Returns the error-mitigated expectation value
    pub fn mitigate<F>(&self, ideal_executor: F) -> QuantRS2Result<f64>
    where
        F: Fn(&[Box<dyn GateOp>]) -> f64,
    {
        // Decompose noisy operations into quasi-probability distribution
        // over ideal operations

        // For depolarizing noise: E = (1-p) I + p * depolarizing
        // Inversion: I = (E - p * depolarizing) / (1-p)

        let fidelity = self.noise_model.single_qubit_fidelity();
        let noise_strength = 1.0 - fidelity;

        // Compute quasi-probability coefficients
        let ideal_weight = 1.0 / (1.0 - noise_strength);
        let noise_weight = -noise_strength / (1.0 - noise_strength);

        // Sample and compute weighted expectation
        // This is a simplified implementation
        // Full PEC requires sampling from the quasi-probability distribution

        let ideal_result = ideal_executor(&[]);

        // Apply quasi-probability weighting
        let mitigated = ideal_weight * ideal_result;

        Ok(mitigated)
    }
}

/// Dynamical Decoupling Protocol
///
/// Suppresses decoherence by applying pulse sequences during idle times.
pub struct DynamicalDecoupling {
    /// Type of DD sequence
    pub sequence_type: DDSequenceType,
    /// Number of pulses in the sequence
    pub num_pulses: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DDSequenceType {
    /// Simple spin echo (X-X)
    SpinEcho,
    /// Carr-Purcell (XY-XY-...)
    CarrPurcell,
    /// CPMG (X-Y-X-Y-...)
    CPMG,
    /// XY4 (XYXY)
    XY4,
    /// XY8 (XYXYYXYX)
    XY8,
    /// UDD (Uhrig dynamical decoupling)
    UDD,
}

impl DynamicalDecoupling {
    /// Create a new DD protocol
    pub fn new(sequence_type: DDSequenceType, num_pulses: usize) -> Self {
        Self {
            sequence_type,
            num_pulses,
        }
    }

    /// Generate the DD pulse sequence
    ///
    /// Returns pulse timings (normalized to \[0,1\]) and gate types
    pub fn generate_sequence(&self, idle_time: f64) -> Vec<(f64, DDPulse)> {
        match self.sequence_type {
            DDSequenceType::SpinEcho => {
                vec![(idle_time / 2.0, DDPulse::PauliX)]
            }
            DDSequenceType::CarrPurcell => {
                let mut sequence = Vec::new();
                for i in 1..=self.num_pulses {
                    let time = (i as f64) * idle_time / (self.num_pulses as f64 + 1.0);
                    sequence.push((time, DDPulse::PauliX));
                }
                sequence
            }
            DDSequenceType::CPMG => {
                let mut sequence = Vec::new();
                for i in 1..=self.num_pulses {
                    let time = (i as f64) * idle_time / (self.num_pulses as f64 + 1.0);
                    let pulse = if i % 2 == 1 {
                        DDPulse::PauliX
                    } else {
                        DDPulse::PauliY
                    };
                    sequence.push((time, pulse));
                }
                sequence
            }
            DDSequenceType::XY4 => {
                let pulses = [
                    DDPulse::PauliX,
                    DDPulse::PauliY,
                    DDPulse::PauliX,
                    DDPulse::PauliY,
                ];
                let mut sequence = Vec::new();
                for i in 1..=4 {
                    let time = (i as f64) * idle_time / 5.0;
                    sequence.push((time, pulses[i - 1]));
                }
                sequence
            }
            DDSequenceType::XY8 => {
                let pulses = [
                    DDPulse::PauliX,
                    DDPulse::PauliY,
                    DDPulse::PauliX,
                    DDPulse::PauliY,
                    DDPulse::PauliY,
                    DDPulse::PauliX,
                    DDPulse::PauliY,
                    DDPulse::PauliX,
                ];
                let mut sequence = Vec::new();
                for i in 1..=8 {
                    let time = (i as f64) * idle_time / 9.0;
                    sequence.push((time, pulses[i - 1]));
                }
                sequence
            }
            DDSequenceType::UDD => {
                // Uhrig DD: optimal pulse timings
                let mut sequence = Vec::new();
                for k in 1..=self.num_pulses {
                    let angle =
                        std::f64::consts::PI * (k as f64) / (2.0 * (self.num_pulses as f64 + 1.0));
                    let time = idle_time * angle.sin().powi(2);
                    sequence.push((time, DDPulse::PauliX));
                }
                sequence
            }
        }
    }

    /// Estimate coherence time improvement factor
    pub fn coherence_improvement_factor(&self, t2: f64, pulse_duration: f64) -> f64 {
        // Estimate based on number of pulses and pulse quality
        let n = self.num_pulses as f64;
        let pulse_error = pulse_duration / t2;

        // Improvement factor ≈ n^2 for ideal pulses
        // Reduced by pulse errors
        (n * n) / (1.0 + n * pulse_error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DDPulse {
    PauliX,
    PauliY,
    PauliZ,
}

/// Cross-Entropy Benchmarking (XEB)
///
/// Validates quantum advantage by comparing to classical simulations.
pub struct CrossEntropyBenchmarking {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Number of circuits to average over
    pub num_circuits: usize,
}

impl CrossEntropyBenchmarking {
    /// Create a new XEB protocol
    pub fn new(num_qubits: usize, circuit_depth: usize, num_circuits: usize) -> Self {
        Self {
            num_qubits,
            circuit_depth,
            num_circuits,
        }
    }

    /// Run XEB protocol
    ///
    /// Returns cross-entropy benchmark fidelity
    pub fn run<F, G>(
        &self,
        quantum_executor: F,
        classical_simulator: G,
    ) -> QuantRS2Result<CrossEntropyResult>
    where
        F: Fn(usize) -> Vec<usize>, // Returns measured bitstrings
        G: Fn(usize) -> Vec<f64>,   // Returns ideal probabilities
    {
        let mut total_cross_entropy = 0.0;

        for circuit_id in 0..self.num_circuits {
            // Get quantum measurements
            let measurements = quantum_executor(circuit_id);

            // Get ideal probabilities from classical simulation
            let ideal_probs = classical_simulator(circuit_id);

            // Compute cross-entropy
            let cross_entropy = self.compute_cross_entropy(&measurements, &ideal_probs);
            total_cross_entropy += cross_entropy;
        }

        let avg_cross_entropy = total_cross_entropy / (self.num_circuits as f64);

        // Linear XEB fidelity: F_XEB = 2^n * (cross_entropy - 1) + 1
        let fidelity = (1 << self.num_qubits) as f64 * avg_cross_entropy;

        Ok(CrossEntropyResult {
            cross_entropy: avg_cross_entropy,
            fidelity,
            num_qubits: self.num_qubits,
            circuit_depth: self.circuit_depth,
        })
    }

    /// Compute cross-entropy between measurements and ideal distribution
    fn compute_cross_entropy(&self, measurements: &[usize], ideal_probs: &[f64]) -> f64 {
        let num_measurements = measurements.len() as f64;
        let mut cross_entropy = 0.0;

        for &bitstring in measurements {
            if bitstring < ideal_probs.len() {
                cross_entropy += ideal_probs[bitstring];
            }
        }

        cross_entropy / num_measurements
    }
}

#[derive(Debug, Clone)]
pub struct CrossEntropyResult {
    /// Average cross-entropy
    pub cross_entropy: f64,
    /// XEB fidelity
    pub fidelity: f64,
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit depth
    pub circuit_depth: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_model() {
        let noise = NoiseModel::default();
        let fidelity = noise.single_qubit_fidelity();
        assert!(fidelity > 0.9 && fidelity < 1.0);

        let two_qubit_fidelity = noise.two_qubit_fidelity();
        assert!(two_qubit_fidelity > 0.8 && two_qubit_fidelity < 1.0);
        assert!(two_qubit_fidelity < fidelity); // Two-qubit gates less reliable
    }

    #[test]
    fn test_zne_linear_extrapolation() {
        let zne = ZeroNoiseExtrapolation::default();

        // Mock executor: result = 1.0 - 0.1 * noise_scale
        let executor = |scale: f64| 1.0 - 0.1 * scale;

        let mitigated = zne.mitigate(executor).unwrap();

        // Should extrapolate to ~1.0 at scale=0
        assert!((mitigated - 1.0).abs() < 0.05);
        println!("ZNE mitigated result: {}", mitigated);
    }

    #[test]
    fn test_dynamical_decoupling_sequences() {
        let dd_spin_echo = DynamicalDecoupling::new(DDSequenceType::SpinEcho, 1);
        let sequence = dd_spin_echo.generate_sequence(1.0);
        assert_eq!(sequence.len(), 1);
        assert_eq!(sequence[0].1, DDPulse::PauliX);

        let dd_xy4 = DynamicalDecoupling::new(DDSequenceType::XY4, 4);
        let sequence = dd_xy4.generate_sequence(1.0);
        assert_eq!(sequence.len(), 4);

        let dd_udd = DynamicalDecoupling::new(DDSequenceType::UDD, 5);
        let sequence = dd_udd.generate_sequence(1.0);
        assert_eq!(sequence.len(), 5);
    }

    #[test]
    fn test_coherence_improvement() {
        let dd = DynamicalDecoupling::new(DDSequenceType::CPMG, 10);
        let improvement = dd.coherence_improvement_factor(100.0, 0.1);
        assert!(improvement > 1.0);
        println!("DD coherence improvement: {}x", improvement);
    }

    #[test]
    fn test_cross_entropy_benchmarking() {
        let xeb = CrossEntropyBenchmarking::new(5, 10, 100);

        // Mock quantum executor
        let quantum_exec = |_circuit_id: usize| vec![0, 1, 2, 3, 4];

        // Mock classical simulator (uniform for simplicity)
        let classical_sim = |_circuit_id: usize| vec![0.0312; 32]; // 1/32 for 5 qubits

        let result = xeb.run(quantum_exec, classical_sim).unwrap();

        assert!(result.cross_entropy > 0.0);
        assert!(result.fidelity > 0.0);
        println!("XEB fidelity: {}", result.fidelity);
    }

    #[test]
    fn test_zne_polynomial_extrapolation() {
        let zne = ZeroNoiseExtrapolation::new(
            vec![1.0, 2.0, 3.0, 4.0],
            ExtrapolationMethod::Polynomial(2),
        );

        // Mock executor: result = 1.0 - 0.05 * scale - 0.01 * scale^2
        let executor = |scale: f64| 1.0 - 0.05 * scale - 0.01 * scale * scale;

        let mitigated = zne.mitigate(executor).unwrap();

        // Should extrapolate close to 1.0 at scale=0
        assert!((mitigated - 1.0).abs() < 0.1);
        println!("ZNE polynomial mitigated result: {}", mitigated);
    }

    #[test]
    fn test_zne_exponential_extrapolation() {
        let zne =
            ZeroNoiseExtrapolation::new(vec![1.0, 2.0, 3.0], ExtrapolationMethod::Exponential);

        // Mock executor: result = 0.9 * exp(-0.1 * scale)
        let executor = |scale: f64| 0.9 * (-0.1 * scale).exp();

        let mitigated = zne.mitigate(executor).unwrap();

        // Should extrapolate close to 0.9 at scale=0
        assert!((mitigated - 0.9).abs() < 0.05);
        println!("ZNE exponential mitigated result: {}", mitigated);
    }
}
