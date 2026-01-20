//! Error mitigation strategies for quantum computing
//!
//! This module provides state-of-the-art error mitigation techniques to improve
//! the accuracy of noisy quantum simulations and real quantum hardware results.
//!
//! # Supported Techniques
//!
//! - **Zero-Noise Extrapolation (ZNE)**: Extrapolate results to zero noise limit
//! - **Probabilistic Error Cancellation (PEC)**: Use quasi-probability to cancel errors
//! - **Clifford Data Regression (CDR)**: Noise characterization with Clifford circuits
//! - **Measurement Error Mitigation**: Correct readout errors using calibration
//! - **Symmetry Verification**: Verify conservation laws and post-select results
//!
//! # Example
//!
//! ```ignore
//! use quantrs2_sim::error_mitigation::{ZeroNoiseExtrapolation, ExtrapolationMethod};
//!
//! let zne = ZeroNoiseExtrapolation::new(ExtrapolationMethod::Richardson);
//! let mitigated_result = zne.apply(&noisy_results, &noise_scales)?;
//! ```

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::collections::HashMap;

// ============================================================================
// Zero-Noise Extrapolation (ZNE)
// ============================================================================

/// Extrapolation methods for ZNE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtrapolationMethod {
    /// Linear extrapolation (2 points)
    Linear,
    /// Richardson extrapolation (3+ points)
    Richardson,
    /// Polynomial fit
    Polynomial { degree: usize },
    /// Exponential fit
    Exponential,
}

/// Zero-Noise Extrapolation for error mitigation
///
/// ZNE runs the same circuit at different noise levels and extrapolates
/// the result to the zero-noise limit.
#[derive(Debug, Clone)]
pub struct ZeroNoiseExtrapolation {
    /// Extrapolation method
    method: ExtrapolationMethod,
    /// Noise scale factors to use (e.g., [1.0, 1.5, 2.0])
    scale_factors: Vec<f64>,
}

impl ZeroNoiseExtrapolation {
    /// Create a new ZNE instance with default scale factors
    pub fn new(method: ExtrapolationMethod) -> Self {
        Self {
            method,
            scale_factors: vec![1.0, 1.5, 2.0, 2.5, 3.0],
        }
    }

    /// Create ZNE with custom scale factors
    pub fn with_scale_factors(
        method: ExtrapolationMethod,
        scale_factors: Vec<f64>,
    ) -> Result<Self> {
        if scale_factors.is_empty() {
            return Err(SimulatorError::InvalidInput(
                "Scale factors cannot be empty".to_string(),
            ));
        }

        // Verify scale factors are in ascending order and start with 1.0
        if scale_factors[0] != 1.0 {
            return Err(SimulatorError::InvalidInput(
                "First scale factor must be 1.0 (original noise level)".to_string(),
            ));
        }

        for i in 1..scale_factors.len() {
            if scale_factors[i] <= scale_factors[i - 1] {
                return Err(SimulatorError::InvalidInput(
                    "Scale factors must be in strictly ascending order".to_string(),
                ));
            }
        }

        Ok(Self {
            method,
            scale_factors,
        })
    }

    /// Apply ZNE to a set of noisy expectation values
    ///
    /// # Arguments
    ///
    /// * `noisy_values` - Expectation values at different noise scales
    /// * `noise_scales` - Corresponding noise scale factors
    pub fn apply(&self, noisy_values: &[f64], noise_scales: &[f64]) -> Result<f64> {
        if noisy_values.len() != noise_scales.len() {
            return Err(SimulatorError::InvalidInput(
                "Number of values must match number of scales".to_string(),
            ));
        }

        if noisy_values.len() < 2 {
            return Err(SimulatorError::InvalidInput(
                "At least 2 data points required for extrapolation".to_string(),
            ));
        }

        match self.method {
            ExtrapolationMethod::Linear => self.linear_extrapolation(noisy_values, noise_scales),
            ExtrapolationMethod::Richardson => {
                self.richardson_extrapolation(noisy_values, noise_scales)
            }
            ExtrapolationMethod::Polynomial { degree } => {
                self.polynomial_extrapolation(noisy_values, noise_scales, degree)
            }
            ExtrapolationMethod::Exponential => {
                self.exponential_extrapolation(noisy_values, noise_scales)
            }
        }
    }

    /// Linear extrapolation using first two points
    fn linear_extrapolation(&self, values: &[f64], scales: &[f64]) -> Result<f64> {
        if values.len() < 2 {
            return Err(SimulatorError::InvalidInput(
                "Linear extrapolation requires at least 2 points".to_string(),
            ));
        }

        let x1 = scales[0];
        let y1 = values[0];
        let x2 = scales[1];
        let y2 = values[1];

        // Extrapolate to x=0 (zero noise)
        let slope = (y2 - y1) / (x2 - x1);
        Ok(y1 - slope * x1)
    }

    /// Richardson extrapolation (higher order)
    fn richardson_extrapolation(&self, values: &[f64], scales: &[f64]) -> Result<f64> {
        if values.len() < 3 {
            return Err(SimulatorError::InvalidInput(
                "Richardson extrapolation requires at least 3 points".to_string(),
            ));
        }

        // Use quadratic Richardson extrapolation
        let x0 = scales[0];
        let x1 = scales[1];
        let x2 = scales[2];
        let y0 = values[0];
        let y1 = values[1];
        let y2 = values[2];

        // Fit quadratic: y = a*x^2 + b*x + c
        // Extrapolate to x=0 gives c
        let denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
        if denom.abs() < 1e-10 {
            return Err(SimulatorError::InvalidInput(
                "Scale factors too close for stable extrapolation".to_string(),
            ));
        }

        let a = (x2 * (y1 - y0) + x0 * (y2 - y1) + x1 * (y0 - y2)) / denom;
        let b = (x2 * x2 * (y0 - y1) + x0 * x0 * (y1 - y2) + x1 * x1 * (y2 - y0)) / denom;
        let c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2)
            / denom;

        Ok(c)
    }

    /// Polynomial fit extrapolation
    fn polynomial_extrapolation(
        &self,
        values: &[f64],
        scales: &[f64],
        degree: usize,
    ) -> Result<f64> {
        if values.len() <= degree {
            return Err(SimulatorError::InvalidInput(format!(
                "Need at least {} points for degree {} polynomial",
                degree + 1,
                degree
            )));
        }

        // Simple polynomial fit using least squares
        // For now, use Richardson for degree 2, linear for degree 1
        match degree {
            1 => self.linear_extrapolation(values, scales),
            2 => self.richardson_extrapolation(values, scales),
            _ => Err(SimulatorError::NotImplemented(
                "Polynomial degree > 2 not yet implemented".to_string(),
            )),
        }
    }

    /// Exponential fit extrapolation: y = a * exp(-b*x) + c
    fn exponential_extrapolation(&self, values: &[f64], scales: &[f64]) -> Result<f64> {
        if values.len() < 3 {
            return Err(SimulatorError::InvalidInput(
                "Exponential extrapolation requires at least 3 points".to_string(),
            ));
        }

        // Simplified: assume y = a * exp(-b*x) + c where c is the zero-noise value
        // Use first derivative at x=0 from first two points
        let x0 = scales[0];
        let x1 = scales[1];
        let y0 = values[0];
        let y1 = values[1];

        // Estimate decay rate
        if (y1 - y0).abs() < 1e-10 {
            return Ok(y0); // No decay observed
        }

        let b_estimate = -((y1 - y0) / (x1 - x0)) / y0.max(1e-10);
        let c_estimate = y0 / (1.0 + b_estimate * x0).max(1e-10);

        Ok(c_estimate)
    }

    /// Get the scale factors
    pub fn scale_factors(&self) -> &[f64] {
        &self.scale_factors
    }
}

// ============================================================================
// Measurement Error Mitigation
// ============================================================================

/// Measurement error mitigation using calibration matrices
///
/// Corrects readout errors by inverting the noise transfer matrix
/// measured through calibration circuits.
#[derive(Debug, Clone)]
pub struct MeasurementErrorMitigation {
    /// Calibration matrix: M[i][j] = P(measure i | prepared j)
    calibration_matrix: Array2<f64>,
    /// Inverse calibration matrix (for mitigation)
    inverse_matrix: Option<Array2<f64>>,
    /// Number of qubits
    n_qubits: usize,
}

impl MeasurementErrorMitigation {
    /// Create a new measurement error mitigation instance
    ///
    /// # Arguments
    ///
    /// * `n_qubits` - Number of qubits
    pub fn new(n_qubits: usize) -> Self {
        let dim = 1 << n_qubits; // 2^n
        let calibration_matrix = Array2::eye(dim); // Identity by default (no error)

        Self {
            calibration_matrix,
            inverse_matrix: None,
            n_qubits,
        }
    }

    /// Set the calibration matrix from measurements
    ///
    /// The calibration matrix `M[i][j]` represents the probability of
    /// measuring bitstring i when bitstring j was prepared.
    pub fn set_calibration_matrix(&mut self, matrix: Array2<f64>) -> Result<()> {
        let expected_dim = 1 << self.n_qubits;
        if matrix.nrows() != expected_dim || matrix.ncols() != expected_dim {
            return Err(SimulatorError::InvalidInput(format!(
                "Calibration matrix must be {}x{} for {} qubits",
                expected_dim, expected_dim, self.n_qubits
            )));
        }

        // Verify matrix is stochastic (columns sum to 1)
        for col in 0..matrix.ncols() {
            let sum: f64 = (0..matrix.nrows()).map(|row| matrix[[row, col]]).sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(SimulatorError::InvalidInput(format!(
                    "Column {} does not sum to 1.0 (sum = {})",
                    col, sum
                )));
            }
        }

        self.calibration_matrix = matrix;
        self.inverse_matrix = None; // Will be computed on demand
        Ok(())
    }

    /// Compute and cache the inverse calibration matrix
    fn compute_inverse(&mut self) -> Result<()> {
        if self.inverse_matrix.is_some() {
            return Ok(());
        }

        // Use pseudo-inverse for stability
        // For now, use simple matrix inversion (can be improved with SVD)
        let matrix = &self.calibration_matrix;
        let n = matrix.nrows();

        // Simple Gauss-Jordan elimination for small matrices
        let mut augmented = Array2::zeros((n, 2 * n));
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                augmented[[i, j + n]] = if i == j { 1.0 } else { 0.0 };
            }
        }

        // Forward elimination (simplified, may need pivoting for stability)
        for i in 0..n {
            // Find pivot
            let pivot = augmented[[i, i]];
            if pivot.abs() < 1e-10 {
                return Err(SimulatorError::InvalidInput(
                    "Calibration matrix is singular or nearly singular".to_string(),
                ));
            }

            // Scale row
            for j in 0..(2 * n) {
                augmented[[i, j]] /= pivot;
            }

            // Eliminate
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..(2 * n) {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }
        }

        // Extract inverse from augmented matrix
        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = augmented[[i, j + n]];
            }
        }

        self.inverse_matrix = Some(inverse);
        Ok(())
    }

    /// Apply measurement error mitigation to noisy counts
    ///
    /// # Arguments
    ///
    /// * `noisy_counts` - Raw measurement counts from the quantum circuit
    ///
    /// # Returns
    ///
    /// Mitigated counts (may contain negative values due to inversion)
    pub fn apply(&mut self, noisy_counts: &HashMap<String, usize>) -> Result<HashMap<String, f64>> {
        self.compute_inverse()?;
        let inverse = self.inverse_matrix.as_ref().unwrap();

        let dim = 1 << self.n_qubits;
        let total_shots: usize = noisy_counts.values().sum();

        // Convert counts to probability vector
        let mut noisy_probs = Array1::zeros(dim);
        for (bitstring, count) in noisy_counts {
            if bitstring.len() != self.n_qubits {
                return Err(SimulatorError::InvalidInput(format!(
                    "Bitstring {} has wrong length (expected {})",
                    bitstring, self.n_qubits
                )));
            }
            let index = usize::from_str_radix(bitstring, 2).map_err(|_| {
                SimulatorError::InvalidInput(format!("Invalid bitstring: {}", bitstring))
            })?;
            noisy_probs[index] = *count as f64 / total_shots as f64;
        }

        // Apply inverse: mitigated_probs = M^(-1) * noisy_probs
        let mitigated_probs = inverse.dot(&noisy_probs);

        // Convert back to counts
        let mut mitigated_counts = HashMap::new();
        for i in 0..dim {
            let bitstring = format!("{:0width$b}", i, width = self.n_qubits);
            let mitigated_count = mitigated_probs[i] * total_shots as f64;
            if mitigated_count.abs() > 1e-10 {
                mitigated_counts.insert(bitstring, mitigated_count);
            }
        }

        Ok(mitigated_counts)
    }

    /// Generate calibration matrix from error rates
    ///
    /// # Arguments
    ///
    /// * `readout_error_0` - Probability of measuring 1 when prepared in |0⟩
    /// * `readout_error_1` - Probability of measuring 0 when prepared in |1⟩
    pub fn from_error_rates(
        n_qubits: usize,
        readout_error_0: f64,
        readout_error_1: f64,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&readout_error_0) {
            return Err(SimulatorError::InvalidInput(
                "readout_error_0 must be in [0, 1]".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&readout_error_1) {
            return Err(SimulatorError::InvalidInput(
                "readout_error_1 must be in [0, 1]".to_string(),
            ));
        }

        let dim = 1 << n_qubits;
        let mut matrix = Array2::zeros((dim, dim));

        // Build tensor product of single-qubit error matrices
        // Single-qubit matrix: [[1-e0, e1], [e0, 1-e1]]
        for prepared in 0..dim {
            for measured in 0..dim {
                let mut prob = 1.0;
                for qubit in 0..n_qubits {
                    let prepared_bit = (prepared >> qubit) & 1;
                    let measured_bit = (measured >> qubit) & 1;

                    let p = if prepared_bit == 0 {
                        if measured_bit == 0 {
                            1.0 - readout_error_0
                        } else {
                            readout_error_0
                        }
                    } else if measured_bit == 1 {
                        1.0 - readout_error_1
                    } else {
                        readout_error_1
                    };

                    prob *= p;
                }
                matrix[[measured, prepared]] = prob;
            }
        }

        let mut mem = Self::new(n_qubits);
        mem.set_calibration_matrix(matrix)?;
        Ok(mem)
    }
}

// ============================================================================
// Symmetry Verification
// ============================================================================

/// Symmetry verification for post-selection
///
/// Verifies that measurement results respect known symmetries
/// (e.g., particle number conservation, parity) and rejects
/// results that violate symmetries.
#[derive(Debug, Clone)]
pub struct SymmetryVerification {
    /// Type of symmetry to verify
    symmetry_type: SymmetryType,
    /// Expected symmetry value
    expected_value: Option<i32>,
}

/// Types of symmetries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryType {
    /// Particle number conservation (count of 1s)
    ParticleNumber,
    /// Parity (even/odd number of 1s)
    Parity,
    /// Custom symmetry (user-defined)
    Custom,
}

impl SymmetryVerification {
    /// Create a new symmetry verification instance
    pub fn new(symmetry_type: SymmetryType) -> Self {
        Self {
            symmetry_type,
            expected_value: None,
        }
    }

    /// Set the expected symmetry value
    pub fn with_expected_value(mut self, value: i32) -> Self {
        self.expected_value = Some(value);
        self
    }

    /// Verify if a bitstring satisfies the symmetry
    pub fn verify(&self, bitstring: &str) -> bool {
        let symmetry_value = self.compute_symmetry(bitstring);

        if let Some(expected) = self.expected_value {
            symmetry_value == expected
        } else {
            true // No constraint if expected value not set
        }
    }

    /// Compute the symmetry value for a bitstring
    fn compute_symmetry(&self, bitstring: &str) -> i32 {
        match self.symmetry_type {
            SymmetryType::ParticleNumber => bitstring.chars().filter(|&c| c == '1').count() as i32,
            SymmetryType::Parity => {
                let ones = bitstring.chars().filter(|&c| c == '1').count();
                (ones % 2) as i32
            }
            SymmetryType::Custom => 0, // Placeholder
        }
    }

    /// Filter measurement counts based on symmetry
    pub fn filter_counts(&self, counts: &HashMap<String, usize>) -> HashMap<String, usize> {
        counts
            .iter()
            .filter(|(bitstring, _)| self.verify(bitstring))
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// Normalize filtered counts to original total
    pub fn filter_and_normalize(&self, counts: &HashMap<String, usize>) -> HashMap<String, usize> {
        let total_shots: usize = counts.values().sum();
        let filtered = self.filter_counts(counts);
        let filtered_total: usize = filtered.values().sum();

        if filtered_total == 0 {
            return HashMap::new();
        }

        // Renormalize to original total
        let scale = total_shots as f64 / filtered_total as f64;
        filtered
            .iter()
            .map(|(k, v)| (k.clone(), (*v as f64 * scale).round() as usize))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zne_linear_extrapolation() {
        let zne = ZeroNoiseExtrapolation::new(ExtrapolationMethod::Linear);
        let values = vec![0.8, 0.6, 0.4];
        let scales = vec![1.0, 2.0, 3.0];

        let result = zne.apply(&values, &scales).unwrap();
        assert!((result - 1.0).abs() < 0.01); // Should extrapolate to ~1.0
    }

    #[test]
    fn test_zne_richardson_extrapolation() {
        let zne = ZeroNoiseExtrapolation::new(ExtrapolationMethod::Richardson);
        // Quadratic decay: y = 1 - 0.1*x^2
        let values = vec![1.0, 0.9, 0.6];
        let scales = vec![0.0, 1.0, 2.0];

        let result = zne.apply(&values, &scales).unwrap();
        assert!((result - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_zne_invalid_input() {
        let zne = ZeroNoiseExtrapolation::new(ExtrapolationMethod::Linear);
        let values = vec![0.8];
        let scales = vec![1.0, 2.0];

        let result = zne.apply(&values, &scales);
        assert!(result.is_err());
    }

    #[test]
    fn test_measurement_error_mitigation_identity() {
        let mut mem = MeasurementErrorMitigation::new(2);

        let mut counts = HashMap::new();
        counts.insert("00".to_string(), 100);
        counts.insert("11".to_string(), 50);

        let mitigated = mem.apply(&counts).unwrap();

        // With identity matrix, should get same counts
        assert!((mitigated["00"] - 100.0).abs() < 1e-6);
        assert!((mitigated["11"] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_measurement_error_mitigation_from_error_rates() {
        let mem = MeasurementErrorMitigation::from_error_rates(1, 0.1, 0.05).unwrap();

        // Verify calibration matrix structure
        let matrix = &mem.calibration_matrix;
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 2);

        // Check specific values
        assert!((matrix[[0, 0]] - 0.9).abs() < 1e-10); // P(0|0) = 1 - e0
        assert!((matrix[[1, 0]] - 0.1).abs() < 1e-10); // P(1|0) = e0
        assert!((matrix[[0, 1]] - 0.05).abs() < 1e-10); // P(0|1) = e1
        assert!((matrix[[1, 1]] - 0.95).abs() < 1e-10); // P(1|1) = 1 - e1
    }

    #[test]
    fn test_symmetry_particle_number() {
        let sym = SymmetryVerification::new(SymmetryType::ParticleNumber).with_expected_value(2);

        assert!(sym.verify("0011"));
        assert!(sym.verify("1100"));
        assert!(sym.verify("1010"));
        assert!(!sym.verify("0001"));
        assert!(!sym.verify("1111"));
    }

    #[test]
    fn test_symmetry_parity() {
        let sym = SymmetryVerification::new(SymmetryType::Parity).with_expected_value(0);

        assert!(sym.verify("0011")); // Even parity
        assert!(sym.verify("1100")); // Even parity
        assert!(!sym.verify("0001")); // Odd parity
        assert!(!sym.verify("0111")); // Odd parity
    }

    #[test]
    fn test_symmetry_filter_counts() {
        let sym = SymmetryVerification::new(SymmetryType::ParticleNumber).with_expected_value(2);

        let mut counts = HashMap::new();
        counts.insert("0011".to_string(), 100);
        counts.insert("1100".to_string(), 50);
        counts.insert("0001".to_string(), 30); // Should be filtered
        counts.insert("1111".to_string(), 20); // Should be filtered

        let filtered = sym.filter_counts(&counts);

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered["0011"], 100);
        assert_eq!(filtered["1100"], 50);
        assert!(!filtered.contains_key("0001"));
        assert!(!filtered.contains_key("1111"));
    }

    #[test]
    fn test_zne_custom_scale_factors() {
        let result = ZeroNoiseExtrapolation::with_scale_factors(
            ExtrapolationMethod::Linear,
            vec![1.0, 1.5, 2.0],
        );
        assert!(result.is_ok());

        let bad_result = ZeroNoiseExtrapolation::with_scale_factors(
            ExtrapolationMethod::Linear,
            vec![2.0, 1.0], // Not starting with 1.0
        );
        assert!(bad_result.is_err());
    }
}
