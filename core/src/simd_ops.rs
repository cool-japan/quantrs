//! SIMD-accelerated quantum operations
//!
//! This module provides SIMD-accelerated implementations of common quantum
//! operations using SciRS2's SIMD utilities.

// Note: SIMD operations would use scirs2_core::simd when available

use crate::error::QuantRS2Result;
use num_complex::Complex64;

/// Apply a phase rotation to a quantum state vector using SIMD when available
///
/// This function applies the phase rotation e^(i*theta) to each amplitude.
pub fn apply_phase_simd(amplitudes: &mut [Complex64], theta: f64) {
    let phase_factor = Complex64::new(theta.cos(), theta.sin());

    // Note: Full SIMD implementation would require complex number SIMD support
    // For now, we apply the operation element-wise
    for amp in amplitudes.iter_mut() {
        *amp *= phase_factor;
    }
}

/// Compute the inner product of two quantum state vectors
///
/// This computes ⟨ψ|φ⟩ = Σ conj(ψ[i]) * φ[i]
pub fn inner_product(state1: &[Complex64], state2: &[Complex64]) -> QuantRS2Result<Complex64> {
    if state1.len() != state2.len() {
        return Err(crate::error::QuantRS2Error::InvalidInput(
            "State vectors must have the same length".to_string(),
        ));
    }

    // Compute inner product
    // Note: A full SIMD implementation would vectorize this operation
    let result = state1
        .iter()
        .zip(state2.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();

    Ok(result)
}

/// Normalize a quantum state vector in-place
///
/// This ensures that the sum of squared magnitudes equals 1.
pub fn normalize_simd(amplitudes: &mut [Complex64]) -> QuantRS2Result<()> {
    // Compute norm squared
    let norm_sqr: f64 = amplitudes.iter().map(|c| c.norm_sqr()).sum();

    if norm_sqr == 0.0 {
        return Err(crate::error::QuantRS2Error::InvalidInput(
            "Cannot normalize zero vector".to_string(),
        ));
    }

    let norm = norm_sqr.sqrt();

    // Normalize each amplitude
    for amp in amplitudes.iter_mut() {
        *amp /= norm;
    }

    Ok(())
}

/// Apply a controlled phase rotation
///
/// This applies a phase rotation to amplitudes where the control qubit is |1⟩.
pub fn controlled_phase_simd(
    amplitudes: &mut [Complex64],
    control_qubit: usize,
    target_qubit: usize,
    theta: f64,
) -> QuantRS2Result<()> {
    let num_qubits = (amplitudes.len() as f64).log2() as usize;

    if control_qubit >= num_qubits || target_qubit >= num_qubits {
        return Err(crate::error::QuantRS2Error::InvalidInput(
            "Qubit index out of range".to_string(),
        ));
    }

    let phase_factor = Complex64::new(theta.cos(), theta.sin());
    let control_mask = 1 << control_qubit;
    let target_mask = 1 << target_qubit;

    // Apply phase to states where both control and target are |1⟩
    for (idx, amp) in amplitudes.iter_mut().enumerate() {
        if (idx & control_mask) != 0 && (idx & target_mask) != 0 {
            *amp *= phase_factor;
        }
    }

    Ok(())
}

/// Compute expectation value of a Pauli-Z measurement
///
/// This computes ⟨ψ|Z|ψ⟩ for a single qubit.
pub fn expectation_z_simd(amplitudes: &[Complex64], qubit: usize) -> QuantRS2Result<f64> {
    let num_qubits = (amplitudes.len() as f64).log2() as usize;

    if qubit >= num_qubits {
        return Err(crate::error::QuantRS2Error::InvalidInput(
            "Qubit index out of range".to_string(),
        ));
    }

    let qubit_mask = 1 << qubit;
    let mut expectation = 0.0;

    // Sum probabilities with appropriate signs
    for (idx, amp) in amplitudes.iter().enumerate() {
        let sign = if (idx & qubit_mask) == 0 { 1.0 } else { -1.0 };
        expectation += sign * amp.norm_sqr();
    }

    Ok(expectation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_simd() {
        let mut state = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, -1.0),
        ];

        normalize_simd(&mut state).unwrap();

        let norm_sqr: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm_sqr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inner_product() {
        let state1 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let state2 = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];

        let result = inner_product(&state1, &state2).unwrap();
        assert_eq!(result, Complex64::new(0.0, 0.0));
    }

    #[test]
    fn test_expectation_z() {
        // |0⟩ state
        let state0 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let exp0 = expectation_z_simd(&state0, 0).unwrap();
        assert!((exp0 - 1.0).abs() < 1e-10);

        // |1⟩ state
        let state1 = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        let exp1 = expectation_z_simd(&state1, 0).unwrap();
        assert!((exp1 + 1.0).abs() < 1e-10);

        // |+⟩ state
        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
        let state_plus = vec![
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
        ];
        let exp_plus = expectation_z_simd(&state_plus, 0).unwrap();
        assert!(exp_plus.abs() < 1e-10);
    }
}
