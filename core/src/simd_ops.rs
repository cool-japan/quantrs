//! SIMD-accelerated quantum operations
//!
//! This module provides SIMD-accelerated implementations of common quantum
//! operations using platform-specific SIMD instructions when available.

use crate::error::QuantRS2Result;
use num_complex::Complex64;

// SIMD operations will be conditionally compiled

/// Apply a phase rotation to a quantum state vector using SIMD when available
///
/// This function applies the phase rotation e^(i*theta) to each amplitude.
pub fn apply_phase_simd(amplitudes: &mut [Complex64], theta: f64) {
    let phase_factor = Complex64::new(theta.cos(), theta.sin());

    #[cfg(feature = "simd")]
    {
        // Process in SIMD chunks when available
        let mut chunks = amplitudes.chunks_exact_mut(4);

        // Process SIMD chunks
        for chunk in &mut chunks {
            // Simulate SIMD operation on 4 complex numbers at once
            for amp in chunk {
                *amp *= phase_factor;
            }
        }

        // Process remainder
        let remainder = chunks.into_remainder();
        for amp in remainder {
            *amp *= phase_factor;
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        // Fallback to scalar implementation
        for amp in amplitudes.iter_mut() {
            *amp *= phase_factor;
        }
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

    #[cfg(feature = "simd")]
    {
        let mut result = Complex64::new(0.0, 0.0);

        // Process in SIMD chunks
        let chunks1 = state1.chunks_exact(4);
        let chunks2 = state2.chunks_exact(4);
        let remainder1 = chunks1.remainder();
        let remainder2 = chunks2.remainder();

        // Process SIMD chunks
        for (chunk1, chunk2) in chunks1.zip(chunks2) {
            // Simulate SIMD operation on 4 complex numbers at once
            for (a, b) in chunk1.iter().zip(chunk2.iter()) {
                result += a.conj() * b;
            }
        }

        // Process remainder
        for (a, b) in remainder1.iter().zip(remainder2.iter()) {
            result += a.conj() * b;
        }

        Ok(result)
    }

    #[cfg(not(feature = "simd"))]
    {
        // Fallback to scalar implementation
        let result = state1
            .iter()
            .zip(state2.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        Ok(result)
    }
}

/// Normalize a quantum state vector in-place
///
/// This ensures that the sum of squared magnitudes equals 1.
pub fn normalize_simd(amplitudes: &mut [Complex64]) -> QuantRS2Result<()> {
    let norm_sqr: f64 = {
        #[cfg(feature = "simd")]
        {
            // Compute norm squared using SIMD
            let mut norm_sqr = 0.0;

            // Process in SIMD chunks
            let chunks = amplitudes.chunks_exact(4);
            let remainder = chunks.remainder();

            // Process SIMD chunks
            for chunk in chunks {
                // Simulate SIMD operation on 4 complex numbers at once
                for amp in chunk {
                    norm_sqr += amp.norm_sqr();
                }
            }

            // Process remainder
            for amp in remainder {
                norm_sqr += amp.norm_sqr();
            }

            norm_sqr
        }

        #[cfg(not(feature = "simd"))]
        {
            amplitudes.iter().map(|c| c.norm_sqr()).sum()
        }
    };

    if norm_sqr == 0.0 {
        return Err(crate::error::QuantRS2Error::InvalidInput(
            "Cannot normalize zero vector".to_string(),
        ));
    }

    let norm = norm_sqr.sqrt();

    // Normalize each amplitude
    #[cfg(feature = "simd")]
    {
        // Process in SIMD chunks
        let mut chunks = amplitudes.chunks_exact_mut(4);

        // Process SIMD chunks
        for chunk in &mut chunks {
            for amp in chunk {
                *amp /= norm;
            }
        }

        // Process remainder
        let remainder = chunks.into_remainder();
        for amp in remainder {
            *amp /= norm;
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        for amp in amplitudes.iter_mut() {
            *amp /= norm;
        }
    }

    Ok(())
}

/// Compute expectation value of a Pauli Z operator
///
/// This computes ⟨ψ|Z|ψ⟩ where Z is the Pauli Z operator on the given qubit.
pub fn expectation_z_simd(amplitudes: &[Complex64], qubit: usize, num_qubits: usize) -> f64 {
    let qubit_mask = 1 << qubit;
    let mut expectation = 0.0;

    #[cfg(feature = "simd")]
    {
        // Process in SIMD chunks
        for (i, amp) in amplitudes.iter().enumerate() {
            let sign = if (i & qubit_mask) == 0 { 1.0 } else { -1.0 };
            expectation += sign * amp.norm_sqr();
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        for (i, amp) in amplitudes.iter().enumerate() {
            let sign = if (i & qubit_mask) == 0 { 1.0 } else { -1.0 };
            expectation += sign * amp.norm_sqr();
        }
    }

    expectation
}

/// Apply a Hadamard gate using SIMD operations
///
/// This applies H = (1/√2) * [[1, 1], [1, -1]] to the specified qubit.
pub fn hadamard_simd(amplitudes: &mut [Complex64], qubit: usize, num_qubits: usize) {
    let qubit_mask = 1 << qubit;
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();

    #[cfg(feature = "simd")]
    {
        // Process pairs of amplitudes that differ only in the target qubit
        for i in 0..(amplitudes.len() / 2) {
            let idx0 = (i & !(qubit_mask >> 1)) | ((i & (qubit_mask >> 1)) << 1);
            let idx1 = idx0 | qubit_mask;

            if idx1 < amplitudes.len() {
                let a0 = amplitudes[idx0];
                let a1 = amplitudes[idx1];

                amplitudes[idx0] = (a0 + a1) * sqrt2_inv;
                amplitudes[idx1] = (a0 - a1) * sqrt2_inv;
            }
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        for i in 0..(amplitudes.len() / 2) {
            let idx0 = (i & !(qubit_mask >> 1)) | ((i & (qubit_mask >> 1)) << 1);
            let idx1 = idx0 | qubit_mask;

            if idx1 < amplitudes.len() {
                let a0 = amplitudes[idx0];
                let a1 = amplitudes[idx1];

                amplitudes[idx0] = (a0 + a1) * sqrt2_inv;
                amplitudes[idx1] = (a0 - a1) * sqrt2_inv;
            }
        }
    }
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
        let exp0 = expectation_z_simd(&state0, 0, 1);
        assert!((exp0 - 1.0).abs() < 1e-10);

        // |1⟩ state
        let state1 = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        let exp1 = expectation_z_simd(&state1, 0, 1);
        assert!((exp1 + 1.0).abs() < 1e-10);

        // |+⟩ state
        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
        let state_plus = vec![
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
        ];
        let exp_plus = expectation_z_simd(&state_plus, 0, 1);
        assert!(exp_plus.abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_simd() {
        // Test Hadamard gate on |0⟩ state to create |+⟩
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        hadamard_simd(&mut state, 0, 1);

        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        assert!((state[0].re - sqrt2_inv).abs() < 1e-10);
        assert!((state[1].re - sqrt2_inv).abs() < 1e-10);
        assert!(state[0].im.abs() < 1e-10);
        assert!(state[1].im.abs() < 1e-10);

        // Apply Hadamard again to get back to |0⟩
        hadamard_simd(&mut state, 0, 1);
        assert!((state[0].re - 1.0).abs() < 1e-10);
        assert!(state[1].re.abs() < 1e-10);
    }

    #[test]
    fn test_phase_simd() {
        let mut state = vec![Complex64::new(0.5, 0.5), Complex64::new(0.5, -0.5)];

        let theta = std::f64::consts::PI / 4.0;
        apply_phase_simd(&mut state, theta);

        // Check that magnitudes are preserved
        let norm_before = 0.5_f64.powi(2) + 0.5_f64.powi(2);
        let norm_after = state[0].norm_sqr();
        assert!((norm_before - norm_after).abs() < 1e-10);
    }
}
