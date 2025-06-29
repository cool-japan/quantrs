//! SIMD-accelerated quantum operations
//!
//! This module provides SIMD-accelerated implementations of common quantum
//! operations using SciRS2's unified SIMD operations.

use crate::error::QuantRS2Result;
use num_complex::Complex64;
// use scirs2_core::simd_ops::SimdUnifiedOps;
use crate::simd_ops_stubs::{SimdComplex64, SimdF64};
use ndarray::{ArrayView1, ArrayViewMut1};

// All SIMD operations now use SciRS2's unified trait

/// Apply a phase rotation to a quantum state vector using SIMD when available
///
/// This function applies the phase rotation e^(i*theta) to each amplitude.
pub fn apply_phase_simd(amplitudes: &mut [Complex64], theta: f64) {
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();

    // Extract real and imaginary parts for SIMD operations
    let len = amplitudes.len();
    let mut real_parts = Vec::with_capacity(len);
    let mut imag_parts = Vec::with_capacity(len);

    for amp in amplitudes.iter() {
        real_parts.push(amp.re);
        imag_parts.push(amp.im);
    }

    // Apply phase rotation using SIMD
    // (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    let mut new_real = vec![0.0; len];
    let mut new_imag = vec![0.0; len];

    // Compute real part: a*cos - b*sin
    let real_view = ArrayView1::from(&real_parts);
    let imag_view = ArrayView1::from(&imag_parts);
    let mut new_real_view = ArrayViewMut1::from(&mut new_real);
    let mut new_imag_view = ArrayViewMut1::from(&mut new_imag);

    // Use SciRS2 SIMD operations
    let real_cos = <f64 as SimdF64>::simd_scalar_mul(&real_view, cos_theta);
    let imag_sin = <f64 as SimdF64>::simd_scalar_mul(&imag_view, sin_theta);
    let new_real_arr = <f64 as SimdF64>::simd_sub_arrays(&real_cos.view(), &imag_sin.view());

    // Compute imaginary part: a*sin + b*cos
    let real_sin = <f64 as SimdF64>::simd_scalar_mul(&real_view, sin_theta);
    let imag_cos = <f64 as SimdF64>::simd_scalar_mul(&imag_view, cos_theta);
    let new_imag_arr = <f64 as SimdF64>::simd_add_arrays(&real_sin.view(), &imag_cos.view());

    // Copy results
    new_real_view.assign(&new_real_arr);
    new_imag_view.assign(&new_imag_arr);

    // Reconstruct complex numbers
    for (i, amp) in amplitudes.iter_mut().enumerate() {
        *amp = Complex64::new(new_real[i], new_imag[i]);
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

    let len = state1.len();

    // Extract real and imaginary parts
    let mut state1_real = Vec::with_capacity(len);
    let mut state1_imag = Vec::with_capacity(len);
    let mut state2_real = Vec::with_capacity(len);
    let mut state2_imag = Vec::with_capacity(len);

    for (a, b) in state1.iter().zip(state2.iter()) {
        state1_real.push(a.re);
        state1_imag.push(a.im);
        state2_real.push(b.re);
        state2_imag.push(b.im);
    }

    // Compute inner product using SIMD
    // ⟨ψ|φ⟩ = Σ (a_r - i*a_i) * (b_r + i*b_i)
    //       = Σ (a_r*b_r + a_i*b_i) + i*(a_r*b_i - a_i*b_r)

    let state1_real_view = ArrayView1::from(&state1_real);
    let state1_imag_view = ArrayView1::from(&state1_imag);
    let state2_real_view = ArrayView1::from(&state2_real);
    let state2_imag_view = ArrayView1::from(&state2_imag);

    // Compute real part: a_r*b_r + a_i*b_i
    let rr_product = <f64 as SimdF64>::simd_mul_arrays(&state1_real_view, &state2_real_view);
    let ii_product = <f64 as SimdF64>::simd_mul_arrays(&state1_imag_view, &state2_imag_view);
    let real_sum = <f64 as SimdF64>::simd_add_arrays(&rr_product.view(), &ii_product.view());
    let real_part = <f64 as SimdF64>::simd_sum_array(&real_sum.view());

    // Compute imaginary part: a_r*b_i - a_i*b_r
    let ri_product = <f64 as SimdF64>::simd_mul_arrays(&state1_real_view, &state2_imag_view);
    let ir_product = <f64 as SimdF64>::simd_mul_arrays(&state1_imag_view, &state2_real_view);
    let imag_diff = <f64 as SimdF64>::simd_sub_arrays(&ri_product.view(), &ir_product.view());
    let imag_part = <f64 as SimdF64>::simd_sum_array(&imag_diff.view());

    Ok(Complex64::new(real_part, -imag_part)) // Negative because of conjugate
}

/// Normalize a quantum state vector in-place
///
/// This ensures that the sum of squared magnitudes equals 1.
pub fn normalize_simd(amplitudes: &mut [Complex64]) -> QuantRS2Result<()> {
    let len = amplitudes.len();

    // Extract real and imaginary parts
    let mut real_parts = Vec::with_capacity(len);
    let mut imag_parts = Vec::with_capacity(len);

    for amp in amplitudes.iter() {
        real_parts.push(amp.re);
        imag_parts.push(amp.im);
    }

    // Compute norm squared using SIMD: Σ(re² + im²)
    let real_view = ArrayView1::from(&real_parts);
    let imag_view = ArrayView1::from(&imag_parts);

    let real_squared = <f64 as SimdF64>::simd_mul_arrays(&real_view, &real_view);
    let imag_squared = <f64 as SimdF64>::simd_mul_arrays(&imag_view, &imag_view);
    let sum_squared = <f64 as SimdF64>::simd_add_arrays(&real_squared.view(), &imag_squared.view());

    let norm_sqr = <f64 as SimdF64>::simd_sum_array(&sum_squared.view());

    if norm_sqr == 0.0 {
        return Err(crate::error::QuantRS2Error::InvalidInput(
            "Cannot normalize zero vector".to_string(),
        ));
    }

    let norm = norm_sqr.sqrt();
    let norm_inv = 1.0 / norm;

    // Normalize using SIMD
    let mut normalized_real = vec![0.0; len];
    let mut normalized_imag = vec![0.0; len];
    let mut normalized_real_view = ArrayViewMut1::from(&mut normalized_real);
    let mut normalized_imag_view = ArrayViewMut1::from(&mut normalized_imag);

    let normalized_real_arr = <f64 as SimdF64>::simd_scalar_mul(&real_view, norm_inv);
    let normalized_imag_arr = <f64 as SimdF64>::simd_scalar_mul(&imag_view, norm_inv);
    normalized_real_view.assign(&normalized_real_arr);
    normalized_imag_view.assign(&normalized_imag_arr);

    // Reconstruct normalized complex numbers
    for (i, amp) in amplitudes.iter_mut().enumerate() {
        *amp = Complex64::new(normalized_real[i], normalized_imag[i]);
    }

    Ok(())
}

/// Compute expectation value of a Pauli Z operator
///
/// This computes ⟨ψ|Z|ψ⟩ where Z is the Pauli Z operator on the given qubit.
pub fn expectation_z_simd(amplitudes: &[Complex64], qubit: usize, _num_qubits: usize) -> f64 {
    let qubit_mask = 1 << qubit;
    let len = amplitudes.len();

    // Compute norm squared and signs
    let mut norm_sqrs = Vec::with_capacity(len);
    let mut signs = Vec::with_capacity(len);

    for (i, amp) in amplitudes.iter().enumerate() {
        norm_sqrs.push(amp.re * amp.re + amp.im * amp.im);
        signs.push(if (i & qubit_mask) == 0 { 1.0 } else { -1.0 });
    }

    // Compute expectation using SIMD
    let norm_view = ArrayView1::from(&norm_sqrs);
    let sign_view = ArrayView1::from(&signs);

    let weighted_arr = <f64 as SimdF64>::simd_mul_arrays(&norm_view, &sign_view);
    <f64 as SimdF64>::simd_sum_array(&weighted_arr.view())
}

/// Apply a Hadamard gate using SIMD operations
///
/// This applies H = (1/√2) * [[1, 1], [1, -1]] to the specified qubit.
pub fn hadamard_simd(amplitudes: &mut [Complex64], qubit: usize, _num_qubits: usize) {
    let qubit_mask = 1 << qubit;
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();

    // Collect pairs of indices that need to be processed
    let mut idx0_list = Vec::new();
    let mut idx1_list = Vec::new();

    for i in 0..(amplitudes.len() / 2) {
        let idx0 = (i & !(qubit_mask >> 1)) | ((i & (qubit_mask >> 1)) << 1);
        let idx1 = idx0 | qubit_mask;

        if idx1 < amplitudes.len() {
            idx0_list.push(idx0);
            idx1_list.push(idx1);
        }
    }

    let pair_count = idx0_list.len();
    if pair_count == 0 {
        return;
    }

    // Extract amplitude pairs
    let mut a0_real = Vec::with_capacity(pair_count);
    let mut a0_imag = Vec::with_capacity(pair_count);
    let mut a1_real = Vec::with_capacity(pair_count);
    let mut a1_imag = Vec::with_capacity(pair_count);

    for i in 0..pair_count {
        let a0 = amplitudes[idx0_list[i]];
        let a1 = amplitudes[idx1_list[i]];
        a0_real.push(a0.re);
        a0_imag.push(a0.im);
        a1_real.push(a1.re);
        a1_imag.push(a1.im);
    }

    // Apply Hadamard using SIMD
    let a0_real_view = ArrayView1::from(&a0_real);
    let a0_imag_view = ArrayView1::from(&a0_imag);
    let a1_real_view = ArrayView1::from(&a1_real);
    let a1_imag_view = ArrayView1::from(&a1_imag);

    let mut new_a0_real = vec![0.0; pair_count];
    let mut new_a0_imag = vec![0.0; pair_count];
    let mut new_a1_real = vec![0.0; pair_count];
    let mut new_a1_imag = vec![0.0; pair_count];

    let mut new_a0_real_view = ArrayViewMut1::from(&mut new_a0_real);
    let mut new_a0_imag_view = ArrayViewMut1::from(&mut new_a0_imag);
    let mut new_a1_real_view = ArrayViewMut1::from(&mut new_a1_real);
    let mut new_a1_imag_view = ArrayViewMut1::from(&mut new_a1_imag);

    // Compute (a0 + a1) * sqrt2_inv
    let real_sum = <f64 as SimdF64>::simd_add_arrays(&a0_real_view, &a1_real_view);
    let new_a0_real_arr = <f64 as SimdF64>::simd_scalar_mul(&real_sum.view(), sqrt2_inv);
    let imag_sum = <f64 as SimdF64>::simd_add_arrays(&a0_imag_view, &a1_imag_view);
    let new_a0_imag_arr = <f64 as SimdF64>::simd_scalar_mul(&imag_sum.view(), sqrt2_inv);

    // Compute (a0 - a1) * sqrt2_inv
    let real_diff = <f64 as SimdF64>::simd_sub_arrays(&a0_real_view, &a1_real_view);
    let new_a1_real_arr = <f64 as SimdF64>::simd_scalar_mul(&real_diff.view(), sqrt2_inv);
    let imag_diff = <f64 as SimdF64>::simd_sub_arrays(&a0_imag_view, &a1_imag_view);
    let new_a1_imag_arr = <f64 as SimdF64>::simd_scalar_mul(&imag_diff.view(), sqrt2_inv);

    new_a0_real_view.assign(&new_a0_real_arr);
    new_a0_imag_view.assign(&new_a0_imag_arr);
    new_a1_real_view.assign(&new_a1_real_arr);
    new_a1_imag_view.assign(&new_a1_imag_arr);

    // Write back results
    for i in 0..pair_count {
        amplitudes[idx0_list[i]] = Complex64::new(new_a0_real[i], new_a0_imag[i]);
        amplitudes[idx1_list[i]] = Complex64::new(new_a1_real[i], new_a1_imag[i]);
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

    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let control_mask = 1 << control_qubit;
    let target_mask = 1 << target_qubit;

    // Collect indices where phase needs to be applied
    let mut indices = Vec::new();
    for idx in 0..amplitudes.len() {
        if (idx & control_mask) != 0 && (idx & target_mask) != 0 {
            indices.push(idx);
        }
    }

    if indices.is_empty() {
        return Ok(());
    }

    // Extract amplitudes to be rotated
    let count = indices.len();
    let mut real_parts = Vec::with_capacity(count);
    let mut imag_parts = Vec::with_capacity(count);

    for &idx in &indices {
        let amp = amplitudes[idx];
        real_parts.push(amp.re);
        imag_parts.push(amp.im);
    }

    // Apply phase rotation using SIMD
    let real_view = ArrayView1::from(&real_parts);
    let imag_view = ArrayView1::from(&imag_parts);

    let mut new_real = vec![0.0; count];
    let mut new_imag = vec![0.0; count];
    let mut new_real_view = ArrayViewMut1::from(&mut new_real);
    let mut new_imag_view = ArrayViewMut1::from(&mut new_imag);

    // Compute real part: a*cos - b*sin
    let real_cos = <f64 as SimdF64>::simd_scalar_mul(&real_view, cos_theta);
    let imag_sin = <f64 as SimdF64>::simd_scalar_mul(&imag_view, sin_theta);
    let new_real_arr = <f64 as SimdF64>::simd_sub_arrays(&real_cos.view(), &imag_sin.view());

    // Compute imaginary part: a*sin + b*cos
    let real_sin = <f64 as SimdF64>::simd_scalar_mul(&real_view, sin_theta);
    let imag_cos = <f64 as SimdF64>::simd_scalar_mul(&imag_view, cos_theta);
    let new_imag_arr = <f64 as SimdF64>::simd_add_arrays(&real_sin.view(), &imag_cos.view());

    new_real_view.assign(&new_real_arr);
    new_imag_view.assign(&new_imag_arr);

    // Write back results
    for (i, &idx) in indices.iter().enumerate() {
        amplitudes[idx] = Complex64::new(new_real[i], new_imag[i]);
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
