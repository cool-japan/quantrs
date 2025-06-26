//! SIMD-accelerated operations for quantum state vector simulation
//!
//! This module provides SIMD-optimized implementations of quantum gate operations
//! for improved performance on modern CPUs using SciRS2 SIMD operations.

use num_complex::Complex64;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Simplified SIMD-like structure for complex operations
/// NOTE: This is being deprecated in favor of SciRS2 SIMD operations.
/// New code should use scirs2_core::simd_ops::SimdUnifiedOps directly.
#[derive(Clone, Copy, Debug)]
#[deprecated(note = "Use scirs2_core::simd_ops::SimdUnifiedOps instead")]
pub struct ComplexVec4 {
    re: [f64; 4],
    im: [f64; 4],
}

impl ComplexVec4 {
    /// Create a new ComplexVec4 from four Complex64 values
    pub fn new(values: [Complex64; 4]) -> Self {
        let mut re = [0.0; 4];
        let mut im = [0.0; 4];
        
        for i in 0..4 {
            re[i] = values[i].re;
            im[i] = values[i].im;
        }
        
        Self { re, im }
    }
    
    /// Create a new ComplexVec4 where all elements have the same value
    pub fn splat(value: Complex64) -> Self {
        Self {
            re: [value.re, value.re, value.re, value.re],
            im: [value.im, value.im, value.im, value.im],
        }
    }
    
    /// Get the element at the specified index
    pub fn get(&self, idx: usize) -> Complex64 {
        assert!(idx < 4, "Index out of bounds");
        Complex64::new(self.re[idx], self.im[idx])
    }
    
    /// Multiply by another ComplexVec4
    pub fn mul(&self, other: &ComplexVec4) -> ComplexVec4 {
        let mut result = ComplexVec4 {
            re: [0.0; 4],
            im: [0.0; 4],
        };
        
        for i in 0..4 {
            result.re[i] = self.re[i] * other.re[i] - self.im[i] * other.im[i];
            result.im[i] = self.re[i] * other.im[i] + self.im[i] * other.re[i];
        }
        
        result
    }
    
    /// Add another ComplexVec4
    pub fn add(&self, other: &ComplexVec4) -> ComplexVec4 {
        let mut result = ComplexVec4 {
            re: [0.0; 4],
            im: [0.0; 4],
        };
        
        for i in 0..4 {
            result.re[i] = self.re[i] + other.re[i];
            result.im[i] = self.im[i] + other.im[i];
        }
        
        result
    }
    
    /// Subtract another ComplexVec4
    pub fn sub(&self, other: &ComplexVec4) -> ComplexVec4 {
        let mut result = ComplexVec4 {
            re: [0.0; 4],
            im: [0.0; 4],
        };
        
        for i in 0..4 {
            result.re[i] = self.re[i] - other.re[i];
            result.im[i] = self.im[i] - other.im[i];
        }
        
        result
    }
    
    /// Negate all elements
    pub fn neg(&self) -> ComplexVec4 {
        let mut result = ComplexVec4 {
            re: [0.0; 4],
            im: [0.0; 4],
        };
        
        for i in 0..4 {
            result.re[i] = -self.re[i];
            result.im[i] = -self.im[i];
        }
        
        result
    }
}

// ============================================================================
// NEW SCIRS2-BASED SIMD IMPLEMENTATIONS
// ============================================================================

/// Apply a single-qubit gate using SciRS2 SIMD operations
///
/// This function uses the SciRS2 SimdUnifiedOps trait for better performance
/// and compliance with the SciRS2 integration policy.
pub fn apply_single_qubit_gate_simd_v2(
    matrix: &[Complex64; 4],
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    let len = in_amps0.len();
    
    // Extract matrix elements
    let m00 = matrix[0];
    let m01 = matrix[1];
    let m10 = matrix[2];
    let m11 = matrix[3];
    
    // Extract real and imaginary parts for SIMD operations
    let mut a0_real: Vec<f64> = in_amps0.iter().map(|c| c.re).collect();
    let mut a0_imag: Vec<f64> = in_amps0.iter().map(|c| c.im).collect();
    let mut a1_real: Vec<f64> = in_amps1.iter().map(|c| c.re).collect();
    let mut a1_imag: Vec<f64> = in_amps1.iter().map(|c| c.im).collect();
    
    let a0_real_view = ArrayView1::from(&a0_real);
    let a0_imag_view = ArrayView1::from(&a0_imag);
    let a1_real_view = ArrayView1::from(&a1_real);
    let a1_imag_view = ArrayView1::from(&a1_imag);
    
    // Compute new_a0 = m00 * a0 + m01 * a1
    // Real part: m00.re * a0.re - m00.im * a0.im + m01.re * a1.re - m01.im * a1.im
    let term1 = f64::simd_scalar_mul(&a0_real_view, m00.re);
    let term2 = f64::simd_scalar_mul(&a0_imag_view, m00.im);
    let term3 = f64::simd_scalar_mul(&a1_real_view, m01.re);
    let term4 = f64::simd_scalar_mul(&a1_imag_view, m01.im);
    
    let temp1 = f64::simd_sub(&term1.view(), &term2.view());
    let temp2 = f64::simd_sub(&term3.view(), &term4.view());
    let new_a0_real = f64::simd_add(&temp1.view(), &temp2.view());
    
    // Imaginary part: m00.re * a0.im + m00.im * a0.re + m01.re * a1.im + m01.im * a1.re
    let term5 = f64::simd_scalar_mul(&a0_imag_view, m00.re);
    let term6 = f64::simd_scalar_mul(&a0_real_view, m00.im);
    let term7 = f64::simd_scalar_mul(&a1_imag_view, m01.re);
    let term8 = f64::simd_scalar_mul(&a1_real_view, m01.im);
    
    let temp3 = f64::simd_add(&term5.view(), &term6.view());
    let temp4 = f64::simd_add(&term7.view(), &term8.view());
    let new_a0_imag = f64::simd_add(&temp3.view(), &temp4.view());
    
    // Compute new_a1 = m10 * a0 + m11 * a1
    let term9 = f64::simd_scalar_mul(&a0_real_view, m10.re);
    let term10 = f64::simd_scalar_mul(&a0_imag_view, m10.im);
    let term11 = f64::simd_scalar_mul(&a1_real_view, m11.re);
    let term12 = f64::simd_scalar_mul(&a1_imag_view, m11.im);
    
    let temp5 = f64::simd_sub(&term9.view(), &term10.view());
    let temp6 = f64::simd_sub(&term11.view(), &term12.view());
    let new_a1_real = f64::simd_add(&temp5.view(), &temp6.view());
    
    let term13 = f64::simd_scalar_mul(&a0_imag_view, m10.re);
    let term14 = f64::simd_scalar_mul(&a0_real_view, m10.im);
    let term15 = f64::simd_scalar_mul(&a1_imag_view, m11.re);
    let term16 = f64::simd_scalar_mul(&a1_real_view, m11.im);
    
    let temp7 = f64::simd_add(&term13.view(), &term14.view());
    let temp8 = f64::simd_add(&term15.view(), &term16.view());
    let new_a1_imag = f64::simd_add(&temp7.view(), &temp8.view());
    
    // Write back results
    for i in 0..len {
        out_amps0[i] = Complex64::new(new_a0_real[i], new_a0_imag[i]);
        out_amps1[i] = Complex64::new(new_a1_real[i], new_a1_imag[i]);
    }
}

/// Apply Hadamard gate using SciRS2 SIMD operations
pub fn apply_h_gate_simd_v2(
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    let sqrt2_inv = std::f64::consts::FRAC_1_SQRT_2;
    let len = in_amps0.len();
    
    // Extract real and imaginary parts
    let a0_real: Vec<f64> = in_amps0.iter().map(|c| c.re).collect();
    let a0_imag: Vec<f64> = in_amps0.iter().map(|c| c.im).collect();
    let a1_real: Vec<f64> = in_amps1.iter().map(|c| c.re).collect();
    let a1_imag: Vec<f64> = in_amps1.iter().map(|c| c.im).collect();
    
    let a0_real_view = ArrayView1::from(&a0_real);
    let a0_imag_view = ArrayView1::from(&a0_imag);
    let a1_real_view = ArrayView1::from(&a1_real);
    let a1_imag_view = ArrayView1::from(&a1_imag);
    
    // Hadamard: new_a0 = (a0 + a1) / sqrt(2), new_a1 = (a0 - a1) / sqrt(2)
    let sum_real = f64::simd_add(&a0_real_view, &a1_real_view);
    let sum_imag = f64::simd_add(&a0_imag_view, &a1_imag_view);
    let diff_real = f64::simd_sub(&a0_real_view, &a1_real_view);
    let diff_imag = f64::simd_sub(&a0_imag_view, &a1_imag_view);
    
    let new_a0_real = f64::simd_scalar_mul(&sum_real.view(), sqrt2_inv);
    let new_a0_imag = f64::simd_scalar_mul(&sum_imag.view(), sqrt2_inv);
    let new_a1_real = f64::simd_scalar_mul(&diff_real.view(), sqrt2_inv);
    let new_a1_imag = f64::simd_scalar_mul(&diff_imag.view(), sqrt2_inv);
    
    // Write back results
    for i in 0..len {
        out_amps0[i] = Complex64::new(new_a0_real[i], new_a0_imag[i]);
        out_amps1[i] = Complex64::new(new_a1_real[i], new_a1_imag[i]);
    }
}

// ============================================================================
// LEGACY IMPLEMENTATIONS (to be removed after full migration)
// ============================================================================

/// Apply a single-qubit gate to multiple amplitudes using SIMD-like operations
///
/// This function processes 4 pairs of amplitudes at once using SIMD-like operations
///
/// # Arguments
///
/// * `matrix` - The 2x2 matrix representation of the gate
/// * `in_amps0` - The first set of input amplitudes (corresponding to bit=0)
/// * `in_amps1` - The second set of input amplitudes (corresponding to bit=1)
/// * `out_amps0` - Output buffer for the first set of amplitudes
/// * `out_amps1` - Output buffer for the second set of amplitudes
pub fn apply_single_qubit_gate_simd(
    matrix: &[Complex64; 4],
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    // Process elements in chunks of 4
    let chunks = in_amps0.len() / 4;
    
    // Extract matrix elements for SIMD-like operations
    let m00 = ComplexVec4::splat(matrix[0]);
    let m01 = ComplexVec4::splat(matrix[1]);
    let m10 = ComplexVec4::splat(matrix[2]);
    let m11 = ComplexVec4::splat(matrix[3]);
    
    for chunk in 0..chunks {
        let offset = chunk * 4;
        
        // Load 4 complex numbers from in_amps0 and in_amps1
        let a0 = ComplexVec4::new([
            in_amps0[offset],
            in_amps0[offset + 1],
            in_amps0[offset + 2],
            in_amps0[offset + 3],
        ]);
        
        let a1 = ComplexVec4::new([
            in_amps1[offset],
            in_amps1[offset + 1],
            in_amps1[offset + 2],
            in_amps1[offset + 3],
        ]);
        
        // Compute complex multiplications
        let m00a0 = m00.mul(&a0);
        let m01a1 = m01.mul(&a1);
        let m10a0 = m10.mul(&a0);
        let m11a1 = m11.mul(&a1);
        
        // Compute new amplitudes
        let new_a0 = m00a0.add(&m01a1);
        let new_a1 = m10a0.add(&m11a1);
        
        // Store the results
        for i in 0..4 {
            out_amps0[offset + i] = new_a0.get(i);
            out_amps1[offset + i] = new_a1.get(i);
        }
    }
    
    // Handle remaining elements (less than 4)
    let remainder_start = chunks * 4;
    for i in remainder_start..in_amps0.len() {
        let a0 = in_amps0[i];
        let a1 = in_amps1[i];
        
        out_amps0[i] = matrix[0] * a0 + matrix[1] * a1;
        out_amps1[i] = matrix[2] * a0 + matrix[3] * a1;
    }
}

/// Apply X gate to multiple amplitudes using SIMD-like operations
///
/// This is a specialized implementation for the Pauli X gate, which simply swaps
/// amplitudes, making it very efficient to implement.
///
/// # Arguments
///
/// * `in_amps0` - The first set of input amplitudes (corresponding to bit=0)
/// * `in_amps1` - The second set of input amplitudes (corresponding to bit=1)
/// * `out_amps0` - Output buffer for the first set of amplitudes
/// * `out_amps1` - Output buffer for the second set of amplitudes
pub fn apply_x_gate_simd(
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    // Simply swap the amplitudes using copy_from_slice
    out_amps0[..in_amps0.len()].copy_from_slice(&in_amps1[..in_amps0.len()]);
    out_amps1[..in_amps0.len()].copy_from_slice(in_amps0);
}

/// Apply Z gate to multiple amplitudes using SIMD-like operations
///
/// This is a specialized implementation for the Pauli Z gate, which only flips the
/// sign of amplitudes where the target bit is 1.
///
/// # Arguments
///
/// * `in_amps0` - The first set of input amplitudes (corresponding to bit=0)
/// * `in_amps1` - The second set of input amplitudes (corresponding to bit=1)
/// * `out_amps0` - Output buffer for the first set of amplitudes
/// * `out_amps1` - Output buffer for the second set of amplitudes
pub fn apply_z_gate_simd(
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    // For Z gate, a0 stays the same, a1 gets negated
    for i in 0..in_amps0.len() {
        out_amps0[i] = in_amps0[i];
        out_amps1[i] = -in_amps1[i];
    }
}

/// Apply Hadamard gate using SIMD-like operations
///
/// This is a specialized implementation for the Hadamard gate using the matrix:
/// H = 1/√2 * [[1, 1], [1, -1]]
pub fn apply_h_gate_simd(
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    use std::f64::consts::FRAC_1_SQRT_2;
    let h_coeff = Complex64::new(FRAC_1_SQRT_2, 0.0);
    
    // Process elements in chunks of 4
    let chunks = in_amps0.len() / 4;
    let h_vec = ComplexVec4::splat(h_coeff);
    
    for chunk in 0..chunks {
        let offset = chunk * 4;
        
        let a0 = ComplexVec4::new([
            in_amps0[offset],
            in_amps0[offset + 1],
            in_amps0[offset + 2],
            in_amps0[offset + 3],
        ]);
        
        let a1 = ComplexVec4::new([
            in_amps1[offset],
            in_amps1[offset + 1],
            in_amps1[offset + 2],
            in_amps1[offset + 3],
        ]);
        
        // H|0⟩ = 1/√2(|0⟩ + |1⟩), H|1⟩ = 1/√2(|0⟩ - |1⟩)
        let sum = a0.add(&a1);
        let diff = a0.sub(&a1);
        
        let new_a0 = h_vec.mul(&sum);
        let new_a1 = h_vec.mul(&diff);
        
        for i in 0..4 {
            out_amps0[offset + i] = new_a0.get(i);
            out_amps1[offset + i] = new_a1.get(i);
        }
    }
    
    // Handle remaining elements
    let remainder_start = chunks * 4;
    for i in remainder_start..in_amps0.len() {
        let a0 = in_amps0[i];
        let a1 = in_amps1[i];
        
        out_amps0[i] = h_coeff * (a0 + a1);
        out_amps1[i] = h_coeff * (a0 - a1);
    }
}

/// Apply Y gate using SIMD-like operations
///
/// Y gate: [[0, -i], [i, 0]]
pub fn apply_y_gate_simd(
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    let i_pos = Complex64::new(0.0, 1.0);
    let i_neg = Complex64::new(0.0, -1.0);
    
    // Process elements in chunks of 4
    let chunks = in_amps0.len() / 4;
    let i_pos_vec = ComplexVec4::splat(i_pos);
    let i_neg_vec = ComplexVec4::splat(i_neg);
    
    for chunk in 0..chunks {
        let offset = chunk * 4;
        
        let a0 = ComplexVec4::new([
            in_amps0[offset],
            in_amps0[offset + 1],
            in_amps0[offset + 2],
            in_amps0[offset + 3],
        ]);
        
        let a1 = ComplexVec4::new([
            in_amps1[offset],
            in_amps1[offset + 1],
            in_amps1[offset + 2],
            in_amps1[offset + 3],
        ]);
        
        // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
        let new_a0 = i_neg_vec.mul(&a1);
        let new_a1 = i_pos_vec.mul(&a0);
        
        for i in 0..4 {
            out_amps0[offset + i] = new_a0.get(i);
            out_amps1[offset + i] = new_a1.get(i);
        }
    }
    
    // Handle remaining elements
    let remainder_start = chunks * 4;
    for i in remainder_start..in_amps0.len() {
        let a0 = in_amps0[i];
        let a1 = in_amps1[i];
        
        out_amps0[i] = i_neg * a1;
        out_amps1[i] = i_pos * a0;
    }
}

/// Apply phase gate (S gate) using SIMD-like operations
///
/// S gate: [[1, 0], [0, i]]
pub fn apply_s_gate_simd(
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    let i_phase = Complex64::new(0.0, 1.0);
    
    // Process elements in chunks of 4
    let chunks = in_amps0.len() / 4;
    let i_vec = ComplexVec4::splat(i_phase);
    
    for chunk in 0..chunks {
        let offset = chunk * 4;
        
        let a1 = ComplexVec4::new([
            in_amps1[offset],
            in_amps1[offset + 1],
            in_amps1[offset + 2],
            in_amps1[offset + 3],
        ]);
        
        let new_a1 = i_vec.mul(&a1);
        
        // Copy a0 unchanged, multiply a1 by i
        for i in 0..4 {
            out_amps0[offset + i] = in_amps0[offset + i];
            out_amps1[offset + i] = new_a1.get(i);
        }
    }
    
    // Handle remaining elements
    let remainder_start = chunks * 4;
    for i in remainder_start..in_amps0.len() {
        out_amps0[i] = in_amps0[i];
        out_amps1[i] = i_phase * in_amps1[i];
    }
}

/// Apply rotation-X gate using SIMD-like operations
///
/// RX(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
pub fn apply_rx_gate_simd(
    angle: f64,
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    let half_angle = angle / 2.0;
    let cos_val = Complex64::new(half_angle.cos(), 0.0);
    let neg_i_sin_val = Complex64::new(0.0, -half_angle.sin());
    
    // Process elements in chunks of 4
    let chunks = in_amps0.len() / 4;
    let cos_vec = ComplexVec4::splat(cos_val);
    let neg_i_sin_vec = ComplexVec4::splat(neg_i_sin_val);
    
    for chunk in 0..chunks {
        let offset = chunk * 4;
        
        let a0 = ComplexVec4::new([
            in_amps0[offset],
            in_amps0[offset + 1],
            in_amps0[offset + 2],
            in_amps0[offset + 3],
        ]);
        
        let a1 = ComplexVec4::new([
            in_amps1[offset],
            in_amps1[offset + 1],
            in_amps1[offset + 2],
            in_amps1[offset + 3],
        ]);
        
        let cos_a0 = cos_vec.mul(&a0);
        let neg_i_sin_a1 = neg_i_sin_vec.mul(&a1);
        let neg_i_sin_a0 = neg_i_sin_vec.mul(&a0);
        let cos_a1 = cos_vec.mul(&a1);
        
        let new_a0 = cos_a0.add(&neg_i_sin_a1);
        let new_a1 = neg_i_sin_a0.add(&cos_a1);
        
        for i in 0..4 {
            out_amps0[offset + i] = new_a0.get(i);
            out_amps1[offset + i] = new_a1.get(i);
        }
    }
    
    // Handle remaining elements
    let remainder_start = chunks * 4;
    for i in remainder_start..in_amps0.len() {
        let a0 = in_amps0[i];
        let a1 = in_amps1[i];
        
        out_amps0[i] = cos_val * a0 + neg_i_sin_val * a1;
        out_amps1[i] = neg_i_sin_val * a0 + cos_val * a1;
    }
}

/// SIMD-optimized wrapper function for applying gates
///
/// This function uses SciRS2 SIMD implementations for all gate operations.
pub fn apply_single_qubit_gate_optimized(
    matrix: &[Complex64; 4],
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    use std::f64::consts::FRAC_1_SQRT_2;
    
    // Special-case optimizations for common gates
    if *matrix == [
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(-FRAC_1_SQRT_2, 0.0),
    ] {
        // Hadamard gate - use specialized implementation
        apply_h_gate_simd_v2(in_amps0, in_amps1, out_amps0, out_amps1);
    } else {
        // All other gates use the general SciRS2 SIMD implementation
        apply_single_qubit_gate_simd_v2(matrix, in_amps0, in_amps1, out_amps0, out_amps1);
    }
}

/// Apply rotation-Y gate using SIMD-like operations
///
/// RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
pub fn apply_ry_gate_simd(
    angle: f64,
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    let half_angle = angle / 2.0;
    let cos_val = Complex64::new(half_angle.cos(), 0.0);
    let sin_val = Complex64::new(half_angle.sin(), 0.0);
    let neg_sin_val = Complex64::new(-half_angle.sin(), 0.0);
    
    // Process elements in chunks of 4
    let chunks = in_amps0.len() / 4;
    let cos_vec = ComplexVec4::splat(cos_val);
    let sin_vec = ComplexVec4::splat(sin_val);
    let neg_sin_vec = ComplexVec4::splat(neg_sin_val);
    
    for chunk in 0..chunks {
        let offset = chunk * 4;
        
        let a0 = ComplexVec4::new([
            in_amps0[offset],
            in_amps0[offset + 1],
            in_amps0[offset + 2],
            in_amps0[offset + 3],
        ]);
        
        let a1 = ComplexVec4::new([
            in_amps1[offset],
            in_amps1[offset + 1],
            in_amps1[offset + 2],
            in_amps1[offset + 3],
        ]);
        
        let cos_a0 = cos_vec.mul(&a0);
        let neg_sin_a1 = neg_sin_vec.mul(&a1);
        let sin_a0 = sin_vec.mul(&a0);
        let cos_a1 = cos_vec.mul(&a1);
        
        let new_a0 = cos_a0.add(&neg_sin_a1);
        let new_a1 = sin_a0.add(&cos_a1);
        
        for i in 0..4 {
            out_amps0[offset + i] = new_a0.get(i);
            out_amps1[offset + i] = new_a1.get(i);
        }
    }
    
    // Handle remaining elements
    let remainder_start = chunks * 4;
    for i in remainder_start..in_amps0.len() {
        let a0 = in_amps0[i];
        let a1 = in_amps1[i];
        
        out_amps0[i] = cos_val * a0 + neg_sin_val * a1;
        out_amps1[i] = sin_val * a0 + cos_val * a1;
    }
}

/// Apply rotation-Z gate using SIMD-like operations
///
/// RZ(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
pub fn apply_rz_gate_simd(
    angle: f64,
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    let half_angle = angle / 2.0;
    let exp_neg_i = Complex64::new(half_angle.cos(), -half_angle.sin());
    let exp_pos_i = Complex64::new(half_angle.cos(), half_angle.sin());
    
    // Process elements in chunks of 4
    let chunks = in_amps0.len() / 4;
    let exp_neg_vec = ComplexVec4::splat(exp_neg_i);
    let exp_pos_vec = ComplexVec4::splat(exp_pos_i);
    
    for chunk in 0..chunks {
        let offset = chunk * 4;
        
        let a0 = ComplexVec4::new([
            in_amps0[offset],
            in_amps0[offset + 1],
            in_amps0[offset + 2],
            in_amps0[offset + 3],
        ]);
        
        let a1 = ComplexVec4::new([
            in_amps1[offset],
            in_amps1[offset + 1],
            in_amps1[offset + 2],
            in_amps1[offset + 3],
        ]);
        
        let new_a0 = exp_neg_vec.mul(&a0);
        let new_a1 = exp_pos_vec.mul(&a1);
        
        for i in 0..4 {
            out_amps0[offset + i] = new_a0.get(i);
            out_amps1[offset + i] = new_a1.get(i);
        }
    }
    
    // Handle remaining elements
    let remainder_start = chunks * 4;
    for i in remainder_start..in_amps0.len() {
        out_amps0[i] = exp_neg_i * in_amps0[i];
        out_amps1[i] = exp_pos_i * in_amps1[i];
    }
}

/// Apply T gate using SIMD-like operations
///
/// T gate: [[1, 0], [0, e^(iπ/4)]]
pub fn apply_t_gate_simd(
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    use std::f64::consts::FRAC_PI_4;
    let t_phase = Complex64::new(FRAC_PI_4.cos(), FRAC_PI_4.sin());
    
    // Process elements in chunks of 4
    let chunks = in_amps0.len() / 4;
    let t_vec = ComplexVec4::splat(t_phase);
    
    for chunk in 0..chunks {
        let offset = chunk * 4;
        
        let a1 = ComplexVec4::new([
            in_amps1[offset],
            in_amps1[offset + 1],
            in_amps1[offset + 2],
            in_amps1[offset + 3],
        ]);
        
        let new_a1 = t_vec.mul(&a1);
        
        // Copy a0 unchanged, multiply a1 by t_phase
        for i in 0..4 {
            out_amps0[offset + i] = in_amps0[offset + i];
            out_amps1[offset + i] = new_a1.get(i);
        }
    }
    
    // Handle remaining elements
    let remainder_start = chunks * 4;
    for i in remainder_start..in_amps0.len() {
        out_amps0[i] = in_amps0[i];
        out_amps1[i] = t_phase * in_amps1[i];
    }
}

/// Gate fusion structure for combining adjacent single-qubit gates
#[derive(Debug, Clone)]
pub struct GateFusion {
    /// Fused matrix representation
    pub fused_matrix: [Complex64; 4],
    /// Target qubit
    pub target: usize,
    /// Number of gates fused
    pub gate_count: usize,
}

impl GateFusion {
    /// Create a new gate fusion starting with an identity gate
    pub fn new(target: usize) -> Self {
        Self {
            fused_matrix: [
                Complex64::new(1.0, 0.0), // I[0,0]
                Complex64::new(0.0, 0.0), // I[0,1]
                Complex64::new(0.0, 0.0), // I[1,0]
                Complex64::new(1.0, 0.0), // I[1,1]
            ],
            target,
            gate_count: 0,
        }
    }
    
    /// Fuse another gate into this fusion
    pub fn fuse_gate(&mut self, gate_matrix: &[Complex64; 4]) {
        // Matrix multiplication: new_matrix = gate_matrix * fused_matrix
        let m = &self.fused_matrix;
        let g = gate_matrix;
        
        self.fused_matrix = [
            g[0] * m[0] + g[1] * m[2], // (0,0)
            g[0] * m[1] + g[1] * m[3], // (0,1)
            g[2] * m[0] + g[3] * m[2], // (1,0)
            g[2] * m[1] + g[3] * m[3], // (1,1)
        ];
        
        self.gate_count += 1;
    }
    
    /// Check if this fusion can be applied using a specialized SIMD kernel
    pub fn can_use_specialized_kernel(&self) -> bool {
        use std::f64::consts::FRAC_1_SQRT_2;
        
        // Check for common gate patterns after fusion
        let m = &self.fused_matrix;
        
        // Identity gate (no-op)
        if (m[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10
            && m[1].norm() < 1e-10
            && m[2].norm() < 1e-10
            && (m[3] - Complex64::new(1.0, 0.0)).norm() < 1e-10
        {
            return true;
        }
        
        // X gate
        if m[0].norm() < 1e-10
            && (m[1] - Complex64::new(1.0, 0.0)).norm() < 1e-10
            && (m[2] - Complex64::new(1.0, 0.0)).norm() < 1e-10
            && m[3].norm() < 1e-10
        {
            return true;
        }
        
        // Y gate
        if m[0].norm() < 1e-10
            && (m[1] - Complex64::new(0.0, -1.0)).norm() < 1e-10
            && (m[2] - Complex64::new(0.0, 1.0)).norm() < 1e-10
            && m[3].norm() < 1e-10
        {
            return true;
        }
        
        // Z gate
        if (m[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10
            && m[1].norm() < 1e-10
            && m[2].norm() < 1e-10
            && (m[3] - Complex64::new(-1.0, 0.0)).norm() < 1e-10
        {
            return true;
        }
        
        // Hadamard gate
        if (m[0] - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-10
            && (m[1] - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-10
            && (m[2] - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-10
            && (m[3] - Complex64::new(-FRAC_1_SQRT_2, 0.0)).norm() < 1e-10
        {
            return true;
        }
        
        false
    }
    
    /// Apply the fused gate using SIMD optimization
    pub fn apply_simd(
        &self,
        in_amps0: &[Complex64],
        in_amps1: &[Complex64],
        out_amps0: &mut [Complex64],
        out_amps1: &mut [Complex64],
    ) {
        apply_single_qubit_gate_optimized(
            &self.fused_matrix,
            in_amps0,
            in_amps1,
            out_amps0,
            out_amps1,
        );
    }
}

/// Vectorized CNOT gate application using SIMD for processing multiple pairs
///
/// This processes control/target pairs in parallel where possible
pub fn apply_cnot_vectorized(
    state: &mut [Complex64],
    control_indices: &[usize],
    target_indices: &[usize],
    num_qubits: usize,
) {
    let dim = 1 << num_qubits;
    let mut new_state = vec![Complex64::new(0.0, 0.0); dim];
    
    // Process all CNOT gates in parallel
    new_state
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, new_amp)| {
            let mut final_idx = i;
            
            // Apply all CNOT gates in sequence
            for (&control_idx, &target_idx) in control_indices.iter().zip(target_indices.iter()) {
                if (final_idx >> control_idx) & 1 == 1 {
                    final_idx ^= 1 << target_idx;
                }
            }
            
            *new_amp = state[final_idx];
        });
    
    state.copy_from_slice(&new_state);
}

/// Scalar implementation of apply_single_qubit_gate for fallback
///
/// # Arguments
///
/// * `matrix` - The 2x2 matrix representation of the gate
/// * `in_amps0` - The first set of input amplitudes (corresponding to bit=0)
/// * `in_amps1` - The second set of input amplitudes (corresponding to bit=1)
/// * `out_amps0` - Output buffer for the first set of amplitudes
/// * `out_amps1` - Output buffer for the second set of amplitudes
pub fn apply_single_qubit_gate_scalar(
    matrix: &[Complex64; 4],
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    for i in 0..in_amps0.len() {
        let a0 = in_amps0[i];
        let a1 = in_amps1[i];
        
        out_amps0[i] = matrix[0] * a0 + matrix[1] * a1;
        out_amps1[i] = matrix[2] * a0 + matrix[3] * a1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;
    
    #[test]
    fn test_x_gate_scalar() {
        // X gate matrix
        let x_matrix = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        
        // Test data
        let in_amps0 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)];
        let in_amps1 = vec![Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)];
        let mut out_amps0 = [Complex64::new(0.0, 0.0); 2];
        let mut out_amps1 = [Complex64::new(0.0, 0.0); 2];
        
        // Apply gate
        apply_single_qubit_gate_scalar(
            &x_matrix,
            &in_amps0,
            &in_amps1,
            &mut out_amps0,
            &mut out_amps1,
        );
        
        // Check results
        assert_eq!(out_amps0[0], Complex64::new(0.0, 0.0));
        assert_eq!(out_amps1[0], Complex64::new(1.0, 0.0));
        assert_eq!(out_amps0[1], Complex64::new(0.5, 0.0));
        assert_eq!(out_amps1[1], Complex64::new(0.5, 0.0));
    }
    
    #[test]
    fn test_hadamard_gate_scalar() {
        // Hadamard gate matrix
        let h_matrix = [
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(-FRAC_1_SQRT_2, 0.0),
        ];
        
        // Test data
        let in_amps0 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let in_amps1 = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        let mut out_amps0 = [Complex64::new(0.0, 0.0); 2];
        let mut out_amps1 = [Complex64::new(0.0, 0.0); 2];
        
        // Apply gate
        apply_single_qubit_gate_scalar(
            &h_matrix,
            &in_amps0,
            &in_amps1,
            &mut out_amps0,
            &mut out_amps1,
        );
        
        // Check results - applying H to |0> should give (|0> + |1>)/sqrt(2)
        assert!((out_amps0[0] - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-10);
        assert!((out_amps1[0] - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-10);
        
        // Applying H to |1> should give (|0> - |1>)/sqrt(2)
        assert!((out_amps0[1] - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-10);
        assert!((out_amps1[1] - Complex64::new(-FRAC_1_SQRT_2, 0.0)).norm() < 1e-10);
    }
    
    #[test]
    fn test_optimized_gate_wrapper() {
        // Hadamard gate matrix
        let h_matrix = [
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(-FRAC_1_SQRT_2, 0.0),
        ];
        
        // Test data
        let in_amps0 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let in_amps1 = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        let mut out_amps0 = [Complex64::new(0.0, 0.0); 2];
        let mut out_amps1 = [Complex64::new(0.0, 0.0); 2];
        
        // Apply gate using the optimized wrapper
        apply_single_qubit_gate_optimized(
            &h_matrix,
            &in_amps0,
            &in_amps1,
            &mut out_amps0,
            &mut out_amps1,
        );
        
        // Check results - applying H to |0> should give (|0> + |1>)/sqrt(2)
        assert!((out_amps0[0] - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-10);
        assert!((out_amps1[0] - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-10);
        
        // Applying H to |1> should give (|0> - |1>)/sqrt(2)
        assert!((out_amps0[1] - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-10);
        assert!((out_amps1[1] - Complex64::new(-FRAC_1_SQRT_2, 0.0)).norm() < 1e-10);
    }
    
    #[test]
    fn test_complex_vec4() {
        // Test splat creation
        let a = ComplexVec4::splat(Complex64::new(1.0, 2.0));
        for i in 0..4 {
            assert_eq!(a.get(i), Complex64::new(1.0, 2.0));
        }
        
        // Test new creation
        let b = ComplexVec4::new([
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
        ]);
        
        assert_eq!(b.get(0), Complex64::new(1.0, 2.0));
        assert_eq!(b.get(1), Complex64::new(3.0, 4.0));
        assert_eq!(b.get(2), Complex64::new(5.0, 6.0));
        assert_eq!(b.get(3), Complex64::new(7.0, 8.0));
        
        // Test multiplication
        let c = a.mul(&b);
        assert!((c.get(0) - Complex64::new(1.0, 2.0) * Complex64::new(1.0, 2.0)).norm() < 1e-10);
        assert!((c.get(1) - Complex64::new(1.0, 2.0) * Complex64::new(3.0, 4.0)).norm() < 1e-10);
        assert!((c.get(2) - Complex64::new(1.0, 2.0) * Complex64::new(5.0, 6.0)).norm() < 1e-10);
        assert!((c.get(3) - Complex64::new(1.0, 2.0) * Complex64::new(7.0, 8.0)).norm() < 1e-10);
        
        // Test addition
        let d = a.add(&b);
        assert!((d.get(0) - (Complex64::new(1.0, 2.0) + Complex64::new(1.0, 2.0))).norm() < 1e-10);
        assert!((d.get(1) - (Complex64::new(1.0, 2.0) + Complex64::new(3.0, 4.0))).norm() < 1e-10);
        assert!((d.get(2) - (Complex64::new(1.0, 2.0) + Complex64::new(5.0, 6.0))).norm() < 1e-10);
        assert!((d.get(3) - (Complex64::new(1.0, 2.0) + Complex64::new(7.0, 8.0))).norm() < 1e-10);
        
        // Test negation
        let e = b.neg();
        assert!((e.get(0) - (-Complex64::new(1.0, 2.0))).norm() < 1e-10);
        assert!((e.get(1) - (-Complex64::new(3.0, 4.0))).norm() < 1e-10);
        assert!((e.get(2) - (-Complex64::new(5.0, 6.0))).norm() < 1e-10);
        assert!((e.get(3) - (-Complex64::new(7.0, 8.0))).norm() < 1e-10);
    }
}