//! SIMD-accelerated operations for quantum state vector simulation
//!
//! This module provides SIMD-optimized implementations of quantum gate operations
//! for improved performance on modern CPUs.

use num_complex::Complex64;

/// Simplified SIMD-like structure for complex operations
/// This serves as a fallback implementation when SIMD is not available
#[derive(Clone, Copy, Debug)]
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

/// SIMD-optimized wrapper function for applying gates
///
/// This function uses the SIMD-like implementation.
///
/// # Arguments
///
/// * `matrix` - The 2x2 matrix representation of the gate
/// * `in_amps0` - The first set of input amplitudes (corresponding to bit=0)
/// * `in_amps1` - The second set of input amplitudes (corresponding to bit=1)
/// * `out_amps0` - Output buffer for the first set of amplitudes
/// * `out_amps1` - Output buffer for the second set of amplitudes
pub fn apply_single_qubit_gate_optimized(
    matrix: &[Complex64; 4],
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    // Special-case optimizations for common gates
    if *matrix
        == [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]
    {
        // X gate
        apply_x_gate_simd(in_amps0, in_amps1, out_amps0, out_amps1);
        return;
    } else if *matrix
        == [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ]
    {
        // Z gate
        apply_z_gate_simd(in_amps0, in_amps1, out_amps0, out_amps1);
        return;
    }

    // Generic gate
    apply_single_qubit_gate_simd(matrix, in_amps0, in_amps1, out_amps0, out_amps1);
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
        let mut out_amps0 = vec![Complex64::new(0.0, 0.0); 2];
        let mut out_amps1 = vec![Complex64::new(0.0, 0.0); 2];

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
        let mut out_amps0 = vec![Complex64::new(0.0, 0.0); 2];
        let mut out_amps1 = vec![Complex64::new(0.0, 0.0); 2];

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
        let mut out_amps0 = vec![Complex64::new(0.0, 0.0); 2];
        let mut out_amps1 = vec![Complex64::new(0.0, 0.0); 2];

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
