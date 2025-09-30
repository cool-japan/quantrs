//! Enhanced Complex Number SIMD Operations using SciRS2
//!
//! This module provides advanced SIMD implementations specifically optimized
//! for complex number arithmetic in quantum state vector operations.
//! It leverages SciRS2's SimdUnifiedOps for maximum performance.

use scirs2_core::ndarray::{Array1, ArrayView1, ArrayViewMut1};
use scirs2_core::Complex64;
use quantrs2_core::platform::PlatformCapabilities;
use scirs2_core::simd_ops::SimdUnifiedOps;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Complex SIMD vector operations using native SciRS2 primitives
#[derive(Debug, Clone)]
pub struct ComplexSimdVector {
    /// Real components (SIMD-aligned)
    real: Vec<f64>,
    /// Imaginary components (SIMD-aligned)
    imag: Vec<f64>,
    /// Number of complex elements
    length: usize,
    /// SIMD lane width for this platform
    simd_width: usize,
}

impl ComplexSimdVector {
    /// Create a new SIMD-aligned complex vector
    pub fn new(length: usize) -> Self {
        let simd_width = Self::detect_simd_width();
        let aligned_length = Self::align_length(length, simd_width);

        Self {
            real: vec![0.0; aligned_length],
            imag: vec![0.0; aligned_length],
            length,
            simd_width,
        }
    }

    /// Create from slice of Complex64
    pub fn from_slice(data: &[Complex64]) -> Self {
        let mut vec = Self::new(data.len());
        for (i, &complex) in data.iter().enumerate() {
            vec.real[i] = complex.re;
            vec.imag[i] = complex.im;
        }
        vec
    }

    /// Convert back to Complex64 slice
    pub fn to_complex_vec(&self) -> Vec<Complex64> {
        (0..self.length)
            .map(|i| Complex64::new(self.real[i], self.imag[i]))
            .collect()
    }

    /// Detect optimal SIMD width for current hardware using PlatformCapabilities
    pub fn detect_simd_width() -> usize {
        PlatformCapabilities::detect().optimal_simd_width_f64()
    }

    /// Align length to SIMD boundary
    fn align_length(length: usize, simd_width: usize) -> usize {
        ((length + simd_width - 1) / simd_width) * simd_width
    }

    /// Get real part as array view
    pub fn real_view(&self) -> ArrayView1<'_, f64> {
        ArrayView1::from(&self.real[..self.length])
    }

    /// Get imaginary part as array view
    pub fn imag_view(&self) -> ArrayView1<'_, f64> {
        ArrayView1::from(&self.imag[..self.length])
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// High-performance complex arithmetic operations using SciRS2 SIMD
pub struct ComplexSimdOps;

impl ComplexSimdOps {
    /// Complex multiplication: c = a * b, vectorized
    pub fn complex_mul_simd(
        a: &ComplexSimdVector,
        b: &ComplexSimdVector,
        c: &mut ComplexSimdVector,
    ) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());

        let a_real = a.real_view();
        let a_imag = a.imag_view();
        let b_real = b.real_view();
        let b_imag = b.imag_view();

        // Complex multiplication: (a_r + i*a_i) * (b_r + i*b_i)
        // = (a_r*b_r - a_i*b_i) + i*(a_r*b_i + a_i*b_r)

        // Real part: a_r*b_r - a_i*b_i using SciRS2 SIMD operations
        let ar_br = f64::simd_mul(&a_real, &b_real);
        let ai_bi = f64::simd_mul(&a_imag, &b_imag);
        let real_result = f64::simd_sub(&ar_br.view(), &ai_bi.view());

        // Imaginary part: a_r*b_i + a_i*b_r using SciRS2 SIMD operations
        let ar_bi = f64::simd_mul(&a_real, &b_imag);
        let ai_br = f64::simd_mul(&a_imag, &b_real);
        let imag_result = f64::simd_add(&ar_bi.view(), &ai_br.view());

        // Store results
        for i in 0..c.length {
            c.real[i] = real_result[i];
            c.imag[i] = imag_result[i];
        }
    }

    /// Complex addition: c = a + b, vectorized
    pub fn complex_add_simd(
        a: &ComplexSimdVector,
        b: &ComplexSimdVector,
        c: &mut ComplexSimdVector,
    ) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());

        let a_real = a.real_view();
        let a_imag = a.imag_view();
        let b_real = b.real_view();
        let b_imag = b.imag_view();

        let real_result = f64::simd_add(&a_real, &b_real);
        let imag_result = f64::simd_add(&a_imag, &b_imag);

        for i in 0..c.length {
            c.real[i] = real_result[i];
            c.imag[i] = imag_result[i];
        }
    }

    /// Complex subtraction: c = a - b, vectorized
    pub fn complex_sub_simd(
        a: &ComplexSimdVector,
        b: &ComplexSimdVector,
        c: &mut ComplexSimdVector,
    ) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());

        let a_real = a.real_view();
        let a_imag = a.imag_view();
        let b_real = b.real_view();
        let b_imag = b.imag_view();

        let real_result = f64::simd_sub(&a_real, &b_real);
        let imag_result = f64::simd_sub(&a_imag, &b_imag);

        for i in 0..c.length {
            c.real[i] = real_result[i];
            c.imag[i] = imag_result[i];
        }
    }

    /// Scalar complex multiplication: c = a * scalar, vectorized
    pub fn complex_scalar_mul_simd(
        a: &ComplexSimdVector,
        scalar: Complex64,
        c: &mut ComplexSimdVector,
    ) {
        assert_eq!(a.len(), c.len());

        let a_real = a.real_view();
        let a_imag = a.imag_view();

        // (a_r + i*a_i) * (s_r + i*s_i) = (a_r*s_r - a_i*s_i) + i*(a_r*s_i + a_i*s_r)
        let ar_sr = f64::simd_scalar_mul(&a_real, scalar.re);
        let ai_si = f64::simd_scalar_mul(&a_imag, scalar.im);
        let real_result = f64::simd_sub(&ar_sr.view(), &ai_si.view());

        let ar_si = f64::simd_scalar_mul(&a_real, scalar.im);
        let ai_sr = f64::simd_scalar_mul(&a_imag, scalar.re);
        let imag_result = f64::simd_add(&ar_si.view(), &ai_sr.view());

        for i in 0..c.length {
            c.real[i] = real_result[i];
            c.imag[i] = imag_result[i];
        }
    }

    /// Complex conjugate: c = conj(a), vectorized
    pub fn complex_conj_simd(a: &ComplexSimdVector, c: &mut ComplexSimdVector) {
        assert_eq!(a.len(), c.len());

        let a_real = a.real_view();
        let a_imag = a.imag_view();

        // Copy real part unchanged
        for i in 0..c.length {
            c.real[i] = a.real[i];
        }

        // Negate imaginary part
        let zero_array = Array1::zeros(a.length);
        let negated_imag = f64::simd_sub(&zero_array.view(), &a_imag);

        for i in 0..c.length {
            c.imag[i] = negated_imag[i];
        }
    }

    /// Complex magnitude squared: |a|^2, vectorized
    pub fn complex_norm_squared_simd(a: &ComplexSimdVector) -> Vec<f64> {
        let a_real = a.real_view();
        let a_imag = a.imag_view();

        let real_sq = f64::simd_mul(&a_real, &a_real);
        let imag_sq = f64::simd_mul(&a_imag, &a_imag);
        let norm_sq = f64::simd_add(&real_sq.view(), &imag_sq.view());

        norm_sq.to_vec()
    }
}

/// Enhanced single-qubit gate application with native complex SIMD
pub fn apply_single_qubit_gate_complex_simd(
    matrix: &[Complex64; 4],
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    let len = in_amps0.len();

    // Convert to SIMD vectors
    let a0_simd = ComplexSimdVector::from_slice(in_amps0);
    let a1_simd = ComplexSimdVector::from_slice(in_amps1);

    let mut result0_simd = ComplexSimdVector::new(len);
    let mut result1_simd = ComplexSimdVector::new(len);

    // Temporary vectors for intermediate results
    let mut temp0_simd = ComplexSimdVector::new(len);
    let mut temp1_simd = ComplexSimdVector::new(len);

    // Compute: out_a0 = matrix[0] * a0 + matrix[1] * a1
    ComplexSimdOps::complex_scalar_mul_simd(&a0_simd, matrix[0], &mut temp0_simd);
    ComplexSimdOps::complex_scalar_mul_simd(&a1_simd, matrix[1], &mut temp1_simd);
    ComplexSimdOps::complex_add_simd(&temp0_simd, &temp1_simd, &mut result0_simd);

    // Compute: out_a1 = matrix[2] * a0 + matrix[3] * a1
    ComplexSimdOps::complex_scalar_mul_simd(&a0_simd, matrix[2], &mut temp0_simd);
    ComplexSimdOps::complex_scalar_mul_simd(&a1_simd, matrix[3], &mut temp1_simd);
    ComplexSimdOps::complex_add_simd(&temp0_simd, &temp1_simd, &mut result1_simd);

    // Convert back to Complex64 arrays
    let result0 = result0_simd.to_complex_vec();
    let result1 = result1_simd.to_complex_vec();

    out_amps0.copy_from_slice(&result0);
    out_amps1.copy_from_slice(&result1);
}

/// Enhanced Hadamard gate with complex SIMD optimizations
pub fn apply_hadamard_gate_complex_simd(
    in_amps0: &[Complex64],
    in_amps1: &[Complex64],
    out_amps0: &mut [Complex64],
    out_amps1: &mut [Complex64],
) {
    let len = in_amps0.len();
    let sqrt2_inv = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);

    let a0_simd = ComplexSimdVector::from_slice(in_amps0);
    let a1_simd = ComplexSimdVector::from_slice(in_amps1);

    let mut sum_simd = ComplexSimdVector::new(len);
    let mut diff_simd = ComplexSimdVector::new(len);
    let mut result0_simd = ComplexSimdVector::new(len);
    let mut result1_simd = ComplexSimdVector::new(len);

    // Hadamard: out_a0 = (a0 + a1) / sqrt(2), out_a1 = (a0 - a1) / sqrt(2)
    ComplexSimdOps::complex_add_simd(&a0_simd, &a1_simd, &mut sum_simd);
    ComplexSimdOps::complex_sub_simd(&a0_simd, &a1_simd, &mut diff_simd);

    ComplexSimdOps::complex_scalar_mul_simd(&sum_simd, sqrt2_inv, &mut result0_simd);
    ComplexSimdOps::complex_scalar_mul_simd(&diff_simd, sqrt2_inv, &mut result1_simd);

    let result0 = result0_simd.to_complex_vec();
    let result1 = result1_simd.to_complex_vec();

    out_amps0.copy_from_slice(&result0);
    out_amps1.copy_from_slice(&result1);
}

/// Optimized CNOT gate for multiple qubit pairs using complex SIMD
pub fn apply_cnot_complex_simd(
    state: &mut [Complex64],
    control_qubit: usize,
    target_qubit: usize,
    num_qubits: usize,
) {
    let dim = 1 << num_qubits;
    let control_mask = 1 << control_qubit;
    let target_mask = 1 << target_qubit;

    // Process state in SIMD chunks where possible
    let chunk_size = ComplexSimdVector::detect_simd_width();
    let num_chunks = dim / (chunk_size * 2); // Process pairs of indices

    for chunk in 0..num_chunks {
        let base_idx = chunk * chunk_size * 2;
        let mut chunk_data = vec![Complex64::new(0.0, 0.0); chunk_size * 2];

        // Collect indices that need swapping
        let mut swap_indices = Vec::new();
        for i in 0..chunk_size {
            let idx = base_idx + i;
            if idx < dim && (idx & control_mask) != 0 {
                let swapped_idx = idx ^ target_mask;
                if swapped_idx < dim {
                    swap_indices.push((idx, swapped_idx));
                    chunk_data[i * 2] = state[idx];
                    chunk_data[i * 2 + 1] = state[swapped_idx];
                }
            }
        }

        // Apply swaps using SIMD operations
        if !swap_indices.is_empty() {
            let chunk_simd = ComplexSimdVector::from_slice(&chunk_data);
            for (i, (idx, swapped_idx)) in swap_indices.iter().enumerate() {
                state[*idx] = chunk_simd.to_complex_vec()[i * 2 + 1];
                state[*swapped_idx] = chunk_simd.to_complex_vec()[i * 2];
            }
        }
    }

    // Handle remaining elements with scalar operations
    let remaining_start = num_chunks * chunk_size * 2;
    for i in remaining_start..dim {
        if (i & control_mask) != 0 {
            let swapped_i = i ^ target_mask;
            if swapped_i < dim {
                let temp = state[i];
                state[i] = state[swapped_i];
                state[swapped_i] = temp;
            }
        }
    }
}

/// Performance benchmarking for complex SIMD operations
pub fn benchmark_complex_simd_operations() -> std::collections::HashMap<String, f64> {
    use std::time::Instant;
    let mut results = std::collections::HashMap::new();

    // Test vector sizes
    let sizes = vec![1024, 4096, 16384, 65536];

    for &size in &sizes {
        let a = ComplexSimdVector::from_slice(&vec![Complex64::new(1.0, 0.5); size]);
        let b = ComplexSimdVector::from_slice(&vec![Complex64::new(0.5, 1.0); size]);
        let mut c = ComplexSimdVector::new(size);

        // Benchmark complex multiplication
        let start = Instant::now();
        for _ in 0..1000 {
            ComplexSimdOps::complex_mul_simd(&a, &b, &mut c);
        }
        let mul_time = start.elapsed().as_nanos() as f64 / 1000.0;
        results.insert(format!("complex_mul_simd_{}", size), mul_time);

        // Benchmark complex addition
        let start = Instant::now();
        for _ in 0..1000 {
            ComplexSimdOps::complex_add_simd(&a, &b, &mut c);
        }
        let add_time = start.elapsed().as_nanos() as f64 / 1000.0;
        results.insert(format!("complex_add_simd_{}", size), add_time);

        // Benchmark scalar multiplication
        let scalar = Complex64::new(2.0, 1.0);
        let start = Instant::now();
        for _ in 0..1000 {
            ComplexSimdOps::complex_scalar_mul_simd(&a, scalar, &mut c);
        }
        let scalar_mul_time = start.elapsed().as_nanos() as f64 / 1000.0;
        results.insert(format!("complex_scalar_mul_simd_{}", size), scalar_mul_time);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn test_complex_simd_vector_creation() {
        let data = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
        ];

        let simd_vec = ComplexSimdVector::from_slice(&data);
        assert_eq!(simd_vec.len(), 3);

        let result = simd_vec.to_complex_vec();
        for (i, &expected) in data.iter().enumerate() {
            assert!((result[i] - expected).norm() < 1e-10);
        }
    }

    #[test]
    fn test_complex_multiplication_simd() {
        let a = ComplexSimdVector::from_slice(&vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
        ]);
        let b = ComplexSimdVector::from_slice(&vec![
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
        ]);
        let mut c = ComplexSimdVector::new(2);

        ComplexSimdOps::complex_mul_simd(&a, &b, &mut c);
        let result = c.to_complex_vec();

        // Verify: (1+2i)*(5+6i) = 5+6i+10i-12 = -7+16i
        let expected0 = Complex64::new(-7.0, 16.0);
        assert!((result[0] - expected0).norm() < 1e-10);

        // Verify: (3+4i)*(7+8i) = 21+24i+28i-32 = -11+52i
        let expected1 = Complex64::new(-11.0, 52.0);
        assert!((result[1] - expected1).norm() < 1e-10);
    }

    #[test]
    fn test_hadamard_gate_complex_simd() {
        let in_amps0 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let in_amps1 = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        let mut out_amps0 = vec![Complex64::new(0.0, 0.0); 2];
        let mut out_amps1 = vec![Complex64::new(0.0, 0.0); 2];

        apply_hadamard_gate_complex_simd(&in_amps0, &in_amps1, &mut out_amps0, &mut out_amps1);

        // H|0⟩ = (|0⟩ + |1⟩)/√2
        let expected = Complex64::new(FRAC_1_SQRT_2, 0.0);
        assert!((out_amps0[0] - expected).norm() < 1e-10);
        assert!((out_amps1[0] - expected).norm() < 1e-10);

        // H|1⟩ = (|0⟩ - |1⟩)/√2
        assert!((out_amps0[1] - expected).norm() < 1e-10);
        assert!((out_amps1[1] - (-expected)).norm() < 1e-10);
    }

    #[test]
    fn test_single_qubit_gate_complex_simd() {
        // X gate matrix
        let x_matrix = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let in_amps0 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)];
        let in_amps1 = vec![Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)];
        let mut out_amps0 = vec![Complex64::new(0.0, 0.0); 2];
        let mut out_amps1 = vec![Complex64::new(0.0, 0.0); 2];

        apply_single_qubit_gate_complex_simd(
            &x_matrix,
            &in_amps0,
            &in_amps1,
            &mut out_amps0,
            &mut out_amps1,
        );

        // X gate should swap amplitudes
        assert!((out_amps0[0] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((out_amps1[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((out_amps0[1] - Complex64::new(0.5, 0.0)).norm() < 1e-10);
        assert!((out_amps1[1] - Complex64::new(0.5, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_complex_norm_squared_simd() {
        let data = vec![
            Complex64::new(3.0, 4.0), // |3+4i|² = 9+16 = 25
            Complex64::new(1.0, 1.0), // |1+i|² = 1+1 = 2
        ];

        let simd_vec = ComplexSimdVector::from_slice(&data);
        let norms = ComplexSimdOps::complex_norm_squared_simd(&simd_vec);

        assert!((norms[0] - 25.0).abs() < 1e-10);
        assert!((norms[1] - 2.0).abs() < 1e-10);
    }
}
