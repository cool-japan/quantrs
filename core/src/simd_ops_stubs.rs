//! SIMD-like batch operations stubs replacing scirs2_core::simd_ops
//!
//! Implements explicit loop-unrolled vectorized operations on f64 and Complex64
//! arrays. Uses 4-wide manual unrolling to exploit CPU instruction-level
//! parallelism without requiring nightly std::simd.

use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::Complex64;

/// Lane width used for manual loop unrolling (matches common SIMD register width)
const UNROLL: usize = 4;

/// Trait for SIMD-like batch operations on f64
pub trait SimdF64 {
    fn simd_add(self, other: f64) -> f64;
    fn simd_sub(self, other: f64) -> f64;
    fn simd_mul(self, other: f64) -> f64;
    fn simd_scalar_mul(view: &ArrayView1<f64>, scalar: f64) -> Array1<f64>;
    fn simd_add_arrays(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64>;
    fn simd_sub_arrays(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64>;
    fn simd_mul_arrays(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64>;
    fn simd_dot(a: &[f64], b: &[f64]) -> f64;
    fn simd_sum(slice: &[f64]) -> f64;
    fn simd_sum_array(a: &ArrayView1<f64>) -> f64;
    fn simd_max(a: &[f64]) -> f64;
    fn simd_min(a: &[f64]) -> f64;
    fn simd_fmadd(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64>;
}

impl SimdF64 for f64 {
    #[inline(always)]
    fn simd_add(self, other: f64) -> f64 {
        self + other
    }

    #[inline(always)]
    fn simd_sub(self, other: f64) -> f64 {
        self - other
    }

    #[inline(always)]
    fn simd_mul(self, other: f64) -> f64 {
        self * other
    }

    /// Scalar-multiply every element, 4-wide unrolled
    #[inline]
    fn simd_scalar_mul(view: &ArrayView1<f64>, scalar: f64) -> Array1<f64> {
        let n = view.len();
        let slice = view.as_slice().unwrap_or(&[]);

        // Fast path: contiguous memory — unrolled loop
        if !slice.is_empty() {
            let mut out = vec![0.0f64; n];
            let chunks = n / UNROLL;
            let rem = n % UNROLL;
            let base = chunks * UNROLL;

            for i in 0..chunks {
                let j = i * UNROLL;
                out[j] = slice[j] * scalar;
                out[j + 1] = slice[j + 1] * scalar;
                out[j + 2] = slice[j + 2] * scalar;
                out[j + 3] = slice[j + 3] * scalar;
            }
            for k in 0..rem {
                out[base + k] = slice[base + k] * scalar;
            }
            return Array1::from(out);
        }

        // Non-contiguous fallback
        view.mapv(|x| x * scalar)
    }

    /// Element-wise addition, 4-wide unrolled
    #[inline]
    fn simd_add_arrays(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(a.len(), b.len(), "simd_add_arrays: length mismatch");
        let n = a.len();

        match (a.as_slice(), b.as_slice()) {
            (Some(sa), Some(sb)) => {
                let mut out = vec![0.0f64; n];
                let chunks = n / UNROLL;
                let rem = n % UNROLL;
                let base = chunks * UNROLL;

                for i in 0..chunks {
                    let j = i * UNROLL;
                    out[j] = sa[j] + sb[j];
                    out[j + 1] = sa[j + 1] + sb[j + 1];
                    out[j + 2] = sa[j + 2] + sb[j + 2];
                    out[j + 3] = sa[j + 3] + sb[j + 3];
                }
                for k in 0..rem {
                    out[base + k] = sa[base + k] + sb[base + k];
                }
                Array1::from(out)
            }
            _ => a + b,
        }
    }

    /// Element-wise subtraction, 4-wide unrolled
    #[inline]
    fn simd_sub_arrays(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(a.len(), b.len(), "simd_sub_arrays: length mismatch");
        let n = a.len();

        match (a.as_slice(), b.as_slice()) {
            (Some(sa), Some(sb)) => {
                let mut out = vec![0.0f64; n];
                let chunks = n / UNROLL;
                let rem = n % UNROLL;
                let base = chunks * UNROLL;

                for i in 0..chunks {
                    let j = i * UNROLL;
                    out[j] = sa[j] - sb[j];
                    out[j + 1] = sa[j + 1] - sb[j + 1];
                    out[j + 2] = sa[j + 2] - sb[j + 2];
                    out[j + 3] = sa[j + 3] - sb[j + 3];
                }
                for k in 0..rem {
                    out[base + k] = sa[base + k] - sb[base + k];
                }
                Array1::from(out)
            }
            _ => a - b,
        }
    }

    /// Element-wise multiplication, 4-wide unrolled
    #[inline]
    fn simd_mul_arrays(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(a.len(), b.len(), "simd_mul_arrays: length mismatch");
        let n = a.len();

        match (a.as_slice(), b.as_slice()) {
            (Some(sa), Some(sb)) => {
                let mut out = vec![0.0f64; n];
                let chunks = n / UNROLL;
                let rem = n % UNROLL;
                let base = chunks * UNROLL;

                for i in 0..chunks {
                    let j = i * UNROLL;
                    out[j] = sa[j] * sb[j];
                    out[j + 1] = sa[j + 1] * sb[j + 1];
                    out[j + 2] = sa[j + 2] * sb[j + 2];
                    out[j + 3] = sa[j + 3] * sb[j + 3];
                }
                for k in 0..rem {
                    out[base + k] = sa[base + k] * sb[base + k];
                }
                Array1::from(out)
            }
            _ => a * b,
        }
    }

    /// Dot product with 4-wide accumulator unrolling to reduce dependency chains
    #[inline]
    fn simd_dot(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "simd_dot: length mismatch");
        let n = a.len();
        let chunks = n / UNROLL;
        let rem = n % UNROLL;
        let base = chunks * UNROLL;

        // Four independent accumulators — breaks scalar dependency chain
        let mut acc0 = 0.0f64;
        let mut acc1 = 0.0f64;
        let mut acc2 = 0.0f64;
        let mut acc3 = 0.0f64;

        for i in 0..chunks {
            let j = i * UNROLL;
            acc0 += a[j] * b[j];
            acc1 += a[j + 1] * b[j + 1];
            acc2 += a[j + 2] * b[j + 2];
            acc3 += a[j + 3] * b[j + 3];
        }

        let mut tail = acc0 + acc1 + acc2 + acc3;
        for k in 0..rem {
            tail += a[base + k] * b[base + k];
        }
        tail
    }

    /// Horizontal sum with 4-wide accumulator unrolling
    #[inline]
    fn simd_sum(slice: &[f64]) -> f64 {
        let n = slice.len();
        let chunks = n / UNROLL;
        let rem = n % UNROLL;
        let base = chunks * UNROLL;

        let mut acc0 = 0.0f64;
        let mut acc1 = 0.0f64;
        let mut acc2 = 0.0f64;
        let mut acc3 = 0.0f64;

        for i in 0..chunks {
            let j = i * UNROLL;
            acc0 += slice[j];
            acc1 += slice[j + 1];
            acc2 += slice[j + 2];
            acc3 += slice[j + 3];
        }

        let mut total = acc0 + acc1 + acc2 + acc3;
        for k in 0..rem {
            total += slice[base + k];
        }
        total
    }

    #[inline]
    fn simd_sum_array(a: &ArrayView1<f64>) -> f64 {
        match a.as_slice() {
            Some(s) => <f64 as SimdF64>::simd_sum(s),
            None => a.sum(),
        }
    }

    /// Maximum value with 4-wide unrolled comparison
    #[inline]
    fn simd_max(a: &[f64]) -> f64 {
        if a.is_empty() {
            return f64::NEG_INFINITY;
        }
        let n = a.len();
        let chunks = n / UNROLL;
        let rem = n % UNROLL;
        let base = chunks * UNROLL;

        let mut m0 = f64::NEG_INFINITY;
        let mut m1 = f64::NEG_INFINITY;
        let mut m2 = f64::NEG_INFINITY;
        let mut m3 = f64::NEG_INFINITY;

        for i in 0..chunks {
            let j = i * UNROLL;
            m0 = m0.max(a[j]);
            m1 = m1.max(a[j + 1]);
            m2 = m2.max(a[j + 2]);
            m3 = m3.max(a[j + 3]);
        }

        let mut max = m0.max(m1).max(m2).max(m3);
        for k in 0..rem {
            max = max.max(a[base + k]);
        }
        max
    }

    /// Minimum value with 4-wide unrolled comparison
    #[inline]
    fn simd_min(a: &[f64]) -> f64 {
        if a.is_empty() {
            return f64::INFINITY;
        }
        let n = a.len();
        let chunks = n / UNROLL;
        let rem = n % UNROLL;
        let base = chunks * UNROLL;

        let mut m0 = f64::INFINITY;
        let mut m1 = f64::INFINITY;
        let mut m2 = f64::INFINITY;
        let mut m3 = f64::INFINITY;

        for i in 0..chunks {
            let j = i * UNROLL;
            m0 = m0.min(a[j]);
            m1 = m1.min(a[j + 1]);
            m2 = m2.min(a[j + 2]);
            m3 = m3.min(a[j + 3]);
        }

        let mut min = m0.min(m1).min(m2).min(m3);
        for k in 0..rem {
            min = min.min(a[base + k]);
        }
        min
    }

    /// Fused multiply-add: out\[i\] = a\[i\]\*b\[i\] + c\[i\], 4-wide unrolled
    #[inline]
    fn simd_fmadd(a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
        let n = a.len();
        assert_eq!(n, b.len(), "simd_fmadd: a/b length mismatch");
        assert_eq!(n, c.len(), "simd_fmadd: a/c length mismatch");

        let mut out = vec![0.0f64; n];
        let chunks = n / UNROLL;
        let rem = n % UNROLL;
        let base = chunks * UNROLL;

        for i in 0..chunks {
            let j = i * UNROLL;
            out[j] = a[j] * b[j] + c[j];
            out[j + 1] = a[j + 1] * b[j + 1] + c[j + 1];
            out[j + 2] = a[j + 2] * b[j + 2] + c[j + 2];
            out[j + 3] = a[j + 3] * b[j + 3] + c[j + 3];
        }
        for k in 0..rem {
            out[base + k] = a[base + k] * b[base + k] + c[base + k];
        }
        out
    }
}

/// Trait for SIMD-like batch operations on Complex64
pub trait SimdComplex64 {
    fn simd_add(self, other: Complex64) -> Complex64;
    fn simd_sub(self, other: Complex64) -> Complex64;
    fn simd_mul(self, other: Complex64) -> Complex64;
    fn simd_scalar_mul(self, scalar: Complex64) -> Complex64;
    fn simd_dot(a: &[Complex64], b: &[Complex64]) -> Complex64;
    fn simd_sum(slice: &[Complex64]) -> Complex64;
    fn simd_sum_array(a: &ArrayView1<Complex64>) -> Complex64;
}

impl SimdComplex64 for Complex64 {
    #[inline(always)]
    fn simd_add(self, other: Complex64) -> Complex64 {
        self + other
    }

    #[inline(always)]
    fn simd_sub(self, other: Complex64) -> Complex64 {
        self - other
    }

    #[inline(always)]
    fn simd_mul(self, other: Complex64) -> Complex64 {
        self * other
    }

    #[inline(always)]
    fn simd_scalar_mul(self, scalar: Complex64) -> Complex64 {
        self * scalar
    }

    /// Complex dot product with 4-wide unrolled accumulation
    #[inline]
    fn simd_dot(a: &[Complex64], b: &[Complex64]) -> Complex64 {
        assert_eq!(a.len(), b.len(), "simd_dot complex: length mismatch");
        let n = a.len();
        let chunks = n / UNROLL;
        let rem = n % UNROLL;
        let base = chunks * UNROLL;

        let zero = Complex64::new(0.0, 0.0);
        let mut acc0 = zero;
        let mut acc1 = zero;
        let mut acc2 = zero;
        let mut acc3 = zero;

        for i in 0..chunks {
            let j = i * UNROLL;
            acc0 += a[j] * b[j];
            acc1 += a[j + 1] * b[j + 1];
            acc2 += a[j + 2] * b[j + 2];
            acc3 += a[j + 3] * b[j + 3];
        }

        let mut total = acc0 + acc1 + acc2 + acc3;
        for k in 0..rem {
            total += a[base + k] * b[base + k];
        }
        total
    }

    /// Horizontal sum with 4-wide accumulator unrolling
    #[inline]
    fn simd_sum(slice: &[Complex64]) -> Complex64 {
        let n = slice.len();
        let chunks = n / UNROLL;
        let rem = n % UNROLL;
        let base = chunks * UNROLL;

        let zero = Complex64::new(0.0, 0.0);
        let mut acc0 = zero;
        let mut acc1 = zero;
        let mut acc2 = zero;
        let mut acc3 = zero;

        for i in 0..chunks {
            let j = i * UNROLL;
            acc0 += slice[j];
            acc1 += slice[j + 1];
            acc2 += slice[j + 2];
            acc3 += slice[j + 3];
        }

        let mut total = acc0 + acc1 + acc2 + acc3;
        for k in 0..rem {
            total += slice[base + k];
        }
        total
    }

    #[inline]
    fn simd_sum_array(a: &ArrayView1<Complex64>) -> Complex64 {
        match a.as_slice() {
            Some(s) => <Complex64 as SimdComplex64>::simd_sum(s),
            None => a.sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_simd_dot_basic() {
        let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = <f64 as SimdF64>::simd_dot(&a, &b);
        let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(
            (result - expected).abs() < 1e-12,
            "simd_dot mismatch: {result} vs {expected}"
        );
    }

    #[test]
    fn test_simd_sum_unrolled() {
        let data: Vec<f64> = (0..17).map(|i| i as f64).collect();
        let result = <f64 as SimdF64>::simd_sum(&data);
        let expected: f64 = data.iter().sum();
        assert!((result - expected).abs() < 1e-12);
    }

    #[test]
    fn test_simd_fmadd() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0f64, 2.0, 2.0, 2.0, 2.0];
        let c = vec![0.5f64, 0.5, 0.5, 0.5, 0.5];
        let result = <f64 as SimdF64>::simd_fmadd(&a, &b, &c);
        let expected: Vec<f64> = a
            .iter()
            .zip(b.iter())
            .zip(c.iter())
            .map(|((ai, bi), ci)| ai * bi + ci)
            .collect();
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-12);
        }
    }

    #[test]
    fn test_simd_add_arrays_unrolled() {
        let a = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = array![9.0f64, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let result = <f64 as SimdF64>::simd_add_arrays(&a.view(), &b.view());
        for v in result.iter() {
            assert!((v - 10.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_simd_max_min() {
        let data = vec![3.0f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
        assert!(((<f64 as SimdF64>::simd_max(&data)) - 9.0).abs() < 1e-12);
        assert!(((<f64 as SimdF64>::simd_min(&data)) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_complex_simd_dot() {
        let a = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.5, 0.5),
        ];
        let b = a.clone();
        let result = <Complex64 as SimdComplex64>::simd_dot(&a, &b);
        let expected: Complex64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result.re - expected.re).abs() < 1e-12);
        assert!((result.im - expected.im).abs() < 1e-12);
    }
}
