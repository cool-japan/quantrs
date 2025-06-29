//! Temporary SIMD operations stubs to replace scirs2_core::simd_ops
//! TODO: Replace with scirs2_core when regex dependency issue is fixed

use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;

/// Trait for SIMD operations on f64
pub trait SimdF64 {
    fn simd_add(self, other: f64) -> f64;
    fn simd_sub(self, other: f64) -> f64;
    fn simd_mul(self, other: f64) -> f64;
    fn simd_scalar_mul(view: &ArrayView1<f64>, scalar: f64) -> Array1<f64>;
    fn simd_add_arrays(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64>;
    fn simd_sub_arrays(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64>;
    fn simd_mul_arrays(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64>;
    fn simd_sum(slice: &[f64]) -> f64;
    fn simd_sum_array(a: &ArrayView1<f64>) -> f64;
}

impl SimdF64 for f64 {
    #[inline]
    fn simd_add(self, other: f64) -> f64 {
        self + other
    }

    #[inline]
    fn simd_sub(self, other: f64) -> f64 {
        self - other
    }

    #[inline]
    fn simd_mul(self, other: f64) -> f64 {
        self * other
    }

    #[inline]
    fn simd_scalar_mul(view: &ArrayView1<f64>, scalar: f64) -> Array1<f64> {
        view.mapv(|x| x * scalar)
    }

    #[inline]
    fn simd_add_arrays(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        a + b
    }

    #[inline]
    fn simd_sub_arrays(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        a - b
    }

    #[inline]
    fn simd_mul_arrays(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        a * b
    }

    #[inline]
    fn simd_sum(slice: &[f64]) -> f64 {
        slice.iter().sum()
    }

    #[inline]
    fn simd_sum_array(a: &ArrayView1<f64>) -> f64 {
        a.sum()
    }
}

/// Trait for SIMD operations on Complex64
pub trait SimdComplex64 {
    fn simd_add(self, other: Complex64) -> Complex64;
    fn simd_sub(self, other: Complex64) -> Complex64;
    fn simd_mul(self, other: Complex64) -> Complex64;
    fn simd_scalar_mul(self, scalar: Complex64) -> Complex64;
    fn simd_sum(slice: &[Complex64]) -> Complex64;
}

impl SimdComplex64 for Complex64 {
    #[inline]
    fn simd_add(self, other: Complex64) -> Complex64 {
        self + other
    }

    #[inline]
    fn simd_sub(self, other: Complex64) -> Complex64 {
        self - other
    }

    #[inline]
    fn simd_mul(self, other: Complex64) -> Complex64 {
        self * other
    }

    #[inline]
    fn simd_scalar_mul(self, scalar: Complex64) -> Complex64 {
        self * scalar
    }

    #[inline]
    fn simd_sum(slice: &[Complex64]) -> Complex64 {
        slice.iter().sum()
    }
}
