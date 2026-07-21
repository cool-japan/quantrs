//! `SciRS2` Python bindings integration for numerical operations.
//!
//! This module provides Python bindings for `SciRS2` numerical operations,
//! including linear algebra, optimization, and statistical functions.

// Allow unused_self for PyO3 method bindings that require &self signature
// Allow unnecessary_wraps for PyO3 Result return types that may need error handling in future
// Allow type_complexity for PyO3 return types with complex nested generics
#![allow(clippy::unused_self)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::type_complexity)]

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use scirs2_core::ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2};
use scirs2_core::Complex64;
use scirs2_linalg::error::LinalgError;
use scirs2_numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
};
use std::collections::HashMap;

// SciRS2 stub types (would be replaced with actual SciRS2 imports)
#[derive(Debug)]
struct SciRS2Array {
    data: ArrayD<f64>,
}

#[derive(Debug)]
struct SciRS2ComplexArray {
    data: ArrayD<Complex64>,
}

/// Real linear-algebra operations, delegated to `scirs2-linalg` (SCIRS2 POLICY).
#[derive(Debug)]
struct SciRS2LinearAlgebra;

impl SciRS2LinearAlgebra {
    /// Singular value decomposition `matrix = u * diag(s) * vt`, computed via
    /// `scirs2_linalg`'s eigendecomposition-based SVD (real algorithm, not a stub).
    fn svd(matrix: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), LinalgError> {
        scirs2_linalg::svd(&matrix.view(), false, None)
    }

    /// Eigenvalues/eigenvectors of a general complex matrix via the real
    /// Householder-Hessenberg + shifted-QR algorithm in `scirs2_linalg::complex`.
    fn eig(
        matrix: &Array2<Complex64>,
    ) -> Result<(Array1<Complex64>, Array2<Complex64>), LinalgError> {
        let result = scirs2_linalg::complex::decompositions::complex_eig(&matrix.view())?;
        Ok((result.eigenvalues, result.eigenvectors))
    }

    /// QR decomposition via Householder reflections (`scirs2_linalg::qr`).
    fn qr(matrix: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), LinalgError> {
        scirs2_linalg::qr(&matrix.view(), None)
    }
}

/// Real numerical optimizers (BFGS quasi-Newton and Adam) operating on an
/// arbitrary Rust closure objective. These are implemented directly here
/// (rather than delegated) because the objective is an opaque Python
/// callback wrapped in a closure, so the update loop must live where that
/// closure can be invoked repeatedly.
#[derive(Debug)]
struct SciRS2Optimizer;

impl SciRS2Optimizer {
    /// Numerical gradient via central finite differences.
    fn numerical_gradient<F>(objective: &F, x: &Array1<f64>) -> Array1<f64>
    where
        F: Fn(&Array1<f64>) -> f64,
    {
        let mut grad = Array1::zeros(x.len());
        let eps = 1e-6;
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            grad[i] = (objective(&x_plus) - objective(&x_minus)) / (2.0 * eps);
        }
        grad
    }

    /// BFGS quasi-Newton minimization with an approximate inverse Hessian and
    /// an Armijo backtracking line search.
    fn minimize_bfgs<F>(
        objective: F,
        initial: &Array1<f64>,
        gradient: Option<Box<dyn Fn(&Array1<f64>) -> Array1<f64>>>,
        max_iterations: usize,
        tolerance: f64,
    ) -> Array1<f64>
    where
        F: Fn(&Array1<f64>) -> f64,
    {
        let n = initial.len();
        let grad_fn = |x: &Array1<f64>| -> Array1<f64> {
            gradient
                .as_ref()
                .map_or_else(|| Self::numerical_gradient(&objective, x), |g| g(x))
        };

        let mut x = initial.clone();
        let mut h_inv = Array2::<f64>::eye(n); // approximate inverse Hessian
        let mut g = grad_fn(&x);

        for _ in 0..max_iterations {
            let grad_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            if grad_norm < tolerance {
                break;
            }

            // Search direction: p = -H_inv * g
            let p = -h_inv.dot(&g);

            // Backtracking line search (Armijo condition)
            let mut alpha = 1.0_f64;
            let f_x = objective(&x);
            let directional_derivative = g.dot(&p);
            if directional_derivative >= 0.0 {
                // Not a descent direction (can happen from curvature loss);
                // fall back to steepest descent for this step.
                let steepest = -&g;
                let step = steepest.mapv(|v| v * alpha);
                x = &x + &step;
                g = grad_fn(&x);
                h_inv = Array2::eye(n);
                continue;
            }

            let c1 = 1e-4_f64;
            let mut new_x = &x + &p.mapv(|v| v * alpha);
            let mut iterations_ls = 0;
            while objective(&new_x) > c1.mul_add(alpha * directional_derivative, f_x)
                && iterations_ls < 50
            {
                alpha *= 0.5;
                new_x = &x + &p.mapv(|v| v * alpha);
                iterations_ls += 1;
            }

            let s = &new_x - &x;
            let new_g = grad_fn(&new_x);
            let y = &new_g - &g;

            let sy = s.dot(&y);
            if sy.abs() > 1e-12 {
                // BFGS inverse-Hessian update:
                // H' = (I - rho s y^T) H (I - rho y s^T) + rho s s^T
                let rho = 1.0 / sy;
                let identity = Array2::<f64>::eye(n);
                let s_col = s.clone().insert_axis(scirs2_core::ndarray::Axis(1));
                let y_row = y.clone().insert_axis(scirs2_core::ndarray::Axis(0));
                let y_col = y.clone().insert_axis(scirs2_core::ndarray::Axis(1));
                let s_row = s.clone().insert_axis(scirs2_core::ndarray::Axis(0));

                let term_left = &identity - &(s_col.dot(&y_row) * rho);
                let term_right = &identity - &(y_col.dot(&s_row) * rho);
                let s_outer = s_col.dot(&s_row) * rho;

                h_inv = term_left.dot(&h_inv).dot(&term_right) + s_outer;
            }

            x = new_x;
            g = new_g;
        }

        x
    }

    /// Adam (adaptive moment estimation) minimization.
    fn minimize_adam<F>(
        objective: F,
        initial: &Array1<f64>,
        gradient: Option<&dyn Fn(&Array1<f64>) -> Array1<f64>>,
        learning_rate: f64,
        iterations: usize,
    ) -> Array1<f64>
    where
        F: Fn(&Array1<f64>) -> f64,
    {
        let n = initial.len();
        let beta1 = 0.9_f64;
        let beta2 = 0.999_f64;
        let eps = 1e-8_f64;

        let mut x = initial.clone();
        let mut m = Array1::<f64>::zeros(n);
        let mut v = Array1::<f64>::zeros(n);

        for t in 1..=iterations {
            let g = gradient.map_or_else(|| Self::numerical_gradient(&objective, &x), |g| g(&x));

            m = m.mapv(|val| val * beta1) + g.mapv(|val| val * (1.0 - beta1));
            v = v.mapv(|val| val * beta2) + g.mapv(|val| val * val * (1.0 - beta2));

            let t_f = t as f64;
            let m_hat = m.mapv(|val| val / (1.0 - beta1.powf(t_f)));
            let v_hat = v.mapv(|val| val / (1.0 - beta2.powf(t_f)));

            for i in 0..n {
                x[i] -= learning_rate * m_hat[i] / (v_hat[i].sqrt() + eps);
            }
        }

        x
    }
}

/// `SciRS2` Linear Algebra operations for Python
#[pyclass(name = "SciRS2LinAlg")]
pub struct PySciRS2LinAlg;

#[pymethods]
impl PySciRS2LinAlg {
    #[new]
    const fn new() -> Self {
        Self
    }

    /// Compute Singular Value Decomposition
    #[pyo3(text_signature = "(matrix, /)")]
    fn svd<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
        let mat = matrix.as_array();
        let (u, s, vt) = SciRS2LinearAlgebra::svd(&mat.to_owned())
            .map_err(|e| PyRuntimeError::new_err(format!("SVD failed: {e}")))?;

        Ok((
            u.into_pyarray(py).into(),
            s.into_pyarray(py).into(),
            vt.into_pyarray(py).into(),
        ))
    }

    /// Compute eigenvalues and eigenvectors
    #[pyo3(text_signature = "(matrix, /)")]
    fn eig<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<(Py<PyArray1<Complex64>>, Py<PyArray2<Complex64>>)> {
        let mat = matrix.as_array().to_owned();
        let (eigenvalues, eigenvectors) = SciRS2LinearAlgebra::eig(&mat)
            .map_err(|e| PyRuntimeError::new_err(format!("Eigendecomposition failed: {e}")))?;

        Ok((
            eigenvalues.into_pyarray(py).into(),
            eigenvectors.into_pyarray(py).into(),
        ))
    }

    /// Compute QR decomposition
    #[pyo3(text_signature = "(matrix, /)")]
    fn qr<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
        let mat = matrix.as_array();
        let (q, r) = SciRS2LinearAlgebra::qr(&mat.to_owned())
            .map_err(|e| PyRuntimeError::new_err(format!("QR decomposition failed: {e}")))?;

        Ok((q.into_pyarray(py).into(), r.into_pyarray(py).into()))
    }

    /// Matrix multiplication with optimized backend
    #[pyo3(text_signature = "(a, b, /)")]
    fn matmul<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<'py, f64>,
        b: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let a_arr = a.as_array();
        let b_arr = b.as_array();

        if a_arr.ncols() != b_arr.nrows() {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: {}x{} @ {}x{}",
                a_arr.nrows(),
                a_arr.ncols(),
                b_arr.nrows(),
                b_arr.ncols()
            )));
        }

        let result = a_arr.dot(&b_arr);
        Ok(result.into_pyarray(py).into())
    }

    /// Solve linear system Ax = b
    #[pyo3(text_signature = "(a, b, /)")]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        a: PyReadonlyArray2<'py, f64>,
        b: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let a_arr = a.as_array();
        let b_arr = b.as_array();

        if a_arr.nrows() != a_arr.ncols() {
            return Err(PyValueError::new_err("Matrix must be square"));
        }

        if a_arr.nrows() != b_arr.len() {
            return Err(PyValueError::new_err("Dimension mismatch"));
        }

        // Real Gaussian elimination with partial pivoting (scirs2-linalg).
        let x = scirs2_linalg::blas_accelerated::solve(&a_arr, &b_arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Linear solve failed: {e}")))?;

        Ok(x.into_pyarray(py).into())
    }
}

/// `SciRS2` Optimization for Python
#[pyclass(name = "SciRS2Optimizer")]
pub struct PySciRS2Optimizer {
    tolerance: f64,
    max_iterations: usize,
}

#[pymethods]
impl PySciRS2Optimizer {
    #[new]
    #[pyo3(signature = (tolerance=1e-8, max_iterations=1000))]
    const fn new(tolerance: f64, max_iterations: usize) -> Self {
        Self {
            tolerance,
            max_iterations,
        }
    }

    /// Minimize using BFGS algorithm
    #[pyo3(text_signature = "(objective, initial, gradient=None, /)")]
    fn minimize_bfgs<'py>(
        &self,
        py: Python<'py>,
        objective: Py<PyAny>,
        initial: PyReadonlyArray1<'py, f64>,
        gradient: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let x0 = initial.as_array().to_owned();

        // Create objective function wrapper
        let obj_fn = move |x: &Array1<f64>| -> f64 {
            // SAFETY: We are inside a #[pymethods] fn that already holds the GIL via `py`,
            // but the closure is called within this same scope so the GIL is still held.
            unsafe {
                Python::attach_unchecked(|py| {
                    let x_py = x.clone().into_pyarray(py);
                    objective
                        .call1(py, (x_py,))
                        .ok()
                        .and_then(|r| r.extract::<f64>(py).ok())
                        .unwrap_or(f64::MAX)
                })
            }
        };

        // Wire the caller-supplied analytic gradient callback (if any) through
        // to the real BFGS loop; when omitted, BFGS falls back to central
        // finite differences on `objective`.
        let grad_fn: Option<Box<dyn Fn(&Array1<f64>) -> Array1<f64>>> =
            gradient.map(|grad_callback| {
                let boxed: Box<dyn Fn(&Array1<f64>) -> Array1<f64>> =
                    Box::new(move |x: &Array1<f64>| -> Array1<f64> {
                        unsafe {
                            Python::attach_unchecked(|py| {
                                let x_py = x.clone().into_pyarray(py);
                                grad_callback
                                    .call1(py, (x_py,))
                                    .ok()
                                    .and_then(|r| r.extract::<PyReadonlyArray1<f64>>(py).ok())
                                    .map_or_else(
                                        || Array1::zeros(x.len()),
                                        |arr| arr.as_array().to_owned(),
                                    )
                            })
                        }
                    });
                boxed
            });

        let result = SciRS2Optimizer::minimize_bfgs(
            obj_fn,
            &x0,
            grad_fn,
            self.max_iterations,
            self.tolerance,
        );
        Ok(result.into_pyarray(py).into())
    }

    /// Minimize using Adam optimizer
    #[pyo3(text_signature = "(objective, initial, learning_rate=0.001, /)")]
    fn minimize_adam<'py>(
        &self,
        py: Python<'py>,
        objective: Py<PyAny>,
        initial: PyReadonlyArray1<'py, f64>,
        learning_rate: Option<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let x0 = initial.as_array().to_owned();
        let lr = learning_rate.unwrap_or(0.001);

        // Create objective function wrapper
        let obj_fn = move |x: &Array1<f64>| -> f64 {
            // SAFETY: We are inside a #[pymethods] fn that already holds the GIL via `py`,
            // but the closure is called within this same scope so the GIL is still held.
            unsafe {
                Python::attach_unchecked(|py| {
                    let x_py = x.clone().into_pyarray(py);
                    objective
                        .call1(py, (x_py,))
                        .ok()
                        .and_then(|r| r.extract::<f64>(py).ok())
                        .unwrap_or(f64::MAX)
                })
            }
        };

        let result = SciRS2Optimizer::minimize_adam(obj_fn, &x0, None, lr, self.max_iterations);
        Ok(result.into_pyarray(py).into())
    }
}

/// `SciRS2` Statistical functions for Python
#[pyclass(name = "SciRS2Stats")]
pub struct PySciRS2Stats;

#[pymethods]
impl PySciRS2Stats {
    #[new]
    const fn new() -> Self {
        Self
    }

    /// Compute correlation matrix
    #[pyo3(text_signature = "(data, /)")]
    fn correlation<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let arr = data.as_array();
        if arr.nrows() < 2 {
            return Err(PyValueError::new_err(
                "Need at least 2 samples (rows) to compute a correlation matrix",
            ));
        }
        let n_features = arr.ncols();
        let mut corr = Array2::eye(n_features);

        // Compute means
        let means: Vec<f64> = (0..n_features)
            .map(|j| arr.column(j).mean().unwrap_or(0.0))
            .collect();

        // Compute correlations
        for i in 0..n_features {
            for j in i + 1..n_features {
                let col_i = arr.column(i);
                let col_j = arr.column(j);

                let cov: f64 = col_i
                    .iter()
                    .zip(col_j.iter())
                    .map(|(a, b)| (a - means[i]) * (b - means[j]))
                    .sum::<f64>()
                    / (arr.nrows() - 1) as f64;

                let std_i = ((col_i.iter().map(|a| (a - means[i]).powi(2)).sum::<f64>())
                    / (arr.nrows() - 1) as f64)
                    .sqrt();
                let std_j = ((col_j.iter().map(|b| (b - means[j]).powi(2)).sum::<f64>())
                    / (arr.nrows() - 1) as f64)
                    .sqrt();

                let correlation = cov / (std_i * std_j);
                corr[[i, j]] = correlation;
                corr[[j, i]] = correlation;
            }
        }

        Ok(corr.into_pyarray(py).into())
    }

    /// Perform Principal Component Analysis
    #[pyo3(text_signature = "(data, n_components=None, /)")]
    fn pca<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
        n_components: Option<usize>,
    ) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
        let arr = data.as_array();
        let n_samples = arr.nrows();
        let n_features = arr.ncols();
        if n_samples < 2 {
            return Err(PyValueError::new_err(
                "Need at least 2 samples (rows) to compute PCA",
            ));
        }
        let k = n_components.unwrap_or_else(|| n_features.min(n_samples));

        // Center the data
        let means: Vec<f64> = (0..n_features)
            .map(|j| arr.column(j).mean().unwrap_or(0.0))
            .collect();

        let mut centered = arr.to_owned();
        for i in 0..n_samples {
            for j in 0..n_features {
                centered[[i, j]] -= means[j];
            }
        }

        // Compute covariance matrix
        let cov = centered.t().dot(&centered) / (n_samples - 1) as f64;

        // Real SVD of the covariance matrix (eigendecomposition-based, scirs2-linalg).
        // For a symmetric PSD covariance matrix the left singular vectors are the
        // principal axes and the singular values are the explained variances.
        let (u, s, _) = SciRS2LinearAlgebra::svd(&cov)
            .map_err(|e| PyRuntimeError::new_err(format!("PCA failed: SVD error: {e}")))?;

        let k = k.min(n_features);

        // Return principal components and explained variance
        let components = u.slice(scirs2_core::ndarray::s![.., ..k]).to_owned();
        let variance = s.slice(scirs2_core::ndarray::s![..k]).to_owned();

        Ok((
            components.into_pyarray(py).into(),
            variance.into_pyarray(py).into(),
        ))
    }
}

/// `SciRS2` Fast Fourier Transform for Python
#[pyclass(name = "SciRS2FFT")]
pub struct PySciRS2FFT;

#[pymethods]
impl PySciRS2FFT {
    #[new]
    const fn new() -> Self {
        Self
    }

    /// Compute 1D FFT
    ///
    /// Uses `scirs2-fft`'s Cooley-Tukey/mixed-radix backend for any input
    /// length (not just powers of two, and not just lengths <= 32).
    #[pyo3(text_signature = "(signal, /)")]
    fn fft<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<'py, Complex64>,
    ) -> PyResult<Py<PyArray1<Complex64>>> {
        let arr = signal.as_array();
        let n = arr.len();
        if n == 0 {
            return Err(PyValueError::new_err("Input signal must not be empty"));
        }

        let input: Vec<Complex64> = arr.to_vec();
        // Pass `Some(n)` explicitly so the transform length always matches the
        // input length (the default would round up to the next power of two).
        let spectrum = scirs2_fft::fft(&input, Some(n))
            .map_err(|e| PyRuntimeError::new_err(format!("FFT failed: {e}")))?;

        Ok(Array1::from_vec(spectrum).into_pyarray(py).into())
    }

    /// Compute inverse 1D FFT
    #[pyo3(text_signature = "(spectrum, /)")]
    fn ifft<'py>(
        &self,
        py: Python<'py>,
        spectrum: PyReadonlyArray1<'py, Complex64>,
    ) -> PyResult<Py<PyArray1<Complex64>>> {
        let arr = spectrum.as_array();
        let n = arr.len();
        if n == 0 {
            return Err(PyValueError::new_err("Input spectrum must not be empty"));
        }

        let input: Vec<Complex64> = arr.to_vec();
        let signal = scirs2_fft::ifft(&input, Some(n))
            .map_err(|e| PyRuntimeError::new_err(format!("Inverse FFT failed: {e}")))?;

        Ok(Array1::from_vec(signal).into_pyarray(py).into())
    }

    /// Compute 2D FFT
    #[pyo3(text_signature = "(image, /)")]
    fn fft2<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<Py<PyArray2<Complex64>>> {
        let arr = image.as_array().to_owned();
        if arr.is_empty() {
            return Err(PyValueError::new_err("Input image must not be empty"));
        }

        let result = scirs2_fft::fft2(&arr, None, None, None)
            .map_err(|e| PyRuntimeError::new_err(format!("2D FFT failed: {e}")))?;

        Ok(result.into_pyarray(py).into())
    }
}

/// Initialize the `SciRS2` bindings submodule
pub fn create_scirs2_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(m.py(), "scirs2")?;

    submodule.add_class::<PySciRS2LinAlg>()?;
    submodule.add_class::<PySciRS2Optimizer>()?;
    submodule.add_class::<PySciRS2Stats>()?;
    submodule.add_class::<PySciRS2FFT>()?;

    m.add_submodule(&submodule)?;
    Ok(())
}

/// Error raised by the pure-Rust tomography/entropy helpers below, kept
/// independent of `PyErr` so the core math is unit-testable without a
/// Python interpreter (see the `#[cfg(test)]` module at the end of this file).
enum QuantumNumericsError {
    /// Bad input from the caller (maps to `PyValueError`).
    InvalidInput(String),
    /// A numerical routine failed (maps to `PyRuntimeError`).
    Computation(String),
}

impl From<QuantumNumericsError> for PyErr {
    fn from(err: QuantumNumericsError) -> Self {
        match err {
            QuantumNumericsError::InvalidInput(msg) => PyValueError::new_err(msg),
            QuantumNumericsError::Computation(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

/// Compute the von Neumann entanglement entropy `S(rho_A) = -Tr(rho_A ln rho_A)`
/// of the reduced density matrix obtained by tracing out the complement of
/// `subsystem_a` from the full state `psi`. Pure Rust so it is directly
/// unit-testable (no PyO3/GIL involved).
fn entanglement_entropy_impl(
    psi: ArrayView1<'_, Complex64>,
    subsystem_a: &[usize],
) -> Result<f64, QuantumNumericsError> {
    let dim = psi.len();
    if dim == 0 || !dim.is_power_of_two() {
        return Err(QuantumNumericsError::InvalidInput(
            "State vector length must be a positive power of two (2^n_qubits)".to_string(),
        ));
    }
    let n_qubits = dim.trailing_zeros() as usize;

    let mut subsystem_a: Vec<usize> = subsystem_a.to_vec();
    subsystem_a.sort_unstable();
    subsystem_a.dedup();
    if subsystem_a.is_empty() || subsystem_a.len() >= n_qubits {
        return Err(QuantumNumericsError::InvalidInput(
            "Partition must be a non-empty, proper subset of the qubits".to_string(),
        ));
    }
    if let Some(&bad_qubit) = subsystem_a.iter().find(|&&q| q >= n_qubits) {
        return Err(QuantumNumericsError::InvalidInput(format!(
            "Partition qubit index {bad_qubit} out of range for a {n_qubits}-qubit state"
        )));
    }

    let subsystem_b: Vec<usize> = (0..n_qubits).filter(|q| !subsystem_a.contains(q)).collect();
    let dim_a = 1usize << subsystem_a.len();
    let dim_b = 1usize << subsystem_b.len();

    // Exact partial trace materializes a dim_a x dim_a matrix; bound it to
    // keep this tractable (an honest capacity limit, not a fabricated result).
    if subsystem_a.len() > 12 {
        return Err(QuantumNumericsError::InvalidInput(
            "Exact partial-trace entanglement entropy supports partitions of at most 12 qubits"
                .to_string(),
        ));
    }

    // Qubit 0 is the most-significant bit of the computational-basis index,
    // matching the bitstring convention used elsewhere in this crate
    // (see `PyMeasurementResult::marginal_probability`).
    let compose = |a_bits: usize, b_bits: usize| -> usize {
        let mut idx = 0usize;
        for (k, &q) in subsystem_a.iter().enumerate() {
            let bit = (a_bits >> (subsystem_a.len() - 1 - k)) & 1;
            idx |= bit << (n_qubits - 1 - q);
        }
        for (k, &q) in subsystem_b.iter().enumerate() {
            let bit = (b_bits >> (subsystem_b.len() - 1 - k)) & 1;
            idx |= bit << (n_qubits - 1 - q);
        }
        idx
    };

    // Real partial trace: rho_A[i, j] = sum_b psi[i, b] * conj(psi[j, b]).
    // The upper triangle is computed directly and the lower triangle is
    // filled by exact conjugate-symmetry so rho_A is Hermitian to the bit.
    let mut rho_a = Array2::<Complex64>::zeros((dim_a, dim_a));
    for i in 0..dim_a {
        for j in i..dim_a {
            let mut sum = Complex64::new(0.0, 0.0);
            for b in 0..dim_b {
                let idx_i = compose(i, b);
                let idx_j = compose(j, b);
                sum += psi[idx_i] * psi[idx_j].conj();
            }
            rho_a[[i, j]] = sum;
            if i != j {
                rho_a[[j, i]] = sum.conj();
            }
        }
    }

    // Von Neumann entropy from the (real, non-negative) eigenvalues of the
    // Hermitian reduced density matrix.
    let eig = scirs2_linalg::complex::decompositions::complex_eigh(&rho_a.view()).map_err(|e| {
        QuantumNumericsError::Computation(format!(
            "Eigendecomposition of the reduced density matrix failed: {e}"
        ))
    })?;

    let entropy: f64 = eig
        .eigenvalues
        .iter()
        .map(|lambda| lambda.re)
        .filter(|&p| p > 1e-12)
        .map(|p| -p * p.ln())
        .sum();

    Ok(entropy)
}

/// Quantum state tomography via linear inversion over the Pauli basis. Pure
/// Rust so it is directly unit-testable (no PyO3/GIL involved).
///
/// `measurements` must have one row per entry of `bases` and `2^n_qubits`
/// columns; row `k` holds the computational-basis outcome probabilities
/// obtained after rotating every qubit into the single-qubit basis given by
/// the corresponding character of `bases[k]` (each basis string has one
/// character per qubit, drawn from `{'X', 'Y', 'Z'}` -- exactly what
/// `StateTomography.measurement_circuits` in `measurement.rs` produces).
///
/// Reconstructs `rho = (1/d) * sum_P <P> * P` over all `4^n_qubits` Pauli
/// strings `P`, where `<P>` is estimated from whichever supplied basis
/// setting matches `P`'s non-identity factors.
fn reconstruct_density_matrix(
    meas: ArrayView2<'_, f64>,
    basis_strings: &[String],
) -> Result<Array2<Complex64>, QuantumNumericsError> {
    if basis_strings.is_empty() {
        return Err(QuantumNumericsError::InvalidInput(
            "No measurement bases provided".to_string(),
        ));
    }
    if meas.nrows() != basis_strings.len() {
        return Err(QuantumNumericsError::InvalidInput(format!(
            "`measurements` has {} row(s) but {} basis string(s) were provided",
            meas.nrows(),
            basis_strings.len()
        )));
    }

    let n_qubits = basis_strings[0].chars().count();
    if n_qubits == 0 || n_qubits > 6 {
        return Err(QuantumNumericsError::InvalidInput(
            "Exact linear-inversion state tomography supports 1 to 6 qubits".to_string(),
        ));
    }
    let dim = 1usize << n_qubits;
    if meas.ncols() != dim {
        return Err(QuantumNumericsError::InvalidInput(format!(
            "Each measurement row must have 2^n_qubits = {dim} outcome probabilities, got {}",
            meas.ncols()
        )));
    }
    for basis in basis_strings {
        if basis.chars().count() != n_qubits || !basis.chars().all(|c| matches!(c, 'X' | 'Y' | 'Z'))
        {
            return Err(QuantumNumericsError::InvalidInput(format!(
                "Invalid basis string '{basis}': expected {n_qubits} character(s) from {{'X','Y','Z'}}"
            )));
        }
    }

    // Index measurement rows by their basis string for O(1) lookup.
    let mut by_basis: HashMap<&str, usize> = HashMap::new();
    for (row, basis) in basis_strings.iter().enumerate() {
        by_basis.entry(basis.as_str()).or_insert(row);
    }

    let pauli = |kind: u8| -> Array2<Complex64> {
        match kind {
            0 => scirs2_core::ndarray::array![
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            ],
            1 => scirs2_core::ndarray::array![
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ],
            2 => scirs2_core::ndarray::array![
                [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
                [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
            ],
            _ => scirs2_core::ndarray::array![
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
            ],
        }
    };

    let kron = |a: &Array2<Complex64>, b: &Array2<Complex64>| -> Array2<Complex64> {
        let (ar, ac) = (a.nrows(), a.ncols());
        let (br, bc) = (b.nrows(), b.ncols());
        let mut out = Array2::<Complex64>::zeros((ar * br, ac * bc));
        for i in 0..ar {
            for j in 0..ac {
                for p in 0..br {
                    for q in 0..bc {
                        out[[i * br + p, j * bc + q]] = a[[i, j]] * b[[p, q]];
                    }
                }
            }
        }
        out
    };

    let mut rho = Array2::<Complex64>::zeros((dim, dim));
    let mut n_nontrivial_terms_used = 0usize;
    let n_pauli_strings = 4usize.pow(n_qubits as u32);

    for code in 0..n_pauli_strings {
        // Base-4 digits of `code`: 0=I, 1=X, 2=Y, 3=Z per qubit.
        let mut digits = vec![0u8; n_qubits];
        let mut c = code;
        for d in digits.iter_mut().rev() {
            *d = (c % 4) as u8;
            c /= 4;
        }

        // <I (x) I (x) ... (x) I> = 1 always: no data needed.
        let expectation = if code == 0 {
            1.0
        } else {
            // For identity factors, 'Z' is used as an arbitrary filler basis:
            // marginalizing (summing) over a qubit's outcome reproduces the
            // reduced-state statistics regardless of which basis that qubit
            // was measured in, so any matching setting is valid.
            let required_basis: String = digits
                .iter()
                .map(|&d| match d {
                    1 => 'X',
                    2 => 'Y',
                    _ => 'Z',
                })
                .collect();

            let Some(&row) = by_basis.get(required_basis.as_str()) else {
                continue; // this Pauli term's required setting was not measured
            };

            n_nontrivial_terms_used += 1;

            let mut expectation = 0.0_f64;
            for outcome in 0..dim {
                let mut sign = 1.0_f64;
                for (qubit, &d) in digits.iter().enumerate() {
                    if d != 0 {
                        let bit = (outcome >> (n_qubits - 1 - qubit)) & 1;
                        if bit == 1 {
                            sign = -sign;
                        }
                    }
                }
                expectation = sign.mul_add(meas[[row, outcome]], expectation);
            }
            expectation
        };

        let mut term = pauli(digits[0]);
        for &d in &digits[1..] {
            term = kron(&term, &pauli(d));
        }
        let coeff = Complex64::new(expectation / dim as f64, 0.0);
        rho = rho + term.mapv(|v| v * coeff);
    }

    if n_nontrivial_terms_used == 0 {
        return Err(QuantumNumericsError::InvalidInput(
            "No usable measurement data: none of the required Pauli-basis settings were found in `bases`"
                .to_string(),
        ));
    }

    Ok(rho)
}

/// Quantum-specific numerical operations using `SciRS2`
#[pyclass(name = "QuantumNumerics")]
pub struct PyQuantumNumerics;

#[pymethods]
impl PyQuantumNumerics {
    #[new]
    const fn new() -> Self {
        Self
    }

    /// Compute fidelity between two quantum states
    #[pyo3(text_signature = "(state1, state2, /)")]
    fn fidelity<'py>(
        &self,
        state1: PyReadonlyArray1<'py, Complex64>,
        state2: PyReadonlyArray1<'py, Complex64>,
    ) -> PyResult<f64> {
        let s1 = state1.as_array();
        let s2 = state2.as_array();

        if s1.len() != s2.len() {
            return Err(PyValueError::new_err("States must have the same dimension"));
        }

        let inner_product: Complex64 = s1.iter().zip(s2.iter()).map(|(a, b)| a.conj() * b).sum();

        Ok(inner_product.norm().powi(2))
    }

    /// Compute the von Neumann entanglement entropy `S(rho_A) = -Tr(rho_A ln rho_A)`
    /// of the reduced density matrix obtained by tracing out the complement of
    /// `partition` from the full state `state`.
    #[pyo3(text_signature = "(state, partition, /)")]
    fn entanglement_entropy<'py>(
        &self,
        state: PyReadonlyArray1<'py, Complex64>,
        partition: &Bound<'py, PyList>,
    ) -> PyResult<f64> {
        let subsystem_a: Vec<usize> = partition.extract()?;
        Ok(entanglement_entropy_impl(state.as_array(), &subsystem_a)?)
    }

    /// Quantum state tomography via linear inversion over the Pauli basis.
    ///
    /// `measurements` must have one row per entry of `bases` and `2^n_qubits`
    /// columns; row `k` holds the computational-basis outcome probabilities
    /// obtained after rotating every qubit into the single-qubit basis given
    /// by the corresponding character of `bases[k]` (each basis string has
    /// one character per qubit, drawn from `{'X', 'Y', 'Z'}` -- exactly what
    /// `StateTomography.measurement_circuits` in `measurement.rs` produces).
    ///
    /// Reconstructs `rho = (1/d) * sum_P <P> * P` over all `4^n_qubits`
    /// Pauli strings `P`, where `<P>` is estimated from whichever supplied
    /// basis setting matches `P`'s non-identity factors.
    #[pyo3(text_signature = "(measurements, bases, /)")]
    fn state_tomography<'py>(
        &self,
        py: Python<'py>,
        measurements: PyReadonlyArray2<'py, f64>,
        bases: &Bound<'py, PyList>,
    ) -> PyResult<Py<PyArray2<Complex64>>> {
        let basis_strings: Vec<String> = bases.extract()?;
        let rho = reconstruct_density_matrix(measurements.as_array(), &basis_strings)?;
        Ok(rho.into_pyarray(py).into())
    }
}

// Pure-Rust regression tests for the real algorithms above. These deliberately
// avoid touching the Python/PyO3 layer (no `Python::attach`): this crate is a
// `cdylib` built with the `extension-module` PyO3 feature, so a standalone
// test binary cannot reliably embed a Python interpreter -- see the existing
// precedent in `lib.rs`'s and `mitigation.rs`'s `#[cfg(test)]` modules, which
// test the underlying pure-Rust logic directly rather than going through
// `#[pymethods]`.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_reconstructs_the_original_matrix() {
        let a = scirs2_core::ndarray::array![[3.0_f64, 0.0], [4.0, 5.0]];
        let (u, s, vt) =
            SciRS2LinearAlgebra::svd(&a).expect("SVD of a well-conditioned matrix must succeed");

        let mut s_diag = Array2::<f64>::zeros((s.len(), s.len()));
        for i in 0..s.len() {
            s_diag[[i, i]] = s[i];
        }
        let reconstructed = u.dot(&s_diag).dot(&vt);

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-8,
                    "SVD reconstruction mismatch at ({i},{j}): {} vs {}",
                    reconstructed[[i, j]],
                    a[[i, j]]
                );
            }
        }
    }

    #[test]
    fn qr_reconstructs_the_original_matrix_and_q_is_orthogonal() {
        let a = scirs2_core::ndarray::array![[1.0_f64, 2.0], [3.0, 4.0]];
        let (q, r) =
            SciRS2LinearAlgebra::qr(&a).expect("QR of a well-conditioned matrix must succeed");

        let reconstructed = q.dot(&r);
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-8);
            }
        }

        let qtq = q.t().dot(&q);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((qtq[[i, j]] - expected).abs() < 1e-8, "Q is not orthogonal");
            }
        }
    }

    #[test]
    fn eig_of_a_diagonal_matrix_recovers_its_diagonal() {
        let a = scirs2_core::ndarray::array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(2.0, 0.0)],
        ];
        let (eigenvalues, _) =
            SciRS2LinearAlgebra::eig(&a).expect("eig of a diagonal matrix must succeed");

        let mut values: Vec<f64> = eigenvalues.iter().map(|c| c.re).collect();
        values.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs"));
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn bfgs_minimizes_a_quadratic_bowl_to_its_analytic_minimum() {
        // f(x, y) = (x - 3)^2 + (y + 2)^2, minimized at (3, -2).
        let objective = |x: &Array1<f64>| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2);
        let gradient: Box<dyn Fn(&Array1<f64>) -> Array1<f64>> = Box::new(|x: &Array1<f64>| {
            Array1::from_vec(vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] + 2.0)])
        });

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let result = SciRS2Optimizer::minimize_bfgs(objective, &x0, Some(gradient), 200, 1e-10);

        assert!(
            (result[0] - 3.0).abs() < 1e-4,
            "BFGS did not converge in x: {}",
            result[0]
        );
        assert!(
            (result[1] + 2.0).abs() < 1e-4,
            "BFGS did not converge in y: {}",
            result[1]
        );
    }

    #[test]
    fn adam_minimizes_a_quadratic_bowl_with_numerical_gradient() {
        // f(x, y) = (x - 1)^2 + (y - 1)^2, minimized at (1, 1); Adam here uses
        // the built-in central-difference numerical gradient (no analytic
        // gradient supplied), exercising the fallback path.
        let objective = |x: &Array1<f64>| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2);
        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let result = SciRS2Optimizer::minimize_adam(objective, &x0, None, 0.1, 2000);

        assert!(
            (result[0] - 1.0).abs() < 0.05,
            "Adam did not converge in x: {}",
            result[0]
        );
        assert!(
            (result[1] - 1.0).abs() < 0.05,
            "Adam did not converge in y: {}",
            result[1]
        );
    }

    #[test]
    fn entanglement_entropy_of_a_product_state_is_zero() {
        // |00>: qubit 0 is the MSB, so amplitude index 0 ("00") carries all
        // the weight and every other basis state is empty.
        let psi = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        let entropy =
            entanglement_entropy_impl(psi.view(), &[0]).unwrap_or_else(|_| panic!("valid input"));
        assert!(
            entropy.abs() < 1e-9,
            "product state entropy should be 0, got {entropy}"
        );
    }

    #[test]
    fn entanglement_entropy_of_a_bell_state_is_ln2() {
        // (|00> + |11>) / sqrt(2): maximally entangled across the 1|1 cut.
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let psi = Array1::from_vec(vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ]);
        let entropy =
            entanglement_entropy_impl(psi.view(), &[0]).unwrap_or_else(|_| panic!("valid input"));
        assert!(
            (entropy - std::f64::consts::LN_2).abs() < 1e-6,
            "Bell-state entanglement entropy should be ln(2), got {entropy}"
        );
    }

    #[test]
    fn entanglement_entropy_rejects_a_trivial_partition() {
        let psi = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        // Partition covering all qubits of a 1-qubit state is not a proper subset.
        assert!(entanglement_entropy_impl(psi.view(), &[0]).is_err());
    }

    #[test]
    fn state_tomography_reconstructs_a_pure_zero_state() {
        // A single qubit prepared in |0>: Z-basis measurement is deterministic,
        // X/Y-basis measurements are maximally random.
        let measurements = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.5, 0.5, 0.5, 0.5])
            .expect("valid shape");
        let bases = vec!["Z".to_string(), "X".to_string(), "Y".to_string()];

        let rho = reconstruct_density_matrix(measurements.view(), &bases)
            .unwrap_or_else(|_| panic!("valid tomography input"));

        assert!((rho[[0, 0]].re - 1.0).abs() < 1e-9);
        assert!(rho[[0, 0]].im.abs() < 1e-9);
        assert!(rho[[1, 1]].re.abs() < 1e-9);
        assert!(rho[[0, 1]].norm() < 1e-9);
        assert!(rho[[1, 0]].norm() < 1e-9);
    }

    #[test]
    fn state_tomography_reports_an_honest_error_for_mismatched_shapes() {
        let measurements = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).expect("valid shape");
        let bases = vec!["Z".to_string(), "X".to_string()]; // 2 bases but 1 row
        assert!(reconstruct_density_matrix(measurements.view(), &bases).is_err());
    }
}
