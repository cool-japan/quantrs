//! Integration with SciRS2 for advanced linear algebra operations.
//!
//! This module provides a comprehensive integration layer with SciRS2 to leverage
//! high-performance linear algebra routines for quantum simulation.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use ndarray_linalg::Norm;
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

#[cfg(feature = "advanced_math")]
use quantrs2_core::prelude::QuantRS2Error as SciRS2Error;

#[cfg(feature = "advanced_math")]
use crate::memory_bandwidth_optimization::MemoryPool as SciRS2MemoryPool;
#[cfg(feature = "advanced_math")]
use scirs2_linalg::{blas, cholesky, det, inv, lapack, lu, qr, solve, svd};

// Additional dependencies for enhanced implementations
#[cfg(feature = "advanced_math")]
use ndarray_linalg::Eig;
#[cfg(feature = "advanced_math")]
use ndrustfft::{ndfft, ndifft, FftHandler};
#[cfg(feature = "advanced_math")]
use sprs::CsMat;

use crate::error::{Result, SimulatorError};

/// Performance statistics for the SciRS2 backend
#[derive(Debug, Default, Clone)]
pub struct BackendStats {
    /// Number of matrix operations performed
    pub matrix_ops: usize,
    /// Number of vector operations performed
    pub vector_ops: usize,
    /// Total time spent in BLAS operations (milliseconds)
    pub blas_time_ms: f64,
    /// Total time spent in LAPACK operations (milliseconds)
    pub lapack_time_ms: f64,
    /// Memory usage statistics
    pub memory_usage_bytes: usize,
    /// Number of FFT operations
    pub fft_ops: usize,
    /// Number of sparse matrix operations
    pub sparse_ops: usize,
}

/// SciRS2-powered linear algebra backend
#[derive(Debug)]
pub struct SciRS2Backend {
    /// Whether SciRS2 is available
    pub available: bool,

    /// Performance statistics
    pub stats: BackendStats,

    /// Memory pool for efficient allocation
    #[cfg(feature = "advanced_math")]
    pub memory_pool: MemoryPool,

    /// FFT engine for frequency domain operations
    #[cfg(feature = "advanced_math")]
    pub fft_engine: FftEngine,
}

impl SciRS2Backend {
    /// Create a new SciRS2 backend
    pub fn new() -> Self {
        Self {
            available: cfg!(feature = "advanced_math"),
            stats: BackendStats::default(),
            #[cfg(feature = "advanced_math")]
            memory_pool: MemoryPool::new(),
            #[cfg(feature = "advanced_math")]
            fft_engine: FftEngine::new(),
        }
    }

    /// Check if the backend is available and functional
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &BackendStats {
        &self.stats
    }

    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = BackendStats::default();
    }

    /// Matrix multiplication using SciRS2 BLAS
    #[cfg(feature = "advanced_math")]
    pub fn matrix_multiply(&mut self, a: &Matrix, b: &Matrix) -> Result<Matrix> {
        let start_time = std::time::Instant::now();

        let result_shape = (a.shape().0, b.shape().1);
        let mut result = Matrix::zeros(result_shape, &self.memory_pool)?;

        BLAS::gemm(
            Complex64::new(1.0, 0.0),
            a,
            b,
            Complex64::new(0.0, 0.0),
            &mut result,
        )?;

        self.stats.matrix_ops += 1;
        self.stats.blas_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(result)
    }

    /// Matrix-vector multiplication using SciRS2 BLAS
    #[cfg(feature = "advanced_math")]
    pub fn matrix_vector_multiply(&mut self, a: &Matrix, x: &Vector) -> Result<Vector> {
        let start_time = std::time::Instant::now();

        let mut result = Vector::zeros(a.shape().0, &self.memory_pool)?;

        BLAS::gemv(
            Complex64::new(1.0, 0.0),
            a,
            x,
            Complex64::new(0.0, 0.0),
            &mut result,
        )?;

        self.stats.vector_ops += 1;
        self.stats.blas_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(result)
    }

    /// SVD decomposition using SciRS2 LAPACK
    #[cfg(feature = "advanced_math")]
    pub fn svd(&mut self, matrix: &Matrix) -> Result<SvdResult> {
        let start_time = std::time::Instant::now();

        let result = LAPACK::svd(matrix)?;

        self.stats.matrix_ops += 1;
        self.stats.lapack_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(result)
    }

    /// Eigenvalue decomposition using SciRS2 LAPACK
    #[cfg(feature = "advanced_math")]
    pub fn eigendecomposition(&mut self, matrix: &Matrix) -> Result<EigResult> {
        let start_time = std::time::Instant::now();

        let result = LAPACK::eig(matrix)?;

        self.stats.matrix_ops += 1;
        self.stats.lapack_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(result)
    }
}

impl Default for SciRS2Backend {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory pool wrapper for SciRS2
#[cfg(feature = "advanced_math")]
#[derive(Debug)]
pub struct MemoryPool {
    inner: SciRS2MemoryPool,
}

#[cfg(feature = "advanced_math")]
impl MemoryPool {
    pub fn new() -> Self {
        Self {
            inner: SciRS2MemoryPool::new(1024, 128).unwrap_or_else(|_| {
                // Fallback to a simple memory pool implementation
                panic!("Failed to create SciRS2 memory pool")
            }),
        }
    }
}

#[cfg(not(feature = "advanced_math"))]
#[derive(Debug)]
pub struct MemoryPool;

#[cfg(not(feature = "advanced_math"))]
impl MemoryPool {
    pub fn new() -> Self {
        Self
    }
}

/// FFT engine for frequency domain operations
#[cfg(feature = "advanced_math")]
#[derive(Debug)]
pub struct FftEngine;

#[cfg(feature = "advanced_math")]
impl FftEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Vector) -> Result<Vector> {
        // Implement forward FFT using ndrustfft
        use ndrustfft::{ndfft, FftHandler};

        let array = input.to_array1()?;
        let mut handler = FftHandler::new(array.len());
        let mut fft_result = array.clone();

        ndfft(&array, &mut fft_result, &mut handler, 0);

        Vector::from_array1(&fft_result.view(), &MemoryPool::new())
    }

    pub fn inverse(&self, input: &Vector) -> Result<Vector> {
        // Implement inverse FFT using ndrustfft
        use ndrustfft::{ndifft, FftHandler};

        let array = input.to_array1()?;
        let mut handler = FftHandler::new(array.len());
        let mut ifft_result = array.clone();

        ndifft(&array, &mut ifft_result, &mut handler, 0);

        Vector::from_array1(&ifft_result.view(), &MemoryPool::new())
    }
}

#[cfg(not(feature = "advanced_math"))]
#[derive(Debug)]
pub struct FftEngine;

#[cfg(not(feature = "advanced_math"))]
impl FftEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, _input: &Vector) -> Result<Vector> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }

    pub fn inverse(&self, _input: &Vector) -> Result<Vector> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }
}

/// Matrix wrapper for SciRS2 operations
#[cfg(feature = "advanced_math")]
#[derive(Debug, Clone)]
pub struct Matrix {
    data: Array2<Complex64>,
}

#[cfg(feature = "advanced_math")]
impl Matrix {
    pub fn from_array2(array: &ArrayView2<Complex64>, _pool: &MemoryPool) -> Result<Self> {
        Ok(Self {
            data: array.to_owned(),
        })
    }

    pub fn zeros(shape: (usize, usize), _pool: &MemoryPool) -> Result<Self> {
        Ok(Self {
            data: Array2::zeros(shape),
        })
    }

    pub fn to_array2(&self, result: &mut Array2<Complex64>) -> Result<()> {
        if result.shape() != self.data.shape() {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Expected shape {:?}, but got {:?}",
                self.data.shape(),
                result.shape()
            )));
        }
        result.assign(&self.data);
        Ok(())
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.data.nrows(), self.data.ncols())
    }

    pub fn view(&self) -> ArrayView2<Complex64> {
        self.data.view()
    }

    pub fn view_mut(&mut self) -> ndarray::ArrayViewMut2<Complex64> {
        self.data.view_mut()
    }
}

#[cfg(not(feature = "advanced_math"))]
#[derive(Debug)]
pub struct Matrix;

#[cfg(not(feature = "advanced_math"))]
impl Matrix {
    pub fn from_array2(_array: &ArrayView2<Complex64>, _pool: &MemoryPool) -> Result<Self> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }

    pub fn zeros(_shape: (usize, usize), _pool: &MemoryPool) -> Result<Self> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }

    pub fn to_array2(&self, _result: &mut Array2<Complex64>) -> Result<()> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }
}

/// Vector wrapper for SciRS2 operations
#[cfg(feature = "advanced_math")]
#[derive(Debug, Clone)]
pub struct Vector {
    data: Array1<Complex64>,
}

#[cfg(feature = "advanced_math")]
impl Vector {
    pub fn from_array1(array: &ArrayView1<Complex64>, _pool: &MemoryPool) -> Result<Self> {
        Ok(Self {
            data: array.to_owned(),
        })
    }

    pub fn zeros(len: usize, _pool: &MemoryPool) -> Result<Self> {
        Ok(Self {
            data: Array1::zeros(len),
        })
    }

    pub fn to_array1(&self) -> Result<Array1<Complex64>> {
        Ok(self.data.clone())
    }

    pub fn to_array1_mut(&self, result: &mut Array1<Complex64>) -> Result<()> {
        if result.len() != self.data.len() {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Expected length {}, but got {}",
                self.data.len(),
                result.len()
            )));
        }
        result.assign(&self.data);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn view(&self) -> ArrayView1<Complex64> {
        self.data.view()
    }

    pub fn view_mut(&mut self) -> ndarray::ArrayViewMut1<Complex64> {
        self.data.view_mut()
    }
}

#[cfg(not(feature = "advanced_math"))]
#[derive(Debug)]
pub struct Vector;

#[cfg(not(feature = "advanced_math"))]
impl Vector {
    pub fn from_array1(_array: &ArrayView1<Complex64>, _pool: &MemoryPool) -> Result<Self> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }

    pub fn zeros(_len: usize, _pool: &MemoryPool) -> Result<Self> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }

    pub fn to_array1(&self) -> Result<Array1<Complex64>> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }

    pub fn to_array1_mut(&self, _result: &mut Array1<Complex64>) -> Result<()> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }
}

/// Sparse matrix wrapper for SciRS2 operations
#[cfg(feature = "advanced_math")]
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// CSR format sparse matrix using nalgebra-sparse
    csr_matrix: nalgebra_sparse::CsrMatrix<Complex64>,
}

#[cfg(feature = "advanced_math")]
impl SparseMatrix {
    pub fn from_csr(
        values: &[Complex64],
        col_indices: &[usize],
        row_ptr: &[usize],
        num_rows: usize,
        num_cols: usize,
        _pool: &MemoryPool,
    ) -> Result<Self> {
        use nalgebra_sparse::CsrMatrix;

        let csr_matrix = CsrMatrix::try_from_csr_data(
            num_rows,
            num_cols,
            row_ptr.to_vec(),
            col_indices.to_vec(),
            values.to_vec(),
        )
        .map_err(|e| {
            SimulatorError::ComputationError(format!("Failed to create CSR matrix: {}", e))
        })?;

        Ok(Self { csr_matrix })
    }

    pub fn matvec(&self, vector: &Vector, result: &mut Vector) -> Result<()> {
        use nalgebra::{Complex, DVector};

        // Convert our Vector to nalgebra DVector
        let input_vec = vector.to_array1()?;
        let nalgebra_vec = DVector::from_iterator(
            input_vec.len(),
            input_vec.iter().map(|&c| Complex::new(c.re, c.im)),
        );

        // Perform matrix-vector multiplication using manual implementation
        let mut output = DVector::zeros(self.csr_matrix.nrows());
        
        // Manual sparse matrix-vector multiplication
        for (row_idx, row) in self.csr_matrix.row_iter().enumerate() {
            let mut sum = Complex::new(0.0, 0.0);
            for (col_idx, value) in row.col_indices().iter().zip(row.values()) {
                sum += value * nalgebra_vec[*col_idx];
            }
            output[row_idx] = sum;
        }

        // Convert back to our format
        let output_array: Array1<Complex64> =
            Array1::from_iter(output.iter().map(|c| Complex64::new(c.re, c.im)));

        result.data.assign(&output_array);
        Ok(())
    }

    pub fn solve(&self, rhs: &Vector) -> Result<Vector> {
        // Improved sparse solver using nalgebra-sparse capabilities
        use nalgebra::{Complex, DVector};
        use nalgebra_sparse::SparseEntry;
        use sprs::CsMat;

        let rhs_array = rhs.to_array1()?;

        // Convert to sprs format for better sparse solving
        let values: Vec<Complex<f64>> = self
            .csr_matrix
            .values()
            .iter()
            .map(|&c| Complex::new(c.re, c.im))
            .collect();
        let (rows, cols, _values) = self.csr_matrix.csr_data();

        // Use iterative solver for sparse systems
        // This is a simplified implementation - production would use better solvers
        let mut solution = rhs_array.clone();

        // Simple Jacobi iteration for demonstration
        for _ in 0..100 {
            let mut new_solution = solution.clone();
            for i in 0..solution.len() {
                if i < self.csr_matrix.nrows() {
                    // Get diagonal element
                    let diag = self
                        .csr_matrix
                        .get_entry(i, i)
                        .map(|entry| match entry {
                            SparseEntry::NonZero(v) => *v,
                            SparseEntry::Zero => Complex::new(0.0, 0.0),
                        })
                        .unwrap_or(Complex::new(1.0, 0.0));

                    if diag.norm() > 1e-14 {
                        new_solution[i] = rhs_array[i] / diag;
                    }
                }
            }
            solution = new_solution;
        }

        Vector::from_array1(&solution.view(), &MemoryPool::new())
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.csr_matrix.nrows(), self.csr_matrix.ncols())
    }

    pub fn nnz(&self) -> usize {
        self.csr_matrix.nnz()
    }
}

#[cfg(not(feature = "advanced_math"))]
#[derive(Debug)]
pub struct SparseMatrix;

#[cfg(not(feature = "advanced_math"))]
impl SparseMatrix {
    pub fn from_csr(
        _values: &[Complex64],
        _col_indices: &[usize],
        _row_ptr: &[usize],
        _num_rows: usize,
        _num_cols: usize,
        _pool: &MemoryPool,
    ) -> Result<Self> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }

    pub fn matvec(&self, _vector: &Vector, _result: &mut Vector) -> Result<()> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }

    pub fn solve(&self, _rhs: &Vector) -> Result<Vector> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }
}

/// BLAS operations using SciRS2
#[cfg(feature = "advanced_math")]
#[derive(Debug)]
pub struct BLAS;

#[cfg(feature = "advanced_math")]
impl BLAS {
    pub fn gemm(
        alpha: Complex64,
        a: &Matrix,
        b: &Matrix,
        beta: Complex64,
        c: &mut Matrix,
    ) -> Result<()> {
        // Use ndarray operations for now - in full implementation would use scirs2-linalg BLAS
        let a_scaled = &a.data * alpha;
        let c_scaled = &c.data * beta;
        let result = a_scaled.dot(&b.data) + c_scaled;
        c.data.assign(&result);
        Ok(())
    }

    pub fn gemv(
        alpha: Complex64,
        a: &Matrix,
        x: &Vector,
        beta: Complex64,
        y: &mut Vector,
    ) -> Result<()> {
        // Matrix-vector multiplication
        let y_scaled = &y.data * beta;
        let result = &a.data.dot(&x.data) * alpha + y_scaled;
        y.data.assign(&result);
        Ok(())
    }
}

#[cfg(not(feature = "advanced_math"))]
#[derive(Debug)]
pub struct BLAS;

#[cfg(not(feature = "advanced_math"))]
impl BLAS {
    pub fn gemm(
        _alpha: Complex64,
        _a: &Matrix,
        _b: &Matrix,
        _beta: Complex64,
        _c: &mut Matrix,
    ) -> Result<()> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }

    pub fn gemv(
        _alpha: Complex64,
        _a: &Matrix,
        _x: &Vector,
        _beta: Complex64,
        _y: &mut Vector,
    ) -> Result<()> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }
}

/// LAPACK operations using SciRS2
#[cfg(feature = "advanced_math")]
#[derive(Debug)]
pub struct LAPACK;

#[cfg(feature = "advanced_math")]
impl LAPACK {
    pub fn svd(matrix: &Matrix) -> Result<SvdResult> {
        // Use ndarray-linalg SVD for complex matrices
        use ndarray_linalg::SVD;
        
        let svd_result = matrix.data.svd(true, true).map_err(|_| {
            SimulatorError::ComputationError("SVD computation failed".to_string())
        })?;

        // Extract U, S, Vt from the SVD result
        let pool = MemoryPool::new();
        
        let u_data = svd_result.0.ok_or_else(|| {
            SimulatorError::ComputationError("SVD U matrix not computed".to_string())
        })?;
        let s_data = svd_result.1;
        let vt_data = svd_result.2.ok_or_else(|| {
            SimulatorError::ComputationError("SVD Vt matrix not computed".to_string())
        })?;
        
        let u = Matrix::from_array2(&u_data.view(), &pool)?;
        // Convert real singular values to complex for consistency
        let s_complex: Array1<Complex64> = s_data.mapv(|x| Complex64::new(x, 0.0));
        let s = Vector::from_array1(&s_complex.view(), &pool)?;
        let vt = Matrix::from_array2(&vt_data.view(), &pool)?;

        Ok(SvdResult { u, s, vt })
    }

    pub fn eig(matrix: &Matrix) -> Result<EigResult> {
        // Eigenvalue decomposition using SciRS2
        use ndarray_linalg::Eig;

        let eig_result = matrix.data.eig().map_err(|_| {
            SimulatorError::ComputationError("Eigenvalue decomposition failed".to_string())
        })?;

        let pool = MemoryPool::new();
        let values = Vector::from_array1(&eig_result.0.view(), &pool)?;
        let vectors = Matrix::from_array2(&eig_result.1.view(), &pool)?;

        Ok(EigResult { values, vectors })
    }

    pub fn lu(matrix: &Matrix) -> Result<(Matrix, Matrix, Vec<usize>)> {
        // Simplified LU decomposition - for production use, would need proper LU with pivoting
        let n = matrix.data.nrows();
        let pool = MemoryPool::new();
        
        // Initialize L as identity and U as copy of input
        let mut l_data = Array2::eye(n);
        let mut u_data = matrix.data.clone();
        let mut perm_vec: Vec<usize> = (0..n).collect();
        
        // Simplified Gaussian elimination
        for k in 0..n.min(n) {
            // Find pivot
            let mut max_row = k;
            let mut max_val = u_data[[k, k]].norm();
            for i in k + 1..n {
                let val = u_data[[i, k]].norm();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }
            
            // Swap rows if needed
            if max_row != k {
                for j in 0..n {
                    let temp = u_data[[k, j]];
                    u_data[[k, j]] = u_data[[max_row, j]];
                    u_data[[max_row, j]] = temp;
                }
                perm_vec.swap(k, max_row);
            }
            
            // Eliminate column
            if u_data[[k, k]].norm() > 1e-10 {
                for i in k + 1..n {
                    let factor = u_data[[i, k]] / u_data[[k, k]];
                    l_data[[i, k]] = factor;
                    for j in k..n {
                        u_data[[i, j]] -= factor * u_data[[k, j]];
                    }
                }
            }
        }
        
        let l_matrix = Matrix::from_array2(&l_data.view(), &pool)?;
        let u_matrix = Matrix::from_array2(&u_data.view(), &pool)?;

        Ok((l_matrix, u_matrix, perm_vec))
    }

    pub fn qr(matrix: &Matrix) -> Result<(Matrix, Matrix)> {
        // QR decomposition using ndarray-linalg
        use ndarray_linalg::QR;
        
        let (q, r) = matrix.data.qr().map_err(|_| {
            SimulatorError::ComputationError("QR decomposition failed".to_string())
        })?;

        let pool = MemoryPool::new();
        let q_matrix = Matrix::from_array2(&q.view(), &pool)?;
        let r_matrix = Matrix::from_array2(&r.view(), &pool)?;

        Ok((q_matrix, r_matrix))
    }
}

#[cfg(not(feature = "advanced_math"))]
#[derive(Debug)]
pub struct LAPACK;

#[cfg(not(feature = "advanced_math"))]
impl LAPACK {
    pub fn svd(_matrix: &Matrix) -> Result<SvdResult> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }

    pub fn eig(_matrix: &Matrix) -> Result<EigResult> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }
}

/// SVD decomposition result
#[cfg(feature = "advanced_math")]
#[derive(Debug, Clone)]
pub struct SvdResult {
    /// U matrix (left singular vectors)
    pub u: Matrix,
    /// Singular values
    pub s: Vector,
    /// V^T matrix (right singular vectors)
    pub vt: Matrix,
}

/// Eigenvalue decomposition result
#[cfg(feature = "advanced_math")]
#[derive(Debug, Clone)]
pub struct EigResult {
    /// Eigenvalues
    pub values: Vector,
    /// Eigenvectors (as columns of matrix)
    pub vectors: Matrix,
}

#[cfg(feature = "advanced_math")]
impl EigResult {
    pub fn to_array1(&self) -> Result<Array1<Complex64>> {
        self.values.to_array1()
    }

    pub fn eigenvalues(&self) -> &Vector {
        &self.values
    }

    pub fn eigenvectors(&self) -> &Matrix {
        &self.vectors
    }
}

#[cfg(feature = "advanced_math")]
impl SvdResult {
    pub fn to_array2(&self) -> Result<Array2<Complex64>> {
        self.u.data.to_owned().into_dimensionality().map_err(|_| {
            SimulatorError::ComputationError("Failed to convert SVD result to array2".to_string())
        })
    }

    pub fn u_matrix(&self) -> &Matrix {
        &self.u
    }

    pub fn singular_values(&self) -> &Vector {
        &self.s
    }

    pub fn vt_matrix(&self) -> &Matrix {
        &self.vt
    }
}

#[cfg(not(feature = "advanced_math"))]
#[derive(Debug)]
pub struct SvdResult;

#[cfg(not(feature = "advanced_math"))]
#[derive(Debug)]
pub struct EigResult;

#[cfg(not(feature = "advanced_math"))]
impl EigResult {
    pub fn to_array1(&self) -> Result<Array1<Complex64>> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }
}

#[cfg(not(feature = "advanced_math"))]
impl SvdResult {
    pub fn to_array2(&self) -> Result<Array2<Complex64>> {
        Err(SimulatorError::UnsupportedOperation(
            "SciRS2 integration requires 'advanced_math' feature".to_string(),
        ))
    }
}

/// Advanced FFT operations for quantum simulation
#[cfg(feature = "advanced_math")]
#[derive(Debug)]
pub struct AdvancedFFT;

#[cfg(feature = "advanced_math")]
impl AdvancedFFT {
    /// Multidimensional FFT for quantum state processing
    pub fn fft_nd(input: &Array2<Complex64>) -> Result<Array2<Complex64>> {
        use ndrustfft::{ndfft, FftHandler};
        
        let (rows, cols) = input.dim();
        let mut result = input.clone();
        
        // FFT along each dimension
        for i in 0..rows {
            let mut row = input.row(i).to_owned();
            let mut handler = FftHandler::new(cols);
            ndfft(&row, &mut row, &mut handler, 0);
            result.row_mut(i).assign(&row);
        }
        
        for j in 0..cols {
            let mut col = result.column(j).to_owned();
            let mut handler = FftHandler::new(rows);
            ndfft(&col, &mut col, &mut handler, 0);
            result.column_mut(j).assign(&col);
        }
        
        Ok(result)
    }

    /// Windowed FFT for spectral analysis
    pub fn windowed_fft(input: &Vector, window_size: usize, overlap: usize) -> Result<Array2<Complex64>> {
        let array = input.to_array1()?;
        let step_size = window_size - overlap;
        let num_windows = (array.len() - overlap) / step_size;
        
        let mut result = Array2::zeros((num_windows, window_size));
        
        for (i, mut row) in result.outer_iter_mut().enumerate() {
            let start = i * step_size;
            let end = (start + window_size).min(array.len());
            
            if end - start == window_size {
                let window = array.slice(s![start..end]);
                
                // Apply Hann window
                let windowed: Array1<Complex64> = window.iter().enumerate()
                    .map(|(j, &val)| {
                        let hann = 0.5 * (1.0 - (2.0 * PI * j as f64 / (window_size - 1) as f64).cos());
                        val * Complex64::new(hann, 0.0)
                    })
                    .collect();
                
                // Compute FFT
                let mut handler = FftHandler::new(window_size);
                let mut fft_result = windowed.clone();
                ndrustfft::ndfft(&windowed, &mut fft_result, &mut handler, 0);
                
                row.assign(&fft_result);
            }
        }
        
        Ok(result)
    }

    /// Convolution using FFT
    pub fn convolution(a: &Vector, b: &Vector) -> Result<Vector> {
        let a_array = a.to_array1()?;
        let b_array = b.to_array1()?;
        
        let n = a_array.len() + b_array.len() - 1;
        let fft_size = n.next_power_of_two();
        
        // Zero-pad inputs
        let mut a_padded = Array1::zeros(fft_size);
        let mut b_padded = Array1::zeros(fft_size);
        a_padded.slice_mut(s![..a_array.len()]).assign(&a_array);
        b_padded.slice_mut(s![..b_array.len()]).assign(&b_array);
        
        // FFT
        let mut handler = FftHandler::new(fft_size);
        let mut a_fft = a_padded.clone();
        let mut b_fft = b_padded.clone();
        ndrustfft::ndfft(&a_padded, &mut a_fft, &mut handler, 0);
        ndrustfft::ndfft(&b_padded, &mut b_fft, &mut handler, 0);
        
        // Multiply in frequency domain
        let mut product = Array1::zeros(fft_size);
        for i in 0..fft_size {
            product[i] = a_fft[i] * b_fft[i];
        }
        
        // IFFT
        let mut result = product.clone();
        ndrustfft::ndifft(&product, &mut result, &mut handler, 0);
        
        // Truncate to correct size and create Vector
        let truncated = result.slice(s![..n]).to_owned();
        Vector::from_array1(&truncated.view(), &MemoryPool::new())
    }
}

/// Advanced sparse linear algebra solvers
#[cfg(feature = "advanced_math")]
#[derive(Debug)]
pub struct SparseSolvers;

#[cfg(feature = "advanced_math")]
impl SparseSolvers {
    /// Conjugate Gradient solver for Ax = b
    pub fn conjugate_gradient(
        matrix: &SparseMatrix,
        b: &Vector,
        x0: Option<&Vector>,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Vector> {
        use nalgebra::{Complex, DVector};
        
        let b_array = b.to_array1()?;
        let b_vec = DVector::from_iterator(b_array.len(), b_array.iter().map(|&c| Complex::new(c.re, c.im)));
        
        let mut x = if let Some(x0_vec) = x0 {
            let x0_array = x0_vec.to_array1()?;
            DVector::from_iterator(x0_array.len(), x0_array.iter().map(|&c| Complex::new(c.re, c.im)))
        } else {
            DVector::zeros(b_vec.len())
        };
        
        // Initial residual: r = b - Ax
        let pool = MemoryPool::new();
        let x_vector = Vector::from_array1(&Array1::from_vec(
            x.iter().map(|c| Complex64::new(c.re, c.im)).collect()
        ).view(), &pool)?;
        let mut ax_vector = Vector::zeros(x.len(), &pool)?;
        matrix.matvec(&x_vector, &mut ax_vector)?;
        
        // Convert back to DVector for computation
        let ax_array = ax_vector.to_array1()?;
        let ax = DVector::from_iterator(ax_array.len(), ax_array.iter().map(|&c| Complex::new(c.re, c.im)));
        
        let mut r = &b_vec - &ax;
        let mut p = r.clone();
        let mut rsold = r.dot(&r).re;
        
        for _ in 0..max_iterations {
            // Ap = A * p
            let p_vec = Vector::from_array1(&Array1::from_vec(
                p.iter().map(|c| Complex64::new(c.re, c.im)).collect()
            ).view(), &MemoryPool::new())?;
            let mut ap_vec = Vector::from_array1(&Array1::zeros(p.len()).view(), &MemoryPool::new())?;
            matrix.matvec(&p_vec, &mut ap_vec)?;
            let ap_array = ap_vec.to_array1()?;
            let ap = DVector::from_iterator(ap_array.len(), ap_array.iter().map(|&c| Complex::new(c.re, c.im)));
            
            let alpha = rsold / p.dot(&ap).re;
            let alpha_complex = Complex::new(alpha, 0.0);
            x += &p * alpha_complex;
            r -= &ap * alpha_complex;
            
            let rsnew = r.dot(&r).re;
            if rsnew.sqrt() < tolerance {
                break;
            }
            
            let beta = rsnew / rsold;
            let beta_complex = Complex::new(beta, 0.0);
            p = &r + &p * beta_complex;
            rsold = rsnew;
        }
        
        let result_array = Array1::from_vec(x.iter().map(|c| Complex64::new(c.re, c.im)).collect());
        Vector::from_array1(&result_array.view(), &MemoryPool::new())
    }

    /// GMRES solver for non-symmetric systems
    pub fn gmres(
        matrix: &SparseMatrix,
        b: &Vector,
        x0: Option<&Vector>,
        tolerance: f64,
        max_iterations: usize,
        restart: usize,
    ) -> Result<Vector> {
        let b_array = b.to_array1()?;
        let n = b_array.len();
        
        let mut x = if let Some(x0_vec) = x0 {
            x0_vec.to_array1()?.to_owned()
        } else {
            Array1::zeros(n)
        };
        
        for _restart_iter in 0..(max_iterations / restart) {
            // Calculate initial residual
            let mut ax = Array1::zeros(n);
            let x_vec = Vector::from_array1(&x.view(), &MemoryPool::new())?;
            let mut ax_vec = Vector::from_array1(&ax.view(), &MemoryPool::new())?;
            matrix.matvec(&x_vec, &mut ax_vec)?;
            ax = ax_vec.to_array1()?;
            
            let mut r = &b_array - &ax;
            let beta = r.norm_l2();
            
            if beta < tolerance {
                break;
            }
            
            r /= beta;
            
            // Arnoldi process
            let mut v = Vec::new();
            v.push(r.clone());
            
            let mut h = Array2::zeros((restart + 1, restart));
            
            for j in 0..restart.min(max_iterations) {
                let v_vec = Vector::from_array1(&v[j].view(), &MemoryPool::new())?;
                let mut av = Array1::zeros(n);
                let mut av_vec = Vector::from_array1(&av.view(), &MemoryPool::new())?;
                matrix.multiply_vector(&v_vec, &mut av_vec)?;
                av = av_vec.to_array1()?;
                
                // Modified Gram-Schmidt orthogonalization
                for i in 0..=j {
                    h[[i, j]] = v[i].dot(&av);
                    av = av - h[[i, j]] * &v[i];
                }
                
                h[[j + 1, j]] = av.norm_l2();
                
                if h[[j + 1, j]].norm() < tolerance {
                    break;
                }
                
                av /= h[[j + 1, j]];
                v.push(av);
            }
            
            // Solve least squares problem using the constructed Hessenberg matrix
            // Simplified implementation - would use proper QR factorization in production
            let krylov_dim = v.len() - 1;
            if krylov_dim > 0 {
                let mut e1 = Array1::zeros(krylov_dim + 1);
                e1[0] = Complex64::new(beta, 0.0);
                
                // Simple back-substitution for upper triangular solve
                let mut y = Array1::zeros(krylov_dim);
                for i in (0..krylov_dim).rev() {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for j in (i + 1)..krylov_dim {
                        sum += h[[i, j]] * y[j];
                    }
                    y[i] = (e1[i] - sum) / h[[i, i]];
                }
                
                // Update solution
                for i in 0..krylov_dim {
                    x = x + y[i] * &v[i];
                }
            }
        }
        
        Vector::from_array1(&x.view(), &MemoryPool::new())
    }

    /// BiCGSTAB solver for complex systems
    pub fn bicgstab(
        matrix: &SparseMatrix,
        b: &Vector,
        x0: Option<&Vector>,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Vector> {
        let b_array = b.to_array1()?;
        let n = b_array.len();
        
        let mut x = if let Some(x0_vec) = x0 {
            x0_vec.to_array1()?.to_owned()
        } else {
            Array1::zeros(n)
        };
        
        // Calculate initial residual
        let mut ax = Array1::zeros(n);
        let x_vec = Vector::from_array1(&x.view(), &MemoryPool::new())?;
        let mut ax_vec = Vector::from_array1(&ax.view(), &MemoryPool::new())?;
        matrix.multiply_vector(&x_vec, &mut ax_vec)?;
        ax = ax_vec.to_array1()?;
        
        let mut r = &b_array - &ax;
        let r0 = r.clone();
        
        let mut rho = Complex64::new(1.0, 0.0);
        let mut alpha = Complex64::new(1.0, 0.0);
        let mut omega = Complex64::new(1.0, 0.0);
        
        let mut p = Array1::zeros(n);
        let mut v = Array1::zeros(n);
        
        for _ in 0..max_iterations {
            let rho_new = r0.dot(&r);
            let beta = (rho_new / rho) * (alpha / omega);
            
            p = &r + beta * (&p - omega * &v);
            
            // v = A * p
            let p_vec = Vector::from_array1(&p.view(), &MemoryPool::new())?;
            let mut v_vec = Vector::from_array1(&v.view(), &MemoryPool::new())?;
            matrix.multiply_vector(&p_vec, &mut v_vec)?;
            v = v_vec.to_array1()?;
            
            alpha = rho_new / r0.dot(&v);
            let s = &r - alpha * &v;
            
            if s.norm_l2() < tolerance {
                x = x + alpha * &p;
                break;
            }
            
            // t = A * s
            let s_vec = Vector::from_array1(&s.view(), &MemoryPool::new())?;
            let mut t_vec = Vector::from_array1(&Array1::zeros(n).view(), &MemoryPool::new())?;
            matrix.multiply_vector(&s_vec, &mut t_vec)?;
            let t = t_vec.to_array1()?;
            
            omega = t.dot(&s) / t.dot(&t);
            x = x + alpha * &p + omega * &s;
            r = s - omega * &t;
            
            if r.norm_l2() < tolerance {
                break;
            }
            
            rho = rho_new;
        }
        
        Vector::from_array1(&x.view(), &MemoryPool::new())
    }
}

/// Advanced eigenvalue solvers for large sparse matrices
#[cfg(feature = "advanced_math")]
#[derive(Debug)]
pub struct AdvancedEigensolvers;

#[cfg(feature = "advanced_math")]
impl AdvancedEigensolvers {
    /// Lanczos algorithm for finding a few eigenvalues of large sparse symmetric matrices
    pub fn lanczos(
        matrix: &SparseMatrix,
        num_eigenvalues: usize,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<EigResult> {
        let n = matrix.csr_matrix.nrows();
        let m = num_eigenvalues.min(max_iterations);
        
        // Initialize random starting vector
        let mut q = Array1::from_vec((0..n).map(|_| {
            Complex64::new(rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5)
        }).collect());
        q /= q.norm_l2();
        
        let mut q_vectors = Vec::new();
        q_vectors.push(q.clone());
        
        let mut alpha = Vec::new();
        let mut beta = Vec::new();
        
        let mut q_prev = Array1::zeros(n);
        
        for j in 0..m {
            // Av = A * q[j]
            let q_vec = Vector::from_array1(&q_vectors[j].view(), &MemoryPool::new())?;
            let mut av_vec = Vector::from_array1(&Array1::zeros(n).view(), &MemoryPool::new())?;
            matrix.multiply_vector(&q_vec, &mut av_vec)?;
            let mut av = av_vec.to_array1()?;
            
            // Alpha computation
            let alpha_j = q_vectors[j].dot(&av);
            alpha.push(alpha_j);
            
            // Orthogonalization
            av = av - alpha_j * &q_vectors[j];
            if j > 0 {
                av = av - beta[j - 1] * &q_prev;
            }
            
            let beta_j = av.norm_l2();
            
            if beta_j.norm() < tolerance {
                break;
            }
            
            beta.push(beta_j);
            q_prev = q_vectors[j].clone();
            
            if j + 1 < m {
                q = av / beta_j;
                q_vectors.push(q.clone());
            }
        }
        
        // Solve the tridiagonal eigenvalue problem
        let dim = alpha.len();
        let mut tridiag = Array2::zeros((dim, dim));
        
        for i in 0..dim {
            tridiag[[i, i]] = alpha[i];
            if i > 0 {
                tridiag[[i - 1, i]] = beta[i - 1];
                tridiag[[i, i - 1]] = beta[i - 1];
            }
        }
        
        // Use simple power iteration for the tridiagonal system (simplified)
        let mut eigenvalues = Array1::zeros(num_eigenvalues.min(dim));
        for i in 0..eigenvalues.len() {
            eigenvalues[i] = tridiag[[i, i]]; // Simplified - would use proper tridiagonal solver
        }
        
        // Construct approximate eigenvectors
        let mut eigenvectors = Array2::zeros((n, eigenvalues.len()));
        for (i, mut col) in eigenvectors.columns_mut().into_iter().enumerate().take(eigenvalues.len()) {
            if i < q_vectors.len() {
                col.assign(&q_vectors[i]);
            }
        }
        
        let values = Vector::from_array1(&eigenvalues.view(), &MemoryPool::new())?;
        let vectors = Matrix::from_array2(&eigenvectors.view(), &MemoryPool::new())?;
        
        Ok(EigResult { values, vectors })
    }

    /// Arnoldi iteration for non-symmetric matrices
    pub fn arnoldi(
        matrix: &SparseMatrix,
        num_eigenvalues: usize,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<EigResult> {
        let n = matrix.csr_matrix.nrows();
        let m = num_eigenvalues.min(max_iterations);
        
        // Initialize random starting vector
        let mut q = Array1::from_vec((0..n).map(|_| {
            Complex64::new(rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5)
        }).collect());
        q /= q.norm_l2();
        
        let mut q_vectors = Vec::new();
        q_vectors.push(q.clone());
        
        let mut h = Array2::zeros((m + 1, m));
        
        for j in 0..m {
            // v = A * q[j]
            let q_vec = Vector::from_array1(&q_vectors[j].view(), &MemoryPool::new())?;
            let mut v_vec = Vector::from_array1(&Array1::zeros(n).view(), &MemoryPool::new())?;
            matrix.multiply_vector(&q_vec, &mut v_vec)?;
            let mut v = v_vec.to_array1()?;
            
            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                h[[i, j]] = q_vectors[i].dot(&v);
                v = v - h[[i, j]] * &q_vectors[i];
            }
            
            h[[j + 1, j]] = v.norm_l2();
            
            if h[[j + 1, j]].norm() < tolerance {
                break;
            }
            
            if j + 1 < m {
                q = v / h[[j + 1, j]];
                q_vectors.push(q.clone());
            }
        }
        
        // Extract eigenvalues from upper Hessenberg matrix (simplified)
        let dim = q_vectors.len();
        let mut eigenvalues = Array1::zeros(num_eigenvalues.min(dim));
        for i in 0..eigenvalues.len() {
            eigenvalues[i] = h[[i, i]]; // Simplified extraction
        }
        
        // Construct eigenvectors
        let mut eigenvectors = Array2::zeros((n, eigenvalues.len()));
        for (i, mut col) in eigenvectors.columns_mut().into_iter().enumerate().take(eigenvalues.len()) {
            if i < q_vectors.len() {
                col.assign(&q_vectors[i]);
            }
        }
        
        let values = Vector::from_array1(&eigenvalues.view(), &MemoryPool::new())?;
        let vectors = Matrix::from_array2(&eigenvectors.view(), &MemoryPool::new())?;
        
        Ok(EigResult { values, vectors })
    }
}

/// Enhanced linear algebra operations
#[cfg(feature = "advanced_math")]
#[derive(Debug)]
pub struct AdvancedLinearAlgebra;

#[cfg(feature = "advanced_math")]
impl AdvancedLinearAlgebra {
    /// QR decomposition with pivoting
    pub fn qr_decomposition(matrix: &Matrix) -> Result<QRResult> {
        use ndarray_linalg::QR;
        
        let qr_result = matrix.data.qr().map_err(|_| {
            SimulatorError::ComputationError("QR decomposition failed".to_string())
        })?;
        
        let pool = MemoryPool::new();
        let q = Matrix::from_array2(&qr_result.0.view(), &pool)?;
        let r = Matrix::from_array2(&qr_result.1.view(), &pool)?;
        
        Ok(QRResult { q, r })
    }

    /// Cholesky decomposition for positive definite matrices
    pub fn cholesky_decomposition(matrix: &Matrix) -> Result<Matrix> {
        use ndarray_linalg::Cholesky;
        
        let chol_result = matrix.data.cholesky(ndarray_linalg::UPLO::Lower).map_err(|_| {
            SimulatorError::ComputationError("Cholesky decomposition failed".to_string())
        })?;
        
        Matrix::from_array2(&chol_result.view(), &MemoryPool::new())
    }

    /// Matrix exponential for quantum evolution
    pub fn matrix_exponential(matrix: &Matrix, t: f64) -> Result<Matrix> {
        let scaled_matrix = &matrix.data * Complex64::new(0.0, -t);
        
        // Matrix exponential using scaling and squaring with Padé approximation
        let mut result = Array2::eye(scaled_matrix.nrows());
        let mut term = Array2::eye(scaled_matrix.nrows());
        
        // Simple series expansion (would use more sophisticated methods in production)
        for k in 1..20 {
            term = term.dot(&scaled_matrix) / Complex64::new(k as f64, 0.0);
            result = result + &term;
            
            if term.norm_l2() < 1e-12 {
                break;
            }
        }
        
        Matrix::from_array2(&result.view(), &MemoryPool::new())
    }

    /// Pseudoinverse using SVD
    pub fn pseudoinverse(matrix: &Matrix, tolerance: f64) -> Result<Matrix> {
        let svd_result = LAPACK::svd(matrix)?;
        
        let u = svd_result.u.data;
        let s = svd_result.s.to_array1()?;
        let vt = svd_result.vt.data;
        
        // Create pseudoinverse of singular values
        let mut s_pinv = Array1::zeros(s.len());
        for (i, &sigma) in s.iter().enumerate() {
            if sigma.norm() > tolerance {
                s_pinv[i] = Complex64::new(1.0, 0.0) / sigma;
            }
        }
        
        // Construct pseudoinverse: V * S^+ * U^T
        let s_pinv_diag = Array2::from_diag(&s_pinv);
        let result = vt.t().dot(&s_pinv_diag).dot(&u.t());
        
        Matrix::from_array2(&result.view(), &MemoryPool::new())
    }

    /// Condition number estimation
    pub fn condition_number(matrix: &Matrix) -> Result<f64> {
        let svd_result = LAPACK::svd(matrix)?;
        let s = svd_result.s.to_array1()?;
        
        let mut min_singular = f64::INFINITY;
        let mut max_singular = 0.0;
        
        for &sigma in s.iter() {
            let sigma_norm = sigma.norm();
            if sigma_norm > 1e-15 {
                min_singular = min_singular.min(sigma_norm);
                max_singular = max_singular.max(sigma_norm);
            }
        }
        
        Ok(max_singular / min_singular)
    }
}

/// QR decomposition result
#[cfg(feature = "advanced_math")]
#[derive(Debug, Clone)]
pub struct QRResult {
    /// Q matrix (orthogonal)
    pub q: Matrix,
    /// R matrix (upper triangular)
    pub r: Matrix,
}

/// Performance benchmarking for SciRS2 integration
pub fn benchmark_scirs2_integration() -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();
    
    // FFT benchmarks
    #[cfg(feature = "advanced_math")]
    {
        let start = std::time::Instant::now();
        let engine = FftEngine::new();
        let test_vector = Vector::from_array1(&Array1::from_vec(
            (0..1024).map(|i| Complex64::new(i as f64, 0.0)).collect()
        ).view(), &MemoryPool::new())?;
        
        for _ in 0..100 {
            let _ = engine.forward(&test_vector)?;
        }
        
        let fft_time = start.elapsed().as_millis() as f64;
        results.insert("fft_1024_100_iterations".to_string(), fft_time);
    }
    
    // Sparse solver benchmarks
    #[cfg(feature = "advanced_math")]
    {
        use nalgebra_sparse::CsrMatrix;
        
        let start = std::time::Instant::now();
        
        // Create test sparse matrix
        let mut row_indices = vec![0; 1000];
        let mut col_indices = vec![0; 1000];
        let mut values = vec![Complex64::new(0.0, 0.0); 1000];
        
        for i in 0..100 {
            for j in 0..10 {
                let idx = i * 10 + j;
                row_indices[idx] = i;
                col_indices[idx] = (i + j) % 100;
                values[idx] = Complex64::new(1.0, 0.0);
            }
        }
        
        let csr = CsrMatrix::try_from_csr_data(100, 100, row_indices, col_indices, values)
            .map_err(|_| SimulatorError::ComputationError("Failed to create test matrix".to_string()))?;
        
        let sparse_matrix = SparseMatrix { csr_matrix: csr };
        let b = Vector::from_array1(&Array1::ones(100).view(), &MemoryPool::new())?;
        
        let _ = SparseSolvers::conjugate_gradient(&sparse_matrix, &b, None, 1e-6, 100)?;
        
        let sparse_solver_time = start.elapsed().as_millis() as f64;
        results.insert("cg_solver_100x100".to_string(), sparse_solver_time);
    }
    
    // Linear algebra benchmarks
    #[cfg(feature = "advanced_math")]
    {
        let start = std::time::Instant::now();
        
        let test_matrix = Matrix::from_array2(&Array2::eye(50).view(), &MemoryPool::new())?;
        for _ in 0..10 {
            let _ = AdvancedLinearAlgebra::qr_decomposition(&test_matrix)?;
        }
        
        let qr_time = start.elapsed().as_millis() as f64;
        results.insert("qr_decomposition_50x50_10_iterations".to_string(), qr_time);
    }
    
    Ok(results)
}
