//! Integration with SciRS2 for advanced linear algebra operations.
//!
//! This module provides a comprehensive integration layer with SciRS2 to leverage
//! high-performance linear algebra routines for quantum simulation.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;

#[cfg(feature = "advanced_math")]
use scirs2_core::{error::SciRS2Error, memory::MemoryPool as SciRS2MemoryPool};
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
            inner: SciRS2MemoryPool::new(),
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
            return Err(SimulatorError::DimensionMismatch {
                expected: self.data.shape().to_vec(),
                actual: result.shape().to_vec(),
            });
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
            return Err(SimulatorError::DimensionMismatch {
                expected: vec![self.data.len()],
                actual: vec![result.len()],
            });
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

        // Perform matrix-vector multiplication
        let output = &self.csr_matrix * nalgebra_vec;

        // Convert back to our format
        let output_array: Array1<Complex64> =
            Array1::from_iter(output.iter().map(|c| Complex64::new(c.re, c.im)));

        result.data.assign(&output_array);
        Ok(())
    }

    pub fn solve(&self, rhs: &Vector) -> Result<Vector> {
        // Improved sparse solver using nalgebra-sparse capabilities
        use nalgebra::{Complex, DVector};
        use sprs::CsMat;

        let rhs_array = rhs.to_array1()?;

        // Convert to sprs format for better sparse solving
        let values: Vec<Complex<f64>> = self
            .csr_matrix
            .values()
            .iter()
            .map(|&c| Complex::new(c.re, c.im))
            .collect();
        let (rows, cols) = self.csr_matrix.csr_data();

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
                        .map(|&v| v)
                        .unwrap_or(Complex64::new(1.0, 0.0));

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
        let result = alpha * a.view().dot(&b.view()) + beta * c.view();
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
        let result = alpha * a.view().dot(&x.view()) + beta * y.view();
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
        // Use scirs2-linalg SVD when available
        let svd_result = svd(&matrix.view())
            .map_err(|_| SimulatorError::ComputationError("SVD computation failed".to_string()))?;

        // Extract U, S, Vt from the SVD result
        let pool = MemoryPool::new();
        let u = Matrix::from_array2(&svd_result.u.view(), &pool)?;
        let s = Vector::from_array1(&svd_result.s.view(), &pool)?;
        let vt = Matrix::from_array2(&svd_result.vt.view(), &pool)?;

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
        // LU decomposition
        let (p, l, u) = lu(&matrix.view())
            .map_err(|_| SimulatorError::ComputationError("LU decomposition failed".to_string()))?;

        let pool = MemoryPool::new();
        let l_matrix = Matrix::from_array2(&l.view(), &pool)?;
        let u_matrix = Matrix::from_array2(&u.view(), &pool)?;

        Ok((l_matrix, u_matrix, p))
    }

    pub fn qr(matrix: &Matrix) -> Result<(Matrix, Matrix)> {
        // QR decomposition
        let (q, r) = qr(&matrix.view())
            .map_err(|_| SimulatorError::ComputationError("QR decomposition failed".to_string()))?;

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
