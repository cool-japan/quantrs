//! Integration with SciRS2 for advanced linear algebra operations.
//!
//! This module provides a comprehensive integration layer with SciRS2 to leverage
//! high-performance linear algebra routines for quantum simulation.

use num_complex::Complex64;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{SimulatorError, Result};

/// SciRS2-powered linear algebra backend
pub struct SciRS2Backend {
    /// Whether SciRS2 is available
    pub available: bool,
    
    /// Performance statistics
    pub stats: BackendStats,
}

/// Performance statistics for the backend
#[derive(Debug, Clone, Default)]
pub struct BackendStats {
    /// Number of matrix operations
    pub matrix_ops: usize,
    /// Number of vector operations  
    pub vector_ops: usize,
    /// Number of FFT operations
    pub fft_ops: usize,
    /// Total time spent in SciRS2 operations
    pub total_time_ms: f64,
    /// Memory usage statistics
    pub memory_usage: MemoryStats,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Peak memory usage (bytes)
    pub peak_usage: usize,
    /// Current memory usage (bytes)
    pub current_usage: usize,
    /// Number of allocations
    pub allocations: usize,
    /// Number of deallocations
    pub deallocations: usize,
}

impl SciRS2Backend {
    /// Create a new SciRS2 backend
    pub fn new() -> Result<Self> {
        Ok(Self {
            #[cfg(feature = "advanced_math")]
            available: true,
            #[cfg(not(feature = "advanced_math"))]
            available: false,
            stats: BackendStats::default(),
        })
    }

    /// Check if SciRS2 backend is available
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
}

/// High-performance matrix operations using SciRS2
impl SciRS2Backend {
    /// Matrix-vector multiplication using BLAS Level 2
    pub fn gemv(
        &mut self,
        matrix: &ArrayView2<Complex64>,
        vector: &ArrayView1<Complex64>,
        result: &mut Array1<Complex64>,
    ) -> Result<()> {
        #[cfg(feature = "advanced_math")]
        {
            let start = std::time::Instant::now();
            
            // Fallback implementation for now - SciRS2 API is still evolving
            if matrix.nrows() != result.len() || matrix.ncols() != vector.len() {
                return Err(SimulatorError::DimensionMismatch(
                    "Matrix and vector dimensions don't match".to_string()
                ));
            }
            
            for i in 0..matrix.nrows() {
                let mut sum = Complex64::new(0.0, 0.0);
                for j in 0..matrix.ncols() {
                    sum += matrix[[i, j]] * vector[j];
                }
                result[i] = sum;
            }
            
            self.stats.matrix_ops += 1;
            self.stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            
            Ok(())
        }
        
        #[cfg(not(feature = "advanced_math"))]
        {
            // Fallback implementation
            if matrix.nrows() != result.len() || matrix.ncols() != vector.len() {
                return Err(SimulatorError::DimensionMismatch(
                    "Matrix and vector dimensions don't match".to_string()
                ));
            }
            
            for i in 0..matrix.nrows() {
                let mut sum = Complex64::new(0.0, 0.0);
                for j in 0..matrix.ncols() {
                    sum += matrix[[i, j]] * vector[j];
                }
                result[i] = sum;
            }
            
            self.stats.matrix_ops += 1;
            Ok(())
        }
    }

    /// Matrix-matrix multiplication using BLAS Level 3
    pub fn gemm(
        &mut self,
        a: &ArrayView2<Complex64>,
        b: &ArrayView2<Complex64>,
        result: &mut Array2<Complex64>,
    ) -> Result<()> {
        #[cfg(feature = "advanced_math")]
        {
            let start = std::time::Instant::now();
            
            // Convert to SciRS2 format
            let sci_a = Matrix::from_array2(a, &self.memory_pool)?;
            let sci_b = Matrix::from_array2(b, &self.memory_pool)?;
            let mut sci_result = Matrix::zeros((a.nrows(), b.ncols()), &self.memory_pool)?;
            
            // Perform BLAS operation
            BLAS::gemm(
                Complex64::new(1.0, 0.0), // alpha
                &sci_a,
                &sci_b,
                Complex64::new(0.0, 0.0), // beta
                &mut sci_result,
            )?;
            
            // Convert back to ndarray
            sci_result.to_array2(result)?;
            
            self.stats.matrix_ops += 1;
            self.stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            
            Ok(())
        }
        
        #[cfg(not(feature = "advanced_math"))]
        {
            // Fallback implementation
            if a.ncols() != b.nrows() {
                return Err(SimulatorError::DimensionMismatch(
                    "Matrix dimensions don't match for multiplication".to_string()
                ));
            }
            
            result.fill(Complex64::new(0.0, 0.0));
            
            for i in 0..a.nrows() {
                for j in 0..b.ncols() {
                    for k in 0..a.ncols() {
                        result[[i, j]] += a[[i, k]] * b[[k, j]];
                    }
                }
            }
            
            self.stats.matrix_ops += 1;
            Ok(())
        }
    }

    /// Singular Value Decomposition using LAPACK
    pub fn svd(
        &mut self,
        matrix: &ArrayView2<Complex64>,
    ) -> Result<(Array2<Complex64>, Array1<f64>, Array2<Complex64>)> {
        #[cfg(feature = "advanced_math")]
        {
            let start = std::time::Instant::now();
            
            // Convert to SciRS2 format
            let sci_matrix = Matrix::from_array2(matrix, &self.memory_pool)?;
            
            // Perform SVD
            let (u, s, vt) = LAPACK::svd(&sci_matrix)?;
            
            // Convert back to ndarray
            let u_array = u.to_array2()?;
            let s_array = s.to_array1()?;
            let vt_array = vt.to_array2()?;
            
            self.stats.matrix_ops += 1;
            self.stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            
            Ok((u_array, s_array, vt_array))
        }
        
        #[cfg(not(feature = "advanced_math"))]
        {
            // For now, return an error - proper fallback would require implementing SVD
            Err(SimulatorError::UnsupportedOperation(
                "SVD requires SciRS2 backend (enable 'advanced_math' feature)".to_string()
            ))
        }
    }

    /// QR decomposition using LAPACK
    pub fn qr(
        &mut self,
        matrix: &ArrayView2<Complex64>,
    ) -> Result<(Array2<Complex64>, Array2<Complex64>)> {
        #[cfg(feature = "advanced_math")]
        {
            let start = std::time::Instant::now();
            
            // Convert to SciRS2 format
            let sci_matrix = Matrix::from_array2(matrix, &self.memory_pool)?;
            
            // Perform QR decomposition
            let (q, r) = LAPACK::qr(&sci_matrix)?;
            
            // Convert back to ndarray
            let q_array = q.to_array2()?;
            let r_array = r.to_array2()?;
            
            self.stats.matrix_ops += 1;
            self.stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            
            Ok((q_array, r_array))
        }
        
        #[cfg(not(feature = "advanced_math"))]
        {
            Err(SimulatorError::UnsupportedOperation(
                "QR decomposition requires SciRS2 backend (enable 'advanced_math' feature)".to_string()
            ))
        }
    }

    /// Eigenvalue decomposition using LAPACK
    pub fn eig(
        &mut self,
        matrix: &ArrayView2<Complex64>,
    ) -> Result<(Array1<Complex64>, Array2<Complex64>)> {
        #[cfg(feature = "advanced_math")]
        {
            let start = std::time::Instant::now();
            
            // Convert to SciRS2 format
            let sci_matrix = Matrix::from_array2(matrix, &self.memory_pool)?;
            
            // Perform eigenvalue decomposition
            let (eigenvalues, eigenvectors) = LAPACK::eig(&sci_matrix)?;
            
            // Convert back to ndarray
            let eigenvalues_array = eigenvalues.to_array1()?;
            let eigenvectors_array = eigenvectors.to_array2()?;
            
            self.stats.matrix_ops += 1;
            self.stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            
            Ok((eigenvalues_array, eigenvectors_array))
        }
        
        #[cfg(not(feature = "advanced_math"))]
        {
            Err(SimulatorError::UnsupportedOperation(
                "Eigenvalue decomposition requires SciRS2 backend (enable 'advanced_math' feature)".to_string()
            ))
        }
    }
}

/// FFT operations using SciRS2
impl SciRS2Backend {
    /// Forward FFT for quantum Fourier transform
    pub fn fft(&mut self, input: &ArrayView1<Complex64>) -> Result<Array1<Complex64>> {
        #[cfg(feature = "advanced_math")]
        {
            let start = std::time::Instant::now();
            
            // Convert to SciRS2 format
            let sci_input = Vector::from_array1(input, &self.memory_pool)?;
            
            // Perform FFT
            let sci_output = self.fft_engine.forward(&sci_input)?;
            
            // Convert back to ndarray
            let output = sci_output.to_array1()?;
            
            self.stats.fft_ops += 1;
            self.stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            
            Ok(output)
        }
        
        #[cfg(not(feature = "advanced_math"))]
        {
            // Simple fallback DFT implementation (very slow)
            let n = input.len();
            let mut output = Array1::zeros(n);
            
            for k in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for j in 0..n {
                    let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                    let twiddle = Complex64::from_polar(1.0, angle);
                    sum += input[j] * twiddle;
                }
                output[k] = sum;
            }
            
            self.stats.fft_ops += 1;
            Ok(output)
        }
    }

    /// Inverse FFT
    pub fn ifft(&mut self, input: &ArrayView1<Complex64>) -> Result<Array1<Complex64>> {
        #[cfg(feature = "advanced_math")]
        {
            let start = std::time::Instant::now();
            
            // Convert to SciRS2 format
            let sci_input = Vector::from_array1(input, &self.memory_pool)?;
            
            // Perform inverse FFT
            let sci_output = self.fft_engine.inverse(&sci_input)?;
            
            // Convert back to ndarray
            let output = sci_output.to_array1()?;
            
            self.stats.fft_ops += 1;
            self.stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            
            Ok(output)
        }
        
        #[cfg(not(feature = "advanced_math"))]
        {
            // Simple fallback inverse DFT implementation
            let n = input.len();
            let mut output = Array1::zeros(n);
            
            for k in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for j in 0..n {
                    let angle = 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                    let twiddle = Complex64::from_polar(1.0, angle);
                    sum += input[j] * twiddle;
                }
                output[k] = sum / n as f64;
            }
            
            self.stats.fft_ops += 1;
            Ok(output)
        }
    }
}

/// Sparse matrix operations using SciRS2
impl SciRS2Backend {
    /// Sparse matrix-vector multiplication
    pub fn sparse_matvec(
        &mut self,
        matrix: &crate::sparse::CSRMatrix,
        vector: &ArrayView1<Complex64>,
        result: &mut Array1<Complex64>,
    ) -> Result<()> {
        #[cfg(feature = "advanced_math")]
        {
            let start = std::time::Instant::now();
            
            // Convert to SciRS2 sparse format
            let sci_matrix = SparseMatrix::from_csr(
                &matrix.values,
                &matrix.col_indices,
                &matrix.row_ptr,
                matrix.num_rows,
                matrix.num_cols,
                &self.memory_pool,
            )?;
            
            let sci_vector = Vector::from_array1(vector, &self.memory_pool)?;
            let mut sci_result = Vector::zeros(result.len(), &self.memory_pool)?;
            
            // Perform sparse matrix-vector multiplication
            sci_matrix.matvec(&sci_vector, &mut sci_result)?;
            
            // Convert back to ndarray
            sci_result.to_array1(result)?;
            
            self.stats.matrix_ops += 1;
            self.stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            
            Ok(())
        }
        
        #[cfg(not(feature = "advanced_math"))]
        {
            // Fallback to manual CSR multiplication
            let owned_vector = vector.to_owned();
            matrix.matvec(&owned_vector).map(|res| {
                result.assign(&res);
            })
        }
    }

    /// Sparse linear system solver
    pub fn sparse_solve(
        &mut self,
        matrix: &crate::sparse::CSRMatrix,
        rhs: &ArrayView1<Complex64>,
    ) -> Result<Array1<Complex64>> {
        #[cfg(feature = "advanced_math")]
        {
            let start = std::time::Instant::now();
            
            // Convert to SciRS2 sparse format
            let sci_matrix = SparseMatrix::from_csr(
                &matrix.values,
                &matrix.col_indices,
                &matrix.row_ptr,
                matrix.num_rows,
                matrix.num_cols,
                &self.memory_pool,
            )?;
            
            let sci_rhs = Vector::from_array1(rhs, &self.memory_pool)?;
            
            // Solve the linear system
            let sci_solution = sci_matrix.solve(&sci_rhs)?;
            
            // Convert back to ndarray
            let solution = sci_solution.to_array1()?;
            
            self.stats.matrix_ops += 1;
            self.stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            
            Ok(solution)
        }
        
        #[cfg(not(feature = "advanced_math"))]
        {
            Err(SimulatorError::UnsupportedOperation(
                "Sparse solver requires SciRS2 backend (enable 'advanced_math' feature)".to_string()
            ))
        }
    }
}

/// Optimization routines using SciRS2
impl SciRS2Backend {
    /// Minimize a function using SciRS2 optimizers
    pub fn minimize<F>(
        &mut self,
        objective: F,
        initial_params: &[f64],
        method: OptimizationMethod,
    ) -> Result<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync,
    {
        #[cfg(feature = "advanced_math")]
        {
            let start = std::time::Instant::now();
            
            // Create SciRS2 optimizer
            let mut optimizer = match method {
                OptimizationMethod::LBFGS { max_iter, tolerance } => {
                    Optimizer::lbfgs(max_iter, tolerance, &self.memory_pool)?
                },
                OptimizationMethod::ConjugateGradient { max_iter, tolerance } => {
                    Optimizer::cg(max_iter, tolerance, &self.memory_pool)?
                },
                OptimizationMethod::GradientDescent { max_iter, learning_rate } => {
                    Optimizer::gd(max_iter, learning_rate, &self.memory_pool)?
                },
            };
            
            // Run optimization
            let result = optimizer.minimize(objective, initial_params)?;
            
            self.stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            
            Ok(OptimizationResult {
                optimal_params: result.params,
                optimal_value: result.value,
                iterations: result.iterations,
                converged: result.converged,
                function_evaluations: result.func_evals,
            })
        }
        
        #[cfg(not(feature = "advanced_math"))]
        {
            Err(SimulatorError::UnsupportedOperation(
                "Optimization requires SciRS2 backend (enable 'advanced_math' feature)".to_string()
            ))
        }
    }
}

/// Optimization methods available in SciRS2
#[derive(Debug, Clone)]
pub enum OptimizationMethod {
    /// Limited-memory BFGS
    LBFGS { max_iter: usize, tolerance: f64 },
    /// Conjugate gradient
    ConjugateGradient { max_iter: usize, tolerance: f64 },
    /// Gradient descent
    GradientDescent { max_iter: usize, learning_rate: f64 },
}

/// Result of optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimal parameters found
    pub optimal_params: Vec<f64>,
    /// Optimal function value
    pub optimal_value: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Number of function evaluations
    pub function_evaluations: usize,
}

/// Global SciRS2 backend instance
static mut GLOBAL_BACKEND: Option<SciRS2Backend> = None;
static BACKEND_INIT: std::sync::Once = std::sync::Once::new();

/// Get global SciRS2 backend instance
pub fn get_backend() -> &'static mut SciRS2Backend {
    unsafe {
        BACKEND_INIT.call_once(|| {
            GLOBAL_BACKEND = Some(SciRS2Backend::new().unwrap_or_else(|_| {
                // Fallback if SciRS2 initialization fails
                SciRS2Backend {
                    available: false,
                    stats: BackendStats::default(),
                }
            }));
        });
        
        GLOBAL_BACKEND.as_mut().unwrap()
    }
}

/// Benchmark SciRS2 operations
pub fn benchmark_scirs2_ops() -> Result<BenchmarkResults> {
    let mut backend = SciRS2Backend::new()?;
    let mut results = BenchmarkResults::default();
    
    // Benchmark matrix-vector multiplication
    let sizes = vec![64, 128, 256, 512, 1024];
    for size in sizes {
        let matrix = Array2::from_elem((size, size), Complex64::new(1.0, 0.0));
        let vector = Array1::from_elem(size, Complex64::new(1.0, 0.0));
        let mut result = Array1::zeros(size);
        
        let start = std::time::Instant::now();
        backend.gemv(&matrix.view(), &vector.view(), &mut result)?;
        let elapsed = start.elapsed();
        
        results.gemv_times.push((size, elapsed.as_nanos() as f64 / 1e6));
    }
    
    // Benchmark FFT
    for &size in &[64, 128, 256, 512, 1024, 2048] {
        let input = Array1::from_elem(size, Complex64::new(1.0, 0.0));
        
        let start = std::time::Instant::now();
        let _output = backend.fft(&input.view())?;
        let elapsed = start.elapsed();
        
        results.fft_times.push((size, elapsed.as_nanos() as f64 / 1e6));
    }
    
    results.backend_stats = backend.get_stats().clone();
    Ok(results)
}

/// Benchmark results
#[derive(Debug, Clone, Default)]
pub struct BenchmarkResults {
    /// GEMV benchmark times (size, time_ms)
    pub gemv_times: Vec<(usize, f64)>,
    /// FFT benchmark times (size, time_ms)
    pub fft_times: Vec<(usize, f64)>,
    /// Backend statistics
    pub backend_stats: BackendStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let backend = SciRS2Backend::new();
        assert!(backend.is_ok());
    }

    #[test]
    fn test_fallback_gemv() {
        let mut backend = SciRS2Backend::new().unwrap();
        
        let matrix = Array2::from_elem((3, 3), Complex64::new(1.0, 0.0));
        let vector = Array1::from_elem(3, Complex64::new(2.0, 0.0));
        let mut result = Array1::zeros(3);
        
        let res = backend.gemv(&matrix.view(), &vector.view(), &mut result);
        assert!(res.is_ok());
        
        // Each element should be 3 * 2 = 6
        for &val in result.iter() {
            assert_eq!(val, Complex64::new(6.0, 0.0));
        }
    }

    #[test]
    fn test_fallback_fft() {
        let mut backend = SciRS2Backend::new().unwrap();
        
        let input = Array1::from_elem(4, Complex64::new(1.0, 0.0));
        let result = backend.fft(&input.view());
        
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 4);
    }
}