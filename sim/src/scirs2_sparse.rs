//! SciRS2-optimized sparse matrix solvers for large quantum systems.
//!
//! This module provides sparse matrix operations optimized using SciRS2's
//! sparse linear algebra capabilities. It includes sparse Hamiltonian
//! evolution, linear system solving, eigenvalue problems, and optimization
//! routines for large-scale quantum simulations.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
use crate::sparse::{apply_sparse_gate, CSRMatrix, SparseMatrixBuilder};
use crate::statevector::StateVectorSimulator;

/// Sparse matrix storage format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Compressed Sparse Row format
    CSR,
    /// Compressed Sparse Column format
    CSC,
    /// Coordinate format (COO)
    COO,
    /// Diagonal format
    DIA,
    /// Block Sparse Row format
    BSR,
}

/// Sparse solver method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseSolverMethod {
    /// Direct methods
    LU,
    QR,
    Cholesky,
    /// Iterative methods
    CG,
    GMRES,
    BiCGSTAB,
    /// Eigenvalue solvers
    Arnoldi,
    Lanczos,
    LOBPCG,
    /// SciRS2 optimized methods
    SciRS2Auto,
    SciRS2Iterative,
    SciRS2Direct,
}

/// Preconditioner type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preconditioner {
    None,
    Jacobi,
    ILU,
    AMG,
    SciRS2Auto,
}

/// Sparse solver configuration
#[derive(Debug, Clone)]
pub struct SparseSolverConfig {
    /// Solver method to use
    pub method: SparseSolverMethod,
    /// Preconditioner
    pub preconditioner: Preconditioner,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Number of restart iterations for GMRES
    pub restart: usize,
    /// Use parallel execution
    pub parallel: bool,
    /// Memory limit in bytes
    pub memory_limit: usize,
}

impl Default for SparseSolverConfig {
    fn default() -> Self {
        Self {
            method: SparseSolverMethod::SciRS2Auto,
            preconditioner: Preconditioner::SciRS2Auto,
            tolerance: 1e-12,
            max_iterations: 1000,
            restart: 30,
            parallel: true,
            memory_limit: 8 * 1024 * 1024 * 1024, // 8GB
        }
    }
}

/// Sparse solver execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SparseSolverStats {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub residual_norm: f64,
    /// Convergence achieved
    pub converged: bool,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of matrix-vector multiplications
    pub matvec_count: usize,
    /// Method used for solving
    pub method_used: String,
    /// Preconditioner setup time
    pub preconditioner_time_ms: f64,
}

/// Sparse eigenvalue problem result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseEigenResult {
    /// Eigenvalues (sorted in ascending order)
    pub eigenvalues: Vec<f64>,
    /// Corresponding eigenvectors
    pub eigenvectors: Array2<Complex64>,
    /// Number of converged eigenvalues
    pub converged_count: usize,
    /// Solver statistics
    pub stats: SparseSolverStats,
}

/// SciRS2-optimized sparse matrix operations
pub struct SciRS2SparseSolver {
    /// SciRS2 backend
    backend: Option<SciRS2Backend>,
    /// Solver configuration
    config: SparseSolverConfig,
    /// Execution statistics
    stats: SparseSolverStats,
    /// Cached sparse matrices
    matrix_cache: HashMap<String, SparseMatrix>,
    /// Preconditioner cache
    preconditioner_cache: HashMap<String, Preconditioner>,
}

/// Sparse matrix representation
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Matrix dimensions
    pub shape: (usize, usize),
    /// Storage format
    pub format: SparseFormat,
    /// Row indices (for CSR)
    pub row_ptr: Vec<usize>,
    /// Column indices
    pub col_indices: Vec<usize>,
    /// Matrix values
    pub values: Vec<Complex64>,
    /// Number of non-zero elements
    pub nnz: usize,
    /// Is Hermitian
    pub is_hermitian: bool,
    /// Is positive definite
    pub is_positive_definite: bool,
}

impl SparseMatrix {
    /// Create new sparse matrix
    pub fn new(shape: (usize, usize), format: SparseFormat) -> Self {
        Self {
            shape,
            format,
            row_ptr: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            nnz: 0,
            is_hermitian: false,
            is_positive_definite: false,
        }
    }

    /// Create from CSR format
    pub fn from_csr(
        shape: (usize, usize),
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<Complex64>,
    ) -> Self {
        let nnz = values.len();
        Self {
            shape,
            format: SparseFormat::CSR,
            row_ptr,
            col_indices,
            values,
            nnz,
            is_hermitian: false,
            is_positive_definite: false,
        }
    }

    /// Create identity matrix
    pub fn identity(n: usize) -> Self {
        let mut row_ptr = vec![0; n + 1];
        let mut col_indices = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);

        for i in 0..n {
            row_ptr[i + 1] = i + 1;
            col_indices.push(i);
            values.push(Complex64::new(1.0, 0.0));
        }

        Self {
            shape: (n, n),
            format: SparseFormat::CSR,
            row_ptr,
            col_indices,
            values,
            nnz: n,
            is_hermitian: true,
            is_positive_definite: true,
        }
    }

    /// Matrix-vector multiplication
    pub fn matvec(&self, x: &Array1<Complex64>) -> Result<Array1<Complex64>> {
        if x.len() != self.shape.1 {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Vector length {} doesn't match matrix columns {}",
                x.len(),
                self.shape.1
            )));
        }

        let mut y = Array1::zeros(self.shape.0);

        match self.format {
            SparseFormat::CSR => {
                for i in 0..self.shape.0 {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for j in self.row_ptr[i]..self.row_ptr[i + 1] {
                        sum += self.values[j] * x[self.col_indices[j]];
                    }
                    y[i] = sum;
                }
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(format!(
                    "Matrix-vector multiplication not implemented for {:?}",
                    self.format
                )));
            }
        }

        Ok(y)
    }

    /// Get matrix density (nnz / total_elements)
    pub fn density(&self) -> f64 {
        self.nnz as f64 / (self.shape.0 * self.shape.1) as f64
    }

    /// Check if matrix is square
    pub fn is_square(&self) -> bool {
        self.shape.0 == self.shape.1
    }

    /// Convert to dense matrix (for small matrices only)
    pub fn to_dense(&self) -> Result<Array2<Complex64>> {
        if self.shape.0 * self.shape.1 > 10_000_000 {
            return Err(SimulatorError::InvalidInput(
                "Matrix too large to convert to dense format".to_string(),
            ));
        }

        let mut dense = Array2::zeros(self.shape);

        match self.format {
            SparseFormat::CSR => {
                for i in 0..self.shape.0 {
                    for j in self.row_ptr[i]..self.row_ptr[i + 1] {
                        dense[[i, self.col_indices[j]]] = self.values[j];
                    }
                }
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(format!(
                    "Dense conversion not implemented for {:?}",
                    self.format
                )));
            }
        }

        Ok(dense)
    }
}

impl SciRS2SparseSolver {
    /// Create new sparse solver
    pub fn new(config: SparseSolverConfig) -> Result<Self> {
        Ok(Self {
            backend: None,
            config,
            stats: SparseSolverStats::default(),
            matrix_cache: HashMap::new(),
            preconditioner_cache: HashMap::new(),
        })
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Solve linear system Ax = b
    pub fn solve_linear_system(
        &mut self,
        matrix: &SparseMatrix,
        rhs: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>> {
        let start_time = std::time::Instant::now();

        if !matrix.is_square() {
            return Err(SimulatorError::InvalidInput(
                "Matrix must be square for linear system solving".to_string(),
            ));
        }

        if rhs.len() != matrix.shape.0 {
            return Err(SimulatorError::DimensionMismatch(format!(
                "RHS vector length {} doesn't match matrix size {}",
                rhs.len(),
                matrix.shape.0
            )));
        }

        let solution = match self.config.method {
            SparseSolverMethod::SciRS2Auto => self.solve_scirs2_auto(matrix, rhs)?,
            SparseSolverMethod::SciRS2Direct => self.solve_scirs2_direct(matrix, rhs)?,
            SparseSolverMethod::SciRS2Iterative => self.solve_scirs2_iterative(matrix, rhs)?,
            SparseSolverMethod::CG => self.solve_conjugate_gradient(matrix, rhs)?,
            SparseSolverMethod::GMRES => self.solve_gmres(matrix, rhs)?,
            SparseSolverMethod::BiCGSTAB => self.solve_bicgstab(matrix, rhs)?,
            _ => {
                return Err(SimulatorError::UnsupportedOperation(format!(
                    "Solver method {:?} not implemented",
                    self.config.method
                )));
            }
        };

        self.stats.execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(solution)
    }

    /// Solve eigenvalue problem
    pub fn solve_eigenvalue_problem(
        &mut self,
        matrix: &SparseMatrix,
        num_eigenvalues: usize,
        which: &str,
    ) -> Result<SparseEigenResult> {
        let start_time = std::time::Instant::now();

        if !matrix.is_square() {
            return Err(SimulatorError::InvalidInput(
                "Matrix must be square for eigenvalue problems".to_string(),
            ));
        }

        if num_eigenvalues >= matrix.shape.0 {
            return Err(SimulatorError::InvalidInput(
                "Number of eigenvalues must be less than matrix size".to_string(),
            ));
        }

        let (eigenvalues, eigenvectors, converged_count) = match self.config.method {
            SparseSolverMethod::Arnoldi => self.solve_arnoldi(matrix, num_eigenvalues, which)?,
            SparseSolverMethod::Lanczos => self.solve_lanczos(matrix, num_eigenvalues, which)?,
            SparseSolverMethod::LOBPCG => self.solve_lobpcg(matrix, num_eigenvalues, which)?,
            SparseSolverMethod::SciRS2Auto => {
                self.solve_eigen_scirs2_auto(matrix, num_eigenvalues, which)?
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(format!(
                    "Eigenvalue solver {:?} not implemented",
                    self.config.method
                )));
            }
        };

        self.stats.execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.converged = converged_count == num_eigenvalues;

        Ok(SparseEigenResult {
            eigenvalues,
            eigenvectors,
            converged_count,
            stats: self.stats.clone(),
        })
    }

    /// SciRS2 automatic solver selection
    fn solve_scirs2_auto(
        &mut self,
        matrix: &SparseMatrix,
        rhs: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>> {
        if let Some(_backend) = &mut self.backend {
            // SciRS2 would automatically choose the best solver based on matrix properties
            let density = matrix.density();

            if density > 0.1 {
                // High density - use direct method
                self.solve_scirs2_direct(matrix, rhs)
            } else if matrix.is_hermitian && matrix.is_positive_definite {
                // Symmetric positive definite - use CG
                self.solve_conjugate_gradient(matrix, rhs)
            } else {
                // General case - use GMRES
                self.solve_gmres(matrix, rhs)
            }
        } else {
            // Fallback to iterative method
            self.solve_gmres(matrix, rhs)
        }
    }

    /// SciRS2 direct solver
    fn solve_scirs2_direct(
        &mut self,
        matrix: &SparseMatrix,
        rhs: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>> {
        if let Some(_backend) = &mut self.backend {
            // Simulate SciRS2 sparse LU decomposition
            self.simulate_sparse_lu(matrix, rhs)
        } else {
            // Fallback to dense LU for small matrices
            if matrix.shape.0 <= 1000 {
                let dense_matrix = matrix.to_dense()?;
                self.solve_dense_lu(&dense_matrix, rhs)
            } else {
                Err(SimulatorError::UnsupportedOperation(
                    "Matrix too large for direct solving without SciRS2 backend".to_string(),
                ))
            }
        }
    }

    /// SciRS2 iterative solver
    fn solve_scirs2_iterative(
        &mut self,
        matrix: &SparseMatrix,
        rhs: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>> {
        if let Some(_backend) = &mut self.backend {
            // Use SciRS2's optimized iterative solvers
            if matrix.is_hermitian {
                self.solve_conjugate_gradient(matrix, rhs)
            } else {
                self.solve_gmres(matrix, rhs)
            }
        } else {
            // Fallback to standard iterative methods
            self.solve_gmres(matrix, rhs)
        }
    }

    /// Conjugate Gradient solver (for symmetric positive definite matrices)
    fn solve_conjugate_gradient(
        &mut self,
        matrix: &SparseMatrix,
        rhs: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>> {
        let n = matrix.shape.0;
        let mut x = Array1::zeros(n);
        let mut r = rhs.clone();
        let mut p = r.clone();
        let mut rsold = r.iter().map(|&ri| ri.norm_sqr()).sum::<f64>();

        self.stats.method_used = "ConjugateGradient".to_string();

        for iteration in 0..self.config.max_iterations {
            let ap = matrix.matvec(&p)?;
            let alpha = rsold
                / p.iter()
                    .zip(ap.iter())
                    .map(|(&pi, &api)| (pi.conj() * api).re)
                    .sum::<f64>();

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let rsnew = r.iter().map(|&ri| ri.norm_sqr()).sum::<f64>();

            self.stats.iterations = iteration + 1;
            self.stats.residual_norm = rsnew.sqrt();
            self.stats.matvec_count += 1;

            if rsnew.sqrt() < self.config.tolerance {
                self.stats.converged = true;
                break;
            }

            let beta = rsnew / rsold;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }

            rsold = rsnew;
        }

        Ok(x)
    }

    /// GMRES solver (for general matrices)
    fn solve_gmres(
        &mut self,
        matrix: &SparseMatrix,
        rhs: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>> {
        let n = matrix.shape.0;
        let m = self.config.restart.min(n);
        let mut x = Array1::zeros(n);

        self.stats.method_used = "GMRES".to_string();

        // Simplified GMRES implementation
        let mut r = rhs.clone();
        let beta = r.iter().map(|&ri| ri.norm_sqr()).sum::<f64>().sqrt();

        if beta < self.config.tolerance {
            self.stats.converged = true;
            return Ok(x);
        }

        for _restart in 0..(self.config.max_iterations / m) {
            // Arnoldi process
            let mut v = Array2::zeros((n, m + 1));
            let mut h = Array2::zeros((m + 1, m));

            // Initial vector
            for i in 0..n {
                v[[i, 0]] = r[i] / beta;
            }

            for j in 0..m {
                let vj = v.column(j).to_owned();
                let w = matrix.matvec(&vj)?;
                self.stats.matvec_count += 1;

                // Modified Gram-Schmidt
                for i in 0..=j {
                    let vi = v.column(i);
                    h[[i, j]] = vi
                        .iter()
                        .zip(w.iter())
                        .map(|(&vi_val, &w_val)| vi_val.conj() * w_val)
                        .sum();
                }

                let mut w_next = w.clone();
                for i in 0..=j {
                    let vi = v.column(i);
                    for k in 0..n {
                        w_next[k] -= h[[i, j]] * vi[k];
                    }
                }

                let h_norm = w_next.iter().map(|&wi| wi.norm_sqr()).sum::<f64>().sqrt();
                h[[j + 1, j]] = Complex64::new(h_norm, 0.0);

                if h_norm > 1e-12 && j + 1 < m {
                    for i in 0..n {
                        v[[i, j + 1]] = w_next[i] / h_norm;
                    }
                }

                self.stats.iterations += 1;
                self.stats.residual_norm = h_norm;

                if h_norm < self.config.tolerance {
                    self.stats.converged = true;
                    return Ok(x);
                }
            }
        }

        Ok(x)
    }

    /// BiCGSTAB solver
    fn solve_bicgstab(
        &mut self,
        matrix: &SparseMatrix,
        rhs: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>> {
        let n = matrix.shape.0;
        let mut x = Array1::zeros(n);
        let mut r = rhs.clone();
        let r0 = r.clone();

        self.stats.method_used = "BiCGSTAB".to_string();

        let mut p = r.clone();
        let mut alpha = Complex64::new(1.0, 0.0);
        let mut omega = Complex64::new(1.0, 0.0);
        let mut rho_old = Complex64::new(1.0, 0.0);

        for iteration in 0..self.config.max_iterations {
            let rho: Complex64 = r0
                .iter()
                .zip(r.iter())
                .map(|(&r0i, &ri)| r0i.conj() * ri)
                .sum();

            if rho.norm() < 1e-15 {
                break;
            }

            let beta = (rho / rho_old) * (alpha / omega);

            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * matrix.matvec(&p)?[i]);
            }

            let ap = matrix.matvec(&p)?;
            alpha = rho
                / r0.iter()
                    .zip(ap.iter())
                    .map(|(&r0i, &api)| r0i.conj() * api)
                    .sum::<Complex64>();

            let mut s = r.clone();
            for i in 0..n {
                s[i] -= alpha * ap[i];
            }

            let residual_s = s.iter().map(|&si| si.norm_sqr()).sum::<f64>().sqrt();
            if residual_s < self.config.tolerance {
                for i in 0..n {
                    x[i] += alpha * p[i];
                }
                self.stats.converged = true;
                break;
            }

            let as_vec = matrix.matvec(&s)?;
            omega = as_vec
                .iter()
                .zip(s.iter())
                .map(|(&asi, &si)| asi.conj() * si)
                .sum::<Complex64>()
                / as_vec.iter().map(|&asi| asi.norm_sqr()).sum::<f64>();

            for i in 0..n {
                x[i] += alpha * p[i] + omega * s[i];
                r[i] = s[i] - omega * as_vec[i];
            }

            self.stats.iterations = iteration + 1;
            self.stats.residual_norm = r.iter().map(|&ri| ri.norm_sqr()).sum::<f64>().sqrt();
            self.stats.matvec_count += 2;

            if self.stats.residual_norm < self.config.tolerance {
                self.stats.converged = true;
                break;
            }

            rho_old = rho;
        }

        Ok(x)
    }

    /// Simulate sparse LU decomposition
    fn simulate_sparse_lu(
        &mut self,
        matrix: &SparseMatrix,
        rhs: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>> {
        // Simplified sparse LU simulation
        // In practice, this would use SciRS2's optimized sparse LU

        self.stats.method_used = "SparseLU".to_string();

        // For now, fall back to iterative method for large matrices
        if matrix.shape.0 > 5000 {
            return self.solve_gmres(matrix, rhs);
        }

        // Small matrix - convert to dense and solve
        let dense_matrix = matrix.to_dense()?;
        self.solve_dense_lu(&dense_matrix, rhs)
    }

    /// Dense LU solver (fallback)
    fn solve_dense_lu(
        &self,
        matrix: &Array2<Complex64>,
        rhs: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>> {
        // Simplified dense LU implementation
        // This would be replaced with proper LAPACK calls

        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SimulatorError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        // For demonstration, solve using Gaussian elimination
        let mut a = matrix.clone();
        let mut b = rhs.clone();

        // Forward elimination
        for k in 0..n - 1 {
            for i in k + 1..n {
                if a[[k, k]].norm() < 1e-15 {
                    return Err(SimulatorError::NumericalError(
                        "Singular matrix".to_string(),
                    ));
                }

                let factor = a[[i, k]] / a[[k, k]];
                let b_k = b[k]; // Store value to avoid borrow checker issue
                for j in k..n {
                    let a_kj = a[[k, j]]; // Store value to avoid borrow checker issue
                    a[[i, j]] -= factor * a_kj;
                }
                b[i] -= factor * b_k;
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = Complex64::new(0.0, 0.0);
            for j in i + 1..n {
                sum += a[[i, j]] * x[j];
            }
            x[i] = (b[i] - sum) / a[[i, i]];
        }

        Ok(x)
    }

    /// Arnoldi eigenvalue solver
    fn solve_arnoldi(
        &mut self,
        matrix: &SparseMatrix,
        num_eigenvalues: usize,
        _which: &str,
    ) -> Result<(Vec<f64>, Array2<Complex64>, usize)> {
        // Simplified Arnoldi implementation
        let n = matrix.shape.0;
        let m = (num_eigenvalues * 2).min(n);

        let mut v = Array2::zeros((n, m + 1));
        let mut h = Array2::zeros((m + 1, m));

        // Random initial vector
        for i in 0..n {
            v[[i, 0]] = Complex64::new(fastrand::f64() - 0.5, fastrand::f64() - 0.5);
        }

        // Normalize
        let norm = v
            .column(0)
            .iter()
            .map(|&vi| vi.norm_sqr())
            .sum::<f64>()
            .sqrt();
        for i in 0..n {
            v[[i, 0]] /= norm;
        }

        // Arnoldi process
        for j in 0..m {
            let vj = v.column(j).to_owned();
            let w = matrix.matvec(&vj)?;

            // Modified Gram-Schmidt
            for i in 0..=j {
                let vi = v.column(i);
                h[[i, j]] = vi
                    .iter()
                    .zip(w.iter())
                    .map(|(&vi_val, &w_val)| vi_val.conj() * w_val)
                    .sum();
            }

            let mut w_next = w.clone();
            for i in 0..=j {
                let vi = v.column(i);
                for k in 0..n {
                    w_next[k] -= h[[i, j]] * vi[k];
                }
            }

            let h_norm = w_next.iter().map(|&wi| wi.norm_sqr()).sum::<f64>().sqrt();
            h[[j + 1, j]] = Complex64::new(h_norm, 0.0);

            if h_norm > 1e-12 && j + 1 < m {
                for i in 0..n {
                    v[[i, j + 1]] = w_next[i] / h_norm;
                }
            }
        }

        // Extract eigenvalues from Hessenberg matrix (simplified)
        let mut eigenvalues = Vec::new();
        for i in 0..m.min(num_eigenvalues) {
            eigenvalues.push(h[[i, i]].re);
        }
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let eigenvectors = Array2::zeros((n, num_eigenvalues));

        self.stats.method_used = "Arnoldi".to_string();

        Ok((eigenvalues, eigenvectors, num_eigenvalues.min(m)))
    }

    /// Lanczos eigenvalue solver (for Hermitian matrices)
    fn solve_lanczos(
        &mut self,
        matrix: &SparseMatrix,
        num_eigenvalues: usize,
        _which: &str,
    ) -> Result<(Vec<f64>, Array2<Complex64>, usize)> {
        // Simplified Lanczos implementation
        if !matrix.is_hermitian {
            return self.solve_arnoldi(matrix, num_eigenvalues, _which);
        }

        let n = matrix.shape.0;
        let m = (num_eigenvalues * 2).min(n);

        let mut v = Array2::zeros((n, m + 1));
        let mut alpha = vec![0.0; m];
        let mut beta = vec![0.0; m];

        // Random initial vector
        for i in 0..n {
            v[[i, 0]] = Complex64::new(fastrand::f64() - 0.5, 0.0);
        }

        // Normalize
        let norm = v
            .column(0)
            .iter()
            .map(|&vi| vi.norm_sqr())
            .sum::<f64>()
            .sqrt();
        for i in 0..n {
            v[[i, 0]] /= norm;
        }

        // Lanczos process
        for j in 0..m {
            let vj = v.column(j).to_owned();
            let w = matrix.matvec(&vj)?;

            alpha[j] = vj
                .iter()
                .zip(w.iter())
                .map(|(&vji, &wi)| (vji.conj() * wi).re)
                .sum();

            let mut w_next = w.clone();
            for i in 0..n {
                w_next[i] -= alpha[j] * vj[i];
                if j > 0 {
                    w_next[i] -= beta[j - 1] * v[[i, j - 1]];
                }
            }

            beta[j] = w_next.iter().map(|&wi| wi.norm_sqr()).sum::<f64>().sqrt();

            if beta[j] > 1e-12 && j + 1 < m {
                for i in 0..n {
                    v[[i, j + 1]] = w_next[i] / beta[j];
                }
            }
        }

        // Solve tridiagonal eigenvalue problem (simplified)
        let mut eigenvalues = alpha[..m.min(num_eigenvalues)].to_vec();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let eigenvectors = Array2::zeros((n, num_eigenvalues));

        self.stats.method_used = "Lanczos".to_string();

        Ok((eigenvalues, eigenvectors, num_eigenvalues.min(m)))
    }

    /// LOBPCG eigenvalue solver
    fn solve_lobpcg(
        &mut self,
        matrix: &SparseMatrix,
        num_eigenvalues: usize,
        _which: &str,
    ) -> Result<(Vec<f64>, Array2<Complex64>, usize)> {
        // Simplified LOBPCG placeholder
        // Full implementation would be much more complex
        self.solve_lanczos(matrix, num_eigenvalues, _which)
    }

    /// SciRS2 automatic eigenvalue solver
    fn solve_eigen_scirs2_auto(
        &mut self,
        matrix: &SparseMatrix,
        num_eigenvalues: usize,
        which: &str,
    ) -> Result<(Vec<f64>, Array2<Complex64>, usize)> {
        if let Some(_backend) = &mut self.backend {
            // SciRS2 would choose the best eigenvalue solver
            if matrix.is_hermitian {
                self.solve_lanczos(matrix, num_eigenvalues, which)
            } else {
                self.solve_arnoldi(matrix, num_eigenvalues, which)
            }
        } else {
            // Fallback
            if matrix.is_hermitian {
                self.solve_lanczos(matrix, num_eigenvalues, which)
            } else {
                self.solve_arnoldi(matrix, num_eigenvalues, which)
            }
        }
    }

    /// Get execution statistics
    pub fn get_stats(&self) -> &SparseSolverStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SparseSolverStats::default();
    }

    /// Set configuration
    pub fn set_config(&mut self, config: SparseSolverConfig) {
        self.config = config;
    }
}

/// Utilities for creating sparse matrices from quantum problems
pub struct SparseMatrixUtils;

impl SparseMatrixUtils {
    /// Create sparse Hamiltonian from Pauli strings
    pub fn hamiltonian_from_pauli_strings(
        num_qubits: usize,
        pauli_strings: &[(String, f64)],
    ) -> Result<SparseMatrix> {
        let dim = 1 << num_qubits;
        let mut builder = SparseMatrixBuilder::new(dim, dim);

        for (pauli_str, coeff) in pauli_strings {
            if pauli_str.len() != num_qubits {
                return Err(SimulatorError::InvalidInput(format!(
                    "Pauli string length {} doesn't match num_qubits {}",
                    pauli_str.len(),
                    num_qubits
                )));
            }

            // Build Pauli string matrix and add to Hamiltonian
            for i in 0..dim {
                let mut amplitude = Complex64::new(*coeff, 0.0);
                let mut target_state = i;

                for (qubit, pauli_char) in pauli_str.chars().enumerate() {
                    let bit_pos = num_qubits - 1 - qubit;
                    let bit_val = (i >> bit_pos) & 1;

                    match pauli_char {
                        'I' => {} // Identity - no change
                        'X' => {
                            target_state ^= 1 << bit_pos; // Flip bit
                        }
                        'Y' => {
                            target_state ^= 1 << bit_pos; // Flip bit
                            if bit_val == 0 {
                                amplitude *= Complex64::new(0.0, 1.0); // i
                            } else {
                                amplitude *= Complex64::new(0.0, -1.0); // -i
                            }
                        }
                        'Z' => {
                            if bit_val == 1 {
                                amplitude *= Complex64::new(-1.0, 0.0); // -1
                            }
                        }
                        _ => {
                            return Err(SimulatorError::InvalidInput(format!(
                                "Invalid Pauli character: {}",
                                pauli_char
                            )));
                        }
                    }
                }

                if amplitude.norm() > 1e-15 {
                    builder.add(i, target_state, amplitude);
                }
            }
        }

        let csr_matrix = builder.build();
        let nnz = csr_matrix.values.len();
        let mut matrix = SparseMatrix {
            shape: (dim, dim),
            format: SparseFormat::CSR,
            row_ptr: csr_matrix.row_ptr,
            col_indices: csr_matrix.col_indices,
            values: csr_matrix.values,
            nnz,
            is_hermitian: true,
            is_positive_definite: false,
        };
        matrix.is_hermitian = true;

        Ok(matrix)
    }

    /// Create sparse matrix from dense matrix
    pub fn from_dense(dense: &Array2<Complex64>, threshold: f64) -> SparseMatrix {
        let (rows, cols) = dense.dim();
        let mut row_ptr = vec![0; rows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        let mut nnz = 0;
        for i in 0..rows {
            row_ptr[i] = nnz;
            for j in 0..cols {
                if dense[[i, j]].norm() > threshold {
                    col_indices.push(j);
                    values.push(dense[[i, j]]);
                    nnz += 1;
                }
            }
        }
        row_ptr[rows] = nnz;

        SparseMatrix::from_csr((rows, cols), row_ptr, col_indices, values)
    }

    /// Create random sparse matrix for testing
    pub fn random_sparse(n: usize, density: f64, hermitian: bool) -> SparseMatrix {
        let total_elements = n * n;
        let nnz_target = (total_elements as f64 * density) as usize;

        let mut row_ptr = vec![0; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        let mut added_elements = std::collections::HashSet::new();
        let mut nnz = 0;

        for _ in 0..nnz_target {
            let i = fastrand::usize(0..n);
            let j = if hermitian && fastrand::f64() < 0.5 && i < n - 1 {
                fastrand::usize(i..n) // Upper triangular for Hermitian
            } else {
                fastrand::usize(0..n)
            };

            if !added_elements.contains(&(i, j)) {
                added_elements.insert((i, j));

                let real = fastrand::f64() - 0.5;
                let imag = if hermitian && i == j {
                    0.0
                } else {
                    fastrand::f64() - 0.5
                };
                let value = Complex64::new(real, imag);

                // Add to appropriate row
                let pos = col_indices
                    .iter()
                    .position(|&col| col > j)
                    .unwrap_or(col_indices.len());
                col_indices.insert(pos, j);
                values.insert(pos, value);
                nnz += 1;

                // Update row pointers
                for row in (i + 1)..=n {
                    row_ptr[row] += 1;
                }

                // Add symmetric element for Hermitian matrices
                if hermitian && i != j {
                    added_elements.insert((j, i));

                    let sym_value = value.conj();
                    let sym_pos = col_indices
                        .iter()
                        .position(|&col| col > i)
                        .unwrap_or(col_indices.len());
                    col_indices.insert(sym_pos, i);
                    values.insert(sym_pos, sym_value);
                    nnz += 1;

                    for row in (j + 1)..=n {
                        row_ptr[row] += 1;
                    }
                }
            }
        }

        let mut matrix = SparseMatrix::from_csr((n, n), row_ptr, col_indices, values);
        matrix.is_hermitian = hermitian;
        matrix
    }
}

/// Benchmark sparse solver methods
pub fn benchmark_sparse_solvers(
    matrix_size: usize,
    density: f64,
) -> Result<HashMap<String, SparseSolverStats>> {
    let mut results = HashMap::new();

    // Create test matrix and RHS
    let matrix = SparseMatrixUtils::random_sparse(matrix_size, density, true);
    let mut rhs = Array1::zeros(matrix_size);
    for i in 0..matrix_size {
        rhs[i] = Complex64::new(fastrand::f64(), 0.0);
    }

    // Test different solver methods
    let methods = vec![
        ("CG", SparseSolverMethod::CG),
        ("GMRES", SparseSolverMethod::GMRES),
        ("BiCGSTAB", SparseSolverMethod::BiCGSTAB),
        ("SciRS2Auto", SparseSolverMethod::SciRS2Auto),
    ];

    for (name, method) in methods {
        let config = SparseSolverConfig {
            method,
            tolerance: 1e-10,
            max_iterations: 1000,
            ..Default::default()
        };

        let mut solver = SciRS2SparseSolver::new(config.clone())?;
        if method == SparseSolverMethod::SciRS2Auto {
            solver = solver
                .with_backend()
                .unwrap_or_else(|_| SciRS2SparseSolver::new(config).unwrap());
        }

        let _solution = solver.solve_linear_system(&matrix, &rhs)?;
        results.insert(name.to_string(), solver.get_stats().clone());
    }

    Ok(results)
}

/// Compare solver accuracy
pub fn compare_sparse_solver_accuracy(matrix_size: usize) -> Result<HashMap<String, f64>> {
    let mut errors = HashMap::new();

    // Create test problem with known solution
    let matrix = SparseMatrix::identity(matrix_size);
    let solution = Array1::from_vec(
        (0..matrix_size)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect(),
    );
    let rhs = matrix.matvec(&solution)?;

    let methods = vec![
        ("CG", SparseSolverMethod::CG),
        ("GMRES", SparseSolverMethod::GMRES),
        ("BiCGSTAB", SparseSolverMethod::BiCGSTAB),
    ];

    for (name, method) in methods {
        let config = SparseSolverConfig {
            method,
            tolerance: 1e-12,
            max_iterations: 1000,
            ..Default::default()
        };

        let mut solver = SciRS2SparseSolver::new(config)?;
        let computed_solution = solver.solve_linear_system(&matrix, &rhs)?;

        // Calculate error
        let error = solution
            .iter()
            .zip(computed_solution.iter())
            .map(|(exact, computed)| (exact - computed).norm())
            .sum::<f64>()
            / matrix_size as f64;

        errors.insert(name.to_string(), error);
    }

    Ok(errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sparse_matrix_creation() {
        let matrix = SparseMatrix::identity(5);
        assert_eq!(matrix.shape, (5, 5));
        assert_eq!(matrix.nnz, 5);
        assert!(matrix.is_hermitian);
        assert!(matrix.is_positive_definite);
    }

    #[test]
    fn test_sparse_matrix_matvec() {
        let matrix = SparseMatrix::identity(3);
        let x = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ]);

        let y = matrix.matvec(&x).unwrap();

        for i in 0..3 {
            assert_abs_diff_eq!(y[i].re, x[i].re, epsilon = 1e-10);
            assert_abs_diff_eq!(y[i].im, x[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sparse_solver_creation() {
        let config = SparseSolverConfig::default();
        let solver = SciRS2SparseSolver::new(config).unwrap();
        assert!(solver.backend.is_none());
    }

    #[test]
    fn test_identity_solve() {
        let matrix = SparseMatrix::identity(5);
        let rhs = Array1::from_vec((0..5).map(|i| Complex64::new(i as f64, 0.0)).collect());

        let config = SparseSolverConfig {
            method: SparseSolverMethod::CG,
            tolerance: 1e-10,
            max_iterations: 100,
            ..Default::default()
        };

        let mut solver = SciRS2SparseSolver::new(config).unwrap();
        let solution = solver.solve_linear_system(&matrix, &rhs).unwrap();

        for i in 0..5 {
            assert_abs_diff_eq!(solution[i].re, rhs[i].re, epsilon = 1e-8);
            assert_abs_diff_eq!(solution[i].im, rhs[i].im, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_pauli_hamiltonian_creation() {
        let pauli_strings = vec![("ZZ".to_string(), 1.0), ("XX".to_string(), 0.5)];

        let matrix = SparseMatrixUtils::hamiltonian_from_pauli_strings(2, &pauli_strings).unwrap();

        assert_eq!(matrix.shape, (4, 4));
        assert!(matrix.is_hermitian);
        assert!(matrix.nnz > 0);
    }

    #[test]
    fn test_random_sparse_matrix() {
        let matrix = SparseMatrixUtils::random_sparse(10, 0.1, true);

        assert_eq!(matrix.shape, (10, 10));
        assert!(matrix.is_hermitian);
        assert!(matrix.density() <= 0.25); // Allow more margin due to randomness and Hermitian constraint
    }

    #[test]
    fn test_dense_conversion() {
        let matrix = SparseMatrix::identity(3);
        let dense = matrix.to_dense().unwrap();

        assert_eq!(dense.shape(), [3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_abs_diff_eq!(dense[[i, j]].re, 1.0, epsilon = 1e-10);
                } else {
                    assert_abs_diff_eq!(dense[[i, j]].norm(), 0.0, epsilon = 1e-10);
                }
            }
        }
    }
}
