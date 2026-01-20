//! Matrix operations for quantum gates using SciRS2
//!
//! This module provides efficient matrix operations for quantum computing,
//! including sparse/dense conversions, tensor products, and specialized
//! quantum operations.

use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::Complex64;
// use scirs2_sparse::csr::CsrMatrix;
use crate::linalg_stubs::CsrMatrix;
use std::fmt::Debug;

/// Trait for quantum matrix operations
pub trait QuantumMatrix: Debug + Send + Sync {
    /// Get the dimension of the matrix (assumed square)
    fn dim(&self) -> usize;

    /// Convert to dense representation
    fn to_dense(&self) -> Array2<Complex64>;

    /// Convert to sparse representation
    fn to_sparse(&self) -> QuantRS2Result<CsrMatrix<Complex64>>;

    /// Check if the matrix is unitary
    fn is_unitary(&self, tolerance: f64) -> QuantRS2Result<bool>;

    /// Compute the tensor product with another matrix
    fn tensor_product(&self, other: &dyn QuantumMatrix) -> QuantRS2Result<Array2<Complex64>>;

    /// Apply to a state vector
    fn apply(&self, state: &ArrayView1<Complex64>) -> QuantRS2Result<Array1<Complex64>>;
}

/// Dense matrix representation
#[derive(Debug, Clone)]
pub struct DenseMatrix {
    data: Array2<Complex64>,
}

impl DenseMatrix {
    /// Create a new dense matrix
    pub fn new(data: Array2<Complex64>) -> QuantRS2Result<Self> {
        if data.nrows() != data.ncols() {
            return Err(QuantRS2Error::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }
        Ok(Self { data })
    }

    /// Create from a flat vector (column-major order)
    pub fn from_vec(data: Vec<Complex64>, dim: usize) -> QuantRS2Result<Self> {
        if data.len() != dim * dim {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Expected {} elements, got {}",
                dim * dim,
                data.len()
            )));
        }
        let matrix = Array2::from_shape_vec((dim, dim), data)
            .map_err(|e| QuantRS2Error::InvalidInput(e.to_string()))?;
        Self::new(matrix)
    }

    /// Get a reference to the underlying array
    pub const fn as_array(&self) -> &Array2<Complex64> {
        &self.data
    }

    /// Check if matrix is hermitian
    pub fn is_hermitian(&self, tolerance: f64) -> bool {
        let n = self.data.nrows();
        for i in 0..n {
            for j in i..n {
                let diff = (self.data[[i, j]] - self.data[[j, i]].conj()).norm();
                if diff > tolerance {
                    return false;
                }
            }
        }
        true
    }
}

impl QuantumMatrix for DenseMatrix {
    fn dim(&self) -> usize {
        self.data.nrows()
    }

    fn to_dense(&self) -> Array2<Complex64> {
        self.data.clone()
    }

    fn to_sparse(&self) -> QuantRS2Result<CsrMatrix<Complex64>> {
        let n = self.dim();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();

        let tolerance = 1e-14;
        for i in 0..n {
            for j in 0..n {
                let val = self.data[[i, j]];
                if val.norm() > tolerance {
                    rows.push(i);
                    cols.push(j);
                    data.push(val);
                }
            }
        }

        CsrMatrix::new(data, rows, cols, (n, n)).map_err(|e| QuantRS2Error::InvalidInput(e))
    }

    fn is_unitary(&self, tolerance: f64) -> QuantRS2Result<bool> {
        let n = self.dim();
        let conj_transpose = self.data.t().mapv(|x| x.conj());
        let product = self.data.dot(&conj_transpose);

        // Check if product is identity
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (product[[i, j]] - expected).norm();
                if diff > tolerance {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn tensor_product(&self, other: &dyn QuantumMatrix) -> QuantRS2Result<Array2<Complex64>> {
        let other_dense = other.to_dense();
        let n1 = self.dim();
        let n2 = other_dense.nrows();
        let n = n1 * n2;

        let mut result = Array2::zeros((n, n));

        for i1 in 0..n1 {
            for j1 in 0..n1 {
                let val1 = self.data[[i1, j1]];
                for i2 in 0..n2 {
                    for j2 in 0..n2 {
                        let val2 = other_dense[[i2, j2]];
                        result[[i1 * n2 + i2, j1 * n2 + j2]] = val1 * val2;
                    }
                }
            }
        }

        Ok(result)
    }

    fn apply(&self, state: &ArrayView1<Complex64>) -> QuantRS2Result<Array1<Complex64>> {
        if state.len() != self.dim() {
            return Err(QuantRS2Error::InvalidInput(format!(
                "State dimension {} doesn't match matrix dimension {}",
                state.len(),
                self.dim()
            )));
        }
        Ok(self.data.dot(state))
    }
}

/// Sparse matrix representation for quantum gates
#[derive(Clone)]
pub struct SparseMatrix {
    csr: CsrMatrix<Complex64>,
    dim: usize,
}

impl Debug for SparseMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseMatrix")
            .field("dim", &self.dim)
            .field("nnz", &self.csr.nnz())
            .finish()
    }
}

impl SparseMatrix {
    /// Create a new sparse matrix
    pub fn new(csr: CsrMatrix<Complex64>) -> QuantRS2Result<Self> {
        let (rows, cols) = csr.shape();
        if rows != cols {
            return Err(QuantRS2Error::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }
        Ok(Self { csr, dim: rows })
    }

    /// Create from triplets
    pub fn from_triplets(
        rows: Vec<usize>,
        cols: Vec<usize>,
        data: Vec<Complex64>,
        dim: usize,
    ) -> QuantRS2Result<Self> {
        let csr = CsrMatrix::new(data, rows, cols, (dim, dim))
            .map_err(|e| QuantRS2Error::InvalidInput(e))?;
        Self::new(csr)
    }
}

impl QuantumMatrix for SparseMatrix {
    fn dim(&self) -> usize {
        self.dim
    }

    fn to_dense(&self) -> Array2<Complex64> {
        self.csr.to_dense()
    }

    fn to_sparse(&self) -> QuantRS2Result<CsrMatrix<Complex64>> {
        Ok(self.csr.clone())
    }

    fn is_unitary(&self, tolerance: f64) -> QuantRS2Result<bool> {
        // Convert to dense for unitary check
        let dense = DenseMatrix::new(self.to_dense())?;
        dense.is_unitary(tolerance)
    }

    fn tensor_product(&self, other: &dyn QuantumMatrix) -> QuantRS2Result<Array2<Complex64>> {
        // For now, convert to dense and compute
        // TODO: Implement sparse tensor product
        let dense = DenseMatrix::new(self.to_dense())?;
        dense.tensor_product(other)
    }

    fn apply(&self, state: &ArrayView1<Complex64>) -> QuantRS2Result<Array1<Complex64>> {
        if state.len() != self.dim() {
            return Err(QuantRS2Error::InvalidInput(format!(
                "State dimension {} doesn't match matrix dimension {}",
                state.len(),
                self.dim()
            )));
        }
        // Convert to dense and apply
        let dense = self.to_dense();
        Ok(dense.dot(state))
    }
}

/// Compute the partial trace of a matrix
pub fn partial_trace(
    matrix: &Array2<Complex64>,
    keep_qubits: &[usize],
    total_qubits: usize,
) -> QuantRS2Result<Array2<Complex64>> {
    let full_dim = 1 << total_qubits;
    if matrix.nrows() != full_dim || matrix.ncols() != full_dim {
        return Err(QuantRS2Error::InvalidInput(format!(
            "Matrix dimension {} doesn't match {} qubits",
            matrix.nrows(),
            total_qubits
        )));
    }

    let keep_dim = 1 << keep_qubits.len();
    let trace_qubits: Vec<usize> = (0..total_qubits)
        .filter(|q| !keep_qubits.contains(q))
        .collect();
    let trace_dim = 1 << trace_qubits.len();

    let mut result = Array2::zeros((keep_dim, keep_dim));

    // Iterate over all basis states
    for i in 0..keep_dim {
        for j in 0..keep_dim {
            let mut sum = Complex64::new(0.0, 0.0);

            // Sum over traced out qubits
            for t in 0..trace_dim {
                let row_idx = reconstruct_index(i, t, keep_qubits, &trace_qubits, total_qubits);
                let col_idx = reconstruct_index(j, t, keep_qubits, &trace_qubits, total_qubits);
                sum += matrix[[row_idx, col_idx]];
            }

            result[[i, j]] = sum;
        }
    }

    Ok(result)
}

/// Helper function to reconstruct full index from partial indices
fn reconstruct_index(
    keep_idx: usize,
    trace_idx: usize,
    keep_qubits: &[usize],
    trace_qubits: &[usize],
    _total_qubits: usize,
) -> usize {
    let mut index = 0;

    // Set bits for kept qubits
    for (i, &q) in keep_qubits.iter().enumerate() {
        if (keep_idx >> i) & 1 == 1 {
            index |= 1 << q;
        }
    }

    // Set bits for traced qubits
    for (i, &q) in trace_qubits.iter().enumerate() {
        if (trace_idx >> i) & 1 == 1 {
            index |= 1 << q;
        }
    }

    index
}

/// Compute the tensor product of multiple matrices efficiently
pub fn tensor_product_many(matrices: &[&dyn QuantumMatrix]) -> QuantRS2Result<Array2<Complex64>> {
    if matrices.is_empty() {
        return Err(QuantRS2Error::InvalidInput(
            "Cannot compute tensor product of empty list".to_string(),
        ));
    }

    if matrices.len() == 1 {
        return Ok(matrices[0].to_dense());
    }

    let mut result = matrices[0].to_dense();
    for matrix in matrices.iter().skip(1) {
        let dense_result = DenseMatrix::new(result)?;
        result = dense_result.tensor_product(*matrix)?;
    }

    Ok(result)
}

/// Check if two matrices are approximately equal
pub fn matrices_approx_equal(
    a: &ArrayView2<Complex64>,
    b: &ArrayView2<Complex64>,
    tolerance: f64,
) -> bool {
    if a.shape() != b.shape() {
        return false;
    }

    for (x, y) in a.iter().zip(b.iter()) {
        if (x - y).norm() > tolerance {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::Complex64;

    #[test]
    fn test_dense_matrix_creation() {
        let data = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .expect("Matrix data creation should succeed");

        let matrix = DenseMatrix::new(data).expect("DenseMatrix creation should succeed");
        assert_eq!(matrix.dim(), 2);
    }

    #[test]
    fn test_unitary_check() {
        // Hadamard gate
        let sqrt2 = 1.0 / 2.0_f64.sqrt();
        let data = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(sqrt2, 0.0),
                Complex64::new(sqrt2, 0.0),
                Complex64::new(sqrt2, 0.0),
                Complex64::new(-sqrt2, 0.0),
            ],
        )
        .expect("Hadamard matrix data creation should succeed");

        let matrix = DenseMatrix::new(data).expect("DenseMatrix creation should succeed");
        assert!(matrix
            .is_unitary(1e-10)
            .expect("Unitary check should succeed"));
    }

    #[test]
    fn test_tensor_product() {
        // Identity ⊗ Pauli-X
        let id = DenseMatrix::new(
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .expect("Identity matrix data creation should succeed"),
        )
        .expect("Identity DenseMatrix creation should succeed");

        let x = DenseMatrix::new(
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .expect("Pauli-X matrix data creation should succeed"),
        )
        .expect("Pauli-X DenseMatrix creation should succeed");

        let result = id
            .tensor_product(&x)
            .expect("Tensor product should succeed");
        assert_eq!(result.shape(), &[4, 4]);

        // Check specific values
        assert_eq!(result[[0, 1]], Complex64::new(1.0, 0.0));
        assert_eq!(result[[2, 3]], Complex64::new(1.0, 0.0));
    }
}
