//! Sparse matrix operations for efficient quantum circuit simulation.
//!
//! This module provides sparse matrix representations and operations
//! optimized for quantum gates, especially for circuits with limited connectivity.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::HashMap;

use crate::error::{Result, SimulatorError};

/// Compressed Sparse Row (CSR) matrix format
#[derive(Debug, Clone)]
pub struct CSRMatrix {
    /// Non-zero values
    pub values: Vec<Complex64>,
    /// Column indices for each value
    pub col_indices: Vec<usize>,
    /// Row pointer array
    pub row_ptr: Vec<usize>,
    /// Number of rows
    pub num_rows: usize,
    /// Number of columns
    pub num_cols: usize,
}

impl CSRMatrix {
    /// Create a new CSR matrix
    #[must_use]
    pub fn new(
        values: Vec<Complex64>,
        col_indices: Vec<usize>,
        row_ptr: Vec<usize>,
        num_rows: usize,
        num_cols: usize,
    ) -> Self {
        assert_eq!(values.len(), col_indices.len());
        assert_eq!(row_ptr.len(), num_rows + 1);

        Self {
            values,
            col_indices,
            row_ptr,
            num_rows,
            num_cols,
        }
    }

    /// Create from a dense matrix
    #[must_use]
    pub fn from_dense(matrix: &Array2<Complex64>) -> Self {
        let num_rows = matrix.nrows();
        let num_cols = matrix.ncols();
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0];

        for i in 0..num_rows {
            for j in 0..num_cols {
                let val = matrix[[i, j]];
                if val.norm() > 1e-15 {
                    values.push(val);
                    col_indices.push(j);
                }
            }
            row_ptr.push(values.len());
        }

        Self::new(values, col_indices, row_ptr, num_rows, num_cols)
    }

    /// Convert to dense matrix
    #[must_use]
    pub fn to_dense(&self) -> Array2<Complex64> {
        let mut dense = Array2::zeros((self.num_rows, self.num_cols));

        for i in 0..self.num_rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            for idx in start..end {
                dense[[i, self.col_indices[idx]]] = self.values[idx];
            }
        }

        dense
    }

    /// Get number of non-zero elements
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Matrix-vector multiplication
    pub fn matvec(&self, vec: &Array1<Complex64>) -> Result<Array1<Complex64>> {
        if vec.len() != self.num_cols {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Vector length {} doesn't match matrix columns {}",
                vec.len(),
                self.num_cols
            )));
        }

        let mut result = Array1::zeros(self.num_rows);

        for i in 0..self.num_rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            let mut sum = Complex64::new(0.0, 0.0);
            for idx in start..end {
                sum += self.values[idx] * vec[self.col_indices[idx]];
            }
            result[i] = sum;
        }

        Ok(result)
    }

    /// Sparse matrix multiplication
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.num_cols != other.num_rows {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Matrix dimensions incompatible: {}x{} * {}x{}",
                self.num_rows, self.num_cols, other.num_rows, other.num_cols
            )));
        }

        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0];

        // Convert other to CSC for efficient column access
        let other_csc = other.to_csc();

        for i in 0..self.num_rows {
            let mut row_values: HashMap<usize, Complex64> = HashMap::new();

            let a_start = self.row_ptr[i];
            let a_end = self.row_ptr[i + 1];

            for a_idx in a_start..a_end {
                let k = self.col_indices[a_idx];
                let a_val = self.values[a_idx];

                // Multiply row i of A with column k of B
                let b_start = other_csc.col_ptr[k];
                let b_end = other_csc.col_ptr[k + 1];

                for b_idx in b_start..b_end {
                    let j = other_csc.row_indices[b_idx];
                    let b_val = other_csc.values[b_idx];

                    *row_values.entry(j).or_insert(Complex64::new(0.0, 0.0)) += a_val * b_val;
                }
            }

            // Sort by column index and add to result
            let mut sorted_cols: Vec<_> = row_values.into_iter().collect();
            sorted_cols.sort_by_key(|(col, _)| *col);

            for (col, val) in sorted_cols {
                if val.norm() > 1e-15 {
                    values.push(val);
                    col_indices.push(col);
                }
            }

            row_ptr.push(values.len());
        }

        Ok(Self::new(
            values,
            col_indices,
            row_ptr,
            self.num_rows,
            other.num_cols,
        ))
    }

    /// Convert to Compressed Sparse Column (CSC) format
    fn to_csc(&self) -> CSCMatrix {
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_ptr = vec![0; self.num_cols + 1];

        // Count elements per column
        for &col in &self.col_indices {
            col_ptr[col + 1] += 1;
        }

        // Cumulative sum to get column pointers
        for i in 1..=self.num_cols {
            col_ptr[i] += col_ptr[i - 1];
        }

        // Temporary array to track current position in each column
        let mut current_pos = col_ptr[0..self.num_cols].to_vec();
        values.resize(self.nnz(), Complex64::new(0.0, 0.0));
        row_indices.resize(self.nnz(), 0);

        // Fill CSC arrays
        for i in 0..self.num_rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            for idx in start..end {
                let col = self.col_indices[idx];
                let pos = current_pos[col];

                values[pos] = self.values[idx];
                row_indices[pos] = i;
                current_pos[col] += 1;
            }
        }

        CSCMatrix {
            values,
            row_indices,
            col_ptr,
            num_rows: self.num_rows,
            num_cols: self.num_cols,
        }
    }
}

/// Compressed Sparse Column (CSC) matrix format
#[derive(Debug, Clone)]
struct CSCMatrix {
    values: Vec<Complex64>,
    row_indices: Vec<usize>,
    col_ptr: Vec<usize>,
    num_rows: usize,
    num_cols: usize,
}

/// Sparse matrix builder for incremental construction
#[derive(Debug)]
pub struct SparseMatrixBuilder {
    triplets: Vec<(usize, usize, Complex64)>,
    num_rows: usize,
    num_cols: usize,
}

impl SparseMatrixBuilder {
    /// Create a new builder
    #[must_use]
    pub const fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            triplets: Vec::new(),
            num_rows,
            num_cols,
        }
    }

    /// Add an element to the matrix
    pub fn add(&mut self, row: usize, col: usize, value: Complex64) {
        if row < self.num_rows && col < self.num_cols && value.norm() > 1e-15 {
            self.triplets.push((row, col, value));
        }
    }

    /// Set value at specific position (alias for add)
    pub fn set_value(&mut self, row: usize, col: usize, value: Complex64) {
        self.add(row, col, value);
    }

    /// Build the CSR matrix
    #[must_use]
    pub fn build(mut self) -> CSRMatrix {
        // Sort by row, then column
        self.triplets.sort_by_key(|(r, c, _)| (*r, *c));

        // Combine duplicates
        let mut combined_triplets = Vec::new();
        let mut last_pos: Option<(usize, usize)> = None;

        for (r, c, v) in self.triplets {
            if Some((r, c)) == last_pos {
                if let Some(last) = combined_triplets.last_mut() {
                    let (_, _, ref mut last_val) = last;
                    *last_val += v;
                }
            } else {
                combined_triplets.push((r, c, v));
                last_pos = Some((r, c));
            }
        }

        // Build CSR arrays
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0];
        let mut current_row = 0;

        for (r, c, v) in combined_triplets {
            while current_row < r {
                row_ptr.push(values.len());
                current_row += 1;
            }

            if v.norm() > 1e-15 {
                values.push(v);
                col_indices.push(c);
            }
        }

        while row_ptr.len() <= self.num_rows {
            row_ptr.push(values.len());
        }

        CSRMatrix::new(values, col_indices, row_ptr, self.num_rows, self.num_cols)
    }
}

/// Sparse quantum gate representations
pub struct SparseGates;

impl SparseGates {
    /// Create sparse Pauli X gate
    #[must_use]
    pub fn x() -> CSRMatrix {
        let mut builder = SparseMatrixBuilder::new(2, 2);
        builder.add(0, 1, Complex64::new(1.0, 0.0));
        builder.add(1, 0, Complex64::new(1.0, 0.0));
        builder.build()
    }

    /// Create sparse Pauli Y gate
    #[must_use]
    pub fn y() -> CSRMatrix {
        let mut builder = SparseMatrixBuilder::new(2, 2);
        builder.add(0, 1, Complex64::new(0.0, -1.0));
        builder.add(1, 0, Complex64::new(0.0, 1.0));
        builder.build()
    }

    /// Create sparse Pauli Z gate
    #[must_use]
    pub fn z() -> CSRMatrix {
        let mut builder = SparseMatrixBuilder::new(2, 2);
        builder.add(0, 0, Complex64::new(1.0, 0.0));
        builder.add(1, 1, Complex64::new(-1.0, 0.0));
        builder.build()
    }

    /// Create sparse CNOT gate
    #[must_use]
    pub fn cnot() -> CSRMatrix {
        let mut builder = SparseMatrixBuilder::new(4, 4);
        builder.add(0, 0, Complex64::new(1.0, 0.0));
        builder.add(1, 1, Complex64::new(1.0, 0.0));
        builder.add(2, 3, Complex64::new(1.0, 0.0));
        builder.add(3, 2, Complex64::new(1.0, 0.0));
        builder.build()
    }

    /// Create sparse CZ gate
    #[must_use]
    pub fn cz() -> CSRMatrix {
        let mut builder = SparseMatrixBuilder::new(4, 4);
        builder.add(0, 0, Complex64::new(1.0, 0.0));
        builder.add(1, 1, Complex64::new(1.0, 0.0));
        builder.add(2, 2, Complex64::new(1.0, 0.0));
        builder.add(3, 3, Complex64::new(-1.0, 0.0));
        builder.build()
    }

    /// Create sparse rotation gate
    pub fn rotation(axis: &str, angle: f64) -> Result<CSRMatrix> {
        let (c, s) = (angle.cos(), angle.sin());
        let half_angle = angle / 2.0;
        let (ch, sh) = (half_angle.cos(), half_angle.sin());

        let mut builder = SparseMatrixBuilder::new(2, 2);

        match axis {
            "x" | "X" => {
                builder.add(0, 0, Complex64::new(ch, 0.0));
                builder.add(0, 1, Complex64::new(0.0, -sh));
                builder.add(1, 0, Complex64::new(0.0, -sh));
                builder.add(1, 1, Complex64::new(ch, 0.0));
            }
            "y" | "Y" => {
                builder.add(0, 0, Complex64::new(ch, 0.0));
                builder.add(0, 1, Complex64::new(-sh, 0.0));
                builder.add(1, 0, Complex64::new(sh, 0.0));
                builder.add(1, 1, Complex64::new(ch, 0.0));
            }
            "z" | "Z" => {
                builder.add(0, 0, Complex64::new(ch, -sh));
                builder.add(1, 1, Complex64::new(ch, sh));
            }
            _ => {
                return Err(SimulatorError::InvalidConfiguration(format!(
                    "Unknown rotation axis: {axis}"
                )))
            }
        }

        Ok(builder.build())
    }

    /// Create sparse controlled rotation gate
    pub fn controlled_rotation(axis: &str, angle: f64) -> Result<CSRMatrix> {
        let single_qubit = Self::rotation(axis, angle)?;

        let mut builder = SparseMatrixBuilder::new(4, 4);

        // |00⟩ and |01⟩ states unchanged
        builder.add(0, 0, Complex64::new(1.0, 0.0));
        builder.add(1, 1, Complex64::new(1.0, 0.0));

        // Apply rotation to |10⟩ and |11⟩ states
        builder.add(2, 2, single_qubit.values[0]);
        if single_qubit.values.len() > 1 {
            builder.add(2, 3, single_qubit.values[1]);
        }
        if single_qubit.values.len() > 2 {
            builder.add(3, 2, single_qubit.values[2]);
        }
        if single_qubit.values.len() > 3 {
            builder.add(3, 3, single_qubit.values[3]);
        }

        Ok(builder.build())
    }
}

/// Apply sparse gate to state vector at specific qubits
pub fn apply_sparse_gate(
    state: &mut Array1<Complex64>,
    gate: &CSRMatrix,
    qubits: &[usize],
    num_qubits: usize,
) -> Result<()> {
    let gate_qubits = qubits.len();
    let gate_dim = 1 << gate_qubits;

    if gate.num_rows != gate_dim || gate.num_cols != gate_dim {
        return Err(SimulatorError::DimensionMismatch(format!(
            "Gate dimension {} doesn't match qubit count {}",
            gate.num_rows, gate_qubits
        )));
    }

    // Create bit masks for the target qubits
    let mut masks = vec![0usize; gate_qubits];
    for (i, &qubit) in qubits.iter().enumerate() {
        masks[i] = 1 << qubit;
    }

    // Apply gate to all basis states
    let state_dim = 1 << num_qubits;
    let mut new_state = Array1::zeros(state_dim);

    for i in 0..state_dim {
        // Extract indices for gate qubits
        let mut gate_idx = 0;
        for (j, &mask) in masks.iter().enumerate() {
            if i & mask != 0 {
                gate_idx |= 1 << j;
            }
        }

        // Apply sparse gate row
        let row_start = gate.row_ptr[gate_idx];
        let row_end = gate.row_ptr[gate_idx + 1];

        for idx in row_start..row_end {
            let gate_col = gate.col_indices[idx];
            let gate_val = gate.values[idx];

            // Reconstruct global index
            let mut j = i;
            for (k, &mask) in masks.iter().enumerate() {
                if gate_col & (1 << k) != 0 {
                    j |= mask;
                } else {
                    j &= !mask;
                }
            }

            new_state[i] += gate_val * state[j];
        }
    }

    state.assign(&new_state);
    Ok(())
}

/// Optimize gate sequence using sparsity
pub fn optimize_sparse_gates(gates: Vec<CSRMatrix>) -> Result<CSRMatrix> {
    if gates.is_empty() {
        return Err(SimulatorError::InvalidInput(
            "Empty gate sequence".to_string(),
        ));
    }

    let mut result = gates[0].clone();
    for gate in gates.into_iter().skip(1) {
        result = result.matmul(&gate)?;

        // Threshold small values
        result.values.retain(|&v| v.norm() > 1e-15);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_matrix_construction() {
        let mut builder = SparseMatrixBuilder::new(3, 3);
        builder.add(0, 0, Complex64::new(1.0, 0.0));
        builder.add(1, 1, Complex64::new(2.0, 0.0));
        builder.add(2, 2, Complex64::new(3.0, 0.0));
        builder.add(0, 2, Complex64::new(4.0, 0.0));

        let sparse = builder.build();
        assert_eq!(sparse.nnz(), 4);
        assert_eq!(sparse.num_rows, 3);
        assert_eq!(sparse.num_cols, 3);
    }

    #[test]
    fn test_sparse_gates() {
        let x = SparseGates::x();
        assert_eq!(x.nnz(), 2);

        let cnot = SparseGates::cnot();
        assert_eq!(cnot.nnz(), 4);

        let rz = SparseGates::rotation("z", 0.5).expect("Failed to create rotation gate");
        assert_eq!(rz.nnz(), 2);
    }

    #[test]
    fn test_sparse_matvec() {
        let x = SparseGates::x();
        let vec = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);

        let result = x
            .matvec(&vec)
            .expect("Failed to perform matrix-vector multiplication");
        assert!((result[0] - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((result[1] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_sparse_matmul() {
        let x = SparseGates::x();
        let z = SparseGates::z();

        let xz = x
            .matmul(&z)
            .expect("Failed to perform matrix multiplication");
        let y_expected = SparseGates::y();

        // X * Z = -iY
        assert_eq!(xz.nnz(), y_expected.nnz());
    }

    #[test]
    fn test_csr_to_dense() {
        let cnot = SparseGates::cnot();
        let dense = cnot.to_dense();

        assert_eq!(dense.shape(), &[4, 4]);
        assert!((dense[[0, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((dense[[3, 2]] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }
}
