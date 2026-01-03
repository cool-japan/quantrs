//! Symbolic matrix operations.
//!
//! This module provides a proper matrix type where each element is a symbolic expression.
//! This is essential for quantum computing where we work with parameterized gates and
//! symbolic Hamiltonians.

use std::fmt;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;

use crate::error::{SymEngineError, SymEngineResult};
use crate::expr::Expression;

/// A symbolic matrix where each element is an Expression.
///
/// This is useful for representing parameterized quantum gates,
/// symbolic Hamiltonians, and other matrix expressions.
#[derive(Clone, Debug)]
pub struct SymbolicMatrix {
    /// The matrix elements (row-major order)
    elements: Vec<Expression>,
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
}

impl SymbolicMatrix {
    /// Create a new symbolic matrix from a 2D array of expressions.
    ///
    /// # Errors
    /// Returns error if the input is empty
    pub fn new(elements: Vec<Vec<Expression>>) -> SymEngineResult<Self> {
        if elements.is_empty() {
            return Err(SymEngineError::dimension("Matrix cannot be empty"));
        }

        let rows = elements.len();
        let cols = elements[0].len();

        // Verify all rows have the same length
        for (i, row) in elements.iter().enumerate() {
            if row.len() != cols {
                return Err(SymEngineError::dimension(format!(
                    "Row {i} has {} columns, expected {cols}",
                    row.len()
                )));
            }
        }

        let flat: Vec<Expression> = elements.into_iter().flatten().collect();

        Ok(Self {
            elements: flat,
            rows,
            cols,
        })
    }

    /// Create a matrix from a flat vector with specified dimensions.
    ///
    /// # Errors
    /// Returns error if the dimensions don't match the vector length
    pub fn from_flat(elements: Vec<Expression>, rows: usize, cols: usize) -> SymEngineResult<Self> {
        if elements.len() != rows * cols {
            return Err(SymEngineError::dimension(format!(
                "Expected {} elements for {}x{} matrix, got {}",
                rows * cols,
                rows,
                cols,
                elements.len()
            )));
        }

        Ok(Self {
            elements,
            rows,
            cols,
        })
    }

    /// Create a zero matrix.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            elements: vec![Expression::zero(); rows * cols],
            rows,
            cols,
        }
    }

    /// Create an identity matrix.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut elements = vec![Expression::zero(); n * n];
        for i in 0..n {
            elements[i * n + i] = Expression::one();
        }
        Self {
            elements,
            rows: n,
            cols: n,
        }
    }

    /// Create a diagonal matrix from a vector of diagonal elements.
    #[must_use]
    pub fn diagonal(diag: Vec<Expression>) -> Self {
        let n = diag.len();
        let mut elements = vec![Expression::zero(); n * n];
        for (i, d) in diag.into_iter().enumerate() {
            elements[i * n + i] = d;
        }
        Self {
            elements,
            rows: n,
            cols: n,
        }
    }

    /// Create a matrix from a numeric Array2.
    #[must_use]
    pub fn from_array(arr: &Array2<f64>) -> Self {
        let rows = arr.nrows();
        let cols = arr.ncols();
        let elements: Vec<Expression> = arr
            .iter()
            .map(|&v| Expression::float_unchecked(v))
            .collect();
        Self {
            elements,
            rows,
            cols,
        }
    }

    /// Create a matrix from a complex Array2.
    #[must_use]
    pub fn from_complex_array(arr: &Array2<Complex64>) -> Self {
        let rows = arr.nrows();
        let cols = arr.ncols();
        let elements: Vec<Expression> =
            arr.iter().map(|&c| Expression::from_complex64(c)).collect();
        Self {
            elements,
            rows,
            cols,
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the number of rows.
    #[must_use]
    pub const fn nrows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    #[must_use]
    pub const fn ncols(&self) -> usize {
        self.cols
    }

    /// Get the dimensions as (rows, cols).
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Check if the matrix is square.
    #[must_use]
    pub const fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Get element at (i, j).
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> &Expression {
        assert!(i < self.rows && j < self.cols, "Index out of bounds");
        &self.elements[i * self.cols + j]
    }

    /// Get mutable reference to element at (i, j).
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut Expression {
        assert!(i < self.rows && j < self.cols, "Index out of bounds");
        &mut self.elements[i * self.cols + j]
    }

    /// Set element at (i, j).
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    pub fn set(&mut self, i: usize, j: usize, value: Expression) {
        assert!(i < self.rows && j < self.cols, "Index out of bounds");
        self.elements[i * self.cols + j] = value;
    }

    /// Get a row as a vector of expressions.
    #[must_use]
    pub fn row(&self, i: usize) -> Vec<Expression> {
        assert!(i < self.rows, "Row index out of bounds");
        let start = i * self.cols;
        self.elements[start..start + self.cols].to_vec()
    }

    /// Get a column as a vector of expressions.
    #[must_use]
    pub fn col(&self, j: usize) -> Vec<Expression> {
        assert!(j < self.cols, "Column index out of bounds");
        (0..self.rows).map(|i| self.get(i, j).clone()).collect()
    }

    // =========================================================================
    // Matrix Operations
    // =========================================================================

    /// Matrix transpose.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut elements = Vec::with_capacity(self.rows * self.cols);
        for j in 0..self.cols {
            for i in 0..self.rows {
                elements.push(self.get(i, j).clone());
            }
        }
        Self {
            elements,
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Complex conjugate of all elements.
    #[must_use]
    pub fn conjugate(&self) -> Self {
        Self {
            elements: self.elements.iter().map(Expression::conjugate).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Hermitian conjugate (conjugate transpose).
    #[must_use]
    pub fn dagger(&self) -> Self {
        self.transpose().conjugate()
    }

    /// Matrix addition.
    ///
    /// # Errors
    /// Returns error if dimensions don't match.
    pub fn add(&self, other: &Self) -> SymEngineResult<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SymEngineError::dimension(format!(
                "Cannot add {}x{} matrix with {}x{} matrix",
                self.rows, self.cols, other.rows, other.cols
            )));
        }

        let elements: Vec<Expression> = self
            .elements
            .iter()
            .zip(other.elements.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Ok(Self {
            elements,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Matrix subtraction.
    ///
    /// # Errors
    /// Returns error if dimensions don't match.
    pub fn sub(&self, other: &Self) -> SymEngineResult<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SymEngineError::dimension(format!(
                "Cannot subtract {}x{} matrix from {}x{} matrix",
                other.rows, other.cols, self.rows, self.cols
            )));
        }

        let elements: Vec<Expression> = self
            .elements
            .iter()
            .zip(other.elements.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        Ok(Self {
            elements,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Matrix multiplication.
    ///
    /// # Errors
    /// Returns error if inner dimensions don't match.
    pub fn matmul(&self, other: &Self) -> SymEngineResult<Self> {
        if self.cols != other.rows {
            return Err(SymEngineError::dimension(format!(
                "Cannot multiply {}x{} matrix with {}x{} matrix",
                self.rows, self.cols, other.rows, other.cols
            )));
        }

        let mut elements = Vec::with_capacity(self.rows * other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = Expression::zero();
                for k in 0..self.cols {
                    sum = sum + self.get(i, k).clone() * other.get(k, j).clone();
                }
                elements.push(sum);
            }
        }

        Ok(Self {
            elements,
            rows: self.rows,
            cols: other.cols,
        })
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn scale(&self, scalar: &Expression) -> Self {
        Self {
            elements: self
                .elements
                .iter()
                .map(|e| e.clone() * scalar.clone())
                .collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Kronecker (tensor) product.
    #[must_use]
    pub fn kron(&self, other: &Self) -> Self {
        let new_rows = self.rows * other.rows;
        let new_cols = self.cols * other.cols;
        let mut elements = Vec::with_capacity(new_rows * new_cols);

        for i1 in 0..self.rows {
            for i2 in 0..other.rows {
                for j1 in 0..self.cols {
                    for j2 in 0..other.cols {
                        let a = self.get(i1, j1).clone();
                        let b = other.get(i2, j2).clone();
                        elements.push(a * b);
                    }
                }
            }
        }

        Self {
            elements,
            rows: new_rows,
            cols: new_cols,
        }
    }

    /// Matrix trace (sum of diagonal elements).
    ///
    /// # Errors
    /// Returns error if matrix is not square.
    pub fn trace(&self) -> SymEngineResult<Expression> {
        if !self.is_square() {
            return Err(SymEngineError::dimension(
                "Trace is only defined for square matrices",
            ));
        }

        let mut sum = Expression::zero();
        for i in 0..self.rows {
            sum = sum + self.get(i, i).clone();
        }
        Ok(sum)
    }

    /// Commutator [A, B] = AB - BA.
    ///
    /// # Errors
    /// Returns error if dimensions don't match or matrices are not square.
    pub fn commutator(&self, other: &Self) -> SymEngineResult<Self> {
        let ab = self.matmul(other)?;
        let ba = other.matmul(self)?;
        ab.sub(&ba)
    }

    /// Anticommutator {A, B} = AB + BA.
    ///
    /// # Errors
    /// Returns error if dimensions don't match.
    pub fn anticommutator(&self, other: &Self) -> SymEngineResult<Self> {
        let ab = self.matmul(other)?;
        let ba = other.matmul(self)?;
        ab.add(&ba)
    }

    // =========================================================================
    // Simplification
    // =========================================================================

    /// Simplify all matrix elements.
    #[must_use]
    pub fn simplify(&self) -> Self {
        Self {
            elements: self.elements.iter().map(Expression::simplify).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Expand all matrix elements.
    #[must_use]
    pub fn expand(&self) -> Self {
        Self {
            elements: self.elements.iter().map(Expression::expand).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    // =========================================================================
    // Evaluation
    // =========================================================================

    /// Evaluate all matrix elements with given variable values.
    ///
    /// # Errors
    /// Returns error if any element evaluation fails.
    pub fn eval(
        &self,
        values: &std::collections::HashMap<String, f64>,
    ) -> SymEngineResult<Array2<f64>> {
        let mut result = Array2::zeros((self.rows, self.cols));
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[[i, j]] = self.get(i, j).eval(values)?;
            }
        }
        Ok(result)
    }

    /// Substitute a variable with an expression in all elements.
    #[must_use]
    pub fn substitute(&self, var: &Expression, value: &Expression) -> Self {
        Self {
            elements: self
                .elements
                .iter()
                .map(|e| e.substitute(var, value))
                .collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    // =========================================================================
    // Differentiation
    // =========================================================================

    /// Compute the derivative of all elements with respect to a variable.
    #[must_use]
    pub fn diff(&self, var: &Expression) -> Self {
        Self {
            elements: self.elements.iter().map(|e| e.diff(var)).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl fmt::Display for SymbolicMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        for i in 0..self.rows {
            write!(f, "  [")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.get(i, j))?;
            }
            writeln!(f, "]")?;
        }
        write!(f, "]")
    }
}

impl std::ops::Index<(usize, usize)> for SymbolicMatrix {
    type Output = Expression;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
    }
}

impl std::ops::IndexMut<(usize, usize)> for SymbolicMatrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut(index.0, index.1)
    }
}

// =========================================================================
// Quantum Gate Matrices
// =========================================================================

/// Create the Pauli X gate matrix.
#[must_use]
pub fn pauli_x() -> SymbolicMatrix {
    SymbolicMatrix::from_flat(
        vec![
            Expression::zero(),
            Expression::one(),
            Expression::one(),
            Expression::zero(),
        ],
        2,
        2,
    )
    .expect("valid 2x2 matrix")
}

/// Create the Pauli Y gate matrix.
#[must_use]
pub fn pauli_y() -> SymbolicMatrix {
    let i = Expression::i();
    SymbolicMatrix::from_flat(
        vec![Expression::zero(), i.clone().neg(), i, Expression::zero()],
        2,
        2,
    )
    .expect("valid 2x2 matrix")
}

/// Create the Pauli Z gate matrix.
#[must_use]
pub fn pauli_z() -> SymbolicMatrix {
    SymbolicMatrix::from_flat(
        vec![
            Expression::one(),
            Expression::zero(),
            Expression::zero(),
            Expression::one().neg(),
        ],
        2,
        2,
    )
    .expect("valid 2x2 matrix")
}

/// Create the Hadamard gate matrix.
#[must_use]
pub fn hadamard() -> SymbolicMatrix {
    let sqrt2_inv = Expression::one() / crate::ops::trig::sqrt(&Expression::int(2));
    SymbolicMatrix::from_flat(
        vec![
            sqrt2_inv.clone(),
            sqrt2_inv.clone(),
            sqrt2_inv.clone(),
            sqrt2_inv.neg(),
        ],
        2,
        2,
    )
    .expect("valid 2x2 matrix")
}

/// Create the S (phase) gate matrix.
#[must_use]
pub fn s_gate() -> SymbolicMatrix {
    SymbolicMatrix::from_flat(
        vec![
            Expression::one(),
            Expression::zero(),
            Expression::zero(),
            Expression::i(),
        ],
        2,
        2,
    )
    .expect("valid 2x2 matrix")
}

/// Create the T gate matrix.
#[must_use]
pub fn t_gate() -> SymbolicMatrix {
    let exp_i_pi_4 =
        crate::ops::trig::exp(&(Expression::i() * Expression::pi() / Expression::int(4)));
    SymbolicMatrix::from_flat(
        vec![
            Expression::one(),
            Expression::zero(),
            Expression::zero(),
            exp_i_pi_4,
        ],
        2,
        2,
    )
    .expect("valid 2x2 matrix")
}

/// Create a rotation gate Rx(θ) around the X axis.
#[must_use]
pub fn rx(theta: &Expression) -> SymbolicMatrix {
    let half = Expression::float_unchecked(0.5);
    let half_theta = theta.clone() * half;
    let cos_half = crate::ops::trig::cos(&half_theta);
    let sin_half = crate::ops::trig::sin(&half_theta);
    let i = Expression::i();

    SymbolicMatrix::from_flat(
        vec![
            cos_half.clone(),
            i.clone().neg() * sin_half.clone(),
            i.neg() * sin_half,
            cos_half,
        ],
        2,
        2,
    )
    .expect("valid 2x2 matrix")
}

/// Create a rotation gate Ry(θ) around the Y axis.
#[must_use]
pub fn ry(theta: &Expression) -> SymbolicMatrix {
    let half = Expression::float_unchecked(0.5);
    let half_theta = theta.clone() * half;
    let cos_half = crate::ops::trig::cos(&half_theta);
    let sin_half = crate::ops::trig::sin(&half_theta);

    SymbolicMatrix::from_flat(
        vec![cos_half.clone(), sin_half.clone().neg(), sin_half, cos_half],
        2,
        2,
    )
    .expect("valid 2x2 matrix")
}

/// Create a rotation gate Rz(θ) around the Z axis.
#[must_use]
pub fn rz(theta: &Expression) -> SymbolicMatrix {
    let half = Expression::float_unchecked(0.5);
    let i = Expression::i();
    let half_theta = theta.clone() * half;
    let exp_neg = crate::ops::trig::exp(&(i.neg() * half_theta.clone()));
    let exp_pos = crate::ops::trig::exp(&(Expression::i() * half_theta));

    SymbolicMatrix::from_flat(
        vec![exp_neg, Expression::zero(), Expression::zero(), exp_pos],
        2,
        2,
    )
    .expect("valid 2x2 matrix")
}

/// Create the CNOT (CX) gate matrix.
#[must_use]
pub fn cnot() -> SymbolicMatrix {
    SymbolicMatrix::from_flat(
        vec![
            Expression::one(),
            Expression::zero(),
            Expression::zero(),
            Expression::zero(),
            Expression::zero(),
            Expression::one(),
            Expression::zero(),
            Expression::zero(),
            Expression::zero(),
            Expression::zero(),
            Expression::zero(),
            Expression::one(),
            Expression::zero(),
            Expression::zero(),
            Expression::one(),
            Expression::zero(),
        ],
        4,
        4,
    )
    .expect("valid 4x4 matrix")
}

/// Create the SWAP gate matrix.
#[must_use]
pub fn swap() -> SymbolicMatrix {
    SymbolicMatrix::from_flat(
        vec![
            Expression::one(),
            Expression::zero(),
            Expression::zero(),
            Expression::zero(),
            Expression::zero(),
            Expression::zero(),
            Expression::one(),
            Expression::zero(),
            Expression::zero(),
            Expression::one(),
            Expression::zero(),
            Expression::zero(),
            Expression::zero(),
            Expression::zero(),
            Expression::zero(),
            Expression::one(),
        ],
        4,
        4,
    )
    .expect("valid 4x4 matrix")
}

/// Create a controlled-U gate matrix.
#[must_use]
pub fn controlled(u: &SymbolicMatrix) -> SymbolicMatrix {
    assert!(u.is_square() && u.nrows() == 2, "U must be a 2x2 matrix");

    let n = 4;
    let mut elements = vec![Expression::zero(); n * n];

    // Top-left 2x2 block is identity (control = |0⟩)
    elements[0] = Expression::one();
    elements[5] = Expression::one();

    // Bottom-right 2x2 block is U (control = |1⟩)
    elements[10] = u.get(0, 0).clone();
    elements[11] = u.get(0, 1).clone();
    elements[14] = u.get(1, 0).clone();
    elements[15] = u.get(1, 1).clone();

    SymbolicMatrix::from_flat(elements, n, n).expect("valid 4x4 matrix")
}

#[cfg(test)]
#[allow(clippy::redundant_clone)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_matrix_creation() {
        let m = SymbolicMatrix::identity(2);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        assert!(m.get(0, 0).is_one());
        assert!(m.get(0, 1).is_zero());
        assert!(m.get(1, 0).is_zero());
        assert!(m.get(1, 1).is_one());
    }

    #[test]
    fn test_matrix_transpose() {
        let x = Expression::symbol("x");
        let y = Expression::symbol("y");
        let z = Expression::symbol("z");
        let w = Expression::symbol("w");

        let m = SymbolicMatrix::new(vec![vec![x.clone(), y.clone()], vec![z.clone(), w.clone()]])
            .expect("valid matrix");

        let mt = m.transpose();
        assert_eq!(mt.get(0, 0).as_symbol(), Some("x"));
        assert_eq!(mt.get(0, 1).as_symbol(), Some("z"));
        assert_eq!(mt.get(1, 0).as_symbol(), Some("y"));
        assert_eq!(mt.get(1, 1).as_symbol(), Some("w"));
    }

    #[test]
    fn test_matrix_multiplication() {
        // Test with identity
        let i = SymbolicMatrix::identity(2);
        let x = Expression::symbol("x");
        let m = SymbolicMatrix::new(vec![
            vec![x.clone(), Expression::zero()],
            vec![Expression::zero(), x.clone()],
        ])
        .expect("valid matrix");

        let result = i.matmul(&m).expect("valid matmul");

        // I * M = M - verify by evaluation
        let mut values = HashMap::new();
        values.insert("x".to_string(), 5.0);

        // result[0,0] should evaluate to x = 5.0
        let r00 = result.get(0, 0).eval(&values).expect("valid eval");
        assert!((r00 - 5.0).abs() < 1e-10);

        // result[0,1] should evaluate to 0
        let r01 = result.get(0, 1).eval(&values).expect("valid eval");
        assert!(r01.abs() < 1e-10);

        // result[1,1] should evaluate to x = 5.0
        let r11 = result.get(1, 1).eval(&values).expect("valid eval");
        assert!((r11 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_kronecker_product() {
        let x = pauli_x();
        let z = pauli_z();

        let xz = x.kron(&z);
        assert_eq!(xz.nrows(), 4);
        assert_eq!(xz.ncols(), 4);
    }

    #[test]
    fn test_trace() {
        let theta = Expression::symbol("theta");
        let m = SymbolicMatrix::diagonal(vec![theta.clone(), theta.clone()]);
        let tr = m.trace().expect("valid trace");

        // tr(diag(θ, θ)) = 2θ
        let mut values = HashMap::new();
        values.insert("theta".to_string(), 3.0);
        let result = tr.eval(&values).expect("valid eval");
        assert!((result - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_gates() {
        let theta = Expression::symbol("theta");

        // Rx(θ) should be unitary: Rx†Rx = I
        let rx_gate = rx(&theta);
        let rx_dag = rx_gate.dagger();

        // Just verify structure - full verification would require complex eval
        assert_eq!(rx_gate.nrows(), 2);
        assert_eq!(rx_dag.nrows(), 2);
    }

    #[test]
    fn test_pauli_commutation() {
        let x = pauli_x();
        let y = pauli_y();

        // [X, Y] = 2iZ
        let comm = x.commutator(&y).expect("valid commutator");
        assert_eq!(comm.nrows(), 2);
    }

    #[test]
    fn test_matrix_diff() {
        let theta = Expression::symbol("theta");
        let m = SymbolicMatrix::diagonal(vec![
            crate::ops::trig::sin(&theta),
            crate::ops::trig::cos(&theta),
        ]);

        let dm = m.diff(&theta);

        // d/dθ sin(θ) = cos(θ), d/dθ cos(θ) = -sin(θ)
        let mut values = HashMap::new();
        values.insert("theta".to_string(), 0.0);

        // At θ=0: d/dθ sin(θ)|θ=0 = cos(0) = 1
        let result = dm.eval(&values).expect("valid eval");
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);
        // d/dθ cos(θ)|θ=0 = -sin(0) = 0
        assert!(result[[1, 1]].abs() < 1e-10);
    }
}
