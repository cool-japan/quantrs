//! Ndarray integration with SciRS2.
//!
//! This module provides conversion between symbolic matrices and
//! SciRS2's ndarray types.

use std::fmt::Write;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;

use crate::error::{SymEngineError, SymEngineResult};
use crate::expr::Expression;

/// Convert a symbolic matrix expression to a numeric Array2<Complex64>.
///
/// # Arguments
/// * `expr` - The matrix expression
/// * `values` - Variable values for evaluation
///
/// # Errors
/// Returns an error if the expression is not a matrix or evaluation fails.
pub fn to_array2(
    expr: &Expression,
    values: &std::collections::HashMap<String, f64>,
) -> SymEngineResult<Array2<Complex64>> {
    // TODO: Implement proper matrix parsing and evaluation
    Err(SymEngineError::not_impl(
        "Matrix to Array2 conversion not yet implemented",
    ))
}

/// Convert a numeric Array2<Complex64> to a symbolic matrix expression.
pub fn from_array2(arr: &Array2<Complex64>) -> Expression {
    let (rows, cols) = arr.dim();

    let mut matrix_str = String::from("Matrix([");

    for i in 0..rows {
        matrix_str.push('[');
        for j in 0..cols {
            let c = arr[[i, j]];
            if c.im.abs() < 1e-15 {
                let _ = write!(matrix_str, "{}", c.re);
            } else if c.re.abs() < 1e-15 {
                let _ = write!(matrix_str, "{}*I", c.im);
            } else {
                let _ = write!(matrix_str, "({}+{}*I)", c.re, c.im);
            }
            if j < cols - 1 {
                matrix_str.push_str(", ");
            }
        }
        matrix_str.push(']');
        if i < rows - 1 {
            matrix_str.push_str(", ");
        }
    }

    matrix_str.push_str("])");

    Expression::new(matrix_str)
}

/// Convert a symbolic vector expression to a numeric Array1<Complex64>.
///
/// # Errors
/// Returns an error if conversion fails.
pub fn to_array1(
    expr: &Expression,
    values: &std::collections::HashMap<String, f64>,
) -> SymEngineResult<Array1<Complex64>> {
    // TODO: Implement proper vector parsing and evaluation
    Err(SymEngineError::not_impl(
        "Vector to Array1 conversion not yet implemented",
    ))
}

/// Convert a numeric Array1<Complex64> to a symbolic column vector expression.
pub fn from_array1(arr: &Array1<Complex64>) -> Expression {
    let n = arr.len();

    let mut matrix_str = String::from("Matrix([");

    for (i, c) in arr.iter().enumerate() {
        matrix_str.push('[');
        if c.im.abs() < 1e-15 {
            let _ = write!(matrix_str, "{}", c.re);
        } else if c.re.abs() < 1e-15 {
            let _ = write!(matrix_str, "{}*I", c.im);
        } else {
            let _ = write!(matrix_str, "({}+{}*I)", c.re, c.im);
        }
        matrix_str.push(']');
        if i < n - 1 {
            matrix_str.push_str(", ");
        }
    }

    matrix_str.push_str("])");

    Expression::new(matrix_str)
}

/// Compute the gradient at given values as an Array1<f64>.
///
/// This is useful for integration with SciRS2 optimization routines.
pub fn gradient_array(
    expr: &Expression,
    params: &[Expression],
    values: &std::collections::HashMap<String, f64>,
) -> SymEngineResult<Array1<f64>> {
    let grad_vec = crate::optimization::gradient_at(expr, params, values)?;
    Ok(Array1::from_vec(grad_vec))
}

/// Compute the Hessian at given values as an Array2<f64>.
///
/// This is useful for integration with SciRS2 optimization routines.
pub fn hessian_array(
    expr: &Expression,
    params: &[Expression],
    values: &std::collections::HashMap<String, f64>,
) -> SymEngineResult<Array2<f64>> {
    let hess_vec = crate::optimization::hessian_at(expr, params, values)?;
    let n = params.len();
    let mut arr = Array2::zeros((n, n));

    for (i, row) in hess_vec.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            arr[[i, j]] = val;
        }
    }

    Ok(arr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_from_array2() {
        let arr: Array2<Complex64> = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
            [Complex64::new(0.0, -1.0), Complex64::new(1.0, 0.0)],
        ];

        let expr = from_array2(&arr);
        // Matrix expressions are stored as symbolic strings
        assert!(expr.to_string().contains("Matrix"));
    }

    #[test]
    fn test_from_array1() {
        let arr: Array1<Complex64> = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0),];

        let expr = from_array1(&arr);
        // Vector expressions are stored as symbolic matrix strings
        assert!(expr.to_string().contains("Matrix"));
    }

    #[test]
    fn test_gradient_array() {
        let x = Expression::symbol("x");
        let expr = x.clone() * x.clone(); // x^2
        let params = vec![x];

        let mut values = std::collections::HashMap::new();
        values.insert("x".to_string(), 3.0);

        let grad = gradient_array(&expr, &params, &values).expect("should compute");
        assert!((grad[0] - 6.0).abs() < 1e-6); // d/dx(x^2) = 2x = 6 at x=3
    }
}
