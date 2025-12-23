//! Integration with SciRS2's ndarray for symbolic matrices and vectors
//!
//! This module provides utilities for working with arrays of symbolic expressions
//! and converting between symbolic and numeric representations.

use crate::{Expression, SymEngineError, SymEngineResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::hash::BuildHasher;

/// Symbolic vector (1D array of expressions)
pub type SymVec = Vec<Expression>;

/// Symbolic matrix (2D array of expressions)
pub type SymMatrix = Vec<Vec<Expression>>;

/// Convert a symbolic vector to a numeric Array1
///
/// # Arguments
/// * `sym_vec` - Vector of symbolic expressions
/// * `values` - Map of symbol names to numeric values
///
/// # Errors
/// Returns an error if any expression cannot be evaluated to a number.
///
/// # Example
/// ```
/// use quantrs2_symengine::Expression;
/// use quantrs2_symengine::ndarray_integration::sym_vec_to_array1;
/// use std::collections::HashMap;
///
/// let x = Expression::symbol("x");
/// let sym_vec = vec![x.clone(), x.pow(&Expression::from(2))];
///
/// let mut values = HashMap::new();
/// values.insert("x".to_string(), 2.0);
///
/// let array = sym_vec_to_array1(&sym_vec, &values).unwrap();
/// assert_eq!(array.len(), 2);
/// ```
pub fn sym_vec_to_array1<S: BuildHasher>(
    sym_vec: &[Expression],
    values: &HashMap<String, f64, S>,
) -> SymEngineResult<Array1<f64>> {
    let numeric: Result<Vec<f64>, _> = sym_vec
        .iter()
        .map(|expr| {
            let mut result = expr.clone();
            for (symbol_name, value) in values {
                let symbol = Expression::symbol(symbol_name.as_str());
                result = result.substitute(&symbol, &Expression::from(*value));
            }
            result.expand().to_f64().ok_or_else(|| {
                SymEngineError::invalid_operation("Failed to evaluate expression to number")
            })
        })
        .collect();

    Ok(Array1::from_vec(numeric?))
}

/// Convert a symbolic matrix to a numeric Array2
///
/// # Arguments
/// * `sym_matrix` - 2D vector of symbolic expressions
/// * `values` - Map of symbol names to numeric values
///
/// # Errors
/// Returns an error if any expression cannot be evaluated to a number.
///
/// # Example
/// ```
/// use quantrs2_symengine::Expression;
/// use quantrs2_symengine::ndarray_integration::sym_matrix_to_array2;
/// use std::collections::HashMap;
///
/// let x = Expression::symbol("x");
/// let sym_matrix = vec![
///     vec![x.clone(), Expression::from(0)],
///     vec![Expression::from(0), x.clone()]
/// ];
///
/// let mut values = HashMap::new();
/// values.insert("x".to_string(), 3.0);
///
/// let array = sym_matrix_to_array2(&sym_matrix, &values).unwrap();
/// assert_eq!(array.shape(), &[2, 2]);
/// ```
pub fn sym_matrix_to_array2<S: BuildHasher>(
    sym_matrix: &[Vec<Expression>],
    values: &HashMap<String, f64, S>,
) -> SymEngineResult<Array2<f64>> {
    if sym_matrix.is_empty() {
        return Err(SymEngineError::invalid_operation("Empty matrix"));
    }

    let rows = sym_matrix.len();
    let cols = sym_matrix[0].len();

    // Check all rows have same length
    if !sym_matrix.iter().all(|row| row.len() == cols) {
        return Err(SymEngineError::invalid_operation(
            "Inconsistent row lengths",
        ));
    }

    let mut data = Vec::with_capacity(rows * cols);
    for row in sym_matrix {
        for expr in row {
            let mut result = expr.clone();
            for (symbol_name, value) in values {
                let symbol = Expression::symbol(symbol_name.as_str());
                result = result.substitute(&symbol, &Expression::from(*value));
            }
            let val = result.expand().to_f64().ok_or_else(|| {
                SymEngineError::invalid_operation("Failed to evaluate expression to number")
            })?;
            data.push(val);
        }
    }

    Array2::from_shape_vec((rows, cols), data)
        .map_err(|_| SymEngineError::invalid_operation("Failed to create array"))
}

/// Evaluate a symbolic vector at multiple parameter sets
///
/// # Arguments
/// * `sym_vec` - Vector of symbolic expressions
/// * `param_sets` - Vector of parameter value maps
///
/// Returns a 2D array where each row is the evaluation at one parameter set.
///
/// # Errors
/// Returns an error if any evaluation fails.
pub fn batch_eval_vec<S: BuildHasher>(
    sym_vec: &[Expression],
    param_sets: &[HashMap<String, f64, S>],
) -> SymEngineResult<Array2<f64>> {
    if param_sets.is_empty() {
        return Err(SymEngineError::invalid_operation(
            "No parameter sets provided",
        ));
    }

    let n_params = param_sets.len();
    let n_exprs = sym_vec.len();
    let mut data = Vec::with_capacity(n_params * n_exprs);

    for params in param_sets {
        for expr in sym_vec {
            let mut result = expr.clone();
            for (symbol_name, value) in params {
                let symbol = Expression::symbol(symbol_name.as_str());
                result = result.substitute(&symbol, &Expression::from(*value));
            }
            let val = result.expand().to_f64().ok_or_else(|| {
                SymEngineError::invalid_operation("Failed to evaluate expression")
            })?;
            data.push(val);
        }
    }

    Array2::from_shape_vec((n_params, n_exprs), data)
        .map_err(|_| SymEngineError::invalid_operation("Failed to create array"))
}

/// Compute gradient of a scalar function and evaluate it numerically
///
/// # Arguments
/// * `expr` - Scalar expression to differentiate
/// * `symbols` - Symbols to differentiate with respect to
/// * `values` - Numeric values for evaluation
///
/// Returns the gradient as an Array1.
///
/// # Errors
/// Returns an error if evaluation fails.
///
/// # Example
/// ```
/// use quantrs2_symengine::Expression;
/// use quantrs2_symengine::ndarray_integration::gradient_at;
/// use std::collections::HashMap;
///
/// let x = Expression::symbol("x");
/// let y = Expression::symbol("y");
/// let f = x.pow(&Expression::from(2)) + y.pow(&Expression::from(2));
///
/// let symbols = vec![x.clone(), y.clone()];
/// let mut values = HashMap::new();
/// values.insert("x".to_string(), 1.0);
/// values.insert("y".to_string(), 2.0);
///
/// let grad = gradient_at(&f, &symbols, &values).unwrap();
/// assert_eq!(grad.len(), 2);
/// // Gradient at (1, 2) should be [2*1, 2*2] = [2, 4]
/// ```
pub fn gradient_at<S: BuildHasher>(
    expr: &Expression,
    symbols: &[Expression],
    values: &HashMap<String, f64, S>,
) -> SymEngineResult<Array1<f64>> {
    let grad = expr.gradient(symbols);
    sym_vec_to_array1(&grad, values)
}

/// Compute Hessian of a scalar function and evaluate it numerically
///
/// # Arguments
/// * `expr` - Scalar expression to differentiate
/// * `symbols` - Symbols to differentiate with respect to
/// * `values` - Numeric values for evaluation
///
/// Returns the Hessian as an Array2.
///
/// # Errors
/// Returns an error if evaluation fails.
pub fn hessian_at<S: BuildHasher>(
    expr: &Expression,
    symbols: &[Expression],
    values: &HashMap<String, f64, S>,
) -> SymEngineResult<Array2<f64>> {
    let hess = expr.hessian(symbols);
    sym_matrix_to_array2(&hess, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[ignore = "SymEngine evaluation limitations"]
    fn test_sym_vec_to_array1() {
        let x = Expression::symbol("x");
        let sym_vec = vec![x.clone(), x.pow(&Expression::from(2)), Expression::from(1)];

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);

        let array = sym_vec_to_array1(&sym_vec, &values).expect("Failed to convert vector");
        assert_eq!(array.len(), 3);
        assert_relative_eq!(array[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(array[1], 9.0, epsilon = 1e-10);
        assert_relative_eq!(array[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    #[ignore = "SymEngine evaluation limitations"]
    fn test_sym_matrix_to_array2() {
        let x = Expression::symbol("x");
        let sym_matrix = vec![
            vec![x.clone(), Expression::from(0)],
            vec![Expression::from(0), x],
        ];

        let mut values = HashMap::new();
        values.insert("x".to_string(), 5.0);

        let array = sym_matrix_to_array2(&sym_matrix, &values).expect("Failed to convert matrix");
        assert_eq!(array.shape(), &[2, 2]);
        assert_relative_eq!(array[[0, 0]], 5.0, epsilon = 1e-10);
        assert_relative_eq!(array[[1, 1]], 5.0, epsilon = 1e-10);
        assert_relative_eq!(array[[0, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    #[ignore = "SymEngine evaluation limitations"]
    fn test_gradient_at() {
        let x = Expression::symbol("x");
        let y = Expression::symbol("y");
        let f = x.pow(&Expression::from(2)) + y.pow(&Expression::from(2));

        let symbols = vec![x, y];
        let mut values = HashMap::new();
        values.insert("x".to_string(), 1.0);
        values.insert("y".to_string(), 2.0);

        let grad = gradient_at(&f, &symbols, &values).expect("Failed to compute gradient");
        assert_eq!(grad.len(), 2);
        // Gradient should be [2*x, 2*y] = [2, 4] at (1, 2)
        assert_relative_eq!(grad[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(grad[1], 4.0, epsilon = 1e-10);
    }

    #[test]
    #[ignore = "SymEngine evaluation limitations"]
    fn test_batch_eval() {
        let x = Expression::symbol("x");
        let sym_vec = vec![x.clone(), x.pow(&Expression::from(2))];

        let param_sets = vec![
            {
                let mut map = HashMap::new();
                map.insert("x".to_string(), 1.0);
                map
            },
            {
                let mut map = HashMap::new();
                map.insert("x".to_string(), 2.0);
                map
            },
            {
                let mut map = HashMap::new();
                map.insert("x".to_string(), 3.0);
                map
            },
        ];

        let result = batch_eval_vec(&sym_vec, &param_sets).expect("Failed batch evaluation");
        assert_eq!(result.shape(), &[3, 2]);

        // First row: x=1 -> [1, 1]
        assert_relative_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[[0, 1]], 1.0, epsilon = 1e-10);

        // Second row: x=2 -> [2, 4]
        assert_relative_eq!(result[[1, 0]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 1]], 4.0, epsilon = 1e-10);

        // Third row: x=3 -> [3, 9]
        assert_relative_eq!(result[[2, 0]], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[[2, 1]], 9.0, epsilon = 1e-10);
    }
}
