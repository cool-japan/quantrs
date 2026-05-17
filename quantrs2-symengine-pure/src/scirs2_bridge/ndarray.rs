//! Ndarray integration with SciRS2.
//!
//! This module provides conversion between symbolic matrices and
//! SciRS2's ndarray types.

use std::fmt::Write;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;

use crate::error::{SymEngineError, SymEngineResult};
use crate::expr::Expression;

/// Parse a single matrix cell string (as produced by `from_array2`) into a `Complex64`.
///
/// Recognises the three forms emitted by `from_array2`:
/// - `"{re}"` — pure real
/// - `"{im}*I"` — pure imaginary
/// - `"({re}+{im}*I)"` — general (note: negative imaginary looks like `(1+-2*I)`)
fn parse_cell(s: &str) -> Result<Complex64, SymEngineError> {
    let s = s.trim();

    // Strip optional outer parentheses — the general format is `({re}+{im}*I)`.
    let s = if s.starts_with('(') && s.ends_with(')') {
        &s[1..s.len() - 1]
    } else {
        s
    };

    // Determine form by checking whether the string ends with "*I".
    if let Some(without_i) = s.strip_suffix("*I") {
        // Could be:
        //   pure imaginary  "{im}*I"        → without_i has no '+' (except in exponent)
        //   general complex "{re}+{im}*I"   → without_i contains a '+' split point
        if let Some(plus_pos) = find_split_plus(without_i) {
            // General complex: re = without_i[..plus_pos], im = without_i[plus_pos+1..]
            let re_str = &without_i[..plus_pos];
            let im_str = &without_i[plus_pos + 1..];
            let re = re_str
                .trim()
                .parse::<f64>()
                .map_err(|_| SymEngineError::parse(format!("cannot parse real part: {re_str}")))?;
            let im = im_str
                .trim()
                .parse::<f64>()
                .map_err(|_| SymEngineError::parse(format!("cannot parse imaginary coefficient: {im_str}")))?;
            return Ok(Complex64::new(re, im));
        }
        // Pure imaginary: no '+' separator found
        let im = without_i
            .trim()
            .parse::<f64>()
            .map_err(|_| SymEngineError::parse(format!("cannot parse imaginary coefficient: {without_i}")))?;
        return Ok(Complex64::new(0.0, im));
    }

    // Pure real fallback
    let re = s
        .parse::<f64>()
        .map_err(|_| SymEngineError::parse(format!("cannot parse cell value: {s}")))?;
    Ok(Complex64::new(re, 0.0))
}

/// Find the position of the '+' that separates a real part from an imaginary part.
///
/// We scan left-to-right and stop at a '+' that is not preceded by 'e' or 'E'
/// (to avoid splitting scientific notation like `1e+10`), and that appears after
/// at least one digit/dot character.
fn find_split_plus(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    // We skip index 0 — the real part must have at least one character before '+'.
    for i in 1..bytes.len() {
        if bytes[i] == b'+' {
            // Exclude exponent markers in scientific notation
            let prev = bytes[i - 1];
            if prev == b'e' || prev == b'E' {
                continue;
            }
            return Some(i);
        }
    }
    None
}

/// Parse the string representation of a symbolic matrix expression into a
/// `Vec<Vec<Complex64>>` row-major matrix.
///
/// Accepts the format produced by [`from_array2`] / [`from_array1`]:
///
/// ```text
/// Matrix([[cell, cell, ...], [cell, ...], ...])
/// ```
fn parse_matrix_expr(expr: &Expression) -> SymEngineResult<Vec<Vec<Complex64>>> {
    let raw = expr
        .as_symbol()
        .ok_or_else(|| SymEngineError::parse("expression is not a matrix symbol"))?;

    // Strip optional "Matrix(" prefix and matching ")"
    let inner = if raw.starts_with("Matrix(") && raw.ends_with(')') {
        &raw["Matrix(".len()..raw.len() - 1]
    } else {
        raw
    };

    // Expect outer "[...]"
    let inner = inner.trim();
    if !inner.starts_with('[') || !inner.ends_with(']') {
        return Err(SymEngineError::parse(format!(
            "expected outer '[...]' in matrix expression, got: {inner}"
        )));
    }
    let inner = &inner[1..inner.len() - 1];

    // Split into row strings by scanning bracket nesting
    let rows_strs = split_rows(inner);

    let mut rows: Vec<Vec<Complex64>> = Vec::with_capacity(rows_strs.len());
    for row_str in rows_strs {
        let row_str = row_str.trim();
        if !row_str.starts_with('[') || !row_str.ends_with(']') {
            return Err(SymEngineError::parse(format!(
                "expected row '[...]', got: {row_str}"
            )));
        }
        let cells_str = &row_str[1..row_str.len() - 1];
        let cells = split_cells(cells_str);
        let row: Vec<Complex64> = cells
            .iter()
            .map(|c| parse_cell(c.trim()))
            .collect::<Result<_, _>>()?;
        rows.push(row);
    }

    Ok(rows)
}

/// Split the contents of the outer `[...]` into individual `[row]` strings.
///
/// We track bracket depth so that nested `[cell]` groups are handled correctly.
fn split_rows(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth: usize = 0;
    let mut start: usize = 0;
    let bytes = s.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'[' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            b']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    parts.push(&s[start..=i]);
                }
            }
            _ => {}
        }
    }

    parts
}

/// Split a flat cell list (contents between `[` and `]` of a row) by commas,
/// respecting nested parentheses so that `(1+-2*I)` is not split.
fn split_cells(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth: usize = 0;
    let mut start: usize = 0;
    let bytes = s.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    // Push the final segment
    parts.push(&s[start..]);
    parts
}

/// Convert a symbolic matrix expression to a numeric `Array2<Complex64>`.
///
/// The expression is expected to be in the format produced by [`from_array2`],
/// i.e. `Matrix([[cell, cell, ...], [cell, ...], ...])`.
///
/// The `values` map is accepted for API uniformity but the matrix representation
/// already contains fully evaluated numeric cells; symbolic cells are not currently
/// supported.
///
/// # Errors
/// Returns an error if the expression is not a matrix symbol or cell parsing fails.
pub fn to_array2(
    expr: &Expression,
    _values: &std::collections::HashMap<String, f64>,
) -> SymEngineResult<Array2<Complex64>> {
    let rows = parse_matrix_expr(expr)?;

    if rows.is_empty() {
        return Ok(Array2::zeros((0, 0)));
    }

    let nrows = rows.len();
    let ncols = rows[0].len();

    // Validate uniform column count
    for (i, row) in rows.iter().enumerate() {
        if row.len() != ncols {
            return Err(SymEngineError::dimension(format!(
                "row {i} has {} columns, expected {ncols}",
                row.len()
            )));
        }
    }

    let flat: Vec<Complex64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((nrows, ncols), flat)
        .map_err(|e| SymEngineError::dimension(e.to_string()))
}

/// Convert a numeric `Array2<Complex64>` to a symbolic matrix expression.
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

/// Convert a symbolic vector expression to a numeric `Array1<Complex64>`.
///
/// The expression is expected to be in the format produced by [`from_array1`],
/// i.e. a column-vector matrix `Matrix([[c1], [c2], ...])`.  Each row must
/// contain exactly one cell.
///
/// The `values` map is accepted for API uniformity (see [`to_array2`]).
///
/// # Errors
/// Returns an error if the expression is not a matrix symbol or cell parsing fails.
pub fn to_array1(
    expr: &Expression,
    _values: &std::collections::HashMap<String, f64>,
) -> SymEngineResult<Array1<Complex64>> {
    let rows = parse_matrix_expr(expr)?;

    let flat: Vec<Complex64> = rows
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            if row.len() == 1 {
                Ok(row[0])
            } else {
                Err(SymEngineError::dimension(format!(
                    "row {i} has {} cells; expected 1 for Array1 conversion",
                    row.len()
                )))
            }
        })
        .collect::<Result<_, _>>()?;

    Ok(Array1::from_vec(flat))
}

/// Convert a numeric `Array1<Complex64>` to a symbolic column vector expression.
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

/// Compute the gradient at given values as an `Array1<f64>`.
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

/// Compute the Hessian at given values as an `Array2<f64>`.
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
    use std::collections::HashMap;

    /// Helper: build a values map (empty, since our matrices are fully numeric).
    fn no_values() -> HashMap<String, f64> {
        HashMap::new()
    }

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

    // =========================================================================
    // to_array1 / to_array2 round-trip tests
    // =========================================================================

    #[test]
    fn test_to_array1_real() {
        // Build a column-vector expression via from_array1 then round-trip through to_array1.
        let src: Array1<Complex64> = array![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        let expr = from_array1(&src);
        let arr = to_array1(&expr, &no_values()).expect("to_array1 should succeed");
        assert_eq!(arr.len(), 3);
        assert!((arr[0].re - 1.0).abs() < 1e-10);
        assert!((arr[1].re - 2.0).abs() < 1e-10);
        assert!((arr[2].re - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_array1_complex() {
        let src: Array1<Complex64> = array![
            Complex64::new(1.0, 2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(4.0, 0.0),
        ];
        let expr = from_array1(&src);
        let arr = to_array1(&expr, &no_values()).expect("to_array1 complex should succeed");
        assert_eq!(arr.len(), 3);
        assert!((arr[0].re - 1.0).abs() < 1e-10);
        assert!((arr[0].im - 2.0).abs() < 1e-10);
        assert!((arr[1].re - 0.0).abs() < 1e-10);
        assert!((arr[1].im - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_array2_2x2_real() {
        // Round-trip: from_array2 → Expression → to_array2
        let src: Array2<Complex64> = array![
            [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
        ];
        let expr = from_array2(&src);
        let arr = to_array2(&expr, &no_values()).expect("to_array2 should succeed");
        assert_eq!(arr.shape(), &[2, 2]);
        assert!((arr[[0, 0]].re - 1.0).abs() < 1e-10);
        assert!((arr[[0, 1]].re - 2.0).abs() < 1e-10);
        assert!((arr[[1, 0]].re - 3.0).abs() < 1e-10);
        assert!((arr[[1, 1]].re - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_array2_2x2_complex() {
        let src: Array2<Complex64> = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
            [Complex64::new(0.0, -1.0), Complex64::new(1.0, 0.0)],
        ];
        let expr = from_array2(&src);
        let arr = to_array2(&expr, &no_values()).expect("to_array2 complex should succeed");
        assert_eq!(arr.shape(), &[2, 2]);
        // (0,1) should be pure imaginary 0+1i
        assert!((arr[[0, 1]].re - 0.0).abs() < 1e-10);
        assert!((arr[[0, 1]].im - 1.0).abs() < 1e-10);
        // (1,0) should be pure imaginary 0-1i
        assert!((arr[[1, 0]].re - 0.0).abs() < 1e-10);
        assert!((arr[[1, 0]].im - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_to_array2_general_complex() {
        let src: Array2<Complex64> = array![[Complex64::new(3.0, 4.0)]];
        let expr = from_array2(&src);
        let arr = to_array2(&expr, &no_values()).expect("to_array2 general complex should succeed");
        assert_eq!(arr.shape(), &[1, 1]);
        assert!((arr[[0, 0]].re - 3.0).abs() < 1e-10);
        assert!((arr[[0, 0]].im - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_array2_negative_imaginary() {
        // Negative imaginary: from_array2 emits "(2+-3*I)" for Complex(2, -3)
        let src: Array2<Complex64> = array![[Complex64::new(2.0, -3.0)]];
        let expr = from_array2(&src);
        let arr = to_array2(&expr, &no_values())
            .expect("to_array2 negative imaginary should succeed");
        assert_eq!(arr.shape(), &[1, 1]);
        assert!((arr[[0, 0]].re - 2.0).abs() < 1e-10);
        assert!((arr[[0, 0]].im - (-3.0)).abs() < 1e-10);
    }
}
