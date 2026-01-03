//! Expression evaluation module.
//!
//! This module provides numeric evaluation of symbolic expressions.
//!
//! ## Real Evaluation
//!
//! For expressions without complex numbers:
//! ```ignore
//! use quantrs2_symengine_pure::eval::evaluate;
//! let expr = Expression::symbol("x") * Expression::int(2);
//! let result = evaluate(&expr, &values)?;
//! ```
//!
//! ## Complex Evaluation
//!
//! For expressions with imaginary unit `I`:
//! ```ignore
//! use quantrs2_symengine_pure::eval::evaluate_complex;
//! let expr = Expression::i() * Expression::symbol("x");
//! let result = evaluate_complex(&expr, &values)?;
//! ```

use std::collections::HashMap;

use scirs2_core::Complex64;

use crate::error::{SymEngineError, SymEngineResult};
use crate::expr::{ExprLang, Expression};

/// Evaluate an expression with given variable values
pub fn evaluate(expr: &Expression, values: &HashMap<String, f64>) -> SymEngineResult<f64> {
    let rec_expr = expr.as_rec_expr();
    let root_idx = rec_expr.as_ref().len() - 1;
    evaluate_rec(rec_expr.as_ref(), root_idx, values)
}

/// Recursive evaluation helper
fn evaluate_rec(
    nodes: &[ExprLang],
    idx: usize,
    values: &HashMap<String, f64>,
) -> SymEngineResult<f64> {
    let node = &nodes[idx];

    match node {
        ExprLang::Num(s) => {
            let name = s.as_str();
            // Try to parse as number first
            if let Ok(n) = name.parse::<f64>() {
                return Ok(n);
            }
            // Handle special constants
            match name {
                "pi" => Ok(std::f64::consts::PI),
                "e" => Ok(std::f64::consts::E),
                "I" => Err(SymEngineError::eval(
                    "Cannot evaluate complex unit i as real",
                )),
                _ => values
                    .get(name)
                    .copied()
                    .ok_or_else(|| SymEngineError::eval(format!("Undefined variable: {name}"))),
            }
        }

        ExprLang::Add([a, b]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            let vb = evaluate_rec(nodes, usize::from(*b), values)?;
            Ok(va + vb)
        }

        ExprLang::Mul([a, b]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            let vb = evaluate_rec(nodes, usize::from(*b), values)?;
            Ok(va * vb)
        }

        ExprLang::Div([a, b]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            let vb = evaluate_rec(nodes, usize::from(*b), values)?;
            if vb.abs() < 1e-15 {
                return Err(SymEngineError::DivisionByZero);
            }
            Ok(va / vb)
        }

        ExprLang::Pow([a, b]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            let vb = evaluate_rec(nodes, usize::from(*b), values)?;
            Ok(va.powf(vb))
        }

        ExprLang::Neg([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            Ok(-va)
        }

        ExprLang::Inv([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            if va.abs() < 1e-15 {
                return Err(SymEngineError::DivisionByZero);
            }
            Ok(1.0 / va)
        }

        ExprLang::Abs([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            Ok(va.abs())
        }

        ExprLang::Sin([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            Ok(va.sin())
        }

        ExprLang::Cos([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            Ok(va.cos())
        }

        ExprLang::Tan([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            Ok(va.tan())
        }

        ExprLang::Exp([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            Ok(va.exp())
        }

        ExprLang::Log([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            if va <= 0.0 {
                return Err(SymEngineError::Undefined(
                    "log of non-positive number".into(),
                ));
            }
            Ok(va.ln())
        }

        ExprLang::Sqrt([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            if va < 0.0 {
                return Err(SymEngineError::Undefined("sqrt of negative number".into()));
            }
            Ok(va.sqrt())
        }

        ExprLang::Asin([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            Ok(va.asin())
        }

        ExprLang::Acos([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            Ok(va.acos())
        }

        ExprLang::Atan([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            Ok(va.atan())
        }

        ExprLang::Sinh([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            Ok(va.sinh())
        }

        ExprLang::Cosh([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            Ok(va.cosh())
        }

        ExprLang::Tanh([a]) => {
            let va = evaluate_rec(nodes, usize::from(*a), values)?;
            Ok(va.tanh())
        }

        // Complex operations - need complex evaluation for proper handling
        ExprLang::Re([a]) | ExprLang::Im([a]) | ExprLang::Conj([a]) => {
            // For real numbers, re(x) = x, im(x) = 0, conj(x) = x
            evaluate_rec(nodes, usize::from(*a), values)
        }

        // Quantum operations cannot be evaluated numerically (symbolic only)
        ExprLang::Commutator([_, _])
        | ExprLang::Anticommutator([_, _])
        | ExprLang::TensorProduct([_, _])
        | ExprLang::Trace([_])
        | ExprLang::Dagger([_])
        | ExprLang::Determinant([_])
        | ExprLang::Transpose([_]) => Err(SymEngineError::eval(
            "Cannot evaluate symbolic quantum operation numerically",
        )),
    }
}

/// Batch evaluation for VQE optimization loops
pub fn evaluate_batch(
    expr: &Expression,
    values_list: &[HashMap<String, f64>],
) -> Vec<SymEngineResult<f64>> {
    values_list.iter().map(|v| evaluate(expr, v)).collect()
}

// =========================================================================
// Complex Evaluation
// =========================================================================

/// Evaluate an expression to a complex number.
///
/// This function can handle expressions containing the imaginary unit `I`,
/// which is essential for quantum computing applications.
///
/// # Arguments
/// * `expr` - The expression to evaluate
/// * `values` - Map of variable names to real values
///
/// # Returns
/// The complex result of the evaluation.
///
/// # Errors
/// Returns error if evaluation fails (undefined variables, etc.)
///
/// # Example
/// ```ignore
/// use quantrs2_symengine_pure::eval::evaluate_complex;
/// use std::collections::HashMap;
///
/// let expr = Expression::i() * Expression::symbol("x");
/// let mut values = HashMap::new();
/// values.insert("x".to_string(), 2.0);
///
/// let result = evaluate_complex(&expr, &values)?;
/// assert!((result.re - 0.0).abs() < 1e-10);
/// assert!((result.im - 2.0).abs() < 1e-10);
/// ```
pub fn evaluate_complex(
    expr: &Expression,
    values: &HashMap<String, f64>,
) -> SymEngineResult<Complex64> {
    let rec_expr = expr.as_rec_expr();
    let root_idx = rec_expr.as_ref().len() - 1;
    evaluate_complex_rec(rec_expr.as_ref(), root_idx, values)
}

/// Recursive complex evaluation helper
fn evaluate_complex_rec(
    nodes: &[ExprLang],
    idx: usize,
    values: &HashMap<String, f64>,
) -> SymEngineResult<Complex64> {
    let node = &nodes[idx];

    match node {
        ExprLang::Num(s) => {
            let name = s.as_str();
            // Try to parse as number first
            if let Ok(n) = name.parse::<f64>() {
                return Ok(Complex64::new(n, 0.0));
            }
            // Handle special constants
            match name {
                "pi" => Ok(Complex64::new(std::f64::consts::PI, 0.0)),
                "e" => Ok(Complex64::new(std::f64::consts::E, 0.0)),
                "I" => Ok(Complex64::new(0.0, 1.0)), // imaginary unit
                _ => values
                    .get(name)
                    .copied()
                    .map(|v| Complex64::new(v, 0.0))
                    .ok_or_else(|| SymEngineError::eval(format!("Undefined variable: {name}"))),
            }
        }

        ExprLang::Add([a, b]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            let vb = evaluate_complex_rec(nodes, usize::from(*b), values)?;
            Ok(va + vb)
        }

        ExprLang::Mul([a, b]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            let vb = evaluate_complex_rec(nodes, usize::from(*b), values)?;
            Ok(va * vb)
        }

        ExprLang::Div([a, b]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            let vb = evaluate_complex_rec(nodes, usize::from(*b), values)?;
            if vb.norm() < 1e-15 {
                return Err(SymEngineError::DivisionByZero);
            }
            Ok(va / vb)
        }

        ExprLang::Pow([a, b]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            let vb = evaluate_complex_rec(nodes, usize::from(*b), values)?;
            Ok(va.powc(vb))
        }

        ExprLang::Neg([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(-va)
        }

        ExprLang::Inv([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            if va.norm() < 1e-15 {
                return Err(SymEngineError::DivisionByZero);
            }
            Ok(Complex64::new(1.0, 0.0) / va)
        }

        ExprLang::Abs([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(Complex64::new(va.norm(), 0.0))
        }

        ExprLang::Sin([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(va.sin())
        }

        ExprLang::Cos([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(va.cos())
        }

        ExprLang::Tan([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(va.tan())
        }

        ExprLang::Exp([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(va.exp())
        }

        ExprLang::Log([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            if va.norm() < 1e-15 {
                return Err(SymEngineError::Undefined("log of zero".into()));
            }
            Ok(va.ln())
        }

        ExprLang::Sqrt([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(va.sqrt())
        }

        ExprLang::Asin([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(va.asin())
        }

        ExprLang::Acos([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(va.acos())
        }

        ExprLang::Atan([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(va.atan())
        }

        ExprLang::Sinh([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(va.sinh())
        }

        ExprLang::Cosh([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(va.cosh())
        }

        ExprLang::Tanh([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(va.tanh())
        }

        // Complex-specific operations
        ExprLang::Re([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(Complex64::new(va.re, 0.0))
        }

        ExprLang::Im([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(Complex64::new(va.im, 0.0))
        }

        ExprLang::Conj([a]) => {
            let va = evaluate_complex_rec(nodes, usize::from(*a), values)?;
            Ok(va.conj())
        }

        // Quantum operations cannot be evaluated numerically (symbolic only)
        ExprLang::Commutator([_, _])
        | ExprLang::Anticommutator([_, _])
        | ExprLang::TensorProduct([_, _])
        | ExprLang::Trace([_])
        | ExprLang::Dagger([_])
        | ExprLang::Determinant([_])
        | ExprLang::Transpose([_]) => Err(SymEngineError::eval(
            "Cannot evaluate symbolic quantum operation numerically",
        )),
    }
}

/// Batch complex evaluation for VQE optimization loops
pub fn evaluate_complex_batch(
    expr: &Expression,
    values_list: &[HashMap<String, f64>],
) -> Vec<SymEngineResult<Complex64>> {
    values_list
        .iter()
        .map(|v| evaluate_complex(expr, v))
        .collect()
}

/// Evaluate a complex expression with complex variable values.
///
/// This allows substituting complex values for variables, which is useful
/// for advanced quantum computing scenarios.
pub fn evaluate_complex_with_complex_values(
    expr: &Expression,
    values: &HashMap<String, Complex64>,
) -> SymEngineResult<Complex64> {
    let rec_expr = expr.as_rec_expr();
    let root_idx = rec_expr.as_ref().len() - 1;
    evaluate_complex_full_rec(rec_expr.as_ref(), root_idx, values)
}

/// Recursive complex evaluation with complex values
fn evaluate_complex_full_rec(
    nodes: &[ExprLang],
    idx: usize,
    values: &HashMap<String, Complex64>,
) -> SymEngineResult<Complex64> {
    let node = &nodes[idx];

    match node {
        ExprLang::Num(s) => {
            let name = s.as_str();
            // Try to parse as number first
            if let Ok(n) = name.parse::<f64>() {
                return Ok(Complex64::new(n, 0.0));
            }
            // Handle special constants
            match name {
                "pi" => Ok(Complex64::new(std::f64::consts::PI, 0.0)),
                "e" => Ok(Complex64::new(std::f64::consts::E, 0.0)),
                "I" => Ok(Complex64::new(0.0, 1.0)),
                _ => values
                    .get(name)
                    .copied()
                    .ok_or_else(|| SymEngineError::eval(format!("Undefined variable: {name}"))),
            }
        }

        ExprLang::Add([a, b]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            let vb = evaluate_complex_full_rec(nodes, usize::from(*b), values)?;
            Ok(va + vb)
        }

        ExprLang::Mul([a, b]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            let vb = evaluate_complex_full_rec(nodes, usize::from(*b), values)?;
            Ok(va * vb)
        }

        ExprLang::Div([a, b]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            let vb = evaluate_complex_full_rec(nodes, usize::from(*b), values)?;
            if vb.norm() < 1e-15 {
                return Err(SymEngineError::DivisionByZero);
            }
            Ok(va / vb)
        }

        ExprLang::Pow([a, b]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            let vb = evaluate_complex_full_rec(nodes, usize::from(*b), values)?;
            Ok(va.powc(vb))
        }

        ExprLang::Neg([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(-va)
        }

        ExprLang::Inv([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            if va.norm() < 1e-15 {
                return Err(SymEngineError::DivisionByZero);
            }
            Ok(Complex64::new(1.0, 0.0) / va)
        }

        ExprLang::Abs([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(Complex64::new(va.norm(), 0.0))
        }

        ExprLang::Sin([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(va.sin())
        }

        ExprLang::Cos([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(va.cos())
        }

        ExprLang::Tan([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(va.tan())
        }

        ExprLang::Exp([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(va.exp())
        }

        ExprLang::Log([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            if va.norm() < 1e-15 {
                return Err(SymEngineError::Undefined("log of zero".into()));
            }
            Ok(va.ln())
        }

        ExprLang::Sqrt([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(va.sqrt())
        }

        ExprLang::Asin([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(va.asin())
        }

        ExprLang::Acos([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(va.acos())
        }

        ExprLang::Atan([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(va.atan())
        }

        ExprLang::Sinh([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(va.sinh())
        }

        ExprLang::Cosh([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(va.cosh())
        }

        ExprLang::Tanh([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(va.tanh())
        }

        ExprLang::Re([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(Complex64::new(va.re, 0.0))
        }

        ExprLang::Im([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(Complex64::new(va.im, 0.0))
        }

        ExprLang::Conj([a]) => {
            let va = evaluate_complex_full_rec(nodes, usize::from(*a), values)?;
            Ok(va.conj())
        }

        ExprLang::Commutator([_, _])
        | ExprLang::Anticommutator([_, _])
        | ExprLang::TensorProduct([_, _])
        | ExprLang::Trace([_])
        | ExprLang::Dagger([_])
        | ExprLang::Determinant([_])
        | ExprLang::Transpose([_]) => Err(SymEngineError::eval(
            "Cannot evaluate symbolic quantum operation numerically",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_constant() {
        let c = Expression::int(42);
        let result = evaluate(&c, &HashMap::new()).expect("should evaluate");
        assert!((result - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_variable() {
        let x = Expression::symbol("x");
        let mut values = HashMap::new();
        values.insert("x".to_string(), 2.5);

        let result = evaluate(&x, &values).expect("should evaluate");
        assert!((result - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expression() {
        let x = Expression::symbol("x");
        let expr = x.clone() * x; // x^2
        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);

        let result = evaluate(&expr, &values).expect("should evaluate");
        assert!((result - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_trig() {
        let x = Expression::symbol("x");
        let sin_x = crate::ops::trig::sin(&x);
        let mut values = HashMap::new();
        values.insert("x".to_string(), std::f64::consts::PI / 2.0);

        let result = evaluate(&sin_x, &values).expect("should evaluate");
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_division_by_zero() {
        let one = Expression::one();
        let zero = Expression::zero();
        let expr = one / zero;

        let result = evaluate(&expr, &HashMap::new());
        assert!(result.is_err());
    }

    // =========================================================================
    // Complex Evaluation Tests
    // =========================================================================

    #[test]
    fn test_eval_complex_imaginary_unit() {
        let i = Expression::i();
        let result = evaluate_complex(&i, &HashMap::new()).expect("should evaluate");
        assert!((result.re - 0.0).abs() < 1e-10);
        assert!((result.im - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_complex_i_squared() {
        // i^2 = -1
        let i = Expression::i();
        let i_squared = i.clone() * i;
        let result = evaluate_complex(&i_squared, &HashMap::new()).expect("should evaluate");
        assert!((result.re - (-1.0)).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);
    }

    #[test]
    fn test_eval_complex_expression() {
        // (3 + 2i) represented as 3 + 2*I
        let three = Expression::int(3);
        let two = Expression::int(2);
        let i = Expression::i();
        let expr = three + two * i;

        let result = evaluate_complex(&expr, &HashMap::new()).expect("should evaluate");
        assert!((result.re - 3.0).abs() < 1e-10);
        assert!((result.im - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_complex_with_variable() {
        // x * I with x = 5
        let x = Expression::symbol("x");
        let i = Expression::i();
        let expr = x * i;

        let mut values = HashMap::new();
        values.insert("x".to_string(), 5.0);

        let result = evaluate_complex(&expr, &values).expect("should evaluate");
        assert!((result.re - 0.0).abs() < 1e-10);
        assert!((result.im - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_complex_exp() {
        // e^(i*pi) = -1 (Euler's identity)
        let i = Expression::i();
        let pi = Expression::float_unchecked(std::f64::consts::PI);
        let expr = crate::ops::trig::exp(&(i * pi));

        let result = evaluate_complex(&expr, &HashMap::new()).expect("should evaluate");
        assert!((result.re - (-1.0)).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);
    }

    #[test]
    fn test_eval_complex_conjugate() {
        // conj(3 + 2i) = 3 - 2i
        let three = Expression::int(3);
        let two = Expression::int(2);
        let i = Expression::i();
        let z = three + two * i;
        let conj_z = crate::ops::complex::conj(&z);

        let result = evaluate_complex(&conj_z, &HashMap::new()).expect("should evaluate");
        assert!((result.re - 3.0).abs() < 1e-10);
        assert!((result.im - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_eval_complex_real_part() {
        // re(3 + 2i) = 3
        let three = Expression::int(3);
        let two = Expression::int(2);
        let i = Expression::i();
        let z = three + two * i;
        let re_z = crate::ops::complex::re(&z);

        let result = evaluate_complex(&re_z, &HashMap::new()).expect("should evaluate");
        assert!((result.re - 3.0).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);
    }

    #[test]
    fn test_eval_complex_imag_part() {
        // im(3 + 2i) = 2
        let three = Expression::int(3);
        let two = Expression::int(2);
        let i = Expression::i();
        let z = three + two * i;
        let im_z = crate::ops::complex::im(&z);

        let result = evaluate_complex(&im_z, &HashMap::new()).expect("should evaluate");
        assert!((result.re - 2.0).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);
    }

    #[test]
    fn test_eval_complex_with_complex_values() {
        // z with z = 1 + 2i
        let z = Expression::symbol("z");

        let mut values = HashMap::new();
        values.insert("z".to_string(), Complex64::new(1.0, 2.0));

        let result = evaluate_complex_with_complex_values(&z, &values).expect("should evaluate");
        assert!((result.re - 1.0).abs() < 1e-10);
        assert!((result.im - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_complex_abs() {
        // |3 + 4i| = 5
        let three = Expression::int(3);
        let four = Expression::int(4);
        let i = Expression::i();
        let z = three + four * i;
        let abs_z = crate::ops::trig::abs(&z);

        let result = evaluate_complex(&abs_z, &HashMap::new()).expect("should evaluate");
        assert!((result.re - 5.0).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);
    }
}
