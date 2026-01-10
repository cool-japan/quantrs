//! Complex number integration with SciRS2.
//!
//! This module provides conversion between symbolic expressions and
//! SciRS2's Complex64 type.

use scirs2_core::Complex64;

use crate::error::{SymEngineError, SymEngineResult};
use crate::expr::Expression;

/// Convert a symbolic expression to a Complex64 if possible.
///
/// # Errors
/// Returns an error if the expression cannot be evaluated to a complex number.
pub fn to_complex64(expr: &Expression) -> SymEngineResult<Complex64> {
    // First try to evaluate as real
    if let Some(re) = expr.to_f64() {
        return Ok(Complex64::new(re, 0.0));
    }

    // Check if it's a pure imaginary (i * value)
    // This is a simplified check - full implementation would parse the expression tree

    Err(SymEngineError::eval(
        "Cannot convert symbolic expression to Complex64 - try evaluating with specific values",
    ))
}

/// Create an expression from a Complex64.
///
/// Returns a simplified expression when possible (e.g., real for pure real numbers).
pub fn from_complex64(c: Complex64) -> Expression {
    Expression::from_complex64(c)
}

/// Evaluate a symbolic expression to a Complex64 with given variable values.
///
/// # Arguments
/// * `expr` - The expression to evaluate
/// * `values` - Map of variable names to complex values
///
/// # Errors
/// Returns an error if evaluation fails.
pub fn eval_complex(
    expr: &Expression,
    values: &std::collections::HashMap<String, Complex64>,
) -> SymEngineResult<Complex64> {
    // For now, only support real evaluation
    // TODO: Implement full complex evaluation

    let real_values: std::collections::HashMap<String, f64> = values
        .iter()
        .filter(|(_, v)| v.im.abs() < 1e-15)
        .map(|(k, v)| (k.clone(), v.re))
        .collect();

    if real_values.len() != values.len() {
        return Err(SymEngineError::not_impl(
            "Complex variable evaluation not yet implemented",
        ));
    }

    let result = expr.eval(&real_values)?;
    Ok(Complex64::new(result, 0.0))
}

/// Create complex arithmetic expressions.
pub mod complex_ops {
    use super::*;

    /// Create a complex number expression a + bi
    pub fn complex(re: f64, im: f64) -> Expression {
        Expression::from_complex64(Complex64::new(re, im))
    }

    /// Create a pure imaginary number i*b
    pub fn imag(b: f64) -> Expression {
        Expression::float_unchecked(b) * Expression::i()
    }

    /// Polar form: r * e^(iθ)
    pub fn polar(r: f64, theta: f64) -> Expression {
        let c = Complex64::from_polar(r, theta);
        Expression::from_complex64(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_complex64_real() {
        let c = Complex64::new(2.5, 0.0);
        let expr = from_complex64(c);
        assert!(expr.is_number());
    }

    #[test]
    fn test_from_complex64_pure_imag() {
        let c = Complex64::new(0.0, 2.0);
        let expr = from_complex64(c);
        // Should be 2 * I
        assert!(!expr.is_symbol());
    }

    #[test]
    fn test_from_complex64_general() {
        let c = Complex64::new(3.0, 4.0);
        let expr = from_complex64(c);
        // Should be 3 + 4*I
        assert!(!expr.is_symbol());
    }

    #[test]
    fn test_complex_ops() {
        use complex_ops::*;

        let c = complex(1.0, 2.0);
        let i = imag(3.0);
        let p = polar(1.0, std::f64::consts::FRAC_PI_2);

        assert!(!c.is_symbol());
        assert!(!i.is_symbol());
        assert!(!p.is_symbol());
    }
}
