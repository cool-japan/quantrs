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
/// Supports full complex arithmetic including expressions containing the
/// imaginary unit `I`, complex variable substitution, and all standard
/// mathematical functions.
///
/// # Arguments
/// * `expr` - The expression to evaluate
/// * `values` - Map of variable names to complex values
///
/// # Errors
/// Returns an error if evaluation fails (undefined variable, division by zero, etc.).
pub fn eval_complex(
    expr: &Expression,
    values: &std::collections::HashMap<String, Complex64>,
) -> SymEngineResult<Complex64> {
    crate::eval::evaluate_complex_with_complex_values(expr, values)
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

    // =========================================================================
    // eval_complex tests (Stub 2c)
    // =========================================================================

    #[test]
    fn test_eval_complex_pure_real() {
        // Expression 7.0 with no variables → Complex(7.0, 0.0)
        let expr = Expression::float_unchecked(7.0);
        let values = std::collections::HashMap::new();
        let result = eval_complex(&expr, &values).expect("should evaluate pure real");
        assert!((result.re - 7.0).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);
    }

    #[test]
    fn test_eval_complex_real_plus_imag() {
        // Expression 2.0 + 3.0*I → Complex(2.0, 3.0)
        let two = Expression::float_unchecked(2.0);
        let three = Expression::float_unchecked(3.0);
        let i = Expression::i();
        let expr = two + three * i;

        let values = std::collections::HashMap::new();
        let result = eval_complex(&expr, &values).expect("should evaluate 2+3i");
        assert!((result.re - 2.0).abs() < 1e-10);
        assert!((result.im - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_complex_with_complex_var() {
        // z where z = 1 + 2i
        let z = Expression::symbol("z");
        let mut values = std::collections::HashMap::new();
        values.insert("z".to_string(), Complex64::new(1.0, 2.0));

        let result = eval_complex(&z, &values).expect("should evaluate complex variable");
        assert!((result.re - 1.0).abs() < 1e-10);
        assert!((result.im - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_complex_pure_imaginary() {
        // Expression 5.0 * I → Complex(0.0, 5.0)
        let five = Expression::float_unchecked(5.0);
        let i = Expression::i();
        let expr = five * i;

        let values = std::collections::HashMap::new();
        let result = eval_complex(&expr, &values).expect("should evaluate pure imaginary");
        assert!(result.re.abs() < 1e-10);
        assert!((result.im - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_complex_i_squared() {
        // I * I = -1
        let i1 = Expression::i();
        let i2 = Expression::i();
        let expr = i1 * i2;

        let values = std::collections::HashMap::new();
        let result = eval_complex(&expr, &values).expect("I*I should be -1");
        assert!((result.re - (-1.0)).abs() < 1e-10);
        assert!(result.im.abs() < 1e-10);
    }
}
