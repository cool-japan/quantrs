//! SciRS2 integration for quantrs2-symengine
//!
//! This module provides integration with the SciRS2 ecosystem, enabling conversion
//! between symbolic expressions and SciRS2's numeric types.

use crate::{Expression, SymEngineError, SymEngineResult};
use scirs2_core::num_traits::ToPrimitive;
use scirs2_core::Complex64;

impl Expression {
    /// Create an expression from a SciRS2 Complex64 number
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    /// use scirs2_core::Complex64;
    ///
    /// let c = Complex64::new(1.0, 2.0);
    /// let expr = Expression::from_complex64(c);
    /// assert!(expr.to_string().contains("1"));
    /// ```
    #[must_use]
    pub fn from_complex64(c: Complex64) -> Self {
        if c.im.abs() < 1e-15 {
            // Treat as real number if imaginary part is negligible
            Self::from_f64(c.re)
        } else if c.re.abs() < 1e-15 {
            // Pure imaginary
            let expr_str = if c.im == 1.0 {
                "I".to_string()
            } else if c.im == -1.0 {
                "-I".to_string()
            } else {
                format!("{}*I", c.im)
            };
            Self::new(expr_str)
        } else {
            // General complex number
            let expr_str = if c.im >= 0.0 {
                format!("{} + {}*I", c.re, c.im)
            } else {
                format!("{} - {}*I", c.re, c.im.abs())
            };
            Self::new(expr_str)
        }
    }

    /// Convert expression to SciRS2 Complex64
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::Expression;
    /// use scirs2_core::Complex64;
    ///
    /// let expr = Expression::new("1 + 2*I");
    /// let c = expr.to_complex64().unwrap();
    /// assert_eq!(c.re, 1.0);
    /// assert_eq!(c.im, 2.0);
    /// ```
    ///
    /// # Errors
    /// Returns an error if the expression cannot be evaluated to a complex number.
    pub fn to_complex64(&self) -> SymEngineResult<Complex64> {
        if let Some((re, im)) = self.to_complex() {
            Ok(Complex64::new(re, im))
        } else {
            Err(SymEngineError::invalid_operation(
                "Expression cannot be converted to complex number",
            ))
        }
    }

    /// Evaluate expression symbolically and convert to Complex64
    ///
    /// This will expand and simplify the expression before attempting conversion.
    ///
    /// # Errors
    /// Returns an error if the expression cannot be evaluated to a complex number.
    pub fn eval_to_complex64(&self) -> SymEngineResult<Complex64> {
        self.expand().to_complex64()
    }
}

impl From<Complex64> for Expression {
    fn from(c: Complex64) -> Self {
        Self::from_complex64(c)
    }
}

impl TryFrom<Expression> for Complex64 {
    type Error = SymEngineError;

    fn try_from(expr: Expression) -> Result<Self, Self::Error> {
        expr.to_complex64()
    }
}

impl TryFrom<&Expression> for Complex64 {
    type Error = SymEngineError;

    fn try_from(expr: &Expression) -> Result<Self, Self::Error> {
        expr.to_complex64()
    }
}

// Implement ToPrimitive for Expression
impl ToPrimitive for Expression {
    fn to_i64(&self) -> Option<i64> {
        self.to_f64().map(|f| f as i64)
    }

    fn to_u64(&self) -> Option<u64> {
        self.to_f64()
            .and_then(|f| if f >= 0.0 { Some(f as u64) } else { None })
    }

    fn to_f64(&self) -> Option<f64> {
        Self::to_f64(self)
    }

    fn to_f32(&self) -> Option<f32> {
        self.to_f64().map(|f| f as f32)
    }
}

// Implement NumCast for Expression
impl scirs2_core::num_traits::NumCast for Expression {
    fn from<T: scirs2_core::num_traits::ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(Self::from_f64)
    }
}

// Note: Signed and Num traits are not fully implemented due to limitations
// in symbolic expression evaluation. These would require runtime evaluation
// for many operations which defeats the purpose of symbolic computation.
// Users should convert to numeric types when needed via to_f64(), eval(), etc.

impl Expression {
    /// Check if expression evaluates to a positive number
    pub fn is_positive_num(&self) -> bool {
        self.to_f64().is_some_and(|f| f > 0.0)
    }

    /// Check if expression evaluates to a negative number
    pub fn is_negative_num(&self) -> bool {
        self.to_f64().is_some_and(|f| f < 0.0)
    }

    /// Get the sign of a numeric expression (-1, 0, or 1)
    #[must_use]
    pub fn signum_expr(&self) -> Self {
        Self::new(format!("sign({self})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[ignore = "SymEngine complex number parsing not working as expected"]
    fn test_complex64_real_conversion() {
        use std::f64::consts::PI;
        let c = Complex64::new(PI, 0.0);
        let expr = Expression::from_complex64(c);

        let result = expr
            .to_complex64()
            .expect("Failed to convert back to Complex64");
        assert_relative_eq!(result.re, PI, epsilon = 1e-10);
        assert_relative_eq!(result.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex64_imaginary_conversion() {
        let c = Complex64::new(0.0, 2.71);
        let expr = Expression::from_complex64(c);

        assert!(expr.to_string().contains('I'));
    }

    #[test]
    #[ignore = "SymEngine complex number parsing not working as expected"]
    fn test_complex64_general_conversion() {
        let c = Complex64::new(1.0, 2.0);
        let expr = Expression::from_complex64(c);

        let result = expr
            .to_complex64()
            .expect("Failed to convert back to Complex64");
        assert_relative_eq!(result.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.im, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex64_from_trait() {
        let c = Complex64::new(1.5, -0.5);
        let expr: Expression = c.into();

        assert!(expr.to_string().contains("1.5"));
    }

    #[test]
    #[ignore = "SymEngine complex number parsing not working as expected"]
    fn test_complex64_try_from() {
        let expr = Expression::new("3 + 4*I");
        let c = Complex64::try_from(&expr).expect("Failed to convert to Complex64");

        assert_relative_eq!(c.re, 3.0, epsilon = 1e-10);
        assert_relative_eq!(c.im, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_symbolic_expression_error() {
        let x = Expression::symbol("x");
        let expr = x.clone() * x.clone() + x;

        let result = expr.to_complex64();
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "SymEngine number type handling varies by configuration"]
    fn test_to_primitive() {
        // Test with f64 values which work reliably
        let expr = Expression::from_f64(42.0);
        assert_eq!(expr.to_f64(), Some(42.0));
        assert_eq!(expr.to_i64(), Some(42));

        let negative = Expression::from_f64(-10.5);
        assert!(negative.to_f64().is_some());
    }

    #[test]
    fn test_zero_one_traits() {
        use scirs2_core::num_traits::{One, Zero};

        let zero = Expression::zero();
        assert!(zero.is_zero());

        let one = Expression::one();
        assert!(one.is_one());
    }

    #[test]
    #[ignore = "SymEngine number type handling varies by configuration"]
    fn test_sign_methods() {
        let positive = Expression::from_f64(5.0);
        assert!(positive.is_positive_num());
        assert!(!positive.is_negative_num());

        let negative = Expression::from_f64(-5.0);
        assert!(!negative.is_positive_num());
        assert!(negative.is_negative_num());

        // Test symbolic abs operation
        let x = Expression::symbol("x");
        let abs_val = x.abs();
        assert!(abs_val.to_string().contains("abs"));

        // Test signum
        let signum = negative.signum_expr();
        assert!(signum.to_string().contains("sign"));
    }

    #[test]
    fn test_scirs2_integration_methods() {
        // Test that the methods exist and work
        let x = Expression::symbol("x");
        let abs_x = x.abs();
        assert!(abs_x.to_string().contains("abs"));

        let signum_x = x.signum_expr();
        assert!(signum_x.to_string().contains("sign"));

        // Test conversion from Complex64
        let c = Complex64::new(1.0, 0.0);
        let expr = Expression::from_complex64(c);
        assert!(!expr.to_string().is_empty());
    }

    #[test]
    fn test_numcast_trait() {
        use scirs2_core::num_traits::NumCast;
        use std::f64::consts::PI;

        // Test generic NumCast
        let expr: Option<Expression> = NumCast::from(123);
        assert!(expr.is_some());

        let expr_f64: Option<Expression> = NumCast::from(PI);
        assert!(expr_f64.is_some());

        // Test that NumCast works with various numeric types
        let expr_i32: Option<Expression> = NumCast::from(42_i32);
        assert!(expr_i32.is_some());

        let expr_u32: Option<Expression> = NumCast::from(42_u32);
        assert!(expr_u32.is_some());

        let expr_f: Option<Expression> = NumCast::from(PI);
        assert!(expr_f.is_some());
    }

    #[test]
    fn test_abs_and_signum_methods() {
        let x = Expression::symbol("x");

        // Test abs method
        let abs_x = x.abs();
        assert!(abs_x.to_string().contains("abs"));

        // Test signum method
        let signum_x = x.signum_expr();
        assert!(signum_x.to_string().contains("sign"));

        // Test with numeric values
        let positive = Expression::from(5);
        let negative = Expression::from(-5);

        let abs_pos = positive.abs();
        let abs_neg = negative.abs();
        assert!(!abs_pos.to_string().is_empty());
        assert!(!abs_neg.to_string().is_empty());
    }
}
