//! Additional mathematical operations for `SymEngine` expressions.

use crate::Expression;
use scirs2_core::num_traits::{One, Zero}; // SciRS2 POLICY compliant

/// Mathematical constants
pub mod constants {
    use crate::Expression;

    /// Euler's number (e ≈ 2.71828...)
    #[must_use]
    pub fn e() -> Expression {
        Expression::new("E")
    }

    /// Pi (π ≈ 3.14159...)
    #[must_use]
    pub fn pi() -> Expression {
        Expression::new("pi")
    }

    /// Imaginary unit (i)
    #[must_use]
    pub fn i() -> Expression {
        Expression::new("I")
    }

    /// Golden ratio (φ ≈ 1.618...)
    #[must_use]
    pub fn golden_ratio() -> Expression {
        Expression::new("GoldenRatio")
    }

    /// Catalan's constant (≈ 0.915...)
    #[must_use]
    pub fn catalan() -> Expression {
        Expression::new("Catalan")
    }

    /// Euler-Mascheroni constant (γ ≈ 0.577...)
    #[must_use]
    pub fn euler_gamma() -> Expression {
        Expression::new("EulerGamma")
    }
}

/// Trigonometric functions
pub mod trig {
    use crate::{Expression, SymEngineResult};

    /// Sine function
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn sin(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("sin({expr})")))
    }

    /// Cosine function
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn cos(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("cos({expr})")))
    }

    /// Tangent function
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn tan(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("tan({expr})")))
    }

    /// Arcsine function
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn asin(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("asin({expr})")))
    }

    /// Arccosine function
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn acos(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("acos({expr})")))
    }

    /// Arctangent function
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn atan(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("atan({expr})")))
    }

    /// Two-argument arctangent function
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn atan2(y: &Expression, x: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("atan2({y}, {x})")))
    }

    /// Hyperbolic sine
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn sinh(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("sinh({expr})")))
    }

    /// Hyperbolic cosine
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn cosh(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("cosh({expr})")))
    }

    /// Hyperbolic tangent
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn tanh(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("tanh({expr})")))
    }
}

/// Exponential and logarithmic functions
pub mod exp_log {
    use crate::{Expression, SymEngineResult};

    /// Exponential function (e^x)
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn exp(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("exp({expr})")))
    }

    /// Natural logarithm (ln)
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn ln(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("log({expr})")))
    }

    /// Logarithm with specified base
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn log(expr: &Expression, base: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("log({expr}, {base})")))
    }

    /// Base-10 logarithm
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn log10(expr: &Expression) -> SymEngineResult<Expression> {
        log(expr, &Expression::from(10))
    }

    /// Base-2 logarithm
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn log2(expr: &Expression) -> SymEngineResult<Expression> {
        log(expr, &Expression::from(2))
    }

    /// Power function (base^exponent)
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn pow(base: &Expression, exponent: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("({base})^({exponent})")))
    }

    /// Square root
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn sqrt(expr: &Expression) -> SymEngineResult<Expression> {
        pow(expr, &Expression::new("1/2"))
    }

    /// Cube root
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn cbrt(expr: &Expression) -> SymEngineResult<Expression> {
        pow(expr, &Expression::new("1/3"))
    }

    /// nth root
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn root(expr: &Expression, n: &Expression) -> SymEngineResult<Expression> {
        pow(expr, &(Expression::from(1) / n.clone()))
    }
}

/// Special mathematical functions
pub mod special {
    use crate::{Expression, SymEngineResult};

    /// Gamma function
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn gamma(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("gamma({expr})")))
    }

    /// Logarithm of gamma function
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn log_gamma(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("loggamma({expr})")))
    }

    /// Beta function
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn beta(a: &Expression, b: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("beta({a}, {b})")))
    }

    /// Error function
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn erf(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("erf({expr})")))
    }

    /// Complementary error function
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn erfc(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("erfc({expr})")))
    }

    /// Bessel function of the first kind
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn bessel_j(n: &Expression, x: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("besselj({n}, {x})")))
    }

    /// Bessel function of the second kind
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn bessel_y(n: &Expression, x: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("bessely({n}, {x})")))
    }

    /// Modified Bessel function of the first kind
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn bessel_i(n: &Expression, x: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("besseli({n}, {x})")))
    }

    /// Modified Bessel function of the second kind
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn bessel_k(n: &Expression, x: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("besselk({n}, {x})")))
    }
}

/// Calculus operations
pub mod calculus {
    use crate::{Expression, SymEngineResult};

    /// Differentiate expression with respect to a symbol
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn diff(expr: &Expression, symbol: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("diff({expr}, {symbol})")))
    }

    /// Partial differentiate (alias for diff)
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn partial(expr: &Expression, symbol: &Expression) -> SymEngineResult<Expression> {
        diff(expr, symbol)
    }

    /// Higher-order differentiation
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn diff_n(expr: &Expression, symbol: &Expression, n: u32) -> SymEngineResult<Expression> {
        let mut result = expr.clone();
        for _ in 0..n {
            result = diff(&result, symbol)?;
        }
        Ok(result)
    }

    /// Integrate expression (indefinite integral)
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn integrate(expr: &Expression, symbol: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("integrate({expr}, {symbol})")))
    }

    /// Definite integral
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn integrate_definite(
        expr: &Expression,
        symbol: &Expression,
        lower: &Expression,
        upper: &Expression,
    ) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!(
            "integrate({expr}, ({symbol}, {lower}, {upper}))"
        )))
    }

    /// Limit of expression as symbol approaches value
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn limit(
        expr: &Expression,
        symbol: &Expression,
        value: &Expression,
    ) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("limit({expr}, {symbol}, {value})")))
    }

    /// Left limit
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn limit_left(
        expr: &Expression,
        symbol: &Expression,
        value: &Expression,
    ) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!(
            "limit({expr}, {symbol}, {value}, '-')"
        )))
    }

    /// Right limit
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn limit_right(
        expr: &Expression,
        symbol: &Expression,
        value: &Expression,
    ) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!(
            "limit({expr}, {symbol}, {value}, '+')"
        )))
    }

    /// Series expansion around a point
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn series(
        expr: &Expression,
        symbol: &Expression,
        point: &Expression,
        n_terms: u32,
    ) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!(
            "series({expr}, {symbol}, {point}, {n_terms})"
        )))
    }
}

/// Linear algebra operations
pub mod linalg {
    use crate::{Expression, SymEngineResult};

    /// Create a matrix from nested vectors
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn matrix(rows: Vec<Vec<Expression>>) -> SymEngineResult<Expression> {
        let rows_str: Vec<String> = rows
            .into_iter()
            .map(|row| {
                let elements: Vec<String> = row.into_iter().map(|e| e.to_string()).collect();
                format!("[{}]", elements.join(", "))
            })
            .collect();
        Ok(Expression::new(format!(
            "Matrix([{}])",
            rows_str.join(", ")
        )))
    }

    /// Create an identity matrix of size n×n
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn identity(n: usize) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("eye({n})")))
    }

    /// Create a zero matrix of size m×n
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn zeros(m: usize, n: usize) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("zeros({m}, {n})")))
    }

    /// Create a ones matrix of size m×n
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn ones(m: usize, n: usize) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("ones({m}, {n})")))
    }

    /// Matrix determinant
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn det(matrix: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("det({matrix})")))
    }

    /// Matrix trace
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn trace(matrix: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("trace({matrix})")))
    }

    /// Matrix transpose
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn transpose(matrix: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("transpose({matrix})")))
    }

    /// Matrix inverse
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn inverse(matrix: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("inverse({matrix})")))
    }
}

/// Number theory functions
pub mod number_theory {
    use crate::{Expression, SymEngineResult};

    /// Greatest common divisor
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn gcd(a: &Expression, b: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("gcd({a}, {b})")))
    }

    /// Least common multiple
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn lcm(a: &Expression, b: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("lcm({a}, {b})")))
    }

    /// Factorial
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn factorial(n: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("factorial({n})")))
    }

    /// Binomial coefficient
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn binomial(n: &Expression, k: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("binomial({n}, {k})")))
    }

    /// Prime factorization
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn factor(n: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("factor({n})")))
    }

    /// Check if number is prime
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn is_prime(n: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("isprime({n})")))
    }

    /// Next prime number
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn next_prime(n: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("nextprime({n})")))
    }

    /// Previous prime number
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn prev_prime(n: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("prevprime({n})")))
    }
}

impl Zero for Expression {
    fn zero() -> Self {
        Self::from(0)
    }

    fn is_zero(&self) -> bool {
        // Simplify and check multiple representations
        let simplified = self.simplify();
        let s = simplified.to_string();

        // Check various zero representations
        if s == "0" || s == "0.0" || s == "-0" || s == "+0" {
            return true;
        }

        // Try to convert to f64 and check if it's zero
        if let Some(val) = simplified.to_f64() {
            return val.abs() < 1e-10;
        }

        false
    }
}

impl One for Expression {
    fn one() -> Self {
        Self::from(1)
    }

    fn is_one(&self) -> bool {
        // This is a simplified check - in practice, you'd want to expand/simplify first
        self.to_string() == "1"
    }
}

/// Polynomial operations
pub mod polynomial {
    use crate::{Expression, SymEngineResult};

    /// Collect terms of an expression with respect to a symbol
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn collect(expr: &Expression, symbol: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("collect({expr}, {symbol})")))
    }

    /// Factor a polynomial expression
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn factor(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("factor({expr})")))
    }

    /// Expand a polynomial expression (trigonometric)
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn expand_trig(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("expand_trig({expr})")))
    }

    /// Expand complex expressions
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn expand_complex(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("expand_complex({expr})")))
    }

    /// Get the degree of a polynomial with respect to a symbol
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn degree(expr: &Expression, symbol: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("degree({expr}, {symbol})")))
    }

    /// Get the leading coefficient of a polynomial
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn leading_coeff(expr: &Expression, symbol: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("LC({expr}, {symbol})")))
    }

    /// Extract coefficient of a term
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn coeff(
        expr: &Expression,
        symbol: &Expression,
        n: &Expression,
    ) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("coeff({expr}, {symbol}, {n})")))
    }

    /// Polynomial division (quotient)
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn div_poly(
        dividend: &Expression,
        divisor: &Expression,
        symbol: &Expression,
    ) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!(
            "quo({dividend}, {divisor}, {symbol})"
        )))
    }

    /// Polynomial division (remainder)
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn rem_poly(
        dividend: &Expression,
        divisor: &Expression,
        symbol: &Expression,
    ) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!(
            "rem({dividend}, {divisor}, {symbol})"
        )))
    }

    /// Polynomial GCD
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn gcd_poly(a: &Expression, b: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("gcd({a}, {b})")))
    }

    /// Polynomial LCM
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn lcm_poly(a: &Expression, b: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("lcm({a}, {b})")))
    }
}

/// Simplification operations
pub mod simplification {
    use crate::{Expression, SymEngineResult};

    /// Simplify a general expression
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn simplify(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("simplify({expr})")))
    }

    /// Simplify trigonometric expressions
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn trigsimp(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("trigsimp({expr})")))
    }

    /// Simplify rational expressions
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn ratsimp(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("ratsimp({expr})")))
    }

    /// Simplify powers
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn powsimp(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("powsimp({expr})")))
    }

    /// Cancel common factors in a rational expression
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn cancel(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("cancel({expr})")))
    }

    /// Combine fractions with a common denominator
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn together(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("together({expr})")))
    }

    /// Separate expression into numerator and denominator
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn apart(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("apart({expr})")))
    }
}

/// Expression manipulation
pub mod manipulation {
    use crate::{Expression, SymEngineResult};

    /// Get all free symbols in an expression
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn free_symbols(expr: &Expression) -> SymEngineResult<Vec<String>> {
        // This is a placeholder - actual implementation would parse the expression
        Ok(vec![expr.to_string()])
    }

    /// Rewrite expression in terms of different basis functions
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn rewrite(expr: &Expression, target_func: &str) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("rewrite({expr}, {target_func})")))
    }

    /// Normal form of a boolean expression
    ///
    /// # Errors
    /// Returns an error if the underlying `SymEngine` operation fails.
    pub fn normal_form(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("to_cnf({expr})")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Expression;

    #[test]
    fn test_constants() {
        let pi = constants::pi();
        assert!(pi.to_string().contains("pi"));

        let e = constants::e();
        assert!(e.to_string().contains('E'));
    }

    #[test]
    fn test_trig_functions() {
        let x = Expression::symbol("x");
        let sin_x = trig::sin(&x).expect("Failed to create sin expression in test_trig_functions");
        assert!(sin_x.to_string().contains("sin"));

        let cos_x = trig::cos(&x).expect("Failed to create cos expression in test_trig_functions");
        assert!(cos_x.to_string().contains("cos"));
    }

    #[test]
    fn test_exp_log_functions() {
        let x = Expression::symbol("x");
        let exp_x =
            exp_log::exp(&x).expect("Failed to create exp expression in test_exp_log_functions");
        assert!(exp_x.to_string().contains("exp"));

        let ln_x =
            exp_log::ln(&x).expect("Failed to create ln expression in test_exp_log_functions");
        assert!(ln_x.to_string().contains("log"));
    }

    #[test]
    fn test_calculus_operations() {
        let x = Expression::symbol("x");
        let x_squared = &x * x.clone();
        let derivative = calculus::diff(&x_squared, &x)
            .expect("Failed to create diff expression in test_calculus_operations");
        assert!(derivative.to_string().contains("diff"));
    }

    #[test]
    fn test_traits() {
        let zero = Expression::zero();
        assert!(zero.is_zero());

        let one = Expression::one();
        assert!(one.is_one());
    }
}
