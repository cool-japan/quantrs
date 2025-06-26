//! Additional mathematical operations for SymEngine expressions.

use crate::Expression;
use num_traits::{Zero, One};

/// Mathematical constants
pub mod constants {
    use crate::Expression;
    
    /// Euler's number (e ≈ 2.71828...)
    pub fn e() -> Expression {
        Expression::new("E")
    }
    
    /// Pi (π ≈ 3.14159...)
    pub fn pi() -> Expression {
        Expression::new("pi")
    }
    
    /// Imaginary unit (i)
    pub fn i() -> Expression {
        Expression::new("I")
    }
    
    /// Golden ratio (φ ≈ 1.618...)
    pub fn golden_ratio() -> Expression {
        Expression::new("GoldenRatio")
    }
    
    /// Catalan's constant (≈ 0.915...)
    pub fn catalan() -> Expression {
        Expression::new("Catalan")
    }
    
    /// Euler-Mascheroni constant (γ ≈ 0.577...)
    pub fn euler_gamma() -> Expression {
        Expression::new("EulerGamma")
    }
}

/// Trigonometric functions
pub mod trig {
    use crate::{Expression, SymEngineResult};
    
    /// Sine function
    pub fn sin(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("sin({})", expr)))
    }
    
    /// Cosine function
    pub fn cos(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("cos({})", expr)))
    }
    
    /// Tangent function
    pub fn tan(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("tan({})", expr)))
    }
    
    /// Arcsine function
    pub fn asin(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("asin({})", expr)))
    }
    
    /// Arccosine function
    pub fn acos(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("acos({})", expr)))
    }
    
    /// Arctangent function
    pub fn atan(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("atan({})", expr)))
    }
    
    /// Two-argument arctangent function
    pub fn atan2(y: &Expression, x: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("atan2({}, {})", y, x)))
    }
    
    /// Hyperbolic sine
    pub fn sinh(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("sinh({})", expr)))
    }
    
    /// Hyperbolic cosine
    pub fn cosh(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("cosh({})", expr)))
    }
    
    /// Hyperbolic tangent
    pub fn tanh(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("tanh({})", expr)))
    }
}

/// Exponential and logarithmic functions
pub mod exp_log {
    use crate::{Expression, SymEngineResult};
    
    /// Exponential function (e^x)
    pub fn exp(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("exp({})", expr)))
    }
    
    /// Natural logarithm (ln)
    pub fn ln(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("log({})", expr)))
    }
    
    /// Logarithm with specified base
    pub fn log(expr: &Expression, base: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("log({}, {})", expr, base)))
    }
    
    /// Base-10 logarithm
    pub fn log10(expr: &Expression) -> SymEngineResult<Expression> {
        log(expr, &Expression::from(10))
    }
    
    /// Base-2 logarithm
    pub fn log2(expr: &Expression) -> SymEngineResult<Expression> {
        log(expr, &Expression::from(2))
    }
    
    /// Power function (base^exponent)
    pub fn pow(base: &Expression, exponent: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("({})^({})", base, exponent)))
    }
    
    /// Square root
    pub fn sqrt(expr: &Expression) -> SymEngineResult<Expression> {
        pow(expr, &Expression::new("1/2"))
    }
    
    /// Cube root
    pub fn cbrt(expr: &Expression) -> SymEngineResult<Expression> {
        pow(expr, &Expression::new("1/3"))
    }
    
    /// nth root
    pub fn root(expr: &Expression, n: &Expression) -> SymEngineResult<Expression> {
        pow(expr, &(Expression::from(1) / n.clone()))
    }
}

/// Special mathematical functions
pub mod special {
    use crate::{Expression, SymEngineResult};
    
    /// Gamma function
    pub fn gamma(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("gamma({})", expr)))
    }
    
    /// Logarithm of gamma function
    pub fn log_gamma(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("loggamma({})", expr)))
    }
    
    /// Beta function
    pub fn beta(a: &Expression, b: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("beta({}, {})", a, b)))
    }
    
    /// Error function
    pub fn erf(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("erf({})", expr)))
    }
    
    /// Complementary error function
    pub fn erfc(expr: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("erfc({})", expr)))
    }
    
    /// Bessel function of the first kind
    pub fn bessel_j(n: &Expression, x: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("besselj({}, {})", n, x)))
    }
    
    /// Bessel function of the second kind
    pub fn bessel_y(n: &Expression, x: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("bessely({}, {})", n, x)))
    }
    
    /// Modified Bessel function of the first kind
    pub fn bessel_i(n: &Expression, x: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("besseli({}, {})", n, x)))
    }
    
    /// Modified Bessel function of the second kind
    pub fn bessel_k(n: &Expression, x: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("besselk({}, {})", n, x)))
    }
}

/// Calculus operations
pub mod calculus {
    use crate::{Expression, SymEngineResult};
    
    /// Differentiate expression with respect to a symbol
    pub fn diff(expr: &Expression, symbol: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("diff({}, {})", expr, symbol)))
    }
    
    /// Partial differentiate (alias for diff)
    pub fn partial(expr: &Expression, symbol: &Expression) -> SymEngineResult<Expression> {
        diff(expr, symbol)
    }
    
    /// Higher-order differentiation
    pub fn diff_n(expr: &Expression, symbol: &Expression, n: u32) -> SymEngineResult<Expression> {
        let mut result = expr.clone();
        for _ in 0..n {
            result = diff(&result, symbol)?;
        }
        Ok(result)
    }
    
    /// Integrate expression (indefinite integral)
    pub fn integrate(expr: &Expression, symbol: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("integrate({}, {})", expr, symbol)))
    }
    
    /// Definite integral
    pub fn integrate_definite(
        expr: &Expression,
        symbol: &Expression,
        lower: &Expression,
        upper: &Expression,
    ) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!(
            "integrate({}, ({}, {}, {}))",
            expr, symbol, lower, upper
        )))
    }
    
    /// Limit of expression as symbol approaches value
    pub fn limit(
        expr: &Expression,
        symbol: &Expression,
        value: &Expression,
    ) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!(
            "limit({}, {}, {})",
            expr, symbol, value
        )))
    }
    
    /// Left limit
    pub fn limit_left(
        expr: &Expression,
        symbol: &Expression,
        value: &Expression,
    ) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!(
            "limit({}, {}, {}, '-')",
            expr, symbol, value
        )))
    }
    
    /// Right limit
    pub fn limit_right(
        expr: &Expression,
        symbol: &Expression,
        value: &Expression,
    ) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!(
            "limit({}, {}, {}, '+')",
            expr, symbol, value
        )))
    }
    
    /// Series expansion around a point
    pub fn series(
        expr: &Expression,
        symbol: &Expression,
        point: &Expression,
        n_terms: u32,
    ) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!(
            "series({}, {}, {}, {})",
            expr, symbol, point, n_terms
        )))
    }
}

/// Linear algebra operations
pub mod linalg {
    use crate::{Expression, SymEngineResult};
    
    /// Create a matrix from nested vectors
    pub fn matrix(rows: Vec<Vec<Expression>>) -> SymEngineResult<Expression> {
        let rows_str: Vec<String> = rows
            .into_iter()
            .map(|row| {
                let elements: Vec<String> = row.into_iter().map(|e| e.to_string()).collect();
                format!("[{}]", elements.join(", "))
            })
            .collect();
        Ok(Expression::new(format!("Matrix([{}])", rows_str.join(", "))))
    }
    
    /// Create an identity matrix of size n×n
    pub fn identity(n: usize) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("eye({})", n)))
    }
    
    /// Create a zero matrix of size m×n
    pub fn zeros(m: usize, n: usize) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("zeros({}, {})", m, n)))
    }
    
    /// Create a ones matrix of size m×n
    pub fn ones(m: usize, n: usize) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("ones({}, {})", m, n)))
    }
    
    /// Matrix determinant
    pub fn det(matrix: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("det({})", matrix)))
    }
    
    /// Matrix trace
    pub fn trace(matrix: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("trace({})", matrix)))
    }
    
    /// Matrix transpose
    pub fn transpose(matrix: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("transpose({})", matrix)))
    }
    
    /// Matrix inverse
    pub fn inverse(matrix: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("inverse({})", matrix)))
    }
}

/// Number theory functions
pub mod number_theory {
    use crate::{Expression, SymEngineResult};
    
    /// Greatest common divisor
    pub fn gcd(a: &Expression, b: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("gcd({}, {})", a, b)))
    }
    
    /// Least common multiple
    pub fn lcm(a: &Expression, b: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("lcm({}, {})", a, b)))
    }
    
    /// Factorial
    pub fn factorial(n: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("factorial({})", n)))
    }
    
    /// Binomial coefficient
    pub fn binomial(n: &Expression, k: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("binomial({}, {})", n, k)))
    }
    
    /// Prime factorization
    pub fn factor(n: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("factor({})", n)))
    }
    
    /// Check if number is prime
    pub fn is_prime(n: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("isprime({})", n)))
    }
    
    /// Next prime number
    pub fn next_prime(n: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("nextprime({})", n)))
    }
    
    /// Previous prime number
    pub fn prev_prime(n: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("prevprime({})", n)))
    }
}

impl Zero for Expression {
    fn zero() -> Self {
        Expression::from(0)
    }
    
    fn is_zero(&self) -> bool {
        // This is a simplified check - in practice, you'd want to expand/simplify first
        self.to_string() == "0"
    }
}

impl One for Expression {
    fn one() -> Self {
        Expression::from(1)
    }
    
    fn is_one(&self) -> bool {
        // This is a simplified check - in practice, you'd want to expand/simplify first
        self.to_string() == "1"
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
        assert!(e.to_string().contains("E"));
    }

    #[test]
    fn test_trig_functions() {
        let x = Expression::symbol("x");
        let sin_x = trig::sin(&x).unwrap();
        assert!(sin_x.to_string().contains("sin"));
        
        let cos_x = trig::cos(&x).unwrap();
        assert!(cos_x.to_string().contains("cos"));
    }

    #[test]
    fn test_exp_log_functions() {
        let x = Expression::symbol("x");
        let exp_x = exp_log::exp(&x).unwrap();
        assert!(exp_x.to_string().contains("exp"));
        
        let ln_x = exp_log::ln(&x).unwrap();
        assert!(ln_x.to_string().contains("log"));
    }

    #[test]
    fn test_calculus_operations() {
        let x = Expression::symbol("x");
        let x_squared = &x * x.clone();
        let derivative = calculus::diff(&x_squared, &x).unwrap();
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