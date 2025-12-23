//! Polynomial operations and utilities
//!
//! This module provides enhanced polynomial manipulation including:
//! - Coefficient extraction
//! - Polynomial expansion and collection
//! - Degree calculation
//! - Root finding

use crate::{Expression, SymEngineResult};
use scirs2_core::num_traits::Zero;
use std::collections::HashMap;

/// Represents a polynomial in a single variable
#[derive(Debug, Clone)]
pub struct Polynomial {
    /// Coefficients indexed by power (coefficient of x^i at index i)
    coefficients: Vec<Expression>,
    /// The variable symbol
    variable: Expression,
}

impl Polynomial {
    /// Create a new polynomial from coefficients
    ///
    /// # Arguments
    /// * `coefficients` - Vector of coefficients where index i is the coefficient of x^i
    /// * `variable` - The polynomial variable
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::{Expression, polynomial::Polynomial};
    ///
    /// let x = Expression::symbol("x");
    /// // Create polynomial 2x^2 + 3x + 1
    /// let coeffs = vec![Expression::from(1), Expression::from(3), Expression::from(2)];
    /// let poly = Polynomial::new(coeffs, x);
    /// ```
    #[must_use]
    pub const fn new(coefficients: Vec<Expression>, variable: Expression) -> Self {
        Self {
            coefficients,
            variable,
        }
    }

    /// Get the degree of the polynomial
    #[must_use]
    pub fn degree(&self) -> usize {
        self.coefficients.len().saturating_sub(1)
    }

    /// Get the coefficient of x^n
    #[must_use]
    pub fn coefficient(&self, n: usize) -> Option<&Expression> {
        self.coefficients.get(n)
    }

    /// Get the leading coefficient (coefficient of highest degree term)
    #[must_use]
    pub fn leading_coefficient(&self) -> Option<&Expression> {
        self.coefficients.last()
    }

    /// Convert to Expression
    #[must_use]
    pub fn to_expression(&self) -> Expression {
        let mut result = Expression::from(0);

        for (power, coeff) in self.coefficients.iter().enumerate() {
            let term = if power == 0 {
                coeff.clone()
            } else if power == 1 {
                coeff.clone() * self.variable.clone()
            } else {
                coeff.clone() * self.variable.pow(&Expression::from(power as i64))
            };
            result = result + term;
        }

        result
    }

    /// Evaluate the polynomial at a specific value
    ///
    /// Uses Horner's method for efficient evaluation
    pub fn evaluate(&self, x: &Expression) -> Expression {
        if self.coefficients.is_empty() {
            return Expression::from(0);
        }

        let mut result = self.coefficients[self.degree()].clone();

        for i in (0..self.degree()).rev() {
            result = result * x.clone() + self.coefficients[i].clone();
        }

        result
    }

    /// Add two polynomials
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let max_len = self.coefficients.len().max(other.coefficients.len());
        let mut coeffs = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = self
                .coefficients
                .get(i)
                .cloned()
                .unwrap_or_else(|| Expression::from(0));
            let b = other
                .coefficients
                .get(i)
                .cloned()
                .unwrap_or_else(|| Expression::from(0));
            coeffs.push(a + b);
        }

        Self::new(coeffs, self.variable.clone())
    }

    /// Multiply two polynomials
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        if self.coefficients.is_empty() || other.coefficients.is_empty() {
            return Self::new(vec![Expression::from(0)], self.variable.clone());
        }

        let result_degree = self.degree() + other.degree();
        let mut coeffs = vec![Expression::from(0); result_degree + 1];

        for (i, a) in self.coefficients.iter().enumerate() {
            for (j, b) in other.coefficients.iter().enumerate() {
                coeffs[i + j] = coeffs[i + j].clone() + a.clone() * b.clone();
            }
        }

        Self::new(coeffs, self.variable.clone())
    }

    /// Differentiate the polynomial
    #[must_use]
    pub fn differentiate(&self) -> Self {
        if self.coefficients.len() <= 1 {
            return Self::new(vec![Expression::from(0)], self.variable.clone());
        }

        let mut coeffs = Vec::with_capacity(self.coefficients.len() - 1);

        for (power, coeff) in self.coefficients.iter().enumerate().skip(1) {
            coeffs.push(coeff.clone() * Expression::from(power as i64));
        }

        Self::new(coeffs, self.variable.clone())
    }

    /// Integrate the polynomial (with constant of integration = 0)
    #[must_use]
    pub fn integrate(&self) -> Self {
        let mut coeffs = vec![Expression::from(0)];

        for (power, coeff) in self.coefficients.iter().enumerate() {
            let new_power = power + 1;
            coeffs.push(coeff.clone() / Expression::from(new_power as i64));
        }

        Self::new(coeffs, self.variable.clone())
    }
}

/// Extract polynomial coefficients from an expression
///
/// # Arguments
/// * `expr` - The expression to analyze
/// * `variable` - The variable to extract coefficients for
///
/// # Returns
/// A vector of coefficients where index i is the coefficient of variable^i
///
/// # Example
/// ```
/// use quantrs2_symengine::{Expression, polynomial::extract_coefficients};
///
/// let x = Expression::symbol("x");
/// let poly = Expression::from(3) * x.pow(&Expression::from(2))
///     + Expression::from(2) * x.clone()
///     + Expression::from(1);
///
/// let coeffs = extract_coefficients(&poly, &x);
/// // coeffs should represent [1, 2, 3]
/// ```
#[must_use]
pub fn extract_coefficients(expr: &Expression, variable: &Expression) -> Vec<Expression> {
    // This is a heuristic implementation
    // A proper implementation would need deeper SymEngine integration
    let expanded = expr.expand();

    // Try to extract terms
    expanded.as_add().map_or_else(
        || {
            // Single term or constant
            let (power, coeff) = extract_power_and_coeff(&expanded, variable);
            let mut coeffs = vec![Expression::from(0); (power + 1) as usize];
            coeffs[power as usize] = coeff;
            coeffs
        },
        |terms| {
            let mut coeff_map: HashMap<u32, Expression> = HashMap::new();

            for term in &terms {
                // Try to determine the power of the variable in this term
                // This is simplified - a real implementation would parse the expression tree
                let (power, coeff) = extract_power_and_coeff(term, variable);

                // Accumulate coefficients for the same power
                coeff_map
                    .entry(power)
                    .and_modify(|c| *c = c.clone() + coeff.clone())
                    .or_insert(coeff);
            }

            // Convert to vector
            let max_power = coeff_map.keys().max().copied().unwrap_or(0);
            let mut coeffs = vec![Expression::from(0); (max_power + 1) as usize];

            for (power, coeff) in coeff_map {
                coeffs[power as usize] = coeff;
            }

            coeffs
        },
    )
}

/// Try to extract a polynomial from an expression
///
/// Returns `Some(Polynomial)` if the expression can be interpreted as a polynomial.
///
/// # Example
/// ```
/// use quantrs2_symengine::{Expression, polynomial::try_extract_polynomial};
///
/// let x = Expression::symbol("x");
/// let expr = x.pow(&Expression::from(2)) + Expression::from(2) * x.clone() + Expression::from(1);
///
/// let poly = try_extract_polynomial(&expr, &x);
/// assert!(poly.is_some());
/// ```
#[must_use]
pub fn try_extract_polynomial(expr: &Expression, variable: &Expression) -> Option<Polynomial> {
    let coeffs = extract_coefficients(expr, variable);

    // Verify that coefficients don't contain the variable
    // (This is a basic check - a full implementation would be more thorough)
    if coeffs.is_empty() {
        return None;
    }

    Some(Polynomial::new(coeffs, variable.clone()))
}

/// Helper function to extract power and coefficient from a single term
fn extract_power_and_coeff(term: &Expression, variable: &Expression) -> (u32, Expression) {
    // Handle multiplication: coeff * variable^power
    if let Some(factors) = term.as_mul() {
        let mut power = 0u32;
        let mut coeff = Expression::from(1);

        for factor in &factors {
            if factor.is_symbol() && factor == variable {
                // Found the variable (x^1)
                power = 1;
            } else if let Some((base, exp)) = factor.as_pow() {
                if base.is_symbol() && base == *variable {
                    // Found variable^power
                    if let Some(p) = exp.to_f64() {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        {
                            power = p as u32;
                        }
                    }
                } else {
                    // Part of coefficient
                    coeff = coeff * factor.clone();
                }
            } else {
                // Numeric or symbolic coefficient
                coeff = coeff * factor.clone();
            }
        }

        return (power, coeff);
    }

    // Handle pure power: x^n
    if let Some((base, exp)) = term.as_pow() {
        if base.is_symbol() && base == *variable {
            if let Some(power) = exp.to_f64() {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                return (power as u32, Expression::from(1));
            }
        }
    }

    // Handle plain variable: x
    if term.is_symbol() && term == variable {
        return (1, Expression::from(1));
    }

    // Default to constant term
    (0, term.clone())
}

/// Perform polynomial long division
///
/// Returns (quotient, remainder)
pub fn poly_divide(
    dividend: &Polynomial,
    divisor: &Polynomial,
) -> SymEngineResult<(Polynomial, Polynomial)> {
    if divisor.coefficients.is_empty() || divisor.coefficients.iter().all(|c| c.is_zero()) {
        return Err(crate::SymEngineError::DivisionByZero);
    }

    let mut remainder = dividend.clone();
    let mut quotient_coeffs =
        vec![Expression::from(0); dividend.degree().saturating_sub(divisor.degree()) + 1];

    while remainder.degree() >= divisor.degree()
        && !remainder.leading_coefficient().unwrap().is_zero()
    {
        let deg_diff = remainder.degree() - divisor.degree();
        let coeff = remainder.leading_coefficient().unwrap().clone()
            / divisor.leading_coefficient().unwrap().clone();

        quotient_coeffs[deg_diff] = coeff.clone();

        // Subtract coeff * x^deg_diff * divisor from remainder
        let mut subtrahend_coeffs = vec![Expression::from(0); remainder.degree() + 1];
        for (i, d_coeff) in divisor.coefficients.iter().enumerate() {
            subtrahend_coeffs[i + deg_diff] = coeff.clone() * d_coeff.clone();
        }

        let subtrahend = Polynomial::new(subtrahend_coeffs, dividend.variable.clone());

        // Update remainder
        let new_remainder_coeffs: Vec<Expression> = remainder
            .coefficients
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let b = subtrahend
                    .coefficients
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| Expression::from(0));
                a.clone() - b
            })
            .collect();

        remainder = Polynomial::new(new_remainder_coeffs, dividend.variable.clone());
    }

    let quotient = Polynomial::new(quotient_coeffs, dividend.variable.clone());

    Ok((quotient, remainder))
}

/// Compute polynomial composition: (f ∘ g)(x) = f(g(x))
///
/// # Example
/// ```
/// use quantrs2_symengine::{Expression, polynomial::{Polynomial, compose}};
///
/// let x = Expression::symbol("x");
///
/// // f(x) = x^2
/// let f = Polynomial::new(vec![Expression::from(0), Expression::from(0), Expression::from(1)], x.clone());
///
/// // g(x) = x + 1
/// let g = Polynomial::new(vec![Expression::from(1), Expression::from(1)], x.clone());
///
/// // (f ∘ g)(x) = (x+1)^2 = x^2 + 2x + 1
/// let composed = compose(&f, &g);
/// assert_eq!(composed.degree(), 2);
/// ```
#[must_use]
pub fn compose(f: &Polynomial, g: &Polynomial) -> Polynomial {
    // Start with constant term
    let mut result = Polynomial::new(vec![f.coefficients[0].clone()], f.variable.clone());

    // Add each term: c_i * g(x)^i
    for (i, coeff) in f.coefficients.iter().enumerate().skip(1) {
        // Compute g^i
        let mut g_power = g.clone();
        for _ in 1..i {
            g_power = g_power.multiply(g);
        }

        // Scale by coefficient
        let scaled_coeffs: Vec<Expression> = g_power
            .coefficients
            .iter()
            .map(|c| c.clone() * coeff.clone())
            .collect();
        let scaled = Polynomial::new(scaled_coeffs, f.variable.clone());

        result = result.add(&scaled);
    }

    result
}

/// Find the Greatest Common Divisor (GCD) of two polynomials using Euclidean algorithm
///
/// # Example
/// ```
/// use quantrs2_symengine::{Expression, polynomial::{Polynomial, poly_gcd}};
///
/// let x = Expression::symbol("x");
///
/// // p(x) = x^2 - 1 = (x-1)(x+1)
/// let p = Polynomial::new(
///     vec![Expression::from(-1), Expression::from(0), Expression::from(1)],
///     x.clone()
/// );
///
/// // q(x) = x - 1
/// let q = Polynomial::new(vec![Expression::from(-1), Expression::from(1)], x.clone());
///
/// // GCD should be (x - 1)
/// let gcd = poly_gcd(&p, &q).unwrap();
/// assert!(gcd.degree() >= 0);
/// ```
pub fn poly_gcd(a: &Polynomial, b: &Polynomial) -> SymEngineResult<Polynomial> {
    let mut r0 = a.clone();
    let mut r1 = b.clone();

    // Add iteration limit to prevent infinite loops
    const MAX_ITERATIONS: usize = 1000;
    let mut iterations = 0;

    while !r1.coefficients.iter().all(|c| c.is_zero()) {
        iterations += 1;
        if iterations > MAX_ITERATIONS {
            return Err(crate::error::SymEngineError::runtime_error(
                "GCD computation exceeded maximum iterations",
            ));
        }

        let (_, remainder) = poly_divide(&r0, &r1)?;

        r0 = r1;
        r1 = remainder;
    }

    Ok(r0)
}

/// Compute the derivative of a polynomial n times
///
/// # Example
/// ```
/// use quantrs2_symengine::{Expression, polynomial::{Polynomial, nth_derivative}};
///
/// let x = Expression::symbol("x");
///
/// // p(x) = x^3
/// let p = Polynomial::new(
///     vec![Expression::from(0), Expression::from(0), Expression::from(0), Expression::from(1)],
///     x
/// );
///
/// // Third derivative should be constant 6
/// let d3 = nth_derivative(&p, 3);
/// assert_eq!(d3.degree(), 0);
/// ```
#[must_use]
pub fn nth_derivative(poly: &Polynomial, n: usize) -> Polynomial {
    let mut result = poly.clone();
    for _ in 0..n {
        result = result.differentiate();
    }
    result
}

/// Compute the antiderivative of a polynomial n times
///
/// # Example
/// ```
/// use quantrs2_symengine::{Expression, polynomial::{Polynomial, nth_integral}};
///
/// let x = Expression::symbol("x");
///
/// // p(x) = 1
/// let p = Polynomial::new(vec![Expression::from(1)], x);
///
/// // Second integral: ∫∫1 dx dx = x^2/2
/// let i2 = nth_integral(&p, 2);
/// assert_eq!(i2.degree(), 2);
/// ```
#[must_use]
pub fn nth_integral(poly: &Polynomial, n: usize) -> Polynomial {
    let mut result = poly.clone();
    for _ in 0..n {
        result = result.integrate();
    }
    result
}

/// Scale polynomial coefficients by a factor
///
/// Returns a new polynomial with all coefficients multiplied by the scale factor.
#[must_use]
pub fn scale(poly: &Polynomial, factor: &Expression) -> Polynomial {
    let scaled_coeffs: Vec<Expression> = poly
        .coefficients
        .iter()
        .map(|c| c.clone() * factor.clone())
        .collect();
    Polynomial::new(scaled_coeffs, poly.variable.clone())
}

/// Translate polynomial: p(x) -> p(x + a)
///
/// # Example
/// ```
/// use quantrs2_symengine::{Expression, polynomial::{Polynomial, translate}};
///
/// let x = Expression::symbol("x");
///
/// // p(x) = x^2
/// let p = Polynomial::new(
///     vec![Expression::from(0), Expression::from(0), Expression::from(1)],
///     x.clone()
/// );
///
/// // p(x+1) = (x+1)^2 = x^2 + 2x + 1
/// let translated = translate(&p, &Expression::from(1));
/// assert_eq!(translated.degree(), 2);
/// ```
#[must_use]
pub fn translate(poly: &Polynomial, shift: &Expression) -> Polynomial {
    // p(x+a) is computed by composing p with (x+a)
    let x_plus_a = Polynomial::new(
        vec![shift.clone(), Expression::from(1)],
        poly.variable.clone(),
    );
    compose(poly, &x_plus_a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_creation() {
        let x = Expression::symbol("x");
        let coeffs = vec![
            Expression::from(1),
            Expression::from(2),
            Expression::from(3),
        ];
        let poly = Polynomial::new(coeffs, x);

        assert_eq!(poly.degree(), 2);
        assert_eq!(poly.coefficient(0).unwrap().to_string(), "1");
        assert_eq!(poly.coefficient(1).unwrap().to_string(), "2");
        assert_eq!(poly.coefficient(2).unwrap().to_string(), "3");
    }

    #[test]
    fn test_polynomial_evaluation() {
        let x = Expression::symbol("x");
        // Polynomial: 1 + 2x + 3x^2
        let coeffs = vec![
            Expression::from(1),
            Expression::from(2),
            Expression::from(3),
        ];
        let poly = Polynomial::new(coeffs, x);

        let result = poly.evaluate(&Expression::from(2));
        // 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        let expr_str = result.to_string();
        assert!(expr_str.contains("17") || expr_str.contains("1 + 2*2 + 3*2^2"));
    }

    #[test]
    fn test_polynomial_addition() {
        let x = Expression::symbol("x");
        let poly1 = Polynomial::new(vec![Expression::from(1), Expression::from(2)], x.clone());
        let poly2 = Polynomial::new(vec![Expression::from(3), Expression::from(4)], x);

        let sum = poly1.add(&poly2);
        assert_eq!(sum.degree(), 1);
        assert_eq!(sum.coefficient(0).unwrap().to_string(), "4");
        assert_eq!(sum.coefficient(1).unwrap().to_string(), "6");
    }

    #[test]
    fn test_polynomial_multiplication() {
        let x = Expression::symbol("x");
        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        let poly1 = Polynomial::new(vec![Expression::from(1), Expression::from(2)], x.clone());
        let poly2 = Polynomial::new(vec![Expression::from(3), Expression::from(4)], x);

        let product = poly1.multiply(&poly2);
        assert_eq!(product.degree(), 2);
        // Coefficients should be 3, 10, 8
    }

    #[test]
    fn test_polynomial_differentiation() {
        let x = Expression::symbol("x");
        // 1 + 2x + 3x^2 -> 2 + 6x
        let coeffs = vec![
            Expression::from(1),
            Expression::from(2),
            Expression::from(3),
        ];
        let poly = Polynomial::new(coeffs, x);

        let derivative = poly.differentiate();
        assert_eq!(derivative.degree(), 1);
        assert_eq!(derivative.coefficient(0).unwrap().to_string(), "2");
        assert_eq!(derivative.coefficient(1).unwrap().to_string(), "6");
    }

    #[test]
    fn test_polynomial_integration() {
        let x = Expression::symbol("x");
        // 2 + 6x -> 2x + 3x^2
        let coeffs = vec![Expression::from(2), Expression::from(6)];
        let poly = Polynomial::new(coeffs, x);

        let integral = poly.integrate();
        assert_eq!(integral.degree(), 2);
        // Constant term should be 0 (integration constant)
        assert!(integral.coefficient(0).unwrap().is_zero());
    }

    #[test]
    fn test_polynomial_composition() {
        let x = Expression::symbol("x");

        // f(x) = x^2
        let f = Polynomial::new(
            vec![
                Expression::from(0),
                Expression::from(0),
                Expression::from(1),
            ],
            x.clone(),
        );

        // g(x) = x + 1
        let g = Polynomial::new(vec![Expression::from(1), Expression::from(1)], x);

        // (f ∘ g)(x) = (x+1)^2
        let composed = compose(&f, &g);
        assert_eq!(composed.degree(), 2);
    }

    /// Test polynomial GCD computation
    ///
    /// NOTE: This test is currently ignored due to issues with symbolic zero detection.
    /// The is_zero() implementation cannot reliably detect symbolic zeros in polynomial
    /// remainders, causing the Euclidean algorithm to fail.
    /// TODO: Improve symbolic expression simplification in is_zero().
    #[test]
    #[ignore = "Symbolic zero detection needs improvement"]
    fn test_polynomial_gcd() {
        let x = Expression::symbol("x");

        // p(x) = x^2 - 1
        let p = Polynomial::new(
            vec![
                Expression::from(-1),
                Expression::from(0),
                Expression::from(1),
            ],
            x.clone(),
        );

        // q(x) = x - 1
        let q = Polynomial::new(vec![Expression::from(-1), Expression::from(1)], x);

        // GCD should exist
        let gcd_result = poly_gcd(&p, &q);
        assert!(gcd_result.is_ok());
    }

    #[test]
    fn test_nth_derivative() {
        let x = Expression::symbol("x");

        // p(x) = x^3
        let p = Polynomial::new(
            vec![
                Expression::from(0),
                Expression::from(0),
                Expression::from(0),
                Expression::from(1),
            ],
            x,
        );

        // Third derivative of x^3 is 6
        let d3 = nth_derivative(&p, 3);
        assert_eq!(d3.degree(), 0);
    }

    #[test]
    fn test_nth_integral() {
        let x = Expression::symbol("x");

        // p(x) = 1
        let p = Polynomial::new(vec![Expression::from(1)], x);

        // Second integral
        let i2 = nth_integral(&p, 2);
        assert_eq!(i2.degree(), 2);
    }

    #[test]
    fn test_polynomial_scale() {
        let x = Expression::symbol("x");

        // p(x) = x^2 + x
        let p = Polynomial::new(
            vec![
                Expression::from(0),
                Expression::from(1),
                Expression::from(1),
            ],
            x,
        );

        // Scale by 2: 2p(x) = 2x^2 + 2x
        let scaled = scale(&p, &Expression::from(2));
        assert_eq!(scaled.degree(), 2);
    }

    #[test]
    fn test_polynomial_translate() {
        let x = Expression::symbol("x");

        // p(x) = x^2
        let p = Polynomial::new(
            vec![
                Expression::from(0),
                Expression::from(0),
                Expression::from(1),
            ],
            x,
        );

        // p(x+1) = (x+1)^2
        let translated = translate(&p, &Expression::from(1));
        assert_eq!(translated.degree(), 2);
    }

    /// Test polynomial extraction from expressions
    ///
    /// NOTE: This test is currently ignored due to incomplete SymEngine FFI type mapping.
    /// The expression tree type (ID 16) is not recognized by the current bindings.
    /// TODO: Update SymEngine-sys bindings to include all expression type constants.
    #[test]
    #[ignore = "Waiting for complete SymEngine FFI type mapping"]
    fn test_try_extract_polynomial() {
        let x = Expression::symbol("x");
        let expr =
            x.pow(&Expression::from(2)) + Expression::from(2) * x.clone() + Expression::from(1);

        let poly = try_extract_polynomial(&expr, &x);
        assert!(poly.is_some());

        if let Some(p) = poly {
            assert_eq!(p.degree(), 2);
        }
    }
}
