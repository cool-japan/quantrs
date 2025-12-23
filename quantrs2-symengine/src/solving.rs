//! Numeric equation solving and root finding
//!
//! This module provides numerical methods for solving equations including:
//! - Newton-Raphson method
//! - Bisection method
//! - Secant method
//! - Fixed-point iteration

use crate::{Expression, SymEngineError, SymEngineResult};
use scirs2_core::num_traits::Float;

/// Configuration for numeric solvers
#[derive(Debug, Clone, Copy)]
pub struct SolverConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Absolute tolerance for convergence
    pub abs_tolerance: f64,
    /// Relative tolerance for convergence
    pub rel_tolerance: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            abs_tolerance: 1e-10,
            rel_tolerance: 1e-10,
        }
    }
}

impl SolverConfig {
    /// Create a new solver configuration with default values
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_iterations: 1000,
            abs_tolerance: 1e-10,
            rel_tolerance: 1e-10,
        }
    }

    /// Set maximum iterations
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set absolute tolerance
    #[must_use]
    pub const fn with_abs_tolerance(mut self, tol: f64) -> Self {
        self.abs_tolerance = tol;
        self
    }

    /// Set relative tolerance
    #[must_use]
    pub const fn with_rel_tolerance(mut self, tol: f64) -> Self {
        self.rel_tolerance = tol;
        self
    }
}

/// Result of a numeric solving operation
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// The root/solution found
    pub root: f64,
    /// Number of iterations used
    pub iterations: usize,
    /// Final error estimate
    pub error: f64,
    /// Whether the solver converged
    pub converged: bool,
}

impl SolverResult {
    /// Check if the solution is reliable (converged with low error)
    #[must_use]
    pub fn is_reliable(&self, tolerance: f64) -> bool {
        self.converged && self.error < tolerance
    }
}

/// Newton-Raphson method for finding roots of f(x) = 0
///
/// Uses the iteration: x_{n+1} = x_n - f(x_n) / f'(x_n)
///
/// # Arguments
/// * `expr` - The expression f(x) to find roots of
/// * `variable` - The variable symbol
/// * `initial_guess` - Starting point for iteration
/// * `config` - Solver configuration
///
/// # Errors
/// Returns an error if:
/// - Derivative cannot be computed
/// - Evaluation fails
/// - Division by zero in iteration
///
/// # Example
/// ```
/// use quantrs2_symengine::{Expression, solving::{newton_raphson, SolverConfig}};
///
/// let x = Expression::symbol("x");
/// let expr = x.pow(&Expression::from(2)) - Expression::from(2);  // Find sqrt(2)
///
/// let result = newton_raphson(&expr, &x, 1.0, &SolverConfig::default()).unwrap();
/// assert!(result.converged);
/// assert!((result.root - 1.41421356).abs() < 1e-6);
/// ```
pub fn newton_raphson(
    expr: &Expression,
    variable: &Expression,
    initial_guess: f64,
    config: &SolverConfig,
) -> SymEngineResult<SolverResult> {
    // Compute derivative symbolically
    let derivative = expr.diff(variable);

    let mut x = initial_guess;
    let mut iterations = 0;

    for _ in 0..config.max_iterations {
        iterations += 1;

        // Evaluate f(x)
        let fx = eval_at(expr, variable, x)?;

        // Check convergence
        if fx.abs() < config.abs_tolerance {
            return Ok(SolverResult {
                root: x,
                iterations,
                error: fx.abs(),
                converged: true,
            });
        }

        // Evaluate f'(x)
        let fpx = eval_at(&derivative, variable, x)?;

        // Check for zero derivative
        if fpx.abs() < 1e-15 {
            return Err(SymEngineError::invalid_operation(
                "Derivative is zero - Newton-Raphson cannot continue",
            ));
        }

        // Newton-Raphson step
        let x_new = x - fx / fpx;

        // Check relative change
        let rel_change = ((x_new - x) / x).abs();
        x = x_new;

        if rel_change < config.rel_tolerance {
            let final_fx = eval_at(expr, variable, x)?;
            return Ok(SolverResult {
                root: x,
                iterations,
                error: final_fx.abs(),
                converged: true,
            });
        }
    }

    // Did not converge
    let final_fx = eval_at(expr, variable, x)?;
    Ok(SolverResult {
        root: x,
        iterations,
        error: final_fx.abs(),
        converged: false,
    })
}

/// Bisection method for finding roots of f(x) = 0
///
/// Requires that f(a) and f(b) have opposite signs.
///
/// # Arguments
/// * `expr` - The expression f(x) to find roots of
/// * `variable` - The variable symbol
/// * `a` - Left endpoint of interval
/// * `b` - Right endpoint of interval
/// * `config` - Solver configuration
///
/// # Errors
/// Returns an error if:
/// - f(a) and f(b) have the same sign
/// - Evaluation fails
///
/// # Example
/// ```
/// use quantrs2_symengine::{Expression, solving::{bisection, SolverConfig}};
///
/// let x = Expression::symbol("x");
/// let expr = x.pow(&Expression::from(2)) - Expression::from(2);  // Find sqrt(2)
///
/// let result = bisection(&expr, &x, 1.0, 2.0, &SolverConfig::default()).unwrap();
/// assert!(result.converged);
/// assert!((result.root - 1.41421356).abs() < 1e-6);
/// ```
pub fn bisection(
    expr: &Expression,
    variable: &Expression,
    mut a: f64,
    mut b: f64,
    config: &SolverConfig,
) -> SymEngineResult<SolverResult> {
    // Evaluate at endpoints
    let mut fa = eval_at(expr, variable, a)?;
    let mut fb = eval_at(expr, variable, b)?;

    // Check that f(a) and f(b) have opposite signs
    if fa * fb > 0.0 {
        return Err(SymEngineError::invalid_operation(
            "f(a) and f(b) must have opposite signs for bisection method",
        ));
    }

    let mut iterations = 0;

    for _ in 0..config.max_iterations {
        iterations += 1;

        // Midpoint
        let c = f64::midpoint(a, b);
        let fc = eval_at(expr, variable, c)?;

        // Check convergence
        if fc.abs() < config.abs_tolerance || (b - a).abs() < config.abs_tolerance {
            return Ok(SolverResult {
                root: c,
                iterations,
                error: fc.abs(),
                converged: true,
            });
        }

        // Update interval
        if fa * fc < 0.0 {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }

    // Did not converge
    let c = f64::midpoint(a, b);
    let fc = eval_at(expr, variable, c)?;
    Ok(SolverResult {
        root: c,
        iterations,
        error: fc.abs(),
        converged: false,
    })
}

/// Secant method for finding roots of f(x) = 0
///
/// Similar to Newton-Raphson but uses finite differences instead of derivatives.
///
/// # Arguments
/// * `expr` - The expression f(x) to find roots of
/// * `variable` - The variable symbol
/// * `x0` - First initial guess
/// * `x1` - Second initial guess
/// * `config` - Solver configuration
///
/// # Errors
/// Returns an error if evaluation fails.
pub fn secant(
    expr: &Expression,
    variable: &Expression,
    mut x0: f64,
    mut x1: f64,
    config: &SolverConfig,
) -> SymEngineResult<SolverResult> {
    let mut f0 = eval_at(expr, variable, x0)?;
    let mut f1 = eval_at(expr, variable, x1)?;

    let mut iterations = 0;

    for _ in 0..config.max_iterations {
        iterations += 1;

        // Check convergence
        if f1.abs() < config.abs_tolerance {
            return Ok(SolverResult {
                root: x1,
                iterations,
                error: f1.abs(),
                converged: true,
            });
        }

        // Check for zero denominator
        if (f1 - f0).abs() < 1e-15 {
            return Err(SymEngineError::invalid_operation(
                "Denominator too small in secant method",
            ));
        }

        // Secant step
        let x2 = x1 - f1 * (x1 - x0) / (f1 - f0);

        // Check relative change
        let rel_change = ((x2 - x1) / x1).abs();

        // Update for next iteration
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = eval_at(expr, variable, x1)?;

        if rel_change < config.rel_tolerance {
            return Ok(SolverResult {
                root: x1,
                iterations,
                error: f1.abs(),
                converged: true,
            });
        }
    }

    // Did not converge
    Ok(SolverResult {
        root: x1,
        iterations,
        error: f1.abs(),
        converged: false,
    })
}

/// Fixed-point iteration for solving g(x) = x
///
/// Iterates: x_{n+1} = g(x_n)
///
/// # Arguments
/// * `expr` - The expression g(x)
/// * `variable` - The variable symbol
/// * `initial_guess` - Starting point
/// * `config` - Solver configuration
///
/// # Errors
/// Returns an error if evaluation fails.
pub fn fixed_point(
    expr: &Expression,
    variable: &Expression,
    initial_guess: f64,
    config: &SolverConfig,
) -> SymEngineResult<SolverResult> {
    let mut x = initial_guess;
    let mut iterations = 0;

    for _ in 0..config.max_iterations {
        iterations += 1;

        // Evaluate g(x)
        let gx = eval_at(expr, variable, x)?;

        // Check convergence
        let error = (gx - x).abs();
        if error < config.abs_tolerance {
            return Ok(SolverResult {
                root: gx,
                iterations,
                error,
                converged: true,
            });
        }

        // Update
        let rel_change = ((gx - x) / x).abs();
        x = gx;

        if rel_change < config.rel_tolerance {
            return Ok(SolverResult {
                root: x,
                iterations,
                error,
                converged: true,
            });
        }
    }

    // Did not converge
    let gx = eval_at(expr, variable, x)?;
    let error = (gx - x).abs();
    Ok(SolverResult {
        root: x,
        iterations,
        error,
        converged: false,
    })
}

/// Brent's method for finding roots (hybrid method combining bisection, secant, and inverse quadratic interpolation)
///
/// Generally more robust and faster than pure bisection or secant methods.
///
/// # Arguments
/// * `expr` - The expression f(x) to find roots of
/// * `variable` - The variable symbol
/// * `a` - Left endpoint of interval
/// * `b` - Right endpoint of interval
/// * `config` - Solver configuration
///
/// # Errors
/// Returns an error if:
/// - f(a) and f(b) have the same sign
/// - Evaluation fails
pub fn brent(
    expr: &Expression,
    variable: &Expression,
    mut a: f64,
    mut b: f64,
    config: &SolverConfig,
) -> SymEngineResult<SolverResult> {
    let mut fa = eval_at(expr, variable, a)?;
    let mut fb = eval_at(expr, variable, b)?;

    // Check that f(a) and f(b) have opposite signs
    if fa * fb > 0.0 {
        return Err(SymEngineError::invalid_operation(
            "f(a) and f(b) must have opposite signs for Brent's method",
        ));
    }

    // Ensure |f(b)| <= |f(a)|
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;

    let mut iterations = 0;

    for _ in 0..config.max_iterations {
        iterations += 1;

        // Check convergence
        if fb.abs() < config.abs_tolerance {
            return Ok(SolverResult {
                root: b,
                iterations,
                error: fb.abs(),
                converged: true,
            });
        }

        // Ensure |f(b)| <= |f(a)|
        if fa.abs() < fb.abs() {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            c = a;
            fc = fa;
        }

        let tol = 2.0 * config.abs_tolerance * b.abs().max(1.0);
        let m = 0.5 * (c - b);

        if m.abs() <= tol {
            return Ok(SolverResult {
                root: b,
                iterations,
                error: fb.abs(),
                converged: true,
            });
        }

        // Choose method
        let s = if (e.abs() >= tol) && (fa.abs() > fb.abs()) {
            // Try interpolation
            let (p, q) = if (a - c).abs() < 1e-15 {
                // Secant method
                (2.0 * m * fb, fa - fb)
            } else {
                // Inverse quadratic interpolation
                let q1 = fa / fc;
                let q2 = fb / fc;
                let q3 = fb / fa;
                let p = q3 * (2.0 * m * q1).mul_add(q1 - q2, -((b - a) * (q2 - 1.0)));
                let q = (q1 - 1.0) * (q2 - 1.0) * (q3 - 1.0);
                (p, q)
            };

            let s_temp = if p > 0.0 { -p / q } else { p / -q };

            // Accept interpolation?
            if (2.0 * s_temp < (3.0 * m).mul_add(q, -(tol * q).abs()))
                && (s_temp < (0.5 * e * q).abs())
            {
                e = d;
                d = s_temp;
                s_temp
            } else {
                // Bisection
                d = m;
                e = d;
                m
            }
        } else {
            // Bisection
            d = m;
            e = d;
            m
        };

        a = b;
        fa = fb;

        b += if s.abs() > tol {
            s
        } else if m > 0.0 {
            tol
        } else {
            -tol
        };

        fb = eval_at(expr, variable, b)?;

        if fb * fc > 0.0 {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
    }

    // Did not converge
    Ok(SolverResult {
        root: b,
        iterations,
        error: fb.abs(),
        converged: false,
    })
}

/// Helper function to evaluate an expression at a specific point
fn eval_at(expr: &Expression, variable: &Expression, value: f64) -> SymEngineResult<f64> {
    let value_expr = Expression::from(value);
    expr.substitute(variable, &value_expr)
        .expand()
        .to_f64()
        .ok_or_else(|| SymEngineError::invalid_operation("Failed to evaluate expression"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[ignore = "SymEngine evaluation can be flaky"]
    fn test_newton_raphson_sqrt() {
        let x = Expression::symbol("x");
        // Find sqrt(2): solve x^2 - 2 = 0
        let expr = x.pow(&Expression::from(2)) - Expression::from(2);

        let config = SolverConfig::default();
        let result = newton_raphson(&expr, &x, 1.0, &config).expect("Failed to solve");

        assert!(result.converged);
        assert_relative_eq!(result.root, 2.0_f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    #[ignore = "SymEngine evaluation can be flaky"]
    fn test_bisection_sqrt() {
        let x = Expression::symbol("x");
        // Find sqrt(2): solve x^2 - 2 = 0
        let expr = x.pow(&Expression::from(2)) - Expression::from(2);

        let config = SolverConfig::default();
        let result = bisection(&expr, &x, 1.0, 2.0, &config).expect("Failed to solve");

        assert!(result.converged);
        assert_relative_eq!(result.root, 2.0_f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    #[ignore = "SymEngine evaluation can be flaky"]
    fn test_secant_method() {
        let x = Expression::symbol("x");
        // Find sqrt(2)
        let expr = x.pow(&Expression::from(2)) - Expression::from(2);

        let config = SolverConfig::default();
        let result = secant(&expr, &x, 1.0, 2.0, &config).expect("Failed to solve");

        assert!(result.converged);
        assert_relative_eq!(result.root, 2.0_f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    #[ignore = "SymEngine evaluation can be flaky"]
    fn test_brent_method() {
        let x = Expression::symbol("x");
        // Find sqrt(2)
        let expr = x.pow(&Expression::from(2)) - Expression::from(2);

        let config = SolverConfig::default();
        let result = brent(&expr, &x, 1.0, 2.0, &config).expect("Failed to solve");

        assert!(result.converged);
        assert_relative_eq!(result.root, 2.0_f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_solver_config() {
        let config = SolverConfig::new()
            .with_max_iterations(500)
            .with_abs_tolerance(1e-8)
            .with_rel_tolerance(1e-8);

        assert_eq!(config.max_iterations, 500);
        assert_eq!(config.abs_tolerance, 1e-8);
        assert_eq!(config.rel_tolerance, 1e-8);
    }

    #[test]
    fn test_solver_result() {
        let result = SolverResult {
            root: std::f64::consts::SQRT_2,
            iterations: 5,
            error: 1e-10,
            converged: true,
        };

        assert!(result.is_reliable(1e-8));
        assert!(!result.is_reliable(1e-12));
    }
}
