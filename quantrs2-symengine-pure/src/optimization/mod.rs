//! VQE/QAOA optimization support.
//!
//! This module provides specialized support for variational quantum
//! algorithm optimization.

use std::collections::HashMap;

use crate::error::{SymEngineError, SymEngineResult};
use crate::expr::Expression;

/// Compute the gradient of an expression at specific parameter values.
///
/// This is optimized for VQE optimization loops where we need to
/// compute gradients at many different parameter values.
pub fn gradient_at(
    expr: &Expression,
    params: &[Expression],
    values: &HashMap<String, f64>,
) -> SymEngineResult<Vec<f64>> {
    let gradient = expr.gradient(params);
    gradient.iter().map(|g| g.eval(values)).collect()
}

/// Compute the Hessian matrix at specific parameter values.
pub fn hessian_at(
    expr: &Expression,
    params: &[Expression],
    values: &HashMap<String, f64>,
) -> SymEngineResult<Vec<Vec<f64>>> {
    let hessian = expr.hessian(params);
    hessian
        .iter()
        .map(|row| row.iter().map(|h| h.eval(values)).collect())
        .collect()
}

/// Parameter-shift rule for quantum gradient estimation.
///
/// For a parametric gate U(θ), the gradient can be computed as:
/// ∂⟨H⟩/∂θ = (⟨H⟩(θ+s) - ⟨H⟩(θ-s)) / (2 sin(s))
///
/// where s is typically π/2.
pub struct ParameterShiftRule {
    /// Shift amount (default: π/2)
    pub shift: f64,
}

impl ParameterShiftRule {
    /// Create a new parameter-shift rule with default shift (π/2)
    #[must_use]
    pub const fn new() -> Self {
        Self {
            shift: std::f64::consts::FRAC_PI_2,
        }
    }

    /// Create with custom shift
    #[must_use]
    pub const fn with_shift(shift: f64) -> Self {
        Self { shift }
    }

    /// Compute gradient using parameter-shift rule
    ///
    /// # Arguments
    /// * `energy_fn` - Function that computes energy expectation value
    /// * `params` - Current parameter values
    ///
    /// # Returns
    /// Vector of gradient components
    pub fn compute_gradient<F>(&self, energy_fn: F, params: &[f64]) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = params.len();
        let mut gradient = vec![0.0; n];
        let s = self.shift;
        let denominator = 2.0 * s.sin();

        for i in 0..n {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();

            params_plus[i] += s;
            params_minus[i] -= s;

            let e_plus = energy_fn(&params_plus);
            let e_minus = energy_fn(&params_minus);

            gradient[i] = (e_plus - e_minus) / denominator;
        }

        gradient
    }
}

impl Default for ParameterShiftRule {
    fn default() -> Self {
        Self::new()
    }
}

/// VQE optimizer state
pub struct VqeOptimizer {
    /// Energy expression (symbolic Hamiltonian expectation value)
    pub energy: Expression,
    /// Parameters
    pub params: Vec<Expression>,
    /// Learning rate
    pub learning_rate: f64,
}

impl VqeOptimizer {
    /// Create a new VQE optimizer
    #[allow(clippy::missing_const_for_fn)] // Expression contains non-const internals
    pub fn new(energy: Expression, params: Vec<Expression>, learning_rate: f64) -> Self {
        Self {
            energy,
            params,
            learning_rate,
        }
    }

    /// Compute gradient at current parameter values
    pub fn compute_gradient(&self, values: &HashMap<String, f64>) -> SymEngineResult<Vec<f64>> {
        gradient_at(&self.energy, &self.params, values)
    }

    /// Perform one optimization step
    pub fn step(&self, values: &mut HashMap<String, f64>) -> SymEngineResult<f64> {
        let gradient = self.compute_gradient(values)?;

        // Update parameters: θ_new = θ_old - lr * ∇E
        for (param, grad) in self.params.iter().zip(gradient.iter()) {
            if let Some(name) = param.as_symbol() {
                if let Some(value) = values.get_mut(name) {
                    *value -= self.learning_rate * grad;
                }
            }
        }

        // Return new energy
        self.energy.eval(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::trig;

    #[test]
    fn test_gradient_at() {
        // E(θ) = θ^2
        let theta = Expression::symbol("theta");
        let energy = theta.clone() * theta.clone();
        let params = vec![theta];

        let mut values = HashMap::new();
        values.insert("theta".to_string(), 3.0);

        let grad = gradient_at(&energy, &params, &values).expect("should compute gradient");

        // ∂(θ²)/∂θ = 2θ, at θ=3 should be 6
        assert!((grad[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_parameter_shift_rule() {
        let psr = ParameterShiftRule::new();

        // Test with f(θ) = sin(θ), where f'(θ) = cos(θ)
        let gradient = psr.compute_gradient(
            |params| params[0].sin(),
            &[0.0], // θ = 0
        );

        // cos(0) = 1
        assert!((gradient[0] - 1.0).abs() < 1e-6);
    }
}
