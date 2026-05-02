//! VQE Natural Gradient optimisation entry point.
//!
//! This module re-exports the [`QuantumAutoDiff`] engine from [`crate::autodiff`]
//! and provides a convenience wrapper that wires together the Quantum Fisher
//! Information Matrix (QFIM) computation and the linear-system solve needed by
//! the Quantum Natural Gradient (QNG) optimiser.
//!
//! ## Typical usage
//!
//! ```rust,ignore
//! use quantrs2_ml::vqe_natural_gradient::VQENaturalGradient;
//!
//! // Define the cost function (e.g. VQE energy expectation)
//! let cost_fn = |params: &[f64]| -> f64 {
//!     params[0].cos() + params[1].sin()  // placeholder
//! };
//!
//! let vqe_ng = VQENaturalGradient::new(cost_fn, 1e-3);
//! let params = vec![0.1_f64, 0.2];
//! let gradients = vec![-0.1_f64, 0.05];
//!
//! let nat_grads = vqe_ng.natural_gradients(&params, &gradients).unwrap();
//! ```

pub use crate::autodiff::QuantumAutoDiff;

use crate::error::{MLError, Result};

/// Convenience wrapper for VQE Quantum Natural Gradient computation.
///
/// Holds the cost-function executor and the Tikhonov regularisation
/// coefficient used when inverting the QFIM.
pub struct VQENaturalGradient {
    /// Inner `QuantumAutoDiff` engine carrying the executor and QFIM logic.
    inner: QuantumAutoDiff,
    /// Tikhonov regularisation added to the diagonal of the QFIM.
    regularization: f64,
}

impl VQENaturalGradient {
    /// Create a new VQE natural-gradient optimiser.
    ///
    /// # Arguments
    /// * `executor` — cost function `f(θ) → f64` (e.g. VQE energy estimator).
    /// * `regularization` — Tikhonov coefficient added to the QFIM diagonal
    ///   before inversion to ensure positive-definiteness.
    pub fn new<F>(executor: F, regularization: f64) -> Self
    where
        F: Fn(&[f64]) -> f64 + 'static,
    {
        Self {
            inner: QuantumAutoDiff::new(executor),
            regularization,
        }
    }

    /// Compute the natural gradients for the given parameter vector.
    ///
    /// 1. Compute ordinary (Euclidean) gradients via the parameter-shift rule.
    /// 2. Build the QFIM using the 4-point parameter-shift formula.
    /// 3. Solve `F · ñ = g` for the natural gradient `ñ`.
    ///
    /// # Errors
    /// Returns [`MLError`] if the QFIM is singular or any circuit evaluation fails.
    pub fn natural_gradients(&self, params: &[f64], euclidean_grads: &[f64]) -> Result<Vec<f64>> {
        self.inner
            .natural_gradients(params, euclidean_grads, self.regularization)
    }

    /// Compute parameter-shift gradients (Euclidean) for the current executor.
    ///
    /// # Errors
    /// Returns [`MLError`] on evaluation failure.
    pub fn parameter_shift_gradients(&self, params: &[f64]) -> Result<Vec<f64>> {
        use std::f64::consts::PI;
        self.inner
            .parameter_shift_gradients(params, PI / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_vqe_ng_natural_gradients_identity() {
        // For a diagonal QFIM = I (executor yields constant), nat_grad ≈ eucl_grad
        let executor = |params: &[f64]| -> f64 { params[0].cos() + params[1].sin() };
        let reg = 1e-3;
        let vqe_ng = VQENaturalGradient::new(executor, reg);

        let params = vec![PI / 4.0, PI / 6.0];
        let grads = vec![0.1, 0.2];

        let nat_grads = vqe_ng
            .natural_gradients(&params, &grads)
            .expect("natural_gradients should succeed");

        assert_eq!(nat_grads.len(), 2);
        // Natural gradients should be finite
        for g in &nat_grads {
            assert!(g.is_finite(), "natural gradient is not finite: {g}");
        }
    }

    #[test]
    fn test_vqe_ng_parameter_shift() {
        let executor = |params: &[f64]| -> f64 { params[0].powi(2) };
        let vqe_ng = VQENaturalGradient::new(executor, 1e-3);
        let params = vec![1.0_f64];

        let grads = vqe_ng
            .parameter_shift_gradients(&params)
            .expect("parameter_shift_gradients should succeed");

        assert_eq!(grads.len(), 1);
        assert!(grads[0].is_finite());
    }
}
