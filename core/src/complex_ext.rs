//! Extended complex number operations for quantum computing
//!
//! This module provides enhanced complex number operations by leveraging
//! SciRS2's type conversion and complex number utilities.

use scirs2_core::Complex64;

/// Extension trait for Complex64 that adds quantum-specific operations
pub trait QuantumComplexExt {
    /// Calculate the probability (squared magnitude) of a quantum amplitude
    fn probability(&self) -> f64;

    /// Normalize the complex number to have unit magnitude
    fn normalize(&self) -> Complex64;

    /// Check if two complex numbers are approximately equal within tolerance
    fn approx_eq(&self, other: &Complex64, tolerance: f64) -> bool;

    /// Calculate the fidelity between two quantum amplitudes
    fn fidelity(&self, other: &Complex64) -> f64;
}

impl QuantumComplexExt for Complex64 {
    fn probability(&self) -> f64 {
        self.norm_sqr()
    }

    fn normalize(&self) -> Complex64 {
        let mag = self.norm();
        if mag > 0.0 {
            self / mag
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    fn approx_eq(&self, other: &Complex64, tolerance: f64) -> bool {
        (self - other).norm() < tolerance
    }

    fn fidelity(&self, other: &Complex64) -> f64 {
        (self.conj() * other).norm()
    }
}

/// Helper functions for creating common quantum states
pub mod quantum_states {
    use super::*;

    /// Create a complex number representing the |0⟩ state amplitude
    pub fn zero_state() -> Complex64 {
        Complex64::new(1.0, 0.0)
    }

    /// Create a complex number representing the |1⟩ state amplitude
    pub fn one_state() -> Complex64 {
        Complex64::new(0.0, 0.0)
    }

    /// Create a complex number representing the |+⟩ state amplitude component
    pub fn plus_state_component() -> Complex64 {
        Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0)
    }

    /// Create a complex number representing the |-⟩ state amplitude component
    pub fn minus_state_component() -> Complex64 {
        Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0)
    }

    /// Create a phase factor e^(i*theta)
    pub fn phase_factor(theta: f64) -> Complex64 {
        Complex64::new(theta.cos(), theta.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability() {
        let c = Complex64::new(0.6, 0.8);
        assert!((c.probability() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let c = Complex64::new(3.0, 4.0);
        let normalized = c.normalize();
        assert!((normalized.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_approx_eq() {
        let c1 = Complex64::new(1.0, 0.0);
        let c2 = Complex64::new(1.0000001, 0.0);
        assert!(c1.approx_eq(&c2, 1e-6));
        assert!(!c1.approx_eq(&c2, 1e-8));
    }

    #[test]
    fn test_phase_factor() {
        use std::f64::consts::PI;
        let phase = quantum_states::phase_factor(PI / 2.0);
        assert!(phase.re.abs() < 1e-10);
        assert!((phase.im - 1.0).abs() < 1e-10);
    }
}
