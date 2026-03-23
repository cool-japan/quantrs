//! # QuantumNeRF - compute_quantum_spherical_harmonic_group Methods
//!
//! This module contains method implementations for `QuantumNeRF`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex32, Complex64};
use std::f64::consts::PI;

use super::quantumnerf_type::QuantumNeRF;

impl QuantumNeRF {
    /// Compute quantum spherical harmonic
    pub(super) fn compute_quantum_spherical_harmonic(
        &self,
        l: usize,
        m: i32,
        theta: f64,
        phi: f64,
    ) -> Result<Complex64> {
        let associated_legendre =
            self.compute_associated_legendre(l, m.abs() as usize, theta.cos());
        let normalization = self.compute_spherical_harmonic_normalization(l, m.abs() as usize);
        let phase = Complex64::from_polar(1.0, m as f64 * phi);
        let quantum_enhancement = 1.0 + self.config.quantum_enhancement_level * 0.1;
        Ok(normalization * associated_legendre * phase * quantum_enhancement)
    }
    /// Compute associated Legendre polynomial (simplified)
    pub(super) fn compute_associated_legendre(&self, l: usize, m: usize, x: f64) -> f64 {
        match (l, m) {
            (0, 0) => 1.0,
            (1, 0) => x,
            (1, 1) => -(1.0 - x * x).sqrt(),
            (2, 0) => 0.5 * (3.0 * x * x - 1.0),
            (2, 1) => -3.0 * x * (1.0 - x * x).sqrt(),
            (2, 2) => 3.0 * (1.0 - x * x),
            _ => 1.0,
        }
    }
    /// Compute spherical harmonic normalization
    pub(super) fn compute_spherical_harmonic_normalization(&self, l: usize, m: usize) -> f64 {
        let factorial_ratio =
            (1..=l - m).product::<usize>() as f64 / (1..=l + m).product::<usize>() as f64;
        ((2.0 * l as f64 + 1.0) * factorial_ratio / (4.0 * PI)).sqrt()
    }
}
