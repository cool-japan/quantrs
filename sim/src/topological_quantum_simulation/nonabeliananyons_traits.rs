//! # NonAbelianAnyons - Trait Implementations
//!
//! This module contains trait implementations for `NonAbelianAnyons`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `AnyonModelImplementation`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, Array3, Array4, Axis};
use scirs2_core::Complex64;

use super::functions::AnyonModelImplementation;
use super::types::{AnyonType, NonAbelianAnyons};

impl Default for NonAbelianAnyons {
    fn default() -> Self {
        Self::new()
    }
}

impl AnyonModelImplementation for NonAbelianAnyons {
    fn get_anyon_types(&self) -> Vec<AnyonType> {
        self.anyon_types.clone()
    }
    fn fusion_coefficients(&self, a: &AnyonType, b: &AnyonType, c: &AnyonType) -> Complex64 {
        if let Some(outcomes) = a.fusion_rules.get(&b.label) {
            if outcomes.contains(&c.label) {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        } else {
            Complex64::new(0.0, 0.0)
        }
    }
    fn braiding_matrix(&self, a: &AnyonType, b: &AnyonType) -> Array2<Complex64> {
        let dim = (a.quantum_dimension * b.quantum_dimension) as usize;
        let mut matrix = Array2::eye(dim);
        if !a.is_abelian || !b.is_abelian {
            let phase = a.r_matrix * b.r_matrix.conj();
            matrix[[0, 0]] = phase;
            if dim > 1 {
                matrix[[1, 1]] = phase.conj();
            }
        }
        matrix
    }
    fn f_matrix(
        &self,
        _a: &AnyonType,
        _b: &AnyonType,
        _c: &AnyonType,
        _d: &AnyonType,
    ) -> Array2<Complex64> {
        Array2::eye(2)
    }
    fn is_abelian(&self) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "Non-Abelian Anyons"
    }
}
