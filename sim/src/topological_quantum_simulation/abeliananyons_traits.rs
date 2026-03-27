//! # AbelianAnyons - Trait Implementations
//!
//! This module contains trait implementations for `AbelianAnyons`.
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
use super::types::{AbelianAnyons, AnyonType};

impl Default for AbelianAnyons {
    fn default() -> Self {
        Self::new()
    }
}

impl AnyonModelImplementation for AbelianAnyons {
    fn get_anyon_types(&self) -> Vec<AnyonType> {
        self.anyon_types.clone()
    }
    fn fusion_coefficients(&self, a: &AnyonType, b: &AnyonType, c: &AnyonType) -> Complex64 {
        if a.topological_charge + b.topological_charge == c.topological_charge {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }
    fn braiding_matrix(&self, a: &AnyonType, b: &AnyonType) -> Array2<Complex64> {
        let phase = a.r_matrix * b.r_matrix.conj();
        Array2::from_shape_vec((1, 1), vec![phase])
            .expect("AbelianAnyons::braiding_matrix: 1x1 matrix shape is always valid")
    }
    fn f_matrix(
        &self,
        _a: &AnyonType,
        _b: &AnyonType,
        _c: &AnyonType,
        _d: &AnyonType,
    ) -> Array2<Complex64> {
        Array2::eye(1)
    }
    fn is_abelian(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "Abelian Anyons"
    }
}
