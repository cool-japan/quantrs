//! # ParafermionAnyons - Trait Implementations
//!
//! This module contains trait implementations for `ParafermionAnyons`.
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
use super::types::{AnyonType, ParafermionAnyons};

impl Default for ParafermionAnyons {
    fn default() -> Self {
        Self::new()
    }
}

impl AnyonModelImplementation for ParafermionAnyons {
    fn get_anyon_types(&self) -> Vec<AnyonType> {
        self.anyon_types.clone()
    }
    fn fusion_coefficients(&self, _a: &AnyonType, _b: &AnyonType, _c: &AnyonType) -> Complex64 {
        Complex64::new(1.0, 0.0)
    }
    fn braiding_matrix(&self, _a: &AnyonType, _b: &AnyonType) -> Array2<Complex64> {
        Array2::eye(1)
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
        false
    }
    fn name(&self) -> &'static str {
        "Parafermion Anyons"
    }
}
