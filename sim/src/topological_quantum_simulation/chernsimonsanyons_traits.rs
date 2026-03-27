//! # ChernSimonsAnyons - Trait Implementations
//!
//! This module contains trait implementations for `ChernSimonsAnyons`.
//!
//! ## Implemented Traits
//!
//! - `AnyonModelImplementation`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, Array3, Array4, Axis};
use scirs2_core::Complex64;

use super::functions::AnyonModelImplementation;
use super::types::{AnyonType, ChernSimonsAnyons};

impl AnyonModelImplementation for ChernSimonsAnyons {
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
        self.level <= 2
    }
    fn name(&self) -> &'static str {
        "Chern-Simons Anyons"
    }
}
