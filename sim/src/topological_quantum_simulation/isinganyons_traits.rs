//! # IsingAnyons - Trait Implementations
//!
//! This module contains trait implementations for `IsingAnyons`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `AnyonModelImplementation`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, Array3, Array4, Axis};
use scirs2_core::Complex64;
use std::f64::consts::PI;

use super::functions::AnyonModelImplementation;
use super::types::{AnyonType, IsingAnyons};

impl Default for IsingAnyons {
    fn default() -> Self {
        Self::new()
    }
}

impl AnyonModelImplementation for IsingAnyons {
    fn get_anyon_types(&self) -> Vec<AnyonType> {
        self.anyon_types.clone()
    }
    fn fusion_coefficients(&self, a: &AnyonType, b: &AnyonType, c: &AnyonType) -> Complex64 {
        match (a.label.as_str(), b.label.as_str(), c.label.as_str()) {
            ("sigma", "sigma", "vacuum" | "psi") => Complex64::new(1.0, 0.0),
            ("psi", "psi", "vacuum") => Complex64::new(1.0, 0.0),
            ("sigma", "psi", "sigma") | ("psi", "sigma", "sigma") => Complex64::new(1.0, 0.0),
            ("vacuum", _, label) | (_, "vacuum", label) if label == a.label || label == b.label => {
                Complex64::new(1.0, 0.0)
            }
            _ => Complex64::new(0.0, 0.0),
        }
    }
    fn braiding_matrix(&self, a: &AnyonType, b: &AnyonType) -> Array2<Complex64> {
        let phase = a.r_matrix * b.r_matrix.conj();
        if a.label == "sigma" && b.label == "sigma" {
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 1.0) * (PI / 8.0).exp(),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, -1.0) * (PI / 8.0).exp(),
                ],
            )
            .expect("IsingAnyons::braiding_matrix: 2x2 matrix shape is always valid")
        } else {
            Array2::from_shape_vec((1, 1), vec![phase])
                .expect("IsingAnyons::braiding_matrix: 1x1 matrix shape is always valid")
        }
    }
    fn f_matrix(
        &self,
        _a: &AnyonType,
        _b: &AnyonType,
        _c: &AnyonType,
        _d: &AnyonType,
    ) -> Array2<Complex64> {
        let sqrt_2_inv = 1.0 / 2.0_f64.sqrt();
        Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(sqrt_2_inv, 0.0),
                Complex64::new(sqrt_2_inv, 0.0),
                Complex64::new(sqrt_2_inv, 0.0),
                Complex64::new(-sqrt_2_inv, 0.0),
            ],
        )
        .expect("IsingAnyons::f_matrix: 2x2 matrix shape is always valid")
    }
    fn is_abelian(&self) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "Ising Anyons"
    }
}
