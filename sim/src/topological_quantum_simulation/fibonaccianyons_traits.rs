//! # FibonacciAnyons - Trait Implementations
//!
//! This module contains trait implementations for `FibonacciAnyons`.
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
use super::types::{AnyonType, FibonacciAnyons};

impl Default for FibonacciAnyons {
    fn default() -> Self {
        Self::new()
    }
}

impl AnyonModelImplementation for FibonacciAnyons {
    fn get_anyon_types(&self) -> Vec<AnyonType> {
        self.anyon_types.clone()
    }
    fn fusion_coefficients(&self, a: &AnyonType, b: &AnyonType, c: &AnyonType) -> Complex64 {
        match (a.label.as_str(), b.label.as_str(), c.label.as_str()) {
            ("tau", "tau", "vacuum" | "tau") => Complex64::new(1.0, 0.0),
            ("vacuum", _, label) | (_, "vacuum", label) if label == a.label || label == b.label => {
                Complex64::new(1.0, 0.0)
            }
            _ => Complex64::new(0.0, 0.0),
        }
    }
    fn braiding_matrix(&self, a: &AnyonType, b: &AnyonType) -> Array2<Complex64> {
        if a.label == "tau" && b.label == "tau" {
            let phi = f64::midpoint(1.0, 5.0_f64.sqrt());
            let phase = Complex64::new(0.0, 1.0) * (4.0 * PI / 5.0).exp();
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    phase,
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    phase * Complex64::new(-1.0 / phi, 0.0),
                ],
            )
            .expect("FibonacciAnyons::braiding_matrix: 2x2 matrix shape is always valid")
        } else {
            Array2::eye(1)
        }
    }
    fn f_matrix(
        &self,
        _a: &AnyonType,
        _b: &AnyonType,
        _c: &AnyonType,
        _d: &AnyonType,
    ) -> Array2<Complex64> {
        let phi = f64::midpoint(1.0, 5.0_f64.sqrt());
        let inv_phi = 1.0 / phi;
        Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(inv_phi, 0.0),
                Complex64::new(inv_phi.sqrt(), 0.0),
                Complex64::new(inv_phi.sqrt(), 0.0),
                Complex64::new(-inv_phi, 0.0),
            ],
        )
        .expect("FibonacciAnyons::f_matrix: 2x2 matrix shape is always valid")
    }
    fn is_abelian(&self) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "Fibonacci Anyons"
    }
}
