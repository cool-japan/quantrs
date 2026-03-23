//! # QuantumNeRF - setup_camera_matrix_group Methods
//!
//! This module contains method implementations for `QuantumNeRF`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::CameraMatrix;

use super::quantumnerf_type::QuantumNeRF;

impl QuantumNeRF {
    /// Setup camera matrix
    pub(super) fn setup_camera_matrix(
        &self,
        position: &Array1<f64>,
        direction: &Array1<f64>,
        up: &Array1<f64>,
        fov: f64,
    ) -> Result<CameraMatrix> {
        let forward = direction / direction.dot(direction).sqrt();
        let right = Self::cross_product(&forward, up);
        let right = &right / right.dot(&right).sqrt();
        let up_corrected = Self::cross_product(&right, &forward);
        Ok(CameraMatrix {
            position: position.clone(),
            forward,
            right,
            up: up_corrected,
            fov,
        })
    }
    /// Cross product helper
    pub(super) fn cross_product(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        Array1::from_vec(vec![
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])
    }
}
