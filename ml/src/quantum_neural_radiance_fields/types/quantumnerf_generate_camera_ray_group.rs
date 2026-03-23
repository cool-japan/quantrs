//! # QuantumNeRF - generate_camera_ray_group Methods
//!
//! This module contains method implementations for `QuantumNeRF`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{CameraMatrix, Ray};

use super::quantumnerf_type::QuantumNeRF;

impl QuantumNeRF {
    /// Generate camera ray for pixel
    pub(super) fn generate_camera_ray(
        &self,
        camera: &CameraMatrix,
        pixel_x: usize,
        pixel_y: usize,
        image_width: usize,
        image_height: usize,
        fov: f64,
    ) -> Result<Ray> {
        let aspect_ratio = image_width as f64 / image_height as f64;
        let ndc_x = (2.0 * pixel_x as f64 / image_width as f64 - 1.0) * aspect_ratio;
        let ndc_y = 1.0 - 2.0 * pixel_y as f64 / image_height as f64;
        let tan_half_fov = (fov / 2.0).tan();
        let camera_x = ndc_x * tan_half_fov;
        let camera_y = ndc_y * tan_half_fov;
        let ray_direction = &camera.forward + camera_x * &camera.right + camera_y * &camera.up;
        let ray_direction = &ray_direction / ray_direction.dot(&ray_direction).sqrt();
        Ok(Ray {
            origin: camera.position.clone(),
            direction: ray_direction,
            near: 0.1,
            far: 10.0,
        })
    }
}
