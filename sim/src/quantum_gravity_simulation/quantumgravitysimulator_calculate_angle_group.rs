//! # QuantumGravitySimulator - calculate_angle_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::SpacetimeVertex;

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Calculate angle between three vertices
    pub(super) fn calculate_angle(
        &self,
        v1: &SpacetimeVertex,
        v2: &SpacetimeVertex,
        v3: &SpacetimeVertex,
    ) -> Result<f64> {
        let vec1: Vec<f64> = v1
            .coordinates
            .iter()
            .zip(&v2.coordinates)
            .map(|(a, b)| a - b)
            .collect();
        let vec2: Vec<f64> = v3
            .coordinates
            .iter()
            .zip(&v2.coordinates)
            .map(|(a, b)| a - b)
            .collect();
        let dot_product: f64 = vec1.iter().zip(&vec2).map(|(a, b)| a * b).sum();
        let norm1: f64 = vec1.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm2: f64 = vec2.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        if norm1 * norm2 == 0.0 {
            return Ok(0.0);
        }
        let cos_angle = dot_product / (norm1 * norm2);
        Ok(cos_angle.acos())
    }
}
