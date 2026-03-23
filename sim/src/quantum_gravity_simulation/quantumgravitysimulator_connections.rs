//! # QuantumGravitySimulator - connections Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::SpacetimeVertex;

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Check if two vertices are causally connected
    pub(super) fn is_causally_connected(
        &self,
        v1: &SpacetimeVertex,
        v2: &SpacetimeVertex,
    ) -> Result<bool> {
        let time_diff = v2.time - v1.time;
        if time_diff <= 0.0 {
            return Ok(false);
        }
        let spatial_distance_sq: f64 = v1.coordinates[1..]
            .iter()
            .zip(&v2.coordinates[1..])
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let spatial_distance = spatial_distance_sq.sqrt();
        let light_travel_time = spatial_distance / self.config.speed_of_light;
        Ok(time_diff >= light_travel_time)
    }
}
