//! # QuantumGravitySimulator - calculate_uv_fixed_point_energy_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::RGTrajectory;

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Calculate UV fixed point energy
    pub(super) fn calculate_uv_fixed_point_energy(&self, trajectory: &RGTrajectory) -> Result<f64> {
        let max_energy = trajectory.energy_scales.last().copied().unwrap_or(1.0);
        Ok(max_energy * self.config.reduced_planck_constant * self.config.speed_of_light)
    }
}
