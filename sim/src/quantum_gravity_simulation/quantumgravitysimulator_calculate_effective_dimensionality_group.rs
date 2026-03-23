//! # QuantumGravitySimulator - calculate_effective_dimensionality_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::RGTrajectory;

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Calculate effective dimensionality from RG flow
    pub(super) fn calculate_effective_dimensionality(
        &self,
        trajectory: &RGTrajectory,
    ) -> Result<f64> {
        if let Some(newton_evolution) = trajectory.coupling_evolution.get("newton_constant") {
            let initial_g = newton_evolution.first().copied().unwrap_or(1.0);
            let final_g = newton_evolution.last().copied().unwrap_or(1.0);
            if final_g > 0.0 && initial_g > 0.0 {
                let dimension = 2.0f64.mul_add((final_g / initial_g).ln(), 4.0);
                Ok(dimension.clamp(2.0, 6.0))
            } else {
                Ok(4.0)
            }
        } else {
            Ok(4.0)
        }
    }
}
