//! # QuantumGravitySimulator - calculate_beta_function_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Calculate beta function for RG flow
    pub(super) fn calculate_beta_function(
        &self,
        coupling: &str,
        value: f64,
        energy_scale: &f64,
    ) -> Result<f64> {
        match coupling {
            "newton_constant" => Ok(2.0f64.mul_add(value, 0.1 * value.powi(2) * energy_scale.ln())),
            "cosmological_constant" => Ok((-2.0f64).mul_add(value, 0.01 * value.powi(2))),
            "r_squared" => Ok((-2.0f64).mul_add(value, 0.001 * value.powi(3))),
            _ => Ok(0.0),
        }
    }
}
