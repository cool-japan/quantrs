//! # QuantumGravitySimulator - calculate_holographic_energy_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;
use std::f64::consts::PI;

use super::types::HolographicDuality;

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Calculate holographic energy
    pub(super) fn calculate_holographic_energy(&self, duality: &HolographicDuality) -> Result<f64> {
        let temperature = duality.bulk_geometry.temperature;
        let central_charge = duality.boundary_theory.central_charge;
        if temperature > 0.0 {
            Ok(PI * central_charge * temperature.powi(4) / 120.0)
        } else {
            Ok(central_charge * 0.1)
        }
    }
}
