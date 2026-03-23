//! # QuantumGravitySimulator - calculate_cdt_ground_state_energy_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::SimplicialComplex;

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Calculate CDT ground state energy
    pub(super) fn calculate_cdt_ground_state_energy(
        &self,
        complex: &SimplicialComplex,
    ) -> Result<f64> {
        let total_action: f64 = complex.simplices.iter().map(|s| s.action).sum();
        Ok(-total_action)
    }
}
