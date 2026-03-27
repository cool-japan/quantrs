//! # QuantumGravitySimulator - calculate_lqg_ground_state_energy_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::SpinNetwork;

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Calculate LQG ground state energy
    pub(super) fn calculate_lqg_ground_state_energy(
        &self,
        spin_network: &SpinNetwork,
    ) -> Result<f64> {
        let mut energy = 0.0;
        for holonomy in spin_network.holonomies.values() {
            let trace = holonomy.matrix[[0, 0]] + holonomy.matrix[[1, 1]];
            energy += -trace.re;
        }
        for node in &spin_network.nodes {
            let curvature_contribution = node
                .quantum_numbers
                .iter()
                .map(|&j| j * (j + 1.0))
                .sum::<f64>();
            energy += curvature_contribution * self.config.planck_length.powi(-2);
        }
        Ok(
            energy * self.config.reduced_planck_constant * self.config.speed_of_light
                / self.config.planck_length,
        )
    }
}
