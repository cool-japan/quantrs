//! # QuantumGravitySimulator - calculate_total_volume_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::SpinNetwork;

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Calculate total volume from spin network
    pub(super) fn calculate_total_volume(&self, spin_network: &SpinNetwork) -> Result<f64> {
        let mut total_volume = 0.0;
        for node in &spin_network.nodes {
            let j_sum: f64 = node.quantum_numbers.iter().sum();
            let volume_eigenvalue = self.config.planck_length.powi(3) * j_sum.sqrt();
            total_volume += volume_eigenvalue;
        }
        Ok(total_volume)
    }
}
