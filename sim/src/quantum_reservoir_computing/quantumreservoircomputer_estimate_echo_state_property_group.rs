//! # QuantumReservoirComputer - estimate_echo_state_property_group Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::random::prelude::*;

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Estimate echo state property
    pub(super) fn estimate_echo_state_property(&self) -> Result<f64> {
        let coupling = self.config.coupling_strength;
        let estimated_spectral_radius = coupling.tanh();
        Ok(if estimated_spectral_radius < 1.0 {
            1.0
        } else {
            1.0 / estimated_spectral_radius
        })
    }
}
