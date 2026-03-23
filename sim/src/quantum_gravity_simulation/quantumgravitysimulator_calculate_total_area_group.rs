//! # QuantumGravitySimulator - calculate_total_area_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;
use std::f64::consts::PI;

use super::types::SpinNetwork;

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Calculate total area from spin network
    pub(super) fn calculate_total_area(&self, spin_network: &SpinNetwork) -> Result<f64> {
        let mut total_area = 0.0;
        for edge in &spin_network.edges {
            let j = edge.spin;
            let area_eigenvalue = (8.0
                * PI
                * self.config.gravitational_constant
                * self.config.reduced_planck_constant
                / self.config.speed_of_light.powi(3))
            .sqrt()
                * (j * (j + 1.0)).sqrt();
            total_area += area_eigenvalue;
        }
        Ok(total_area)
    }
}
