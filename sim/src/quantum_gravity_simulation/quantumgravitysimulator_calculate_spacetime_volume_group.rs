//! # QuantumGravitySimulator - calculate_spacetime_volume_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::SimplicialComplex;

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Calculate spacetime volume from CDT
    pub(super) fn calculate_spacetime_volume(&self, complex: &SimplicialComplex) -> Result<f64> {
        let total_volume: f64 = complex.simplices.iter().map(|s| s.volume).sum();
        Ok(total_volume)
    }
    /// Calculate Hausdorff dimension from CDT
    pub(super) fn calculate_hausdorff_dimension(&self, complex: &SimplicialComplex) -> Result<f64> {
        let num_vertices = complex.vertices.len() as f64;
        let typical_length = self.config.planck_length * num_vertices.powf(1.0 / 4.0);
        let volume = self.calculate_spacetime_volume(complex)?;
        if typical_length > 0.0 {
            Ok(volume.log(typical_length))
        } else {
            Ok(4.0)
        }
    }
}
