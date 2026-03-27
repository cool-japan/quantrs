//! # QuantumGravitySimulator - calculate_ads_volume_group Methods
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
    /// Calculate `AdS` volume
    pub(super) fn calculate_ads_volume(&self, duality: &HolographicDuality) -> Result<f64> {
        let ads_radius = duality.bulk_geometry.ads_radius;
        let dimension = 5;
        let _half_dim = f64::from(dimension) / 2.0;
        let gamma_approx = if dimension % 2 == 0 {
            (1..=(dimension / 2)).map(f64::from).product::<f64>()
        } else {
            let n = dimension / 2;
            PI.sqrt() * (1..=(2 * n)).map(f64::from).product::<f64>()
                / (4.0_f64.powi(n) * (1..=n).map(f64::from).product::<f64>())
        };
        Ok(PI.powi(dimension / 2) * ads_radius.powi(dimension) / gamma_approx)
    }
}
