//! # QuantumGravitySimulator - measure_as_geometry_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::{GeometryMeasurements, RGTrajectory, TopologyMeasurements};

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Measure Asymptotic Safety geometry
    pub(super) fn measure_as_geometry(
        &self,
        trajectory: &RGTrajectory,
    ) -> Result<GeometryMeasurements> {
        let area_spectrum: Vec<f64> = (1..=10)
            .map(|n| f64::from(n) * self.config.planck_length.powi(2))
            .collect();
        let volume_spectrum: Vec<f64> = (1..=10)
            .map(|n| f64::from(n) * self.config.planck_length.powi(4))
            .collect();
        let length_spectrum: Vec<f64> = (1..=10)
            .map(|n| f64::from(n) * self.config.planck_length)
            .collect();
        let discrete_curvature = if let Some(cosmo_evolution) =
            trajectory.coupling_evolution.get("cosmological_constant")
        {
            cosmo_evolution.last().copied().unwrap_or(0.0) / self.config.planck_length.powi(2)
        } else {
            0.0
        };
        let topology_measurements = TopologyMeasurements {
            euler_characteristic: 0,
            betti_numbers: vec![1, 0, 0, 0],
            homology_groups: vec![
                "Z".to_string(),
                "0".to_string(),
                "0".to_string(),
                "0".to_string(),
            ],
            fundamental_group: "trivial".to_string(),
        };
        Ok(GeometryMeasurements {
            area_spectrum,
            volume_spectrum,
            length_spectrum,
            discrete_curvature,
            topology_measurements,
        })
    }
}
