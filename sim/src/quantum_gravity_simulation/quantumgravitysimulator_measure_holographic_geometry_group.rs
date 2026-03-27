//! # QuantumGravitySimulator - measure_holographic_geometry_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::{GeometryMeasurements, HolographicDuality, TopologyMeasurements};

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Measure holographic geometry
    pub(super) fn measure_holographic_geometry(
        &self,
        duality: &HolographicDuality,
    ) -> Result<GeometryMeasurements> {
        let area_spectrum: Vec<f64> = duality
            .entanglement_structure
            .rt_surfaces
            .iter()
            .map(|surface| surface.area)
            .collect();
        let volume_spectrum: Vec<f64> = duality
            .entanglement_structure
            .rt_surfaces
            .iter()
            .map(|surface| surface.boundary_region.volume)
            .collect();
        let length_spectrum: Vec<f64> = (1..=10)
            .map(|n| f64::from(n) * duality.bulk_geometry.ads_radius / 10.0)
            .collect();
        let discrete_curvature = -1.0 / duality.bulk_geometry.ads_radius.powi(2);
        let topology_measurements = TopologyMeasurements {
            euler_characteristic: 0,
            betti_numbers: vec![1, 0, 0, 0, 0],
            homology_groups: vec!["Z".to_string(); 5],
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
