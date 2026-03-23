//! # QuantumGravitySimulator - measure_cdt_geometry_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::types::{GeometryMeasurements, SimplicialComplex, TopologyMeasurements};

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Measure CDT geometry properties
    pub(super) fn measure_cdt_geometry(
        &self,
        complex: &SimplicialComplex,
    ) -> Result<GeometryMeasurements> {
        let area_spectrum: Vec<f64> = complex
            .time_slices
            .iter()
            .map(|slice| slice.spatial_volume.powf(2.0 / 3.0))
            .collect();
        let volume_spectrum: Vec<f64> = complex
            .time_slices
            .iter()
            .map(|slice| slice.spatial_volume)
            .collect();
        let length_spectrum: Vec<f64> = complex
            .simplices
            .iter()
            .map(|_| self.config.planck_length)
            .collect();
        let discrete_curvature: f64 = complex
            .time_slices
            .iter()
            .map(|slice| slice.curvature)
            .sum::<f64>()
            / complex.time_slices.len() as f64;
        let topology_measurements = TopologyMeasurements {
            euler_characteristic: self.calculate_euler_characteristic(complex)?,
            betti_numbers: vec![1, 0, 0, 1],
            homology_groups: vec![
                "Z".to_string(),
                "0".to_string(),
                "0".to_string(),
                "Z".to_string(),
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
    /// Calculate Euler characteristic of simplicial complex
    pub(super) fn calculate_euler_characteristic(
        &self,
        complex: &SimplicialComplex,
    ) -> Result<i32> {
        let vertices = complex.vertices.len() as i32;
        let edges = complex
            .simplices
            .iter()
            .map(|s| s.vertices.len() * (s.vertices.len() - 1) / 2)
            .sum::<usize>() as i32;
        let faces = complex.simplices.len() as i32;
        Ok(vertices - edges + faces)
    }
}
