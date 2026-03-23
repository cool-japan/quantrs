//! # QuantumGravitySimulator - measure_quantum_geometry_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;
use std::f64::consts::PI;

use super::types::{GeometryMeasurements, SpinNetwork, TopologyMeasurements};

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Measure quantum geometry properties
    pub(super) fn measure_quantum_geometry(
        &self,
        spin_network: &SpinNetwork,
    ) -> Result<GeometryMeasurements> {
        let area_spectrum: Vec<f64> = spin_network
            .edges
            .iter()
            .map(|edge| (edge.spin * (edge.spin + 1.0)).sqrt() * self.config.planck_length.powi(2))
            .collect();
        let volume_spectrum: Vec<f64> = spin_network
            .nodes
            .iter()
            .map(|node| {
                node.quantum_numbers.iter().sum::<f64>().sqrt() * self.config.planck_length.powi(3)
            })
            .collect();
        let length_spectrum: Vec<f64> = spin_network.edges.iter().map(|edge| edge.length).collect();
        let discrete_curvature = self.calculate_discrete_curvature(spin_network)?;
        let topology_measurements = TopologyMeasurements {
            euler_characteristic: (spin_network.nodes.len() as i32)
                - (spin_network.edges.len() as i32)
                + 1,
            betti_numbers: vec![1, 0, 0],
            homology_groups: vec!["Z".to_string(), "0".to_string(), "0".to_string()],
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
    /// Calculate discrete curvature from spin network
    pub(super) fn calculate_discrete_curvature(&self, spin_network: &SpinNetwork) -> Result<f64> {
        let mut total_curvature = 0.0;
        for node in &spin_network.nodes {
            let expected_angle = 2.0 * PI;
            let actual_angle: f64 = node
                .quantum_numbers
                .iter()
                .map(|&j| 2.0 * (j * PI / node.valence as f64))
                .sum();
            let curvature = (expected_angle - actual_angle) / self.config.planck_length.powi(2);
            total_curvature += curvature;
        }
        Ok(total_curvature / spin_network.nodes.len() as f64)
    }
}
