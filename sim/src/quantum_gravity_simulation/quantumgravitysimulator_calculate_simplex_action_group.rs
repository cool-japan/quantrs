//! # QuantumGravitySimulator - calculate_simplex_action_group Methods
//!
//! This module contains method implementations for `QuantumGravitySimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;
use std::f64::consts::PI;

use super::types::{SimplexType, SpacetimeVertex};

use super::quantumgravitysimulator_type::QuantumGravitySimulator;

impl QuantumGravitySimulator {
    /// Calculate Einstein-Hilbert action for a simplex
    pub(super) fn calculate_simplex_action(
        &self,
        vertices: &[SpacetimeVertex],
        simplex_vertices: &[usize],
        _simplex_type: SimplexType,
    ) -> Result<f64> {
        let volume = self.calculate_simplex_volume(vertices, simplex_vertices)?;
        let curvature = self.calculate_simplex_curvature(vertices, simplex_vertices)?;
        let einstein_hilbert_term =
            volume * curvature / (16.0 * PI * self.config.gravitational_constant);
        let cosmological_term = self.config.cosmological_constant * volume;
        Ok(einstein_hilbert_term + cosmological_term)
    }
    /// Calculate volume of a simplex
    pub(super) fn calculate_simplex_volume(
        &self,
        vertices: &[SpacetimeVertex],
        simplex_vertices: &[usize],
    ) -> Result<f64> {
        if simplex_vertices.len() < 2 {
            return Ok(0.0);
        }
        let mut volume = 1.0;
        for i in 1..simplex_vertices.len() {
            let v1 = &vertices[simplex_vertices[0]];
            let v2 = &vertices[simplex_vertices[i]];
            let distance_sq: f64 = v1
                .coordinates
                .iter()
                .zip(&v2.coordinates)
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            volume *= distance_sq.sqrt();
        }
        Ok(volume
            * self
                .config
                .planck_length
                .powi(self.config.spatial_dimensions as i32 + 1))
    }
    /// Calculate discrete curvature of a simplex
    pub(super) fn calculate_simplex_curvature(
        &self,
        vertices: &[SpacetimeVertex],
        simplex_vertices: &[usize],
    ) -> Result<f64> {
        if simplex_vertices.len() < 3 {
            return Ok(0.0);
        }
        let mut curvature = 0.0;
        let num_vertices = simplex_vertices.len();
        for i in 0..num_vertices {
            for j in (i + 1)..num_vertices {
                for k in (j + 1)..num_vertices {
                    let v1 = &vertices[simplex_vertices[i]];
                    let v2 = &vertices[simplex_vertices[j]];
                    let v3 = &vertices[simplex_vertices[k]];
                    let angle = self.calculate_angle(v1, v2, v3)?;
                    curvature += (PI - angle) / (PI * self.config.planck_length.powi(2));
                }
            }
        }
        Ok(curvature)
    }
}
