//! # TopologicalQuantumSimulator - calculate_topological_invariants_group Methods
//!
//! This module contains method implementations for `TopologicalQuantumSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use std::f64::consts::PI;

use super::types::{AnyonModel, TopologicalInvariants};

use super::topologicalquantumsimulator_type::TopologicalQuantumSimulator;

impl TopologicalQuantumSimulator {
    /// Calculate topological invariants
    pub fn calculate_topological_invariants(&mut self) -> Result<TopologicalInvariants> {
        let mut invariants = TopologicalInvariants::default();
        invariants.chern_number = self.calculate_chern_number()?;
        invariants.winding_number = self.calculate_winding_number()?;
        invariants.z2_invariant = self.calculate_z2_invariant()?;
        invariants.berry_phase = self.calculate_berry_phase()?;
        invariants.hall_conductivity = f64::from(invariants.chern_number) * 2.0 * PI / 137.0;
        invariants.topological_entanglement_entropy =
            self.calculate_topological_entanglement_entropy()?;
        self.state.topological_invariants = invariants.clone();
        Ok(invariants)
    }
    /// Calculate Chern number
    pub(super) fn calculate_chern_number(&self) -> Result<i32> {
        let magnetic_flux = self.config.magnetic_field * self.lattice.sites.len() as f64;
        let flux_quanta = (magnetic_flux / (2.0 * PI)).round() as i32;
        Ok(flux_quanta)
    }
    /// Calculate winding number
    pub(super) fn calculate_winding_number(&self) -> Result<i32> {
        match self.config.dimensions.len() {
            1 => Ok(1),
            _ => Ok(0),
        }
    }
    /// Calculate Z2 invariant
    pub(super) fn calculate_z2_invariant(&self) -> Result<bool> {
        let time_reversal_broken = self.config.magnetic_field.abs() > 1e-10;
        Ok(!time_reversal_broken)
    }
    /// Calculate topological entanglement entropy
    pub(super) fn calculate_topological_entanglement_entropy(&self) -> Result<f64> {
        let total_quantum_dimension: f64 = self
            .anyon_model
            .get_anyon_types()
            .iter()
            .map(|anyon| anyon.quantum_dimension * anyon.quantum_dimension)
            .sum();
        let gamma = match self.config.anyon_model {
            AnyonModel::Abelian => 0.0,
            AnyonModel::Fibonacci => 0.5 * (5.0_f64.sqrt() - 1.0) / 2.0,
            AnyonModel::Ising => 0.5,
            _ => 0.5,
        };
        Ok(-gamma * total_quantum_dimension.ln())
    }
}
