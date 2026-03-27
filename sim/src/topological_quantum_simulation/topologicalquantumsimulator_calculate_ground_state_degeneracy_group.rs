//! # TopologicalQuantumSimulator - calculate_ground_state_degeneracy_group Methods
//!
//! This module contains method implementations for `TopologicalQuantumSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{AnyonModel, TopologicalConfig, TopologicalLattice};

use super::topologicalquantumsimulator_type::TopologicalQuantumSimulator;

impl TopologicalQuantumSimulator {
    /// Calculate ground state degeneracy
    pub(super) fn calculate_ground_state_degeneracy(
        config: &TopologicalConfig,
        lattice: &TopologicalLattice,
    ) -> usize {
        match config.anyon_model {
            AnyonModel::Abelian => {
                let genus = Self::calculate_genus(lattice);
                2_usize.pow(genus as u32)
            }
            AnyonModel::Fibonacci => {
                let num_qubits = lattice.sites.len() / 2;
                let golden_ratio = f64::midpoint(1.0, 5.0_f64.sqrt());
                (golden_ratio.powi(num_qubits as i32) / 5.0_f64.sqrt()).round() as usize
            }
            AnyonModel::Ising => {
                let num_majoranas = lattice.sites.len();
                2_usize.pow((num_majoranas / 2) as u32)
            }
            _ => 1,
        }
    }
    /// Calculate topological genus
    pub(super) fn calculate_genus(lattice: &TopologicalLattice) -> usize {
        let vertices = lattice.sites.len();
        let edges = lattice.bonds.len();
        let faces = lattice.plaquettes.len() + 1;
        let euler_characteristic = vertices as i32 - edges as i32 + faces as i32;
        let genus = (2 - euler_characteristic) / 2;
        genus.max(0) as usize
    }
}
