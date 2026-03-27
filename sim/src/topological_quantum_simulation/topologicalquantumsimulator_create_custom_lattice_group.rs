//! # TopologicalQuantumSimulator - create_custom_lattice_group Methods
//!
//! This module contains method implementations for `TopologicalQuantumSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};

use super::types::TopologicalLattice;

use super::topologicalquantumsimulator_type::TopologicalQuantumSimulator;

impl TopologicalQuantumSimulator {
    /// Create custom lattice
    pub(super) fn create_custom_lattice(dimensions: &[usize]) -> Result<TopologicalLattice> {
        Self::create_square_lattice(dimensions)
    }
}
