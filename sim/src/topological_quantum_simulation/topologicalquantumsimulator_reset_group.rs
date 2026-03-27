//! # TopologicalQuantumSimulator - reset_group Methods
//!
//! This module contains method implementations for `TopologicalQuantumSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};

use super::types::TopologicalSimulationStats;

use super::topologicalquantumsimulator_type::TopologicalQuantumSimulator;

impl TopologicalQuantumSimulator {
    /// Reset simulation
    pub fn reset(&mut self) -> Result<()> {
        self.state = Self::create_initial_topological_state(&self.config, &self.lattice)?;
        self.braiding_history.clear();
        self.stats = TopologicalSimulationStats::default();
        Ok(())
    }
}
