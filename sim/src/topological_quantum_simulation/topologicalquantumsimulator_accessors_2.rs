//! # TopologicalQuantumSimulator - accessors Methods
//!
//! This module contains method implementations for `TopologicalQuantumSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TopologicalSimulationStats;

use super::topologicalquantumsimulator_type::TopologicalQuantumSimulator;

impl TopologicalQuantumSimulator {
    /// Get simulation statistics
    #[must_use]
    pub const fn get_stats(&self) -> &TopologicalSimulationStats {
        &self.stats
    }
}
