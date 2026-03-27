//! # TopologicalQuantumSimulator - accessors Methods
//!
//! This module contains method implementations for `TopologicalQuantumSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TopologicalState;

use super::topologicalquantumsimulator_type::TopologicalQuantumSimulator;

impl TopologicalQuantumSimulator {
    /// Get current topological state
    #[must_use]
    pub const fn get_state(&self) -> &TopologicalState {
        &self.state
    }
}
