//! # TopologicalQuantumSimulator - accessors Methods
//!
//! This module contains method implementations for `TopologicalQuantumSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::BraidingOperation;

use super::topologicalquantumsimulator_type::TopologicalQuantumSimulator;

impl TopologicalQuantumSimulator {
    /// Get braiding history
    #[must_use]
    pub fn get_braiding_history(&self) -> &[BraidingOperation] {
        &self.braiding_history
    }
}
