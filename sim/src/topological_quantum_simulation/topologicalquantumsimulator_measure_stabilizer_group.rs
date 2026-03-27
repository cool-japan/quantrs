//! # TopologicalQuantumSimulator - measure_stabilizer_group Methods
//!
//! This module contains method implementations for `TopologicalQuantumSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};

use super::types::{StabilizerType, SyndromeDetector};

use super::topologicalquantumsimulator_type::TopologicalQuantumSimulator;

impl TopologicalQuantumSimulator {
    /// Measure stabilizer
    pub(super) fn measure_stabilizer(&self, detector: &SyndromeDetector) -> Result<bool> {
        let probability = match detector.stabilizer_type {
            StabilizerType::PauliX => 0.1,
            StabilizerType::PauliZ => 0.1,
            StabilizerType::XZ => 0.05,
        };
        Ok(fastrand::f64() < probability)
    }
}
