//! # Quantumresourceconfig - Trait Implementations
//!
//! This module contains trait implementations for `Quantumresourceconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::Duration;

impl Default for QuantumResourceConfig {
    fn default() -> Self {
        Self {
            max_qubits: 1000,
            max_circuit_depth: 10000,
            max_execution_time: Duration::from_secs(3600),
            allocation_strategy: ResourceAllocationStrategy::Balanced,
        }
    }
}
