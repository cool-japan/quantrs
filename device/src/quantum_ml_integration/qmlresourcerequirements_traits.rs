//! # QMLResourceRequirements - Trait Implementations
//!
//! This module contains trait implementations for `QMLResourceRequirements`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

// Import types from sibling modules
use super::types::*;
// Merged into super::types
// Merged into super::types

impl Default for QMLResourceRequirements {
    fn default() -> Self {
        Self {
            quantum_resources: QuantumResourceRequirements {
                qubits_needed: 4,
                circuits_per_epoch: 100,
                required_fidelity: 0.95,
                required_coherence: Duration::from_micros(100),
                preferred_backend: None,
            },
            classical_resources: ClassicalResourceRequirements {
                cpu_cores: 4,
                memory_mb: 8192,
                gpu_requirements: None,
                storage_mb: 1024,
                network_bandwidth: None,
            },
            time_constraints: TimeConstraints {
                max_training_time: Some(Duration::from_secs(3600)),
                deadline: None,
                priority_scheduling: false,
            },
            cost_constraints: CostConstraints {
                max_cost: Some(100.0),
                cost_per_hour_limit: Some(10.0),
                budget_allocation: None,
            },
        }
    }
}
