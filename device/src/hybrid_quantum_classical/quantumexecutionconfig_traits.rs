//! # Quantumexecutionconfig - Trait Implementations
//!
//! This module contains trait implementations for `Quantumexecutionconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for QuantumExecutionConfig {
    fn default() -> Self {
        Self {
            strategy: QuantumExecutionStrategy::AdaptiveBackend,
            backend_selection: BackendSelectionConfig::default(),
            circuit_optimization: CircuitOptimizationConfig::default(),
            error_mitigation: QuantumErrorMitigationConfig::default(),
            resource_management: QuantumResourceConfig::default(),
        }
    }
}
