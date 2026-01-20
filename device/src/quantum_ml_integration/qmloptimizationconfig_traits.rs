//! # QMLOptimizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `QMLOptimizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

// Import types from sibling modules
use super::types::*;
// Merged into super::types
// Merged into super::types

impl Default for QMLOptimizationConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::Adam,
            optimizer_params: [
                ("beta1".to_string(), 0.9),
                ("beta2".to_string(), 0.999),
                ("epsilon".to_string(), 1e-8),
            ]
            .iter()
            .cloned()
            .collect(),
            enable_parameter_sharing: false,
            circuit_optimization: CircuitOptimizationConfig {
                enable_gate_fusion: true,
                enable_compression: true,
                max_depth: None,
                allowed_gates: None,
                topology_aware: true,
            },
            hardware_aware: true,
            multi_objective: MultiObjectiveConfig {
                enabled: false,
                objective_weights: HashMap::new(),
                pareto_exploration: false,
                constraint_handling: ConstraintHandling::Penalty,
            },
        }
    }
}
