//! # EnhancedResourceEstimator - estimate_pattern_impact_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::parallel_ops_stubs::*;

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Estimate pattern resource impact
    pub(super) fn estimate_pattern_impact(pattern_name: &str) -> f64 {
        match pattern_name {
            "QFT" => 2.5,
            "Grover" => 1.8,
            "QAOA" => 2.0,
            "VQE" => 2.2,
            "Entanglement" => 1.5,
            _ => 1.0,
        }
    }
}
