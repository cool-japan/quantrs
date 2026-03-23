//! # QuantumAdvantageMetrics - Trait Implementations
//!
//! This module contains trait implementations for `QuantumAdvantageMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::QuantumAdvantageMetrics;

impl Default for QuantumAdvantageMetrics {
    fn default() -> Self {
        Self {
            theoretical_speedup: 1.0,
            practical_advantage: 1.0,
            complexity_class: "NP".to_string(),
            quantum_resource_requirements: 0,
            classical_resource_requirements: 0,
        }
    }
}
