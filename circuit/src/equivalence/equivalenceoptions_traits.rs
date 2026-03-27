//! # EquivalenceOptions - Trait Implementations
//!
//! This module contains trait implementations for `EquivalenceOptions`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::{COMPLEX_TOLERANCE, DEFAULT_TOLERANCE};
use super::types::EquivalenceOptions;

impl Default for EquivalenceOptions {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
            ignore_global_phase: true,
            check_all_states: true,
            max_unitary_qubits: 10,
            enable_adaptive_tolerance: true,
            enable_statistical_analysis: true,
            enable_stability_analysis: true,
            enable_graph_comparison: false,
            confidence_level: 0.95,
            max_condition_number: 1e12,
            scirs2_config: None,
            complex_tolerance: COMPLEX_TOLERANCE,
            enable_parallel_computation: true,
        }
    }
}
