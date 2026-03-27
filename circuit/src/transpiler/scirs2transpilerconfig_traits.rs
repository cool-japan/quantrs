//! # SciRS2TranspilerConfig - Trait Implementations
//!
//! This module contains trait implementations for `SciRS2TranspilerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, HashSet, VecDeque};

use super::types::SciRS2TranspilerConfig;

impl Default for SciRS2TranspilerConfig {
    fn default() -> Self {
        let mut cost_weights = HashMap::new();
        cost_weights.insert("depth".to_string(), 0.4);
        cost_weights.insert("gates".to_string(), 0.3);
        cost_weights.insert("error".to_string(), 0.3);
        Self {
            enable_parallel_graph_optimization: true,
            buffer_pool_size: 64 * 1024 * 1024,
            chunk_size: 1024,
            enable_connectivity_analysis: true,
            convergence_threshold: 1e-6,
            max_graph_iterations: 100,
            enable_ml_guidance: false,
            cost_weights,
            enable_spectral_analysis: true,
        }
    }
}
