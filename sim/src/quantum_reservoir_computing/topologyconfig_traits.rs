//! # TopologyConfig - Trait Implementations
//!
//! This module contains trait implementations for `TopologyConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::TopologyConfig;

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            connectivity_density: 0.1,
            clustering_coefficient: 0.3,
            rewiring_probability: 0.1,
            power_law_exponent: 2.5,
            hierarchical_levels: 3,
            modularity_strength: 0.5,
            num_modules: 4,
            enable_adaptive: false,
            adaptation_rate: 0.01,
            min_connection_strength: 0.1,
            max_connection_strength: 2.0,
            pruning_threshold: 0.05,
        }
    }
}
