//! # GraphConfig - Trait Implementations
//!
//! This module contains trait implementations for `GraphConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{CommunityDetectionParams, GraphAlgorithm, GraphConfig, QuantumWalkParams};

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            algorithm_type: GraphAlgorithm::QuantumInspiredRandomWalk,
            num_vertices: 100,
            connectivity: 0.1,
            walk_params: QuantumWalkParams::default(),
            community_params: CommunityDetectionParams::default(),
        }
    }
}
