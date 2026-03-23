//! # EnhancedResourceEstimator - classify_topology_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::parallel_ops_stubs::*;

use super::types::TopologyType;

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Classify topology type
    pub(super) fn classify_topology(_connectivity: &[Vec<usize>], density: f64) -> TopologyType {
        if density < 0.1 {
            TopologyType::Sparse
        } else if density < 0.3 {
            TopologyType::Regular
        } else if density < 0.6 {
            TopologyType::Dense
        } else {
            TopologyType::AllToAll
        }
    }
}
