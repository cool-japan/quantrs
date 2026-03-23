//! # ResourceManagementConfig - Trait Implementations
//!
//! This module contains trait implementations for `ResourceManagementConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ResourceAllocationStrategy, ResourceManagementConfig};

impl Default for ResourceManagementConfig {
    fn default() -> Self {
        Self {
            max_overhead_ratio: 0.3,
            allocation_strategy: ResourceAllocationStrategy::Adaptive,
            enable_dynamic_adjustment: true,
            enforce_constraints: true,
            performance_weight: 0.7,
        }
    }
}
