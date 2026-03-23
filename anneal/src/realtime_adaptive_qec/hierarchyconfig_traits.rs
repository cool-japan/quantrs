//! # HierarchyConfig - Trait Implementations
//!
//! This module contains trait implementations for `HierarchyConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{HierarchyCommunication, HierarchyConfig};

impl Default for HierarchyConfig {
    fn default() -> Self {
        Self {
            enable_hierarchy: true,
            num_levels: 3,
            level_thresholds: vec![0.01, 0.05, 0.1],
            level_resources: vec![0.1, 0.3, 0.6],
            communication_protocol: HierarchyCommunication::Cascade,
        }
    }
}
