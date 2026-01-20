//! # Classicalcomputationconfig - Trait Implementations
//!
//! This module contains trait implementations for `Classicalcomputationconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for ClassicalComputationConfig {
    fn default() -> Self {
        Self {
            strategy: ClassicalProcessingStrategy::Parallel,
            resource_allocation: ClassicalResourceConfig::default(),
            caching_config: ClassicalCachingConfig::default(),
            parallel_processing: ClassicalParallelConfig::default(),
            data_management: DataManagementConfig::default(),
        }
    }
}
