//! # Classicalparallelconfig - Trait Implementations
//!
//! This module contains trait implementations for `Classicalparallelconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for ClassicalParallelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: ParallelizationStrategy::DataParallel,
            work_distribution: WorkDistributionAlgorithm::WorkStealing,
            load_balancing: LoadBalancingConfig::default(),
        }
    }
}
