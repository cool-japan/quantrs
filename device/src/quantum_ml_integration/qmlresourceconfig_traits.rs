//! # QMLResourceConfig - Trait Implementations
//!
//! This module contains trait implementations for `QMLResourceConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

// Import types from merged types module
use super::types::*;

impl Default for QMLResourceConfig {
    fn default() -> Self {
        Self {
            max_circuits_per_step: 1000,
            memory_limit_mb: 8192,
            parallel_config: ParallelExecutionConfig {
                enable_parallel_circuits: true,
                max_workers: 4,
                batch_processing: BatchProcessingConfig {
                    dynamic_batch_size: true,
                    min_batch_size: 8,
                    max_batch_size: 128,
                    adaptation_strategy: BatchAdaptationStrategy::Performance,
                },
                load_balancing: LoadBalancingStrategy::Performance,
            },
            caching_strategy: CachingStrategy::LRU,
            resource_priorities: ResourcePriorities {
                weights: [
                    ("quantum".to_string(), 0.4),
                    ("classical".to_string(), 0.3),
                    ("memory".to_string(), 0.2),
                    ("network".to_string(), 0.1),
                ]
                .iter()
                .cloned()
                .collect(),
                dynamic_adjustment: true,
                performance_reallocation: true,
            },
        }
    }
}
