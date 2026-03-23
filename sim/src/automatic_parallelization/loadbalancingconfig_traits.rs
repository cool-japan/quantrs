//! # LoadBalancingConfig - Trait Implementations
//!
//! This module contains trait implementations for `LoadBalancingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant};

use super::types::{LoadBalancingConfig, WorkStealingStrategy};

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_balancing: true,
            work_stealing_strategy: WorkStealingStrategy::Adaptive,
            monitoring_interval: Duration::from_millis(100),
            rebalancing_threshold: 0.2,
        }
    }
}
