//! # Loadbalancingconfig - Trait Implementations
//!
//! This module contains trait implementations for `Loadbalancingconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::Duration;

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rebalancing_frequency: Duration::from_secs(30),
            load_threshold: 0.8,
            migration_cost_threshold: 0.1,
        }
    }
}
