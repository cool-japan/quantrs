//! # UsageMetricsConfig - Trait Implementations
//!
//! This module contains trait implementations for `UsageMetricsConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::UsageMetricsConfig;

impl Default for UsageMetricsConfig {
    fn default() -> Self {
        Self {
            track_computations: true,
            track_resource_utilization: true,
            track_algorithm_usage: true,
            track_success_rates: true,
        }
    }
}
