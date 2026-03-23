//! # CicdConfig - Trait Implementations
//!
//! This module contains trait implementations for `CicdConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CicdConfig, CicdPlatform, Comparison, PerformanceGate};

impl Default for CicdConfig {
    fn default() -> Self {
        Self {
            platform: CicdPlatform::GitHubActions,
            webhook_endpoints: vec![],
            deployment_tracking: true,
            performance_gates: vec![
                PerformanceGate {
                    metric: "response_time_p95".to_string(),
                    threshold: 1000.0,
                    comparison: Comparison::LessThan,
                },
                PerformanceGate {
                    metric: "error_rate".to_string(),
                    threshold: 0.01,
                    comparison: Comparison::LessThan,
                },
            ],
        }
    }
}
