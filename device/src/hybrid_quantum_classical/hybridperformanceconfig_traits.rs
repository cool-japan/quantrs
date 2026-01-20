//! # Hybridperformanceconfig - Trait Implementations
//!
//! This module contains trait implementations for `Hybridperformanceconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for HybridPerformanceConfig {
    fn default() -> Self {
        Self {
            optimization_targets: vec![PerformanceTarget::BalancedPerformance],
            profiling: ProfilingConfig::default(),
            benchmarking: BenchmarkingConfig::default(),
            resource_monitoring: ResourceMonitoringConfig::default(),
        }
    }
}
