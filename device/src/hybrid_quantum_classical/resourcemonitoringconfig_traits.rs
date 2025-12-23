//! # Resourcemonitoringconfig - Trait Implementations
//!
//! This module contains trait implementations for `Resourcemonitoringconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for ResourceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            granularity: MonitoringGranularity::Process,
            metrics: vec![
                ResourceMetric::CPUUsage,
                ResourceMetric::MemoryUsage,
                ResourceMetric::QuantumResourceUsage,
            ],
            alerting: AlertingConfig::default(),
        }
    }
}
