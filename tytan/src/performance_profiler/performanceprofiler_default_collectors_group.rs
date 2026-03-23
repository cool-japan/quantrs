//! # PerformanceProfiler - default_collectors_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;

use super::functions::MetricsCollector;
use super::types::{CPUCollector, MemoryCollector, TimeCollector};

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Get default collectors
    pub(super) fn default_collectors() -> Vec<Box<dyn MetricsCollector>> {
        vec![
            Box::new(TimeCollector),
            Box::new(MemoryCollector),
            Box::new(CPUCollector),
        ]
    }
}
