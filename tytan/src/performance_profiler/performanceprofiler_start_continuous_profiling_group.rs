//! # PerformanceProfiler - start_continuous_profiling_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;
use std::time::{Duration, Instant};

use super::types::ContinuousProfiler;

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Continuous profiling mode
    pub fn start_continuous_profiling(
        &mut self,
        duration: Duration,
    ) -> Result<ContinuousProfiler, String> {
        if !self.config.enabled {
            return Err("Profiling not enabled".to_string());
        }
        let profiler = ContinuousProfiler::new(duration, self.config.sampling_interval);
        Ok(profiler)
    }
}
