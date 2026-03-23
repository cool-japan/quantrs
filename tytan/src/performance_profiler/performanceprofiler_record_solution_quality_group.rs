//! # PerformanceProfiler - record_solution_quality_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Record solution quality
    pub fn record_solution_quality(&mut self, quality: f64) {
        if !self.config.enabled {
            return;
        }
        if let Some(ref mut context) = self.current_profile {
            let elapsed = context.profile.start_time.elapsed();
            context
                .profile
                .metrics
                .quality_metrics
                .quality_timeline
                .push((elapsed, quality));
            if context
                .profile
                .metrics
                .quality_metrics
                .quality_timeline
                .len()
                == 1
            {
                context
                    .profile
                    .metrics
                    .quality_metrics
                    .time_to_first_solution = elapsed;
            }
        }
    }
}
