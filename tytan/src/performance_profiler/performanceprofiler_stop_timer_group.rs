//! # PerformanceProfiler - stop_timer_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;
use std::time::{Duration, Instant};

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Stop timer
    pub fn stop_timer(&mut self, name: &str) -> Option<Duration> {
        if !self.config.enabled {
            return None;
        }
        if let Some(ref mut context) = self.current_profile {
            if let Some(start_time) = context.timers.remove(name) {
                let duration = start_time.elapsed();
                match name {
                    "qubo_generation" => {
                        context.profile.metrics.time_metrics.qubo_generation_time = duration;
                    }
                    "compilation" => {
                        context.profile.metrics.time_metrics.compilation_time = duration;
                    }
                    "solving" => {
                        context.profile.metrics.time_metrics.solving_time = duration;
                    }
                    "post_processing" => {
                        context.profile.metrics.time_metrics.post_processing_time = duration;
                    }
                    _ => {
                        context
                            .metrics_buffer
                            .time_samples
                            .push((name.to_string(), duration));
                    }
                }
                return Some(duration);
            }
        }
        None
    }
}
