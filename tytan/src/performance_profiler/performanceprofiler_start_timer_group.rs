//! # PerformanceProfiler - start_timer_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;
use std::time::{Duration, Instant};

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Start timer
    pub fn start_timer(&mut self, name: &str) {
        if !self.config.enabled {
            return;
        }
        if let Some(ref mut context) = self.current_profile {
            context.timers.insert(name.to_string(), Instant::now());
        }
    }
}
