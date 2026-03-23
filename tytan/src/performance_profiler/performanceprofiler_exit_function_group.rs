//! # PerformanceProfiler - exit_function_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::thread;
use std::time::{Duration, Instant};

use super::types::{EventType, ProfileEvent};

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Record function exit
    pub(super) fn exit_function(&mut self, name: &str) {
        if let Some(ref mut context) = self.current_profile {
            if let Some((_, enter_time)) = context.call_stack.pop() {
                let duration = enter_time.elapsed();
                let event = ProfileEvent {
                    timestamp: Instant::now(),
                    event_type: EventType::FunctionReturn,
                    name: name.to_string(),
                    duration: Some(duration),
                    data: HashMap::new(),
                    thread_id: thread::current().id(),
                };
                context.profile.events.push(event);
                *context
                    .profile
                    .metrics
                    .time_metrics
                    .function_times
                    .entry(name.to_string())
                    .or_insert(Duration::from_secs(0)) += duration;
            }
        }
    }
}
