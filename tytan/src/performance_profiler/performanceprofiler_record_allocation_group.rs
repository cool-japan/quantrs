//! # PerformanceProfiler - record_allocation_group Methods
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
    /// Record memory allocation
    pub fn record_allocation(&mut self, size: usize) {
        if !self.config.enabled || !self.config.profile_memory {
            return;
        }
        if let Some(ref mut context) = self.current_profile {
            context.profile.metrics.memory_metrics.allocations += 1;
            context.profile.metrics.memory_metrics.largest_allocation = context
                .profile
                .metrics
                .memory_metrics
                .largest_allocation
                .max(size);
            let event = ProfileEvent {
                timestamp: Instant::now(),
                event_type: EventType::MemoryAlloc,
                name: "allocation".to_string(),
                duration: None,
                data: {
                    let mut data = HashMap::new();
                    data.insert("size".to_string(), size.to_string());
                    data
                },
                thread_id: thread::current().id(),
            };
            context.profile.events.push(event);
        }
    }
}
