//! # PerformanceProfiler - record_deallocation_group Methods
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
    /// Record memory deallocation
    pub fn record_deallocation(&mut self, size: usize) {
        if !self.config.enabled || !self.config.profile_memory {
            return;
        }
        if let Some(ref mut context) = self.current_profile {
            context.profile.metrics.memory_metrics.deallocations += 1;
            let event = ProfileEvent {
                timestamp: Instant::now(),
                event_type: EventType::MemoryFree,
                name: "deallocation".to_string(),
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
