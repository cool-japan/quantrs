//! # PerformanceProfiler - enter_function_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::thread;
use std::time::{Duration, Instant};

use super::types::{EventType, FunctionGuard, ProfileEvent};

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Record function entry
    pub fn enter_function(&mut self, name: &str) -> FunctionGuard {
        if !self.config.enabled {
            return FunctionGuard {
                profiler: None,
                name: String::new(),
            };
        }
        if let Some(ref mut context) = self.current_profile {
            let event = ProfileEvent {
                timestamp: Instant::now(),
                event_type: EventType::FunctionCall,
                name: name.to_string(),
                duration: None,
                data: HashMap::new(),
                thread_id: thread::current().id(),
            };
            context.profile.events.push(event);
            context.call_stack.push((name.to_string(), Instant::now()));
        }
        FunctionGuard {
            profiler: Some(std::ptr::from_mut::<Self>(self)),
            name: name.to_string(),
        }
    }
}
