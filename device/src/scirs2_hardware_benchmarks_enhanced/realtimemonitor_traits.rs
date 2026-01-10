//! # RealtimeMonitor - Trait Implementations
//!
//! This module contains trait implementations for `RealtimeMonitor`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use std::sync::{Arc, Mutex};

use super::types::{AlertManager, BenchmarkDashboard, RealtimeMonitor};

impl Default for RealtimeMonitor {
    fn default() -> Self {
        Self {
            dashboard: Arc::new(Mutex::new(BenchmarkDashboard::new())),
            alert_manager: Arc::new(AlertManager::new()),
        }
    }
}
