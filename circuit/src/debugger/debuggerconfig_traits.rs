//! # DebuggerConfig - Trait Implementations
//!
//! This module contains trait implementations for `DebuggerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::DebuggerConfig;

impl Default for DebuggerConfig {
    fn default() -> Self {
        Self {
            enable_step_mode: true,
            enable_auto_visualization: true,
            enable_profiling: true,
            enable_memory_tracking: true,
            enable_error_detection: true,
            max_history_entries: 1000,
            visualization_frequency: Duration::from_millis(100),
            profiling_sample_rate: 1.0,
            memory_warning_threshold: 0.8,
            gate_timeout: Duration::from_secs(30),
        }
    }
}
