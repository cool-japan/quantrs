//! # TrainingPerformanceMonitor - Trait Implementations
//!
//! This module contains trait implementations for `TrainingPerformanceMonitor`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

// Import types from sibling modules
use super::types::*;
// Merged into super::types
// Merged into super::types

impl std::fmt::Debug for TrainingPerformanceMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingPerformanceMonitor")
            .field("performance_metrics", &self.performance_metrics)
            .field("alert_manager", &self.alert_manager)
            .field("config", &self.config)
            .finish()
    }
}
