//! # SchedulingOptimizationSettings - Trait Implementations
//!
//! This module contains trait implementations for `SchedulingOptimizationSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::traits::ProviderOptimizer;
use super::types::*;
use crate::prelude::CloudProvider;
use crate::DeviceResult;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

impl Default for SchedulingOptimizationSettings {
    fn default() -> Self {
        Self {
            queue_optimization: true,
            batch_optimization: true,
            deadline_awareness: true,
            cost_aware_scheduling: true,
            load_balancing: true,
        }
    }
}
