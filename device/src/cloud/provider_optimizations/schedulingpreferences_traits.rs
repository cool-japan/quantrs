//! # SchedulingPreferences - Trait Implementations
//!
//! This module contains trait implementations for `SchedulingPreferences`.
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

impl Default for SchedulingPreferences {
    fn default() -> Self {
        Self {
            preferred_time_slots: Vec::new(),
            deadline_flexibility: 0.5,
            priority_level: SchedulingPriority::Normal,
            preemption_policy: PreemptionPolicy::None,
        }
    }
}
