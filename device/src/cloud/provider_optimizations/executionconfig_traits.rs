//! # ExecutionConfig - Trait Implementations
//!
//! This module contains trait implementations for `ExecutionConfig`.
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

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            provider: CloudProvider::IBM,
            backend: "ibm_brisbane".to_string(),
            optimization_settings: OptimizationSettings::default(),
            resource_allocation: ResourceAllocation::default(),
            scheduling_preferences: SchedulingPreferences::default(),
        }
    }
}
