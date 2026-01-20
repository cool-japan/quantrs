//! # CostOptimizationSettings - Trait Implementations
//!
//! This module contains trait implementations for `CostOptimizationSettings`.
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

impl Default for CostOptimizationSettings {
    fn default() -> Self {
        Self {
            provider_comparison: true,
            spot_instance_usage: false,
            volume_discounts: true,
            off_peak_scheduling: true,
            resource_sharing: false,
        }
    }
}
