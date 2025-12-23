//! # OptimizationSettings - Trait Implementations
//!
//! This module contains trait implementations for `OptimizationSettings`.
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

impl Default for OptimizationSettings {
    fn default() -> Self {
        Self {
            circuit_optimization: CircuitOptimizationSettings::default(),
            hardware_optimization: HardwareOptimizationSettings::default(),
            scheduling_optimization: SchedulingOptimizationSettings::default(),
            cost_optimization: CostOptimizationSettings::default(),
        }
    }
}
