//! # HardwareOptimizationSettings - Trait Implementations
//!
//! This module contains trait implementations for `HardwareOptimizationSettings`.
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

impl Default for HardwareOptimizationSettings {
    fn default() -> Self {
        Self {
            qubit_mapping: QubitMappingStrategy::NoiseAdaptive,
            routing_optimization: RoutingOptimizationStrategy::FidelityAware,
            calibration_optimization: CalibrationOptimizationStrategy::Dynamic,
            noise_adaptation: NoiseAdaptationStrategy::Statistical,
        }
    }
}
