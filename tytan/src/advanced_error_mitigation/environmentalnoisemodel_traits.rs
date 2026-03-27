//! # EnvironmentalNoiseModel - Trait Implementations
//!
//! This module contains trait implementations for `EnvironmentalNoiseModel`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::EnvironmentalNoiseModel;

impl Default for EnvironmentalNoiseModel {
    fn default() -> Self {
        Self {
            temperature_noise: 0.0,
            magnetic_noise: 0.0,
            electric_noise: 0.0,
            vibration_sensitivity: 0.0,
            control_line_noise: HashMap::new(),
        }
    }
}
