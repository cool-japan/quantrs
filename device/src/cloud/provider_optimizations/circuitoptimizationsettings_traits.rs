//! # CircuitOptimizationSettings - Trait Implementations
//!
//! This module contains trait implementations for `CircuitOptimizationSettings`.
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

impl Default for CircuitOptimizationSettings {
    fn default() -> Self {
        Self {
            gate_fusion: true,
            gate_cancellation: true,
            circuit_compression: true,
            transpilation_level: TranspilationLevel::Intermediate,
            error_mitigation: ErrorMitigationSettings::default(),
        }
    }
}
