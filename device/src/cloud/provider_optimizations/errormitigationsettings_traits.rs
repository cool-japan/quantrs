//! # ErrorMitigationSettings - Trait Implementations
//!
//! This module contains trait implementations for `ErrorMitigationSettings`.
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

impl Default for ErrorMitigationSettings {
    fn default() -> Self {
        Self {
            zero_noise_extrapolation: false,
            readout_error_mitigation: true,
            gate_error_mitigation: false,
            decoherence_mitigation: false,
            crosstalk_mitigation: false,
        }
    }
}
