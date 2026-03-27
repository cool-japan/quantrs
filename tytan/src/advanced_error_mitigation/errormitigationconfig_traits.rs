//! # ErrorMitigationConfig - Trait Implementations
//!
//! This module contains trait implementations for `ErrorMitigationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, SystemTime};

use super::types::ErrorMitigationConfig;

impl Default for ErrorMitigationConfig {
    fn default() -> Self {
        Self {
            real_time_monitoring: true,
            adaptive_protocols: true,
            device_calibration: true,
            syndrome_prediction: true,
            qec_integration: true,
            monitoring_interval: Duration::from_millis(100),
            calibration_interval: Duration::from_secs(3600),
            noise_update_threshold: 0.05,
            mitigation_threshold: 0.1,
            history_retention: Duration::from_secs(24 * 3600),
        }
    }
}
