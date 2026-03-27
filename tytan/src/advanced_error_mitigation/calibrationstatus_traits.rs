//! # CalibrationStatus - Trait Implementations
//!
//! This module contains trait implementations for `CalibrationStatus`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, SystemTime};

use super::types::{CalibrationOverallStatus, CalibrationStatus};

impl Default for CalibrationStatus {
    fn default() -> Self {
        Self {
            overall_status: CalibrationOverallStatus::Good,
            individual_calibrations: HashMap::new(),
            last_full_calibration: SystemTime::now(),
            next_scheduled_calibration: SystemTime::now(),
            drift_indicators: HashMap::new(),
        }
    }
}
