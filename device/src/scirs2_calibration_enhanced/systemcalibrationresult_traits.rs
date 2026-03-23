//! # SystemCalibrationResult - Trait Implementations
//!
//! This module contains trait implementations for `SystemCalibrationResult`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;
use std::fmt;

use super::types::SystemCalibrationResult;

impl fmt::Display for SystemCalibrationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "System Calibration Result:")?;
        writeln!(f, "  Device: {}", self.report.device_name)?;
        writeln!(
            f,
            "  Average fidelity: {:.4}",
            self.report.summary.average_fidelity
        )?;
        writeln!(
            f,
            "  Worst-case fidelity: {:.4}",
            self.report.summary.worst_case_fidelity
        )?;
        writeln!(f, "  Calibration time: {:?}", self.calibration_time)?;
        writeln!(
            f,
            "  Quality score: {:.2}%",
            self.quality_metrics.overall_quality * 100.0
        )?;
        writeln!(f, "  Recommendations: {}", self.recommendations.len())?;
        Ok(())
    }
}
