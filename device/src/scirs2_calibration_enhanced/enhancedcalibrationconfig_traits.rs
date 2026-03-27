//! # EnhancedCalibrationConfig - Trait Implementations
//!
//! This module contains trait implementations for `EnhancedCalibrationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;

use super::types::{
    AnalysisOptions, CalibrationConfig, CalibrationObjective, EnhancedCalibrationConfig,
    IdentificationMethod, PerformanceThresholds,
};

impl Default for EnhancedCalibrationConfig {
    fn default() -> Self {
        Self {
            base_config: CalibrationConfig::default(),
            enable_ml_identification: true,
            enable_adaptive_protocols: true,
            enable_drift_tracking: true,
            enable_error_characterization: true,
            enable_auto_recalibration: true,
            enable_visual_reports: true,
            identification_methods: vec![
                IdentificationMethod::ProcessTomography,
                IdentificationMethod::GateSetTomography,
                IdentificationMethod::RandomizedBenchmarking,
            ],
            calibration_objectives: vec![
                CalibrationObjective::MaximizeFidelity,
                CalibrationObjective::MinimizeDrift,
                CalibrationObjective::OptimizeSpeed,
            ],
            performance_thresholds: PerformanceThresholds::default(),
            analysis_options: AnalysisOptions::default(),
        }
    }
}
