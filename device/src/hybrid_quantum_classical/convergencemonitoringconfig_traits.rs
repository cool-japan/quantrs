//! # Convergencemonitoringconfig - Trait Implementations
//!
//! This module contains trait implementations for `Convergencemonitoringconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for ConvergenceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            frequency: MonitoringFrequency::EveryIteration,
            metrics: vec![
                ConvergenceMetric::ObjectiveValue,
                ConvergenceMetric::GradientNorm,
                ConvergenceMetric::ExecutionTime,
            ],
            visualization: VisualizationConfig::default(),
        }
    }
}
