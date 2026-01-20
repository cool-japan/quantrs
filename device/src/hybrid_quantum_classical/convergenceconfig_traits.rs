//! # Convergenceconfig - Trait Implementations
//!
//! This module contains trait implementations for `Convergenceconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            criteria: vec![
                ConvergenceCriterion::ValueTolerance(1e-6),
                ConvergenceCriterion::MaxIterations(1000),
            ],
            early_stopping: EarlyStoppingConfig::default(),
            monitoring: ConvergenceMonitoringConfig::default(),
        }
    }
}
