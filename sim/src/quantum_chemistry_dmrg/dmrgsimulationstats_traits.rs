//! # DMRGSimulationStats - Trait Implementations
//!
//! This module contains trait implementations for `DMRGSimulationStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{AccuracyMetrics, DMRGSimulationStats};

impl Default for DMRGSimulationStats {
    fn default() -> Self {
        Self {
            total_calculations: 0,
            average_convergence_time: 0.0,
            success_rate: 0.0,
            memory_efficiency: 0.0,
            computational_efficiency: 0.0,
            accuracy_metrics: AccuracyMetrics::default(),
        }
    }
}
