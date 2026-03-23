//! # ProtocolAdaptationStrategy - Trait Implementations
//!
//! This module contains trait implementations for `ProtocolAdaptationStrategy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    AdaptationAlgorithm, AdaptationTrigger, ProtocolAdaptationStrategy, RollbackStrategy,
};

impl Default for ProtocolAdaptationStrategy {
    fn default() -> Self {
        Self {
            adaptation_algorithm: AdaptationAlgorithm::ThresholdBased,
            learning_rate: 0.01,
            history_window: 100,
            adaptation_triggers: vec![
                AdaptationTrigger::ErrorRateThreshold(0.05),
                AdaptationTrigger::PerformanceDegradation(0.2),
            ],
            rollback_strategy: RollbackStrategy::RevertToPrevious,
        }
    }
}
