//! # Feedbackcontrolconfig - Trait Implementations
//!
//! This module contains trait implementations for `Feedbackcontrolconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::Duration;

impl Default for FeedbackControlConfig {
    fn default() -> Self {
        Self {
            enable_realtime_feedback: false,
            target_latency: Duration::from_millis(100),
            control_frequency: 10.0,
            feedback_algorithms: vec![FeedbackAlgorithm::PID],
            adaptive_control: AdaptiveControlConfig::default(),
            state_estimation: StateEstimationConfig::default(),
        }
    }
}
