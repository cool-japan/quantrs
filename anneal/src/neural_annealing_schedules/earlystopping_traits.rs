//! # EarlyStopping - Trait Implementations
//!
//! This module contains trait implementations for `EarlyStopping`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::EarlyStopping;

impl Default for EarlyStopping {
    fn default() -> Self {
        Self {
            patience: 10,
            min_delta: 1e-4,
            monitor_metric: "validation_loss".to_string(),
            current_patience: 0,
            best_metric: f64::INFINITY,
        }
    }
}
