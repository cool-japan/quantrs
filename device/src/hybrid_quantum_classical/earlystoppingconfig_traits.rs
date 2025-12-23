//! # EarlyStoppingConfig - Trait Implementations
//!
//! This module contains trait implementations for `EarlyStoppingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            patience: 50,
            min_improvement: 1e-8,
            restoration_strategy: RestorationStrategy::BestSoFar,
        }
    }
}
