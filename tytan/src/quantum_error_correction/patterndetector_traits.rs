//! # PatternDetector - Trait Implementations
//!
//! This module contains trait implementations for `PatternDetector`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;
use scirs2_core::random::prelude::*;

use super::types::PatternDetector;

impl Clone for PatternDetector {
    fn clone(&self) -> Self {
        Self {
            patterns: self.patterns.clone(),
            pattern_frequency: self.pattern_frequency.clone(),
            prediction_model: None,
        }
    }
}
