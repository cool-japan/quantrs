//! # MLFusionPredictor - Trait Implementations
//!
//! This module contains trait implementations for `MLFusionPredictor`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "advanced_math")]
use quantrs2_circuit::prelude::*;

use super::types::MLFusionPredictor;

impl Default for MLFusionPredictor {
    fn default() -> Self {
        Self::new()
    }
}
