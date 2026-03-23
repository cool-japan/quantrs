//! # CircuitPatternAnalyzer - Trait Implementations
//!
//! This module contains trait implementations for `CircuitPatternAnalyzer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "advanced_math")]
use quantrs2_circuit::prelude::*;

use super::types::CircuitPatternAnalyzer;

impl Default for CircuitPatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
