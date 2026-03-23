//! # QuantumGraphScheduler - Trait Implementations
//!
//! This module contains trait implementations for `QuantumGraphScheduler`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::QuantumGraphScheduler;

impl Default for QuantumGraphScheduler {
    fn default() -> Self {
        Self::new(100)
    }
}
