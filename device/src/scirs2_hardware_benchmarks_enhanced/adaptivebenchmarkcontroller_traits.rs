//! # AdaptiveBenchmarkController - Trait Implementations
//!
//! This module contains trait implementations for `AdaptiveBenchmarkController`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use std::sync::{Arc, Mutex};

use super::types::{AdaptationEngine, AdaptiveBenchmarkController};

impl Default for AdaptiveBenchmarkController {
    fn default() -> Self {
        Self {
            adaptation_engine: Arc::new(AdaptationEngine::new()),
        }
    }
}
