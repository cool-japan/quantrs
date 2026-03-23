//! # CompilerConfig - Trait Implementations
//!
//! This module contains trait implementations for `CompilerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant};

use super::types::CompilerConfig;

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            aggressive_optimization: false,
            cache_embeddings: true,
            parallel_compilation: true,
            max_compilation_time: Duration::from_secs(300),
            optimization_tolerance: 1e-6,
            seed: None,
        }
    }
}
