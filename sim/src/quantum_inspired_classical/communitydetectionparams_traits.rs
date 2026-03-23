//! # CommunityDetectionParams - Trait Implementations
//!
//! This module contains trait implementations for `CommunityDetectionParams`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::CommunityDetectionParams;

impl Default for CommunityDetectionParams {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            num_iterations: 100,
            modularity_threshold: 0.01,
        }
    }
}
