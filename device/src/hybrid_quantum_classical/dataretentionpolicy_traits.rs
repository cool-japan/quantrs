//! # Dataretentionpolicy - Trait Implementations
//!
//! This module contains trait implementations for `Dataretentionpolicy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::Duration;

impl Default for DataRetentionPolicy {
    fn default() -> Self {
        Self {
            retain_intermediate: true,
            retention_duration: Duration::from_secs(3600),
            cleanup_strategy: CleanupStrategy::TimeBasedCleanup,
        }
    }
}
