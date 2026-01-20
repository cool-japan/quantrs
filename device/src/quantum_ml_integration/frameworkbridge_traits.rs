//! # FrameworkBridge - Trait Implementations
//!
//! This module contains trait implementations for `FrameworkBridge`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

// Import types from sibling modules
use super::types::*;
// Merged into super::types
// Merged into super::types

impl std::fmt::Debug for FrameworkBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameworkBridge")
            .field("framework_type", &self.framework_type)
            .field("conversion_cache", &self.conversion_cache)
            .field("performance_metrics", &self.performance_metrics)
            .finish()
    }
}
