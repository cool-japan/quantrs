//! # QMLDataPipeline - Trait Implementations
//!
//! This module contains trait implementations for `QMLDataPipeline`.
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

impl std::fmt::Debug for QMLDataPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QMLDataPipeline")
            .field("data_cache", &self.data_cache)
            .field("config", &self.config)
            .finish()
    }
}
