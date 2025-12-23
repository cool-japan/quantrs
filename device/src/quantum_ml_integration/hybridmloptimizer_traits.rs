//! # HybridMLOptimizer - Trait Implementations
//!
//! This module contains trait implementations for `HybridMLOptimizer`.
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

impl std::fmt::Debug for HybridMLOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridMLOptimizer")
            .field("config", &self.config)
            .field("optimization_history", &self.optimization_history)
            .field("performance_analytics", &self.performance_analytics)
            .finish()
    }
}
