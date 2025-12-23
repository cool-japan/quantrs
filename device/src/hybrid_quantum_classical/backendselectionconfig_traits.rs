//! # Backendselectionconfig - Trait Implementations
//!
//! This module contains trait implementations for `Backendselectionconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for BackendSelectionConfig {
    fn default() -> Self {
        Self {
            criteria: vec![
                SelectionCriterion::Fidelity,
                SelectionCriterion::QueueTime,
                SelectionCriterion::Availability,
            ],
            preferred_backends: vec![],
            fallback_strategy: FallbackStrategy::BestAvailable,
            enable_dynamic_selection: true,
        }
    }
}
