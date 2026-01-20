//! # AlertManager - Trait Implementations
//!
//! This module contains trait implementations for `AlertManager`.
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

impl std::fmt::Debug for AlertManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlertManager")
            .field("config", &self.config)
            .field("active_alerts", &self.active_alerts)
            .field("alert_history", &self.alert_history)
            .finish()
    }
}
