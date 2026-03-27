//! # GridSettings - Trait Implementations
//!
//! This module contains trait implementations for `GridSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::GridSettings;

impl Default for GridSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            size: 20,
            color: "#e0e0e0".to_string(),
            snap: true,
        }
    }
}
