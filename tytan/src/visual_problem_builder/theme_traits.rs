//! # Theme - Trait Implementations
//!
//! This module contains trait implementations for `Theme`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{GridSettings, Theme};

impl Default for Theme {
    fn default() -> Self {
        Self {
            primary_color: "#007acc".to_string(),
            secondary_color: "#ffa500".to_string(),
            background_color: "#ffffff".to_string(),
            text_color: "#000000".to_string(),
            grid: GridSettings::default(),
        }
    }
}
