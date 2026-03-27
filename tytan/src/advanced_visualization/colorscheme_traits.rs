//! # ColorScheme - Trait Implementations
//!
//! This module contains trait implementations for `ColorScheme`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ColorScheme;

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            primary: "#1f77b4".to_string(),
            secondary: "#ff7f0e".to_string(),
            accent: "#2ca02c".to_string(),
            background: "#ffffff".to_string(),
            text: "#000000".to_string(),
            grid: "#cccccc".to_string(),
            energy_high: "#d62728".to_string(),
            energy_low: "#2ca02c".to_string(),
            convergence: "#1f77b4".to_string(),
            divergence: "#d62728".to_string(),
        }
    }
}
