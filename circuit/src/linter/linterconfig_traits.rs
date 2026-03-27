//! # LinterConfig - Trait Implementations
//!
//! This module contains trait implementations for `LinterConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{LinterConfig, Severity, StyleStrictness};

impl Default for LinterConfig {
    fn default() -> Self {
        Self {
            enable_pattern_detection: true,
            enable_antipattern_detection: true,
            enable_style_checking: true,
            enable_optimization_analysis: true,
            enable_complexity_analysis: true,
            enable_best_practices: true,
            severity_threshold: Severity::Info,
            max_analysis_depth: 1000,
            enable_scirs2_analysis: true,
            pattern_confidence_threshold: 0.8,
            enable_auto_fix: true,
            performance_threshold: 0.1,
            style_strictness: StyleStrictness::Moderate,
        }
    }
}
