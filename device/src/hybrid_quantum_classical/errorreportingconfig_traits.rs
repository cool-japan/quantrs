//! # Errorreportingconfig - Trait Implementations
//!
//! This module contains trait implementations for `Errorreportingconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for ErrorReportingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level: ErrorReportingLevel::Error,
            channels: vec![ErrorReportingChannel::Log],
            include_diagnostics: true,
        }
    }
}
