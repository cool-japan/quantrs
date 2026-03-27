//! # ReportingOptions - Trait Implementations
//!
//! This module contains trait implementations for `ReportingOptions`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ExportFormat, ReportingOptions};

impl Default for ReportingOptions {
    fn default() -> Self {
        Self {
            generate_plots: true,
            include_raw_data: false,
            include_confidence_intervals: true,
            export_format: ExportFormat::JSON,
        }
    }
}
