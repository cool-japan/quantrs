//! # ReportingOptions - Trait Implementations
//!
//! This module contains trait implementations for `ReportingOptions`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;

use super::types::{ExportFormat, ReportingOptions};

impl Default for ReportingOptions {
    fn default() -> Self {
        Self {
            detailed_reports: true,
            include_visualizations: true,
            export_format: ExportFormat::JSON,
            enable_dashboard: true,
        }
    }
}
