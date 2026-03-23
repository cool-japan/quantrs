//! # ExportSettings - Trait Implementations
//!
//! This module contains trait implementations for `ExportSettings`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ExportFormat, ExportSettings};
use std::time::Duration;

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            enable_export: false,
            export_formats: vec![ExportFormat::JSON],
            export_destinations: vec![],
            export_frequency: Duration::from_secs(3600),
            compression_enabled: true,
        }
    }
}
