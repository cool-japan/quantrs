//! # Visualizationconfig - Trait Implementations
//!
//! This module contains trait implementations for `Visualizationconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::Duration;

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            enable_plotting: false,
            plot_types: vec![PlotType::ConvergencePlot],
            update_frequency: Duration::from_secs(1),
            export_format: ExportFormat::PNG,
        }
    }
}
