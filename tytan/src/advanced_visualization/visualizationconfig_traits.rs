//! # VisualizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `VisualizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

use super::types::{ColorScheme, ExportFormat, RenderingQuality, VisualizationConfig};

impl Default for VisualizationConfig {
    fn default() -> Self {
        let mut color_schemes = HashMap::new();
        color_schemes.insert("default".to_string(), ColorScheme::default());
        color_schemes.insert("high_contrast".to_string(), ColorScheme::high_contrast());
        color_schemes.insert(
            "colorblind_friendly".to_string(),
            ColorScheme::colorblind_friendly(),
        );
        Self {
            interactive_mode: true,
            real_time_updates: true,
            enable_3d_rendering: true,
            quantum_state_viz: true,
            performance_dashboard: true,
            update_frequency: Duration::from_millis(100),
            max_data_points: 10000,
            export_formats: vec![
                ExportFormat::PNG,
                ExportFormat::SVG,
                ExportFormat::HTML,
                ExportFormat::JSON,
            ],
            rendering_quality: RenderingQuality::High,
            color_schemes,
        }
    }
}
