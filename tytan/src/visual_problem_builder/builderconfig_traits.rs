//! # BuilderConfig - Trait Implementations
//!
//! This module contains trait implementations for `BuilderConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{BuilderConfig, ExportFormat, Theme, VariableType};

impl Default for BuilderConfig {
    fn default() -> Self {
        Self {
            real_time_validation: true,
            auto_save_interval: Some(std::time::Duration::from_secs(30)),
            max_problem_size: 10000,
            default_variable_type: VariableType::Binary,
            supported_formats: vec![
                ExportFormat::Python,
                ExportFormat::Rust,
                ExportFormat::JSON,
                ExportFormat::QUBO,
            ],
            theme: Theme::default(),
        }
    }
}
