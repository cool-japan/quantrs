//! # AnalysisOptions - Trait Implementations
//!
//! This module contains trait implementations for `AnalysisOptions`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{AnalysisOptions, TypeCheckingLevel};

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            type_checking: TypeCheckingLevel::Strict,
            data_flow_analysis: true,
            control_flow_analysis: true,
            dead_code_elimination: true,
            constant_propagation: true,
            loop_optimization: true,
        }
    }
}
