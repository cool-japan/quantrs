//! # EnhancedHardwareBenchmark - generate_visualizations_group Methods
//!
//! This module contains method implementations for `EnhancedHardwareBenchmark`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::{
    buffer_pool::BufferPool,
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};
use scirs2_core::parallel_ops::*;

use super::types::{BenchmarkVisualizations, ComprehensiveBenchmarkResult, VisualAnalyzer};

use super::enhancedhardwarebenchmark_type::EnhancedHardwareBenchmark;

impl EnhancedHardwareBenchmark {
    pub(super) fn generate_visualizations(
        result: &ComprehensiveBenchmarkResult,
    ) -> QuantRS2Result<BenchmarkVisualizations> {
        VisualAnalyzer::generate_visualizations(result)
    }
}
