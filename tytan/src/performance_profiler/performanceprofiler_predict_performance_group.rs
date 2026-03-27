//! # PerformanceProfiler - predict_performance_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;

use super::types::{PerformancePrediction, PerformancePredictor, ProblemCharacteristics};

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Predict performance characteristics
    pub fn predict_performance(
        &self,
        problem_characteristics: &ProblemCharacteristics,
    ) -> PerformancePrediction {
        let predictor = PerformancePredictor::new(&self.profiles);
        predictor.predict(problem_characteristics)
    }
}
