//! # PerformanceProfiler - compare_profiles_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;
use std::collections::{BTreeMap, HashMap};

use super::types::{
    ComparisonReport, MemoryComparison, MemoryMetrics, Profile, QualityComparison, QualityMetrics,
    TimeComparison, TimeMetrics,
};

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Compare profiles
    pub fn compare_profiles(&self, profile1: &Profile, profile2: &Profile) -> ComparisonReport {
        ComparisonReport {
            time_comparison: Self::compare_time_metrics(
                &profile1.metrics.time_metrics,
                &profile2.metrics.time_metrics,
            ),
            memory_comparison: Self::compare_memory_metrics(
                &profile1.metrics.memory_metrics,
                &profile2.metrics.memory_metrics,
            ),
            quality_comparison: Self::compare_quality_metrics(
                &profile1.metrics.quality_metrics,
                &profile2.metrics.quality_metrics,
            ),
            regressions: Vec::new(),
            improvements: Vec::new(),
        }
    }
    /// Compare time metrics
    fn compare_time_metrics(m1: &TimeMetrics, m2: &TimeMetrics) -> TimeComparison {
        TimeComparison {
            total_time_diff: m2.total_time.as_secs_f64() - m1.total_time.as_secs_f64(),
            total_time_ratio: m2.total_time.as_secs_f64() / m1.total_time.as_secs_f64(),
            qubo_time_diff: m2.qubo_generation_time.as_secs_f64()
                - m1.qubo_generation_time.as_secs_f64(),
            solving_time_diff: m2.solving_time.as_secs_f64() - m1.solving_time.as_secs_f64(),
            function_diffs: BTreeMap::new(),
        }
    }
    /// Compare memory metrics
    fn compare_memory_metrics(m1: &MemoryMetrics, m2: &MemoryMetrics) -> MemoryComparison {
        MemoryComparison {
            peak_memory_diff: m2.peak_memory as i64 - m1.peak_memory as i64,
            peak_memory_ratio: m2.peak_memory as f64 / m1.peak_memory as f64,
            avg_memory_diff: m2.avg_memory as i64 - m1.avg_memory as i64,
            allocations_diff: m2.allocations as i64 - m1.allocations as i64,
        }
    }
    /// Compare quality metrics
    fn compare_quality_metrics(m1: &QualityMetrics, m2: &QualityMetrics) -> QualityComparison {
        QualityComparison {
            convergence_rate_diff: m2.convergence_rate - m1.convergence_rate,
            time_to_best_diff: m2.time_to_best_solution.as_secs_f64()
                - m1.time_to_best_solution.as_secs_f64(),
            final_quality_diff: 0.0,
        }
    }
}
