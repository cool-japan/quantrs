//! # PerformanceProfiler - benchmark_compare_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;

use super::types::{BenchmarkComparison, MetricComparison, PerformanceTrend, Profile};

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Benchmark comparison
    pub fn benchmark_compare(&self, profiles: &[Profile]) -> BenchmarkComparison {
        let mut comparison = BenchmarkComparison {
            profiles: profiles.iter().map(|p| p.id.clone()).collect(),
            metrics_comparison: Vec::new(),
            regression_analysis: Vec::new(),
            performance_trends: Vec::new(),
        };
        if profiles.len() < 2 {
            return comparison;
        }
        let times: Vec<f64> = profiles
            .iter()
            .map(|p| p.metrics.time_metrics.total_time.as_secs_f64())
            .collect();
        comparison.metrics_comparison.push(MetricComparison {
            metric_name: "total_time".to_string(),
            values: times.clone(),
            trend: if times.len() >= 2 {
                if times[times.len() - 1] < times[0] {
                    PerformanceTrend::Improving
                } else if times[times.len() - 1] > times[0] * 1.1 {
                    PerformanceTrend::Degrading
                } else {
                    PerformanceTrend::Stable
                }
            } else {
                PerformanceTrend::Unknown
            },
            variance: Self::calculate_variance(&times),
        });
        let memory: Vec<f64> = profiles
            .iter()
            .map(|p| p.metrics.memory_metrics.peak_memory as f64)
            .collect();
        comparison.metrics_comparison.push(MetricComparison {
            metric_name: "peak_memory".to_string(),
            values: memory.clone(),
            trend: if memory.len() >= 2 {
                if memory[memory.len() - 1] < memory[0] {
                    PerformanceTrend::Improving
                } else if memory[memory.len() - 1] > memory[0] * 1.1 {
                    PerformanceTrend::Degrading
                } else {
                    PerformanceTrend::Stable
                }
            } else {
                PerformanceTrend::Unknown
            },
            variance: Self::calculate_variance(&memory),
        });
        comparison
    }
    /// Calculate variance
    fn calculate_variance(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
}
