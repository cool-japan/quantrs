//! # EnhancedBenchmarkConfig - Trait Implementations
//!
//! This module contains trait implementations for `EnhancedBenchmarkConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;

use super::types::{
    AnalysisMethod, BenchmarkConfig, BenchmarkSuite, EnhancedBenchmarkConfig, LayerFidelity,
    PerformanceMetric, ReportingOptions,
};

impl Default for EnhancedBenchmarkConfig {
    fn default() -> Self {
        Self {
            base_config: BenchmarkConfig::default(),
            enable_ml_prediction: true,
            enable_significance_testing: true,
            enable_comparative_analysis: true,
            enable_realtime_monitoring: true,
            enable_adaptive_protocols: true,
            enable_visual_analytics: true,
            benchmark_suites: vec![
                BenchmarkSuite::QuantumVolume,
                BenchmarkSuite::RandomizedBenchmarking,
                BenchmarkSuite::CrossEntropyBenchmarking,
                BenchmarkSuite::LayerFidelity,
            ],
            performance_metrics: vec![
                PerformanceMetric::GateFidelity,
                PerformanceMetric::CircuitDepth,
                PerformanceMetric::ExecutionTime,
                PerformanceMetric::ErrorRate,
            ],
            analysis_methods: vec![
                AnalysisMethod::StatisticalTesting,
                AnalysisMethod::RegressionAnalysis,
                AnalysisMethod::TimeSeriesAnalysis,
            ],
            reporting_options: ReportingOptions::default(),
        }
    }
}
