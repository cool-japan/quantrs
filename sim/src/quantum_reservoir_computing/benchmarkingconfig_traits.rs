//! # BenchmarkingConfig - Trait Implementations
//!
//! This module contains trait implementations for `BenchmarkingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    BenchmarkDataset, BenchmarkingConfig, ComparisonMethod, CrossValidationStrategy,
    PerformanceMetric, StatisticalTest,
};

impl Default for BenchmarkingConfig {
    fn default() -> Self {
        Self {
            enable_comprehensive: true,
            datasets: vec![
                BenchmarkDataset::MackeyGlass,
                BenchmarkDataset::Lorenz,
                BenchmarkDataset::Sine,
                BenchmarkDataset::Chaotic,
            ],
            metrics: vec![
                PerformanceMetric::MSE,
                PerformanceMetric::MAE,
                PerformanceMetric::R2,
                PerformanceMetric::MemoryCapacity,
            ],
            statistical_tests: vec![
                StatisticalTest::TTest,
                StatisticalTest::WilcoxonRankSum,
                StatisticalTest::KruskalWallis,
            ],
            comparison_methods: vec![
                ComparisonMethod::ESN,
                ComparisonMethod::LSTM,
                ComparisonMethod::GRU,
            ],
            num_runs: 10,
            confidence_level: 0.95,
            enable_cross_validation: true,
            cv_strategy: CrossValidationStrategy::KFold,
        }
    }
}
