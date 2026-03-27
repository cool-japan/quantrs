//! # BenchmarkConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `BenchmarkConfiguration`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{BenchmarkConfiguration, TemperatureProfile};
use crate::applications::unified::SolverType;
#[allow(unused_imports)]
use crate::applications::{ApplicationError, ApplicationResult};

impl Default for BenchmarkConfiguration {
    fn default() -> Self {
        Self {
            benchmark_industries: vec![
                "finance".to_string(),
                "logistics".to_string(),
                "energy".to_string(),
                "manufacturing".to_string(),
                "healthcare".to_string(),
                "telecommunications".to_string(),
            ],
            problem_sizes: vec![5, 10, 20, 50, 100, 200, 500],
            solver_types: vec![SolverType::Classical, SolverType::QuantumSimulator],
            repetitions: 10,
            max_test_time: 300.0,
            enable_memory_profiling: true,
            enable_cpu_profiling: true,
            enable_scalability_analysis: true,
            enable_parallel_execution: true,
            temperature_profiles: vec![
                TemperatureProfile::Linear,
                TemperatureProfile::Exponential,
                TemperatureProfile::Logarithmic,
            ],
            convergence_thresholds: vec![0.01, 0.001, 0.0001],
        }
    }
}
