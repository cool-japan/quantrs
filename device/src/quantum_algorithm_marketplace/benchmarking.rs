//! Benchmarking Configuration Types

use serde::{Deserialize, Serialize};

/// Benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkingConfig {
    /// Benchmark metrics configuration
    pub metrics_config: BenchmarkMetricsConfig,
    /// Hardware testing configuration
    pub hardware_testing: HardwareTestingConfig,
    /// Algorithm comparison configuration
    pub comparison_config: AlgorithmComparisonConfig,
    /// Continuous benchmarking configuration
    pub continuous_benchmarking: ContinuousBenchmarkingConfig,
}

/// Benchmark metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetricsConfig {
    pub metrics: Vec<String>,
    pub aggregation_methods: Vec<String>,
}

/// Hardware testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareTestingConfig {
    pub test_backends: Vec<String>,
    pub test_duration: u64,
}

/// Algorithm comparison configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmComparisonConfig {
    pub comparison_metrics: Vec<String>,
    pub significance_threshold: f64,
}

/// Continuous benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousBenchmarkingConfig {
    pub enable_continuous: bool,
    pub benchmark_frequency: u64,
}
