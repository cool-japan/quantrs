//! Test results and metrics types.
//!
//! This module defines structures for test results, summaries,
//! performance data, and quality metrics.

use std::time::Duration;

use super::types::{TestFailure, TestResult};

/// Test results container
#[derive(Debug, Clone)]
pub struct TestResults {
    /// Individual test results
    pub test_results: Vec<TestResult>,
    /// Summary statistics
    pub summary: TestSummary,
    /// Failures
    pub failures: Vec<TestFailure>,
    /// Performance data
    pub performance: PerformanceData,
}

/// Test summary statistics
#[derive(Debug, Clone)]
pub struct TestSummary {
    /// Total tests run
    pub total_tests: usize,
    /// Passed tests
    pub passed: usize,
    /// Failed tests
    pub failed: usize,
    /// Skipped tests
    pub skipped: usize,
    /// Average runtime
    pub avg_runtime: Duration,
    /// Success rate
    pub success_rate: f64,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Quality metrics for test results
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Average solution quality
    pub avg_quality: f64,
    /// Best solution quality
    pub best_quality: f64,
    /// Worst solution quality
    pub worst_quality: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Constraint satisfaction rate
    pub constraint_satisfaction_rate: f64,
}

/// Performance data container
#[derive(Debug, Clone)]
pub struct PerformanceData {
    /// Runtime statistics
    pub runtime_stats: RuntimeStats,
    /// Memory statistics
    pub memory_stats: MemoryStats,
    /// Convergence data
    pub convergence_data: ConvergenceData,
}

/// Runtime statistics
#[derive(Debug, Clone)]
pub struct RuntimeStats {
    /// Total runtime
    pub total_time: Duration,
    /// QUBO generation time
    pub qubo_generation_time: Duration,
    /// Solving time
    pub solving_time: Duration,
    /// Validation time
    pub validation_time: Duration,
    /// Time per test
    pub time_per_test: Vec<(String, Duration)>,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage
    pub peak_memory: usize,
    /// Average memory usage
    pub avg_memory: usize,
    /// Memory per test
    pub memory_per_test: Vec<(String, usize)>,
}

/// Convergence tracking data
#[derive(Debug, Clone)]
pub struct ConvergenceData {
    /// Convergence curves
    pub curves: Vec<ConvergenceCurve>,
    /// Average iterations to convergence
    pub avg_iterations: f64,
    /// Convergence rate
    pub convergence_rate: f64,
}

/// Individual convergence curve
#[derive(Debug, Clone)]
pub struct ConvergenceCurve {
    /// Test ID
    pub test_id: String,
    /// Iteration data
    pub iterations: Vec<IterationData>,
    /// Converged
    pub converged: bool,
}

/// Iteration data point
#[derive(Debug, Clone)]
pub struct IterationData {
    /// Iteration number
    pub iteration: usize,
    /// Best objective value
    pub best_value: f64,
    /// Current value
    pub current_value: f64,
    /// Temperature (if applicable)
    pub temperature: Option<f64>,
}

impl Default for TestResults {
    fn default() -> Self {
        Self {
            test_results: Vec::new(),
            summary: TestSummary::default(),
            failures: Vec::new(),
            performance: PerformanceData::default(),
        }
    }
}

impl Default for TestSummary {
    fn default() -> Self {
        Self {
            total_tests: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            avg_runtime: Duration::from_secs(0),
            success_rate: 0.0,
            quality_metrics: QualityMetrics::default(),
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            avg_quality: 0.0,
            best_quality: f64::NEG_INFINITY,
            worst_quality: f64::INFINITY,
            std_dev: 0.0,
            constraint_satisfaction_rate: 0.0,
        }
    }
}

impl Default for PerformanceData {
    fn default() -> Self {
        Self {
            runtime_stats: RuntimeStats::default(),
            memory_stats: MemoryStats::default(),
            convergence_data: ConvergenceData::default(),
        }
    }
}

impl Default for RuntimeStats {
    fn default() -> Self {
        Self {
            total_time: Duration::from_secs(0),
            qubo_generation_time: Duration::from_secs(0),
            solving_time: Duration::from_secs(0),
            validation_time: Duration::from_secs(0),
            time_per_test: Vec::new(),
        }
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            peak_memory: 0,
            avg_memory: 0,
            memory_per_test: Vec::new(),
        }
    }
}

impl Default for ConvergenceData {
    fn default() -> Self {
        Self {
            curves: Vec::new(),
            avg_iterations: 0.0,
            convergence_rate: 0.0,
        }
    }
}
