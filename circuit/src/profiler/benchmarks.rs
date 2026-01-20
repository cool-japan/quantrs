//! Benchmarking engine and regression detection
//!
//! This module provides comprehensive benchmarking capabilities including
//! benchmark suite management, regression detection, baseline management,
//! and performance trend analysis.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

// Import types from sibling modules
use super::analyzers::*;
use super::metrics::*;

pub struct BenchmarkEngine {
    /// Benchmark suites
    pub benchmark_suites: HashMap<String, BenchmarkSuite>,
    /// Benchmark results
    pub benchmark_results: HashMap<String, BenchmarkResult>,
    /// Comparison results
    pub comparison_results: ComparisonResults,
    /// Benchmark configuration
    pub config: BenchmarkConfig,
}

/// Benchmark suite definition
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    /// Suite name
    pub name: String,
    /// Benchmark tests
    pub tests: Vec<BenchmarkTest>,
    /// Suite configuration
    pub config: BenchmarkSuiteConfig,
    /// Suite metadata
    pub metadata: HashMap<String, String>,
}

/// Individual benchmark test
#[derive(Debug, Clone)]
pub struct BenchmarkTest {
    /// Test name
    pub name: String,
    /// Test type
    pub test_type: BenchmarkTestType,
    /// Test parameters
    pub parameters: HashMap<String, f64>,
    /// Expected performance range
    pub expected_range: (f64, f64),
}

/// Types of benchmark tests
#[derive(Debug, Clone)]
pub enum BenchmarkTestType {
    /// Performance test
    Performance,
    /// Stress test
    Stress,
    /// Load test
    Load,
    /// Endurance test
    Endurance,
    /// Accuracy test
    Accuracy,
}

/// Benchmark suite configuration
#[derive(Debug, Clone)]
pub struct BenchmarkSuiteConfig {
    /// Number of iterations
    pub iterations: usize,
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Timeout per test
    pub test_timeout: Duration,
    /// Statistical confidence level
    pub confidence_level: f64,
}

/// Benchmark execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Result timestamp
    pub timestamp: SystemTime,
    /// Suite name
    pub suite_name: String,
    /// Test results
    pub test_results: HashMap<String, TestResult>,
    /// Overall score
    pub overall_score: f64,
    /// Execution duration
    pub execution_duration: Duration,
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test name
    pub test_name: String,
    /// Performance score
    pub score: f64,
    /// Execution time
    pub execution_time: Duration,
    /// Pass/fail status
    pub passed: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Test metadata
    pub metadata: HashMap<String, String>,
}

/// Comparison results between benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResults {
    /// Baseline benchmark
    pub baseline: String,
    /// Comparison benchmarks
    pub comparisons: HashMap<String, ComparisonSummary>,
    /// Statistical significance tests
    pub significance_tests: HashMap<String, f64>,
    /// Performance regression analysis
    pub regression_analysis: RegressionAnalysisResults,
}

/// Summary of benchmark comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    /// Performance improvement
    pub improvement: f64,
    /// Statistical significance
    pub significance: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Effect size
    pub effect_size: f64,
}

/// Regression analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysisResults {
    /// Detected regressions
    pub regressions: Vec<PerformanceRegression>,
    /// Regression severity
    pub severity_summary: HashMap<RegressionSeverity, usize>,
    /// Trend analysis
    pub trend_analysis: TrendAnalysisResults,
}

/// Performance regression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    /// Test name
    pub test_name: String,
    /// Regression magnitude
    pub magnitude: f64,
    /// Regression severity
    pub severity: RegressionSeverity,
    /// Statistical confidence
    pub confidence: f64,
    /// Probable cause
    pub probable_cause: Option<String>,
}

/// Regression severity levels
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    /// Minor regression
    Minor,
    /// Moderate regression
    Moderate,
    /// Major regression
    Major,
    /// Critical regression
    Critical,
}

/// Trend analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisResults {
    /// Performance trends
    pub trends: HashMap<String, TrendDirection>,
    /// Trend strengths
    pub trend_strengths: HashMap<String, f64>,
    /// Forecast confidence
    pub forecast_confidence: HashMap<String, f64>,
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Default iterations
    pub default_iterations: usize,
    /// Default timeout
    pub default_timeout: Duration,
    /// Enable statistical analysis
    pub enable_statistical_analysis: bool,
    /// Comparison baseline
    pub comparison_baseline: Option<String>,
    /// Auto-regression detection
    pub auto_regression_detection: bool,
}

/// Regression detector for performance monitoring
#[derive(Debug, Clone)]
pub struct RegressionDetector {
    /// Detection algorithms
    pub algorithms: HashMap<String, RegressionDetectionAlgorithm>,
    /// Detected regressions
    pub detected_regressions: Vec<PerformanceRegression>,
    /// Detection configuration
    pub config: RegressionDetectionConfig,
    /// Baseline management
    pub baseline_manager: BaselineManager,
}

/// Regression detection algorithm
#[derive(Debug, Clone)]
pub struct RegressionDetectionAlgorithm {
    /// Algorithm type
    pub algorithm_type: RegressionAlgorithmType,
    /// Algorithm parameters
    pub parameters: HashMap<String, f64>,
    /// Detection sensitivity
    pub sensitivity: f64,
    /// False positive rate
    pub false_positive_rate: f64,
}

/// Types of regression detection algorithms
#[derive(Debug, Clone)]
pub enum RegressionAlgorithmType {
    /// Statistical change point detection
    ChangePointDetection,
    /// Control chart analysis
    ControlChart,
    /// Trend analysis
    TrendAnalysis,
    /// Machine learning based
    MachineLearning,
    /// Composite algorithm
    Composite,
}

/// Regression detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetectionConfig {
    /// Enable continuous monitoring
    pub enable_continuous_monitoring: bool,
    /// Detection window size
    pub detection_window: Duration,
    /// Minimum regression magnitude
    pub min_regression_magnitude: f64,
    /// Confidence threshold
    pub confidence_threshold: f64,
}

/// Baseline management for regression detection
#[derive(Debug, Clone)]
pub struct BaselineManager {
    /// Current baselines
    pub baselines: HashMap<String, PerformanceBaseline>,
    /// Baseline update policy
    pub update_policy: BaselineUpdatePolicy,
    /// Baseline validation
    pub validation_results: BaselineValidationResults,
}

/// Performance baseline definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Baseline name
    pub name: String,
    /// Baseline values
    pub values: HashMap<String, f64>,
    /// Baseline timestamp
    pub timestamp: SystemTime,
    /// Baseline confidence
    pub confidence: f64,
    /// Baseline validity period
    pub validity_period: Duration,
}

/// Baseline update policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineUpdatePolicy {
    /// Update frequency
    pub update_frequency: Duration,
    /// Minimum data points for update
    pub min_data_points: usize,
    /// Update threshold
    pub update_threshold: f64,
    /// Auto-update enabled
    pub auto_update: bool,
}

/// Baseline validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineValidationResults {
    /// Validation status
    pub status: ValidationStatus,
    /// Validation score
    pub score: f64,
    /// Validation timestamp
    pub timestamp: SystemTime,
    /// Validation errors
    pub errors: Vec<String>,
}

/// Validation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Valid baseline
    Valid,
    /// Invalid baseline
    Invalid,
    /// Needs validation
    NeedsValidation,
    /// Validation in progress
    Validating,
}
