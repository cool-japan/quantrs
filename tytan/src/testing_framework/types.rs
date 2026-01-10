//! Core types for the testing framework.
//!
//! This module defines fundamental types, enums, and traits used throughout
//! the testing framework.

use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Test suite
#[derive(Debug, Clone)]
pub struct TestSuite {
    /// Test categories
    pub categories: Vec<TestCategory>,
    /// Individual test cases
    pub test_cases: Vec<TestCase>,
    /// Benchmarks
    pub benchmarks: Vec<Benchmark>,
}

#[derive(Debug, Clone)]
pub struct TestCategory {
    /// Category name
    pub name: String,
    /// Description
    pub description: String,
    /// Problem types
    pub problem_types: Vec<ProblemType>,
    /// Difficulty levels
    pub difficulties: Vec<Difficulty>,
    /// Tags
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProblemType {
    /// Max-cut problem
    MaxCut,
    /// Traveling salesman
    TSP,
    /// Graph coloring
    GraphColoring,
    /// Number partitioning
    NumberPartitioning,
    /// Knapsack
    Knapsack,
    /// Set cover
    SetCover,
    /// Vehicle routing
    VRP,
    /// Job scheduling
    JobScheduling,
    /// Portfolio optimization
    Portfolio,
    /// Ising model
    Ising,
    /// Custom problem
    Custom { name: String },
}

#[derive(Debug, Clone)]
pub enum Difficulty {
    /// Easy problems
    Easy,
    /// Medium difficulty
    Medium,
    /// Hard problems
    Hard,
    /// Very hard (NP-hard instances)
    VeryHard,
    /// Stress test
    Extreme,
}

#[derive(Debug, Clone)]
pub struct TestCase {
    /// Test ID
    pub id: String,
    /// Problem type
    pub problem_type: ProblemType,
    /// Problem size
    pub size: usize,
    /// QUBO matrix
    pub qubo: Array2<f64>,
    /// Variable mapping
    pub var_map: HashMap<String, usize>,
    /// Known optimal solution (if available)
    pub optimal_solution: Option<HashMap<String, bool>>,
    /// Optimal value
    pub optimal_value: Option<f64>,
    /// Constraints
    pub constraints: Vec<Constraint>,
    /// Metadata
    pub metadata: TestMetadata,
}

#[derive(Debug, Clone)]
pub struct TestMetadata {
    /// Generation method
    pub generation_method: String,
    /// Difficulty estimate
    pub difficulty: Difficulty,
    /// Expected runtime
    pub expected_runtime: Duration,
    /// Notes
    pub notes: String,
    /// Tags
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Variables involved
    pub variables: Vec<String>,
    /// Parameters
    pub parameters: HashMap<String, f64>,
    /// Penalty weight
    pub penalty: f64,
}

#[derive(Debug, Clone)]
pub enum ConstraintType {
    /// Linear equality
    LinearEquality { target: f64 },
    /// Linear inequality
    LinearInequality { bound: f64, is_upper: bool },
    /// One-hot encoding
    OneHot,
    /// At most k
    AtMostK { k: usize },
    /// At least k
    AtLeastK { k: usize },
    /// Exactly k
    ExactlyK { k: usize },
    /// Custom constraint
    Custom { name: String },
}

#[derive(Debug, Clone)]
pub struct Benchmark {
    /// Benchmark name
    pub name: String,
    /// Test cases
    pub test_cases: Vec<String>,
    /// Performance metrics to collect
    pub metrics: Vec<PerformanceMetric>,
    /// Baseline results
    pub baseline: Option<BenchmarkResults>,
}

#[derive(Debug, Clone)]
pub enum PerformanceMetric {
    /// Solving time
    SolveTime,
    /// Solution quality
    SolutionQuality,
    /// Constraint violations
    ConstraintViolations,
    /// Memory usage
    MemoryUsage,
    /// Convergence rate
    ConvergenceRate,
    /// Sample efficiency
    SampleEfficiency,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Benchmark name
    pub name: String,
    /// Results per metric
    pub metrics: HashMap<String, f64>,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

#[derive(Debug, Clone)]
pub enum FailureType {
    /// Timeout
    Timeout,
    /// Constraint violation
    ConstraintViolation,
    /// Invalid solution
    InvalidSolution,
    /// Sampler error
    SamplerError,
    /// Validation error
    ValidationError,
    /// Unexpected error
    UnexpectedError,
}

/// Regression testing report
#[derive(Debug, Clone)]
pub struct RegressionReport {
    /// Performance regressions detected
    pub regressions: Vec<RegressionIssue>,
    /// Performance improvements detected
    pub improvements: Vec<RegressionIssue>,
    /// Number of baseline tests
    pub baseline_tests: usize,
    /// Number of current tests
    pub current_tests: usize,
}

/// Individual regression issue
#[derive(Debug, Clone)]
pub struct RegressionIssue {
    /// Test ID
    pub test_id: String,
    /// Metric that regressed (quality, runtime, etc.)
    pub metric: String,
    /// Baseline value
    pub baseline_value: f64,
    /// Current value
    pub current_value: f64,
    /// Percentage change
    pub change_percent: f64,
}

/// CI/CD integration report
#[derive(Debug, Clone)]
pub struct CIReport {
    /// Overall CI status
    pub status: CIStatus,
    /// Test pass rate
    pub passed_rate: f64,
    /// Total number of tests
    pub total_tests: usize,
    /// Number of failed tests
    pub failed_tests: usize,
    /// Number of critical failures
    pub critical_failures: usize,
    /// Average runtime
    pub avg_runtime: Duration,
    /// Overall quality score (0-100)
    pub quality_score: f64,
}

/// CI status enumeration
#[derive(Debug, Clone)]
pub enum CIStatus {
    /// All tests passed with good performance
    Pass,
    /// Tests passed but with warnings
    Warning,
    /// Critical failures detected
    Fail,
}

/// Extended performance metrics
#[derive(Debug, Clone)]
pub struct ExtendedPerformanceMetrics {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Memory utilization (MB)
    pub memory_usage: f64,
    /// GPU utilization (if applicable)
    pub gpu_utilization: Option<f64>,
    /// Energy consumption (Joules)
    pub energy_consumption: f64,
    /// Iterations per second
    pub iterations_per_second: f64,
    /// Solution quality trend
    pub quality_trend: QualityTrend,
}

/// Quality trend analysis
#[derive(Debug, Clone)]
pub enum QualityTrend {
    /// Quality improving over time
    Improving,
    /// Quality stable
    Stable,
    /// Quality degrading
    Degrading,
    /// Insufficient data
    Unknown,
}

/// Test execution environment
#[derive(Debug, Clone)]
pub struct TestEnvironment {
    /// Operating system
    pub os: String,
    /// CPU model
    pub cpu_model: String,
    /// Available memory (GB)
    pub memory_gb: f64,
    /// GPU information (if available)
    pub gpu_info: Option<String>,
    /// Rust version
    pub rust_version: String,
    /// Compilation flags
    pub compile_flags: Vec<String>,
}

/// Sampler comparison results
#[derive(Debug, Clone)]
pub struct SamplerComparison {
    /// First sampler name
    pub sampler1_name: String,
    /// Second sampler name
    pub sampler2_name: String,
    /// Individual test comparisons
    pub test_comparisons: Vec<TestComparison>,
    /// Average quality improvement (sampler2 vs sampler1)
    pub avg_quality_improvement: f64,
    /// Average runtime ratio (sampler2 / sampler1)
    pub avg_runtime_ratio: f64,
    /// Overall winner
    pub winner: String,
}

/// Individual test comparison
#[derive(Debug, Clone)]
pub struct TestComparison {
    /// Test ID
    pub test_id: String,
    /// First sampler quality
    pub sampler1_quality: f64,
    /// Second sampler quality
    pub sampler2_quality: f64,
    /// Quality improvement (positive means sampler2 is better)
    pub quality_improvement: f64,
    /// First sampler runtime
    pub sampler1_runtime: Duration,
    /// Second sampler runtime
    pub sampler2_runtime: Duration,
    /// Runtime ratio (sampler2 / sampler1)
    pub runtime_ratio: f64,
}

/// Test generator trait
pub trait TestGenerator: Send + Sync {
    /// Generate test cases
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String>;

    /// Generator name
    fn name(&self) -> &str;

    /// Supported problem types
    fn supported_types(&self) -> Vec<ProblemType>;
}

#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Problem type
    pub problem_type: ProblemType,
    /// Problem size
    pub size: usize,
    /// Difficulty
    pub difficulty: Difficulty,
    /// Random seed
    pub seed: Option<u64>,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
}

/// Validator trait
pub trait Validator: Send + Sync {
    /// Validate test result
    fn validate(&self, test_case: &TestCase, result: &TestResult) -> ValidationResult;

    /// Validator name
    fn name(&self) -> &str;
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Overall valid
    pub is_valid: bool,
    /// Validation checks
    pub checks: Vec<ValidationCheck>,
    /// Warnings
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ValidationCheck {
    /// Check name
    pub name: String,
    /// Passed
    pub passed: bool,
    /// Message
    pub message: String,
    /// Details
    pub details: Option<String>,
}

/// Test result
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test case ID
    pub test_id: String,
    /// Sampler used
    pub sampler: String,
    /// Solution found
    pub solution: HashMap<String, bool>,
    /// Objective value
    pub objective_value: f64,
    /// Constraints satisfied
    pub constraints_satisfied: bool,
    /// Validation results
    pub validation: ValidationResult,
    /// Runtime
    pub runtime: Duration,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Test failure record
#[derive(Debug, Clone)]
pub struct TestFailure {
    /// Test ID
    pub test_id: String,
    /// Failure type
    pub failure_type: FailureType,
    /// Error message
    pub message: String,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
    /// Context
    pub context: HashMap<String, String>,
}
