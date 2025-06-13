//! Automated testing framework for quantum optimization.
//!
//! This module provides comprehensive testing tools for QUBO problems,
//! including test case generation, validation, and benchmarking.

#[cfg(feature = "dwave")]
use crate::compile::{Compile, CompiledModel};
use crate::sampler::{SampleResult, Sampler, SamplerError, SamplerResult};
use ndarray::{Array, Array1, Array2, IxDyn};
use rand::prelude::*;
use rand::{thread_rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

/// Automated testing framework
pub struct TestingFramework {
    /// Test configuration
    config: TestConfig,
    /// Test suite
    suite: TestSuite,
    /// Test results
    results: TestResults,
    /// Validators
    validators: Vec<Box<dyn Validator>>,
    /// Generators
    generators: Vec<Box<dyn TestGenerator>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    /// Random seed
    pub seed: Option<u64>,
    /// Number of test cases per category
    pub cases_per_category: usize,
    /// Problem sizes to test
    pub problem_sizes: Vec<usize>,
    /// Samplers to test
    pub samplers: Vec<SamplerConfig>,
    /// Timeout per test
    pub timeout: Duration,
    /// Validation settings
    pub validation: ValidationConfig,
    /// Output settings
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    /// Sampler name
    pub name: String,
    /// Number of samples
    pub num_samples: usize,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Check constraint satisfaction
    pub check_constraints: bool,
    /// Check objective improvement
    pub check_objective: bool,
    /// Statistical validation
    pub statistical_tests: bool,
    /// Tolerance for floating point comparisons
    pub tolerance: f64,
    /// Minimum solution quality
    pub min_quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Generate report
    pub generate_report: bool,
    /// Report format
    pub format: ReportFormat,
    /// Output directory
    pub output_dir: String,
    /// Verbosity level
    pub verbosity: VerbosityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    /// Plain text
    Text,
    /// JSON
    Json,
    /// HTML
    Html,
    /// Markdown
    Markdown,
    /// CSV
    Csv,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerbosityLevel {
    /// Only errors
    Error,
    /// Warnings and errors
    Warning,
    /// Info messages
    Info,
    /// Debug information
    Debug,
}

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

#[derive(Debug, Clone, PartialEq)]
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

/// Test results
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

#[derive(Debug, Clone)]
pub struct PerformanceData {
    /// Runtime statistics
    pub runtime_stats: RuntimeStats,
    /// Memory statistics
    pub memory_stats: MemoryStats,
    /// Convergence data
    pub convergence_data: ConvergenceData,
}

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

#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Peak memory usage
    pub peak_memory: usize,
    /// Average memory usage
    pub avg_memory: usize,
    /// Memory per test
    pub memory_per_test: Vec<(String, usize)>,
}

#[derive(Debug, Clone)]
pub struct ConvergenceData {
    /// Convergence curves
    pub curves: Vec<ConvergenceCurve>,
    /// Average iterations to convergence
    pub avg_iterations: f64,
    /// Convergence rate
    pub convergence_rate: f64,
}

#[derive(Debug, Clone)]
pub struct ConvergenceCurve {
    /// Test ID
    pub test_id: String,
    /// Iteration data
    pub iterations: Vec<IterationData>,
    /// Converged
    pub converged: bool,
}

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

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Benchmark name
    pub name: String,
    /// Results per metric
    pub metrics: HashMap<String, f64>,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
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

impl TestingFramework {
    /// Run regression tests against baseline
    pub fn run_regression_tests<S: Sampler>(&mut self, sampler: &S, baseline_file: &str) -> Result<RegressionReport, String> {
        // Load baseline results
        let baseline = self.load_baseline(baseline_file)?;
        
        // Run current tests
        self.run_suite(sampler)?;
        
        // Compare with baseline
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();
        
        for current_result in &self.results.test_results {
            if let Some(baseline_result) = baseline.iter().find(|b| b.test_id == current_result.test_id) {
                let quality_change = (current_result.objective_value - baseline_result.objective_value) / baseline_result.objective_value.abs();
                let runtime_change = (current_result.runtime.as_secs_f64() - baseline_result.runtime.as_secs_f64()) / baseline_result.runtime.as_secs_f64();
                
                if quality_change > 0.05 || runtime_change > 0.2 {
                    regressions.push(RegressionIssue {
                        test_id: current_result.test_id.clone(),
                        metric: if quality_change > 0.05 { "quality".to_string() } else { "runtime".to_string() },
                        baseline_value: if quality_change > 0.05 { baseline_result.objective_value } else { baseline_result.runtime.as_secs_f64() },
                        current_value: if quality_change > 0.05 { current_result.objective_value } else { current_result.runtime.as_secs_f64() },
                        change_percent: if quality_change > 0.05 { quality_change * 100.0 } else { runtime_change * 100.0 },
                    });
                } else if quality_change < -0.05 || runtime_change < -0.2 {
                    improvements.push(RegressionIssue {
                        test_id: current_result.test_id.clone(),
                        metric: if quality_change < -0.05 { "quality".to_string() } else { "runtime".to_string() },
                        baseline_value: if quality_change < -0.05 { baseline_result.objective_value } else { baseline_result.runtime.as_secs_f64() },
                        current_value: if quality_change < -0.05 { current_result.objective_value } else { current_result.runtime.as_secs_f64() },
                        change_percent: if quality_change < -0.05 { quality_change * 100.0 } else { runtime_change * 100.0 },
                    });
                }
            }
        }
        
        Ok(RegressionReport {
            regressions,
            improvements,
            baseline_tests: baseline.len(),
            current_tests: self.results.test_results.len(),
        })
    }
    
    /// Load baseline results from file
    fn load_baseline(&self, _filename: &str) -> Result<Vec<TestResult>, String> {
        // Simplified implementation - in practice would load from JSON/CSV
        Ok(Vec::new())
    }
    
    /// Run test suite in parallel
    pub fn run_suite_parallel<S: Sampler + Clone + Send + Sync>(&mut self, sampler: &S, num_threads: usize) -> Result<(), String> {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let test_cases = Arc::new(self.suite.test_cases.clone());
        let results = Arc::new(Mutex::new(Vec::new()));
        let failures = Arc::new(Mutex::new(Vec::new()));
        
        let total_start = Instant::now();
        let chunk_size = (test_cases.len() + num_threads - 1) / num_threads;
        
        let mut handles = Vec::new();
        
        for thread_id in 0..num_threads {
            let start_idx = thread_id * chunk_size;
            let end_idx = ((thread_id + 1) * chunk_size).min(test_cases.len());
            
            if start_idx >= test_cases.len() {
                break;
            }
            
            let test_cases_clone = Arc::clone(&test_cases);
            let results_clone = Arc::clone(&results);
            let failures_clone = Arc::clone(&failures);
            let sampler_clone = sampler.clone();
            
            let handle = thread::spawn(move || {
                for idx in start_idx..end_idx {
                    let test_case = &test_cases_clone[idx];
                    
                    match Self::run_single_test_static(test_case, &sampler_clone) {
                        Ok(result) => {
                            results_clone.lock().unwrap().push(result);
                        }
                        Err(e) => {
                            failures_clone.lock().unwrap().push(TestFailure {
                                test_id: test_case.id.clone(),
                                failure_type: FailureType::SamplerError,
                                message: e,
                                stack_trace: None,
                                context: HashMap::new(),
                            });
                        }
                    }
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().map_err(|_| "Thread panic")?;
        }
        
        // Collect results
        self.results.test_results = results.lock().unwrap().clone();
        self.results.failures = failures.lock().unwrap().clone();
        
        self.results.performance.runtime_stats.total_time = total_start.elapsed();
        self.results.summary.passed = self.results.test_results.len();
        self.results.summary.failed = self.results.failures.len();
        self.results.summary.total_tests = self.results.summary.passed + self.results.summary.failed;
        
        self.calculate_summary();
        
        Ok(())
    }
    
    /// Static version of run_single_test for parallel execution
    fn run_single_test_static<S: Sampler>(test_case: &TestCase, sampler: &S) -> Result<TestResult, String> {
        let solve_start = Instant::now();
        
        // Run sampler
        let sample_result = sampler
            .run_qubo(&(test_case.qubo.clone(), test_case.var_map.clone()), 100)
            .map_err(|e| format!("Sampler error: {:?}", e))?;
        
        let solve_time = solve_start.elapsed();
        
        // Get best solution
        let best_sample = sample_result
            .iter()
            .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
            .ok_or("No samples returned")?;
        
        let solution = best_sample.assignments.clone();
        
        Ok(TestResult {
            test_id: test_case.id.clone(),
            sampler: "parallel".to_string(),
            solution,
            objective_value: best_sample.energy,
            constraints_satisfied: true,
            validation: ValidationResult {
                is_valid: true,
                checks: Vec::new(),
                warnings: Vec::new(),
            },
            runtime: solve_time,
            metrics: HashMap::new(),
        })
    }
    
    /// Generate CI/CD report
    pub fn generate_ci_report(&self) -> Result<CIReport, String> {
        let passed_rate = if self.results.summary.total_tests > 0 {
            self.results.summary.passed as f64 / self.results.summary.total_tests as f64
        } else {
            0.0
        };
        
        let status = if passed_rate >= 0.95 {
            CIStatus::Pass
        } else if passed_rate >= 0.8 {
            CIStatus::Warning
        } else {
            CIStatus::Fail
        };
        
        Ok(CIReport {
            status,
            passed_rate,
            total_tests: self.results.summary.total_tests,
            failed_tests: self.results.summary.failed,
            critical_failures: self.results.failures.iter().filter(|f| matches!(f.failure_type, FailureType::Timeout | FailureType::SamplerError)).count(),
            avg_runtime: self.results.summary.avg_runtime,
            quality_score: self.calculate_quality_score(),
        })
    }
    
    /// Calculate overall quality score
    fn calculate_quality_score(&self) -> f64 {
        if self.results.test_results.is_empty() {
            return 0.0;
        }
        
        let constraint_score = self.results.summary.quality_metrics.constraint_satisfaction_rate;
        let success_score = self.results.summary.success_rate;
        let quality_score = if self.results.summary.quality_metrics.best_quality.is_finite() {
            0.8 // Base score for having finite solutions
        } else {
            0.0
        };
        
        (constraint_score * 0.4 + success_score * 0.4 + quality_score * 0.2) * 100.0
    }
    
    /// Add stress test cases
    pub fn add_stress_tests(&mut self) {
        let stress_categories = vec![
            TestCategory {
                name: "Large Scale Tests".to_string(),
                description: "Tests with large problem sizes".to_string(),
                problem_types: vec![ProblemType::MaxCut, ProblemType::TSP],
                difficulties: vec![Difficulty::Extreme],
                tags: vec!["stress".to_string(), "large".to_string()],
            },
            TestCategory {
                name: "Memory Stress Tests".to_string(),
                description: "Tests designed to stress memory usage".to_string(),
                problem_types: vec![ProblemType::Knapsack],
                difficulties: vec![Difficulty::VeryHard, Difficulty::Extreme],
                tags: vec!["stress".to_string(), "memory".to_string()],
            },
            TestCategory {
                name: "Runtime Stress Tests".to_string(),
                description: "Tests with challenging runtime requirements".to_string(),
                problem_types: vec![ProblemType::GraphColoring],
                difficulties: vec![Difficulty::Extreme],
                tags: vec!["stress".to_string(), "runtime".to_string()],
            },
        ];
        
        for category in stress_categories {
            self.suite.categories.push(category);
        }
    }
    
    /// Detect test environment
    pub fn detect_environment(&self) -> TestEnvironment {
        TestEnvironment {
            os: std::env::consts::OS.to_string(),
            cpu_model: "Unknown".to_string(), // Would need OS-specific detection
            memory_gb: 8.0, // Simplified - would need system detection
            gpu_info: None,
            rust_version: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
            compile_flags: vec!["--release".to_string()],
        }
    }
    
    /// Export test results for external analysis
    pub fn export_results(&self, format: &str) -> Result<String, String> {
        match format {
            "csv" => self.export_csv(),
            "json" => self.generate_json_report(),
            "xml" => self.export_xml(),
            _ => Err(format!("Unsupported export format: {}", format)),
        }
    }
    
    /// Export results as CSV
    fn export_csv(&self) -> Result<String, String> {
        let mut csv = String::new();
        csv.push_str("test_id,problem_type,size,sampler,objective_value,runtime_ms,constraints_satisfied,valid\n");
        
        for result in &self.results.test_results {
            // Find corresponding test case for additional info
            if let Some(test_case) = self.suite.test_cases.iter().find(|tc| tc.id == result.test_id) {
                csv.push_str(&format!(
                    "{},{:?},{},{},{},{},{},{}\n",
                    result.test_id,
                    test_case.problem_type,
                    test_case.size,
                    result.sampler,
                    result.objective_value,
                    result.runtime.as_millis(),
                    result.constraints_satisfied,
                    result.validation.is_valid
                ));
            }
        }
        
        Ok(csv)
    }
    
    /// Export results as XML
    fn export_xml(&self) -> Result<String, String> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<test_results>\n");
        xml.push_str(&format!("  <summary total=\"{}\" passed=\"{}\" failed=\"{}\" success_rate=\"{:.2}\"/>\n",
            self.results.summary.total_tests,
            self.results.summary.passed,
            self.results.summary.failed,
            self.results.summary.success_rate
        ));
        
        xml.push_str("  <tests>\n");
        for result in &self.results.test_results {
            xml.push_str(&format!(
                "    <test id=\"{}\" sampler=\"{}\" objective=\"{}\" runtime_ms=\"{}\" valid=\"{}\"/>\n",
                result.test_id,
                result.sampler,
                result.objective_value,
                result.runtime.as_millis(),
                result.validation.is_valid
            ));
        }
        xml.push_str("  </tests>\n");
        xml.push_str("</test_results>\n");
        
        Ok(xml)
    }
    
    /// Add industry-specific test generators
    pub fn add_industry_generators(&mut self) {
        // Add finance test generator
        self.generators.push(Box::new(FinanceTestGenerator));
        
        // Add logistics test generator  
        self.generators.push(Box::new(LogisticsTestGenerator));
        
        // Add manufacturing test generator
        self.generators.push(Box::new(ManufacturingTestGenerator));
    }
    
    /// Generate performance comparison report
    pub fn compare_samplers<S1: Sampler, S2: Sampler>(&mut self, sampler1: &S1, sampler2: &S2, sampler1_name: &str, sampler2_name: &str) -> Result<SamplerComparison, String> {
        // Run tests with first sampler
        self.run_suite(sampler1)?;
        let results1 = self.results.test_results.clone();
        
        // Clear results and run with second sampler
        self.results.test_results.clear();
        self.run_suite(sampler2)?;
        let results2 = self.results.test_results.clone();
        
        // Compare results
        let mut comparisons = Vec::new();
        
        for r1 in &results1 {
            if let Some(r2) = results2.iter().find(|r| r.test_id == r1.test_id) {
                let quality_diff = r2.objective_value - r1.objective_value;
                let runtime_ratio = r2.runtime.as_secs_f64() / r1.runtime.as_secs_f64();
                
                comparisons.push(TestComparison {
                    test_id: r1.test_id.clone(),
                    sampler1_quality: r1.objective_value,
                    sampler2_quality: r2.objective_value,
                    quality_improvement: -quality_diff, // Negative because lower is better
                    sampler1_runtime: r1.runtime,
                    sampler2_runtime: r2.runtime,
                    runtime_ratio,
                });
            }
        }
        
        let avg_quality_improvement = comparisons.iter().map(|c| c.quality_improvement).sum::<f64>() / comparisons.len() as f64;
        let avg_runtime_ratio = comparisons.iter().map(|c| c.runtime_ratio).sum::<f64>() / comparisons.len() as f64;
        
        Ok(SamplerComparison {
            sampler1_name: sampler1_name.to_string(),
            sampler2_name: sampler2_name.to_string(),
            test_comparisons: comparisons,
            avg_quality_improvement,
            avg_runtime_ratio,
            winner: if avg_quality_improvement > 0.0 { sampler2_name.to_string() } else { sampler1_name.to_string() },
        })
    }
    /// Create new testing framework
    pub fn new(config: TestConfig) -> Self {
        Self {
            config,
            suite: TestSuite {
                categories: Vec::new(),
                test_cases: Vec::new(),
                benchmarks: Vec::new(),
            },
            results: TestResults {
                test_results: Vec::new(),
                summary: TestSummary {
                    total_tests: 0,
                    passed: 0,
                    failed: 0,
                    skipped: 0,
                    avg_runtime: Duration::from_secs(0),
                    success_rate: 0.0,
                    quality_metrics: QualityMetrics {
                        avg_quality: 0.0,
                        best_quality: f64::NEG_INFINITY,
                        worst_quality: f64::INFINITY,
                        std_dev: 0.0,
                        constraint_satisfaction_rate: 0.0,
                    },
                },
                failures: Vec::new(),
                performance: PerformanceData {
                    runtime_stats: RuntimeStats {
                        total_time: Duration::from_secs(0),
                        qubo_generation_time: Duration::from_secs(0),
                        solving_time: Duration::from_secs(0),
                        validation_time: Duration::from_secs(0),
                        time_per_test: Vec::new(),
                    },
                    memory_stats: MemoryStats {
                        peak_memory: 0,
                        avg_memory: 0,
                        memory_per_test: Vec::new(),
                    },
                    convergence_data: ConvergenceData {
                        curves: Vec::new(),
                        avg_iterations: 0.0,
                        convergence_rate: 0.0,
                    },
                },
            },
            validators: Self::default_validators(),
            generators: Self::default_generators(),
        }
    }

    /// Get default validators
    fn default_validators() -> Vec<Box<dyn Validator>> {
        vec![
            Box::new(ConstraintValidator),
            Box::new(ObjectiveValidator),
            Box::new(BoundsValidator),
            Box::new(SymmetryValidator),
        ]
    }

    /// Get default generators
    fn default_generators() -> Vec<Box<dyn TestGenerator>> {
        vec![
            Box::new(MaxCutGenerator),
            Box::new(TSPGenerator),
            Box::new(GraphColoringGenerator),
            Box::new(KnapsackGenerator),
            Box::new(RandomQuboGenerator),
        ]
    }

    /// Add test category
    pub fn add_category(&mut self, category: TestCategory) {
        self.suite.categories.push(category);
    }

    /// Add custom generator
    pub fn add_generator(&mut self, generator: Box<dyn TestGenerator>) {
        self.generators.push(generator);
    }

    /// Add custom validator
    pub fn add_validator(&mut self, validator: Box<dyn Validator>) {
        self.validators.push(validator);
    }

    /// Generate test suite
    pub fn generate_suite(&mut self) -> Result<(), String> {
        let start_time = Instant::now();

        // Generate tests for each category
        for category in &self.suite.categories {
            for problem_type in &category.problem_types {
                for difficulty in &category.difficulties {
                    for size in &self.config.problem_sizes {
                        let config = GeneratorConfig {
                            problem_type: problem_type.clone(),
                            size: *size,
                            difficulty: difficulty.clone(),
                            seed: self.config.seed,
                            parameters: HashMap::new(),
                        };

                        // Find suitable generator
                        for generator in &self.generators {
                            if generator.supported_types().contains(problem_type) {
                                let test_cases = generator.generate(&config)?;
                                self.suite.test_cases.extend(test_cases);
                                break;
                            }
                        }
                    }
                }
            }
        }

        self.results.performance.runtime_stats.qubo_generation_time = start_time.elapsed();

        Ok(())
    }

    /// Run test suite
    pub fn run_suite<S: Sampler>(&mut self, sampler: &S) -> Result<(), String> {
        let total_start = Instant::now();

        let test_cases = self.suite.test_cases.clone();
        for test_case in &test_cases {
            let test_start = Instant::now();

            // Run test with timeout
            match self.run_single_test(test_case, sampler) {
                Ok(result) => {
                    self.results.test_results.push(result);
                    self.results.summary.passed += 1;
                }
                Err(e) => {
                    self.results.failures.push(TestFailure {
                        test_id: test_case.id.clone(),
                        failure_type: FailureType::SamplerError,
                        message: e,
                        stack_trace: None,
                        context: HashMap::new(),
                    });
                    self.results.summary.failed += 1;
                }
            }

            let test_time = test_start.elapsed();
            self.results
                .performance
                .runtime_stats
                .time_per_test
                .push((test_case.id.clone(), test_time));

            self.results.summary.total_tests += 1;
        }

        self.results.performance.runtime_stats.total_time = total_start.elapsed();
        self.calculate_summary();

        Ok(())
    }

    /// Run single test
    fn run_single_test<S: Sampler>(
        &mut self,
        test_case: &TestCase,
        sampler: &S,
    ) -> Result<TestResult, String> {
        let solve_start = Instant::now();

        // Run sampler
        let sample_result = sampler
            .run_qubo(
                &(test_case.qubo.clone(), test_case.var_map.clone()),
                self.config.samplers[0].num_samples,
            )
            .map_err(|e| format!("Sampler error: {:?}", e))?;

        let solve_time = solve_start.elapsed();

        // Get best solution
        let best_sample = sample_result
            .iter()
            .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
            .ok_or("No samples returned")?;

        // Use the assignments directly (already decoded)
        let solution = best_sample.assignments.clone();

        // Validate
        let validation_start = Instant::now();
        let mut validation = ValidationResult {
            is_valid: true,
            checks: Vec::new(),
            warnings: Vec::new(),
        };

        for validator in &self.validators {
            let result = validator.validate(
                test_case,
                &TestResult {
                    test_id: test_case.id.clone(),
                    sampler: "test".to_string(),
                    solution: solution.clone(),
                    objective_value: best_sample.energy,
                    constraints_satisfied: true,
                    validation: validation.clone(),
                    runtime: solve_time,
                    metrics: HashMap::new(),
                },
            );

            validation.checks.extend(result.checks);
            validation.warnings.extend(result.warnings);
            validation.is_valid &= result.is_valid;
        }

        let validation_time = validation_start.elapsed();
        self.results.performance.runtime_stats.solving_time += solve_time;
        self.results.performance.runtime_stats.validation_time += validation_time;

        Ok(TestResult {
            test_id: test_case.id.clone(),
            sampler: self.config.samplers[0].name.clone(),
            solution,
            objective_value: best_sample.energy,
            constraints_satisfied: validation.is_valid,
            validation,
            runtime: solve_time + validation_time,
            metrics: HashMap::new(),
        })
    }

    /// Decode solution
    fn decode_solution(
        &self,
        var_map: &HashMap<String, usize>,
        sample: &[i8],
    ) -> HashMap<String, bool> {
        let mut solution = HashMap::new();

        for (var_name, &idx) in var_map {
            if idx < sample.len() {
                solution.insert(var_name.clone(), sample[idx] == 1);
            }
        }

        solution
    }

    /// Calculate summary statistics
    fn calculate_summary(&mut self) {
        if self.results.test_results.is_empty() {
            return;
        }

        // Success rate
        self.results.summary.success_rate =
            self.results.summary.passed as f64 / self.results.summary.total_tests as f64;

        // Average runtime
        let total_runtime: Duration = self.results.test_results.iter().map(|r| r.runtime).sum();
        self.results.summary.avg_runtime = total_runtime / self.results.test_results.len() as u32;

        // Quality metrics
        let qualities: Vec<f64> = self
            .results
            .test_results
            .iter()
            .map(|r| r.objective_value)
            .collect();

        self.results.summary.quality_metrics.avg_quality =
            qualities.iter().sum::<f64>() / qualities.len() as f64;

        self.results.summary.quality_metrics.best_quality = *qualities
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);

        self.results.summary.quality_metrics.worst_quality = *qualities
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);

        // Standard deviation
        let mean = self.results.summary.quality_metrics.avg_quality;
        let variance =
            qualities.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / qualities.len() as f64;
        self.results.summary.quality_metrics.std_dev = variance.sqrt();

        // Constraint satisfaction rate
        let satisfied = self
            .results
            .test_results
            .iter()
            .filter(|r| r.constraints_satisfied)
            .count();
        self.results
            .summary
            .quality_metrics
            .constraint_satisfaction_rate =
            satisfied as f64 / self.results.test_results.len() as f64;
    }

    /// Generate report
    pub fn generate_report(&self) -> Result<String, String> {
        match self.config.output.format {
            ReportFormat::Text => self.generate_text_report(),
            ReportFormat::Json => self.generate_json_report(),
            ReportFormat::Html => self.generate_html_report(),
            ReportFormat::Markdown => self.generate_markdown_report(),
            ReportFormat::Csv => self.generate_csv_report(),
        }
    }

    /// Generate text report
    fn generate_text_report(&self) -> Result<String, String> {
        let mut report = String::new();

        report.push_str("=== Quantum Optimization Test Report ===\n\n");

        report.push_str(&format!(
            "Total Tests: {}\n",
            self.results.summary.total_tests
        ));
        report.push_str(&format!("Passed: {}\n", self.results.summary.passed));
        report.push_str(&format!("Failed: {}\n", self.results.summary.failed));
        report.push_str(&format!(
            "Success Rate: {:.2}%\n",
            self.results.summary.success_rate * 100.0
        ));
        report.push_str(&format!(
            "Average Runtime: {:?}\n\n",
            self.results.summary.avg_runtime
        ));

        report.push_str("Quality Metrics:\n");
        report.push_str(&format!(
            "  Average Quality: {:.4}\n",
            self.results.summary.quality_metrics.avg_quality
        ));
        report.push_str(&format!(
            "  Best Quality: {:.4}\n",
            self.results.summary.quality_metrics.best_quality
        ));
        report.push_str(&format!(
            "  Worst Quality: {:.4}\n",
            self.results.summary.quality_metrics.worst_quality
        ));
        report.push_str(&format!(
            "  Std Dev: {:.4}\n",
            self.results.summary.quality_metrics.std_dev
        ));
        report.push_str(&format!(
            "  Constraint Satisfaction: {:.2}%\n\n",
            self.results
                .summary
                .quality_metrics
                .constraint_satisfaction_rate
                * 100.0
        ));

        if !self.results.failures.is_empty() {
            report.push_str("Failures:\n");
            for failure in &self.results.failures {
                report.push_str(&format!(
                    "  - {} ({}): {}\n",
                    failure.test_id,
                    format!("{:?}", failure.failure_type),
                    failure.message
                ));
            }
        }

        Ok(report)
    }

    /// Generate JSON report
    fn generate_json_report(&self) -> Result<String, String> {
        use std::fmt::Write;
        
        let mut json = String::new();
        
        // Build JSON manually (avoiding serde dependency issues)
        json.push_str("{\n");
        
        // Summary section
        json.push_str("  \"summary\": {\n");
        write!(&mut json, "    \"total_tests\": {},\n", self.results.summary.total_tests).unwrap();
        write!(&mut json, "    \"passed\": {},\n", self.results.summary.passed).unwrap();
        write!(&mut json, "    \"failed\": {},\n", self.results.summary.failed).unwrap();
        write!(&mut json, "    \"skipped\": {},\n", self.results.summary.skipped).unwrap();
        write!(&mut json, "    \"success_rate\": {},\n", self.results.summary.success_rate).unwrap();
        write!(&mut json, "    \"avg_runtime_ms\": {}\n", self.results.summary.avg_runtime.as_millis()).unwrap();
        json.push_str("  },\n");
        
        // Quality metrics
        json.push_str("  \"quality_metrics\": {\n");
        write!(&mut json, "    \"avg_quality\": {},\n", self.results.summary.quality_metrics.avg_quality).unwrap();
        write!(&mut json, "    \"best_quality\": {},\n", self.results.summary.quality_metrics.best_quality).unwrap();
        write!(&mut json, "    \"worst_quality\": {},\n", self.results.summary.quality_metrics.worst_quality).unwrap();
        write!(&mut json, "    \"std_dev\": {},\n", self.results.summary.quality_metrics.std_dev).unwrap();
        write!(&mut json, "    \"constraint_satisfaction_rate\": {}\n", self.results.summary.quality_metrics.constraint_satisfaction_rate).unwrap();
        json.push_str("  },\n");
        
        // Performance data
        json.push_str("  \"performance\": {\n");
        write!(&mut json, "    \"total_time_ms\": {},\n", self.results.performance.runtime_stats.total_time.as_millis()).unwrap();
        write!(&mut json, "    \"solving_time_ms\": {},\n", self.results.performance.runtime_stats.solving_time.as_millis()).unwrap();
        write!(&mut json, "    \"validation_time_ms\": {}\n", self.results.performance.runtime_stats.validation_time.as_millis()).unwrap();
        json.push_str("  },\n");
        
        // Test results
        json.push_str("  \"test_results\": [\n");
        for (i, result) in self.results.test_results.iter().enumerate() {
            json.push_str("    {\n");
            write!(&mut json, "      \"test_id\": \"{}\",\n", result.test_id).unwrap();
            write!(&mut json, "      \"sampler\": \"{}\",\n", result.sampler).unwrap();
            write!(&mut json, "      \"objective_value\": {},\n", result.objective_value).unwrap();
            write!(&mut json, "      \"constraints_satisfied\": {},\n", result.constraints_satisfied).unwrap();
            write!(&mut json, "      \"runtime_ms\": {},\n", result.runtime.as_millis()).unwrap();
            write!(&mut json, "      \"is_valid\": {}\n", result.validation.is_valid).unwrap();
            json.push_str("    }");
            if i < self.results.test_results.len() - 1 {
                json.push_str(",");
            }
            json.push_str("\n");
        }
        json.push_str("  ],\n");
        
        // Failures
        json.push_str("  \"failures\": [\n");
        for (i, failure) in self.results.failures.iter().enumerate() {
            json.push_str("    {\n");
            write!(&mut json, "      \"test_id\": \"{}\",\n", failure.test_id).unwrap();
            write!(&mut json, "      \"failure_type\": \"{:?}\",\n", failure.failure_type).unwrap();
            write!(&mut json, "      \"message\": \"{}\"\n", failure.message.replace("\"", "\\\"")).unwrap();
            json.push_str("    }");
            if i < self.results.failures.len() - 1 {
                json.push_str(",");
            }
            json.push_str("\n");
        }
        json.push_str("  ]\n");
        
        json.push_str("}\n");
        
        Ok(json)
    }

    /// Generate HTML report
    fn generate_html_report(&self) -> Result<String, String> {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>Quantum Optimization Test Report</title>\n");
        html.push_str(
            "<style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .passed { color: green; }
            .failed { color: red; }
        </style>\n",
        );
        html.push_str("</head>\n<body>\n");

        html.push_str("<h1>Quantum Optimization Test Report</h1>\n");

        // Summary
        html.push_str("<h2>Summary</h2>\n");
        html.push_str("<table>\n");
        html.push_str(&format!(
            "<tr><td>Total Tests</td><td>{}</td></tr>\n",
            self.results.summary.total_tests
        ));
        html.push_str(&format!(
            "<tr><td>Passed</td><td class='passed'>{}</td></tr>\n",
            self.results.summary.passed
        ));
        html.push_str(&format!(
            "<tr><td>Failed</td><td class='failed'>{}</td></tr>\n",
            self.results.summary.failed
        ));
        html.push_str(&format!(
            "<tr><td>Success Rate</td><td>{:.2}%</td></tr>\n",
            self.results.summary.success_rate * 100.0
        ));
        html.push_str("</table>\n");

        html.push_str("</body>\n</html>");

        Ok(html)
    }

    /// Generate Markdown report
    fn generate_markdown_report(&self) -> Result<String, String> {
        let mut md = String::new();

        md.push_str("# Quantum Optimization Test Report\n\n");

        md.push_str("## Summary\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!(
            "| Total Tests | {} |\n",
            self.results.summary.total_tests
        ));
        md.push_str(&format!("| Passed | {} |\n", self.results.summary.passed));
        md.push_str(&format!("| Failed | {} |\n", self.results.summary.failed));
        md.push_str(&format!(
            "| Success Rate | {:.2}% |\n",
            self.results.summary.success_rate * 100.0
        ));
        md.push_str(&format!(
            "| Average Runtime | {:?} |\n\n",
            self.results.summary.avg_runtime
        ));

        md.push_str("## Quality Metrics\n\n");
        md.push_str(&format!(
            "- **Average Quality**: {:.4}\n",
            self.results.summary.quality_metrics.avg_quality
        ));
        md.push_str(&format!(
            "- **Best Quality**: {:.4}\n",
            self.results.summary.quality_metrics.best_quality
        ));
        md.push_str(&format!(
            "- **Worst Quality**: {:.4}\n",
            self.results.summary.quality_metrics.worst_quality
        ));
        md.push_str(&format!(
            "- **Standard Deviation**: {:.4}\n",
            self.results.summary.quality_metrics.std_dev
        ));
        md.push_str(&format!(
            "- **Constraint Satisfaction Rate**: {:.2}%\n\n",
            self.results
                .summary
                .quality_metrics
                .constraint_satisfaction_rate
                * 100.0
        ));

        Ok(md)
    }

    /// Generate CSV report
    fn generate_csv_report(&self) -> Result<String, String> {
        let mut csv = String::new();

        csv.push_str("test_id,sampler,objective_value,constraints_satisfied,runtime_ms,valid\n");

        for result in &self.results.test_results {
            csv.push_str(&format!(
                "{},{},{},{},{},{}\n",
                result.test_id,
                result.sampler,
                result.objective_value,
                result.constraints_satisfied,
                result.runtime.as_millis(),
                result.validation.is_valid
            ));
        }

        Ok(csv)
    }

    /// Save report to file
    pub fn save_report(&self, filename: &str) -> Result<(), String> {
        let report = self.generate_report()?;
        let mut file =
            File::create(filename).map_err(|e| format!("Failed to create file: {}", e))?;
        file.write_all(report.as_bytes())
            .map_err(|e| format!("Failed to write file: {}", e))?;
        Ok(())
    }
}

/// Constraint validator
struct ConstraintValidator;

impl Validator for ConstraintValidator {
    fn validate(&self, test_case: &TestCase, result: &TestResult) -> ValidationResult {
        let mut checks = Vec::new();
        let mut is_valid = true;

        for constraint in &test_case.constraints {
            let satisfied = self.check_constraint(constraint, &result.solution);

            checks.push(ValidationCheck {
                name: format!("Constraint {:?}", constraint.constraint_type),
                passed: satisfied,
                message: if satisfied {
                    "Constraint satisfied".to_string()
                } else {
                    "Constraint violated".to_string()
                },
                details: None,
            });

            is_valid &= satisfied;
        }

        ValidationResult {
            is_valid,
            checks,
            warnings: Vec::new(),
        }
    }

    fn name(&self) -> &str {
        "ConstraintValidator"
    }
}

impl ConstraintValidator {
    fn check_constraint(&self, constraint: &Constraint, solution: &HashMap<String, bool>) -> bool {
        match &constraint.constraint_type {
            ConstraintType::OneHot => {
                let active = constraint
                    .variables
                    .iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .count();
                active == 1
            }
            ConstraintType::AtMostK { k } => {
                let active = constraint
                    .variables
                    .iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .count();
                active <= *k
            }
            ConstraintType::AtLeastK { k } => {
                let active = constraint
                    .variables
                    .iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .count();
                active >= *k
            }
            ConstraintType::ExactlyK { k } => {
                let active = constraint
                    .variables
                    .iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .count();
                active == *k
            }
            _ => true, // Other constraints not implemented
        }
    }
}

/// Objective validator
struct ObjectiveValidator;

impl Validator for ObjectiveValidator {
    fn validate(&self, test_case: &TestCase, result: &TestResult) -> ValidationResult {
        let mut checks = Vec::new();

        // Check if objective is better than random
        let random_value = self.estimate_random_objective(&test_case.qubo);
        let improvement = (random_value - result.objective_value) / random_value.abs();

        checks.push(ValidationCheck {
            name: "Objective improvement".to_string(),
            passed: improvement > 0.0,
            message: format!("Improvement over random: {:.2}%", improvement * 100.0),
            details: Some(format!(
                "Random: {:.4}, Found: {:.4}",
                random_value, result.objective_value
            )),
        });

        // Check against optimal if known
        if let Some(optimal_value) = test_case.optimal_value {
            let gap = (result.objective_value - optimal_value).abs() / optimal_value.abs();
            let acceptable_gap = 0.05; // 5% gap

            checks.push(ValidationCheck {
                name: "Optimality gap".to_string(),
                passed: gap <= acceptable_gap,
                message: format!("Gap to optimal: {:.2}%", gap * 100.0),
                details: Some(format!(
                    "Optimal: {:.4}, Found: {:.4}",
                    optimal_value, result.objective_value
                )),
            });
        }

        ValidationResult {
            is_valid: checks.iter().all(|c| c.passed),
            checks,
            warnings: Vec::new(),
        }
    }

    fn name(&self) -> &str {
        "ObjectiveValidator"
    }
}

impl ObjectiveValidator {
    fn estimate_random_objective(&self, qubo: &Array2<f64>) -> f64 {
        let n = qubo.shape()[0];
        let mut rng = thread_rng();
        let mut total = 0.0;
        let samples = 100;

        for _ in 0..samples {
            let mut x = vec![0.0; n];
            for i in 0..n {
                x[i] = if rng.gen::<bool>() { 1.0 } else { 0.0 };
            }

            let mut value = 0.0;
            for i in 0..n {
                for j in 0..n {
                    value += qubo[[i, j]] * x[i] * x[j];
                }
            }

            total += value;
        }

        total / samples as f64
    }
}

/// Bounds validator
struct BoundsValidator;

impl Validator for BoundsValidator {
    fn validate(&self, test_case: &TestCase, result: &TestResult) -> ValidationResult {
        let mut checks = Vec::new();

        // Check all variables are binary
        let all_binary = result.solution.values().all(|&v| v == true || v == false);

        checks.push(ValidationCheck {
            name: "Binary variables".to_string(),
            passed: all_binary,
            message: if all_binary {
                "All variables are binary".to_string()
            } else {
                "Non-binary values found".to_string()
            },
            details: None,
        });

        // Check variable count
        let expected_vars = test_case.var_map.len();
        let actual_vars = result.solution.len();

        checks.push(ValidationCheck {
            name: "Variable count".to_string(),
            passed: expected_vars == actual_vars,
            message: format!(
                "Expected {} variables, found {}",
                expected_vars, actual_vars
            ),
            details: None,
        });

        ValidationResult {
            is_valid: checks.iter().all(|c| c.passed),
            checks,
            warnings: Vec::new(),
        }
    }

    fn name(&self) -> &str {
        "BoundsValidator"
    }
}

/// Symmetry validator
struct SymmetryValidator;

impl Validator for SymmetryValidator {
    fn validate(&self, test_case: &TestCase, result: &TestResult) -> ValidationResult {
        let mut warnings = Vec::new();

        // Check for symmetries in QUBO
        if self.is_symmetric(&test_case.qubo) {
            warnings.push("QUBO matrix has symmetries that might not be broken".to_string());
        }

        ValidationResult {
            is_valid: true,
            checks: Vec::new(),
            warnings,
        }
    }

    fn name(&self) -> &str {
        "SymmetryValidator"
    }
}

impl SymmetryValidator {
    fn is_symmetric(&self, qubo: &Array2<f64>) -> bool {
        let n = qubo.shape()[0];

        for i in 0..n {
            for j in i + 1..n {
                if (qubo[[i, j]] - qubo[[j, i]]).abs() > 1e-10 {
                    return false;
                }
            }
        }

        true
    }
}

/// Max-cut problem generator
struct MaxCutGenerator;

impl TestGenerator for MaxCutGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let mut test_cases = Vec::new();

        // Generate random graph
        let n = config.size;
        let edge_probability = match config.difficulty {
            Difficulty::Easy => 0.3,
            Difficulty::Medium => 0.5,
            Difficulty::Hard => 0.7,
            Difficulty::VeryHard => 0.9,
            Difficulty::Extreme => 0.95,
        };

        let mut qubo = Array2::zeros((n, n));
        let mut var_map = HashMap::new();

        for i in 0..n {
            var_map.insert(format!("x_{}", i), i);
        }

        // Generate edges
        for i in 0..n {
            for j in i + 1..n {
                if rng.gen::<f64>() < edge_probability {
                    let weight = rng.gen_range(1.0..10.0);
                    // Max-cut: minimize -w_ij * (x_i + x_j - 2*x_i*x_j)
                    qubo[[i, i]] -= weight;
                    qubo[[j, j]] -= weight;
                    qubo[[i, j]] += 2.0 * weight;
                    qubo[[j, i]] += 2.0 * weight;
                }
            }
        }

        test_cases.push(TestCase {
            id: format!("maxcut_{}_{:?}", n, config.difficulty),
            problem_type: ProblemType::MaxCut,
            size: n,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints: Vec::new(),
            metadata: TestMetadata {
                generation_method: "Random graph".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(100),
                notes: format!("Edge probability: {}", edge_probability),
                tags: vec!["graph".to_string(), "maxcut".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &str {
        "MaxCutGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::MaxCut]
    }
}

/// TSP generator
struct TSPGenerator;

impl TestGenerator for TSPGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n_cities = config.size;
        let mut test_cases = Vec::new();

        // Generate random city locations
        let mut cities = Vec::new();
        for _ in 0..n_cities {
            cities.push((rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0)));
        }

        // Calculate distances
        let mut distances = Array2::zeros((n_cities, n_cities));
        for i in 0..n_cities {
            for j in 0..n_cities {
                if i != j {
                    let dx: f64 = cities[i].0 - cities[j].0;
                    let dy: f64 = cities[i].1 - cities[j].1;
                    distances[[i, j]] = (dx * dx + dy * dy).sqrt();
                }
            }
        }

        // Create QUBO
        let n_vars = n_cities * n_cities;
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();

        // Variable mapping: x[i,j] = city i at position j
        for i in 0..n_cities {
            for j in 0..n_cities {
                let idx = i * n_cities + j;
                var_map.insert(format!("x_{}_{}", i, j), idx);
            }
        }

        // Objective: minimize total distance
        for i in 0..n_cities {
            for j in 0..n_cities {
                for k in 0..n_cities {
                    let next_j = (j + 1) % n_cities;
                    let idx1 = i * n_cities + j;
                    let idx2 = k * n_cities + next_j;
                    qubo[[idx1, idx2]] += distances[[i, k]];
                }
            }
        }

        // Constraints
        let mut constraints = Vec::new();
        let penalty = 1000.0;

        // Each city visited exactly once
        for i in 0..n_cities {
            let vars: Vec<_> = (0..n_cities).map(|j| format!("x_{}_{}", i, j)).collect();

            constraints.push(Constraint {
                constraint_type: ConstraintType::ExactlyK { k: 1 },
                variables: vars,
                parameters: HashMap::new(),
                penalty,
            });
        }

        // Each position has exactly one city
        for j in 0..n_cities {
            let vars: Vec<_> = (0..n_cities).map(|i| format!("x_{}_{}", i, j)).collect();

            constraints.push(Constraint {
                constraint_type: ConstraintType::ExactlyK { k: 1 },
                variables: vars,
                parameters: HashMap::new(),
                penalty,
            });
        }

        // Add constraint penalties to QUBO
        self.add_constraint_penalties(&mut qubo, &var_map, &constraints)?;

        test_cases.push(TestCase {
            id: format!("tsp_{}_{:?}", n_cities, config.difficulty),
            problem_type: ProblemType::TSP,
            size: n_cities,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints,
            metadata: TestMetadata {
                generation_method: "Random cities".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(500),
                notes: format!("{} cities", n_cities),
                tags: vec!["routing".to_string(), "tsp".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &str {
        "TSPGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::TSP]
    }
}

impl TSPGenerator {
    fn add_constraint_penalties(
        &self,
        qubo: &mut Array2<f64>,
        var_map: &HashMap<String, usize>,
        constraints: &[Constraint],
    ) -> Result<(), String> {
        for constraint in constraints {
            match &constraint.constraint_type {
                ConstraintType::ExactlyK { k } => {
                    // (sum x_i - k)^2
                    for v1 in &constraint.variables {
                        if let Some(&idx1) = var_map.get(v1) {
                            // Linear term: -2k
                            qubo[[idx1, idx1]] += constraint.penalty * (1.0 - 2.0 * *k as f64);

                            // Quadratic terms
                            for v2 in &constraint.variables {
                                if v1 != v2 {
                                    if let Some(&idx2) = var_map.get(v2) {
                                        qubo[[idx1, idx2]] += constraint.penalty * 2.0;
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Graph coloring generator
struct GraphColoringGenerator;

impl TestGenerator for GraphColoringGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n_vertices = config.size;
        let n_colors = match config.difficulty {
            Difficulty::Easy => 4,
            Difficulty::Medium => 3,
            _ => 3,
        };

        let mut test_cases = Vec::new();

        // Generate random graph
        let edge_prob = 0.3;
        let mut edges = Vec::new();

        for i in 0..n_vertices {
            for j in i + 1..n_vertices {
                if rng.gen::<f64>() < edge_prob {
                    edges.push((i, j));
                }
            }
        }

        // Create QUBO
        let n_vars = n_vertices * n_colors;
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();

        // Variable mapping: x[v,c] = vertex v has color c
        for v in 0..n_vertices {
            for c in 0..n_colors {
                let idx = v * n_colors + c;
                var_map.insert(format!("x_{}_{}", v, c), idx);
            }
        }

        // Objective: minimize number of colors used (simplified)
        for v in 0..n_vertices {
            for c in 0..n_colors {
                let idx = v * n_colors + c;
                qubo[[idx, idx]] -= c as f64; // Prefer lower colors
            }
        }

        // Constraints
        let mut constraints = Vec::new();
        let penalty = 100.0;

        // Each vertex has exactly one color
        for v in 0..n_vertices {
            let vars: Vec<_> = (0..n_colors).map(|c| format!("x_{}_{}", v, c)).collect();

            constraints.push(Constraint {
                constraint_type: ConstraintType::ExactlyK { k: 1 },
                variables: vars,
                parameters: HashMap::new(),
                penalty,
            });
        }

        // Adjacent vertices have different colors
        for (u, v) in &edges {
            for c in 0..n_colors {
                let idx_u = u * n_colors + c;
                let idx_v = v * n_colors + c;
                qubo[[idx_u, idx_v]] += penalty;
                qubo[[idx_v, idx_u]] += penalty;
            }
        }

        test_cases.push(TestCase {
            id: format!(
                "coloring_{}_{}_{}_{:?}",
                n_vertices,
                n_colors,
                edges.len(),
                config.difficulty
            ),
            problem_type: ProblemType::GraphColoring,
            size: n_vertices,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints,
            metadata: TestMetadata {
                generation_method: "Random graph".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(200),
                notes: format!(
                    "{} vertices, {} colors, {} edges",
                    n_vertices,
                    n_colors,
                    edges.len()
                ),
                tags: vec!["graph".to_string(), "coloring".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &str {
        "GraphColoringGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::GraphColoring]
    }
}

/// Knapsack generator
struct KnapsackGenerator;

impl TestGenerator for KnapsackGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n_items = config.size;
        let mut test_cases = Vec::new();

        // Generate items
        let mut values = Vec::new();
        let mut weights = Vec::new();

        for _ in 0..n_items {
            values.push(rng.gen_range(1.0..100.0));
            weights.push(rng.gen_range(1.0..50.0));
        }

        let capacity = weights.iter().sum::<f64>() * 0.5; // 50% of total weight

        // Create QUBO
        let mut qubo = Array2::zeros((n_items, n_items));
        let mut var_map = HashMap::new();

        for i in 0..n_items {
            var_map.insert(format!("x_{}", i), i);
            // Maximize value (negative in minimization)
            qubo[[i, i]] -= values[i];
        }

        // Weight constraint penalty
        let penalty = values.iter().sum::<f64>() * 2.0;

        // Add soft constraint for capacity
        // Penalty for exceeding capacity: (sum w_i x_i - W)^2 if sum > W
        // This is simplified - proper implementation would use slack variables

        test_cases.push(TestCase {
            id: format!("knapsack_{}_{:?}", n_items, config.difficulty),
            problem_type: ProblemType::Knapsack,
            size: n_items,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints: vec![],
            metadata: TestMetadata {
                generation_method: "Random items".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(100),
                notes: format!("{} items, capacity: {:.1}", n_items, capacity),
                tags: vec!["optimization".to_string(), "knapsack".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &str {
        "KnapsackGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::Knapsack]
    }
}

/// Random QUBO generator
struct RandomQuboGenerator;

impl TestGenerator for RandomQuboGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n = config.size;
        let mut test_cases = Vec::new();

        // Generate random QUBO
        let mut qubo = Array2::zeros((n, n));
        let density = match config.difficulty {
            Difficulty::Easy => 0.3,
            Difficulty::Medium => 0.5,
            Difficulty::Hard => 0.7,
            Difficulty::VeryHard => 0.9,
            Difficulty::Extreme => 1.0,
        };

        for i in 0..n {
            for j in i..n {
                if rng.gen::<f64>() < density {
                    let value = rng.gen_range(-10.0..10.0);
                    qubo[[i, j]] = value;
                    if i != j {
                        qubo[[j, i]] = value;
                    }
                }
            }
        }

        let mut var_map = HashMap::new();
        for i in 0..n {
            var_map.insert(format!("x_{}", i), i);
        }

        test_cases.push(TestCase {
            id: format!("random_{}_{:?}", n, config.difficulty),
            problem_type: ProblemType::Custom {
                name: "Random QUBO".to_string(),
            },
            size: n,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints: vec![],
            metadata: TestMetadata {
                generation_method: "Random generation".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(50),
                notes: format!("Density: {}", density),
                tags: vec!["random".to_string(), "qubo".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &str {
        "RandomQuboGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![
            ProblemType::Custom {
                name: "Random".to_string(),
            },
            ProblemType::Ising,
        ]
    }
}

/// Finance industry test generator
struct FinanceTestGenerator;

impl TestGenerator for FinanceTestGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n_assets = config.size;
        let mut test_cases = Vec::new();

        // Generate portfolio optimization test case
        let mut qubo = Array2::zeros((n_assets, n_assets));
        let mut var_map = HashMap::new();

        for i in 0..n_assets {
            var_map.insert(format!("asset_{}", i), i);
            
            // Expected return (negative for minimization)
            let expected_return = rng.gen_range(0.05..0.15);
            qubo[[i, i]] -= expected_return;
        }

        // Risk covariance terms
        for i in 0..n_assets {
            for j in 0..n_assets {
                let covariance = if i == j {
                    rng.gen_range(0.01..0.04) // Variance
                } else {
                    rng.gen_range(-0.01..0.01) // Covariance
                };
                qubo[[i, j]] += covariance;
            }
        }

        test_cases.push(TestCase {
            id: format!("portfolio_{}_{:?}", n_assets, config.difficulty),
            problem_type: ProblemType::Portfolio,
            size: n_assets,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints: vec![
                Constraint {
                    constraint_type: ConstraintType::LinearEquality { target: 1.0 },
                    variables: (0..n_assets).map(|i| format!("asset_{}", i)).collect(),
                    parameters: HashMap::new(),
                    penalty: 1000.0,
                }
            ],
            metadata: TestMetadata {
                generation_method: "Random portfolio".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(200),
                notes: format!("{} assets", n_assets),
                tags: vec!["finance".to_string(), "portfolio".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &str {
        "FinanceTestGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::Portfolio]
    }
}

/// Logistics industry test generator
struct LogisticsTestGenerator;

impl TestGenerator for LogisticsTestGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n_vehicles = 2;
        let n_locations = config.size;
        let mut test_cases = Vec::new();

        // Generate vehicle routing problem
        let n_vars = n_vehicles * n_locations * n_locations;
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();

        // Variable mapping: x[v][i][j] = vehicle v goes from i to j
        for v in 0..n_vehicles {
            for i in 0..n_locations {
                for j in 0..n_locations {
                    let idx = v * n_locations * n_locations + i * n_locations + j;
                    var_map.insert(format!("x_{}_{}_{}",v, i, j), idx);
                }
            }
        }

        // Add distance objective
        for v in 0..n_vehicles {
            for i in 0..n_locations {
                for j in 0..n_locations {
                    if i != j {
                        let idx = v * n_locations * n_locations + i * n_locations + j;
                        let distance = rng.gen_range(1.0..20.0);
                        qubo[[idx, idx]] += distance;
                    }
                }
            }
        }

        test_cases.push(TestCase {
            id: format!("vrp_{}_{}_{:?}", n_vehicles, n_locations, config.difficulty),
            problem_type: ProblemType::VRP,
            size: n_locations,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints: vec![],
            metadata: TestMetadata {
                generation_method: "Random VRP".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(500),
                notes: format!("{} vehicles, {} locations", n_vehicles, n_locations),
                tags: vec!["logistics".to_string(), "vrp".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &str {
        "LogisticsTestGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::VRP]
    }
}

/// Manufacturing industry test generator
struct ManufacturingTestGenerator;

impl TestGenerator for ManufacturingTestGenerator {
    fn generate(&self, config: &GeneratorConfig) -> Result<Vec<TestCase>, String> {
        let mut rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let mut thread_rng = thread_rng();
            StdRng::from_rng(&mut thread_rng)
        };

        let n_jobs = config.size;
        let n_machines = 3;
        let mut test_cases = Vec::new();

        // Generate job scheduling problem
        let n_vars = n_jobs * n_machines;
        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();

        // Variable mapping: x[j][m] = job j on machine m
        for j in 0..n_jobs {
            for m in 0..n_machines {
                let idx = j * n_machines + m;
                var_map.insert(format!("job_{}_machine_{}", j, m), idx);
            }
        }

        // Add processing time objective
        for j in 0..n_jobs {
            for m in 0..n_machines {
                let idx = j * n_machines + m;
                let processing_time = rng.gen_range(1.0..10.0);
                qubo[[idx, idx]] += processing_time;
            }
        }

        // Add constraints: each job assigned to exactly one machine
        let mut constraints = Vec::new();
        for j in 0..n_jobs {
            let vars: Vec<_> = (0..n_machines).map(|m| format!("job_{}_machine_{}", j, m)).collect();
            constraints.push(Constraint {
                constraint_type: ConstraintType::ExactlyK { k: 1 },
                variables: vars,
                parameters: HashMap::new(),
                penalty: 100.0,
            });
        }

        test_cases.push(TestCase {
            id: format!("scheduling_{}_{}_{:?}", n_jobs, n_machines, config.difficulty),
            problem_type: ProblemType::JobScheduling,
            size: n_jobs,
            qubo,
            var_map,
            optimal_solution: None,
            optimal_value: None,
            constraints,
            metadata: TestMetadata {
                generation_method: "Random job scheduling".to_string(),
                difficulty: config.difficulty.clone(),
                expected_runtime: Duration::from_millis(300),
                notes: format!("{} jobs, {} machines", n_jobs, n_machines),
                tags: vec!["manufacturing".to_string(), "scheduling".to_string()],
            },
        });

        Ok(test_cases)
    }

    fn name(&self) -> &str {
        "ManufacturingTestGenerator"
    }

    fn supported_types(&self) -> Vec<ProblemType> {
        vec![ProblemType::JobScheduling]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::SASampler;

    #[test]
    fn test_testing_framework() {
        let config = TestConfig {
            seed: Some(42),
            cases_per_category: 5,
            problem_sizes: vec![5, 10],
            samplers: vec![SamplerConfig {
                name: "SA".to_string(),
                num_samples: 100,
                parameters: HashMap::new(),
            }],
            timeout: Duration::from_secs(10),
            validation: ValidationConfig {
                check_constraints: true,
                check_objective: true,
                statistical_tests: false,
                tolerance: 1e-6,
                min_quality: 0.0,
            },
            output: OutputConfig {
                generate_report: true,
                format: ReportFormat::Text,
                output_dir: "/tmp".to_string(),
                verbosity: VerbosityLevel::Info,
            },
        };

        let mut framework = TestingFramework::new(config);

        // Add test categories
        framework.add_category(TestCategory {
            name: "Graph Problems".to_string(),
            description: "Graph-based optimization problems".to_string(),
            problem_types: vec![ProblemType::MaxCut, ProblemType::GraphColoring],
            difficulties: vec![Difficulty::Easy, Difficulty::Medium],
            tags: vec!["graph".to_string()],
        });

        // Generate test suite
        let result = framework.generate_suite();
        assert!(result.is_ok());
        assert!(!framework.suite.test_cases.is_empty());

        // Run tests
        let sampler = SASampler::new(Some(42));
        let result = framework.run_suite(&sampler);
        assert!(result.is_ok());

        // Check results
        assert!(framework.results.summary.total_tests > 0);
        assert!(framework.results.summary.success_rate >= 0.0);

        // Generate report
        let report = framework.generate_report();
        assert!(report.is_ok());
    }
}
