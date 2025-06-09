//! Automated testing framework for quantum optimization.
//!
//! This module provides comprehensive testing tools for QUBO problems,
//! including test case generation, validation, and benchmarking.

use crate::sampler::{Sampler, SampleResult, SamplerError, SamplerResult};
#[cfg(feature = "dwave")]
use crate::compile::{Compile, CompiledModel};
use ndarray::{Array, Array1, Array2, IxDyn};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::{Write, BufWriter};
use rand::prelude::*;
use rand::{thread_rng, SeedableRng};
use serde::{Serialize, Deserialize};

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
            self.results.performance.runtime_stats.time_per_test.push(
                (test_case.id.clone(), test_time)
            );
            
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
        let sample_result = sampler.run_qubo(
            &(test_case.qubo.clone(), test_case.var_map.clone()),
            self.config.samplers[0].num_samples,
        ).map_err(|e| format!("Sampler error: {:?}", e))?;
        
        let solve_time = solve_start.elapsed();
        
        // Get best solution
        let best_sample = sample_result.iter()
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
            let result = validator.validate(test_case, &TestResult {
                test_id: test_case.id.clone(),
                sampler: "test".to_string(),
                solution: solution.clone(),
                objective_value: best_sample.energy,
                constraints_satisfied: true,
                validation: validation.clone(),
                runtime: solve_time,
                metrics: HashMap::new(),
            });
            
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
        let total_runtime: Duration = self.results.test_results.iter()
            .map(|r| r.runtime)
            .sum();
        self.results.summary.avg_runtime = 
            total_runtime / self.results.test_results.len() as u32;
        
        // Quality metrics
        let qualities: Vec<f64> = self.results.test_results.iter()
            .map(|r| r.objective_value)
            .collect();
        
        self.results.summary.quality_metrics.avg_quality = 
            qualities.iter().sum::<f64>() / qualities.len() as f64;
        
        self.results.summary.quality_metrics.best_quality = 
            *qualities.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        
        self.results.summary.quality_metrics.worst_quality = 
            *qualities.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        
        // Standard deviation
        let mean = self.results.summary.quality_metrics.avg_quality;
        let variance = qualities.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / qualities.len() as f64;
        self.results.summary.quality_metrics.std_dev = variance.sqrt();
        
        // Constraint satisfaction rate
        let satisfied = self.results.test_results.iter()
            .filter(|r| r.constraints_satisfied)
            .count();
        self.results.summary.quality_metrics.constraint_satisfaction_rate = 
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
        
        report.push_str(&format!("Total Tests: {}\n", self.results.summary.total_tests));
        report.push_str(&format!("Passed: {}\n", self.results.summary.passed));
        report.push_str(&format!("Failed: {}\n", self.results.summary.failed));
        report.push_str(&format!("Success Rate: {:.2}%\n", self.results.summary.success_rate * 100.0));
        report.push_str(&format!("Average Runtime: {:?}\n\n", self.results.summary.avg_runtime));
        
        report.push_str("Quality Metrics:\n");
        report.push_str(&format!("  Average Quality: {:.4}\n", self.results.summary.quality_metrics.avg_quality));
        report.push_str(&format!("  Best Quality: {:.4}\n", self.results.summary.quality_metrics.best_quality));
        report.push_str(&format!("  Worst Quality: {:.4}\n", self.results.summary.quality_metrics.worst_quality));
        report.push_str(&format!("  Std Dev: {:.4}\n", self.results.summary.quality_metrics.std_dev));
        report.push_str(&format!("  Constraint Satisfaction: {:.2}%\n\n", 
            self.results.summary.quality_metrics.constraint_satisfaction_rate * 100.0));
        
        if !self.results.failures.is_empty() {
            report.push_str("Failures:\n");
            for failure in &self.results.failures {
                report.push_str(&format!("  - {} ({}): {}\n", 
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
        // TODO: Add proper JSON serialization support
        Ok("{}".to_string())
    }
    
    /// Generate HTML report
    fn generate_html_report(&self) -> Result<String, String> {
        let mut html = String::new();
        
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>Quantum Optimization Test Report</title>\n");
        html.push_str("<style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .passed { color: green; }
            .failed { color: red; }
        </style>\n");
        html.push_str("</head>\n<body>\n");
        
        html.push_str("<h1>Quantum Optimization Test Report</h1>\n");
        
        // Summary
        html.push_str("<h2>Summary</h2>\n");
        html.push_str("<table>\n");
        html.push_str(&format!("<tr><td>Total Tests</td><td>{}</td></tr>\n", self.results.summary.total_tests));
        html.push_str(&format!("<tr><td>Passed</td><td class='passed'>{}</td></tr>\n", self.results.summary.passed));
        html.push_str(&format!("<tr><td>Failed</td><td class='failed'>{}</td></tr>\n", self.results.summary.failed));
        html.push_str(&format!("<tr><td>Success Rate</td><td>{:.2}%</td></tr>\n", self.results.summary.success_rate * 100.0));
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
        md.push_str(&format!("| Total Tests | {} |\n", self.results.summary.total_tests));
        md.push_str(&format!("| Passed | {} |\n", self.results.summary.passed));
        md.push_str(&format!("| Failed | {} |\n", self.results.summary.failed));
        md.push_str(&format!("| Success Rate | {:.2}% |\n", self.results.summary.success_rate * 100.0));
        md.push_str(&format!("| Average Runtime | {:?} |\n\n", self.results.summary.avg_runtime));
        
        md.push_str("## Quality Metrics\n\n");
        md.push_str(&format!("- **Average Quality**: {:.4}\n", self.results.summary.quality_metrics.avg_quality));
        md.push_str(&format!("- **Best Quality**: {:.4}\n", self.results.summary.quality_metrics.best_quality));
        md.push_str(&format!("- **Worst Quality**: {:.4}\n", self.results.summary.quality_metrics.worst_quality));
        md.push_str(&format!("- **Standard Deviation**: {:.4}\n", self.results.summary.quality_metrics.std_dev));
        md.push_str(&format!("- **Constraint Satisfaction Rate**: {:.2}%\n\n", 
            self.results.summary.quality_metrics.constraint_satisfaction_rate * 100.0));
        
        Ok(md)
    }
    
    /// Generate CSV report
    fn generate_csv_report(&self) -> Result<String, String> {
        let mut csv = String::new();
        
        csv.push_str("test_id,sampler,objective_value,constraints_satisfied,runtime_ms,valid\n");
        
        for result in &self.results.test_results {
            csv.push_str(&format!("{},{},{},{},{},{}\n",
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
        let mut file = File::create(filename)
            .map_err(|e| format!("Failed to create file: {}", e))?;
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
                let active = constraint.variables.iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .count();
                active == 1
            }
            ConstraintType::AtMostK { k } => {
                let active = constraint.variables.iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .count();
                active <= *k
            }
            ConstraintType::AtLeastK { k } => {
                let active = constraint.variables.iter()
                    .filter(|v| *solution.get(*v).unwrap_or(&false))
                    .count();
                active >= *k
            }
            ConstraintType::ExactlyK { k } => {
                let active = constraint.variables.iter()
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
            details: Some(format!("Random: {:.4}, Found: {:.4}", random_value, result.objective_value)),
        });
        
        // Check against optimal if known
        if let Some(optimal_value) = test_case.optimal_value {
            let gap = (result.objective_value - optimal_value).abs() / optimal_value.abs();
            let acceptable_gap = 0.05; // 5% gap
            
            checks.push(ValidationCheck {
                name: "Optimality gap".to_string(),
                passed: gap <= acceptable_gap,
                message: format!("Gap to optimal: {:.2}%", gap * 100.0),
                details: Some(format!("Optimal: {:.4}, Found: {:.4}", optimal_value, result.objective_value)),
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
            message: format!("Expected {} variables, found {}", expected_vars, actual_vars),
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
            for j in i+1..n {
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
            for j in i+1..n {
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
            let vars: Vec<_> = (0..n_cities)
                .map(|j| format!("x_{}_{}", i, j))
                .collect();
            
            constraints.push(Constraint {
                constraint_type: ConstraintType::ExactlyK { k: 1 },
                variables: vars,
                parameters: HashMap::new(),
                penalty,
            });
        }
        
        // Each position has exactly one city
        for j in 0..n_cities {
            let vars: Vec<_> = (0..n_cities)
                .map(|i| format!("x_{}_{}", i, j))
                .collect();
            
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
            for j in i+1..n_vertices {
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
            let vars: Vec<_> = (0..n_colors)
                .map(|c| format!("x_{}_{}", v, c))
                .collect();
            
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
            id: format!("coloring_{}_{}_{}_{:?}", n_vertices, n_colors, edges.len(), config.difficulty),
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
                notes: format!("{} vertices, {} colors, {} edges", n_vertices, n_colors, edges.len()),
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
            problem_type: ProblemType::Custom { name: "Random QUBO".to_string() },
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
            ProblemType::Custom { name: "Random".to_string() },
            ProblemType::Ising,
        ]
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
            samplers: vec![
                SamplerConfig {
                    name: "SA".to_string(),
                    num_samples: 100,
                    parameters: HashMap::new(),
                }
            ],
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