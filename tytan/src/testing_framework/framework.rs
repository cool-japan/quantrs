//! Core testing framework implementation.
//!
//! This module provides the main TestingFramework struct and its implementation
//! for running tests, managing test suites, and generating reports.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::sampler::Sampler;

use super::config::{ReportFormat, TestConfig};
use super::generators::{
    default_generators, FinanceTestGenerator, LogisticsTestGenerator, ManufacturingTestGenerator,
};
use super::reports;
use super::results::{ConvergenceData, MemoryStats, PerformanceData, RuntimeStats, TestResults};
use super::types::{
    CIReport, CIStatus, Difficulty, FailureType, GeneratorConfig, ProblemType, RegressionIssue,
    RegressionReport, SamplerComparison, TestCase, TestCategory, TestComparison, TestEnvironment,
    TestFailure, TestGenerator, TestResult, TestSuite, ValidationResult, Validator,
};
use super::validators::default_validators;

/// Automated testing framework
pub struct TestingFramework {
    /// Test configuration
    pub config: TestConfig,
    /// Test suite
    pub suite: TestSuite,
    /// Test results
    pub results: TestResults,
    /// Validators
    validators: Vec<Box<dyn Validator>>,
    /// Generators
    generators: Vec<Box<dyn TestGenerator>>,
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
            results: TestResults::default(),
            validators: default_validators(),
            generators: default_generators(),
        }
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
            .map_err(|e| format!("Sampler error: {e:?}"))?;

        let solve_time = solve_start.elapsed();

        // Get best solution
        let best_sample = sample_result
            .iter()
            .min_by(|a, b| {
                a.energy
                    .partial_cmp(&b.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
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
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(&0.0);

        self.results.summary.quality_metrics.worst_quality = *qualities
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
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
        reports::generate_report(&self.config.output.format, &self.results, &self.suite)
    }

    /// Save report to file
    pub fn save_report(&self, filename: &str) -> Result<(), String> {
        let report = self.generate_report()?;
        reports::save_report(&report, filename)
    }

    /// Run regression tests against baseline
    pub fn run_regression_tests<S: Sampler>(
        &mut self,
        sampler: &S,
        baseline_file: &str,
    ) -> Result<RegressionReport, String> {
        // Load baseline results
        let baseline = self.load_baseline(baseline_file)?;

        // Run current tests
        self.run_suite(sampler)?;

        // Compare with baseline
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();

        for current_result in &self.results.test_results {
            if let Some(baseline_result) = baseline
                .iter()
                .find(|b| b.test_id == current_result.test_id)
            {
                let quality_change = (current_result.objective_value
                    - baseline_result.objective_value)
                    / baseline_result.objective_value.abs();
                let runtime_change = (current_result.runtime.as_secs_f64()
                    - baseline_result.runtime.as_secs_f64())
                    / baseline_result.runtime.as_secs_f64();

                if quality_change > 0.05 || runtime_change > 0.2 {
                    regressions.push(RegressionIssue {
                        test_id: current_result.test_id.clone(),
                        metric: if quality_change > 0.05 {
                            "quality".to_string()
                        } else {
                            "runtime".to_string()
                        },
                        baseline_value: if quality_change > 0.05 {
                            baseline_result.objective_value
                        } else {
                            baseline_result.runtime.as_secs_f64()
                        },
                        current_value: if quality_change > 0.05 {
                            current_result.objective_value
                        } else {
                            current_result.runtime.as_secs_f64()
                        },
                        change_percent: if quality_change > 0.05 {
                            quality_change * 100.0
                        } else {
                            runtime_change * 100.0
                        },
                    });
                } else if quality_change < -0.05 || runtime_change < -0.2 {
                    improvements.push(RegressionIssue {
                        test_id: current_result.test_id.clone(),
                        metric: if quality_change < -0.05 {
                            "quality".to_string()
                        } else {
                            "runtime".to_string()
                        },
                        baseline_value: if quality_change < -0.05 {
                            baseline_result.objective_value
                        } else {
                            baseline_result.runtime.as_secs_f64()
                        },
                        current_value: if quality_change < -0.05 {
                            current_result.objective_value
                        } else {
                            current_result.runtime.as_secs_f64()
                        },
                        change_percent: if quality_change < -0.05 {
                            quality_change * 100.0
                        } else {
                            runtime_change * 100.0
                        },
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
    const fn load_baseline(&self, _filename: &str) -> Result<Vec<TestResult>, String> {
        // Simplified implementation - in practice would load from JSON/CSV
        Ok(Vec::new())
    }

    /// Run test suite in parallel
    pub fn run_suite_parallel<S: Sampler + Clone + Send + Sync + 'static>(
        &mut self,
        sampler: &S,
        num_threads: usize,
    ) -> Result<(), String> {
        let test_cases = Arc::new(self.suite.test_cases.clone());
        let results = Arc::new(Mutex::new(Vec::new()));
        let failures = Arc::new(Mutex::new(Vec::new()));

        let total_start = Instant::now();
        let chunk_size = test_cases.len().div_ceil(num_threads);

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
                            if let Ok(mut guard) = results_clone.lock() {
                                guard.push(result);
                            }
                        }
                        Err(e) => {
                            if let Ok(mut guard) = failures_clone.lock() {
                                guard.push(TestFailure {
                                    test_id: test_case.id.clone(),
                                    failure_type: FailureType::SamplerError,
                                    message: e,
                                    stack_trace: None,
                                    context: HashMap::new(),
                                });
                            }
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
        self.results.test_results = results
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        self.results.failures = failures
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();

        self.results.performance.runtime_stats.total_time = total_start.elapsed();
        self.results.summary.passed = self.results.test_results.len();
        self.results.summary.failed = self.results.failures.len();
        self.results.summary.total_tests =
            self.results.summary.passed + self.results.summary.failed;

        self.calculate_summary();

        Ok(())
    }

    /// Static version of run_single_test for parallel execution
    fn run_single_test_static<S: Sampler>(
        test_case: &TestCase,
        sampler: &S,
    ) -> Result<TestResult, String> {
        let solve_start = Instant::now();

        // Run sampler
        let sample_result = sampler
            .run_qubo(&(test_case.qubo.clone(), test_case.var_map.clone()), 100)
            .map_err(|e| format!("Sampler error: {e:?}"))?;

        let solve_time = solve_start.elapsed();

        // Get best solution
        let best_sample = sample_result
            .iter()
            .min_by(|a, b| {
                a.energy
                    .partial_cmp(&b.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
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
            critical_failures: self
                .results
                .failures
                .iter()
                .filter(|f| {
                    matches!(
                        f.failure_type,
                        FailureType::Timeout | FailureType::SamplerError
                    )
                })
                .count(),
            avg_runtime: self.results.summary.avg_runtime,
            quality_score: self.calculate_quality_score(),
        })
    }

    /// Calculate overall quality score
    fn calculate_quality_score(&self) -> f64 {
        if self.results.test_results.is_empty() {
            return 0.0;
        }

        let constraint_score = self
            .results
            .summary
            .quality_metrics
            .constraint_satisfaction_rate;
        let success_score = self.results.summary.success_rate;
        let quality_score = if self
            .results
            .summary
            .quality_metrics
            .best_quality
            .is_finite()
        {
            0.8 // Base score for having finite solutions
        } else {
            0.0
        };

        (constraint_score.mul_add(0.4, success_score * 0.4) + quality_score * 0.2) * 100.0
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
            memory_gb: 8.0,                   // Simplified - would need system detection
            gpu_info: None,
            rust_version: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
            compile_flags: vec!["--release".to_string()],
        }
    }

    /// Export test results for external analysis
    pub fn export_results(&self, format: &str) -> Result<String, String> {
        match format {
            "csv" => reports::export_csv(&self.results, &self.suite),
            "json" => reports::generate_json_report(&self.results),
            "xml" => reports::export_xml(&self.results),
            _ => Err(format!("Unsupported export format: {format}")),
        }
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
    pub fn compare_samplers<S1: Sampler, S2: Sampler>(
        &mut self,
        sampler1: &S1,
        sampler2: &S2,
        sampler1_name: &str,
        sampler2_name: &str,
    ) -> Result<SamplerComparison, String> {
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

        let avg_quality_improvement = comparisons
            .iter()
            .map(|c| c.quality_improvement)
            .sum::<f64>()
            / comparisons.len() as f64;
        let avg_runtime_ratio =
            comparisons.iter().map(|c| c.runtime_ratio).sum::<f64>() / comparisons.len() as f64;

        Ok(SamplerComparison {
            sampler1_name: sampler1_name.to_string(),
            sampler2_name: sampler2_name.to_string(),
            test_comparisons: comparisons,
            avg_quality_improvement,
            avg_runtime_ratio,
            winner: if avg_quality_improvement > 0.0 {
                sampler2_name.to_string()
            } else {
                sampler1_name.to_string()
            },
        })
    }
}
