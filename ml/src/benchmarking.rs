//! Unified benchmarking framework for quantum machine learning
//!
//! This module provides comprehensive benchmarking capabilities for quantum ML
//! algorithms, including performance metrics, scalability analysis, and
//! quantum advantage assessment.

use crate::circuit_integration::QuantumMLExecutor;
use crate::error::{MLError, Result};
use crate::simulator_backends::{Backend, BackendCapabilities, SimulatorBackend};
use ndarray::{Array1, Array2};
use quantrs2_core::prelude::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Comprehensive benchmarking framework
pub struct BenchmarkFramework {
    /// Registered benchmarks
    benchmarks: HashMap<String, Box<dyn Benchmark>>,
    /// Results storage
    results: BenchmarkResults,
    /// Configuration
    config: BenchmarkConfig,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of repetitions per benchmark
    pub repetitions: usize,
    /// Warm-up runs before timing
    pub warmup_runs: usize,
    /// Maximum time per benchmark (seconds)
    pub max_time_per_benchmark: f64,
    /// Include memory profiling
    pub profile_memory: bool,
    /// Include convergence analysis
    pub analyze_convergence: bool,
    /// Statistical confidence level
    pub confidence_level: f64,
    /// Output directory for benchmark results
    pub output_directory: String,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            repetitions: 10,
            warmup_runs: 3,
            max_time_per_benchmark: 300.0, // 5 minutes
            profile_memory: true,
            analyze_convergence: true,
            confidence_level: 0.95,
            output_directory: "/tmp/quantum_ml_benchmarks".to_string(),
        }
    }
}

/// Benchmark trait
pub trait Benchmark: Send + Sync {
    /// Get benchmark name
    fn name(&self) -> &str;

    /// Get benchmark description
    fn description(&self) -> &str;

    /// Setup benchmark (called once before all runs)
    fn setup(&mut self) -> Result<()>;

    /// Run single benchmark iteration
    fn run(&mut self, backend: &Backend) -> Result<BenchmarkRunResult>;

    /// Cleanup after benchmark
    fn cleanup(&mut self) -> Result<()>;

    /// Get expected scaling behavior
    fn expected_scaling(&self) -> ScalingType;

    /// Get benchmark category
    fn category(&self) -> BenchmarkCategory;
}

/// Benchmark run result
#[derive(Debug, Clone)]
pub struct BenchmarkRunResult {
    /// Execution time
    pub execution_time: Duration,
    /// Memory usage (bytes)
    pub memory_usage: Option<usize>,
    /// Algorithm-specific metrics
    pub metrics: HashMap<String, f64>,
    /// Final result value
    pub result_value: Option<f64>,
    /// Success flag
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Scaling behavior types
#[derive(Debug, Clone, Copy)]
pub enum ScalingType {
    /// Polynomial scaling O(n^k)
    Polynomial(f64),
    /// Exponential scaling O(2^n)
    Exponential,
    /// Logarithmic scaling O(log n)
    Logarithmic,
    /// Linear scaling O(n)
    Linear,
    /// Constant time O(1)
    Constant,
}

/// Benchmark categories
#[derive(Debug, Clone, Copy)]
pub enum BenchmarkCategory {
    /// Algorithm performance
    Algorithm,
    /// Hardware simulation
    Hardware,
    /// Memory usage
    Memory,
    /// Scalability
    Scalability,
    /// Accuracy
    Accuracy,
    /// Quantum advantage
    QuantumAdvantage,
}

/// Aggregated benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Results by benchmark name
    results: HashMap<String, Vec<BenchmarkRunResult>>,
    /// Statistical summaries
    summaries: HashMap<String, BenchmarkSummary>,
    /// Metadata
    metadata: BenchmarkMetadata,
}

/// Statistical summary of benchmark runs
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    /// Mean execution time
    pub mean_time: Duration,
    /// Standard deviation of execution time
    pub std_time: Duration,
    /// Minimum execution time
    pub min_time: Duration,
    /// Maximum execution time
    pub max_time: Duration,
    /// Mean memory usage
    pub mean_memory: Option<usize>,
    /// Success rate
    pub success_rate: f64,
    /// Confidence interval for mean time
    pub confidence_interval: (Duration, Duration),
}

/// Benchmark metadata
#[derive(Debug, Clone)]
pub struct BenchmarkMetadata {
    /// System information
    pub system_info: SystemInfo,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
    /// Configuration used
    pub config: BenchmarkConfig,
}

/// System information
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Available memory (bytes)
    pub available_memory: usize,
    /// Operating system
    pub os: String,
    /// Architecture
    pub arch: String,
}

impl BenchmarkFramework {
    /// Create new benchmark framework
    pub fn new() -> Self {
        Self {
            benchmarks: HashMap::new(),
            results: BenchmarkResults {
                results: HashMap::new(),
                summaries: HashMap::new(),
                metadata: BenchmarkMetadata {
                    system_info: SystemInfo::collect(),
                    timestamp: std::time::SystemTime::now(),
                    config: BenchmarkConfig::default(),
                },
            },
            config: BenchmarkConfig::default(),
        }
    }

    /// Set benchmark configuration
    pub fn with_config(mut self, config: BenchmarkConfig) -> Self {
        self.config = config.clone();
        self.results.metadata.config = config;
        self
    }

    /// Register a benchmark
    pub fn register_benchmark(&mut self, name: impl Into<String>, benchmark: Box<dyn Benchmark>) {
        self.benchmarks.insert(name.into(), benchmark);
    }

    /// Run all benchmarks
    pub fn run_all_benchmarks(&mut self, backends: &[&Backend]) -> Result<&BenchmarkResults> {
        // Collect benchmark names first to avoid borrow conflicts
        let benchmark_names: Vec<String> = self.benchmarks.keys().cloned().collect();
        for name in benchmark_names {
            println!("Running benchmark: {}", name);

            // Extract benchmark temporarily to avoid borrow conflicts
            if let Some(mut benchmark) = self.benchmarks.remove(&name) {
                self.run_benchmark(&name, benchmark.as_mut(), backends)?;
                self.benchmarks.insert(name, benchmark);
            }
        }

        self.compute_summaries()?;
        Ok(&self.results)
    }

    /// Run specific benchmark
    pub fn run_benchmark(
        &mut self,
        name: &str,
        benchmark: &mut dyn Benchmark,
        backends: &[&Backend],
    ) -> Result<()> {
        benchmark.setup()?;

        for backend in backends {
            let backend_name = format!("{}_{}", name, backend.name());
            let mut runs = Vec::new();

            // Warm-up runs
            for _ in 0..self.config.warmup_runs {
                let _ = benchmark.run(*backend);
            }

            // Actual benchmark runs
            for run_idx in 0..self.config.repetitions {
                let start_time = Instant::now();

                let result = match benchmark.run(*backend) {
                    Ok(mut result) => {
                        result.execution_time = start_time.elapsed();
                        result
                    }
                    Err(e) => BenchmarkRunResult {
                        execution_time: start_time.elapsed(),
                        memory_usage: None,
                        metrics: HashMap::new(),
                        result_value: None,
                        success: false,
                        error_message: Some(e.to_string()),
                    },
                };

                runs.push(result);

                // Check timeout
                if start_time.elapsed().as_secs_f64() > self.config.max_time_per_benchmark {
                    println!(
                        "Benchmark {} timed out after {} runs",
                        backend_name,
                        run_idx + 1
                    );
                    break;
                }
            }

            self.results.results.insert(backend_name, runs);
        }

        benchmark.cleanup()?;
        Ok(())
    }

    /// Compute statistical summaries
    fn compute_summaries(&mut self) -> Result<()> {
        for (name, runs) in &self.results.results {
            let summary = self.compute_summary(runs)?;
            self.results.summaries.insert(name.clone(), summary);
        }
        Ok(())
    }

    /// Compute summary statistics for benchmark runs
    fn compute_summary(&self, runs: &[BenchmarkRunResult]) -> Result<BenchmarkSummary> {
        if runs.is_empty() {
            return Err(MLError::InvalidConfiguration(
                "No benchmark runs".to_string(),
            ));
        }

        let successful_runs: Vec<_> = runs.iter().filter(|r| r.success).collect();
        let success_rate = successful_runs.len() as f64 / runs.len() as f64;

        if successful_runs.is_empty() {
            return Ok(BenchmarkSummary {
                mean_time: Duration::from_secs(0),
                std_time: Duration::from_secs(0),
                min_time: Duration::from_secs(0),
                max_time: Duration::from_secs(0),
                mean_memory: None,
                success_rate,
                confidence_interval: (Duration::from_secs(0), Duration::from_secs(0)),
            });
        }

        let times: Vec<f64> = successful_runs
            .iter()
            .map(|r| r.execution_time.as_secs_f64())
            .collect();

        let mean_time_secs = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times
            .iter()
            .map(|t| (t - mean_time_secs).powi(2))
            .sum::<f64>()
            / times.len() as f64;
        let std_time_secs = variance.sqrt();

        let min_time_secs = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_time_secs = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Compute confidence interval (assuming normal distribution)
        let z_score = 1.96; // 95% confidence
        let margin = z_score * std_time_secs / (times.len() as f64).sqrt();
        let ci_lower = (mean_time_secs - margin).max(0.0);
        let ci_upper = mean_time_secs + margin;

        // Memory statistics
        let memories: Vec<usize> = successful_runs
            .iter()
            .filter_map(|r| r.memory_usage)
            .collect();
        let mean_memory = if memories.is_empty() {
            None
        } else {
            Some(memories.iter().sum::<usize>() / memories.len())
        };

        Ok(BenchmarkSummary {
            mean_time: Duration::from_secs_f64(mean_time_secs),
            std_time: Duration::from_secs_f64(std_time_secs),
            min_time: Duration::from_secs_f64(min_time_secs),
            max_time: Duration::from_secs_f64(max_time_secs),
            mean_memory,
            success_rate,
            confidence_interval: (
                Duration::from_secs_f64(ci_lower),
                Duration::from_secs_f64(ci_upper),
            ),
        })
    }

    /// Generate benchmark report
    pub fn generate_report(&self) -> BenchmarkReport {
        BenchmarkReport::new(&self.results)
    }
}

/// System information collector
impl SystemInfo {
    fn collect() -> Self {
        Self {
            cpu_cores: num_cpus::get(),
            available_memory: Self::get_available_memory(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
        }
    }

    fn get_available_memory() -> usize {
        // Simplified - would use proper system calls
        8 * 1024 * 1024 * 1024 // 8 GB default
    }
}

/// Quantum ML algorithm benchmarks
pub mod algorithm_benchmarks {
    use super::*;
    use crate::qnn::QuantumNeuralNetwork;
    use crate::variational::VariationalCircuit;

    /// VQE benchmark
    pub struct VQEBenchmark {
        num_qubits: usize,
        num_parameters: usize,
        circuit: Option<VariationalCircuit>,
    }

    impl VQEBenchmark {
        pub fn new(num_qubits: usize, num_parameters: usize) -> Self {
            Self {
                num_qubits,
                num_parameters,
                circuit: None,
            }
        }
    }

    impl Benchmark for VQEBenchmark {
        fn name(&self) -> &str {
            "VQE"
        }

        fn description(&self) -> &str {
            "Variational Quantum Eigensolver benchmark"
        }

        fn setup(&mut self) -> Result<()> {
            // Setup VQE circuit
            self.circuit = Some(VariationalCircuit::new(
                self.num_qubits,
                2 * self.num_qubits, // Default number of parameters
                2,                   // Default number of layers
                crate::variational::AnsatzType::HardwareEfficient,
            )?);
            Ok(())
        }

        fn run(&mut self, backend: &Backend) -> Result<BenchmarkRunResult> {
            let start = Instant::now();
            let parameters: Array1<f64> = Array1::zeros(self.num_parameters);

            // Simulate VQE optimization step
            let mut metrics = HashMap::new();
            metrics.insert("num_qubits".to_string(), self.num_qubits as f64);
            metrics.insert("num_parameters".to_string(), self.num_parameters as f64);

            // Would run actual VQE here
            let result_value = Some(fastrand::f64() - 0.5); // Placeholder energy

            Ok(BenchmarkRunResult {
                execution_time: start.elapsed(),
                memory_usage: Some(self.estimate_memory_usage()),
                metrics,
                result_value,
                success: true,
                error_message: None,
            })
        }

        fn cleanup(&mut self) -> Result<()> {
            self.circuit = None;
            Ok(())
        }

        fn expected_scaling(&self) -> ScalingType {
            ScalingType::Exponential
        }

        fn category(&self) -> BenchmarkCategory {
            BenchmarkCategory::Algorithm
        }
    }

    impl VQEBenchmark {
        fn estimate_memory_usage(&self) -> usize {
            // Rough estimate: 2^n * 16 bytes for statevector
            2_usize.pow(self.num_qubits as u32) * 16
        }
    }

    /// QAOA benchmark
    pub struct QAOABenchmark {
        num_qubits: usize,
        num_layers: usize,
        problem_size: usize,
    }

    impl QAOABenchmark {
        pub fn new(num_qubits: usize, num_layers: usize, problem_size: usize) -> Self {
            Self {
                num_qubits,
                num_layers,
                problem_size,
            }
        }
    }

    impl Benchmark for QAOABenchmark {
        fn name(&self) -> &str {
            "QAOA"
        }

        fn description(&self) -> &str {
            "Quantum Approximate Optimization Algorithm benchmark"
        }

        fn setup(&mut self) -> Result<()> {
            Ok(())
        }

        fn run(&mut self, _backend: &Backend) -> Result<BenchmarkRunResult> {
            let start = Instant::now();

            let mut metrics = HashMap::new();
            metrics.insert("num_qubits".to_string(), self.num_qubits as f64);
            metrics.insert("num_layers".to_string(), self.num_layers as f64);
            metrics.insert("problem_size".to_string(), self.problem_size as f64);

            // Simulate QAOA execution
            let result_value = Some(fastrand::f64()); // Placeholder approximation ratio

            Ok(BenchmarkRunResult {
                execution_time: start.elapsed(),
                memory_usage: Some(2_usize.pow(self.num_qubits as u32) * 16),
                metrics,
                result_value,
                success: true,
                error_message: None,
            })
        }

        fn cleanup(&mut self) -> Result<()> {
            Ok(())
        }

        fn expected_scaling(&self) -> ScalingType {
            ScalingType::Polynomial(2.0)
        }

        fn category(&self) -> BenchmarkCategory {
            BenchmarkCategory::Algorithm
        }
    }

    /// Quantum Neural Network benchmark
    pub struct QNNBenchmark {
        num_qubits: usize,
        num_layers: usize,
        training_samples: usize,
    }

    impl QNNBenchmark {
        pub fn new(num_qubits: usize, num_layers: usize, training_samples: usize) -> Self {
            Self {
                num_qubits,
                num_layers,
                training_samples,
            }
        }
    }

    impl Benchmark for QNNBenchmark {
        fn name(&self) -> &str {
            "QNN"
        }

        fn description(&self) -> &str {
            "Quantum Neural Network training benchmark"
        }

        fn setup(&mut self) -> Result<()> {
            Ok(())
        }

        fn run(&mut self, _backend: &Backend) -> Result<BenchmarkRunResult> {
            let start = Instant::now();

            let mut metrics = HashMap::new();
            metrics.insert("num_qubits".to_string(), self.num_qubits as f64);
            metrics.insert("num_layers".to_string(), self.num_layers as f64);
            metrics.insert("training_samples".to_string(), self.training_samples as f64);

            // Simulate QNN training
            let result_value = Some(1.0 - fastrand::f64() * 0.3); // Placeholder accuracy

            Ok(BenchmarkRunResult {
                execution_time: start.elapsed(),
                memory_usage: Some(2_usize.pow(self.num_qubits as u32) * 16),
                metrics,
                result_value,
                success: true,
                error_message: None,
            })
        }

        fn cleanup(&mut self) -> Result<()> {
            Ok(())
        }

        fn expected_scaling(&self) -> ScalingType {
            ScalingType::Polynomial(3.0)
        }

        fn category(&self) -> BenchmarkCategory {
            BenchmarkCategory::Algorithm
        }
    }
}

/// Benchmark report generation
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Summary statistics
    summaries: HashMap<String, BenchmarkSummary>,
    /// Comparative analysis
    comparisons: Vec<BenchmarkComparison>,
    /// Scaling analysis
    scaling_analysis: HashMap<String, ScalingAnalysis>,
    /// Recommendations
    recommendations: Vec<String>,
}

/// Benchmark comparison
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    /// Benchmark names being compared
    benchmarks: Vec<String>,
    /// Comparison metrics
    metrics: HashMap<String, f64>,
    /// Winner (if applicable)
    winner: Option<String>,
}

/// Scaling analysis result
#[derive(Debug, Clone)]
pub struct ScalingAnalysis {
    /// Observed scaling type
    observed_scaling: ScalingType,
    /// Expected scaling type
    expected_scaling: ScalingType,
    /// Scaling coefficient
    scaling_coefficient: f64,
    /// R-squared for fit
    r_squared: f64,
}

impl BenchmarkReport {
    /// Create new benchmark report
    pub fn new(results: &BenchmarkResults) -> Self {
        let mut report = Self {
            summaries: results.summaries.clone(),
            comparisons: Vec::new(),
            scaling_analysis: HashMap::new(),
            recommendations: Vec::new(),
        };

        report.analyze_comparisons();
        report.analyze_scaling();
        report.generate_recommendations();

        report
    }

    /// Analyze benchmark comparisons
    fn analyze_comparisons(&mut self) {
        // Group benchmarks by algorithm type and compare across backends
        let mut algorithm_groups: HashMap<String, Vec<String>> = HashMap::new();

        for benchmark_name in self.summaries.keys() {
            let algorithm = benchmark_name.split('_').next().unwrap_or(benchmark_name);
            algorithm_groups
                .entry(algorithm.to_string())
                .or_insert_with(Vec::new)
                .push(benchmark_name.clone());
        }

        for (algorithm, benchmarks) in algorithm_groups {
            if benchmarks.len() > 1 {
                let comparison = self.compare_benchmarks(&benchmarks);
                self.comparisons.push(comparison);
            }
        }
    }

    /// Compare benchmarks
    fn compare_benchmarks(&self, benchmark_names: &[String]) -> BenchmarkComparison {
        let mut metrics = HashMap::new();
        let mut winner = None;
        let mut best_time = Duration::from_secs(u64::MAX);

        for name in benchmark_names {
            if let Some(summary) = self.summaries.get(name) {
                metrics.insert(
                    format!("{}_mean_time", name),
                    summary.mean_time.as_secs_f64(),
                );
                metrics.insert(format!("{}_success_rate", name), summary.success_rate);

                if summary.mean_time < best_time && summary.success_rate > 0.8 {
                    best_time = summary.mean_time;
                    winner = Some(name.clone());
                }
            }
        }

        BenchmarkComparison {
            benchmarks: benchmark_names.to_vec(),
            metrics,
            winner,
        }
    }

    /// Analyze scaling behavior
    fn analyze_scaling(&mut self) {
        // Placeholder - would implement actual scaling analysis
        for benchmark_name in self.summaries.keys() {
            let analysis = ScalingAnalysis {
                observed_scaling: ScalingType::Exponential,
                expected_scaling: ScalingType::Exponential,
                scaling_coefficient: 2.0,
                r_squared: 0.95,
            };
            self.scaling_analysis
                .insert(benchmark_name.clone(), analysis);
        }
    }

    /// Generate recommendations
    fn generate_recommendations(&mut self) {
        // Analyze results and generate recommendations
        for comparison in &self.comparisons {
            if let Some(ref winner) = comparison.winner {
                self.recommendations.push(format!(
                    "For {} algorithms, {} backend shows best performance",
                    comparison.benchmarks[0].split('_').next().unwrap_or(""),
                    winner.split('_').last().unwrap_or("")
                ));
            }
        }

        // Add general recommendations
        self.recommendations
            .push("Use statevector backend for small circuits (<20 qubits)".to_string());
        self.recommendations
            .push("Use MPS backend for large circuits with limited entanglement".to_string());
        self.recommendations
            .push("Consider GPU acceleration for repeated circuit evaluations".to_string());
    }

    /// Export report to string
    pub fn to_string(&self) -> String {
        let mut report = String::new();

        report.push_str("# Quantum ML Benchmark Report\n\n");

        // Summary statistics
        report.push_str("## Summary Statistics\n\n");
        for (name, summary) in &self.summaries {
            report.push_str(&format!(
                "### {}\n- Mean time: {:.3}s\n- Success rate: {:.1}%\n- Memory: {}\n\n",
                name,
                summary.mean_time.as_secs_f64(),
                summary.success_rate * 100.0,
                summary
                    .mean_memory
                    .map(|m| format!("{:.1} MB", m as f64 / 1024.0 / 1024.0))
                    .unwrap_or_else(|| "N/A".to_string())
            ));
        }

        // Comparisons
        report.push_str("## Benchmark Comparisons\n\n");
        for comparison in &self.comparisons {
            report.push_str(&format!(
                "### Comparing: {}\n",
                comparison.benchmarks.join(", ")
            ));
            if let Some(ref winner) = comparison.winner {
                report.push_str(&format!("**Winner: {}**\n\n", winner));
            }
        }

        // Recommendations
        report.push_str("## Recommendations\n\n");
        for (i, rec) in self.recommendations.iter().enumerate() {
            report.push_str(&format!("{}. {}\n", i + 1, rec));
        }

        report
    }
}

/// Utility functions for benchmark creation
pub mod benchmark_utils {
    use super::*;
    use crate::simulator_backends::{MPSBackend, StatevectorBackend};

    /// Create standard benchmark suite
    pub fn create_standard_suite() -> BenchmarkFramework {
        let mut framework = BenchmarkFramework::new();

        // Add VQE benchmarks
        framework.register_benchmark(
            "vqe_4q",
            Box::new(algorithm_benchmarks::VQEBenchmark::new(4, 8)),
        );
        framework.register_benchmark(
            "vqe_8q",
            Box::new(algorithm_benchmarks::VQEBenchmark::new(8, 16)),
        );
        framework.register_benchmark(
            "vqe_12q",
            Box::new(algorithm_benchmarks::VQEBenchmark::new(12, 24)),
        );

        // Add QAOA benchmarks
        framework.register_benchmark(
            "qaoa_6q",
            Box::new(algorithm_benchmarks::QAOABenchmark::new(6, 3, 10)),
        );
        framework.register_benchmark(
            "qaoa_10q",
            Box::new(algorithm_benchmarks::QAOABenchmark::new(10, 3, 20)),
        );

        // Add QNN benchmarks
        framework.register_benchmark(
            "qnn_4q",
            Box::new(algorithm_benchmarks::QNNBenchmark::new(4, 2, 100)),
        );
        framework.register_benchmark(
            "qnn_8q",
            Box::new(algorithm_benchmarks::QNNBenchmark::new(8, 3, 100)),
        );

        framework
    }

    /// Create backends for benchmarking
    pub fn create_benchmark_backends() -> Vec<Backend> {
        let mut backends: Vec<Backend> = Vec::new();

        // Statevector backend
        backends.push(Backend::Statevector(StatevectorBackend::new(15)));

        // MPS backend
        backends.push(Backend::MPS(MPSBackend::new(64, 50)));

        // GPU backend if available
        #[cfg(feature = "gpu")]
        {
            use crate::simulator_backends::GPUBackend;
            if let Ok(gpu_backend) = GPUBackend::new(0, 20) {
                backends.push(Backend::GPU(gpu_backend));
            }
        }

        backends
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator_backends::StatevectorBackend;

    #[test]
    fn test_benchmark_framework() {
        let mut framework = BenchmarkFramework::new();
        framework.register_benchmark(
            "test",
            Box::new(algorithm_benchmarks::VQEBenchmark::new(4, 8)),
        );

        assert_eq!(framework.benchmarks.len(), 1);
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig {
            repetitions: 5,
            warmup_runs: 1,
            ..Default::default()
        };

        assert_eq!(config.repetitions, 5);
        assert_eq!(config.warmup_runs, 1);
    }

    #[test]
    fn test_vqe_benchmark() {
        let mut benchmark = algorithm_benchmarks::VQEBenchmark::new(4, 8);
        let backend = StatevectorBackend::new(10);

        let setup_result = benchmark.setup();
        assert!(setup_result.is_ok());

        let run_result = benchmark.run(&Backend::Statevector(backend));
        assert!(run_result.is_ok());

        let cleanup_result = benchmark.cleanup();
        assert!(cleanup_result.is_ok());
    }

    #[test]
    fn test_system_info() {
        let info = SystemInfo::collect();
        assert!(info.cpu_cores > 0);
        assert!(info.available_memory > 0);
    }
}
