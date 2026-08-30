//! `SciRS2` statistical tools for circuit benchmarking
//!
//! This module leverages `SciRS2`'s advanced statistical analysis capabilities to provide
//! comprehensive benchmarking, performance analysis, and statistical insights for quantum circuits.

use crate::builder::Circuit;
use crate::noise_models::{NoiseAnalysisResult, NoiseAnalyzer, NoiseModel};
use crate::simulator_interface::{ExecutionResult, SimulatorBackend};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// Placeholder types representing SciRS2 statistical interface
// In the real implementation, these would be imported from SciRS2

/// Statistical distribution types supported by `SciRS2`
#[derive(Debug, Clone, PartialEq)]
pub enum Distribution {
    /// Normal distribution
    Normal { mean: f64, std_dev: f64 },
    /// Uniform distribution
    Uniform { min: f64, max: f64 },
    /// Exponential distribution
    Exponential { rate: f64 },
    /// Beta distribution
    Beta { alpha: f64, beta: f64 },
    /// Gamma distribution
    Gamma { shape: f64, scale: f64 },
    /// Poisson distribution
    Poisson { lambda: f64 },
    /// Chi-squared distribution
    ChiSquared { degrees_of_freedom: usize },
    /// Student's t-distribution
    StudentT { degrees_of_freedom: usize },
}

/// Statistical test types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatisticalTest {
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    /// Anderson-Darling test
    AndersonDarling,
    /// Shapiro-Wilk test for normality
    ShapiroWilk,
    /// Mann-Whitney U test
    MannWhitney,
    /// Wilcoxon signed-rank test
    Wilcoxon,
    /// Chi-squared goodness of fit
    ChiSquaredGoodnessOfFit,
    /// ANOVA F-test
    ANOVA,
    /// Kruskal-Wallis test
    KruskalWallis,
}

/// Hypothesis test result
#[derive(Debug, Clone)]
pub struct HypothesisTestResult {
    /// Test statistic value
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical value at significance level
    pub critical_value: f64,
    /// Whether null hypothesis is rejected
    pub reject_null: bool,
    /// Significance level used
    pub significance_level: f64,
    /// Effect size (if applicable)
    pub effect_size: Option<f64>,
    /// Confidence interval
    pub confidence_interval: Option<(f64, f64)>,
}

/// Descriptive statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats {
    /// Sample size
    pub count: usize,
    /// Mean
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Variance
    pub variance: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Median (50th percentile)
    pub median: f64,
    /// First quartile (25th percentile)
    pub q1: f64,
    /// Third quartile (75th percentile)
    pub q3: f64,
    /// Interquartile range
    pub iqr: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// Mode (most frequent value)
    pub mode: Option<f64>,
}

/// Benchmarking configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of benchmark runs
    pub num_runs: usize,
    /// Warm-up runs to exclude from statistics
    pub warmup_runs: usize,
    /// Timeout per run
    pub timeout: Duration,
    /// Significance level for statistical tests
    pub significance_level: f64,
    /// Whether to collect detailed timing data
    pub collect_timing: bool,
    /// Whether to collect memory usage data
    pub collect_memory: bool,
    /// Whether to collect error statistics
    pub collect_errors: bool,
    /// Random seed for reproducible benchmarks
    pub seed: Option<u64>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_runs: 100,
            warmup_runs: 10,
            timeout: Duration::from_secs(60),
            significance_level: 0.05,
            collect_timing: true,
            collect_memory: false,
            collect_errors: true,
            seed: None,
        }
    }
}

/// Circuit benchmarking suite using `SciRS2` statistical tools
pub struct CircuitBenchmark {
    /// Benchmark configuration
    config: BenchmarkConfig,
    /// Collected benchmark data
    benchmark_data: Vec<BenchmarkRun>,
    /// Statistical analyzer
    stats_analyzer: StatisticalAnalyzer,
}

/// Single benchmark run data
#[derive(Debug, Clone)]
pub struct BenchmarkRun {
    /// Run identifier
    pub run_id: usize,
    /// Execution time
    pub execution_time: Duration,
    /// Memory usage in bytes
    pub memory_usage: Option<usize>,
    /// Success/failure status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Circuit metrics
    pub circuit_metrics: CircuitMetrics,
    /// Execution results
    pub execution_results: Option<ExecutionResult>,
    /// Noise analysis results
    pub noise_analysis: Option<NoiseAnalysisResult>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Circuit performance metrics
#[derive(Debug, Clone)]
pub struct CircuitMetrics {
    /// Circuit depth
    pub depth: usize,
    /// Total gate count
    pub gate_count: usize,
    /// Gate count by type
    pub gate_counts: HashMap<String, usize>,
    /// Two-qubit gate count
    pub two_qubit_gates: usize,
    /// Circuit fidelity estimate
    pub fidelity: Option<f64>,
    /// Error rate estimate
    pub error_rate: Option<f64>,
}

/// Comprehensive benchmark report
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Benchmark configuration used
    pub config: BenchmarkConfig,
    /// Total runs completed
    pub completed_runs: usize,
    /// Success rate
    pub success_rate: f64,
    /// Timing statistics
    pub timing_stats: DescriptiveStats,
    /// Raw per-run timing samples (successful runs only), retained so later
    /// baseline comparisons can operate on the real distribution rather than
    /// a single summary statistic.
    pub timing_samples: Vec<f64>,
    /// Memory statistics (if collected)
    pub memory_stats: Option<DescriptiveStats>,
    /// Performance regression analysis
    pub regression_analysis: Option<RegressionAnalysis>,
    /// Distribution fitting results
    pub distribution_fit: Option<DistributionFit>,
    /// Outlier analysis
    pub outlier_analysis: OutlierAnalysis,
    /// Performance comparison with baseline
    pub baseline_comparison: Option<BaselineComparison>,
    /// Statistical test results
    pub statistical_tests: Vec<HypothesisTestResult>,
    /// Performance insights and recommendations
    pub insights: Vec<PerformanceInsight>,
}

/// Regression analysis results
#[derive(Debug, Clone)]
pub struct RegressionAnalysis {
    /// Linear regression slope
    pub slope: f64,
    /// Y-intercept
    pub intercept: f64,
    /// R-squared correlation coefficient
    pub r_squared: f64,
    /// P-value for slope significance
    pub slope_p_value: f64,
    /// Whether there's a significant trend
    pub significant_trend: bool,
    /// Predicted performance degradation per run
    pub degradation_per_run: f64,
}

/// Distribution fitting analysis
#[derive(Debug, Clone)]
pub struct DistributionFit {
    /// Best fitting distribution
    pub best_distribution: Distribution,
    /// Goodness of fit score
    pub goodness_of_fit: f64,
    /// P-value for fit test
    pub fit_p_value: f64,
    /// Alternative distributions tested
    pub alternative_fits: Vec<(Distribution, f64)>,
}

/// Outlier detection and analysis
#[derive(Debug, Clone)]
pub struct OutlierAnalysis {
    /// Number of outliers detected
    pub num_outliers: usize,
    /// Outlier indices
    pub outlier_indices: Vec<usize>,
    /// Outlier detection method used
    pub detection_method: OutlierDetectionMethod,
    /// Outlier threshold used
    pub threshold: f64,
    /// Impact of outliers on statistics
    pub outlier_impact: OutlierImpact,
}

/// Outlier detection methods
#[derive(Debug, Clone, PartialEq)]
pub enum OutlierDetectionMethod {
    /// Interquartile range method
    IQR { multiplier: f64 },
    /// Z-score method
    ZScore { threshold: f64 },
    /// Modified Z-score (median absolute deviation)
    ModifiedZScore { threshold: f64 },
    /// Isolation forest
    IsolationForest,
    /// Local outlier factor
    LocalOutlierFactor,
}

/// Impact of outliers on statistical measures
#[derive(Debug, Clone)]
pub struct OutlierImpact {
    /// Change in mean when outliers removed
    pub mean_change: f64,
    /// Change in standard deviation when outliers removed
    pub std_dev_change: f64,
    /// Change in median when outliers removed
    pub median_change: f64,
    /// Relative impact percentage
    pub relative_impact: f64,
}

/// Baseline performance comparison
#[derive(Debug, Clone)]
pub struct BaselineComparison {
    /// Baseline benchmark name
    pub baseline_name: String,
    /// Performance improvement/degradation factor
    pub performance_factor: f64,
    /// Statistical significance of difference
    pub significance: HypothesisTestResult,
    /// Confidence interval for difference
    pub difference_ci: (f64, f64),
    /// Effect size
    pub effect_size: f64,
    /// Practical significance assessment
    pub practical_significance: PracticalSignificance,
}

/// Practical significance assessment
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PracticalSignificance {
    /// Negligible difference
    Negligible,
    /// Small effect
    Small,
    /// Medium effect
    Medium,
    /// Large effect
    Large,
    /// Very large effect
    VeryLarge,
}

/// Performance insights and recommendations
#[derive(Debug, Clone)]
pub struct PerformanceInsight {
    /// Insight category
    pub category: InsightCategory,
    /// Insight message
    pub message: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Supporting evidence
    pub evidence: Vec<String>,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Performance insight categories
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsightCategory {
    /// Performance degradation detected
    PerformanceDegradation,
    /// Performance improvement detected
    PerformanceImprovement,
    /// High variability in results
    HighVariability,
    /// Outliers detected
    OutliersDetected,
    /// Memory usage concerns
    MemoryUsage,
    /// Error rate concerns
    ErrorRate,
    /// Circuit optimization opportunity
    OptimizationOpportunity,
}

impl CircuitBenchmark {
    /// Create a new circuit benchmark suite
    #[must_use]
    pub const fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            benchmark_data: Vec::new(),
            stats_analyzer: StatisticalAnalyzer::new(),
        }
    }

    /// Run comprehensive benchmark suite
    pub fn run_benchmark<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
        simulator: &dyn SimulatorExecutor,
        noise_model: Option<&NoiseModel>,
    ) -> QuantRS2Result<BenchmarkReport> {
        self.benchmark_data.clear();

        let total_runs = self.config.num_runs + self.config.warmup_runs;

        for run_id in 0..total_runs {
            let is_warmup = run_id < self.config.warmup_runs;

            match self.run_single_benchmark(circuit, simulator, noise_model, run_id) {
                Ok(run_data) => {
                    if !is_warmup {
                        self.benchmark_data.push(run_data);
                    }
                }
                Err(e) => {
                    if !is_warmup {
                        // Record failed run
                        let failed_run = BenchmarkRun {
                            run_id,
                            execution_time: Duration::from_millis(0),
                            memory_usage: None,
                            success: false,
                            error_message: Some(e.to_string()),
                            circuit_metrics: self.calculate_circuit_metrics(circuit),
                            execution_results: None,
                            noise_analysis: None,
                            custom_metrics: HashMap::new(),
                        };
                        self.benchmark_data.push(failed_run);
                    }
                }
            }
        }

        self.generate_benchmark_report()
    }

    /// Run a single benchmark iteration
    fn run_single_benchmark<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        simulator: &dyn SimulatorExecutor,
        noise_model: Option<&NoiseModel>,
        run_id: usize,
    ) -> QuantRS2Result<BenchmarkRun> {
        let start_time = Instant::now();
        let start_memory = if self.config.collect_memory {
            Some(self.get_memory_usage())
        } else {
            None
        };

        // Actually execute the circuit through the injected simulator so the
        // measured execution_time reflects real work, and the captured
        // ExecutionResult reflects what the simulator produced.
        let execution_outcome = simulator.execute(circuit as &dyn std::any::Any);

        let end_time = Instant::now();
        let end_memory = if self.config.collect_memory {
            Some(self.get_memory_usage())
        } else {
            None
        };

        let execution_time = end_time - start_time;
        let memory_usage = match (start_memory, end_memory) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        };

        let (success, error_message, execution_results) = match execution_outcome {
            Ok(result) => (true, None, Some(result)),
            Err(e) => (false, Some(e.to_string()), None),
        };

        // Perform real noise analysis when a noise model is provided, by
        // registering it with a `NoiseAnalyzer` and running its full
        // circuit-noise analysis (gate errors, decoherence, readout,
        // crosstalk) against the actual circuit.
        let noise_analysis = if let Some(noise) = noise_model {
            let mut analyzer = NoiseAnalyzer::new();
            let device_key = format!("__circuit_benchmark_run_{run_id}__");
            analyzer.add_noise_model(device_key.clone(), noise.clone());
            Some(analyzer.analyze_circuit_noise(circuit, &device_key)?)
        } else {
            None
        };

        Ok(BenchmarkRun {
            run_id,
            execution_time,
            memory_usage,
            success,
            error_message,
            circuit_metrics: self.calculate_circuit_metrics(circuit),
            execution_results,
            noise_analysis,
            custom_metrics: HashMap::new(),
        })
    }

    /// Calculate circuit metrics
    fn calculate_circuit_metrics<const N: usize>(&self, circuit: &Circuit<N>) -> CircuitMetrics {
        let gate_count = circuit.gates().len();
        let mut gate_counts = HashMap::new();
        let mut two_qubit_gates = 0;

        for gate in circuit.gates() {
            let gate_name = gate.name();
            *gate_counts.entry(gate_name.to_string()).or_insert(0) += 1;

            if gate.qubits().len() == 2 {
                two_qubit_gates += 1;
            }
        }

        CircuitMetrics {
            depth: gate_count, // Simplified depth calculation
            gate_count,
            gate_counts,
            two_qubit_gates,
            fidelity: None,
            error_rate: None,
        }
    }

    /// Get current memory usage (placeholder)
    const fn get_memory_usage(&self) -> usize {
        // In real implementation, this would use system APIs to get memory usage
        0
    }

    /// Generate comprehensive benchmark report
    fn generate_benchmark_report(&self) -> QuantRS2Result<BenchmarkReport> {
        let completed_runs = self.benchmark_data.len();
        let successful_runs: Vec<_> = self
            .benchmark_data
            .iter()
            .filter(|run| run.success)
            .collect();

        let success_rate = successful_runs.len() as f64 / completed_runs as f64;

        // Extract timing data
        let timing_data: Vec<f64> = successful_runs
            .iter()
            .map(|run| run.execution_time.as_secs_f64())
            .collect();

        let timing_stats = self
            .stats_analyzer
            .calculate_descriptive_stats(&timing_data)?;

        // Extract memory data if available
        let memory_stats = if self.config.collect_memory {
            let memory_data: Vec<f64> = successful_runs
                .iter()
                .filter_map(|run| run.memory_usage.map(|m| m as f64))
                .collect();

            if memory_data.is_empty() {
                None
            } else {
                Some(
                    self.stats_analyzer
                        .calculate_descriptive_stats(&memory_data)?,
                )
            }
        } else {
            None
        };

        // Perform regression analysis to detect performance trends
        let regression_analysis = self
            .stats_analyzer
            .perform_regression_analysis(&timing_data)?;

        // Fit distributions to timing data
        let distribution_fit = self.stats_analyzer.fit_distributions(&timing_data)?;

        // Detect outliers
        let outlier_analysis = self.stats_analyzer.detect_outliers(
            &timing_data,
            OutlierDetectionMethod::IQR { multiplier: 1.5 },
        )?;

        // Generate insights
        let insights = self.generate_performance_insights(
            &timing_stats,
            &regression_analysis,
            &outlier_analysis,
            success_rate,
        );

        Ok(BenchmarkReport {
            config: self.config.clone(),
            completed_runs,
            success_rate,
            timing_stats,
            timing_samples: timing_data,
            memory_stats,
            regression_analysis: Some(regression_analysis),
            distribution_fit: Some(distribution_fit),
            outlier_analysis,
            baseline_comparison: None,
            statistical_tests: Vec::new(),
            insights,
        })
    }

    /// Generate performance insights based on statistical analysis
    fn generate_performance_insights(
        &self,
        timing_stats: &DescriptiveStats,
        regression: &RegressionAnalysis,
        outliers: &OutlierAnalysis,
        success_rate: f64,
    ) -> Vec<PerformanceInsight> {
        let mut insights = Vec::new();

        // Check for performance degradation
        if regression.significant_trend && regression.slope > 0.0 {
            insights.push(PerformanceInsight {
                category: InsightCategory::PerformanceDegradation,
                message: format!(
                    "Significant performance degradation detected: {:.4} seconds per run increase",
                    regression.degradation_per_run
                ),
                confidence: 1.0 - regression.slope_p_value,
                evidence: vec![
                    format!("Linear trend slope: {:.6}", regression.slope),
                    format!("R-squared: {:.4}", regression.r_squared),
                    format!("P-value: {:.4}", regression.slope_p_value),
                ],
                recommendations: vec![
                    "Investigate potential memory leaks".to_string(),
                    "Check for resource contention".to_string(),
                    "Profile execution to identify bottlenecks".to_string(),
                ],
            });
        }

        // Check for high variability
        let coefficient_of_variation = timing_stats.std_dev / timing_stats.mean;
        if coefficient_of_variation > 0.2 {
            insights.push(PerformanceInsight {
                category: InsightCategory::HighVariability,
                message: format!(
                    "High performance variability detected: CV = {:.2}%",
                    coefficient_of_variation * 100.0
                ),
                confidence: 0.8,
                evidence: vec![
                    format!("Standard deviation: {:.4} seconds", timing_stats.std_dev),
                    format!("Mean: {:.4} seconds", timing_stats.mean),
                    format!(
                        "Coefficient of variation: {:.2}%",
                        coefficient_of_variation * 100.0
                    ),
                ],
                recommendations: vec![
                    "Increase warm-up runs to stabilize performance".to_string(),
                    "Check for system load variations".to_string(),
                    "Consider running benchmarks in isolated environment".to_string(),
                ],
            });
        }

        // Check for outliers
        if outliers.num_outliers > 0 {
            let outlier_percentage =
                outliers.num_outliers as f64 / timing_stats.count as f64 * 100.0;
            insights.push(PerformanceInsight {
                category: InsightCategory::OutliersDetected,
                message: format!(
                    "Performance outliers detected: {} outliers ({:.1}% of runs)",
                    outliers.num_outliers, outlier_percentage
                ),
                confidence: 0.9,
                evidence: vec![
                    format!("Number of outliers: {}", outliers.num_outliers),
                    format!("Outlier percentage: {:.1}%", outlier_percentage),
                    format!("Detection method: {:?}", outliers.detection_method),
                ],
                recommendations: vec![
                    "Investigate causes of outlier runs".to_string(),
                    "Consider removing outliers from performance metrics".to_string(),
                    "Check for system interruptions during benchmarking".to_string(),
                ],
            });
        }

        // Check success rate
        if success_rate < 0.95 {
            insights.push(PerformanceInsight {
                category: InsightCategory::ErrorRate,
                message: format!("Low success rate detected: {:.1}%", success_rate * 100.0),
                confidence: 1.0,
                evidence: vec![
                    format!("Success rate: {:.1}%", success_rate * 100.0),
                    format!(
                        "Failed runs: {}",
                        timing_stats.count - (timing_stats.count as f64 * success_rate) as usize
                    ),
                ],
                recommendations: vec![
                    "Investigate failure causes".to_string(),
                    "Check circuit validity and simulator compatibility".to_string(),
                    "Increase timeout limits if timeouts are occurring".to_string(),
                ],
            });
        }

        insights
    }

    /// Compare with baseline benchmark
    pub fn compare_with_baseline(
        &self,
        baseline: &BenchmarkReport,
    ) -> QuantRS2Result<BaselineComparison> {
        if self.benchmark_data.is_empty() {
            return Err(QuantRS2Error::InvalidInput(
                "No benchmark data available for comparison".to_string(),
            ));
        }

        let current_timing: Vec<f64> = self
            .benchmark_data
            .iter()
            .filter(|run| run.success)
            .map(|run| run.execution_time.as_secs_f64())
            .collect();

        let baseline_mean = baseline.timing_stats.mean;
        let current_mean = self
            .stats_analyzer
            .calculate_descriptive_stats(&current_timing)?
            .mean;

        let performance_factor = current_mean / baseline_mean;

        // Perform statistical test for significance using the real retained
        // per-run baseline timing samples (falling back to the summary mean
        // only if the baseline report predates sample retention).
        let baseline_samples: Vec<f64> = if baseline.timing_samples.is_empty() {
            vec![baseline_mean]
        } else {
            baseline.timing_samples.clone()
        };
        let significance = self.stats_analyzer.mann_whitney_test(
            &current_timing,
            &baseline_samples,
            self.config.significance_level,
        )?;

        // Calculate effect size (Cohen's d)
        let effect_size = (current_mean - baseline_mean) / baseline.timing_stats.std_dev;

        // Assess practical significance
        let practical_significance = match effect_size.abs() {
            x if x < 0.2 => PracticalSignificance::Negligible,
            x if x < 0.5 => PracticalSignificance::Small,
            x if x < 0.8 => PracticalSignificance::Medium,
            x if x < 1.2 => PracticalSignificance::Large,
            _ => PracticalSignificance::VeryLarge,
        };

        // Compute a real normal-approximation confidence interval for the
        // difference in means (current - baseline), using the pooled
        // standard error from both samples' variances.
        let current_variance = self
            .stats_analyzer
            .calculate_descriptive_stats(&current_timing)?
            .variance;
        let baseline_variance = if baseline_samples.len() > 1 {
            self.stats_analyzer
                .calculate_descriptive_stats(&baseline_samples)?
                .variance
        } else {
            baseline.timing_stats.std_dev * baseline.timing_stats.std_dev
        };
        let n_current = current_timing.len() as f64;
        let n_baseline = baseline_samples.len().max(1) as f64;
        let standard_error = (current_variance / n_current + baseline_variance / n_baseline).sqrt();
        let z_critical = inverse_normal_cdf(1.0 - self.config.significance_level / 2.0);
        let mean_difference = current_mean - baseline_mean;
        let difference_ci = (
            z_critical.mul_add(-standard_error, mean_difference),
            z_critical.mul_add(standard_error, mean_difference),
        );

        Ok(BaselineComparison {
            baseline_name: "baseline".to_string(),
            performance_factor,
            significance,
            difference_ci,
            effect_size,
            practical_significance,
        })
    }
}

/// Statistical analyzer using `SciRS2` capabilities
pub struct StatisticalAnalyzer;

impl StatisticalAnalyzer {
    /// Create a new statistical analyzer
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Calculate descriptive statistics
    pub fn calculate_descriptive_stats(&self, data: &[f64]) -> QuantRS2Result<DescriptiveStats> {
        if data.is_empty() {
            return Err(QuantRS2Error::InvalidInput("Empty data".to_string()));
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = data.len();
        let mean = data.iter().sum::<f64>() / count as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();
        let min = sorted_data[0];
        let max = sorted_data[count - 1];

        let median = if count % 2 == 0 {
            f64::midpoint(sorted_data[count / 2 - 1], sorted_data[count / 2])
        } else {
            sorted_data[count / 2]
        };

        let q1 = self.percentile(&sorted_data, 0.25);
        let q3 = self.percentile(&sorted_data, 0.75);
        let iqr = q3 - q1;

        // Calculate skewness and kurtosis
        let skewness = self.calculate_skewness(data, mean, std_dev);
        let kurtosis = self.calculate_kurtosis(data, mean, std_dev);

        Ok(DescriptiveStats {
            count,
            mean,
            std_dev,
            variance,
            min,
            max,
            median,
            q1,
            q3,
            iqr,
            skewness,
            kurtosis,
            mode: None, // Would implement mode calculation
        })
    }

    /// Calculate percentile
    fn percentile(&self, sorted_data: &[f64], p: f64) -> f64 {
        let index = (p * (sorted_data.len() - 1) as f64).round() as usize;
        sorted_data[index.min(sorted_data.len() - 1)]
    }

    /// Calculate skewness
    fn calculate_skewness(&self, data: &[f64], mean: f64, std_dev: f64) -> f64 {
        let n = data.len() as f64;
        let skew_sum = data
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>();
        skew_sum / n
    }

    /// Calculate kurtosis
    fn calculate_kurtosis(&self, data: &[f64], mean: f64, std_dev: f64) -> f64 {
        let n = data.len() as f64;
        let kurt_sum = data
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>();
        kurt_sum / n - 3.0 // Excess kurtosis
    }

    /// Perform linear regression analysis
    pub fn perform_regression_analysis(&self, data: &[f64]) -> QuantRS2Result<RegressionAnalysis> {
        if data.len() < 3 {
            return Err(QuantRS2Error::InvalidInput(
                "Insufficient data for regression".to_string(),
            ));
        }

        let n = data.len() as f64;
        let x_values: Vec<f64> = (0..data.len()).map(|i| i as f64).collect();

        let x_mean = x_values.iter().sum::<f64>() / n;
        let y_mean = data.iter().sum::<f64>() / n;

        let numerator: f64 = x_values
            .iter()
            .zip(data.iter())
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();

        let denominator: f64 = x_values.iter().map(|x| (x - x_mean).powi(2)).sum();

        let slope = numerator / denominator;
        let intercept = slope.mul_add(-x_mean, y_mean);

        // Calculate R-squared
        let ss_tot: f64 = data.iter().map(|y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = x_values
            .iter()
            .zip(data.iter())
            .map(|(x, y)| {
                let predicted = slope * x + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        let r_squared = 1.0 - (ss_res / ss_tot);

        // Calculate a real p-value for the slope estimate from Student's
        // t-distribution: t = slope / se(slope), with n - 2 residual
        // degrees of freedom, using the standard OLS standard-error formula
        // se(slope) = sqrt(MSE / sum((x - x_mean)^2)).
        let degrees_of_freedom = n - 2.0;
        let slope_p_value = if degrees_of_freedom > 0.0 && denominator > 0.0 {
            let mean_squared_error = (ss_res / degrees_of_freedom).max(0.0);
            let standard_error_slope = (mean_squared_error / denominator).sqrt();
            if standard_error_slope > 0.0 {
                let t_statistic = slope / standard_error_slope;
                student_t_two_sided_p_value(t_statistic, degrees_of_freedom)
            } else if slope.abs() > 0.0 {
                // Zero residual variance with a non-zero slope: perfect
                // linear fit, so the trend is maximally significant.
                0.0
            } else {
                1.0
            }
        } else {
            1.0
        };
        let significant_trend = slope_p_value < 0.05;

        Ok(RegressionAnalysis {
            slope,
            intercept,
            r_squared,
            slope_p_value,
            significant_trend,
            degradation_per_run: slope,
        })
    }

    /// Fit probability distributions to data
    pub fn fit_distributions(&self, data: &[f64]) -> QuantRS2Result<DistributionFit> {
        let stats = self.calculate_descriptive_stats(data)?;

        // Fit normal distribution
        let normal_dist = Distribution::Normal {
            mean: stats.mean,
            std_dev: stats.std_dev,
        };

        // Real goodness-of-fit assessment: a one-sample Kolmogorov-Smirnov
        // test of the empirical distribution against the fitted normal.
        let (ks_statistic, fit_p_value) =
            Self::kolmogorov_smirnov_normal_test(data, stats.mean, stats.std_dev);
        let goodness_of_fit = (1.0 - ks_statistic).clamp(0.0, 1.0);

        Ok(DistributionFit {
            best_distribution: normal_dist,
            goodness_of_fit,
            fit_p_value,
            alternative_fits: Vec::new(),
        })
    }

    /// One-sample Kolmogorov-Smirnov test of `data` against a
    /// `Normal(mean, std_dev)` distribution. Returns `(D statistic, p-value)`
    /// using the asymptotic Kolmogorov distribution for the p-value.
    fn kolmogorov_smirnov_normal_test(data: &[f64], mean: f64, std_dev: f64) -> (f64, f64) {
        if data.is_empty() || std_dev <= 0.0 {
            return (1.0, 0.0);
        }

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len() as f64;

        let mut max_diff = 0.0_f64;
        for (i, &value) in sorted.iter().enumerate() {
            let z = (value - mean) / std_dev;
            let cdf = normal_cdf(z);
            let empirical_upper = (i as f64 + 1.0) / n;
            let empirical_lower = i as f64 / n;
            max_diff = max_diff.max((empirical_upper - cdf).abs());
            max_diff = max_diff.max((cdf - empirical_lower).abs());
        }

        let p_value = kolmogorov_smirnov_p_value(max_diff, sorted.len());
        (max_diff, p_value)
    }

    /// Detect outliers using specified method
    pub fn detect_outliers(
        &self,
        data: &[f64],
        method: OutlierDetectionMethod,
    ) -> QuantRS2Result<OutlierAnalysis> {
        let (outlier_indices, threshold) = match method {
            OutlierDetectionMethod::IQR { multiplier } => {
                (self.detect_outliers_iqr(data, multiplier)?, multiplier)
            }
            OutlierDetectionMethod::ZScore { threshold } => {
                (self.detect_outliers_zscore(data, threshold)?, threshold)
            }
            OutlierDetectionMethod::ModifiedZScore { threshold } => (
                self.detect_outliers_modified_zscore(data, threshold)?,
                threshold,
            ),
            OutlierDetectionMethod::IsolationForest
            | OutlierDetectionMethod::LocalOutlierFactor => {
                // These require a trained ensemble/density model that this
                // analyzer does not implement; report honestly instead of
                // silently claiming zero outliers were found.
                return Err(QuantRS2Error::UnsupportedOperation(format!(
                    "Outlier detection method {method:?} is not yet implemented"
                )));
            }
        };

        let num_outliers = outlier_indices.len();

        // Calculate outlier impact
        let outlier_impact = if num_outliers > 0 {
            self.calculate_outlier_impact(data, &outlier_indices)?
        } else {
            OutlierImpact {
                mean_change: 0.0,
                std_dev_change: 0.0,
                median_change: 0.0,
                relative_impact: 0.0,
            }
        };

        Ok(OutlierAnalysis {
            num_outliers,
            outlier_indices,
            detection_method: method,
            threshold,
            outlier_impact,
        })
    }

    /// Detect outliers using IQR method
    fn detect_outliers_iqr(&self, data: &[f64], multiplier: f64) -> QuantRS2Result<Vec<usize>> {
        let stats = self.calculate_descriptive_stats(data)?;
        let lower_bound = multiplier.mul_add(-stats.iqr, stats.q1);
        let upper_bound = multiplier.mul_add(stats.iqr, stats.q3);

        Ok(data
            .iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                if value < lower_bound || value > upper_bound {
                    Some(i)
                } else {
                    None
                }
            })
            .collect())
    }

    /// Detect outliers using Z-score method
    fn detect_outliers_zscore(&self, data: &[f64], threshold: f64) -> QuantRS2Result<Vec<usize>> {
        let stats = self.calculate_descriptive_stats(data)?;

        Ok(data
            .iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                let z_score = (value - stats.mean) / stats.std_dev;
                if z_score.abs() > threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect())
    }

    /// Detect outliers using the modified Z-score method (median absolute
    /// deviation based), which is more robust to extreme values than the
    /// mean/std-dev based Z-score method above.
    fn detect_outliers_modified_zscore(
        &self,
        data: &[f64],
        threshold: f64,
    ) -> QuantRS2Result<Vec<usize>> {
        if data.is_empty() {
            return Err(QuantRS2Error::InvalidInput("Empty data".to_string()));
        }

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = Self::median_of_sorted(&sorted);

        let mut abs_deviations: Vec<f64> = data.iter().map(|&v| (v - median).abs()).collect();
        abs_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = Self::median_of_sorted(&abs_deviations);

        if mad == 0.0 {
            // No spread to normalize against: fall back to flagging any
            // value that differs at all from the median, rather than
            // dividing by zero or silently reporting no outliers.
            return Ok(data
                .iter()
                .enumerate()
                .filter_map(|(i, &value)| {
                    if (value - median).abs() > f64::EPSILON {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect());
        }

        // 0.6745 is the constant that makes the MAD comparable to the
        // standard deviation for normally distributed data (Iglewicz & Hoaglin).
        Ok(data
            .iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                let modified_z = 0.6745 * (value - median) / mad;
                if modified_z.abs() > threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect())
    }

    /// Median of an already-sorted slice
    fn median_of_sorted(sorted: &[f64]) -> f64 {
        let count = sorted.len();
        if count == 0 {
            return 0.0;
        }
        if count % 2 == 0 {
            f64::midpoint(sorted[count / 2 - 1], sorted[count / 2])
        } else {
            sorted[count / 2]
        }
    }

    /// Calculate impact of outliers on statistics
    fn calculate_outlier_impact(
        &self,
        data: &[f64],
        outlier_indices: &[usize],
    ) -> QuantRS2Result<OutlierImpact> {
        let original_stats = self.calculate_descriptive_stats(data)?;

        // Create data without outliers
        let filtered_data: Vec<f64> = data
            .iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                if outlier_indices.contains(&i) {
                    None
                } else {
                    Some(value)
                }
            })
            .collect();

        let filtered_stats = self.calculate_descriptive_stats(&filtered_data)?;

        let mean_change = (original_stats.mean - filtered_stats.mean).abs();
        let std_dev_change = (original_stats.std_dev - filtered_stats.std_dev).abs();
        let median_change = (original_stats.median - filtered_stats.median).abs();
        let relative_impact = mean_change / original_stats.mean * 100.0;

        Ok(OutlierImpact {
            mean_change,
            std_dev_change,
            median_change,
            relative_impact,
        })
    }

    /// Perform Mann-Whitney U test
    pub fn mann_whitney_test(
        &self,
        sample1: &[f64],
        sample2: &[f64],
        significance_level: f64,
    ) -> QuantRS2Result<HypothesisTestResult> {
        let n1 = sample1.len();
        let n2 = sample2.len();
        if n1 == 0 || n2 == 0 {
            return Err(QuantRS2Error::InvalidInput(
                "Mann-Whitney U test requires two non-empty samples".to_string(),
            ));
        }

        // Combine both samples, tagging which group each value came from,
        // then rank the combined data (ties receive the average of the
        // ranks they span).
        let mut combined: Vec<(f64, u8)> = sample1
            .iter()
            .map(|&value| (value, 0u8))
            .chain(sample2.iter().map(|&value| (value, 1u8)))
            .collect();
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let n_total = combined.len();
        let mut ranks = vec![0.0_f64; n_total];
        let mut tie_correction_sum = 0.0_f64;
        let mut i = 0;
        while i < n_total {
            let mut j = i;
            while j + 1 < n_total && (combined[j + 1].0 - combined[i].0).abs() < f64::EPSILON {
                j += 1;
            }
            // Average (1-based) rank shared by the tied block [i, j]
            let average_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
            for rank in ranks.iter_mut().take(j + 1).skip(i) {
                *rank = average_rank;
            }
            let tie_group_size = (j - i + 1) as f64;
            tie_correction_sum += tie_group_size.powi(3) - tie_group_size;
            i = j + 1;
        }

        let rank_sum_sample1: f64 = combined
            .iter()
            .zip(ranks.iter())
            .filter(|((_, group), _)| *group == 0)
            .map(|(_, &rank)| rank)
            .sum();

        let n1_f = n1 as f64;
        let n2_f = n2 as f64;
        let n_total_f = n1_f + n2_f;

        let u1 = rank_sum_sample1 - n1_f * (n1_f + 1.0) / 2.0;
        let u2 = n1_f * n2_f - u1;
        let u_statistic = u1.min(u2);

        let mean_u = n1_f * n2_f / 2.0;
        let tie_term = if n_total_f > 1.0 {
            tie_correction_sum / (n_total_f * (n_total_f - 1.0))
        } else {
            0.0
        };
        let variance_u = (n1_f * n2_f / 12.0) * (n_total_f + 1.0 - tie_term);
        let std_dev_u = variance_u.max(0.0).sqrt();

        let p_value = if std_dev_u > 0.0 {
            let difference = u1 - mean_u;
            // Continuity correction toward the mean
            let z_score = if difference > 0.0 {
                (difference - 0.5) / std_dev_u
            } else if difference < 0.0 {
                (difference + 0.5) / std_dev_u
            } else {
                0.0
            };
            (2.0 * (1.0 - normal_cdf(z_score.abs()))).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let critical_value = inverse_normal_cdf(1.0 - significance_level / 2.0);
        let reject_null = p_value < significance_level;

        // Rank-biserial correlation as the effect size for a Mann-Whitney U test
        let effect_size = 1.0 - (2.0 * u1) / (n1_f * n2_f);

        Ok(HypothesisTestResult {
            test_statistic: u_statistic,
            p_value,
            critical_value,
            reject_null,
            significance_level,
            effect_size: Some(effect_size),
            confidence_interval: None,
        })
    }
}

/// Trait for simulator execution (placeholder)
pub trait SimulatorExecutor {
    fn execute(&self, circuit: &dyn std::any::Any) -> QuantRS2Result<ExecutionResult>;
}

impl Default for StatisticalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------
// Real statistical primitives backing the hypothesis tests, regression
// significance, and goodness-of-fit tests above. These are standard
// numerical-analysis approximations (Abramowitz & Stegun error function,
// Acklam's inverse normal CDF, Lanczos log-gamma, and the Numerical
// Recipes continued-fraction for the regularized incomplete beta
// function), not statistical placeholders.
// ---------------------------------------------------------------------

/// Error function approximation (Abramowitz & Stegun 7.1.26), maximum
/// absolute error ~1.5e-7.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let a1 = 0.254_829_592_f64;
    let a2 = -0.284_496_736_f64;
    let a3 = 1.421_413_741_f64;
    let a4 = -1.453_152_027_f64;
    let a5 = 1.061_405_429_f64;
    let p = 0.327_591_1_f64;

    let t = 1.0 / p.mul_add(x, 1.0);
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    let y = 1.0 - poly * (-x * x).exp();

    sign * y
}

/// Standard normal cumulative distribution function.
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Inverse standard normal CDF (quantile function) via Peter Acklam's
/// rational approximation, accurate to about 1.15e-9.
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_69e+02,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239e+00,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e+01,
        1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00,
        4.374_664_141_464_968e+00,
        2.938_163_982_698_783e+00,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-03,
        3.224_671_290_700_398e-01,
        2.445_134_137_142_996e+00,
        3.754_408_661_907_416e+00,
    ];

    let p_low = 0.024_85;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Natural log of the gamma function via the Lanczos approximation
/// (g = 7, n = 9), accurate to double precision over the domain used here.
fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection formula for arguments below 0.5
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COEFFICIENTS[0];
        let t = x + 7.5;
        for (i, coeff) in COEFFICIENTS.iter().enumerate().skip(1) {
            a += coeff / (x + i as f64);
        }
        0.5_f64.mul_add(
            (2.0 * std::f64::consts::PI).ln(),
            (x + 0.5) * t.ln() - t + a.ln(),
        )
    }
}

/// Continued-fraction evaluation used by the regularized incomplete beta
/// function (Numerical Recipes `betacf`).
fn incomplete_beta_continued_fraction(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITERATIONS: usize = 200;
    const EPSILON: f64 = 1e-12;
    const FP_MIN: f64 = 1e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FP_MIN {
        d = FP_MIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITERATIONS {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        let aa_even = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa_even * d;
        if d.abs() < FP_MIN {
            d = FP_MIN;
        }
        c = 1.0 + aa_even / c;
        if c.abs() < FP_MIN {
            c = FP_MIN;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa_odd = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa_odd * d;
        if d.abs() < FP_MIN {
            d = FP_MIN;
        }
        c = 1.0 + aa_odd / c;
        if c.abs() < FP_MIN {
            c = FP_MIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < EPSILON {
            break;
        }
    }

    h
}

/// Regularized incomplete beta function `I_x(a, b)`.
fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let ln_beta_fn = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let front = (a.mul_add(x.ln(), ln_beta_fn) + b * (1.0 - x).ln()).exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        front * incomplete_beta_continued_fraction(x, a, b) / a
    } else {
        1.0 - front * incomplete_beta_continued_fraction(1.0 - x, b, a) / b
    }
}

/// Two-sided p-value for a Student's t-distributed statistic with the
/// given (real-valued) degrees of freedom, computed via the regularized
/// incomplete beta function: `p = I_{df / (df + t^2)}(df/2, 1/2)`.
fn student_t_two_sided_p_value(t_statistic: f64, degrees_of_freedom: f64) -> f64 {
    if degrees_of_freedom <= 0.0 {
        return 1.0;
    }
    let x = degrees_of_freedom / (degrees_of_freedom + t_statistic * t_statistic);
    incomplete_beta(x, degrees_of_freedom / 2.0, 0.5).clamp(0.0, 1.0)
}

/// Asymptotic p-value for the one-sample Kolmogorov-Smirnov statistic `d`
/// computed from `n` observations, using the Kolmogorov distribution's
/// alternating-series form (Marsaglia-Kolmogorov asymptotic expansion).
fn kolmogorov_smirnov_p_value(d: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let n_f = n as f64;
    let lambda = (n_f.sqrt() + 0.12 + 0.11 / n_f.sqrt()) * d;
    if lambda < 0.2 {
        return 1.0;
    }

    let mut sum = 0.0_f64;
    for k in 1..=100_i32 {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        sum += sign * (-2.0 * f64::from(k * k) * lambda * lambda).exp();
    }

    (2.0 * sum).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptive_stats() {
        let analyzer = StatisticalAnalyzer::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let stats = analyzer
            .calculate_descriptive_stats(&data)
            .expect("calculate_descriptive_stats should succeed");
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_outlier_detection_iqr() {
        let analyzer = StatisticalAnalyzer::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100.0 is an outlier

        let outliers = analyzer
            .detect_outliers_iqr(&data, 1.5)
            .expect("outlier detection should succeed");
        assert_eq!(outliers.len(), 1);
        assert_eq!(outliers[0], 5); // Index of 100.0
    }

    #[test]
    fn test_regression_analysis() {
        let analyzer = StatisticalAnalyzer::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Perfect linear trend

        let regression = analyzer
            .perform_regression_analysis(&data)
            .expect("perform_regression_analysis should succeed");
        assert!((regression.slope - 1.0).abs() < 1e-10);
        assert!((regression.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.num_runs, 100);
        assert_eq!(config.warmup_runs, 10);
        assert_eq!(config.significance_level, 0.05);
    }

    #[test]
    fn test_distribution_creation() {
        let normal = Distribution::Normal {
            mean: 0.0,
            std_dev: 1.0,
        };
        match normal {
            Distribution::Normal { mean, std_dev } => {
                assert_eq!(mean, 0.0);
                assert_eq!(std_dev, 1.0);
            }
            _ => panic!("Wrong distribution type"),
        }
    }

    /// A mock simulator that records whether/how many times it was invoked,
    /// used to prove `run_single_benchmark` actually calls the injected
    /// simulator instead of fabricating timing/results around no-op work.
    struct RecordingSimulator {
        calls: std::cell::Cell<usize>,
    }

    impl SimulatorExecutor for RecordingSimulator {
        fn execute(&self, _circuit: &dyn std::any::Any) -> QuantRS2Result<ExecutionResult> {
            self.calls.set(self.calls.get() + 1);
            Ok(ExecutionResult {
                measurements: HashMap::new(),
                final_state: None,
                execution_stats: crate::simulator_interface::ExecutionStats {
                    execution_time: Duration::from_millis(1),
                    memory_used: 0,
                    shots: 1,
                    success_rate: 1.0,
                },
                backend_results: HashMap::new(),
            })
        }
    }

    #[test]
    fn test_run_single_benchmark_executes_simulator_and_noise_analysis() {
        let mut circ: Circuit<2> = Circuit::new();
        circ.h(0)
            .expect("h gate should apply")
            .cnot(0, 1)
            .expect("cnot gate should apply");

        let benchmark = CircuitBenchmark::new(BenchmarkConfig::default());
        let simulator = RecordingSimulator {
            calls: std::cell::Cell::new(0),
        };
        let noise_model = NoiseModel::ibm_quantum();

        let run = benchmark
            .run_single_benchmark(&circ, &simulator, Some(&noise_model), 0)
            .expect("run_single_benchmark should succeed");

        assert_eq!(
            simulator.calls.get(),
            1,
            "run_single_benchmark must actually invoke simulator.execute()"
        );
        assert!(run.success);
        assert!(
            run.execution_results.is_some(),
            "execution_results must be populated from the simulator's real output"
        );

        let noise_analysis = run
            .noise_analysis
            .expect("noise analysis must be computed when a noise model is supplied");
        assert!(noise_analysis.total_error >= 0.0);
        assert!(noise_analysis.total_fidelity <= 1.0);
        assert!(
            !noise_analysis.gate_errors.is_empty(),
            "noise analysis should report per-gate errors for a circuit with gates"
        );
    }

    #[test]
    fn test_mann_whitney_clear_difference_rejects_null() {
        let analyzer = StatisticalAnalyzer::new();
        let low = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let high = vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0];

        let result = analyzer
            .mann_whitney_test(&low, &high, 0.05)
            .expect("mann_whitney_test should succeed");

        // U statistic for two completely non-overlapping equal-size samples is 0.
        assert_eq!(result.test_statistic, 0.0);
        assert!(
            result.p_value < 0.01,
            "p-value should be tiny for clearly separated distributions, got {}",
            result.p_value
        );
        assert!(result.reject_null);
        assert!((result.critical_value - 1.959_963_985).abs() < 1e-6);
    }

    #[test]
    fn test_mann_whitney_identical_distributions_do_not_reject() {
        let analyzer = StatisticalAnalyzer::new();
        let sample = vec![1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0];
        let other = sample.clone();

        let result = analyzer
            .mann_whitney_test(&sample, &other, 0.05)
            .expect("mann_whitney_test should succeed");

        assert!(
            result.p_value > 0.05,
            "identical distributions should not show a significant difference, got p={}",
            result.p_value
        );
        assert!(!result.reject_null);
    }

    #[test]
    fn test_regression_slope_p_value_is_real() {
        let analyzer = StatisticalAnalyzer::new();

        // Perfect, noiseless linear trend: p-value should indicate strong significance.
        let trending = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let strong = analyzer
            .perform_regression_analysis(&trending)
            .expect("perform_regression_analysis should succeed");
        assert!(strong.slope_p_value < 1e-6);
        assert!(strong.significant_trend);

        // Perfectly flat data (zero slope, zero residual variance): the
        // real t-test must report no significant trend, unlike the old
        // heuristic which only checked `slope.abs() > 0.001`.
        let flat = vec![5.0; 10];
        let none = analyzer
            .perform_regression_analysis(&flat)
            .expect("perform_regression_analysis should succeed");
        assert!((none.slope_p_value - 1.0).abs() < 1e-9);
        assert!(!none.significant_trend);
    }

    #[test]
    fn test_fit_distributions_not_hardcoded() {
        let analyzer = StatisticalAnalyzer::new();
        let data = vec![
            -2.0, -1.5, -1.2, -0.8, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0,
        ];

        let fit = analyzer
            .fit_distributions(&data)
            .expect("fit_distributions should succeed");

        assert!((0.0..=1.0).contains(&fit.goodness_of_fit));
        assert!((0.0..=1.0).contains(&fit.fit_p_value));
        assert!(
            (fit.goodness_of_fit - 0.8).abs() > 1e-9 || (fit.fit_p_value - 0.3).abs() > 1e-9,
            "goodness_of_fit/fit_p_value must be computed from the data, not the old hardcoded placeholders"
        );
    }

    #[test]
    fn test_detect_outliers_modified_zscore() {
        let analyzer = StatisticalAnalyzer::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100.0 is a clear outlier

        let outliers = analyzer
            .detect_outliers_modified_zscore(&data, 3.5)
            .expect("modified z-score detection should succeed");
        assert_eq!(outliers, vec![5]);
    }

    #[test]
    fn test_detect_outliers_unsupported_method_errors_honestly() {
        let analyzer = StatisticalAnalyzer::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let isolation_forest_result =
            analyzer.detect_outliers(&data, OutlierDetectionMethod::IsolationForest);
        assert!(
            isolation_forest_result.is_err(),
            "unimplemented IsolationForest must return an honest error, not silently report zero outliers"
        );

        let lof_result =
            analyzer.detect_outliers(&data, OutlierDetectionMethod::LocalOutlierFactor);
        assert!(lof_result.is_err());
    }

    #[test]
    fn test_benchmark_report_retains_timing_samples_for_baseline_comparison() {
        let mut circ: Circuit<1> = Circuit::new();
        circ.h(0).expect("h gate should apply");

        let config = BenchmarkConfig {
            num_runs: 20,
            warmup_runs: 2,
            ..BenchmarkConfig::default()
        };

        let simulator = RecordingSimulator {
            calls: std::cell::Cell::new(0),
        };

        let mut baseline_benchmark = CircuitBenchmark::new(config.clone());
        let baseline_report = baseline_benchmark
            .run_benchmark(&circ, &simulator, None)
            .expect("run_benchmark should succeed");

        assert_eq!(
            baseline_report.timing_samples.len(),
            baseline_report.completed_runs,
            "the real per-run timing samples must be retained on the report"
        );

        let mut current_benchmark = CircuitBenchmark::new(config);
        current_benchmark
            .run_benchmark(&circ, &simulator, None)
            .expect("run_benchmark should succeed");

        let comparison = current_benchmark
            .compare_with_baseline(&baseline_report)
            .expect("compare_with_baseline should succeed");

        // With the real Mann-Whitney implementation operating on the full
        // retained baseline distribution, the U statistic must reflect an
        // actual rank comparison rather than the old fixed 0.0 placeholder
        // derived from a single scalar baseline sample.
        assert!(comparison.significance.test_statistic >= 0.0);
        assert!((0.0..=1.0).contains(&comparison.significance.p_value));
        assert!(comparison.difference_ci.0 <= comparison.difference_ci.1);
    }
}
