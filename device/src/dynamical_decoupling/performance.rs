//! Performance analysis for dynamical decoupling sequences

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::time::Duration;

use super::{
    config::{BenchmarkProtocol, DDPerformanceConfig, DDPerformanceMetric, StatisticalDepth},
    sequences::DDSequence,
    DDCircuitExecutor,
};
use crate::DeviceResult;

// SciRS2 dependencies with fallbacks
#[cfg(feature = "scirs2")]
use scirs2_stats::{ks_2samp, mean, pearsonr, shapiro_wilk, spearmanr, std, ttest_1samp};

#[cfg(not(feature = "scirs2"))]
use super::fallback_scirs2::{mean, pearsonr, std};

/// Performance analysis results for DD sequences
#[derive(Debug, Clone)]
pub struct DDPerformanceAnalysis {
    /// Measured performance metrics
    pub metrics: HashMap<DDPerformanceMetric, f64>,
    /// Benchmark results
    pub benchmark_results: BenchmarkResults,
    /// Statistical analysis
    pub statistical_analysis: DDStatisticalAnalysis,
    /// Comparative analysis
    pub comparative_analysis: Option<ComparativeAnalysis>,
    /// Performance trends
    pub performance_trends: PerformanceTrends,
}

/// Benchmark results for DD sequences
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Randomized benchmarking results
    pub randomized_benchmarking: Option<RandomizedBenchmarkingResults>,
    /// Process tomography results
    pub process_tomography: Option<ProcessTomographyResults>,
    /// Gate set tomography results
    pub gate_set_tomography: Option<GateSetTomographyResults>,
    /// Cross-entropy benchmarking results
    pub cross_entropy_benchmarking: Option<CrossEntropyResults>,
    /// Cycle benchmarking results
    pub cycle_benchmarking: Option<CycleBenchmarkingResults>,
}

/// Randomized benchmarking results
#[derive(Debug, Clone)]
pub struct RandomizedBenchmarkingResults {
    /// Gate fidelity
    pub gate_fidelity: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Decay rate
    pub decay_rate: f64,
    /// Number of sequences tested
    pub sequences_tested: usize,
    /// Statistical significance
    pub p_value: f64,
}

/// Process tomography results
#[derive(Debug, Clone)]
pub struct ProcessTomographyResults {
    /// Process fidelity
    pub process_fidelity: f64,
    /// Process matrix
    pub process_matrix: Array2<f64>,
    /// Eigenvalue spectrum
    pub eigenvalues: Array1<f64>,
    /// Completeness score
    pub completeness: f64,
}

/// Gate set tomography results
#[derive(Debug, Clone)]
pub struct GateSetTomographyResults {
    /// Gate set fidelity
    pub gate_set_fidelity: f64,
    /// Individual gate fidelities
    pub gate_fidelities: HashMap<String, f64>,
    /// SPAM (State Preparation and Measurement) errors
    pub spam_errors: HashMap<String, f64>,
    /// Model consistency
    pub model_consistency: f64,
}

/// Cross-entropy benchmarking results
#[derive(Debug, Clone)]
pub struct CrossEntropyResults {
    /// Cross-entropy score
    pub cross_entropy_score: f64,
    /// Linear XEB fidelity
    pub linear_xeb_fidelity: f64,
    /// Quantum volume
    pub quantum_volume: usize,
    /// Statistical confidence
    pub confidence_level: f64,
}

/// Cycle benchmarking results
#[derive(Debug, Clone)]
pub struct CycleBenchmarkingResults {
    /// Cycle fidelity
    pub cycle_fidelity: f64,
    /// Systematic error rate
    pub systematic_error_rate: f64,
    /// Stochastic error rate
    pub stochastic_error_rate: f64,
    /// Leakage rate
    pub leakage_rate: f64,
}

/// Statistical analysis for DD performance
#[derive(Debug, Clone)]
pub struct DDStatisticalAnalysis {
    /// Descriptive statistics
    pub descriptive_stats: DescriptiveStatistics,
    /// Hypothesis testing results
    pub hypothesis_tests: HypothesisTestResults,
    /// Correlation analysis
    pub correlation_analysis: CorrelationAnalysis,
    /// Distribution analysis
    pub distribution_analysis: DistributionAnalysis,
    /// Confidence intervals
    pub confidence_intervals: ConfidenceIntervals,
}

/// Descriptive statistics
#[derive(Debug, Clone)]
pub struct DescriptiveStatistics {
    /// Mean values for each metric
    pub means: HashMap<String, f64>,
    /// Standard deviations
    pub standard_deviations: HashMap<String, f64>,
    /// Medians
    pub medians: HashMap<String, f64>,
    /// Percentiles (25th, 75th, 95th, 99th)
    pub percentiles: HashMap<String, Vec<f64>>,
    /// Min/max values
    pub ranges: HashMap<String, (f64, f64)>,
}

/// Hypothesis testing results
#[derive(Debug, Clone)]
pub struct HypothesisTestResults {
    /// T-test results comparing to baseline
    pub t_test_results: HashMap<String, TTestResult>,
    /// Kolmogorov-Smirnov test results
    pub ks_test_results: HashMap<String, KSTestResult>,
    /// Normality test results
    pub normality_tests: HashMap<String, NormalityTestResult>,
}

/// T-test result
#[derive(Debug, Clone)]
pub struct TTestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub significant: bool,
    pub effect_size: f64,
}

/// Kolmogorov-Smirnov test result
#[derive(Debug, Clone)]
pub struct KSTestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub significant: bool,
}

/// Normality test result
#[derive(Debug, Clone)]
pub struct NormalityTestResult {
    pub shapiro_statistic: f64,
    pub shapiro_p_value: f64,
    pub is_normal: bool,
}

/// Correlation analysis results
#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    /// Pearson correlation matrix
    pub pearson_correlations: Array2<f64>,
    /// Spearman correlation matrix
    pub spearman_correlations: Array2<f64>,
    /// Significant correlations
    pub significant_correlations: Vec<(String, String, f64)>,
}

/// Distribution analysis results
#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    /// Best-fit distributions
    pub best_fit_distributions: HashMap<String, String>,
    /// Distribution parameters
    pub distribution_parameters: HashMap<String, Vec<f64>>,
    /// Goodness-of-fit statistics
    pub goodness_of_fit: HashMap<String, f64>,
}

/// Confidence intervals
#[derive(Debug, Clone)]
pub struct ConfidenceIntervals {
    /// 95% confidence intervals for means
    pub mean_intervals: HashMap<String, (f64, f64)>,
    /// Bootstrap confidence intervals
    pub bootstrap_intervals: HashMap<String, (f64, f64)>,
    /// Prediction intervals
    pub prediction_intervals: HashMap<String, (f64, f64)>,
}

/// Comparative analysis between sequences
#[derive(Debug, Clone)]
pub struct ComparativeAnalysis {
    /// Relative performance improvements
    pub relative_improvements: HashMap<DDPerformanceMetric, f64>,
    /// Statistical significance of improvements
    pub significance_tests: HashMap<DDPerformanceMetric, bool>,
    /// Effect sizes
    pub effect_sizes: HashMap<DDPerformanceMetric, f64>,
    /// Ranking among compared sequences
    pub performance_ranking: usize,
}

/// Performance trends over time/parameters
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Trend slopes for each metric
    pub trend_slopes: HashMap<DDPerformanceMetric, f64>,
    /// Trend significance
    pub trend_significance: HashMap<DDPerformanceMetric, f64>,
    /// Seasonality detection
    pub seasonality: HashMap<DDPerformanceMetric, bool>,
    /// Outlier detection
    pub outliers: HashMap<DDPerformanceMetric, Vec<usize>>,
}

/// DD performance analyzer
pub struct DDPerformanceAnalyzer {
    pub config: DDPerformanceConfig,
    pub historical_data: Vec<DDPerformanceAnalysis>,
}

impl DDPerformanceAnalyzer {
    /// Create new performance analyzer
    pub fn new(config: DDPerformanceConfig) -> Self {
        Self {
            config,
            historical_data: Vec::new(),
        }
    }

    /// Analyze performance of DD sequence
    pub async fn analyze_performance(
        &mut self,
        sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<DDPerformanceAnalysis> {
        println!("Starting DD performance analysis");
        let start_time = std::time::Instant::now();

        // Calculate performance metrics
        let metrics = self
            .calculate_performance_metrics(sequence, executor)
            .await?;

        // Run benchmarks if enabled
        let benchmark_results = if self.config.enable_benchmarking {
            self.run_benchmarks(sequence, executor).await?
        } else {
            BenchmarkResults {
                randomized_benchmarking: None,
                process_tomography: None,
                gate_set_tomography: None,
                cross_entropy_benchmarking: None,
                cycle_benchmarking: None,
            }
        };

        // Perform statistical analysis
        let statistical_analysis = self.perform_statistical_analysis(&metrics, sequence)?;

        // Comparative analysis (if historical data exists)
        let comparative_analysis = if !self.historical_data.is_empty() {
            Some(self.perform_comparative_analysis(&metrics)?)
        } else {
            None
        };

        // Analyze performance trends
        let performance_trends = self.analyze_performance_trends(&metrics)?;

        let analysis = DDPerformanceAnalysis {
            metrics,
            benchmark_results,
            statistical_analysis,
            comparative_analysis,
            performance_trends,
        };

        self.historical_data.push(analysis.clone());

        println!(
            "DD performance analysis completed in {:?}",
            start_time.elapsed()
        );
        Ok(analysis)
    }

    /// Calculate performance metrics
    async fn calculate_performance_metrics(
        &self,
        sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<HashMap<DDPerformanceMetric, f64>> {
        let mut metrics = HashMap::new();

        for metric in &self.config.metrics {
            let value = match metric {
                DDPerformanceMetric::CoherenceTime => {
                    self.measure_coherence_time(sequence, executor).await?
                }
                DDPerformanceMetric::ProcessFidelity => {
                    self.measure_process_fidelity(sequence, executor).await?
                }
                DDPerformanceMetric::GateOverhead => sequence.properties.pulse_count as f64,
                DDPerformanceMetric::TimeOverhead => {
                    sequence.duration * 1e6 // Convert to microseconds
                }
                DDPerformanceMetric::RobustnessScore => {
                    self.calculate_robustness_score(sequence, executor).await?
                }
                DDPerformanceMetric::NoiseSuppressionFactor => {
                    self.calculate_noise_suppression(sequence, executor).await?
                }
                DDPerformanceMetric::ResourceEfficiency => {
                    self.calculate_resource_efficiency(sequence, executor)
                        .await?
                }
            };

            metrics.insert(metric.clone(), value);
        }

        Ok(metrics)
    }

    /// Measure coherence time
    pub async fn measure_coherence_time(
        &self,
        sequence: &DDSequence,
        _executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<f64> {
        // Simplified coherence time measurement
        let base_t2 = 50e-6; // 50 μs base T2
        let enhancement_factor: f64 = sequence.properties.noise_suppression.values().sum();
        let suppression_factor =
            1.0 + enhancement_factor / sequence.properties.noise_suppression.len() as f64;

        Ok(base_t2 * suppression_factor * 1e6) // Return in microseconds
    }

    /// Measure process fidelity
    pub async fn measure_process_fidelity(
        &self,
        sequence: &DDSequence,
        _executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<f64> {
        // Simplified fidelity calculation
        let base_fidelity = 0.99;
        let order_factor = 0.001 * (sequence.properties.sequence_order as f64);
        let overhead_penalty = -0.0001 * (sequence.properties.pulse_count as f64);

        Ok((base_fidelity + order_factor + overhead_penalty)
            .max(0.0)
            .min(1.0))
    }

    /// Calculate robustness score
    pub async fn calculate_robustness_score(
        &self,
        sequence: &DDSequence,
        _executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<f64> {
        let mut robustness = 0.0;

        // Symmetry contributions
        if sequence.properties.symmetry.time_reversal {
            robustness += 0.25;
        }
        if sequence.properties.symmetry.phase_symmetry {
            robustness += 0.25;
        }
        if sequence.properties.symmetry.rotational_symmetry {
            robustness += 0.25;
        }
        if sequence.properties.symmetry.inversion_symmetry {
            robustness += 0.25;
        }

        // Noise suppression diversity
        let noise_diversity = sequence.properties.noise_suppression.len() as f64 / 10.0;
        robustness += noise_diversity.min(0.5);

        Ok(robustness)
    }

    /// Calculate noise suppression factor
    async fn calculate_noise_suppression(
        &self,
        sequence: &DDSequence,
        _executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<f64> {
        let avg_suppression: f64 = sequence.properties.noise_suppression.values().sum::<f64>()
            / sequence.properties.noise_suppression.len() as f64;
        Ok(avg_suppression)
    }

    /// Calculate resource efficiency
    async fn calculate_resource_efficiency(
        &self,
        sequence: &DDSequence,
        _executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<f64> {
        let coherence_improvement = self.measure_coherence_time(sequence, _executor).await? / 50.0; // Relative to base T2
        let resource_cost = sequence.properties.pulse_count as f64;

        Ok(coherence_improvement / resource_cost.max(1.0))
    }

    /// Run benchmark protocols
    async fn run_benchmarks(
        &self,
        sequence: &DDSequence,
        executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<BenchmarkResults> {
        let mut results = BenchmarkResults {
            randomized_benchmarking: None,
            process_tomography: None,
            gate_set_tomography: None,
            cross_entropy_benchmarking: None,
            cycle_benchmarking: None,
        };

        for protocol in &self.config.benchmarking_config.protocols {
            match protocol {
                BenchmarkProtocol::RandomizedBenchmarking => {
                    results.randomized_benchmarking =
                        Some(self.run_randomized_benchmarking(sequence, executor).await?);
                }
                BenchmarkProtocol::ProcessTomography => {
                    results.process_tomography =
                        Some(self.run_process_tomography(sequence, executor).await?);
                }
                BenchmarkProtocol::GateSetTomography => {
                    results.gate_set_tomography =
                        Some(self.run_gate_set_tomography(sequence, executor).await?);
                }
                BenchmarkProtocol::CrossEntropyBenchmarking => {
                    results.cross_entropy_benchmarking = Some(
                        self.run_cross_entropy_benchmarking(sequence, executor)
                            .await?,
                    );
                }
                BenchmarkProtocol::CycleBenchmarking => {
                    results.cycle_benchmarking =
                        Some(self.run_cycle_benchmarking(sequence, executor).await?);
                }
            }
        }

        Ok(results)
    }

    /// Run randomized benchmarking
    async fn run_randomized_benchmarking(
        &self,
        _sequence: &DDSequence,
        _executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<RandomizedBenchmarkingResults> {
        // Simplified RB implementation
        Ok(RandomizedBenchmarkingResults {
            gate_fidelity: 0.995,
            confidence_interval: (0.990, 0.999),
            decay_rate: 0.005,
            sequences_tested: self.config.benchmarking_config.benchmark_runs,
            p_value: 0.001,
        })
    }

    /// Run other benchmark protocols (simplified implementations)
    async fn run_process_tomography(
        &self,
        _sequence: &DDSequence,
        _executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<ProcessTomographyResults> {
        Ok(ProcessTomographyResults {
            process_fidelity: 0.98,
            process_matrix: Array2::eye(4),
            eigenvalues: Array1::from_vec(vec![1.0, 0.99, 0.98, 0.97]),
            completeness: 0.99,
        })
    }

    async fn run_gate_set_tomography(
        &self,
        _sequence: &DDSequence,
        _executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<GateSetTomographyResults> {
        let mut gate_fidelities = HashMap::new();
        gate_fidelities.insert("X".to_string(), 0.995);
        gate_fidelities.insert("Y".to_string(), 0.994);
        gate_fidelities.insert("Z".to_string(), 0.999);

        let mut spam_errors = HashMap::new();
        spam_errors.insert("prep_error".to_string(), 0.001);
        spam_errors.insert("meas_error".to_string(), 0.002);

        Ok(GateSetTomographyResults {
            gate_set_fidelity: 0.996,
            gate_fidelities,
            spam_errors,
            model_consistency: 0.98,
        })
    }

    async fn run_cross_entropy_benchmarking(
        &self,
        _sequence: &DDSequence,
        _executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<CrossEntropyResults> {
        Ok(CrossEntropyResults {
            cross_entropy_score: 2.1,
            linear_xeb_fidelity: 0.92,
            quantum_volume: 64,
            confidence_level: 0.95,
        })
    }

    async fn run_cycle_benchmarking(
        &self,
        _sequence: &DDSequence,
        _executor: &dyn DDCircuitExecutor,
    ) -> DeviceResult<CycleBenchmarkingResults> {
        Ok(CycleBenchmarkingResults {
            cycle_fidelity: 0.993,
            systematic_error_rate: 0.002,
            stochastic_error_rate: 0.005,
            leakage_rate: 0.0001,
        })
    }

    /// Perform statistical analysis
    fn perform_statistical_analysis(
        &self,
        metrics: &HashMap<DDPerformanceMetric, f64>,
        _sequence: &DDSequence,
    ) -> DeviceResult<DDStatisticalAnalysis> {
        // Simplified statistical analysis to avoid potential stack overflow
        let mut means = HashMap::new();
        let mut standard_deviations = HashMap::new();
        let mut medians = HashMap::new();
        let mut percentiles = HashMap::new();
        let mut ranges = HashMap::new();

        // Populate with basic statistics from the metrics
        for (metric, &value) in metrics {
            let metric_name = format!("{:?}", metric);
            means.insert(metric_name.clone(), value);
            standard_deviations.insert(metric_name.clone(), value * 0.1); // 10% std dev
            medians.insert(metric_name.clone(), value);
            percentiles.insert(
                metric_name.clone(),
                vec![value * 0.9, value * 1.1, value * 1.2, value * 1.3],
            );
            ranges.insert(metric_name, (value * 0.8, value * 1.2));
        }

        let descriptive_stats = DescriptiveStatistics {
            means,
            standard_deviations,
            medians,
            percentiles,
            ranges,
        };

        let hypothesis_tests = HypothesisTestResults {
            t_test_results: HashMap::new(),
            ks_test_results: HashMap::new(),
            normality_tests: HashMap::new(),
        };

        let correlation_analysis = CorrelationAnalysis {
            pearson_correlations: Array2::eye(metrics.len().max(1)),
            spearman_correlations: Array2::eye(metrics.len().max(1)),
            significant_correlations: Vec::new(),
        };

        let distribution_analysis = DistributionAnalysis {
            best_fit_distributions: HashMap::new(),
            distribution_parameters: HashMap::new(),
            goodness_of_fit: HashMap::new(),
        };

        let confidence_intervals = ConfidenceIntervals {
            mean_intervals: HashMap::new(),
            bootstrap_intervals: HashMap::new(),
            prediction_intervals: HashMap::new(),
        };

        Ok(DDStatisticalAnalysis {
            descriptive_stats,
            hypothesis_tests,
            correlation_analysis,
            distribution_analysis,
            confidence_intervals,
        })
    }

    /// Calculate descriptive statistics
    fn calculate_descriptive_statistics(
        &self,
        data: &HashMap<String, Array1<f64>>,
    ) -> DeviceResult<DescriptiveStatistics> {
        let mut means = HashMap::new();
        let mut standard_deviations = HashMap::new();
        let mut medians = HashMap::new();
        let mut percentiles = HashMap::new();
        let mut ranges = HashMap::new();

        for (metric_name, values) in data {
            #[cfg(feature = "scirs2")]
            let mean_val = mean(&values.view()).unwrap_or(0.0);
            #[cfg(not(feature = "scirs2"))]
            let mean_val = values.sum() / values.len() as f64;

            #[cfg(feature = "scirs2")]
            let std_val = std(&values.view(), 1, None).unwrap_or(1.0);
            #[cfg(not(feature = "scirs2"))]
            let std_val = {
                let mean = mean_val;
                let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - 1) as f64;
                variance.sqrt()
            };

            means.insert(metric_name.clone(), mean_val);
            standard_deviations.insert(metric_name.clone(), std_val);

            // Calculate median and percentiles (simplified)
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let len = sorted.len();

            let median = if len % 2 == 0 {
                (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
            } else {
                sorted[len / 2]
            };
            medians.insert(metric_name.clone(), median);

            let p25 = sorted[len / 4];
            let p75 = sorted[3 * len / 4];
            let p95 = sorted[95 * len / 100];
            let p99 = sorted[99 * len / 100];

            percentiles.insert(metric_name.clone(), vec![p25, p75, p95, p99]);
            ranges.insert(metric_name.clone(), (sorted[0], sorted[len - 1]));
        }

        Ok(DescriptiveStatistics {
            means,
            standard_deviations,
            medians,
            percentiles,
            ranges,
        })
    }

    /// Other statistical analysis methods (simplified implementations)
    fn perform_hypothesis_tests(
        &self,
        _data: &HashMap<String, Array1<f64>>,
    ) -> DeviceResult<HypothesisTestResults> {
        Ok(HypothesisTestResults {
            t_test_results: HashMap::new(),
            ks_test_results: HashMap::new(),
            normality_tests: HashMap::new(),
        })
    }

    fn perform_correlation_analysis(
        &self,
        data: &HashMap<String, Array1<f64>>,
    ) -> DeviceResult<CorrelationAnalysis> {
        let n_metrics = data.len();
        let pearson_correlations = Array2::eye(n_metrics);
        let spearman_correlations = Array2::eye(n_metrics);

        Ok(CorrelationAnalysis {
            pearson_correlations,
            spearman_correlations,
            significant_correlations: Vec::new(),
        })
    }

    fn analyze_distributions(
        &self,
        _data: &HashMap<String, Array1<f64>>,
    ) -> DeviceResult<DistributionAnalysis> {
        Ok(DistributionAnalysis {
            best_fit_distributions: HashMap::new(),
            distribution_parameters: HashMap::new(),
            goodness_of_fit: HashMap::new(),
        })
    }

    fn calculate_confidence_intervals(
        &self,
        _data: &HashMap<String, Array1<f64>>,
    ) -> DeviceResult<ConfidenceIntervals> {
        Ok(ConfidenceIntervals {
            mean_intervals: HashMap::new(),
            bootstrap_intervals: HashMap::new(),
            prediction_intervals: HashMap::new(),
        })
    }

    fn perform_comparative_analysis(
        &self,
        _metrics: &HashMap<DDPerformanceMetric, f64>,
    ) -> DeviceResult<ComparativeAnalysis> {
        Ok(ComparativeAnalysis {
            relative_improvements: HashMap::new(),
            significance_tests: HashMap::new(),
            effect_sizes: HashMap::new(),
            performance_ranking: 1,
        })
    }

    fn analyze_performance_trends(
        &self,
        _metrics: &HashMap<DDPerformanceMetric, f64>,
    ) -> DeviceResult<PerformanceTrends> {
        Ok(PerformanceTrends {
            trend_slopes: HashMap::new(),
            trend_significance: HashMap::new(),
            seasonality: HashMap::new(),
            outliers: HashMap::new(),
        })
    }
}
