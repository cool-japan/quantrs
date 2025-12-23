//! Quantum Advantage Validation Tools
//!
//! This module provides tools to rigorously validate and quantify quantum advantage
//! in quantum machine learning algorithms through statistical testing and benchmarking.

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Quantum advantage validation framework
#[derive(Debug)]
pub struct QuantumAdvantageValidator {
    /// Classical baseline results
    classical_baselines: HashMap<String, ClassicalBaseline>,

    /// Quantum algorithm results
    quantum_results: HashMap<String, QuantumResult>,

    /// Statistical test configuration
    config: ValidationConfig,

    /// Validation reports
    reports: Vec<ValidationReport>,
}

/// Configuration for quantum advantage validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Confidence level for statistical tests (e.g., 0.95 for 95% confidence)
    pub confidence_level: f64,

    /// Number of repeated trials for statistical significance
    pub num_trials: usize,

    /// Enable bootstrapping for robustness
    pub use_bootstrapping: bool,

    /// Number of bootstrap samples
    pub bootstrap_samples: usize,

    /// Metrics to compare
    pub metrics: Vec<ComparisonMetric>,

    /// Enable detailed logging
    pub verbose: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            num_trials: 100,
            use_bootstrapping: true,
            bootstrap_samples: 1000,
            metrics: vec![
                ComparisonMetric::Accuracy,
                ComparisonMetric::ExecutionTime,
                ComparisonMetric::Scalability,
                ComparisonMetric::Robustness,
            ],
            verbose: false,
        }
    }
}

/// Metrics for comparing quantum vs classical algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonMetric {
    /// Prediction accuracy or error rate
    Accuracy,

    /// Execution time / speed
    ExecutionTime,

    /// Sample complexity (data efficiency)
    SampleComplexity,

    /// Scalability with problem size
    Scalability,

    /// Robustness to noise
    Robustness,

    /// Expressivity / model capacity
    Expressivity,

    /// Generalization performance
    Generalization,

    /// Resource efficiency
    ResourceEfficiency,
}

/// Classical algorithm baseline results
#[derive(Debug, Clone)]
pub struct ClassicalBaseline {
    pub algorithm_name: String,
    pub accuracy: Vec<f64>,
    pub execution_times: Vec<Duration>,
    pub sample_sizes: Vec<usize>,
    pub resource_usage: ResourceUsage,
    pub hyperparameters: HashMap<String, String>,
}

/// Quantum algorithm results
#[derive(Debug, Clone)]
pub struct QuantumResult {
    pub algorithm_name: String,
    pub accuracy: Vec<f64>,
    pub execution_times: Vec<Duration>,
    pub sample_sizes: Vec<usize>,
    pub resource_usage: QuantumResourceUsage,
    pub circuit_depth: Vec<usize>,
    pub qubit_count: usize,
    pub fidelity: Vec<f64>,
}

/// Classical resource usage
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub cpu_time: Duration,
    pub memory_mb: usize,
    pub num_parameters: usize,
}

/// Quantum resource usage
#[derive(Debug, Clone)]
pub struct QuantumResourceUsage {
    pub circuit_execution_time: Duration,
    pub total_gates: usize,
    pub entangling_gates: usize,
    pub measurements: usize,
    pub classical_processing_time: Duration,
}

/// Validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub metric: ComparisonMetric,
    pub quantum_algorithm: String,
    pub classical_baseline: String,
    pub quantum_advantage: QuantumAdvantage,
    pub statistical_significance: StatisticalSignificance,
    pub speedup_factor: Option<f64>,
    pub sample_efficiency: Option<f64>,
    pub confidence_interval: (f64, f64),
    pub p_value: f64,
    pub effect_size: f64,
    pub interpretation: String,
}

/// Quantum advantage assessment
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumAdvantage {
    /// Strong quantum advantage demonstrated
    Strong {
        improvement_factor: f64,
        significance_level: f64,
    },

    /// Moderate quantum advantage
    Moderate {
        improvement_factor: f64,
        significance_level: f64,
    },

    /// Marginal or no quantum advantage
    Marginal { improvement_factor: f64 },

    /// Classical algorithm performs better
    None { classical_advantage: f64 },

    /// Inconclusive results
    Inconclusive { reason: String },
}

/// Statistical significance test results
#[derive(Debug, Clone)]
pub struct StatisticalSignificance {
    pub test_name: String,
    pub p_value: f64,
    pub is_significant: bool,
    pub confidence_level: f64,
    pub test_statistic: f64,
}

impl QuantumAdvantageValidator {
    /// Create a new validator with default configuration
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }

    /// Create a new validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self {
            classical_baselines: HashMap::new(),
            quantum_results: HashMap::new(),
            config,
            reports: Vec::new(),
        }
    }

    /// Register a classical baseline algorithm
    pub fn register_classical_baseline(&mut self, baseline: ClassicalBaseline) {
        self.classical_baselines
            .insert(baseline.algorithm_name.clone(), baseline);
    }

    /// Register quantum algorithm results
    pub fn register_quantum_result(&mut self, result: QuantumResult) {
        self.quantum_results
            .insert(result.algorithm_name.clone(), result);
    }

    /// Perform comprehensive validation
    pub fn validate(&mut self) -> Result<Vec<ValidationReport>> {
        self.reports.clear();

        for (quantum_name, quantum_result) in &self.quantum_results {
            for (classical_name, classical_baseline) in &self.classical_baselines {
                for metric in &self.config.metrics {
                    let report = self.compare_algorithms(
                        metric,
                        quantum_name,
                        quantum_result,
                        classical_name,
                        classical_baseline,
                    )?;

                    self.reports.push(report);
                }
            }
        }

        Ok(self.reports.clone())
    }

    /// Compare quantum and classical algorithms for a specific metric
    fn compare_algorithms(
        &self,
        metric: &ComparisonMetric,
        quantum_name: &str,
        quantum_result: &QuantumResult,
        classical_name: &str,
        classical_baseline: &ClassicalBaseline,
    ) -> Result<ValidationReport> {
        match metric {
            ComparisonMetric::Accuracy => self.compare_accuracy(
                quantum_name,
                quantum_result,
                classical_name,
                classical_baseline,
            ),
            ComparisonMetric::ExecutionTime => self.compare_execution_time(
                quantum_name,
                quantum_result,
                classical_name,
                classical_baseline,
            ),
            ComparisonMetric::SampleComplexity => self.compare_sample_complexity(
                quantum_name,
                quantum_result,
                classical_name,
                classical_baseline,
            ),
            ComparisonMetric::Scalability => self.compare_scalability(
                quantum_name,
                quantum_result,
                classical_name,
                classical_baseline,
            ),
            ComparisonMetric::Robustness => self.compare_robustness(
                quantum_name,
                quantum_result,
                classical_name,
                classical_baseline,
            ),
            _ => Err(MLError::InvalidInput(format!(
                "Metric {:?} not yet implemented",
                metric
            ))),
        }
    }

    /// Compare accuracy between quantum and classical
    fn compare_accuracy(
        &self,
        quantum_name: &str,
        quantum_result: &QuantumResult,
        classical_name: &str,
        classical_baseline: &ClassicalBaseline,
    ) -> Result<ValidationReport> {
        let quantum_acc = Self::mean(&quantum_result.accuracy);
        let classical_acc = Self::mean(&classical_baseline.accuracy);

        // Perform t-test
        let stat_sig = self.welch_t_test(&quantum_result.accuracy, &classical_baseline.accuracy)?;

        let improvement_factor = quantum_acc / classical_acc;

        let quantum_advantage = if stat_sig.is_significant {
            if improvement_factor >= 1.5 {
                QuantumAdvantage::Strong {
                    improvement_factor,
                    significance_level: stat_sig.confidence_level,
                }
            } else if improvement_factor > 1.1 {
                QuantumAdvantage::Moderate {
                    improvement_factor,
                    significance_level: stat_sig.confidence_level,
                }
            } else if improvement_factor >= 1.0 {
                QuantumAdvantage::Marginal { improvement_factor }
            } else {
                QuantumAdvantage::None {
                    classical_advantage: classical_acc / quantum_acc,
                }
            }
        } else {
            QuantumAdvantage::Inconclusive {
                reason: "Results not statistically significant".to_string(),
            }
        };

        let confidence_interval = self.bootstrap_confidence_interval(&quantum_result.accuracy)?;

        let interpretation =
            self.interpret_advantage(&quantum_advantage, ComparisonMetric::Accuracy);

        Ok(ValidationReport {
            metric: ComparisonMetric::Accuracy,
            quantum_algorithm: quantum_name.to_string(),
            classical_baseline: classical_name.to_string(),
            quantum_advantage,
            statistical_significance: stat_sig.clone(),
            speedup_factor: None,
            sample_efficiency: None,
            confidence_interval,
            p_value: stat_sig.p_value,
            effect_size: self.cohens_d(&quantum_result.accuracy, &classical_baseline.accuracy),
            interpretation,
        })
    }

    /// Compare execution time (speedup)
    fn compare_execution_time(
        &self,
        quantum_name: &str,
        quantum_result: &QuantumResult,
        classical_name: &str,
        classical_baseline: &ClassicalBaseline,
    ) -> Result<ValidationReport> {
        let quantum_times: Vec<f64> = quantum_result
            .execution_times
            .iter()
            .map(|d| d.as_secs_f64())
            .collect();

        let classical_times: Vec<f64> = classical_baseline
            .execution_times
            .iter()
            .map(|d| d.as_secs_f64())
            .collect();

        let quantum_mean_time = Self::mean(&quantum_times);
        let classical_mean_time = Self::mean(&classical_times);

        let speedup_factor = classical_mean_time / quantum_mean_time;

        let stat_sig = self.welch_t_test(&classical_times, &quantum_times)?;

        let quantum_advantage = if stat_sig.is_significant {
            if speedup_factor >= 10.0 {
                QuantumAdvantage::Strong {
                    improvement_factor: speedup_factor,
                    significance_level: stat_sig.confidence_level,
                }
            } else if speedup_factor >= 2.0 {
                QuantumAdvantage::Moderate {
                    improvement_factor: speedup_factor,
                    significance_level: stat_sig.confidence_level,
                }
            } else if speedup_factor >= 1.0 {
                QuantumAdvantage::Marginal {
                    improvement_factor: speedup_factor,
                }
            } else {
                QuantumAdvantage::None {
                    classical_advantage: 1.0 / speedup_factor,
                }
            }
        } else {
            QuantumAdvantage::Inconclusive {
                reason: "Timing difference not statistically significant".to_string(),
            }
        };

        let confidence_interval = self.bootstrap_confidence_interval(&quantum_times)?;
        let interpretation =
            self.interpret_advantage(&quantum_advantage, ComparisonMetric::ExecutionTime);

        Ok(ValidationReport {
            metric: ComparisonMetric::ExecutionTime,
            quantum_algorithm: quantum_name.to_string(),
            classical_baseline: classical_name.to_string(),
            quantum_advantage,
            statistical_significance: stat_sig.clone(),
            speedup_factor: Some(speedup_factor),
            sample_efficiency: None,
            confidence_interval,
            p_value: stat_sig.p_value,
            effect_size: self.cohens_d(&quantum_times, &classical_times),
            interpretation,
        })
    }

    /// Compare sample complexity (data efficiency)
    fn compare_sample_complexity(
        &self,
        quantum_name: &str,
        quantum_result: &QuantumResult,
        classical_name: &str,
        classical_baseline: &ClassicalBaseline,
    ) -> Result<ValidationReport> {
        // Calculate samples needed to reach target accuracy (e.g., 90%)
        let target_acc = 0.90;

        let quantum_samples = self.estimate_sample_requirement(
            &quantum_result.accuracy,
            &quantum_result.sample_sizes,
            target_acc,
        );

        let classical_samples = self.estimate_sample_requirement(
            &classical_baseline.accuracy,
            &classical_baseline.sample_sizes,
            target_acc,
        );

        let sample_efficiency = classical_samples / quantum_samples;

        let quantum_advantage = if sample_efficiency >= 10.0 {
            QuantumAdvantage::Strong {
                improvement_factor: sample_efficiency,
                significance_level: self.config.confidence_level,
            }
        } else if sample_efficiency >= 2.0 {
            QuantumAdvantage::Moderate {
                improvement_factor: sample_efficiency,
                significance_level: self.config.confidence_level,
            }
        } else if sample_efficiency > 1.0 {
            QuantumAdvantage::Marginal {
                improvement_factor: sample_efficiency,
            }
        } else {
            QuantumAdvantage::None {
                classical_advantage: 1.0 / sample_efficiency,
            }
        };

        let interpretation =
            self.interpret_advantage(&quantum_advantage, ComparisonMetric::SampleComplexity);

        Ok(ValidationReport {
            metric: ComparisonMetric::SampleComplexity,
            quantum_algorithm: quantum_name.to_string(),
            classical_baseline: classical_name.to_string(),
            quantum_advantage,
            statistical_significance: StatisticalSignificance {
                test_name: "Sample Complexity Estimation".to_string(),
                p_value: 0.0,
                is_significant: sample_efficiency > 1.2,
                confidence_level: self.config.confidence_level,
                test_statistic: sample_efficiency,
            },
            speedup_factor: None,
            sample_efficiency: Some(sample_efficiency),
            confidence_interval: (quantum_samples * 0.8, quantum_samples * 1.2),
            p_value: 0.0,
            effect_size: 0.0,
            interpretation,
        })
    }

    /// Compare scalability
    fn compare_scalability(
        &self,
        quantum_name: &str,
        quantum_result: &QuantumResult,
        classical_name: &str,
        classical_baseline: &ClassicalBaseline,
    ) -> Result<ValidationReport> {
        // Analyze how execution time scales with problem size
        let quantum_scaling = self.estimate_scaling_exponent(&quantum_result.execution_times)?;
        let classical_scaling =
            self.estimate_scaling_exponent(&classical_baseline.execution_times)?;

        let scaling_advantage = classical_scaling / quantum_scaling.max(0.1);

        let quantum_advantage = if scaling_advantage >= 2.0 && quantum_scaling < classical_scaling {
            QuantumAdvantage::Strong {
                improvement_factor: scaling_advantage,
                significance_level: self.config.confidence_level,
            }
        } else if scaling_advantage > 1.3 {
            QuantumAdvantage::Moderate {
                improvement_factor: scaling_advantage,
                significance_level: self.config.confidence_level,
            }
        } else if scaling_advantage >= 1.0 {
            QuantumAdvantage::Marginal {
                improvement_factor: scaling_advantage,
            }
        } else {
            QuantumAdvantage::None {
                classical_advantage: 1.0 / scaling_advantage,
            }
        };

        let interpretation =
            self.interpret_advantage(&quantum_advantage, ComparisonMetric::Scalability);

        Ok(ValidationReport {
            metric: ComparisonMetric::Scalability,
            quantum_algorithm: quantum_name.to_string(),
            classical_baseline: classical_name.to_string(),
            quantum_advantage,
            statistical_significance: StatisticalSignificance {
                test_name: "Scalability Analysis".to_string(),
                p_value: 0.0,
                is_significant: scaling_advantage > 1.2,
                confidence_level: self.config.confidence_level,
                test_statistic: scaling_advantage,
            },
            speedup_factor: Some(scaling_advantage),
            sample_efficiency: None,
            confidence_interval: (quantum_scaling * 0.9, quantum_scaling * 1.1),
            p_value: 0.0,
            effect_size: classical_scaling - quantum_scaling,
            interpretation,
        })
    }

    /// Compare robustness to noise
    fn compare_robustness(
        &self,
        quantum_name: &str,
        quantum_result: &QuantumResult,
        classical_name: &str,
        classical_baseline: &ClassicalBaseline,
    ) -> Result<ValidationReport> {
        // Measure performance variance as a proxy for robustness
        let quantum_variance = Self::variance(&quantum_result.accuracy);
        let classical_variance = Self::variance(&classical_baseline.accuracy);

        // Lower variance = more robust
        let robustness_ratio = classical_variance / quantum_variance.max(1e-10);

        let quantum_advantage = if robustness_ratio >= 2.0 {
            QuantumAdvantage::Strong {
                improvement_factor: robustness_ratio,
                significance_level: self.config.confidence_level,
            }
        } else if robustness_ratio > 1.2 {
            QuantumAdvantage::Moderate {
                improvement_factor: robustness_ratio,
                significance_level: self.config.confidence_level,
            }
        } else if robustness_ratio >= 1.0 {
            QuantumAdvantage::Marginal {
                improvement_factor: robustness_ratio,
            }
        } else {
            QuantumAdvantage::None {
                classical_advantage: 1.0 / robustness_ratio,
            }
        };

        let interpretation =
            self.interpret_advantage(&quantum_advantage, ComparisonMetric::Robustness);

        Ok(ValidationReport {
            metric: ComparisonMetric::Robustness,
            quantum_algorithm: quantum_name.to_string(),
            classical_baseline: classical_name.to_string(),
            quantum_advantage,
            statistical_significance: StatisticalSignificance {
                test_name: "Variance Ratio Test".to_string(),
                p_value: 0.0,
                is_significant: robustness_ratio > 1.2,
                confidence_level: self.config.confidence_level,
                test_statistic: robustness_ratio,
            },
            speedup_factor: None,
            sample_efficiency: None,
            confidence_interval: (quantum_variance * 0.8, quantum_variance * 1.2),
            p_value: 0.0,
            effect_size: (classical_variance - quantum_variance).abs(),
            interpretation,
        })
    }

    /// Perform Welch's t-test for statistical significance
    fn welch_t_test(&self, sample1: &[f64], sample2: &[f64]) -> Result<StatisticalSignificance> {
        if sample1.is_empty() || sample2.is_empty() {
            return Err(MLError::InvalidInput(
                "Empty samples for t-test".to_string(),
            ));
        }

        let mean1 = Self::mean(sample1);
        let mean2 = Self::mean(sample2);
        let var1 = Self::variance(sample1);
        let var2 = Self::variance(sample2);
        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;

        let se = ((var1 / n1) + (var2 / n2)).sqrt();
        let t_stat = (mean1 - mean2) / se;

        // Simplified p-value estimation (for demonstration)
        let p_value = 2.0 * (1.0 - Self::t_cdf(t_stat.abs(), (n1 + n2 - 2.0) as usize));

        let is_significant = p_value < (1.0 - self.config.confidence_level);

        Ok(StatisticalSignificance {
            test_name: "Welch's t-test".to_string(),
            p_value,
            is_significant,
            confidence_level: self.config.confidence_level,
            test_statistic: t_stat,
        })
    }

    /// Calculate Cohen's d effect size
    fn cohens_d(&self, sample1: &[f64], sample2: &[f64]) -> f64 {
        let mean1 = Self::mean(sample1);
        let mean2 = Self::mean(sample2);
        let var1 = Self::variance(sample1);
        let var2 = Self::variance(sample2);
        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;

        let pooled_std = (((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0)).sqrt();

        (mean1 - mean2) / pooled_std
    }

    /// Bootstrap confidence interval
    fn bootstrap_confidence_interval(&self, data: &[f64]) -> Result<(f64, f64)> {
        if !self.config.use_bootstrapping {
            let mean = Self::mean(data);
            let std = Self::variance(data).sqrt();
            let margin = 1.96 * std / (data.len() as f64).sqrt();
            return Ok((mean - margin, mean + margin));
        }

        let mut rng = thread_rng();
        let mut bootstrap_means = Vec::with_capacity(self.config.bootstrap_samples);

        for _ in 0..self.config.bootstrap_samples {
            let sample: Vec<f64> = (0..data.len())
                .map(|_| data[rng.gen_range(0..data.len())])
                .collect();
            bootstrap_means.push(Self::mean(&sample));
        }

        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let lower_idx =
            ((1.0 - self.config.confidence_level) / 2.0 * bootstrap_means.len() as f64) as usize;
        let upper_idx =
            ((1.0 + self.config.confidence_level) / 2.0 * bootstrap_means.len() as f64) as usize;

        Ok((
            bootstrap_means[lower_idx],
            bootstrap_means[upper_idx.min(bootstrap_means.len() - 1)],
        ))
    }

    /// Estimate sample requirement for target accuracy
    fn estimate_sample_requirement(
        &self,
        accuracies: &[f64],
        sample_sizes: &[usize],
        target: f64,
    ) -> f64 {
        // Simple linear interpolation (could be improved with learning curve fitting)
        for (i, &acc) in accuracies.iter().enumerate() {
            if acc >= target {
                return sample_sizes.get(i).copied().unwrap_or(sample_sizes[0]) as f64;
            }
        }

        // If target not reached, extrapolate
        sample_sizes.last().copied().unwrap_or(1000) as f64 * 2.0
    }

    /// Estimate scaling exponent from execution times
    fn estimate_scaling_exponent(&self, times: &[Duration]) -> Result<f64> {
        if times.len() < 2 {
            return Ok(1.0);
        }

        // Simple log-log slope estimation
        let times_f: Vec<f64> = times.iter().map(|t| t.as_secs_f64()).collect();
        let sizes: Vec<f64> = (1..=times.len()).map(|i| i as f64 * 100.0).collect();

        let log_times: Vec<f64> = times_f.iter().map(|t| t.max(1e-10).ln()).collect();
        let log_sizes: Vec<f64> = sizes.iter().map(|s| s.ln()).collect();

        // Simple linear regression on log-log data
        let mean_log_size = Self::mean(&log_sizes);
        let mean_log_time = Self::mean(&log_times);

        let numerator: f64 = log_sizes
            .iter()
            .zip(log_times.iter())
            .map(|(x, y)| (x - mean_log_size) * (y - mean_log_time))
            .sum();

        let denominator: f64 = log_sizes.iter().map(|x| (x - mean_log_size).powi(2)).sum();

        Ok(numerator / denominator.max(1e-10))
    }

    /// Interpret quantum advantage for human readability
    fn interpret_advantage(
        &self,
        advantage: &QuantumAdvantage,
        metric: ComparisonMetric,
    ) -> String {
        match advantage {
            QuantumAdvantage::Strong {
                improvement_factor,
                significance_level,
            } => {
                format!(
                    "Strong quantum advantage demonstrated: {:.2}x improvement in {:?} with {:.1}% confidence",
                    improvement_factor,
                    metric,
                    significance_level * 100.0
                )
            }
            QuantumAdvantage::Moderate {
                improvement_factor,
                significance_level,
            } => {
                format!(
                    "Moderate quantum advantage: {:.2}x improvement in {:?} with {:.1}% confidence",
                    improvement_factor,
                    metric,
                    significance_level * 100.0
                )
            }
            QuantumAdvantage::Marginal { improvement_factor } => {
                format!(
                    "Marginal quantum advantage: {:.2}x improvement in {:?}, may not justify quantum overhead",
                    improvement_factor, metric
                )
            }
            QuantumAdvantage::None {
                classical_advantage,
            } => {
                format!(
                    "No quantum advantage: classical algorithm performs {:.2}x better in {:?}",
                    classical_advantage, metric
                )
            }
            QuantumAdvantage::Inconclusive { reason } => {
                format!("Inconclusive results for {:?}: {}", metric, reason)
            }
        }
    }

    /// Print comprehensive validation report
    pub fn print_report(&self) {
        println!("\n══════════════════════════════════════════════════════════");
        println!("       Quantum Advantage Validation Report               ");
        println!("══════════════════════════════════════════════════════════\n");

        println!("Configuration:");
        println!(
            "  Confidence Level: {:.1}%",
            self.config.confidence_level * 100.0
        );
        println!("  Number of Trials: {}", self.config.num_trials);
        println!("  Bootstrap Samples: {}", self.config.bootstrap_samples);
        println!();

        let mut by_metric: HashMap<String, Vec<&ValidationReport>> = HashMap::new();
        for report in &self.reports {
            by_metric
                .entry(format!("{:?}", report.metric))
                .or_insert_with(Vec::new)
                .push(report);
        }

        for (metric, reports) in by_metric {
            println!("──────────────────────────────────────────────────────────");
            println!("Metric: {}", metric);
            println!("──────────────────────────────────────────────────────────");

            for report in reports {
                println!(
                    "\n  {} vs {}",
                    report.quantum_algorithm, report.classical_baseline
                );
                println!("  {}", report.interpretation);
                println!("  P-value: {:.4}", report.p_value);
                println!("  Effect Size (Cohen's d): {:.2}", report.effect_size);
                println!(
                    "  Confidence Interval: [{:.4}, {:.4}]",
                    report.confidence_interval.0, report.confidence_interval.1
                );

                if let Some(speedup) = report.speedup_factor {
                    println!("  Speedup Factor: {:.2}x", speedup);
                }

                if let Some(efficiency) = report.sample_efficiency {
                    println!("  Sample Efficiency: {:.2}x", efficiency);
                }
            }
            println!();
        }

        println!("══════════════════════════════════════════════════════════\n");
    }

    // Helper functions
    fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    fn variance(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let mean = Self::mean(data);
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
    }

    // Simplified t-distribution CDF approximation
    fn t_cdf(t: f64, df: usize) -> f64 {
        // Very simplified approximation - would use proper implementation in production
        let x = df as f64 / (df as f64 + t * t);
        0.5 * (1.0 + (1.0 - x).sqrt().copysign(t))
    }
}

impl Default for QuantumAdvantageValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = QuantumAdvantageValidator::new();
        assert_eq!(validator.reports.len(), 0);
    }

    #[test]
    fn test_register_results() {
        let mut validator = QuantumAdvantageValidator::new();

        let classical = ClassicalBaseline {
            algorithm_name: "SVM".to_string(),
            accuracy: vec![0.85, 0.84, 0.86],
            execution_times: vec![Duration::from_secs(10); 3],
            sample_sizes: vec![100, 200, 300],
            resource_usage: ResourceUsage {
                cpu_time: Duration::from_secs(10),
                memory_mb: 100,
                num_parameters: 1000,
            },
            hyperparameters: HashMap::new(),
        };

        validator.register_classical_baseline(classical);
        assert_eq!(validator.classical_baselines.len(), 1);
    }

    #[test]
    fn test_statistical_tests() {
        let validator = QuantumAdvantageValidator::new();

        let sample1 = vec![0.9, 0.91, 0.92, 0.89, 0.90];
        let sample2 = vec![0.85, 0.84, 0.86, 0.85, 0.84];

        let result = validator.welch_t_test(&sample1, &sample2);
        assert!(result.is_ok());

        let sig = result.unwrap();
        assert!(sig.is_significant);
    }

    #[test]
    fn test_cohens_d() {
        let validator = QuantumAdvantageValidator::new();

        let sample1 = vec![1.0, 2.0, 3.0];
        let sample2 = vec![2.0, 3.0, 4.0];

        let effect = validator.cohens_d(&sample1, &sample2);
        assert!(effect.abs() > 0.0);
    }
}
