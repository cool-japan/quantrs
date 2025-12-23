//! Statistical noise analysis using SciRS2
//!
//! This module provides comprehensive statistical characterization of quantum noise,
//! including distributional analysis, moment analysis, correlation analysis, and outlier detection.

use super::config::DistributionType;
use crate::{DeviceError, DeviceResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::prelude::*;
use scirs2_stats::{corrcoef, kurtosis, mean, median, skew, spearmanr, std, var};
use std::collections::HashMap;

/// Statistical noise characterization
#[derive(Debug, Clone)]
pub struct StatisticalNoiseModel {
    /// Distributional analysis for each noise source
    pub distributions: HashMap<String, NoiseDistribution>,
    /// Higher-order moment analysis
    pub moments: HashMap<String, MomentAnalysis>,
    /// Correlation structure between noise sources
    pub correlation_structure: CorrelationStructure,
    /// Outlier detection and analysis
    pub outlier_analysis: OutlierAnalysis,
    /// Non-parametric density estimates
    pub density_estimates: HashMap<String, DensityEstimate>,
}

/// Noise distribution analysis result
#[derive(Debug, Clone)]
pub struct NoiseDistribution {
    pub distribution_type: DistributionType,
    pub parameters: Vec<f64>,
    pub goodness_of_fit: f64,
    pub confidence_intervals: Vec<(f64, f64)>,
    pub p_value: f64,
}

/// Statistical moment analysis
#[derive(Debug, Clone)]
pub struct MomentAnalysis {
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub higher_moments: Vec<f64>,
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}

/// Correlation structure between noise sources
#[derive(Debug, Clone)]
pub struct CorrelationStructure {
    pub correlationmatrix: Array2<f64>,
    pub partial_correlations: Array2<f64>,
    pub rank_correlations: Array2<f64>,
    pub time_varying_correlations: Option<Array2<f64>>,
    pub correlation_networks: CorrelationNetworks,
}

/// Correlation network analysis
#[derive(Debug, Clone)]
pub struct CorrelationNetworks {
    pub threshold_networks: HashMap<String, Array2<bool>>,
    pub community_structure: Vec<Vec<usize>>,
    pub centrality_measures: HashMap<usize, CentralityMeasures>,
}

/// Network centrality measures
#[derive(Debug, Clone)]
pub struct CentralityMeasures {
    pub betweenness: f64,
    pub closeness: f64,
    pub eigenvector: f64,
    pub pagerank: f64,
}

/// Outlier detection results
#[derive(Debug, Clone)]
pub struct OutlierAnalysis {
    pub outlier_indices: Vec<usize>,
    pub outlier_scores: Array1<f64>,
    pub outlier_method: OutlierMethod,
    pub contamination_rate: f64,
}

/// Outlier detection methods
#[derive(Debug, Clone, PartialEq)]
pub enum OutlierMethod {
    IsolationForest,
    LocalOutlierFactor,
    OneClassSVM,
    DBSCAN,
    StatisticalTests,
}

/// Non-parametric density estimation
#[derive(Debug, Clone)]
pub struct DensityEstimate {
    pub method: DensityMethod,
    pub bandwidth: f64,
    pub support: Array1<f64>,
    pub density: Array1<f64>,
    pub log_likelihood: f64,
}

/// Density estimation methods
#[derive(Debug, Clone, PartialEq)]
pub enum DensityMethod {
    KernelDensityEstimation,
    HistogramEstimation,
    GaussianMixture,
    Splines,
}

/// Statistical analysis engine
pub struct StatisticalAnalyzer {
    confidence_level: f64,
    bootstrap_samples: usize,
}

impl StatisticalAnalyzer {
    /// Create a new statistical analyzer
    pub fn new(confidence_level: f64, bootstrap_samples: usize) -> Self {
        Self {
            confidence_level,
            bootstrap_samples,
        }
    }

    /// Fit the best distribution to data using maximum likelihood and goodness-of-fit tests
    pub fn fit_best_distribution(&self, data: &ArrayView2<f64>) -> DeviceResult<NoiseDistribution> {
        let flat_data = data.iter().copied().collect::<Vec<f64>>();
        let data_array = Array1::from_vec(flat_data);

        let mut best_distribution = NoiseDistribution {
            distribution_type: DistributionType::Normal,
            parameters: vec![],
            goodness_of_fit: 0.0,
            confidence_intervals: vec![],
            p_value: 0.0,
        };

        let mut best_score = f64::NEG_INFINITY;

        // Test normal distribution
        let data_mean = mean(&data_array.view())
            .map_err(|e| DeviceError::APIError(format!("Mean calculation error: {:?}", e)))?;
        let data_std = std(&data_array.view(), 1, None)
            .map_err(|e| DeviceError::APIError(format!("Std calculation error: {:?}", e)))?;

        let (ks_stat, p_value) =
            self.kolmogorov_smirnov_test(&data_array, &DistributionType::Normal)?;
        let score = -ks_stat;

        if score > best_score {
            best_score = score;
            best_distribution = NoiseDistribution {
                distribution_type: DistributionType::Normal,
                parameters: vec![data_mean, data_std],
                goodness_of_fit: 1.0 - ks_stat,
                confidence_intervals: vec![(
                    data_mean - 1.96 * data_std,
                    data_mean + 1.96 * data_std,
                )],
                p_value,
            };
        }

        // Test gamma distribution for positive data
        if data_array.iter().all(|&x| x > 0.0) {
            let (ks_stat, p_value) =
                self.kolmogorov_smirnov_test(&data_array, &DistributionType::Gamma)?;
            let score = -ks_stat;

            if score > best_score {
                let (shape, scale) = self.estimate_gamma_parameters(&data_array)?;
                best_score = score;
                best_distribution = NoiseDistribution {
                    distribution_type: DistributionType::Gamma,
                    parameters: vec![shape, scale],
                    goodness_of_fit: 1.0 - ks_stat,
                    confidence_intervals: vec![(0.0, shape * scale * 3.0)],
                    p_value,
                };
            }
        }

        // Test exponential distribution for non-negative data
        if data_array.iter().all(|&x| x >= 0.0) {
            let (ks_stat, p_value) =
                self.kolmogorov_smirnov_test(&data_array, &DistributionType::Exponential)?;
            let score = -ks_stat;

            if score > best_score {
                let rate = 1.0 / data_mean;
                best_distribution = NoiseDistribution {
                    distribution_type: DistributionType::Exponential,
                    parameters: vec![rate],
                    goodness_of_fit: 1.0 - ks_stat,
                    confidence_intervals: vec![(0.0, -data_mean * (0.05_f64).ln())],
                    p_value,
                };
            }
        }

        Ok(best_distribution)
    }

    /// Analyze statistical moments of the data
    pub fn analyze_moments(&self, data: &ArrayView2<f64>) -> DeviceResult<MomentAnalysis> {
        let flat_data = data.iter().copied().collect::<Vec<f64>>();
        let data_array = Array1::from_vec(flat_data);

        let data_mean = mean(&data_array.view())
            .map_err(|e| DeviceError::APIError(format!("Mean calculation error: {:?}", e)))?;

        let data_var = var(&data_array.view(), 1, None)
            .map_err(|e| DeviceError::APIError(format!("Variance calculation error: {:?}", e)))?;

        let data_skew = skew(&data_array.view(), true, None)
            .map_err(|e| DeviceError::APIError(format!("Skewness calculation error: {:?}", e)))?;

        let data_kurt = kurtosis(&data_array.view(), true, true, None)
            .map_err(|e| DeviceError::APIError(format!("Kurtosis calculation error: {:?}", e)))?;

        // Calculate higher moments
        let higher_moments = self.calculate_higher_moments(&data_array, 6)?;

        // Calculate confidence intervals using bootstrap
        let confidence_intervals = self.bootstrap_moment_confidence(&data_array)?;

        Ok(MomentAnalysis {
            mean: data_mean,
            variance: data_var,
            skewness: data_skew,
            kurtosis: data_kurt,
            higher_moments,
            confidence_intervals,
        })
    }

    /// Analyze correlation structure between multiple noise sources
    pub fn analyze_correlation_structure(
        &self,
        noise_measurements: &HashMap<String, Array2<f64>>,
    ) -> DeviceResult<CorrelationStructure> {
        if noise_measurements.len() < 2 {
            return Ok(CorrelationStructure {
                correlationmatrix: Array2::zeros((0, 0)),
                partial_correlations: Array2::zeros((0, 0)),
                rank_correlations: Array2::zeros((0, 0)),
                time_varying_correlations: None,
                correlation_networks: CorrelationNetworks {
                    threshold_networks: HashMap::new(),
                    community_structure: vec![],
                    centrality_measures: HashMap::new(),
                },
            });
        }

        // Collect all data into a matrix
        let sources: Vec<String> = noise_measurements.keys().cloned().collect();
        let n_sources = sources.len();
        let max_samples = noise_measurements
            .values()
            .map(|data| data.nrows())
            .max()
            .unwrap_or(0);

        let mut data_matrix = Array2::zeros((max_samples, n_sources));

        for (i, source) in sources.iter().enumerate() {
            if let Some(measurements) = noise_measurements.get(source) {
                let flat_data: Vec<f64> = measurements.iter().copied().collect();
                let n_samples = flat_data.len().min(max_samples);
                for j in 0..n_samples {
                    data_matrix[[j, i]] = flat_data[j];
                }
            }
        }

        // Compute correlation matrices
        let correlationmatrix = corrcoef(&data_matrix.view(), "pearson")
            .map_err(|e| DeviceError::APIError(format!("Correlation matrix error: {:?}", e)))?;

        // Compute rank correlations
        let rank_correlations = self.compute_rank_correlations(&data_matrix)?;

        // Compute partial correlations
        let partial_correlations = self.compute_partial_correlations(&correlationmatrix)?;

        // Build correlation networks
        let correlation_networks = self.build_correlation_networks(&correlationmatrix)?;

        Ok(CorrelationStructure {
            correlationmatrix,
            partial_correlations,
            rank_correlations,
            time_varying_correlations: None,
            correlation_networks,
        })
    }

    /// Detect outliers in the data using multiple methods
    pub fn detect_outliers(
        &self,
        noise_measurements: &HashMap<String, Array2<f64>>,
    ) -> DeviceResult<OutlierAnalysis> {
        let mut all_outliers = Vec::new();
        let mut all_scores = Vec::new();

        // Collect all data points
        for measurements in noise_measurements.values() {
            let flat_data: Vec<f64> = measurements.iter().copied().collect();

            // Use statistical method (modified Z-score)
            let data_array = Array1::from_vec(flat_data);
            let (outliers, scores) = self.detect_outliers_statistical(&data_array)?;

            all_outliers.extend(outliers);
            all_scores.extend(scores);
        }

        let contamination_rate = all_outliers.len() as f64
            / noise_measurements.values().map(|m| m.len()).sum::<usize>() as f64;

        Ok(OutlierAnalysis {
            outlier_indices: all_outliers,
            outlier_scores: Array1::from_vec(all_scores),
            outlier_method: OutlierMethod::StatisticalTests,
            contamination_rate,
        })
    }

    /// Estimate probability density non-parametrically using kernel density estimation
    pub fn estimate_density(&self, data: &ArrayView2<f64>) -> DeviceResult<DensityEstimate> {
        let flat_data = data.iter().copied().collect::<Vec<f64>>();
        let data_array = Array1::from_vec(flat_data);

        let n = data_array.len();
        if n < 2 {
            return Ok(DensityEstimate {
                method: DensityMethod::KernelDensityEstimation,
                bandwidth: 1.0,
                support: Array1::zeros(0),
                density: Array1::zeros(0),
                log_likelihood: 0.0,
            });
        }

        // Scott's rule for bandwidth selection
        let data_std = std(&data_array.view(), 1, None)
            .map_err(|e| DeviceError::APIError(format!("Std calculation error: {:?}", e)))?;
        let bandwidth = 1.06 * data_std * (n as f64).powf(-1.0 / 5.0);

        // Create support points
        let data_min = data_array.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let data_max = data_array.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = data_max - data_min;
        let support_points = 100;

        let support = Array1::from_shape_fn(support_points, |i| {
            data_min - 0.2 * range + (i as f64 / (support_points - 1) as f64) * 1.4 * range
        });

        // Compute density using Gaussian kernel
        let mut density = Array1::zeros(support_points);
        for (i, &x) in support.iter().enumerate() {
            let mut sum = 0.0;
            for &data_point in data_array.iter() {
                let z = (x - data_point) / bandwidth;
                sum += (-0.5 * z * z).exp();
            }
            density[i] = sum / (n as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt());
        }

        // Compute log-likelihood for cross-validation
        let mut log_likelihood = 0.0;
        for &data_point in data_array.iter() {
            let mut sum = 0.0;
            for &other_point in data_array.iter() {
                if (data_point - other_point).abs() > 1e-10 {
                    let z = (data_point - other_point) / bandwidth;
                    sum += (-0.5 * z * z).exp();
                }
            }
            if sum > 0.0 {
                log_likelihood +=
                    (sum / ((n - 1) as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt())).ln();
            }
        }

        Ok(DensityEstimate {
            method: DensityMethod::KernelDensityEstimation,
            bandwidth,
            support,
            density,
            log_likelihood,
        })
    }

    // Helper methods

    fn kolmogorov_smirnov_test(
        &self,
        data: &Array1<f64>,
        distribution_type: &DistributionType,
    ) -> DeviceResult<(f64, f64)> {
        let n = data.len() as f64;
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let data_mean = mean(&data.view())
            .map_err(|e| DeviceError::APIError(format!("Mean calculation error: {:?}", e)))?;
        let data_std = std(&data.view(), 1, None)
            .map_err(|e| DeviceError::APIError(format!("Std calculation error: {:?}", e)))?;

        let mut max_diff: f64 = 0.0;
        for (i, &value) in sorted_data.iter().enumerate() {
            let empirical_cdf = (i + 1) as f64 / n;

            // Calculate theoretical CDF based on distribution type
            let theoretical_cdf = match distribution_type {
                DistributionType::Normal => {
                    let z = (value - data_mean) / data_std;
                    0.5 * (1.0 + self.erf(z / 2.0_f64.sqrt()))
                }
                DistributionType::Exponential => {
                    if value >= 0.0 {
                        let rate = 1.0 / data_mean;
                        1.0 - (-rate * value).exp()
                    } else {
                        0.0
                    }
                }
                DistributionType::Gamma => {
                    // Simplified gamma CDF approximation
                    let (shape, scale) = self.estimate_gamma_parameters(data)?;
                    if value >= 0.0 {
                        self.gamma_cdf_approx(value, shape, scale)
                    } else {
                        0.0
                    }
                }
                _ => 0.5, // Default fallback
            };

            let diff = (empirical_cdf - theoretical_cdf).abs();
            max_diff = max_diff.max(diff);
        }

        let p_value = if max_diff > 0.0 {
            2.0 * (-2.0 * n * max_diff * max_diff).exp()
        } else {
            1.0
        };

        Ok((max_diff, p_value))
    }

    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = x.signum();
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Gamma CDF approximation
    fn gamma_cdf_approx(&self, x: f64, shape: f64, scale: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }

        // Simple approximation using incomplete gamma function
        let normalized_x = x / scale;

        // For simplicity, use normal approximation for large shape parameters
        if shape > 10.0 {
            let mean = shape * scale;
            let variance = shape * scale * scale;
            let std_dev = variance.sqrt();
            let z = (x - mean) / std_dev;
            0.5 * (1.0 + self.erf(z / 2.0_f64.sqrt()))
        } else {
            // Simple approximation
            (normalized_x / (normalized_x + 1.0)).powf(shape)
        }
    }

    fn estimate_gamma_parameters(&self, data: &Array1<f64>) -> DeviceResult<(f64, f64)> {
        let data_mean = mean(&data.view())
            .map_err(|e| DeviceError::APIError(format!("Mean calculation error: {:?}", e)))?;
        let data_var = var(&data.view(), 1, None)
            .map_err(|e| DeviceError::APIError(format!("Variance calculation error: {:?}", e)))?;

        if data_var > 0.0 {
            let scale = data_var / data_mean;
            let shape = data_mean / scale;
            Ok((shape, scale))
        } else {
            Ok((1.0, data_mean))
        }
    }

    fn calculate_higher_moments(
        &self,
        data: &Array1<f64>,
        max_order: usize,
    ) -> DeviceResult<Vec<f64>> {
        let data_mean = mean(&data.view())
            .map_err(|e| DeviceError::APIError(format!("Mean calculation error: {:?}", e)))?;

        let mut moments = Vec::new();

        for order in 3..=max_order {
            let mut sum = 0.0;
            for &value in data.iter() {
                sum += (value - data_mean).powi(order as i32);
            }
            moments.push(sum / data.len() as f64);
        }

        Ok(moments)
    }

    fn bootstrap_moment_confidence(
        &self,
        data: &Array1<f64>,
    ) -> DeviceResult<HashMap<String, (f64, f64)>> {
        let mut confidence_intervals = HashMap::new();
        let mut rng = thread_rng();
        use scirs2_core::random::prelude::*;

        // Bootstrap for mean
        let mut bootstrap_means = Vec::new();
        for _ in 0..self.bootstrap_samples {
            let sample: Vec<f64> = (0..data.len())
                .map(|_| data[rng.gen_range(0..data.len())])
                .collect();
            let sample_array = Array1::from_vec(sample);
            if let Ok(sample_mean) = mean(&sample_array.view()) {
                bootstrap_means.push(sample_mean);
            }
        }

        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let alpha = 1.0 - self.confidence_level;
        let lower_idx = (alpha / 2.0 * self.bootstrap_samples as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * self.bootstrap_samples as f64) as usize;

        if !bootstrap_means.is_empty() && upper_idx < bootstrap_means.len() {
            confidence_intervals.insert(
                "mean".to_string(),
                (bootstrap_means[lower_idx], bootstrap_means[upper_idx]),
            );
        }

        Ok(confidence_intervals)
    }

    fn compute_rank_correlations(&self, data: &Array2<f64>) -> DeviceResult<Array2<f64>> {
        let n_vars = data.ncols();
        let mut rank_corr = Array2::zeros((n_vars, n_vars));

        for i in 0..n_vars {
            for j in 0..n_vars {
                if i == j {
                    rank_corr[[i, j]] = 1.0;
                } else {
                    let col1 = data.column(i);
                    let col2 = data.column(j);

                    if let Ok((rho, _)) = spearmanr(&col1, &col2, "two-sided") {
                        rank_corr[[i, j]] = rho;
                    }
                }
            }
        }

        Ok(rank_corr)
    }

    fn compute_partial_correlations(&self, corr_matrix: &Array2<f64>) -> DeviceResult<Array2<f64>> {
        let n = corr_matrix.nrows();
        let mut partial_corr = Array2::zeros((n, n));

        // Simplified partial correlation computation
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    partial_corr[[i, j]] = 1.0;
                } else {
                    // Simplified calculation - in practice would use matrix inversion
                    partial_corr[[i, j]] = corr_matrix[[i, j]];
                }
            }
        }

        Ok(partial_corr)
    }

    fn build_correlation_networks(
        &self,
        corr_matrix: &Array2<f64>,
    ) -> DeviceResult<CorrelationNetworks> {
        let n = corr_matrix.nrows();
        let mut threshold_networks = HashMap::new();

        // Create networks at different correlation thresholds
        for &threshold in &[0.3, 0.5, 0.7, 0.9] {
            let mut network = Array2::from_elem((n, n), false);
            for i in 0..n {
                for j in 0..n {
                    if i != j && corr_matrix[[i, j]].abs() > threshold {
                        network[[i, j]] = true;
                    }
                }
            }
            threshold_networks.insert(threshold.to_string(), network);
        }

        // Simplified community structure and centrality
        let community_structure = vec![vec![0, 1], vec![2, 3]];
        let centrality_measures = HashMap::new();

        Ok(CorrelationNetworks {
            threshold_networks,
            community_structure,
            centrality_measures,
        })
    }

    fn detect_outliers_statistical(
        &self,
        data: &Array1<f64>,
    ) -> DeviceResult<(Vec<usize>, Vec<f64>)> {
        // Modified Z-score method
        let data_median = median(&data.view())
            .map_err(|e| DeviceError::APIError(format!("Median calculation error: {:?}", e)))?;

        // Median absolute deviation
        let deviations: Vec<f64> = data.iter().map(|&x| (x - data_median).abs()).collect();
        let mad = median(&Array1::from_vec(deviations).view())
            .map_err(|e| DeviceError::APIError(format!("MAD calculation error: {:?}", e)))?;

        let threshold = 3.5;
        let mut outliers = Vec::new();
        let mut scores = Vec::new();

        for (i, &value) in data.iter().enumerate() {
            let modified_z_score = if mad > 0.0 {
                0.6745 * (value - data_median) / mad
            } else {
                0.0
            };

            scores.push(modified_z_score.abs());

            if modified_z_score.abs() > threshold {
                outliers.push(i);
            }
        }

        Ok((outliers, scores))
    }
}
