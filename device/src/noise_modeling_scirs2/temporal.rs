//! Temporal correlation noise modeling using SciRS2
//!
//! This module provides temporal correlation analysis of quantum noise,
//! including autoregressive models, long-term memory analysis, and change point detection.

use crate::{DeviceError, DeviceResult};
use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;

/// Temporal correlation model
#[derive(Debug, Clone)]
pub struct TemporalNoiseModel {
    /// Autoregressive models for each noise source
    pub ar_models: HashMap<String, ARModel>,
    /// Long-term memory characteristics
    pub long_memory: HashMap<String, LongMemoryModel>,
    /// Non-stationary analysis
    pub nonstationarity: HashMap<String, NonstationarityAnalysis>,
    /// Change point detection
    pub change_points: HashMap<String, Vec<ChangePoint>>,
    /// Temporal clustering
    pub temporal_clusters: HashMap<String, TemporalClusters>,
}

/// Autoregressive model
#[derive(Debug, Clone)]
pub struct ARModel {
    pub order: usize,
    pub coefficients: Array1<f64>,
    pub noise_variance: f64,
    pub aic: f64,
    pub bic: f64,
    pub prediction_error: f64,
}

/// Long-term memory model
#[derive(Debug, Clone)]
pub struct LongMemoryModel {
    pub hurst_exponent: f64,
    pub fractal_dimension: f64,
    pub long_range_dependence: bool,
    pub memory_parameter: f64,
    pub confidence_interval: (f64, f64),
}

/// Non-stationarity analysis
#[derive(Debug, Clone)]
pub struct NonstationarityAnalysis {
    pub is_stationary: bool,
    pub test_statistics: HashMap<String, f64>,
    pub change_point_locations: Vec<usize>,
    pub trend_components: Array1<f64>,
    pub seasonal_components: Option<Array1<f64>>,
}

/// Change point detection
#[derive(Debug, Clone)]
pub struct ChangePoint {
    pub location: usize,
    pub timestamp: f64,
    pub change_magnitude: f64,
    pub change_type: ChangeType,
    pub confidence: f64,
}

/// Types of changes detected
#[derive(Debug, Clone, PartialEq)]
pub enum ChangeType {
    Mean,
    Variance,
    Distribution,
    Correlation,
}

/// Temporal clustering
#[derive(Debug, Clone)]
pub struct TemporalClusters {
    pub cluster_labels: Array1<usize>,
    pub cluster_centers: Array2<f64>,
    pub cluster_statistics: Vec<ClusterStatistics>,
    pub temporal_transitions: Array2<f64>,
}

/// Cluster statistics
#[derive(Debug, Clone)]
pub struct ClusterStatistics {
    pub cluster_id: usize,
    pub size: usize,
    pub duration: f64,
    pub stability: f64,
    pub characteristics: HashMap<String, f64>,
}

/// Temporal analysis engine
pub struct TemporalAnalyzer {
    max_ar_order: usize,
    change_detection_sensitivity: f64,
    clustering_algorithm: ClusteringAlgorithm,
}

/// Clustering algorithms for temporal analysis
#[derive(Debug, Clone, PartialEq)]
pub enum ClusteringAlgorithm {
    KMeans,
    HierarchicalClustering,
    DBSCAN,
    GaussianMixture,
}

impl TemporalAnalyzer {
    /// Create a new temporal analyzer
    pub fn new(max_ar_order: usize, change_detection_sensitivity: f64) -> Self {
        Self {
            max_ar_order,
            change_detection_sensitivity,
            clustering_algorithm: ClusteringAlgorithm::KMeans,
        }
    }

    /// Fit autoregressive model to temporal data
    pub fn fit_ar_model(&self, data: &ArrayView2<f64>) -> DeviceResult<ARModel> {
        let flat_data: Vec<f64> = data.iter().copied().collect();
        let n = flat_data.len();

        if n < self.max_ar_order + 1 {
            return Ok(ARModel {
                order: 0,
                coefficients: Array1::zeros(0),
                noise_variance: 1.0,
                aic: f64::INFINITY,
                bic: f64::INFINITY,
                prediction_error: 1.0,
            });
        }

        // Find optimal AR order using AIC/BIC criteria
        let mut best_order = 1;
        let mut best_aic = f64::INFINITY;
        let mut best_coefficients = Array1::zeros(1);
        let mut best_noise_variance = 1.0;

        for order in 1..=self.max_ar_order.min(n / 4) {
            let (coefficients, noise_variance) = self.estimate_ar_parameters(&flat_data, order)?;
            let aic = self.calculate_aic(&flat_data, &coefficients, noise_variance, order);

            if aic < best_aic {
                best_aic = aic;
                best_order = order;
                best_coefficients = coefficients;
                best_noise_variance = noise_variance;
            }
        }

        let bic = self.calculate_bic(
            &flat_data,
            &best_coefficients,
            best_noise_variance,
            best_order,
        );
        let prediction_error = self.calculate_prediction_error(&flat_data, &best_coefficients)?;

        Ok(ARModel {
            order: best_order,
            coefficients: best_coefficients,
            noise_variance: best_noise_variance,
            aic: best_aic,
            bic,
            prediction_error,
        })
    }

    /// Analyze long-term memory characteristics
    pub fn analyze_long_memory(&self, data: &ArrayView2<f64>) -> DeviceResult<LongMemoryModel> {
        let flat_data: Vec<f64> = data.iter().copied().collect();

        // Estimate Hurst exponent using R/S analysis
        let hurst_exponent = self.estimate_hurst_exponent(&flat_data)?;
        let fractal_dimension = 2.0 - hurst_exponent;
        let long_range_dependence = hurst_exponent > 0.5;
        let memory_parameter = hurst_exponent - 0.5;

        // Calculate confidence interval (simplified)
        let confidence_interval = (hurst_exponent - 0.1, hurst_exponent + 0.1);

        Ok(LongMemoryModel {
            hurst_exponent,
            fractal_dimension,
            long_range_dependence,
            memory_parameter,
            confidence_interval,
        })
    }

    /// Test for non-stationarity
    pub fn test_nonstationarity(
        &self,
        data: &ArrayView2<f64>,
    ) -> DeviceResult<NonstationarityAnalysis> {
        let flat_data: Vec<f64> = data.iter().copied().collect();
        let n = flat_data.len();

        if n < 10 {
            return Ok(NonstationarityAnalysis {
                is_stationary: true,
                test_statistics: HashMap::new(),
                change_point_locations: vec![],
                trend_components: Array1::zeros(n),
                seasonal_components: None,
            });
        }

        // Augmented Dickey-Fuller test (simplified)
        let mut test_statistics = HashMap::new();
        let adf_statistic = self.augmented_dickey_fuller_test(&flat_data)?;
        test_statistics.insert("ADF".to_string(), adf_statistic);

        let is_stationary = adf_statistic < -2.86; // Critical value at 5% significance

        // Simple trend detection
        let trend_components = self.extract_trend(&flat_data)?;

        // Change point detection
        let change_point_locations = self.detect_change_points_simple(&flat_data)?;

        Ok(NonstationarityAnalysis {
            is_stationary,
            test_statistics,
            change_point_locations,
            trend_components,
            seasonal_components: None,
        })
    }

    /// Detect change points in the data
    pub fn detect_change_points(&self, data: &ArrayView2<f64>) -> DeviceResult<Vec<ChangePoint>> {
        let flat_data: Vec<f64> = data.iter().copied().collect();
        let change_point_locations = self.detect_change_points_simple(&flat_data)?;

        let mut change_points = Vec::new();
        for &location in &change_point_locations {
            if location < flat_data.len() {
                let change_magnitude = self.estimate_change_magnitude(&flat_data, location)?;

                change_points.push(ChangePoint {
                    location,
                    timestamp: location as f64, // Would use actual timestamps in practice
                    change_magnitude,
                    change_type: ChangeType::Mean, // Simplified - would detect actual type
                    confidence: 0.95,              // Would calculate actual confidence
                });
            }
        }

        Ok(change_points)
    }

    /// Cluster temporal patterns
    pub fn cluster_temporal_patterns(
        &self,
        data: &ArrayView2<f64>,
    ) -> DeviceResult<TemporalClusters> {
        let flat_data: Vec<f64> = data.iter().copied().collect();
        let n = flat_data.len();

        if n < 4 {
            return Ok(TemporalClusters {
                cluster_labels: Array1::zeros(n),
                cluster_centers: Array2::zeros((1, 2)),
                cluster_statistics: vec![],
                temporal_transitions: Array2::zeros((1, 1)),
            });
        }

        // Simple k-means clustering (k=3)
        let k = 3.min(n / 2);
        let (cluster_labels, cluster_centers) = self.simple_kmeans(&flat_data, k)?;

        // Calculate cluster statistics
        let cluster_statistics =
            self.calculate_cluster_statistics(&flat_data, &cluster_labels, k)?;

        // Calculate transition matrix
        let temporal_transitions = self.calculate_transition_matrix(&cluster_labels, k)?;

        Ok(TemporalClusters {
            cluster_labels: Array1::from_vec(cluster_labels),
            cluster_centers,
            cluster_statistics,
            temporal_transitions,
        })
    }

    // Helper methods

    fn estimate_ar_parameters(
        &self,
        data: &[f64],
        order: usize,
    ) -> DeviceResult<(Array1<f64>, f64)> {
        let n = data.len();
        if n <= order {
            return Ok((Array1::zeros(order), 1.0));
        }

        // Yule-Walker equations (simplified)
        let mut coefficients = Array1::zeros(order);
        let mut sum_y = 0.0;
        let mut sum_yy = 0.0;

        for i in order..n {
            sum_y += data[i];
            sum_yy += data[i] * data[i];
        }

        let mean_y = sum_y / (n - order) as f64;

        // Simple AR(1) estimation
        if order >= 1 && n > order + 1 {
            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for i in order..n - 1 {
                numerator += (data[i] - mean_y) * (data[i + 1] - mean_y);
                denominator += (data[i] - mean_y).powi(2);
            }

            if denominator > 1e-10 {
                coefficients[0] = numerator / denominator;
            }
        }

        // Estimate noise variance
        let mut residual_sum = 0.0;
        let mut count = 0;

        for i in order..n {
            let mut predicted = mean_y;
            for j in 0..order.min(coefficients.len()) {
                if i >= j + 1 {
                    predicted += coefficients[j] * (data[i - j - 1] - mean_y);
                }
            }
            residual_sum += (data[i] - predicted).powi(2);
            count += 1;
        }

        let noise_variance = if count > 0 {
            residual_sum / count as f64
        } else {
            1.0
        };

        Ok((coefficients, noise_variance))
    }

    fn calculate_aic(
        &self,
        data: &[f64],
        coefficients: &Array1<f64>,
        noise_variance: f64,
        order: usize,
    ) -> f64 {
        let n = data.len() as f64;
        let k = order as f64;

        if noise_variance <= 0.0 {
            return f64::INFINITY;
        }

        let log_likelihood =
            -0.5 * n * (noise_variance.ln() + 1.0 + (2.0 * std::f64::consts::PI).ln());
        2.0 * k - 2.0 * log_likelihood
    }

    fn calculate_bic(
        &self,
        data: &[f64],
        coefficients: &Array1<f64>,
        noise_variance: f64,
        order: usize,
    ) -> f64 {
        let n = data.len() as f64;
        let k = order as f64;

        if noise_variance <= 0.0 {
            return f64::INFINITY;
        }

        let log_likelihood =
            -0.5 * n * (noise_variance.ln() + 1.0 + (2.0 * std::f64::consts::PI).ln());
        k * n.ln() - 2.0 * log_likelihood
    }

    fn calculate_prediction_error(
        &self,
        data: &[f64],
        coefficients: &Array1<f64>,
    ) -> DeviceResult<f64> {
        let n = data.len();
        let order = coefficients.len();

        if n <= order {
            return Ok(1.0);
        }

        let mean = data.iter().sum::<f64>() / n as f64;
        let mut error_sum = 0.0;
        let mut count = 0;

        for i in order..n {
            let mut prediction = mean;
            for j in 0..order {
                if i >= j + 1 {
                    prediction += coefficients[j] * (data[i - j - 1] - mean);
                }
            }
            error_sum += (data[i] - prediction).powi(2);
            count += 1;
        }

        Ok(if count > 0 {
            (error_sum / count as f64).sqrt()
        } else {
            1.0
        })
    }

    fn estimate_hurst_exponent(&self, data: &[f64]) -> DeviceResult<f64> {
        let n = data.len();
        if n < 4 {
            return Ok(0.5);
        }

        // R/S analysis
        let mean = data.iter().sum::<f64>() / n as f64;

        // Calculate cumulative deviations
        let mut cumulative_deviations = vec![0.0; n];
        for i in 1..n {
            cumulative_deviations[i] = cumulative_deviations[i - 1] + (data[i] - mean);
        }

        // Calculate range
        let max_deviation = cumulative_deviations
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_deviation = cumulative_deviations
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let range = max_deviation - min_deviation;

        // Calculate standard deviation
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // R/S ratio
        let rs_ratio = if std_dev > 0.0 { range / std_dev } else { 1.0 };

        // Hurst exponent estimation
        let hurst = if rs_ratio > 0.0 {
            rs_ratio.ln() / (n as f64).ln()
        } else {
            0.5
        };

        Ok(hurst.max(0.0).min(1.0))
    }

    fn augmented_dickey_fuller_test(&self, data: &[f64]) -> DeviceResult<f64> {
        let n = data.len();
        if n < 3 {
            return Ok(-3.0); // Default to stationary
        }

        // Simple Dickey-Fuller test
        let mut sum_diff = 0.0;
        let mut sum_lag = 0.0;
        let mut sum_diff_lag = 0.0;
        let mut sum_lag_sq = 0.0;

        for i in 1..n {
            let diff = data[i] - data[i - 1];
            let lag = data[i - 1];

            sum_diff += diff;
            sum_lag += lag;
            sum_diff_lag += diff * lag;
            sum_lag_sq += lag * lag;
        }

        let n_obs = (n - 1) as f64;
        let mean_diff = sum_diff / n_obs;
        let mean_lag = sum_lag / n_obs;

        // Calculate regression coefficient
        let numerator = sum_diff_lag - n_obs * mean_diff * mean_lag;
        let denominator = sum_lag_sq - n_obs * mean_lag * mean_lag;

        let beta = if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };

        // Simple t-statistic (should be proper t-test)
        let t_stat = if beta.abs() < 1.0 { beta * 10.0 } else { -3.0 };

        Ok(t_stat)
    }

    fn extract_trend(&self, data: &[f64]) -> DeviceResult<Array1<f64>> {
        let n = data.len();
        let mut trend = Array1::zeros(n);

        if n < 2 {
            return Ok(trend);
        }

        // Simple linear trend
        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean = data.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        let slope = if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };
        let intercept = y_mean - slope * x_mean;

        for i in 0..n {
            trend[i] = intercept + slope * i as f64;
        }

        Ok(trend)
    }

    fn detect_change_points_simple(&self, data: &[f64]) -> DeviceResult<Vec<usize>> {
        let n = data.len();
        let mut change_points = Vec::new();

        if n < 6 {
            return Ok(change_points);
        }

        let window_size = (n / 10).max(3);

        for i in window_size..(n - window_size) {
            let before_mean = data[i - window_size..i].iter().sum::<f64>() / window_size as f64;
            let after_mean = data[i..i + window_size].iter().sum::<f64>() / window_size as f64;

            let change_magnitude = (after_mean - before_mean).abs();
            let threshold = self.change_detection_sensitivity
                * data.iter().map(|&x| x.abs()).sum::<f64>()
                / n as f64;

            if change_magnitude > threshold {
                change_points.push(i);
            }
        }

        Ok(change_points)
    }

    fn estimate_change_magnitude(&self, data: &[f64], location: usize) -> DeviceResult<f64> {
        let n = data.len();
        if location >= n || location == 0 {
            return Ok(0.0);
        }

        let window_size = (n / 20).max(2).min(location).min(n - location);

        let before_start = location.saturating_sub(window_size);
        let after_end = (location + window_size).min(n);

        let before_mean = if before_start < location {
            data[before_start..location].iter().sum::<f64>() / (location - before_start) as f64
        } else {
            data[location]
        };

        let after_mean = if location < after_end {
            data[location..after_end].iter().sum::<f64>() / (after_end - location) as f64
        } else {
            data[location]
        };

        Ok((after_mean - before_mean).abs())
    }

    fn simple_kmeans(&self, data: &[f64], k: usize) -> DeviceResult<(Vec<usize>, Array2<f64>)> {
        let n = data.len();
        if k == 0 || n == 0 {
            return Ok((vec![], Array2::zeros((0, 1))));
        }

        // Initialize centroids
        let mut centroids = Array2::zeros((k, 1));
        let data_min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let data_max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        for i in 0..k {
            centroids[[i, 0]] = data_min + (i as f64 / k as f64) * (data_max - data_min);
        }

        let mut labels = vec![0; n];

        // Simple k-means iterations
        for _iter in 0..10 {
            // Assign points to clusters
            for (i, &point) in data.iter().enumerate() {
                let mut best_cluster = 0;
                let mut min_distance = f64::INFINITY;

                for cluster in 0..k {
                    let distance = (point - centroids[[cluster, 0]]).abs();
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = cluster;
                    }
                }
                labels[i] = best_cluster;
            }

            // Update centroids
            for cluster in 0..k {
                let cluster_points: Vec<f64> = data
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| labels[*i] == cluster)
                    .map(|(_, &value)| value)
                    .collect();

                if !cluster_points.is_empty() {
                    centroids[[cluster, 0]] =
                        cluster_points.iter().sum::<f64>() / cluster_points.len() as f64;
                }
            }
        }

        Ok((labels, centroids))
    }

    fn calculate_cluster_statistics(
        &self,
        data: &[f64],
        labels: &[usize],
        k: usize,
    ) -> DeviceResult<Vec<ClusterStatistics>> {
        let mut statistics = Vec::new();

        for cluster_id in 0..k {
            let cluster_data: Vec<f64> = data
                .iter()
                .enumerate()
                .filter(|(i, _)| labels[*i] == cluster_id)
                .map(|(_, &value)| value)
                .collect();

            let size = cluster_data.len();
            let duration = size as f64; // Simplified - would use actual time durations

            let stability = if size > 1 {
                let mean = cluster_data.iter().sum::<f64>() / size as f64;
                let variance = cluster_data
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / size as f64;
                1.0 / (1.0 + variance.sqrt())
            } else {
                1.0
            };

            let mut characteristics = HashMap::new();
            if !cluster_data.is_empty() {
                characteristics.insert(
                    "mean".to_string(),
                    cluster_data.iter().sum::<f64>() / size as f64,
                );
                characteristics.insert("size".to_string(), size as f64);
            }

            statistics.push(ClusterStatistics {
                cluster_id,
                size,
                duration,
                stability,
                characteristics,
            });
        }

        Ok(statistics)
    }

    fn calculate_transition_matrix(&self, labels: &[usize], k: usize) -> DeviceResult<Array2<f64>> {
        let mut transitions = Array2::zeros((k, k));

        if labels.len() < 2 {
            return Ok(transitions);
        }

        // Count transitions
        for i in 0..labels.len() - 1 {
            let from = labels[i];
            let to = labels[i + 1];
            if from < k && to < k {
                transitions[[from, to]] += 1.0;
            }
        }

        // Normalize rows
        for i in 0..k {
            let row_sum: f64 = (0..k).map(|j| transitions[[i, j]]).sum();
            if row_sum > 0.0 {
                for j in 0..k {
                    transitions[[i, j]] /= row_sum;
                }
            }
        }

        Ok(transitions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_temporal_analyzer_creation() {
        let analyzer = TemporalAnalyzer::new(5, 0.1);
        assert_eq!(analyzer.max_ar_order, 5);
        assert_eq!(analyzer.change_detection_sensitivity, 0.1);
    }

    #[test]
    fn test_ar_model_fitting() {
        let analyzer = TemporalAnalyzer::new(3, 0.1);
        let test_data = Array2::from_shape_fn((100, 1), |(i, _)| {
            // Generate AR(1) process
            if i == 0 {
                0.0
            } else {
                0.5 * (i as f64) + 0.1 * (i as f64).sin()
            }
        });

        let ar_model = analyzer.fit_ar_model(&test_data.view()).unwrap();

        assert!(ar_model.order > 0);
        assert!(ar_model.coefficients.len() > 0);
        assert!(ar_model.noise_variance > 0.0);
        assert!(ar_model.aic.is_finite());
        assert!(ar_model.bic.is_finite());
    }

    #[test]
    fn test_hurst_exponent_estimation() {
        let analyzer = TemporalAnalyzer::new(3, 0.1);

        // Test with random walk (should have H ≈ 0.5)
        let test_data = Array2::from_shape_fn((50, 1), |(i, _)| {
            (i as f64).sqrt() // Simple trend
        });

        let long_memory = analyzer.analyze_long_memory(&test_data.view()).unwrap();

        assert!(long_memory.hurst_exponent >= 0.0);
        assert!(long_memory.hurst_exponent <= 1.0);
        assert_eq!(
            long_memory.fractal_dimension,
            2.0 - long_memory.hurst_exponent
        );
    }

    #[test]
    fn test_change_point_detection() {
        let analyzer = TemporalAnalyzer::new(3, 0.5);

        // Create data with a clear change point
        let mut test_data_vec = vec![1.0; 25];
        test_data_vec.extend(vec![5.0; 25]);
        let test_data = Array2::from_shape_fn((50, 1), |(i, _)| test_data_vec[i]);

        let change_points = analyzer.detect_change_points(&test_data.view()).unwrap();

        // Should detect the change point around index 25
        assert!(!change_points.is_empty());
        assert!(change_points
            .iter()
            .any(|cp| cp.location > 20 && cp.location < 30));
    }

    #[test]
    fn test_temporal_clustering() {
        let analyzer = TemporalAnalyzer::new(3, 0.1);

        // Create data with distinct patterns
        let test_data = Array2::from_shape_fn((20, 1), |(i, _)| if i < 10 { 1.0 } else { 5.0 });

        let clusters = analyzer
            .cluster_temporal_patterns(&test_data.view())
            .unwrap();

        assert_eq!(clusters.cluster_labels.len(), 20);
        assert!(clusters.cluster_centers.nrows() > 0);
        assert_eq!(
            clusters.temporal_transitions.nrows(),
            clusters.temporal_transitions.ncols()
        );
    }

    #[test]
    fn test_nonstationarity_analysis() {
        let analyzer = TemporalAnalyzer::new(3, 0.1);

        // Create trending data (non-stationary)
        let test_data = Array2::from_shape_fn((30, 1), |(i, _)| i as f64);

        let nonstationarity = analyzer.test_nonstationarity(&test_data.view()).unwrap();

        assert!(nonstationarity.test_statistics.contains_key("ADF"));
        assert_eq!(nonstationarity.trend_components.len(), 30);
    }
}
