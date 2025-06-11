//! Utility functions and helpers for noise modeling

use std::collections::HashMap;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use crate::DeviceResult;
use super::types::*;
use super::config::*;

/// Utility functions for noise modeling operations
pub struct NoiseModelingUtils;

impl NoiseModelingUtils {
    /// Create feature engineering configuration from calibration data
    pub fn create_feature_engineering(
        calibration_data: &HashMap<String, Array2<f64>>,
        config: &SciRS2NoiseConfig,
    ) -> DeviceResult<FeatureEngineering> {
        let temporal_features = Self::extract_temporal_features(calibration_data)?;
        let spectral_features = Self::extract_spectral_features(calibration_data)?;
        let spatial_features = Self::extract_spatial_features(calibration_data)?;
        let interaction_features = Self::extract_interaction_features(&temporal_features, &spectral_features)?;
        let feature_selection_results = Self::perform_feature_selection(
            calibration_data,
            &temporal_features,
            &spectral_features,
            &spatial_features,
        )?;

        Ok(FeatureEngineering {
            temporal_features,
            spectral_features,
            spatial_features,
            interaction_features,
            feature_selection_results,
        })
    }

    /// Extract temporal feature names
    fn extract_temporal_features(
        calibration_data: &HashMap<String, Array2<f64>>,
    ) -> DeviceResult<Vec<String>> {
        let mut features = Vec::new();
        
        // Basic temporal features
        features.push("mean_drift".to_string());
        features.push("variance_trend".to_string());
        features.push("autocorrelation_lag1".to_string());
        features.push("autocorrelation_lag5".to_string());
        features.push("temporal_stability".to_string());
        features.push("change_point_density".to_string());
        features.push("long_term_memory".to_string());
        
        // Adaptive features based on data characteristics
        if let Some((_, data)) = calibration_data.iter().next() {
            if data.ncols() > 100 {
                features.push("seasonal_component".to_string());
                features.push("trend_component".to_string());
            }
            
            if data.nrows() > 10 {
                features.push("cross_correlation_max".to_string());
                features.push("cross_correlation_lag".to_string());
            }
        }
        
        Ok(features)
    }

    /// Extract spectral feature names
    fn extract_spectral_features(
        calibration_data: &HashMap<String, Array2<f64>>,
    ) -> DeviceResult<Vec<String>> {
        let mut features = Vec::new();
        
        // Basic spectral features
        features.push("dominant_frequency".to_string());
        features.push("spectral_centroid".to_string());
        features.push("spectral_bandwidth".to_string());
        features.push("spectral_flatness".to_string());
        features.push("spectral_rolloff".to_string());
        features.push("noise_color_exponent".to_string());
        features.push("harmonic_ratio".to_string());
        
        // Frequency band power features
        for i in 0..5 {
            features.push(format!("band_power_{}", i));
        }
        
        // Cross-spectral features
        if calibration_data.len() > 1 {
            features.push("coherence_max".to_string());
            features.push("phase_coupling".to_string());
            features.push("cross_spectral_density".to_string());
        }
        
        Ok(features)
    }

    /// Extract spatial feature names
    fn extract_spatial_features(
        calibration_data: &HashMap<String, Array2<f64>>,
    ) -> DeviceResult<Vec<String>> {
        let mut features = Vec::new();
        
        // Basic spatial features
        features.push("spatial_correlation_range".to_string());
        features.push("spatial_variance".to_string());
        features.push("spatial_anisotropy".to_string());
        features.push("nugget_effect".to_string());
        features.push("spatial_clustering".to_string());
        
        // Geometric features
        features.push("distance_to_center".to_string());
        features.push("nearest_neighbor_distance".to_string());
        features.push("local_density".to_string());
        
        // Advanced spatial features
        if let Some((_, data)) = calibration_data.iter().next() {
            if data.nrows() > 5 {
                features.push("spatial_hotspots".to_string());
                features.push("spatial_gradients".to_string());
                features.push("regional_effects".to_string());
            }
        }
        
        Ok(features)
    }

    /// Extract interaction features
    fn extract_interaction_features(
        temporal_features: &[String],
        spectral_features: &[String],
    ) -> DeviceResult<Vec<String>> {
        let mut interactions = Vec::new();
        
        // Temporal-spectral interactions
        interactions.push("temporal_spectral_coupling".to_string());
        interactions.push("time_frequency_localization".to_string());
        interactions.push("spectral_temporal_stability".to_string());
        
        // Cross-domain features
        interactions.push("wavelet_coefficients_variance".to_string());
        interactions.push("short_time_fourier_features".to_string());
        
        // Higher-order interactions
        if temporal_features.len() > 3 && spectral_features.len() > 3 {
            interactions.push("multivariate_coupling".to_string());
            interactions.push("nonlinear_interactions".to_string());
        }
        
        Ok(interactions)
    }

    /// Perform feature selection
    fn perform_feature_selection(
        calibration_data: &HashMap<String, Array2<f64>>,
        temporal_features: &[String],
        spectral_features: &[String],
        spatial_features: &[String],
    ) -> DeviceResult<HashMap<String, f64>> {
        let mut selection_results = HashMap::new();
        
        // Simplified feature selection scoring
        for feature in temporal_features {
            selection_results.insert(feature.clone(), rand::random::<f64>());
        }
        
        for feature in spectral_features {
            selection_results.insert(feature.clone(), rand::random::<f64>());
        }
        
        for feature in spatial_features {
            selection_results.insert(feature.clone(), rand::random::<f64>());
        }
        
        Ok(selection_results)
    }

    /// Create noise prediction model structure
    pub fn create_prediction_model(
        feature_engineering: FeatureEngineering,
        config: &SciRS2NoiseConfig,
    ) -> DeviceResult<NoisePredictionModel> {
        let prediction_horizon = Self::determine_prediction_horizon(config)?;
        let adaptive_components = Self::create_adaptive_components(config)?;
        let change_detection = Self::create_change_detection_config(config)?;
        let online_learning = Self::create_online_learning_config(config)?;
        let performance_monitoring = Self::create_performance_monitoring(config)?;

        Ok(NoisePredictionModel {
            feature_engineering,
            prediction_horizon,
            adaptive_components,
            change_detection,
            online_learning,
            performance_monitoring,
        })
    }

    /// Determine optimal prediction horizon
    fn determine_prediction_horizon(config: &SciRS2NoiseConfig) -> DeviceResult<usize> {
        // Base horizon on sampling frequency and requirements
        let base_horizon = (config.sampling_frequency / 100.0) as usize; // 1% of sampling rate
        Ok(base_horizon.max(10).min(1000)) // Reasonable bounds
    }

    /// Create adaptive components configuration
    fn create_adaptive_components(config: &SciRS2NoiseConfig) -> DeviceResult<AdaptiveComponents> {
        let adaptation_rate = if config.enable_adaptive_modeling { 0.01 } else { 0.0 };
        let forgetting_factor = 0.95;
        let adaptation_triggers = vec![
            "performance_degradation".to_string(),
            "distribution_shift".to_string(),
            "correlation_change".to_string(),
        ];
        let model_update_frequency = 100; // Update every 100 samples

        Ok(AdaptiveComponents {
            adaptation_rate,
            forgetting_factor,
            adaptation_triggers,
            model_update_frequency,
        })
    }

    /// Create change detection configuration
    fn create_change_detection_config(config: &SciRS2NoiseConfig) -> DeviceResult<ChangeDetection> {
        Ok(ChangeDetection {
            detection_method: "cusum".to_string(),
            change_threshold: 3.0, // 3-sigma threshold
            detection_delay: 5,     // Allow 5 samples delay
            false_alarm_rate: 0.01, // 1% false alarm rate
        })
    }

    /// Create online learning configuration
    fn create_online_learning_config(config: &SciRS2NoiseConfig) -> DeviceResult<OnlineLearning> {
        Ok(OnlineLearning {
            learning_rate: 0.001,
            batch_size: 32,
            memory_buffer_size: 1000,
            update_strategy: "sgd".to_string(),
        })
    }

    /// Create performance monitoring configuration
    fn create_performance_monitoring(config: &SciRS2NoiseConfig) -> DeviceResult<PerformanceMonitoring> {
        let monitoring_metrics = vec![
            "rmse".to_string(),
            "mae".to_string(),
            "r2".to_string(),
            "prediction_interval_coverage".to_string(),
        ];
        
        let mut performance_history = HashMap::new();
        for metric in &monitoring_metrics {
            performance_history.insert(metric.clone(), Array1::zeros(100));
        }
        
        let mut degradation_thresholds = HashMap::new();
        degradation_thresholds.insert("rmse".to_string(), 0.2);
        degradation_thresholds.insert("r2".to_string(), 0.8);
        
        let alert_system = AlertSystem {
            alert_levels: vec!["warning".to_string(), "critical".to_string()],
            notification_channels: vec!["email".to_string(), "log".to_string()],
            escalation_rules: HashMap::new(),
        };

        Ok(PerformanceMonitoring {
            monitoring_metrics,
            performance_history,
            degradation_thresholds,
            alert_system,
        })
    }

    /// Compute summary statistics for noise data
    pub fn compute_summary_statistics(
        data: &HashMap<String, Array2<f64>>,
    ) -> DeviceResult<HashMap<String, NoiseStatistics>> {
        let mut statistics = HashMap::new();
        
        for (noise_type, values) in data {
            let stats = Self::compute_single_statistics(values)?;
            statistics.insert(noise_type.clone(), stats);
        }
        
        Ok(statistics)
    }

    /// Compute statistics for a single noise source
    fn compute_single_statistics(data: &Array2<f64>) -> DeviceResult<NoiseStatistics> {
        let flattened: Vec<f64> = data.iter().copied().collect();
        let n = flattened.len() as f64;
        
        // Basic statistics
        let mean = flattened.iter().sum::<f64>() / n;
        let variance = flattened.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        
        // Min/max
        let min_val = flattened.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = flattened.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Percentiles (simplified)
        let mut sorted = flattened.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p25_idx = (0.25 * n) as usize;
        let p50_idx = (0.50 * n) as usize;
        let p75_idx = (0.75 * n) as usize;
        
        let percentile_25 = sorted.get(p25_idx).copied().unwrap_or(mean);
        let median = sorted.get(p50_idx).copied().unwrap_or(mean);
        let percentile_75 = sorted.get(p75_idx).copied().unwrap_or(mean);
        
        Ok(NoiseStatistics {
            mean,
            std_dev,
            variance,
            min_val,
            max_val,
            median,
            percentile_25,
            percentile_75,
            skewness: Self::compute_skewness(&flattened, mean, std_dev),
            kurtosis: Self::compute_kurtosis(&flattened, mean, std_dev),
        })
    }

    /// Compute skewness
    fn compute_skewness(data: &[f64], mean: f64, std_dev: f64) -> f64 {
        if std_dev < 1e-8 {
            return 0.0;
        }
        
        let n = data.len() as f64;
        let sum_cubed = data.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum::<f64>();
        sum_cubed / n
    }

    /// Compute kurtosis
    fn compute_kurtosis(data: &[f64], mean: f64, std_dev: f64) -> f64 {
        if std_dev < 1e-8 {
            return 0.0;
        }
        
        let n = data.len() as f64;
        let sum_fourth = data.iter().map(|x| ((x - mean) / std_dev).powi(4)).sum::<f64>();
        (sum_fourth / n) - 3.0 // Excess kurtosis
    }

    /// Estimate optimal model parameters
    pub fn estimate_model_parameters(
        data: &HashMap<String, Array2<f64>>,
        config: &SciRS2NoiseConfig,
    ) -> DeviceResult<HashMap<String, ModelParameters>> {
        let mut parameters = HashMap::new();
        
        for (noise_type, values) in data {
            let params = Self::estimate_single_model_parameters(values, config)?;
            parameters.insert(noise_type.clone(), params);
        }
        
        Ok(parameters)
    }

    /// Estimate parameters for a single noise source
    fn estimate_single_model_parameters(
        data: &Array2<f64>,
        config: &SciRS2NoiseConfig,
    ) -> DeviceResult<ModelParameters> {
        let num_qubits = data.nrows();
        let num_samples = data.ncols();
        
        // Estimate noise correlation length
        let correlation_length = Self::estimate_correlation_length(data)?;
        
        // Estimate characteristic time scales
        let characteristic_time = Self::estimate_characteristic_time(data, config.sampling_frequency)?;
        
        // Estimate noise amplitude
        let noise_amplitude = Self::estimate_noise_amplitude(data)?;
        
        // Model complexity estimation
        let model_complexity = Self::estimate_model_complexity(num_qubits, num_samples);
        
        Ok(ModelParameters {
            correlation_length,
            characteristic_time,
            noise_amplitude,
            model_complexity,
            num_components: Self::estimate_num_components(data)?,
            regularization_strength: Self::estimate_regularization(num_qubits, num_samples),
        })
    }

    /// Estimate correlation length
    fn estimate_correlation_length(data: &Array2<f64>) -> DeviceResult<f64> {
        // Simple estimation based on autocorrelation decay
        if data.ncols() < 10 {
            return Ok(1.0);
        }
        
        let first_row = data.row(0);
        let autocorr_1 = Self::compute_autocorrelation(&first_row, 1)?;
        let autocorr_5 = Self::compute_autocorrelation(&first_row, 5)?;
        
        // Exponential decay assumption: exp(-lag/tau)
        if autocorr_1 > 1e-8 {
            let tau = -1.0 / autocorr_1.ln();
            Ok(tau.max(0.1).min(100.0))
        } else {
            Ok(1.0)
        }
    }

    /// Compute autocorrelation at a given lag
    fn compute_autocorrelation(data: &ArrayView1<f64>, lag: usize) -> DeviceResult<f64> {
        if data.len() <= lag {
            return Ok(0.0);
        }
        
        let n = data.len() - lag;
        let mean = data.mean().unwrap_or(0.0);
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..n {
            let x_i = data[i] - mean;
            let x_lag = data[i + lag] - mean;
            numerator += x_i * x_lag;
            denominator += x_i * x_i;
        }
        
        if denominator > 1e-8 {
            Ok(numerator / denominator)
        } else {
            Ok(0.0)
        }
    }

    /// Estimate characteristic time scale
    fn estimate_characteristic_time(data: &Array2<f64>, sampling_freq: f64) -> DeviceResult<f64> {
        // Estimate from spectral characteristics
        let dt = 1.0 / sampling_freq;
        let time_scale = dt * (data.ncols() as f64 / 10.0); // Rough estimate
        Ok(time_scale.max(dt).min(1.0)) // Reasonable bounds
    }

    /// Estimate noise amplitude
    fn estimate_noise_amplitude(data: &Array2<f64>) -> DeviceResult<f64> {
        // Use RMS value as amplitude estimate
        let mean_square = data.iter().map(|x| x.powi(2)).sum::<f64>() / (data.len() as f64);
        Ok(mean_square.sqrt())
    }

    /// Estimate model complexity
    fn estimate_model_complexity(num_qubits: usize, num_samples: usize) -> usize {
        // Heuristic for model complexity based on data size
        let base_complexity = (num_qubits as f64).sqrt() as usize;
        let sample_factor = if num_samples > 1000 { 2 } else { 1 };
        (base_complexity * sample_factor).max(1).min(20)
    }

    /// Estimate number of components for decomposition
    fn estimate_num_components(data: &Array2<f64>) -> DeviceResult<usize> {
        // Simple heuristic based on data rank and size
        let min_dim = data.nrows().min(data.ncols());
        let num_components = (min_dim / 3).max(2).min(10);
        Ok(num_components)
    }

    /// Estimate regularization strength
    fn estimate_regularization(num_qubits: usize, num_samples: usize) -> f64 {
        // Regularization strength based on overfitting risk
        let ratio = num_qubits as f64 / num_samples as f64;
        if ratio > 0.5 {
            0.1 // High regularization for small sample/high dimension
        } else if ratio > 0.1 {
            0.01 // Moderate regularization
        } else {
            0.001 // Low regularization for large samples
        }
    }
}

/// Summary statistics for noise data
#[derive(Debug, Clone)]
pub struct NoiseStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub median: f64,
    pub percentile_25: f64,
    pub percentile_75: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Model parameters for noise modeling
#[derive(Debug, Clone)]
pub struct ModelParameters {
    pub correlation_length: f64,
    pub characteristic_time: f64,
    pub noise_amplitude: f64,
    pub model_complexity: usize,
    pub num_components: usize,
    pub regularization_strength: f64,
}
