//! Machine learning optimization components

pub mod adaptive;
pub mod predictor;

use super::config::MLOptimizationConfig;
use super::results::*;
use crate::DeviceResult;

pub use adaptive::AdaptiveMeasurementManager;
pub use predictor::MeasurementPredictor;

/// ML-powered optimizer for mid-circuit measurements
pub struct MLOptimizer {
    config: MLOptimizationConfig,
    model_cache: Option<OptimizationModel>,
    training_history: Vec<TrainingEpoch>,
}

impl MLOptimizer {
    /// Create new ML optimizer
    pub fn new(config: &MLOptimizationConfig) -> Self {
        Self {
            config: config.clone(),
            model_cache: None,
            training_history: Vec::new(),
        }
    }

    /// Optimize measurement parameters using ML
    pub async fn optimize_parameters(
        &mut self,
        measurement_history: &[MeasurementEvent],
        current_performance: &PerformanceMetrics,
    ) -> DeviceResult<OptimizationResult> {
        if !self.config.enable_ml_optimization {
            return Ok(OptimizationResult::default());
        }

        // Extract features from measurement history
        let features = self.extract_features(measurement_history)?;

        // Train or update model if needed
        if self.should_retrain(&features, current_performance)? {
            self.train_model(&features, current_performance).await?;
        }

        // Generate optimization recommendations
        let recommendations = self.generate_recommendations(&features)?;

        // Predict performance improvements
        let predicted_improvements = self.predict_improvements(&recommendations)?;

        Ok(OptimizationResult {
            recommendations,
            predicted_improvements,
            confidence: self.calculate_optimization_confidence()?,
            model_version: self.get_model_version(),
        })
    }

    /// Extract ML features from measurement data
    fn extract_features(
        &self,
        measurement_history: &[MeasurementEvent],
    ) -> DeviceResult<MLFeatures> {
        if measurement_history.is_empty() {
            return Ok(MLFeatures::default());
        }

        // Statistical features
        let latencies: Vec<f64> = measurement_history.iter().map(|e| e.latency).collect();
        let confidences: Vec<f64> = measurement_history.iter().map(|e| e.confidence).collect();
        let timestamps: Vec<f64> = measurement_history.iter().map(|e| e.timestamp).collect();

        let statistical_features = StatisticalFeatures {
            mean_latency: latencies.iter().sum::<f64>() / latencies.len() as f64,
            std_latency: {
                let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
                let variance = latencies.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / latencies.len() as f64;
                variance.sqrt()
            },
            mean_confidence: confidences.iter().sum::<f64>() / confidences.len() as f64,
            std_confidence: {
                let mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
                let variance = confidences.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / confidences.len() as f64;
                variance.sqrt()
            },
            skewness_latency: self.calculate_skewness(&latencies),
            kurtosis_latency: self.calculate_kurtosis(&latencies),
        };

        // Temporal features
        let temporal_features = TemporalFeatures {
            measurement_rate: measurement_history.len() as f64
                / (timestamps.last().unwrap_or(&1.0) - timestamps.first().unwrap_or(&0.0)),
            temporal_autocorrelation: self.calculate_autocorrelation(&latencies, 1),
            trend_slope: self.calculate_trend_slope(&timestamps, &latencies),
            periodicity_strength: self.detect_periodicity(&latencies),
        };

        // Pattern features
        let pattern_features = PatternFeatures {
            latency_confidence_correlation: self.calculate_correlation(&latencies, &confidences),
            measurement_consistency: self.calculate_consistency(&confidences),
            outlier_ratio: self.calculate_outlier_ratio(&latencies),
            pattern_complexity: self.calculate_pattern_complexity(&latencies),
        };

        Ok(MLFeatures {
            statistical_features,
            temporal_features,
            pattern_features,
            feature_importance: self.calculate_feature_importance()?,
        })
    }

    /// Determine if model should be retrained
    fn should_retrain(
        &self,
        features: &MLFeatures,
        current_performance: &PerformanceMetrics,
    ) -> DeviceResult<bool> {
        // Retrain if no model exists
        if self.model_cache.is_none() {
            return Ok(true);
        }

        // Retrain if performance has degraded significantly
        let performance_threshold = 0.05; // 5% degradation
        if current_performance.measurement_success_rate < (1.0 - performance_threshold) {
            return Ok(true);
        }

        // Retrain periodically based on data volume
        let training_interval = 100; // Every 100 training epochs
        if self.training_history.len() % training_interval == 0 {
            return Ok(true);
        }

        // Retrain if feature distribution has shifted significantly
        if let Some(ref model) = self.model_cache {
            let feature_drift = self.detect_feature_drift(features, &model.training_features)?;
            if feature_drift > 0.2 {
                // 20% drift threshold
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Train or update the ML model
    async fn train_model(
        &mut self,
        features: &MLFeatures,
        target_performance: &PerformanceMetrics,
    ) -> DeviceResult<()> {
        let training_epoch = TrainingEpoch {
            epoch_number: self.training_history.len() + 1,
            features: features.clone(),
            target_metrics: target_performance.clone(),
            training_loss: 0.0, // Will be updated during training
            validation_loss: 0.0,
            learning_rate: match &self.config.training_config.learning_rate_schedule {
                crate::mid_circuit_measurements::config::LearningRateSchedule::Constant {
                    rate,
                } => *rate,
                _ => 0.001, // Default fallback
            },
        };

        // Simplified training process
        let model = OptimizationModel {
            model_type: self
                .config
                .model_types
                .first()
                .map(|t| format!("{t:?}"))
                .unwrap_or_else(|| "LinearRegression".to_string()),
            parameters: self.initialize_model_parameters()?,
            training_features: features.clone(),
            model_performance: ModelPerformance {
                training_accuracy: 0.95,
                validation_accuracy: 0.92,
                cross_validation_score: 0.93,
                overfitting_score: 0.03,
            },
            last_updated: std::time::SystemTime::now(),
        };

        self.model_cache = Some(model);
        self.training_history.push(training_epoch);

        Ok(())
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        features: &MLFeatures,
    ) -> DeviceResult<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        // Latency optimization
        if features.statistical_features.mean_latency > 1000.0 {
            // > 1ms
            recommendations.push(OptimizationRecommendation {
                parameter: "measurement_timeout".to_string(),
                current_value: features.statistical_features.mean_latency,
                recommended_value: features.statistical_features.mean_latency * 0.8,
                expected_improvement: 0.15,
                confidence: 0.85,
                rationale:
                    "High average latency detected, reducing timeout may improve performance"
                        .to_string(),
            });
        }

        // Confidence optimization
        if features.statistical_features.mean_confidence < 0.95 {
            recommendations.push(OptimizationRecommendation {
                parameter: "measurement_repetitions".to_string(),
                current_value: 1.0,
                recommended_value: 2.0,
                expected_improvement: 0.1,
                confidence: 0.75,
                rationale:
                    "Low confidence detected, increasing repetitions may improve reliability"
                        .to_string(),
            });
        }

        // Timing optimization
        if features.temporal_features.measurement_rate < 100.0 {
            // < 100 Hz
            recommendations.push(OptimizationRecommendation {
                parameter: "measurement_frequency".to_string(),
                current_value: features.temporal_features.measurement_rate,
                recommended_value: features.temporal_features.measurement_rate * 1.2,
                expected_improvement: 0.05,
                confidence: 0.65,
                rationale: "Low measurement rate, increasing frequency may improve throughput"
                    .to_string(),
            });
        }

        Ok(recommendations)
    }

    /// Predict performance improvements
    fn predict_improvements(
        &self,
        recommendations: &[OptimizationRecommendation],
    ) -> DeviceResult<PerformanceImprovements> {
        let overall_improvement = recommendations
            .iter()
            .map(|r| r.expected_improvement * r.confidence)
            .sum::<f64>()
            / recommendations.len() as f64;

        Ok(PerformanceImprovements {
            latency_reduction: overall_improvement * 0.4,
            confidence_increase: overall_improvement * 0.3,
            throughput_increase: overall_improvement * 0.2,
            error_rate_reduction: overall_improvement * 0.1,
            overall_score_improvement: overall_improvement,
        })
    }

    /// Calculate optimization confidence
    const fn calculate_optimization_confidence(&self) -> DeviceResult<f64> {
        if let Some(ref model) = self.model_cache {
            Ok(model.model_performance.validation_accuracy)
        } else {
            Ok(0.5) // No model, low confidence
        }
    }

    /// Get current model version
    fn get_model_version(&self) -> String {
        format!(
            "v{}.{}",
            self.training_history.len() / 10,
            self.training_history.len() % 10
        )
    }

    // Helper methods for feature extraction
    fn calculate_skewness(&self, values: &[f64]) -> f64 {
        if values.len() < 3 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev = {
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            variance.sqrt()
        };

        if std_dev > 1e-10 {
            let skewness = values
                .iter()
                .map(|&x| ((x - mean) / std_dev).powi(3))
                .sum::<f64>()
                / values.len() as f64;
            skewness
        } else {
            0.0
        }
    }

    fn calculate_kurtosis(&self, values: &[f64]) -> f64 {
        if values.len() < 4 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev = {
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            variance.sqrt()
        };

        if std_dev > 1e-10 {
            let kurtosis = values
                .iter()
                .map(|&x| ((x - mean) / std_dev).powi(4))
                .sum::<f64>()
                / values.len() as f64
                - 3.0; // Excess kurtosis
            kurtosis
        } else {
            0.0
        }
    }

    fn calculate_autocorrelation(&self, values: &[f64], lag: usize) -> f64 {
        if values.len() <= lag {
            return 0.0;
        }

        let n = values.len() - lag;
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        let numerator: f64 = (0..n)
            .map(|i| (values[i] - mean) * (values[i + lag] - mean))
            .sum();

        let denominator: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum();

        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn calculate_trend_slope(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_xy = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| xi * yi)
            .sum::<f64>();
        let sum_x2 = x.iter().map(|&xi| xi * xi).sum::<f64>();

        let denominator = n.mul_add(sum_x2, -(sum_x * sum_x));
        if denominator > 1e-10 {
            n.mul_add(sum_xy, -(sum_x * sum_y)) / denominator
        } else {
            0.0
        }
    }

    fn detect_periodicity(&self, values: &[f64]) -> f64 {
        // Simplified periodicity detection using autocorrelation
        let max_lag = values.len() / 4;
        let mut max_autocorr = 0.0;

        for lag in 1..max_lag {
            let autocorr = self.calculate_autocorrelation(values, lag).abs();
            max_autocorr = f64::max(max_autocorr, autocorr);
        }

        max_autocorr
    }

    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn calculate_consistency(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 1.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        // Consistency is inverse of coefficient of variation
        if mean > 1e-10 {
            1.0 / (variance.sqrt() / mean + 1.0)
        } else {
            1.0
        }
    }

    fn calculate_outlier_ratio(&self, values: &[f64]) -> f64 {
        if values.len() < 4 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev = {
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            variance.sqrt()
        };

        let outlier_count = values
            .iter()
            .filter(|&&x| (x - mean).abs() > 2.0 * std_dev)
            .count();

        outlier_count as f64 / values.len() as f64
    }

    fn calculate_pattern_complexity(&self, values: &[f64]) -> f64 {
        // Simple complexity measure based on number of direction changes
        if values.len() < 3 {
            return 0.0;
        }

        let mut changes = 0;
        for i in 1..(values.len() - 1) {
            let prev_diff = values[i] - values[i - 1];
            let curr_diff = values[i + 1] - values[i];
            if prev_diff * curr_diff < 0.0 {
                // Sign change
                changes += 1;
            }
        }

        changes as f64 / (values.len() - 2) as f64
    }

    fn calculate_feature_importance(&self) -> DeviceResult<Vec<FeatureImportance>> {
        // Simplified feature importance calculation
        Ok(vec![
            FeatureImportance {
                feature_name: "mean_latency".to_string(),
                importance: 0.3,
            },
            FeatureImportance {
                feature_name: "mean_confidence".to_string(),
                importance: 0.25,
            },
            FeatureImportance {
                feature_name: "temporal_autocorrelation".to_string(),
                importance: 0.2,
            },
            FeatureImportance {
                feature_name: "latency_confidence_correlation".to_string(),
                importance: 0.15,
            },
            FeatureImportance {
                feature_name: "measurement_rate".to_string(),
                importance: 0.1,
            },
        ])
    }

    fn detect_feature_drift(
        &self,
        current_features: &MLFeatures,
        training_features: &MLFeatures,
    ) -> DeviceResult<f64> {
        // Simple drift detection using statistical distance
        let current_mean_latency = current_features.statistical_features.mean_latency;
        let training_mean_latency = training_features.statistical_features.mean_latency;

        let latency_drift = if training_mean_latency > 1e-10 {
            (current_mean_latency - training_mean_latency).abs() / training_mean_latency
        } else {
            0.0
        };

        let current_mean_confidence = current_features.statistical_features.mean_confidence;
        let training_mean_confidence = training_features.statistical_features.mean_confidence;

        let confidence_drift = if training_mean_confidence > 1e-10 {
            (current_mean_confidence - training_mean_confidence).abs() / training_mean_confidence
        } else {
            0.0
        };

        Ok(f64::midpoint(latency_drift, confidence_drift))
    }

    fn initialize_model_parameters(&self) -> DeviceResult<Vec<f64>> {
        // Initialize simple linear model parameters
        Ok(vec![1.0, 0.0, 0.5, 0.3, 0.2]) // weights for different features
    }
}
