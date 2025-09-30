//! Anomaly detection components

use super::super::results::*;
use crate::DeviceResult;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

/// Anomaly detector for measurement data
pub struct AnomalyDetector {
    threshold: f64,
    window_size: usize,
}

impl AnomalyDetector {
    /// Create new anomaly detector
    pub fn new() -> Self {
        Self {
            threshold: 2.0, // Standard deviations
            window_size: 50,
        }
    }

    /// Create anomaly detector with custom parameters
    pub fn with_parameters(threshold: f64, window_size: usize) -> Self {
        Self {
            threshold,
            window_size,
        }
    }

    /// Detect anomalies in measurement data
    pub fn detect(
        &self,
        latencies: &[f64],
        confidences: &[f64],
    ) -> DeviceResult<AnomalyDetectionResults> {
        if latencies.is_empty() || confidences.is_empty() {
            return Ok(AnomalyDetectionResults::default());
        }

        // Detect statistical anomalies
        let statistical_anomalies = self.detect_statistical_anomalies(latencies, confidences)?;

        // Detect temporal anomalies
        let temporal_anomalies = self.detect_temporal_anomalies(latencies)?;

        // Detect pattern anomalies
        let pattern_anomalies = self.detect_pattern_anomalies(latencies, confidences)?;

        // Calculate anomaly scores
        let anomaly_scores = self.calculate_anomaly_scores(latencies, confidences)?;

        // Generate anomaly summary
        let anomaly_summary = self.generate_anomaly_summary(
            &statistical_anomalies,
            &temporal_anomalies,
            &pattern_anomalies,
        )?;

        Ok(AnomalyDetectionResults {
            anomalies: vec![], // TODO: Convert different anomaly types to AnomalyEvent
            anomaly_scores,
            thresholds: HashMap::new(), // Placeholder - would need to be computed
            method_performance:
                crate::mid_circuit_measurements::results::AnomalyMethodPerformance {
                    precision: HashMap::new(),
                    recall: HashMap::new(),
                    f1_scores: HashMap::new(),
                    false_positive_rates: HashMap::new(),
                },
        })
    }

    /// Detect statistical anomalies using z-score method
    fn detect_statistical_anomalies(
        &self,
        latencies: &[f64],
        confidences: &[f64],
    ) -> DeviceResult<Vec<StatisticalAnomaly>> {
        let mut anomalies = Vec::new();

        // Detect anomalies in latencies
        let latency_anomalies = self.detect_zscore_anomalies(latencies, "latency")?;
        anomalies.extend(latency_anomalies);

        // Detect anomalies in confidences
        let confidence_anomalies = self.detect_zscore_anomalies(confidences, "confidence")?;
        anomalies.extend(confidence_anomalies);

        Ok(anomalies)
    }

    /// Detect z-score based anomalies
    fn detect_zscore_anomalies(
        &self,
        values: &[f64],
        metric_name: &str,
    ) -> DeviceResult<Vec<StatisticalAnomaly>> {
        if values.len() < 3 {
            return Ok(vec![]);
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let mut anomalies = Vec::new();

        for (index, &value) in values.iter().enumerate() {
            let z_score = if std_dev > 1e-10 {
                (value - mean) / std_dev
            } else {
                0.0
            };

            if z_score.abs() > self.threshold {
                anomalies.push(StatisticalAnomaly {
                    index,
                    value,
                    z_score,
                    p_value: self.calculate_p_value(z_score),
                    metric_type: metric_name.to_string(),
                    anomaly_severity: self.classify_severity(z_score.abs()),
                });
            }
        }

        Ok(anomalies)
    }

    /// Detect temporal anomalies using moving window
    fn detect_temporal_anomalies(&self, values: &[f64]) -> DeviceResult<Vec<TemporalAnomaly>> {
        if values.len() < self.window_size * 2 {
            return Ok(vec![]);
        }

        let mut anomalies = Vec::new();

        for i in self.window_size..(values.len() - self.window_size) {
            let before_window = &values[(i - self.window_size)..i];
            let after_window = &values[i..(i + self.window_size)];

            let before_mean = before_window.iter().sum::<f64>() / before_window.len() as f64;
            let after_mean = after_window.iter().sum::<f64>() / after_window.len() as f64;

            let change_magnitude = (after_mean - before_mean).abs();
            let relative_change = if before_mean > 1e-10 {
                change_magnitude / before_mean
            } else {
                change_magnitude
            };

            // Detect significant changes
            if relative_change > 0.3 {
                // 30% change threshold
                anomalies.push(TemporalAnomaly {
                    start_index: i - self.window_size,
                    end_index: i + self.window_size,
                    change_point: i,
                    magnitude: change_magnitude,
                    direction: if after_mean > before_mean {
                        ChangeDirection::Increase
                    } else {
                        ChangeDirection::Decrease
                    },
                    confidence: self.calculate_change_confidence(relative_change),
                });
            }
        }

        Ok(anomalies)
    }

    /// Detect pattern anomalies
    fn detect_pattern_anomalies(
        &self,
        latencies: &[f64],
        confidences: &[f64],
    ) -> DeviceResult<Vec<PatternAnomaly>> {
        let mut anomalies = Vec::new();

        // Detect correlation anomalies
        if latencies.len() == confidences.len() && latencies.len() > 10 {
            let correlation = self.calculate_correlation(latencies, confidences);

            // Expected negative correlation (higher latency -> lower confidence)
            if correlation > -0.1 {
                // Unexpectedly positive or weak correlation
                anomalies.push(PatternAnomaly {
                    pattern_type: PatternType::CorrelationAnomaly,
                    description: format!(
                        "Unexpected correlation between latency and confidence: {:.3}",
                        correlation
                    ),
                    severity: if correlation > 0.3 {
                        AnomalySeverity::High
                    } else {
                        AnomalySeverity::Medium
                    },
                    affected_indices: (0..latencies.len()).collect(),
                    confidence: 0.8,
                });
            }
        }

        // Detect sequence anomalies
        let sequence_anomalies = self.detect_sequence_anomalies(latencies)?;
        anomalies.extend(sequence_anomalies);

        Ok(anomalies)
    }

    /// Detect sequence anomalies (repeated patterns, outlier sequences)
    fn detect_sequence_anomalies(&self, values: &[f64]) -> DeviceResult<Vec<PatternAnomaly>> {
        let mut anomalies = Vec::new();

        if values.len() < 5 {
            return Ok(anomalies);
        }

        // Detect constant sequences (potential stuck measurements)
        let mut constant_start = 0;
        let mut constant_length = 1;

        for i in 1..values.len() {
            if (values[i] - values[i - 1]).abs() < 1e-6 {
                constant_length += 1;
            } else {
                if constant_length >= 5 {
                    // 5+ identical measurements
                    anomalies.push(PatternAnomaly {
                        pattern_type: PatternType::ConstantSequence,
                        description: format!(
                            "Constant sequence of {} identical values: {:.3}",
                            constant_length, values[constant_start]
                        ),
                        severity: AnomalySeverity::Medium,
                        affected_indices: (constant_start..constant_start + constant_length)
                            .collect(),
                        confidence: 0.9,
                    });
                }
                constant_start = i;
                constant_length = 1;
            }
        }

        // Check final sequence
        if constant_length >= 5 {
            anomalies.push(PatternAnomaly {
                pattern_type: PatternType::ConstantSequence,
                description: format!(
                    "Constant sequence of {} identical values: {:.3}",
                    constant_length, values[constant_start]
                ),
                severity: AnomalySeverity::Medium,
                affected_indices: (constant_start..constant_start + constant_length).collect(),
                confidence: 0.9,
            });
        }

        Ok(anomalies)
    }

    /// Calculate anomaly scores for all measurements
    fn calculate_anomaly_scores(
        &self,
        latencies: &[f64],
        confidences: &[f64],
    ) -> DeviceResult<Array1<f64>> {
        let n = latencies.len().min(confidences.len());
        let mut scores = Array1::zeros(n);

        if n == 0 {
            return Ok(scores);
        }

        // Calculate z-scores for latencies
        let latency_mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let latency_std = {
            let variance = latencies
                .iter()
                .map(|&x| (x - latency_mean).powi(2))
                .sum::<f64>()
                / latencies.len() as f64;
            variance.sqrt()
        };

        // Calculate z-scores for confidences
        let confidence_mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
        let confidence_std = {
            let variance = confidences
                .iter()
                .map(|&x| (x - confidence_mean).powi(2))
                .sum::<f64>()
                / confidences.len() as f64;
            variance.sqrt()
        };

        for i in 0..n {
            let latency_zscore = if latency_std > 1e-10 {
                (latencies[i] - latency_mean) / latency_std
            } else {
                0.0
            };

            let confidence_zscore = if confidence_std > 1e-10 {
                (confidences[i] - confidence_mean) / confidence_std
            } else {
                0.0
            };

            // Combine scores (higher is more anomalous)
            scores[i] = (latency_zscore.abs() + confidence_zscore.abs()) / 2.0;
        }

        Ok(scores)
    }

    /// Generate comprehensive anomaly summary
    fn generate_anomaly_summary(
        &self,
        statistical_anomalies: &[StatisticalAnomaly],
        temporal_anomalies: &[TemporalAnomaly],
        pattern_anomalies: &[PatternAnomaly],
    ) -> DeviceResult<AnomalySummary> {
        let total_anomalies =
            statistical_anomalies.len() + temporal_anomalies.len() + pattern_anomalies.len();

        // Count by severity
        let high_severity = statistical_anomalies
            .iter()
            .filter(|a| matches!(a.anomaly_severity, AnomalySeverity::High))
            .count()
            + pattern_anomalies
                .iter()
                .filter(|a| matches!(a.severity, AnomalySeverity::High))
                .count();

        let medium_severity = statistical_anomalies
            .iter()
            .filter(|a| matches!(a.anomaly_severity, AnomalySeverity::Medium))
            .count()
            + pattern_anomalies
                .iter()
                .filter(|a| matches!(a.severity, AnomalySeverity::Medium))
                .count();

        let low_severity = total_anomalies - high_severity - medium_severity;

        // Calculate overall anomaly rate
        let anomaly_rate = if total_anomalies > 0 {
            total_anomalies as f64 / 100.0 // Assuming base of 100 measurements
        } else {
            0.0
        };

        // Determine dominant anomaly types
        let mut anomaly_types = Vec::new();
        if !statistical_anomalies.is_empty() {
            anomaly_types.push("Statistical".to_string());
        }
        if !temporal_anomalies.is_empty() {
            anomaly_types.push("Temporal".to_string());
        }
        if !pattern_anomalies.is_empty() {
            anomaly_types.push("Pattern".to_string());
        }

        // Generate recommendations
        let mut recommendations = Vec::new();
        if high_severity > 0 {
            recommendations.push("Investigate high-severity anomalies immediately".to_string());
        }
        if temporal_anomalies.len() > 3 {
            recommendations.push("Review measurement timing and calibration".to_string());
        }
        if pattern_anomalies
            .iter()
            .any(|a| matches!(a.pattern_type, PatternType::ConstantSequence))
        {
            recommendations.push("Check for stuck measurement equipment".to_string());
        }

        Ok(AnomalySummary {
            total_anomalies,
            anomaly_rate,
            severity_distribution: vec![
                ("High".to_string(), high_severity),
                ("Medium".to_string(), medium_severity),
                ("Low".to_string(), low_severity),
            ],
            anomaly_types,
            recommendations,
        })
    }

    /// Calculate p-value for z-score (simplified)
    fn calculate_p_value(&self, z_score: f64) -> f64 {
        // Simplified p-value calculation
        let abs_z = z_score.abs();
        if abs_z > 3.0 {
            0.001
        } else if abs_z > 2.5 {
            0.01
        } else if abs_z > 2.0 {
            0.05
        } else {
            0.1
        }
    }

    /// Classify anomaly severity based on z-score
    fn classify_severity(&self, abs_z_score: f64) -> AnomalySeverity {
        if abs_z_score > 3.0 {
            AnomalySeverity::High
        } else if abs_z_score > 2.5 {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        }
    }

    /// Calculate change confidence
    fn calculate_change_confidence(&self, relative_change: f64) -> f64 {
        // Higher relative changes have higher confidence
        (relative_change * 2.0).min(1.0)
    }

    /// Calculate Pearson correlation
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
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AnomalyDetectionResults {
    fn default() -> Self {
        Self {
            anomalies: vec![],
            anomaly_scores: Array1::zeros(0),
            thresholds: HashMap::new(),
            method_performance:
                crate::mid_circuit_measurements::results::AnomalyMethodPerformance {
                    precision: HashMap::new(),
                    recall: HashMap::new(),
                    f1_scores: HashMap::new(),
                    false_positive_rates: HashMap::new(),
                },
        }
    }
}

impl Default for AnomalySummary {
    fn default() -> Self {
        Self {
            total_anomalies: 0,
            anomaly_rate: 0.0,
            severity_distribution: vec![],
            anomaly_types: vec![],
            recommendations: vec![],
        }
    }
}
