//! Process monitoring and anomaly detection

use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::super::results::*;
use crate::DeviceResult;

impl ProcessAnomalyDetector {
    /// Create new anomaly detector
    pub const fn new(threshold: f64, algorithm: AnomalyDetectionAlgorithm) -> Self {
        Self {
            historical_data: Vec::new(),
            threshold,
            algorithm,
        }
    }

    /// Add process metrics to historical data
    pub fn add_process_metrics(&mut self, metrics: ProcessMetrics) {
        self.historical_data.push(metrics);

        // Keep only recent data (e.g., last 1000 measurements)
        if self.historical_data.len() > 1000 {
            self.historical_data.remove(0);
        }
    }

    /// Detect anomalies in current process metrics
    pub fn detect_anomaly(&self, current_metrics: &ProcessMetrics) -> DeviceResult<bool> {
        if self.historical_data.len() < 10 {
            return Ok(false); // Not enough data for detection
        }

        match self.algorithm {
            AnomalyDetectionAlgorithm::StatisticalThreshold => {
                self.detect_statistical_anomaly(current_metrics)
            }
            AnomalyDetectionAlgorithm::IsolationForest => {
                self.detect_isolation_forest_anomaly(current_metrics)
            }
            AnomalyDetectionAlgorithm::OneClassSVM => {
                self.detect_one_class_svm_anomaly(current_metrics)
            }
            AnomalyDetectionAlgorithm::LocalOutlierFactor => {
                self.detect_lof_anomaly(current_metrics)
            }
        }
    }

    /// Statistical threshold-based anomaly detection
    fn detect_statistical_anomaly(&self, current_metrics: &ProcessMetrics) -> DeviceResult<bool> {
        // Calculate statistics for historical data
        let historical_fidelities: Vec<f64> = self
            .historical_data
            .iter()
            .map(|m| m.process_fidelity)
            .collect();

        let historical_unitarities: Vec<f64> =
            self.historical_data.iter().map(|m| m.unitarity).collect();

        let mean_fidelity =
            historical_fidelities.iter().sum::<f64>() / historical_fidelities.len() as f64;
        let std_fidelity = {
            let variance = historical_fidelities
                .iter()
                .map(|&x| (x - mean_fidelity).powi(2))
                .sum::<f64>()
                / historical_fidelities.len() as f64;
            variance.sqrt()
        };

        let mean_unitarity =
            historical_unitarities.iter().sum::<f64>() / historical_unitarities.len() as f64;
        let std_unitarity = {
            let variance = historical_unitarities
                .iter()
                .map(|&x| (x - mean_unitarity).powi(2))
                .sum::<f64>()
                / historical_unitarities.len() as f64;
            variance.sqrt()
        };

        // Check if current metrics are beyond threshold
        let fidelity_z_score = if std_fidelity > 1e-12 {
            (current_metrics.process_fidelity - mean_fidelity).abs() / std_fidelity
        } else {
            0.0
        };

        let unitarity_z_score = if std_unitarity > 1e-12 {
            (current_metrics.unitarity - mean_unitarity).abs() / std_unitarity
        } else {
            0.0
        };

        Ok(fidelity_z_score > self.threshold || unitarity_z_score > self.threshold)
    }

    /// Isolation forest anomaly detection (simplified)
    fn detect_isolation_forest_anomaly(
        &self,
        current_metrics: &ProcessMetrics,
    ) -> DeviceResult<bool> {
        // Simplified isolation forest
        let features = self.extract_features(current_metrics);
        let anomaly_score = self.calculate_isolation_score(&features);
        Ok(anomaly_score > self.threshold)
    }

    /// One-class SVM anomaly detection (simplified)
    fn detect_one_class_svm_anomaly(&self, current_metrics: &ProcessMetrics) -> DeviceResult<bool> {
        // Simplified one-class SVM
        let features = self.extract_features(current_metrics);
        let distance_from_hyperplane = self.calculate_svm_distance(&features);
        Ok(distance_from_hyperplane < -self.threshold)
    }

    /// Local outlier factor anomaly detection (simplified)
    fn detect_lof_anomaly(&self, current_metrics: &ProcessMetrics) -> DeviceResult<bool> {
        // Simplified LOF calculation
        let features = self.extract_features(current_metrics);
        let lof_score = self.calculate_lof_score(&features);
        Ok(lof_score > self.threshold)
    }

    /// Extract features from process metrics
    fn extract_features(&self, metrics: &ProcessMetrics) -> Vec<f64> {
        vec![
            metrics.process_fidelity,
            metrics.average_gate_fidelity,
            metrics.unitarity,
            metrics.entangling_power,
            metrics.non_unitality,
            metrics.channel_capacity,
            metrics.coherent_information,
            metrics.diamond_norm_distance,
        ]
    }

    /// Calculate isolation score (simplified)
    fn calculate_isolation_score(&self, features: &[f64]) -> f64 {
        // Simplified isolation score based on distance from historical mean
        let mut total_distance = 0.0;

        for (i, &feature) in features.iter().enumerate() {
            let historical_values: Vec<f64> = self
                .historical_data
                .iter()
                .map(|m| self.extract_features(m)[i])
                .collect();

            if !historical_values.is_empty() {
                let mean = historical_values.iter().sum::<f64>() / historical_values.len() as f64;
                total_distance += (feature - mean).abs();
            }
        }

        total_distance / features.len() as f64
    }

    /// Calculate SVM distance (simplified)
    fn calculate_svm_distance(&self, features: &[f64]) -> f64 {
        // Simplified SVM distance calculation
        let mut weighted_sum = 0.0;

        for (i, &feature) in features.iter().enumerate() {
            let weight = 1.0 / (i + 1) as f64; // Simple weight scheme
            weighted_sum += weight * feature;
        }

        weighted_sum - 0.5 // Assume hyperplane at 0.5
    }

    /// Calculate LOF score (simplified)
    fn calculate_lof_score(&self, features: &[f64]) -> f64 {
        // Simplified LOF calculation
        let k = 5.min(self.historical_data.len()); // Number of nearest neighbors

        if k == 0 {
            return 1.0;
        }

        // Calculate distances to historical data
        let mut distances: Vec<f64> = self
            .historical_data
            .iter()
            .map(|historical_metrics| {
                let historical_features = self.extract_features(historical_metrics);
                self.euclidean_distance(features, &historical_features)
            })
            .collect();

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate local reachability density
        let k_distance = distances[k.min(distances.len()) - 1];
        let lrd = if k_distance > 1e-12 {
            k as f64 / k_distance
        } else {
            f64::INFINITY
        };

        // Return LOF score (simplified)
        1.0 / lrd.max(1e-12)
    }

    /// Calculate Euclidean distance between feature vectors
    fn euclidean_distance(&self, features1: &[f64], features2: &[f64]) -> f64 {
        features1
            .iter()
            .zip(features2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl ProcessDriftDetector {
    /// Create new drift detector
    pub const fn new(
        reference_metrics: ProcessMetrics,
        sensitivity: f64,
        method: DriftDetectionMethod,
    ) -> Self {
        Self {
            reference_metrics,
            sensitivity,
            method,
        }
    }

    /// Detect drift in current process metrics
    pub fn detect_drift(&self, historical_data: &[ProcessMetrics]) -> DeviceResult<bool> {
        if historical_data.is_empty() {
            return Ok(false);
        }

        match self.method {
            DriftDetectionMethod::StatisticalTest => self.detect_statistical_drift(historical_data),
            DriftDetectionMethod::ChangePointDetection => {
                self.detect_change_point_drift(historical_data)
            }
            DriftDetectionMethod::KLDivergence => self.detect_kl_divergence_drift(historical_data),
            DriftDetectionMethod::WassersteinDistance => {
                self.detect_wasserstein_drift(historical_data)
            }
        }
    }

    /// Statistical test-based drift detection
    fn detect_statistical_drift(&self, historical_data: &[ProcessMetrics]) -> DeviceResult<bool> {
        // Compare recent data with reference using t-test
        let recent_fidelities: Vec<f64> = historical_data.iter()
            .rev()
            .take(20) // Take last 20 measurements
            .map(|m| m.process_fidelity)
            .collect();

        if recent_fidelities.is_empty() {
            return Ok(false);
        }

        let recent_mean = recent_fidelities.iter().sum::<f64>() / recent_fidelities.len() as f64;
        let reference_fidelity = self.reference_metrics.process_fidelity;

        let difference = (recent_mean - reference_fidelity).abs();
        let threshold = self.sensitivity * reference_fidelity;

        Ok(difference > threshold)
    }

    /// Change point detection-based drift detection
    fn detect_change_point_drift(&self, historical_data: &[ProcessMetrics]) -> DeviceResult<bool> {
        if historical_data.len() < 10 {
            return Ok(false);
        }

        let fidelities: Vec<f64> = historical_data.iter().map(|m| m.process_fidelity).collect();

        // CUSUM (Cumulative Sum) change point detection
        let reference_mean = self.reference_metrics.process_fidelity;
        let mut cusum_pos = 0.0;
        let mut cusum_neg = 0.0;
        let threshold = self.sensitivity * 0.1; // Threshold for CUSUM

        for &fidelity in &fidelities {
            let deviation = fidelity - reference_mean;
            cusum_pos = (cusum_pos + deviation).max(0.0);
            cusum_neg = (cusum_neg - deviation).max(0.0);

            if cusum_pos > threshold || cusum_neg > threshold {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// KL divergence-based drift detection
    fn detect_kl_divergence_drift(&self, historical_data: &[ProcessMetrics]) -> DeviceResult<bool> {
        // Simplified KL divergence calculation
        let recent_data: Vec<f64> = historical_data
            .iter()
            .rev()
            .take(50)
            .map(|m| m.process_fidelity)
            .collect();

        if recent_data.len() < 10 {
            return Ok(false);
        }

        // Create histograms
        let reference_hist = self.create_histogram(&[self.reference_metrics.process_fidelity]);
        let recent_hist = self.create_histogram(&recent_data);

        let kl_divergence = self.calculate_kl_divergence(&reference_hist, &recent_hist);

        Ok(kl_divergence > self.sensitivity)
    }

    /// Wasserstein distance-based drift detection
    fn detect_wasserstein_drift(&self, historical_data: &[ProcessMetrics]) -> DeviceResult<bool> {
        // Simplified Wasserstein distance
        let recent_data: Vec<f64> = historical_data
            .iter()
            .rev()
            .take(50)
            .map(|m| m.process_fidelity)
            .collect();

        if recent_data.is_empty() {
            return Ok(false);
        }

        let reference_value = self.reference_metrics.process_fidelity;
        let recent_mean = recent_data.iter().sum::<f64>() / recent_data.len() as f64;

        let wasserstein_distance = (reference_value - recent_mean).abs();

        Ok(wasserstein_distance > self.sensitivity * 0.1)
    }

    /// Create histogram from data
    fn create_histogram(&self, data: &[f64]) -> Vec<f64> {
        let num_bins = 10;
        let mut histogram = vec![0.0; num_bins];

        if data.is_empty() {
            return histogram;
        }

        let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-12 {
            histogram[0] = 1.0;
            return histogram;
        }

        let bin_width = (max_val - min_val) / num_bins as f64;

        for &value in data {
            let bin_idx = ((value - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1);
            histogram[bin_idx] += 1.0;
        }

        // Normalize
        let total = histogram.iter().sum::<f64>();
        if total > 1e-12 {
            for bin in &mut histogram {
                *bin /= total;
            }
        }

        histogram
    }

    /// Calculate KL divergence between two distributions
    fn calculate_kl_divergence(&self, p: &[f64], q: &[f64]) -> f64 {
        let mut kl_div = 0.0;
        let epsilon = 1e-12; // Small value to avoid log(0)

        for (p_i, q_i) in p.iter().zip(q.iter()) {
            let p_safe = p_i.max(epsilon);
            let q_safe = q_i.max(epsilon);
            kl_div += p_safe * (p_safe / q_safe).ln();
        }

        kl_div
    }
}

/// Generate process monitoring result
pub fn generate_monitoring_result(
    current_metrics: ProcessMetrics,
    anomaly_detector: &ProcessAnomalyDetector,
    drift_detector: &ProcessDriftDetector,
    experimental_conditions: ExperimentalConditions,
) -> DeviceResult<ProcessMonitoringResult> {
    // Detect anomalies
    let is_anomalous = anomaly_detector.detect_anomaly(&current_metrics)?;
    let anomaly_score = if is_anomalous { 0.8 } else { 0.1 };

    // Detect drift
    let has_drift = drift_detector.detect_drift(&anomaly_detector.historical_data)?;
    let drift_indicator = if has_drift { 0.7 } else { 0.05 };

    // Determine alert level
    let alert_level = if anomaly_score > 0.7 || drift_indicator > 0.6 {
        AlertLevel::Critical
    } else if anomaly_score > 0.4 || drift_indicator > 0.3 {
        AlertLevel::Warning
    } else {
        AlertLevel::Normal
    };

    Ok(ProcessMonitoringResult {
        current_metrics,
        experimental_conditions,
        anomaly_score,
        drift_indicator,
        alert_level,
    })
}
