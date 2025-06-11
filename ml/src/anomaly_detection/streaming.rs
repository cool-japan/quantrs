//! Real-time streaming anomaly detection

use crate::error::{MLError, Result};
use ndarray::{Array1, Array2, Axis};
use std::collections::VecDeque;

use super::config::*;
use super::core::QuantumAnomalyDetector;

/// Streaming anomaly detector for real-time processing
pub struct StreamingAnomalyDetector {
    detector: QuantumAnomalyDetector,
    buffer: VecDeque<Array1<f64>>,
    config: RealtimeConfig,
    drift_detector: Option<DriftDetector>,
}

/// Drift detection for concept drift in streaming data
#[derive(Debug)]
pub struct DriftDetector {
    threshold: f64,
    warning_threshold: f64,
    drift_detected: bool,
    warning_detected: bool,
    error_rate_history: VecDeque<f64>,
}

impl StreamingAnomalyDetector {
    /// Create new streaming anomaly detector
    pub fn new(detector: QuantumAnomalyDetector, config: RealtimeConfig) -> Self {
        let drift_detector = if config.drift_detection {
            Some(DriftDetector::new(0.05, 0.02))
        } else {
            None
        };

        StreamingAnomalyDetector {
            detector,
            buffer: VecDeque::with_capacity(config.buffer_size),
            config,
            drift_detector,
        }
    }

    /// Process a single sample
    pub fn process_sample(&mut self, sample: Array1<f64>) -> Result<f64> {
        // Add sample to buffer
        self.buffer.push_back(sample.clone());

        // Remove old samples if buffer is full
        while self.buffer.len() > self.config.buffer_size {
            self.buffer.pop_front();
        }

        // Detect anomaly if buffer has enough samples
        if self.buffer.len() >= self.config.buffer_size / 2 {
            let data = self.buffer_to_array()?;
            let result = self.detector.detect(&data)?;
            let anomaly_score = result.anomaly_scores[result.anomaly_scores.len() - 1];

            // Check for drift if enabled
            if let Some(ref mut drift_detector) = self.drift_detector {
                let is_anomaly = result.anomaly_labels[result.anomaly_labels.len() - 1] == 1;
                drift_detector.update(is_anomaly);

                if drift_detector.is_drift_detected() {
                    // Handle drift (could retrain model here)
                    drift_detector.reset();
                }
            }

            return Ok(anomaly_score);
        }

        // Not enough data for detection
        Ok(0.0)
    }

    /// Process a batch of samples
    pub fn process_batch(&mut self, batch: &Array2<f64>) -> Result<Array1<f64>> {
        let mut scores = Array1::zeros(batch.nrows());

        for (i, sample) in batch.outer_iter().enumerate() {
            scores[i] = self.process_sample(sample.to_owned())?;
        }

        Ok(scores)
    }

    /// Update the underlying detector with new labeled data
    pub fn update_detector(
        &mut self,
        data: &Array2<f64>,
        labels: Option<&Array1<i32>>,
    ) -> Result<()> {
        if self.config.online_learning {
            self.detector.update(data, labels)?;
        }
        Ok(())
    }

    /// Check if drift is detected
    pub fn is_drift_detected(&self) -> bool {
        self.drift_detector
            .as_ref()
            .map(|d| d.is_drift_detected())
            .unwrap_or(false)
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the buffer
    pub fn clear_buffer(&mut self) {
        self.buffer.clear();
    }

    // Helper methods

    fn buffer_to_array(&self) -> Result<Array2<f64>> {
        if self.buffer.is_empty() {
            return Err(MLError::DataError("Buffer is empty".to_string()));
        }

        let n_samples = self.buffer.len();
        let n_features = self.buffer[0].len();

        let data = Array2::from_shape_vec(
            (n_samples, n_features),
            self.buffer.iter().flat_map(|s| s.iter().cloned()).collect(),
        )
        .map_err(|e| MLError::DataError(e.to_string()))?;

        Ok(data)
    }
}

impl DriftDetector {
    /// Create new drift detector
    pub fn new(drift_threshold: f64, warning_threshold: f64) -> Self {
        DriftDetector {
            threshold: drift_threshold,
            warning_threshold,
            drift_detected: false,
            warning_detected: false,
            error_rate_history: VecDeque::with_capacity(1000),
        }
    }

    /// Update with a new prediction result
    pub fn update(&mut self, is_error: bool) {
        let error_rate = if is_error { 1.0 } else { 0.0 };
        self.error_rate_history.push_back(error_rate);

        // Keep only recent history
        while self.error_rate_history.len() > 100 {
            self.error_rate_history.pop_front();
        }

        // Calculate moving average error rate
        let avg_error_rate =
            self.error_rate_history.iter().sum::<f64>() / self.error_rate_history.len() as f64;

        // Check for warning level
        if avg_error_rate > self.warning_threshold {
            self.warning_detected = true;
        }

        // Check for drift
        if avg_error_rate > self.threshold {
            self.drift_detected = true;
        }
    }

    /// Check if drift is detected
    pub fn is_drift_detected(&self) -> bool {
        self.drift_detected
    }

    /// Check if warning is detected
    pub fn is_warning_detected(&self) -> bool {
        self.warning_detected
    }

    /// Reset the drift detector
    pub fn reset(&mut self) {
        self.drift_detected = false;
        self.warning_detected = false;
        self.error_rate_history.clear();
    }

    /// Get current error rate
    pub fn current_error_rate(&self) -> f64 {
        if self.error_rate_history.is_empty() {
            0.0
        } else {
            self.error_rate_history.iter().sum::<f64>() / self.error_rate_history.len() as f64
        }
    }
}
