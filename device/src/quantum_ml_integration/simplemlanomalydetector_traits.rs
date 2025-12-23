//! # SimpleMLAnomalyDetector - Trait Implementations
//!
//! This module contains trait implementations for `SimpleMLAnomalyDetector`.
//!
//! ## Implemented Traits
//!
//! - `AnomalyDetector`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

// Import types from sibling modules
use super::types::*;
// Merged into super::types
// Merged into super::types
// Import traits from functions module
use super::functions::AnomalyDetector;

use std::time::Instant;

impl AnomalyDetector for SimpleMLAnomalyDetector {
    fn detect(&self, data: &[(Instant, f64)]) -> Vec<DetectedAnomaly> {
        if data.len() < 3 {
            return Vec::new();
        }
        let values: Vec<f64> = data.iter().map(|(_, v)| *v).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        let std_dev = variance.sqrt();
        data.iter()
            .enumerate()
            .filter_map(|(i, &(timestamp, value))| {
                if (value - mean).abs() > self.threshold * std_dev {
                    Some(DetectedAnomaly {
                        anomaly_type: AnomalyType::PerformanceDegradation,
                        severity: (value - mean).abs() / std_dev,
                        description: format!(
                            "Value {} deviates significantly from mean {}",
                            value, mean
                        ),
                        timestamp,
                        affected_metrics: vec!["performance".to_string()],
                    })
                } else {
                    None
                }
            })
            .collect()
    }
    fn update(&mut self, _data: &[(Instant, f64)]) {}
    fn threshold(&self) -> f64 {
        self.threshold
    }
    fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }
}
