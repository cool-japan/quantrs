//! ML Monitoring Configuration Types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ML monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLMonitoringConfig {
    /// Real-time monitoring
    pub enable_real_time_monitoring: bool,
    /// Performance tracking
    pub performance_tracking: bool,
    /// Model drift detection
    pub drift_detection: DriftDetectionConfig,
    /// Anomaly detection
    pub anomaly_detection: bool,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,
}

/// Drift detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetectionConfig {
    /// Enable drift detection
    pub enable_detection: bool,
    /// Detection methods
    pub detection_methods: Vec<DriftDetectionMethod>,
    /// Detection window size
    pub window_size: usize,
    /// Significance threshold
    pub significance_threshold: f64,
}

/// Drift detection methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftDetectionMethod {
    ADWIN,
    DDM,
    EDDM,
    PageHinkley,
    KolmogorovSmirnov,
    PopulationStabilityIndex,
}
