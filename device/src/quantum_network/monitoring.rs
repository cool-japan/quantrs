//! Network monitoring and analytics

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkErrorModelConfig {
    pub error_model_type: String,
    pub error_parameters: HashMap<String, f64>,
    pub temporal_correlation: f64,
    pub spatial_correlation: f64,
}

impl Default for NetworkErrorModelConfig {
    fn default() -> Self {
        let mut error_parameters = HashMap::new();
        error_parameters.insert("bit_flip_rate".to_string(), 0.01);
        error_parameters.insert("phase_flip_rate".to_string(), 0.01);

        Self {
            error_model_type: "pauli".to_string(),
            error_parameters,
            temporal_correlation: 0.1,
            spatial_correlation: 0.05,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPerformanceMonitoringConfig {
    pub performance_metrics: Vec<String>,
    pub monitoring_interval: Duration,
    pub real_time_monitoring: bool,
    pub historical_data_retention: Duration,
}

impl Default for NetworkPerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            performance_metrics: vec![
                "latency".to_string(),
                "throughput".to_string(),
                "fidelity".to_string(),
            ],
            monitoring_interval: Duration::from_secs(1),
            real_time_monitoring: true,
            historical_data_retention: Duration::from_secs(3600),
        }
    }
}
