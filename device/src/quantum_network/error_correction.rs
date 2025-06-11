//! Error correction and fault tolerance for quantum networks

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardErrorCorrectionConfig {
    pub fec_algorithm: String,
    pub code_rate: f64,
    pub interleaving: bool,
}

impl Default for ForwardErrorCorrectionConfig {
    fn default() -> Self {
        Self {
            fec_algorithm: "Reed-Solomon".to_string(),
            code_rate: 0.8,
            interleaving: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ARQConfig {
    pub arq_type: String,
    pub timeout_value: Duration,
    pub maximum_retransmissions: u32,
}

impl Default for ARQConfig {
    fn default() -> Self {
        Self {
            arq_type: "Go-Back-N".to_string(),
            timeout_value: Duration::from_millis(100),
            maximum_retransmissions: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridErrorCorrectionConfig {
    pub fec_arq_combination: String,
    pub adaptive_switching: bool,
    pub performance_threshold: f64,
    pub optimization_strategy: String,
}

impl Default for HybridErrorCorrectionConfig {
    fn default() -> Self {
        Self {
            fec_arq_combination: "parallel".to_string(),
            adaptive_switching: true,
            performance_threshold: 0.95,
            optimization_strategy: "adaptive".to_string(),
        }
    }
}
