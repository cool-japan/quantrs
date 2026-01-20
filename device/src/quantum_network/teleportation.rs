//! Quantum teleportation protocols and configurations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatePreparationConfig {
    pub preparation_method: String,
    pub state_fidelity: f64,
    pub verification_protocol: String,
}

impl Default for StatePreparationConfig {
    fn default() -> Self {
        Self {
            preparation_method: "direct".to_string(),
            state_fidelity: 0.99,
            verification_protocol: "process_tomography".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleportationMeasurementConfig {
    pub measurement_basis: String,
    pub measurement_efficiency: f64,
    pub detection_probability: f64,
    pub timing_precision: Duration,
}

impl Default for TeleportationMeasurementConfig {
    fn default() -> Self {
        Self {
            measurement_basis: "Bell".to_string(),
            measurement_efficiency: 0.95,
            detection_probability: 0.9,
            timing_precision: Duration::from_nanos(100),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleportationClassicalConfig {
    pub communication_channel: String,
    pub message_encoding: String,
    pub compression_enabled: bool,
}

impl Default for TeleportationClassicalConfig {
    fn default() -> Self {
        Self {
            communication_channel: "optical".to_string(),
            message_encoding: "binary".to_string(),
            compression_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeleportationFidelityConfig {
    pub minimum_fidelity: f64,
    pub average_fidelity_target: f64,
    pub fidelity_estimation_method: String,
    pub verification_shots: u32,
}

impl Default for TeleportationFidelityConfig {
    fn default() -> Self {
        Self {
            minimum_fidelity: 0.9,
            average_fidelity_target: 0.95,
            fidelity_estimation_method: "process_tomography".to_string(),
            verification_shots: 1000,
        }
    }
}
