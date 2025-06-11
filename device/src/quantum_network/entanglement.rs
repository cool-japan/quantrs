//! Entanglement distribution and management protocols

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementGenerationConfig {
    pub source_type: String,
    pub generation_rate: f64,
    pub fidelity_target: f64,
    pub entanglement_type: String,
}

impl Default for EntanglementGenerationConfig {
    fn default() -> Self {
        Self {
            source_type: "SPDC".to_string(),
            generation_rate: 1000.0,
            fidelity_target: 0.95,
            entanglement_type: "Bell_state".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementPurificationConfig {
    pub purification_protocol: String,
    pub fidelity_target: f64,
    pub maximum_rounds: u32,
    pub resource_efficiency: f64,
}

impl Default for EntanglementPurificationConfig {
    fn default() -> Self {
        Self {
            purification_protocol: "DEJMPS".to_string(),
            fidelity_target: 0.99,
            maximum_rounds: 5,
            resource_efficiency: 0.8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementSwappingConfig {
    pub swapping_protocol: String,
    pub fidelity_preservation: f64,
    pub network_topology_aware: bool,
}

impl Default for EntanglementSwappingConfig {
    fn default() -> Self {
        Self {
            swapping_protocol: "standard".to_string(),
            fidelity_preservation: 0.95,
            network_topology_aware: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementStorageConfig {
    pub storage_method: String,
    pub storage_time: Duration,
    pub retrieval_efficiency: f64,
}

impl Default for EntanglementStorageConfig {
    fn default() -> Self {
        Self {
            storage_method: "quantum_memory".to_string(),
            storage_time: Duration::from_millis(100),
            retrieval_efficiency: 0.9,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementQualityConfig {
    pub fidelity_threshold: f64,
    pub entanglement_verification: String,
    pub quality_monitoring_interval: Duration,
}

impl Default for EntanglementQualityConfig {
    fn default() -> Self {
        Self {
            fidelity_threshold: 0.9,
            entanglement_verification: "process_tomography".to_string(),
            quality_monitoring_interval: Duration::from_secs(1),
        }
    }
}
