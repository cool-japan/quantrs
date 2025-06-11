//! Quantum Key Distribution (QKD) protocols and configurations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyGenerationConfig {
    pub protocol: String,
    pub key_length: u32,
    pub generation_rate: f64,
    pub basis_choices: Vec<String>,
    pub error_correction: bool,
    pub privacy_amplification: bool,
}

impl Default for KeyGenerationConfig {
    fn default() -> Self {
        Self {
            protocol: "BB84".to_string(),
            key_length: 256,
            generation_rate: 1000.0,
            basis_choices: vec!["Z".to_string(), "X".to_string()],
            error_correction: true,
            privacy_amplification: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyAmplificationConfig {
    pub hash_function: String,
    pub compression_ratio: f64,
    pub security_parameter: u32,
    pub randomness_extraction: String,
    pub verification_enabled: bool,
}

impl Default for PrivacyAmplificationConfig {
    fn default() -> Self {
        Self {
            hash_function: "SHA-256".to_string(),
            compression_ratio: 0.5,
            security_parameter: 128,
            randomness_extraction: "leftover_hash".to_string(),
            verification_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReconciliationConfig {
    pub algorithm: String,
    pub efficiency: f64,
    pub parity_bits: u32,
    pub syndrome_length: u32,
    pub interactive_rounds: u32,
}

impl Default for ErrorReconciliationConfig {
    fn default() -> Self {
        Self {
            algorithm: "Cascade".to_string(),
            efficiency: 1.2,
            parity_bits: 1024,
            syndrome_length: 512,
            interactive_rounds: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKDSecurityConfig {
    pub security_proofs: Vec<String>,
    pub threat_model: String,
    pub privacy_parameter: f64,
    pub soundness_parameter: f64,
}

impl Default for QKDSecurityConfig {
    fn default() -> Self {
        Self {
            security_proofs: vec!["Finite-key".to_string()],
            threat_model: "Individual attack".to_string(),
            privacy_parameter: 0.11,
            soundness_parameter: 128.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceParameters {
    pub max_distance: f64,
    pub attenuation_coefficient: f64,
    pub dark_count_rate: f64,
}

impl Default for DistanceParameters {
    fn default() -> Self {
        Self {
            max_distance: 100.0,
            attenuation_coefficient: 0.2,
            dark_count_rate: 1000.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKDPerformanceConfig {
    pub target_key_rate: f64,
    pub distance_parameters: DistanceParameters,
    pub channel_model: String,
    pub optimization_targets: Vec<String>,
}

impl Default for QKDPerformanceConfig {
    fn default() -> Self {
        Self {
            target_key_rate: 1000.0,
            distance_parameters: DistanceParameters::default(),
            channel_model: "QBER".to_string(),
            optimization_targets: vec!["key_rate".to_string(), "efficiency".to_string()],
        }
    }
}
