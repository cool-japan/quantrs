//! Photonic Quantum Communication Protocols
//!
//! This module implements quantum communication protocols specifically designed for
//! photonic systems, including quantum key distribution, teleportation, and networking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

use super::continuous_variable::{CVResult, Complex, GaussianState};
use super::gate_based::{PhotonicQubitEncoding, PhotonicQubitState};
use super::{PhotonicMode, PhotonicSystemType};
use crate::DeviceResult;

/// Photonic protocol errors
#[derive(Error, Debug)]
pub enum PhotonicProtocolError {
    #[error("Protocol execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    #[error("Security violation: {0}")]
    SecurityViolation(String),
    #[error("Network communication error: {0}")]
    NetworkError(String),
    #[error("Protocol not supported: {0}")]
    UnsupportedProtocol(String),
}

/// Types of photonic quantum protocols
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhotonicProtocolType {
    /// Quantum Key Distribution
    QKD { variant: QKDVariant },
    /// Quantum Teleportation
    Teleportation,
    /// Quantum State Distribution
    StateDistribution,
    /// Quantum Clock Synchronization
    ClockSynchronization,
    /// Quantum Sensing Networks
    SensingNetwork,
    /// Quantum Internet Protocols
    QuantumInternet { protocol_version: String },
}

/// QKD protocol variants
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QKDVariant {
    /// BB84 protocol
    BB84,
    /// B92 protocol
    B92,
    /// SARG04 protocol
    SARG04,
    /// Continuous Variable QKD
    CVQKD { modulation: CVModulation },
    /// Measurement Device Independent QKD
    MDIQKD,
    /// Twin Field QKD
    TwinField,
}

/// CV QKD modulation schemes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CVModulation {
    /// Gaussian modulation
    Gaussian { variance: f64 },
    /// Discrete modulation
    Discrete { constellation_size: usize },
    /// Heterodyne detection
    Heterodyne,
    /// Homodyne detection
    Homodyne,
}

/// Protocol execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolContext {
    /// Protocol identifier
    pub protocol_id: String,
    /// Participating parties
    pub parties: Vec<ProtocolParty>,
    /// Security parameters
    pub security_params: SecurityParameters,
    /// Network configuration
    pub network_config: NetworkConfiguration,
    /// Start time
    #[serde(with = "instant_serde")]
    pub start_time: Instant,
}

/// Party in a quantum protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolParty {
    /// Party identifier
    pub party_id: String,
    /// Role in protocol
    pub role: PartyRole,
    /// Communication endpoints
    pub endpoints: Vec<String>,
    /// Capabilities
    pub capabilities: PartyCapabilities,
}

/// Role of a party in quantum protocols
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PartyRole {
    /// Alice (sender)
    Alice,
    /// Bob (receiver)
    Bob,
    /// Charlie (trusted third party)
    Charlie,
    /// Eve (eavesdropper - for security analysis)
    Eve,
    /// Network node
    NetworkNode { node_id: String },
}

/// Capabilities of a protocol party
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartyCapabilities {
    /// Supported encodings
    pub encodings: Vec<PhotonicQubitEncoding>,
    /// Detection efficiency
    pub detection_efficiency: f64,
    /// Maximum transmission rate
    pub max_rate: f64,
    /// Supported wavelengths
    pub wavelengths: Vec<f64>,
    /// Distance limitations
    pub max_distance: Option<f64>,
}

/// Security parameters for protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityParameters {
    /// Required security level (bits)
    pub security_level: usize,
    /// Error tolerance
    pub error_tolerance: f64,
    /// Privacy amplification parameters
    pub privacy_amplification: PrivacyAmplificationParams,
    /// Authentication method
    pub authentication: AuthenticationMethod,
}

/// Privacy amplification parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyAmplificationParams {
    /// Hash function family
    pub hash_family: String,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Number of rounds
    pub rounds: usize,
}

/// Authentication methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuthenticationMethod {
    /// Classical authentication
    Classical { algorithm: String },
    /// Quantum authentication
    Quantum { protocol: String },
    /// Post-quantum cryptography
    PostQuantum { algorithm: String },
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfiguration {
    /// Topology type
    pub topology: NetworkTopology,
    /// Channel characteristics
    pub channels: Vec<QuantumChannel>,
    /// Routing configuration
    pub routing: RoutingConfig,
}

/// Network topology types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NetworkTopology {
    /// Point-to-point
    PointToPoint,
    /// Star network
    Star { hub: String },
    /// Ring network
    Ring,
    /// Mesh network
    Mesh,
    /// Tree network
    Tree { root: String },
}

/// Quantum communication channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumChannel {
    /// Channel identifier
    pub channel_id: String,
    /// Source party
    pub source: String,
    /// Destination party
    pub destination: String,
    /// Channel characteristics
    pub characteristics: ChannelCharacteristics,
}

/// Physical characteristics of quantum channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelCharacteristics {
    /// Loss rate (dB/km)
    pub loss_rate: f64,
    /// Dark count rate
    pub dark_count_rate: f64,
    /// Channel length (km)
    pub length: f64,
    /// Wavelength (nm)
    pub wavelength: f64,
    /// Detector efficiency
    pub detector_efficiency: f64,
}

/// Routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Routing algorithm
    pub algorithm: RoutingAlgorithm,
    /// Maximum hops
    pub max_hops: usize,
    /// Load balancing enabled
    pub load_balancing: bool,
}

/// Routing algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RoutingAlgorithm {
    /// Shortest path
    ShortestPath,
    /// Minimum loss
    MinimumLoss,
    /// Load balanced
    LoadBalanced,
    /// Quantum-aware routing
    QuantumAware,
}

/// Protocol execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolResult {
    /// Whether protocol succeeded
    pub success: bool,
    /// Generated key (for QKD)
    pub key: Option<Vec<u8>>,
    /// Transmitted state (for teleportation)
    pub transmitted_state: Option<PhotonicQubitState>,
    /// Protocol metrics
    pub metrics: ProtocolMetrics,
    /// Security analysis
    pub security_analysis: SecurityAnalysis,
}

/// Protocol performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolMetrics {
    /// Execution time
    pub execution_time: Duration,
    /// Key rate (bits/second)
    pub key_rate: Option<f64>,
    /// Error rate
    pub error_rate: f64,
    /// Fidelity
    pub fidelity: f64,
    /// Throughput
    pub throughput: f64,
}

/// Security analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAnalysis {
    /// Information leakage estimate
    pub information_leakage: f64,
    /// Eavesdropping detection
    pub eavesdropping_detected: bool,
    /// Security proof validity
    pub security_proof_valid: bool,
    /// Achieved security level
    pub achieved_security: f64,
}

/// Photonic protocol engine
pub struct PhotonicProtocolEngine {
    /// Active protocols
    pub active_protocols: HashMap<String, ProtocolContext>,
    /// Protocol statistics
    pub statistics: ProtocolStatistics,
    /// Security monitor
    pub security_monitor: SecurityMonitor,
}

/// Protocol execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolStatistics {
    /// Total protocols executed
    pub total_protocols: usize,
    /// Success rate
    pub success_rate: f64,
    /// Average key rate
    pub average_key_rate: f64,
    /// Average fidelity
    pub average_fidelity: f64,
    /// Security violations detected
    pub security_violations: usize,
}

/// Security monitoring system
#[derive(Debug, Clone)]
pub struct SecurityMonitor {
    /// Threat detection enabled
    pub threat_detection: bool,
    /// Anomaly detection threshold
    pub anomaly_threshold: f64,
    /// Security events
    pub security_events: Vec<SecurityEvent>,
}

/// Security event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    /// Event timestamp
    #[serde(with = "instant_serde")]
    pub timestamp: Instant,
    /// Event type
    pub event_type: SecurityEventType,
    /// Severity level
    pub severity: SecuritySeverity,
    /// Description
    pub description: String,
    /// Affected protocols
    pub affected_protocols: Vec<String>,
}

mod instant_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration_since_epoch = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        duration_since_epoch.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Instant::now()) // Simplified
    }
}

/// Types of security events
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SecurityEventType {
    /// Eavesdropping attempt detected
    EavesdroppingDetected,
    /// Anomalous error rate
    AnomalousErrorRate,
    /// Authentication failure
    AuthenticationFailure,
    /// Protocol violation
    ProtocolViolation,
    /// Network intrusion
    NetworkIntrusion,
}

/// Security event severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl PhotonicProtocolEngine {
    pub fn new() -> Self {
        Self {
            active_protocols: HashMap::new(),
            statistics: ProtocolStatistics {
                total_protocols: 0,
                success_rate: 0.0,
                average_key_rate: 0.0,
                average_fidelity: 0.0,
                security_violations: 0,
            },
            security_monitor: SecurityMonitor {
                threat_detection: true,
                anomaly_threshold: 0.05,
                security_events: Vec::new(),
            },
        }
    }

    /// Execute a quantum protocol
    pub fn execute_protocol(
        &mut self,
        protocol_type: PhotonicProtocolType,
        context: ProtocolContext,
    ) -> Result<ProtocolResult, PhotonicProtocolError> {
        let start_time = Instant::now();

        // Register protocol
        self.active_protocols
            .insert(context.protocol_id.clone(), context.clone());

        // Execute based on protocol type
        let result = match protocol_type {
            PhotonicProtocolType::QKD { variant } => self.execute_qkd_protocol(variant, &context),
            PhotonicProtocolType::Teleportation => self.execute_teleportation_protocol(&context),
            PhotonicProtocolType::StateDistribution => {
                self.execute_state_distribution_protocol(&context)
            }
            PhotonicProtocolType::ClockSynchronization => {
                self.execute_clock_sync_protocol(&context)
            }
            PhotonicProtocolType::SensingNetwork => self.execute_sensing_protocol(&context),
            PhotonicProtocolType::QuantumInternet { protocol_version } => {
                self.execute_quantum_internet_protocol(&context, &protocol_version)
            }
        }?;

        // Update statistics
        self.update_statistics(&result);

        // Remove from active protocols
        self.active_protocols.remove(&context.protocol_id);

        Ok(result)
    }

    /// Execute QKD protocol
    fn execute_qkd_protocol(
        &mut self,
        variant: QKDVariant,
        context: &ProtocolContext,
    ) -> Result<ProtocolResult, PhotonicProtocolError> {
        match variant {
            QKDVariant::BB84 => self.execute_bb84(context),
            QKDVariant::CVQKD { modulation } => self.execute_cvqkd(context, modulation),
            _ => Err(PhotonicProtocolError::UnsupportedProtocol(format!(
                "QKD variant {:?} not implemented",
                variant
            ))),
        }
    }

    /// Execute BB84 protocol
    fn execute_bb84(
        &mut self,
        context: &ProtocolContext,
    ) -> Result<ProtocolResult, PhotonicProtocolError> {
        let start_time = Instant::now();

        // Simulate BB84 protocol execution
        let key_length = 256; // bits
        let mut raw_key = Vec::new();
        let mut error_count = 0;

        // Simulate photon transmission and measurement
        for _ in 0..key_length * 2 {
            // Send 2x for basis reconciliation
            let bit = rand::random::<bool>();
            let basis = rand::random::<bool>(); // 0: rectilinear, 1: diagonal

            // Simulate channel loss and errors
            let channel_loss = 0.05; // 5% loss
            let error_rate = 0.01; // 1% error rate

            if rand::random::<f64>() > channel_loss {
                let received_bit = if rand::random::<f64>() < error_rate {
                    !bit // Flip bit due to error
                } else {
                    bit
                };

                if rand::random::<bool>() {
                    // Bob chooses same basis 50% of time
                    raw_key.push(received_bit as u8);
                    if received_bit != bit {
                        error_count += 1;
                    }
                }
            }
        }

        // Trim to desired key length
        raw_key.truncate(key_length);
        let final_error_rate = error_count as f64 / raw_key.len() as f64;

        // Security analysis
        let security_analysis = SecurityAnalysis {
            information_leakage: final_error_rate * 0.5, // Simplified estimate
            eavesdropping_detected: final_error_rate > 0.11, // BB84 threshold
            security_proof_valid: final_error_rate <= 0.11,
            achieved_security: if final_error_rate <= 0.11 {
                context.security_params.security_level as f64
            } else {
                0.0
            },
        };

        let metrics = ProtocolMetrics {
            execution_time: start_time.elapsed(),
            key_rate: Some(raw_key.len() as f64 / start_time.elapsed().as_secs_f64()),
            error_rate: final_error_rate,
            fidelity: 1.0 - final_error_rate,
            throughput: raw_key.len() as f64,
        };

        Ok(ProtocolResult {
            success: security_analysis.security_proof_valid,
            key: if security_analysis.security_proof_valid {
                Some(raw_key)
            } else {
                None
            },
            transmitted_state: None,
            metrics,
            security_analysis,
        })
    }

    /// Execute CV-QKD protocol
    fn execute_cvqkd(
        &mut self,
        context: &ProtocolContext,
        modulation: CVModulation,
    ) -> Result<ProtocolResult, PhotonicProtocolError> {
        let start_time = Instant::now();

        // Simulate CV-QKD protocol
        let key_length = 256;
        let mut key = Vec::new();

        // Generate coherent states for CV-QKD
        for _ in 0..key_length {
            let x_quad = match modulation {
                CVModulation::Gaussian { variance } => {
                    // Generate Gaussian distributed quadrature
                    self.generate_gaussian_random(0.0, variance)
                }
                CVModulation::Discrete { constellation_size } => {
                    // Generate discrete constellation point
                    let point = (rand::random::<f64>() * constellation_size as f64) as usize;
                    (point as f64 - constellation_size as f64 / 2.0) * 0.5
                }
                _ => rand::random::<f64>() - 0.5,
            };

            // Quantize to bits (simplified)
            key.push(if x_quad > 0.0 { 1 } else { 0 });
        }

        let error_rate = 0.02; // Lower error rate for CV-QKD
        let security_analysis = SecurityAnalysis {
            information_leakage: error_rate * 0.3,
            eavesdropping_detected: error_rate > 0.05,
            security_proof_valid: error_rate <= 0.05,
            achieved_security: if error_rate <= 0.05 {
                context.security_params.security_level as f64
            } else {
                0.0
            },
        };

        let metrics = ProtocolMetrics {
            execution_time: start_time.elapsed(),
            key_rate: Some(key.len() as f64 / start_time.elapsed().as_secs_f64()),
            error_rate,
            fidelity: 1.0 - error_rate,
            throughput: key.len() as f64,
        };

        Ok(ProtocolResult {
            success: security_analysis.security_proof_valid,
            key: if security_analysis.security_proof_valid {
                Some(key)
            } else {
                None
            },
            transmitted_state: None,
            metrics,
            security_analysis,
        })
    }

    /// Execute teleportation protocol
    fn execute_teleportation_protocol(
        &self,
        context: &ProtocolContext,
    ) -> Result<ProtocolResult, PhotonicProtocolError> {
        let start_time = Instant::now();

        // Simulate quantum teleportation
        let input_state = PhotonicQubitState::plus(PhotonicQubitEncoding::Polarization);
        let fidelity = 0.95; // Typical teleportation fidelity

        let mut output_state = input_state.clone();
        output_state.amplitude_0 *= (fidelity as f64).sqrt();
        output_state.amplitude_1 *= (fidelity as f64).sqrt();

        let metrics = ProtocolMetrics {
            execution_time: start_time.elapsed(),
            key_rate: None,
            error_rate: 1.0 - fidelity,
            fidelity,
            throughput: 1.0, // One qubit teleported
        };

        let security_analysis = SecurityAnalysis {
            information_leakage: 0.0, // Perfect security in principle
            eavesdropping_detected: false,
            security_proof_valid: true,
            achieved_security: f64::INFINITY, // Information-theoretic security
        };

        Ok(ProtocolResult {
            success: fidelity > 0.9,
            key: None,
            transmitted_state: Some(output_state),
            metrics,
            security_analysis,
        })
    }

    /// Execute other protocols (placeholder implementations)
    fn execute_state_distribution_protocol(
        &self,
        context: &ProtocolContext,
    ) -> Result<ProtocolResult, PhotonicProtocolError> {
        // Placeholder implementation
        Ok(self.create_placeholder_result(context))
    }

    fn execute_clock_sync_protocol(
        &self,
        context: &ProtocolContext,
    ) -> Result<ProtocolResult, PhotonicProtocolError> {
        // Placeholder implementation
        Ok(self.create_placeholder_result(context))
    }

    fn execute_sensing_protocol(
        &self,
        context: &ProtocolContext,
    ) -> Result<ProtocolResult, PhotonicProtocolError> {
        // Placeholder implementation
        Ok(self.create_placeholder_result(context))
    }

    fn execute_quantum_internet_protocol(
        &self,
        context: &ProtocolContext,
        _protocol_version: &str,
    ) -> Result<ProtocolResult, PhotonicProtocolError> {
        // Placeholder implementation
        Ok(self.create_placeholder_result(context))
    }

    /// Create placeholder result for unimplemented protocols
    fn create_placeholder_result(&self, _context: &ProtocolContext) -> ProtocolResult {
        ProtocolResult {
            success: true,
            key: None,
            transmitted_state: None,
            metrics: ProtocolMetrics {
                execution_time: Duration::from_millis(100),
                key_rate: None,
                error_rate: 0.01,
                fidelity: 0.99,
                throughput: 1.0,
            },
            security_analysis: SecurityAnalysis {
                information_leakage: 0.01,
                eavesdropping_detected: false,
                security_proof_valid: true,
                achieved_security: 128.0,
            },
        }
    }

    /// Update protocol statistics
    fn update_statistics(&mut self, result: &ProtocolResult) {
        self.statistics.total_protocols += 1;

        let success_count = if result.success { 1.0 } else { 0.0 };
        self.statistics.success_rate = (self.statistics.success_rate
            * (self.statistics.total_protocols - 1) as f64
            + success_count)
            / self.statistics.total_protocols as f64;

        if let Some(key_rate) = result.metrics.key_rate {
            self.statistics.average_key_rate = (self.statistics.average_key_rate
                * (self.statistics.total_protocols - 1) as f64
                + key_rate)
                / self.statistics.total_protocols as f64;
        }

        self.statistics.average_fidelity = (self.statistics.average_fidelity
            * (self.statistics.total_protocols - 1) as f64
            + result.metrics.fidelity)
            / self.statistics.total_protocols as f64;

        if result.security_analysis.eavesdropping_detected {
            self.statistics.security_violations += 1;
        }
    }

    /// Generate Gaussian random number (simplified)
    fn generate_gaussian_random(&self, mean: f64, variance: f64) -> f64 {
        // Simple Box-Muller transform
        let u1 = rand::random::<f64>();
        let u2 = rand::random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + variance.sqrt() * z
    }

    /// Get protocol statistics
    pub fn get_statistics(&self) -> &ProtocolStatistics {
        &self.statistics
    }

    /// Get active protocols
    pub fn get_active_protocols(&self) -> Vec<String> {
        self.active_protocols.keys().cloned().collect()
    }
}

impl Default for PhotonicProtocolEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_engine_creation() {
        let engine = PhotonicProtocolEngine::new();
        assert_eq!(engine.statistics.total_protocols, 0);
        assert_eq!(engine.active_protocols.len(), 0);
    }

    #[test]
    fn test_bb84_protocol() {
        let mut engine = PhotonicProtocolEngine::new();

        let context = ProtocolContext {
            protocol_id: "test_bb84".to_string(),
            parties: vec![],
            security_params: SecurityParameters {
                security_level: 128,
                error_tolerance: 0.11,
                privacy_amplification: PrivacyAmplificationParams {
                    hash_family: "SHA256".to_string(),
                    compression_ratio: 0.5,
                    rounds: 1,
                },
                authentication: AuthenticationMethod::Classical {
                    algorithm: "HMAC".to_string(),
                },
            },
            network_config: NetworkConfiguration {
                topology: NetworkTopology::PointToPoint,
                channels: vec![],
                routing: RoutingConfig {
                    algorithm: RoutingAlgorithm::ShortestPath,
                    max_hops: 1,
                    load_balancing: false,
                },
            },
            start_time: Instant::now(),
        };

        let result = engine
            .execute_protocol(
                PhotonicProtocolType::QKD {
                    variant: QKDVariant::BB84,
                },
                context,
            )
            .unwrap();

        assert!(result.metrics.fidelity > 0.0);
        assert!(result.metrics.execution_time > Duration::ZERO);
    }

    #[test]
    fn test_teleportation_protocol() {
        let mut engine = PhotonicProtocolEngine::new();

        let context = ProtocolContext {
            protocol_id: "test_teleport".to_string(),
            parties: vec![],
            security_params: SecurityParameters {
                security_level: 128,
                error_tolerance: 0.05,
                privacy_amplification: PrivacyAmplificationParams {
                    hash_family: "SHA256".to_string(),
                    compression_ratio: 0.5,
                    rounds: 1,
                },
                authentication: AuthenticationMethod::Quantum {
                    protocol: "QAUTH".to_string(),
                },
            },
            network_config: NetworkConfiguration {
                topology: NetworkTopology::PointToPoint,
                channels: vec![],
                routing: RoutingConfig {
                    algorithm: RoutingAlgorithm::QuantumAware,
                    max_hops: 1,
                    load_balancing: false,
                },
            },
            start_time: Instant::now(),
        };

        let result = engine
            .execute_protocol(PhotonicProtocolType::Teleportation, context)
            .unwrap();

        assert!(result.success);
        assert!(result.transmitted_state.is_some());
        assert!(result.metrics.fidelity > 0.9);
    }

    #[test]
    fn test_cvqkd_protocol() {
        let mut engine = PhotonicProtocolEngine::new();

        let context = ProtocolContext {
            protocol_id: "test_cvqkd".to_string(),
            parties: vec![],
            security_params: SecurityParameters {
                security_level: 256,
                error_tolerance: 0.05,
                privacy_amplification: PrivacyAmplificationParams {
                    hash_family: "SHA3".to_string(),
                    compression_ratio: 0.7,
                    rounds: 2,
                },
                authentication: AuthenticationMethod::PostQuantum {
                    algorithm: "Dilithium".to_string(),
                },
            },
            network_config: NetworkConfiguration {
                topology: NetworkTopology::PointToPoint,
                channels: vec![],
                routing: RoutingConfig {
                    algorithm: RoutingAlgorithm::MinimumLoss,
                    max_hops: 1,
                    load_balancing: false,
                },
            },
            start_time: Instant::now(),
        };

        let result = engine
            .execute_protocol(
                PhotonicProtocolType::QKD {
                    variant: QKDVariant::CVQKD {
                        modulation: CVModulation::Gaussian { variance: 1.0 },
                    },
                },
                context,
            )
            .unwrap();

        assert!(result.key.is_some());
        assert!(result.metrics.key_rate.is_some());
    }
}
