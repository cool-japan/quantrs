//! Photonic quantum computing device interfaces
//!
//! This module provides support for photonic quantum computers, including
//! continuous variable systems, gate-based photonic systems, and measurement-based
//! quantum computing approaches.

use crate::{CircuitExecutor, CircuitResult, DeviceError, DeviceResult, QuantumDevice};
use quantrs2_circuit::prelude::Circuit;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

pub mod client;
pub mod config;
pub mod continuous_variable;
pub mod cv_gates;
pub mod device;
pub mod encoding;
pub mod error_correction;
pub mod gate_based;
pub mod measurement_based;
pub mod noise_models;
pub mod optimization;
pub mod protocols;
pub mod squeezed_states;

pub use client::PhotonicClient;
pub use config::{PhotonicConfig, PhotonicSystem};
pub use device::PhotonicQuantumDeviceImpl;

/// Types of photonic quantum computing systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhotonicSystemType {
    /// Continuous variable quantum computing
    ContinuousVariable,
    /// Gate-based photonic quantum computing
    GateBased,
    /// Measurement-based quantum computing (one-way quantum computing)
    MeasurementBased,
    /// Hybrid photonic-matter systems
    Hybrid,
}

/// Photonic quantum computing modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhotonicMode {
    /// Position quadrature
    Position,
    /// Momentum quadrature
    Momentum,
    /// Number states (Fock states)
    Number,
    /// Coherent states
    Coherent,
    /// Squeezed states
    Squeezed,
    /// Cat states
    Cat,
}

/// Configuration for photonic quantum devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicDeviceConfig {
    /// Type of photonic system
    pub system_type: PhotonicSystemType,
    /// Number of optical modes
    pub mode_count: usize,
    /// Cutoff dimension for Fock space truncation
    pub cutoff_dimension: Option<usize>,
    /// Squeezing parameter range
    pub squeezing_range: Option<(f64, f64)>,
    /// Loss rate per mode
    pub loss_rate: Option<f64>,
    /// Thermal photon number
    pub thermal_photons: Option<f64>,
    /// Detection efficiency
    pub detection_efficiency: Option<f64>,
    /// Gate fidelity
    pub gate_fidelity: Option<f64>,
    /// Measurement fidelity
    pub measurement_fidelity: Option<f64>,
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Enable hardware acceleration
    pub hardware_acceleration: bool,
    /// Custom hardware parameters
    pub hardware_params: HashMap<String, String>,
}

impl Default for PhotonicDeviceConfig {
    fn default() -> Self {
        Self {
            system_type: PhotonicSystemType::ContinuousVariable,
            mode_count: 8,
            cutoff_dimension: Some(10),
            squeezing_range: Some((-2.0, 2.0)),
            loss_rate: Some(0.01),
            thermal_photons: Some(0.1),
            detection_efficiency: Some(0.9),
            gate_fidelity: Some(0.99),
            measurement_fidelity: Some(0.95),
            max_execution_time: Some(Duration::from_secs(300)),
            hardware_acceleration: true,
            hardware_params: HashMap::new(),
        }
    }
}

/// Result of photonic quantum circuit execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicCircuitResult {
    /// Standard circuit result
    pub circuit_result: CircuitResult,
    /// Photonic-specific results
    pub photonic_data: PhotonicMeasurementData,
    /// Execution metadata
    pub execution_metadata: PhotonicExecutionMetadata,
}

/// Photonic measurement data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhotonicMeasurementData {
    /// Quadrature measurements
    pub quadratures: Vec<(f64, f64)>, // (x, p) pairs
    /// Photon number measurements
    pub photon_numbers: Vec<usize>,
    /// Homodyne detection results
    pub homodyne_results: Vec<f64>,
    /// Heterodyne detection results
    pub heterodyne_results: Vec<(f64, f64)>,
    /// Correlation functions
    pub correlations: HashMap<String, f64>,
    /// Fidelity estimates
    pub fidelities: HashMap<String, f64>,
}

/// Photonic execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicExecutionMetadata {
    /// System type used
    pub system_type: PhotonicSystemType,
    /// Number of modes used
    pub modes_used: usize,
    /// Actual execution time
    pub execution_time: Duration,
    /// Loss rate during execution
    pub measured_loss_rate: Option<f64>,
    /// Thermal noise during execution
    pub thermal_noise: Option<f64>,
    /// Gate sequence applied
    pub gate_sequence: Vec<String>,
    /// Optimization applied
    pub optimizations_applied: Vec<String>,
}

impl Default for PhotonicExecutionMetadata {
    fn default() -> Self {
        Self {
            system_type: PhotonicSystemType::ContinuousVariable,
            modes_used: 0,
            execution_time: Duration::from_millis(0),
            measured_loss_rate: None,
            thermal_noise: None,
            gate_sequence: Vec::new(),
            optimizations_applied: Vec::new(),
        }
    }
}

/// Trait for photonic quantum devices
#[async_trait::async_trait]
pub trait PhotonicQuantumDevice: QuantumDevice + CircuitExecutor {
    /// Get the photonic system type
    async fn system_type(&self) -> DeviceResult<PhotonicSystemType>;

    /// Get the number of optical modes
    async fn mode_count(&self) -> DeviceResult<usize>;

    /// Get the cutoff dimension for Fock space
    async fn cutoff_dimension(&self) -> DeviceResult<Option<usize>>;

    /// Check if continuous variable operations are supported
    async fn supports_cv_operations(&self) -> DeviceResult<bool>;

    /// Check if gate-based operations are supported
    async fn supports_gate_based(&self) -> DeviceResult<bool>;

    /// Check if measurement-based operations are supported
    async fn supports_measurement_based(&self) -> DeviceResult<bool>;

    /// Get supported quadrature measurement precision
    async fn quadrature_precision(&self) -> DeviceResult<f64>;

    /// Get photon detection efficiency
    async fn detection_efficiency(&self) -> DeviceResult<f64>;

    /// Execute a photonic circuit with detailed results
    async fn execute_photonic_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        shots: usize,
        config: Option<PhotonicDeviceConfig>,
    ) -> DeviceResult<PhotonicCircuitResult>;

    /// Perform quadrature measurements
    async fn measure_quadratures(
        &self,
        modes: &[usize],
        angles: &[f64],
    ) -> DeviceResult<Vec<(f64, f64)>>;

    /// Perform photon number measurements
    async fn measure_photon_numbers(&self, modes: &[usize]) -> DeviceResult<Vec<usize>>;

    /// Perform homodyne detection
    async fn homodyne_detection(
        &self,
        mode: usize,
        phase: f64,
        shots: usize,
    ) -> DeviceResult<Vec<f64>>;

    /// Perform heterodyne detection
    async fn heterodyne_detection(
        &self,
        mode: usize,
        shots: usize,
    ) -> DeviceResult<Vec<(f64, f64)>>;

    /// Calculate correlation functions
    async fn calculate_correlations(
        &self,
        modes: &[(usize, usize)],
        correlation_type: &str,
    ) -> DeviceResult<HashMap<String, f64>>;

    /// Estimate state fidelity
    async fn estimate_fidelity(
        &self,
        target_state: &str,
        measurement_data: &PhotonicMeasurementData,
    ) -> DeviceResult<f64>;
}

/// Create a photonic quantum device
pub async fn create_photonic_device(
    client: PhotonicClient,
    config: PhotonicDeviceConfig,
) -> DeviceResult<impl PhotonicQuantumDevice> {
    PhotonicQuantumDeviceImpl::new("default_photonic_device".to_string(), client, config).await
}

/// Validate photonic device configuration
pub fn validate_photonic_config(config: &PhotonicDeviceConfig) -> DeviceResult<()> {
    if config.mode_count == 0 {
        return Err(DeviceError::InvalidInput(
            "Mode count must be greater than 0".to_string(),
        ));
    }

    if let Some(cutoff) = config.cutoff_dimension {
        if cutoff == 0 {
            return Err(DeviceError::InvalidInput(
                "Cutoff dimension must be greater than 0".to_string(),
            ));
        }
    }

    if let Some((min_squeeze, max_squeeze)) = config.squeezing_range {
        if min_squeeze >= max_squeeze {
            return Err(DeviceError::InvalidInput(
                "Invalid squeezing range: min must be less than max".to_string(),
            ));
        }
    }

    if let Some(loss_rate) = config.loss_rate {
        if !(0.0..=1.0).contains(&loss_rate) {
            return Err(DeviceError::InvalidInput(
                "Loss rate must be between 0 and 1".to_string(),
            ));
        }
    }

    if let Some(efficiency) = config.detection_efficiency {
        if !(0.0..=1.0).contains(&efficiency) {
            return Err(DeviceError::InvalidInput(
                "Detection efficiency must be between 0 and 1".to_string(),
            ));
        }
    }

    Ok(())
}

/// Photonic gate operations
pub mod gates {
    use super::*;

    /// Displacement gate parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DisplacementGate {
        pub alpha: f64,  // Displacement amplitude
        pub phi: f64,    // Displacement phase
        pub mode: usize, // Target mode
    }

    /// Squeezing gate parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SqueezingGate {
        pub r: f64,      // Squeezing parameter
        pub phi: f64,    // Squeezing angle
        pub mode: usize, // Target mode
    }

    /// Two-mode squeezing gate parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TwoModeSqueezingGate {
        pub r: f64,       // Squeezing parameter
        pub phi: f64,     // Squeezing phase
        pub mode1: usize, // First mode
        pub mode2: usize, // Second mode
    }

    /// Beamsplitter gate parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BeamsplitterGate {
        pub theta: f64,   // Transmission angle
        pub phi: f64,     // Phase
        pub mode1: usize, // First mode
        pub mode2: usize, // Second mode
    }

    /// Phase rotation gate parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PhaseRotationGate {
        pub phi: f64,    // Rotation angle
        pub mode: usize, // Target mode
    }

    /// Kerr gate parameters (non-linear)
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct KerrGate {
        pub kappa: f64,  // Kerr parameter
        pub mode: usize, // Target mode
    }

    /// Cross-Kerr gate parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CrossKerrGate {
        pub kappa: f64,   // Cross-Kerr parameter
        pub mode1: usize, // First mode
        pub mode2: usize, // Second mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_photonic_config_validation() {
        let valid_config = PhotonicDeviceConfig::default();
        assert!(validate_photonic_config(&valid_config).is_ok());

        let invalid_config = PhotonicDeviceConfig {
            mode_count: 0,
            ..Default::default()
        };
        assert!(validate_photonic_config(&invalid_config).is_err());
    }

    #[test]
    fn test_photonic_system_types() {
        let cv_system = PhotonicSystemType::ContinuousVariable;
        assert_eq!(cv_system, PhotonicSystemType::ContinuousVariable);

        let gb_system = PhotonicSystemType::GateBased;
        assert_eq!(gb_system, PhotonicSystemType::GateBased);
    }

    #[test]
    fn test_photonic_modes() {
        let position_mode = PhotonicMode::Position;
        assert_eq!(position_mode, PhotonicMode::Position);

        let squeezed_mode = PhotonicMode::Squeezed;
        assert_eq!(squeezed_mode, PhotonicMode::Squeezed);
    }
}
