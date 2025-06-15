//! Neutral atom quantum computing device interfaces
//!
//! This module provides support for neutral atom quantum computers, including
//! Rydberg atom systems, optical tweezer arrays, and neutral atom gate operations.

use crate::{CircuitExecutor, CircuitResult, DeviceError, DeviceResult, QuantumDevice};
use quantrs2_circuit::prelude::Circuit;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

pub mod client;
pub mod config;
pub mod device;
pub mod gate_operations;
pub mod protocols;
pub mod rydberg;
pub mod tweezer_arrays;

pub use client::NeutralAtomClient;
pub use device::NeutralAtomDevice;

/// Types of neutral atom quantum computing systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeutralAtomSystemType {
    /// Rydberg atom systems
    Rydberg,
    /// Optical tweezer arrays
    OpticalTweezer,
    /// Magnetic trap arrays
    MagneticTrap,
    /// Hybrid neutral atom systems
    Hybrid,
}

/// Neutral atom state encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AtomStateEncoding {
    /// Ground and excited states
    GroundExcited,
    /// Hyperfine states
    Hyperfine,
    /// Clock states
    Clock,
    /// Zeeman states
    Zeeman,
}

/// Configuration for neutral atom quantum devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutralAtomDeviceConfig {
    /// Type of neutral atom system
    pub system_type: NeutralAtomSystemType,
    /// Number of atoms in the array
    pub atom_count: usize,
    /// Atom spacing in micrometers
    pub atom_spacing: f64,
    /// State encoding scheme
    pub state_encoding: AtomStateEncoding,
    /// Rydberg blockade radius in micrometers
    pub blockade_radius: Option<f64>,
    /// Laser wavelength in nanometers
    pub laser_wavelength: Option<f64>,
    /// Trap depth in microkelvin
    pub trap_depth: Option<f64>,
    /// Gate fidelity
    pub gate_fidelity: Option<f64>,
    /// Measurement fidelity
    pub measurement_fidelity: Option<f64>,
    /// Loading efficiency
    pub loading_efficiency: Option<f64>,
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Enable hardware acceleration
    pub hardware_acceleration: bool,
    /// Custom hardware parameters
    pub hardware_params: HashMap<String, String>,
}

impl Default for NeutralAtomDeviceConfig {
    fn default() -> Self {
        Self {
            system_type: NeutralAtomSystemType::Rydberg,
            atom_count: 100,
            atom_spacing: 5.0,
            state_encoding: AtomStateEncoding::GroundExcited,
            blockade_radius: Some(8.0),
            laser_wavelength: Some(480.0),
            trap_depth: Some(1000.0),
            gate_fidelity: Some(0.995),
            measurement_fidelity: Some(0.99),
            loading_efficiency: Some(0.95),
            max_execution_time: Some(Duration::from_secs(60)),
            hardware_acceleration: true,
            hardware_params: HashMap::new(),
        }
    }
}

/// Result of neutral atom quantum circuit execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutralAtomCircuitResult {
    /// Standard circuit result
    pub circuit_result: CircuitResult,
    /// Neutral atom-specific results
    pub neutral_atom_data: NeutralAtomMeasurementData,
    /// Execution metadata
    pub execution_metadata: NeutralAtomExecutionMetadata,
}

/// Neutral atom measurement data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutralAtomMeasurementData {
    /// Atom positions in the array
    pub atom_positions: Vec<(f64, f64, f64)>,
    /// Atom state measurements
    pub atom_states: Vec<String>,
    /// Rydberg excitation patterns
    pub rydberg_patterns: Vec<Vec<bool>>,
    /// Correlation measurements
    pub correlations: HashMap<String, f64>,
    /// Fidelity estimates
    pub fidelities: HashMap<String, f64>,
    /// Loading success rates
    pub loading_success: Vec<bool>,
}

/// Neutral atom execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutralAtomExecutionMetadata {
    /// System type used
    pub system_type: NeutralAtomSystemType,
    /// Number of atoms used
    pub atoms_used: usize,
    /// Actual execution time
    pub execution_time: Duration,
    /// Gate sequence applied
    pub gate_sequence: Vec<String>,
    /// Optimization applied
    pub optimizations_applied: Vec<String>,
    /// Temperature during execution
    pub temperature: Option<f64>,
    /// Laser power used
    pub laser_power: Option<f64>,
}

impl Default for NeutralAtomMeasurementData {
    fn default() -> Self {
        Self {
            atom_positions: Vec::new(),
            atom_states: Vec::new(),
            rydberg_patterns: Vec::new(),
            correlations: HashMap::new(),
            fidelities: HashMap::new(),
            loading_success: Vec::new(),
        }
    }
}

impl Default for NeutralAtomExecutionMetadata {
    fn default() -> Self {
        Self {
            system_type: NeutralAtomSystemType::Rydberg,
            atoms_used: 0,
            execution_time: Duration::from_millis(0),
            gate_sequence: Vec::new(),
            optimizations_applied: Vec::new(),
            temperature: None,
            laser_power: None,
        }
    }
}

/// Trait for neutral atom quantum devices
#[async_trait::async_trait]
pub trait NeutralAtomQuantumDevice: QuantumDevice + CircuitExecutor {
    /// Get the neutral atom system type
    async fn system_type(&self) -> DeviceResult<NeutralAtomSystemType>;

    /// Get the number of atoms in the array
    async fn atom_count(&self) -> DeviceResult<usize>;

    /// Get the atom spacing
    async fn atom_spacing(&self) -> DeviceResult<f64>;

    /// Get the state encoding scheme
    async fn state_encoding(&self) -> DeviceResult<AtomStateEncoding>;

    /// Get the Rydberg blockade radius
    async fn blockade_radius(&self) -> DeviceResult<Option<f64>>;

    /// Check if Rydberg gates are supported
    async fn supports_rydberg_gates(&self) -> DeviceResult<bool>;

    /// Check if optical tweezer manipulation is supported
    async fn supports_tweezer_manipulation(&self) -> DeviceResult<bool>;

    /// Get loading efficiency
    async fn loading_efficiency(&self) -> DeviceResult<f64>;

    /// Get gate fidelity
    async fn gate_fidelity(&self) -> DeviceResult<f64>;

    /// Execute a neutral atom circuit with detailed results
    async fn execute_neutral_atom_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        shots: usize,
        config: Option<NeutralAtomDeviceConfig>,
    ) -> DeviceResult<NeutralAtomCircuitResult>;

    /// Load atoms into the trap array
    async fn load_atoms(&self, positions: &[(f64, f64, f64)]) -> DeviceResult<Vec<bool>>;

    /// Move atoms using optical tweezers
    async fn move_atoms(
        &self,
        atom_indices: &[usize],
        new_positions: &[(f64, f64, f64)],
    ) -> DeviceResult<()>;

    /// Perform Rydberg excitation
    async fn rydberg_excitation(
        &self,
        atom_indices: &[usize],
        excitation_time: Duration,
        laser_power: f64,
    ) -> DeviceResult<Vec<bool>>;

    /// Perform global Rydberg operations
    async fn global_rydberg_operation(
        &self,
        operation: &str,
        parameters: &HashMap<String, f64>,
    ) -> DeviceResult<()>;

    /// Measure atom states
    async fn measure_atom_states(&self, atom_indices: &[usize]) -> DeviceResult<Vec<String>>;

    /// Calculate atom correlations
    async fn calculate_atom_correlations(
        &self,
        atom_pairs: &[(usize, usize)],
        correlation_type: &str,
    ) -> DeviceResult<HashMap<String, f64>>;

    /// Estimate state fidelity
    async fn estimate_fidelity(
        &self,
        target_state: &str,
        measurement_data: &NeutralAtomMeasurementData,
    ) -> DeviceResult<f64>;
}

/// Create a neutral atom quantum device
pub fn create_neutral_atom_device(
    client: NeutralAtomClient,
    device_id: String,
    config: NeutralAtomDeviceConfig,
) -> DeviceResult<NeutralAtomDevice> {
    Ok(NeutralAtomDevice::new(client, device_id, config))
}

/// Validate neutral atom device configuration
pub fn validate_neutral_atom_config(config: &NeutralAtomDeviceConfig) -> DeviceResult<()> {
    if config.atom_count == 0 {
        return Err(DeviceError::InvalidInput(
            "Atom count must be greater than 0".to_string(),
        ));
    }

    if config.atom_spacing <= 0.0 {
        return Err(DeviceError::InvalidInput(
            "Atom spacing must be positive".to_string(),
        ));
    }

    if let Some(blockade_radius) = config.blockade_radius {
        if blockade_radius <= 0.0 {
            return Err(DeviceError::InvalidInput(
                "Blockade radius must be positive".to_string(),
            ));
        }
    }

    if let Some(fidelity) = config.gate_fidelity {
        if fidelity < 0.0 || fidelity > 1.0 {
            return Err(DeviceError::InvalidInput(
                "Gate fidelity must be between 0 and 1".to_string(),
            ));
        }
    }

    if let Some(efficiency) = config.loading_efficiency {
        if efficiency < 0.0 || efficiency > 1.0 {
            return Err(DeviceError::InvalidInput(
                "Loading efficiency must be between 0 and 1".to_string(),
            ));
        }
    }

    Ok(())
}

/// Neutral atom gate operations
pub mod gates {
    use super::*;

    /// Rydberg excitation gate parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RydbergExcitationGate {
        pub atom_index: usize,
        pub excitation_time: Duration,
        pub laser_power: f64,
        pub detuning: f64,
    }

    /// Rydberg blockade gate parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RydbergBlockadeGate {
        pub control_atom: usize,
        pub target_atom: usize,
        pub blockade_strength: f64,
        pub interaction_time: Duration,
    }

    /// Global Rydberg gate parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GlobalRydbergGate {
        pub operation_type: String,
        pub laser_power: f64,
        pub pulse_duration: Duration,
        pub phase: f64,
    }

    /// Optical tweezer movement parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TweezerMovementGate {
        pub atom_index: usize,
        pub start_position: (f64, f64, f64),
        pub end_position: (f64, f64, f64),
        pub movement_time: Duration,
    }

    /// Hyperfine state manipulation parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HyperfineGate {
        pub atom_index: usize,
        pub target_state: String,
        pub microwave_frequency: f64,
        pub pulse_duration: Duration,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_atom_config_validation() {
        let valid_config = NeutralAtomDeviceConfig::default();
        assert!(validate_neutral_atom_config(&valid_config).is_ok());

        let invalid_config = NeutralAtomDeviceConfig {
            atom_count: 0,
            ..Default::default()
        };
        assert!(validate_neutral_atom_config(&invalid_config).is_err());
    }

    #[test]
    fn test_neutral_atom_system_types() {
        let rydberg_system = NeutralAtomSystemType::Rydberg;
        assert_eq!(rydberg_system, NeutralAtomSystemType::Rydberg);

        let tweezer_system = NeutralAtomSystemType::OpticalTweezer;
        assert_eq!(tweezer_system, NeutralAtomSystemType::OpticalTweezer);
    }

    #[test]
    fn test_atom_state_encoding() {
        let ground_excited = AtomStateEncoding::GroundExcited;
        assert_eq!(ground_excited, AtomStateEncoding::GroundExcited);

        let hyperfine = AtomStateEncoding::Hyperfine;
        assert_eq!(hyperfine, AtomStateEncoding::Hyperfine);
    }
}