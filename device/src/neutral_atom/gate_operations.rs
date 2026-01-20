//! Gate operations for neutral atom quantum devices
//!
//! This module provides implementations of quantum gate operations
//! specific to neutral atom systems, including Rydberg gates,
//! optical tweezer manipulations, and hyperfine state operations.

use crate::{DeviceError, DeviceResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Gate operation types for neutral atoms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeutralAtomGateType {
    /// Single-qubit rotation gates
    SingleQubitRotation,
    /// Rydberg excitation gate
    RydbergExcitation,
    /// Rydberg blockade gate
    RydbergBlockade,
    /// Global Rydberg gate
    GlobalRydberg,
    /// Optical tweezer movement
    TweezerMovement,
    /// Hyperfine state manipulation
    HyperfineManipulation,
    /// Measurement operation
    Measurement,
}

/// Parameters for neutral atom gate operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutralAtomGateParams {
    /// Gate type
    pub gate_type: NeutralAtomGateType,
    /// Target atom indices
    pub target_atoms: Vec<usize>,
    /// Gate duration
    pub duration: Duration,
    /// Gate-specific parameters
    pub parameters: HashMap<String, f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Single-qubit rotation gate parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleQubitRotationParams {
    /// Rotation angle (radians)
    pub angle: f64,
    /// Rotation axis
    pub axis: RotationAxis,
    /// Laser power (mW)
    pub laser_power: f64,
    /// Pulse duration
    pub pulse_duration: Duration,
}

/// Rotation axis for single-qubit gates
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RotationAxis {
    /// X-axis rotation
    X,
    /// Y-axis rotation
    Y,
    /// Z-axis rotation
    Z,
    /// Arbitrary axis
    Arbitrary { x: f64, y: f64, z: f64 },
}

/// Rydberg excitation gate parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RydbergExcitationParams {
    /// Target atom index
    pub atom_index: usize,
    /// Excitation time
    pub excitation_time: Duration,
    /// Laser power (mW)
    pub laser_power: f64,
    /// Detuning from resonance (MHz)
    pub detuning: f64,
    /// Rabi frequency (MHz)
    pub rabi_frequency: f64,
}

/// Rydberg blockade gate parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RydbergBlockadeParams {
    /// Control atom index
    pub control_atom: usize,
    /// Target atom index
    pub target_atom: usize,
    /// Blockade strength (MHz)
    pub blockade_strength: f64,
    /// Interaction time
    pub interaction_time: Duration,
    /// Gate phase
    pub phase: f64,
}

/// Global Rydberg operation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalRydbergParams {
    /// Operation type
    pub operation_type: String,
    /// Laser power (mW)
    pub laser_power: f64,
    /// Pulse duration
    pub pulse_duration: Duration,
    /// Phase (radians)
    pub phase: f64,
    /// Frequency sweep parameters
    pub frequency_sweep: Option<FrequencySweepParams>,
}

/// Frequency sweep parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencySweepParams {
    /// Start frequency (MHz)
    pub start_frequency: f64,
    /// End frequency (MHz)
    pub end_frequency: f64,
    /// Sweep rate (MHz/μs)
    pub sweep_rate: f64,
}

/// Optical tweezer movement parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TweezerMovementParams {
    /// Atom index to move
    pub atom_index: usize,
    /// Start position (x, y, z) in micrometers
    pub start_position: (f64, f64, f64),
    /// End position (x, y, z) in micrometers
    pub end_position: (f64, f64, f64),
    /// Movement time
    pub movement_time: Duration,
    /// Movement trajectory
    pub trajectory: MovementTrajectory,
}

/// Movement trajectory types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MovementTrajectory {
    /// Linear movement
    Linear,
    /// Smooth acceleration/deceleration
    Smooth,
    /// Custom trajectory with waypoints
    Custom(Vec<(f64, f64, f64)>),
}

/// Hyperfine state manipulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperfineManipulationParams {
    /// Target atom index
    pub atom_index: usize,
    /// Initial hyperfine state
    pub initial_state: String,
    /// Target hyperfine state
    pub target_state: String,
    /// Microwave frequency (MHz)
    pub microwave_frequency: f64,
    /// Pulse duration
    pub pulse_duration: Duration,
    /// Pulse power (dBm)
    pub pulse_power: f64,
}

/// Measurement operation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementParams {
    /// Atoms to measure
    pub target_atoms: Vec<usize>,
    /// Measurement type
    pub measurement_type: MeasurementType,
    /// Integration time
    pub integration_time: Duration,
    /// Measurement basis
    pub basis: MeasurementBasis,
}

/// Measurement types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementType {
    /// Standard state detection
    StateDetection,
    /// Fluorescence measurement
    Fluorescence,
    /// Ionization measurement
    Ionization,
    /// Correlation measurement
    Correlation,
}

/// Measurement basis
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementBasis {
    /// Computational basis (|0⟩, |1⟩)
    Computational,
    /// X basis (|+⟩, |-⟩)
    X,
    /// Y basis (|+i⟩, |-i⟩)
    Y,
    /// Custom basis
    Custom(String),
}

/// Gate operation builder
pub struct NeutralAtomGateBuilder {
    gate_type: Option<NeutralAtomGateType>,
    target_atoms: Vec<usize>,
    duration: Option<Duration>,
    parameters: HashMap<String, f64>,
    metadata: HashMap<String, String>,
}

impl NeutralAtomGateBuilder {
    /// Create a new gate builder
    pub fn new() -> Self {
        Self {
            gate_type: None,
            target_atoms: Vec::new(),
            duration: None,
            parameters: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the gate type
    #[must_use]
    pub const fn gate_type(mut self, gate_type: NeutralAtomGateType) -> Self {
        self.gate_type = Some(gate_type);
        self
    }

    /// Add target atoms
    #[must_use]
    pub fn target_atoms(mut self, atoms: &[usize]) -> Self {
        self.target_atoms.extend_from_slice(atoms);
        self
    }

    /// Set gate duration
    #[must_use]
    pub const fn duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Add a parameter
    #[must_use]
    pub fn parameter(mut self, key: &str, value: f64) -> Self {
        self.parameters.insert(key.to_string(), value);
        self
    }

    /// Add metadata
    #[must_use]
    pub fn metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Build the gate parameters
    pub fn build(self) -> DeviceResult<NeutralAtomGateParams> {
        let gate_type = self
            .gate_type
            .ok_or_else(|| DeviceError::InvalidInput("Gate type must be specified".to_string()))?;

        let duration = self.duration.ok_or_else(|| {
            DeviceError::InvalidInput("Gate duration must be specified".to_string())
        })?;

        if self.target_atoms.is_empty() {
            return Err(DeviceError::InvalidInput(
                "At least one target atom must be specified".to_string(),
            ));
        }

        Ok(NeutralAtomGateParams {
            gate_type,
            target_atoms: self.target_atoms,
            duration,
            parameters: self.parameters,
            metadata: self.metadata,
        })
    }
}

impl Default for NeutralAtomGateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate gate operation parameters
pub fn validate_gate_params(params: &NeutralAtomGateParams) -> DeviceResult<()> {
    // Check that target atoms are valid
    if params.target_atoms.is_empty() {
        return Err(DeviceError::InvalidInput(
            "Gate operation must have at least one target atom".to_string(),
        ));
    }

    // Check duration is positive
    if params.duration.is_zero() {
        return Err(DeviceError::InvalidInput(
            "Gate duration must be positive".to_string(),
        ));
    }

    // Validate gate-specific parameters
    match params.gate_type {
        NeutralAtomGateType::SingleQubitRotation => {
            validate_single_qubit_rotation_params(params)?;
        }
        NeutralAtomGateType::RydbergExcitation => {
            validate_rydberg_excitation_params(params)?;
        }
        NeutralAtomGateType::RydbergBlockade => {
            validate_rydberg_blockade_params(params)?;
        }
        NeutralAtomGateType::TweezerMovement => {
            validate_tweezer_movement_params(params)?;
        }
        _ => {} // Other gate types have their own validation
    }

    Ok(())
}

fn validate_single_qubit_rotation_params(params: &NeutralAtomGateParams) -> DeviceResult<()> {
    if params.target_atoms.len() != 1 {
        return Err(DeviceError::InvalidInput(
            "Single-qubit rotation must target exactly one atom".to_string(),
        ));
    }

    if !params.parameters.contains_key("angle") {
        return Err(DeviceError::InvalidInput(
            "Single-qubit rotation must specify rotation angle".to_string(),
        ));
    }

    Ok(())
}

fn validate_rydberg_excitation_params(params: &NeutralAtomGateParams) -> DeviceResult<()> {
    if params.target_atoms.len() != 1 {
        return Err(DeviceError::InvalidInput(
            "Rydberg excitation must target exactly one atom".to_string(),
        ));
    }

    if !params.parameters.contains_key("laser_power") {
        return Err(DeviceError::InvalidInput(
            "Rydberg excitation must specify laser power".to_string(),
        ));
    }

    Ok(())
}

fn validate_rydberg_blockade_params(params: &NeutralAtomGateParams) -> DeviceResult<()> {
    if params.target_atoms.len() != 2 {
        return Err(DeviceError::InvalidInput(
            "Rydberg blockade must target exactly two atoms".to_string(),
        ));
    }

    if !params.parameters.contains_key("blockade_strength") {
        return Err(DeviceError::InvalidInput(
            "Rydberg blockade must specify blockade strength".to_string(),
        ));
    }

    Ok(())
}

fn validate_tweezer_movement_params(params: &NeutralAtomGateParams) -> DeviceResult<()> {
    if params.target_atoms.len() != 1 {
        return Err(DeviceError::InvalidInput(
            "Tweezer movement must target exactly one atom".to_string(),
        ));
    }

    let required_params = ["start_x", "start_y", "start_z", "end_x", "end_y", "end_z"];
    for param in &required_params {
        if !params.parameters.contains_key(*param) {
            return Err(DeviceError::InvalidInput(format!(
                "Tweezer movement must specify {param}"
            )));
        }
    }

    Ok(())
}

/// Create a single-qubit rotation gate
pub fn create_single_qubit_rotation(
    atom_index: usize,
    angle: f64,
    axis: RotationAxis,
    duration: Duration,
) -> DeviceResult<NeutralAtomGateParams> {
    let axis_str = match axis {
        RotationAxis::X => "x".to_string(),
        RotationAxis::Y => "y".to_string(),
        RotationAxis::Z => "z".to_string(),
        RotationAxis::Arbitrary { x, y, z } => format!("arbitrary_{x}_{y}_{z}"),
    };

    NeutralAtomGateBuilder::new()
        .gate_type(NeutralAtomGateType::SingleQubitRotation)
        .target_atoms(&[atom_index])
        .duration(duration)
        .parameter("angle", angle)
        .metadata("axis", &axis_str)
        .build()
}

/// Create a Rydberg excitation gate
pub fn create_rydberg_excitation(
    atom_index: usize,
    laser_power: f64,
    excitation_time: Duration,
    detuning: f64,
) -> DeviceResult<NeutralAtomGateParams> {
    NeutralAtomGateBuilder::new()
        .gate_type(NeutralAtomGateType::RydbergExcitation)
        .target_atoms(&[atom_index])
        .duration(excitation_time)
        .parameter("laser_power", laser_power)
        .parameter("detuning", detuning)
        .build()
}

/// Create a Rydberg blockade gate
pub fn create_rydberg_blockade(
    control_atom: usize,
    target_atom: usize,
    blockade_strength: f64,
    interaction_time: Duration,
) -> DeviceResult<NeutralAtomGateParams> {
    NeutralAtomGateBuilder::new()
        .gate_type(NeutralAtomGateType::RydbergBlockade)
        .target_atoms(&[control_atom, target_atom])
        .duration(interaction_time)
        .parameter("blockade_strength", blockade_strength)
        .parameter("control_atom", control_atom as f64)
        .parameter("target_atom", target_atom as f64)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_builder() {
        let gate = NeutralAtomGateBuilder::new()
            .gate_type(NeutralAtomGateType::SingleQubitRotation)
            .target_atoms(&[0])
            .duration(Duration::from_micros(100))
            .parameter("angle", std::f64::consts::PI)
            .build()
            .expect("Failed to build single-qubit rotation gate");

        assert_eq!(gate.gate_type, NeutralAtomGateType::SingleQubitRotation);
        assert_eq!(gate.target_atoms, vec![0]);
        assert_eq!(gate.duration, Duration::from_micros(100));
        assert_eq!(gate.parameters.get("angle"), Some(&std::f64::consts::PI));
    }

    #[test]
    fn test_gate_validation() {
        let gate = NeutralAtomGateParams {
            gate_type: NeutralAtomGateType::SingleQubitRotation,
            target_atoms: vec![0],
            duration: Duration::from_micros(100),
            parameters: [("angle".to_string(), std::f64::consts::PI)].into(),
            metadata: HashMap::new(),
        };

        assert!(validate_gate_params(&gate).is_ok());
    }

    #[test]
    fn test_create_single_qubit_rotation() {
        let gate = create_single_qubit_rotation(
            0,
            std::f64::consts::PI,
            RotationAxis::X,
            Duration::from_micros(100),
        )
        .expect("Failed to create single-qubit rotation gate");

        assert_eq!(gate.gate_type, NeutralAtomGateType::SingleQubitRotation);
        assert_eq!(gate.target_atoms, vec![0]);
        assert_eq!(gate.parameters.get("angle"), Some(&std::f64::consts::PI));
    }
}
