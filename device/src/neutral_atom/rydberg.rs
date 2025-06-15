//! Rydberg atom quantum computing implementation
//!
//! This module provides implementations for Rydberg atom quantum computing,
//! including Rydberg excitation, blockade interactions, and global operations.

use crate::{DeviceError, DeviceResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Rydberg atom configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RydbergConfig {
    /// Principal quantum number for Rydberg state
    pub principal_quantum_number: u32,
    /// Rydberg excitation wavelength (nm)
    pub excitation_wavelength: f64,
    /// Blockade radius (μm)
    pub blockade_radius: f64,
    /// Maximum Rabi frequency (MHz)
    pub max_rabi_frequency: f64,
    /// Laser linewidth (kHz)
    pub laser_linewidth: f64,
    /// Atom spacing (μm)
    pub atom_spacing: f64,
    /// Temperature (nK)
    pub temperature: f64,
}

/// Rydberg quantum states
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RydbergState {
    /// Ground state
    Ground,
    /// Rydberg excited state
    Rydberg,
    /// Superposition state
    Superposition { amplitude_ground: f64, amplitude_rydberg: f64 },
}

/// Rydberg gate types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RydbergGateType {
    /// Single-atom excitation
    SingleExcitation,
    /// Controlled-Z gate using blockade
    ControlledZ,
    /// Controlled-Phase gate
    ControlledPhase,
    /// Multi-atom excitation
    MultiExcitation,
    /// Global rotation
    GlobalRotation,
    /// Adiabatic passage
    AdiabaticPassage,
}

/// Rydberg gate parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RydbergGateParams {
    /// Gate type
    pub gate_type: RydbergGateType,
    /// Target atoms
    pub target_atoms: Vec<usize>,
    /// Laser parameters
    pub laser_params: RydbergLaserParams,
    /// Timing parameters
    pub timing: RydbergTimingParams,
    /// Additional parameters
    pub additional_params: HashMap<String, f64>,
}

/// Rydberg laser parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RydbergLaserParams {
    /// Rabi frequency (MHz)
    pub rabi_frequency: f64,
    /// Detuning from resonance (MHz)
    pub detuning: f64,
    /// Laser power (mW)
    pub power: f64,
    /// Beam waist (μm)
    pub beam_waist: f64,
    /// Polarization
    pub polarization: LaserPolarization,
    /// Phase (radians)
    pub phase: f64,
}

/// Laser polarization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LaserPolarization {
    /// Linear polarization
    Linear,
    /// Circular left polarization
    CircularLeft,
    /// Circular right polarization
    CircularRight,
    /// Elliptical polarization
    Elliptical { ratio: f64, angle: f64 },
}

/// Rydberg timing parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RydbergTimingParams {
    /// Pulse duration
    pub pulse_duration: Duration,
    /// Rise time
    pub rise_time: Duration,
    /// Fall time
    pub fall_time: Duration,
    /// Wait time after pulse
    pub wait_time: Duration,
}

/// Rydberg interaction types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RydbergInteractionType {
    /// Van der Waals interaction
    VanDerWaals,
    /// Dipole-dipole interaction
    DipoleDipole,
    /// Förster resonance
    Forster,
    /// Long-range interaction
    LongRange,
}

/// Rydberg interaction parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RydbergInteraction {
    /// Interaction type
    pub interaction_type: RydbergInteractionType,
    /// Interaction strength (MHz)
    pub strength: f64,
    /// Range (μm)
    pub range: f64,
    /// Atom pairs involved
    pub atom_pairs: Vec<(usize, usize)>,
}

/// Rydberg pulse sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RydbergPulseSequence {
    /// Sequence name
    pub name: String,
    /// Individual pulses
    pub pulses: Vec<RydbergPulse>,
    /// Total sequence duration
    pub total_duration: Duration,
    /// Repetitions
    pub repetitions: usize,
}

/// Individual Rydberg pulse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RydbergPulse {
    /// Pulse ID
    pub pulse_id: String,
    /// Start time
    pub start_time: Duration,
    /// Pulse parameters
    pub parameters: RydbergGateParams,
    /// Pulse shape
    pub pulse_shape: PulseShape,
}

/// Pulse shapes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PulseShape {
    /// Rectangular pulse
    Rectangular,
    /// Gaussian pulse
    Gaussian,
    /// Blackman pulse
    Blackman,
    /// Hanning pulse
    Hanning,
    /// Custom pulse shape
    Custom(String),
}

/// Rydberg system simulator
pub struct RydbergSimulator {
    config: RydbergConfig,
    atom_states: Vec<RydbergState>,
    interactions: Vec<RydbergInteraction>,
}

impl RydbergSimulator {
    /// Create a new Rydberg simulator
    pub fn new(config: RydbergConfig, num_atoms: usize) -> Self {
        let atom_states = vec![RydbergState::Ground; num_atoms];
        
        Self {
            config,
            atom_states,
            interactions: Vec::new(),
        }
    }

    /// Add interaction between atoms
    pub fn add_interaction(&mut self, interaction: RydbergInteraction) {
        self.interactions.push(interaction);
    }

    /// Apply a Rydberg gate
    pub fn apply_gate(&mut self, gate_params: &RydbergGateParams) -> DeviceResult<()> {
        match gate_params.gate_type {
            RydbergGateType::SingleExcitation => {
                self.apply_single_excitation(gate_params)?;
            }
            RydbergGateType::ControlledZ => {
                self.apply_controlled_z(gate_params)?;
            }
            RydbergGateType::GlobalRotation => {
                self.apply_global_rotation(gate_params)?;
            }
            _ => {
                return Err(DeviceError::UnsupportedOperation(format!(
                    "Gate type {:?} not yet implemented",
                    gate_params.gate_type
                )));
            }
        }

        Ok(())
    }

    /// Apply single atom excitation
    fn apply_single_excitation(&mut self, gate_params: &RydbergGateParams) -> DeviceResult<()> {
        if gate_params.target_atoms.len() != 1 {
            return Err(DeviceError::InvalidInput(
                "Single excitation requires exactly one target atom".to_string(),
            ));
        }

        let atom_index = gate_params.target_atoms[0];
        if atom_index >= self.atom_states.len() {
            return Err(DeviceError::InvalidInput(format!(
                "Atom index {} out of bounds",
                atom_index
            )));
        }

        // Calculate excitation probability based on laser parameters
        let rabi_freq = gate_params.laser_params.rabi_frequency;
        let duration = gate_params.timing.pulse_duration.as_secs_f64() * 1e6; // Convert to μs
        let excitation_prob = (rabi_freq * duration * std::f64::consts::PI / 2.0).sin().powi(2);

        // Apply excitation (simplified model)
        if excitation_prob > 0.5 {
            self.atom_states[atom_index] = RydbergState::Rydberg;
        } else {
            self.atom_states[atom_index] = RydbergState::Superposition {
                amplitude_ground: (1.0 - excitation_prob).sqrt(),
                amplitude_rydberg: excitation_prob.sqrt(),
            };
        }

        Ok(())
    }

    /// Apply controlled-Z gate using blockade
    fn apply_controlled_z(&mut self, gate_params: &RydbergGateParams) -> DeviceResult<()> {
        if gate_params.target_atoms.len() != 2 {
            return Err(DeviceError::InvalidInput(
                "Controlled-Z gate requires exactly two target atoms".to_string(),
            ));
        }

        let control_atom = gate_params.target_atoms[0];
        let target_atom = gate_params.target_atoms[1];

        // Check if atoms are within blockade radius
        let distance = self.calculate_atom_distance(control_atom, target_atom)?;
        if distance > self.config.blockade_radius {
            return Err(DeviceError::InvalidInput(
                "Atoms are too far apart for blockade interaction".to_string(),
            ));
        }

        // Apply blockade interaction (simplified)
        match (&self.atom_states[control_atom], &self.atom_states[target_atom]) {
            (RydbergState::Rydberg, RydbergState::Ground) => {
                // |10⟩ state - no change due to blockade
            }
            (RydbergState::Ground, RydbergState::Rydberg) => {
                // |01⟩ state - no change due to blockade
            }
            (RydbergState::Rydberg, RydbergState::Rydberg) => {
                // |11⟩ state - blockade prevents double excitation
                // Apply phase shift
                // Implementation would include proper phase tracking
            }
            _ => {
                // Other states handled with more complex logic
            }
        }

        Ok(())
    }

    /// Apply global rotation to all atoms
    fn apply_global_rotation(&mut self, gate_params: &RydbergGateParams) -> DeviceResult<()> {
        let rotation_angle = gate_params
            .additional_params
            .get("rotation_angle")
            .copied()
            .unwrap_or(std::f64::consts::PI / 2.0);

        for i in 0..self.atom_states.len() {
            match &self.atom_states[i] {
                RydbergState::Ground => {
                    let excitation_prob = (rotation_angle / 2.0).sin().powi(2);
                    self.atom_states[i] = RydbergState::Superposition {
                        amplitude_ground: (1.0 - excitation_prob).sqrt(),
                        amplitude_rydberg: excitation_prob.sqrt(),
                    };
                }
                RydbergState::Rydberg => {
                    let ground_prob = (rotation_angle / 2.0).sin().powi(2);
                    self.atom_states[i] = RydbergState::Superposition {
                        amplitude_ground: ground_prob.sqrt(),
                        amplitude_rydberg: (1.0 - ground_prob).sqrt(),
                    };
                }
                RydbergState::Superposition { .. } => {
                    // More complex rotation logic for superposition states
                    // Implementation would include proper quantum state evolution
                }
            }
        }

        Ok(())
    }

    /// Calculate distance between two atoms
    fn calculate_atom_distance(&self, atom1: usize, atom2: usize) -> DeviceResult<f64> {
        if atom1 >= self.atom_states.len() || atom2 >= self.atom_states.len() {
            return Err(DeviceError::InvalidInput("Atom index out of bounds".to_string()));
        }

        // Simplified distance calculation assuming regular spacing
        let distance = ((atom1 as f64 - atom2 as f64).abs() * self.config.atom_spacing);
        Ok(distance)
    }

    /// Get current atom states
    pub fn get_atom_states(&self) -> &[RydbergState] {
        &self.atom_states
    }

    /// Measure atom states
    pub fn measure_atoms(&mut self, atom_indices: &[usize]) -> DeviceResult<Vec<bool>> {
        let mut results = Vec::new();

        for &atom_index in atom_indices {
            if atom_index >= self.atom_states.len() {
                return Err(DeviceError::InvalidInput(format!(
                    "Atom index {} out of bounds",
                    atom_index
                )));
            }

            let measurement_result = match &self.atom_states[atom_index] {
                RydbergState::Ground => false,
                RydbergState::Rydberg => true,
                RydbergState::Superposition { amplitude_rydberg, .. } => {
                    // Probabilistic measurement
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    
                    let mut hasher = DefaultHasher::new();
                    atom_index.hash(&mut hasher);
                    let hash = hasher.finish();
                    let random_value = (hash % 1000) as f64 / 1000.0;
                    
                    random_value < amplitude_rydberg.powi(2)
                }
            };

            // Collapse the wavefunction after measurement
            self.atom_states[atom_index] = if measurement_result {
                RydbergState::Rydberg
            } else {
                RydbergState::Ground
            };

            results.push(measurement_result);
        }

        Ok(results)
    }

    /// Reset all atoms to ground state
    pub fn reset(&mut self) {
        for state in &mut self.atom_states {
            *state = RydbergState::Ground;
        }
    }

    /// Calculate system energy
    pub fn calculate_energy(&self) -> f64 {
        let mut energy = 0.0;

        // Add single-atom energies
        for state in &self.atom_states {
            match state {
                RydbergState::Rydberg => {
                    energy += 1.0; // Rydberg excitation energy (normalized)
                }
                RydbergState::Superposition { amplitude_rydberg, .. } => {
                    energy += amplitude_rydberg.powi(2);
                }
                _ => {} // Ground state has zero energy
            }
        }

        // Add interaction energies
        for interaction in &self.interactions {
            for &(atom1, atom2) in &interaction.atom_pairs {
                let interaction_energy = match (&self.atom_states[atom1], &self.atom_states[atom2]) {
                    (RydbergState::Rydberg, RydbergState::Rydberg) => interaction.strength,
                    _ => 0.0,
                };
                energy += interaction_energy;
            }
        }

        energy
    }
}

impl Default for RydbergConfig {
    fn default() -> Self {
        Self {
            principal_quantum_number: 70,
            excitation_wavelength: 480.0,
            blockade_radius: 8.0,
            max_rabi_frequency: 10.0,
            laser_linewidth: 1.0,
            atom_spacing: 5.0,
            temperature: 1.0,
        }
    }
}

impl Default for RydbergLaserParams {
    fn default() -> Self {
        Self {
            rabi_frequency: 1.0,
            detuning: 0.0,
            power: 1.0,
            beam_waist: 1.0,
            polarization: LaserPolarization::Linear,
            phase: 0.0,
        }
    }
}

impl Default for RydbergTimingParams {
    fn default() -> Self {
        Self {
            pulse_duration: Duration::from_micros(1),
            rise_time: Duration::from_nanos(100),
            fall_time: Duration::from_nanos(100),
            wait_time: Duration::from_micros(1),
        }
    }
}

/// Create a single excitation gate
pub fn create_single_excitation_gate(
    atom_index: usize,
    rabi_frequency: f64,
    duration: Duration,
) -> RydbergGateParams {
    RydbergGateParams {
        gate_type: RydbergGateType::SingleExcitation,
        target_atoms: vec![atom_index],
        laser_params: RydbergLaserParams {
            rabi_frequency,
            ..Default::default()
        },
        timing: RydbergTimingParams {
            pulse_duration: duration,
            ..Default::default()
        },
        additional_params: HashMap::new(),
    }
}

/// Create a controlled-Z gate
pub fn create_controlled_z_gate(
    control_atom: usize,
    target_atom: usize,
    interaction_strength: f64,
    duration: Duration,
) -> RydbergGateParams {
    let mut additional_params = HashMap::new();
    additional_params.insert("interaction_strength".to_string(), interaction_strength);

    RydbergGateParams {
        gate_type: RydbergGateType::ControlledZ,
        target_atoms: vec![control_atom, target_atom],
        laser_params: RydbergLaserParams::default(),
        timing: RydbergTimingParams {
            pulse_duration: duration,
            ..Default::default()
        },
        additional_params,
    }
}

/// Create a global rotation gate
pub fn create_global_rotation_gate(
    num_atoms: usize,
    rotation_angle: f64,
    duration: Duration,
) -> RydbergGateParams {
    let target_atoms: Vec<usize> = (0..num_atoms).collect();
    let mut additional_params = HashMap::new();
    additional_params.insert("rotation_angle".to_string(), rotation_angle);

    RydbergGateParams {
        gate_type: RydbergGateType::GlobalRotation,
        target_atoms,
        laser_params: RydbergLaserParams::default(),
        timing: RydbergTimingParams {
            pulse_duration: duration,
            ..Default::default()
        },
        additional_params,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rydberg_simulator_creation() {
        let config = RydbergConfig::default();
        let simulator = RydbergSimulator::new(config, 5);
        
        assert_eq!(simulator.get_atom_states().len(), 5);
        assert!(simulator.get_atom_states().iter().all(|s| *s == RydbergState::Ground));
    }

    #[test]
    fn test_single_excitation_gate() {
        let config = RydbergConfig::default();
        let mut simulator = RydbergSimulator::new(config, 3);
        
        let gate = create_single_excitation_gate(0, 1.0, Duration::from_micros(1));
        assert!(simulator.apply_gate(&gate).is_ok());
    }

    #[test]
    fn test_measurement() {
        let config = RydbergConfig::default();
        let mut simulator = RydbergSimulator::new(config, 2);
        
        let results = simulator.measure_atoms(&[0, 1]).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|&r| !r)); // All should be false (ground state)
    }

    #[test]
    fn test_energy_calculation() {
        let config = RydbergConfig::default();
        let simulator = RydbergSimulator::new(config, 2);
        
        let energy = simulator.calculate_energy();
        assert_eq!(energy, 0.0); // All atoms in ground state
    }
}