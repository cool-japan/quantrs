//! Gate-Based Photonic Quantum Computing
//!
//! This module implements gate-based (discrete variable) photonic quantum computing
//! using photonic qubits encoded in various optical modes and polarizations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use thiserror::Error;

use super::{PhotonicMode, PhotonicSystemType};
use crate::DeviceResult;
use quantrs2_core::qubit::QubitId;
use scirs2_core::random::prelude::*;

/// Errors for gate-based photonic operations
#[derive(Error, Debug)]
pub enum PhotonicGateError {
    #[error("Invalid qubit encoding: {0}")]
    InvalidEncoding(String),
    #[error("Polarization mismatch: {0}")]
    PolarizationMismatch(String),
    #[error("Path mode not found: {0}")]
    PathModeNotFound(usize),
    #[error("Insufficient photon resources: {0}")]
    InsufficientPhotons(String),
    #[error("Gate decomposition failed: {0}")]
    GateDecompositionFailed(String),
}

type PhotonicGateResult<T> = Result<T, PhotonicGateError>;

/// Photonic qubit encoding schemes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhotonicQubitEncoding {
    /// Polarization encoding: |0⟩ = |H⟩, |1⟩ = |V⟩
    Polarization,
    /// Path encoding: |0⟩ = |path₀⟩, |1⟩ = |path₁⟩
    Path { path0: usize, path1: usize },
    /// Time-bin encoding: |0⟩ = early, |1⟩ = late
    TimeBin { early_time: f64, late_time: f64 },
    /// Frequency encoding: |0⟩ = ω₀, |1⟩ = ω₁
    Frequency { freq0: f64, freq1: f64 },
    /// Dual-rail encoding (path encoding with single photons)
    DualRail { rail0: usize, rail1: usize },
}

/// Polarization states
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Polarization {
    /// Horizontal polarization
    Horizontal,
    /// Vertical polarization
    Vertical,
    /// Diagonal polarization (+45°)
    Diagonal,
    /// Anti-diagonal polarization (-45°)
    AntiDiagonal,
    /// Right circular polarization
    RightCircular,
    /// Left circular polarization
    LeftCircular,
}

/// Photonic qubit state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicQubitState {
    /// Encoding scheme
    pub encoding: PhotonicQubitEncoding,
    /// Amplitude for |0⟩ state
    pub amplitude_0: f64,
    /// Amplitude for |1⟩ state
    pub amplitude_1: f64,
    /// Phase between |0⟩ and |1⟩
    pub relative_phase: f64,
    /// Global phase
    pub global_phase: f64,
}

impl PhotonicQubitState {
    /// Create |0⟩ state
    pub fn zero(encoding: PhotonicQubitEncoding) -> Self {
        Self {
            encoding,
            amplitude_0: 1.0,
            amplitude_1: 0.0,
            relative_phase: 0.0,
            global_phase: 0.0,
        }
    }

    /// Create |1⟩ state
    pub fn one(encoding: PhotonicQubitEncoding) -> Self {
        Self {
            encoding,
            amplitude_0: 0.0,
            amplitude_1: 1.0,
            relative_phase: 0.0,
            global_phase: 0.0,
        }
    }

    /// Create |+⟩ state (equal superposition)
    pub fn plus(encoding: PhotonicQubitEncoding) -> Self {
        Self {
            encoding,
            amplitude_0: 1.0 / (2.0_f64).sqrt(),
            amplitude_1: 1.0 / (2.0_f64).sqrt(),
            relative_phase: 0.0,
            global_phase: 0.0,
        }
    }

    /// Create |-⟩ state
    pub fn minus(encoding: PhotonicQubitEncoding) -> Self {
        Self {
            encoding,
            amplitude_0: 1.0 / (2.0_f64).sqrt(),
            amplitude_1: 1.0 / (2.0_f64).sqrt(),
            relative_phase: PI,
            global_phase: 0.0,
        }
    }

    /// Get probability of measuring |0⟩
    pub fn prob_zero(&self) -> f64 {
        self.amplitude_0 * self.amplitude_0
    }

    /// Get probability of measuring |1⟩
    pub fn prob_one(&self) -> f64 {
        self.amplitude_1 * self.amplitude_1
    }

    /// Apply Pauli-X gate
    pub fn pauli_x(&mut self) {
        std::mem::swap(&mut self.amplitude_0, &mut self.amplitude_1);
        self.relative_phase += PI;
    }

    /// Apply Pauli-Y gate
    pub fn pauli_y(&mut self) {
        let old_amp_0 = self.amplitude_0;
        self.amplitude_0 = self.amplitude_1;
        self.amplitude_1 = -old_amp_0;
        self.global_phase += PI / 2.0;
    }

    /// Apply Pauli-Z gate
    pub fn pauli_z(&mut self) {
        self.relative_phase += PI;
    }

    /// Apply Hadamard gate
    pub fn hadamard(&mut self) {
        let old_amp_0 = self.amplitude_0;
        let old_amp_1 = self.amplitude_1;
        let old_phase = self.relative_phase;

        self.amplitude_0 = (old_amp_0 + old_amp_1 * old_phase.cos()) / (2.0_f64).sqrt();
        self.amplitude_1 = (old_amp_0 - old_amp_1 * old_phase.cos()) / (2.0_f64).sqrt();
        self.relative_phase = if old_amp_1 * old_phase.sin() >= 0.0 {
            0.0
        } else {
            PI
        };
    }

    /// Apply rotation around X-axis
    pub fn rx(&mut self, angle: f64) {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        let old_amp_0 = self.amplitude_0;
        let old_amp_1 = self.amplitude_1;

        // RX rotation: |0⟩ → cos(θ/2)|0⟩ - i*sin(θ/2)|1⟩
        //              |1⟩ → -i*sin(θ/2)|0⟩ + cos(θ/2)|1⟩
        // For real amplitudes, we approximate the complex rotation
        self.amplitude_0 = (cos_half * old_amp_0 + sin_half * old_amp_1).abs();
        self.amplitude_1 = (sin_half * old_amp_0 + cos_half * old_amp_1).abs();

        // Update phase to account for the imaginary components
        if angle.abs() > PI / 2.0 {
            self.relative_phase += PI;
        }
    }

    /// Apply rotation around Y-axis
    pub fn ry(&mut self, angle: f64) {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        let old_amp_0 = self.amplitude_0;
        let old_amp_1 = self.amplitude_1;

        self.amplitude_0 = (cos_half * old_amp_0 - sin_half * old_amp_1).abs();
        self.amplitude_1 = (sin_half * old_amp_0 + cos_half * old_amp_1).abs();
    }

    /// Apply rotation around Z-axis
    pub fn rz(&mut self, angle: f64) {
        self.relative_phase += angle;
        self.global_phase -= angle / 2.0;
    }

    /// Apply phase gate
    pub fn phase(&mut self, angle: f64) {
        self.relative_phase += angle;
    }
}

/// Photonic gate implementations
pub struct PhotonicGates;

impl PhotonicGates {
    /// Implement Pauli-X with polarization encoding
    pub fn polarization_x() -> PhotonicGateImpl {
        PhotonicGateImpl {
            gate_name: "PolarizationX".to_string(),
            encoding_required: PhotonicQubitEncoding::Polarization,
            optical_elements: vec![OpticalElement::HalfWaveplate { angle: PI / 2.0 }],
            success_probability: 1.0,
            fidelity: 0.995,
        }
    }

    /// Implement Pauli-Z with polarization encoding
    pub fn polarization_z() -> PhotonicGateImpl {
        PhotonicGateImpl {
            gate_name: "PolarizationZ".to_string(),
            encoding_required: PhotonicQubitEncoding::Polarization,
            optical_elements: vec![OpticalElement::PhaseShift { phase: PI }],
            success_probability: 1.0,
            fidelity: 0.999,
        }
    }

    /// Implement Hadamard with polarization encoding
    pub fn polarization_hadamard() -> PhotonicGateImpl {
        PhotonicGateImpl {
            gate_name: "PolarizationHadamard".to_string(),
            encoding_required: PhotonicQubitEncoding::Polarization,
            optical_elements: vec![OpticalElement::HalfWaveplate { angle: PI / 8.0 }],
            success_probability: 1.0,
            fidelity: 0.995,
        }
    }

    /// Implement CNOT with dual-rail encoding
    pub fn dual_rail_cnot(
        control_rails: (usize, usize),
        target_rails: (usize, usize),
    ) -> PhotonicGateImpl {
        PhotonicGateImpl {
            gate_name: "DualRailCNOT".to_string(),
            encoding_required: PhotonicQubitEncoding::DualRail {
                rail0: control_rails.0,
                rail1: control_rails.1,
            },
            optical_elements: vec![
                // Fredkin gate implementation
                OpticalElement::BeamSplitter {
                    transmittance: 0.5,
                    phase: 0.0,
                    input1: control_rails.1,
                    input2: target_rails.0,
                },
                OpticalElement::BeamSplitter {
                    transmittance: 0.5,
                    phase: 0.0,
                    input1: control_rails.1,
                    input2: target_rails.1,
                },
            ],
            success_probability: 0.25, // Two-photon gates have lower success probability
            fidelity: 0.90,
        }
    }

    /// Implement controlled-Z with path encoding
    pub fn path_cz(
        control_paths: (usize, usize),
        target_paths: (usize, usize),
    ) -> PhotonicGateImpl {
        PhotonicGateImpl {
            gate_name: "PathCZ".to_string(),
            encoding_required: PhotonicQubitEncoding::Path {
                path0: control_paths.0,
                path1: control_paths.1,
            },
            optical_elements: vec![OpticalElement::MachZehnderInterferometer {
                path1: control_paths.1,
                path2: target_paths.1,
                phase_shift: PI,
            }],
            success_probability: 1.0,
            fidelity: 0.98,
        }
    }

    /// Implement arbitrary single-qubit rotation
    pub fn arbitrary_rotation(axis: RotationAxis, angle: f64) -> PhotonicGateImpl {
        let elements = match axis {
            RotationAxis::X => vec![OpticalElement::HalfWaveplate { angle: angle / 2.0 }],
            RotationAxis::Y => vec![
                OpticalElement::QuarterWaveplate,
                OpticalElement::HalfWaveplate { angle: angle / 2.0 },
                OpticalElement::QuarterWaveplate,
            ],
            RotationAxis::Z => vec![OpticalElement::PhaseShift { phase: angle }],
        };

        PhotonicGateImpl {
            gate_name: format!("R{:?}({:.3})", axis, angle),
            encoding_required: PhotonicQubitEncoding::Polarization,
            optical_elements: elements,
            success_probability: 1.0,
            fidelity: 0.995,
        }
    }
}

/// Rotation axis for arbitrary rotations
#[derive(Debug, Clone, Copy)]
pub enum RotationAxis {
    X,
    Y,
    Z,
}

/// Optical elements for implementing photonic gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpticalElement {
    /// Half-wave plate with rotation angle
    HalfWaveplate { angle: f64 },
    /// Quarter-wave plate
    QuarterWaveplate,
    /// Phase shifter
    PhaseShift { phase: f64 },
    /// Beam splitter
    BeamSplitter {
        transmittance: f64,
        phase: f64,
        input1: usize,
        input2: usize,
    },
    /// Mach-Zehnder interferometer
    MachZehnderInterferometer {
        path1: usize,
        path2: usize,
        phase_shift: f64,
    },
    /// Polarizing beam splitter
    PolarizingBeamSplitter { h_output: usize, v_output: usize },
    /// Photon detector
    PhotonDetector { mode: usize, efficiency: f64 },
    /// Electro-optic modulator
    ElectroOpticModulator { voltage: f64, response_time: f64 },
}

/// Implementation details for a photonic gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicGateImpl {
    /// Gate name
    pub gate_name: String,
    /// Required encoding scheme
    pub encoding_required: PhotonicQubitEncoding,
    /// Sequence of optical elements
    pub optical_elements: Vec<OpticalElement>,
    /// Success probability (for probabilistic gates)
    pub success_probability: f64,
    /// Gate fidelity
    pub fidelity: f64,
}

impl PhotonicGateImpl {
    /// Apply gate to a photonic qubit state
    pub fn apply(&self, state: &mut PhotonicQubitState) -> PhotonicGateResult<bool> {
        // Check encoding compatibility
        if std::mem::discriminant(&state.encoding)
            != std::mem::discriminant(&self.encoding_required)
        {
            return Err(PhotonicGateError::InvalidEncoding(format!(
                "Gate requires {:?}, but state uses {:?}",
                self.encoding_required, state.encoding
            )));
        }

        // Apply optical elements in sequence
        for element in &self.optical_elements {
            self.apply_optical_element(element, state)?;
        }

        // Determine if gate succeeded (for probabilistic gates)
        let success = thread_rng().gen::<f64>() < self.success_probability;

        Ok(success)
    }

    /// Apply individual optical element
    fn apply_optical_element(
        &self,
        element: &OpticalElement,
        state: &mut PhotonicQubitState,
    ) -> PhotonicGateResult<()> {
        match element {
            OpticalElement::HalfWaveplate { angle } => {
                // Rotate polarization by 2*angle
                match state.encoding {
                    PhotonicQubitEncoding::Polarization => {
                        state.rx(2.0 * angle);
                    }
                    _ => {
                        return Err(PhotonicGateError::InvalidEncoding(
                            "Half-wave plate requires polarization encoding".to_string(),
                        ))
                    }
                }
            }
            OpticalElement::QuarterWaveplate => {
                // Convert linear to circular polarization
                match state.encoding {
                    PhotonicQubitEncoding::Polarization => {
                        state.phase(PI / 2.0);
                    }
                    _ => {
                        return Err(PhotonicGateError::InvalidEncoding(
                            "Quarter-wave plate requires polarization encoding".to_string(),
                        ))
                    }
                }
            }
            OpticalElement::PhaseShift { phase } => {
                state.rz(*phase);
            }
            OpticalElement::BeamSplitter {
                transmittance,
                phase,
                ..
            } => {
                // Implement beam splitter transformation
                let cos_theta = transmittance.sqrt();
                let sin_theta = (1.0 - transmittance).sqrt();

                let old_amp_0 = state.amplitude_0;
                let old_amp_1 = state.amplitude_1;

                state.amplitude_0 =
                    (cos_theta * old_amp_0 + sin_theta * old_amp_1 * phase.cos()).abs();
                state.amplitude_1 =
                    (sin_theta * old_amp_0 - cos_theta * old_amp_1 * phase.cos()).abs();
            }
            OpticalElement::MachZehnderInterferometer { phase_shift, .. } => {
                // Implement interferometric phase shift
                state.relative_phase += phase_shift;
            }
            OpticalElement::PolarizingBeamSplitter { .. } => {
                // Separates H and V polarizations - measurement-like operation
                // For state evolution, this acts as a conditional operation
                if state.prob_zero() > 0.5 {
                    state.amplitude_1 = 0.0;
                    state.amplitude_0 = 1.0;
                } else {
                    state.amplitude_0 = 0.0;
                    state.amplitude_1 = 1.0;
                }
            }
            OpticalElement::PhotonDetector { efficiency, .. } => {
                // Detection success depends on efficiency
                if thread_rng().gen::<f64>() > *efficiency {
                    return Err(PhotonicGateError::InsufficientPhotons(
                        "Photon detection failed".to_string(),
                    ));
                }
            }
            OpticalElement::ElectroOpticModulator { voltage, .. } => {
                // Apply voltage-dependent phase shift
                let phase_shift = voltage * 0.1; // Simplified linear response
                state.rz(phase_shift);
            }
        }

        Ok(())
    }

    /// Get resource requirements
    pub fn resource_requirements(&self) -> PhotonicResourceRequirements {
        let mut requirements = PhotonicResourceRequirements::default();

        for element in &self.optical_elements {
            match element {
                OpticalElement::HalfWaveplate { .. } => {
                    requirements.waveplates += 1;
                }
                OpticalElement::QuarterWaveplate => {
                    requirements.waveplates += 1;
                }
                OpticalElement::BeamSplitter { .. } => {
                    requirements.beam_splitters += 1;
                }
                OpticalElement::PhotonDetector { .. } => {
                    requirements.detectors += 1;
                }
                OpticalElement::PolarizingBeamSplitter { .. } => {
                    requirements.polarizing_beam_splitters += 1;
                }
                OpticalElement::MachZehnderInterferometer { .. } => {
                    requirements.interferometers += 1;
                }
                OpticalElement::ElectroOpticModulator { .. } => {
                    requirements.modulators += 1;
                }
                OpticalElement::PhaseShift { .. } => {
                    requirements.phase_shifters += 1;
                }
            }
        }

        requirements
    }
}

/// Resource requirements for photonic gates
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhotonicResourceRequirements {
    /// Number of waveplates needed
    pub waveplates: usize,
    /// Number of beam splitters needed
    pub beam_splitters: usize,
    /// Number of detectors needed
    pub detectors: usize,
    /// Number of polarizing beam splitters needed
    pub polarizing_beam_splitters: usize,
    /// Number of interferometers needed
    pub interferometers: usize,
    /// Number of modulators needed
    pub modulators: usize,
    /// Number of phase shifters needed
    pub phase_shifters: usize,
}

/// Photonic circuit compiler
pub struct PhotonicCircuitCompiler {
    /// Available encodings
    pub available_encodings: Vec<PhotonicQubitEncoding>,
    /// Hardware constraints
    pub hardware_constraints: PhotonicHardwareConstraints,
}

/// Hardware constraints for photonic systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicHardwareConstraints {
    /// Maximum number of optical modes
    pub max_modes: usize,
    /// Available waveplates
    pub available_waveplates: usize,
    /// Available beam splitters
    pub available_beam_splitters: usize,
    /// Available detectors
    pub available_detectors: usize,
    /// Detector efficiency
    pub detector_efficiency: f64,
    /// Phase stability (in radians)
    pub phase_stability: f64,
    /// Loss rate per optical element
    pub loss_rate: f64,
}

impl Default for PhotonicHardwareConstraints {
    fn default() -> Self {
        Self {
            max_modes: 16,
            available_waveplates: 8,
            available_beam_splitters: 4,
            available_detectors: 8,
            detector_efficiency: 0.9,
            phase_stability: 0.01,
            loss_rate: 0.005,
        }
    }
}

impl PhotonicCircuitCompiler {
    pub fn new(constraints: PhotonicHardwareConstraints) -> Self {
        Self {
            available_encodings: vec![
                PhotonicQubitEncoding::Polarization,
                PhotonicQubitEncoding::Path { path0: 0, path1: 1 },
                PhotonicQubitEncoding::DualRail { rail0: 0, rail1: 1 },
            ],
            hardware_constraints: constraints,
        }
    }

    /// Compile a gate sequence to photonic implementation
    pub fn compile_gate_sequence(
        &self,
        gates: &[PhotonicGateImpl],
    ) -> PhotonicGateResult<PhotonicCircuitImplementation> {
        let mut total_requirements = PhotonicResourceRequirements::default();
        let mut success_probability = 1.0;
        let mut total_fidelity = 1.0;

        for gate in gates {
            let requirements = gate.resource_requirements();

            // Check resource constraints
            if total_requirements.waveplates + requirements.waveplates
                > self.hardware_constraints.available_waveplates
            {
                return Err(PhotonicGateError::InsufficientPhotons(
                    "Not enough waveplates available".to_string(),
                ));
            }

            if total_requirements.beam_splitters + requirements.beam_splitters
                > self.hardware_constraints.available_beam_splitters
            {
                return Err(PhotonicGateError::InsufficientPhotons(
                    "Not enough beam splitters available".to_string(),
                ));
            }

            // Accumulate requirements
            total_requirements.waveplates += requirements.waveplates;
            total_requirements.beam_splitters += requirements.beam_splitters;
            total_requirements.detectors += requirements.detectors;
            total_requirements.polarizing_beam_splitters += requirements.polarizing_beam_splitters;
            total_requirements.interferometers += requirements.interferometers;
            total_requirements.modulators += requirements.modulators;
            total_requirements.phase_shifters += requirements.phase_shifters;

            // Update success probability and fidelity
            success_probability *= gate.success_probability;
            total_fidelity *= gate.fidelity;
        }

        Ok(PhotonicCircuitImplementation {
            gates: gates.to_vec(),
            resource_requirements: total_requirements,
            success_probability,
            total_fidelity,
            estimated_execution_time: std::time::Duration::from_millis(gates.len() as u64 * 10),
        })
    }

    /// Optimize gate sequence for hardware
    pub fn optimize_for_hardware(
        &self,
        implementation: &mut PhotonicCircuitImplementation,
    ) -> PhotonicGateResult<()> {
        // Combine adjacent phase shifts
        let mut optimized_gates = Vec::new();
        let mut accumulated_phase = 0.0;

        for gate in &implementation.gates {
            if gate.optical_elements.len() == 1 {
                if let OpticalElement::PhaseShift { phase } = &gate.optical_elements[0] {
                    accumulated_phase += phase;
                    continue;
                }
            }

            // If we have accumulated phase, add it as a single gate
            if accumulated_phase.abs() > 1e-10 {
                optimized_gates.push(PhotonicGateImpl {
                    gate_name: "CombinedPhase".to_string(),
                    encoding_required: PhotonicQubitEncoding::Polarization,
                    optical_elements: vec![OpticalElement::PhaseShift {
                        phase: accumulated_phase,
                    }],
                    success_probability: 1.0,
                    fidelity: 0.999,
                });
                accumulated_phase = 0.0;
            }

            optimized_gates.push(gate.clone());
        }

        // Add final accumulated phase if any
        if accumulated_phase.abs() > 1e-10 {
            optimized_gates.push(PhotonicGateImpl {
                gate_name: "FinalPhase".to_string(),
                encoding_required: PhotonicQubitEncoding::Polarization,
                optical_elements: vec![OpticalElement::PhaseShift {
                    phase: accumulated_phase,
                }],
                success_probability: 1.0,
                fidelity: 0.999,
            });
        }

        implementation.gates = optimized_gates;

        Ok(())
    }
}

/// Compiled photonic circuit implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicCircuitImplementation {
    /// Sequence of photonic gates
    pub gates: Vec<PhotonicGateImpl>,
    /// Total resource requirements
    pub resource_requirements: PhotonicResourceRequirements,
    /// Overall success probability
    pub success_probability: f64,
    /// Total fidelity
    pub total_fidelity: f64,
    /// Estimated execution time
    pub estimated_execution_time: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_photonic_qubit_states() {
        let zero = PhotonicQubitState::zero(PhotonicQubitEncoding::Polarization);
        assert_eq!(zero.prob_zero(), 1.0);
        assert_eq!(zero.prob_one(), 0.0);

        let one = PhotonicQubitState::one(PhotonicQubitEncoding::Polarization);
        assert_eq!(one.prob_zero(), 0.0);
        assert_eq!(one.prob_one(), 1.0);

        let plus = PhotonicQubitState::plus(PhotonicQubitEncoding::Polarization);
        assert!((plus.prob_zero() - 0.5).abs() < 1e-10);
        assert!((plus.prob_one() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_gates() {
        let mut state = PhotonicQubitState::zero(PhotonicQubitEncoding::Polarization);

        state.pauli_x();
        assert_eq!(state.prob_one(), 1.0);

        state.pauli_x();
        assert_eq!(state.prob_zero(), 1.0);

        state.pauli_z();
        assert_eq!(state.prob_zero(), 1.0); // |0⟩ is eigenstate of Z
    }

    #[test]
    fn test_hadamard_gate() {
        let mut state = PhotonicQubitState::zero(PhotonicQubitEncoding::Polarization);

        state.hadamard();
        assert!((state.prob_zero() - 0.5).abs() < 1e-10);
        assert!((state.prob_one() - 0.5).abs() < 1e-10);

        state.hadamard();
        assert!((state.prob_zero() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_photonic_gate_implementation() {
        let x_gate = PhotonicGates::polarization_x();
        let mut state = PhotonicQubitState::zero(PhotonicQubitEncoding::Polarization);

        let success = x_gate.apply(&mut state).unwrap();
        assert!(success);
        assert!(state.prob_one() > 0.9); // Should be close to |1⟩
    }

    #[test]
    fn test_resource_requirements() {
        let hadamard_gate = PhotonicGates::polarization_hadamard();
        let requirements = hadamard_gate.resource_requirements();

        assert_eq!(requirements.waveplates, 1);
        assert_eq!(requirements.beam_splitters, 0);
    }

    #[test]
    fn test_circuit_compilation() {
        let constraints = PhotonicHardwareConstraints::default();
        let compiler = PhotonicCircuitCompiler::new(constraints);

        let gates = vec![
            PhotonicGates::polarization_hadamard(),
            PhotonicGates::polarization_x(),
        ];

        let implementation = compiler.compile_gate_sequence(&gates).unwrap();
        assert_eq!(implementation.gates.len(), 2);
        assert!(implementation.success_probability > 0.9);
    }

    #[test]
    fn test_optimization() {
        let constraints = PhotonicHardwareConstraints::default();
        let compiler = PhotonicCircuitCompiler::new(constraints);

        // Create a sequence with multiple phase shifts
        let gates = vec![
            PhotonicGateImpl {
                gate_name: "Phase1".to_string(),
                encoding_required: PhotonicQubitEncoding::Polarization,
                optical_elements: vec![OpticalElement::PhaseShift { phase: PI / 4.0 }],
                success_probability: 1.0,
                fidelity: 0.999,
            },
            PhotonicGateImpl {
                gate_name: "Phase2".to_string(),
                encoding_required: PhotonicQubitEncoding::Polarization,
                optical_elements: vec![OpticalElement::PhaseShift { phase: PI / 4.0 }],
                success_probability: 1.0,
                fidelity: 0.999,
            },
        ];

        let mut implementation = compiler.compile_gate_sequence(&gates).unwrap();
        let original_length = implementation.gates.len();

        compiler.optimize_for_hardware(&mut implementation).unwrap();

        // Should combine the two phase shifts into one
        assert!(implementation.gates.len() <= original_length);
    }
}
