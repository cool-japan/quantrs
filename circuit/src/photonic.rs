//! Photonic quantum circuit support
//!
//! This module provides specialized support for photonic quantum computing,
//! including linear optical elements, measurement-based computation,
//! and continuous variable quantum computation.

use crate::builder::Circuit;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

/// Photonic mode representing optical field modes
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhotonicMode {
    pub id: u32,
    pub polarization: Polarization,
    pub frequency: Option<f64>, // Optional frequency specification
}

impl PhotonicMode {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            polarization: Polarization::Horizontal,
            frequency: None,
        }
    }

    pub fn with_polarization(mut self, polarization: Polarization) -> Self {
        self.polarization = polarization;
        self
    }

    pub fn with_frequency(mut self, frequency: f64) -> Self {
        self.frequency = Some(frequency);
        self
    }
}

/// Polarization states for photonic modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Polarization {
    Horizontal,
    Vertical,
    Diagonal,
    AntiDiagonal,
    LeftCircular,
    RightCircular,
}

/// Linear optical elements for photonic circuits
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhotonicGate {
    /// Beam splitter with reflectivity parameter
    BeamSplitter {
        mode1: PhotonicMode,
        mode2: PhotonicMode,
        reflectivity: f64, // 0.0 = fully transmissive, 1.0 = fully reflective
        phase: f64,
    },
    /// Phase shifter
    PhaseShifter { mode: PhotonicMode, phase: f64 },
    /// Polarization rotator
    PolarizationRotator {
        mode: PhotonicMode,
        angle: f64, // Rotation angle
    },
    /// Half-wave plate
    HalfWavePlate {
        mode: PhotonicMode,
        angle: f64, // Fast axis angle
    },
    /// Quarter-wave plate
    QuarterWavePlate { mode: PhotonicMode, angle: f64 },
    /// Polarizing beam splitter
    PolarizingBeamSplitter {
        input: PhotonicMode,
        h_output: PhotonicMode, // Horizontal polarization output
        v_output: PhotonicMode, // Vertical polarization output
    },
    /// Mach-Zehnder interferometer
    MachZehnder {
        input1: PhotonicMode,
        input2: PhotonicMode,
        output1: PhotonicMode,
        output2: PhotonicMode,
        phase_shift: f64,
    },
    /// Hong-Ou-Mandel effect (two-photon interference)
    HongOuMandel {
        mode1: PhotonicMode,
        mode2: PhotonicMode,
    },
    /// Photonic controlled gate (using ancilla photons)
    PhotonicCNOT {
        control: PhotonicMode,
        target: PhotonicMode,
        ancilla: Vec<PhotonicMode>,
    },
    /// Kerr effect (nonlinear phase shift)
    KerrGate { mode: PhotonicMode, strength: f64 },
}

impl PhotonicGate {
    /// Get the modes involved in this gate
    pub fn modes(&self) -> Vec<PhotonicMode> {
        match self {
            Self::BeamSplitter { mode1, mode2, .. } => vec![*mode1, *mode2],
            Self::PhaseShifter { mode, .. } => vec![*mode],
            Self::PolarizationRotator { mode, .. } => vec![*mode],
            Self::HalfWavePlate { mode, .. } => vec![*mode],
            Self::QuarterWavePlate { mode, .. } => vec![*mode],
            Self::PolarizingBeamSplitter {
                input,
                h_output,
                v_output,
                ..
            } => {
                vec![*input, *h_output, *v_output]
            }
            Self::MachZehnder {
                input1,
                input2,
                output1,
                output2,
                ..
            } => {
                vec![*input1, *input2, *output1, *output2]
            }
            Self::HongOuMandel { mode1, mode2, .. } => vec![*mode1, *mode2],
            Self::PhotonicCNOT {
                control,
                target,
                ancilla,
                ..
            } => {
                let mut modes = vec![*control, *target];
                modes.extend(ancilla);
                modes
            }
            Self::KerrGate { mode, .. } => vec![*mode],
        }
    }

    /// Get gate name
    pub fn name(&self) -> &'static str {
        match self {
            Self::BeamSplitter { .. } => "BS",
            Self::PhaseShifter { .. } => "PS",
            Self::PolarizationRotator { .. } => "PR",
            Self::HalfWavePlate { .. } => "HWP",
            Self::QuarterWavePlate { .. } => "QWP",
            Self::PolarizingBeamSplitter { .. } => "PBS",
            Self::MachZehnder { .. } => "MZ",
            Self::HongOuMandel { .. } => "HOM",
            Self::PhotonicCNOT { .. } => "PCNOT",
            Self::KerrGate { .. } => "KERR",
        }
    }
}

/// Photonic measurement operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhotonicMeasurement {
    /// Photon number measurement
    PhotonNumber {
        mode: PhotonicMode,
        detector_efficiency: f64,
    },
    /// Homodyne measurement (measures quadrature)
    Homodyne {
        mode: PhotonicMode,
        local_oscillator_phase: f64,
        detection_efficiency: f64,
    },
    /// Heterodyne measurement (measures both quadratures)
    Heterodyne {
        mode: PhotonicMode,
        detection_efficiency: f64,
    },
    /// Polarization measurement
    Polarization {
        mode: PhotonicMode,
        measurement_basis: PolarizationBasis,
    },
    /// Coincidence detection
    Coincidence {
        modes: Vec<PhotonicMode>,
        time_window: f64, // nanoseconds
    },
}

/// Polarization measurement bases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolarizationBasis {
    Linear,   // H/V
    Diagonal, // D/A
    Circular, // L/R
}

/// Photonic circuit representation
#[derive(Debug, Clone)]
pub struct PhotonicCircuit {
    /// Number of photonic modes
    pub num_modes: usize,
    /// Photonic gates in the circuit
    pub gates: Vec<PhotonicGate>,
    /// Measurements in the circuit
    pub measurements: Vec<PhotonicMeasurement>,
    /// Mode mapping (logical to physical modes)
    pub mode_mapping: HashMap<u32, u32>,
}

impl PhotonicCircuit {
    /// Create a new photonic circuit
    pub fn new(num_modes: usize) -> Self {
        Self {
            num_modes,
            gates: Vec::new(),
            measurements: Vec::new(),
            mode_mapping: HashMap::new(),
        }
    }

    /// Add a photonic gate
    pub fn add_gate(&mut self, gate: PhotonicGate) -> QuantRS2Result<()> {
        // Validate that all modes are within bounds
        for mode in gate.modes() {
            if mode.id as usize >= self.num_modes {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Mode {} exceeds circuit size {}",
                    mode.id, self.num_modes
                )));
            }
        }

        self.gates.push(gate);
        Ok(())
    }

    /// Add a measurement
    pub fn add_measurement(&mut self, measurement: PhotonicMeasurement) -> QuantRS2Result<()> {
        self.measurements.push(measurement);
        Ok(())
    }

    /// Get circuit depth (simplified)
    pub fn depth(&self) -> usize {
        // For photonic circuits, depth is more complex due to parallelism
        // This is a simplified calculation
        self.gates.len()
    }

    /// Validate the photonic circuit
    pub fn validate(&self) -> QuantRS2Result<()> {
        // Check for mode conflicts
        let mut mode_usage = HashMap::new();

        for (layer, gate) in self.gates.iter().enumerate() {
            for mode in gate.modes() {
                if let Some(&last_usage) = mode_usage.get(&mode.id) {
                    if last_usage == layer {
                        return Err(QuantRS2Error::InvalidInput(format!(
                            "Mode {} used multiple times in layer {}",
                            mode.id, layer
                        )));
                    }
                }
                mode_usage.insert(mode.id, layer);
            }
        }

        Ok(())
    }
}

/// Builder for photonic circuits
#[derive(Clone)]
pub struct PhotonicCircuitBuilder {
    circuit: PhotonicCircuit,
}

impl PhotonicCircuitBuilder {
    /// Create a new builder
    pub fn new(num_modes: usize) -> Self {
        Self {
            circuit: PhotonicCircuit::new(num_modes),
        }
    }

    /// Add a beam splitter
    pub fn beam_splitter(
        &mut self,
        mode1: u32,
        mode2: u32,
        reflectivity: f64,
        phase: f64,
    ) -> QuantRS2Result<&mut Self> {
        let gate = PhotonicGate::BeamSplitter {
            mode1: PhotonicMode::new(mode1),
            mode2: PhotonicMode::new(mode2),
            reflectivity,
            phase,
        };
        self.circuit.add_gate(gate)?;
        Ok(self)
    }

    /// Add a phase shifter
    pub fn phase_shifter(&mut self, mode: u32, phase: f64) -> QuantRS2Result<&mut Self> {
        let gate = PhotonicGate::PhaseShifter {
            mode: PhotonicMode::new(mode),
            phase,
        };
        self.circuit.add_gate(gate)?;
        Ok(self)
    }

    /// Add a Mach-Zehnder interferometer
    pub fn mach_zehnder(
        &mut self,
        input1: u32,
        input2: u32,
        output1: u32,
        output2: u32,
        phase_shift: f64,
    ) -> QuantRS2Result<&mut Self> {
        let gate = PhotonicGate::MachZehnder {
            input1: PhotonicMode::new(input1),
            input2: PhotonicMode::new(input2),
            output1: PhotonicMode::new(output1),
            output2: PhotonicMode::new(output2),
            phase_shift,
        };
        self.circuit.add_gate(gate)?;
        Ok(self)
    }

    /// Add Hong-Ou-Mandel interference
    pub fn hong_ou_mandel(&mut self, mode1: u32, mode2: u32) -> QuantRS2Result<&mut Self> {
        let gate = PhotonicGate::HongOuMandel {
            mode1: PhotonicMode::new(mode1),
            mode2: PhotonicMode::new(mode2),
        };
        self.circuit.add_gate(gate)?;
        Ok(self)
    }

    /// Add photon number measurement
    pub fn measure_photon_number(&mut self, mode: u32) -> QuantRS2Result<&mut Self> {
        let measurement = PhotonicMeasurement::PhotonNumber {
            mode: PhotonicMode::new(mode),
            detector_efficiency: 1.0,
        };
        self.circuit.add_measurement(measurement)?;
        Ok(self)
    }

    /// Build the final circuit
    pub fn build(self) -> QuantRS2Result<PhotonicCircuit> {
        self.circuit.validate()?;
        Ok(self.circuit)
    }
}

/// Conversion between photonic and standard quantum circuits
pub struct PhotonicConverter;

impl PhotonicConverter {
    /// Convert a standard quantum circuit to photonic representation
    pub fn quantum_to_photonic<const N: usize>(
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<PhotonicCircuit> {
        let mut photonic_circuit = PhotonicCircuit::new(N * 2); // Dual-rail encoding

        for gate in circuit.gates() {
            let photonic_gates = Self::convert_gate(gate.as_ref())?;
            for pg in photonic_gates {
                photonic_circuit.add_gate(pg)?;
            }
        }

        Ok(photonic_circuit)
    }

    /// Convert a quantum gate to photonic representation
    fn convert_gate(gate: &dyn GateOp) -> QuantRS2Result<Vec<PhotonicGate>> {
        let mut photonic_gates = Vec::new();
        let gate_name = gate.name();
        let qubits = gate.qubits();

        match gate_name {
            "H" => {
                // Hadamard gate using beam splitters and phase shifters
                let qubit = qubits[0].id();
                let mode0 = qubit * 2; // |0⟩ rail
                let mode1 = qubit * 2 + 1; // |1⟩ rail

                // Beam splitter with 50:50 ratio
                photonic_gates.push(PhotonicGate::BeamSplitter {
                    mode1: PhotonicMode::new(mode0),
                    mode2: PhotonicMode::new(mode1),
                    reflectivity: 0.5,
                    phase: 0.0,
                });
            }
            "X" => {
                // Pauli-X swaps the rails
                let qubit = qubits[0].id();
                let mode0 = qubit * 2;
                let mode1 = qubit * 2 + 1;

                // Swap using beam splitters
                photonic_gates.push(PhotonicGate::BeamSplitter {
                    mode1: PhotonicMode::new(mode0),
                    mode2: PhotonicMode::new(mode1),
                    reflectivity: 1.0, // Full reflection = swap
                    phase: 0.0,
                });
            }
            "Z" => {
                // Pauli-Z adds phase to |1⟩ rail
                let qubit = qubits[0].id();
                let mode1 = qubit * 2 + 1;

                photonic_gates.push(PhotonicGate::PhaseShifter {
                    mode: PhotonicMode::new(mode1),
                    phase: PI,
                });
            }
            "CNOT" => {
                // CNOT using photonic controlled gates (requires ancilla photons)
                let control_qubit = qubits[0].id();
                let target_qubit = qubits[1].id();

                let control_mode = control_qubit * 2 + 1; // Control on |1⟩ rail
                let target_mode0 = target_qubit * 2;
                let target_mode1 = target_qubit * 2 + 1;

                // Simplified photonic CNOT (would need more complex implementation)
                photonic_gates.push(PhotonicGate::PhotonicCNOT {
                    control: PhotonicMode::new(control_mode),
                    target: PhotonicMode::new(target_mode0),
                    ancilla: vec![PhotonicMode::new(target_mode1)],
                });
            }
            _ => {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Gate {} not supported in photonic conversion",
                    gate_name
                )));
            }
        }

        Ok(photonic_gates)
    }
}

/// Continuous variable quantum computation support
#[derive(Debug, Clone)]
pub struct CVCircuit {
    /// Number of modes
    pub num_modes: usize,
    /// CV gates
    pub gates: Vec<CVGate>,
    /// Position/momentum measurements
    pub measurements: Vec<CVMeasurement>,
}

/// Continuous variable gates
#[derive(Debug, Clone, PartialEq)]
pub enum CVGate {
    /// Displacement operator D(α)
    Displacement {
        mode: u32,
        amplitude: f64,
        phase: f64,
    },
    /// Squeezing operator S(r)
    Squeezing {
        mode: u32,
        squeezing_parameter: f64,
        squeezing_angle: f64,
    },
    /// Two-mode squeezing
    TwoModeSqueezing {
        mode1: u32,
        mode2: u32,
        squeezing_parameter: f64,
    },
    /// Rotation gate (phase space rotation)
    Rotation { mode: u32, angle: f64 },
    /// Beam splitter (linear transformation)
    CVBeamSplitter {
        mode1: u32,
        mode2: u32,
        theta: f64, // Beam splitter angle
        phi: f64,   // Phase shift
    },
    /// Kerr gate (cubic phase)
    CVKerr { mode: u32, strength: f64 },
    /// Controlled displacement
    ControlledDisplacement {
        control_mode: u32,
        target_mode: u32,
        strength: f64,
    },
}

/// CV measurements
#[derive(Debug, Clone, PartialEq)]
pub enum CVMeasurement {
    /// Homodyne detection (position/momentum)
    CVHomodyne {
        mode: u32,
        angle: f64, // 0 = position, π/2 = momentum
    },
    /// Heterodyne detection
    CVHeterodyne { mode: u32 },
    /// Photon number measurement
    CVPhotonNumber { mode: u32 },
}

impl CVCircuit {
    /// Create new CV circuit
    pub fn new(num_modes: usize) -> Self {
        Self {
            num_modes,
            gates: Vec::new(),
            measurements: Vec::new(),
        }
    }

    /// Add a displacement gate
    pub fn displacement(&mut self, mode: u32, amplitude: f64, phase: f64) -> QuantRS2Result<()> {
        self.gates.push(CVGate::Displacement {
            mode,
            amplitude,
            phase,
        });
        Ok(())
    }

    /// Add squeezing
    pub fn squeezing(&mut self, mode: u32, r: f64, angle: f64) -> QuantRS2Result<()> {
        self.gates.push(CVGate::Squeezing {
            mode,
            squeezing_parameter: r,
            squeezing_angle: angle,
        });
        Ok(())
    }

    /// Add beam splitter
    pub fn beam_splitter(
        &mut self,
        mode1: u32,
        mode2: u32,
        theta: f64,
        phi: f64,
    ) -> QuantRS2Result<()> {
        self.gates.push(CVGate::CVBeamSplitter {
            mode1,
            mode2,
            theta,
            phi,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::Hadamard;

    #[test]
    fn test_photonic_circuit_creation() {
        let mut circuit = PhotonicCircuit::new(4);

        let bs_gate = PhotonicGate::BeamSplitter {
            mode1: PhotonicMode::new(0),
            mode2: PhotonicMode::new(1),
            reflectivity: 0.5,
            phase: 0.0,
        };

        assert!(circuit.add_gate(bs_gate).is_ok());
        assert_eq!(circuit.gates.len(), 1);
    }

    #[test]
    fn test_photonic_builder() {
        let mut builder = PhotonicCircuitBuilder::new(4);

        builder.beam_splitter(0, 1, 0.5, 0.0).unwrap();
        builder.phase_shifter(1, PI / 2.0).unwrap();
        builder.mach_zehnder(0, 1, 2, 3, PI / 4.0).unwrap();
        let result = builder.build();

        assert!(result.is_ok());
        let circuit = result.unwrap();
        assert_eq!(circuit.gates.len(), 3);
    }

    #[test]
    fn test_quantum_to_photonic_conversion() {
        let mut quantum_circuit = Circuit::<2>::new();
        quantum_circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .unwrap();

        let photonic_result = PhotonicConverter::quantum_to_photonic(&quantum_circuit);
        assert!(photonic_result.is_ok());

        let photonic_circuit = photonic_result.unwrap();
        assert_eq!(photonic_circuit.num_modes, 4); // Dual-rail encoding
        assert!(!photonic_circuit.gates.is_empty());
    }

    #[test]
    fn test_cv_circuit() {
        let mut cv_circuit = CVCircuit::new(2);

        assert!(cv_circuit.displacement(0, 1.0, 0.0).is_ok());
        assert!(cv_circuit.squeezing(1, 0.5, PI / 4.0).is_ok());
        assert!(cv_circuit.beam_splitter(0, 1, PI / 4.0, 0.0).is_ok());

        assert_eq!(cv_circuit.gates.len(), 3);
    }

    #[test]
    fn test_photonic_modes() {
        let mode = PhotonicMode::new(0)
            .with_polarization(Polarization::Vertical)
            .with_frequency(532e12); // Green light

        assert_eq!(mode.id, 0);
        assert_eq!(mode.polarization, Polarization::Vertical);
        assert_eq!(mode.frequency, Some(532e12));
    }

    #[test]
    fn test_hong_ou_mandel() {
        let mut builder = PhotonicCircuitBuilder::new(2);
        builder.hong_ou_mandel(0, 1).unwrap();
        let result = builder.build();

        assert!(result.is_ok());
        let circuit = result.unwrap();
        assert_eq!(circuit.gates.len(), 1);

        match &circuit.gates[0] {
            PhotonicGate::HongOuMandel { mode1, mode2 } => {
                assert_eq!(mode1.id, 0);
                assert_eq!(mode2.id, 1);
            }
            _ => panic!("Expected HongOuMandel gate"),
        }
    }
}
