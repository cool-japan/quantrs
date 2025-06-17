//! Device-specific gate calibration data structures
//!
//! This module provides comprehensive gate calibration tracking for quantum devices,
//! including error rates, gate fidelities, timing information, and hardware-specific
//! parameters. This data is essential for circuit optimization and error mitigation.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

/// Complete calibration data for a quantum device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCalibration {
    /// Device identifier
    pub device_id: String,
    /// Timestamp of calibration
    pub timestamp: SystemTime,
    /// Calibration validity duration
    pub valid_duration: Duration,
    /// Qubit-specific calibrations
    pub qubit_calibrations: HashMap<QubitId, QubitCalibration>,
    /// Single-qubit gate calibrations
    pub single_qubit_gates: HashMap<String, SingleQubitGateCalibration>,
    /// Two-qubit gate calibrations
    pub two_qubit_gates: HashMap<(QubitId, QubitId), TwoQubitGateCalibration>,
    /// Multi-qubit gate calibrations
    pub multi_qubit_gates: HashMap<Vec<QubitId>, MultiQubitGateCalibration>,
    /// Readout calibration data
    pub readout_calibration: ReadoutCalibration,
    /// Cross-talk matrix between qubits
    pub crosstalk_matrix: CrosstalkMatrix,
    /// Device topology and connectivity
    pub topology: DeviceTopology,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Default for DeviceCalibration {
    fn default() -> Self {
        Self {
            device_id: String::new(),
            timestamp: SystemTime::UNIX_EPOCH,
            valid_duration: Duration::from_secs(3600), // 1 hour default
            qubit_calibrations: HashMap::new(),
            single_qubit_gates: HashMap::new(),
            two_qubit_gates: HashMap::new(),
            multi_qubit_gates: HashMap::new(),
            readout_calibration: ReadoutCalibration::default(),
            crosstalk_matrix: CrosstalkMatrix::default(),
            topology: DeviceTopology::default(),
            metadata: HashMap::new(),
        }
    }
}

impl DeviceCalibration {
    /// Get single-qubit gate fidelity for a specific qubit
    pub fn single_qubit_fidelity(&self, qubit: usize) -> Option<f64> {
        let qubit_id = QubitId(qubit as u32);

        // Try to get fidelity from single-qubit gates (prefer X gate as representative)
        if let Some(x_gate) = self.single_qubit_gates.get("X") {
            if let Some(gate_data) = x_gate.qubit_data.get(&qubit_id) {
                return Some(gate_data.fidelity);
            }
        }

        // Fallback: try other common single-qubit gates
        for gate_name in &["H", "Y", "Z", "RX", "RY", "RZ"] {
            if let Some(gate) = self.single_qubit_gates.get(*gate_name) {
                if let Some(gate_data) = gate.qubit_data.get(&qubit_id) {
                    return Some(gate_data.fidelity);
                }
            }
        }

        // Default fallback
        None
    }

    /// Get two-qubit gate fidelity between two qubits
    pub fn gate_fidelity(&self, q1: usize, q2: usize) -> Option<f64> {
        let qubit1 = QubitId(q1 as u32);
        let qubit2 = QubitId(q2 as u32);

        // Look for a two-qubit gate between these qubits (try both directions)
        for gate in self.two_qubit_gates.values() {
            if (gate.control == qubit1 && gate.target == qubit2)
                || (gate.control == qubit2 && gate.target == qubit1)
            {
                return Some(gate.fidelity);
            }
        }

        // Default fallback
        None
    }
}

/// Calibration data for individual qubits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QubitCalibration {
    /// Qubit identifier
    pub qubit_id: QubitId,
    /// Qubit frequency (Hz)
    pub frequency: f64,
    /// Anharmonicity (Hz)
    pub anharmonicity: f64,
    /// T1 coherence time (microseconds)
    pub t1: f64,
    /// T2 coherence time (microseconds)
    pub t2: f64,
    /// T2* coherence time (microseconds)
    pub t2_star: Option<f64>,
    /// Readout assignment error
    pub readout_error: f64,
    /// Thermal population
    pub thermal_population: f64,
    /// Operating temperature (mK)
    pub temperature: Option<f64>,
    /// Additional qubit-specific parameters
    pub parameters: HashMap<String, f64>,
}

/// Single-qubit gate calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleQubitGateCalibration {
    /// Gate name (e.g., "X", "Y", "Z", "H", "RX", "RY", "RZ")
    pub gate_name: String,
    /// Per-qubit calibration data
    pub qubit_data: HashMap<QubitId, SingleQubitGateData>,
    /// Default gate parameters
    pub default_parameters: GateParameters,
}

/// Single-qubit gate data for a specific qubit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleQubitGateData {
    /// Gate error rate
    pub error_rate: f64,
    /// Gate fidelity
    pub fidelity: f64,
    /// Gate duration (nanoseconds)
    pub duration: f64,
    /// Drive amplitude
    pub amplitude: f64,
    /// Drive frequency (Hz)
    pub frequency: f64,
    /// Phase correction
    pub phase: f64,
    /// Pulse shape parameters
    pub pulse_shape: PulseShape,
    /// Calibrated gate matrix (if different from ideal)
    pub calibrated_matrix: Option<Vec<Complex64>>,
    /// Parameter-dependent calibrations (e.g., for rotation angles)
    pub parameter_calibrations: Option<ParameterCalibration>,
}

/// Two-qubit gate calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoQubitGateCalibration {
    /// Gate name (e.g., "CNOT", "CZ", "ISwap")
    pub gate_name: String,
    /// Control qubit
    pub control: QubitId,
    /// Target qubit
    pub target: QubitId,
    /// Gate error rate
    pub error_rate: f64,
    /// Gate fidelity
    pub fidelity: f64,
    /// Gate duration (nanoseconds)
    pub duration: f64,
    /// Coupling strength (MHz)
    pub coupling_strength: f64,
    /// Cross-resonance parameters (for CR gates)
    pub cross_resonance: Option<CrossResonanceParameters>,
    /// Calibrated gate matrix
    pub calibrated_matrix: Option<Vec<Complex64>>,
    /// Direction-specific calibration (some gates work better in one direction)
    pub directional: bool,
    /// Alternative calibration for reversed direction
    pub reversed_calibration: Option<Box<TwoQubitGateCalibration>>,
}

/// Multi-qubit gate calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiQubitGateCalibration {
    /// Gate name (e.g., "Toffoli", "Fredkin")
    pub gate_name: String,
    /// Qubits involved
    pub qubits: Vec<QubitId>,
    /// Gate error rate
    pub error_rate: f64,
    /// Gate fidelity
    pub fidelity: f64,
    /// Gate duration (nanoseconds)
    pub duration: f64,
    /// Decomposition used on hardware
    pub decomposition: GateDecomposition,
    /// Native implementation available
    pub is_native: bool,
}

/// Readout calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadoutCalibration {
    /// Per-qubit readout data
    pub qubit_readout: HashMap<QubitId, QubitReadoutData>,
    /// Readout mitigation matrix
    pub mitigation_matrix: Option<Vec<Vec<f64>>>,
    /// Readout duration (nanoseconds)
    pub duration: f64,
    /// Integration time (nanoseconds)
    pub integration_time: f64,
}

impl Default for ReadoutCalibration {
    fn default() -> Self {
        Self {
            qubit_readout: HashMap::new(),
            mitigation_matrix: None,
            duration: 1000.0,        // 1 microsecond default
            integration_time: 500.0, // 500 ns default
        }
    }
}

/// Qubit-specific readout data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QubitReadoutData {
    /// Probability of reading 0 when prepared in |0⟩
    pub p0_given_0: f64,
    /// Probability of reading 1 when prepared in |1⟩
    pub p1_given_1: f64,
    /// Readout resonator frequency (Hz)
    pub resonator_frequency: f64,
    /// Optimal readout amplitude
    pub readout_amplitude: f64,
    /// Optimal readout phase
    pub readout_phase: f64,
    /// Signal-to-noise ratio
    pub snr: f64,
}

/// Cross-talk matrix between qubits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrosstalkMatrix {
    /// Matrix of crosstalk coefficients
    /// Entry (i,j) represents crosstalk from qubit i to qubit j
    pub matrix: Vec<Vec<f64>>,
    /// Measurement method used
    pub measurement_method: String,
    /// Threshold for significant crosstalk
    pub significance_threshold: f64,
}

impl Default for CrosstalkMatrix {
    fn default() -> Self {
        Self {
            matrix: Vec::new(),
            measurement_method: "default".to_string(),
            significance_threshold: 0.01,
        }
    }
}

/// Device topology and connectivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceTopology {
    /// Number of qubits
    pub num_qubits: usize,
    /// Coupling map: which qubits can interact
    pub coupling_map: Vec<(QubitId, QubitId)>,
    /// Physical layout (e.g., "linear", "grid", "heavy-hex")
    pub layout_type: String,
    /// Physical coordinates of qubits (if applicable)
    pub qubit_coordinates: Option<HashMap<QubitId, (f64, f64)>>,
}

impl Default for DeviceTopology {
    fn default() -> Self {
        Self {
            num_qubits: 0,
            coupling_map: Vec::new(),
            layout_type: "unknown".to_string(),
            qubit_coordinates: None,
        }
    }
}

/// Gate parameters that can be calibrated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateParameters {
    /// Amplitude scaling factor
    pub amplitude_scale: f64,
    /// Phase offset
    pub phase_offset: f64,
    /// Duration scaling factor
    pub duration_scale: f64,
    /// DRAG coefficient (for single-qubit gates)
    pub drag_coefficient: Option<f64>,
    /// Additional hardware-specific parameters
    pub custom_parameters: HashMap<String, f64>,
}

/// Pulse shape for gate implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PulseShape {
    /// Gaussian pulse
    Gaussian { sigma: f64, cutoff: f64 },
    /// Gaussian with DRAG correction
    GaussianDRAG { sigma: f64, beta: f64, cutoff: f64 },
    /// Square pulse
    Square { rise_time: f64 },
    /// Cosine-shaped pulse
    Cosine { rise_time: f64 },
    /// Custom pulse shape
    Custom {
        name: String,
        parameters: HashMap<String, f64>,
    },
}

/// Parameter-dependent calibration (e.g., for rotation gates)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterCalibration {
    /// Calibration points (parameter value -> calibration data)
    pub calibration_points: Vec<(f64, SingleQubitGateData)>,
    /// Interpolation method
    pub interpolation: InterpolationMethod,
    /// Valid parameter range
    pub valid_range: (f64, f64),
}

/// Interpolation method for parameter-dependent calibrations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Linear interpolation
    Linear,
    /// Cubic spline interpolation
    CubicSpline,
    /// Polynomial interpolation
    Polynomial { degree: usize },
    /// Nearest neighbor
    NearestNeighbor,
}

/// Cross-resonance parameters for CNOT gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossResonanceParameters {
    /// Drive frequency (Hz)
    pub drive_frequency: f64,
    /// Drive amplitude
    pub drive_amplitude: f64,
    /// Pulse duration (ns)
    pub pulse_duration: f64,
    /// Echo pulse parameters
    pub echo_amplitude: f64,
    pub echo_duration: f64,
    /// ZX interaction rate (MHz)
    pub zx_interaction_rate: f64,
}

/// Gate decomposition for multi-qubit gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDecomposition {
    /// Sequence of gates in decomposition
    pub gates: Vec<DecomposedGate>,
    /// Total error from decomposition
    pub decomposition_error: f64,
    /// Optimal decomposition for this device
    pub is_optimal: bool,
}

/// Individual gate in a decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposedGate {
    /// Gate name
    pub gate_name: String,
    /// Qubits acted on
    pub qubits: Vec<QubitId>,
    /// Parameters (if any)
    pub parameters: Vec<f64>,
}

/// Calibration manager for handling device calibrations
#[derive(Debug, Clone)]
pub struct CalibrationManager {
    /// Current calibrations for each device
    calibrations: HashMap<String, DeviceCalibration>,
    /// Calibration history
    history: Vec<(String, SystemTime, DeviceCalibration)>,
    /// Maximum history size
    max_history: usize,
}

impl CalibrationManager {
    /// Create a new calibration manager
    pub fn new() -> Self {
        Self {
            calibrations: HashMap::new(),
            history: Vec::new(),
            max_history: 100,
        }
    }

    /// Load calibration from file
    pub fn load_calibration(&mut self, path: &str) -> QuantRS2Result<()> {
        let data = std::fs::read_to_string(path).map_err(|e| {
            QuantRS2Error::InvalidInput(format!("Failed to read calibration: {}", e))
        })?;

        let calibration: DeviceCalibration = serde_json::from_str(&data).map_err(|e| {
            QuantRS2Error::InvalidInput(format!("Failed to parse calibration: {}", e))
        })?;

        self.update_calibration(calibration);
        Ok(())
    }

    /// Save calibration to file
    pub fn save_calibration(&self, device_id: &str, path: &str) -> QuantRS2Result<()> {
        let calibration = self.get_calibration(device_id).ok_or_else(|| {
            QuantRS2Error::InvalidInput(format!("No calibration for device {}", device_id))
        })?;

        let data = serde_json::to_string_pretty(calibration).map_err(|e| {
            QuantRS2Error::InvalidInput(format!("Failed to serialize calibration: {}", e))
        })?;

        std::fs::write(path, data).map_err(|e| {
            QuantRS2Error::InvalidInput(format!("Failed to write calibration: {}", e))
        })?;

        Ok(())
    }

    /// Update calibration for a device
    pub fn update_calibration(&mut self, calibration: DeviceCalibration) {
        let device_id = calibration.device_id.clone();
        let timestamp = calibration.timestamp;

        // Store in history
        if let Some(old_cal) = self.calibrations.get(&device_id) {
            self.history
                .push((device_id.clone(), timestamp, old_cal.clone()));

            // Trim history if needed
            if self.history.len() > self.max_history {
                self.history.remove(0);
            }
        }

        // Update current calibration
        self.calibrations.insert(device_id, calibration);
    }

    /// Get current calibration for a device
    pub fn get_calibration(&self, device_id: &str) -> Option<&DeviceCalibration> {
        self.calibrations.get(device_id)
    }

    /// Check if calibration is still valid
    pub fn is_calibration_valid(&self, device_id: &str) -> bool {
        if let Some(cal) = self.calibrations.get(device_id) {
            let elapsed = SystemTime::now()
                .duration_since(cal.timestamp)
                .unwrap_or(Duration::from_secs(u64::MAX));

            elapsed < cal.valid_duration
        } else {
            false
        }
    }

    /// Get the latest calibration across all devices
    pub fn get_latest_calibration(&self) -> Option<&DeviceCalibration> {
        self.calibrations.values().max_by_key(|cal| cal.timestamp)
    }

    /// Get gate fidelity for a specific gate on specific qubits
    pub fn get_gate_fidelity(
        &self,
        device_id: &str,
        gate_name: &str,
        qubits: &[QubitId],
    ) -> Option<f64> {
        let cal = self.calibrations.get(device_id)?;

        match qubits.len() {
            1 => {
                let gate_cal = cal.single_qubit_gates.get(gate_name)?;
                gate_cal.qubit_data.get(&qubits[0]).map(|d| d.fidelity)
            }
            2 => cal
                .two_qubit_gates
                .get(&(qubits[0], qubits[1]))
                .filter(|g| g.gate_name == gate_name)
                .map(|g| g.fidelity),
            _ => cal
                .multi_qubit_gates
                .get(qubits)
                .filter(|g| g.gate_name == gate_name)
                .map(|g| g.fidelity),
        }
    }

    /// Get gate duration
    pub fn get_gate_duration(
        &self,
        device_id: &str,
        gate_name: &str,
        qubits: &[QubitId],
    ) -> Option<f64> {
        let cal = self.calibrations.get(device_id)?;

        match qubits.len() {
            1 => {
                let gate_cal = cal.single_qubit_gates.get(gate_name)?;
                gate_cal.qubit_data.get(&qubits[0]).map(|d| d.duration)
            }
            2 => cal
                .two_qubit_gates
                .get(&(qubits[0], qubits[1]))
                .filter(|g| g.gate_name == gate_name)
                .map(|g| g.duration),
            _ => cal
                .multi_qubit_gates
                .get(qubits)
                .filter(|g| g.gate_name == gate_name)
                .map(|g| g.duration),
        }
    }
}

/// Builder for creating device calibrations
pub struct CalibrationBuilder {
    device_id: String,
    timestamp: SystemTime,
    valid_duration: Duration,
    qubit_calibrations: HashMap<QubitId, QubitCalibration>,
    single_qubit_gates: HashMap<String, SingleQubitGateCalibration>,
    two_qubit_gates: HashMap<(QubitId, QubitId), TwoQubitGateCalibration>,
    multi_qubit_gates: HashMap<Vec<QubitId>, MultiQubitGateCalibration>,
    readout_calibration: Option<ReadoutCalibration>,
    crosstalk_matrix: Option<CrosstalkMatrix>,
    topology: Option<DeviceTopology>,
    metadata: HashMap<String, String>,
}

impl CalibrationBuilder {
    /// Create a new calibration builder
    pub fn new(device_id: String) -> Self {
        Self {
            device_id,
            timestamp: SystemTime::now(),
            valid_duration: Duration::from_secs(24 * 3600), // 24 hours default
            qubit_calibrations: HashMap::new(),
            single_qubit_gates: HashMap::new(),
            two_qubit_gates: HashMap::new(),
            multi_qubit_gates: HashMap::new(),
            readout_calibration: None,
            crosstalk_matrix: None,
            topology: None,
            metadata: HashMap::new(),
        }
    }

    /// Set validity duration
    pub fn valid_duration(mut self, duration: Duration) -> Self {
        self.valid_duration = duration;
        self
    }

    /// Add qubit calibration
    pub fn add_qubit_calibration(mut self, calibration: QubitCalibration) -> Self {
        self.qubit_calibrations
            .insert(calibration.qubit_id, calibration);
        self
    }

    /// Add single-qubit gate calibration
    pub fn add_single_qubit_gate(
        mut self,
        gate_name: String,
        calibration: SingleQubitGateCalibration,
    ) -> Self {
        self.single_qubit_gates.insert(gate_name, calibration);
        self
    }

    /// Add two-qubit gate calibration
    pub fn add_two_qubit_gate(
        mut self,
        control: QubitId,
        target: QubitId,
        calibration: TwoQubitGateCalibration,
    ) -> Self {
        self.two_qubit_gates.insert((control, target), calibration);
        self
    }

    /// Set readout calibration
    pub fn readout_calibration(mut self, calibration: ReadoutCalibration) -> Self {
        self.readout_calibration = Some(calibration);
        self
    }

    /// Set crosstalk matrix
    pub fn crosstalk_matrix(mut self, matrix: CrosstalkMatrix) -> Self {
        self.crosstalk_matrix = Some(matrix);
        self
    }

    /// Set device topology
    pub fn topology(mut self, topology: DeviceTopology) -> Self {
        self.topology = Some(topology);
        self
    }

    /// Add metadata
    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Build the calibration
    pub fn build(self) -> QuantRS2Result<DeviceCalibration> {
        let readout_calibration = self
            .readout_calibration
            .ok_or_else(|| QuantRS2Error::InvalidInput("Readout calibration required".into()))?;

        let crosstalk_matrix = self
            .crosstalk_matrix
            .ok_or_else(|| QuantRS2Error::InvalidInput("Crosstalk matrix required".into()))?;

        let topology = self
            .topology
            .ok_or_else(|| QuantRS2Error::InvalidInput("Device topology required".into()))?;

        Ok(DeviceCalibration {
            device_id: self.device_id,
            timestamp: self.timestamp,
            valid_duration: self.valid_duration,
            qubit_calibrations: self.qubit_calibrations,
            single_qubit_gates: self.single_qubit_gates,
            two_qubit_gates: self.two_qubit_gates,
            multi_qubit_gates: self.multi_qubit_gates,
            readout_calibration,
            crosstalk_matrix,
            topology,
            metadata: self.metadata,
        })
    }
}

/// Create a default calibration for ideal simulation
pub fn create_ideal_calibration(device_id: String, num_qubits: usize) -> DeviceCalibration {
    let mut builder = CalibrationBuilder::new(device_id);

    // Add ideal qubit calibrations
    for i in 0..num_qubits {
        let qubit_id = QubitId(i as u32);
        builder = builder.add_qubit_calibration(QubitCalibration {
            qubit_id,
            frequency: 5e9,          // 5 GHz
            anharmonicity: -300e6,   // -300 MHz
            t1: 100_000.0,           // 100 μs
            t2: 100_000.0,           // 100 μs
            t2_star: Some(50_000.0), // 50 μs
            readout_error: 0.001,    // 0.1%
            thermal_population: 0.01,
            temperature: Some(20.0), // 20 mK
            parameters: HashMap::new(),
        });
    }

    // Add ideal single-qubit gates
    for gate_name in ["X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ"] {
        let mut qubit_data = HashMap::new();

        for i in 0..num_qubits {
            qubit_data.insert(
                QubitId(i as u32),
                SingleQubitGateData {
                    error_rate: 0.001,
                    fidelity: 0.999,
                    duration: 20.0, // 20 ns
                    amplitude: 1.0,
                    frequency: 5e9,
                    phase: 0.0,
                    pulse_shape: PulseShape::GaussianDRAG {
                        sigma: 5.0,
                        beta: 0.5,
                        cutoff: 2.0,
                    },
                    calibrated_matrix: None,
                    parameter_calibrations: None,
                },
            );
        }

        builder = builder.add_single_qubit_gate(
            gate_name.to_string(),
            SingleQubitGateCalibration {
                gate_name: gate_name.to_string(),
                qubit_data,
                default_parameters: GateParameters {
                    amplitude_scale: 1.0,
                    phase_offset: 0.0,
                    duration_scale: 1.0,
                    drag_coefficient: Some(0.5),
                    custom_parameters: HashMap::new(),
                },
            },
        );
    }

    // Add ideal two-qubit gates (nearest neighbor)
    for i in 0..num_qubits - 1 {
        let control = QubitId(i as u32);
        let target = QubitId((i + 1) as u32);

        builder = builder.add_two_qubit_gate(
            control,
            target,
            TwoQubitGateCalibration {
                gate_name: "CNOT".to_string(),
                control,
                target,
                error_rate: 0.01,
                fidelity: 0.99,
                duration: 200.0,         // 200 ns
                coupling_strength: 30.0, // 30 MHz
                cross_resonance: Some(CrossResonanceParameters {
                    drive_frequency: 4.8e9,
                    drive_amplitude: 0.5,
                    pulse_duration: 180.0,
                    echo_amplitude: 0.25,
                    echo_duration: 90.0,
                    zx_interaction_rate: 3.0,
                }),
                calibrated_matrix: None,
                directional: true,
                reversed_calibration: None,
            },
        );
    }

    // Add ideal readout
    let mut qubit_readout = HashMap::new();
    for i in 0..num_qubits {
        qubit_readout.insert(
            QubitId(i as u32),
            QubitReadoutData {
                p0_given_0: 0.999,
                p1_given_1: 0.999,
                resonator_frequency: 6.5e9,
                readout_amplitude: 0.1,
                readout_phase: 0.0,
                snr: 10.0,
            },
        );
    }

    builder = builder.readout_calibration(ReadoutCalibration {
        qubit_readout,
        mitigation_matrix: None,
        duration: 2000.0,         // 2 μs
        integration_time: 1500.0, // 1.5 μs
    });

    // Add ideal crosstalk (none)
    let mut matrix = vec![vec![0.0; num_qubits]; num_qubits];
    for i in 0..num_qubits {
        matrix[i][i] = 1.0;
    }

    builder = builder.crosstalk_matrix(CrosstalkMatrix {
        matrix,
        measurement_method: "Ideal".to_string(),
        significance_threshold: 0.01,
    });

    // Add linear topology
    let mut coupling_map = Vec::new();
    for i in 0..num_qubits - 1 {
        coupling_map.push((QubitId(i as u32), QubitId((i + 1) as u32)));
    }

    builder = builder.topology(DeviceTopology {
        num_qubits,
        coupling_map,
        layout_type: "linear".to_string(),
        qubit_coordinates: None,
    });

    builder.build().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_builder() {
        let cal = CalibrationBuilder::new("test_device".to_string())
            .add_qubit_calibration(QubitCalibration {
                qubit_id: QubitId(0),
                frequency: 5e9,
                anharmonicity: -300e6,
                t1: 50_000.0,
                t2: 40_000.0,
                t2_star: Some(30_000.0),
                readout_error: 0.02,
                thermal_population: 0.02,
                temperature: Some(15.0),
                parameters: HashMap::new(),
            })
            .readout_calibration(ReadoutCalibration {
                qubit_readout: HashMap::new(),
                mitigation_matrix: None,
                duration: 2000.0,
                integration_time: 1500.0,
            })
            .crosstalk_matrix(CrosstalkMatrix {
                matrix: vec![vec![1.0]],
                measurement_method: "Test".to_string(),
                significance_threshold: 0.01,
            })
            .topology(DeviceTopology {
                num_qubits: 1,
                coupling_map: vec![],
                layout_type: "single".to_string(),
                qubit_coordinates: None,
            })
            .build()
            .unwrap();

        assert_eq!(cal.device_id, "test_device");
        assert_eq!(cal.qubit_calibrations.len(), 1);
    }

    #[test]
    fn test_calibration_manager() {
        let mut manager = CalibrationManager::new();
        let cal = create_ideal_calibration("test_device".to_string(), 5);

        manager.update_calibration(cal);

        assert!(manager.is_calibration_valid("test_device"));
        assert_eq!(
            manager.get_gate_fidelity("test_device", "X", &[QubitId(0)]),
            Some(0.999)
        );
        assert_eq!(
            manager.get_gate_duration("test_device", "CNOT", &[QubitId(0), QubitId(1)]),
            Some(200.0)
        );
    }

    #[test]
    fn test_ideal_calibration() {
        let cal = create_ideal_calibration("ideal".to_string(), 10);

        assert_eq!(cal.qubit_calibrations.len(), 10);
        assert!(cal.single_qubit_gates.contains_key("X"));
        assert!(cal.single_qubit_gates.contains_key("RZ"));
        assert_eq!(cal.topology.coupling_map.len(), 9); // 9 connections for 10 qubits
    }
}
