//! IBM Quantum calibration data and backend properties.
//!
//! This module provides access to IBM Quantum backend calibration data,
//! including gate error rates, T1/T2 times, readout errors, and more.
//!
//! ## Example
//!
//! ```rust,ignore
//! use quantrs2_device::ibm_calibration::{CalibrationData, QubitProperties, GateProperties};
//!
//! // Get calibration data for a backend
//! let calibration = CalibrationData::fetch(&client, "ibm_brisbane").await?;
//!
//! // Check qubit coherence times
//! let t1 = calibration.qubit(0).t1();
//! let t2 = calibration.qubit(0).t2();
//!
//! // Get gate error rates
//! let cx_error = calibration.gate_error("cx", &[0, 1])?;
//!
//! // Find best qubits
//! let best_qubits = calibration.best_qubits(5)?;
//! ```

use std::collections::HashMap;
#[cfg(feature = "ibm")]
use std::time::SystemTime;

use crate::{DeviceError, DeviceResult};

/// Calibration data for an IBM Quantum backend
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// Backend name
    pub backend_name: String,
    /// Calibration timestamp
    pub last_update_date: String,
    /// Qubit properties
    pub qubits: Vec<QubitCalibration>,
    /// Gate calibration data
    pub gates: HashMap<String, Vec<GateCalibration>>,
    /// General backend properties
    pub general: GeneralProperties,
}

impl CalibrationData {
    /// Create calibration data from a backend
    #[cfg(feature = "ibm")]
    pub async fn fetch(
        client: &crate::ibm::IBMQuantumClient,
        backend_name: &str,
    ) -> DeviceResult<Self> {
        // In a real implementation, this would fetch from IBM Quantum API
        // For now, create placeholder data
        let backend = client.get_backend(backend_name).await?;

        let mut qubits = Vec::new();
        for i in 0..backend.n_qubits {
            qubits.push(QubitCalibration {
                qubit_id: i,
                t1: Duration::from_micros(100 + (i as u64 * 5)), // Placeholder
                t2: Duration::from_micros(80 + (i as u64 * 3)),
                frequency: 5.0 + (i as f64 * 0.1), // GHz
                anharmonicity: -0.34,              // GHz
                readout_error: 0.01 + (i as f64 * 0.001),
                readout_length: Duration::from_nanos(500),
                prob_meas0_prep1: 0.02,
                prob_meas1_prep0: 0.01,
            });
        }

        let mut gates = HashMap::new();

        // Single-qubit gates
        let mut sx_gates = Vec::new();
        let mut x_gates = Vec::new();
        let mut rz_gates = Vec::new();

        for i in 0..backend.n_qubits {
            sx_gates.push(GateCalibration {
                gate_name: "sx".to_string(),
                qubits: vec![i],
                gate_error: 0.0002 + (i as f64 * 0.00001),
                gate_length: Duration::from_nanos(35),
                parameters: HashMap::new(),
            });

            x_gates.push(GateCalibration {
                gate_name: "x".to_string(),
                qubits: vec![i],
                gate_error: 0.0003 + (i as f64 * 0.00001),
                gate_length: Duration::from_nanos(35),
                parameters: HashMap::new(),
            });

            rz_gates.push(GateCalibration {
                gate_name: "rz".to_string(),
                qubits: vec![i],
                gate_error: 0.0, // Virtual gate
                gate_length: Duration::from_nanos(0),
                parameters: HashMap::new(),
            });
        }

        gates.insert("sx".to_string(), sx_gates);
        gates.insert("x".to_string(), x_gates);
        gates.insert("rz".to_string(), rz_gates);

        // Two-qubit gates (CX) for connected pairs
        let mut cx_gates = Vec::new();
        for i in 0..backend.n_qubits.saturating_sub(1) {
            cx_gates.push(GateCalibration {
                gate_name: "cx".to_string(),
                qubits: vec![i, i + 1],
                gate_error: 0.005 + (i as f64 * 0.0005),
                gate_length: Duration::from_nanos(300),
                parameters: HashMap::new(),
            });
        }
        gates.insert("cx".to_string(), cx_gates);

        Ok(Self {
            backend_name: backend_name.to_string(),
            last_update_date: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs().to_string())
                .unwrap_or_else(|_| "0".to_string()),
            qubits,
            gates,
            general: GeneralProperties {
                backend_name: backend_name.to_string(),
                backend_version: backend.version,
                n_qubits: backend.n_qubits,
                basis_gates: vec![
                    "id".to_string(),
                    "rz".to_string(),
                    "sx".to_string(),
                    "x".to_string(),
                    "cx".to_string(),
                ],
                supported_instructions: vec![
                    "cx".to_string(),
                    "id".to_string(),
                    "rz".to_string(),
                    "sx".to_string(),
                    "x".to_string(),
                    "measure".to_string(),
                    "reset".to_string(),
                    "delay".to_string(),
                ],
                local: false,
                simulator: backend.simulator,
                conditional: true,
                open_pulse: true,
                memory: true,
                max_shots: 100000,
                coupling_map: (0..backend.n_qubits.saturating_sub(1))
                    .map(|i| (i, i + 1))
                    .collect(),
                dynamic_reprate_enabled: true,
                rep_delay_range: (0.0, 500.0),
                default_rep_delay: 250.0,
                max_experiments: 300,
                processor_type: ProcessorType::Eagle,
            },
        })
    }

    #[cfg(not(feature = "ibm"))]
    pub async fn fetch(
        _client: &crate::ibm::IBMQuantumClient,
        backend_name: &str,
    ) -> DeviceResult<Self> {
        Err(DeviceError::UnsupportedDevice(format!(
            "IBM support not enabled for {}",
            backend_name
        )))
    }

    /// Get qubit calibration data
    pub fn qubit(&self, qubit_id: usize) -> Option<&QubitCalibration> {
        self.qubits.get(qubit_id)
    }

    /// Get gate error rate
    pub fn gate_error(&self, gate_name: &str, qubits: &[usize]) -> Option<f64> {
        self.gates.get(gate_name).and_then(|gates| {
            gates
                .iter()
                .find(|g| g.qubits == qubits)
                .map(|g| g.gate_error)
        })
    }

    /// Get gate length
    pub fn gate_length(&self, gate_name: &str, qubits: &[usize]) -> Option<Duration> {
        self.gates.get(gate_name).and_then(|gates| {
            gates
                .iter()
                .find(|g| g.qubits == qubits)
                .map(|g| g.gate_length)
        })
    }

    /// Find the best N qubits based on T1, T2, and readout error
    pub fn best_qubits(&self, n: usize) -> DeviceResult<Vec<usize>> {
        if n > self.qubits.len() {
            return Err(DeviceError::InvalidInput(format!(
                "Requested {} qubits but only {} available",
                n,
                self.qubits.len()
            )));
        }

        let mut scored_qubits: Vec<(usize, f64)> = self
            .qubits
            .iter()
            .enumerate()
            .map(|(i, q)| {
                // Score based on T1, T2 (higher is better) and readout error (lower is better)
                let t1_score = q.t1.as_micros() as f64 / 200.0; // Normalize to ~1
                let t2_score = q.t2.as_micros() as f64 / 150.0;
                let readout_score = 1.0 - q.readout_error * 10.0; // Invert and scale
                let score = t1_score + t2_score + readout_score;
                (i, score)
            })
            .collect();

        scored_qubits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored_qubits.into_iter().take(n).map(|(i, _)| i).collect())
    }

    /// Find the best connected qubit pairs for two-qubit gates
    pub fn best_cx_pairs(&self, n: usize) -> DeviceResult<Vec<(usize, usize)>> {
        let cx_gates = self
            .gates
            .get("cx")
            .ok_or_else(|| DeviceError::CalibrationError("No CX gate data".to_string()))?;

        let mut scored_pairs: Vec<((usize, usize), f64)> = cx_gates
            .iter()
            .filter_map(|g| {
                if g.qubits.len() == 2 {
                    Some(((g.qubits[0], g.qubits[1]), 1.0 - g.gate_error * 100.0))
                } else {
                    None
                }
            })
            .collect();

        scored_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored_pairs
            .into_iter()
            .take(n)
            .map(|(pair, _)| pair)
            .collect())
    }

    /// Calculate expected circuit fidelity based on calibration data
    pub fn estimate_circuit_fidelity(&self, gates: &[(String, Vec<usize>)]) -> f64 {
        let mut fidelity = 1.0;

        for (gate_name, qubits) in gates {
            if let Some(error) = self.gate_error(gate_name, qubits) {
                fidelity *= 1.0 - error;
            }
        }

        // Account for readout errors
        let used_qubits: std::collections::HashSet<usize> =
            gates.iter().flat_map(|(_, q)| q.iter().copied()).collect();

        for qubit in used_qubits {
            if let Some(q) = self.qubit(qubit) {
                fidelity *= 1.0 - q.readout_error;
            }
        }

        fidelity
    }

    /// Get average single-qubit gate error
    pub fn avg_single_qubit_error(&self) -> f64 {
        let sx_gates = self.gates.get("sx");
        if let Some(gates) = sx_gates {
            let total: f64 = gates.iter().map(|g| g.gate_error).sum();
            total / gates.len() as f64
        } else {
            0.0
        }
    }

    /// Get average two-qubit gate error
    pub fn avg_two_qubit_error(&self) -> f64 {
        let cx_gates = self.gates.get("cx");
        if let Some(gates) = cx_gates {
            let total: f64 = gates.iter().map(|g| g.gate_error).sum();
            total / gates.len() as f64
        } else {
            0.0
        }
    }

    /// Get average T1 time
    pub fn avg_t1(&self) -> Duration {
        if self.qubits.is_empty() {
            return Duration::from_secs(0);
        }
        let total: u128 = self.qubits.iter().map(|q| q.t1.as_micros()).sum();
        Duration::from_micros((total / self.qubits.len() as u128) as u64)
    }

    /// Get average T2 time
    pub fn avg_t2(&self) -> Duration {
        if self.qubits.is_empty() {
            return Duration::from_secs(0);
        }
        let total: u128 = self.qubits.iter().map(|q| q.t2.as_micros()).sum();
        Duration::from_micros((total / self.qubits.len() as u128) as u64)
    }

    /// Get average readout error
    pub fn avg_readout_error(&self) -> f64 {
        if self.qubits.is_empty() {
            return 0.0;
        }
        let total: f64 = self.qubits.iter().map(|q| q.readout_error).sum();
        total / self.qubits.len() as f64
    }
}

/// Duration type alias for calibration data
pub type Duration = std::time::Duration;

/// Calibration data for a single qubit
#[derive(Debug, Clone)]
pub struct QubitCalibration {
    /// Qubit identifier
    pub qubit_id: usize,
    /// T1 relaxation time
    pub t1: Duration,
    /// T2 dephasing time
    pub t2: Duration,
    /// Qubit frequency in GHz
    pub frequency: f64,
    /// Anharmonicity in GHz
    pub anharmonicity: f64,
    /// Readout assignment error
    pub readout_error: f64,
    /// Readout duration
    pub readout_length: Duration,
    /// Probability of measuring 0 when prepared in 1
    pub prob_meas0_prep1: f64,
    /// Probability of measuring 1 when prepared in 0
    pub prob_meas1_prep0: f64,
}

impl QubitCalibration {
    /// Get T1 time in microseconds
    pub fn t1_us(&self) -> f64 {
        self.t1.as_micros() as f64
    }

    /// Get T2 time in microseconds
    pub fn t2_us(&self) -> f64 {
        self.t2.as_micros() as f64
    }

    /// Calculate quality score (0-1, higher is better)
    pub fn quality_score(&self) -> f64 {
        let t1_score = (self.t1.as_micros() as f64 / 200.0).min(1.0);
        let t2_score = (self.t2.as_micros() as f64 / 150.0).min(1.0);
        let readout_score = 1.0 - self.readout_error.min(1.0);
        (t1_score + t2_score + readout_score) / 3.0
    }
}

/// Calibration data for a gate
#[derive(Debug, Clone)]
pub struct GateCalibration {
    /// Gate name
    pub gate_name: String,
    /// Qubits this gate acts on
    pub qubits: Vec<usize>,
    /// Gate error rate
    pub gate_error: f64,
    /// Gate duration
    pub gate_length: Duration,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
}

impl GateCalibration {
    /// Get gate length in nanoseconds
    pub fn gate_length_ns(&self) -> f64 {
        self.gate_length.as_nanos() as f64
    }

    /// Calculate gate fidelity (1 - error)
    pub fn fidelity(&self) -> f64 {
        1.0 - self.gate_error
    }
}

/// General backend properties
#[derive(Debug, Clone)]
pub struct GeneralProperties {
    /// Backend name
    pub backend_name: String,
    /// Backend version
    pub backend_version: String,
    /// Number of qubits
    pub n_qubits: usize,
    /// Basis gates supported
    pub basis_gates: Vec<String>,
    /// All supported instructions
    pub supported_instructions: Vec<String>,
    /// Whether the backend runs locally
    pub local: bool,
    /// Whether this is a simulator
    pub simulator: bool,
    /// Whether conditional operations are supported
    pub conditional: bool,
    /// Whether OpenPulse is supported
    pub open_pulse: bool,
    /// Whether memory (mid-circuit measurement) is supported
    pub memory: bool,
    /// Maximum number of shots per job
    pub max_shots: usize,
    /// Coupling map (connected qubit pairs)
    pub coupling_map: Vec<(usize, usize)>,
    /// Whether dynamic repetition rate is enabled
    pub dynamic_reprate_enabled: bool,
    /// Repetition delay range in microseconds
    pub rep_delay_range: (f64, f64),
    /// Default repetition delay in microseconds
    pub default_rep_delay: f64,
    /// Maximum number of experiments per job
    pub max_experiments: usize,
    /// Processor type
    pub processor_type: ProcessorType,
}

/// IBM Quantum processor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorType {
    /// Falcon processor (7-27 qubits)
    Falcon,
    /// Hummingbird processor (65 qubits)
    Hummingbird,
    /// Eagle processor (127 qubits)
    Eagle,
    /// Osprey processor (433 qubits)
    Osprey,
    /// Condor processor (1121 qubits)
    Condor,
    /// Simulator
    Simulator,
    /// Unknown processor type
    Unknown,
}

impl ProcessorType {
    /// Get typical T1 time for this processor type
    pub fn typical_t1(&self) -> Duration {
        match self {
            Self::Falcon => Duration::from_micros(80),
            Self::Hummingbird => Duration::from_micros(100),
            Self::Eagle => Duration::from_micros(150),
            Self::Osprey => Duration::from_micros(200),
            Self::Condor => Duration::from_micros(200),
            Self::Simulator | Self::Unknown => Duration::from_micros(100),
        }
    }

    /// Get typical two-qubit gate error for this processor type
    pub fn typical_cx_error(&self) -> f64 {
        match self {
            Self::Falcon => 0.01,
            Self::Hummingbird => 0.008,
            Self::Eagle => 0.005,
            Self::Osprey => 0.004,
            Self::Condor => 0.003,
            Self::Simulator => 0.0,
            Self::Unknown => 0.01,
        }
    }
}

// =============================================================================
// Pulse Calibration Write Support
// =============================================================================

/// Custom pulse calibration definition for a gate
#[derive(Debug, Clone)]
pub struct CustomCalibration {
    /// Gate name (e.g., "x", "sx", "cx", "custom_gate")
    pub gate_name: String,
    /// Target qubits for this calibration
    pub qubits: Vec<usize>,
    /// Pulse schedule definition
    pub pulse_schedule: PulseSchedule,
    /// Parameters that can be varied (e.g., rotation angles)
    pub parameters: Vec<String>,
    /// Description of this calibration
    pub description: Option<String>,
}

/// Pulse schedule definition
#[derive(Debug, Clone)]
pub struct PulseSchedule {
    /// Name of this schedule
    pub name: String,
    /// Sequence of pulse instructions
    pub instructions: Vec<PulseInstruction>,
    /// Total duration in dt (device time units)
    pub duration_dt: u64,
    /// Sample rate in Hz
    pub dt: f64,
}

/// Individual pulse instruction
#[derive(Debug, Clone)]
pub enum PulseInstruction {
    /// Play a pulse on a channel
    Play {
        /// Pulse waveform
        pulse: PulseWaveform,
        /// Target channel
        channel: PulseChannel,
        /// Start time in dt
        t0: u64,
        /// Name identifier
        name: Option<String>,
    },
    /// Set frequency of a channel
    SetFrequency {
        /// Frequency in Hz
        frequency: f64,
        /// Target channel
        channel: PulseChannel,
        /// Start time in dt
        t0: u64,
    },
    /// Shift frequency of a channel
    ShiftFrequency {
        /// Frequency shift in Hz
        frequency: f64,
        /// Target channel
        channel: PulseChannel,
        /// Start time in dt
        t0: u64,
    },
    /// Set phase of a channel
    SetPhase {
        /// Phase in radians
        phase: f64,
        /// Target channel
        channel: PulseChannel,
        /// Start time in dt
        t0: u64,
    },
    /// Shift phase of a channel
    ShiftPhase {
        /// Phase shift in radians
        phase: f64,
        /// Target channel
        channel: PulseChannel,
        /// Start time in dt
        t0: u64,
    },
    /// Delay on a channel
    Delay {
        /// Duration in dt
        duration: u64,
        /// Target channel
        channel: PulseChannel,
        /// Start time in dt
        t0: u64,
    },
    /// Acquire measurement data
    Acquire {
        /// Duration in dt
        duration: u64,
        /// Qubit index
        qubit: usize,
        /// Memory slot
        memory_slot: usize,
        /// Start time in dt
        t0: u64,
    },
    /// Barrier across channels
    Barrier {
        /// Channels to synchronize
        channels: Vec<PulseChannel>,
        /// Start time in dt
        t0: u64,
    },
}

/// Pulse waveform types
#[derive(Debug, Clone)]
pub enum PulseWaveform {
    /// Gaussian pulse
    Gaussian {
        /// Amplitude (complex, represented as (real, imag))
        amp: (f64, f64),
        /// Duration in dt
        duration: u64,
        /// Standard deviation in dt
        sigma: f64,
        /// Optional name
        name: Option<String>,
    },
    /// Gaussian square (flat-top) pulse
    GaussianSquare {
        /// Amplitude
        amp: (f64, f64),
        /// Duration in dt
        duration: u64,
        /// Standard deviation for rise/fall
        sigma: f64,
        /// Flat-top width in dt
        width: u64,
        /// Rise/fall shape: "gaussian" or "cos"
        risefall_shape: String,
        /// Optional name
        name: Option<String>,
    },
    /// DRAG (Derivative Removal by Adiabatic Gate) pulse
    Drag {
        /// Amplitude
        amp: (f64, f64),
        /// Duration in dt
        duration: u64,
        /// Standard deviation in dt
        sigma: f64,
        /// DRAG coefficient (beta)
        beta: f64,
        /// Optional name
        name: Option<String>,
    },
    /// Constant pulse
    Constant {
        /// Amplitude
        amp: (f64, f64),
        /// Duration in dt
        duration: u64,
        /// Optional name
        name: Option<String>,
    },
    /// Custom waveform from samples
    Waveform {
        /// Complex samples (real, imag) pairs
        samples: Vec<(f64, f64)>,
        /// Optional name
        name: Option<String>,
    },
}

impl PulseWaveform {
    /// Get the duration of this waveform in dt
    pub fn duration(&self) -> u64 {
        match self {
            Self::Gaussian { duration, .. } => *duration,
            Self::GaussianSquare { duration, .. } => *duration,
            Self::Drag { duration, .. } => *duration,
            Self::Constant { duration, .. } => *duration,
            Self::Waveform { samples, .. } => samples.len() as u64,
        }
    }

    /// Get the name of this waveform
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Gaussian { name, .. } => name.as_deref(),
            Self::GaussianSquare { name, .. } => name.as_deref(),
            Self::Drag { name, .. } => name.as_deref(),
            Self::Constant { name, .. } => name.as_deref(),
            Self::Waveform { name, .. } => name.as_deref(),
        }
    }
}

/// Pulse channel types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PulseChannel {
    /// Drive channel for single-qubit gates
    Drive(usize),
    /// Control channel for two-qubit gates
    Control(usize),
    /// Measure channel for readout
    Measure(usize),
    /// Acquire channel for data acquisition
    Acquire(usize),
}

impl std::fmt::Display for PulseChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Drive(idx) => write!(f, "d{}", idx),
            Self::Control(idx) => write!(f, "u{}", idx),
            Self::Measure(idx) => write!(f, "m{}", idx),
            Self::Acquire(idx) => write!(f, "a{}", idx),
        }
    }
}

/// Validation result for calibration
#[derive(Debug, Clone)]
pub struct CalibrationValidation {
    /// Whether the calibration is valid
    pub is_valid: bool,
    /// Warning messages
    pub warnings: Vec<String>,
    /// Error messages
    pub errors: Vec<String>,
}

impl CalibrationValidation {
    /// Create a valid result
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            warnings: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Add a warning
    pub fn add_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    /// Add an error
    pub fn add_error(&mut self, msg: impl Into<String>) {
        self.is_valid = false;
        self.errors.push(msg.into());
    }
}

/// Backend constraints for pulse calibrations
#[derive(Debug, Clone)]
pub struct PulseBackendConstraints {
    /// Maximum amplitude (typically 1.0)
    pub max_amplitude: f64,
    /// Minimum pulse duration in dt
    pub min_pulse_duration: u64,
    /// Maximum pulse duration in dt
    pub max_pulse_duration: u64,
    /// Pulse granularity (must be multiple of this)
    pub pulse_granularity: u64,
    /// Available drive channels
    pub drive_channels: Vec<usize>,
    /// Available control channels
    pub control_channels: Vec<usize>,
    /// Available measure channels
    pub measure_channels: Vec<usize>,
    /// Qubit frequency limits (min, max) in GHz
    pub frequency_range: (f64, f64),
    /// Device time unit (dt) in seconds
    pub dt_seconds: f64,
    /// Supported pulse waveform types
    pub supported_waveforms: Vec<String>,
}

impl Default for PulseBackendConstraints {
    fn default() -> Self {
        Self {
            max_amplitude: 1.0,
            min_pulse_duration: 16,
            max_pulse_duration: 16384,
            pulse_granularity: 16,
            drive_channels: (0..127).collect(),
            control_channels: (0..127).collect(),
            measure_channels: (0..127).collect(),
            frequency_range: (4.5, 5.5),
            dt_seconds: 2.22e-10, // ~4.5 GHz sampling rate
            supported_waveforms: vec![
                "Gaussian".to_string(),
                "GaussianSquare".to_string(),
                "Drag".to_string(),
                "Constant".to_string(),
                "Waveform".to_string(),
            ],
        }
    }
}

/// Manager for custom pulse calibrations
#[derive(Debug, Clone)]
pub struct CalibrationManager {
    /// Backend name
    pub backend_name: String,
    /// Custom calibrations
    pub custom_calibrations: Vec<CustomCalibration>,
    /// Backend constraints
    pub constraints: PulseBackendConstraints,
    /// Default calibrations from backend
    pub defaults: Option<CalibrationData>,
}

impl CalibrationManager {
    /// Create a new CalibrationManager for a backend
    pub fn new(backend_name: impl Into<String>) -> Self {
        Self {
            backend_name: backend_name.into(),
            custom_calibrations: Vec::new(),
            constraints: PulseBackendConstraints::default(),
            defaults: None,
        }
    }

    /// Create with calibration data from backend
    pub fn with_defaults(backend_name: impl Into<String>, defaults: CalibrationData) -> Self {
        let mut manager = Self::new(backend_name);
        manager.defaults = Some(defaults);
        manager
    }

    /// Set backend constraints
    pub fn with_constraints(mut self, constraints: PulseBackendConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Add a custom calibration
    pub fn add_calibration(&mut self, calibration: CustomCalibration) -> DeviceResult<()> {
        let validation = self.validate_calibration(&calibration)?;
        if !validation.is_valid {
            return Err(DeviceError::CalibrationError(format!(
                "Invalid calibration: {}",
                validation.errors.join(", ")
            )));
        }
        self.custom_calibrations.push(calibration);
        Ok(())
    }

    /// Remove a custom calibration by gate name and qubits
    pub fn remove_calibration(&mut self, gate_name: &str, qubits: &[usize]) -> bool {
        let initial_len = self.custom_calibrations.len();
        self.custom_calibrations
            .retain(|c| !(c.gate_name == gate_name && c.qubits == qubits));
        self.custom_calibrations.len() < initial_len
    }

    /// Get a custom calibration
    pub fn get_calibration(&self, gate_name: &str, qubits: &[usize]) -> Option<&CustomCalibration> {
        self.custom_calibrations
            .iter()
            .find(|c| c.gate_name == gate_name && c.qubits == qubits)
    }

    /// Validate a calibration against backend constraints
    pub fn validate_calibration(
        &self,
        calibration: &CustomCalibration,
    ) -> DeviceResult<CalibrationValidation> {
        let mut result = CalibrationValidation::valid();

        // Check qubits are within range
        if let Some(defaults) = &self.defaults {
            for &qubit in &calibration.qubits {
                if qubit >= defaults.qubits.len() {
                    result.add_error(format!(
                        "Qubit {} is out of range (max {})",
                        qubit,
                        defaults.qubits.len() - 1
                    ));
                }
            }
        }

        // Check pulse schedule
        let schedule = &calibration.pulse_schedule;

        // Validate duration is within limits
        if schedule.duration_dt < self.constraints.min_pulse_duration {
            result.add_error(format!(
                "Schedule duration {} dt is below minimum {} dt",
                schedule.duration_dt, self.constraints.min_pulse_duration
            ));
        }

        if schedule.duration_dt > self.constraints.max_pulse_duration {
            result.add_error(format!(
                "Schedule duration {} dt exceeds maximum {} dt",
                schedule.duration_dt, self.constraints.max_pulse_duration
            ));
        }

        // Validate each instruction
        for instruction in &schedule.instructions {
            self.validate_instruction(instruction, &mut result);
        }

        Ok(result)
    }

    /// Validate a single pulse instruction
    fn validate_instruction(
        &self,
        instruction: &PulseInstruction,
        result: &mut CalibrationValidation,
    ) {
        match instruction {
            PulseInstruction::Play { pulse, channel, .. } => {
                // Check amplitude
                let amp = match pulse {
                    PulseWaveform::Gaussian { amp, .. } => *amp,
                    PulseWaveform::GaussianSquare { amp, .. } => *amp,
                    PulseWaveform::Drag { amp, .. } => *amp,
                    PulseWaveform::Constant { amp, .. } => *amp,
                    PulseWaveform::Waveform { samples, .. } => {
                        // Check all samples
                        for (i, sample) in samples.iter().enumerate() {
                            let magnitude = (sample.0 * sample.0 + sample.1 * sample.1).sqrt();
                            if magnitude > self.constraints.max_amplitude {
                                result.add_error(format!(
                                    "Waveform sample {} has amplitude {:.4} exceeding max {:.4}",
                                    i, magnitude, self.constraints.max_amplitude
                                ));
                            }
                        }
                        (0.0, 0.0) // Already checked
                    }
                };

                let magnitude = (amp.0 * amp.0 + amp.1 * amp.1).sqrt();
                if magnitude > self.constraints.max_amplitude {
                    result.add_error(format!(
                        "Pulse amplitude {:.4} exceeds maximum {:.4}",
                        magnitude, self.constraints.max_amplitude
                    ));
                }

                // Validate channel exists
                self.validate_channel(channel, result);

                // Check pulse duration granularity
                let duration = pulse.duration();
                if duration % self.constraints.pulse_granularity != 0 {
                    result.add_warning(format!(
                        "Pulse duration {} dt is not a multiple of granularity {} dt",
                        duration, self.constraints.pulse_granularity
                    ));
                }
            }
            PulseInstruction::SetFrequency {
                frequency, channel, ..
            } => {
                let freq_ghz = frequency / 1e9;
                if freq_ghz < self.constraints.frequency_range.0
                    || freq_ghz > self.constraints.frequency_range.1
                {
                    result.add_error(format!(
                        "Frequency {:.3} GHz is outside allowed range ({:.3}, {:.3}) GHz",
                        freq_ghz,
                        self.constraints.frequency_range.0,
                        self.constraints.frequency_range.1
                    ));
                }
                self.validate_channel(channel, result);
            }
            PulseInstruction::ShiftFrequency { channel, .. } => {
                self.validate_channel(channel, result);
            }
            PulseInstruction::SetPhase { channel, .. } => {
                self.validate_channel(channel, result);
            }
            PulseInstruction::ShiftPhase { channel, .. } => {
                self.validate_channel(channel, result);
            }
            PulseInstruction::Delay {
                duration, channel, ..
            } => {
                if *duration > self.constraints.max_pulse_duration {
                    result.add_warning(format!("Delay duration {} dt may be too long", duration));
                }
                self.validate_channel(channel, result);
            }
            PulseInstruction::Acquire { qubit, .. } => {
                if let Some(defaults) = &self.defaults {
                    if *qubit >= defaults.qubits.len() {
                        result.add_error(format!("Acquire qubit {} is out of range", qubit));
                    }
                }
            }
            PulseInstruction::Barrier { channels, .. } => {
                for channel in channels {
                    self.validate_channel(channel, result);
                }
            }
        }
    }

    /// Validate a pulse channel
    fn validate_channel(&self, channel: &PulseChannel, result: &mut CalibrationValidation) {
        match channel {
            PulseChannel::Drive(idx) => {
                if !self.constraints.drive_channels.contains(idx) {
                    result.add_error(format!("Drive channel d{} is not available", idx));
                }
            }
            PulseChannel::Control(idx) => {
                if !self.constraints.control_channels.contains(idx) {
                    result.add_error(format!("Control channel u{} is not available", idx));
                }
            }
            PulseChannel::Measure(idx) => {
                if !self.constraints.measure_channels.contains(idx) {
                    result.add_error(format!("Measure channel m{} is not available", idx));
                }
            }
            PulseChannel::Acquire(_) => {
                // Acquire channels are usually same as measure
            }
        }
    }

    /// Generate QASM 3.0 defcal statements for custom calibrations
    pub fn generate_defcal_statements(&self) -> String {
        let mut output = String::new();

        // Header comment
        output.push_str("// Custom pulse calibrations\n");
        output.push_str("// Generated by QuantRS2 CalibrationManager\n\n");

        for cal in &self.custom_calibrations {
            output.push_str(&self.calibration_to_defcal(cal));
            output.push('\n');
        }

        output
    }

    /// Convert a single calibration to defcal statement
    fn calibration_to_defcal(&self, calibration: &CustomCalibration) -> String {
        let mut output = String::new();

        // Add description as comment
        if let Some(desc) = &calibration.description {
            output.push_str(&format!("// {}\n", desc));
        }

        // Build parameter list
        let params = if calibration.parameters.is_empty() {
            String::new()
        } else {
            format!("({})", calibration.parameters.join(", "))
        };

        // Build qubit list
        let qubits = calibration
            .qubits
            .iter()
            .map(|q| format!("$q{}", q))
            .collect::<Vec<_>>()
            .join(", ");

        // Start defcal block
        output.push_str(&format!(
            "defcal {}{} {} {{\n",
            calibration.gate_name, params, qubits
        ));

        // Add instructions
        for instruction in &calibration.pulse_schedule.instructions {
            output.push_str(&format!(
                "    {};\n",
                self.instruction_to_openpulse(instruction)
            ));
        }

        output.push_str("}\n");
        output
    }

    /// Convert a pulse instruction to OpenPulse statement
    fn instruction_to_openpulse(&self, instruction: &PulseInstruction) -> String {
        match instruction {
            PulseInstruction::Play {
                pulse,
                channel,
                t0,
                name,
            } => {
                let pulse_str = self.waveform_to_openpulse(pulse);
                let name_comment = name
                    .as_ref()
                    .map(|n| format!(" // {}", n))
                    .unwrap_or_default();
                format!("play({}, {}) @ {}{}", pulse_str, channel, t0, name_comment)
            }
            PulseInstruction::SetFrequency {
                frequency,
                channel,
                t0,
            } => {
                format!("set_frequency({}, {:.6e}) @ {}", channel, frequency, t0)
            }
            PulseInstruction::ShiftFrequency {
                frequency,
                channel,
                t0,
            } => {
                format!("shift_frequency({}, {:.6e}) @ {}", channel, frequency, t0)
            }
            PulseInstruction::SetPhase { phase, channel, t0 } => {
                format!("set_phase({}, {:.6}) @ {}", channel, phase, t0)
            }
            PulseInstruction::ShiftPhase { phase, channel, t0 } => {
                format!("shift_phase({}, {:.6}) @ {}", channel, phase, t0)
            }
            PulseInstruction::Delay {
                duration,
                channel,
                t0,
            } => {
                format!("delay({}, {}) @ {}", channel, duration, t0)
            }
            PulseInstruction::Acquire {
                duration,
                qubit,
                memory_slot,
                t0,
            } => {
                format!(
                    "acquire({}, {}, c{}) @ {}",
                    duration, qubit, memory_slot, t0
                )
            }
            PulseInstruction::Barrier { channels, t0 } => {
                let channels_str = channels
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("barrier({}) @ {}", channels_str, t0)
            }
        }
    }

    /// Convert a waveform to OpenPulse expression
    fn waveform_to_openpulse(&self, waveform: &PulseWaveform) -> String {
        match waveform {
            PulseWaveform::Gaussian {
                amp,
                duration,
                sigma,
                name,
            } => {
                let name_str = name
                    .as_ref()
                    .map(|n| format!(", name=\"{}\"", n))
                    .unwrap_or_default();
                format!(
                    "gaussian({}, {}, {:.2}, {:.2}{})",
                    duration, amp.0, amp.1, sigma, name_str
                )
            }
            PulseWaveform::GaussianSquare {
                amp,
                duration,
                sigma,
                width,
                risefall_shape,
                name,
            } => {
                let name_str = name
                    .as_ref()
                    .map(|n| format!(", name=\"{}\"", n))
                    .unwrap_or_default();
                format!(
                    "gaussian_square({}, {}, {:.2}, {:.2}, {}, \"{}\"{name_str})",
                    duration, amp.0, amp.1, sigma, width, risefall_shape
                )
            }
            PulseWaveform::Drag {
                amp,
                duration,
                sigma,
                beta,
                name,
            } => {
                let name_str = name
                    .as_ref()
                    .map(|n| format!(", name=\"{}\"", n))
                    .unwrap_or_default();
                format!(
                    "drag({}, {}, {:.2}, {:.2}, {:.4}{})",
                    duration, amp.0, amp.1, sigma, beta, name_str
                )
            }
            PulseWaveform::Constant {
                amp,
                duration,
                name,
            } => {
                let name_str = name
                    .as_ref()
                    .map(|n| format!(", name=\"{}\"", n))
                    .unwrap_or_default();
                format!(
                    "constant({}, {}, {:.2}{})",
                    duration, amp.0, amp.1, name_str
                )
            }
            PulseWaveform::Waveform { samples, name } => {
                let name_str = name.as_deref().unwrap_or("custom");
                format!("waveform(\"{}\", {} samples)", name_str, samples.len())
            }
        }
    }

    /// Convert to IBM-compatible JSON format for upload
    pub fn to_ibm_format(&self) -> DeviceResult<serde_json::Value> {
        let mut calibrations = Vec::new();

        for cal in &self.custom_calibrations {
            let schedule_json = self.schedule_to_ibm_json(&cal.pulse_schedule)?;
            calibrations.push(serde_json::json!({
                "gate_name": cal.gate_name,
                "qubits": cal.qubits,
                "schedule": schedule_json,
                "parameters": cal.parameters,
            }));
        }

        Ok(serde_json::json!({
            "backend": self.backend_name,
            "calibrations": calibrations,
        }))
    }

    /// Convert pulse schedule to IBM JSON format
    fn schedule_to_ibm_json(&self, schedule: &PulseSchedule) -> DeviceResult<serde_json::Value> {
        let mut instructions = Vec::new();

        for inst in &schedule.instructions {
            instructions.push(self.instruction_to_ibm_json(inst)?);
        }

        Ok(serde_json::json!({
            "name": schedule.name,
            "instructions": instructions,
            "duration": schedule.duration_dt,
            "dt": schedule.dt,
        }))
    }

    /// Convert instruction to IBM JSON format
    fn instruction_to_ibm_json(
        &self,
        instruction: &PulseInstruction,
    ) -> DeviceResult<serde_json::Value> {
        match instruction {
            PulseInstruction::Play {
                pulse,
                channel,
                t0,
                name,
            } => Ok(serde_json::json!({
                "name": "play",
                "t0": t0,
                "ch": channel.to_string(),
                "pulse": self.waveform_to_ibm_json(pulse)?,
                "label": name,
            })),
            PulseInstruction::SetFrequency {
                frequency,
                channel,
                t0,
            } => Ok(serde_json::json!({
                "name": "setf",
                "t0": t0,
                "ch": channel.to_string(),
                "frequency": frequency,
            })),
            PulseInstruction::ShiftFrequency {
                frequency,
                channel,
                t0,
            } => Ok(serde_json::json!({
                "name": "shiftf",
                "t0": t0,
                "ch": channel.to_string(),
                "frequency": frequency,
            })),
            PulseInstruction::SetPhase { phase, channel, t0 } => Ok(serde_json::json!({
                "name": "setp",
                "t0": t0,
                "ch": channel.to_string(),
                "phase": phase,
            })),
            PulseInstruction::ShiftPhase { phase, channel, t0 } => Ok(serde_json::json!({
                "name": "fc",
                "t0": t0,
                "ch": channel.to_string(),
                "phase": phase,
            })),
            PulseInstruction::Delay {
                duration,
                channel,
                t0,
            } => Ok(serde_json::json!({
                "name": "delay",
                "t0": t0,
                "ch": channel.to_string(),
                "duration": duration,
            })),
            PulseInstruction::Acquire {
                duration,
                qubit,
                memory_slot,
                t0,
            } => Ok(serde_json::json!({
                "name": "acquire",
                "t0": t0,
                "duration": duration,
                "qubits": [qubit],
                "memory_slot": [memory_slot],
            })),
            PulseInstruction::Barrier { channels, t0 } => {
                let ch_strs: Vec<String> = channels.iter().map(|c| c.to_string()).collect();
                Ok(serde_json::json!({
                    "name": "barrier",
                    "t0": t0,
                    "channels": ch_strs,
                }))
            }
        }
    }

    /// Convert waveform to IBM JSON format
    fn waveform_to_ibm_json(&self, waveform: &PulseWaveform) -> DeviceResult<serde_json::Value> {
        match waveform {
            PulseWaveform::Gaussian {
                amp,
                duration,
                sigma,
                name,
            } => Ok(serde_json::json!({
                "pulse_type": "Gaussian",
                "parameters": {
                    "amp": [amp.0, amp.1],
                    "duration": duration,
                    "sigma": sigma,
                },
                "name": name,
            })),
            PulseWaveform::GaussianSquare {
                amp,
                duration,
                sigma,
                width,
                risefall_shape,
                name,
            } => Ok(serde_json::json!({
                "pulse_type": "GaussianSquare",
                "parameters": {
                    "amp": [amp.0, amp.1],
                    "duration": duration,
                    "sigma": sigma,
                    "width": width,
                    "risefall_sigma_ratio": risefall_shape,
                },
                "name": name,
            })),
            PulseWaveform::Drag {
                amp,
                duration,
                sigma,
                beta,
                name,
            } => Ok(serde_json::json!({
                "pulse_type": "Drag",
                "parameters": {
                    "amp": [amp.0, amp.1],
                    "duration": duration,
                    "sigma": sigma,
                    "beta": beta,
                },
                "name": name,
            })),
            PulseWaveform::Constant {
                amp,
                duration,
                name,
            } => Ok(serde_json::json!({
                "pulse_type": "Constant",
                "parameters": {
                    "amp": [amp.0, amp.1],
                    "duration": duration,
                },
                "name": name,
            })),
            PulseWaveform::Waveform { samples, name } => {
                // Convert samples to lists
                let real: Vec<f64> = samples.iter().map(|(r, _)| *r).collect();
                let imag: Vec<f64> = samples.iter().map(|(_, i)| *i).collect();
                Ok(serde_json::json!({
                    "pulse_type": "Waveform",
                    "samples": {
                        "real": real,
                        "imag": imag,
                    },
                    "name": name,
                }))
            }
        }
    }

    /// Get the number of custom calibrations
    pub fn len(&self) -> usize {
        self.custom_calibrations.len()
    }

    /// Check if there are no custom calibrations
    pub fn is_empty(&self) -> bool {
        self.custom_calibrations.is_empty()
    }

    /// List all custom calibration gate names
    pub fn calibration_names(&self) -> Vec<(&str, &[usize])> {
        self.custom_calibrations
            .iter()
            .map(|c| (c.gate_name.as_str(), c.qubits.as_slice()))
            .collect()
    }
}

/// Builder for creating custom pulse calibrations
#[derive(Debug, Clone)]
pub struct CalibrationBuilder {
    gate_name: String,
    qubits: Vec<usize>,
    instructions: Vec<PulseInstruction>,
    parameters: Vec<String>,
    description: Option<String>,
    dt: f64,
}

impl CalibrationBuilder {
    /// Create a new calibration builder
    pub fn new(gate_name: impl Into<String>, qubits: Vec<usize>) -> Self {
        Self {
            gate_name: gate_name.into(),
            qubits,
            instructions: Vec::new(),
            parameters: Vec::new(),
            description: None,
            dt: 2.22e-10, // Default dt
        }
    }

    /// Set the device time unit
    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Add a parameter
    pub fn parameter(mut self, param: impl Into<String>) -> Self {
        self.parameters.push(param.into());
        self
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a Gaussian pulse
    pub fn gaussian(
        mut self,
        channel: PulseChannel,
        t0: u64,
        duration: u64,
        amp: (f64, f64),
        sigma: f64,
    ) -> Self {
        self.instructions.push(PulseInstruction::Play {
            pulse: PulseWaveform::Gaussian {
                amp,
                duration,
                sigma,
                name: None,
            },
            channel,
            t0,
            name: None,
        });
        self
    }

    /// Add a DRAG pulse
    pub fn drag(
        mut self,
        channel: PulseChannel,
        t0: u64,
        duration: u64,
        amp: (f64, f64),
        sigma: f64,
        beta: f64,
    ) -> Self {
        self.instructions.push(PulseInstruction::Play {
            pulse: PulseWaveform::Drag {
                amp,
                duration,
                sigma,
                beta,
                name: None,
            },
            channel,
            t0,
            name: None,
        });
        self
    }

    /// Add a Gaussian square pulse
    pub fn gaussian_square(
        mut self,
        channel: PulseChannel,
        t0: u64,
        duration: u64,
        amp: (f64, f64),
        sigma: f64,
        width: u64,
    ) -> Self {
        self.instructions.push(PulseInstruction::Play {
            pulse: PulseWaveform::GaussianSquare {
                amp,
                duration,
                sigma,
                width,
                risefall_shape: "gaussian".to_string(),
                name: None,
            },
            channel,
            t0,
            name: None,
        });
        self
    }

    /// Add a constant pulse
    pub fn constant(
        mut self,
        channel: PulseChannel,
        t0: u64,
        duration: u64,
        amp: (f64, f64),
    ) -> Self {
        self.instructions.push(PulseInstruction::Play {
            pulse: PulseWaveform::Constant {
                amp,
                duration,
                name: None,
            },
            channel,
            t0,
            name: None,
        });
        self
    }

    /// Add a phase shift
    pub fn shift_phase(mut self, channel: PulseChannel, t0: u64, phase: f64) -> Self {
        self.instructions
            .push(PulseInstruction::ShiftPhase { phase, channel, t0 });
        self
    }

    /// Add a frequency shift
    pub fn shift_frequency(mut self, channel: PulseChannel, t0: u64, frequency: f64) -> Self {
        self.instructions.push(PulseInstruction::ShiftFrequency {
            frequency,
            channel,
            t0,
        });
        self
    }

    /// Add a delay
    pub fn delay(mut self, channel: PulseChannel, t0: u64, duration: u64) -> Self {
        self.instructions.push(PulseInstruction::Delay {
            duration,
            channel,
            t0,
        });
        self
    }

    /// Add a barrier
    pub fn barrier(mut self, channels: Vec<PulseChannel>, t0: u64) -> Self {
        self.instructions
            .push(PulseInstruction::Barrier { channels, t0 });
        self
    }

    /// Build the custom calibration
    pub fn build(self) -> CustomCalibration {
        // Calculate total duration
        let duration_dt = self
            .instructions
            .iter()
            .map(|i| match i {
                PulseInstruction::Play { pulse, t0, .. } => t0 + pulse.duration(),
                PulseInstruction::Delay { duration, t0, .. } => t0 + duration,
                PulseInstruction::Acquire { duration, t0, .. } => t0 + duration,
                _ => 0,
            })
            .max()
            .unwrap_or(0);

        CustomCalibration {
            gate_name: self.gate_name.clone(),
            qubits: self.qubits,
            pulse_schedule: PulseSchedule {
                name: self.gate_name,
                instructions: self.instructions,
                duration_dt,
                dt: self.dt,
            },
            parameters: self.parameters,
            description: self.description,
        }
    }
}

/// Instruction properties (Qiskit Target compatibility)
#[derive(Debug, Clone)]
pub struct InstructionProperties {
    /// Duration in seconds
    pub duration: Option<f64>,
    /// Error rate
    pub error: Option<f64>,
    /// Calibration data
    pub calibration: Option<String>,
}

impl Default for InstructionProperties {
    fn default() -> Self {
        Self {
            duration: None,
            error: None,
            calibration: None,
        }
    }
}

/// Target representation (Qiskit Target compatibility)
#[derive(Debug, Clone)]
pub struct Target {
    /// Number of qubits
    pub num_qubits: usize,
    /// Description
    pub description: String,
    /// Instruction properties map
    pub instruction_properties: HashMap<String, HashMap<Vec<usize>, InstructionProperties>>,
    /// Coupling map
    pub coupling_map: Vec<(usize, usize)>,
}

impl Target {
    /// Create a new Target from calibration data
    pub fn from_calibration(calibration: &CalibrationData) -> Self {
        let mut instruction_properties = HashMap::new();

        // Add gate properties
        for (gate_name, gates) in &calibration.gates {
            let mut props = HashMap::new();
            for gate in gates {
                props.insert(
                    gate.qubits.clone(),
                    InstructionProperties {
                        duration: Some(gate.gate_length.as_secs_f64()),
                        error: Some(gate.gate_error),
                        calibration: None,
                    },
                );
            }
            instruction_properties.insert(gate_name.clone(), props);
        }

        // Add measure properties
        let mut measure_props = HashMap::new();
        for qubit in &calibration.qubits {
            measure_props.insert(
                vec![qubit.qubit_id],
                InstructionProperties {
                    duration: Some(qubit.readout_length.as_secs_f64()),
                    error: Some(qubit.readout_error),
                    calibration: None,
                },
            );
        }
        instruction_properties.insert("measure".to_string(), measure_props);

        Self {
            num_qubits: calibration.qubits.len(),
            description: format!("Target for {}", calibration.backend_name),
            instruction_properties,
            coupling_map: calibration.general.coupling_map.clone(),
        }
    }

    /// Check if an instruction is supported on given qubits
    pub fn instruction_supported(&self, instruction: &str, qubits: &[usize]) -> bool {
        self.instruction_properties
            .get(instruction)
            .is_some_and(|props| props.contains_key(qubits))
    }

    /// Get instruction properties
    pub fn get_instruction_properties(
        &self,
        instruction: &str,
        qubits: &[usize],
    ) -> Option<&InstructionProperties> {
        self.instruction_properties
            .get(instruction)
            .and_then(|props| props.get(qubits))
    }
}

#[cfg(test)]
#[allow(clippy::pedantic, clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_qubit_calibration_quality_score() {
        let qubit = QubitCalibration {
            qubit_id: 0,
            t1: Duration::from_micros(100),
            t2: Duration::from_micros(75),
            frequency: 5.0,
            anharmonicity: -0.34,
            readout_error: 0.01,
            readout_length: Duration::from_nanos(500),
            prob_meas0_prep1: 0.02,
            prob_meas1_prep0: 0.01,
        };

        let score = qubit.quality_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_gate_calibration_fidelity() {
        let gate = GateCalibration {
            gate_name: "cx".to_string(),
            qubits: vec![0, 1],
            gate_error: 0.005,
            gate_length: Duration::from_nanos(300),
            parameters: HashMap::new(),
        };

        assert!((gate.fidelity() - 0.995).abs() < 1e-10);
    }

    #[test]
    fn test_processor_type_typical_values() {
        let eagle = ProcessorType::Eagle;
        assert!(eagle.typical_t1().as_micros() > 100);
        assert!(eagle.typical_cx_error() < 0.01);
    }

    #[test]
    fn test_target_instruction_supported() {
        let mut instruction_properties = HashMap::new();
        let mut cx_props = HashMap::new();
        cx_props.insert(
            vec![0, 1],
            InstructionProperties {
                duration: Some(3e-7),
                error: Some(0.005),
                calibration: None,
            },
        );
        instruction_properties.insert("cx".to_string(), cx_props);

        let target = Target {
            num_qubits: 5,
            description: "Test target".to_string(),
            instruction_properties,
            coupling_map: vec![(0, 1), (1, 2)],
        };

        assert!(target.instruction_supported("cx", &[0, 1]));
        assert!(!target.instruction_supported("cx", &[0, 2]));
    }

    // =========================================================================
    // CalibrationManager Tests
    // =========================================================================

    #[test]
    fn test_calibration_manager_new() {
        let manager = CalibrationManager::new("ibm_brisbane");
        assert_eq!(manager.backend_name, "ibm_brisbane");
        assert!(manager.is_empty());
    }

    #[test]
    fn test_calibration_builder_drag_pulse() {
        let cal = CalibrationBuilder::new("x", vec![0])
            .description("Custom X gate with DRAG pulse")
            .drag(PulseChannel::Drive(0), 0, 160, (0.5, 0.0), 40.0, 0.1)
            .build();

        assert_eq!(cal.gate_name, "x");
        assert_eq!(cal.qubits, vec![0]);
        assert_eq!(cal.pulse_schedule.instructions.len(), 1);
        assert_eq!(cal.pulse_schedule.duration_dt, 160);
    }

    #[test]
    fn test_calibration_builder_gaussian_pulse() {
        let cal = CalibrationBuilder::new("sx", vec![0])
            .gaussian(PulseChannel::Drive(0), 0, 80, (0.25, 0.0), 20.0)
            .build();

        assert_eq!(cal.gate_name, "sx");
        assert_eq!(cal.pulse_schedule.duration_dt, 80);
    }

    #[test]
    fn test_calibration_builder_multi_instruction() {
        let cal = CalibrationBuilder::new("cx", vec![0, 1])
            .description("Cross-resonance CNOT gate")
            .shift_phase(PulseChannel::Drive(0), 0, std::f64::consts::PI / 2.0)
            .gaussian_square(PulseChannel::Control(0), 0, 1024, (0.8, 0.0), 64.0, 896)
            .shift_phase(PulseChannel::Drive(1), 1024, -std::f64::consts::PI / 2.0)
            .build();

        assert_eq!(cal.gate_name, "cx");
        assert_eq!(cal.qubits, vec![0, 1]);
        assert_eq!(cal.pulse_schedule.instructions.len(), 3);
        assert_eq!(cal.pulse_schedule.duration_dt, 1024);
    }

    #[test]
    fn test_calibration_manager_add_and_get() {
        let mut manager = CalibrationManager::new("test_backend");

        let cal = CalibrationBuilder::new("x", vec![0])
            .drag(PulseChannel::Drive(0), 0, 160, (0.5, 0.0), 40.0, 0.1)
            .build();

        manager
            .add_calibration(cal)
            .expect("Should add calibration");
        assert_eq!(manager.len(), 1);

        let retrieved = manager.get_calibration("x", &[0]);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.map(|c| &c.gate_name), Some(&"x".to_string()));
    }

    #[test]
    fn test_calibration_manager_remove() {
        let mut manager = CalibrationManager::new("test_backend");

        let cal1 = CalibrationBuilder::new("x", vec![0])
            .drag(PulseChannel::Drive(0), 0, 160, (0.5, 0.0), 40.0, 0.1)
            .build();
        let cal2 = CalibrationBuilder::new("x", vec![1])
            .drag(PulseChannel::Drive(1), 0, 160, (0.5, 0.0), 40.0, 0.1)
            .build();

        manager.add_calibration(cal1).expect("add");
        manager.add_calibration(cal2).expect("add");
        assert_eq!(manager.len(), 2);

        assert!(manager.remove_calibration("x", &[0]));
        assert_eq!(manager.len(), 1);

        assert!(manager.get_calibration("x", &[0]).is_none());
        assert!(manager.get_calibration("x", &[1]).is_some());
    }

    #[test]
    fn test_calibration_validation_amplitude_error() {
        let manager = CalibrationManager::new("test_backend");

        // Create calibration with amplitude > 1.0
        let cal = CalibrationBuilder::new("x", vec![0])
            .drag(PulseChannel::Drive(0), 0, 160, (1.5, 0.0), 40.0, 0.1)
            .build();

        let result = manager.validate_calibration(&cal).expect("validation");
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_calibration_validation_duration_warning() {
        let mut constraints = PulseBackendConstraints::default();
        constraints.pulse_granularity = 16;

        let manager = CalibrationManager::new("test_backend").with_constraints(constraints);

        // Duration 100 is not multiple of 16
        let cal = CalibrationBuilder::new("x", vec![0])
            .gaussian(PulseChannel::Drive(0), 0, 100, (0.5, 0.0), 25.0)
            .build();

        let result = manager.validate_calibration(&cal).expect("validation");
        // This should produce a warning, but still be valid
        assert!(result.is_valid);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_generate_defcal_statements() {
        let mut manager = CalibrationManager::new("test_backend");

        let cal = CalibrationBuilder::new("x", vec![0])
            .description("Custom X gate")
            .drag(PulseChannel::Drive(0), 0, 160, (0.5, 0.0), 40.0, 0.1)
            .build();

        manager.add_calibration(cal).expect("add");

        let defcal = manager.generate_defcal_statements();
        assert!(defcal.contains("defcal x $q0"));
        assert!(defcal.contains("drag("));
        assert!(defcal.contains("Custom X gate"));
    }

    #[test]
    fn test_to_ibm_format() {
        let mut manager = CalibrationManager::new("ibm_brisbane");

        let cal = CalibrationBuilder::new("sx", vec![0])
            .gaussian(PulseChannel::Drive(0), 0, 80, (0.25, 0.0), 20.0)
            .build();

        manager.add_calibration(cal).expect("add");

        let json = manager.to_ibm_format().expect("should serialize");
        let obj = json.as_object().expect("should be object");

        assert_eq!(
            obj.get("backend").and_then(|v| v.as_str()),
            Some("ibm_brisbane")
        );
        assert!(obj.get("calibrations").is_some());
    }

    #[test]
    fn test_pulse_waveform_duration() {
        let gaussian = PulseWaveform::Gaussian {
            amp: (0.5, 0.0),
            duration: 160,
            sigma: 40.0,
            name: None,
        };
        assert_eq!(gaussian.duration(), 160);

        let drag = PulseWaveform::Drag {
            amp: (0.5, 0.0),
            duration: 80,
            sigma: 20.0,
            beta: 0.1,
            name: None,
        };
        assert_eq!(drag.duration(), 80);

        let waveform = PulseWaveform::Waveform {
            samples: vec![(0.1, 0.0), (0.2, 0.0), (0.3, 0.0)],
            name: Some("custom".to_string()),
        };
        assert_eq!(waveform.duration(), 3);
    }

    #[test]
    fn test_pulse_channel_display() {
        assert_eq!(format!("{}", PulseChannel::Drive(0)), "d0");
        assert_eq!(format!("{}", PulseChannel::Control(5)), "u5");
        assert_eq!(format!("{}", PulseChannel::Measure(2)), "m2");
        assert_eq!(format!("{}", PulseChannel::Acquire(3)), "a3");
    }

    #[test]
    fn test_calibration_builder_with_parameters() {
        let cal = CalibrationBuilder::new("rz", vec![0])
            .parameter("theta")
            .shift_phase(PulseChannel::Drive(0), 0, 0.0)
            .build();

        assert_eq!(cal.parameters, vec!["theta"]);
    }

    #[test]
    fn test_calibration_names() {
        let mut manager = CalibrationManager::new("test_backend");

        let cal1 = CalibrationBuilder::new("x", vec![0])
            .drag(PulseChannel::Drive(0), 0, 160, (0.5, 0.0), 40.0, 0.1)
            .build();
        let cal2 = CalibrationBuilder::new("cx", vec![0, 1])
            .gaussian_square(PulseChannel::Control(0), 0, 1024, (0.8, 0.0), 64.0, 896)
            .build();

        manager.add_calibration(cal1).expect("add");
        manager.add_calibration(cal2).expect("add");

        let names = manager.calibration_names();
        assert_eq!(names.len(), 2);
        assert!(names
            .iter()
            .any(|(name, qubits)| *name == "x" && *qubits == [0]));
        assert!(names
            .iter()
            .any(|(name, qubits)| *name == "cx" && *qubits == [0, 1]));
    }
}
