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
