//! Hardware specification types for transpiler

use crate::routing::CouplingMap;
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hardware backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareBackend {
    Superconducting,
    TrappedIon,
    Photonic,
    NeutralAtom,
    SiliconDots,
    Topological,
    Hybrid,
}

/// Hardware specification with advanced capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpec {
    /// Device name/identifier
    pub name: String,

    /// Hardware backend type
    pub backend_type: HardwareBackend,

    /// Maximum number of qubits
    pub max_qubits: usize,

    /// Qubit connectivity topology
    pub coupling_map: CouplingMap,

    /// Native gate set with fidelities
    pub native_gates: NativeGateSet,

    /// Gate error rates
    pub gate_errors: HashMap<String, f64>,

    /// Qubit coherence times (T1, T2)
    pub coherence_times: HashMap<usize, (f64, f64)>,

    /// Gate durations in nanoseconds
    pub gate_durations: HashMap<String, f64>,

    /// Readout fidelity per qubit
    pub readout_fidelity: HashMap<usize, f64>,

    /// Cross-talk parameters
    pub crosstalk_matrix: Option<Array2<f64>>,

    /// Calibration timestamp
    pub calibration_timestamp: std::time::SystemTime,

    /// Advanced hardware features
    pub advanced_features: AdvancedHardwareFeatures,
}

impl Default for HardwareSpec {
    fn default() -> Self {
        Self {
            name: "Generic Quantum Device".to_string(),
            backend_type: HardwareBackend::Superconducting,
            max_qubits: 27,
            coupling_map: CouplingMap::grid(3, 9),
            native_gates: NativeGateSet::default(),
            gate_errors: HashMap::new(),
            coherence_times: HashMap::new(),
            gate_durations: HashMap::new(),
            readout_fidelity: HashMap::new(),
            crosstalk_matrix: None,
            calibration_timestamp: std::time::SystemTime::now(),
            advanced_features: AdvancedHardwareFeatures::default(),
        }
    }
}

/// Advanced hardware features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedHardwareFeatures {
    /// Support for mid-circuit measurements
    pub mid_circuit_measurement: bool,

    /// Support for conditional operations
    pub conditional_operations: bool,

    /// Support for parameterized gates
    pub parameterized_gates: bool,

    /// Support for pulse-level control
    pub pulse_control: bool,

    /// Support for error mitigation
    pub error_mitigation: ErrorMitigationSupport,

    /// Quantum volume
    pub quantum_volume: Option<u64>,

    /// CLOPS (Circuit Layer Operations Per Second)
    pub clops: Option<f64>,
}

impl Default for AdvancedHardwareFeatures {
    fn default() -> Self {
        Self {
            mid_circuit_measurement: false,
            conditional_operations: false,
            parameterized_gates: true,
            pulse_control: false,
            error_mitigation: ErrorMitigationSupport::default(),
            quantum_volume: None,
            clops: None,
        }
    }
}

/// Error mitigation support levels
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ErrorMitigationSupport {
    pub zero_noise_extrapolation: bool,
    pub probabilistic_error_cancellation: bool,
    pub symmetry_verification: bool,
    pub virtual_distillation: bool,
    pub clifford_data_regression: bool,
}

/// Native gate set with advanced properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeGateSet {
    /// Single-qubit gates with properties
    pub single_qubit: HashMap<String, GateProperties>,

    /// Two-qubit gates with properties
    pub two_qubit: HashMap<String, GateProperties>,

    /// Multi-qubit gates with properties
    pub multi_qubit: HashMap<String, GateProperties>,

    /// Basis gate decompositions
    pub decompositions: HashMap<String, GateDecomposition>,
}

impl Default for NativeGateSet {
    fn default() -> Self {
        let mut single_qubit = HashMap::new();
        single_qubit.insert("X".to_string(), GateProperties::default());
        single_qubit.insert("Y".to_string(), GateProperties::default());
        single_qubit.insert("Z".to_string(), GateProperties::default());
        single_qubit.insert("H".to_string(), GateProperties::default());
        single_qubit.insert("S".to_string(), GateProperties::default());
        single_qubit.insert("T".to_string(), GateProperties::default());

        let mut two_qubit = HashMap::new();
        two_qubit.insert("CNOT".to_string(), GateProperties::default());
        two_qubit.insert("CZ".to_string(), GateProperties::default());

        Self {
            single_qubit,
            two_qubit,
            multi_qubit: HashMap::new(),
            decompositions: HashMap::new(),
        }
    }
}

/// Gate properties including noise characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateProperties {
    pub fidelity: f64,
    pub duration: f64,
    pub error_rate: f64,
    pub calibrated: bool,
    pub pulse_sequence: Option<String>,
}

impl Default for GateProperties {
    fn default() -> Self {
        Self {
            fidelity: 0.999,
            duration: 20e-9,
            error_rate: 0.001,
            calibrated: true,
            pulse_sequence: None,
        }
    }
}

/// Gate decomposition rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDecomposition {
    pub target_gate: String,
    pub decomposition: Vec<DecomposedGate>,
    pub cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposedGate {
    pub gate_type: String,
    pub qubits: Vec<usize>,
    pub parameters: Vec<f64>,
}
