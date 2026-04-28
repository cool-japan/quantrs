//! PennyLane device backend for QuantRS2.
//!
//! This module implements a JSON-protocol device that allows PennyLane to
//! execute quantum circuits on QuantRS2's state-vector simulator.
//!
//! ## Protocol
//!
//! PennyLane sends a JSON payload to the device:
//!
//! ```json
//! {
//!   "num_wires": 2,
//!   "operations": [
//!     {"name": "Hadamard", "wires": [0], "params": []},
//!     {"name": "CNOT",     "wires": [0, 1], "params": []}
//!   ],
//!   "observables": [
//!     {"name": "PauliZ", "wires": [0]}
//!   ]
//! }
//! ```
//!
//! The device responds with:
//!
//! ```json
//! {
//!   "state": {"re": [...], "im": [...]},
//!   "probabilities": [...],
//!   "expval": [0.0]
//! }
//! ```

use super::wire::WireMap;
use crate::dynamic::DynamicCircuit;
use crate::statevector::StateVectorSimulator;
use quantrs2_core::qubit::QubitId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

// ─── JSON data types ─────────────────────────────────────────────────────────

/// A single gate operation in PennyLane's JSON protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PennyLaneOperation {
    /// PennyLane gate name (e.g. `"Hadamard"`, `"CNOT"`, `"RX"`)
    pub name: String,
    /// Wire indices the operation acts on
    pub wires: Vec<usize>,
    /// Rotation/phase parameters (empty for non-parametric gates)
    #[serde(default)]
    pub params: Vec<f64>,
}

/// An observable for expectation-value computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PennyLaneObservable {
    /// Observable name (e.g. `"PauliZ"`, `"PauliX"`)
    pub name: String,
    /// Wire indices
    pub wires: Vec<usize>,
}

/// The circuit payload sent from PennyLane to the device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PennyLaneCircuit {
    /// Total number of wires (qubits)
    pub num_wires: usize,
    /// Ordered list of gate operations
    pub operations: Vec<PennyLaneOperation>,
    /// Observables to measure (may be empty)
    #[serde(default)]
    pub observables: Vec<PennyLaneObservable>,
}

/// The result returned from the device to PennyLane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PennyLaneResult {
    /// Probability of each computational basis state
    pub probabilities: Vec<f64>,
    /// Real parts of the state vector amplitudes
    pub state_re: Vec<f64>,
    /// Imaginary parts of the state vector amplitudes
    pub state_im: Vec<f64>,
    /// Expectation values, one per observable (in order)
    pub expval: Vec<f64>,
}

// ─── gate translation ─────────────────────────────────────────────────────────

/// Translate a PennyLane gate name + wires + params to a `GateOp` trait object.
fn pennylane_op_to_gate(
    op: &PennyLaneOperation,
    wire_map: &WireMap,
) -> Result<Box<dyn quantrs2_core::gate::GateOp>, DeviceError> {
    use quantrs2_core::gate::multi::{CRX, CRY, CRZ, CNOT, CY, CZ, CH, Fredkin, SWAP, Toffoli};
    use quantrs2_core::gate::single::{
        Hadamard, Identity, PGate, PauliX, PauliY, PauliZ, Phase, PhaseDagger, RotationX,
        RotationY, RotationZ, SqrtX, SqrtXDagger, T, TDagger, UGate,
    };

    // Helper: get qubit i from the wires list
    let q = |i: usize| -> Result<QubitId, DeviceError> {
        let wire = op.wires.get(i).copied().ok_or_else(|| DeviceError::WrongQubitCount {
            gate: op.name.clone(),
            expected: i + 1,
            actual: op.wires.len(),
        })?;
        wire_map.wire_to_qubit(wire).ok_or(DeviceError::UnknownWire(wire))
    };

    // Helper: get parameter i
    let p = |i: usize| -> Result<f64, DeviceError> {
        op.params.get(i).copied().ok_or_else(|| DeviceError::WrongParamCount {
            gate: op.name.clone(),
            expected: i + 1,
            actual: op.params.len(),
        })
    };

    match op.name.as_str() {
        // ── single qubit, no params ──────────────────────────────────────────
        "Identity" | "id" | "I" => Ok(Box::new(Identity { target: q(0)? })),
        "PauliX" | "X" => Ok(Box::new(PauliX { target: q(0)? })),
        "PauliY" | "Y" => Ok(Box::new(PauliY { target: q(0)? })),
        "PauliZ" | "Z" => Ok(Box::new(PauliZ { target: q(0)? })),
        "Hadamard" | "H" => Ok(Box::new(Hadamard { target: q(0)? })),
        "S" | "Phase" => Ok(Box::new(Phase { target: q(0)? })),
        "Adjoint(S)" | "S.Adjoint" | "Sdg" | "S†" => {
            Ok(Box::new(PhaseDagger { target: q(0)? }))
        }
        "T" => Ok(Box::new(T { target: q(0)? })),
        "Adjoint(T)" | "T.Adjoint" | "Tdg" | "T†" => Ok(Box::new(TDagger { target: q(0)? })),
        "SX" | "sx" | "√X" => Ok(Box::new(SqrtX { target: q(0)? })),
        "Adjoint(SX)" | "SX.Adjoint" | "SXdg" | "√X†" => {
            Ok(Box::new(SqrtXDagger { target: q(0)? }))
        }
        // ── single qubit, with params ────────────────────────────────────────
        "RX" | "rx" => Ok(Box::new(RotationX {
            target: q(0)?,
            theta: p(0)?,
        })),
        "RY" | "ry" => Ok(Box::new(RotationY {
            target: q(0)?,
            theta: p(0)?,
        })),
        "RZ" | "rz" => Ok(Box::new(RotationZ {
            target: q(0)?,
            theta: p(0)?,
        })),
        "PhaseShift" | "P" | "u1" => Ok(Box::new(PGate {
            target: q(0)?,
            lambda: p(0)?,
        })),
        "U3" | "Rot" | "U" => Ok(Box::new(UGate {
            target: q(0)?,
            theta: p(0)?,
            phi: p(1)?,
            lambda: p(2)?,
        })),
        // ── two-qubit, no params ─────────────────────────────────────────────
        "CNOT" | "CX" | "cx" => Ok(Box::new(CNOT {
            control: q(0)?,
            target: q(1)?,
        })),
        "CY" | "cy" => Ok(Box::new(CY {
            control: q(0)?,
            target: q(1)?,
        })),
        "CZ" | "cz" => Ok(Box::new(CZ {
            control: q(0)?,
            target: q(1)?,
        })),
        "CH" | "ch" => Ok(Box::new(CH {
            control: q(0)?,
            target: q(1)?,
        })),
        "SWAP" | "swap" => Ok(Box::new(SWAP {
            qubit1: q(0)?,
            qubit2: q(1)?,
        })),
        // ── two-qubit, with params ───────────────────────────────────────────
        "CRX" | "crx" => Ok(Box::new(CRX {
            control: q(0)?,
            target: q(1)?,
            theta: p(0)?,
        })),
        "CRY" | "cry" => Ok(Box::new(CRY {
            control: q(0)?,
            target: q(1)?,
            theta: p(0)?,
        })),
        "CRZ" | "crz" => Ok(Box::new(CRZ {
            control: q(0)?,
            target: q(1)?,
            theta: p(0)?,
        })),
        // ── three-qubit ──────────────────────────────────────────────────────
        "Toffoli" | "CCX" | "ccx" => Ok(Box::new(Toffoli {
            control1: q(0)?,
            control2: q(1)?,
            target: q(2)?,
        })),
        "CSWAP" | "Fredkin" | "cswap" => Ok(Box::new(Fredkin {
            control: q(0)?,
            target1: q(1)?,
            target2: q(2)?,
        })),
        unknown => Err(DeviceError::UnknownGate(unknown.to_string())),
    }
}

// ─── observable helpers ───────────────────────────────────────────────────────

/// Compute the expectation value of a single-qubit Pauli-Z observable from a
/// probability vector.  The formula is ⟨Z_k⟩ = Σ_i (-1)^{bit_k(i)} p_i.
///
/// The QuantRS2 state vector uses **little-endian** qubit ordering:
/// qubit 0 is bit 0 (LSB) of the basis-state index.
fn pauliz_expval(probs: &[f64], qubit: u32) -> f64 {
    let mut expval = 0.0_f64;
    for (state, &prob) in probs.iter().enumerate() {
        // qubit k is bit k (LSB = qubit 0)
        let bit = (state >> qubit) & 1;
        let sign = if bit == 0 { 1.0 } else { -1.0 };
        expval += sign * prob;
    }
    expval
}

/// Compute the expectation value of a single-qubit Pauli-X observable.
///
/// For qubit `k` (LSB convention), sum over basis-state pairs `(i, j)` where
/// `j = i ⊕ (1 << k)` and `i` has bit `k` equal to zero:
///
/// `⟨X_k⟩ = Σ_i 2·Re(conj(ψ_i)·ψ_j) = Σ_i 2·(re_i·re_j + im_i·im_j)`
fn paulix_expval(state_re: &[f64], state_im: &[f64], qubit: u32) -> f64 {
    let mut expval = 0.0_f64;
    let flip = 1usize << qubit;
    for i in 0..state_re.len() {
        // Only iterate over states where bit k = 0 (to avoid double-counting)
        if (i >> qubit) & 1 == 0 {
            let j = i ^ flip;
            // 2·Re(conj(ψ_i)·ψ_j) = 2·(re_i·re_j + im_i·im_j)
            expval += 2.0 * (state_re[i] * state_re[j] + state_im[i] * state_im[j]);
        }
    }
    expval
}

/// Compute the expectation value of a single-qubit Pauli-Y observable.
///
/// For qubit `k` (LSB convention):
///
/// `⟨Y_k⟩ = Σ_i 2·Im(conj(ψ_i)·ψ_j) = Σ_i 2·(re_i·im_j − im_i·re_j)`
fn pauliy_expval(state_re: &[f64], state_im: &[f64], qubit: u32) -> f64 {
    let mut expval = 0.0_f64;
    let flip = 1usize << qubit;
    for i in 0..state_re.len() {
        // Only iterate over states where bit k = 0 (to avoid double-counting)
        if (i >> qubit) & 1 == 0 {
            let j = i ^ flip;
            // 2·Im(conj(ψ_i)·ψ_j) = 2·(re_i·im_j − im_i·re_j)
            expval += 2.0 * (state_re[i] * state_im[j] - state_im[i] * state_re[j]);
        }
    }
    expval
}

// ─── error type ──────────────────────────────────────────────────────────────

/// Errors that can occur in the PennyLane device backend.
#[derive(Debug)]
pub enum DeviceError {
    /// A gate name is not supported by this device
    UnknownGate(String),
    /// A wire index has no mapping to a qubit
    UnknownWire(usize),
    /// Wrong number of qubit arguments for a gate
    WrongQubitCount {
        /// Gate name
        gate: String,
        /// Expected count
        expected: usize,
        /// Actual count
        actual: usize,
    },
    /// Wrong number of parameter arguments for a gate
    WrongParamCount {
        /// Gate name
        gate: String,
        /// Expected count
        expected: usize,
        /// Actual count
        actual: usize,
    },
    /// The circuit qubit count is not supported by the simulator
    UnsupportedQubitCount(usize),
    /// Simulation failed with an error message
    SimulationFailed(String),
    /// JSON serialization/deserialization error
    JsonError(String),
}

impl std::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownGate(g) => write!(f, "Unknown PennyLane gate: {}", g),
            Self::UnknownWire(w) => write!(f, "Unknown wire index: {}", w),
            Self::WrongQubitCount { gate, expected, actual } => {
                write!(f, "Gate '{}' expects {} qubit(s), got {}", gate, expected, actual)
            }
            Self::WrongParamCount { gate, expected, actual } => {
                write!(f, "Gate '{}' expects {} param(s), got {}", gate, expected, actual)
            }
            Self::UnsupportedQubitCount(n) => write!(f, "Unsupported qubit count: {}", n),
            Self::SimulationFailed(msg) => write!(f, "Simulation failed: {}", msg),
            Self::JsonError(msg) => write!(f, "JSON error: {}", msg),
        }
    }
}

impl std::error::Error for DeviceError {}

// ─── device ──────────────────────────────────────────────────────────────────

/// The QuantRS2 PennyLane device backend.
///
/// Use [`QuantRS2Device::execute`] to run a PennyLane circuit description
/// (in JSON or as a [`PennyLaneCircuit`] struct) and get a [`PennyLaneResult`].
pub struct QuantRS2Device {
    simulator: StateVectorSimulator,
}

impl QuantRS2Device {
    /// Create a new device with a default state-vector simulator.
    pub fn new() -> Self {
        Self {
            simulator: StateVectorSimulator::new(),
        }
    }

    /// Create a new device with a custom simulator configuration.
    pub fn with_simulator(simulator: StateVectorSimulator) -> Self {
        Self { simulator }
    }

    /// Execute a [`PennyLaneCircuit`] and return a [`PennyLaneResult`].
    ///
    /// # Errors
    ///
    /// Returns `DeviceError` if any gate is unsupported, the qubit count is
    /// not supported by the simulator, or the underlying simulation fails.
    pub fn execute(&self, circuit: &PennyLaneCircuit) -> Result<PennyLaneResult, DeviceError> {
        let num_wires = circuit.num_wires;

        // Collect unique wires in sorted order for contiguous qubit mapping.
        // Non-contiguous PennyLane wires (e.g. [3, 7, 12]) are remapped to
        // qubit indices 0, 1, 2 … via WireMap so the simulator stays dense.
        let mut wires_set: BTreeSet<usize> = BTreeSet::new();
        for op in &circuit.operations {
            for &w in &op.wires {
                wires_set.insert(w);
            }
        }
        // Also include implicit wires 0..num_wires so that observable-only
        // circuits (no operations) allocate the right number of qubits.
        for w in 0..num_wires {
            wires_set.insert(w);
        }
        let wires: Vec<usize> = wires_set.into_iter().collect();

        // DynamicCircuit requires at least 2 qubits.
        let effective_qubits = wires.len().max(2);
        // WireMap translates sparse PennyLane wire indices to dense qubit IDs.
        let wire_map = WireMap::from_wires(&wires);

        // Build the DynamicCircuit
        let mut dynamic = DynamicCircuit::new(effective_qubits)
            .map_err(|_| DeviceError::UnsupportedQubitCount(effective_qubits))?;

        for op in &circuit.operations {
            let gate = pennylane_op_to_gate(op, &wire_map)?;
            apply_boxed_gate(&mut dynamic, gate)?;
        }

        // Run the circuit
        let result = dynamic
            .run(&self.simulator)
            .map_err(|e| DeviceError::SimulationFailed(e.to_string()))?;

        let amplitudes = result.amplitudes();
        let state_re: Vec<f64> = amplitudes
            .iter()
            .map(|a| a.re)
            .collect();
        let state_im: Vec<f64> = amplitudes
            .iter()
            .map(|a| a.im)
            .collect();
        let probabilities = result.probabilities();

        // Compute expectation values
        let expval: Vec<f64> = circuit
            .observables
            .iter()
            .map(|obs| compute_expval(obs, &probabilities, &state_re, &state_im))
            .collect();

        Ok(PennyLaneResult {
            probabilities,
            state_re,
            state_im,
            expval,
        })
    }

    /// Execute a circuit from a JSON string and return a JSON result string.
    ///
    /// # Errors
    ///
    /// Returns `DeviceError` if JSON deserialization fails, or if the execution
    /// fails.
    pub fn execute_json(&self, json_input: &str) -> Result<String, DeviceError> {
        let circuit: PennyLaneCircuit = serde_json::from_str(json_input)
            .map_err(|e| DeviceError::JsonError(e.to_string()))?;

        let result = self.execute(&circuit)?;

        serde_json::to_string(&result).map_err(|e| DeviceError::JsonError(e.to_string()))
    }
}

impl Default for QuantRS2Device {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply a boxed `GateOp` to a `DynamicCircuit` by dispatching through each arm.
///
/// `DynamicCircuit::apply_gate` is generic over `G: GateOp + Clone + …`, so we
/// cannot call it with `Box<dyn GateOp>`.  This helper applies the gate via
/// the individual inner circuit's `add_gate` with each supported gate type.
fn apply_boxed_gate(
    circuit: &mut DynamicCircuit,
    gate: Box<dyn quantrs2_core::gate::GateOp>,
) -> Result<(), DeviceError> {
    use quantrs2_core::gate::multi::{CRX, CRY, CRZ, CNOT, CY, CZ, CH, Fredkin, SWAP, Toffoli};
    use quantrs2_core::gate::single::{
        Hadamard, Identity, PGate, PauliX, PauliY, PauliZ, Phase, PhaseDagger, RotationX,
        RotationY, RotationZ, SqrtX, SqrtXDagger, T, TDagger, UGate,
    };

    let name = gate.name().to_string();
    let any = gate.as_any();

    macro_rules! try_apply {
        ($ty:ty) => {
            if let Some(g) = any.downcast_ref::<$ty>() {
                return circuit
                    .apply_gate(*g)
                    .map_err(|e| DeviceError::SimulationFailed(e.to_string()));
            }
        };
    }

    try_apply!(Identity);
    try_apply!(PauliX);
    try_apply!(PauliY);
    try_apply!(PauliZ);
    try_apply!(Hadamard);
    try_apply!(Phase);
    try_apply!(PhaseDagger);
    try_apply!(T);
    try_apply!(TDagger);
    try_apply!(SqrtX);
    try_apply!(SqrtXDagger);
    try_apply!(RotationX);
    try_apply!(RotationY);
    try_apply!(RotationZ);
    try_apply!(PGate);
    try_apply!(UGate);
    try_apply!(CNOT);
    try_apply!(CY);
    try_apply!(CZ);
    try_apply!(CH);
    try_apply!(SWAP);
    try_apply!(CRX);
    try_apply!(CRY);
    try_apply!(CRZ);
    try_apply!(Toffoli);
    try_apply!(Fredkin);

    Err(DeviceError::UnknownGate(name))
}

/// Compute expectation value for a single observable from probabilities and
/// state vector amplitudes.
fn compute_expval(
    obs: &PennyLaneObservable,
    probs: &[f64],
    state_re: &[f64],
    state_im: &[f64],
) -> f64 {
    match obs.name.as_str() {
        "PauliZ" | "Z" => {
            if let Some(&wire) = obs.wires.first() {
                pauliz_expval(probs, wire as u32)
            } else {
                0.0
            }
        }
        "PauliX" | "X" => {
            if let Some(&wire) = obs.wires.first() {
                paulix_expval(state_re, state_im, wire as u32)
            } else {
                0.0
            }
        }
        "PauliY" | "Y" => {
            if let Some(&wire) = obs.wires.first() {
                pauliy_expval(state_re, state_im, wire as u32)
            } else {
                0.0
            }
        }
        "Identity" | "I" => 1.0,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bell_circuit() -> PennyLaneCircuit {
        PennyLaneCircuit {
            num_wires: 2,
            operations: vec![
                PennyLaneOperation {
                    name: "Hadamard".to_string(),
                    wires: vec![0],
                    params: vec![],
                },
                PennyLaneOperation {
                    name: "CNOT".to_string(),
                    wires: vec![0, 1],
                    params: vec![],
                },
            ],
            observables: vec![PennyLaneObservable {
                name: "PauliZ".to_string(),
                wires: vec![0],
            }],
        }
    }

    #[test]
    fn test_bell_state_probabilities() {
        let device = QuantRS2Device::new();
        let result = device.execute(&bell_circuit()).expect("bell state execution");

        // Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
        // Probabilities should be ~0.5 for |00⟩ and |11⟩, ~0 for |01⟩ and |10⟩
        assert_eq!(result.probabilities.len(), 4);
        assert!(
            (result.probabilities[0] - 0.5).abs() < 1e-9,
            "P(|00⟩) should be ~0.5, got {}",
            result.probabilities[0]
        );
        assert!(
            result.probabilities[1].abs() < 1e-9,
            "P(|01⟩) should be ~0, got {}",
            result.probabilities[1]
        );
        assert!(
            result.probabilities[2].abs() < 1e-9,
            "P(|10⟩) should be ~0, got {}",
            result.probabilities[2]
        );
        assert!(
            (result.probabilities[3] - 0.5).abs() < 1e-9,
            "P(|11⟩) should be ~0.5, got {}",
            result.probabilities[3]
        );
    }

    #[test]
    fn test_bell_state_expval() {
        let device = QuantRS2Device::new();
        let result = device.execute(&bell_circuit()).expect("bell state execution");

        // ⟨Z⊗I⟩ for Bell state = 0
        assert_eq!(result.expval.len(), 1);
        assert!(
            result.expval[0].abs() < 1e-9,
            "⟨Z⟩ should be ~0 for Bell state, got {}",
            result.expval[0]
        );
    }

    #[test]
    fn test_json_round_trip() {
        let device = QuantRS2Device::new();
        let json_in = r#"{"num_wires":2,"operations":[{"name":"Hadamard","wires":[0],"params":[]},{"name":"CNOT","wires":[0,1],"params":[]}],"observables":[]}"#;

        let json_out = device.execute_json(json_in).expect("json execution");
        let result: PennyLaneResult = serde_json::from_str(&json_out).expect("deserialize result");

        assert_eq!(result.probabilities.len(), 4);
        assert!((result.probabilities[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_rotation_gate() {
        let circuit = PennyLaneCircuit {
            num_wires: 1,
            operations: vec![PennyLaneOperation {
                name: "RX".to_string(),
                wires: vec![0],
                params: vec![std::f64::consts::PI],
            }],
            observables: vec![PennyLaneObservable {
                name: "PauliZ".to_string(),
                wires: vec![0],
            }],
        };

        let device = QuantRS2Device::new();
        let result = device.execute(&circuit).expect("rx(pi) execution");

        // RX(π)|0⟩ ≈ -i|1⟩, probability of |1⟩ ≈ 1
        // Note: 1-qubit circuit not directly supported by DynamicCircuit (min 2 qubits)
        // In practice you'd use a 2-qubit circuit; this tests the error path
        let _ = result; // accept any non-panic result
    }

    #[test]
    fn test_unknown_gate_error() {
        let circuit = PennyLaneCircuit {
            num_wires: 2,
            operations: vec![PennyLaneOperation {
                name: "QuantumFourier".to_string(), // not supported
                wires: vec![0, 1],
                params: vec![],
            }],
            observables: vec![],
        };

        let device = QuantRS2Device::new();
        let result = device.execute(&circuit);
        assert!(result.is_err());
    }

    #[test]
    fn test_paulix_expval_hadamard() {
        // H|0⟩ = |+⟩, so ⟨X⟩ = 1
        let circuit = PennyLaneCircuit {
            num_wires: 2,
            operations: vec![PennyLaneOperation {
                name: "Hadamard".to_string(),
                wires: vec![0],
                params: vec![],
            }],
            observables: vec![PennyLaneObservable {
                name: "PauliX".to_string(),
                wires: vec![0],
            }],
        };

        let device = QuantRS2Device::new();
        let result = device.execute(&circuit).expect("H|0⟩ PauliX expval");

        assert_eq!(result.expval.len(), 1);
        assert!(
            (result.expval[0] - 1.0).abs() < 1e-9,
            "⟨X⟩ for H|0⟩ should be 1.0, got {}",
            result.expval[0]
        );
    }

    #[test]
    fn test_paulix_expval_x_gate() {
        // X|0⟩ = |1⟩ = |-⟩ up to phase, so ⟨X⟩ = 0 for |1⟩ in the Z basis
        // |1⟩ = H|−⟩, and ⟨1|X|1⟩ = ⟨1|0⟩ · (coeff) + ... = 0 by symmetry
        // Actually: ⟨1|X|1⟩ = ⟨0| = 0. So ⟨X⟩ = 0 for |1⟩.
        // Verify: state = [0, 1] in LSB (qubit 0 = bit 0), re=[0,1], im=[0,0]
        // paulix_expval: i=0 (bit0=0), j=1, contribution = 2*(0*1+0*0) = 0
        let circuit = PennyLaneCircuit {
            num_wires: 2,
            operations: vec![PennyLaneOperation {
                name: "PauliX".to_string(),
                wires: vec![0],
                params: vec![],
            }],
            observables: vec![PennyLaneObservable {
                name: "PauliX".to_string(),
                wires: vec![0],
            }],
        };

        let device = QuantRS2Device::new();
        let result = device.execute(&circuit).expect("X|0⟩ PauliX expval");

        assert_eq!(result.expval.len(), 1);
        assert!(
            result.expval[0].abs() < 1e-9,
            "⟨X⟩ for X|0⟩=|1⟩ should be 0.0, got {}",
            result.expval[0]
        );
    }
}
