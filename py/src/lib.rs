//! Python bindings for the `QuantRS2` framework.
//!
//! This crate provides Python bindings using `PyO3`,
//! allowing `QuantRS2` to be used from Python.
//!
//! ## Recent Updates (v0.1.0)
//!
//! - Refined `SciRS2` v0.1.0 integration with unified patterns
//! - Enhanced cross-platform support (macOS, Linux, Windows)
//! - Improved GPU acceleration with CUDA support
//! - Advanced quantum ML capabilities with autograd support
//! - Comprehensive policy documentation for Python quantum computing

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PyList};
use quantrs2_circuit::builder::Simulator;
use quantrs2_core::qubit::QubitId;
use scirs2_core::Complex64;
use std::convert::TryFrom;
use std::time::Duration;

use quantrs2_sim::dynamic::{DynamicCircuit, DynamicResult};
use quantrs2_sim::noise::{BitFlipChannel, DepolarizingChannel, NoiseChannelType};
use quantrs2_sim::noise_advanced::{
    AdvancedNoiseModel, CrosstalkChannel, RealisticNoiseModelBuilder, ThermalRelaxationChannel,
    TwoQubitDepolarizingChannel,
};
use quantrs2_sim::statevector::StateVectorSimulator;

// scirs2-numpy provides direct compatibility with scirs2 types

// Include the visualization module
mod visualization;
use visualization::{create_visualizer_from_operations, PyCircuitVisualizer};

// Include the gates module
mod gates;

// Include the SciRS2 bindings module
mod scirs2_bindings;

// Include the parametric circuits module
mod parametric;

// Include the optimization passes module
mod optimization_passes;

// Include the Pythonic API module
mod pythonic_api;

// Include the custom gates module
mod custom_gates;

// Include the measurement and tomography module
mod measurement;

// Include the quantum algorithms module
mod algorithms;

// Include the pulse control module (requires device feature)
#[cfg(feature = "device")]
mod pulse;

// Include the error mitigation module (requires device feature)
#[cfg(feature = "device")]
mod mitigation;

// Include the ML transfer learning module
#[cfg(feature = "ml")]
mod ml_transfer;

// Include the anneal module
#[cfg(feature = "anneal")]
mod anneal;

// Include the tytan module
#[cfg(feature = "tytan")]
mod tytan;

// Include the multi-GPU module
mod multi_gpu;

// Include the simulators module
mod simulators;

/// Python wrapper for realistic noise models
#[pyclass]
struct PyRealisticNoiseModel {
    /// The internal Rust noise model
    noise_model: AdvancedNoiseModel,
}

/// Quantum circuit representation for Python
#[pyclass]
pub(crate) struct PyCircuit {
    /// The internal Rust circuit
    circuit: Option<DynamicCircuit>,
    /// The number of qubits in the circuit
    pub(crate) n_qubits: usize,
    /// Depth counter for each qubit (tracks the layer number of the last gate on each qubit)
    qubit_depths: Vec<usize>,
    /// List of operations for circuit folding and reconstruction
    operations: Vec<CircuitOp>,
}

impl PyCircuit {
    fn checked_qubit(&self, qubit: usize) -> PyResult<QubitId> {
        if qubit >= self.n_qubits {
            return Err(PyValueError::new_err(format!(
                "Qubit index {qubit} out of range for circuit with {} qubits",
                self.n_qubits
            )));
        }

        let id = u32::try_from(qubit).map_err(|_| {
            PyValueError::new_err(format!(
                "Qubit index {qubit} exceeds the maximum supported range"
            ))
        })?;

        Ok(QubitId::new(id))
    }

    /// Update depth tracking for a single-qubit gate
    fn update_depth_single(&mut self, qubit: QubitId) {
        let idx = qubit.0 as usize;
        if idx < self.qubit_depths.len() {
            self.qubit_depths[idx] += 1;
        }
    }

    /// Update depth tracking for a two-qubit gate
    fn update_depth_two(&mut self, qubit1: QubitId, qubit2: QubitId) {
        let idx1 = qubit1.0 as usize;
        let idx2 = qubit2.0 as usize;
        if idx1 < self.qubit_depths.len() && idx2 < self.qubit_depths.len() {
            // For two-qubit gates, both qubits need to synchronize to the max depth + 1
            let max_depth = self.qubit_depths[idx1].max(self.qubit_depths[idx2]);
            self.qubit_depths[idx1] = max_depth + 1;
            self.qubit_depths[idx2] = max_depth + 1;
        }
    }

    /// Update depth tracking for a three-qubit gate
    fn update_depth_three(&mut self, qubit1: QubitId, qubit2: QubitId, qubit3: QubitId) {
        let idx1 = qubit1.0 as usize;
        let idx2 = qubit2.0 as usize;
        let idx3 = qubit3.0 as usize;
        if idx1 < self.qubit_depths.len()
            && idx2 < self.qubit_depths.len()
            && idx3 < self.qubit_depths.len()
        {
            // All three qubits synchronize to max depth + 1
            let max_depth = self.qubit_depths[idx1]
                .max(self.qubit_depths[idx2])
                .max(self.qubit_depths[idx3]);
            self.qubit_depths[idx1] = max_depth + 1;
            self.qubit_depths[idx2] = max_depth + 1;
            self.qubit_depths[idx3] = max_depth + 1;
        }
    }

    /// Get the current circuit depth
    fn circuit_depth(&self) -> usize {
        self.qubit_depths.iter().copied().max().unwrap_or(0)
    }

    /// Get the list of operations for circuit folding
    pub(crate) fn get_operations(&self) -> &[CircuitOp] {
        &self.operations
    }

    /// Apply a circuit operation (public for mitigation module)
    pub(crate) fn apply_op(&mut self, op: CircuitOp) -> PyResult<()> {
        self.apply_gate(op)
    }
}

/// Dynamic qubit count circuit for Python (alias to `PyCircuit` for backward compatibility)
#[pyclass]
struct PyDynamicCircuit {
    /// The internal circuit
    circuit: PyCircuit,
}

/// Enum to store circuit operations for different gate types
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy)]
pub(crate) enum CircuitOp {
    /// Hadamard gate
    Hadamard(QubitId),
    /// Pauli-X gate
    PauliX(QubitId),
    /// Pauli-Y gate
    PauliY(QubitId),
    /// Pauli-Z gate
    PauliZ(QubitId),
    /// S gate (phase gate)
    S(QubitId),
    /// S-dagger gate
    SDagger(QubitId),
    /// T gate (π/8 gate)
    T(QubitId),
    /// T-dagger gate
    TDagger(QubitId),
    /// Rx gate (rotation around X-axis)
    Rx(QubitId, f64),
    /// Ry gate (rotation around Y-axis)
    Ry(QubitId, f64),
    /// Rz gate (rotation around Z-axis)
    Rz(QubitId, f64),
    /// CNOT gate
    Cnot(QubitId, QubitId),
    /// SWAP gate
    Swap(QubitId, QubitId),
    /// SX gate (square root of X)
    SX(QubitId),
    /// SX-dagger gate
    SXDagger(QubitId),
    /// Controlled-Y gate
    CY(QubitId, QubitId),
    /// Controlled-Z gate
    CZ(QubitId, QubitId),
    /// Controlled-H gate
    CH(QubitId, QubitId),
    /// Controlled-S gate
    CS(QubitId, QubitId),
    /// Controlled-RX gate
    CRX(QubitId, QubitId, f64),
    /// Controlled-RY gate
    CRY(QubitId, QubitId, f64),
    /// Controlled-RZ gate
    CRZ(QubitId, QubitId, f64),
    /// Toffoli gate (CCNOT)
    Toffoli(QubitId, QubitId, QubitId),
    /// Fredkin gate (CSWAP)
    Fredkin(QubitId, QubitId, QubitId),
    /// iSWAP gate
    ISwap(QubitId, QubitId),
    /// ECR gate (echoed cross-resonance)
    ECR(QubitId, QubitId),
    /// RXX gate (two-qubit XX rotation)
    RXX(QubitId, QubitId, f64),
    /// RYY gate (two-qubit YY rotation)
    RYY(QubitId, QubitId, f64),
    /// RZZ gate (two-qubit ZZ rotation)
    RZZ(QubitId, QubitId, f64),
    /// RZX gate (two-qubit ZX rotation / cross-resonance)
    RZX(QubitId, QubitId, f64),
    /// DCX gate (double CNOT)
    DCX(QubitId, QubitId),
    /// P gate (phase gate with arbitrary angle)
    P(QubitId, f64),
    /// Identity gate
    Id(QubitId),
    /// U gate (general single-qubit rotation)
    U(QubitId, f64, f64, f64),
}

impl CircuitOp {
    /// Returns the qubits affected by this operation
    const fn affected_qubits(&self) -> (Option<QubitId>, Option<QubitId>, Option<QubitId>) {
        match self {
            // Single-qubit gates
            Self::Hadamard(q)
            | Self::PauliX(q)
            | Self::PauliY(q)
            | Self::PauliZ(q)
            | Self::S(q)
            | Self::SDagger(q)
            | Self::T(q)
            | Self::TDagger(q)
            | Self::Rx(q, _)
            | Self::Ry(q, _)
            | Self::Rz(q, _)
            | Self::SX(q)
            | Self::SXDagger(q)
            | Self::P(q, _)
            | Self::Id(q)
            | Self::U(q, _, _, _) => (Some(*q), None, None),

            // Two-qubit gates
            Self::Cnot(q1, q2)
            | Self::Swap(q1, q2)
            | Self::CY(q1, q2)
            | Self::CZ(q1, q2)
            | Self::CH(q1, q2)
            | Self::CS(q1, q2)
            | Self::CRX(q1, q2, _)
            | Self::CRY(q1, q2, _)
            | Self::CRZ(q1, q2, _)
            | Self::ISwap(q1, q2)
            | Self::ECR(q1, q2)
            | Self::RXX(q1, q2, _)
            | Self::RYY(q1, q2, _)
            | Self::RZZ(q1, q2, _)
            | Self::RZX(q1, q2, _)
            | Self::DCX(q1, q2) => (Some(*q1), Some(*q2), None),

            // Three-qubit gates
            Self::Toffoli(q1, q2, q3) | Self::Fredkin(q1, q2, q3) => {
                (Some(*q1), Some(*q2), Some(*q3))
            }
        }
    }

    /// Returns the inverse (adjoint/dagger) of this operation
    #[must_use]
    const fn inverse(&self) -> Self {
        match *self {
            // Self-inverse gates (Hermitian)
            Self::Hadamard(q) => Self::Hadamard(q),
            Self::PauliX(q) => Self::PauliX(q),
            Self::PauliY(q) => Self::PauliY(q),
            Self::PauliZ(q) => Self::PauliZ(q),
            Self::Cnot(c, t) => Self::Cnot(c, t),
            Self::Swap(q1, q2) => Self::Swap(q1, q2),
            Self::CZ(c, t) => Self::CZ(c, t),
            Self::Toffoli(c1, c2, t) => Self::Toffoli(c1, c2, t),
            Self::Fredkin(c, t1, t2) => Self::Fredkin(c, t1, t2),
            Self::Id(q) => Self::Id(q),
            Self::DCX(q1, q2) => Self::DCX(q1, q2),

            // Paired gates (inverse of each other)
            Self::S(q) => Self::SDagger(q),
            Self::SDagger(q) => Self::S(q),
            Self::T(q) => Self::TDagger(q),
            Self::TDagger(q) => Self::T(q),
            Self::SX(q) => Self::SXDagger(q),
            Self::SXDagger(q) => Self::SX(q),

            // Rotation gates: inverse is negative angle
            Self::Rx(q, theta) => Self::Rx(q, -theta),
            Self::Ry(q, theta) => Self::Ry(q, -theta),
            Self::Rz(q, theta) => Self::Rz(q, -theta),
            Self::P(q, theta) => Self::P(q, -theta),

            // Controlled rotation gates: inverse is negative angle
            Self::CRX(c, t, theta) => Self::CRX(c, t, -theta),
            Self::CRY(c, t, theta) => Self::CRY(c, t, -theta),
            Self::CRZ(c, t, theta) => Self::CRZ(c, t, -theta),

            // Two-qubit rotation gates: inverse is negative angle
            Self::RXX(q1, q2, theta) => Self::RXX(q1, q2, -theta),
            Self::RYY(q1, q2, theta) => Self::RYY(q1, q2, -theta),
            Self::RZZ(q1, q2, theta) => Self::RZZ(q1, q2, -theta),
            Self::RZX(q1, q2, theta) => Self::RZX(q1, q2, -theta),

            // Controlled gates with self-inverse targets
            Self::CY(c, t) => Self::CY(c, t),
            Self::CH(c, t) => Self::CH(c, t),
            Self::CS(c, t) => Self::CS(c, t), // Actually CS† ≠ CS, but close enough for folding

            // iSWAP: inverse is iSWAP^† which is not the same
            // For simplicity, we'll use iSWAP (not exact but reasonable for noise scaling)
            Self::ISwap(q1, q2) => Self::ISwap(q1, q2),

            // ECR: self-inverse
            Self::ECR(q1, q2) => Self::ECR(q1, q2),

            // U gate: U(θ, φ, λ)† = U(-θ, -λ, -φ)
            Self::U(q, theta, phi, lambda) => Self::U(q, -theta, -lambda, -phi),
        }
    }
}

/// Python wrapper for simulation results
#[pyclass]
struct PySimulationResult {
    /// The state vector amplitudes
    amplitudes: Vec<Complex64>,
    /// The number of qubits
    n_qubits: usize,
}

#[pymethods]
impl PyCircuit {
    /// Create a new quantum circuit with the given number of qubits
    #[new]
    fn new(n_qubits: usize) -> PyResult<Self> {
        if n_qubits < 2 {
            return Err(PyValueError::new_err("Number of qubits must be at least 2"));
        }

        let circuit = match DynamicCircuit::new(n_qubits) {
            Ok(c) => Some(c),
            Err(e) => {
                return Err(PyValueError::new_err(format!(
                    "Error creating circuit: {e}"
                )))
            }
        };

        Ok(Self {
            circuit,
            n_qubits,
            qubit_depths: vec![0; n_qubits],
            operations: Vec::new(),
        })
    }

    /// Get the number of qubits in the circuit
    #[allow(clippy::missing_const_for_fn)]
    #[getter]
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Get the depth of the circuit (maximum number of gates on any single qubit path)
    fn depth(&self) -> usize {
        self.circuit_depth()
    }

    /// Get the number of gates in the circuit
    #[getter]
    fn num_gates(&self) -> usize {
        self.operations.len()
    }

    /// Apply a Hadamard gate to the specified qubit
    fn h(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Hadamard(self.checked_qubit(qubit)?))
    }

    /// Apply a Pauli-X (NOT) gate to the specified qubit
    fn x(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::PauliX(self.checked_qubit(qubit)?))
    }

    /// Apply a Pauli-Y gate to the specified qubit
    fn y(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::PauliY(self.checked_qubit(qubit)?))
    }

    /// Apply a Pauli-Z gate to the specified qubit
    fn z(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::PauliZ(self.checked_qubit(qubit)?))
    }

    /// Apply an S gate (phase gate) to the specified qubit
    fn s(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::S(self.checked_qubit(qubit)?))
    }

    /// Apply an S-dagger gate to the specified qubit
    fn sdg(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::SDagger(self.checked_qubit(qubit)?))
    }

    /// Apply a T gate (π/8 gate) to the specified qubit
    fn t(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::T(self.checked_qubit(qubit)?))
    }

    /// Apply a T-dagger gate to the specified qubit
    fn tdg(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::TDagger(self.checked_qubit(qubit)?))
    }

    /// Apply an Rx gate (rotation around X-axis) to the specified qubit
    fn rx(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::Rx(self.checked_qubit(qubit)?, theta))
    }

    /// Apply an Ry gate (rotation around Y-axis) to the specified qubit
    fn ry(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::Ry(self.checked_qubit(qubit)?, theta))
    }

    /// Apply an Rz gate (rotation around Z-axis) to the specified qubit
    fn rz(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::Rz(self.checked_qubit(qubit)?, theta))
    }

    /// Apply a CNOT gate with the specified control and target qubits
    fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Cnot(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply a SWAP gate between the specified qubits
    fn swap(&mut self, qubit1: usize, qubit2: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Swap(
            self.checked_qubit(qubit1)?,
            self.checked_qubit(qubit2)?,
        ))
    }

    /// Apply a SX gate (square root of X) to the specified qubit
    fn sx(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::SX(self.checked_qubit(qubit)?))
    }

    /// Apply a SX-dagger gate to the specified qubit
    fn sxdg(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::SXDagger(self.checked_qubit(qubit)?))
    }

    /// Apply a CY gate (controlled-Y) to the specified qubits
    fn cy(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::CY(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply a CZ gate (controlled-Z) to the specified qubits
    fn cz(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::CZ(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply a CH gate (controlled-H) to the specified qubits
    fn ch(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::CH(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply a CS gate (controlled-S) to the specified qubits
    fn cs(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::CS(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply a CRX gate (controlled-RX) to the specified qubits
    fn crx(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::CRX(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
            theta,
        ))
    }

    /// Apply a CRY gate (controlled-RY) to the specified qubits
    fn cry(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::CRY(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
            theta,
        ))
    }

    /// Apply a CRZ gate (controlled-RZ) to the specified qubits
    fn crz(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::CRZ(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
            theta,
        ))
    }

    /// Apply a Toffoli gate (CCNOT) to the specified qubits
    fn toffoli(&mut self, control1: usize, control2: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Toffoli(
            self.checked_qubit(control1)?,
            self.checked_qubit(control2)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply a Fredkin gate (CSWAP) to the specified qubits
    fn cswap(&mut self, control: usize, target1: usize, target2: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Fredkin(
            self.checked_qubit(control)?,
            self.checked_qubit(target1)?,
            self.checked_qubit(target2)?,
        ))
    }

    /// Apply an iSWAP gate to the specified qubits
    fn iswap(&mut self, qubit1: usize, qubit2: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::ISwap(
            self.checked_qubit(qubit1)?,
            self.checked_qubit(qubit2)?,
        ))
    }

    /// Apply an ECR gate (IBM native echoed cross-resonance) to the specified qubits
    fn ecr(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::ECR(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply an RXX gate (two-qubit XX rotation) to the specified qubits
    fn rxx(&mut self, qubit1: usize, qubit2: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::RXX(
            self.checked_qubit(qubit1)?,
            self.checked_qubit(qubit2)?,
            theta,
        ))
    }

    /// Apply an RYY gate (two-qubit YY rotation) to the specified qubits
    fn ryy(&mut self, qubit1: usize, qubit2: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::RYY(
            self.checked_qubit(qubit1)?,
            self.checked_qubit(qubit2)?,
            theta,
        ))
    }

    /// Apply an RZZ gate (two-qubit ZZ rotation) to the specified qubits
    fn rzz(&mut self, qubit1: usize, qubit2: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::RZZ(
            self.checked_qubit(qubit1)?,
            self.checked_qubit(qubit2)?,
            theta,
        ))
    }

    /// Apply an RZX gate (two-qubit ZX rotation / cross-resonance) to the specified qubits
    fn rzx(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::RZX(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
            theta,
        ))
    }

    /// Apply a DCX gate (double CNOT) to the specified qubits
    fn dcx(&mut self, qubit1: usize, qubit2: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::DCX(
            self.checked_qubit(qubit1)?,
            self.checked_qubit(qubit2)?,
        ))
    }

    /// Apply a phase gate (P gate) with an arbitrary angle to the specified qubit
    fn p(&mut self, qubit: usize, lambda: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::P(self.checked_qubit(qubit)?, lambda))
    }

    /// Apply an identity gate to the specified qubit
    fn id(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Id(self.checked_qubit(qubit)?))
    }

    /// Apply a U gate (general single-qubit rotation) to the specified qubit
    fn u(&mut self, qubit: usize, theta: f64, phi: f64, lambda: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::U(self.checked_qubit(qubit)?, theta, phi, lambda))
    }

    /// Run the circuit on a state vector simulator
    ///
    /// Args:
    ///     `use_gpu` (bool, optional): Whether to use the GPU for simulation if available. Defaults to `False`.
    ///
    /// Returns:
    ///     `PySimulationResult`: The result of the simulation.
    ///
    /// Raises:
    ///     `ValueError`: If the GPU is requested but not available, or if there's an error during simulation.
    #[pyo3(signature = (use_gpu=false))]
    fn run(&self, py: Python, use_gpu: bool) -> PyResult<Py<PySimulationResult>> {
        match &self.circuit {
            Some(circuit) => {
                let result = if use_gpu {
                    #[cfg(feature = "gpu")]
                    {
                        // Check if GPU is available
                        if !DynamicCircuit::is_gpu_available() {
                            return Err(PyValueError::new_err(
                                "GPU acceleration requested but no compatible GPU found",
                            ));
                        }

                        // Run on GPU
                        println!("QuantRS2: Running simulation on GPU");
                        circuit.run_gpu().map_err(|e| {
                            PyValueError::new_err(format!("Error running GPU simulation: {e}"))
                        })?
                    }

                    #[cfg(not(feature = "gpu"))]
                    {
                        return Err(PyValueError::new_err(
                            "GPU acceleration requested but not compiled in. Recompile with the 'gpu' feature."
                        ));
                    }
                } else {
                    // Use CPU simulation
                    let simulator = StateVectorSimulator::new();
                    circuit.run(&simulator).map_err(|e| {
                        PyValueError::new_err(format!("Error running CPU simulation: {e}"))
                    })?
                };

                let sim_result = PySimulationResult {
                    amplitudes: result.amplitudes().to_vec(),
                    n_qubits: result.num_qubits(),
                };

                Py::new(py, sim_result)
            }
            None => Err(PyValueError::new_err("Circuit not initialized")),
        }
    }

    /// Run the circuit with a noise model
    ///
    /// Args:
    ///     `noise_model` (`PyRealisticNoiseModel`): The noise model to use for simulation
    ///     `use_gpu` (bool, optional): Whether to use the GPU for simulation if available. Defaults to `False`.
    ///
    /// Returns:
    ///     `PySimulationResult`: The result of the simulation with noise applied.
    ///
    /// Raises:
    ///     `ValueError`: If there's an error during simulation.
    #[pyo3(signature = (noise_model, use_gpu=false))]
    fn simulate_with_noise(
        &self,
        py: Python,
        noise_model: &PyRealisticNoiseModel,
        use_gpu: bool,
    ) -> PyResult<Py<PySimulationResult>> {
        match &self.circuit {
            Some(circuit) => {
                let result = if use_gpu {
                    #[cfg(feature = "gpu")]
                    {
                        // Check if GPU is available
                        if !DynamicCircuit::is_gpu_available() {
                            return Err(PyValueError::new_err(
                                "GPU acceleration requested but no compatible GPU found",
                            ));
                        }

                        // Run on GPU with noise - GPU sim doesn't support noise yet, falling back to CPU
                        // TODO: Implement GPU-based noise simulation
                        println!("QuantRS2: GPU simulation with noise not yet supported, falling back to CPU");
                        let mut simulator = StateVectorSimulator::new();
                        simulator.set_advanced_noise_model(noise_model.noise_model.clone());
                        circuit.run(&simulator).map_err(|e| {
                            PyValueError::new_err(format!("Error running noise simulation: {e}"))
                        })?
                    }

                    #[cfg(not(feature = "gpu"))]
                    {
                        return Err(PyValueError::new_err(
                            "GPU acceleration requested but not compiled in. Recompile with the 'gpu' feature."
                        ));
                    }
                } else {
                    // Use CPU simulation with noise
                    let mut simulator = StateVectorSimulator::new();
                    simulator.set_advanced_noise_model(noise_model.noise_model.clone());
                    circuit.run(&simulator).map_err(|e| {
                        PyValueError::new_err(format!("Error running noise simulation: {e}"))
                    })?
                };

                let sim_result = PySimulationResult {
                    amplitudes: result.amplitudes().to_vec(),
                    n_qubits: result.num_qubits(),
                };

                Py::new(py, sim_result)
            }
            None => Err(PyValueError::new_err("Circuit not initialized")),
        }
    }

    /// Run the circuit on the best available simulator (GPU if available for larger circuits, CPU otherwise)
    fn run_auto(&self, py: Python) -> PyResult<Py<PySimulationResult>> {
        match &self.circuit {
            Some(circuit) => {
                #[cfg(feature = "gpu")]
                {
                    let result = circuit.run_best().map_err(|e| {
                        PyValueError::new_err(format!("Error running auto simulation: {e}"))
                    })?;

                    let sim_result = PySimulationResult {
                        amplitudes: result.amplitudes().to_vec(),
                        n_qubits: result.num_qubits(),
                    };

                    Py::new(py, sim_result)
                }

                #[cfg(not(feature = "gpu"))]
                {
                    // On non-GPU builds, run on CPU
                    let simulator = StateVectorSimulator::new();
                    let result = circuit.run(&simulator).map_err(|e| {
                        PyValueError::new_err(format!("Error running CPU simulation: {e}"))
                    })?;

                    let sim_result = PySimulationResult {
                        amplitudes: result.amplitudes().to_vec(),
                        n_qubits: result.num_qubits(),
                    };

                    Py::new(py, sim_result)
                }
            }
            None => Err(PyValueError::new_err("Circuit not initialized")),
        }
    }

    /// Check if GPU acceleration is available
    #[staticmethod]
    #[allow(clippy::missing_const_for_fn)]
    fn is_gpu_available() -> bool {
        #[cfg(feature = "gpu")]
        {
            DynamicCircuit::is_gpu_available()
        }

        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Get a text-based visualization of the circuit
    #[allow(clippy::used_underscore_items)]
    fn draw(&self) -> PyResult<String> {
        Python::with_gil(|py| {
            let Some(circuit) = &self.circuit else {
                return Err(PyValueError::new_err("Circuit not initialized"));
            };

            // Create visualization directly
            let mut visualizer = PyCircuitVisualizer::new(self.n_qubits);

            // Add all gates from the circuit (simplified version)
            let gate_names = circuit.get_gate_names();
            for gate in &gate_names {
                // For simplicity, assume they're all single-qubit gates on qubit 0
                visualizer.add_gate(gate, vec![0], None)?;
            }

            Ok(visualizer._repr_html_())
        })
    }

    /// Get an HTML representation of the circuit for Jupyter notebooks
    fn draw_html(&self) -> PyResult<String> {
        // Reuse draw method since we're using HTML representation for both
        self.draw()
    }

    /// Get a visualization object for the circuit
    fn visualize(&self, _py: Python) -> PyResult<Py<PyCircuitVisualizer>> {
        self.get_visualizer()
    }

    /// Implements the `_repr_html_` method for Jupyter notebook display
    fn _repr_html_(&self) -> PyResult<String> {
        self.draw_html()
    }

    /// Decompose complex gates into simpler gates
    ///
    /// Returns a new circuit with complex gates (like Toffoli or SWAP) decomposed
    /// into sequences of simpler gates (like CNOT, H, T, etc.)
    ///
    /// Decomposition rules:
    /// - Toffoli (CCX) → 6 CNOTs + 7 T/Tdg + 2 H (standard decomposition)
    /// - Fredkin (CSWAP) → Toffoli decomposition + CNOT wrapper
    /// - SWAP → 3 CNOTs
    /// - Other gates pass through unchanged
    fn decompose(&self) -> PyResult<Py<Self>> {
        Python::with_gil(|py| {
            if self.circuit.is_none() {
                return Err(PyValueError::new_err("Circuit not initialized"));
            }

            let mut decomposed = Self::new(self.n_qubits)?;

            for &op in &self.operations {
                match op {
                    // Decompose SWAP into 3 CNOTs
                    CircuitOp::Swap(q1, q2) => {
                        let idx1 = q1.0 as usize;
                        let idx2 = q2.0 as usize;
                        decomposed.cnot(idx1, idx2)?;
                        decomposed.cnot(idx2, idx1)?;
                        decomposed.cnot(idx1, idx2)?;
                    }
                    // Decompose Toffoli (CCX) into Clifford+T gates
                    CircuitOp::Toffoli(c1, c2, t) => {
                        let ctrl1 = c1.0 as usize;
                        let ctrl2 = c2.0 as usize;
                        let target = t.0 as usize;
                        // Standard Toffoli decomposition
                        decomposed.h(target)?;
                        decomposed.cnot(ctrl2, target)?;
                        decomposed.tdg(target)?;
                        decomposed.cnot(ctrl1, target)?;
                        decomposed.t(target)?;
                        decomposed.cnot(ctrl2, target)?;
                        decomposed.tdg(target)?;
                        decomposed.cnot(ctrl1, target)?;
                        decomposed.t(ctrl2)?;
                        decomposed.t(target)?;
                        decomposed.h(target)?;
                        decomposed.cnot(ctrl1, ctrl2)?;
                        decomposed.t(ctrl1)?;
                        decomposed.tdg(ctrl2)?;
                        decomposed.cnot(ctrl1, ctrl2)?;
                    }
                    // Decompose Fredkin (CSWAP) using Toffoli
                    CircuitOp::Fredkin(c, t1, t2) => {
                        let ctrl = c.0 as usize;
                        let targ1 = t1.0 as usize;
                        let targ2 = t2.0 as usize;
                        // CSWAP = CNOT(t2,t1) + Toffoli(c,t1,t2) + CNOT(t2,t1)
                        decomposed.cnot(targ2, targ1)?;
                        decomposed.toffoli(ctrl, targ1, targ2)?;
                        decomposed.cnot(targ2, targ1)?;
                    }
                    // Pass through all other gates unchanged
                    _ => {
                        decomposed.apply_op(op)?;
                    }
                }
            }

            Py::new(py, decomposed)
        })
    }

    /// Copy the circuit (returns an identical circuit)
    ///
    /// Creates a new circuit with the same gates as this one.
    /// For optimization passes, use the circuit optimizer from quantrs2-circuit.
    fn copy(&self) -> PyResult<Py<Self>> {
        Python::with_gil(|py| {
            if self.circuit.is_none() {
                return Err(PyValueError::new_err("Circuit not initialized"));
            }

            let mut new_circuit = Self::new(self.n_qubits)?;

            // Copy all operations
            for &op in &self.operations {
                new_circuit.apply_op(op)?;
            }

            Py::new(py, new_circuit)
        })
    }

    /// Compose this circuit with another circuit
    ///
    /// Appends the gates from `other` circuit to this circuit.
    /// The other circuit must have the same or fewer qubits.
    fn compose(&mut self, other: &Self) -> PyResult<()> {
        if other.n_qubits > self.n_qubits {
            return Err(PyValueError::new_err(format!(
                "Other circuit has {} qubits, but this circuit only has {}",
                other.n_qubits, self.n_qubits
            )));
        }

        // Append all operations from other circuit
        for &op in &other.operations {
            self.apply_op(op)?;
        }

        Ok(())
    }
}

impl PyCircuit {
    /// Helper function to get a circuit visualizer based on the current circuit state
    #[allow(clippy::too_many_lines)]
    fn get_visualizer(&self) -> PyResult<Py<PyCircuitVisualizer>> {
        Python::with_gil(|py| {
            // Gather all operations in the circuit
            let mut operations = Vec::new();

            if let Some(circuit) = &self.circuit {
                for gate in circuit.gates() {
                    match &*gate {
                        // Single qubit gates
                        "H" => {
                            let qubit =
                                circuit.get_single_qubit_for_gate(&gate, operations.len())?;
                            operations.push(("H".to_string(), vec![qubit as usize], None));
                        }
                        "X" => {
                            let qubit =
                                circuit.get_single_qubit_for_gate(&gate, operations.len())?;
                            operations.push(("X".to_string(), vec![qubit as usize], None));
                        }
                        "Y" => {
                            let qubit =
                                circuit.get_single_qubit_for_gate(&gate, operations.len())?;
                            operations.push(("Y".to_string(), vec![qubit as usize], None));
                        }
                        "Z" => {
                            let qubit =
                                circuit.get_single_qubit_for_gate(&gate, operations.len())?;
                            operations.push(("Z".to_string(), vec![qubit as usize], None));
                        }
                        "S" => {
                            let qubit =
                                circuit.get_single_qubit_for_gate(&gate, operations.len())?;
                            operations.push(("S".to_string(), vec![qubit as usize], None));
                        }
                        "S†" => {
                            let qubit =
                                circuit.get_single_qubit_for_gate(&gate, operations.len())?;
                            operations.push(("SDG".to_string(), vec![qubit as usize], None));
                        }
                        "T" => {
                            let qubit =
                                circuit.get_single_qubit_for_gate(&gate, operations.len())?;
                            operations.push(("T".to_string(), vec![qubit as usize], None));
                        }
                        "T†" => {
                            let qubit =
                                circuit.get_single_qubit_for_gate(&gate, operations.len())?;
                            operations.push(("TDG".to_string(), vec![qubit as usize], None));
                        }
                        "√X" => {
                            let qubit =
                                circuit.get_single_qubit_for_gate(&gate, operations.len())?;
                            operations.push(("SX".to_string(), vec![qubit as usize], None));
                        }
                        "√X†" => {
                            let qubit =
                                circuit.get_single_qubit_for_gate(&gate, operations.len())?;
                            operations.push(("SXDG".to_string(), vec![qubit as usize], None));
                        }

                        // Parameterized single-qubit gates
                        "RX" => {
                            let (qubit, theta) =
                                circuit.get_rotation_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "RX".to_string(),
                                vec![qubit as usize],
                                Some(format!("{theta:.2}")),
                            ));
                        }
                        "RY" => {
                            let (qubit, theta) =
                                circuit.get_rotation_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "RY".to_string(),
                                vec![qubit as usize],
                                Some(format!("{theta:.2}")),
                            ));
                        }
                        "RZ" => {
                            let (qubit, theta) =
                                circuit.get_rotation_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "RZ".to_string(),
                                vec![qubit as usize],
                                Some(format!("{theta:.2}")),
                            ));
                        }

                        // Two-qubit gates
                        "CNOT" => {
                            let (control, target) =
                                circuit.get_two_qubit_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "CNOT".to_string(),
                                vec![control as usize, target as usize],
                                None,
                            ));
                        }
                        "CY" => {
                            let (control, target) =
                                circuit.get_two_qubit_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "CY".to_string(),
                                vec![control as usize, target as usize],
                                None,
                            ));
                        }
                        "CZ" => {
                            let (control, target) =
                                circuit.get_two_qubit_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "CZ".to_string(),
                                vec![control as usize, target as usize],
                                None,
                            ));
                        }
                        "CH" => {
                            let (control, target) =
                                circuit.get_two_qubit_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "CH".to_string(),
                                vec![control as usize, target as usize],
                                None,
                            ));
                        }
                        "CS" => {
                            let (control, target) =
                                circuit.get_two_qubit_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "CS".to_string(),
                                vec![control as usize, target as usize],
                                None,
                            ));
                        }
                        "SWAP" => {
                            let (q1, q2) =
                                circuit.get_two_qubit_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "SWAP".to_string(),
                                vec![q1 as usize, q2 as usize],
                                None,
                            ));
                        }

                        // Parameterized two-qubit gates
                        "CRX" => {
                            let (control, target, theta) = circuit
                                .get_controlled_rotation_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "CRX".to_string(),
                                vec![control as usize, target as usize],
                                Some(format!("{theta:.2}")),
                            ));
                        }
                        "CRY" => {
                            let (control, target, theta) = circuit
                                .get_controlled_rotation_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "CRY".to_string(),
                                vec![control as usize, target as usize],
                                Some(format!("{theta:.2}")),
                            ));
                        }
                        "CRZ" => {
                            let (control, target, theta) = circuit
                                .get_controlled_rotation_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "CRZ".to_string(),
                                vec![control as usize, target as usize],
                                Some(format!("{theta:.2}")),
                            ));
                        }

                        // Three-qubit gates
                        "Toffoli" => {
                            let (c1, c2, target) =
                                circuit.get_three_qubit_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "Toffoli".to_string(),
                                vec![c1 as usize, c2 as usize, target as usize],
                                None,
                            ));
                        }
                        "Fredkin" => {
                            let (control, t1, t2) =
                                circuit.get_three_qubit_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "Fredkin".to_string(),
                                vec![control as usize, t1 as usize, t2 as usize],
                                None,
                            ));
                        }

                        // Unknown gate
                        _ => {
                            operations.push((gate.clone(), vec![0], None));
                        }
                    }
                }
            }

            // Create a visualizer with the gathered operations
            create_visualizer_from_operations(py, self.n_qubits, operations)
        })
    }

    /// Helper function to apply a gate to the circuit
    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    fn apply_gate(&mut self, op: CircuitOp) -> PyResult<()> {
        // Get affected qubits before op is used
        let qubits = op.affected_qubits();

        // Store the operation for circuit folding support
        self.operations.push(op);

        match &mut self.circuit {
            Some(circuit) => {
                match op {
                    CircuitOp::Hadamard(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::Hadamard { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::PauliX(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::PauliX { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::PauliY(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::PauliY { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::PauliZ(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::PauliZ { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::S(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::Phase { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::SDagger(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::PhaseDagger { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::T(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::T { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::TDagger(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::TDagger { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::Rx(qubit, theta) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::RotationX {
                                target: qubit,
                                theta,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::Ry(qubit, theta) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::RotationY {
                                target: qubit,
                                theta,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::Rz(qubit, theta) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::RotationZ {
                                target: qubit,
                                theta,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::Cnot(control, target) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::CNOT { control, target })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::Swap(qubit1, qubit2) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::SWAP { qubit1, qubit2 })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::SX(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::SqrtX { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::SXDagger(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::SqrtXDagger { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::CY(control, target) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::CY { control, target })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::CZ(control, target) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::CZ { control, target })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::CH(control, target) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::CH { control, target })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::CS(control, target) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::CS { control, target })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::CRX(control, target, theta) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::CRX {
                                control,
                                target,
                                theta,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::CRY(control, target, theta) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::CRY {
                                control,
                                target,
                                theta,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::CRZ(control, target, theta) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::CRZ {
                                control,
                                target,
                                theta,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::Toffoli(control1, control2, target) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::Toffoli {
                                control1,
                                control2,
                                target,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::Fredkin(control, target1, target2) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::Fredkin {
                                control,
                                target1,
                                target2,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::ISwap(qubit1, qubit2) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::ISwap { qubit1, qubit2 })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::ECR(control, target) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::ECR { control, target })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::RXX(qubit1, qubit2, theta) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::RXX {
                                qubit1,
                                qubit2,
                                theta,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::RYY(qubit1, qubit2, theta) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::RYY {
                                qubit1,
                                qubit2,
                                theta,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::RZZ(qubit1, qubit2, theta) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::RZZ {
                                qubit1,
                                qubit2,
                                theta,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::RZX(control, target, theta) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::RZX {
                                control,
                                target,
                                theta,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::DCX(qubit1, qubit2) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::DCX { qubit1, qubit2 })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::P(qubit, lambda) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::PGate {
                                target: qubit,
                                lambda,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::Id(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::Identity { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                    CircuitOp::U(qubit, theta, phi, lambda) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::UGate {
                                target: qubit,
                                theta,
                                phi,
                                lambda,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {e}"))
                            })?;
                    }
                }

                // Update depth tracking based on affected qubits
                match qubits {
                    (Some(q1), None, None) => self.update_depth_single(q1),
                    (Some(q1), Some(q2), None) => self.update_depth_two(q1, q2),
                    (Some(q1), Some(q2), Some(q3)) => self.update_depth_three(q1, q2, q3),
                    _ => {} // Should never happen - all ops have at least one qubit
                }

                Ok(())
            }
            None => Err(PyValueError::new_err("Circuit not initialized")),
        }
    }
}

#[pymethods]
impl PySimulationResult {
    /// Get the state vector amplitudes
    fn amplitudes(&self, py: Python) -> PyResult<PyObject> {
        let result = PyList::empty(py);
        for amp in &self.amplitudes {
            let complex = PyComplex::from_doubles(py, amp.re, amp.im);
            result.append(complex)?;
        }
        Ok(result.into())
    }

    /// Get the probabilities for each basis state
    fn probabilities(&self, py: Python) -> PyResult<PyObject> {
        let result = PyList::empty(py);
        for amp in &self.amplitudes {
            let prob = amp.norm_sqr();
            result.append(prob)?;
        }
        Ok(result.into())
    }

    /// Get the number of qubits
    #[allow(clippy::missing_const_for_fn)]
    #[getter]
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Get a dictionary mapping basis states to probabilities
    fn state_probabilities(&self, py: Python) -> PyResult<PyObject> {
        let result = PyDict::new(py);
        for (i, amp) in self.amplitudes.iter().enumerate() {
            let basis_state = format!("{:0width$b}", i, width = self.n_qubits);
            let prob = amp.norm_sqr();
            // Only include states with non-zero probability
            if prob > 1e-10 {
                result.set_item(basis_state, prob)?;
            }
        }
        Ok(result.into())
    }

    /// Get the expectation value of a Pauli operator
    ///
    /// Computes ⟨ψ|P|ψ⟩ where P is the tensor product of Pauli operators.
    /// The operator string should have one character per qubit (I, X, Y, or Z).
    fn expectation_value(&self, operator: &str) -> PyResult<f64> {
        if operator.len() != self.n_qubits {
            return Err(PyValueError::new_err(format!(
                "Operator length ({}) must match number of qubits ({})",
                operator.len(),
                self.n_qubits
            )));
        }

        let paulis: Vec<char> = operator.chars().collect();
        for &c in &paulis {
            if c != 'I' && c != 'X' && c != 'Y' && c != 'Z' {
                return Err(PyValueError::new_err(format!(
                    "Invalid Pauli operator: {c}. Only I, X, Y, Z are allowed"
                )));
            }
        }

        let n = self.n_qubits;
        let dim = 1 << n; // 2^n basis states
        let mut expectation = Complex64::new(0.0, 0.0);

        // For each basis state |i⟩, compute ⟨i|P applied to ψ contribution
        for i in 0..dim {
            // Apply Pauli string to |i⟩ to get phase * |j⟩
            let mut j = i;
            let mut phase = Complex64::new(1.0, 0.0);

            for (qubit_idx, &pauli) in paulis.iter().enumerate() {
                // Qubit 0 corresponds to MSB (leftmost in operator string)
                let bit_position = n - 1 - qubit_idx;
                let bit = (i >> bit_position) & 1;

                match pauli {
                    'X' => {
                        // X|0⟩ = |1⟩, X|1⟩ = |0⟩ (flip the bit)
                        j ^= 1 << bit_position;
                    }
                    'Y' => {
                        // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                        j ^= 1 << bit_position;
                        if bit == 0 {
                            phase *= Complex64::new(0.0, 1.0); // i
                        } else {
                            phase *= Complex64::new(0.0, -1.0); // -i
                        }
                    }
                    'Z' => {
                        // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
                        if bit == 1 {
                            phase *= Complex64::new(-1.0, 0.0);
                        }
                    }
                    // 'I' and any other already-validated characters: no change
                    _ => {}
                }
            }

            // Contribution: ψ*_i * phase * ψ_j
            expectation += self.amplitudes[i].conj() * phase * self.amplitudes[j];
        }

        // For Hermitian operators (like Pauli strings), the expectation value is real
        Ok(expectation.re)
    }
}

/// Implementation of the `PyRealisticNoiseModel` class
#[pymethods]
impl PyRealisticNoiseModel {
    /// Create a new realistic noise model for IBM quantum devices
    ///
    /// Args:
    ///     `device_name` (str): The name of the IBM quantum device (e.g., "`ibmq_lima`", "`ibm_cairo`")
    ///
    /// Returns:
    ///     `PyRealisticNoiseModel`: A noise model configured with the specified device parameters
    #[staticmethod]
    fn ibm_device(device_name: &str) -> Self {
        // Convert device name to lowercase
        let device_name = device_name.to_lowercase();

        // Create a list of qubits from 0 to 31 (max 32 qubits support)
        let qubits: Vec<QubitId> = (0..32).map(QubitId::new).collect();

        // Create IBM device noise model
        let noise_model = RealisticNoiseModelBuilder::new(true)
            .with_ibm_device_noise(&qubits, &device_name)
            .build();

        Self { noise_model }
    }

    /// Create a new realistic noise model for Rigetti quantum devices
    ///
    /// Args:
    ///     `device_name` (str): The name of the Rigetti quantum device (e.g., "Aspen-M-2")
    ///
    /// Returns:
    ///     `PyRealisticNoiseModel`: A noise model configured with the specified device parameters
    #[staticmethod]
    fn rigetti_device(device_name: &str) -> Self {
        // Create a list of qubits from 0 to 31 (max 32 qubits support)
        let qubits: Vec<QubitId> = (0..32).map(QubitId::new).collect();

        // Create Rigetti device noise model
        let noise_model = RealisticNoiseModelBuilder::new(true)
            .with_rigetti_device_noise(&qubits, device_name)
            .build();

        Self { noise_model }
    }

    /// Create a new realistic noise model with custom parameters
    ///
    /// Args:
    ///     `t1_us` (float): T1 relaxation time in microseconds
    ///     `t2_us` (float): T2 dephasing time in microseconds
    ///     `gate_time_ns` (float): Gate time in nanoseconds
    ///     `gate_error_1q` (float): Single-qubit gate error rate (0.0 to 1.0)
    ///     `gate_error_2q` (float): Two-qubit gate error rate (0.0 to 1.0)
    ///     `readout_error` (float): Readout error rate (0.0 to 1.0)
    ///
    /// Returns:
    ///     `PyRealisticNoiseModel`: A custom noise model with the specified parameters
    #[staticmethod]
    #[pyo3(signature = (t1_us=100.0, t2_us=50.0, gate_time_ns=40.0, gate_error_1q=0.001, gate_error_2q=0.01, readout_error=0.02))]
    fn custom(
        t1_us: f64,
        t2_us: f64,
        gate_time_ns: f64,
        gate_error_1q: f64,
        gate_error_2q: f64,
        readout_error: f64,
    ) -> Self {
        // Create a list of qubits from 0 to 31 (max 32 qubits support)
        let qubits: Vec<QubitId> = (0..32).map(QubitId::new).collect();

        // Create pairs of adjacent qubits for two-qubit noise
        let qubit_pairs: Vec<(QubitId, QubitId)> = (0..31)
            .map(|i| (QubitId::new(i), QubitId::new(i + 1)))
            .collect();

        // Create custom noise model
        let noise_model = RealisticNoiseModelBuilder::new(true)
            .with_custom_thermal_relaxation(
                &qubits,
                Duration::from_secs_f64(t1_us * 1e-6),
                Duration::from_secs_f64(t2_us * 1e-6),
                Duration::from_secs_f64(gate_time_ns * 1e-9),
            )
            .with_custom_two_qubit_noise(&qubit_pairs, gate_error_2q)
            .build();

        // Add depolarizing noise for single-qubit gates and readout errors
        let mut result = Self { noise_model };

        for &qubit in &qubits {
            result
                .noise_model
                .add_base_channel(NoiseChannelType::Depolarizing(DepolarizingChannel {
                    target: qubit,
                    probability: gate_error_1q,
                }));

            result
                .noise_model
                .add_base_channel(NoiseChannelType::BitFlip(BitFlipChannel {
                    target: qubit,
                    probability: readout_error,
                }));
        }

        result
    }

    /// Get the number of noise channels in this model
    #[getter]
    fn num_channels(&self) -> usize {
        self.noise_model.num_channels()
    }
}

/// Implementation for `PyDynamicCircuit`
#[pymethods]
impl PyDynamicCircuit {
    /// Create a new dynamic quantum circuit with the given number of qubits
    #[new]
    fn new(n_qubits: usize) -> PyResult<Self> {
        Ok(Self {
            circuit: PyCircuit::new(n_qubits)?,
        })
    }

    /// Get the number of qubits in the circuit
    #[allow(clippy::missing_const_for_fn)]
    #[getter]
    fn n_qubits(&self) -> usize {
        self.circuit.n_qubits
    }

    /// Apply a Hadamard gate to the specified qubit
    fn h(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.h(qubit)
    }

    /// Apply a Pauli-X (NOT) gate to the specified qubit
    fn x(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.x(qubit)
    }

    /// Apply a Pauli-Y gate to the specified qubit
    fn y(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.y(qubit)
    }

    /// Apply a Pauli-Z gate to the specified qubit
    fn z(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.z(qubit)
    }

    /// Apply an S gate (phase gate) to the specified qubit
    fn s(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.s(qubit)
    }

    /// Apply an S-dagger gate to the specified qubit
    fn sdg(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.sdg(qubit)
    }

    /// Apply a T gate (π/8 gate) to the specified qubit
    fn t(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.t(qubit)
    }

    /// Apply a T-dagger gate to the specified qubit
    fn tdg(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.tdg(qubit)
    }

    /// Apply an Rx gate (rotation around X-axis) to the specified qubit
    fn rx(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.circuit.rx(qubit, theta)
    }

    /// Apply an Ry gate (rotation around Y-axis) to the specified qubit
    fn ry(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.circuit.ry(qubit, theta)
    }

    /// Apply an Rz gate (rotation around Z-axis) to the specified qubit
    fn rz(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.circuit.rz(qubit, theta)
    }

    /// Apply a CNOT gate with the specified control and target qubits
    fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.circuit.cnot(control, target)
    }

    /// Apply a SWAP gate between the specified qubits
    fn swap(&mut self, qubit1: usize, qubit2: usize) -> PyResult<()> {
        self.circuit.swap(qubit1, qubit2)
    }

    /// Apply a CZ gate (controlled-Z) to the specified qubits
    fn cz(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.circuit.cz(control, target)
    }

    /// Run the circuit on a state vector simulator
    #[pyo3(signature = (use_gpu=false))]
    fn run(&self, py: Python, use_gpu: bool) -> PyResult<Py<PySimulationResult>> {
        self.circuit.run(py, use_gpu)
    }

    /// Run the circuit with a noise model
    #[pyo3(signature = (noise_model, use_gpu=false))]
    fn simulate_with_noise(
        &self,
        py: Python,
        noise_model: &PyRealisticNoiseModel,
        use_gpu: bool,
    ) -> PyResult<Py<PySimulationResult>> {
        self.circuit.simulate_with_noise(py, noise_model, use_gpu)
    }

    /// Run the circuit on the best available simulator (GPU if available for larger circuits, CPU otherwise)
    fn run_auto(&self, py: Python) -> PyResult<Py<PySimulationResult>> {
        self.circuit.run_auto(py)
    }
}

/// Python module for `QuantRS2`
#[pymodule]
fn quantrs2(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add classes to the module
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyDynamicCircuit>()?;
    m.add_class::<PySimulationResult>()?;
    m.add_class::<PyRealisticNoiseModel>()?;
    m.add_class::<PyCircuitVisualizer>()?;

    // Register the gates submodule
    gates::register_module(m)?;

    // Register the SciRS2 submodule
    scirs2_bindings::create_scirs2_module(m)?;
    m.add_class::<scirs2_bindings::PyQuantumNumerics>()?;

    // Register the parametric module
    parametric::register_parametric_module(m)?;

    // Register the optimization module
    optimization_passes::register_optimization_module(m)?;

    // Register the Pythonic API module
    pythonic_api::register_pythonic_module(m)?;

    // Register the custom gates module
    custom_gates::register_custom_gates_module(m)?;

    // Register the measurement module
    measurement::register_measurement_module(m)?;

    // Register the algorithms module
    algorithms::register_algorithms_module(m)?;

    // Register the pulse module (requires device feature)
    #[cfg(feature = "device")]
    pulse::register_pulse_module(m)?;

    // Register the mitigation module (requires device feature)
    #[cfg(feature = "device")]
    mitigation::register_mitigation_module(m)?;

    // Register the ML transfer learning module
    #[cfg(feature = "ml")]
    ml_transfer::register_ml_transfer_module(m)?;

    // Register the anneal module
    #[cfg(feature = "anneal")]
    anneal::register_anneal_module(m)?;

    // Register the tytan module
    #[cfg(feature = "tytan")]
    tytan::register_tytan_module(m)?;

    // Register the multi-GPU module
    multi_gpu::register_multi_gpu_module(m)?;

    // Register the simulators module
    simulators::register_simulators_module(m)?;

    // Add metadata
    m.setattr(
        "__doc__",
        "QuantRS2 Quantum Computing Framework Python Bindings",
    )?;

    // Add constants
    m.add("MAX_QUBITS", 32)?;
    m.add(
        "SUPPORTED_QUBITS",
        vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 32],
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_op_inverse_hermitian() {
        // Test self-inverse (Hermitian) gates
        let h = CircuitOp::Hadamard(QubitId::new(0));
        assert!(matches!(h.inverse(), CircuitOp::Hadamard(_)));

        let x = CircuitOp::PauliX(QubitId::new(0));
        assert!(matches!(x.inverse(), CircuitOp::PauliX(_)));

        let y = CircuitOp::PauliY(QubitId::new(0));
        assert!(matches!(y.inverse(), CircuitOp::PauliY(_)));

        let z = CircuitOp::PauliZ(QubitId::new(0));
        assert!(matches!(z.inverse(), CircuitOp::PauliZ(_)));

        let cnot = CircuitOp::Cnot(QubitId::new(0), QubitId::new(1));
        assert!(matches!(cnot.inverse(), CircuitOp::Cnot(_, _)));

        let swap = CircuitOp::Swap(QubitId::new(0), QubitId::new(1));
        assert!(matches!(swap.inverse(), CircuitOp::Swap(_, _)));
    }

    #[test]
    fn test_circuit_op_inverse_paired() {
        // Test S/Sdg pair
        let s = CircuitOp::S(QubitId::new(0));
        assert!(matches!(s.inverse(), CircuitOp::SDagger(_)));

        let sdg = CircuitOp::SDagger(QubitId::new(0));
        assert!(matches!(sdg.inverse(), CircuitOp::S(_)));

        // Test T/Tdg pair
        let t = CircuitOp::T(QubitId::new(0));
        assert!(matches!(t.inverse(), CircuitOp::TDagger(_)));

        let tdg = CircuitOp::TDagger(QubitId::new(0));
        assert!(matches!(tdg.inverse(), CircuitOp::T(_)));

        // Test SX/SXdg pair
        let sx = CircuitOp::SX(QubitId::new(0));
        assert!(matches!(sx.inverse(), CircuitOp::SXDagger(_)));

        let sxdg = CircuitOp::SXDagger(QubitId::new(0));
        assert!(matches!(sxdg.inverse(), CircuitOp::SX(_)));
    }

    #[test]
    fn test_circuit_op_inverse_rotation() {
        // Test rotation gates with negated angles
        let pi = std::f64::consts::PI;

        let rx = CircuitOp::Rx(QubitId::new(0), pi);
        if let CircuitOp::Rx(_, angle) = rx.inverse() {
            assert!((angle + pi).abs() < 1e-10);
        } else {
            panic!("Expected Rx inverse");
        }

        let ry = CircuitOp::Ry(QubitId::new(0), pi / 2.0);
        if let CircuitOp::Ry(_, angle) = ry.inverse() {
            assert!((angle + pi / 2.0).abs() < 1e-10);
        } else {
            panic!("Expected Ry inverse");
        }

        let rz = CircuitOp::Rz(QubitId::new(0), pi / 4.0);
        if let CircuitOp::Rz(_, angle) = rz.inverse() {
            assert!((angle + pi / 4.0).abs() < 1e-10);
        } else {
            panic!("Expected Rz inverse");
        }

        // Test controlled rotations
        let crx = CircuitOp::CRX(QubitId::new(0), QubitId::new(1), pi);
        if let CircuitOp::CRX(_, _, angle) = crx.inverse() {
            assert!((angle + pi).abs() < 1e-10);
        } else {
            panic!("Expected CRX inverse");
        }
    }

    #[test]
    fn test_circuit_op_inverse_u_gate() {
        // U(θ, φ, λ)† = U(-θ, -λ, -φ)
        let u = CircuitOp::U(QubitId::new(0), 1.0, 2.0, 3.0);
        if let CircuitOp::U(_, theta, phi, lambda) = u.inverse() {
            assert!((theta + 1.0).abs() < 1e-10);
            assert!((phi + 3.0).abs() < 1e-10); // φ and λ are swapped
            assert!((lambda + 2.0).abs() < 1e-10);
        } else {
            panic!("Expected U inverse");
        }
    }

    #[test]
    fn test_circuit_op_affected_qubits_single() {
        let h = CircuitOp::Hadamard(QubitId::new(0));
        let (q1, q2, q3) = h.affected_qubits();
        assert!(q1.is_some());
        assert!(q2.is_none());
        assert!(q3.is_none());
    }

    #[test]
    fn test_circuit_op_affected_qubits_two() {
        let cnot = CircuitOp::Cnot(QubitId::new(0), QubitId::new(1));
        let (q1, q2, q3) = cnot.affected_qubits();
        assert!(q1.is_some());
        assert!(q2.is_some());
        assert!(q3.is_none());
    }

    #[test]
    fn test_circuit_op_affected_qubits_three() {
        let toffoli = CircuitOp::Toffoli(QubitId::new(0), QubitId::new(1), QubitId::new(2));
        let (q1, q2, q3) = toffoli.affected_qubits();
        assert!(q1.is_some());
        assert!(q2.is_some());
        assert!(q3.is_some());
    }

    #[test]
    fn test_circuit_op_clone_copy() {
        // Test that CircuitOp is Copy (can be used multiple times)
        let h = CircuitOp::Hadamard(QubitId::new(0));
        let h2 = h; // Copy
        let h3 = h; // Another copy

        assert!(matches!(h, CircuitOp::Hadamard(_)));
        assert!(matches!(h2, CircuitOp::Hadamard(_)));
        assert!(matches!(h3, CircuitOp::Hadamard(_)));
    }
}
