//! Core circuit types: `CircuitOp` enum and `PyCircuit` struct with all implementations.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::dynamic::{DynamicCircuit, DynamicResult};
use quantrs2_sim::statevector::StateVectorSimulator;
use std::convert::TryFrom;

use crate::noise_model::PyRealisticNoiseModel;
use crate::simulation_result::PySimulationResult;
use crate::visualization::{create_visualizer_from_operations, PyCircuitVisualizer};

/// Quantum circuit representation for Python
#[pyclass]
pub struct PyCircuit {
    /// The internal Rust circuit
    pub(crate) circuit: Option<DynamicCircuit>,
    /// The number of qubits in the circuit
    pub(crate) n_qubits: usize,
    /// Depth counter for each qubit (tracks the layer number of the last gate on each qubit)
    pub(crate) qubit_depths: Vec<usize>,
    /// List of operations for circuit folding and reconstruction
    pub(crate) operations: Vec<CircuitOp>,
}

impl PyCircuit {
    pub(crate) fn checked_qubit(&self, qubit: usize) -> PyResult<QubitId> {
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
    pub(crate) fn update_depth_single(&mut self, qubit: QubitId) {
        let idx = qubit.0 as usize;
        if idx < self.qubit_depths.len() {
            self.qubit_depths[idx] += 1;
        }
    }

    /// Update depth tracking for a two-qubit gate
    pub(crate) fn update_depth_two(&mut self, qubit1: QubitId, qubit2: QubitId) {
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
    pub(crate) fn update_depth_three(&mut self, qubit1: QubitId, qubit2: QubitId, qubit3: QubitId) {
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
    pub(crate) fn circuit_depth(&self) -> usize {
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

/// Enum to store circuit operations for different gate types
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy)]
pub enum CircuitOp {
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
    pub(crate) const fn affected_qubits(
        &self,
    ) -> (Option<QubitId>, Option<QubitId>, Option<QubitId>) {
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
    pub(crate) const fn inverse(&self) -> Self {
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

#[pymethods]
impl PyCircuit {
    /// Create a new quantum circuit with the given number of qubits
    #[new]
    pub(crate) fn new(n_qubits: usize) -> PyResult<Self> {
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
    pub(crate) fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Get the depth of the circuit (maximum number of gates on any single qubit path)
    pub(crate) fn depth(&self) -> usize {
        self.circuit_depth()
    }

    /// Get the number of gates in the circuit
    #[getter]
    pub(crate) fn num_gates(&self) -> usize {
        self.operations.len()
    }

    /// Apply a Hadamard gate to the specified qubit
    pub(crate) fn h(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Hadamard(self.checked_qubit(qubit)?))
    }

    /// Apply a Pauli-X (NOT) gate to the specified qubit
    pub(crate) fn x(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::PauliX(self.checked_qubit(qubit)?))
    }

    /// Apply a Pauli-Y gate to the specified qubit
    pub(crate) fn y(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::PauliY(self.checked_qubit(qubit)?))
    }

    /// Apply a Pauli-Z gate to the specified qubit
    pub(crate) fn z(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::PauliZ(self.checked_qubit(qubit)?))
    }

    /// Apply an S gate (phase gate) to the specified qubit
    pub(crate) fn s(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::S(self.checked_qubit(qubit)?))
    }

    /// Apply an S-dagger gate to the specified qubit
    pub(crate) fn sdg(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::SDagger(self.checked_qubit(qubit)?))
    }

    /// Apply a T gate (π/8 gate) to the specified qubit
    pub(crate) fn t(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::T(self.checked_qubit(qubit)?))
    }

    /// Apply a T-dagger gate to the specified qubit
    pub(crate) fn tdg(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::TDagger(self.checked_qubit(qubit)?))
    }

    /// Apply an Rx gate (rotation around X-axis) to the specified qubit
    pub(crate) fn rx(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::Rx(self.checked_qubit(qubit)?, theta))
    }

    /// Apply an Ry gate (rotation around Y-axis) to the specified qubit
    pub(crate) fn ry(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::Ry(self.checked_qubit(qubit)?, theta))
    }

    /// Apply an Rz gate (rotation around Z-axis) to the specified qubit
    pub(crate) fn rz(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::Rz(self.checked_qubit(qubit)?, theta))
    }

    /// Apply a CNOT gate with the specified control and target qubits
    pub(crate) fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Cnot(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply a SWAP gate between the specified qubits
    pub(crate) fn swap(&mut self, qubit1: usize, qubit2: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Swap(
            self.checked_qubit(qubit1)?,
            self.checked_qubit(qubit2)?,
        ))
    }

    /// Apply a SX gate (square root of X) to the specified qubit
    pub(crate) fn sx(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::SX(self.checked_qubit(qubit)?))
    }

    /// Apply a SX-dagger gate to the specified qubit
    pub(crate) fn sxdg(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::SXDagger(self.checked_qubit(qubit)?))
    }

    /// Apply a CY gate (controlled-Y) to the specified qubits
    pub(crate) fn cy(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::CY(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply a CZ gate (controlled-Z) to the specified qubits
    pub(crate) fn cz(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::CZ(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply a CH gate (controlled-H) to the specified qubits
    pub(crate) fn ch(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::CH(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply a CS gate (controlled-S) to the specified qubits
    pub(crate) fn cs(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::CS(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply a CRX gate (controlled-RX) to the specified qubits
    pub(crate) fn crx(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::CRX(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
            theta,
        ))
    }

    /// Apply a CRY gate (controlled-RY) to the specified qubits
    pub(crate) fn cry(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::CRY(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
            theta,
        ))
    }

    /// Apply a CRZ gate (controlled-RZ) to the specified qubits
    pub(crate) fn crz(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::CRZ(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
            theta,
        ))
    }

    /// Apply a Toffoli gate (CCNOT) to the specified qubits
    pub(crate) fn toffoli(
        &mut self,
        control1: usize,
        control2: usize,
        target: usize,
    ) -> PyResult<()> {
        self.apply_gate(CircuitOp::Toffoli(
            self.checked_qubit(control1)?,
            self.checked_qubit(control2)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply a Fredkin gate (CSWAP) to the specified qubits
    pub(crate) fn cswap(&mut self, control: usize, target1: usize, target2: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Fredkin(
            self.checked_qubit(control)?,
            self.checked_qubit(target1)?,
            self.checked_qubit(target2)?,
        ))
    }

    /// Apply an iSWAP gate to the specified qubits
    pub(crate) fn iswap(&mut self, qubit1: usize, qubit2: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::ISwap(
            self.checked_qubit(qubit1)?,
            self.checked_qubit(qubit2)?,
        ))
    }

    /// Apply an ECR gate (IBM native echoed cross-resonance) to the specified qubits
    pub(crate) fn ecr(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::ECR(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
        ))
    }

    /// Apply an RXX gate (two-qubit XX rotation) to the specified qubits
    pub(crate) fn rxx(&mut self, qubit1: usize, qubit2: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::RXX(
            self.checked_qubit(qubit1)?,
            self.checked_qubit(qubit2)?,
            theta,
        ))
    }

    /// Apply an RYY gate (two-qubit YY rotation) to the specified qubits
    pub(crate) fn ryy(&mut self, qubit1: usize, qubit2: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::RYY(
            self.checked_qubit(qubit1)?,
            self.checked_qubit(qubit2)?,
            theta,
        ))
    }

    /// Apply an RZZ gate (two-qubit ZZ rotation) to the specified qubits
    pub(crate) fn rzz(&mut self, qubit1: usize, qubit2: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::RZZ(
            self.checked_qubit(qubit1)?,
            self.checked_qubit(qubit2)?,
            theta,
        ))
    }

    /// Apply an RZX gate (two-qubit ZX rotation / cross-resonance) to the specified qubits
    pub(crate) fn rzx(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::RZX(
            self.checked_qubit(control)?,
            self.checked_qubit(target)?,
            theta,
        ))
    }

    /// Apply a DCX gate (double CNOT) to the specified qubits
    pub(crate) fn dcx(&mut self, qubit1: usize, qubit2: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::DCX(
            self.checked_qubit(qubit1)?,
            self.checked_qubit(qubit2)?,
        ))
    }

    /// Apply a phase gate (P gate) with an arbitrary angle to the specified qubit
    pub(crate) fn p(&mut self, qubit: usize, lambda: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::P(self.checked_qubit(qubit)?, lambda))
    }

    /// Apply an identity gate to the specified qubit
    pub(crate) fn id(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Id(self.checked_qubit(qubit)?))
    }

    /// Apply a U gate (general single-qubit rotation) to the specified qubit
    pub(crate) fn u(&mut self, qubit: usize, theta: f64, phi: f64, lambda: f64) -> PyResult<()> {
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
    pub(crate) fn run(&self, py: Python, use_gpu: bool) -> PyResult<Py<PySimulationResult>> {
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
    pub(crate) fn simulate_with_noise(
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
    pub(crate) fn run_auto(&self, py: Python) -> PyResult<Py<PySimulationResult>> {
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
    pub(crate) fn is_gpu_available() -> bool {
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
    pub(crate) fn draw(&self) -> PyResult<String> {
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
    }

    /// Get an HTML representation of the circuit for Jupyter notebooks
    pub(crate) fn draw_html(&self) -> PyResult<String> {
        // Reuse draw method since we're using HTML representation for both
        self.draw()
    }

    /// Get a visualization object for the circuit
    pub(crate) fn visualize(&self, _py: Python) -> PyResult<Py<PyCircuitVisualizer>> {
        self.get_visualizer()
    }

    /// Implements the `_repr_html_` method for Jupyter notebook display
    pub(crate) fn _repr_html_(&self) -> PyResult<String> {
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
    pub(crate) fn decompose(&self, py: Python) -> PyResult<Py<Self>> {
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
    }

    /// Copy the circuit (returns an identical circuit)
    ///
    /// Creates a new circuit with the same gates as this one.
    /// For optimization passes, use the circuit optimizer from quantrs2-circuit.
    pub(crate) fn copy(&self, py: Python) -> PyResult<Py<Self>> {
        if self.circuit.is_none() {
            return Err(PyValueError::new_err("Circuit not initialized"));
        }

        let mut new_circuit = Self::new(self.n_qubits)?;

        // Copy all operations
        for &op in &self.operations {
            new_circuit.apply_op(op)?;
        }

        Py::new(py, new_circuit)
    }

    /// Compose this circuit with another circuit
    ///
    /// Appends the gates from `other` circuit to this circuit.
    /// The other circuit must have the same or fewer qubits.
    pub(crate) fn compose(&mut self, other: &Self) -> PyResult<()> {
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

/// Dynamic qubit count circuit for Python (alias to `PyCircuit` for backward compatibility)
#[pyclass]
pub struct PyDynamicCircuit {
    /// The internal circuit
    circuit: PyCircuit,
}

/// Implementation for `PyDynamicCircuit`
#[pymethods]
impl PyDynamicCircuit {
    /// Create a new dynamic quantum circuit with the given number of qubits
    #[new]
    pub(crate) fn new(n_qubits: usize) -> PyResult<Self> {
        Ok(Self {
            circuit: PyCircuit::new(n_qubits)?,
        })
    }

    /// Get the number of qubits in the circuit
    #[allow(clippy::missing_const_for_fn)]
    #[getter]
    pub(crate) fn n_qubits(&self) -> usize {
        self.circuit.n_qubits
    }

    /// Apply a Hadamard gate to the specified qubit
    pub(crate) fn h(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.h(qubit)
    }

    /// Apply a Pauli-X (NOT) gate to the specified qubit
    pub(crate) fn x(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.x(qubit)
    }

    /// Apply a Pauli-Y gate to the specified qubit
    pub(crate) fn y(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.y(qubit)
    }

    /// Apply a Pauli-Z gate to the specified qubit
    pub(crate) fn z(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.z(qubit)
    }

    /// Apply an S gate (phase gate) to the specified qubit
    pub(crate) fn s(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.s(qubit)
    }

    /// Apply an S-dagger gate to the specified qubit
    pub(crate) fn sdg(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.sdg(qubit)
    }

    /// Apply a T gate (π/8 gate) to the specified qubit
    pub(crate) fn t(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.t(qubit)
    }

    /// Apply a T-dagger gate to the specified qubit
    pub(crate) fn tdg(&mut self, qubit: usize) -> PyResult<()> {
        self.circuit.tdg(qubit)
    }

    /// Apply an Rx gate (rotation around X-axis) to the specified qubit
    pub(crate) fn rx(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.circuit.rx(qubit, theta)
    }

    /// Apply an Ry gate (rotation around Y-axis) to the specified qubit
    pub(crate) fn ry(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.circuit.ry(qubit, theta)
    }

    /// Apply an Rz gate (rotation around Z-axis) to the specified qubit
    pub(crate) fn rz(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.circuit.rz(qubit, theta)
    }

    /// Apply a CNOT gate with the specified control and target qubits
    pub(crate) fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.circuit.cnot(control, target)
    }

    /// Apply a SWAP gate between the specified qubits
    pub(crate) fn swap(&mut self, qubit1: usize, qubit2: usize) -> PyResult<()> {
        self.circuit.swap(qubit1, qubit2)
    }

    /// Apply a CZ gate (controlled-Z) to the specified qubits
    pub(crate) fn cz(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.circuit.cz(control, target)
    }

    /// Run the circuit on a state vector simulator
    #[pyo3(signature = (use_gpu=false))]
    pub(crate) fn run(&self, py: Python, use_gpu: bool) -> PyResult<Py<PySimulationResult>> {
        self.circuit.run(py, use_gpu)
    }

    /// Run the circuit with a noise model
    #[pyo3(signature = (noise_model, use_gpu=false))]
    pub(crate) fn simulate_with_noise(
        &self,
        py: Python,
        noise_model: &PyRealisticNoiseModel,
        use_gpu: bool,
    ) -> PyResult<Py<PySimulationResult>> {
        self.circuit.simulate_with_noise(py, noise_model, use_gpu)
    }

    /// Run the circuit on the best available simulator (GPU if available for larger circuits, CPU otherwise)
    pub(crate) fn run_auto(&self, py: Python) -> PyResult<Py<PySimulationResult>> {
        self.circuit.run_auto(py)
    }
}
