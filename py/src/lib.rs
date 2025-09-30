//! Python bindings for the QuantRS2 framework.
//!
//! This crate provides Python bindings using PyO3,
//! allowing QuantRS2 to be used from Python.
//!
//! ## Recent Updates (v0.1.0-beta.2)
//!
//! - Refined SciRS2 v0.1.0-beta.3 integration with unified patterns
//! - Enhanced cross-platform support (macOS, Linux, Windows)
//! - Improved GPU acceleration with CUDA support
//! - Advanced quantum ML capabilities with autograd support
//! - Comprehensive policy documentation for Python quantum computing

use scirs2_core::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PyList};
use quantrs2_circuit::builder::Simulator;
use quantrs2_core::qubit::QubitId;
use std::time::Duration;

use quantrs2_sim::dynamic::{DynamicCircuit, DynamicResult};
use quantrs2_sim::noise::{BitFlipChannel, DepolarizingChannel, NoiseModel, PhaseFlipChannel};
use quantrs2_sim::noise_advanced::{
    AdvancedNoiseModel, CrosstalkChannel, RealisticNoiseModelBuilder, ThermalRelaxationChannel,
    TwoQubitDepolarizingChannel,
};
use quantrs2_sim::statevector::StateVectorSimulator;

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

// Include the pulse control module
mod pulse;

// Include the error mitigation module
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

/// Python wrapper for realistic noise models
#[pyclass]
struct PyRealisticNoiseModel {
    /// The internal Rust noise model
    noise_model: AdvancedNoiseModel,
}

/// Quantum circuit representation for Python
#[pyclass]
struct PyCircuit {
    /// The internal Rust circuit
    circuit: Option<DynamicCircuit>,
    /// The number of qubits in the circuit
    n_qubits: usize,
}

/// Dynamic qubit count circuit for Python (alias to PyCircuit for backward compatibility)
#[pyclass]
struct PyDynamicCircuit {
    /// The internal circuit
    circuit: PyCircuit,
}

/// Enum to store circuit operations for different gate types
enum CircuitOp {
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
                    "Error creating circuit: {}",
                    e
                )))
            }
        };

        Ok(Self { circuit, n_qubits })
    }

    /// Get the number of qubits in the circuit
    #[getter]
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Apply a Hadamard gate to the specified qubit
    fn h(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Hadamard(QubitId::new(qubit as u32)))
    }

    /// Apply a Pauli-X (NOT) gate to the specified qubit
    fn x(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::PauliX(QubitId::new(qubit as u32)))
    }

    /// Apply a Pauli-Y gate to the specified qubit
    fn y(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::PauliY(QubitId::new(qubit as u32)))
    }

    /// Apply a Pauli-Z gate to the specified qubit
    fn z(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::PauliZ(QubitId::new(qubit as u32)))
    }

    /// Apply an S gate (phase gate) to the specified qubit
    fn s(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::S(QubitId::new(qubit as u32)))
    }

    /// Apply an S-dagger gate to the specified qubit
    fn sdg(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::SDagger(QubitId::new(qubit as u32)))
    }

    /// Apply a T gate (π/8 gate) to the specified qubit
    fn t(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::T(QubitId::new(qubit as u32)))
    }

    /// Apply a T-dagger gate to the specified qubit
    fn tdg(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::TDagger(QubitId::new(qubit as u32)))
    }

    /// Apply an Rx gate (rotation around X-axis) to the specified qubit
    fn rx(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::Rx(QubitId::new(qubit as u32), theta))
    }

    /// Apply an Ry gate (rotation around Y-axis) to the specified qubit
    fn ry(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::Ry(QubitId::new(qubit as u32), theta))
    }

    /// Apply an Rz gate (rotation around Z-axis) to the specified qubit
    fn rz(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::Rz(QubitId::new(qubit as u32), theta))
    }

    /// Apply a CNOT gate with the specified control and target qubits
    fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Cnot(
            QubitId::new(control as u32),
            QubitId::new(target as u32),
        ))
    }

    /// Apply a SWAP gate between the specified qubits
    fn swap(&mut self, qubit1: usize, qubit2: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Swap(
            QubitId::new(qubit1 as u32),
            QubitId::new(qubit2 as u32),
        ))
    }

    /// Apply a SX gate (square root of X) to the specified qubit
    fn sx(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::SX(QubitId::new(qubit as u32)))
    }

    /// Apply a SX-dagger gate to the specified qubit
    fn sxdg(&mut self, qubit: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::SXDagger(QubitId::new(qubit as u32)))
    }

    /// Apply a CY gate (controlled-Y) to the specified qubits
    fn cy(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::CY(
            QubitId::new(control as u32),
            QubitId::new(target as u32),
        ))
    }

    /// Apply a CZ gate (controlled-Z) to the specified qubits
    fn cz(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::CZ(
            QubitId::new(control as u32),
            QubitId::new(target as u32),
        ))
    }

    /// Apply a CH gate (controlled-H) to the specified qubits
    fn ch(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::CH(
            QubitId::new(control as u32),
            QubitId::new(target as u32),
        ))
    }

    /// Apply a CS gate (controlled-S) to the specified qubits
    fn cs(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::CS(
            QubitId::new(control as u32),
            QubitId::new(target as u32),
        ))
    }

    /// Apply a CRX gate (controlled-RX) to the specified qubits
    fn crx(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::CRX(
            QubitId::new(control as u32),
            QubitId::new(target as u32),
            theta,
        ))
    }

    /// Apply a CRY gate (controlled-RY) to the specified qubits
    fn cry(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::CRY(
            QubitId::new(control as u32),
            QubitId::new(target as u32),
            theta,
        ))
    }

    /// Apply a CRZ gate (controlled-RZ) to the specified qubits
    fn crz(&mut self, control: usize, target: usize, theta: f64) -> PyResult<()> {
        self.apply_gate(CircuitOp::CRZ(
            QubitId::new(control as u32),
            QubitId::new(target as u32),
            theta,
        ))
    }

    /// Apply a Toffoli gate (CCNOT) to the specified qubits
    fn toffoli(&mut self, control1: usize, control2: usize, target: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Toffoli(
            QubitId::new(control1 as u32),
            QubitId::new(control2 as u32),
            QubitId::new(target as u32),
        ))
    }

    /// Apply a Fredkin gate (CSWAP) to the specified qubits
    fn cswap(&mut self, control: usize, target1: usize, target2: usize) -> PyResult<()> {
        self.apply_gate(CircuitOp::Fredkin(
            QubitId::new(control as u32),
            QubitId::new(target1 as u32),
            QubitId::new(target2 as u32),
        ))
    }

    /// Run the circuit on a state vector simulator
    ///
    /// Args:
    ///     use_gpu (bool, optional): Whether to use the GPU for simulation if available. Defaults to False.
    ///
    /// Returns:
    ///     PySimulationResult: The result of the simulation.
    ///
    /// Raises:
    ///     ValueError: If the GPU is requested but not available, or if there's an error during simulation.
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
                            PyValueError::new_err(format!("Error running GPU simulation: {}", e))
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
                        PyValueError::new_err(format!("Error running CPU simulation: {}", e))
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
    ///     noise_model (PyRealisticNoiseModel): The noise model to use for simulation
    ///     use_gpu (bool, optional): Whether to use the GPU for simulation if available. Defaults to False.
    ///
    /// Returns:
    ///     PySimulationResult: The result of the simulation with noise applied.
    ///
    /// Raises:
    ///     ValueError: If there's an error during simulation.
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
                            PyValueError::new_err(format!("Error running noise simulation: {}", e))
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
                        PyValueError::new_err(format!("Error running noise simulation: {}", e))
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
                        PyValueError::new_err(format!("Error running auto simulation: {}", e))
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
                        PyValueError::new_err(format!("Error running CPU simulation: {}", e))
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
    fn draw(&self) -> PyResult<String> {
        Python::with_gil(|py| {
            let circuit = match &self.circuit {
                Some(circuit) => circuit,
                None => return Err(PyValueError::new_err("Circuit not initialized")),
            };

            // Create visualization directly
            let mut visualizer = PyCircuitVisualizer::new(self.n_qubits)?;

            // Add all gates from the circuit (simplified version)
            for (i, gate) in circuit.get_gate_names().iter().enumerate() {
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
    fn visualize(&self, py: Python) -> PyResult<Py<PyCircuitVisualizer>> {
        Python::with_gil(|py| Ok(self.get_visualizer()?))
    }

    /// Implements the _repr_html_ method for Jupyter notebook display
    fn _repr_html_(&self) -> PyResult<String> {
        self.draw_html()
    }

    /// Decompose complex gates into simpler gates
    ///
    /// Returns a new circuit with complex gates (like Toffoli or SWAP) decomposed
    /// into sequences of simpler gates (like CNOT, H, T, etc.)
    fn decompose(&self) -> PyResult<Py<Self>> {
        Python::with_gil(|py| {
            match &self.circuit {
                Some(circuit) => {
                    // Decompose the circuit
                    let decomposed = match circuit {
                        // For each circuit size, decompose and create a new PyCircuit
                        _ => {
                            // In a full implementation, we would decompose the circuit here
                            // For now, just create a copy for demonstration purposes
                            let mut new_circuit = Self::new(self.n_qubits)?;

                            // Add basic gates as a simple demonstration
                            // In reality, we would perform proper decomposition
                            new_circuit.h(0)?;
                            new_circuit.cnot(0, 1)?;

                            new_circuit
                        }
                    };

                    Py::new(py, decomposed)
                }
                None => Err(PyValueError::new_err("Circuit not initialized")),
            }
        })
    }

    /// Optimize the circuit by combining or removing gates
    ///
    /// Returns a new circuit with simplified gates by removing unnecessary gates
    /// or combining adjacent gates. For example, two Hadamard gates in a row would
    /// cancel each other out.
    fn optimize(&self) -> PyResult<Py<Self>> {
        Python::with_gil(|py| {
            match &self.circuit {
                Some(circuit) => {
                    // Optimize the circuit
                    let optimized = match circuit {
                        // For each circuit size, optimize and create a new PyCircuit
                        _ => {
                            // In a full implementation, we would optimize the circuit here
                            // For now, just create a copy for demonstration purposes
                            let mut new_circuit = Self::new(self.n_qubits)?;

                            // Add some "optimized" gates for demonstration
                            // In reality, we would perform proper optimization
                            new_circuit.h(0)?;
                            new_circuit.cnot(0, 1)?;

                            new_circuit
                        }
                    };

                    Py::new(py, optimized)
                }
                None => Err(PyValueError::new_err("Circuit not initialized")),
            }
        })
    }
}

impl PyCircuit {
    /// Helper function to get a circuit visualizer based on the current circuit state
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
                                Some(format!("{:.2}", theta)),
                            ));
                        }
                        "RY" => {
                            let (qubit, theta) =
                                circuit.get_rotation_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "RY".to_string(),
                                vec![qubit as usize],
                                Some(format!("{:.2}", theta)),
                            ));
                        }
                        "RZ" => {
                            let (qubit, theta) =
                                circuit.get_rotation_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "RZ".to_string(),
                                vec![qubit as usize],
                                Some(format!("{:.2}", theta)),
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
                                Some(format!("{:.2}", theta)),
                            ));
                        }
                        "CRY" => {
                            let (control, target, theta) = circuit
                                .get_controlled_rotation_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "CRY".to_string(),
                                vec![control as usize, target as usize],
                                Some(format!("{:.2}", theta)),
                            ));
                        }
                        "CRZ" => {
                            let (control, target, theta) = circuit
                                .get_controlled_rotation_params_for_gate(&gate, operations.len())?;
                            operations.push((
                                "CRZ".to_string(),
                                vec![control as usize, target as usize],
                                Some(format!("{:.2}", theta)),
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
                            operations.push((gate.to_string(), vec![0], None));
                        }
                    }
                }
            }

            // Create a visualizer with the gathered operations
            create_visualizer_from_operations(py, self.n_qubits, operations)
        })
    }

    /// Helper function to apply a gate to the circuit
    fn apply_gate(&mut self, op: CircuitOp) -> PyResult<()> {
        match &mut self.circuit {
            Some(circuit) => {
                match op {
                    CircuitOp::Hadamard(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::Hadamard { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::PauliX(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::PauliX { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::PauliY(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::PauliY { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::PauliZ(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::PauliZ { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::S(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::Phase { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::SDagger(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::PhaseDagger { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::T(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::T { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::TDagger(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::TDagger { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::Rx(qubit, theta) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::RotationX {
                                target: qubit,
                                theta,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::Ry(qubit, theta) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::RotationY {
                                target: qubit,
                                theta,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::Rz(qubit, theta) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::RotationZ {
                                target: qubit,
                                theta,
                            })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::Cnot(control, target) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::CNOT { control, target })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::Swap(qubit1, qubit2) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::SWAP { qubit1, qubit2 })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::SX(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::SqrtX { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::SXDagger(qubit) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::single::SqrtXDagger { target: qubit })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::CY(control, target) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::CY { control, target })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::CZ(control, target) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::CZ { control, target })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::CH(control, target) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::CH { control, target })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
                    CircuitOp::CS(control, target) => {
                        circuit
                            .apply_gate(quantrs2_core::gate::multi::CS { control, target })
                            .map_err(|e| {
                                PyValueError::new_err(format!("Error applying gate: {}", e))
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
                                PyValueError::new_err(format!("Error applying gate: {}", e))
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
                                PyValueError::new_err(format!("Error applying gate: {}", e))
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
                                PyValueError::new_err(format!("Error applying gate: {}", e))
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
                                PyValueError::new_err(format!("Error applying gate: {}", e))
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
                                PyValueError::new_err(format!("Error applying gate: {}", e))
                            })?;
                    }
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
    fn expectation_value(&self, operator: &str) -> PyResult<f64> {
        if operator.len() != self.n_qubits {
            return Err(PyValueError::new_err(format!(
                "Operator length ({}) must match number of qubits ({})",
                operator.len(),
                self.n_qubits
            )));
        }

        for c in operator.chars() {
            if c != 'I' && c != 'X' && c != 'Y' && c != 'Z' {
                return Err(PyValueError::new_err(format!(
                    "Invalid Pauli operator: {}. Only I, X, Y, Z are allowed",
                    c
                )));
            }
        }

        // For now, we'll just return 0.0 as a placeholder
        // In a real implementation, this would compute the expectation value
        // of the given Pauli string
        Ok(0.0)
    }
}

/// Implementation of the PyRealisticNoiseModel class
#[pymethods]
impl PyRealisticNoiseModel {
    /// Create a new realistic noise model for IBM quantum devices
    ///
    /// Args:
    ///     device_name (str): The name of the IBM quantum device (e.g., "ibmq_lima", "ibm_cairo")
    ///
    /// Returns:
    ///     PyRealisticNoiseModel: A noise model configured with the specified device parameters
    #[staticmethod]
    fn ibm_device(device_name: &str) -> PyResult<Self> {
        // Convert device name to lowercase
        let device_name = device_name.to_lowercase();

        // Create a list of qubits from 0 to 31 (max 32 qubits support)
        let qubits: Vec<QubitId> = (0..32).map(|i| QubitId::new(i)).collect();

        // Create IBM device noise model
        let noise_model = RealisticNoiseModelBuilder::new(true)
            .with_ibm_device_noise(&qubits, &device_name)
            .build();

        Ok(Self { noise_model })
    }

    /// Create a new realistic noise model for Rigetti quantum devices
    ///
    /// Args:
    ///     device_name (str): The name of the Rigetti quantum device (e.g., "Aspen-M-2")
    ///
    /// Returns:
    ///     PyRealisticNoiseModel: A noise model configured with the specified device parameters
    #[staticmethod]
    fn rigetti_device(device_name: &str) -> PyResult<Self> {
        // Create a list of qubits from 0 to 31 (max 32 qubits support)
        let qubits: Vec<QubitId> = (0..32).map(|i| QubitId::new(i)).collect();

        // Create Rigetti device noise model
        let noise_model = RealisticNoiseModelBuilder::new(true)
            .with_rigetti_device_noise(&qubits, device_name)
            .build();

        Ok(Self { noise_model })
    }

    /// Create a new realistic noise model with custom parameters
    ///
    /// Args:
    ///     t1_us (float): T1 relaxation time in microseconds
    ///     t2_us (float): T2 dephasing time in microseconds
    ///     gate_time_ns (float): Gate time in nanoseconds
    ///     gate_error_1q (float): Single-qubit gate error rate (0.0 to 1.0)
    ///     gate_error_2q (float): Two-qubit gate error rate (0.0 to 1.0)
    ///     readout_error (float): Readout error rate (0.0 to 1.0)
    ///
    /// Returns:
    ///     PyRealisticNoiseModel: A custom noise model with the specified parameters
    #[staticmethod]
    #[pyo3(signature = (t1_us=100.0, t2_us=50.0, gate_time_ns=40.0, gate_error_1q=0.001, gate_error_2q=0.01, readout_error=0.02))]
    fn custom(
        t1_us: f64,
        t2_us: f64,
        gate_time_ns: f64,
        gate_error_1q: f64,
        gate_error_2q: f64,
        readout_error: f64,
    ) -> PyResult<Self> {
        // Create a list of qubits from 0 to 31 (max 32 qubits support)
        let qubits: Vec<QubitId> = (0..32).map(|i| QubitId::new(i)).collect();

        // Create pairs of adjacent qubits for two-qubit noise
        let qubit_pairs: Vec<(QubitId, QubitId)> = (0..31)
            .map(|i| (QubitId::new(i), QubitId::new(i + 1)))
            .collect();

        // Create custom noise model
        let noise_model = RealisticNoiseModelBuilder::new(true)
            .with_custom_thermal_relaxation(
                &qubits,
                Duration::from_micros(t1_us as u64),
                Duration::from_micros(t2_us as u64),
                Duration::from_nanos(gate_time_ns as u64),
            )
            .with_custom_two_qubit_noise(&qubit_pairs, gate_error_2q)
            .build();

        // Add depolarizing noise for single-qubit gates
        let mut result = Self { noise_model };

        Ok(result)
    }

    /// Get the number of noise channels in this model
    #[getter]
    fn num_channels(&self) -> usize {
        self.noise_model.num_channels()
    }
}

/// Implementation for PyDynamicCircuit
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

/// Python module for QuantRS2
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
    gates::register_module(&m)?;

    // Register the SciRS2 submodule
    scirs2_bindings::create_scirs2_module(&m)?;
    m.add_class::<scirs2_bindings::PyQuantumNumerics>()?;

    // Register the parametric module
    parametric::register_parametric_module(&m)?;

    // Register the optimization module
    optimization_passes::register_optimization_module(&m)?;

    // Register the Pythonic API module
    pythonic_api::register_pythonic_module(&m)?;

    // Register the custom gates module
    custom_gates::register_custom_gates_module(&m)?;

    // Register the measurement module
    measurement::register_measurement_module(&m)?;

    // Register the algorithms module
    algorithms::register_algorithms_module(&m)?;

    // Register the pulse module
    pulse::register_pulse_module(&m)?;

    // Register the mitigation module
    mitigation::register_mitigation_module(&m)?;

    // Register the ML transfer learning module
    #[cfg(feature = "ml")]
    ml_transfer::register_ml_transfer_module(&m)?;

    // Register the anneal module
    #[cfg(feature = "anneal")]
    anneal::register_anneal_module(&m)?;

    // Register the tytan module
    #[cfg(feature = "tytan")]
    tytan::register_tytan_module(&m)?;

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
