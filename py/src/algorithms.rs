//! Quantum algorithm templates (VQE, QAOA, QFT, etc.)
//!
//! This module provides ready-to-use implementations of common quantum algorithms:
//! - Variational Quantum Eigensolver (VQE)
//! - Quantum Approximate Optimization Algorithm (QAOA)
//! - Quantum Fourier Transform (QFT)
//! - Grover's Search Algorithm
//! - Quantum Phase Estimation (QPE)

use crate::{PyCircuit, PySimulationResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

/// Variational Quantum Eigensolver (VQE)
#[pyclass(name = "VQE")]
pub struct PyVQE {
    n_qubits: usize,
    ansatz_type: String,
    optimizer: String,
    max_iterations: usize,
}

#[pymethods]
impl PyVQE {
    #[new]
    #[pyo3(signature = (n_qubits, ansatz_type="hardware_efficient", optimizer="COBYLA", max_iterations=100))]
    fn new(
        n_qubits: usize,
        ansatz_type: Option<&str>,
        optimizer: Option<&str>,
        max_iterations: Option<usize>,
    ) -> Self {
        Self {
            n_qubits,
            ansatz_type: ansatz_type.unwrap_or("hardware_efficient").to_string(),
            optimizer: optimizer.unwrap_or("COBYLA").to_string(),
            max_iterations: max_iterations.unwrap_or(100),
        }
    }

    /// Create the ansatz circuit (simplified - returns regular circuit)
    fn create_ansatz(&self, py: Python, num_layers: usize) -> PyResult<Py<PyCircuit>> {
        let mut circuit = PyCircuit::new(self.n_qubits)?;

        match self.ansatz_type.as_str() {
            "hardware_efficient" => {
                // Hardware-efficient ansatz
                for layer in 0..num_layers {
                    // Single-qubit rotations
                    for q in 0..self.n_qubits {
                        circuit.ry(q, layer as f64 * 0.1)?;
                        circuit.rz(q, layer as f64 * 0.2)?;
                    }

                    // Entangling gates
                    for q in 0..self.n_qubits - 1 {
                        circuit.cnot(q, q + 1)?;
                    }
                }

                // Final layer of rotations
                for q in 0..self.n_qubits {
                    circuit.ry(q, 0.3)?;
                }
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown ansatz type: {}",
                    self.ansatz_type
                )));
            }
        }

        Py::new(py, circuit)
    }
}

/// Quantum Approximate Optimization Algorithm (QAOA)
#[pyclass(name = "QAOA")]
pub struct PyQAOA {
    n_qubits: usize,
    p: usize, // Number of QAOA layers
}

#[pymethods]
impl PyQAOA {
    #[new]
    fn new(n_qubits: usize, p: usize) -> Self {
        Self { n_qubits, p }
    }

    /// Create QAOA circuit for MaxCut problem
    fn maxcut_circuit(
        &self,
        py: Python,
        edges: Vec<(usize, usize)>,
        params: Vec<f64>,
    ) -> PyResult<Py<PyCircuit>> {
        if params.len() != 2 * self.p {
            return Err(PyValueError::new_err(format!(
                "Expected {} parameters, got {}",
                2 * self.p,
                params.len()
            )));
        }

        let mut circuit = PyCircuit::new(self.n_qubits)?;

        // Initial state: uniform superposition
        for q in 0..self.n_qubits {
            circuit.h(q)?;
        }

        // QAOA layers
        for layer in 0..self.p {
            let gamma = params[2 * layer];
            let beta = params[2 * layer + 1];

            // Cost Hamiltonian layer (ZZ interactions)
            for &(i, j) in &edges {
                circuit.cnot(i, j)?;
                circuit.rz(j, gamma)?;
                circuit.cnot(i, j)?;
            }

            // Mixer Hamiltonian layer (X rotations)
            for q in 0..self.n_qubits {
                circuit.rx(q, beta)?;
            }
        }

        Py::new(py, circuit)
    }
}

/// Quantum Fourier Transform (QFT)
#[pyclass(name = "QFT")]
pub struct PyQFT {}

#[pymethods]
impl PyQFT {
    #[new]
    fn new() -> Self {
        Self {}
    }

    /// Create QFT circuit
    #[staticmethod]
    fn circuit(py: Python, n_qubits: usize, inverse: Option<bool>) -> PyResult<Py<PyCircuit>> {
        let inverse = inverse.unwrap_or(false);
        let mut circuit = PyCircuit::new(n_qubits)?;

        if inverse {
            // Inverse QFT
            for j in (0..n_qubits).rev() {
                // Controlled phase rotations
                for k in (j + 1..n_qubits).rev() {
                    let angle = -std::f64::consts::PI / (1 << (k - j)) as f64;
                    circuit.crz(k, j, angle)?;
                }
                // Hadamard
                circuit.h(j)?;
            }

            // Swap qubits
            for i in 0..n_qubits / 2 {
                circuit.swap(i, n_qubits - i - 1)?;
            }
        } else {
            // Forward QFT
            for j in 0..n_qubits {
                // Hadamard
                circuit.h(j)?;

                // Controlled phase rotations
                for k in j + 1..n_qubits {
                    let angle = std::f64::consts::PI / (1 << (k - j)) as f64;
                    circuit.crz(k, j, angle)?;
                }
            }

            // Swap qubits
            for i in 0..n_qubits / 2 {
                circuit.swap(i, n_qubits - i - 1)?;
            }
        }

        Py::new(py, circuit)
    }

    /// Apply QFT to specific qubits in a circuit
    #[staticmethod]
    fn apply_to_circuit(
        circuit: &mut PyCircuit,
        qubits: Vec<usize>,
        inverse: Option<bool>,
    ) -> PyResult<()> {
        let inverse = inverse.unwrap_or(false);
        let n = qubits.len();

        if inverse {
            // Inverse QFT on specified qubits
            for j in (0..n).rev() {
                for k in (j + 1..n).rev() {
                    let angle = -std::f64::consts::PI / (1 << (k - j)) as f64;
                    circuit.crz(qubits[k], qubits[j], angle)?;
                }
                circuit.h(qubits[j])?;
            }

            for i in 0..n / 2 {
                circuit.swap(qubits[i], qubits[n - i - 1])?;
            }
        } else {
            // Forward QFT on specified qubits
            for j in 0..n {
                circuit.h(qubits[j])?;

                for k in j + 1..n {
                    let angle = std::f64::consts::PI / (1 << (k - j)) as f64;
                    circuit.crz(qubits[k], qubits[j], angle)?;
                }
            }

            for i in 0..n / 2 {
                circuit.swap(qubits[i], qubits[n - i - 1])?;
            }
        }

        Ok(())
    }
}

/// Grover's Search Algorithm
#[pyclass(name = "Grover")]
pub struct PyGrover {
    n_qubits: usize,
    n_iterations: usize,
}

#[pymethods]
impl PyGrover {
    #[new]
    fn new(n_qubits: usize) -> Self {
        // Calculate optimal number of iterations
        let n_items = 1 << n_qubits;
        let n_iterations = ((std::f64::consts::PI / 4.0) * (n_items as f64).sqrt()) as usize;

        Self {
            n_qubits,
            n_iterations,
        }
    }

    /// Create Grover circuit with marked items
    fn create_circuit(
        &self,
        py: Python,
        marked_items: Vec<usize>,
        iterations: Option<usize>,
    ) -> PyResult<Py<PyCircuit>> {
        let iterations = iterations.unwrap_or(self.n_iterations);
        let mut circuit = PyCircuit::new(self.n_qubits)?;

        // Initialize in uniform superposition
        for q in 0..self.n_qubits {
            circuit.h(q)?;
        }

        // Grover iterations
        for _ in 0..iterations {
            // Oracle for marked items
            for &item in &marked_items {
                // Apply X gates to flip qubits for the marked item
                for q in 0..self.n_qubits {
                    if (item & (1 << q)) == 0 {
                        circuit.x(q)?;
                    }
                }

                // Multi-controlled Z (simplified for 2-3 qubits)
                if self.n_qubits == 2 {
                    circuit.cz(0, 1)?;
                } else if self.n_qubits == 3 {
                    circuit.h(2)?;
                    circuit.toffoli(0, 1, 2)?;
                    circuit.h(2)?;
                }

                // Undo X gates
                for q in 0..self.n_qubits {
                    if (item & (1 << q)) == 0 {
                        circuit.x(q)?;
                    }
                }
            }

            // Diffusion operator
            self.apply_diffusion(&mut circuit)?;
        }

        Py::new(py, circuit)
    }

    /// Apply the diffusion operator
    fn apply_diffusion(&self, circuit: &mut PyCircuit) -> PyResult<()> {
        // Apply Hadamard to all qubits
        for q in 0..self.n_qubits {
            circuit.h(q)?;
        }

        // Apply X to all qubits
        for q in 0..self.n_qubits {
            circuit.x(q)?;
        }

        // Multi-controlled Z gate (simplified for 2-3 qubits)
        if self.n_qubits == 2 {
            circuit.cz(0, 1)?;
        } else if self.n_qubits == 3 {
            // CCZ gate
            circuit.h(2)?;
            circuit.toffoli(0, 1, 2)?;
            circuit.h(2)?;
        } else {
            // For larger systems, would need to decompose multi-controlled Z
            return Err(PyRuntimeError::new_err(
                "Diffusion for >3 qubits not implemented",
            ));
        }

        // Apply X to all qubits
        for q in 0..self.n_qubits {
            circuit.x(q)?;
        }

        // Apply Hadamard to all qubits
        for q in 0..self.n_qubits {
            circuit.h(q)?;
        }

        Ok(())
    }
}

/// Quantum Phase Estimation (QPE)
#[pyclass(name = "QPE")]
pub struct PyQPE {}

#[pymethods]
impl PyQPE {
    #[new]
    fn new() -> Self {
        Self {}
    }

    /// Create QPE circuit for phase estimation
    #[staticmethod]
    fn circuit(
        py: Python,
        n_counting_qubits: usize,
        n_state_qubits: usize,
        unitary_gate: &str,
        phase: f64,
    ) -> PyResult<Py<PyCircuit>> {
        let n_total = n_counting_qubits + n_state_qubits;
        let mut circuit = PyCircuit::new(n_total)?;

        // Initialize counting qubits in superposition
        for q in 0..n_counting_qubits {
            circuit.h(q)?;
        }

        // Initialize eigenstate (for demonstration, use |1⟩)
        if n_state_qubits > 0 {
            circuit.x(n_counting_qubits)?;
        }

        // Controlled unitary operations
        for c in 0..n_counting_qubits {
            let power = 1 << (n_counting_qubits - c - 1);

            // Apply U^(2^k) controlled by qubit c
            for _ in 0..power {
                match unitary_gate {
                    "Z" => {
                        if n_state_qubits > 0 {
                            circuit.cz(c, n_counting_qubits)?;
                        }
                    }
                    "RZ" => {
                        if n_state_qubits > 0 {
                            circuit.crz(c, n_counting_qubits, phase)?;
                        }
                    }
                    _ => {
                        return Err(PyValueError::new_err("Unsupported unitary gate"));
                    }
                }
            }
        }

        // Apply inverse QFT to counting qubits
        PyQFT::apply_to_circuit(&mut circuit, (0..n_counting_qubits).collect(), Some(true))?;

        Py::new(py, circuit)
    }
}

/// Simple helper for creating Hamiltonians
#[pyfunction]
fn create_ising_hamiltonian(
    py: Python,
    n_qubits: usize,
    j_coupling: f64,
    h_field: f64,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let dim = 1 << n_qubits;
    let mut h_matrix = Array2::zeros((dim, dim));

    // Add ZZ coupling terms
    for i in 0..n_qubits - 1 {
        for state in 0..dim {
            let bit_i = (state >> i) & 1;
            let bit_j = (state >> (i + 1)) & 1;
            let sign = if bit_i == bit_j { 1.0 } else { -1.0 };
            h_matrix[[state, state]] += Complex64::new(j_coupling * sign, 0.0);
        }
    }

    // Add Z field terms
    for i in 0..n_qubits {
        for state in 0..dim {
            let bit = (state >> i) & 1;
            let sign = if bit == 0 { 1.0 } else { -1.0 };
            h_matrix[[state, state]] += Complex64::new(h_field * sign, 0.0);
        }
    }

    Ok(h_matrix.into_pyarray(py).into())
}

/// Register the algorithms module
pub fn register_algorithms_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(m.py(), "algorithms")?;

    submodule.add_class::<PyVQE>()?;
    submodule.add_class::<PyQAOA>()?;
    submodule.add_class::<PyQFT>()?;
    submodule.add_class::<PyGrover>()?;
    submodule.add_class::<PyQPE>()?;

    submodule.add_function(wrap_pyfunction!(create_ising_hamiltonian, &submodule)?)?;

    m.add_submodule(&submodule)?;
    Ok(())
}
