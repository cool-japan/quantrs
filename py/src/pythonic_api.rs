//! Pythonic API matching Qiskit/Cirq conventions.
//!
//! This module provides familiar APIs for users coming from Qiskit or Cirq,
//! making it easier to transition to QuantRS2.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use std::f64::consts::PI;

// Re-use types from main module
use crate::{PyCircuit, PySimulationResult};

/// Qiskit-style QuantumCircuit
#[pyclass(name = "QuantumCircuit")]
pub struct PyQuantumCircuit {
    inner: PyCircuit,
    registers: HashMap<String, Vec<usize>>,
    measurements: HashMap<usize, String>,
}

#[pymethods]
impl PyQuantumCircuit {
    /// Create a new quantum circuit
    ///
    /// Args:
    ///     n_qubits: Number of qubits (or QuantumRegister)
    ///     n_clbits: Number of classical bits (optional)
    ///     name: Circuit name (optional)
    #[new]
    #[pyo3(signature = (n_qubits, n_clbits=None, name=None))]
    fn new(
        n_qubits: &Bound<'_, PyAny>,
        n_clbits: Option<usize>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let num_qubits = if let Ok(n) = n_qubits.extract::<usize>() {
            n
        } else if let Ok(qreg) = n_qubits.extract::<PyRef<PyQuantumRegister>>() {
            qreg.size
        } else {
            return Err(PyValueError::new_err(
                "n_qubits must be int or QuantumRegister",
            ));
        };

        let inner = PyCircuit::new(num_qubits)?;

        Ok(Self {
            inner,
            registers: HashMap::new(),
            measurements: HashMap::new(),
        })
    }

    /// Number of qubits
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    /// Add a register to the circuit
    fn add_register(&mut self, register: &PyQuantumRegister) -> PyResult<()> {
        if self.registers.contains_key(&register.name) {
            return Err(PyValueError::new_err(format!(
                "Register {} already exists",
                register.name
            )));
        }

        let start_idx = self
            .registers
            .values()
            .flatten()
            .max()
            .map(|&x| x + 1)
            .unwrap_or(0);
        let indices: Vec<usize> = (start_idx..start_idx + register.size).collect();
        self.registers.insert(register.name.clone(), indices);

        Ok(())
    }

    // Qiskit-style gate methods

    /// Add Hadamard gate
    fn h(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = self.parse_qubit(qubit)?;
        self.inner.h(q)
    }

    /// Add Pauli-X gate
    fn x(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = self.parse_qubit(qubit)?;
        self.inner.x(q)
    }

    /// Add Pauli-Y gate
    fn y(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = self.parse_qubit(qubit)?;
        self.inner.y(q)
    }

    /// Add Pauli-Z gate
    fn z(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = self.parse_qubit(qubit)?;
        self.inner.z(q)
    }

    /// Add CNOT/CX gate
    fn cx(&mut self, control: &Bound<'_, PyAny>, target: &Bound<'_, PyAny>) -> PyResult<()> {
        let c = self.parse_qubit(control)?;
        let t = self.parse_qubit(target)?;
        self.inner.cnot(c, t)
    }

    /// Add CNOT gate (alias for cx)
    fn cnot(&mut self, control: &Bound<'_, PyAny>, target: &Bound<'_, PyAny>) -> PyResult<()> {
        self.cx(control, target)
    }

    /// Add controlled-Y gate
    fn cy(&mut self, control: &Bound<'_, PyAny>, target: &Bound<'_, PyAny>) -> PyResult<()> {
        let c = self.parse_qubit(control)?;
        let t = self.parse_qubit(target)?;
        self.inner.cy(c, t)
    }

    /// Add controlled-Z gate
    fn cz(&mut self, control: &Bound<'_, PyAny>, target: &Bound<'_, PyAny>) -> PyResult<()> {
        let c = self.parse_qubit(control)?;
        let t = self.parse_qubit(target)?;
        self.inner.cz(c, t)
    }

    /// Add RX rotation gate
    fn rx(&mut self, theta: f64, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = self.parse_qubit(qubit)?;
        self.inner.rx(q, theta)
    }

    /// Add RY rotation gate
    fn ry(&mut self, theta: f64, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = self.parse_qubit(qubit)?;
        self.inner.ry(q, theta)
    }

    /// Add RZ rotation gate
    fn rz(&mut self, phi: f64, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = self.parse_qubit(qubit)?;
        self.inner.rz(q, phi)
    }

    /// Add S gate (√Z)
    fn s(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = self.parse_qubit(qubit)?;
        self.inner.s(q)
    }

    /// Add S† gate
    fn sdg(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = self.parse_qubit(qubit)?;
        self.inner.sdg(q)
    }

    /// Add T gate (T = S^(1/2))
    fn t(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = self.parse_qubit(qubit)?;
        self.inner.t(q)
    }

    /// Add T† gate
    fn tdg(&mut self, qubit: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = self.parse_qubit(qubit)?;
        self.inner.tdg(q)
    }

    /// Add SWAP gate
    fn swap(&mut self, qubit1: &Bound<'_, PyAny>, qubit2: &Bound<'_, PyAny>) -> PyResult<()> {
        let q1 = self.parse_qubit(qubit1)?;
        let q2 = self.parse_qubit(qubit2)?;
        self.inner.swap(q1, q2)
    }

    /// Add Toffoli (CCX) gate
    fn ccx(
        &mut self,
        control1: &Bound<'_, PyAny>,
        control2: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let c1 = self.parse_qubit(control1)?;
        let c2 = self.parse_qubit(control2)?;
        let t = self.parse_qubit(target)?;
        self.inner.toffoli(c1, c2, t)
    }

    /// Add Toffoli gate (alias)
    fn toffoli(
        &mut self,
        control1: &Bound<'_, PyAny>,
        control2: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.ccx(control1, control2, target)
    }

    /// Add measurement
    fn measure(&mut self, qubit: &Bound<'_, PyAny>, cbit: &Bound<'_, PyAny>) -> PyResult<()> {
        let q = self.parse_qubit(qubit)?;
        let c = if let Ok(bit) = cbit.extract::<usize>() {
            format!("c{}", bit)
        } else if let Ok(name) = cbit.extract::<String>() {
            name
        } else {
            return Err(PyValueError::new_err("Classical bit must be int or string"));
        };

        self.measurements.insert(q, c);
        Ok(())
    }

    /// Measure all qubits
    fn measure_all(&mut self) -> PyResult<()> {
        for q in 0..self.inner.n_qubits() {
            self.measurements.insert(q, format!("c{}", q));
        }
        Ok(())
    }

    /// Add a barrier (no-op in simulation)
    fn barrier(&mut self, qubits: Option<Vec<usize>>) -> PyResult<()> {
        // Barriers are used for circuit visualization, no-op in simulation
        Ok(())
    }

    /// Compose with another circuit
    fn compose(&mut self, other: &PyQuantumCircuit, qubits: Option<Vec<usize>>) -> PyResult<()> {
        // Simplified composition - would need full implementation
        if other.num_qubits() > self.num_qubits() {
            return Err(PyValueError::new_err("Other circuit has more qubits"));
        }
        Ok(())
    }

    /// Get circuit depth
    fn depth(&self) -> usize {
        // Simplified - would need to track gate operations
        0
    }

    /// Draw the circuit
    fn draw(&self, output: Option<&str>) -> PyResult<String> {
        match output {
            Some("text") | None => self.inner.draw(),
            Some("mpl") => Ok("Matplotlib drawing not yet implemented".to_string()),
            Some("latex") => Ok("LaTeX drawing not yet implemented".to_string()),
            _ => Err(PyValueError::new_err("Unknown output format")),
        }
    }

    /// Execute the circuit (simplified)
    fn execute(
        &self,
        py: Python,
        backend: Option<&str>,
        shots: Option<usize>,
    ) -> PyResult<Py<PySimulationResult>> {
        // For now, just run the simulation
        self.inner.run(py, false)
    }
}

impl PyQuantumCircuit {
    fn parse_qubit(&self, qubit: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(q) = qubit.extract::<usize>() {
            Ok(q)
        } else if let Ok(qbit) = qubit.extract::<PyRef<PyQubit>>() {
            Ok(qbit.index)
        } else if let Ok(tuple) = qubit.extract::<(String, usize)>() {
            let (reg_name, idx) = tuple;
            self.registers
                .get(&reg_name)
                .and_then(|indices| indices.get(idx))
                .copied()
                .ok_or_else(|| {
                    PyValueError::new_err(format!("Invalid register access: {}[{}]", reg_name, idx))
                })
        } else {
            Err(PyValueError::new_err(
                "Qubit must be int, Qubit, or (register, index)",
            ))
        }
    }
}

/// Quantum register for Qiskit compatibility
#[pyclass(name = "QuantumRegister")]
pub struct PyQuantumRegister {
    size: usize,
    name: String,
}

#[pymethods]
impl PyQuantumRegister {
    #[new]
    #[pyo3(signature = (size, name=None))]
    fn new(size: usize, name: Option<String>) -> Self {
        let name = name.unwrap_or_else(|| format!("q{}", size));
        Self { size, name }
    }

    #[getter]
    fn size(&self) -> usize {
        self.size
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __len__(&self) -> usize {
        self.size
    }

    fn __getitem__(&self, index: usize) -> PyResult<PyQubit> {
        if index >= self.size {
            return Err(PyIndexError::new_err("Qubit index out of range"));
        }
        Ok(PyQubit {
            register: self.name.clone(),
            index,
        })
    }
}

/// Classical register for Qiskit compatibility
#[pyclass(name = "ClassicalRegister")]
pub struct PyClassicalRegister {
    size: usize,
    name: String,
}

#[pymethods]
impl PyClassicalRegister {
    #[new]
    #[pyo3(signature = (size, name=None))]
    fn new(size: usize, name: Option<String>) -> Self {
        let name = name.unwrap_or_else(|| format!("c{}", size));
        Self { size, name }
    }

    #[getter]
    fn size(&self) -> usize {
        self.size
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }
}

/// Individual qubit reference
#[pyclass(name = "Qubit")]
pub struct PyQubit {
    register: String,
    index: usize,
}

#[pymethods]
impl PyQubit {
    #[getter]
    fn register(&self) -> &str {
        &self.register
    }

    #[getter]
    fn index(&self) -> usize {
        self.index
    }
}

/// Cirq-style circuit operations
#[pyclass(name = "Circuit")]
pub struct PyCirqCircuit {
    moments: Vec<Vec<(String, Vec<usize>, Option<Vec<f64>>)>>,
    n_qubits: usize,
}

#[pymethods]
impl PyCirqCircuit {
    #[new]
    fn new() -> Self {
        Self {
            moments: Vec::new(),
            n_qubits: 0,
        }
    }

    /// Append operations to the circuit
    fn append(&mut self, operations: &Bound<'_, PyList>) -> PyResult<()> {
        let mut moment = Vec::new();

        for op in operations {
            if let Ok(gate_op) = op.extract::<PyRef<PyGateOperation>>() {
                moment.push((
                    gate_op.gate_type.clone(),
                    gate_op.qubits.clone(),
                    gate_op.params.clone(),
                ));

                // Update qubit count
                if let Some(&max_q) = gate_op.qubits.iter().max() {
                    self.n_qubits = self.n_qubits.max(max_q + 1);
                }
            }
        }

        self.moments.push(moment);
        Ok(())
    }

    /// Get all qubits used in the circuit
    fn all_qubits(&self) -> Vec<usize> {
        (0..self.n_qubits).collect()
    }

    /// Get circuit as a list of moments
    fn moments(&self) -> Vec<Vec<(String, Vec<usize>, Option<Vec<f64>>)>> {
        self.moments.clone()
    }

    /// Convert to QuantRS2 native circuit
    fn to_quantrs(&self, py: Python) -> PyResult<Py<PyCircuit>> {
        let mut circuit = PyCircuit::new(self.n_qubits)?;

        for moment in &self.moments {
            for (gate_type, qubits, params) in moment {
                match gate_type.as_str() {
                    "H" => circuit.h(qubits[0])?,
                    "X" => circuit.x(qubits[0])?,
                    "Y" => circuit.y(qubits[0])?,
                    "Z" => circuit.z(qubits[0])?,
                    "CNOT" => circuit.cnot(qubits[0], qubits[1])?,
                    "RX" => circuit.rx(qubits[0], params.as_ref().unwrap()[0])?,
                    "RY" => circuit.ry(qubits[0], params.as_ref().unwrap()[0])?,
                    "RZ" => circuit.rz(qubits[0], params.as_ref().unwrap()[0])?,
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown gate: {}",
                            gate_type
                        )))
                    }
                }
            }
        }

        Py::new(py, circuit)
    }
}

/// Gate operation for Cirq-style API
#[pyclass(name = "GateOperation")]
pub struct PyGateOperation {
    gate_type: String,
    qubits: Vec<usize>,
    params: Option<Vec<f64>>,
}

/// Cirq-style gates
#[pyclass(name = "Gates")]
pub struct PyCirqGates;

#[pymethods]
impl PyCirqGates {
    #[staticmethod]
    fn H(qubit: usize) -> PyGateOperation {
        PyGateOperation {
            gate_type: "H".to_string(),
            qubits: vec![qubit],
            params: None,
        }
    }

    #[staticmethod]
    fn X(qubit: usize) -> PyGateOperation {
        PyGateOperation {
            gate_type: "X".to_string(),
            qubits: vec![qubit],
            params: None,
        }
    }

    #[staticmethod]
    fn Y(qubit: usize) -> PyGateOperation {
        PyGateOperation {
            gate_type: "Y".to_string(),
            qubits: vec![qubit],
            params: None,
        }
    }

    #[staticmethod]
    fn Z(qubit: usize) -> PyGateOperation {
        PyGateOperation {
            gate_type: "Z".to_string(),
            qubits: vec![qubit],
            params: None,
        }
    }

    #[staticmethod]
    fn CNOT(control: usize, target: usize) -> PyGateOperation {
        PyGateOperation {
            gate_type: "CNOT".to_string(),
            qubits: vec![control, target],
            params: None,
        }
    }

    #[staticmethod]
    fn rx(rads: f64, qubit: usize) -> PyGateOperation {
        PyGateOperation {
            gate_type: "RX".to_string(),
            qubits: vec![qubit],
            params: Some(vec![rads]),
        }
    }

    #[staticmethod]
    fn ry(rads: f64, qubit: usize) -> PyGateOperation {
        PyGateOperation {
            gate_type: "RY".to_string(),
            qubits: vec![qubit],
            params: Some(vec![rads]),
        }
    }

    #[staticmethod]
    fn rz(rads: f64, qubit: usize) -> PyGateOperation {
        PyGateOperation {
            gate_type: "RZ".to_string(),
            qubits: vec![qubit],
            params: Some(vec![rads]),
        }
    }
}

/// Create a line qubit (Cirq-style)
#[pyfunction]
fn LineQubit(x: usize) -> usize {
    x
}

/// Create grid qubits (Cirq-style)
#[pyfunction]
fn GridQubit(row: usize, col: usize, n_cols: usize) -> usize {
    row * n_cols + col
}

/// Register the Pythonic API module
pub fn register_pythonic_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Qiskit-style API
    m.add_class::<PyQuantumCircuit>()?;
    m.add_class::<PyQuantumRegister>()?;
    m.add_class::<PyClassicalRegister>()?;
    m.add_class::<PyQubit>()?;

    // Cirq-style API
    m.add_class::<PyCirqCircuit>()?;
    m.add_class::<PyGateOperation>()?;
    m.add_class::<PyCirqGates>()?;
    m.add_function(wrap_pyfunction!(LineQubit, m)?)?;
    m.add_function(wrap_pyfunction!(GridQubit, m)?)?;

    // Create submodules for better organization
    let qiskit_compat = PyModule::new(m.py(), "qiskit_compat")?;
    qiskit_compat.add_class::<PyQuantumCircuit>()?;
    qiskit_compat.add_class::<PyQuantumRegister>()?;
    qiskit_compat.add_class::<PyClassicalRegister>()?;
    m.add_submodule(&qiskit_compat)?;

    let cirq_compat = PyModule::new(m.py(), "cirq_compat")?;
    cirq_compat.add_class::<PyCirqCircuit>()?;
    cirq_compat.add_class::<PyGateOperation>()?;
    cirq_compat.add_class::<PyCirqGates>()?;
    m.add_submodule(&cirq_compat)?;

    Ok(())
}
