//! Python bindings for advanced quantum simulators.
//!
//! This module provides Python bindings for specialized quantum simulators:
//! - `StabilizerSimulator`: Efficient simulation of Clifford circuits
//! - `TensorNetworkSimulator`: Tensor network-based simulation
//! - `MPSSimulator`: Matrix Product State simulation

// Allow unused_self for PyO3 method bindings that require &self signature
// Allow unnecessary_wraps for PyO3 Result return types that may need error handling in future
// Allow missing_const_for_fn for PyO3 getters that cannot be const due to Python bindings
#![allow(clippy::unused_self)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::missing_const_for_fn)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scirs2_core::Complex64;
use scirs2_numpy::{IntoPyArray, PyArray1};
use std::collections::HashMap;

use quantrs2_sim::mps_simulator::MPSSimulator;
use quantrs2_sim::stabilizer::{StabilizerGate, StabilizerSimulator};
use quantrs2_sim::tensor::{ContractionStrategy, TensorNetworkSimulator};

/// Python wrapper for the Stabilizer Simulator
///
/// Efficiently simulates Clifford circuits using the stabilizer formalism.
/// This simulator can handle circuits with only Clifford gates (H, S, CNOT, X, Y, Z, CZ, SWAP)
/// in polynomial time, enabling simulation of very large circuits.
#[pyclass(name = "StabilizerSimulator")]
pub struct PyStabilizerSimulator {
    simulator: StabilizerSimulator,
}

#[pymethods]
impl PyStabilizerSimulator {
    /// Create a new Stabilizer Simulator
    ///
    /// Args:
    ///     n_qubits (int): Number of qubits in the circuit
    ///
    /// Returns:
    ///     StabilizerSimulator: A new stabilizer simulator instance
    #[new]
    fn new(n_qubits: usize) -> Self {
        Self {
            simulator: StabilizerSimulator::new(n_qubits),
        }
    }

    /// Get the number of qubits
    #[getter]
    fn n_qubits(&self) -> usize {
        self.simulator.num_qubits()
    }

    /// Apply a Hadamard gate
    fn h(&mut self, qubit: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::H(qubit))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply H gate: {e}")))
    }

    /// Apply an S gate (phase gate)
    fn s(&mut self, qubit: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::S(qubit))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply S gate: {e}")))
    }

    /// Apply an S-dagger gate
    fn sdg(&mut self, qubit: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::SDag(qubit))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply S† gate: {e}")))
    }

    /// Apply a Pauli-X gate
    fn x(&mut self, qubit: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::X(qubit))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply X gate: {e}")))
    }

    /// Apply a Pauli-Y gate
    fn y(&mut self, qubit: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::Y(qubit))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply Y gate: {e}")))
    }

    /// Apply a Pauli-Z gate
    fn z(&mut self, qubit: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::Z(qubit))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply Z gate: {e}")))
    }

    /// Apply a CNOT gate
    fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::CNOT(control, target))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply CNOT gate: {e}")))
    }

    /// Apply a CZ gate
    fn cz(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::CZ(control, target))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply CZ gate: {e}")))
    }

    /// Apply a CY gate
    fn cy(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::CY(control, target))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply CY gate: {e}")))
    }

    /// Apply a SWAP gate
    fn swap(&mut self, qubit1: usize, qubit2: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::SWAP(qubit1, qubit2))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply SWAP gate: {e}")))
    }

    /// Apply a sqrt(X) gate
    fn sx(&mut self, qubit: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::SqrtX(qubit))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply √X gate: {e}")))
    }

    /// Apply a sqrt(X)-dagger gate
    fn sxdg(&mut self, qubit: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::SqrtXDag(qubit))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply √X† gate: {e}")))
    }

    /// Apply a sqrt(Y) gate
    fn sy(&mut self, qubit: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::SqrtY(qubit))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply √Y gate: {e}")))
    }

    /// Apply a sqrt(Y)-dagger gate
    fn sydg(&mut self, qubit: usize) -> PyResult<()> {
        self.simulator
            .apply_gate(StabilizerGate::SqrtYDag(qubit))
            .map_err(|e| PyValueError::new_err(format!("Failed to apply √Y† gate: {e}")))
    }

    /// Measure a qubit
    ///
    /// Args:
    ///     qubit (int): The qubit index to measure
    ///
    /// Returns:
    ///     bool: The measurement result (True for |1⟩, False for |0⟩)
    fn measure(&mut self, qubit: usize) -> PyResult<bool> {
        self.simulator
            .measure(qubit)
            .map_err(|e| PyValueError::new_err(format!("Failed to measure qubit: {e}")))
    }

    /// Get the current stabilizer generators
    ///
    /// Returns:
    ///     list[str]: List of stabilizer generator strings in Pauli notation
    fn get_stabilizers(&self) -> Vec<String> {
        self.simulator.get_stabilizers()
    }

    /// Get measurement record
    ///
    /// Returns:
    ///     list[tuple[int, bool]]: List of (qubit, outcome) tuples
    fn get_measurements(&self) -> Vec<(usize, bool)> {
        self.simulator.get_measurements().to_vec()
    }

    /// Reset the simulator to the |0...0⟩ state
    fn reset(&mut self) {
        self.simulator.reset();
    }

    /// Get the state vector (for compatibility with other simulators)
    ///
    /// Note: This is expensive for stabilizer states as it reconstructs the full state vector.
    /// For large systems, prefer using `get_stabilizers()` instead.
    fn get_statevector(&self, py: Python<'_>) -> Py<PyArray1<Complex64>> {
        let state = self.simulator.get_statevector();
        scirs2_core::ndarray::Array1::from_vec(state)
            .into_pyarray(py)
            .into()
    }

    /// Sample measurement outcomes
    ///
    /// Args:
    ///     shots (int): Number of measurement shots
    ///
    /// Returns:
    ///     dict[str, int]: Dictionary mapping bitstrings to counts
    fn sample(&mut self, shots: usize) -> PyResult<HashMap<String, usize>> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let n_qubits = self.simulator.num_qubits();

        for _ in 0..shots {
            // Reset simulator for each shot
            self.simulator.reset();

            // Measure all qubits
            let mut bitstring = String::with_capacity(n_qubits);
            for q in 0..n_qubits {
                let outcome = self
                    .simulator
                    .measure(q)
                    .map_err(|e| PyValueError::new_err(format!("Failed to measure: {e}")))?;
                bitstring.push(if outcome { '1' } else { '0' });
            }

            *counts.entry(bitstring).or_insert(0) += 1;
        }

        Ok(counts)
    }
}

/// Python wrapper for the Tensor Network Simulator
///
/// Uses tensor network contraction for quantum simulation.
/// This approach is efficient for circuits with limited entanglement.
#[pyclass(name = "TensorNetworkSimulator")]
pub struct PyTensorNetworkSimulator {
    simulator: TensorNetworkSimulator,
    n_qubits: usize,
}

#[pymethods]
impl PyTensorNetworkSimulator {
    /// Create a new Tensor Network Simulator
    ///
    /// Args:
    ///     n_qubits (int): Number of qubits
    ///     max_bond_dim (int, optional): Maximum bond dimension. Defaults to 256.
    ///     strategy (str, optional): Contraction strategy ("sequential", "optimal", "greedy").
    ///         Defaults to "greedy".
    ///
    /// Returns:
    ///     TensorNetworkSimulator: A new tensor network simulator instance
    #[new]
    #[pyo3(signature = (n_qubits, max_bond_dim=None, strategy=None))]
    fn new(n_qubits: usize, max_bond_dim: Option<usize>, strategy: Option<&str>) -> PyResult<Self> {
        let strat = match strategy.unwrap_or("greedy") {
            "sequential" => ContractionStrategy::Sequential,
            "optimal" => ContractionStrategy::Optimal,
            "greedy" => ContractionStrategy::Greedy,
            other => {
                return Err(PyValueError::new_err(format!(
                "Unknown contraction strategy: {other}. Use 'sequential', 'optimal', or 'greedy'."
            )))
            }
        };

        let mut sim = TensorNetworkSimulator::new(n_qubits).with_strategy(strat);

        if let Some(bond_dim) = max_bond_dim {
            sim = sim.with_max_bond_dim(bond_dim);
        }

        Ok(Self {
            simulator: sim,
            n_qubits,
        })
    }

    /// Get the number of qubits
    #[getter]
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Initialize the simulator to the |0...0⟩ state
    fn initialize(&mut self) -> PyResult<()> {
        self.simulator
            .initialize_zero_state()
            .map_err(|e| PyValueError::new_err(format!("Failed to initialize state: {e}")))
    }

    /// Get simulation statistics
    fn get_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.simulator.get_stats();
        let dict = PyDict::new(py);
        dict.set_item("contractions", stats.contractions)?;
        dict.set_item("contraction_time_ms", stats.contraction_time_ms)?;
        dict.set_item("max_bond_dimension", stats.max_bond_dimension)?;
        dict.set_item("memory_usage", stats.memory_usage)?;
        dict.set_item("flop_count", stats.flop_count)?;
        Ok(dict)
    }
}

/// Python wrapper for the MPS (Matrix Product State) Simulator
///
/// Uses Matrix Product State representation for efficient simulation
/// of circuits with limited entanglement. MPS is particularly efficient
/// for 1D quantum systems and circuits with local gates.
#[pyclass(name = "MPSSimulator")]
pub struct PyMPSSimulator {
    simulator: MPSSimulator,
    n_qubits: usize,
    max_bond_dim: usize,
}

#[pymethods]
impl PyMPSSimulator {
    /// Create a new MPS Simulator
    ///
    /// Args:
    ///     n_qubits (int): Number of qubits
    ///     max_bond_dim (int, optional): Maximum bond dimension. Defaults to 100.
    ///     truncation_threshold (float, optional): SVD truncation threshold. Defaults to 1e-10.
    ///
    /// Returns:
    ///     MPSSimulator: A new MPS simulator instance
    #[new]
    #[pyo3(signature = (n_qubits, max_bond_dim=None, truncation_threshold=None))]
    fn new(
        n_qubits: usize,
        max_bond_dim: Option<usize>,
        truncation_threshold: Option<f64>,
    ) -> Self {
        let bond_dim = max_bond_dim.unwrap_or(100);
        let mut sim = MPSSimulator::new(bond_dim);
        if let Some(threshold) = truncation_threshold {
            sim.set_truncation_threshold(threshold);
        }

        Self {
            simulator: sim,
            n_qubits,
            max_bond_dim: bond_dim,
        }
    }

    /// Get the number of qubits
    #[getter]
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Get the maximum bond dimension
    #[getter]
    fn max_bond_dim(&self) -> usize {
        self.max_bond_dim
    }
}

/// Register the simulators module
pub fn register_simulators_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent_module.py(), "simulators")?;

    m.add_class::<PyStabilizerSimulator>()?;
    m.add_class::<PyTensorNetworkSimulator>()?;
    m.add_class::<PyMPSSimulator>()?;

    // Add module docstring
    m.setattr(
        "__doc__",
        "Advanced quantum simulators for QuantRS2.\n\n\
         Available simulators:\n\
         - StabilizerSimulator: Efficient Clifford circuit simulation\n\
         - TensorNetworkSimulator: Tensor network-based simulation\n\
         - MPSSimulator: Matrix Product State simulation",
    )?;

    parent_module.add_submodule(&m)?;
    Ok(())
}
