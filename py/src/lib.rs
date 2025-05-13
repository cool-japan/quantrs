//! Python bindings for the QuantRS2 framework.
//!
//! This crate provides Python bindings using PyO3,
//! allowing QuantRS2 to be used from Python.

use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PyList};
use quantrs2_circuit::builder::Simulator;
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::statevector::StateVectorSimulator;

/// Quantum circuit representation for Python
#[pyclass]
struct PyCircuit {
    /// The internal Rust circuit
    circuit: Option<CircuitEnum>,
}

/// Enum to store circuits with different qubit counts
enum CircuitEnum {
    /// 1-qubit circuit
    Q1(Circuit<1>),
    /// 2-qubit circuit
    Q2(Circuit<2>),
    /// 3-qubit circuit
    Q3(Circuit<3>),
    /// 4-qubit circuit
    Q4(Circuit<4>),
    /// 5-qubit circuit
    Q5(Circuit<5>),
    /// 8-qubit circuit
    Q8(Circuit<8>),
    /// 10-qubit circuit
    Q10(Circuit<10>),
    /// 16-qubit circuit
    Q16(Circuit<16>),
}

/// Python wrapper for simulation results
#[pyclass]
struct PySimulationResult {
    /// The state vector amplitudes
    amplitudes: Vec<Complex64>,
    /// The number of qubits
    n_qubits: usize,
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
}

/// Apply a circuit operation to a circuit
fn apply_op<const N: usize>(circuit: &mut Circuit<N>, op: CircuitOp) -> Result<(), String> {
    match op {
        CircuitOp::Hadamard(qubit) => {
            circuit.h(qubit).map_err(|e| e.to_string())?;
        }
        CircuitOp::PauliX(qubit) => {
            circuit.x(qubit).map_err(|e| e.to_string())?;
        }
        CircuitOp::PauliY(qubit) => {
            circuit.y(qubit).map_err(|e| e.to_string())?;
        }
        CircuitOp::PauliZ(qubit) => {
            circuit.z(qubit).map_err(|e| e.to_string())?;
        }
        CircuitOp::S(qubit) => {
            // Use the Phase gate for S
            circuit
                .add_gate(quantrs2_core::gate::single::Phase { target: qubit })
                .map_err(|e| e.to_string())?;
        }
        CircuitOp::SDagger(qubit) => {
            // Use the PhaseDagger gate for S-dagger
            circuit
                .add_gate(quantrs2_core::gate::single::PhaseDagger { target: qubit })
                .map_err(|e| e.to_string())?;
        }
        CircuitOp::T(qubit) => {
            // Use the T gate
            circuit
                .add_gate(quantrs2_core::gate::single::T { target: qubit })
                .map_err(|e| e.to_string())?;
        }
        CircuitOp::TDagger(qubit) => {
            // Use the TDagger gate
            circuit
                .add_gate(quantrs2_core::gate::single::TDagger { target: qubit })
                .map_err(|e| e.to_string())?;
        }
        CircuitOp::Rx(qubit, theta) => {
            circuit.rx(qubit, theta).map_err(|e| e.to_string())?;
        }
        CircuitOp::Ry(qubit, theta) => {
            circuit.ry(qubit, theta).map_err(|e| e.to_string())?;
        }
        CircuitOp::Rz(qubit, theta) => {
            circuit.rz(qubit, theta).map_err(|e| e.to_string())?;
        }
        CircuitOp::Cnot(control, target) => {
            circuit.cnot(control, target).map_err(|e| e.to_string())?;
        }
        CircuitOp::Swap(qubit1, qubit2) => {
            circuit.swap(qubit1, qubit2).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[pymethods]
impl PyCircuit {
    /// Create a new quantum circuit with the given number of qubits
    #[new]
    fn new(n_qubits: usize) -> PyResult<Self> {
        let circuit = match n_qubits {
            1 => Some(CircuitEnum::Q1(Circuit::<1>::new())),
            2 => Some(CircuitEnum::Q2(Circuit::<2>::new())),
            3 => Some(CircuitEnum::Q3(Circuit::<3>::new())),
            4 => Some(CircuitEnum::Q4(Circuit::<4>::new())),
            5 => Some(CircuitEnum::Q5(Circuit::<5>::new())),
            8 => Some(CircuitEnum::Q8(Circuit::<8>::new())),
            10 => Some(CircuitEnum::Q10(Circuit::<10>::new())),
            16 => Some(CircuitEnum::Q16(Circuit::<16>::new())),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported number of qubits: {}. Supported values are 1, 2, 3, 4, 5, 8, 10, and 16.",
                    n_qubits
                )))
            }
        };

        Ok(Self { circuit })
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
        // Check if GPU is requested and available
        if use_gpu {
            #[cfg(feature = "gpu")]
            {
                use quantrs2_sim::gpu::GpuStateVectorSimulator;
                if !GpuStateVectorSimulator::is_available() {
                    return Err(PyValueError::new_err(
                        "GPU acceleration requested but not available on this system",
                    ));
                }

                let simulator =
                    std::panic::catch_unwind(|| GpuStateVectorSimulator::new_blocking())
                        .map_err(|_| PyValueError::new_err("Failed to initialize GPU simulator"))?;

                let simulator = simulator.map_err(|e| {
                    PyValueError::new_err(format!("Failed to initialize GPU simulator: {}", e))
                })?;

                // Run on the GPU simulator
                return match &self.circuit {
                    Some(CircuitEnum::Q1(c)) => {
                        let result = simulator.run(c).map_err(|e| {
                            PyValueError::new_err(format!("Error running GPU simulation: {}", e))
                        })?;
                        let simulation_result = PySimulationResult {
                            amplitudes: result.amplitudes().to_vec(),
                            n_qubits: 1,
                        };
                        Py::new(py, simulation_result)
                    }
                    Some(CircuitEnum::Q2(c)) => {
                        let result = simulator.run(c).map_err(|e| {
                            PyValueError::new_err(format!("Error running GPU simulation: {}", e))
                        })?;
                        let simulation_result = PySimulationResult {
                            amplitudes: result.amplitudes().to_vec(),
                            n_qubits: 2,
                        };
                        Py::new(py, simulation_result)
                    }
                    Some(CircuitEnum::Q3(c)) => {
                        let result = simulator.run(c).map_err(|e| {
                            PyValueError::new_err(format!("Error running GPU simulation: {}", e))
                        })?;
                        let simulation_result = PySimulationResult {
                            amplitudes: result.amplitudes().to_vec(),
                            n_qubits: 3,
                        };
                        Py::new(py, simulation_result)
                    }
                    Some(CircuitEnum::Q4(c)) => {
                        let result = simulator.run(c).map_err(|e| {
                            PyValueError::new_err(format!("Error running GPU simulation: {}", e))
                        })?;
                        let simulation_result = PySimulationResult {
                            amplitudes: result.amplitudes().to_vec(),
                            n_qubits: 4,
                        };
                        Py::new(py, simulation_result)
                    }
                    Some(CircuitEnum::Q5(c)) => {
                        let result = simulator.run(c).map_err(|e| {
                            PyValueError::new_err(format!("Error running GPU simulation: {}", e))
                        })?;
                        let simulation_result = PySimulationResult {
                            amplitudes: result.amplitudes().to_vec(),
                            n_qubits: 5,
                        };
                        Py::new(py, simulation_result)
                    }
                    Some(CircuitEnum::Q8(c)) => {
                        let result = simulator.run(c).map_err(|e| {
                            PyValueError::new_err(format!("Error running GPU simulation: {}", e))
                        })?;
                        let simulation_result = PySimulationResult {
                            amplitudes: result.amplitudes().to_vec(),
                            n_qubits: 8,
                        };
                        Py::new(py, simulation_result)
                    }
                    Some(CircuitEnum::Q10(c)) => {
                        let result = simulator.run(c).map_err(|e| {
                            PyValueError::new_err(format!("Error running GPU simulation: {}", e))
                        })?;
                        let simulation_result = PySimulationResult {
                            amplitudes: result.amplitudes().to_vec(),
                            n_qubits: 10,
                        };
                        Py::new(py, simulation_result)
                    }
                    Some(CircuitEnum::Q16(c)) => {
                        let result = simulator.run(c).map_err(|e| {
                            PyValueError::new_err(format!("Error running GPU simulation: {}", e))
                        })?;
                        let simulation_result = PySimulationResult {
                            amplitudes: result.amplitudes().to_vec(),
                            n_qubits: 16,
                        };
                        Py::new(py, simulation_result)
                    }
                    None => Err(PyValueError::new_err("Circuit not initialized")),
                };
            }

            #[cfg(not(feature = "gpu"))]
            {
                return Err(PyValueError::new_err(
                    "GPU acceleration requested but not compiled in. Recompile with the 'gpu' feature."
                ));
            }
        }

        // Use CPU simulation
        let simulator = StateVectorSimulator::new();
        // Run on the CPU simulator
        match &self.circuit {
            Some(CircuitEnum::Q1(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running CPU simulation: {}", e))
                })?;
                let simulation_result = PySimulationResult {
                    amplitudes: result.amplitudes().to_vec(),
                    n_qubits: 1,
                };
                Py::new(py, simulation_result)
            }
            Some(CircuitEnum::Q2(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running CPU simulation: {}", e))
                })?;
                let simulation_result = PySimulationResult {
                    amplitudes: result.amplitudes().to_vec(),
                    n_qubits: 2,
                };
                Py::new(py, simulation_result)
            }
            Some(CircuitEnum::Q3(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running CPU simulation: {}", e))
                })?;
                let simulation_result = PySimulationResult {
                    amplitudes: result.amplitudes().to_vec(),
                    n_qubits: 3,
                };
                Py::new(py, simulation_result)
            }
            Some(CircuitEnum::Q4(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running CPU simulation: {}", e))
                })?;
                let simulation_result = PySimulationResult {
                    amplitudes: result.amplitudes().to_vec(),
                    n_qubits: 4,
                };
                Py::new(py, simulation_result)
            }
            Some(CircuitEnum::Q5(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running CPU simulation: {}", e))
                })?;
                let simulation_result = PySimulationResult {
                    amplitudes: result.amplitudes().to_vec(),
                    n_qubits: 5,
                };
                Py::new(py, simulation_result)
            }
            Some(CircuitEnum::Q8(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running CPU simulation: {}", e))
                })?;
                let simulation_result = PySimulationResult {
                    amplitudes: result.amplitudes().to_vec(),
                    n_qubits: 8,
                };
                Py::new(py, simulation_result)
            }
            Some(CircuitEnum::Q10(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running CPU simulation: {}", e))
                })?;
                let simulation_result = PySimulationResult {
                    amplitudes: result.amplitudes().to_vec(),
                    n_qubits: 10,
                };
                Py::new(py, simulation_result)
            }
            Some(CircuitEnum::Q16(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running CPU simulation: {}", e))
                })?;
                let simulation_result = PySimulationResult {
                    amplitudes: result.amplitudes().to_vec(),
                    n_qubits: 16,
                };
                Py::new(py, simulation_result)
            }
            None => Err(PyValueError::new_err("Circuit not initialized")),
        }
    }
}

impl PyCircuit {
    /// Helper function to apply a gate to the circuit
    fn apply_gate(&mut self, op: CircuitOp) -> PyResult<()> {
        match &mut self.circuit {
            Some(CircuitEnum::Q1(c)) => {
                apply_op(c, op)
                    .map_err(|e| PyValueError::new_err(format!("Error applying gate: {}", e)))?;
                Ok(())
            }
            Some(CircuitEnum::Q2(c)) => {
                apply_op(c, op)
                    .map_err(|e| PyValueError::new_err(format!("Error applying gate: {}", e)))?;
                Ok(())
            }
            Some(CircuitEnum::Q3(c)) => {
                apply_op(c, op)
                    .map_err(|e| PyValueError::new_err(format!("Error applying gate: {}", e)))?;
                Ok(())
            }
            Some(CircuitEnum::Q4(c)) => {
                apply_op(c, op)
                    .map_err(|e| PyValueError::new_err(format!("Error applying gate: {}", e)))?;
                Ok(())
            }
            Some(CircuitEnum::Q5(c)) => {
                apply_op(c, op)
                    .map_err(|e| PyValueError::new_err(format!("Error applying gate: {}", e)))?;
                Ok(())
            }
            Some(CircuitEnum::Q8(c)) => {
                apply_op(c, op)
                    .map_err(|e| PyValueError::new_err(format!("Error applying gate: {}", e)))?;
                Ok(())
            }
            Some(CircuitEnum::Q10(c)) => {
                apply_op(c, op)
                    .map_err(|e| PyValueError::new_err(format!("Error applying gate: {}", e)))?;
                Ok(())
            }
            Some(CircuitEnum::Q16(c)) => {
                apply_op(c, op)
                    .map_err(|e| PyValueError::new_err(format!("Error applying gate: {}", e)))?;
                Ok(())
            }
            None => Err(PyValueError::new_err("Circuit not initialized")),
        }
    }

    /// Helper function to run a circuit on a given simulator
    fn run_with_simulator<S, const M: usize>(
        &self,
        py: Python,
        simulator: &S,
    ) -> PyResult<Py<PySimulationResult>>
    where
        S: quantrs2_circuit::prelude::Simulator<M>,
        S: quantrs2_circuit::prelude::Simulator<1>
            + quantrs2_circuit::prelude::Simulator<2>
            + quantrs2_circuit::prelude::Simulator<3>
            + quantrs2_circuit::prelude::Simulator<4>
            + quantrs2_circuit::prelude::Simulator<5>
            + quantrs2_circuit::prelude::Simulator<8>
            + quantrs2_circuit::prelude::Simulator<10>
            + quantrs2_circuit::prelude::Simulator<16>,
    {
        let result = match &self.circuit {
            Some(CircuitEnum::Q1(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running circuit simulation: {}", e))
                })?;
                (result.amplitudes().to_vec(), 1)
            }
            Some(CircuitEnum::Q2(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running circuit simulation: {}", e))
                })?;
                (result.amplitudes().to_vec(), 2)
            }
            Some(CircuitEnum::Q3(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running circuit simulation: {}", e))
                })?;
                (result.amplitudes().to_vec(), 3)
            }
            Some(CircuitEnum::Q4(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running circuit simulation: {}", e))
                })?;
                (result.amplitudes().to_vec(), 4)
            }
            Some(CircuitEnum::Q5(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running circuit simulation: {}", e))
                })?;
                (result.amplitudes().to_vec(), 5)
            }
            Some(CircuitEnum::Q8(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running circuit simulation: {}", e))
                })?;
                (result.amplitudes().to_vec(), 8)
            }
            Some(CircuitEnum::Q10(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running circuit simulation: {}", e))
                })?;
                (result.amplitudes().to_vec(), 10)
            }
            Some(CircuitEnum::Q16(c)) => {
                let result = simulator.run(c).map_err(|e| {
                    PyValueError::new_err(format!("Error running circuit simulation: {}", e))
                })?;
                (result.amplitudes().to_vec(), 16)
            }
            None => return Err(PyValueError::new_err("Circuit not initialized")),
        };

        let simulation_result = PySimulationResult {
            amplitudes: result.0,
            n_qubits: result.1,
        };
        Py::new(py, simulation_result)
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
}

/// Python module for QuantRS2
#[pymodule]
fn quantrs2(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add classes to the module
    m.add_class::<PyCircuit>()?;
    m.add_class::<PySimulationResult>()?;

    Ok(())
}
