//! Circuit helper methods for PyCircuit (visualization and gate application).

use crate::circuit_core::{CircuitOp, PyCircuit};
use crate::visualization::{create_visualizer_from_operations, PyCircuitVisualizer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

impl PyCircuit {
    /// Helper function to get a circuit visualizer based on the current circuit state
    #[allow(clippy::too_many_lines)]
    pub(crate) fn get_visualizer(&self) -> PyResult<Py<PyCircuitVisualizer>> {
        Python::attach(|py| {
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
    pub(crate) fn apply_gate(&mut self, op: CircuitOp) -> PyResult<()> {
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
