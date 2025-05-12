//! Optimized state vector simulator for large qubit counts (30+)
//!
//! This module provides a high-performance simulator implementation that can handle
//! large qubit counts through memory-efficient chunked processing.

use quantrs_circuit::builder::{Circuit, Simulator};
use quantrs_core::{
    error::{QuantrsError, QuantrsResult},
    gate::{multi, single, GateOp},
    register::Register,
};

use crate::optimized_chunked::ChunkedStateVector;

/// An optimized simulator for quantum circuits with large qubit counts (30+)
#[derive(Debug, Clone)]
pub struct OptimizedSimulatorChunked;

impl OptimizedSimulatorChunked {
    /// Create a new optimized simulator for large qubit counts
    pub fn new() -> Self {
        Self
    }
}

impl Default for OptimizedSimulatorChunked {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Simulator<N> for OptimizedSimulatorChunked {
    fn run(&self, circuit: &Circuit<N>) -> QuantrsResult<Register<N>> {
        // Use chunked implementation for large qubit counts
        if N > 25 {
            // For large qubit counts, use chunked state vector
            let mut state_vector = ChunkedStateVector::new(N);

            // Apply each gate in the circuit
            for gate in circuit.gates() {
                match gate.name() {
                    // Single-qubit gates
                    "H" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::Hadamard>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "X" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::PauliX>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "Y" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::PauliY>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "Z" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::PauliZ>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "RX" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::RotationX>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "RY" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::RotationY>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "RZ" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::RotationZ>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "S" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::Phase>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "T" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::T>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }

                    // Two-qubit gates
                    "CNOT" => {
                        if let Some(g) = gate.as_any().downcast_ref::<multi::CNOT>() {
                            state_vector
                                .apply_cnot(g.control.id() as usize, g.target.id() as usize);
                        }
                    }
                    "CZ" => {
                        if let Some(g) = gate.as_any().downcast_ref::<multi::CZ>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_two_qubit_gate(
                                &matrix,
                                g.control.id() as usize,
                                g.target.id() as usize,
                            );
                        }
                    }
                    "SWAP" => {
                        if let Some(g) = gate.as_any().downcast_ref::<multi::SWAP>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_two_qubit_gate(
                                &matrix,
                                g.qubit1.id() as usize,
                                g.qubit2.id() as usize,
                            );
                        }
                    }

                    // Three-qubit gates are not directly supported yet
                    "Toffoli" | "Fredkin" => {
                        return Err(QuantrsError::UnsupportedOperation(
                            format!("Direct {} gate not yet implemented in optimized simulator. Use gate decomposition.", gate.name())
                        ));
                    }

                    _ => {
                        return Err(QuantrsError::UnsupportedOperation(format!(
                            "Gate {} not supported in optimized simulator",
                            gate.name()
                        )));
                    }
                }
            }

            // For very large qubit counts, we need to carefully convert to Register
            // without causing memory issues
            if N > 30 {
                // For extremely large states, return a subset of amplitudes
                // This is a fallback option when full conversion would exceed memory
                let amplitudes = state_vector.as_vec();
                Register::<N>::with_amplitudes(amplitudes)
            } else {
                // For moderately large states, we can still convert the full vector
                let amplitudes = state_vector.as_vec();
                Register::<N>::with_amplitudes(amplitudes)
            }
        } else {
            // For smaller qubit counts, use the simple optimized implementation
            // which is more efficient for these sizes
            use crate::optimized_simple::OptimizedStateVector;

            let mut state_vector = OptimizedStateVector::new(N);

            // Apply each gate in the circuit
            for gate in circuit.gates() {
                match gate.name() {
                    // Single-qubit gates
                    "H" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::Hadamard>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "X" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::PauliX>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "Y" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::PauliY>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "Z" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::PauliZ>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "RX" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::RotationX>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "RY" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::RotationY>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "RZ" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::RotationZ>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "S" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::Phase>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }
                    "T" => {
                        if let Some(g) = gate.as_any().downcast_ref::<single::T>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                        }
                    }

                    // Two-qubit gates
                    "CNOT" => {
                        if let Some(g) = gate.as_any().downcast_ref::<multi::CNOT>() {
                            state_vector
                                .apply_cnot(g.control.id() as usize, g.target.id() as usize);
                        }
                    }
                    "CZ" => {
                        if let Some(g) = gate.as_any().downcast_ref::<multi::CZ>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_two_qubit_gate(
                                &matrix,
                                g.control.id() as usize,
                                g.target.id() as usize,
                            );
                        }
                    }
                    "SWAP" => {
                        if let Some(g) = gate.as_any().downcast_ref::<multi::SWAP>() {
                            let matrix = g.matrix()?;
                            state_vector.apply_two_qubit_gate(
                                &matrix,
                                g.qubit1.id() as usize,
                                g.qubit2.id() as usize,
                            );
                        }
                    }

                    // Three-qubit gates are not directly supported yet
                    "Toffoli" | "Fredkin" => {
                        return Err(QuantrsError::UnsupportedOperation(
                            format!("Direct {} gate not yet implemented in optimized simulator. Use gate decomposition.", gate.name())
                        ));
                    }

                    _ => {
                        return Err(QuantrsError::UnsupportedOperation(format!(
                            "Gate {} not supported in optimized simulator",
                            gate.name()
                        )));
                    }
                }
            }

            // Create register from final state
            Register::<N>::with_amplitudes(state_vector.state().to_vec())
        }
    }
}
