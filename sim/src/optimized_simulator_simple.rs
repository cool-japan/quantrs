//! Optimized state vector simulator for quantum circuits (Simplified version)
//!
//! This module provides a high-performance simulator implementation focused on
//! correctness and simplicity for quantum state vector simulation.

use num_complex::Complex64;

use quantrs_core::{
    error::{QuantrsError, QuantrsResult},
    gate::{GateOp, single, multi},
    qubit::QubitId,
    register::Register,
};
use quantrs_circuit::builder::{Circuit, Simulator};

use crate::optimized_simple::OptimizedStateVector;

/// A simplified optimized state vector simulator for quantum circuits
#[derive(Debug, Clone)]
pub struct OptimizedSimulatorSimple;

impl OptimizedSimulatorSimple {
    /// Create a new optimized simulator
    pub fn new() -> Self {
        Self
    }
}

impl Default for OptimizedSimulatorSimple {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Simulator<N> for OptimizedSimulatorSimple {
    fn run(&self, circuit: &Circuit<N>) -> QuantrsResult<Register<N>> {
        // Initialize state vector
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
                },
                "X" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::PauliX>() {
                        let matrix = g.matrix()?;
                        state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                    }
                },
                "Y" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::PauliY>() {
                        let matrix = g.matrix()?;
                        state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                    }
                },
                "Z" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::PauliZ>() {
                        let matrix = g.matrix()?;
                        state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                    }
                },
                "RX" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::RotationX>() {
                        let matrix = g.matrix()?;
                        state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                    }
                },
                "RY" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::RotationY>() {
                        let matrix = g.matrix()?;
                        state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                    }
                },
                "RZ" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::RotationZ>() {
                        let matrix = g.matrix()?;
                        state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                    }
                },
                "S" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::Phase>() {
                        let matrix = g.matrix()?;
                        state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                    }
                },
                "T" => {
                    if let Some(g) = gate.as_any().downcast_ref::<single::T>() {
                        let matrix = g.matrix()?;
                        state_vector.apply_single_qubit_gate(&matrix, g.target.id() as usize);
                    }
                },
                
                // Two-qubit gates
                "CNOT" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CNOT>() {
                        state_vector.apply_cnot(g.control.id() as usize, g.target.id() as usize);
                    }
                },
                "CZ" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::CZ>() {
                        let matrix = g.matrix()?;
                        state_vector.apply_two_qubit_gate(&matrix, g.control.id() as usize, g.target.id() as usize);
                    }
                },
                "SWAP" => {
                    if let Some(g) = gate.as_any().downcast_ref::<multi::SWAP>() {
                        let matrix = g.matrix()?;
                        state_vector.apply_two_qubit_gate(&matrix, g.qubit1.id() as usize, g.qubit2.id() as usize);
                    }
                },
                
                // Three-qubit gates are not directly supported yet
                "Toffoli" | "Fredkin" => {
                    return Err(QuantrsError::UnsupportedOperation(
                        format!("Direct {} gate not yet implemented in optimized simulator. Use gate decomposition.", gate.name())
                    ));
                },
                
                _ => {
                    return Err(QuantrsError::UnsupportedOperation(
                        format!("Gate {} not supported in optimized simulator", gate.name())
                    ));
                },
            }
        }
        
        // Create register from final state
        Register::<N>::with_amplitudes(state_vector.state().to_vec())
    }
}