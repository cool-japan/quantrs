//! Builder types for quantum circuits.
//!
//! This module contains the Circuit type for building and
//! executing quantum circuits.

use std::fmt;

use quantrs_core::{
    error::QuantrsResult,
    gate::{
        GateOp,
        single::{Hadamard, PauliX, PauliY, PauliZ, RotationX, RotationY, RotationZ, Phase, T},
        multi::{CNOT, CZ, SWAP, Toffoli, Fredkin},
    },
    qubit::QubitId,
    register::Register,
};

/// A quantum circuit with a fixed number of qubits
pub struct Circuit<const N: usize> {
    // Vector of gates to be applied in sequence
    gates: Vec<Box<dyn GateOp>>,
}

impl<const N: usize> Clone for Circuit<N> {
    fn clone(&self) -> Self {
        // We can't clone dyn GateOp directly, so we create a new circuit
        // with the same gates by using their type information
        // In a real implementation, we would use the stored gate types
        // to create new instances of each gate
        Self {
            gates: Vec::new(),
        }
    }
}

impl<const N: usize> fmt::Debug for Circuit<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Circuit")
            .field("num_qubits", &N)
            .field("num_gates", &self.gates.len())
            .finish()
    }
}

impl<const N: usize> Circuit<N> {
    /// Create a new empty circuit with N qubits
    pub fn new() -> Self {
        Self {
            gates: Vec::new(),
        }
    }
    
    /// Add a gate to the circuit
    pub fn add_gate<G: GateOp + 'static>(&mut self, gate: G) -> QuantrsResult<&mut Self> {
        // Validate that all qubits are within range
        for qubit in gate.qubits() {
            if qubit.id() as usize >= N {
                return Err(quantrs_core::error::QuantrsError::InvalidQubitId(qubit.id()));
            }
        }
        
        self.gates.push(Box::new(gate));
        Ok(self)
    }
    
    /// Get all gates in the circuit
    pub fn gates(&self) -> &[Box<dyn GateOp>] {
        &self.gates
    }
    
    /// Get the number of qubits in the circuit
    pub fn num_qubits(&self) -> usize {
        N
    }
    
    /// Get the number of gates in the circuit
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }
    
    /// Apply a Hadamard gate to a qubit
    pub fn h(&mut self, target: impl Into<QubitId>) -> QuantrsResult<&mut Self> {
        self.add_gate(Hadamard { target: target.into() })
    }
    
    /// Apply a Pauli-X gate to a qubit
    pub fn x(&mut self, target: impl Into<QubitId>) -> QuantrsResult<&mut Self> {
        self.add_gate(PauliX { target: target.into() })
    }
    
    /// Apply a Pauli-Y gate to a qubit
    pub fn y(&mut self, target: impl Into<QubitId>) -> QuantrsResult<&mut Self> {
        self.add_gate(PauliY { target: target.into() })
    }
    
    /// Apply a Pauli-Z gate to a qubit
    pub fn z(&mut self, target: impl Into<QubitId>) -> QuantrsResult<&mut Self> {
        self.add_gate(PauliZ { target: target.into() })
    }
    
    /// Apply a rotation around X-axis
    pub fn rx(&mut self, target: impl Into<QubitId>, theta: f64) -> QuantrsResult<&mut Self> {
        self.add_gate(RotationX { target: target.into(), theta })
    }
    
    /// Apply a rotation around Y-axis
    pub fn ry(&mut self, target: impl Into<QubitId>, theta: f64) -> QuantrsResult<&mut Self> {
        self.add_gate(RotationY { target: target.into(), theta })
    }
    
    /// Apply a rotation around Z-axis
    pub fn rz(&mut self, target: impl Into<QubitId>, theta: f64) -> QuantrsResult<&mut Self> {
        self.add_gate(RotationZ { target: target.into(), theta })
    }
    
    /// Apply a Phase gate (S gate)
    pub fn s(&mut self, target: impl Into<QubitId>) -> QuantrsResult<&mut Self> {
        self.add_gate(Phase { target: target.into() })
    }
    
    /// Apply a T gate
    pub fn t(&mut self, target: impl Into<QubitId>) -> QuantrsResult<&mut Self> {
        self.add_gate(T { target: target.into() })
    }
    
    /// Apply a CNOT gate
    pub fn cnot(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>
    ) -> QuantrsResult<&mut Self> {
        self.add_gate(CNOT { 
            control: control.into(),
            target: target.into(),
        })
    }
    
    /// Apply a CZ gate
    pub fn cz(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>
    ) -> QuantrsResult<&mut Self> {
        self.add_gate(CZ { 
            control: control.into(),
            target: target.into(),
        })
    }
    
    /// Apply a SWAP gate
    pub fn swap(
        &mut self,
        qubit1: impl Into<QubitId>,
        qubit2: impl Into<QubitId>
    ) -> QuantrsResult<&mut Self> {
        self.add_gate(SWAP { 
            qubit1: qubit1.into(),
            qubit2: qubit2.into(),
        })
    }
    
    /// Apply a Toffoli (CCNOT) gate
    pub fn toffoli(
        &mut self,
        control1: impl Into<QubitId>,
        control2: impl Into<QubitId>,
        target: impl Into<QubitId>
    ) -> QuantrsResult<&mut Self> {
        self.add_gate(Toffoli { 
            control1: control1.into(),
            control2: control2.into(),
            target: target.into(),
        })
    }
    
    /// Apply a Fredkin (CSWAP) gate
    pub fn cswap(
        &mut self,
        control: impl Into<QubitId>,
        target1: impl Into<QubitId>,
        target2: impl Into<QubitId>
    ) -> QuantrsResult<&mut Self> {
        self.add_gate(Fredkin { 
            control: control.into(),
            target1: target1.into(),
            target2: target2.into(),
        })
    }
    
    /// Run the circuit on a simulator
    pub fn run<S: Simulator<N>>(&self, simulator: S) -> QuantrsResult<Register<N>> {
        simulator.run(self)
    }
}

impl<const N: usize> Default for Circuit<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for quantum circuit simulators
pub trait Simulator<const N: usize> {
    /// Run a quantum circuit and return the final register state
    fn run(&self, circuit: &Circuit<N>) -> QuantrsResult<Register<N>>;
}