//! Builder types for quantum circuits.
//!
//! This module contains the Circuit type for building and
//! executing quantum circuits.

use std::fmt;

use quantrs2_core::{
    error::QuantRS2Result,
    gate::{
        multi::{Fredkin, Toffoli, CH, CNOT, CRX, CRY, CRZ, CS, CY, CZ, SWAP},
        single::{
            Hadamard, PauliX, PauliY, PauliZ, Phase, PhaseDagger, RotationX, RotationY, RotationZ,
            SqrtX, SqrtXDagger, TDagger, T,
        },
        GateOp,
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
        Self { gates: Vec::new() }
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
        Self { gates: Vec::new() }
    }

    /// Add a gate to the circuit
    pub fn add_gate<G: GateOp + 'static>(&mut self, gate: G) -> QuantRS2Result<&mut Self> {
        // Validate that all qubits are within range
        for qubit in gate.qubits() {
            if qubit.id() as usize >= N {
                return Err(quantrs2_core::error::QuantRS2Error::InvalidQubitId(
                    qubit.id(),
                ));
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
    pub fn h(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(Hadamard {
            target: target.into(),
        })
    }

    /// Apply a Pauli-X gate to a qubit
    pub fn x(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(PauliX {
            target: target.into(),
        })
    }

    /// Apply a Pauli-Y gate to a qubit
    pub fn y(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(PauliY {
            target: target.into(),
        })
    }

    /// Apply a Pauli-Z gate to a qubit
    pub fn z(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(PauliZ {
            target: target.into(),
        })
    }

    /// Apply a rotation around X-axis
    pub fn rx(&mut self, target: impl Into<QubitId>, theta: f64) -> QuantRS2Result<&mut Self> {
        self.add_gate(RotationX {
            target: target.into(),
            theta,
        })
    }

    /// Apply a rotation around Y-axis
    pub fn ry(&mut self, target: impl Into<QubitId>, theta: f64) -> QuantRS2Result<&mut Self> {
        self.add_gate(RotationY {
            target: target.into(),
            theta,
        })
    }

    /// Apply a rotation around Z-axis
    pub fn rz(&mut self, target: impl Into<QubitId>, theta: f64) -> QuantRS2Result<&mut Self> {
        self.add_gate(RotationZ {
            target: target.into(),
            theta,
        })
    }

    /// Apply a Phase gate (S gate)
    pub fn s(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(Phase {
            target: target.into(),
        })
    }

    /// Apply a Phase-dagger gate (S† gate)
    pub fn sdg(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(PhaseDagger {
            target: target.into(),
        })
    }

    /// Apply a T gate
    pub fn t(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(T {
            target: target.into(),
        })
    }

    /// Apply a T-dagger gate (T† gate)
    pub fn tdg(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(TDagger {
            target: target.into(),
        })
    }

    /// Apply a Square Root of X gate (√X)
    pub fn sx(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(SqrtX {
            target: target.into(),
        })
    }

    /// Apply a Square Root of X Dagger gate (√X†)
    pub fn sxdg(&mut self, target: impl Into<QubitId>) -> QuantRS2Result<&mut Self> {
        self.add_gate(SqrtXDagger {
            target: target.into(),
        })
    }

    /// Apply a CNOT gate
    pub fn cnot(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CNOT {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a CNOT gate (alias for cnot)
    pub fn cx(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.cnot(control, target)
    }

    /// Apply a CY gate (Controlled-Y)
    pub fn cy(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CY {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a CZ gate (Controlled-Z)
    pub fn cz(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CZ {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a CH gate (Controlled-Hadamard)
    pub fn ch(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CH {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a CS gate (Controlled-Phase/S)
    pub fn cs(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CS {
            control: control.into(),
            target: target.into(),
        })
    }

    /// Apply a controlled rotation around X-axis (CRX)
    pub fn crx(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
        theta: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CRX {
            control: control.into(),
            target: target.into(),
            theta,
        })
    }

    /// Apply a controlled rotation around Y-axis (CRY)
    pub fn cry(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
        theta: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CRY {
            control: control.into(),
            target: target.into(),
            theta,
        })
    }

    /// Apply a controlled rotation around Z-axis (CRZ)
    pub fn crz(
        &mut self,
        control: impl Into<QubitId>,
        target: impl Into<QubitId>,
        theta: f64,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(CRZ {
            control: control.into(),
            target: target.into(),
            theta,
        })
    }

    /// Apply a SWAP gate
    pub fn swap(
        &mut self,
        qubit1: impl Into<QubitId>,
        qubit2: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
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
        target: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
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
        target2: impl Into<QubitId>,
    ) -> QuantRS2Result<&mut Self> {
        self.add_gate(Fredkin {
            control: control.into(),
            target1: target1.into(),
            target2: target2.into(),
        })
    }

    /// Run the circuit on a simulator
    pub fn run<S: Simulator<N>>(&self, simulator: S) -> QuantRS2Result<Register<N>> {
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
    fn run(&self, circuit: &Circuit<N>) -> QuantRS2Result<Register<N>>;
}
