//! Gate optimization passes for quantum circuits
//!
//! This module provides various optimization techniques for quantum circuits,
//! including gate fusion, commutation, and peephole optimizations.

pub mod compression;
pub mod fusion;
pub mod lazy_evaluation;
pub mod peephole;
pub mod zx_optimizer;

use crate::error::QuantRS2Result;
use crate::gate::GateOp;
use crate::qubit::QubitId;

/// Trait for optimization passes
pub trait OptimizationPass {
    /// Apply the optimization pass to a sequence of gates
    fn optimize(&self, gates: Vec<Box<dyn GateOp>>) -> QuantRS2Result<Vec<Box<dyn GateOp>>>;

    /// Get the name of this optimization pass
    fn name(&self) -> &str;

    /// Check if this pass is applicable to the given gates
    fn is_applicable(&self, gates: &[Box<dyn GateOp>]) -> bool {
        !gates.is_empty()
    }
}

/// Chain multiple optimization passes together
pub struct OptimizationChain {
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl OptimizationChain {
    /// Create a new optimization chain
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add an optimization pass to the chain
    pub fn add_pass(mut self, pass: Box<dyn OptimizationPass>) -> Self {
        self.passes.push(pass);
        self
    }

    /// Apply all optimization passes in sequence
    pub fn optimize(
        &self,
        mut gates: Vec<Box<dyn GateOp>>,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        for pass in &self.passes {
            if pass.is_applicable(&gates) {
                gates = pass.optimize(gates)?;
            }
        }
        Ok(gates)
    }
}

/// Information about gate connectivity
#[derive(Debug, Clone)]
pub struct GateInfo {
    /// The gate being analyzed
    pub gate: Box<dyn GateOp>,
    /// Index in the gate sequence
    pub index: usize,
    /// Qubits this gate acts on
    pub qubits: Vec<QubitId>,
    /// Whether this gate is parameterized
    pub is_parameterized: bool,
}

/// Check if two gates act on disjoint qubits
pub fn gates_are_disjoint(gate1: &dyn GateOp, gate2: &dyn GateOp) -> bool {
    let qubits1 = gate1.qubits();
    let qubits2 = gate2.qubits();

    for q1 in &qubits1 {
        for q2 in &qubits2 {
            if q1 == q2 {
                return false;
            }
        }
    }
    true
}

/// Check if two gates can be commuted past each other
pub fn gates_can_commute(gate1: &dyn GateOp, gate2: &dyn GateOp) -> bool {
    // Disjoint gates always commute
    if gates_are_disjoint(gate1, gate2) {
        return true;
    }

    // Same single-qubit gates on same qubit might commute
    if gate1.qubits().len() == 1 && gate2.qubits().len() == 1 && gate1.qubits() == gate2.qubits() {
        match (gate1.name(), gate2.name()) {
            // Z-basis gates commute with each other
            ("Z", "Z")
            | ("Z", "S")
            | ("Z", "S†")
            | ("Z", "T")
            | ("Z", "T†")
            | ("S", "Z")
            | ("S", "S")
            | ("S", "S†")
            | ("S", "T")
            | ("S", "T†")
            | ("S†", "Z")
            | ("S†", "S")
            | ("S†", "S†")
            | ("S†", "T")
            | ("S†", "T†")
            | ("T", "Z")
            | ("T", "S")
            | ("T", "S†")
            | ("T", "T")
            | ("T", "T†")
            | ("T†", "Z")
            | ("T†", "S")
            | ("T†", "S†")
            | ("T†", "T")
            | ("T†", "T†")
            | ("RZ", "RZ")
            | ("RZ", "Z")
            | ("RZ", "S")
            | ("RZ", "T") => true,

            // X-basis gates commute with each other
            ("X", "X") | ("RX", "RX") | ("RX", "X") | ("X", "RX") => true,

            // Y-basis gates commute with each other
            ("Y", "Y") | ("RY", "RY") | ("RY", "Y") | ("Y", "RY") => true,

            _ => false,
        }
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::single::{Hadamard, PauliX, PauliZ};
    use crate::qubit::QubitId;

    #[test]
    fn test_gates_are_disjoint() {
        let gate1 = Box::new(PauliX { target: QubitId(0) }) as Box<dyn GateOp>;
        let gate2 = Box::new(PauliZ { target: QubitId(1) }) as Box<dyn GateOp>;
        let gate3 = Box::new(Hadamard { target: QubitId(0) }) as Box<dyn GateOp>;

        assert!(gates_are_disjoint(gate1.as_ref(), gate2.as_ref()));
        assert!(!gates_are_disjoint(gate1.as_ref(), gate3.as_ref()));
    }

    #[test]
    fn test_gates_can_commute() {
        let z1 = Box::new(PauliZ { target: QubitId(0) }) as Box<dyn GateOp>;
        let z2 = Box::new(PauliZ { target: QubitId(0) }) as Box<dyn GateOp>;
        let x1 = Box::new(PauliX { target: QubitId(0) }) as Box<dyn GateOp>;

        assert!(gates_can_commute(z1.as_ref(), z2.as_ref()));
        assert!(!gates_can_commute(z1.as_ref(), x1.as_ref()));
    }
}
