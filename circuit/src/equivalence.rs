//! Circuit equivalence checking algorithms
//!
//! This module provides various methods to check if two quantum circuits
//! are equivalent, including exact and approximate equivalence.

use crate::builder::Circuit;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use ndarray::{array, Array2, ArrayView2};
use num_complex::Complex64;
use std::collections::HashMap;

/// Tolerance for numerical comparisons
const DEFAULT_TOLERANCE: f64 = 1e-10;

/// Result of equivalence check
#[derive(Debug, Clone)]
pub struct EquivalenceResult {
    /// Whether the circuits are equivalent
    pub equivalent: bool,
    /// Type of equivalence check performed
    pub check_type: EquivalenceType,
    /// Maximum difference found (for numerical checks)
    pub max_difference: Option<f64>,
    /// Additional details about the check
    pub details: String,
}

/// Types of equivalence checks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquivalenceType {
    /// Check if circuits produce identical unitaries
    UnitaryEquivalence,
    /// Check if circuits produce same output states for all inputs
    StateVectorEquivalence,
    /// Check if measurement probabilities are identical
    ProbabilisticEquivalence,
    /// Check if circuits have identical gate structure
    StructuralEquivalence,
    /// Check if circuits are equivalent up to a global phase
    GlobalPhaseEquivalence,
}

/// Options for equivalence checking
#[derive(Debug, Clone)]
pub struct EquivalenceOptions {
    /// Numerical tolerance for comparisons
    pub tolerance: f64,
    /// Whether to ignore global phase differences
    pub ignore_global_phase: bool,
    /// Whether to check all computational basis states
    pub check_all_states: bool,
    /// Maximum circuit size for unitary construction
    pub max_unitary_qubits: usize,
}

impl Default for EquivalenceOptions {
    fn default() -> Self {
        EquivalenceOptions {
            tolerance: DEFAULT_TOLERANCE,
            ignore_global_phase: true,
            check_all_states: true,
            max_unitary_qubits: 10,
        }
    }
}

/// Circuit equivalence checker
pub struct EquivalenceChecker {
    options: EquivalenceOptions,
}

impl EquivalenceChecker {
    /// Create a new equivalence checker with options
    pub fn new(options: EquivalenceOptions) -> Self {
        EquivalenceChecker { options }
    }
    
    /// Create a new equivalence checker with default options
    pub fn default() -> Self {
        Self::new(EquivalenceOptions::default())
    }
    
    /// Check if two circuits are equivalent using all methods
    pub fn check_equivalence<const N: usize>(
        &self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        // Try structural equivalence first (fastest)
        if let Ok(result) = self.check_structural_equivalence(circuit1, circuit2) {
            if result.equivalent {
                return Ok(result);
            }
        }
        
        // Try unitary equivalence if circuits are small enough
        if N <= self.options.max_unitary_qubits {
            return self.check_unitary_equivalence(circuit1, circuit2);
        }
        
        // For larger circuits, use state vector equivalence
        self.check_state_vector_equivalence(circuit1, circuit2)
    }
    
    /// Check structural equivalence (exact gate-by-gate match)
    pub fn check_structural_equivalence<const N: usize>(
        &self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        if circuit1.num_gates() != circuit2.num_gates() {
            return Ok(EquivalenceResult {
                equivalent: false,
                check_type: EquivalenceType::StructuralEquivalence,
                max_difference: None,
                details: format!(
                    "Different number of gates: {} vs {}",
                    circuit1.num_gates(),
                    circuit2.num_gates()
                ),
            });
        }
        
        let gates1 = circuit1.gates();
        let gates2 = circuit2.gates();
        
        for (i, (gate1, gate2)) in gates1.iter().zip(gates2.iter()).enumerate() {
            if !self.gates_equal(gate1.as_ref(), gate2.as_ref()) {
                return Ok(EquivalenceResult {
                    equivalent: false,
                    check_type: EquivalenceType::StructuralEquivalence,
                    max_difference: None,
                    details: format!(
                        "Gates differ at position {}: {} vs {}",
                        i,
                        gate1.name(),
                        gate2.name()
                    ),
                });
            }
        }
        
        Ok(EquivalenceResult {
            equivalent: true,
            check_type: EquivalenceType::StructuralEquivalence,
            max_difference: Some(0.0),
            details: "Circuits are structurally identical".to_string(),
        })
    }
    
    /// Check if two gates are equal
    fn gates_equal(&self, gate1: &dyn GateOp, gate2: &dyn GateOp) -> bool {
        // Check gate names
        if gate1.name() != gate2.name() {
            return false;
        }
        
        // Check qubits
        let qubits1 = gate1.qubits();
        let qubits2 = gate2.qubits();
        
        if qubits1.len() != qubits2.len() {
            return false;
        }
        
        for (q1, q2) in qubits1.iter().zip(qubits2.iter()) {
            if q1 != q2 {
                return false;
            }
        }
        
        // TODO: Check parameters for parametric gates
        true
    }
    
    /// Check unitary equivalence
    pub fn check_unitary_equivalence<const N: usize>(
        &self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        if N > self.options.max_unitary_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Circuit too large for unitary construction: {} qubits (max: {})",
                N,
                self.options.max_unitary_qubits
            )));
        }
        
        // Get unitaries for both circuits
        let unitary1 = self.get_circuit_unitary(circuit1)?;
        let unitary2 = self.get_circuit_unitary(circuit2)?;
        
        // Compare unitaries
        let (equivalent, max_diff) = if self.options.ignore_global_phase {
            self.unitaries_equal_up_to_phase(&unitary1, &unitary2)
        } else {
            self.unitaries_equal(&unitary1, &unitary2)
        };
        
        Ok(EquivalenceResult {
            equivalent,
            check_type: if self.options.ignore_global_phase {
                EquivalenceType::GlobalPhaseEquivalence
            } else {
                EquivalenceType::UnitaryEquivalence
            },
            max_difference: Some(max_diff),
            details: if equivalent {
                "Unitaries are equivalent".to_string()
            } else {
                format!("Maximum unitary difference: {:.2e}", max_diff)
            },
        })
    }
    
    /// Get the unitary matrix for a circuit
    fn get_circuit_unitary<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<Array2<Complex64>> {
        let dim = 1 << N;
        let mut unitary = Array2::eye(dim);
        
        // Apply each gate to the unitary
        for gate in circuit.gates() {
            self.apply_gate_to_unitary(&mut unitary, gate.as_ref(), N)?;
        }
        
        Ok(unitary)
    }
    
    /// Apply a gate to a unitary matrix
    fn apply_gate_to_unitary(
        &self,
        unitary: &mut Array2<Complex64>,
        gate: &dyn GateOp,
        num_qubits: usize,
    ) -> QuantRS2Result<()> {
        let gate_matrix = self.get_gate_matrix(gate)?;
        let qubits = gate.qubits();
        
        // Apply the gate matrix to the full unitary
        match qubits.len() {
            1 => {
                let qubit_idx = qubits[0].id() as usize;
                self.apply_single_qubit_gate(unitary, &gate_matrix, qubit_idx, num_qubits)?;
            }
            2 => {
                let control_idx = qubits[0].id() as usize;
                let target_idx = qubits[1].id() as usize;
                self.apply_two_qubit_gate(unitary, &gate_matrix, control_idx, target_idx, num_qubits)?;
            }
            _ => {
                return Err(QuantRS2Error::UnsupportedOperation(
                    format!("Gates with {} qubits not yet supported", qubits.len())
                ));
            }
        }
        
        Ok(())
    }
    
    /// Get the matrix representation of a gate
    fn get_gate_matrix(&self, gate: &dyn GateOp) -> QuantRS2Result<Array2<Complex64>> {
        let c0 = Complex64::new(0.0, 0.0);
        let c1 = Complex64::new(1.0, 0.0);
        let ci = Complex64::new(0.0, 1.0);
        
        match gate.name() {
            "H" => {
                let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
                Ok(array![[c1 * sqrt2_inv, c1 * sqrt2_inv],
                         [c1 * sqrt2_inv, -c1 * sqrt2_inv]])
            }
            "X" => Ok(array![[c0, c1], [c1, c0]]),
            "Y" => Ok(array![[c0, -ci], [ci, c0]]),
            "Z" => Ok(array![[c1, c0], [c0, -c1]]),
            "S" => Ok(array![[c1, c0], [c0, ci]]),
            "T" => Ok(array![[c1, c0], [c0, Complex64::new(1.0/std::f64::consts::SQRT_2, 1.0/std::f64::consts::SQRT_2)]]),
            "CNOT" | "CX" => Ok(array![
                [c1, c0, c0, c0],
                [c0, c1, c0, c0],
                [c0, c0, c0, c1],
                [c0, c0, c1, c0]
            ]),
            "CZ" => Ok(array![
                [c1, c0, c0, c0],
                [c0, c1, c0, c0],
                [c0, c0, c1, c0],
                [c0, c0, c0, -c1]
            ]),
            "SWAP" => Ok(array![
                [c1, c0, c0, c0],
                [c0, c0, c1, c0],
                [c0, c1, c0, c0],
                [c0, c0, c0, c1]
            ]),
            _ => Err(QuantRS2Error::UnsupportedOperation(
                format!("Gate '{}' matrix not yet implemented", gate.name())
            ))
        }
    }
    
    /// Apply a single-qubit gate to a unitary matrix
    fn apply_single_qubit_gate(
        &self,
        unitary: &mut Array2<Complex64>,
        gate_matrix: &Array2<Complex64>,
        qubit_idx: usize,
        num_qubits: usize,
    ) -> QuantRS2Result<()> {
        let dim = 1 << num_qubits;
        let mut new_unitary = Array2::zeros((dim, dim));
        
        // Apply gate to each basis state
        for col in 0..dim {
            for row in 0..dim {
                let mut sum = Complex64::new(0.0, 0.0);
                
                // Check if this element should be affected by the gate
                let row_bit = (row >> qubit_idx) & 1;
                let col_bit = (col >> qubit_idx) & 1;
                
                for k in 0..dim {
                    let k_bit = (k >> qubit_idx) & 1;
                    
                    // Only mix states that differ in the target qubit
                    if (row ^ k) == ((row_bit ^ k_bit) << qubit_idx) {
                        sum += gate_matrix[[row_bit, k_bit]] * unitary[[k, col]];
                    }
                }
                
                new_unitary[[row, col]] = sum;
            }
        }
        
        *unitary = new_unitary;
        Ok(())
    }
    
    /// Apply a two-qubit gate to a unitary matrix
    fn apply_two_qubit_gate(
        &self,
        unitary: &mut Array2<Complex64>,
        gate_matrix: &Array2<Complex64>,
        qubit1_idx: usize,
        qubit2_idx: usize,
        num_qubits: usize,
    ) -> QuantRS2Result<()> {
        let dim = 1 << num_qubits;
        let mut new_unitary = Array2::zeros((dim, dim));
        
        for col in 0..dim {
            for row in 0..dim {
                let mut sum = Complex64::new(0.0, 0.0);
                
                // Extract relevant qubit states
                let row_q1 = (row >> qubit1_idx) & 1;
                let row_q2 = (row >> qubit2_idx) & 1;
                let row_gate_idx = (row_q1 << 1) | row_q2;
                
                let col_q1 = (col >> qubit1_idx) & 1;
                let col_q2 = (col >> qubit2_idx) & 1;
                
                for k in 0..dim {
                    let k_q1 = (k >> qubit1_idx) & 1;
                    let k_q2 = (k >> qubit2_idx) & 1;
                    let k_gate_idx = (k_q1 << 1) | k_q2;
                    
                    // Check if k differs from row only in the gate qubits
                    let diff = row ^ k;
                    let expected_diff = ((row_q1 ^ k_q1) << qubit1_idx) | ((row_q2 ^ k_q2) << qubit2_idx);
                    
                    if diff == expected_diff {
                        sum += gate_matrix[[row_gate_idx, k_gate_idx]] * unitary[[k, col]];
                    }
                }
                
                new_unitary[[row, col]] = sum;
            }
        }
        
        *unitary = new_unitary;
        Ok(())
    }
    
    /// Check if two unitaries are equal
    fn unitaries_equal(
        &self,
        u1: &Array2<Complex64>,
        u2: &Array2<Complex64>,
    ) -> (bool, f64) {
        if u1.shape() != u2.shape() {
            return (false, f64::INFINITY);
        }
        
        let mut max_diff = 0.0;
        for (a, b) in u1.iter().zip(u2.iter()) {
            let diff = (a - b).norm();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > self.options.tolerance {
                return (false, max_diff);
            }
        }
        
        (true, max_diff)
    }
    
    /// Check if two unitaries are equal up to a global phase
    fn unitaries_equal_up_to_phase(
        &self,
        u1: &Array2<Complex64>,
        u2: &Array2<Complex64>,
    ) -> (bool, f64) {
        if u1.shape() != u2.shape() {
            return (false, f64::INFINITY);
        }
        
        // Find the first non-zero element to determine phase
        let mut phase = None;
        for (a, b) in u1.iter().zip(u2.iter()) {
            if a.norm() > self.options.tolerance && b.norm() > self.options.tolerance {
                phase = Some(b / a);
                break;
            }
        }
        
        let phase = match phase {
            Some(p) => p,
            None => return (false, f64::INFINITY),
        };
        
        // Check all elements with phase adjustment
        let mut max_diff = 0.0;
        for (a, b) in u1.iter().zip(u2.iter()) {
            let adjusted = a * phase;
            let diff = (adjusted - b).norm();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > self.options.tolerance {
                return (false, max_diff);
            }
        }
        
        (true, max_diff)
    }
    
    /// Check state vector equivalence
    pub fn check_state_vector_equivalence<const N: usize>(
        &self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        let mut max_diff = 0.0;
        let num_states = if self.options.check_all_states {
            1 << N
        } else {
            // Check a subset of states for large circuits
            std::cmp::min(1 << N, 100)
        };
        
        for state_idx in 0..num_states {
            let state1 = self.apply_circuit_to_state(circuit1, state_idx, N)?;
            let state2 = self.apply_circuit_to_state(circuit2, state_idx, N)?;
            
            let (equal, diff) = if self.options.ignore_global_phase {
                self.states_equal_up_to_phase(&state1, &state2)
            } else {
                self.states_equal(&state1, &state2)
            };
            
            if diff > max_diff {
                max_diff = diff;
            }
            
            if !equal {
                return Ok(EquivalenceResult {
                    equivalent: false,
                    check_type: EquivalenceType::StateVectorEquivalence,
                    max_difference: Some(max_diff),
                    details: format!(
                        "States differ for input |{:0b}>: max difference {:.2e}",
                        state_idx, max_diff
                    ),
                });
            }
        }
        
        Ok(EquivalenceResult {
            equivalent: true,
            check_type: EquivalenceType::StateVectorEquivalence,
            max_difference: Some(max_diff),
            details: format!(
                "Checked {} computational basis states",
                num_states
            ),
        })
    }
    
    /// Apply circuit to a computational basis state
    fn apply_circuit_to_state<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        state_idx: usize,
        num_qubits: usize,
    ) -> QuantRS2Result<Vec<Complex64>> {
        let dim = 1 << num_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[state_idx] = Complex64::new(1.0, 0.0);
        
        // Apply each gate to the state vector
        for gate in circuit.gates() {
            self.apply_gate_to_state(&mut state, gate.as_ref(), num_qubits)?;
        }
        
        Ok(state)
    }
    
    /// Apply a gate to a state vector
    fn apply_gate_to_state(
        &self,
        state: &mut Vec<Complex64>,
        gate: &dyn GateOp,
        num_qubits: usize,
    ) -> QuantRS2Result<()> {
        let gate_matrix = self.get_gate_matrix(gate)?;
        let qubits = gate.qubits();
        
        match qubits.len() {
            1 => {
                let qubit_idx = qubits[0].id() as usize;
                self.apply_single_qubit_gate_to_state(state, &gate_matrix, qubit_idx, num_qubits)?;
            }
            2 => {
                let control_idx = qubits[0].id() as usize;
                let target_idx = qubits[1].id() as usize;
                self.apply_two_qubit_gate_to_state(state, &gate_matrix, control_idx, target_idx, num_qubits)?;
            }
            _ => {
                return Err(QuantRS2Error::UnsupportedOperation(
                    format!("Gates with {} qubits not yet supported", qubits.len())
                ));
            }
        }
        
        Ok(())
    }
    
    /// Apply a single-qubit gate to a state vector
    fn apply_single_qubit_gate_to_state(
        &self,
        state: &mut Vec<Complex64>,
        gate_matrix: &Array2<Complex64>,
        qubit_idx: usize,
        num_qubits: usize,
    ) -> QuantRS2Result<()> {
        let dim = 1 << num_qubits;
        let mut new_state = vec![Complex64::new(0.0, 0.0); dim];
        
        for i in 0..dim {
            let bit = (i >> qubit_idx) & 1;
            
            for j in 0..2 {
                let other_idx = i ^ ((bit ^ j) << qubit_idx);
                new_state[i] += gate_matrix[[bit, j]] * state[other_idx];
            }
        }
        
        *state = new_state;
        Ok(())
    }
    
    /// Apply a two-qubit gate to a state vector
    fn apply_two_qubit_gate_to_state(
        &self,
        state: &mut Vec<Complex64>,
        gate_matrix: &Array2<Complex64>,
        qubit1_idx: usize,
        qubit2_idx: usize,
        num_qubits: usize,
    ) -> QuantRS2Result<()> {
        let dim = 1 << num_qubits;
        let mut new_state = vec![Complex64::new(0.0, 0.0); dim];
        
        for i in 0..dim {
            let bit1 = (i >> qubit1_idx) & 1;
            let bit2 = (i >> qubit2_idx) & 1;
            let gate_row = (bit1 << 1) | bit2;
            
            for gate_col in 0..4 {
                let new_bit1 = (gate_col >> 1) & 1;
                let new_bit2 = gate_col & 1;
                
                let j = i ^ ((bit1 ^ new_bit1) << qubit1_idx) ^ ((bit2 ^ new_bit2) << qubit2_idx);
                new_state[i] += gate_matrix[[gate_row, gate_col]] * state[j];
            }
        }
        
        *state = new_state;
        Ok(())
    }
    
    /// Check if two state vectors are equal
    fn states_equal(&self, s1: &[Complex64], s2: &[Complex64]) -> (bool, f64) {
        if s1.len() != s2.len() {
            return (false, f64::INFINITY);
        }
        
        let mut max_diff = 0.0;
        for (a, b) in s1.iter().zip(s2.iter()) {
            let diff = (a - b).norm();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > self.options.tolerance {
                return (false, max_diff);
            }
        }
        
        (true, max_diff)
    }
    
    /// Check if two state vectors are equal up to a global phase
    fn states_equal_up_to_phase(&self, s1: &[Complex64], s2: &[Complex64]) -> (bool, f64) {
        if s1.len() != s2.len() {
            return (false, f64::INFINITY);
        }
        
        // Find phase from first non-zero element
        let mut phase = None;
        for (a, b) in s1.iter().zip(s2.iter()) {
            if a.norm() > self.options.tolerance && b.norm() > self.options.tolerance {
                phase = Some(b / a);
                break;
            }
        }
        
        let phase = match phase {
            Some(p) => p,
            None => return (false, f64::INFINITY),
        };
        
        // Check all elements with phase adjustment
        let mut max_diff = 0.0;
        for (a, b) in s1.iter().zip(s2.iter()) {
            let adjusted = a * phase;
            let diff = (adjusted - b).norm();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > self.options.tolerance {
                return (false, max_diff);
            }
        }
        
        (true, max_diff)
    }
    
    /// Check probabilistic equivalence (measurement outcomes)
    pub fn check_probabilistic_equivalence<const N: usize>(
        &self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        // For each computational basis state, check measurement probabilities
        let mut max_diff = 0.0;
        
        for state_idx in 0..(1 << N) {
            let probs1 = self.get_measurement_probabilities(circuit1, state_idx, N)?;
            let probs2 = self.get_measurement_probabilities(circuit2, state_idx, N)?;
            
            for (p1, p2) in probs1.iter().zip(probs2.iter()) {
                let diff = (p1 - p2).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > self.options.tolerance {
                    return Ok(EquivalenceResult {
                        equivalent: false,
                        check_type: EquivalenceType::ProbabilisticEquivalence,
                        max_difference: Some(max_diff),
                        details: format!(
                            "Measurement probabilities differ for input |{:0b}>",
                            state_idx
                        ),
                    });
                }
            }
        }
        
        Ok(EquivalenceResult {
            equivalent: true,
            check_type: EquivalenceType::ProbabilisticEquivalence,
            max_difference: Some(max_diff),
            details: "Measurement probabilities match for all inputs".to_string(),
        })
    }
    
    /// Get measurement probabilities for a circuit and input state
    fn get_measurement_probabilities<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        state_idx: usize,
        num_qubits: usize,
    ) -> QuantRS2Result<Vec<f64>> {
        // Apply circuit to get final state
        let final_state = self.apply_circuit_to_state(circuit, state_idx, num_qubits)?;
        
        // Calculate probabilities from amplitudes
        let probs: Vec<f64> = final_state
            .iter()
            .map(|amplitude| amplitude.norm_sqr())
            .collect();
        
        Ok(probs)
    }
}

/// Quick check if two circuits are structurally identical
pub fn circuits_structurally_equal<const N: usize>(
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
) -> bool {
    let checker = EquivalenceChecker::default();
    checker
        .check_structural_equivalence(circuit1, circuit2)
        .map(|result| result.equivalent)
        .unwrap_or(false)
}

/// Quick check if two circuits are equivalent (using default options)
pub fn circuits_equivalent<const N: usize>(
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
) -> QuantRS2Result<bool> {
    let checker = EquivalenceChecker::default();
    Ok(checker.check_equivalence(circuit1, circuit2)?.equivalent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::single::{Hadamard, PauliX, PauliZ};
    use quantrs2_core::gate::multi::CNOT;
    
    #[test]
    fn test_structural_equivalence() {
        let mut circuit1 = Circuit::<2>::new();
        circuit1.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit1.add_gate(CNOT { control: QubitId(0), target: QubitId(1) }).unwrap();
        
        let mut circuit2 = Circuit::<2>::new();
        circuit2.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit2.add_gate(CNOT { control: QubitId(0), target: QubitId(1) }).unwrap();
        
        let checker = EquivalenceChecker::default();
        let result = checker.check_structural_equivalence(&circuit1, &circuit2).unwrap();
        assert!(result.equivalent);
    }
    
    #[test]
    fn test_structural_non_equivalence() {
        let mut circuit1 = Circuit::<2>::new();
        circuit1.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit1.add_gate(CNOT { control: QubitId(0), target: QubitId(1) }).unwrap();
        
        let mut circuit2 = Circuit::<2>::new();
        circuit2.add_gate(PauliX { target: QubitId(0) }).unwrap();
        circuit2.add_gate(CNOT { control: QubitId(0), target: QubitId(1) }).unwrap();
        
        let checker = EquivalenceChecker::default();
        let result = checker.check_structural_equivalence(&circuit1, &circuit2).unwrap();
        assert!(!result.equivalent);
    }
    
    #[test]
    fn test_different_gate_count() {
        let mut circuit1 = Circuit::<2>::new();
        circuit1.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        
        let mut circuit2 = Circuit::<2>::new();
        circuit2.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit2.add_gate(PauliZ { target: QubitId(0) }).unwrap();
        
        let checker = EquivalenceChecker::default();
        let result = checker.check_structural_equivalence(&circuit1, &circuit2).unwrap();
        assert!(!result.equivalent);
        assert!(result.details.contains("Different number of gates"));
    }
}