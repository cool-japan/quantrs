//! Pauli string evolution and operations.
//!
//! This module provides efficient operations for Pauli strings, including:
//! - Pauli string construction and manipulation
//! - Time evolution of Pauli observables
//! - Commutation relations and algebra
//! - Measurement expectation values

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;

use crate::error::{Result, SimulatorError};
use crate::trotter::{
    DynamicCircuit, Hamiltonian, HamiltonianTerm, TrotterDecomposer, TrotterMethod,
};
use quantrs2_core::gate::{multi::*, single::*, GateOp};
use quantrs2_core::qubit::QubitId;

/// Single Pauli operator type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PauliOperator {
    /// Identity operator
    I,
    /// Pauli-X operator
    X,
    /// Pauli-Y operator
    Y,
    /// Pauli-Z operator
    Z,
}

impl fmt::Display for PauliOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PauliOperator::I => write!(f, "I"),
            PauliOperator::X => write!(f, "X"),
            PauliOperator::Y => write!(f, "Y"),
            PauliOperator::Z => write!(f, "Z"),
        }
    }
}

impl PauliOperator {
    /// Parse from string
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "I" => Ok(PauliOperator::I),
            "X" => Ok(PauliOperator::X),
            "Y" => Ok(PauliOperator::Y),
            "Z" => Ok(PauliOperator::Z),
            _ => Err(SimulatorError::InvalidInput(format!(
                "Invalid Pauli operator: {}",
                s
            ))),
        }
    }

    /// Get matrix representation
    pub fn matrix(&self) -> Array2<Complex64> {
        match self {
            PauliOperator::I => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .unwrap(),
            PauliOperator::X => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            PauliOperator::Y => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, -1.0),
                    Complex64::new(0.0, 1.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            PauliOperator::Z => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )
            .unwrap(),
        }
    }

    /// Check if commutes with another Pauli
    pub fn commutes_with(&self, other: &PauliOperator) -> bool {
        match (self, other) {
            (PauliOperator::I, _) | (_, PauliOperator::I) => true,
            (a, b) if a == b => true,
            _ => false,
        }
    }

    /// Multiplication of Pauli operators (returns (result, phase))
    pub fn multiply(&self, other: &PauliOperator) -> (PauliOperator, Complex64) {
        match (self, other) {
            (PauliOperator::I, p) | (p, PauliOperator::I) => (*p, Complex64::new(1.0, 0.0)),
            (PauliOperator::X, PauliOperator::X)
            | (PauliOperator::Y, PauliOperator::Y)
            | (PauliOperator::Z, PauliOperator::Z) => (PauliOperator::I, Complex64::new(1.0, 0.0)),
            (PauliOperator::X, PauliOperator::Y) => (PauliOperator::Z, Complex64::new(0.0, 1.0)),
            (PauliOperator::Y, PauliOperator::X) => (PauliOperator::Z, Complex64::new(0.0, -1.0)),
            (PauliOperator::Y, PauliOperator::Z) => (PauliOperator::X, Complex64::new(0.0, 1.0)),
            (PauliOperator::Z, PauliOperator::Y) => (PauliOperator::X, Complex64::new(0.0, -1.0)),
            (PauliOperator::Z, PauliOperator::X) => (PauliOperator::Y, Complex64::new(0.0, 1.0)),
            (PauliOperator::X, PauliOperator::Z) => (PauliOperator::Y, Complex64::new(0.0, -1.0)),
        }
    }
}

/// A Pauli string is a tensor product of Pauli operators
#[derive(Debug, Clone)]
pub struct PauliString {
    /// Pauli operators at each qubit position  
    pub operators: Vec<PauliOperator>,
    /// Overall coefficient
    pub coefficient: Complex64,
    /// Number of qubits
    pub num_qubits: usize,
}

impl PauliString {
    /// Create a new Pauli string
    pub fn new(num_qubits: usize) -> Self {
        Self {
            operators: vec![PauliOperator::I; num_qubits],
            coefficient: Complex64::new(1.0, 0.0),
            num_qubits,
        }
    }

    /// Create from string representation like "XYZI"
    pub fn from_string(pauli_str: &str, coefficient: Complex64) -> Result<Self> {
        let operators: Result<Vec<_>> = pauli_str
            .chars()
            .map(|c| PauliOperator::from_str(&c.to_string()))
            .collect();

        Ok(Self {
            operators: operators?,
            coefficient,
            num_qubits: pauli_str.len(),
        })
    }

    /// Create from qubit indices and Pauli operators
    pub fn from_ops(
        num_qubits: usize,
        ops: &[(usize, PauliOperator)],
        coefficient: Complex64,
    ) -> Result<Self> {
        let mut pauli_string = Self::new(num_qubits);
        pauli_string.coefficient = coefficient;

        for &(qubit, op) in ops {
            if qubit >= num_qubits {
                return Err(SimulatorError::IndexOutOfBounds(qubit));
            }
            pauli_string.operators[qubit] = op;
        }

        Ok(pauli_string)
    }

    /// Set operator at specific qubit
    pub fn set_operator(&mut self, qubit: usize, op: PauliOperator) -> Result<()> {
        if qubit >= self.num_qubits {
            return Err(SimulatorError::IndexOutOfBounds(qubit));
        }
        self.operators[qubit] = op;
        Ok(())
    }

    /// Get operator at specific qubit
    pub fn get_operator(&self, qubit: usize) -> Result<PauliOperator> {
        if qubit >= self.num_qubits {
            return Err(SimulatorError::IndexOutOfBounds(qubit));
        }
        Ok(self.operators[qubit])
    }

    /// Get non-identity operators
    pub fn non_identity_ops(&self) -> Vec<(usize, PauliOperator)> {
        self.operators
            .iter()
            .enumerate()
            .filter(|(_, &op)| op != PauliOperator::I)
            .map(|(i, &op)| (i, op))
            .collect()
    }

    /// Check if this Pauli string commutes with another
    pub fn commutes_with(&self, other: &PauliString) -> bool {
        if self.num_qubits != other.num_qubits {
            return false;
        }

        let mut anti_commute_count = 0;
        for i in 0..self.num_qubits {
            if !self.operators[i].commutes_with(&other.operators[i]) {
                anti_commute_count += 1;
            }
        }

        // Pauli strings commute if they anti-commute at an even number of positions
        anti_commute_count % 2 == 0
    }

    /// Multiply two Pauli strings
    pub fn multiply(&self, other: &PauliString) -> Result<PauliString> {
        if self.num_qubits != other.num_qubits {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Pauli strings have different lengths: {} vs {}",
                self.num_qubits, other.num_qubits
            )));
        }

        let mut result = PauliString::new(self.num_qubits);
        let mut total_phase = self.coefficient * other.coefficient;

        for i in 0..self.num_qubits {
            let (op, phase) = self.operators[i].multiply(&other.operators[i]);
            result.operators[i] = op;
            total_phase *= phase;
        }

        result.coefficient = total_phase;
        Ok(result)
    }

    /// Get weight (number of non-identity operators)
    pub fn weight(&self) -> usize {
        self.operators
            .iter()
            .filter(|&&op| op != PauliOperator::I)
            .count()
    }

    /// Convert to Pauli string representation
    pub fn pauli_string(&self) -> String {
        self.operators.iter().map(|op| op.to_string()).collect()
    }

    /// Create time evolution circuit for this Pauli string
    pub fn evolution_circuit(&self, time: f64) -> Result<DynamicCircuit> {
        let mut circuit = DynamicCircuit::new(self.num_qubits);

        if self.weight() == 0 {
            // Identity operator - no gates needed
            return Ok(circuit);
        }

        let non_identity = self.non_identity_ops();
        let angle = -2.0 * self.coefficient.re * time;

        if non_identity.len() == 1 {
            // Single-qubit Pauli evolution
            let (qubit, op) = non_identity[0];
            match op {
                PauliOperator::X => circuit.add_gate(Box::new(RotationX {
                    target: QubitId::new(qubit as u32),
                    theta: angle,
                }))?,
                PauliOperator::Y => circuit.add_gate(Box::new(RotationY {
                    target: QubitId::new(qubit as u32),
                    theta: angle,
                }))?,
                PauliOperator::Z => circuit.add_gate(Box::new(RotationZ {
                    target: QubitId::new(qubit as u32),
                    theta: angle,
                }))?,
                PauliOperator::I => {} // Should not happen
            }
        } else {
            // Multi-qubit Pauli string evolution

            // Apply basis rotations to convert all non-identity operators to Z
            for &(qubit, op) in &non_identity {
                match op {
                    PauliOperator::X => circuit.add_gate(Box::new(Hadamard {
                        target: QubitId::new(qubit as u32),
                    }))?,
                    PauliOperator::Y => {
                        circuit.add_gate(Box::new(Hadamard {
                            target: QubitId::new(qubit as u32),
                        }))?;
                        circuit.add_gate(Box::new(Phase {
                            target: QubitId::new(qubit as u32),
                        }))?;
                    }
                    PauliOperator::Z => {} // No basis change needed
                    PauliOperator::I => {} // Should not happen
                }
            }

            // Apply CNOT ladder to disentangle all Z operators to the last qubit
            for i in 0..non_identity.len() - 1 {
                circuit.add_gate(Box::new(CNOT {
                    control: QubitId::new(non_identity[i].0 as u32),
                    target: QubitId::new(non_identity[i + 1].0 as u32),
                }))?;
            }

            // Apply Z rotation on the last qubit
            circuit.add_gate(Box::new(RotationZ {
                target: QubitId::new(non_identity[non_identity.len() - 1].0 as u32),
                theta: angle,
            }))?;

            // Reverse CNOT ladder
            for i in (0..non_identity.len() - 1).rev() {
                circuit.add_gate(Box::new(CNOT {
                    control: QubitId::new(non_identity[i].0 as u32),
                    target: QubitId::new(non_identity[i + 1].0 as u32),
                }))?;
            }

            // Reverse basis rotations
            for &(qubit, op) in non_identity.iter().rev() {
                match op {
                    PauliOperator::X => circuit.add_gate(Box::new(Hadamard {
                        target: QubitId::new(qubit as u32),
                    }))?,
                    PauliOperator::Y => {
                        circuit.add_gate(Box::new(PhaseDagger {
                            target: QubitId::new(qubit as u32),
                        }))?;
                        circuit.add_gate(Box::new(Hadamard {
                            target: QubitId::new(qubit as u32),
                        }))?;
                    }
                    PauliOperator::Z => {} // No basis change needed
                    PauliOperator::I => {} // Should not happen
                }
            }
        }

        Ok(circuit)
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}) {}", self.coefficient, self.pauli_string())
    }
}

/// Sum of Pauli strings (Pauli operator)
#[derive(Debug, Clone)]
pub struct PauliOperatorSum {
    /// Individual Pauli string terms
    pub terms: Vec<PauliString>,
    /// Number of qubits
    pub num_qubits: usize,
}

impl PauliOperatorSum {
    /// Create new empty sum
    pub fn new(num_qubits: usize) -> Self {
        Self {
            terms: Vec::new(),
            num_qubits,
        }
    }

    /// Add a Pauli string term
    pub fn add_term(&mut self, pauli_string: PauliString) -> Result<()> {
        if pauli_string.num_qubits != self.num_qubits {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Pauli string has {} qubits, expected {}",
                pauli_string.num_qubits, self.num_qubits
            )));
        }
        self.terms.push(pauli_string);
        Ok(())
    }

    /// Combine like terms
    pub fn simplify(&mut self) {
        let mut simplified_terms: HashMap<String, Complex64> = HashMap::new();

        for term in &self.terms {
            let key = format!("{}", term);
            *simplified_terms
                .entry(key)
                .or_insert(Complex64::new(0.0, 0.0)) += term.coefficient;
        }

        self.terms.clear();
        for (pauli_str, coeff) in simplified_terms {
            if coeff.norm() > 1e-15 {
                if let Ok(term) = PauliString::from_string(&pauli_str, coeff) {
                    self.terms.push(term);
                }
            }
        }
    }

    /// Convert to Hamiltonian for Trotter evolution
    pub fn to_hamiltonian(&self) -> Hamiltonian {
        let mut ham = Hamiltonian::new(self.num_qubits);

        for term in &self.terms {
            let non_identity = term.non_identity_ops();

            match non_identity.len() {
                0 => {} // Identity term - ignore for Hamiltonian
                1 => {
                    let (qubit, op) = non_identity[0];
                    let pauli_str = match op {
                        PauliOperator::X => "X",
                        PauliOperator::Y => "Y",
                        PauliOperator::Z => "Z",
                        PauliOperator::I => continue,
                    };
                    let _ = ham.add_single_pauli(qubit, pauli_str, term.coefficient.re);
                }
                2 => {
                    let (q1, op1) = non_identity[0];
                    let (q2, op2) = non_identity[1];
                    let pauli1 = match op1 {
                        PauliOperator::X => "X",
                        PauliOperator::Y => "Y",
                        PauliOperator::Z => "Z",
                        PauliOperator::I => continue,
                    };
                    let pauli2 = match op2 {
                        PauliOperator::X => "X",
                        PauliOperator::Y => "Y",
                        PauliOperator::Z => "Z",
                        PauliOperator::I => continue,
                    };
                    let _ = ham.add_two_pauli(q1, q2, pauli1, pauli2, term.coefficient.re);
                }
                _ => {
                    // Multi-qubit case
                    let qubits: Vec<usize> = non_identity.iter().map(|&(q, _)| q).collect();
                    let paulis: Vec<String> = non_identity
                        .iter()
                        .map(|&(_, op)| match op {
                            PauliOperator::X => "X".to_string(),
                            PauliOperator::Y => "Y".to_string(),
                            PauliOperator::Z => "Z".to_string(),
                            PauliOperator::I => "I".to_string(),
                        })
                        .collect();
                    let _ = ham.add_pauli_string(qubits, paulis, term.coefficient.re);
                }
            }
        }

        ham
    }

    /// Time evolution using Trotter decomposition
    pub fn time_evolution_circuit(
        &self,
        time: f64,
        trotter_steps: usize,
        method: TrotterMethod,
    ) -> Result<DynamicCircuit> {
        let hamiltonian = self.to_hamiltonian();
        let decomposer = TrotterDecomposer::new(method, trotter_steps);
        decomposer.decompose(&hamiltonian, time)
    }

    /// Direct time evolution without Trotter approximation (for single terms)
    pub fn exact_evolution_circuit(&self, time: f64) -> Result<DynamicCircuit> {
        let mut circuit = DynamicCircuit::new(self.num_qubits);

        for term in &self.terms {
            let term_circuit = term.evolution_circuit(time)?;
            for gate in term_circuit.gates() {
                circuit.add_gate(gate.clone())?;
            }
        }

        Ok(circuit)
    }
}

impl fmt::Display for PauliOperatorSum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            write!(f, "0")
        } else {
            let term_strs: Vec<String> = self.terms.iter().map(|t| format!("{}", t)).collect();
            write!(f, "{}", term_strs.join(" + "))
        }
    }
}

/// Utilities for common Pauli string operations
pub struct PauliUtils;

impl PauliUtils {
    /// Create a single-qubit Pauli string
    pub fn single_qubit(
        num_qubits: usize,
        qubit: usize,
        op: PauliOperator,
        coeff: Complex64,
    ) -> Result<PauliString> {
        PauliString::from_ops(num_qubits, &[(qubit, op)], coeff)
    }

    /// Create an all-X Pauli string
    pub fn all_x(num_qubits: usize, coeff: Complex64) -> PauliString {
        let mut pauli = PauliString::new(num_qubits);
        pauli.coefficient = coeff;
        for i in 0..num_qubits {
            pauli.operators[i] = PauliOperator::X;
        }
        pauli
    }

    /// Create an all-Z Pauli string
    pub fn all_z(num_qubits: usize, coeff: Complex64) -> PauliString {
        let mut pauli = PauliString::new(num_qubits);
        pauli.coefficient = coeff;
        for i in 0..num_qubits {
            pauli.operators[i] = PauliOperator::Z;
        }
        pauli
    }

    /// Create random Pauli string
    pub fn random(num_qubits: usize, weight: usize, coeff: Complex64) -> Result<PauliString> {
        if weight > num_qubits {
            return Err(SimulatorError::InvalidInput(
                "Weight cannot exceed number of qubits".to_string(),
            ));
        }

        let mut pauli = PauliString::new(num_qubits);
        pauli.coefficient = coeff;

        // Randomly select positions for non-identity operators
        let mut positions: Vec<usize> = (0..num_qubits).collect();
        fastrand::shuffle(&mut positions);

        let ops = [PauliOperator::X, PauliOperator::Y, PauliOperator::Z];

        for i in 0..weight {
            pauli.operators[positions[i]] = ops[fastrand::usize(0..3)];
        }

        Ok(pauli)
    }

    /// Check if a set of Pauli strings are mutually commuting
    pub fn are_mutually_commuting(pauli_strings: &[PauliString]) -> bool {
        for i in 0..pauli_strings.len() {
            for j in i + 1..pauli_strings.len() {
                if !pauli_strings[i].commutes_with(&pauli_strings[j]) {
                    return false;
                }
            }
        }
        true
    }

    /// Find maximal set of mutually commuting Pauli strings
    pub fn maximal_commuting_set(pauli_strings: &[PauliString]) -> Vec<usize> {
        let mut commuting_set = Vec::new();

        for (i, pauli) in pauli_strings.iter().enumerate() {
            let mut can_add = true;
            for &j in &commuting_set {
                if !pauli.commutes_with(&pauli_strings[j]) {
                    can_add = false;
                    break;
                }
            }

            if can_add {
                commuting_set.push(i);
            }
        }

        commuting_set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_operator_multiply() {
        let (result, phase) = PauliOperator::X.multiply(&PauliOperator::Y);
        assert_eq!(result, PauliOperator::Z);
        assert_eq!(phase, Complex64::new(0.0, 1.0));

        let (result, phase) = PauliOperator::Y.multiply(&PauliOperator::X);
        assert_eq!(result, PauliOperator::Z);
        assert_eq!(phase, Complex64::new(0.0, -1.0));
    }

    #[test]
    fn test_pauli_string_creation() {
        let pauli = PauliString::from_string("XYZ", Complex64::new(1.0, 0.0)).unwrap();
        assert_eq!(pauli.num_qubits, 3);
        assert_eq!(pauli.operators[0], PauliOperator::X);
        assert_eq!(pauli.operators[1], PauliOperator::Y);
        assert_eq!(pauli.operators[2], PauliOperator::Z);
    }

    #[test]
    fn test_pauli_string_multiply() {
        let p1 = PauliString::from_string("XY", Complex64::new(1.0, 0.0)).unwrap();
        let p2 = PauliString::from_string("YZ", Complex64::new(1.0, 0.0)).unwrap();

        let result = p1.multiply(&p2).unwrap();
        assert_eq!(result.operators[0], PauliOperator::Z);
        assert_eq!(result.operators[1], PauliOperator::X);
        assert_eq!(result.coefficient, Complex64::new(-1.0, 0.0));
    }

    #[test]
    fn test_pauli_string_commutation() {
        let p1 = PauliString::from_string("XY", Complex64::new(1.0, 0.0)).unwrap();
        let p2 = PauliString::from_string("ZI", Complex64::new(1.0, 0.0)).unwrap();
        let p3 = PauliString::from_string("XI", Complex64::new(1.0, 0.0)).unwrap();

        assert!(!p1.commutes_with(&p2)); // XY and ZI anti-commute at first qubit
        assert!(p1.commutes_with(&p3)); // XY and XI commute
    }

    #[test]
    fn test_pauli_string_weight() {
        let p1 = PauliString::from_string("XIYZ", Complex64::new(1.0, 0.0)).unwrap();
        assert_eq!(p1.weight(), 3);

        let p2 = PauliString::from_string("IIII", Complex64::new(1.0, 0.0)).unwrap();
        assert_eq!(p2.weight(), 0);
    }

    #[test]
    fn test_pauli_operator_sum() {
        let mut sum = PauliOperatorSum::new(2);

        let p1 = PauliString::from_string("XX", Complex64::new(1.0, 0.0)).unwrap();
        let p2 = PauliString::from_string("YY", Complex64::new(0.5, 0.0)).unwrap();

        sum.add_term(p1).unwrap();
        sum.add_term(p2).unwrap();

        assert_eq!(sum.terms.len(), 2);
    }

    #[test]
    fn test_evolution_circuit_single_qubit() {
        let pauli = PauliString::from_string("X", Complex64::new(1.0, 0.0)).unwrap();
        let circuit = pauli.evolution_circuit(1.0).unwrap();
        assert!(circuit.gate_count() > 0);
    }

    #[test]
    fn test_evolution_circuit_multi_qubit() {
        let pauli = PauliString::from_string("XYZ", Complex64::new(1.0, 0.0)).unwrap();
        let circuit = pauli.evolution_circuit(1.0).unwrap();
        assert!(circuit.gate_count() > 0);
    }

    #[test]
    fn test_utils_single_qubit() {
        let pauli =
            PauliUtils::single_qubit(3, 1, PauliOperator::X, Complex64::new(1.0, 0.0)).unwrap();
        assert_eq!(pauli.operators[0], PauliOperator::I);
        assert_eq!(pauli.operators[1], PauliOperator::X);
        assert_eq!(pauli.operators[2], PauliOperator::I);
    }

    #[test]
    fn test_mutually_commuting() {
        let p1 = PauliString::from_string("XX", Complex64::new(1.0, 0.0)).unwrap();
        let p2 = PauliString::from_string("ZZ", Complex64::new(1.0, 0.0)).unwrap();
        let p3 = PauliString::from_string("YY", Complex64::new(1.0, 0.0)).unwrap();

        let paulis = vec![p1, p2, p3];
        assert!(PauliUtils::are_mutually_commuting(&paulis));
    }
}
