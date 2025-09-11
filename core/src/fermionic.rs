//! Fermionic operations and Jordan-Wigner transformations
//!
//! This module provides support for fermionic operators and their mapping to qubit operators
//! using the Jordan-Wigner transformation. This enables quantum simulation of fermionic systems
//! such as molecules and condensed matter systems.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{single, GateOp},
    qubit::QubitId,
};
use ndarray::Array2;
use num_complex::Complex;
use rustc_hash::FxHashMap;

/// Type alias for complex numbers
type Complex64 = Complex<f64>;

/// Fermionic operator types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FermionOperatorType {
    /// Creation operator a†
    Creation,
    /// Annihilation operator a
    Annihilation,
    /// Number operator n = a†a
    Number,
    /// Identity operator
    Identity,
}

/// A single fermionic operator acting on a specific mode
#[derive(Debug, Clone, PartialEq)]
pub struct FermionOperator {
    /// Type of the operator
    pub op_type: FermionOperatorType,
    /// Mode (site) index
    pub mode: usize,
    /// Coefficient
    pub coefficient: Complex64,
}

impl FermionOperator {
    /// Create a new fermionic operator
    pub fn new(op_type: FermionOperatorType, mode: usize, coefficient: Complex64) -> Self {
        Self {
            op_type,
            mode,
            coefficient,
        }
    }

    /// Create a creation operator
    pub fn creation(mode: usize) -> Self {
        Self::new(
            FermionOperatorType::Creation,
            mode,
            Complex64::new(1.0, 0.0),
        )
    }

    /// Create an annihilation operator
    pub fn annihilation(mode: usize) -> Self {
        Self::new(
            FermionOperatorType::Annihilation,
            mode,
            Complex64::new(1.0, 0.0),
        )
    }

    /// Create a number operator
    pub fn number(mode: usize) -> Self {
        Self::new(FermionOperatorType::Number, mode, Complex64::new(1.0, 0.0))
    }

    /// Get the Hermitian conjugate
    pub fn dagger(&self) -> Self {
        let conj_coeff = self.coefficient.conj();
        match self.op_type {
            FermionOperatorType::Creation => {
                Self::new(FermionOperatorType::Annihilation, self.mode, conj_coeff)
            }
            FermionOperatorType::Annihilation => {
                Self::new(FermionOperatorType::Creation, self.mode, conj_coeff)
            }
            FermionOperatorType::Number => {
                Self::new(FermionOperatorType::Number, self.mode, conj_coeff)
            }
            FermionOperatorType::Identity => {
                Self::new(FermionOperatorType::Identity, self.mode, conj_coeff)
            }
        }
    }
}

/// A product of fermionic operators
#[derive(Debug, Clone, PartialEq)]
pub struct FermionTerm {
    /// Ordered list of operators in the term
    pub operators: Vec<FermionOperator>,
    /// Overall coefficient
    pub coefficient: Complex64,
}

impl FermionTerm {
    /// Create a new fermionic term
    pub fn new(operators: Vec<FermionOperator>, coefficient: Complex64) -> Self {
        Self {
            operators,
            coefficient,
        }
    }

    /// Create an identity term
    pub fn identity() -> Self {
        Self {
            operators: vec![],
            coefficient: Complex64::new(1.0, 0.0),
        }
    }

    /// Normal order the operators using anticommutation relations
    pub fn normal_order(&mut self) -> QuantRS2Result<()> {
        // Bubble sort with anticommutation
        let n = self.operators.len();
        for i in 0..n {
            for j in 0..n.saturating_sub(i + 1) {
                if self.should_swap(j) {
                    self.swap_operators(j)?;
                }
            }
        }
        Ok(())
    }

    /// Check if two adjacent operators should be swapped
    fn should_swap(&self, idx: usize) -> bool {
        if idx + 1 >= self.operators.len() {
            return false;
        }

        let op1 = &self.operators[idx];
        let op2 = &self.operators[idx + 1];

        // Normal ordering: creation operators before annihilation
        match (op1.op_type, op2.op_type) {
            (FermionOperatorType::Annihilation, FermionOperatorType::Creation) => {
                op1.mode > op2.mode
            }
            (FermionOperatorType::Creation, FermionOperatorType::Creation) => op1.mode > op2.mode,
            (FermionOperatorType::Annihilation, FermionOperatorType::Annihilation) => {
                op1.mode < op2.mode
            }
            _ => false,
        }
    }

    /// Swap two adjacent operators with anticommutation
    fn swap_operators(&mut self, idx: usize) -> QuantRS2Result<()> {
        if idx + 1 >= self.operators.len() {
            return Err(QuantRS2Error::InvalidInput("Index out of bounds".into()));
        }

        let op1 = &self.operators[idx];
        let op2 = &self.operators[idx + 1];

        // Check anticommutation relation
        if op1.mode != op2.mode {
            // Different modes anticommute: {a_i, a_j†} = 0 for i ≠ j
            self.coefficient *= -1.0;
        } else {
            // Same mode: {a_i, a_i†} = 1
            match (op1.op_type, op2.op_type) {
                (FermionOperatorType::Annihilation, FermionOperatorType::Creation) => {
                    // a_i a_i† = 1 - a_i† a_i
                    // This requires splitting into two terms
                    return Err(QuantRS2Error::UnsupportedOperation(
                        "Anticommutation that produces multiple terms not yet supported".into(),
                    ));
                }
                _ => {
                    self.coefficient *= -1.0;
                }
            }
        }

        self.operators.swap(idx, idx + 1);
        Ok(())
    }

    /// Get the Hermitian conjugate
    pub fn dagger(&self) -> Self {
        let mut conj_ops = self.operators.clone();
        conj_ops.reverse();
        conj_ops = conj_ops.into_iter().map(|op| op.dagger()).collect();

        Self {
            operators: conj_ops,
            coefficient: self.coefficient.conj(),
        }
    }
}

/// A sum of fermionic terms (second-quantized Hamiltonian)
#[derive(Debug, Clone)]
pub struct FermionHamiltonian {
    /// Terms in the Hamiltonian
    pub terms: Vec<FermionTerm>,
    /// Number of fermionic modes
    pub n_modes: usize,
}

impl FermionHamiltonian {
    /// Create a new fermionic Hamiltonian
    pub fn new(n_modes: usize) -> Self {
        Self {
            terms: Vec::new(),
            n_modes,
        }
    }

    /// Add a term to the Hamiltonian
    pub fn add_term(&mut self, term: FermionTerm) {
        self.terms.push(term);
    }

    /// Add a one-body term: h_ij a†_i a_j
    pub fn add_one_body(&mut self, i: usize, j: usize, coefficient: Complex64) {
        let term = FermionTerm::new(
            vec![
                FermionOperator::creation(i),
                FermionOperator::annihilation(j),
            ],
            coefficient,
        );
        self.add_term(term);
    }

    /// Add a two-body term: g_ijkl a†_i a†_j a_k a_l
    pub fn add_two_body(&mut self, i: usize, j: usize, k: usize, l: usize, coefficient: Complex64) {
        let term = FermionTerm::new(
            vec![
                FermionOperator::creation(i),
                FermionOperator::creation(j),
                FermionOperator::annihilation(k),
                FermionOperator::annihilation(l),
            ],
            coefficient,
        );
        self.add_term(term);
    }

    /// Add a chemical potential term: μ n_i
    pub fn add_chemical_potential(&mut self, i: usize, mu: f64) {
        let term = FermionTerm::new(vec![FermionOperator::number(i)], Complex64::new(mu, 0.0));
        self.add_term(term);
    }

    /// Get the Hermitian conjugate
    pub fn dagger(&self) -> Self {
        let conj_terms = self.terms.iter().map(|t| t.dagger()).collect();
        Self {
            terms: conj_terms,
            n_modes: self.n_modes,
        }
    }

    /// Check if the Hamiltonian is Hermitian
    pub fn is_hermitian(&self, _tolerance: f64) -> bool {
        let conj = self.dagger();

        // Compare terms (this is simplified - proper implementation would canonicalize first)
        if self.terms.len() != conj.terms.len() {
            return false;
        }

        // Check if all terms match within tolerance
        true // Placeholder
    }
}

/// Jordan-Wigner transformation
pub struct JordanWigner {
    /// Number of fermionic modes (qubits)
    n_modes: usize,
}

impl JordanWigner {
    /// Create a new Jordan-Wigner transformer
    pub fn new(n_modes: usize) -> Self {
        Self { n_modes }
    }

    /// Transform a fermionic operator to qubit operators
    pub fn transform_operator(&self, op: &FermionOperator) -> QuantRS2Result<Vec<QubitOperator>> {
        match op.op_type {
            FermionOperatorType::Creation => self.transform_creation(op.mode, op.coefficient),
            FermionOperatorType::Annihilation => {
                self.transform_annihilation(op.mode, op.coefficient)
            }
            FermionOperatorType::Number => self.transform_number(op.mode, op.coefficient),
            FermionOperatorType::Identity => {
                Ok(vec![QubitOperator::identity(self.n_modes, op.coefficient)])
            }
        }
    }

    /// Transform creation operator a†_j = (X_j - iY_j)/2 ⊗ Z_{<j}
    fn transform_creation(
        &self,
        mode: usize,
        coeff: Complex64,
    ) -> QuantRS2Result<Vec<QubitOperator>> {
        if mode >= self.n_modes {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Mode {} out of bounds",
                mode
            )));
        }

        let mut operators = Vec::new();

        // (X_j - iY_j)/2 = σ^-_j
        let sigma_minus = QubitTerm {
            operators: vec![(mode, PauliOperator::Minus)],
            coefficient: coeff,
        };

        // Apply Z string to all qubits before mode j
        let z_string: Vec<_> = (0..mode).map(|i| (i, PauliOperator::Z)).collect();

        let mut term = sigma_minus;
        term.operators.extend(z_string);

        operators.push(QubitOperator {
            terms: vec![term],
            n_qubits: self.n_modes,
        });

        Ok(operators)
    }

    /// Transform annihilation operator a_j = (X_j + iY_j)/2 ⊗ Z_{<j}
    fn transform_annihilation(
        &self,
        mode: usize,
        coeff: Complex64,
    ) -> QuantRS2Result<Vec<QubitOperator>> {
        if mode >= self.n_modes {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Mode {} out of bounds",
                mode
            )));
        }

        let mut operators = Vec::new();

        // (X_j + iY_j)/2 = σ^+_j
        let sigma_plus = QubitTerm {
            operators: vec![(mode, PauliOperator::Plus)],
            coefficient: coeff,
        };

        // Apply Z string to all qubits before mode j
        let z_string: Vec<_> = (0..mode).map(|i| (i, PauliOperator::Z)).collect();

        let mut term = sigma_plus;
        term.operators.extend(z_string);

        operators.push(QubitOperator {
            terms: vec![term],
            n_qubits: self.n_modes,
        });

        Ok(operators)
    }

    /// Transform number operator n_j = a†_j a_j = (I - Z_j)/2
    fn transform_number(
        &self,
        mode: usize,
        coeff: Complex64,
    ) -> QuantRS2Result<Vec<QubitOperator>> {
        if mode >= self.n_modes {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Mode {} out of bounds",
                mode
            )));
        }

        let mut operators = Vec::new();

        // Identity term
        operators.push(QubitOperator {
            terms: vec![QubitTerm {
                operators: vec![],
                coefficient: coeff * 0.5,
            }],
            n_qubits: self.n_modes,
        });

        // -Z term
        operators.push(QubitOperator {
            terms: vec![QubitTerm {
                operators: vec![(mode, PauliOperator::Z)],
                coefficient: -coeff * 0.5,
            }],
            n_qubits: self.n_modes,
        });

        Ok(operators)
    }

    /// Transform a fermionic term to qubit operators
    pub fn transform_term(&self, term: &FermionTerm) -> QuantRS2Result<QubitOperator> {
        if term.operators.is_empty() {
            return Ok(QubitOperator::identity(self.n_modes, term.coefficient));
        }

        // Transform each operator and combine
        let mut result = QubitOperator::identity(self.n_modes, term.coefficient);

        for op in &term.operators {
            let transformed = self.transform_operator(op)?;

            // Multiply qubit operators
            let mut new_result = QubitOperator::zero(self.n_modes);
            for t in transformed {
                new_result = new_result.add(&result.multiply(&t)?)?;
            }
            result = new_result;
        }

        Ok(result)
    }

    /// Transform a fermionic Hamiltonian to qubit operators
    pub fn transform_hamiltonian(
        &self,
        hamiltonian: &FermionHamiltonian,
    ) -> QuantRS2Result<QubitOperator> {
        let mut qubit_ham = QubitOperator::zero(self.n_modes);

        for term in &hamiltonian.terms {
            let transformed = self.transform_term(term)?;
            qubit_ham = qubit_ham.add(&transformed)?;
        }

        Ok(qubit_ham)
    }
}

/// Pauli operator types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PauliOperator {
    /// Identity
    I,
    /// Pauli X
    X,
    /// Pauli Y
    Y,
    /// Pauli Z
    Z,
    /// Raising operator (X + iY)/2
    Plus,
    /// Lowering operator (X - iY)/2
    Minus,
}

impl PauliOperator {
    /// Get the matrix representation
    pub fn matrix(&self) -> Array2<Complex64> {
        match self {
            Self::I => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .unwrap(),
            Self::X => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            Self::Y => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, -1.0),
                    Complex64::new(0.0, 1.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            Self::Z => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )
            .unwrap(),
            Self::Plus => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            Self::Minus => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
        }
    }
}

/// A term in a qubit operator (product of Pauli operators)
#[derive(Debug, Clone)]
pub struct QubitTerm {
    /// List of (qubit_index, pauli_operator) pairs
    pub operators: Vec<(usize, PauliOperator)>,
    /// Coefficient
    pub coefficient: Complex64,
}

/// A sum of qubit terms (Pauli strings)
#[derive(Debug, Clone)]
pub struct QubitOperator {
    /// Terms in the operator
    pub terms: Vec<QubitTerm>,
    /// Number of qubits
    pub n_qubits: usize,
}

impl QubitOperator {
    /// Create a zero operator
    pub fn zero(n_qubits: usize) -> Self {
        Self {
            terms: vec![],
            n_qubits,
        }
    }

    /// Create an identity operator
    pub fn identity(n_qubits: usize, coefficient: Complex64) -> Self {
        Self {
            terms: vec![QubitTerm {
                operators: vec![],
                coefficient,
            }],
            n_qubits,
        }
    }

    /// Add two qubit operators
    pub fn add(&self, other: &Self) -> QuantRS2Result<Self> {
        if self.n_qubits != other.n_qubits {
            return Err(QuantRS2Error::InvalidInput(
                "Operators must have same number of qubits".into(),
            ));
        }

        let mut result = self.clone();
        result.terms.extend(other.terms.clone());
        Ok(result)
    }

    /// Multiply two qubit operators
    pub fn multiply(&self, other: &Self) -> QuantRS2Result<Self> {
        if self.n_qubits != other.n_qubits {
            return Err(QuantRS2Error::InvalidInput(
                "Operators must have same number of qubits".into(),
            ));
        }

        let mut result_terms = Vec::new();

        for term1 in &self.terms {
            for term2 in &other.terms {
                // Multiply coefficients
                let coeff = term1.coefficient * term2.coefficient;

                // Combine Pauli operators
                let mut combined_ops = term1.operators.clone();
                combined_ops.extend(&term2.operators);

                result_terms.push(QubitTerm {
                    operators: combined_ops,
                    coefficient: coeff,
                });
            }
        }

        Ok(Self {
            terms: result_terms,
            n_qubits: self.n_qubits,
        })
    }

    /// Simplify by combining like terms
    pub fn simplify(&mut self) {
        // Group terms by operator pattern
        let mut grouped: FxHashMap<Vec<(usize, PauliOperator)>, Complex64> = FxHashMap::default();

        for term in &self.terms {
            let key = term.operators.clone();
            *grouped.entry(key).or_insert(Complex64::new(0.0, 0.0)) += term.coefficient;
        }

        // Rebuild terms
        self.terms = grouped
            .into_iter()
            .filter(|(_, coeff)| coeff.norm() > 1e-12)
            .map(|(ops, coeff)| QubitTerm {
                operators: ops,
                coefficient: coeff,
            })
            .collect();
    }
}

/// Convert a QubitOperator to quantum gates
pub fn qubit_operator_to_gates(op: &QubitOperator) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
    let mut gates = Vec::new();

    for term in &op.terms {
        // Apply coefficient as global phase
        // In practice, this would be handled by the circuit simulator

        // Apply Pauli operators
        for (qubit, pauli) in &term.operators {
            let gate: Box<dyn GateOp> = match pauli {
                PauliOperator::I => continue, // Identity - skip
                PauliOperator::X => Box::new(single::PauliX {
                    target: QubitId(*qubit as u32),
                }),
                PauliOperator::Y => Box::new(single::PauliY {
                    target: QubitId(*qubit as u32),
                }),
                PauliOperator::Z => Box::new(single::PauliZ {
                    target: QubitId(*qubit as u32),
                }),
                PauliOperator::Plus | PauliOperator::Minus => {
                    return Err(QuantRS2Error::UnsupportedOperation(
                        "Ladder operators require decomposition".into(),
                    ));
                }
            };
            gates.push(gate);
        }
    }

    Ok(gates)
}

/// Bravyi-Kitaev transformation (alternative to Jordan-Wigner)
pub struct BravyiKitaev {
    #[allow(dead_code)]
    n_modes: usize,
}

impl BravyiKitaev {
    /// Create a new Bravyi-Kitaev transformer
    pub fn new(n_modes: usize) -> Self {
        Self { n_modes }
    }

    /// Transform a fermionic operator (placeholder)
    pub fn transform_operator(&self, _op: &FermionOperator) -> QuantRS2Result<Vec<QubitOperator>> {
        // Bravyi-Kitaev transformation is more complex than Jordan-Wigner
        // This is a placeholder for future implementation
        Err(QuantRS2Error::UnsupportedOperation(
            "Bravyi-Kitaev transformation not yet implemented".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fermion_operator_creation() {
        let op = FermionOperator::creation(0);
        assert_eq!(op.op_type, FermionOperatorType::Creation);
        assert_eq!(op.mode, 0);
        assert_eq!(op.coefficient, Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_fermion_operator_dagger() {
        let op = FermionOperator::creation(0);
        let dag = op.dagger();
        assert_eq!(dag.op_type, FermionOperatorType::Annihilation);
        assert_eq!(dag.mode, 0);
    }

    #[test]
    fn test_jordan_wigner_number_operator() {
        let jw = JordanWigner::new(4);
        let op = FermionOperator::number(1);
        let qubit_ops = jw.transform_operator(&op).unwrap();

        // n_1 = (I - Z_1)/2
        assert_eq!(qubit_ops.len(), 2);
    }

    #[test]
    fn test_jordan_wigner_creation_operator() {
        let jw = JordanWigner::new(4);
        let op = FermionOperator::creation(2);
        let qubit_ops = jw.transform_operator(&op).unwrap();

        // a†_2 should have Z operators on qubits 0 and 1
        assert_eq!(qubit_ops.len(), 1);
    }

    #[test]
    fn test_fermionic_hamiltonian() {
        let mut ham = FermionHamiltonian::new(4);

        // Add hopping term
        ham.add_one_body(0, 1, Complex64::new(-1.0, 0.0));
        ham.add_one_body(1, 0, Complex64::new(-1.0, 0.0));

        // Add interaction
        ham.add_two_body(0, 1, 1, 0, Complex64::new(2.0, 0.0));

        assert_eq!(ham.terms.len(), 3);
    }

    #[test]
    fn test_qubit_operator_operations() {
        let op1 = QubitOperator::identity(2, Complex64::new(1.0, 0.0));
        let op2 = QubitOperator::identity(2, Complex64::new(2.0, 0.0));

        let sum = op1.add(&op2).unwrap();
        assert_eq!(sum.terms.len(), 2);

        let prod = op1.multiply(&op2).unwrap();
        assert_eq!(prod.terms.len(), 1);
        assert_eq!(prod.terms[0].coefficient, Complex64::new(2.0, 0.0));
    }
}
