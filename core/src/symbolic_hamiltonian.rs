//! Symbolic Hamiltonian construction and manipulation module
//!
//! This module provides tools for constructing and manipulating quantum Hamiltonians
//! using symbolic expressions, enabling automatic differentiation, parameter optimization,
//! and analytical calculations for quantum algorithms like VQE and QAOA.

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::parametric::Parameter;
use crate::qubit::QubitId;
use crate::symbolic::SymbolicExpression;
use num_complex::Complex64;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Pauli operator types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PauliOperator {
    I, // Identity
    X, // Pauli-X
    Y, // Pauli-Y
    Z, // Pauli-Z
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

/// A Pauli string representing a tensor product of Pauli operators
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PauliString {
    /// Map from qubit index to Pauli operator
    pub operators: HashMap<QubitId, PauliOperator>,
    /// Number of qubits this string acts on
    pub n_qubits: usize,
}

impl Hash for PauliString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.n_qubits.hash(state);
        // Hash the operators by sorting them to ensure consistent ordering
        let mut sorted_ops: Vec<_> = self.operators.iter().collect();
        sorted_ops.sort_by_key(|(qubit, _)| qubit.id());
        for (qubit, op) in sorted_ops {
            qubit.hash(state);
            op.hash(state);
        }
    }
}

impl PauliString {
    /// Create a new Pauli string
    pub fn new(n_qubits: usize) -> Self {
        PauliString {
            operators: HashMap::new(),
            n_qubits,
        }
    }

    /// Create an identity string
    pub fn identity(n_qubits: usize) -> Self {
        PauliString::new(n_qubits)
    }

    /// Add a Pauli operator at a specific qubit
    pub fn with_operator(mut self, qubit: QubitId, op: PauliOperator) -> Self {
        if op != PauliOperator::I {
            self.operators.insert(qubit, op);
        }
        self
    }

    /// Get the operator at a specific qubit (returns I if not specified)
    pub fn get_operator(&self, qubit: QubitId) -> PauliOperator {
        self.operators
            .get(&qubit)
            .copied()
            .unwrap_or(PauliOperator::I)
    }

    /// Check if this is the identity string
    pub fn is_identity(&self) -> bool {
        self.operators.is_empty()
    }

    /// Get the weight (number of non-identity operators) of this string
    pub fn weight(&self) -> usize {
        self.operators.len()
    }

    /// Multiply two Pauli strings
    pub fn multiply(&self, other: &PauliString) -> (Complex64, PauliString) {
        if self.n_qubits != other.n_qubits {
            panic!("Cannot multiply Pauli strings of different sizes");
        }

        let mut result = PauliString::new(self.n_qubits);
        let mut phase = Complex64::new(1.0, 0.0);

        // Collect all qubits that have operators in either string
        let mut all_qubits: std::collections::HashSet<QubitId> =
            self.operators.keys().copied().collect();
        all_qubits.extend(other.operators.keys());

        for qubit in all_qubits {
            let op1 = self.get_operator(qubit);
            let op2 = other.get_operator(qubit);
            let (local_phase, result_op) = multiply_pauli_operators(op1, op2);
            phase *= local_phase;

            if result_op != PauliOperator::I {
                result.operators.insert(qubit, result_op);
            }
        }

        (phase, result)
    }

    /// Commute two Pauli strings (returns true if they commute)
    pub fn commutes_with(&self, other: &PauliString) -> bool {
        if self.n_qubits != other.n_qubits {
            return false;
        }

        let mut anticommuting_count = 0;

        for qubit in 0..self.n_qubits {
            let op1 = self.get_operator(QubitId::from(qubit));
            let op2 = other.get_operator(QubitId::from(qubit));

            if !pauli_operators_commute(op1, op2) {
                anticommuting_count += 1;
            }
        }

        anticommuting_count % 2 == 0
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            return write!(f, "I");
        }

        let mut terms = Vec::new();
        for qubit in 0..self.n_qubits {
            if let Some(op) = self.operators.get(&QubitId::from(qubit)) {
                terms.push(format!("{}{}", op, qubit));
            }
        }

        if terms.is_empty() {
            write!(f, "I")
        } else {
            write!(f, "{}", terms.join("⊗"))
        }
    }
}

/// A term in a symbolic Hamiltonian
#[derive(Debug, Clone)]
pub struct SymbolicHamiltonianTerm {
    /// Coefficient (can be symbolic)
    pub coefficient: Parameter,
    /// Pauli string
    pub pauli_string: PauliString,
}

impl SymbolicHamiltonianTerm {
    /// Create a new term
    pub fn new(coefficient: Parameter, pauli_string: PauliString) -> Self {
        SymbolicHamiltonianTerm {
            coefficient,
            pauli_string,
        }
    }

    /// Evaluate the coefficient with given variable values
    pub fn evaluate_coefficient(&self, variables: &HashMap<String, f64>) -> QuantRS2Result<f64> {
        self.coefficient.evaluate(variables)
    }

    /// Get all variable names in the coefficient
    pub fn variables(&self) -> Vec<String> {
        self.coefficient.variables()
    }

    /// Multiply by a scalar
    pub fn scale(&self, scalar: Parameter) -> Self {
        SymbolicHamiltonianTerm {
            coefficient: self.coefficient.clone() * scalar,
            pauli_string: self.pauli_string.clone(),
        }
    }
}

impl fmt::Display for SymbolicHamiltonianTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.pauli_string.is_identity() {
            write!(f, "{}", self.coefficient.to_symbolic_expression())
        } else {
            write!(
                f,
                "{} * {}",
                self.coefficient.to_symbolic_expression(),
                self.pauli_string
            )
        }
    }
}

/// A symbolic quantum Hamiltonian
#[derive(Debug, Clone)]
pub struct SymbolicHamiltonian {
    /// Terms in the Hamiltonian
    pub terms: Vec<SymbolicHamiltonianTerm>,
    /// Number of qubits
    pub n_qubits: usize,
}

impl SymbolicHamiltonian {
    /// Create a new empty Hamiltonian
    pub fn new(n_qubits: usize) -> Self {
        SymbolicHamiltonian {
            terms: Vec::new(),
            n_qubits,
        }
    }

    /// Create a zero Hamiltonian
    pub fn zero(n_qubits: usize) -> Self {
        SymbolicHamiltonian::new(n_qubits)
    }

    /// Create an identity Hamiltonian
    pub fn identity(n_qubits: usize) -> Self {
        let mut hamiltonian = SymbolicHamiltonian::new(n_qubits);
        hamiltonian.add_term(Parameter::constant(1.0), PauliString::identity(n_qubits));
        hamiltonian
    }

    /// Add a term to the Hamiltonian
    pub fn add_term(&mut self, coefficient: Parameter, pauli_string: PauliString) {
        if pauli_string.n_qubits != self.n_qubits {
            panic!("Pauli string size mismatch");
        }
        self.terms
            .push(SymbolicHamiltonianTerm::new(coefficient, pauli_string));
    }

    /// Add a Pauli-X term at a specific qubit
    pub fn add_x(&mut self, qubit: QubitId, coefficient: Parameter) {
        let pauli_string = PauliString::new(self.n_qubits).with_operator(qubit, PauliOperator::X);
        self.add_term(coefficient, pauli_string);
    }

    /// Add a Pauli-Y term at a specific qubit
    pub fn add_y(&mut self, qubit: QubitId, coefficient: Parameter) {
        let pauli_string = PauliString::new(self.n_qubits).with_operator(qubit, PauliOperator::Y);
        self.add_term(coefficient, pauli_string);
    }

    /// Add a Pauli-Z term at a specific qubit
    pub fn add_z(&mut self, qubit: QubitId, coefficient: Parameter) {
        let pauli_string = PauliString::new(self.n_qubits).with_operator(qubit, PauliOperator::Z);
        self.add_term(coefficient, pauli_string);
    }

    /// Add a ZZ interaction term between two qubits
    pub fn add_zz(&mut self, qubit1: QubitId, qubit2: QubitId, coefficient: Parameter) {
        let pauli_string = PauliString::new(self.n_qubits)
            .with_operator(qubit1, PauliOperator::Z)
            .with_operator(qubit2, PauliOperator::Z);
        self.add_term(coefficient, pauli_string);
    }

    /// Add a XX interaction term between two qubits
    pub fn add_xx(&mut self, qubit1: QubitId, qubit2: QubitId, coefficient: Parameter) {
        let pauli_string = PauliString::new(self.n_qubits)
            .with_operator(qubit1, PauliOperator::X)
            .with_operator(qubit2, PauliOperator::X);
        self.add_term(coefficient, pauli_string);
    }

    /// Add a YY interaction term between two qubits
    pub fn add_yy(&mut self, qubit1: QubitId, qubit2: QubitId, coefficient: Parameter) {
        let pauli_string = PauliString::new(self.n_qubits)
            .with_operator(qubit1, PauliOperator::Y)
            .with_operator(qubit2, PauliOperator::Y);
        self.add_term(coefficient, pauli_string);
    }

    /// Add another Hamiltonian to this one
    pub fn add_hamiltonian(&mut self, other: &SymbolicHamiltonian) {
        if other.n_qubits != self.n_qubits {
            panic!("Hamiltonian size mismatch");
        }

        for term in &other.terms {
            self.terms.push(term.clone());
        }
    }

    /// Multiply the Hamiltonian by a scalar
    pub fn scale(&self, scalar: Parameter) -> Self {
        let mut result = SymbolicHamiltonian::new(self.n_qubits);
        for term in &self.terms {
            result.terms.push(term.scale(scalar.clone()));
        }
        result
    }

    /// Get all variable names in the Hamiltonian
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        for term in &self.terms {
            vars.extend(term.variables());
        }
        vars.sort();
        vars.dedup();
        vars
    }

    /// Evaluate the Hamiltonian with given variable values
    pub fn evaluate(
        &self,
        variables: &HashMap<String, f64>,
    ) -> QuantRS2Result<Vec<(f64, PauliString)>> {
        let mut result = Vec::new();
        for term in &self.terms {
            let coeff = term.evaluate_coefficient(variables)?;
            if coeff.abs() > 1e-12 {
                // Filter out nearly zero terms
                result.push((coeff, term.pauli_string.clone()));
            }
        }
        Ok(result)
    }

    /// Compute the commutator [H1, H2] = H1*H2 - H2*H1
    pub fn commutator(&self, other: &SymbolicHamiltonian) -> SymbolicHamiltonian {
        let mut result = SymbolicHamiltonian::new(self.n_qubits);

        // H1*H2 terms
        for term1 in &self.terms {
            for term2 in &other.terms {
                let (phase, pauli_product) = term1.pauli_string.multiply(&term2.pauli_string);
                let coeff = term1.coefficient.clone()
                    * term2.coefficient.clone()
                    * Parameter::complex_constant(phase);
                result.add_term(coeff, pauli_product);
            }
        }

        // -H2*H1 terms
        for term2 in &other.terms {
            for term1 in &self.terms {
                let (phase, pauli_product) = term2.pauli_string.multiply(&term1.pauli_string);
                let coeff = Parameter::constant(-1.0)
                    * term2.coefficient.clone()
                    * term1.coefficient.clone()
                    * Parameter::complex_constant(phase);
                result.add_term(coeff, pauli_product);
            }
        }

        result.simplify()
    }

    /// Simplify the Hamiltonian by combining like terms
    pub fn simplify(&self) -> Self {
        let mut grouped_terms: HashMap<PauliString, Vec<Parameter>> = HashMap::new();

        // Group terms by Pauli string
        for term in &self.terms {
            grouped_terms
                .entry(term.pauli_string.clone())
                .or_insert_with(Vec::new)
                .push(term.coefficient.clone());
        }

        let mut result = SymbolicHamiltonian::new(self.n_qubits);

        // Combine coefficients for each unique Pauli string
        for (pauli_string, coefficients) in grouped_terms {
            if coefficients.len() == 1 {
                result.add_term(coefficients[0].clone(), pauli_string);
            } else {
                // Sum all coefficients
                let mut combined_coeff = coefficients[0].clone();
                for coeff in coefficients.iter().skip(1) {
                    combined_coeff = combined_coeff + coeff.clone();
                }
                result.add_term(combined_coeff, pauli_string);
            }
        }

        result
    }

    /// Compute the expectation value with respect to computational basis states
    pub fn expectation_value(
        &self,
        state_vector: &[Complex64],
    ) -> QuantRS2Result<SymbolicExpression> {
        if state_vector.len() != 1 << self.n_qubits {
            return Err(QuantRS2Error::InvalidInput(
                "State vector size doesn't match number of qubits".to_string(),
            ));
        }

        let mut expectation = SymbolicExpression::zero();

        for term in &self.terms {
            let pauli_expectation = compute_pauli_expectation(&term.pauli_string, state_vector)?;
            let term_contribution = term.coefficient.to_symbolic_expression()
                * SymbolicExpression::complex_constant(pauli_expectation);
            expectation = expectation + term_contribution;
        }

        Ok(expectation)
    }

    /// Compute gradients with respect to parameters
    #[cfg(feature = "symbolic")]
    pub fn gradients(
        &self,
        parameter_names: &[String],
    ) -> QuantRS2Result<HashMap<String, SymbolicHamiltonian>> {
        use crate::symbolic::calculus;

        let mut gradients = HashMap::new();

        for param_name in parameter_names {
            let mut grad_hamiltonian = SymbolicHamiltonian::new(self.n_qubits);

            for term in &self.terms {
                let grad_coeff = term.coefficient.diff(param_name)?;
                grad_hamiltonian.add_term(grad_coeff, term.pauli_string.clone());
            }

            gradients.insert(param_name.clone(), grad_hamiltonian);
        }

        Ok(gradients)
    }
}

impl fmt::Display for SymbolicHamiltonian {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }

        let terms_str: Vec<String> = self.terms.iter().map(|term| term.to_string()).collect();

        write!(f, "{}", terms_str.join(" + "))
    }
}

/// Predefined Hamiltonians for common quantum problems
pub mod hamiltonians {
    use super::*;

    /// Create a transverse field Ising model Hamiltonian
    /// H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ
    pub fn transverse_field_ising(
        n_qubits: usize,
        j_coupling: Parameter,
        h_field: Parameter,
    ) -> SymbolicHamiltonian {
        let mut hamiltonian = SymbolicHamiltonian::new(n_qubits);

        // ZZ coupling terms
        for i in 0..n_qubits - 1 {
            let coupling = Parameter::constant(-1.0) * j_coupling.clone();
            hamiltonian.add_zz(QubitId::from(i), QubitId::from(i + 1), coupling);
        }

        // Transverse field terms
        for i in 0..n_qubits {
            let field = Parameter::constant(-1.0) * h_field.clone();
            hamiltonian.add_x(QubitId::from(i), field);
        }

        hamiltonian
    }

    /// Create a Heisenberg model Hamiltonian
    /// H = J Σᵢ (XᵢXᵢ₊₁ + YᵢYᵢ₊₁ + ZᵢZᵢ₊₁)
    pub fn heisenberg(n_qubits: usize, j_coupling: Parameter) -> SymbolicHamiltonian {
        let mut hamiltonian = SymbolicHamiltonian::new(n_qubits);

        for i in 0..n_qubits - 1 {
            hamiltonian.add_xx(QubitId::from(i), QubitId::from(i + 1), j_coupling.clone());
            hamiltonian.add_yy(QubitId::from(i), QubitId::from(i + 1), j_coupling.clone());
            hamiltonian.add_zz(QubitId::from(i), QubitId::from(i + 1), j_coupling.clone());
        }

        hamiltonian
    }

    /// Create a MaxCut Hamiltonian for QAOA
    /// H = Σ_{(i,j) ∈ E} wᵢⱼ/2 * (1 - ZᵢZⱼ)
    pub fn maxcut(edges: &[(QubitId, QubitId, f64)], n_qubits: usize) -> SymbolicHamiltonian {
        let mut hamiltonian = SymbolicHamiltonian::new(n_qubits);

        // Constant term
        let mut constant_term = 0.0;

        for &(i, j, weight) in edges {
            constant_term += weight / 2.0;
            let coupling = Parameter::constant(-weight / 2.0);
            hamiltonian.add_zz(i, j, coupling);
        }

        // Add constant term
        if constant_term.abs() > 1e-12 {
            hamiltonian.add_term(
                Parameter::constant(constant_term),
                PauliString::identity(n_qubits),
            );
        }

        hamiltonian
    }

    /// Create a number partitioning Hamiltonian
    /// H = (Σᵢ cᵢZᵢ)²
    pub fn number_partitioning(coefficients: &[f64]) -> SymbolicHamiltonian {
        let n_qubits = coefficients.len();
        let mut hamiltonian = SymbolicHamiltonian::new(n_qubits);

        // Diagonal terms cᵢ²Zᵢ²
        for (i, &ci) in coefficients.iter().enumerate() {
            hamiltonian.add_z(QubitId::from(i), Parameter::constant(ci * ci));
        }

        // Cross terms 2cᵢcⱼZᵢZⱼ
        for i in 0..n_qubits {
            for j in i + 1..n_qubits {
                let coeff = 2.0 * coefficients[i] * coefficients[j];
                hamiltonian.add_zz(
                    QubitId::from(i),
                    QubitId::from(j),
                    Parameter::constant(coeff),
                );
            }
        }

        hamiltonian
    }

    /// Create a molecular Hamiltonian (simplified)
    /// For demonstration purposes - real molecular Hamiltonians are more complex
    pub fn molecular_h2(bond_length: Parameter) -> SymbolicHamiltonian {
        let mut hamiltonian = SymbolicHamiltonian::new(4); // 4 qubits for H2

        // Kinetic energy terms (simplified)
        let kinetic_coeff = Parameter::constant(-1.0) / bond_length.clone();
        hamiltonian.add_z(QubitId::from(0), kinetic_coeff.clone());
        hamiltonian.add_z(QubitId::from(1), kinetic_coeff);

        // Electron-electron repulsion
        let ee_coeff = Parameter::constant(1.0) / bond_length.clone();
        hamiltonian.add_zz(QubitId::from(0), QubitId::from(1), ee_coeff);

        // Nuclear-nuclear repulsion
        let nn_coeff = Parameter::constant(1.0) / bond_length;
        hamiltonian.add_term(nn_coeff, PauliString::identity(4));

        hamiltonian
    }
}

// Helper functions

fn multiply_pauli_operators(op1: PauliOperator, op2: PauliOperator) -> (Complex64, PauliOperator) {
    use PauliOperator::*;
    match (op1, op2) {
        (I, op) | (op, I) => (Complex64::new(1.0, 0.0), op),
        (X, X) | (Y, Y) | (Z, Z) => (Complex64::new(1.0, 0.0), I),
        (X, Y) => (Complex64::new(0.0, 1.0), Z),
        (Y, X) => (Complex64::new(0.0, -1.0), Z),
        (Y, Z) => (Complex64::new(0.0, 1.0), X),
        (Z, Y) => (Complex64::new(0.0, -1.0), X),
        (Z, X) => (Complex64::new(0.0, 1.0), Y),
        (X, Z) => (Complex64::new(0.0, -1.0), Y),
    }
}

fn pauli_operators_commute(op1: PauliOperator, op2: PauliOperator) -> bool {
    use PauliOperator::*;
    match (op1, op2) {
        (I, _) | (_, I) => true,
        (X, X) | (Y, Y) | (Z, Z) => true,
        _ => false,
    }
}

fn compute_pauli_expectation(
    pauli_string: &PauliString,
    state_vector: &[Complex64],
) -> QuantRS2Result<Complex64> {
    // This is a simplified implementation
    // In practice, you'd want to use more efficient algorithms
    let n_qubits = pauli_string.n_qubits;
    let n_states = 1 << n_qubits;

    if state_vector.len() != n_states {
        return Err(QuantRS2Error::InvalidInput(
            "State vector size mismatch".to_string(),
        ));
    }

    let mut expectation = Complex64::new(0.0, 0.0);

    for i in 0..n_states {
        for j in 0..n_states {
            let matrix_element = compute_pauli_matrix_element(pauli_string, i, j);
            expectation += state_vector[i].conj() * matrix_element * state_vector[j];
        }
    }

    Ok(expectation)
}

fn compute_pauli_matrix_element(pauli_string: &PauliString, i: usize, j: usize) -> Complex64 {
    // Compute <i|P|j> where P is the Pauli string
    let mut element = Complex64::new(1.0, 0.0);

    for qubit in 0..pauli_string.n_qubits {
        let op = pauli_string.get_operator(QubitId::from(qubit));
        let bit_i = (i >> qubit) & 1;
        let bit_j = (j >> qubit) & 1;

        let local_element = match op {
            PauliOperator::I => {
                if bit_i == bit_j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            }
            PauliOperator::X => {
                if bit_i != bit_j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            }
            PauliOperator::Y => {
                if bit_i != bit_j {
                    if bit_j == 1 {
                        Complex64::new(0.0, 1.0)
                    } else {
                        Complex64::new(0.0, -1.0)
                    }
                } else {
                    Complex64::new(0.0, 0.0)
                }
            }
            PauliOperator::Z => {
                if bit_i == bit_j {
                    if bit_i == 0 {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::new(-1.0, 0.0)
                    }
                } else {
                    Complex64::new(0.0, 0.0)
                }
            }
        };

        element *= local_element;
        if element.norm() < 1e-12 {
            break; // Early termination if element becomes zero
        }
    }

    element
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_string_creation() {
        let pauli_string = PauliString::new(3)
            .with_operator(QubitId::from(0), PauliOperator::X)
            .with_operator(QubitId::from(2), PauliOperator::Z);

        assert_eq!(
            pauli_string.get_operator(QubitId::from(0)),
            PauliOperator::X
        );
        assert_eq!(
            pauli_string.get_operator(QubitId::from(1)),
            PauliOperator::I
        );
        assert_eq!(
            pauli_string.get_operator(QubitId::from(2)),
            PauliOperator::Z
        );
        assert_eq!(pauli_string.weight(), 2);
    }

    #[test]
    fn test_pauli_multiplication() {
        let p1 = PauliString::new(2).with_operator(QubitId::from(0), PauliOperator::X);
        let p2 = PauliString::new(2).with_operator(QubitId::from(0), PauliOperator::Y);

        let (phase, product) = p1.multiply(&p2);
        assert_eq!(phase, Complex64::new(0.0, 1.0)); // i
        assert_eq!(product.get_operator(QubitId::from(0)), PauliOperator::Z);
    }

    #[test]
    fn test_hamiltonian_construction() {
        let mut hamiltonian = SymbolicHamiltonian::new(3);
        hamiltonian.add_x(QubitId::from(0), Parameter::constant(1.0));
        hamiltonian.add_zz(QubitId::from(0), QubitId::from(1), Parameter::variable("J"));

        assert_eq!(hamiltonian.terms.len(), 2);
        let variables = hamiltonian.variables();
        assert!(variables.contains(&"J".to_string()));
    }

    #[test]
    fn test_transverse_field_ising() {
        let hamiltonian = hamiltonians::transverse_field_ising(
            3,
            Parameter::constant(1.0),
            Parameter::constant(0.5),
        );

        // Should have 2 ZZ terms and 3 X terms
        assert_eq!(hamiltonian.terms.len(), 5);
    }

    #[test]
    fn test_hamiltonian_scaling() {
        let mut hamiltonian = SymbolicHamiltonian::new(2);
        hamiltonian.add_x(QubitId::from(0), Parameter::constant(1.0));

        let scaled = hamiltonian.scale(Parameter::constant(2.0));
        let vars = HashMap::new();
        let evaluated = scaled.evaluate(&vars).unwrap();

        assert_eq!(evaluated[0].0, 2.0);
    }
}
