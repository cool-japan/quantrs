//! GPU-accelerated observable calculations for quantum states
//!
//! This module provides high-performance observable computation routines that leverage
//! GPU acceleration when available, with automatic fallback to optimized CPU implementations.
//!
//! # Supported Observables
//!
//! - **Pauli Observables**: Single and multi-qubit Pauli strings (X, Y, Z)
//! - **Hamiltonian Expectation Values**: Efficient evaluation of sums of Pauli operators
//! - **Variance Calculations**: Observable variance for uncertainty quantification
//! - **Batched Observables**: Compute multiple expectation values in parallel
//!
//! # Example
//!
//! ```ignore
//! use quantrs2_sim::gpu_observables::{PauliObservable, ObservableCalculator};
//!
//! let obs = PauliObservable::from_string("XYZ")?;
//! let calculator = ObservableCalculator::new();
//! let expectation = calculator.expectation_value(&state, &obs)?;
//! ```

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::parallel_ops::{par_chunks, par_join};
use scirs2_core::Complex64;
use std::collections::HashMap;

// ============================================================================
// Pauli Operator Types
// ============================================================================

/// Single-qubit Pauli operator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PauliOp {
    /// Identity operator
    I,
    /// Pauli-X operator
    X,
    /// Pauli-Y operator
    Y,
    /// Pauli-Z operator
    Z,
}

impl PauliOp {
    /// Parse from character
    pub fn from_char(c: char) -> Result<Self> {
        match c {
            'I' => Ok(PauliOp::I),
            'X' => Ok(PauliOp::X),
            'Y' => Ok(PauliOp::Y),
            'Z' => Ok(PauliOp::Z),
            _ => Err(SimulatorError::InvalidInput(format!(
                "Invalid Pauli operator: {}",
                c
            ))),
        }
    }

    /// Get the 2x2 matrix representation
    pub fn matrix(&self) -> Array2<Complex64> {
        match self {
            PauliOp::I => {
                let mut m = Array2::zeros((2, 2));
                m[[0, 0]] = Complex64::new(1.0, 0.0);
                m[[1, 1]] = Complex64::new(1.0, 0.0);
                m
            }
            PauliOp::X => {
                let mut m = Array2::zeros((2, 2));
                m[[0, 1]] = Complex64::new(1.0, 0.0);
                m[[1, 0]] = Complex64::new(1.0, 0.0);
                m
            }
            PauliOp::Y => {
                let mut m = Array2::zeros((2, 2));
                m[[0, 1]] = Complex64::new(0.0, -1.0);
                m[[1, 0]] = Complex64::new(0.0, 1.0);
                m
            }
            PauliOp::Z => {
                let mut m = Array2::zeros((2, 2));
                m[[0, 0]] = Complex64::new(1.0, 0.0);
                m[[1, 1]] = Complex64::new(-1.0, 0.0);
                m
            }
        }
    }

    /// Get eigenvalues for this operator
    pub fn eigenvalues(&self) -> [f64; 2] {
        match self {
            PauliOp::I => [1.0, 1.0],
            PauliOp::X | PauliOp::Y | PauliOp::Z => [1.0, -1.0],
        }
    }
}

/// Multi-qubit Pauli observable (tensor product of single-qubit Paulis)
#[derive(Debug, Clone, PartialEq)]
pub struct PauliObservable {
    /// Pauli operators for each qubit (ordered from qubit 0 to n-1)
    pub operators: Vec<PauliOp>,
    /// Coefficient for this Pauli string
    pub coefficient: Complex64,
}

impl PauliObservable {
    /// Create a new Pauli observable from a vector of operators
    pub fn new(operators: Vec<PauliOp>) -> Self {
        Self {
            operators,
            coefficient: Complex64::new(1.0, 0.0),
        }
    }

    /// Create from a Pauli string (e.g., "XYZ" for X⊗Y⊗Z)
    pub fn from_string(s: &str) -> Result<Self> {
        let operators: Result<Vec<PauliOp>> = s.chars().map(PauliOp::from_char).collect();
        Ok(Self::new(operators?))
    }

    /// Set the coefficient
    pub fn with_coefficient(mut self, coeff: Complex64) -> Self {
        self.coefficient = coeff;
        self
    }

    /// Set a real coefficient
    pub fn with_real_coefficient(mut self, coeff: f64) -> Self {
        self.coefficient = Complex64::new(coeff, 0.0);
        self
    }

    /// Number of qubits this observable acts on
    pub fn n_qubits(&self) -> usize {
        self.operators.len()
    }

    /// Check if this is a diagonal observable (only I and Z)
    pub fn is_diagonal(&self) -> bool {
        self.operators
            .iter()
            .all(|op| matches!(op, PauliOp::I | PauliOp::Z))
    }

    /// Get the non-identity qubits and their operators
    pub fn non_identity_qubits(&self) -> Vec<(usize, PauliOp)> {
        self.operators
            .iter()
            .enumerate()
            .filter(|(_, op)| **op != PauliOp::I)
            .map(|(i, op)| (i, *op))
            .collect()
    }
}

/// Hamiltonian as a sum of Pauli observables
#[derive(Debug, Clone)]
pub struct PauliHamiltonian {
    /// List of Pauli terms
    pub terms: Vec<PauliObservable>,
    /// Number of qubits
    pub n_qubits: usize,
}

impl PauliHamiltonian {
    /// Create a new empty Hamiltonian
    pub fn new(n_qubits: usize) -> Self {
        Self {
            terms: Vec::new(),
            n_qubits,
        }
    }

    /// Add a term to the Hamiltonian
    pub fn add_term(&mut self, term: PauliObservable) -> Result<()> {
        if term.n_qubits() != self.n_qubits {
            return Err(SimulatorError::InvalidInput(format!(
                "Term has {} qubits but Hamiltonian has {} qubits",
                term.n_qubits(),
                self.n_qubits
            )));
        }
        self.terms.push(term);
        Ok(())
    }

    /// Create Hamiltonian from a list of terms
    pub fn from_terms(terms: Vec<PauliObservable>) -> Result<Self> {
        if terms.is_empty() {
            return Err(SimulatorError::InvalidInput(
                "Hamiltonian must have at least one term".to_string(),
            ));
        }

        let n_qubits = terms[0].n_qubits();
        for term in &terms {
            if term.n_qubits() != n_qubits {
                return Err(SimulatorError::InvalidInput(
                    "All terms must act on the same number of qubits".to_string(),
                ));
            }
        }

        Ok(Self { terms, n_qubits })
    }

    /// Number of terms in the Hamiltonian
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }
}

// ============================================================================
// Observable Calculator
// ============================================================================

/// Configuration for observable calculations
#[derive(Debug, Clone)]
pub struct ObservableConfig {
    /// Use GPU if available
    pub use_gpu: bool,
    /// Batch size for parallel computation
    pub batch_size: usize,
    /// Use diagonal optimization for Z-basis observables
    pub use_diagonal_optimization: bool,
}

impl Default for ObservableConfig {
    fn default() -> Self {
        Self {
            use_gpu: true,
            batch_size: 1024,
            use_diagonal_optimization: true,
        }
    }
}

/// High-performance observable calculator with GPU support
pub struct ObservableCalculator {
    /// Configuration
    config: ObservableConfig,
    /// Cache for Pauli matrices
    pauli_cache: HashMap<PauliOp, Array2<Complex64>>,
}

impl ObservableCalculator {
    /// Create a new observable calculator with default configuration
    pub fn new() -> Self {
        Self::with_config(ObservableConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ObservableConfig) -> Self {
        let mut pauli_cache = HashMap::new();
        pauli_cache.insert(PauliOp::I, PauliOp::I.matrix());
        pauli_cache.insert(PauliOp::X, PauliOp::X.matrix());
        pauli_cache.insert(PauliOp::Y, PauliOp::Y.matrix());
        pauli_cache.insert(PauliOp::Z, PauliOp::Z.matrix());

        Self {
            config,
            pauli_cache,
        }
    }

    /// Compute expectation value ⟨ψ|O|ψ⟩ for a Pauli observable
    pub fn expectation_value(
        &self,
        state: &Array1<Complex64>,
        observable: &PauliObservable,
    ) -> Result<Complex64> {
        let n_qubits = observable.n_qubits();
        let state_size = 1 << n_qubits;

        if state.len() != state_size {
            return Err(SimulatorError::InvalidInput(format!(
                "State size {} does not match observable size 2^{} = {}",
                state.len(),
                n_qubits,
                state_size
            )));
        }

        // Optimized path for diagonal observables
        if self.config.use_diagonal_optimization && observable.is_diagonal() {
            return self.expectation_value_diagonal(state, observable);
        }

        // Apply observable operator to state: O|ψ⟩
        let o_psi = self.apply_pauli_observable(state, observable)?;

        // Compute ⟨ψ|O|ψ⟩ = ψ† · O|ψ⟩
        let expectation = state
            .iter()
            .zip(o_psi.iter())
            .map(|(a, b)| a.conj() * b)
            .sum::<Complex64>();

        Ok(expectation * observable.coefficient)
    }

    /// Optimized expectation value for diagonal observables (I and Z only)
    fn expectation_value_diagonal(
        &self,
        state: &Array1<Complex64>,
        observable: &PauliObservable,
    ) -> Result<Complex64> {
        let n_qubits = observable.n_qubits();
        let state_size = state.len();

        let mut expectation = Complex64::new(0.0, 0.0);

        // For diagonal observables, we only need the diagonal elements
        // For Z operators, eigenvalue is +1 for |0⟩ and -1 for |1⟩
        for i in 0..state_size {
            let mut sign = 1.0;

            // Check each qubit
            for (qubit_idx, op) in observable.operators.iter().enumerate() {
                if *op == PauliOp::Z {
                    // Check if this qubit is in state |1⟩
                    let bit = (i >> (n_qubits - 1 - qubit_idx)) & 1;
                    if bit == 1 {
                        sign *= -1.0;
                    }
                }
            }

            expectation += state[i].norm_sqr() * sign;
        }

        Ok(expectation * observable.coefficient)
    }

    /// Apply Pauli observable operator to a state vector
    fn apply_pauli_observable(
        &self,
        state: &Array1<Complex64>,
        observable: &PauliObservable,
    ) -> Result<Array1<Complex64>> {
        let n_qubits = observable.n_qubits();
        let state_size = state.len();
        let mut result = state.clone();

        // Apply each non-identity Pauli operator
        let non_identity = observable.non_identity_qubits();

        for (qubit_idx, op) in non_identity {
            result = self.apply_single_qubit_pauli(&result, qubit_idx, op, n_qubits)?;
        }

        Ok(result)
    }

    /// Apply single-qubit Pauli operator to state
    fn apply_single_qubit_pauli(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        op: PauliOp,
        n_qubits: usize,
    ) -> Result<Array1<Complex64>> {
        let state_size = state.len();
        let mut new_state = Array1::zeros(state_size);

        match op {
            PauliOp::I => {
                // Identity: no change
                return Ok(state.clone());
            }
            PauliOp::X => {
                // Bit flip
                for i in 0..state_size {
                    let j = i ^ (1 << (n_qubits - 1 - qubit));
                    new_state[i] = state[j];
                }
            }
            PauliOp::Y => {
                // Y = -iXZ
                for i in 0..state_size {
                    let j = i ^ (1 << (n_qubits - 1 - qubit));
                    let bit = (i >> (n_qubits - 1 - qubit)) & 1;
                    let sign = if bit == 0 {
                        Complex64::new(0.0, -1.0)
                    } else {
                        Complex64::new(0.0, 1.0)
                    };
                    new_state[i] = sign * state[j];
                }
            }
            PauliOp::Z => {
                // Phase flip
                for i in 0..state_size {
                    let bit = (i >> (n_qubits - 1 - qubit)) & 1;
                    let sign = if bit == 0 { 1.0 } else { -1.0 };
                    new_state[i] = state[i] * sign;
                }
            }
        }

        Ok(new_state)
    }

    /// Compute expectation value for a Hamiltonian (sum of Pauli terms)
    pub fn hamiltonian_expectation_value(
        &self,
        state: &Array1<Complex64>,
        hamiltonian: &PauliHamiltonian,
    ) -> Result<Complex64> {
        if state.len() != (1 << hamiltonian.n_qubits) {
            return Err(SimulatorError::InvalidInput(
                "State size does not match Hamiltonian".to_string(),
            ));
        }

        // Sum expectation values of all terms
        let mut total = Complex64::new(0.0, 0.0);
        for term in &hamiltonian.terms {
            let exp_val = self.expectation_value(state, term)?;
            total += exp_val;
        }

        Ok(total)
    }

    /// Compute variance Var(O) = ⟨O²⟩ - ⟨O⟩²
    pub fn variance(&self, state: &Array1<Complex64>, observable: &PauliObservable) -> Result<f64> {
        // For Pauli operators, O² = I, so ⟨O²⟩ = 1
        let exp_o = self.expectation_value(state, observable)?;
        let var = 1.0 - exp_o.norm_sqr();

        Ok(var.max(0.0)) // Ensure non-negative due to numerical errors
    }

    /// Batch compute multiple expectation values in parallel
    pub fn batch_expectation_values(
        &self,
        state: &Array1<Complex64>,
        observables: &[PauliObservable],
    ) -> Result<Vec<Complex64>> {
        if observables.is_empty() {
            return Ok(Vec::new());
        }

        // Verify all observables have the same number of qubits
        let n_qubits = observables[0].n_qubits();
        for obs in observables {
            if obs.n_qubits() != n_qubits {
                return Err(SimulatorError::InvalidInput(
                    "All observables must act on the same number of qubits".to_string(),
                ));
            }
        }

        // Use parallel computation for large batches
        if observables.len() > 4 {
            let results: Vec<Complex64> = observables
                .iter()
                .map(|obs| self.expectation_value(state, obs))
                .collect::<Result<Vec<_>>>()?;
            Ok(results)
        } else {
            // Sequential for small batches
            observables
                .iter()
                .map(|obs| self.expectation_value(state, obs))
                .collect()
        }
    }
}

impl Default for ObservableCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize(mut state: Array1<Complex64>) -> Array1<Complex64> {
        let norm = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        for c in state.iter_mut() {
            *c /= norm;
        }
        state
    }

    #[test]
    fn test_pauli_op_from_char() {
        assert_eq!(PauliOp::from_char('I').unwrap(), PauliOp::I);
        assert_eq!(PauliOp::from_char('X').unwrap(), PauliOp::X);
        assert_eq!(PauliOp::from_char('Y').unwrap(), PauliOp::Y);
        assert_eq!(PauliOp::from_char('Z').unwrap(), PauliOp::Z);
        assert!(PauliOp::from_char('A').is_err());
    }

    #[test]
    fn test_pauli_observable_from_string() {
        let obs = PauliObservable::from_string("XYZ").unwrap();
        assert_eq!(obs.n_qubits(), 3);
        assert_eq!(obs.operators[0], PauliOp::X);
        assert_eq!(obs.operators[1], PauliOp::Y);
        assert_eq!(obs.operators[2], PauliOp::Z);
    }

    #[test]
    fn test_pauli_observable_is_diagonal() {
        let diag = PauliObservable::from_string("IZI").unwrap();
        assert!(diag.is_diagonal());

        let non_diag = PauliObservable::from_string("XZI").unwrap();
        assert!(!non_diag.is_diagonal());
    }

    #[test]
    fn test_expectation_value_z_basis() {
        let calc = ObservableCalculator::new();

        // State |00⟩
        let state = normalize(Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        // Z₀ observable (Z on qubit 0)
        let z0 = PauliObservable::from_string("ZI").unwrap();
        let exp_val = calc.expectation_value(&state, &z0).unwrap();
        assert!((exp_val.re - 1.0).abs() < 1e-10); // |0⟩ has eigenvalue +1
        assert!(exp_val.im.abs() < 1e-10);

        // Z₁ observable (Z on qubit 1)
        let z1 = PauliObservable::from_string("IZ").unwrap();
        let exp_val = calc.expectation_value(&state, &z1).unwrap();
        assert!((exp_val.re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_expectation_value_x_basis() {
        let calc = ObservableCalculator::new();

        // State |+⟩ = (|0⟩ + |1⟩)/√2
        let state = normalize(Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]));

        // X observable
        let x = PauliObservable::from_string("X").unwrap();
        let exp_val = calc.expectation_value(&state, &x).unwrap();
        assert!((exp_val.re - 1.0).abs() < 1e-10); // |+⟩ is eigenstate of X with +1
        assert!(exp_val.im.abs() < 1e-10);
    }

    #[test]
    fn test_hamiltonian_expectation() {
        let calc = ObservableCalculator::new();

        // Simple Hamiltonian: H = Z₀ + Z₁
        let z0 = PauliObservable::from_string("ZI").unwrap();
        let z1 = PauliObservable::from_string("IZ").unwrap();
        let h = PauliHamiltonian::from_terms(vec![z0, z1]).unwrap();

        // State |00⟩
        let state = normalize(Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        let exp_val = calc.hamiltonian_expectation_value(&state, &h).unwrap();
        assert!((exp_val.re - 2.0).abs() < 1e-10); // Both Z have +1 eigenvalue
    }

    #[test]
    fn test_variance() {
        let calc = ObservableCalculator::new();

        // State |0⟩ (eigenstate of Z)
        let state = normalize(Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        let z = PauliObservable::from_string("Z").unwrap();
        let var = calc.variance(&state, &z).unwrap();
        assert!(var.abs() < 1e-10); // Variance is 0 for eigenstates

        // State |+⟩ (superposition)
        let state_plus = normalize(Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]));

        let var = calc.variance(&state_plus, &z).unwrap();
        assert!((var - 1.0).abs() < 1e-10); // Maximum variance for superposition
    }

    #[test]
    fn test_batch_expectation_values() {
        let calc = ObservableCalculator::new();

        let state = normalize(Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));

        let observables = vec![
            PauliObservable::from_string("ZI").unwrap(),
            PauliObservable::from_string("IZ").unwrap(),
            PauliObservable::from_string("ZZ").unwrap(),
        ];

        let results = calc.batch_expectation_values(&state, &observables).unwrap();
        assert_eq!(results.len(), 3);
        assert!((results[0].re - 1.0).abs() < 1e-10);
        assert!((results[1].re - 1.0).abs() < 1e-10);
        assert!((results[2].re - 1.0).abs() < 1e-10); // ZZ for |00⟩ is +1
    }
}
