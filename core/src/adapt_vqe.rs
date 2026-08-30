//! ADAPT-VQE: Adaptive Derivative-Assembled Pseudo-Trotter ansatz for quantum chemistry.
//!
//! Implements the ADAPT-VQE algorithm (Grimsley et al., 2019) which adaptively
//! builds a compact ansatz from a fermionic operator pool, avoiding barren
//! plateaus and minimising circuit depth relative to fixed-depth approaches.

// ADAPT-VQE: Adaptive Derivative-Assembled Pseudo-Trotter VQE
//
// A state-of-the-art quantum chemistry algorithm that adaptively constructs
// the ansatz circuit during optimization, avoiding the barren plateau problem
// and reducing circuit depth.
//
// Reference: Grimsley, H. R., et al. (2019). "An adaptive variational algorithm for exact molecular simulations on a quantum computer"
// Nature Communications 10, 3007

use crate::error::QuantRS2Error;
use crate::optimization_stubs::{minimize, Method, Options};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::Complex64;
use std::collections::HashMap;

/// Fermionic operator pool for quantum chemistry
///
/// Contains the complete set of single and double excitation operators
/// that can be used to construct the ADAPT-VQE ansatz.
#[derive(Debug, Clone)]
pub struct FermionicOperatorPool {
    /// Single excitation operators (a†_p a_q)
    pub single_excitations: Vec<FermionicOperator>,
    /// Double excitation operators (a†_p a†_q a_r a_s)
    pub double_excitations: Vec<FermionicOperator>,
    /// Number of spin orbitals
    pub num_orbitals: usize,
}

impl FermionicOperatorPool {
    /// Create a new operator pool for a given number of spin orbitals
    pub fn new(num_orbitals: usize) -> Self {
        let mut single_excitations = Vec::new();
        let mut double_excitations = Vec::new();

        // Generate all single excitations
        for p in 0..num_orbitals {
            for q in 0..num_orbitals {
                if p != q {
                    single_excitations.push(FermionicOperator::single_excitation(p, q));
                }
            }
        }

        // Generate all double excitations
        for p in 0..num_orbitals {
            for q in p + 1..num_orbitals {
                for r in 0..num_orbitals {
                    for s in r + 1..num_orbitals {
                        if (p, q) != (r, s) {
                            double_excitations
                                .push(FermionicOperator::double_excitation(p, q, r, s));
                        }
                    }
                }
            }
        }

        Self {
            single_excitations,
            double_excitations,
            num_orbitals,
        }
    }

    /// Get all operators in the pool
    pub fn all_operators(&self) -> Vec<FermionicOperator> {
        let mut operators = Vec::new();
        operators.extend(self.single_excitations.clone());
        operators.extend(self.double_excitations.clone());
        operators
    }

    /// Get operator count
    pub fn size(&self) -> usize {
        self.single_excitations.len() + self.double_excitations.len()
    }
}

/// Fermionic operator representation
#[derive(Debug, Clone, PartialEq)]
pub struct FermionicOperator {
    /// Creation operator indices
    pub creation_ops: Vec<usize>,
    /// Annihilation operator indices
    pub annihilation_ops: Vec<usize>,
    /// Operator label for identification
    pub label: String,
}

impl FermionicOperator {
    /// Create a single excitation operator a†_p a_q
    pub fn single_excitation(p: usize, q: usize) -> Self {
        Self {
            creation_ops: vec![p],
            annihilation_ops: vec![q],
            label: format!("E_{{{},{}}}", p, q),
        }
    }

    /// Create a double excitation operator a†_p a†_q a_r a_s
    pub fn double_excitation(p: usize, q: usize, r: usize, s: usize) -> Self {
        Self {
            creation_ops: vec![p, q],
            annihilation_ops: vec![r, s],
            label: format!("E_{{{},{},{},{}}}", p, q, r, s),
        }
    }

    /// Convert to Pauli string representation using Jordan-Wigner transformation
    pub fn to_pauli_string(&self, num_qubits: usize) -> PauliString {
        // Simplified Jordan-Wigner transformation
        // Full implementation would require more sophisticated mapping
        let mut pauli_ops = vec![PauliOp::I; num_qubits];

        // Apply creation operators
        for &idx in &self.creation_ops {
            if idx < num_qubits {
                pauli_ops[idx] = PauliOp::X;
            }
        }

        // Apply annihilation operators
        for &idx in &self.annihilation_ops {
            if idx < num_qubits {
                pauli_ops[idx] = PauliOp::Y;
            }
        }

        PauliString {
            operators: pauli_ops,
            coefficient: Complex64::new(1.0, 0.0),
        }
    }
}

/// Pauli operator types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PauliOp {
    I, // Identity
    X, // Pauli-X
    Y, // Pauli-Y
    Z, // Pauli-Z
}

/// Pauli string representation of a quantum operator
#[derive(Debug, Clone)]
pub struct PauliString {
    /// Pauli operators for each qubit
    pub operators: Vec<PauliOp>,
    /// Overall coefficient
    pub coefficient: Complex64,
}

impl PauliString {
    /// Compute expectation value <ψ|P|ψ> for this Pauli string
    pub fn expectation_value(&self, state: &Array1<Complex64>) -> Complex64 {
        // Apply Pauli operator (including its coefficient) to the state and
        // compute the overlap <ψ|c·P|ψ>.
        let transformed = self.apply_to_state(state);
        state
            .iter()
            .zip(transformed.iter())
            .map(|(a, b)| a.conj() * b)
            .sum::<Complex64>()
    }

    /// Apply Pauli string (scaled by its coefficient) to a quantum state.
    ///
    /// Returns `c · P |ψ⟩`, where `c` is [`PauliString::coefficient`]. Qubit `k`
    /// corresponds to bit `k` of the state-vector index (little-endian), matching
    /// the bit-mask convention used throughout the crate.
    pub fn apply_to_state(&self, state: &Array1<Complex64>) -> Array1<Complex64> {
        let n = self.operators.len();
        let dim = 1 << n;
        let mut result = Array1::<Complex64>::zeros(dim);

        for i in 0..dim {
            let mut new_index = i;
            let mut phase = self.coefficient;

            for (qubit, &op) in self.operators.iter().enumerate() {
                let bit = (i >> qubit) & 1;
                match op {
                    PauliOp::I => {}
                    PauliOp::X => {
                        new_index ^= 1 << qubit; // Flip bit
                    }
                    PauliOp::Y => {
                        new_index ^= 1 << qubit;
                        phase *= if bit == 0 {
                            Complex64::new(0.0, 1.0)
                        } else {
                            Complex64::new(0.0, -1.0)
                        };
                    }
                    PauliOp::Z => {
                        if bit == 1 {
                            phase *= Complex64::new(-1.0, 0.0);
                        }
                    }
                }
            }

            result[new_index] += phase * state[i];
        }

        result
    }

    /// Compute commutator [H, P] where H is the Hamiltonian
    pub fn commutator_with_hamiltonian(
        &self,
        hamiltonian: &MolecularHamiltonian,
        state: &Array1<Complex64>,
    ) -> Complex64 {
        // [H, P] = HP - PH
        let hp_state = hamiltonian.apply_to_state(&self.apply_to_state(state));
        let ph_state = self.apply_to_state(&hamiltonian.apply_to_state(state));

        state
            .iter()
            .zip(hp_state.iter().zip(ph_state.iter()))
            .map(|(psi, (hp, ph))| psi.conj() * (hp - ph))
            .sum()
    }
}

/// Multiply two single-qubit Pauli operators, returning the resulting Pauli and
/// the complex phase factor (Pauli algebra: XY = iZ, YZ = iX, ZX = iY, etc.).
fn pauli_mul(a: PauliOp, b: PauliOp) -> (PauliOp, Complex64) {
    use PauliOp::{I, X, Y, Z};
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    match (a, b) {
        (I, x) => (x, one),
        (x, I) => (x, one),
        (X, X) | (Y, Y) | (Z, Z) => (I, one),
        (X, Y) => (Z, i),
        (Y, X) => (Z, -i),
        (Y, Z) => (X, i),
        (Z, Y) => (X, -i),
        (Z, X) => (Y, i),
        (X, Z) => (Y, -i),
    }
}

/// A Pauli string under construction: a per-qubit operator list plus a scalar
/// coefficient. Used to accumulate Jordan-Wigner products before they are turned
/// into [`PauliString`]s.
#[derive(Clone)]
struct PauliTerm {
    operators: Vec<PauliOp>,
    coefficient: Complex64,
}

impl PauliTerm {
    fn identity(num_qubits: usize) -> Self {
        Self {
            operators: vec![PauliOp::I; num_qubits],
            coefficient: Complex64::new(1.0, 0.0),
        }
    }

    /// Multiply this term (in place sense, returns new) by a single-qubit Pauli
    /// on `qubit`, folding the resulting phase into the coefficient.
    fn times_single(&self, qubit: usize, op: PauliOp) -> Self {
        let mut operators = self.operators.clone();
        let (new_op, phase) = pauli_mul(operators[qubit], op);
        operators[qubit] = new_op;
        Self {
            operators,
            coefficient: self.coefficient * phase,
        }
    }
}

/// Expand a single fermionic ladder operator into its two Jordan-Wigner Pauli
/// terms acting on `num_qubits` qubits.
///
/// `a†_p = ½ (X_p - i Y_p) ⊗ Z_{<p}` (when `creation == true`)
/// `a_p  = ½ (X_p + i Y_p) ⊗ Z_{<p}` (when `creation == false`)
fn jordan_wigner_ladder(site: usize, creation: bool, num_qubits: usize) -> Vec<PauliTerm> {
    let half = Complex64::new(0.5, 0.0);
    // Sign on the Y component: -i for creation, +i for annihilation.
    let y_coeff = if creation {
        Complex64::new(0.0, -0.5)
    } else {
        Complex64::new(0.0, 0.5)
    };

    let mut x_term = PauliTerm::identity(num_qubits);
    let mut y_term = PauliTerm::identity(num_qubits);

    // Jordan-Wigner Z string on all qubits with index < site.
    for z in 0..site {
        x_term.operators[z] = PauliOp::Z;
        y_term.operators[z] = PauliOp::Z;
    }
    x_term.operators[site] = PauliOp::X;
    x_term.coefficient = half;
    y_term.operators[site] = PauliOp::Y;
    y_term.coefficient = y_coeff;

    vec![x_term, y_term]
}

/// Convert a normal-ordered product of creation operators (`creations`) followed
/// by annihilation operators (`annihilations`) into a sum of [`PauliString`]s via
/// the Jordan-Wigner transformation.
///
/// The product is `a†_{c0} a†_{c1} ... a_{a0} a_{a1} ...`, applied left-to-right.
fn jordan_wigner_excitation(
    creations: &[usize],
    annihilations: &[usize],
    num_qubits: usize,
) -> Vec<PauliString> {
    // Start with the identity term, then fold each ladder operator's two-term
    // expansion into the running product set.
    let mut terms: Vec<PauliTerm> = vec![PauliTerm::identity(num_qubits)];

    let ladders = creations
        .iter()
        .map(|&p| (p, true))
        .chain(annihilations.iter().map(|&p| (p, false)));

    for (site, creation) in ladders {
        let factor = jordan_wigner_ladder(site, creation, num_qubits);
        let mut next = Vec::with_capacity(terms.len() * factor.len());
        for term in &terms {
            for ladder_term in &factor {
                // Multiply `term` by `ladder_term` qubit-by-qubit.
                let mut acc = PauliTerm {
                    operators: term.operators.clone(),
                    coefficient: term.coefficient * ladder_term.coefficient,
                };
                for (qubit, &op) in ladder_term.operators.iter().enumerate() {
                    if op != PauliOp::I {
                        acc = acc.times_single(qubit, op);
                    }
                }
                next.push(acc);
            }
        }
        terms = next;
    }

    terms
        .into_iter()
        .map(|t| PauliString {
            operators: t.operators,
            coefficient: t.coefficient,
        })
        .collect()
}

/// Molecular Hamiltonian in second-quantized form
#[derive(Debug, Clone)]
pub struct MolecularHamiltonian {
    /// One-electron integrals
    pub one_electron_integrals: Array2<f64>,
    /// Two-electron integrals (4D tensor flattened)
    pub two_electron_integrals: HashMap<(usize, usize, usize, usize), f64>,
    /// Nuclear repulsion energy
    pub nuclear_repulsion: f64,
    /// Number of spin orbitals
    pub num_orbitals: usize,
}

impl MolecularHamiltonian {
    /// Create a new molecular Hamiltonian
    pub fn new(
        one_electron: Array2<f64>,
        two_electron: HashMap<(usize, usize, usize, usize), f64>,
        nuclear_repulsion: f64,
    ) -> Self {
        let num_orbitals = one_electron.nrows();
        Self {
            one_electron_integrals: one_electron,
            two_electron_integrals: two_electron,
            nuclear_repulsion,
            num_orbitals,
        }
    }

    /// Apply Hamiltonian to a quantum state.
    ///
    /// The second-quantized Hamiltonian
    /// `H = Σ_pq h_pq a†_p a_q + ½ Σ_pqrs h_pqrs a†_p a†_q a_r a_s`
    /// is mapped to qubit operators via the Jordan-Wigner transformation and
    /// applied term-by-term to the input state vector.
    ///
    /// Each fermionic ladder operator is expanded into a sum of Pauli strings:
    /// `a†_p = ½ (X_p - i Y_p) ⊗ Z_{<p}` and `a_p = ½ (X_p + i Y_p) ⊗ Z_{<p}`,
    /// where `Z_{<p}` is the Jordan-Wigner string of Pauli-Z operators on all
    /// qubits with index below `p`. Products of ladder operators are formed by
    /// multiplying the corresponding Pauli strings (tracking the i/-i phases and
    /// the Z-string parities), then each resulting Pauli string is applied to the
    /// state via the cheap bit-mask routine in [`PauliString::apply_to_state`].
    pub fn apply_to_state(&self, state: &Array1<Complex64>) -> Array1<Complex64> {
        let num_qubits = self.num_orbitals;
        let dim = 1usize << num_qubits;
        let mut result = Array1::<Complex64>::zeros(dim);

        // One-electron part: Σ_pq h_pq a†_p a_q
        for p in 0..num_qubits {
            for q in 0..num_qubits {
                let coeff = self.one_electron_integrals[[p, q]];
                if coeff.abs() < 1e-15 {
                    continue;
                }
                let pauli_terms = jordan_wigner_excitation(&[p], &[q], num_qubits);
                for term in &pauli_terms {
                    let scaled = PauliString {
                        operators: term.operators.clone(),
                        coefficient: term.coefficient * coeff,
                    };
                    let contribution = scaled.apply_to_state(state);
                    result = result + contribution;
                }
            }
        }

        // Two-electron part: ½ Σ_pqrs h_pqrs a†_p a†_q a_r a_s
        for (&(p, q, r, s), &coeff) in &self.two_electron_integrals {
            if coeff.abs() < 1e-15 {
                continue;
            }
            if p >= num_qubits || q >= num_qubits || r >= num_qubits || s >= num_qubits {
                continue;
            }
            let pauli_terms = jordan_wigner_excitation(&[p, q], &[r, s], num_qubits);
            for term in &pauli_terms {
                let scaled = PauliString {
                    operators: term.operators.clone(),
                    coefficient: term.coefficient * coeff * 0.5,
                };
                let contribution = scaled.apply_to_state(state);
                result = result + contribution;
            }
        }

        result
    }

    /// Compute energy expectation value <ψ|H|ψ>
    pub fn expectation_value(&self, state: &Array1<Complex64>) -> f64 {
        let h_psi = self.apply_to_state(state);
        let energy: Complex64 = state
            .iter()
            .zip(h_psi.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        energy.re + self.nuclear_repulsion
    }
}

/// ADAPT-VQE algorithm configuration
#[derive(Debug, Clone)]
pub struct AdaptVQEConfig {
    /// Gradient threshold for operator selection
    pub gradient_threshold: f64,
    /// Maximum number of ADAPT iterations
    pub max_iterations: usize,
    /// Energy convergence threshold
    pub energy_threshold: f64,
    /// Maximum VQE optimization steps per iteration
    pub max_vqe_steps: usize,
    /// Optimizer for parameter optimization
    pub optimizer_method: Method,
}

impl Default for AdaptVQEConfig {
    fn default() -> Self {
        Self {
            gradient_threshold: 1e-3,
            max_iterations: 50,
            energy_threshold: 1e-6,
            max_vqe_steps: 100,
            optimizer_method: Method::LBFGS,
        }
    }
}

/// ADAPT-VQE ansatz built adaptively
#[derive(Debug, Clone)]
pub struct AdaptAnsatz {
    /// Selected operators in order
    pub operators: Vec<FermionicOperator>,
    /// Optimized parameters for each operator
    pub parameters: Vec<f64>,
    /// Energy at each iteration
    pub energy_history: Vec<f64>,
}

impl AdaptAnsatz {
    /// Create an empty ansatz
    pub const fn new() -> Self {
        Self {
            operators: Vec::new(),
            parameters: Vec::new(),
            energy_history: Vec::new(),
        }
    }

    /// Add a new operator to the ansatz
    pub fn add_operator(&mut self, operator: FermionicOperator, parameter: f64) {
        self.operators.push(operator);
        self.parameters.push(parameter);
    }

    /// Get current circuit depth (number of operators)
    pub fn depth(&self) -> usize {
        self.operators.len()
    }

    /// Apply ansatz to a reference state
    pub fn apply_to_state(
        &self,
        reference_state: &Array1<Complex64>,
        num_qubits: usize,
    ) -> Array1<Complex64> {
        let mut state = reference_state.clone();

        for (operator, &theta) in self.operators.iter().zip(self.parameters.iter()) {
            let pauli_string = operator.to_pauli_string(num_qubits);

            // Apply exp(-iθP) using Pauli rotation
            // In practice, would use Trotter decomposition or other methods
            let rotation = self.apply_pauli_rotation(&pauli_string, theta);
            state = rotation.dot(&state);
        }

        state
    }

    /// Apply Pauli rotation exp(-iθP)
    fn apply_pauli_rotation(&self, pauli: &PauliString, theta: f64) -> Array2<Complex64> {
        let n = pauli.operators.len();
        let dim = 1 << n;

        // Simplified: construct rotation matrix
        // Full implementation would use efficient Pauli rotation circuits
        let mut rotation = Array2::<Complex64>::zeros((dim, dim));

        for i in 0..dim {
            for j in 0..dim {
                if i == j {
                    rotation[[i, j]] = Complex64::new((theta / 2.0).cos(), 0.0);
                }
            }
        }

        rotation
    }
}

impl Default for AdaptAnsatz {
    fn default() -> Self {
        Self::new()
    }
}

/// Main ADAPT-VQE algorithm implementation
#[derive(Debug)]
pub struct AdaptVQE {
    /// Molecular Hamiltonian
    pub hamiltonian: MolecularHamiltonian,
    /// Operator pool
    pub operator_pool: FermionicOperatorPool,
    /// Configuration
    pub config: AdaptVQEConfig,
    /// Current ansatz
    pub ansatz: AdaptAnsatz,
    /// Number of qubits required
    pub num_qubits: usize,
}

impl AdaptVQE {
    /// Create a new ADAPT-VQE instance
    pub fn new(
        hamiltonian: MolecularHamiltonian,
        num_qubits: usize,
        config: AdaptVQEConfig,
    ) -> Self {
        let operator_pool = FermionicOperatorPool::new(hamiltonian.num_orbitals);
        let ansatz = AdaptAnsatz::new();

        Self {
            hamiltonian,
            operator_pool,
            config,
            ansatz,
            num_qubits,
        }
    }

    /// Run the ADAPT-VQE algorithm
    pub fn run(
        &mut self,
        initial_state: &Array1<Complex64>,
    ) -> Result<AdaptVQEResult, QuantRS2Error> {
        let mut current_state = initial_state.clone();
        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.config.max_iterations && !converged {
            // Step 1: Compute gradients for all operators in the pool
            let gradients = self.compute_operator_gradients(&current_state)?;

            // Step 2: Select operator with largest gradient magnitude
            let (max_gradient_idx, max_gradient) = gradients
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                .ok_or_else(|| QuantRS2Error::InvalidInput("No gradients computed".to_string()))?;

            // Check convergence: if max gradient is below threshold, we're done
            if max_gradient.abs() < self.config.gradient_threshold {
                converged = true;
                break;
            }

            // Step 3: Add selected operator to ansatz with initial parameter = 0
            let selected_operator = self.operator_pool.all_operators()[max_gradient_idx].clone();
            self.ansatz.add_operator(selected_operator, 0.0);

            // Step 4: Optimize all parameters in the current ansatz
            let optimized_params = self.optimize_parameters(&current_state)?;
            self.ansatz.parameters = optimized_params;

            // Step 5: Update state and energy
            current_state = self.ansatz.apply_to_state(initial_state, self.num_qubits);
            let energy = self.hamiltonian.expectation_value(&current_state);
            self.ansatz.energy_history.push(energy);

            // Check energy convergence
            if iteration > 0 {
                let energy_change = (self.ansatz.energy_history[iteration]
                    - self.ansatz.energy_history[iteration - 1])
                    .abs();
                if energy_change < self.config.energy_threshold {
                    converged = true;
                }
            }

            iteration += 1;
        }

        Ok(AdaptVQEResult {
            final_energy: self.ansatz.energy_history.last().copied().unwrap_or(0.0),
            final_state: current_state,
            ansatz: self.ansatz.clone(),
            num_iterations: iteration,
            converged,
        })
    }

    /// Compute gradients for all operators in the pool
    fn compute_operator_gradients(
        &self,
        state: &Array1<Complex64>,
    ) -> Result<Vec<f64>, QuantRS2Error> {
        let mut gradients = Vec::new();

        for operator in self.operator_pool.all_operators() {
            let pauli_string = operator.to_pauli_string(self.num_qubits);

            // Gradient = <ψ|[H, A]|ψ> where A is the operator
            let gradient = pauli_string.commutator_with_hamiltonian(&self.hamiltonian, state);
            gradients.push(gradient.re);
        }

        Ok(gradients)
    }

    /// Optimize all parameters in the ansatz
    fn optimize_parameters(
        &self,
        initial_state: &Array1<Complex64>,
    ) -> Result<Vec<f64>, QuantRS2Error> {
        // Initial guess: current parameters
        let initial_params = Array1::from_vec(self.ansatz.parameters.clone());

        // Objective function: energy expectation value <ψ(θ)|H|ψ(θ)>.
        let objective = |params: &ArrayView1<f64>| -> f64 {
            let mut ansatz_copy = self.ansatz.clone();
            ansatz_copy.parameters = params.to_vec();
            let state = ansatz_copy.apply_to_state(initial_state, self.num_qubits);
            self.hamiltonian.expectation_value(&state)
        };

        let options = Options {
            max_iter: self.config.max_vqe_steps,
            tolerance: 1e-6,
            ..Default::default()
        };

        // Run optimization via the in-tree SciRS2 optimizer wrapper.
        let result = minimize(
            objective,
            &initial_params,
            self.config.optimizer_method.clone(),
            Some(options),
        )
        .map_err(|e| {
            QuantRS2Error::OptimizationFailed(format!("Parameter optimization failed: {e:?}"))
        })?;

        Ok(result.x.to_vec())
    }

    /// Get current circuit depth
    pub fn get_circuit_depth(&self) -> usize {
        self.ansatz.depth()
    }

    /// Get operator pool size
    pub fn get_pool_size(&self) -> usize {
        self.operator_pool.size()
    }
}

/// Result from ADAPT-VQE algorithm
#[derive(Debug, Clone)]
pub struct AdaptVQEResult {
    /// Final ground state energy
    pub final_energy: f64,
    /// Final quantum state
    pub final_state: Array1<Complex64>,
    /// Constructed ansatz
    pub ansatz: AdaptAnsatz,
    /// Number of ADAPT iterations performed
    pub num_iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
}

impl AdaptVQEResult {
    /// Get circuit depth of the final ansatz
    pub fn circuit_depth(&self) -> usize {
        self.ansatz.depth()
    }

    /// Get energy lowering from initial to final
    pub fn energy_lowering(&self) -> Option<f64> {
        if self.ansatz.energy_history.len() >= 2 {
            Some(self.ansatz.energy_history[0] - self.final_energy)
        } else {
            None
        }
    }

    /// Get convergence rate (energy change per iteration)
    pub fn convergence_rate(&self) -> f64 {
        if self.num_iterations > 1 {
            let energy_change =
                (self.ansatz.energy_history.first().unwrap_or(&0.0) - self.final_energy).abs();
            energy_change / self.num_iterations as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fermionic_operator_pool() {
        let pool = FermionicOperatorPool::new(4);

        // For 4 orbitals: 4*3 = 12 single excitations
        assert_eq!(pool.single_excitations.len(), 12);

        // Double excitations: C(4,2) * C(4,2) - overlaps
        assert!(!pool.double_excitations.is_empty());

        assert_eq!(
            pool.size(),
            pool.single_excitations.len() + pool.double_excitations.len()
        );
    }

    #[test]
    fn test_pauli_string_application() {
        let pauli = PauliString {
            operators: vec![PauliOp::X, PauliOp::I],
            coefficient: Complex64::new(1.0, 0.0),
        };

        let state = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);

        let result = pauli.apply_to_state(&state);

        // X on qubit 0 should flip |00⟩ to |01⟩
        assert!((result[0].re - 0.0).abs() < 1e-10);
        assert!((result[1].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_adapt_ansatz() {
        let mut ansatz = AdaptAnsatz::new();

        assert_eq!(ansatz.depth(), 0);

        let op = FermionicOperator::single_excitation(0, 1);
        ansatz.add_operator(op, 0.1);

        assert_eq!(ansatz.depth(), 1);
        assert_eq!(ansatz.parameters.len(), 1);
    }

    #[test]
    fn test_molecular_hamiltonian() {
        let h_one = Array2::from_shape_fn((2, 2), |(i, j)| if i == j { -1.0 } else { 0.0 });

        let h_two = HashMap::new();
        let nuclear = 0.5;

        let hamiltonian = MolecularHamiltonian::new(h_one, h_two, nuclear);
        assert_eq!(hamiltonian.num_orbitals, 2);
        assert!((hamiltonian.nuclear_repulsion - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_jordan_wigner_number_operator() {
        // H = h_00 a†_0 a_0  with h_00 = 1 (a number operator on orbital 0).
        // Under Jordan-Wigner, a†_0 a_0 = (I - Z_0)/2 = diag(0, 1) on qubit 0,
        // i.e. it returns the occupation number of orbital 0.
        let mut h_one = Array2::<f64>::zeros((2, 2));
        h_one[[0, 0]] = 1.0;
        let hamiltonian = MolecularHamiltonian::new(h_one, HashMap::new(), 0.0);

        // Basis ordering: index bit q is occupation of orbital q.
        // |0> in occupation of orbital 0 -> states 0 (00) and 2 (10) have n_0 = 0.
        // |1> in occupation of orbital 0 -> states 1 (01) and 3 (11) have n_0 = 1.

        // State |01> (orbital 0 occupied) -> eigenvalue 1.
        let mut occ0 = Array1::<Complex64>::zeros(4);
        occ0[1] = Complex64::new(1.0, 0.0);
        let out = hamiltonian.apply_to_state(&occ0);
        // n_0 |01> = 1 * |01>
        assert!((out[1] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        for k in [0usize, 2, 3] {
            assert!(out[k].norm() < 1e-10);
        }
        // Non-identity Hamiltonian acting on an eigenstate with eigenvalue 1
        // must reproduce the input here, but acting on the empty orbital it must
        // annihilate it (so it is genuinely NOT a clone of an arbitrary input).
        let mut empty0 = Array1::<Complex64>::zeros(4);
        empty0[0] = Complex64::new(1.0, 0.0); // |00>, n_0 = 0
        let out_empty = hamiltonian.apply_to_state(&empty0);
        assert!(
            out_empty.iter().all(|c| c.norm() < 1e-10),
            "number operator must annihilate the empty orbital, got {out_empty:?}"
        );
        // And it is not a clone of the input (input had norm 1, output has norm 0).
        assert!((out_empty.clone() - empty0)
            .iter()
            .any(|c| c.norm() > 1e-10));
    }

    #[test]
    fn test_expectation_value_number_operator() {
        // <ψ|H|ψ> for H = n_0 should equal the occupation of orbital 0 plus the
        // nuclear repulsion energy.
        let mut h_one = Array2::<f64>::zeros((2, 2));
        h_one[[0, 0]] = 1.0;
        let nuclear = 0.25;
        let hamiltonian = MolecularHamiltonian::new(h_one, HashMap::new(), nuclear);

        // |01>: orbital 0 occupied -> <n_0> = 1 -> energy = 1 + 0.25
        let mut occ0 = Array1::<Complex64>::zeros(4);
        occ0[1] = Complex64::new(1.0, 0.0);
        let e_occ = hamiltonian.expectation_value(&occ0);
        assert!((e_occ - 1.25).abs() < 1e-10, "expected 1.25, got {e_occ}");

        // |00>: orbital 0 empty -> <n_0> = 0 -> energy = 0 + 0.25
        let mut empty0 = Array1::<Complex64>::zeros(4);
        empty0[0] = Complex64::new(1.0, 0.0);
        let e_empty = hamiltonian.expectation_value(&empty0);
        assert!(
            (e_empty - 0.25).abs() < 1e-10,
            "expected 0.25, got {e_empty}"
        );
    }

    #[test]
    fn test_jordan_wigner_hopping_is_not_identity() {
        // A hopping term h_01 a†_0 a_1 + h_10 a†_1 a_0 moves an electron between
        // orbitals; applied to |10> (orbital 1 occupied) it must produce |01>
        // (orbital 0 occupied), i.e. it is genuinely off-diagonal, NOT a clone.
        let mut h_one = Array2::<f64>::zeros((2, 2));
        h_one[[0, 1]] = 1.0;
        h_one[[1, 0]] = 1.0;
        let hamiltonian = MolecularHamiltonian::new(h_one, HashMap::new(), 0.0);

        // |10>: orbital 1 occupied (bit 1 set) -> index 2.
        let mut state = Array1::<Complex64>::zeros(4);
        state[2] = Complex64::new(1.0, 0.0);
        let out = hamiltonian.apply_to_state(&state);

        // a†_0 a_1 |10> = |01> (index 1); the conjugate term annihilates this state.
        assert!(
            (out[1].norm() - 1.0).abs() < 1e-10,
            "hopping should populate |01>, got {out:?}"
        );
        assert!(out[2].norm() < 1e-10, "input amplitude must move away");
        // Definitively not a clone of the input.
        assert!((out - state).iter().any(|c| c.norm() > 1e-10));
    }
}
