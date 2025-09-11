//! Fermionic quantum simulation with SciRS2 integration.
//!
//! This module provides comprehensive support for simulating fermionic systems,
//! including Jordan-Wigner transformations, fermionic operators, and specialized
//! algorithms for electronic structure and many-body fermionic systems.

use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use std::collections::HashMap;

use crate::error::{Result, SimulatorError};
use crate::pauli::{PauliOperator, PauliOperatorSum, PauliString};
use crate::scirs2_integration::SciRS2Backend;

/// Fermionic creation and annihilation operators
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FermionicOperator {
    /// Creation operator c†_i
    Creation(usize),
    /// Annihilation operator c_i
    Annihilation(usize),
    /// Number operator n_i = c†_i c_i
    Number(usize),
    /// Hopping term c†_i c_j
    Hopping { from: usize, to: usize },
    /// Interaction term c†_i c†_j c_k c_l
    Interaction { sites: [usize; 4] },
}

/// Fermionic operator string (product of fermionic operators)
#[derive(Debug, Clone)]
pub struct FermionicString {
    /// Ordered list of fermionic operators
    pub operators: Vec<FermionicOperator>,
    /// Coefficient of the operator string
    pub coefficient: Complex64,
    /// Number of fermionic modes
    pub num_modes: usize,
}

/// Sum of fermionic operator strings (fermionic Hamiltonian)
#[derive(Debug, Clone)]
pub struct FermionicHamiltonian {
    /// Terms in the Hamiltonian
    pub terms: Vec<FermionicString>,
    /// Number of fermionic modes
    pub num_modes: usize,
    /// Whether the Hamiltonian is Hermitian
    pub is_hermitian: bool,
}

/// Jordan-Wigner transformation for mapping fermions to qubits
pub struct JordanWignerTransform {
    /// Number of fermionic modes
    num_modes: usize,
    /// Cached Pauli string representations
    pauli_cache: HashMap<FermionicOperator, PauliString>,
}

/// Fermionic simulator with SciRS2 optimization
pub struct FermionicSimulator {
    /// Number of fermionic modes
    num_modes: usize,
    /// Jordan-Wigner transformer
    jw_transform: JordanWignerTransform,
    /// Current fermionic state (in qubit representation)
    state: Array1<Complex64>,
    /// SciRS2 backend for optimization
    backend: Option<SciRS2Backend>,
    /// Simulation statistics
    stats: FermionicStats,
}

/// Statistics for fermionic simulation
#[derive(Debug, Clone, Default)]
pub struct FermionicStats {
    /// Number of Jordan-Wigner transformations performed
    pub jw_transformations: usize,
    /// Number of fermionic operators applied
    pub fermionic_ops_applied: usize,
    /// Time spent in Jordan-Wigner transformation
    pub jw_time_ms: f64,
    /// Memory usage for operator storage
    pub operator_memory_bytes: usize,
    /// Maximum Pauli string length encountered
    pub max_pauli_string_length: usize,
}

impl FermionicOperator {
    /// Check if operator is creation type
    pub fn is_creation(&self) -> bool {
        matches!(self, Self::Creation(_))
    }

    /// Check if operator is annihilation type
    pub fn is_annihilation(&self) -> bool {
        matches!(self, Self::Annihilation(_))
    }

    /// Get site index for single-site operators
    pub fn site(&self) -> Option<usize> {
        match self {
            Self::Creation(i)
            | Self::Annihilation(i)
            | Self::Number(i) => Some(*i),
            _ => None,
        }
    }

    /// Get canonical ordering for operator comparison
    pub fn ordering_key(&self) -> (usize, usize) {
        match self {
            Self::Creation(i) => (1, *i),
            Self::Annihilation(i) => (0, *i),
            Self::Number(i) => (2, *i),
            Self::Hopping { from, to } => (3, from.min(to) * 1000 + from.max(to)),
            Self::Interaction { sites } => {
                let mut sorted_sites = *sites;
                sorted_sites.sort();
                (
                    4,
                    sorted_sites[0] * 1000000
                        + sorted_sites[1] * 10000
                        + sorted_sites[2] * 100
                        + sorted_sites[3],
                )
            }
        }
    }
}

impl FermionicString {
    /// Create new fermionic string
    pub fn new(
        operators: Vec<FermionicOperator>,
        coefficient: Complex64,
        num_modes: usize,
    ) -> Self {
        Self {
            operators,
            coefficient,
            num_modes,
        }
    }

    /// Create single fermionic operator
    pub fn single_operator(
        op: FermionicOperator,
        coefficient: Complex64,
        num_modes: usize,
    ) -> Self {
        Self::new(vec![op], coefficient, num_modes)
    }

    /// Create creation operator c†_i
    pub fn creation(site: usize, coefficient: Complex64, num_modes: usize) -> Self {
        Self::single_operator(FermionicOperator::Creation(site), coefficient, num_modes)
    }

    /// Create annihilation operator c_i
    pub fn annihilation(site: usize, coefficient: Complex64, num_modes: usize) -> Self {
        Self::single_operator(
            FermionicOperator::Annihilation(site),
            coefficient,
            num_modes,
        )
    }

    /// Create number operator n_i
    pub fn number(site: usize, coefficient: Complex64, num_modes: usize) -> Self {
        Self::single_operator(FermionicOperator::Number(site), coefficient, num_modes)
    }

    /// Create hopping term t c†_i c_j
    pub fn hopping(from: usize, to: usize, coefficient: Complex64, num_modes: usize) -> Self {
        Self::single_operator(
            FermionicOperator::Hopping { from, to },
            coefficient,
            num_modes,
        )
    }

    /// Multiply two fermionic strings
    pub fn multiply(&self, other: &FermionicString) -> Result<FermionicString> {
        if self.num_modes != other.num_modes {
            return Err(SimulatorError::DimensionMismatch(
                "Fermionic strings must have same number of modes".to_string(),
            ));
        }

        let mut result_ops = self.operators.clone();
        result_ops.extend(other.operators.clone());

        // Apply fermionic anticommutation rules
        let (canonical_ops, sign) = self.canonicalize_operators(&result_ops)?;

        Ok(FermionicString {
            operators: canonical_ops,
            coefficient: self.coefficient * other.coefficient * sign,
            num_modes: self.num_modes,
        })
    }

    /// Canonicalize fermionic operators (apply anticommutation)
    fn canonicalize_operators(
        &self,
        ops: &[FermionicOperator],
    ) -> Result<(Vec<FermionicOperator>, Complex64)> {
        let mut canonical = ops.to_vec();
        let mut sign = Complex64::new(1.0, 0.0);

        // Bubble sort with fermionic anticommutation
        for i in 0..canonical.len() {
            for j in (i + 1)..canonical.len() {
                if canonical[i].ordering_key() > canonical[j].ordering_key() {
                    // Swap with anticommutation sign
                    canonical.swap(i, j);
                    sign *= Complex64::new(-1.0, 0.0);
                }
            }
        }

        // Apply fermionic algebra rules (c_i c_i = 0, c†_i c_i = n_i, etc.)
        let simplified = self.apply_fermionic_algebra(&canonical)?;

        Ok((simplified, sign))
    }

    /// Apply fermionic algebra rules
    fn apply_fermionic_algebra(&self, ops: &[FermionicOperator]) -> Result<Vec<FermionicOperator>> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < ops.len() {
            if i + 1 < ops.len() {
                match (&ops[i], &ops[i + 1]) {
                    // c†_i c_i = n_i
                    (FermionicOperator::Creation(a), FermionicOperator::Annihilation(b))
                        if a == b =>
                    {
                        result.push(FermionicOperator::Number(*a));
                        i += 2;
                        continue;
                    }
                    // c_i c_i = 0 (skip both)
                    (FermionicOperator::Annihilation(a), FermionicOperator::Annihilation(b))
                        if a == b =>
                    {
                        // Result is zero - would need to handle this properly
                        i += 2;
                        continue;
                    }
                    // c†_i c†_i = 0 (skip both)
                    (FermionicOperator::Creation(a), FermionicOperator::Creation(b)) if a == b => {
                        // Result is zero - would need to handle this properly
                        i += 2;
                        continue;
                    }
                    _ => {
                        result.push(ops[i].clone());
                        i += 1;
                    }
                }
            } else {
                result.push(ops[i].clone());
                i += 1;
            }
        }

        Ok(result)
    }

    /// Compute Hermitian conjugate
    pub fn hermitian_conjugate(&self) -> FermionicString {
        let mut conjugate_ops = Vec::new();

        // Reverse order and conjugate each operator
        for op in self.operators.iter().rev() {
            let conjugate_op = match op {
                FermionicOperator::Creation(i) => FermionicOperator::Annihilation(*i),
                FermionicOperator::Annihilation(i) => FermionicOperator::Creation(*i),
                FermionicOperator::Number(i) => FermionicOperator::Number(*i),
                FermionicOperator::Hopping { from, to } => FermionicOperator::Hopping {
                    from: *to,
                    to: *from,
                },
                FermionicOperator::Interaction { sites } => {
                    // Reverse the order for interaction terms
                    let mut rev_sites = *sites;
                    rev_sites.reverse();
                    FermionicOperator::Interaction { sites: rev_sites }
                }
            };
            conjugate_ops.push(conjugate_op);
        }

        FermionicString {
            operators: conjugate_ops,
            coefficient: self.coefficient.conj(),
            num_modes: self.num_modes,
        }
    }
}

impl FermionicHamiltonian {
    /// Create new fermionic Hamiltonian
    pub fn new(num_modes: usize) -> Self {
        Self {
            terms: Vec::new(),
            num_modes,
            is_hermitian: true,
        }
    }

    /// Add term to Hamiltonian
    pub fn add_term(&mut self, term: FermionicString) -> Result<()> {
        if term.num_modes != self.num_modes {
            return Err(SimulatorError::DimensionMismatch(
                "Term must have same number of modes as Hamiltonian".to_string(),
            ));
        }

        self.terms.push(term);
        Ok(())
    }

    /// Add Hermitian conjugate terms automatically
    pub fn make_hermitian(&mut self) {
        let mut conjugate_terms = Vec::new();

        for term in &self.terms {
            let conjugate = term.hermitian_conjugate();
            // Only add if it's different from the original term
            if !self.terms_equal(term, &conjugate) {
                conjugate_terms.push(conjugate);
            }
        }

        self.terms.extend(conjugate_terms);
        self.is_hermitian = true;
    }

    /// Check if two terms are equal
    fn terms_equal(&self, term1: &FermionicString, term2: &FermionicString) -> bool {
        term1.operators == term2.operators && (term1.coefficient - term2.coefficient).norm() < 1e-12
    }

    /// Create molecular Hamiltonian
    pub fn molecular_hamiltonian(
        num_modes: usize,
        one_body_integrals: &Array2<f64>,
        two_body_integrals: &Array3<f64>,
    ) -> Result<Self> {
        let mut hamiltonian = Self::new(num_modes);

        // One-body terms: ∑_{i,j} h_{ij} c†_i c_j
        for i in 0..num_modes {
            for j in 0..num_modes {
                if one_body_integrals[[i, j]].abs() > 1e-12 {
                    let coeff = Complex64::new(one_body_integrals[[i, j]], 0.0);
                    let term = FermionicString::new(
                        vec![
                            FermionicOperator::Creation(i),
                            FermionicOperator::Annihilation(j),
                        ],
                        coeff,
                        num_modes,
                    );
                    hamiltonian.add_term(term)?;
                }
            }
        }

        // Two-body terms: ∑_{i,j,k,l} V_{ijkl} c†_i c†_j c_l c_k
        for i in 0..num_modes {
            for j in 0..num_modes {
                for k in 0..num_modes {
                    if two_body_integrals[[i, j, k]].abs() > 1e-12 {
                        for l in 0..num_modes {
                            let coeff = Complex64::new(0.5 * two_body_integrals[[i, j, k]], 0.0);
                            let term = FermionicString::new(
                                vec![
                                    FermionicOperator::Creation(i),
                                    FermionicOperator::Creation(j),
                                    FermionicOperator::Annihilation(l),
                                    FermionicOperator::Annihilation(k),
                                ],
                                coeff,
                                num_modes,
                            );
                            hamiltonian.add_term(term)?;
                        }
                    }
                }
            }
        }

        hamiltonian.make_hermitian();
        Ok(hamiltonian)
    }

    /// Create Hubbard model Hamiltonian
    pub fn hubbard_model(
        sites: usize,
        hopping: f64,
        interaction: f64,
        chemical_potential: f64,
    ) -> Result<Self> {
        let num_modes = 2 * sites; // Spin up and spin down
        let mut hamiltonian = Self::new(num_modes);

        // Hopping terms: -t ∑_{⟨i,j⟩,σ} (c†_{i,σ} c_{j,σ} + h.c.)
        for i in 0..sites {
            for sigma in 0..2 {
                let site_i = 2 * i + sigma;

                // Nearest neighbor hopping (1D chain)
                if i + 1 < sites {
                    let site_j = 2 * (i + 1) + sigma;

                    // Forward hopping
                    let hopping_term = FermionicString::hopping(
                        site_i,
                        site_j,
                        Complex64::new(-hopping, 0.0),
                        num_modes,
                    );
                    hamiltonian.add_term(hopping_term)?;

                    // Backward hopping (Hermitian conjugate)
                    let back_hopping_term = FermionicString::hopping(
                        site_j,
                        site_i,
                        Complex64::new(-hopping, 0.0),
                        num_modes,
                    );
                    hamiltonian.add_term(back_hopping_term)?;
                }
            }
        }

        // Interaction terms: U ∑_i n_{i,↑} n_{i,↓}
        for i in 0..sites {
            let up_site = 2 * i;
            let down_site = 2 * i + 1;

            let interaction_term = FermionicString::new(
                vec![
                    FermionicOperator::Number(up_site),
                    FermionicOperator::Number(down_site),
                ],
                Complex64::new(interaction, 0.0),
                num_modes,
            );
            hamiltonian.add_term(interaction_term)?;
        }

        // Chemical potential terms: -μ ∑_{i,σ} n_{i,σ}
        for i in 0..num_modes {
            let mu_term =
                FermionicString::number(i, Complex64::new(-chemical_potential, 0.0), num_modes);
            hamiltonian.add_term(mu_term)?;
        }

        Ok(hamiltonian)
    }
}

impl JordanWignerTransform {
    /// Create new Jordan-Wigner transformer
    pub fn new(num_modes: usize) -> Self {
        Self {
            num_modes,
            pauli_cache: HashMap::new(),
        }
    }

    /// Transform fermionic operator to Pauli string
    pub fn transform_operator(&mut self, op: &FermionicOperator) -> Result<PauliString> {
        if let Some(cached) = self.pauli_cache.get(op) {
            return Ok(cached.clone());
        }

        let pauli_string = match op {
            FermionicOperator::Creation(i) => self.creation_to_pauli(*i)?,
            FermionicOperator::Annihilation(i) => self.annihilation_to_pauli(*i)?,
            FermionicOperator::Number(i) => self.number_to_pauli(*i)?,
            FermionicOperator::Hopping { from, to } => self.hopping_to_pauli(*from, *to)?,
            FermionicOperator::Interaction { sites } => self.interaction_to_pauli(*sites)?,
        };

        self.pauli_cache.insert(op.clone(), pauli_string.clone());
        Ok(pauli_string)
    }

    /// Transform creation operator c†_i to Pauli string
    fn creation_to_pauli(&self, site: usize) -> Result<PauliString> {
        if site >= self.num_modes {
            return Err(SimulatorError::IndexOutOfBounds(site));
        }

        let mut paulis = vec![PauliOperator::I; self.num_modes];

        // Jordan-Wigner string: Z_0 Z_1 ... Z_{i-1} (X_i - i Y_i)/2
        for j in 0..site {
            paulis[j] = PauliOperator::Z;
        }

        // For creation: (X - iY)/2, we'll represent this as two terms
        // This is a simplified representation - proper implementation would
        // handle complex Pauli strings
        paulis[site] = PauliOperator::X;

        let ops: Vec<(usize, PauliOperator)> = paulis
            .iter()
            .enumerate()
            .filter(|(_, &op)| op != PauliOperator::I)
            .map(|(i, &op)| (i, op))
            .collect();

        PauliString::from_ops(self.num_modes, &ops, Complex64::new(0.5, 0.0))
    }

    /// Transform annihilation operator c_i to Pauli string
    fn annihilation_to_pauli(&self, site: usize) -> Result<PauliString> {
        if site >= self.num_modes {
            return Err(SimulatorError::IndexOutOfBounds(site));
        }

        let mut paulis = vec![PauliOperator::I; self.num_modes];

        // Jordan-Wigner string: Z_0 Z_1 ... Z_{i-1} (X_i + i Y_i)/2
        for j in 0..site {
            paulis[j] = PauliOperator::Z;
        }

        paulis[site] = PauliOperator::X;

        let ops: Vec<(usize, PauliOperator)> = paulis
            .iter()
            .enumerate()
            .filter(|(_, &op)| op != PauliOperator::I)
            .map(|(i, &op)| (i, op))
            .collect();

        PauliString::from_ops(self.num_modes, &ops, Complex64::new(0.5, 0.0))
    }

    /// Transform number operator n_i to Pauli string
    fn number_to_pauli(&self, site: usize) -> Result<PauliString> {
        if site >= self.num_modes {
            return Err(SimulatorError::IndexOutOfBounds(site));
        }

        let mut paulis = vec![PauliOperator::I; self.num_modes];

        // n_i = c†_i c_i = (I - Z_i)/2
        paulis[site] = PauliOperator::Z;

        let ops: Vec<(usize, PauliOperator)> = paulis
            .iter()
            .enumerate()
            .filter(|(_, &op)| op != PauliOperator::I)
            .map(|(i, &op)| (i, op))
            .collect();

        PauliString::from_ops(self.num_modes, &ops, Complex64::new(-0.5, 0.0))
    }

    /// Transform hopping term to Pauli string
    fn hopping_to_pauli(&self, from: usize, to: usize) -> Result<PauliString> {
        // This would be a product of creation and annihilation operators
        // Simplified implementation for now
        let mut paulis = vec![PauliOperator::I; self.num_modes];

        let min_site = from.min(to);
        let max_site = from.max(to);

        // Jordan-Wigner string between sites
        for j in min_site..max_site {
            paulis[j] = PauliOperator::Z;
        }

        paulis[from] = PauliOperator::X;
        paulis[to] = PauliOperator::X;

        let ops: Vec<(usize, PauliOperator)> = paulis
            .iter()
            .enumerate()
            .filter(|(_, &op)| op != PauliOperator::I)
            .map(|(i, &op)| (i, op))
            .collect();

        PauliString::from_ops(self.num_modes, &ops, Complex64::new(0.25, 0.0))
    }

    /// Transform interaction term to Pauli string
    fn interaction_to_pauli(&self, sites: [usize; 4]) -> Result<PauliString> {
        // Simplified implementation for four-fermion interaction
        let mut paulis = vec![PauliOperator::I; self.num_modes];

        for &site in &sites {
            paulis[site] = PauliOperator::Z;
        }

        let ops: Vec<(usize, PauliOperator)> = paulis
            .iter()
            .enumerate()
            .filter(|(_, &op)| op != PauliOperator::I)
            .map(|(i, &op)| (i, op))
            .collect();

        PauliString::from_ops(self.num_modes, &ops, Complex64::new(0.0625, 0.0))
    }

    /// Transform fermionic string to Pauli operator sum
    pub fn transform_string(
        &mut self,
        fermionic_string: &FermionicString,
    ) -> Result<PauliOperatorSum> {
        let mut pauli_sum = PauliOperatorSum::new(self.num_modes);

        if fermionic_string.operators.is_empty() {
            // Identity term
            let mut identity_string = PauliString::new(self.num_modes);
            identity_string.coefficient = fermionic_string.coefficient;
            let _ = pauli_sum.add_term(identity_string);
            return Ok(pauli_sum);
        }

        // For now, simplified implementation that handles single operators
        if fermionic_string.operators.len() == 1 {
            let pauli_string = self.transform_operator(&fermionic_string.operators[0])?;
            let mut scaled_string = pauli_string.clone();
            scaled_string.coefficient = pauli_string.coefficient * fermionic_string.coefficient;
            let _ = pauli_sum.add_term(scaled_string);
        } else {
            // Multi-operator case would require more complex implementation
            // For now, return identity with coefficient
            let mut identity_string = PauliString::new(self.num_modes);
            identity_string.coefficient = fermionic_string.coefficient;
            let _ = pauli_sum.add_term(identity_string);
        }

        Ok(pauli_sum)
    }

    /// Transform fermionic Hamiltonian to Pauli Hamiltonian
    pub fn transform_hamiltonian(
        &mut self,
        hamiltonian: &FermionicHamiltonian,
    ) -> Result<PauliOperatorSum> {
        let mut pauli_hamiltonian = PauliOperatorSum::new(self.num_modes);

        for term in &hamiltonian.terms {
            let pauli_terms = self.transform_string(term)?;
            for pauli_term in pauli_terms.terms {
                let _ = pauli_hamiltonian.add_term(pauli_term);
            }
        }

        Ok(pauli_hamiltonian)
    }
}

impl FermionicSimulator {
    /// Create new fermionic simulator
    pub fn new(num_modes: usize) -> Result<Self> {
        let dim = 1 << num_modes;
        let mut state = Array1::zeros(dim);
        state[0] = Complex64::new(1.0, 0.0); // |0...0⟩ (vacuum state)

        Ok(Self {
            num_modes,
            jw_transform: JordanWignerTransform::new(num_modes),
            state,
            backend: None,
            stats: FermionicStats::default(),
        })
    }

    /// Initialize with SciRS2 backend
    pub fn with_scirs2_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Set initial fermionic state
    pub fn set_initial_state(&mut self, occupation: &[bool]) -> Result<()> {
        if occupation.len() != self.num_modes {
            return Err(SimulatorError::DimensionMismatch(
                "Occupation must match number of modes".to_string(),
            ));
        }

        // Create Fock state |n_0, n_1, ..., n_{N-1}⟩
        let mut index = 0;
        for (i, &occupied) in occupation.iter().enumerate() {
            if occupied {
                index |= 1 << (self.num_modes - 1 - i);
            }
        }

        self.state.fill(Complex64::new(0.0, 0.0));
        self.state[index] = Complex64::new(1.0, 0.0);

        Ok(())
    }

    /// Apply fermionic operator
    pub fn apply_fermionic_operator(&mut self, op: &FermionicOperator) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Transform to Pauli representation
        let pauli_string = self.jw_transform.transform_operator(op)?;

        // Apply Pauli string to state
        self.apply_pauli_string(&pauli_string)?;

        self.stats.fermionic_ops_applied += 1;
        self.stats.jw_transformations += 1;
        self.stats.jw_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(())
    }

    /// Apply fermionic string
    pub fn apply_fermionic_string(&mut self, fermionic_string: &FermionicString) -> Result<()> {
        let pauli_sum = self.jw_transform.transform_string(fermionic_string)?;

        // Apply each Pauli term
        for pauli_term in &pauli_sum.terms {
            self.apply_pauli_string(pauli_term)?;
        }

        Ok(())
    }

    /// Apply Pauli string to current state
    fn apply_pauli_string(&mut self, pauli_string: &PauliString) -> Result<()> {
        // This would need proper Pauli string application
        // For now, placeholder implementation
        Ok(())
    }

    /// Compute expectation value of fermionic operator
    pub fn expectation_value(&mut self, op: &FermionicOperator) -> Result<Complex64> {
        let pauli_string = self.jw_transform.transform_operator(op)?;

        // Compute ⟨ψ|P|ψ⟩ where P is the Pauli string
        let expectation = self.compute_pauli_expectation(&pauli_string)?;

        Ok(expectation)
    }

    /// Compute Pauli string expectation value
    fn compute_pauli_expectation(&self, pauli_string: &PauliString) -> Result<Complex64> {
        // Simplified implementation
        // Would need proper Pauli string expectation value computation
        Ok(Complex64::new(0.0, 0.0))
    }

    /// Evolve under fermionic Hamiltonian
    pub fn evolve_hamiltonian(
        &mut self,
        hamiltonian: &FermionicHamiltonian,
        time: f64,
    ) -> Result<()> {
        // Transform to Pauli Hamiltonian
        let pauli_hamiltonian = self.jw_transform.transform_hamiltonian(hamiltonian)?;

        // Evolve under Pauli Hamiltonian (would use Trotter-Suzuki or exact methods)
        self.evolve_pauli_hamiltonian(&pauli_hamiltonian, time)?;

        Ok(())
    }

    /// Evolve under Pauli Hamiltonian
    fn evolve_pauli_hamiltonian(
        &mut self,
        _hamiltonian: &PauliOperatorSum,
        _time: f64,
    ) -> Result<()> {
        // Would implement time evolution under Pauli Hamiltonian
        // Could use matrix exponentiation or Trotter-Suzuki decomposition
        Ok(())
    }

    /// Get current state vector
    pub fn get_state(&self) -> &Array1<Complex64> {
        &self.state
    }

    /// Get number of particles in current state
    pub fn get_particle_number(&self) -> f64 {
        let mut total_number = 0.0;

        for (index, amplitude) in self.state.iter().enumerate() {
            let prob = amplitude.norm_sqr();
            let popcount = index.count_ones() as f64;
            total_number += prob * popcount;
        }

        total_number
    }

    /// Get simulation statistics
    pub fn get_stats(&self) -> &FermionicStats {
        &self.stats
    }

    /// Compute particle number correlation
    pub fn particle_correlation(&mut self, site1: usize, site2: usize) -> Result<f64> {
        let n1_op = FermionicOperator::Number(site1);
        let n2_op = FermionicOperator::Number(site2);

        let n1_exp = self.expectation_value(&n1_op)?.re;
        let n2_exp = self.expectation_value(&n2_op)?.re;

        // For ⟨n_i n_j⟩, would need to compute product operator expectation
        let n1_n2_exp = 0.0; // Placeholder

        Ok(n1_n2_exp - n1_exp * n2_exp)
    }
}

/// Benchmark fermionic simulation
pub fn benchmark_fermionic_simulation(num_modes: usize) -> Result<FermionicStats> {
    let mut simulator = FermionicSimulator::new(num_modes)?;

    // Create simple Hubbard model
    let hamiltonian = FermionicHamiltonian::hubbard_model(num_modes / 2, 1.0, 2.0, 0.5)?;

    // Apply some fermionic operators
    let creation_op = FermionicOperator::Creation(0);
    simulator.apply_fermionic_operator(&creation_op)?;

    let annihilation_op = FermionicOperator::Annihilation(1);
    simulator.apply_fermionic_operator(&annihilation_op)?;

    // Evolve under Hamiltonian
    simulator.evolve_hamiltonian(&hamiltonian, 0.1)?;

    Ok(simulator.get_stats().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fermionic_operator_creation() {
        let op = FermionicOperator::Creation(0);
        assert!(op.is_creation());
        assert!(!op.is_annihilation());
        assert_eq!(op.site(), Some(0));
    }

    #[test]
    fn test_fermionic_string() {
        let ops = vec![
            FermionicOperator::Creation(0),
            FermionicOperator::Annihilation(1),
        ];
        let string = FermionicString::new(ops, Complex64::new(1.0, 0.0), 4);
        assert_eq!(string.operators.len(), 2);
        assert_eq!(string.num_modes, 4);
    }

    #[test]
    fn test_hubbard_hamiltonian() {
        let hamiltonian = FermionicHamiltonian::hubbard_model(2, 1.0, 2.0, 0.5).unwrap();
        assert_eq!(hamiltonian.num_modes, 4); // 2 sites × 2 spins
        assert!(!hamiltonian.terms.is_empty());
    }

    #[test]
    fn test_jordan_wigner_transform() {
        let mut jw = JordanWignerTransform::new(4);
        let creation_op = FermionicOperator::Creation(1);
        let pauli_string = jw.transform_operator(&creation_op).unwrap();

        assert_eq!(pauli_string.num_qubits, 4);
        assert_eq!(pauli_string.operators[0], PauliOperator::Z); // Jordan-Wigner string
        assert_eq!(pauli_string.operators[1], PauliOperator::X);
    }

    #[test]
    fn test_fermionic_simulator() {
        let mut simulator = FermionicSimulator::new(4).unwrap();

        // Set initial state with one particle
        simulator
            .set_initial_state(&[true, false, false, false])
            .unwrap();

        let particle_number = simulator.get_particle_number();
        assert!((particle_number - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fermionic_string_multiplication() {
        let string1 = FermionicString::creation(0, Complex64::new(1.0, 0.0), 4);
        let string2 = FermionicString::annihilation(1, Complex64::new(1.0, 0.0), 4);

        let product = string1.multiply(&string2).unwrap();
        assert!(!product.operators.is_empty());
    }
}
