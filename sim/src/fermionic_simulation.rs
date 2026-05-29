//! Fermionic quantum simulation with `SciRS2` integration.
//!
//! This module provides comprehensive support for simulating fermionic systems,
//! including Jordan-Wigner transformations, fermionic operators, and specialized
//! algorithms for electronic structure and many-body fermionic systems.

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::Complex64;
use std::collections::HashMap;

use crate::error::{Result, SimulatorError};
use crate::pauli::{PauliOperator, PauliOperatorSum, PauliString};
use crate::scirs2_integration::SciRS2Backend;

/// Fermionic creation and annihilation operators
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FermionicOperator {
    /// Creation operator c†_i
    Creation(usize),
    /// Annihilation operator `c_i`
    Annihilation(usize),
    /// Number operator `n_i` = c†_i `c_i`
    Number(usize),
    /// Hopping term c†_i `c_j`
    Hopping { from: usize, to: usize },
    /// Interaction term c†_i c†_j `c_k` `c_l`
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

/// Fermionic simulator with `SciRS2` optimization
pub struct FermionicSimulator {
    /// Number of fermionic modes
    num_modes: usize,
    /// Jordan-Wigner transformer
    jw_transform: JordanWignerTransform,
    /// Current fermionic state (in qubit representation)
    state: Array1<Complex64>,
    /// `SciRS2` backend for optimization
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
    #[must_use]
    pub const fn is_creation(&self) -> bool {
        matches!(self, Self::Creation(_))
    }

    /// Check if operator is annihilation type
    #[must_use]
    pub const fn is_annihilation(&self) -> bool {
        matches!(self, Self::Annihilation(_))
    }

    /// Get site index for single-site operators
    #[must_use]
    pub const fn site(&self) -> Option<usize> {
        match self {
            Self::Creation(i) | Self::Annihilation(i) | Self::Number(i) => Some(*i),
            _ => None,
        }
    }

    /// Get canonical ordering for operator comparison
    #[must_use]
    pub fn ordering_key(&self) -> (usize, usize) {
        match self {
            Self::Creation(i) => (1, *i),
            Self::Annihilation(i) => (0, *i),
            Self::Number(i) => (2, *i),
            Self::Hopping { from, to } => (3, from.min(to) * 1000 + from.max(to)),
            Self::Interaction { sites } => {
                let mut sorted_sites = *sites;
                sorted_sites.sort_unstable();
                (
                    4,
                    sorted_sites[0] * 1_000_000
                        + sorted_sites[1] * 10_000
                        + sorted_sites[2] * 100
                        + sorted_sites[3],
                )
            }
        }
    }
}

impl FermionicString {
    /// Create new fermionic string
    #[must_use]
    pub const fn new(
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
    #[must_use]
    pub fn single_operator(
        op: FermionicOperator,
        coefficient: Complex64,
        num_modes: usize,
    ) -> Self {
        Self::new(vec![op], coefficient, num_modes)
    }

    /// Create creation operator c†_i
    #[must_use]
    pub fn creation(site: usize, coefficient: Complex64, num_modes: usize) -> Self {
        Self::single_operator(FermionicOperator::Creation(site), coefficient, num_modes)
    }

    /// Create annihilation operator `c_i`
    #[must_use]
    pub fn annihilation(site: usize, coefficient: Complex64, num_modes: usize) -> Self {
        Self::single_operator(
            FermionicOperator::Annihilation(site),
            coefficient,
            num_modes,
        )
    }

    /// Create number operator `n_i`
    #[must_use]
    pub fn number(site: usize, coefficient: Complex64, num_modes: usize) -> Self {
        Self::single_operator(FermionicOperator::Number(site), coefficient, num_modes)
    }

    /// Create hopping term t c†_i `c_j`
    #[must_use]
    pub fn hopping(from: usize, to: usize, coefficient: Complex64, num_modes: usize) -> Self {
        Self::single_operator(
            FermionicOperator::Hopping { from, to },
            coefficient,
            num_modes,
        )
    }

    /// Multiply two fermionic strings
    pub fn multiply(&self, other: &Self) -> Result<Self> {
        if self.num_modes != other.num_modes {
            return Err(SimulatorError::DimensionMismatch(
                "Fermionic strings must have same number of modes".to_string(),
            ));
        }

        let mut result_ops = self.operators.clone();
        result_ops.extend(other.operators.clone());

        // Apply fermionic anticommutation rules
        let (canonical_ops, sign) = self.canonicalize_operators(&result_ops)?;

        Ok(Self {
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
                    }
                    // c_i c_i = 0 (skip both)
                    (FermionicOperator::Annihilation(a), FermionicOperator::Annihilation(b))
                        if a == b =>
                    {
                        // Result is zero - would need to handle this properly
                        i += 2;
                    }
                    // c†_i c†_i = 0 (skip both)
                    (FermionicOperator::Creation(a), FermionicOperator::Creation(b)) if a == b => {
                        // Result is zero - would need to handle this properly
                        i += 2;
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
    #[must_use]
    pub fn hermitian_conjugate(&self) -> Self {
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

        Self {
            operators: conjugate_ops,
            coefficient: self.coefficient.conj(),
            num_modes: self.num_modes,
        }
    }
}

impl FermionicHamiltonian {
    /// Create new fermionic Hamiltonian
    #[must_use]
    pub const fn new(num_modes: usize) -> Self {
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
    #[must_use]
    pub fn new(num_modes: usize) -> Self {
        Self {
            num_modes,
            pauli_cache: HashMap::new(),
        }
    }

    /// Transform a single fermionic operator to its complete Pauli operator sum via JW.
    ///
    /// Returns the full Jordan-Wigner representation as a `PauliOperatorSum` (possibly
    /// containing multiple terms for operators like creation/annihilation).
    ///
    /// JW mappings (n = num_modes, Z_k below denotes Z on mode k):
    ///
    /// - `n_i = (I − Z_i)/2`           → two Pauli strings (constant + Z_i)
    /// - `c†_i = (Z_0⋯Z_{i-1})(X_i − iY_i)/2` → two Pauli strings
    /// - `c_i  = (Z_0⋯Z_{i-1})(X_i + iY_i)/2` → two Pauli strings
    /// - `Hopping c†_from c_to` via full product    → four raw terms reduced to two
    /// - `Interaction c†⋯c_l` via full product
    pub fn transform_operator_to_sum(
        &mut self,
        op: &FermionicOperator,
    ) -> Result<PauliOperatorSum> {
        let mut sum = PauliOperatorSum::new(self.num_modes);
        match op {
            FermionicOperator::Number(site) => {
                // n_i = (I - Z_i)/2 = 0.5·I - 0.5·Z_i
                let identity = PauliString::new(self.num_modes); // coefficient = 1.0, all I
                let mut id = identity;
                id.coefficient = Complex64::new(0.5, 0.0);
                sum.add_term(id)?;

                let z_term =
                    self.single_site_pauli(*site, PauliOperator::Z, Complex64::new(-0.5, 0.0))?;
                sum.add_term(z_term)?;
            }
            FermionicOperator::Creation(site) => {
                // c†_i = (Z_{0..i} X_i)/2 − i(Z_{0..i} Y_i)/2
                let x_term =
                    self.jw_pauli_string(*site, PauliOperator::X, Complex64::new(0.5, 0.0))?;
                let y_term =
                    self.jw_pauli_string(*site, PauliOperator::Y, Complex64::new(0.0, -0.5))?;
                sum.add_term(x_term)?;
                sum.add_term(y_term)?;
            }
            FermionicOperator::Annihilation(site) => {
                // c_i = (Z_{0..i} X_i)/2 + i(Z_{0..i} Y_i)/2
                let x_term =
                    self.jw_pauli_string(*site, PauliOperator::X, Complex64::new(0.5, 0.0))?;
                let y_term =
                    self.jw_pauli_string(*site, PauliOperator::Y, Complex64::new(0.0, 0.5))?;
                sum.add_term(x_term)?;
                sum.add_term(y_term)?;
            }
            FermionicOperator::Hopping { from, to } => {
                // c†_from c_to: multiply the two full JW sums
                let creation_sum =
                    self.transform_operator_to_sum(&FermionicOperator::Creation(*from))?;
                let annihilation_sum =
                    self.transform_operator_to_sum(&FermionicOperator::Annihilation(*to))?;
                for ca in &creation_sum.terms {
                    for an in &annihilation_sum.terms {
                        let product = ca.multiply(an)?;
                        // Skip near-zero terms
                        if product.coefficient.norm() > 1e-15 {
                            sum.add_term(product)?;
                        }
                    }
                }
            }
            FermionicOperator::Interaction { sites } => {
                // c†_{s0} c†_{s1} c_{s2} c_{s3}: full product
                let sums: Vec<PauliOperatorSum> = [
                    FermionicOperator::Creation(sites[0]),
                    FermionicOperator::Creation(sites[1]),
                    FermionicOperator::Annihilation(sites[2]),
                    FermionicOperator::Annihilation(sites[3]),
                ]
                .iter()
                .map(|fop| self.transform_operator_to_sum(fop))
                .collect::<Result<Vec<_>>>()?;

                // Iteratively multiply all operator sums
                let mut current: Vec<PauliString> = sums[0].terms.clone();
                for next_sum in sums.iter().skip(1) {
                    let mut new_current = Vec::new();
                    for ca in &current {
                        for nb in &next_sum.terms {
                            let product = ca.multiply(nb)?;
                            if product.coefficient.norm() > 1e-15 {
                                new_current.push(product);
                            }
                        }
                    }
                    current = new_current;
                }
                for term in current {
                    sum.add_term(term)?;
                }
            }
        }
        Ok(sum)
    }

    /// Build a JW Pauli string: Z_{0}⋯Z_{site-1} op_{site} I_{site+1}⋯ with given coefficient.
    fn jw_pauli_string(
        &self,
        site: usize,
        op: PauliOperator,
        coeff: Complex64,
    ) -> Result<PauliString> {
        if site >= self.num_modes {
            return Err(SimulatorError::IndexOutOfBounds(site));
        }
        let mut paulis = vec![PauliOperator::I; self.num_modes];
        paulis[..site].fill(PauliOperator::Z);
        paulis[site] = op;
        let ops: Vec<(usize, PauliOperator)> = paulis
            .iter()
            .enumerate()
            .filter(|(_, &p)| p != PauliOperator::I)
            .map(|(i, &p)| (i, p))
            .collect();
        PauliString::from_ops(self.num_modes, &ops, coeff)
    }

    /// Build a single-site Pauli string (no JW Z-string): I⋯I op_site I⋯I.
    fn single_site_pauli(
        &self,
        site: usize,
        op: PauliOperator,
        coeff: Complex64,
    ) -> Result<PauliString> {
        if site >= self.num_modes {
            return Err(SimulatorError::IndexOutOfBounds(site));
        }
        PauliString::from_ops(self.num_modes, &[(site, op)], coeff)
    }

    /// Transform fermionic operator to a single representative Pauli string.
    ///
    /// This returns only the X-part of the JW decomposition (for creation/annihilation)
    /// or the Z-only part of the number operator, and is used internally for
    /// Pauli-composition chains.  For complete expectation values use
    /// `transform_operator_to_sum`.
    pub fn transform_operator(&mut self, op: &FermionicOperator) -> Result<PauliString> {
        if let Some(cached) = self.pauli_cache.get(op) {
            return Ok(cached.clone());
        }

        let pauli_string = match op {
            FermionicOperator::Creation(i) => {
                self.jw_pauli_string(*i, PauliOperator::X, Complex64::new(0.5, 0.0))?
            }
            FermionicOperator::Annihilation(i) => {
                self.jw_pauli_string(*i, PauliOperator::X, Complex64::new(0.5, 0.0))?
            }
            FermionicOperator::Number(i) => {
                // Only Z term; identity term handled in transform_operator_to_sum
                self.single_site_pauli(*i, PauliOperator::Z, Complex64::new(-0.5, 0.0))?
            }
            FermionicOperator::Hopping { from, to } => self
                .creation_to_pauli(*from)?
                .multiply(&self.annihilation_to_pauli(*to)?)?,
            FermionicOperator::Interaction { sites } => self
                .creation_to_pauli(sites[0])?
                .multiply(&self.creation_to_pauli(sites[1])?)?
                .multiply(&self.annihilation_to_pauli(sites[2])?)?
                .multiply(&self.annihilation_to_pauli(sites[3])?)?,
        };

        self.pauli_cache.insert(op.clone(), pauli_string.clone());
        Ok(pauli_string)
    }

    /// Transform creation operator c†_i to single Pauli string (X-part of JW).
    fn creation_to_pauli(&self, site: usize) -> Result<PauliString> {
        if site >= self.num_modes {
            return Err(SimulatorError::IndexOutOfBounds(site));
        }
        self.jw_pauli_string(site, PauliOperator::X, Complex64::new(0.5, 0.0))
    }

    /// Transform annihilation operator c_i to single Pauli string (X-part of JW).
    fn annihilation_to_pauli(&self, site: usize) -> Result<PauliString> {
        if site >= self.num_modes {
            return Err(SimulatorError::IndexOutOfBounds(site));
        }
        self.jw_pauli_string(site, PauliOperator::X, Complex64::new(0.5, 0.0))
    }

    /// Transform hopping term c†_from c_to to Pauli string (X-part only).
    fn hopping_to_pauli(&self, from: usize, to: usize) -> Result<PauliString> {
        self.creation_to_pauli(from)?
            .multiply(&self.annihilation_to_pauli(to)?)
    }

    /// Transform four-body interaction c†_{s0} c†_{s1} c_{s2} c_{s3} to Pauli string.
    fn interaction_to_pauli(&self, sites: [usize; 4]) -> Result<PauliString> {
        self.creation_to_pauli(sites[0])?
            .multiply(&self.creation_to_pauli(sites[1])?)?
            .multiply(&self.annihilation_to_pauli(sites[2])?)?
            .multiply(&self.annihilation_to_pauli(sites[3])?)
    }

    /// Transform fermionic string to Pauli operator sum.
    ///
    /// Each fermionic operator is expanded to its complete JW Pauli representation;
    /// successive operator sums are then multiplied together (outer product).
    /// The overall fermionic coefficient is applied last.
    pub fn transform_string(
        &mut self,
        fermionic_string: &FermionicString,
    ) -> Result<PauliOperatorSum> {
        let mut pauli_sum = PauliOperatorSum::new(self.num_modes);

        if fermionic_string.operators.is_empty() {
            let mut identity_string = PauliString::new(self.num_modes);
            identity_string.coefficient = fermionic_string.coefficient;
            pauli_sum.add_term(identity_string)?;
            return Ok(pauli_sum);
        }

        // Expand each fermionic operator to its complete JW sum (possibly multi-term),
        // then multiply the sums together.
        let mut current: Vec<PauliString> = {
            let first_sum = self.transform_operator_to_sum(&fermionic_string.operators[0])?;
            first_sum.terms
        };

        for op in fermionic_string.operators.iter().skip(1) {
            let next_sum = self.transform_operator_to_sum(op)?;
            let mut new_current = Vec::new();
            for ca in &current {
                for nb in &next_sum.terms {
                    let product = ca.multiply(nb)?;
                    if product.coefficient.norm() > 1e-15 {
                        new_current.push(product);
                    }
                }
            }
            current = new_current;
        }

        // Apply the overall fermionic coefficient and collect terms
        for mut term in current {
            term.coefficient *= fermionic_string.coefficient;
            pauli_sum.add_term(term)?;
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

    /// Initialize with `SciRS2` backend
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

    /// Apply a Pauli string in-place to `self.state`.
    ///
    /// For each qubit/mode `q` in order:
    ///   - I → no-op
    ///   - Z → multiply amplitude by −1 for all states with bit `q_bit` set
    ///   - X → swap amplitude pairs (i, i XOR (1<<q_bit))
    ///   - Y → X-then-Z combined with ±i phase: swap and multiply
    ///
    /// The state-vector index bit ordering follows `set_initial_state`:
    /// mode `q` corresponds to bit `n_modes − 1 − q` of the state index.
    ///
    /// After all single-qubit operators are applied the overall coefficient
    /// of the Pauli string is multiplied into every amplitude.
    fn apply_pauli_string(&mut self, pauli_string: &PauliString) -> Result<()> {
        let n = self.num_modes;
        let size = self.state.len();

        for (q, &op) in pauli_string.operators.iter().enumerate() {
            let q_bit = n - 1 - q;
            match op {
                PauliOperator::I => {}
                PauliOperator::Z => {
                    for i in 0..size {
                        if (i >> q_bit) & 1 == 1 {
                            self.state[i] = -self.state[i];
                        }
                    }
                }
                PauliOperator::X => {
                    for i in 0..size {
                        if (i >> q_bit) & 1 == 0 {
                            let j = i | (1 << q_bit);
                            self.state.swap(i, j);
                        }
                    }
                }
                PauliOperator::Y => {
                    // Y = i·X·Z: swap with phase ±i
                    // Y|0⟩ = i|1⟩,  Y|1⟩ = -i|0⟩
                    for i in 0..size {
                        if (i >> q_bit) & 1 == 0 {
                            let j = i | (1 << q_bit);
                            let a = self.state[i];
                            let b = self.state[j];
                            self.state[i] = Complex64::new(0.0, 1.0) * b;
                            self.state[j] = Complex64::new(0.0, -1.0) * a;
                        }
                    }
                }
            }
        }

        // Apply overall coefficient
        let coeff = pauli_string.coefficient;
        for amp in self.state.iter_mut() {
            *amp *= coeff;
        }

        Ok(())
    }

    /// Compute expectation value of a fermionic operator in the current state.
    ///
    /// Uses the complete JW expansion (`transform_operator_to_sum`) so that
    /// all Pauli string terms — including constant identity contributions — are
    /// correctly included in the expectation value.
    pub fn expectation_value(&mut self, op: &FermionicOperator) -> Result<Complex64> {
        let pauli_sum = self.jw_transform.transform_operator_to_sum(op)?;

        let mut total = Complex64::new(0.0, 0.0);
        for term in &pauli_sum.terms {
            total += self.compute_pauli_expectation(term)?;
        }
        Ok(total)
    }

    /// Compute ⟨ψ|P|ψ⟩ for a Pauli string P without mutating `self.state`.
    ///
    /// The Pauli string is applied analytically to a clone of the state vector
    /// and then the inner product with the original state is taken.
    fn compute_pauli_expectation(&self, pauli_string: &PauliString) -> Result<Complex64> {
        let n = self.num_modes;
        let size = self.state.len();
        let mut psi_prime = self.state.clone();

        // Apply each Pauli operator to the cloned state (without the coefficient)
        for (q, &op) in pauli_string.operators.iter().enumerate() {
            let q_bit = n - 1 - q;
            match op {
                PauliOperator::I => {}
                PauliOperator::Z => {
                    for i in 0..size {
                        if (i >> q_bit) & 1 == 1 {
                            psi_prime[i] = -psi_prime[i];
                        }
                    }
                }
                PauliOperator::X => {
                    for i in 0..size {
                        if (i >> q_bit) & 1 == 0 {
                            let j = i | (1 << q_bit);
                            psi_prime.swap(i, j);
                        }
                    }
                }
                PauliOperator::Y => {
                    for i in 0..size {
                        if (i >> q_bit) & 1 == 0 {
                            let j = i | (1 << q_bit);
                            let a = psi_prime[i];
                            let b = psi_prime[j];
                            psi_prime[i] = Complex64::new(0.0, 1.0) * b;
                            psi_prime[j] = Complex64::new(0.0, -1.0) * a;
                        }
                    }
                }
            }
        }

        // ⟨ψ|P|ψ⟩ = coeff * Σ_i conj(ψ[i]) * (P|ψ⟩)[i]
        let raw: Complex64 = self
            .state
            .iter()
            .zip(psi_prime.iter())
            .map(|(&a, &b)| a.conj() * b)
            .sum();

        Ok(pauli_string.coefficient * raw)
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

    /// Apply a single Pauli string action P|ψ⟩ to `target` (in-place).
    ///
    /// The `target` array is modified by the unit Pauli operators (coefficient ignored).
    fn apply_pauli_operators(
        operators: &[PauliOperator],
        num_modes: usize,
        target: &mut Array1<Complex64>,
    ) {
        let size = target.len();
        for (q, &op) in operators.iter().enumerate() {
            let q_bit = num_modes - 1 - q;
            match op {
                PauliOperator::I => {}
                PauliOperator::Z => {
                    for i in 0..size {
                        if (i >> q_bit) & 1 == 1 {
                            target[i] = -target[i];
                        }
                    }
                }
                PauliOperator::X => {
                    for i in 0..size {
                        if (i >> q_bit) & 1 == 0 {
                            let j = i | (1 << q_bit);
                            target.swap(i, j);
                        }
                    }
                }
                PauliOperator::Y => {
                    for i in 0..size {
                        if (i >> q_bit) & 1 == 0 {
                            let j = i | (1 << q_bit);
                            let a = target[i];
                            let b = target[j];
                            target[i] = Complex64::new(0.0, 1.0) * b;
                            target[j] = Complex64::new(0.0, -1.0) * a;
                        }
                    }
                }
            }
        }
    }

    /// Evolve the state under a Pauli Hamiltonian for time `t` using first-order Trotter.
    ///
    /// Before applying the Trotter steps, terms with the same Pauli operator pattern
    /// are merged by summing their coefficients.  This is essential for Hermitian
    /// Hamiltonians derived from JW transformation, where conjugate-pair terms have
    /// imaginary coefficients that must cancel before the exponential is applied.
    ///
    /// For each merged term `c·P` (c real after merging, P² = I Hermitian Pauli):
    ///   exp(−i·t·c·P) = cos(c·t)·I − i·sin(c·t)·P
    ///
    /// For residual terms with complex coefficients (non-Hermitian parts), the
    /// complete formula exp(−i·t·c·P) = cos(|c|t)·I − i·(c/|c|)·sin(|c|t)·P is used.
    fn evolve_pauli_hamiltonian(
        &mut self,
        hamiltonian: &PauliOperatorSum,
        time: f64,
    ) -> Result<()> {
        // Step 1: merge terms by Pauli operator pattern (sum coefficients)
        let mut merged: HashMap<Vec<PauliOperator>, Complex64> = HashMap::new();
        for term in &hamiltonian.terms {
            *merged
                .entry(term.operators.clone())
                .or_insert(Complex64::new(0.0, 0.0)) += term.coefficient;
        }

        let n = self.num_modes;
        let size = self.state.len();

        for (operators, coeff) in &merged {
            let c_re = coeff.re;
            let c_im = coeff.im;
            // Skip negligible terms
            if c_re.abs() < 1e-14 && c_im.abs() < 1e-14 {
                continue;
            }

            // Compute P|ψ⟩ on a clone (unit Pauli action only, no coefficient)
            let mut p_psi = self.state.clone();
            Self::apply_pauli_operators(operators, n, &mut p_psi);

            // exp(−i·t·c·P) where c may be complex and P is a Hermitian Pauli string.
            // For real c: exp(−i·t·c·P) = cos(ct)·I − i·sin(ct)·P  (exactly unitary)
            // For complex c = a+ib:
            //   exp(−i·t·c·P) = cos(|c|t)·I − i·(c/|c|)·sin(|c|t)·P
            // phase = −i·c/|c|·sin(|c|t) = (c_im·sin_t/|c|) + i·(−c_re·sin_t/|c|)
            let magnitude = (c_re * c_re + c_im * c_im).sqrt();
            let theta = magnitude * time;
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            // For a Hermitian Hamiltonian (c purely real after merging), c_im ≈ 0 and
            // phase = (0, −c_re·sin_t/|c|) = (0, −sign(c_re)·sin_t)
            let phase = Complex64::new(c_im * sin_t / magnitude, -c_re * sin_t / magnitude);

            for i in 0..size {
                self.state[i] = cos_t * self.state[i] + phase * p_psi[i];
            }
        }

        Ok(())
    }

    /// Get current state vector
    #[must_use]
    pub const fn get_state(&self) -> &Array1<Complex64> {
        &self.state
    }

    /// Get number of particles in current state
    #[must_use]
    pub fn get_particle_number(&self) -> f64 {
        let mut total_number = 0.0;

        for (index, amplitude) in self.state.iter().enumerate() {
            let prob = amplitude.norm_sqr();
            let popcount = f64::from(index.count_ones());
            total_number += prob * popcount;
        }

        total_number
    }

    /// Get simulation statistics
    #[must_use]
    pub const fn get_stats(&self) -> &FermionicStats {
        &self.stats
    }

    /// Compute connected particle-number correlation ⟨n_i n_j⟩ − ⟨n_i⟩⟨n_j⟩.
    ///
    /// Both single-site expectation values and the joint ⟨n_i n_j⟩ are computed
    /// exactly from the current state vector via the complete Jordan-Wigner expansion.
    pub fn particle_correlation(&mut self, site1: usize, site2: usize) -> Result<f64> {
        // Individual number operator expectations (full JW expansion: I/2 - Z/2)
        let n1 = self
            .expectation_value(&FermionicOperator::Number(site1))?
            .re;
        let n2 = self
            .expectation_value(&FermionicOperator::Number(site2))?
            .re;

        // ⟨n_i n_j⟩ via the product n_i * n_j expanded through transform_string
        // which properly handles the multi-term JW product (I/2-Z_i/2)(I/2-Z_j/2)
        let n1n2_string = FermionicString {
            operators: vec![
                FermionicOperator::Number(site1),
                FermionicOperator::Number(site2),
            ],
            coefficient: Complex64::new(1.0, 0.0),
            num_modes: self.num_modes,
        };
        let pauli_sum = self.jw_transform.transform_string(&n1n2_string)?;

        let mut n1n2 = 0.0_f64;
        for term in &pauli_sum.terms {
            n1n2 += self.compute_pauli_expectation(term)?.re;
        }

        Ok(n1n2 - n1 * n2)
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
        let hamiltonian = FermionicHamiltonian::hubbard_model(2, 1.0, 2.0, 0.5)
            .expect("Failed to create Hubbard model Hamiltonian");
        assert_eq!(hamiltonian.num_modes, 4); // 2 sites × 2 spins
        assert!(!hamiltonian.terms.is_empty());
    }

    #[test]
    fn test_jordan_wigner_transform() {
        let mut jw = JordanWignerTransform::new(4);
        let creation_op = FermionicOperator::Creation(1);
        let pauli_string = jw
            .transform_operator(&creation_op)
            .expect("Failed to transform creation operator via Jordan-Wigner");

        assert_eq!(pauli_string.num_qubits, 4);
        assert_eq!(pauli_string.operators[0], PauliOperator::Z); // Jordan-Wigner string
        assert_eq!(pauli_string.operators[1], PauliOperator::X);
    }

    #[test]
    fn test_fermionic_simulator() {
        let mut simulator =
            FermionicSimulator::new(4).expect("Failed to create fermionic simulator");

        // Set initial state with one particle
        simulator
            .set_initial_state(&[true, false, false, false])
            .expect("Failed to set initial fermionic state");

        let particle_number = simulator.get_particle_number();
        assert!((particle_number - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fermionic_string_multiplication() {
        let string1 = FermionicString::creation(0, Complex64::new(1.0, 0.0), 4);
        let string2 = FermionicString::annihilation(1, Complex64::new(1.0, 0.0), 4);

        let product = string1
            .multiply(&string2)
            .expect("Failed to multiply fermionic strings");
        assert!(!product.operators.is_empty());
    }
}
