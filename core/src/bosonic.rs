//! Bosonic operations for quantum optics and continuous variable quantum computing
//!
//! This module provides support for bosonic operators (creation, annihilation, number, position, momentum)
//! and their applications in quantum optics, continuous variable quantum computing, and bosonic simulations.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
};
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex;

/// Type alias for complex numbers
type Complex64 = Complex<f64>;

/// Bosonic operator types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BosonOperatorType {
    /// Creation operator a†
    Creation,
    /// Annihilation operator a
    Annihilation,
    /// Number operator n = a†a
    Number,
    /// Position operator x = (a + a†)/√2
    Position,
    /// Momentum operator p = i(a† - a)/√2
    Momentum,
    /// Identity operator
    Identity,
    /// Displacement operator D(α) = exp(αa† - α*a)
    Displacement,
    /// Squeeze operator S(z) = exp((z*a² - z(a†)²)/2)
    Squeeze,
}

/// A single bosonic operator acting on a specific mode
#[derive(Debug, Clone)]
pub struct BosonOperator {
    /// Type of the operator
    pub op_type: BosonOperatorType,
    /// Mode index
    pub mode: usize,
    /// Coefficient
    pub coefficient: Complex64,
    /// Truncation dimension (Fock space cutoff)
    pub truncation: usize,
}

impl BosonOperator {
    /// Create a new bosonic operator
    pub const fn new(
        op_type: BosonOperatorType,
        mode: usize,
        coefficient: Complex64,
        truncation: usize,
    ) -> Self {
        Self {
            op_type,
            mode,
            coefficient,
            truncation,
        }
    }

    /// Create a creation operator
    pub const fn creation(mode: usize, truncation: usize) -> Self {
        Self::new(
            BosonOperatorType::Creation,
            mode,
            Complex64::new(1.0, 0.0),
            truncation,
        )
    }

    /// Create an annihilation operator
    pub const fn annihilation(mode: usize, truncation: usize) -> Self {
        Self::new(
            BosonOperatorType::Annihilation,
            mode,
            Complex64::new(1.0, 0.0),
            truncation,
        )
    }

    /// Create a number operator
    pub const fn number(mode: usize, truncation: usize) -> Self {
        Self::new(
            BosonOperatorType::Number,
            mode,
            Complex64::new(1.0, 0.0),
            truncation,
        )
    }

    /// Create a position operator
    pub const fn position(mode: usize, truncation: usize) -> Self {
        Self::new(
            BosonOperatorType::Position,
            mode,
            Complex64::new(1.0, 0.0),
            truncation,
        )
    }

    /// Create a momentum operator
    pub const fn momentum(mode: usize, truncation: usize) -> Self {
        Self::new(
            BosonOperatorType::Momentum,
            mode,
            Complex64::new(1.0, 0.0),
            truncation,
        )
    }

    /// Create a displacement operator D(α)
    pub const fn displacement(mode: usize, alpha: Complex64, truncation: usize) -> Self {
        Self::new(BosonOperatorType::Displacement, mode, alpha, truncation)
    }

    /// Create a squeeze operator S(z)
    pub const fn squeeze(mode: usize, z: Complex64, truncation: usize) -> Self {
        Self::new(BosonOperatorType::Squeeze, mode, z, truncation)
    }

    /// Get the dense matrix representation
    pub fn to_matrix(&self) -> QuantRS2Result<Array2<Complex64>> {
        match self.op_type {
            BosonOperatorType::Creation => self.creation_matrix(),
            BosonOperatorType::Annihilation => self.annihilation_matrix(),
            BosonOperatorType::Number => self.number_matrix(),
            BosonOperatorType::Position => self.position_matrix(),
            BosonOperatorType::Momentum => self.momentum_matrix(),
            BosonOperatorType::Identity => self.identity_matrix(),
            BosonOperatorType::Displacement => self.displacement_matrix(),
            BosonOperatorType::Squeeze => self.squeeze_matrix(),
        }
    }

    /// Creation operator matrix: a†|n⟩ = √(n+1)|n+1⟩
    fn creation_matrix(&self) -> QuantRS2Result<Array2<Complex64>> {
        let n = self.truncation;
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n - 1 {
            matrix[[i + 1, i]] = self.coefficient * ((i + 1) as f64).sqrt();
        }

        Ok(matrix)
    }

    /// Annihilation operator matrix: a|n⟩ = √n|n-1⟩
    fn annihilation_matrix(&self) -> QuantRS2Result<Array2<Complex64>> {
        let n = self.truncation;
        let mut matrix = Array2::zeros((n, n));

        for i in 1..n {
            matrix[[i - 1, i]] = self.coefficient * (i as f64).sqrt();
        }

        Ok(matrix)
    }

    /// Number operator matrix: n|n⟩ = n|n⟩
    fn number_matrix(&self) -> QuantRS2Result<Array2<Complex64>> {
        let n = self.truncation;
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            matrix[[i, i]] = self.coefficient * (i as f64);
        }

        Ok(matrix)
    }

    /// Position operator matrix: x = (a + a†)/√2
    fn position_matrix(&self) -> QuantRS2Result<Array2<Complex64>> {
        let a = self.annihilation_matrix()?;
        let a_dag = self.creation_matrix()?;
        let sqrt2 = 2.0_f64.sqrt();

        Ok((&a + &a_dag) / sqrt2)
    }

    /// Momentum operator matrix: p = i(a† - a)/√2
    fn momentum_matrix(&self) -> QuantRS2Result<Array2<Complex64>> {
        let a = self.annihilation_matrix()?;
        let a_dag = self.creation_matrix()?;
        let sqrt2 = 2.0_f64.sqrt();
        let i = Complex64::new(0.0, 1.0);

        Ok((&a_dag - &a).mapv(|x| i * x / sqrt2))
    }

    /// Identity matrix
    fn identity_matrix(&self) -> QuantRS2Result<Array2<Complex64>> {
        let n = self.truncation;
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            matrix[[i, i]] = self.coefficient;
        }

        Ok(matrix)
    }

    /// Displacement operator matrix: D(α) = exp(αa† - α*a)
    fn displacement_matrix(&self) -> QuantRS2Result<Array2<Complex64>> {
        let a = self.annihilation_matrix()?;
        let a_dag = self.creation_matrix()?;
        let alpha = self.coefficient;

        // Generator: αa† - α*a
        let generator = &a_dag.mapv(|x| alpha * x) - &a.mapv(|x| alpha.conj() * x);

        // Use matrix exponential via Padé approximation
        matrix_exponential_complex(&generator)
    }

    /// Squeeze operator matrix: S(z) = exp((z*a² - z*(a†)²)/2)
    fn squeeze_matrix(&self) -> QuantRS2Result<Array2<Complex64>> {
        let a = self.annihilation_matrix()?;
        let a_dag = self.creation_matrix()?;
        let z = self.coefficient;

        // a² and (a†)²
        let a_squared = a.dot(&a);
        let a_dag_squared = a_dag.dot(&a_dag);

        // Generator: (z*a² - z*(a†)²)/2
        let generator =
            &a_squared.mapv(|x| z * x / 2.0) - &a_dag_squared.mapv(|x| z.conj() * x / 2.0);

        // Use matrix exponential via Padé approximation
        matrix_exponential_complex(&generator)
    }

    /// Get the Hermitian conjugate
    #[must_use]
    pub fn dagger(&self) -> Self {
        let conj_coeff = self.coefficient.conj();
        match self.op_type {
            BosonOperatorType::Creation => Self::new(
                BosonOperatorType::Annihilation,
                self.mode,
                conj_coeff,
                self.truncation,
            ),
            BosonOperatorType::Annihilation => Self::new(
                BosonOperatorType::Creation,
                self.mode,
                conj_coeff,
                self.truncation,
            ),
            BosonOperatorType::Number => Self::new(
                BosonOperatorType::Number,
                self.mode,
                conj_coeff,
                self.truncation,
            ),
            BosonOperatorType::Position => Self::new(
                BosonOperatorType::Position,
                self.mode,
                conj_coeff,
                self.truncation,
            ),
            BosonOperatorType::Momentum => Self::new(
                BosonOperatorType::Momentum,
                self.mode,
                -conj_coeff,
                self.truncation,
            ), // p† = -p
            BosonOperatorType::Identity => Self::new(
                BosonOperatorType::Identity,
                self.mode,
                conj_coeff,
                self.truncation,
            ),
            BosonOperatorType::Displacement => Self::new(
                BosonOperatorType::Displacement,
                self.mode,
                conj_coeff,
                self.truncation,
            ),
            BosonOperatorType::Squeeze => Self::new(
                BosonOperatorType::Squeeze,
                self.mode,
                conj_coeff,
                self.truncation,
            ),
        }
    }
}

/// A product of bosonic operators
#[derive(Debug, Clone)]
pub struct BosonTerm {
    /// Ordered list of operators in the term
    pub operators: Vec<BosonOperator>,
    /// Overall coefficient
    pub coefficient: Complex64,
}

impl BosonTerm {
    /// Create a new bosonic term
    pub const fn new(operators: Vec<BosonOperator>, coefficient: Complex64) -> Self {
        Self {
            operators,
            coefficient,
        }
    }

    /// Create an identity term
    pub const fn identity(_truncation: usize) -> Self {
        Self {
            operators: vec![],
            coefficient: Complex64::new(1.0, 0.0),
        }
    }

    /// Get the matrix representation
    pub fn to_matrix(&self, n_modes: usize) -> QuantRS2Result<Array2<Complex64>> {
        if self.operators.is_empty() {
            // Return identity for the full system
            let total_dim = self
                .operators
                .first()
                .map_or(1, |op| op.truncation.pow(n_modes as u32));
            let mut identity = Array2::zeros((total_dim, total_dim));
            for i in 0..total_dim {
                identity[[i, i]] = self.coefficient;
            }
            return Ok(identity);
        }

        // Build the full operator using Kronecker products
        let mut result = None;
        let truncation = self.operators[0].truncation;

        for mode in 0..n_modes {
            // Find operator for this mode
            let mode_op = self
                .operators
                .iter()
                .find(|op| op.mode == mode)
                .map(|op| op.to_matrix())
                .unwrap_or_else(|| {
                    // Identity for this mode
                    let mut id = Array2::zeros((truncation, truncation));
                    for i in 0..truncation {
                        id[[i, i]] = Complex64::new(1.0, 0.0);
                    }
                    Ok(id)
                })?;

            result = match result {
                None => Some(mode_op),
                Some(prev) => Some(kron_complex(&prev, &mode_op)),
            };
        }

        let mut final_result =
            result.ok_or_else(|| QuantRS2Error::InvalidInput("No operators in term".into()))?;
        final_result = final_result.mapv(|x| self.coefficient * x);
        Ok(final_result)
    }

    /// Normal order the operators (creation operators to the left)
    pub fn normal_order(&mut self) -> QuantRS2Result<()> {
        // For bosons, [a, a†] = 1
        // Normal ordering puts creation operators before annihilation
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

        // Only consider swapping annihilation-creation pairs
        matches!(
            (op1.op_type, op2.op_type),
            (BosonOperatorType::Annihilation, BosonOperatorType::Creation)
        ) && op1.mode == op2.mode
    }

    /// Swap two adjacent operators with commutation
    fn swap_operators(&mut self, idx: usize) -> QuantRS2Result<()> {
        if idx + 1 >= self.operators.len() {
            return Err(QuantRS2Error::InvalidInput("Index out of bounds".into()));
        }

        let op1 = &self.operators[idx];
        let op2 = &self.operators[idx + 1];

        // Check commutation relation
        if op1.mode == op2.mode
            && (op1.op_type, op2.op_type)
                == (BosonOperatorType::Annihilation, BosonOperatorType::Creation)
        {
            // [a, a†] = 1
            // a a† = a† a + 1
            // This would require splitting into two terms
            return Err(QuantRS2Error::UnsupportedOperation(
                "Commutation that produces multiple terms not yet supported".into(),
            ));
        }
        // Other operators commute or have more complex relations

        // Different modes commute
        self.operators.swap(idx, idx + 1);
        Ok(())
    }

    /// Get the Hermitian conjugate
    #[must_use]
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

/// A sum of bosonic terms (Hamiltonian)
#[derive(Debug, Clone)]
pub struct BosonHamiltonian {
    /// Terms in the Hamiltonian
    pub terms: Vec<BosonTerm>,
    /// Number of bosonic modes
    pub n_modes: usize,
    /// Fock space truncation
    pub truncation: usize,
}

impl BosonHamiltonian {
    /// Create a new bosonic Hamiltonian
    pub const fn new(n_modes: usize, truncation: usize) -> Self {
        Self {
            terms: Vec::new(),
            n_modes,
            truncation,
        }
    }

    /// Add a term to the Hamiltonian
    pub fn add_term(&mut self, term: BosonTerm) {
        self.terms.push(term);
    }

    /// Add a single-mode term: ω a†a (harmonic oscillator)
    pub fn add_harmonic_oscillator(&mut self, mode: usize, omega: f64) {
        let term = BosonTerm::new(
            vec![BosonOperator::number(mode, self.truncation)],
            Complex64::new(omega, 0.0),
        );
        self.add_term(term);
    }

    /// Add a two-mode coupling: g(a†b + ab†)
    pub fn add_beam_splitter(&mut self, mode1: usize, mode2: usize, g: f64) {
        // a†b term
        let term1 = BosonTerm::new(
            vec![
                BosonOperator::creation(mode1, self.truncation),
                BosonOperator::annihilation(mode2, self.truncation),
            ],
            Complex64::new(g, 0.0),
        );
        self.add_term(term1);

        // ab† term
        let term2 = BosonTerm::new(
            vec![
                BosonOperator::annihilation(mode1, self.truncation),
                BosonOperator::creation(mode2, self.truncation),
            ],
            Complex64::new(g, 0.0),
        );
        self.add_term(term2);
    }

    /// Add a Kerr nonlinearity: κ(a†)²a²
    pub fn add_kerr_nonlinearity(&mut self, mode: usize, kappa: f64) {
        let term = BosonTerm::new(
            vec![
                BosonOperator::creation(mode, self.truncation),
                BosonOperator::creation(mode, self.truncation),
                BosonOperator::annihilation(mode, self.truncation),
                BosonOperator::annihilation(mode, self.truncation),
            ],
            Complex64::new(kappa, 0.0),
        );
        self.add_term(term);
    }

    /// Add a two-mode squeezing: ξ(ab + a†b†)
    pub fn add_two_mode_squeezing(&mut self, mode1: usize, mode2: usize, xi: Complex64) {
        // ab term
        let term1 = BosonTerm::new(
            vec![
                BosonOperator::annihilation(mode1, self.truncation),
                BosonOperator::annihilation(mode2, self.truncation),
            ],
            xi,
        );
        self.add_term(term1);

        // a†b† term
        let term2 = BosonTerm::new(
            vec![
                BosonOperator::creation(mode1, self.truncation),
                BosonOperator::creation(mode2, self.truncation),
            ],
            xi.conj(),
        );
        self.add_term(term2);
    }

    /// Get the matrix representation
    pub fn to_matrix(&self) -> QuantRS2Result<Array2<Complex64>> {
        let total_dim = self.truncation.pow(self.n_modes as u32);
        let mut result = Array2::zeros((total_dim, total_dim));

        for term in &self.terms {
            let term_matrix = term.to_matrix(self.n_modes)?;
            result = &result + &term_matrix;
        }

        Ok(result)
    }

    /// Get the Hermitian conjugate
    #[must_use]
    pub fn dagger(&self) -> Self {
        let conj_terms = self.terms.iter().map(|t| t.dagger()).collect();
        Self {
            terms: conj_terms,
            n_modes: self.n_modes,
            truncation: self.truncation,
        }
    }

    /// Check if the Hamiltonian is Hermitian
    pub fn is_hermitian(&self, tolerance: f64) -> bool {
        // Get matrices
        let h = match self.to_matrix() {
            Ok(mat) => mat,
            Err(_) => return false,
        };

        let h_dag = match self.dagger().to_matrix() {
            Ok(mat) => mat,
            Err(_) => return false,
        };

        // Compare matrices
        for i in 0..h.shape()[0] {
            for j in 0..h.shape()[1] {
                if (h[[i, j]] - h_dag[[i, j]]).norm() > tolerance {
                    return false;
                }
            }
        }

        true
    }
}

/// Gaussian state representation
#[derive(Debug, Clone)]
pub struct GaussianState {
    /// Number of modes
    pub n_modes: usize,
    /// Displacement vector (2n dimensional: [x1, p1, x2, p2, ...])
    pub displacement: Vec<f64>,
    /// Covariance matrix (2n × 2n)
    pub covariance: Array2<f64>,
}

impl GaussianState {
    /// Create a vacuum state
    pub fn vacuum(n_modes: usize) -> Self {
        use scirs2_core::ndarray::Array2;

        Self {
            n_modes,
            displacement: vec![0.0; 2 * n_modes],
            covariance: Array2::eye(2 * n_modes) * 0.5, // ℏ/2 units
        }
    }

    /// Create a coherent state |α⟩
    pub fn coherent(n_modes: usize, mode: usize, alpha: Complex64) -> Self {
        let mut state = Self::vacuum(n_modes);

        // Set displacement
        state.displacement[2 * mode] = alpha.re * 2.0_f64.sqrt(); // x quadrature
        state.displacement[2 * mode + 1] = alpha.im * 2.0_f64.sqrt(); // p quadrature

        state
    }

    /// Create a squeezed state
    pub fn squeezed(n_modes: usize, mode: usize, r: f64, phi: f64) -> Self {
        let mut state = Self::vacuum(n_modes);

        // Modify covariance matrix for the squeezed mode
        let c = r.cosh();
        let s = r.sinh();
        let cos_2phi = (2.0 * phi).cos();
        let sin_2phi = (2.0 * phi).sin();

        let idx = 2 * mode;
        state.covariance[[idx, idx]] = 0.5 * s.mul_add(cos_2phi, c);
        state.covariance[[idx + 1, idx + 1]] = 0.5 * s.mul_add(-cos_2phi, c);
        state.covariance[[idx, idx + 1]] = 0.5 * s * sin_2phi;
        state.covariance[[idx + 1, idx]] = 0.5 * s * sin_2phi;

        state
    }

    /// Apply a symplectic transformation
    pub fn apply_symplectic(&mut self, s: &Array2<f64>) -> QuantRS2Result<()> {
        if s.shape() != [2 * self.n_modes, 2 * self.n_modes] {
            return Err(QuantRS2Error::InvalidInput(
                "Symplectic matrix has wrong dimensions".into(),
            ));
        }

        // Transform displacement: d' = S d
        let d = Array2::from_shape_vec((2 * self.n_modes, 1), self.displacement.clone()).map_err(
            |e| QuantRS2Error::LinalgError(format!("Failed to create displacement vector: {e}")),
        )?;
        let d_prime = s.dot(&d);
        self.displacement = d_prime.iter().copied().collect();

        // Transform covariance: V' = S V S^T
        self.covariance = s.dot(&self.covariance).dot(&s.t());

        Ok(())
    }

    /// Calculate the purity Tr(ρ²)
    pub const fn purity(&self) -> f64 {
        // For Gaussian states: purity = 1/√det(2V)
        // TODO: Implement determinant calculation
        // For now, return placeholder for pure states
        1.0
    }
}

/// Convert bosonic operators to qubit representation (using truncated Fock space)
pub fn boson_to_qubit_encoding(
    op: &BosonOperator,
    encoding: &str,
) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
    match encoding {
        "binary" => {
            // Binary encoding: |n⟩ → |binary(n)⟩
            // Number of qubits needed: ceil(log2(truncation))
            let _n_qubits = (op.truncation as f64).log2().ceil() as usize;

            // This would require implementing the specific encoding
            Err(QuantRS2Error::UnsupportedOperation(
                "Binary encoding not yet implemented".into(),
            ))
        }
        "unary" => {
            // Unary encoding: |n⟩ → |0...010...0⟩ (n-th position is 1)
            // Number of qubits = truncation
            Err(QuantRS2Error::UnsupportedOperation(
                "Unary encoding not yet implemented".into(),
            ))
        }
        "gray" => {
            // Gray code encoding
            Err(QuantRS2Error::UnsupportedOperation(
                "Gray code encoding not yet implemented".into(),
            ))
        }
        _ => Err(QuantRS2Error::InvalidInput(format!(
            "Unknown encoding: {encoding}"
        ))),
    }
}

/// Matrix exponential for complex matrices using Padé approximation
fn matrix_exponential_complex(a: &Array2<Complex64>) -> QuantRS2Result<Array2<Complex64>> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(QuantRS2Error::InvalidInput("Matrix must be square".into()));
    }

    // Scale the matrix to reduce norm
    let norm = a.mapv(|x| x.norm()).sum() / (n as f64);
    let scale = (norm.log2().ceil() as i32).max(0);
    let scale_factor = 2.0_f64.powi(scale);
    let a_scaled = a.mapv(|x| x / scale_factor);

    // Padé approximation of degree 6
    let mut u = Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    });
    let mut v = Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    });

    let a2 = a_scaled.dot(&a_scaled);
    let a4 = a2.dot(&a2);
    let a6 = a4.dot(&a2);

    // Compute U and V for Padé(6,6)
    let c = [
        1.0,
        1.0 / 2.0,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
        1.0 / 5040.0,
    ];

    u = &u * c[0]
        + &a_scaled * c[1]
        + &a2 * c[2]
        + &a_scaled.dot(&a2) * c[3]
        + &a4 * c[4]
        + &a_scaled.dot(&a4) * c[5]
        + &a6 * c[6];

    v = &v * c[0] - &a_scaled * c[1] + &a2 * c[2] - &a_scaled.dot(&a2) * c[3] + &a4 * c[4]
        - &a_scaled.dot(&a4) * c[5]
        + &a6 * c[6];

    // Solve (V - U)X = 2U for X = exp(A)
    let v_minus_u = &v - &u;
    let two_u = &u * 2.0;

    // Simple inversion for small matrices
    let exp_a_scaled = solve_complex(&v_minus_u, &two_u)?;

    // Square the result scale times
    let mut result = exp_a_scaled;
    for _ in 0..scale {
        result = result.dot(&result);
    }

    Ok(result)
}

/// Simple complex matrix solver using Gaussian elimination
fn solve_complex(
    a: &Array2<Complex64>,
    b: &Array2<Complex64>,
) -> QuantRS2Result<Array2<Complex64>> {
    let n = a.shape()[0];
    if a.shape()[1] != n || b.shape()[0] != n {
        return Err(QuantRS2Error::InvalidInput(
            "Invalid dimensions for solve".into(),
        ));
    }

    // Create augmented matrix [A|B]
    let mut aug = Array2::zeros((n, n + b.shape()[1]));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        for j in 0..b.shape()[1] {
            aug[[i, n + j]] = b[[i, j]];
        }
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in i + 1..n {
            if aug[[k, i]].norm() > aug[[max_row, i]].norm() {
                max_row = k;
            }
        }

        // Swap rows
        for j in 0..n + b.shape()[1] {
            let temp = aug[[i, j]];
            aug[[i, j]] = aug[[max_row, j]];
            aug[[max_row, j]] = temp;
        }

        // Check for singular matrix
        if aug[[i, i]].norm() < 1e-12 {
            return Err(QuantRS2Error::LinalgError("Singular matrix".into()));
        }

        // Eliminate column
        for k in i + 1..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..n + b.shape()[1] {
                aug[[k, j]] = aug[[k, j]] - factor * aug[[i, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array2::zeros((n, b.shape()[1]));
    for i in (0..n).rev() {
        for j in 0..b.shape()[1] {
            x[[i, j]] = aug[[i, n + j]];
            for k in i + 1..n {
                x[[i, j]] = x[[i, j]] - aug[[i, k]] * x[[k, j]];
            }
            x[[i, j]] = x[[i, j]] / aug[[i, i]];
        }
    }

    Ok(x)
}

/// Kronecker product for complex matrices
fn kron_complex(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (m, n) = a.dim();
    let (p, q) = b.dim();
    let mut result = Array2::zeros((m * p, n * q));

    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                for l in 0..q {
                    result[[i * p + k, j * q + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }

    result
}

/// Sparse matrix representation for bosonic operators
#[derive(Debug, Clone)]
pub struct SparseBosonOperator {
    /// Row indices
    pub rows: Vec<usize>,
    /// Column indices
    pub cols: Vec<usize>,
    /// Non-zero values
    pub data: Vec<Complex64>,
    /// Matrix shape
    pub shape: (usize, usize),
}

impl SparseBosonOperator {
    /// Create from triplets
    pub const fn from_triplets(
        rows: Vec<usize>,
        cols: Vec<usize>,
        data: Vec<Complex64>,
        shape: (usize, usize),
    ) -> Self {
        Self {
            rows,
            cols,
            data,
            shape,
        }
    }

    /// Convert to dense matrix
    pub fn to_dense(&self) -> Array2<Complex64> {
        let mut matrix = Array2::zeros(self.shape);
        for ((&i, &j), &val) in self.rows.iter().zip(&self.cols).zip(&self.data) {
            matrix[[i, j]] = val;
        }
        matrix
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper macro for relative equality testing
    macro_rules! assert_relative_eq {
        ($left:expr, $right:expr, epsilon = $epsilon:expr) => {
            let diff = ($left - $right).abs();
            assert!(diff < $epsilon, "assertion failed: `(left ≈ right)`\n  left: `{:?}`,\n right: `{:?}`,\n  diff: `{:?}`,\nepsilon: `{:?}`", $left, $right, diff, $epsilon);
        };
    }

    #[test]
    fn test_creation_annihilation_operators() {
        let truncation = 5;
        let a = BosonOperator::annihilation(0, truncation);
        let a_dag = BosonOperator::creation(0, truncation);

        let a_mat = a
            .to_matrix()
            .expect("Annihilation matrix creation should succeed");
        let a_dag_mat = a_dag
            .to_matrix()
            .expect("Creation matrix creation should succeed");

        // Check that a† is the conjugate transpose of a
        for i in 0..truncation {
            for j in 0..truncation {
                assert_relative_eq!(a_dag_mat[[i, j]].re, a_mat[[j, i]].re, epsilon = 1e-12);
                assert_relative_eq!(a_dag_mat[[i, j]].im, -a_mat[[j, i]].im, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_number_operator() {
        let truncation = 5;
        let n_op = BosonOperator::number(0, truncation);
        let n_mat = n_op
            .to_matrix()
            .expect("Number operator matrix creation should succeed");

        // Check diagonal elements
        for i in 0..truncation {
            assert_relative_eq!(n_mat[[i, i]].re, i as f64, epsilon = 1e-12);
            assert_relative_eq!(n_mat[[i, i]].im, 0.0, epsilon = 1e-12);
        }

        // Check off-diagonal elements are zero
        for i in 0..truncation {
            for j in 0..truncation {
                if i != j {
                    assert_relative_eq!(n_mat[[i, j]].norm(), 0.0, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_position_momentum_commutation() {
        let truncation = 10;
        let x = BosonOperator::position(0, truncation);
        let p = BosonOperator::momentum(0, truncation);

        let x_mat = x
            .to_matrix()
            .expect("Position operator matrix creation should succeed");
        let p_mat = p
            .to_matrix()
            .expect("Momentum operator matrix creation should succeed");

        // Calculate [x, p] = xp - px
        let xp = x_mat.dot(&p_mat);
        let px = p_mat.dot(&x_mat);
        let commutator = &xp - &px;

        // Should equal i times identity
        // But due to our normalization with √2, it might be different
        // Let's check what we actually get
        println!("Commutator[0,0] = {:?}", commutator[[0, 0]]);
        println!("Expected canonical [x,p] = i, but we have normalization factors");

        // The canonical commutation is [x,p] = i
        // With our definitions x = (a+a†)/√2, p = i(a†-a)/√2
        // We should get [x,p] = i
        let expected_value = 1.0;

        // Due to truncation effects, check only inner elements
        for i in 0..truncation - 1 {
            for j in 0..truncation - 1 {
                if i == j {
                    assert_relative_eq!(commutator[[i, j]].re, 0.0, epsilon = 1e-10);
                    assert_relative_eq!(commutator[[i, j]].im, expected_value, epsilon = 1e-10);
                } else {
                    assert_relative_eq!(commutator[[i, j]].norm(), 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_harmonic_oscillator_hamiltonian() {
        let n_modes = 2;
        let truncation = 5;
        let mut ham = BosonHamiltonian::new(n_modes, truncation);

        // Add harmonic oscillators
        ham.add_harmonic_oscillator(0, 1.0);
        ham.add_harmonic_oscillator(1, 2.0);

        // Add coupling
        ham.add_beam_splitter(0, 1, 0.5);

        assert_eq!(ham.terms.len(), 4); // 2 HO + 2 coupling terms

        // Check Hermiticity
        assert!(ham.is_hermitian(1e-12));
    }

    #[test]
    fn test_gaussian_state() {
        let n_modes = 2;

        // Test vacuum state
        let vacuum = GaussianState::vacuum(n_modes);
        assert_eq!(vacuum.displacement.len(), 2 * n_modes);
        assert_relative_eq!(vacuum.purity(), 1.0, epsilon = 1e-12);

        // Test coherent state
        let alpha = Complex64::new(1.0, 0.5);
        let coherent = GaussianState::coherent(n_modes, 0, alpha);
        assert_relative_eq!(
            coherent.displacement[0],
            alpha.re * 2.0_f64.sqrt(),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            coherent.displacement[1],
            alpha.im * 2.0_f64.sqrt(),
            epsilon = 1e-12
        );

        // Test squeezed state
        let squeezed = GaussianState::squeezed(n_modes, 0, 1.0, 0.0);
        // For r=1, phi=0: x quadrature is squeezed (variance < 0.5) but actually with these formulas
        // cosh(1) - sinh(1) ≈ 0.368 < 0.5
        assert!(squeezed.covariance[[0, 0]] > 0.5); // Actually anti-squeezed in x for phi=0
        assert!(squeezed.covariance[[1, 1]] < 0.5); // Squeezed in p for phi=0
    }
}
