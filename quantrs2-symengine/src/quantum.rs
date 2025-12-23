//! Quantum computing specific symbolic operations
//!
//! This module provides symbolic representations and operations for quantum computing,
//! including Pauli matrices, quantum gates, and operator algebra.

use crate::{Expression, SymEngineResult};

/// Pauli matrices and related quantum operators
pub mod pauli {
    use crate::Expression;

    /// Pauli X matrix (σx)
    ///
    /// ```text
    /// σx = [[0, 1],
    ///       [1, 0]]
    /// ```
    #[must_use]
    pub fn sigma_x() -> Expression {
        Expression::new("Matrix([[0, 1], [1, 0]])")
    }

    /// Pauli Y matrix (σy)
    ///
    /// ```text
    /// σy = [[0, -i],
    ///       [i,  0]]
    /// ```
    #[must_use]
    pub fn sigma_y() -> Expression {
        Expression::new("Matrix([[0, -I], [I, 0]])")
    }

    /// Pauli Z matrix (σz)
    ///
    /// ```text
    /// σz = [[1,  0],
    ///       [0, -1]]
    /// ```
    #[must_use]
    pub fn sigma_z() -> Expression {
        Expression::new("Matrix([[1, 0], [0, -1]])")
    }

    /// Identity matrix (2x2)
    ///
    /// ```text
    /// I = [[1, 0],
    ///      [0, 1]]
    /// ```
    #[must_use]
    pub fn identity() -> Expression {
        Expression::new("Matrix([[1, 0], [0, 1]])")
    }

    /// Pauli matrices as a vector [σx, σy, σz]
    #[must_use]
    pub fn sigma_vector() -> Vec<Expression> {
        vec![sigma_x(), sigma_y(), sigma_z()]
    }
}

/// Common quantum gates
pub mod gates {
    use crate::Expression;

    /// Hadamard gate
    ///
    /// ```text
    /// H = 1/√2 * [[1,  1],
    ///             [1, -1]]
    /// ```
    #[must_use]
    pub fn hadamard() -> Expression {
        Expression::new("1/sqrt(2) * Matrix([[1, 1], [1, -1]])")
    }

    /// Phase gate (S gate)
    ///
    /// ```text
    /// S = [[1, 0],
    ///      [0, i]]
    /// ```
    #[must_use]
    pub fn phase() -> Expression {
        Expression::new("Matrix([[1, 0], [0, I]])")
    }

    /// T gate (π/8 gate)
    ///
    /// ```text
    /// T = [[1, 0        ],
    ///      [0, e^(iπ/4)]]
    /// ```
    #[must_use]
    pub fn t_gate() -> Expression {
        Expression::new("Matrix([[1, 0], [0, exp(I*pi/4)]])")
    }

    /// CNOT gate (Controlled-NOT)
    ///
    /// ```text
    /// CNOT = [[1, 0, 0, 0],
    ///         [0, 1, 0, 0],
    ///         [0, 0, 0, 1],
    ///         [0, 0, 1, 0]]
    /// ```
    #[must_use]
    pub fn cnot() -> Expression {
        Expression::new("Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])")
    }

    /// Rotation around X axis
    ///
    /// ```text
    /// Rx(θ) = [[cos(θ/2),   -i*sin(θ/2)],
    ///          [-i*sin(θ/2), cos(θ/2)  ]]
    /// ```
    #[must_use]
    pub fn rx(theta: &Expression) -> Expression {
        Expression::new(format!(
            "Matrix([[cos({theta}/2), -I*sin({theta}/2)], [-I*sin({theta}/2), cos({theta}/2)]])"
        ))
    }

    /// Rotation around Y axis
    ///
    /// ```text
    /// Ry(θ) = [[cos(θ/2), -sin(θ/2)],
    ///          [sin(θ/2),  cos(θ/2)]]
    /// ```
    #[must_use]
    pub fn ry(theta: &Expression) -> Expression {
        Expression::new(format!(
            "Matrix([[cos({theta}/2), -sin({theta}/2)], [sin({theta}/2), cos({theta}/2)]])"
        ))
    }

    /// Rotation around Z axis
    ///
    /// ```text
    /// Rz(θ) = [[e^(-iθ/2), 0        ],
    ///          [0,         e^(iθ/2) ]]
    /// ```
    #[must_use]
    pub fn rz(theta: &Expression) -> Expression {
        Expression::new(format!(
            "Matrix([[exp(-I*{theta}/2), 0], [0, exp(I*{theta}/2)]])"
        ))
    }

    /// General single-qubit rotation (U3 gate)
    ///
    /// ```text
    /// U3(θ, φ, λ) = [[cos(θ/2),          -e^(iλ)*sin(θ/2)    ],
    ///                [e^(iφ)*sin(θ/2),    e^(i(φ+λ))*cos(θ/2)]]
    /// ```
    #[must_use]
    pub fn u3(theta: &Expression, phi: &Expression, lambda: &Expression) -> Expression {
        Expression::new(format!(
            "Matrix([[cos({theta}/2), -exp(I*{lambda})*sin({theta}/2)], [exp(I*{phi})*sin({theta}/2), exp(I*({phi}+{lambda}))*cos({theta}/2)]])"
        ))
    }
}

/// Quantum operator algebra
pub mod operators {
    use crate::{Expression, SymEngineResult};

    /// Commutator [A, B] = AB - BA
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::quantum::operators::commutator;
    /// use quantrs2_symengine::quantum::pauli::{sigma_x, sigma_y};
    ///
    /// let comm = commutator(&sigma_x(), &sigma_y());
    /// // Result: [σx, σy] = 2i*σz
    /// ```
    #[must_use]
    pub fn commutator(a: &Expression, b: &Expression) -> Expression {
        a.clone() * b.clone() - b.clone() * a.clone()
    }

    /// Anticommutator {A, B} = AB + BA
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::quantum::operators::anticommutator;
    /// use quantrs2_symengine::quantum::pauli::{sigma_x, sigma_y};
    ///
    /// let anticomm = anticommutator(&sigma_x(), &sigma_y());
    /// // Result: {σx, σy} = 0
    /// ```
    #[must_use]
    pub fn anticommutator(a: &Expression, b: &Expression) -> Expression {
        a.clone() * b.clone() + b.clone() * a.clone()
    }

    /// Check if an operator is Hermitian (A = A†)
    ///
    /// # Errors
    /// Returns an error if the underlying SymEngine operation fails.
    pub fn is_hermitian(matrix: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!(
            "conjugate(transpose({matrix})) - {matrix}"
        )))
    }

    /// Check if an operator is unitary (A†A = I)
    ///
    /// # Errors
    /// Returns an error if the underlying SymEngine operation fails.
    pub fn is_unitary(matrix: &Expression) -> SymEngineResult<Expression> {
        let conjugate_transpose = Expression::new(format!("conjugate(transpose({matrix}))"));
        Ok(Expression::new(format!("{conjugate_transpose} * {matrix}")))
    }

    /// Matrix trace (sum of diagonal elements)
    ///
    /// # Errors
    /// Returns an error if the underlying SymEngine operation fails.
    pub fn trace(matrix: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("trace({matrix})")))
    }

    /// Tensor product (Kronecker product) of two operators
    ///
    /// # Errors
    /// Returns an error if the underlying SymEngine operation fails.
    pub fn tensor_product(a: &Expression, b: &Expression) -> SymEngineResult<Expression> {
        Ok(Expression::new(format!("kronecker({a}, {b})")))
    }
}

/// Quantum state representations
pub mod states {
    use crate::Expression;

    /// |0⟩ state (computational basis)
    #[must_use]
    pub fn ket_0() -> Expression {
        Expression::new("Matrix([[1], [0]])")
    }

    /// |1⟩ state (computational basis)
    #[must_use]
    pub fn ket_1() -> Expression {
        Expression::new("Matrix([[0], [1]])")
    }

    /// |+⟩ state = (|0⟩ + |1⟩)/√2
    #[must_use]
    pub fn ket_plus() -> Expression {
        Expression::new("1/sqrt(2) * Matrix([[1], [1]])")
    }

    /// |-⟩ state = (|0⟩ - |1⟩)/√2
    #[must_use]
    pub fn ket_minus() -> Expression {
        Expression::new("1/sqrt(2) * Matrix([[1], [-1]])")
    }

    /// |i⟩ state = (|0⟩ + i|1⟩)/√2
    #[must_use]
    pub fn ket_i() -> Expression {
        Expression::new("1/sqrt(2) * Matrix([[1], [I]])")
    }

    /// |-i⟩ state = (|0⟩ - i|1⟩)/√2
    #[must_use]
    pub fn ket_minus_i() -> Expression {
        Expression::new("1/sqrt(2) * Matrix([[1], [-I]])")
    }

    /// Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    #[must_use]
    pub fn bell_phi_plus() -> Expression {
        Expression::new("1/sqrt(2) * Matrix([[1], [0], [0], [1]])")
    }

    /// Bell state |Φ-⟩ = (|00⟩ - |11⟩)/√2
    #[must_use]
    pub fn bell_phi_minus() -> Expression {
        Expression::new("1/sqrt(2) * Matrix([[1], [0], [0], [-1]])")
    }

    /// Bell state |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    #[must_use]
    pub fn bell_psi_plus() -> Expression {
        Expression::new("1/sqrt(2) * Matrix([[0], [1], [1], [0]])")
    }

    /// Bell state |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    #[must_use]
    pub fn bell_psi_minus() -> Expression {
        Expression::new("1/sqrt(2) * Matrix([[0], [1], [-1], [0]])")
    }

    /// General qubit state |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    #[must_use]
    pub fn bloch_state(theta: &Expression, phi: &Expression) -> Expression {
        Expression::new(format!(
            "Matrix([[cos({theta}/2)], [exp(I*{phi})*sin({theta}/2)]])"
        ))
    }
}

/// Quantum Hamiltonian and evolution operators
pub mod hamiltonian {
    use crate::{Expression, SymEngineResult};

    /// Time evolution operator U(t) = e^(-iHt/ℏ)
    ///
    /// # Errors
    /// Returns an error if the underlying SymEngine operation fails.
    pub fn time_evolution(
        hamiltonian: &Expression,
        time: &Expression,
        hbar: Option<&Expression>,
    ) -> SymEngineResult<Expression> {
        let h = hbar.map_or_else(|| Expression::from(1), Clone::clone);
        Ok(Expression::new(format!("exp(-I*{hamiltonian}*{time}/{h})")))
    }

    /// Pauli string Hamiltonian (e.g., H = σx⊗σy⊗σz)
    ///
    /// # Errors
    /// Returns an error if the underlying SymEngine operation fails.
    pub fn pauli_string(pauli_ops: &[Expression]) -> SymEngineResult<Expression> {
        if pauli_ops.is_empty() {
            return Ok(Expression::from(1));
        }

        let mut result = pauli_ops[0].clone();
        for op in &pauli_ops[1..] {
            result = Expression::new(format!("kronecker({result}, {op})"));
        }
        Ok(result)
    }

    /// Ising model Hamiltonian H = -J Σ σz_i σz_{i+1} - h Σ σx_i
    #[must_use]
    pub fn ising_model(n_sites: usize, j: &Expression, h: &Expression) -> Expression {
        // This is a symbolic placeholder - actual implementation would be more complex
        Expression::new(format!("IsingHamiltonian(n={n_sites}, J={j}, h={h})"))
    }

    /// Heisenberg model Hamiltonian H = J Σ (σx_i σx_{i+1} + σy_i σy_{i+1} + σz_i σz_{i+1})
    #[must_use]
    pub fn heisenberg_model(n_sites: usize, j: &Expression) -> Expression {
        Expression::new(format!("HeisenbergHamiltonian(n={n_sites}, J={j})"))
    }
}

/// Advanced quantum operators (creation, annihilation, ladder, spin)
pub mod advanced_operators {
    use crate::Expression;

    /// Bosonic creation operator a†
    ///
    /// Creates a symbolic representation of the bosonic creation operator.
    /// In the number state basis |n⟩, a†|n⟩ = √(n+1)|n+1⟩
    #[must_use]
    pub fn creation() -> Expression {
        Expression::symbol("a_dag")
    }

    /// Bosonic annihilation operator a
    ///
    /// Creates a symbolic representation of the bosonic annihilation operator.
    /// In the number state basis |n⟩, a|n⟩ = √n|n-1⟩
    #[must_use]
    pub fn annihilation() -> Expression {
        Expression::symbol("a")
    }

    /// Number operator n = a†a
    ///
    /// Returns the number of quanta in a mode.
    #[must_use]
    pub fn number_operator() -> Expression {
        let a_dag = creation();
        let a = annihilation();
        a_dag * a
    }

    /// Commutator of creation and annihilation operators [a, a†] = 1
    ///
    /// This is the fundamental bosonic commutation relation.
    #[must_use]
    pub fn bosonic_commutator() -> Expression {
        Expression::from(1)
    }

    /// Position operator x = (a + a†)/√2
    ///
    /// Position operator in terms of ladder operators
    #[must_use]
    pub fn position_operator() -> Expression {
        let a = annihilation();
        let a_dag = creation();
        let sqrt2 = Expression::new("sqrt(2)");
        (a + a_dag) / sqrt2
    }

    /// Momentum operator p = i(a† - a)/√2
    ///
    /// Momentum operator in terms of ladder operators
    #[must_use]
    pub fn momentum_operator() -> Expression {
        let a = annihilation();
        let a_dag = creation();
        let sqrt2 = Expression::new("sqrt(2)");
        let i = Expression::new("I");
        i * (a_dag - a) / sqrt2
    }

    /// Spin raising operator S+ = Sx + iSy
    ///
    /// For spin-1/2: S+ = ℏσ+/2, where σ+ = σx + iσy
    #[must_use]
    pub fn spin_raising() -> Expression {
        Expression::symbol("S_plus")
    }

    /// Spin lowering operator S- = Sx - iSy
    ///
    /// For spin-1/2: S- = ℏσ-/2, where σ- = σx - iσy
    #[must_use]
    pub fn spin_lowering() -> Expression {
        Expression::symbol("S_minus")
    }

    /// Spin x-component operator Sx
    #[must_use]
    pub fn spin_x() -> Expression {
        Expression::symbol("S_x")
    }

    /// Spin y-component operator Sy
    #[must_use]
    pub fn spin_y() -> Expression {
        Expression::symbol("S_y")
    }

    /// Spin z-component operator Sz
    #[must_use]
    pub fn spin_z() -> Expression {
        Expression::symbol("S_z")
    }

    /// Spin squared operator S² = Sx² + Sy² + Sz²
    #[must_use]
    pub fn spin_squared() -> Expression {
        let sx = spin_x();
        let sy = spin_y();
        let sz = spin_z();
        let two = Expression::from(2);
        sx.pow(&two) + sy.pow(&two) + sz.pow(&two)
    }

    /// Angular momentum raising operator L+ = Lx + iLy
    #[must_use]
    pub fn angular_momentum_raising() -> Expression {
        Expression::symbol("L_plus")
    }

    /// Angular momentum lowering operator L- = Lx - iLy
    #[must_use]
    pub fn angular_momentum_lowering() -> Expression {
        Expression::symbol("L_minus")
    }

    /// Angular momentum x-component operator Lx
    #[must_use]
    pub fn angular_momentum_x() -> Expression {
        Expression::symbol("L_x")
    }

    /// Angular momentum y-component operator Ly
    #[must_use]
    pub fn angular_momentum_y() -> Expression {
        Expression::symbol("L_y")
    }

    /// Angular momentum z-component operator Lz
    #[must_use]
    pub fn angular_momentum_z() -> Expression {
        Expression::symbol("L_z")
    }

    /// Angular momentum squared operator L² = Lx² + Ly² + Lz²
    #[must_use]
    pub fn angular_momentum_squared() -> Expression {
        let lx = angular_momentum_x();
        let ly = angular_momentum_y();
        let lz = angular_momentum_z();
        let two = Expression::from(2);
        lx.pow(&two) + ly.pow(&two) + lz.pow(&two)
    }

    /// Fermionic creation operator c†
    ///
    /// Fermions obey anticommutation relations: {c, c†} = 1
    #[must_use]
    pub fn fermionic_creation() -> Expression {
        Expression::symbol("c_dag")
    }

    /// Fermionic annihilation operator c
    ///
    /// Fermions obey anticommutation relations: {c, c} = 0
    #[must_use]
    pub fn fermionic_annihilation() -> Expression {
        Expression::symbol("c")
    }

    /// Fermionic number operator n = c†c
    #[must_use]
    pub fn fermionic_number_operator() -> Expression {
        let c_dag = fermionic_creation();
        let c = fermionic_annihilation();
        c_dag * c
    }

    /// Displacement operator D(α) = exp(αa† - α*a)
    ///
    /// Displaces a coherent state in phase space
    #[must_use]
    pub fn displacement_operator(alpha: &Expression) -> Expression {
        let a = annihilation();
        let a_dag = creation();
        let arg = alpha.clone() * a_dag - alpha.conjugate() * a;
        Expression::new(format!("exp({arg})"))
    }

    /// Squeezing operator S(ζ) = exp[(ζ*a²- ζa†²)/2]
    ///
    /// Squeezes quantum fluctuations in one quadrature
    #[must_use]
    pub fn squeezing_operator(zeta: &Expression) -> Expression {
        let a = annihilation();
        let a_dag = creation();
        let two = Expression::from(2);
        let term1 = zeta.clone() * a.pow(&two);
        let term2 = zeta.conjugate() * a_dag.pow(&two);
        let arg = (term1 - term2) / two;
        Expression::new(format!("exp({arg})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Matrix syntax causes issues with some SymEngine configurations"]
    fn test_pauli_matrices() {
        let sx = pauli::sigma_x();
        let sy = pauli::sigma_y();
        let sz = pauli::sigma_z();
        let id = pauli::identity();

        assert!(sx.to_string().contains("Matrix"));
        assert!(sy.to_string().contains("Matrix"));
        assert!(sz.to_string().contains("Matrix"));
        assert!(id.to_string().contains("Matrix"));
    }

    #[test]
    #[ignore = "Matrix syntax causes issues with some SymEngine configurations"]
    fn test_quantum_gates() {
        let h = gates::hadamard();
        let s = gates::phase();
        let t = gates::t_gate();
        let cnot = gates::cnot();

        assert!(h.to_string().contains("Matrix"));
        assert!(s.to_string().contains("Matrix"));
        assert!(t.to_string().contains("Matrix"));
        assert!(cnot.to_string().contains("Matrix"));
    }

    #[test]
    #[ignore = "Matrix syntax causes issues with some SymEngine configurations"]
    fn test_rotations() {
        let theta = Expression::symbol("theta");
        let rx = gates::rx(&theta);
        let ry = gates::ry(&theta);
        let rz = gates::rz(&theta);

        assert!(rx.to_string().contains("cos"));
        assert!(ry.to_string().contains("sin"));
        assert!(rz.to_string().contains("exp"));
    }

    #[test]
    fn test_commutator() {
        let a = Expression::symbol("A");
        let b = Expression::symbol("B");
        let comm = operators::commutator(&a, &b);

        // [A, B] = AB - BA
        let expected = a.clone() * b.clone() - b * a;
        assert_eq!(comm.expand().to_string(), expected.expand().to_string());
    }

    #[test]
    #[ignore = "Matrix syntax causes issues with some SymEngine configurations"]
    fn test_quantum_states() {
        let ket0 = states::ket_0();
        let ket1 = states::ket_1();
        let ket_plus = states::ket_plus();
        let bell = states::bell_phi_plus();

        assert!(ket0.to_string().contains("Matrix"));
        assert!(ket1.to_string().contains("Matrix"));
        assert!(ket_plus.to_string().contains("sqrt"));
        assert!(bell.to_string().contains("Matrix"));
    }

    #[test]
    #[ignore = "causes segfault in some SymEngine configurations"]
    fn test_bloch_state() {
        let theta = Expression::symbol("theta");
        let phi = Expression::symbol("phi");
        let state = states::bloch_state(&theta, &phi);

        assert!(state.to_string().contains("cos"));
        assert!(state.to_string().contains("exp"));
    }

    #[test]
    fn test_advanced_operators() {
        use advanced_operators::*;

        // Test creation and annihilation operators
        let a = annihilation();
        let a_dag = creation();
        assert!(a.to_string().contains('a'));
        assert!(a_dag.to_string().contains("a_dag"));

        // Test number operator
        let n_op = number_operator();
        assert!(n_op.to_string().contains("a_dag"));

        // Test position and momentum operators
        let x_op = position_operator();
        let p_op = momentum_operator();
        assert!(x_op.to_string().contains("sqrt"));
        assert!(p_op.to_string().contains('I'));
    }

    #[test]
    fn test_spin_operators() {
        use advanced_operators::*;

        let sx = spin_x();
        let sy = spin_y();
        let sz = spin_z();

        assert!(sx.to_string().contains("S_x"));
        assert!(sy.to_string().contains("S_y"));
        assert!(sz.to_string().contains("S_z"));

        let s_squared = spin_squared();
        assert!(!s_squared.to_string().is_empty());
    }

    #[test]
    fn test_angular_momentum_operators() {
        use advanced_operators::*;

        let lx = angular_momentum_x();
        let ly = angular_momentum_y();
        let lz = angular_momentum_z();

        assert!(lx.to_string().contains("L_x"));
        assert!(ly.to_string().contains("L_y"));
        assert!(lz.to_string().contains("L_z"));

        let l_squared = angular_momentum_squared();
        assert!(!l_squared.to_string().is_empty());
    }

    #[test]
    fn test_fermionic_operators() {
        use advanced_operators::*;

        let c = fermionic_annihilation();
        let c_dag = fermionic_creation();

        assert!(c.to_string().contains('c'));
        assert!(c_dag.to_string().contains("c_dag"));

        let n_op = fermionic_number_operator();
        assert!(n_op.to_string().contains("c_dag"));
    }

    #[test]
    fn test_special_operators() {
        use advanced_operators::*;

        let alpha = Expression::symbol("alpha");
        let disp = displacement_operator(&alpha);
        assert!(disp.to_string().contains("exp"));

        let zeta = Expression::symbol("zeta");
        let squeeze = squeezing_operator(&zeta);
        assert!(squeeze.to_string().contains("exp"));
    }
}
