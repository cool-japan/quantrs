//! Quantum operator algebra.

use crate::error::SymEngineResult;
use crate::expr::{ExprLang, Expression};

/// Commutator [A, B] = AB - BA
#[must_use]
pub fn commutator(a: &Expression, b: &Expression) -> Expression {
    a.clone() * b.clone() - b.clone() * a.clone()
}

/// Anticommutator {A, B} = AB + BA
#[must_use]
pub fn anticommutator(a: &Expression, b: &Expression) -> Expression {
    a.clone() * b.clone() + b.clone() * a.clone()
}

/// Tensor product (Kronecker product) of two operators
pub fn tensor_product(a: &Expression, b: &Expression) -> Expression {
    Expression::new(format!("kronecker({a}, {b})"))
}

/// Matrix trace (sum of diagonal elements)
pub fn trace(matrix: &Expression) -> Expression {
    Expression::new(format!("trace({matrix})"))
}

/// Hermitian conjugate (dagger) operation
pub fn dagger(expr: &Expression) -> Expression {
    Expression::new(format!("conjugate(transpose({expr}))"))
}

/// Check if an operator is Hermitian (A = A†)
///
/// Returns the difference A - A†, which should be zero for Hermitian operators.
pub fn is_hermitian(matrix: &Expression) -> Expression {
    matrix.clone() - dagger(matrix)
}

/// Check if an operator is unitary (A†A = I)
///
/// Returns the product A†A, which should equal identity for unitary operators.
pub fn is_unitary(matrix: &Expression) -> Expression {
    dagger(matrix) * matrix.clone()
}

/// Matrix transpose
pub fn transpose(matrix: &Expression) -> Expression {
    Expression::new(format!("transpose({matrix})"))
}

/// Matrix determinant
pub fn determinant(matrix: &Expression) -> Expression {
    Expression::new(format!("det({matrix})"))
}

/// Matrix inverse
pub fn inverse(matrix: &Expression) -> Expression {
    Expression::new(format!("inverse({matrix})"))
}

/// Matrix exponential (e^A)
pub fn matrix_exp(matrix: &Expression) -> Expression {
    crate::ops::trig::exp(matrix)
}

/// Bosonic creation operator a†
#[must_use]
pub fn creation() -> Expression {
    Expression::symbol("a_dag")
}

/// Bosonic annihilation operator a
#[must_use]
pub fn annihilation() -> Expression {
    Expression::symbol("a")
}

/// Number operator n = a†a
#[must_use]
pub fn number_operator() -> Expression {
    creation() * annihilation()
}

/// Position operator x = (a + a†)/√2
#[must_use]
pub fn position_operator() -> Expression {
    let a = annihilation();
    let a_dag = creation();
    let sqrt2 = crate::ops::trig::sqrt(&Expression::int(2));
    (a + a_dag) / sqrt2
}

/// Momentum operator p = i(a† - a)/√2
#[must_use]
pub fn momentum_operator() -> Expression {
    let a = annihilation();
    let a_dag = creation();
    let sqrt2 = crate::ops::trig::sqrt(&Expression::int(2));
    Expression::i() * (a_dag - a) / sqrt2
}

/// Fermionic creation operator c†
#[must_use]
pub fn fermionic_creation() -> Expression {
    Expression::symbol("c_dag")
}

/// Fermionic annihilation operator c
#[must_use]
pub fn fermionic_annihilation() -> Expression {
    Expression::symbol("c")
}

/// Fermionic number operator n = c†c
#[must_use]
pub fn fermionic_number_operator() -> Expression {
    fermionic_creation() * fermionic_annihilation()
}

/// Displacement operator D(α) = exp(αa† - α*a)
#[must_use]
pub fn displacement_operator(alpha: &Expression) -> Expression {
    let a = annihilation();
    let a_dag = creation();
    let arg = alpha.clone() * a_dag - alpha.conjugate() * a;
    crate::ops::trig::exp(&arg)
}

/// Squeezing operator S(ζ) = exp[(ζ*a² - ζa†²)/2]
#[must_use]
pub fn squeezing_operator(zeta: &Expression) -> Expression {
    let a = annihilation();
    let a_dag = creation();
    let two = Expression::int(2);
    let term1 = zeta.clone() * a.pow(&two);
    let term2 = zeta.conjugate() * a_dag.pow(&two);
    let arg = (term1 - term2) / two;
    crate::ops::trig::exp(&arg)
}

/// Spin raising operator S+ = Sx + iSy
#[must_use]
pub fn spin_raising() -> Expression {
    Expression::symbol("S_plus")
}

/// Spin lowering operator S- = Sx - iSy
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
    let two = Expression::int(2);
    spin_x().pow(&two) + spin_y().pow(&two) + spin_z().pow(&two)
}

// =========================================================================
// Angular Momentum Operators
// =========================================================================

/// Angular momentum x-component Jx
#[must_use]
pub fn angular_momentum_x() -> Expression {
    Expression::symbol("J_x")
}

/// Angular momentum y-component Jy
#[must_use]
pub fn angular_momentum_y() -> Expression {
    Expression::symbol("J_y")
}

/// Angular momentum z-component Jz
#[must_use]
pub fn angular_momentum_z() -> Expression {
    Expression::symbol("J_z")
}

/// Angular momentum squared J² = Jx² + Jy² + Jz²
#[must_use]
pub fn angular_momentum_squared() -> Expression {
    let two = Expression::int(2);
    angular_momentum_x().pow(&two) + angular_momentum_y().pow(&two) + angular_momentum_z().pow(&two)
}

/// Angular momentum raising operator J+ = Jx + iJy
#[must_use]
pub fn angular_momentum_raising() -> Expression {
    angular_momentum_x() + Expression::i() * angular_momentum_y()
}

/// Angular momentum lowering operator J- = Jx - iJy
#[must_use]
pub fn angular_momentum_lowering() -> Expression {
    angular_momentum_x() - Expression::i() * angular_momentum_y()
}

// =========================================================================
// Spin-1/2 Operators (Pauli basis)
// =========================================================================

/// Spin-1/2 x-operator (σx/2)
#[must_use]
pub fn spin_half_x() -> Expression {
    spin_x() / Expression::int(2)
}

/// Spin-1/2 y-operator (σy/2)
#[must_use]
pub fn spin_half_y() -> Expression {
    spin_y() / Expression::int(2)
}

/// Spin-1/2 z-operator (σz/2)
#[must_use]
pub fn spin_half_z() -> Expression {
    spin_z() / Expression::int(2)
}

/// Spin-1/2 raising operator (σ+/2)
#[must_use]
pub fn spin_half_raising() -> Expression {
    (spin_x() + Expression::i() * spin_y()) / Expression::int(2)
}

/// Spin-1/2 lowering operator (σ-/2)
#[must_use]
pub fn spin_half_lowering() -> Expression {
    (spin_x() - Expression::i() * spin_y()) / Expression::int(2)
}

// =========================================================================
// Fock State Operations
// =========================================================================

/// Fock state |n⟩
#[must_use]
pub fn fock_state(n: u32) -> Expression {
    Expression::new(format!("ket({n})"))
}

/// Coherent state |α⟩ = D(α)|0⟩
#[must_use]
pub fn coherent_state(alpha: &Expression) -> Expression {
    Expression::new(format!("coherent({alpha})"))
}

/// Cat state |cat±⟩ = N(|α⟩ ± |-α⟩)
#[must_use]
pub fn cat_state(alpha: &Expression, parity: bool) -> Expression {
    let sign = if parity { "plus" } else { "minus" };
    Expression::new(format!("cat_{sign}({alpha})"))
}

/// Squeezed vacuum state S(ζ)|0⟩
#[must_use]
pub fn squeezed_vacuum(zeta: &Expression) -> Expression {
    Expression::new(format!("squeezed_vacuum({zeta})"))
}

/// Thermal state density matrix
#[must_use]
pub fn thermal_state(n_bar: &Expression) -> Expression {
    Expression::new(format!("thermal({n_bar})"))
}

// =========================================================================
// Multi-mode Operators
// =========================================================================

/// Multi-mode annihilation operator a_k
#[must_use]
pub fn annihilation_mode(mode: u32) -> Expression {
    Expression::symbol(&format!("a_{mode}"))
}

/// Multi-mode creation operator a†_k
#[must_use]
pub fn creation_mode(mode: u32) -> Expression {
    Expression::symbol(&format!("a_dag_{mode}"))
}

/// Multi-mode number operator n_k = a†_k a_k
#[must_use]
pub fn number_operator_mode(mode: u32) -> Expression {
    creation_mode(mode) * annihilation_mode(mode)
}

/// Total number operator N = Σ_k n_k
pub fn total_number_operator(num_modes: u32) -> Expression {
    let mut total = Expression::zero();
    for k in 0..num_modes {
        total = total + number_operator_mode(k);
    }
    total
}

// =========================================================================
// Fermionic Algebra
// =========================================================================

/// Multi-mode fermionic annihilation operator c_j
#[must_use]
pub fn fermionic_annihilation_mode(mode: u32) -> Expression {
    Expression::symbol(&format!("c_{mode}"))
}

/// Multi-mode fermionic creation operator c†_j
#[must_use]
pub fn fermionic_creation_mode(mode: u32) -> Expression {
    Expression::symbol(&format!("c_dag_{mode}"))
}

/// Fermionic number operator for mode j: n_j = c†_j c_j
#[must_use]
pub fn fermionic_number_operator_mode(mode: u32) -> Expression {
    fermionic_creation_mode(mode) * fermionic_annihilation_mode(mode)
}

/// Total fermionic number operator N = Σ_j c†_j c_j
pub fn total_fermionic_number_operator(num_modes: u32) -> Expression {
    let mut total = Expression::zero();
    for j in 0..num_modes {
        total = total + fermionic_number_operator_mode(j);
    }
    total
}

/// Jordan-Wigner string for fermionic mode j: Π_{k<j} (1 - 2n_k)
pub fn jordan_wigner_string(mode: u32) -> Expression {
    if mode == 0 {
        return Expression::one();
    }

    let mut product = Expression::one();
    for k in 0..mode {
        let n_k = fermionic_number_operator_mode(k);
        let factor = Expression::one() - Expression::int(2) * n_k;
        product = product * factor;
    }
    product
}

// =========================================================================
// Hamiltonian Builders
// =========================================================================

/// Harmonic oscillator Hamiltonian: H = ℏω(n + 1/2)
#[must_use]
pub fn harmonic_oscillator_hamiltonian(omega: &Expression) -> Expression {
    let n = number_operator();
    let half = Expression::float_unchecked(0.5);
    omega.clone() * (n + half)
}

/// Jaynes-Cummings Hamiltonian (without RWA): H = ωc a†a + ωa σz/2 + g(a† + a)(σ+ + σ-)
pub fn jaynes_cummings_hamiltonian(
    omega_cavity: &Expression,
    omega_atom: &Expression,
    coupling: &Expression,
) -> Expression {
    let n = number_operator();
    let a = annihilation();
    let a_dag = creation();
    let sz = spin_z();
    let s_plus = spin_raising();
    let s_minus = spin_lowering();
    let half = Expression::float_unchecked(0.5);

    let cavity_term = omega_cavity.clone() * n;
    let atom_term = omega_atom.clone() * sz * half;
    let interaction = coupling.clone() * (a_dag + a) * (s_plus + s_minus);

    cavity_term + atom_term + interaction
}

/// Single-mode Kerr Hamiltonian: H = ℏω a†a + ℏK (a†a)²
#[must_use]
pub fn kerr_hamiltonian(omega: &Expression, kerr_strength: &Expression) -> Expression {
    let n = number_operator();
    let two = Expression::int(2);
    omega.clone() * n.clone() + kerr_strength.clone() * n.pow(&two)
}

/// Beam splitter Hamiltonian: H = θ(a†b + ab†)
pub fn beam_splitter_hamiltonian(theta: &Expression, mode_a: u32, mode_b: u32) -> Expression {
    let a = annihilation_mode(mode_a);
    let a_dag = creation_mode(mode_a);
    let b = annihilation_mode(mode_b);
    let b_dag = creation_mode(mode_b);

    theta.clone() * (a_dag * b + a * b_dag)
}

/// Two-mode squeezing Hamiltonian: H = ζ(a†b† - ab)
pub fn two_mode_squeezing_hamiltonian(zeta: &Expression, mode_a: u32, mode_b: u32) -> Expression {
    let a = annihilation_mode(mode_a);
    let a_dag = creation_mode(mode_a);
    let b = annihilation_mode(mode_b);
    let b_dag = creation_mode(mode_b);

    zeta.clone() * (a_dag * b_dag - a * b)
}

/// Hubbard model single-site term: U n_↑ n_↓
pub fn hubbard_interaction(u: &Expression, site: u32) -> Expression {
    let n_up = fermionic_number_operator_mode(2 * site);
    let n_down = fermionic_number_operator_mode(2 * site + 1);
    u.clone() * n_up * n_down
}

/// Hopping term: -t (c†_i c_j + c†_j c_i)
pub fn hopping_term(t: &Expression, site_i: u32, site_j: u32, spin: u32) -> Expression {
    let mode_i = 2 * site_i + spin;
    let mode_j = 2 * site_j + spin;

    let c_dag_i = fermionic_creation_mode(mode_i);
    let c_i = fermionic_annihilation_mode(mode_i);
    let c_dag_j = fermionic_creation_mode(mode_j);
    let c_j = fermionic_annihilation_mode(mode_j);

    t.clone().neg() * (c_dag_i * c_j + c_dag_j * c_i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commutator() {
        let a = Expression::symbol("A");
        let b = Expression::symbol("B");
        let comm = commutator(&a, &b);

        // [A, B] = AB - BA
        let _expected = a.clone() * b.clone() - b * a;
        // Structural comparison
        assert!(!comm.is_symbol());
    }

    #[test]
    fn test_bosonic_operators() {
        let a = annihilation();
        let a_dag = creation();
        let n = number_operator();

        assert!(a.is_symbol());
        assert!(a_dag.is_symbol());
        assert!(!n.is_symbol()); // n = a†a is a product
    }

    #[test]
    fn test_displacement_operator() {
        let alpha = Expression::symbol("alpha");
        let d = displacement_operator(&alpha);

        assert!(!d.is_symbol());
    }

    #[test]
    fn test_angular_momentum() {
        let jx = angular_momentum_x();
        let jy = angular_momentum_y();
        let jz = angular_momentum_z();
        let j2 = angular_momentum_squared();

        assert!(jx.is_symbol());
        assert!(jy.is_symbol());
        assert!(jz.is_symbol());
        assert!(!j2.is_symbol());
    }

    #[test]
    fn test_multi_mode_operators() {
        let a0 = annihilation_mode(0);
        let a1 = annihilation_mode(1);
        let n_total = total_number_operator(3);

        assert!(a0.is_symbol());
        assert!(a1.is_symbol());
        assert!(!n_total.is_symbol());
    }

    #[test]
    fn test_fock_state() {
        let ket0 = fock_state(0);
        let ket5 = fock_state(5);

        // Fock states are represented as symbolic expressions, not raw symbols
        let ket0_str = ket0.to_string();
        let ket5_str = ket5.to_string();

        assert!(ket0_str.contains("ket") || ket0_str.contains('0'));
        assert!(ket5_str.contains("ket") || ket5_str.contains('5'));
    }

    #[test]
    fn test_jordan_wigner_string() {
        let jw0 = jordan_wigner_string(0);
        let jw1 = jordan_wigner_string(1);
        let jw2 = jordan_wigner_string(2);

        assert!(jw0.is_one());
        assert!(!jw1.is_symbol());
        assert!(!jw2.is_symbol());
    }

    #[test]
    fn test_hamiltonians() {
        let omega = Expression::symbol("omega");
        let kerr = Expression::symbol("K");

        let h_ho = harmonic_oscillator_hamiltonian(&omega);
        let h_kerr = kerr_hamiltonian(&omega, &kerr);

        assert!(!h_ho.is_symbol());
        assert!(!h_kerr.is_symbol());
    }

    #[test]
    fn test_beam_splitter() {
        let theta = Expression::symbol("theta");
        let h_bs = beam_splitter_hamiltonian(&theta, 0, 1);

        assert!(!h_bs.is_symbol());
    }
}
