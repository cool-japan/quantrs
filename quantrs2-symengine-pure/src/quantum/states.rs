//! Quantum state representations.

use crate::expr::Expression;

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

/// GHZ state for n qubits: (|00...0⟩ + |11...1⟩)/√2
#[must_use]
pub fn ghz_state(n: usize) -> Expression {
    if n == 0 {
        return Expression::one();
    }

    let dim = 1 << n; // 2^n
    let mut elements = vec!["0".to_string(); dim];
    elements[0] = "1".to_string();
    elements[dim - 1] = "1".to_string();

    let matrix_str = elements
        .iter()
        .map(|e| format!("[{e}]"))
        .collect::<Vec<_>>()
        .join(", ");

    Expression::new(format!("1/sqrt(2) * Matrix([{matrix_str}])"))
}

/// W state for n qubits: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
#[must_use]
pub fn w_state(n: usize) -> Expression {
    if n == 0 {
        return Expression::one();
    }

    let dim = 1 << n; // 2^n
    let mut elements = vec!["0".to_string(); dim];

    // W state has exactly one 1 in each basis state
    for i in 0..n {
        let idx = 1 << (n - 1 - i); // Position of the 1-qubit
        elements[idx] = "1".to_string();
    }

    let matrix_str = elements
        .iter()
        .map(|e| format!("[{e}]"))
        .collect::<Vec<_>>()
        .join(", ");

    Expression::new(format!("1/sqrt({n}) * Matrix([{matrix_str}])"))
}

/// Coherent state |α⟩ (for harmonic oscillator)
///
/// Eigenstate of the annihilation operator: a|α⟩ = α|α⟩
#[must_use]
pub fn coherent_state(alpha: &Expression) -> Expression {
    // |α⟩ = exp(-|α|²/2) Σ_n (α^n/√n!) |n⟩
    // We represent this symbolically
    Expression::new(format!("CoherentState({alpha})"))
}

/// Squeezed vacuum state |ζ⟩
///
/// S(ζ)|0⟩ where S is the squeezing operator
#[must_use]
pub fn squeezed_vacuum(zeta: &Expression) -> Expression {
    Expression::new(format!("SqueezedVacuum({zeta})"))
}

/// Thermal state (density matrix)
///
/// ρ = (1 - e^(-ℏω/kT)) Σ_n e^(-nℏω/kT) |n⟩⟨n|
#[must_use]
pub fn thermal_state(n_thermal: &Expression) -> Expression {
    // n_thermal is the mean photon number
    Expression::new(format!("ThermalState({n_thermal})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_computational_basis() {
        let ket0 = ket_0();
        let ket1 = ket_1();

        // State expressions are stored as symbolic strings
        assert!(!ket0.to_string().is_empty());
        assert!(!ket1.to_string().is_empty());
    }

    #[test]
    fn test_bell_states() {
        let phi_plus = bell_phi_plus();
        let psi_minus = bell_psi_minus();

        assert!(!phi_plus.to_string().is_empty());
        assert!(!psi_minus.to_string().is_empty());
    }

    #[test]
    fn test_ghz_state() {
        let ghz3 = ghz_state(3);
        assert!(ghz3.to_string().contains("Matrix"));
    }

    #[test]
    fn test_w_state() {
        let w3 = w_state(3);
        assert!(w3.to_string().contains("Matrix"));
    }
}
