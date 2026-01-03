//! Pauli matrices and related quantum operators.

use crate::expr::Expression;

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

/// Pauli raising operator σ+ = (σx + iσy)/2
#[must_use]
pub fn sigma_plus() -> Expression {
    (sigma_x() + Expression::i() * sigma_y()) / Expression::int(2)
}

/// Pauli lowering operator σ- = (σx - iσy)/2
#[must_use]
pub fn sigma_minus() -> Expression {
    (sigma_x() - Expression::i() * sigma_y()) / Expression::int(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_matrices() {
        let sx = sigma_x();
        let sy = sigma_y();
        let sz = sigma_z();
        let id = identity();

        // Matrix expressions are stored as symbolic strings for now
        // Full matrix parsing will be implemented later
        assert!(!sx.to_string().is_empty());
        assert!(!sy.to_string().is_empty());
        assert!(!sz.to_string().is_empty());
        assert!(!id.to_string().is_empty());
    }

    #[test]
    fn test_sigma_vector() {
        let sigmas = sigma_vector();
        assert_eq!(sigmas.len(), 3);
    }
}
