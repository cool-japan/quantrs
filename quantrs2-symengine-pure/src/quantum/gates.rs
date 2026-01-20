//! Common quantum gates.

use crate::expr::Expression;
use crate::ops::trig;

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

/// SWAP gate
///
/// ```text
/// SWAP = [[1, 0, 0, 0],
///         [0, 0, 1, 0],
///         [0, 1, 0, 0],
///         [0, 0, 0, 1]]
/// ```
#[must_use]
pub fn swap() -> Expression {
    Expression::new("Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])")
}

/// Rotation around X axis
///
/// ```text
/// Rx(θ) = [[cos(θ/2),   -i*sin(θ/2)],
///          [-i*sin(θ/2), cos(θ/2)  ]]
/// ```
#[must_use]
pub fn rx(theta: &Expression) -> Expression {
    let half = theta.clone() / Expression::int(2);
    let cos_half = trig::cos(&half);
    let sin_half = trig::sin(&half);
    let i_sin = Expression::i().neg() * sin_half;

    // Construct matrix symbolically
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
        "Matrix([[cos({theta}/2), -exp(I*{lambda})*sin({theta}/2)], \
         [exp(I*{phi})*sin({theta}/2), exp(I*({phi}+{lambda}))*cos({theta}/2)]])"
    ))
}

/// Controlled-Z gate
///
/// ```text
/// CZ = [[1, 0, 0, 0],
///       [0, 1, 0, 0],
///       [0, 0, 1, 0],
///       [0, 0, 0, -1]]
/// ```
#[must_use]
pub fn cz() -> Expression {
    Expression::new("Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])")
}

/// Toffoli gate (CCX)
///
/// 8x8 matrix for 3-qubit controlled-controlled-NOT
#[must_use]
pub fn toffoli() -> Expression {
    Expression::new(
        "Matrix([[1,0,0,0,0,0,0,0],\
         [0,1,0,0,0,0,0,0],\
         [0,0,1,0,0,0,0,0],\
         [0,0,0,1,0,0,0,0],\
         [0,0,0,0,1,0,0,0],\
         [0,0,0,0,0,1,0,0],\
         [0,0,0,0,0,0,0,1],\
         [0,0,0,0,0,0,1,0]])",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_gates() {
        let h = hadamard();
        let s = phase();
        let t = t_gate();
        let cx = cnot();

        // Gate expressions are stored as symbolic strings for now
        assert!(!h.to_string().is_empty());
        assert!(!s.to_string().is_empty());
        assert!(!t.to_string().is_empty());
        assert!(!cx.to_string().is_empty());
    }

    #[test]
    fn test_rotation_gates() {
        let theta = Expression::symbol("theta");
        let rx_gate = rx(&theta);
        let ry_gate = ry(&theta);
        let rz_gate = rz(&theta);

        // Rotation gates contain the parameter theta
        assert!(rx_gate.to_string().contains("theta"));
        assert!(ry_gate.to_string().contains("theta"));
        assert!(rz_gate.to_string().contains("theta"));
    }
}
