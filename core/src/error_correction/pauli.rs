//! Pauli operators and strings for quantum error correction

use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use std::fmt;

/// Pauli operator representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

impl Pauli {
    /// Get matrix representation
    pub fn matrix(&self) -> Array2<Complex64> {
        match self {
            Self::I => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .expect("Pauli I matrix: 2x2 shape with 4 elements is always valid"),
            Self::X => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .expect("Pauli X matrix: 2x2 shape with 4 elements is always valid"),
            Self::Y => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, -1.0),
                    Complex64::new(0.0, 1.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .expect("Pauli Y matrix: 2x2 shape with 4 elements is always valid"),
            Self::Z => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )
            .expect("Pauli Z matrix: 2x2 shape with 4 elements is always valid"),
        }
    }

    /// Multiply two Pauli operators
    pub const fn multiply(&self, other: &Self) -> (Complex64, Self) {
        use Pauli::{I, X, Y, Z};
        match (self, other) {
            (I, p) | (p, I) => (Complex64::new(1.0, 0.0), *p),
            (X, X) | (Y, Y) | (Z, Z) => (Complex64::new(1.0, 0.0), I),
            (X, Y) => (Complex64::new(0.0, 1.0), Z),
            (Y, X) => (Complex64::new(0.0, -1.0), Z),
            (Y, Z) => (Complex64::new(0.0, 1.0), X),
            (Z, Y) => (Complex64::new(0.0, -1.0), X),
            (Z, X) => (Complex64::new(0.0, 1.0), Y),
            (X, Z) => (Complex64::new(0.0, -1.0), Y),
        }
    }
}

/// Multi-qubit Pauli operator
#[derive(Debug, Clone, PartialEq)]
pub struct PauliString {
    /// Phase factor (±1, ±i)
    pub phase: Complex64,
    /// Pauli operators for each qubit
    pub paulis: Vec<Pauli>,
}

impl PauliString {
    /// Create a new Pauli string
    pub const fn new(paulis: Vec<Pauli>) -> Self {
        Self {
            phase: Complex64::new(1.0, 0.0),
            paulis,
        }
    }

    /// Create identity on n qubits
    pub fn identity(n: usize) -> Self {
        Self::new(vec![Pauli::I; n])
    }

    /// Get the weight (number of non-identity operators)
    pub fn weight(&self) -> usize {
        self.paulis.iter().filter(|&&p| p != Pauli::I).count()
    }

    /// Multiply two Pauli strings
    pub fn multiply(&self, other: &Self) -> QuantRS2Result<Self> {
        if self.paulis.len() != other.paulis.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Pauli strings must have same length".to_string(),
            ));
        }

        let mut phase = self.phase * other.phase;
        let mut paulis = Vec::with_capacity(self.paulis.len());

        for (p1, p2) in self.paulis.iter().zip(&other.paulis) {
            let (factor, result) = p1.multiply(p2);
            phase *= factor;
            paulis.push(result);
        }

        Ok(Self { phase, paulis })
    }

    /// Check if two Pauli strings commute
    pub fn commutes_with(&self, other: &Self) -> QuantRS2Result<bool> {
        if self.paulis.len() != other.paulis.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Pauli strings must have same length".to_string(),
            ));
        }

        let mut commutation_count = 0;
        for (p1, p2) in self.paulis.iter().zip(&other.paulis) {
            if *p1 != Pauli::I && *p2 != Pauli::I && p1 != p2 {
                commutation_count += 1;
            }
        }

        Ok(commutation_count % 2 == 0)
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let phase_str = if self.phase == Complex64::new(1.0, 0.0) {
            "+".to_string()
        } else if self.phase == Complex64::new(-1.0, 0.0) {
            "-".to_string()
        } else if self.phase == Complex64::new(0.0, 1.0) {
            "+i".to_string()
        } else {
            "-i".to_string()
        };

        write!(f, "{phase_str}")?;
        for p in &self.paulis {
            write!(f, "{p:?}")?;
        }
        Ok(())
    }
}
