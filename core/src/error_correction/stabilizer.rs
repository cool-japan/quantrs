//! Stabilizer code implementation

use super::pauli::{Pauli, PauliString};
use crate::error::{QuantRS2Error, QuantRS2Result};

/// Stabilizer code definition
#[derive(Debug, Clone)]
pub struct StabilizerCode {
    /// Number of physical qubits
    pub n: usize,
    /// Number of logical qubits
    pub k: usize,
    /// Minimum distance
    pub d: usize,
    /// Stabilizer generators
    pub stabilizers: Vec<PauliString>,
    /// Logical X operators
    pub logical_x: Vec<PauliString>,
    /// Logical Z operators
    pub logical_z: Vec<PauliString>,
}

impl StabilizerCode {
    /// Create a new stabilizer code
    pub fn new(
        n: usize,
        k: usize,
        d: usize,
        stabilizers: Vec<PauliString>,
        logical_x: Vec<PauliString>,
        logical_z: Vec<PauliString>,
    ) -> QuantRS2Result<Self> {
        // Validate code parameters
        // Note: For surface codes and other topological codes,
        // some stabilizers may be linearly dependent, so we allow more flexibility
        if stabilizers.len() > 2 * (n - k) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Too many stabilizers: got {}, maximum is {}",
                stabilizers.len(),
                2 * (n - k)
            )));
        }

        if logical_x.len() != k || logical_z.len() != k {
            return Err(QuantRS2Error::InvalidInput(
                "Number of logical operators must equal k".to_string(),
            ));
        }

        // Check that stabilizers commute
        for i in 0..stabilizers.len() {
            for j in i + 1..stabilizers.len() {
                if !stabilizers[i].commutes_with(&stabilizers[j])? {
                    return Err(QuantRS2Error::InvalidInput(
                        "Stabilizers must commute".to_string(),
                    ));
                }
            }
        }

        Ok(Self {
            n,
            k,
            d,
            stabilizers,
            logical_x,
            logical_z,
        })
    }

    /// Create the 3-qubit repetition code
    pub fn repetition_code() -> Self {
        let stabilizers = vec![
            PauliString::new(vec![Pauli::Z, Pauli::Z, Pauli::I]),
            PauliString::new(vec![Pauli::I, Pauli::Z, Pauli::Z]),
        ];

        let logical_x = vec![PauliString::new(vec![Pauli::X, Pauli::X, Pauli::X])];
        let logical_z = vec![PauliString::new(vec![Pauli::Z, Pauli::I, Pauli::I])];

        Self::new(3, 1, 1, stabilizers, logical_x, logical_z)
            .expect("3-qubit repetition code: verified standard code parameters are always valid")
    }

    /// Create the 5-qubit perfect code
    pub fn five_qubit_code() -> Self {
        let stabilizers = vec![
            PauliString::new(vec![Pauli::X, Pauli::Z, Pauli::Z, Pauli::X, Pauli::I]),
            PauliString::new(vec![Pauli::I, Pauli::X, Pauli::Z, Pauli::Z, Pauli::X]),
            PauliString::new(vec![Pauli::X, Pauli::I, Pauli::X, Pauli::Z, Pauli::Z]),
            PauliString::new(vec![Pauli::Z, Pauli::X, Pauli::I, Pauli::X, Pauli::Z]),
        ];

        let logical_x = vec![PauliString::new(vec![
            Pauli::X,
            Pauli::X,
            Pauli::X,
            Pauli::X,
            Pauli::X,
        ])];
        let logical_z = vec![PauliString::new(vec![
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
        ])];

        Self::new(5, 1, 3, stabilizers, logical_x, logical_z)
            .expect("5-qubit perfect code: verified standard code parameters are always valid")
    }

    /// Create the 7-qubit Steane code
    pub fn steane_code() -> Self {
        let stabilizers = vec![
            PauliString::new(vec![
                Pauli::I,
                Pauli::I,
                Pauli::I,
                Pauli::X,
                Pauli::X,
                Pauli::X,
                Pauli::X,
            ]),
            PauliString::new(vec![
                Pauli::I,
                Pauli::X,
                Pauli::X,
                Pauli::I,
                Pauli::I,
                Pauli::X,
                Pauli::X,
            ]),
            PauliString::new(vec![
                Pauli::X,
                Pauli::I,
                Pauli::X,
                Pauli::I,
                Pauli::X,
                Pauli::I,
                Pauli::X,
            ]),
            PauliString::new(vec![
                Pauli::I,
                Pauli::I,
                Pauli::I,
                Pauli::Z,
                Pauli::Z,
                Pauli::Z,
                Pauli::Z,
            ]),
            PauliString::new(vec![
                Pauli::I,
                Pauli::Z,
                Pauli::Z,
                Pauli::I,
                Pauli::I,
                Pauli::Z,
                Pauli::Z,
            ]),
            PauliString::new(vec![
                Pauli::Z,
                Pauli::I,
                Pauli::Z,
                Pauli::I,
                Pauli::Z,
                Pauli::I,
                Pauli::Z,
            ]),
        ];

        let logical_x = vec![PauliString::new(vec![
            Pauli::X,
            Pauli::X,
            Pauli::X,
            Pauli::X,
            Pauli::X,
            Pauli::X,
            Pauli::X,
        ])];
        let logical_z = vec![PauliString::new(vec![
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
        ])];

        Self::new(7, 1, 3, stabilizers, logical_x, logical_z)
            .expect("7-qubit Steane code: verified standard code parameters are always valid")
    }

    /// Get syndrome for a given error
    pub fn syndrome(&self, error: &PauliString) -> QuantRS2Result<Vec<bool>> {
        if error.paulis.len() != self.n {
            return Err(QuantRS2Error::InvalidInput(
                "Error must act on all physical qubits".to_string(),
            ));
        }

        let mut syndrome = Vec::with_capacity(self.stabilizers.len());
        for stabilizer in &self.stabilizers {
            syndrome.push(!stabilizer.commutes_with(error)?);
        }

        Ok(syndrome)
    }
}
