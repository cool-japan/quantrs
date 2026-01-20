//! Hypergraph product codes for quantum LDPC

use super::pauli::{Pauli, PauliString};
use super::stabilizer::StabilizerCode;
use crate::error::QuantRS2Result;
use scirs2_core::ndarray::Array2;

/// Hypergraph product codes for quantum LDPC
#[derive(Debug, Clone)]
pub struct HypergraphProductCode {
    /// Number of physical qubits
    pub n: usize,
    /// Number of logical qubits
    pub k: usize,
    /// X-type stabilizers
    pub x_stabilizers: Vec<PauliString>,
    /// Z-type stabilizers
    pub z_stabilizers: Vec<PauliString>,
}

impl HypergraphProductCode {
    /// Create hypergraph product code from two classical codes
    pub fn new(h1: Array2<u8>, h2: Array2<u8>) -> Self {
        let (m1, n1) = h1.dim();
        let (m2, n2) = h2.dim();

        let n = n1 * m2 + m1 * n2;
        let k = (n1 - m1) * (n2 - m2);

        let mut x_stabilizers = Vec::new();
        let mut z_stabilizers = Vec::new();

        // X-type stabilizers: H1 ⊗ I2
        for i in 0..m1 {
            for j in 0..m2 {
                let mut paulis = vec![Pauli::I; n];

                // Apply H1 to first block
                for l in 0..n1 {
                    if h1[[i, l]] == 1 {
                        paulis[l * m2 + j] = Pauli::X;
                    }
                }

                x_stabilizers.push(PauliString::new(paulis));
            }
        }

        // Z-type stabilizers: I1 ⊗ H2^T
        for i in 0..m1 {
            for j in 0..m2 {
                let mut paulis = vec![Pauli::I; n];

                // Apply H2^T to second block
                for l in 0..n2 {
                    if h2[[j, l]] == 1 {
                        paulis[n1 * m2 + i * n2 + l] = Pauli::Z;
                    }
                }

                z_stabilizers.push(PauliString::new(paulis));
            }
        }

        Self {
            n,
            k,
            x_stabilizers,
            z_stabilizers,
        }
    }

    /// Convert to stabilizer code representation
    pub fn to_stabilizer_code(&self) -> QuantRS2Result<StabilizerCode> {
        let mut stabilizers = self.x_stabilizers.clone();
        stabilizers.extend(self.z_stabilizers.clone());

        // Simplified logical operators
        let logical_x = vec![PauliString::new(vec![Pauli::X; self.n])];
        let logical_z = vec![PauliString::new(vec![Pauli::Z; self.n])];

        StabilizerCode::new(
            self.n,
            self.k,
            3, // Simplified distance
            stabilizers,
            logical_x,
            logical_z,
        )
    }
}
