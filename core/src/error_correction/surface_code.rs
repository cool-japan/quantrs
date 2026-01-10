//! Surface code implementation

use super::pauli::{Pauli, PauliString};
use super::stabilizer::StabilizerCode;
use crate::error::QuantRS2Result;
use std::collections::HashMap;

/// Surface code lattice
#[derive(Debug, Clone)]
pub struct SurfaceCode {
    /// Number of rows in the lattice
    pub rows: usize,
    /// Number of columns in the lattice
    pub cols: usize,
    /// Qubit positions (row, col) -> qubit index
    pub qubit_map: HashMap<(usize, usize), usize>,
    /// Stabilizer plaquettes
    pub x_stabilizers: Vec<Vec<usize>>,
    pub z_stabilizers: Vec<Vec<usize>>,
}

impl SurfaceCode {
    /// Create a new surface code
    pub fn new(rows: usize, cols: usize) -> Self {
        let mut qubit_map = HashMap::new();
        let mut qubit_index = 0;

        // Place qubits on the lattice
        for r in 0..rows {
            for c in 0..cols {
                qubit_map.insert((r, c), qubit_index);
                qubit_index += 1;
            }
        }

        let mut x_stabilizers = Vec::new();
        let mut z_stabilizers = Vec::new();

        // Create X stabilizers (vertex operators)
        for r in 0..rows - 1 {
            for c in 0..cols - 1 {
                if (r + c) % 2 == 0 {
                    let stabilizer = vec![
                        qubit_map[&(r, c)],
                        qubit_map[&(r, c + 1)],
                        qubit_map[&(r + 1, c)],
                        qubit_map[&(r + 1, c + 1)],
                    ];
                    x_stabilizers.push(stabilizer);
                }
            }
        }

        // Create Z stabilizers (plaquette operators)
        for r in 0..rows - 1 {
            for c in 0..cols - 1 {
                if (r + c) % 2 == 1 {
                    let stabilizer = vec![
                        qubit_map[&(r, c)],
                        qubit_map[&(r, c + 1)],
                        qubit_map[&(r + 1, c)],
                        qubit_map[&(r + 1, c + 1)],
                    ];
                    z_stabilizers.push(stabilizer);
                }
            }
        }

        Self {
            rows,
            cols,
            qubit_map,
            x_stabilizers,
            z_stabilizers,
        }
    }

    /// Get the code distance
    pub fn distance(&self) -> usize {
        self.rows.min(self.cols)
    }

    /// Convert to stabilizer code representation
    pub fn to_stabilizer_code(&self) -> QuantRS2Result<StabilizerCode> {
        let n = self.qubit_map.len();
        let mut stabilizers = Vec::new();

        // Add X stabilizers
        for x_stab in &self.x_stabilizers {
            let mut paulis = vec![Pauli::I; n];
            for &qubit in x_stab {
                paulis[qubit] = Pauli::X;
            }
            stabilizers.push(PauliString::new(paulis));
        }

        // Add Z stabilizers
        for z_stab in &self.z_stabilizers {
            let mut paulis = vec![Pauli::I; n];
            for &qubit in z_stab {
                paulis[qubit] = Pauli::Z;
            }
            stabilizers.push(PauliString::new(paulis));
        }

        // Create logical operators (simplified - just use boundary chains)
        let mut logical_x_paulis = vec![Pauli::I; n];
        let mut logical_z_paulis = vec![Pauli::I; n];

        // Logical X: horizontal chain on top boundary
        for c in 0..self.cols {
            if let Some(&qubit) = self.qubit_map.get(&(0, c)) {
                logical_x_paulis[qubit] = Pauli::X;
            }
        }

        // Logical Z: vertical chain on left boundary
        for r in 0..self.rows {
            if let Some(&qubit) = self.qubit_map.get(&(r, 0)) {
                logical_z_paulis[qubit] = Pauli::Z;
            }
        }

        let logical_x = vec![PauliString::new(logical_x_paulis)];
        let logical_z = vec![PauliString::new(logical_z_paulis)];

        StabilizerCode::new(n, 1, self.distance(), stabilizers, logical_x, logical_z)
    }
}
