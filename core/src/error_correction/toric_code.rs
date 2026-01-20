//! Toric code (generalization of surface code on torus)

use super::pauli::{Pauli, PauliString};
use super::stabilizer::StabilizerCode;
use crate::error::QuantRS2Result;
use std::collections::HashMap;

/// Toric code (generalization of surface code on torus)
#[derive(Debug, Clone)]
pub struct ToricCode {
    /// Number of rows in the torus
    pub rows: usize,
    /// Number of columns in the torus
    pub cols: usize,
    /// Qubit mapping
    pub qubit_map: HashMap<(usize, usize), usize>,
}

impl ToricCode {
    /// Create a new toric code
    pub fn new(rows: usize, cols: usize) -> Self {
        let mut qubit_map = HashMap::new();
        let mut qubit_index = 0;

        // Place qubits on torus (two qubits per unit cell)
        for r in 0..rows {
            for c in 0..cols {
                // Horizontal edge qubit
                qubit_map.insert((2 * r, c), qubit_index);
                qubit_index += 1;
                // Vertical edge qubit
                qubit_map.insert((2 * r + 1, c), qubit_index);
                qubit_index += 1;
            }
        }

        Self {
            rows,
            cols,
            qubit_map,
        }
    }

    /// Get number of logical qubits
    pub const fn logical_qubits(&self) -> usize {
        2 // Two logical qubits for torus topology
    }

    /// Get code distance
    pub fn distance(&self) -> usize {
        self.rows.min(self.cols)
    }

    /// Convert to stabilizer code representation
    pub fn to_stabilizer_code(&self) -> QuantRS2Result<StabilizerCode> {
        let n = self.qubit_map.len();
        let mut stabilizers = Vec::new();

        // Vertex stabilizers (X-type) - star operators
        for r in 0..self.rows {
            for c in 0..self.cols {
                let mut paulis = vec![Pauli::I; n];

                // Four edges around vertex with correct torus indexing
                let h_edge_below = (2 * r, c);
                let h_edge_above = (2 * ((r + self.rows - 1) % self.rows), c);
                let v_edge_left = (2 * r + 1, (c + self.cols - 1) % self.cols);
                let v_edge_right = (2 * r + 1, c);

                for &coord in &[h_edge_below, h_edge_above, v_edge_left, v_edge_right] {
                    if let Some(&qubit) = self.qubit_map.get(&coord) {
                        paulis[qubit] = Pauli::X;
                    }
                }

                stabilizers.push(PauliString::new(paulis));
            }
        }

        // Plaquette stabilizers (Z-type) - face operators
        for r in 0..self.rows {
            for c in 0..self.cols {
                let mut paulis = vec![Pauli::I; n];

                // Four edges around plaquette with correct indexing
                let h_edge_top = (2 * r, c);
                let h_edge_bottom = (2 * ((r + 1) % self.rows), c);
                let v_edge_left = (2 * r + 1, c);
                let v_edge_right = (2 * r + 1, (c + 1) % self.cols);

                for &coord in &[h_edge_top, h_edge_bottom, v_edge_left, v_edge_right] {
                    if let Some(&qubit) = self.qubit_map.get(&coord) {
                        paulis[qubit] = Pauli::Z;
                    }
                }

                stabilizers.push(PauliString::new(paulis));
            }
        }

        // Logical operators (horizontal and vertical loops)
        let mut logical_x1 = vec![Pauli::I; n];
        let mut logical_z1 = vec![Pauli::I; n];
        let mut logical_x2 = vec![Pauli::I; n];
        let mut logical_z2 = vec![Pauli::I; n];

        // Horizontal logical loop operators
        for c in 0..self.cols {
            // Logical X along horizontal direction (vertical edges)
            if let Some(&qubit) = self.qubit_map.get(&(1, c)) {
                logical_x1[qubit] = Pauli::X;
            }
            // Logical Z along horizontal direction (horizontal edges)
            if let Some(&qubit) = self.qubit_map.get(&(0, c)) {
                logical_z2[qubit] = Pauli::Z;
            }
        }

        // Vertical logical loop operators
        for r in 0..self.rows {
            // Logical X along vertical direction (horizontal edges)
            if let Some(&qubit) = self.qubit_map.get(&(2 * r, 0)) {
                logical_x2[qubit] = Pauli::X;
            }
            // Logical Z along vertical direction (vertical edges)
            if let Some(&qubit) = self.qubit_map.get(&(2 * r + 1, 0)) {
                logical_z1[qubit] = Pauli::Z;
            }
        }

        let logical_x = vec![PauliString::new(logical_x1), PauliString::new(logical_x2)];
        let logical_z = vec![PauliString::new(logical_z1), PauliString::new(logical_z2)];

        StabilizerCode::new(n, 2, self.distance(), stabilizers, logical_x, logical_z)
    }
}
