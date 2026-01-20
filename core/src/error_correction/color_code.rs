//! Color code implementation

use super::pauli::{Pauli, PauliString};
use super::stabilizer::StabilizerCode;
use crate::error::QuantRS2Result;
use std::collections::HashMap;

/// Color enumeration for face coloring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Red,
    Green,
    Blue,
}

/// Color code
#[derive(Debug, Clone)]
pub struct ColorCode {
    /// Number of physical qubits
    pub n: usize,
    /// Face coloring (red, green, blue)
    pub faces: Vec<(Vec<usize>, Color)>,
    /// Vertex to qubit mapping
    pub vertex_map: HashMap<(i32, i32), usize>,
}

impl ColorCode {
    /// Create a triangular color code
    pub fn triangular(size: usize) -> Self {
        let mut vertex_map = HashMap::new();
        let mut qubit_index = 0;

        // Create hexagonal lattice vertices
        for i in 0..size as i32 {
            for j in 0..size as i32 {
                vertex_map.insert((i, j), qubit_index);
                qubit_index += 1;
            }
        }

        let mut faces = Vec::new();

        // Create colored faces
        for i in 0..size as i32 - 1 {
            for j in 0..size as i32 - 1 {
                // Red face
                if let (Some(&q1), Some(&q2), Some(&q3)) = (
                    vertex_map.get(&(i, j)),
                    vertex_map.get(&(i + 1, j)),
                    vertex_map.get(&(i, j + 1)),
                ) {
                    faces.push((vec![q1, q2, q3], Color::Red));
                }

                // Green face
                if let (Some(&q1), Some(&q2), Some(&q3)) = (
                    vertex_map.get(&(i + 1, j)),
                    vertex_map.get(&(i + 1, j + 1)),
                    vertex_map.get(&(i, j + 1)),
                ) {
                    faces.push((vec![q1, q2, q3], Color::Green));
                }
            }
        }

        Self {
            n: vertex_map.len(),
            faces,
            vertex_map,
        }
    }

    /// Convert to stabilizer code
    pub fn to_stabilizer_code(&self) -> QuantRS2Result<StabilizerCode> {
        let mut x_stabilizers = Vec::new();
        let mut z_stabilizers = Vec::new();

        for (qubits, _color) in &self.faces {
            // X-type stabilizer
            let mut x_paulis = vec![Pauli::I; self.n];
            for &q in qubits {
                x_paulis[q] = Pauli::X;
            }
            x_stabilizers.push(PauliString::new(x_paulis));

            // Z-type stabilizer
            let mut z_paulis = vec![Pauli::I; self.n];
            for &q in qubits {
                z_paulis[q] = Pauli::Z;
            }
            z_stabilizers.push(PauliString::new(z_paulis));
        }

        let mut stabilizers = x_stabilizers;
        stabilizers.extend(z_stabilizers);

        // Simplified logical operators
        let logical_x = vec![PauliString::new(vec![Pauli::X; self.n])];
        let logical_z = vec![PauliString::new(vec![Pauli::Z; self.n])];

        StabilizerCode::new(
            self.n,
            1,
            3, // minimum distance
            stabilizers,
            logical_x,
            logical_z,
        )
    }
}
