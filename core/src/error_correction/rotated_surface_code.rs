//! Rotated planar surface code implementation.
//!
//! The rotated surface code (also called the rotated planar code) is a variant of the
//! surface code with a 45-degree rotation that achieves the same [[d², 1, d]] parameters
//! with fewer boundary ancillas than the standard planar code.
//!
//! # Layout Convention
//!
//! Data qubits sit at integer coordinates `(row, col)` with `0 ≤ row, col < d`.
//! Index: `row * d + col`.
//!
//! Stabilizer ancillas are associated with plaquettes in the 2D checkerboard:
//! - Interior plaquettes at `(r+0.5, c+0.5)` for `0 ≤ r, c < d-1`.
//!   Parity `(r + c) % 2 == 0` → X-type; `(r + c) % 2 == 1` → Z-type.
//! - Boundary half-weight plaquettes:
//!   * Top/bottom edges: X-type
//!   * Left/right edges: Z-type
//!
//! # Encoding
//!
//! `[[d², 1, d]]`: `d²` physical qubits, 1 logical qubit, distance `d`.

use crate::error::QuantRS2Result;
use crate::error_correction::pauli::{Pauli, PauliString};

/// A rotated planar surface code of distance `d`.
///
/// See the module-level documentation for the layout convention.
#[derive(Debug, Clone)]
pub struct RotatedSurfaceCode {
    /// Code distance.
    pub distance: usize,
    /// Data qubit coordinates `(row, col)`.
    pub data_qubits: Vec<(usize, usize)>,
    /// X-ancilla positions stored as `(2*row+1, 2*col+1)` in double-integer coords,
    /// plus boundary half-plaquettes stored with a sentinel value.
    pub x_ancillas: Vec<(usize, usize)>,
    /// Z-ancilla positions (same encoding).
    pub z_ancillas: Vec<(usize, usize)>,
}

impl RotatedSurfaceCode {
    /// Construct a `d × d` rotated planar surface code.
    ///
    /// # Panics
    ///
    /// Panics if `d < 2`.
    pub fn new(d: usize) -> Self {
        assert!(d >= 2, "Surface code distance must be at least 2");

        // Data qubits: d² positions in row-major order
        let data_qubits: Vec<(usize, usize)> = (0..d)
            .flat_map(|r| (0..d).map(move |c| (r, c)))
            .collect();

        // We represent ancilla positions in 2× integer coordinates to avoid floats.
        // A real-coordinate position (r + 0.5, c + 0.5) becomes (2r+1, 2c+1).
        // Boundary half-plaquettes extend beyond the data grid:
        //   Top boundary:    (−0.5, c + 0.5)  →  stored as  (usize::MAX, 2c+1)
        //   Bottom boundary: (d−0.5, c + 0.5) →  stored as  (2d, 2c+1)   [2d > 2*(d-1)+1]
        //   Left boundary:   (r + 0.5, −0.5)  →  stored as  (2r+1, usize::MAX)
        //   Right boundary:  (r + 0.5, d−0.5) →  stored as  (2r+1, 2d)

        let mut x_ancillas: Vec<(usize, usize)> = Vec::new();
        let mut z_ancillas: Vec<(usize, usize)> = Vec::new();

        // 1. Interior plaquettes: (d-1)×(d-1) positions
        for pr in 0..d - 1 {
            for pc in 0..d - 1 {
                let coord = (2 * pr + 1, 2 * pc + 1);
                if (pr + pc) % 2 == 0 {
                    x_ancillas.push(coord);
                } else {
                    z_ancillas.push(coord);
                }
            }
        }

        // 2. Boundary half-plaquettes.
        //
        // For the rotated code the X-boundaries are on the top and bottom edges
        // and Z-boundaries are on the left and right edges.
        //
        // Top edge: X-type half-plaquettes at plaquette column c where
        //   the interior plaquette at (0, c) would be X-type, i.e., (0+c)%2==0 → c even.
        //   Half-plaquette involves data qubits (0, c) and (0, c+1).
        //
        // Bottom edge: X-type half-plaquettes at column c where
        //   the interior plaquette at (d-2, c) would be X-type, i.e., (d-2+c)%2==0.
        //   For even (d-2): c even; for odd (d-2): c odd.
        //   Half-plaquette involves data qubits (d-1, c) and (d-1, c+1).
        //
        // Left edge: Z-type half-plaquettes at row r where
        //   the interior plaquette at (r, 0) would be Z-type, i.e., (r+0)%2==1 → r odd.
        //   Half-plaquette involves data qubits (r, 0) and (r+1, 0).
        //
        // Right edge: Z-type half-plaquettes at row r where
        //   the interior plaquette at (r, d-2) would be Z-type, i.e., (r+d-2)%2==1.
        //   For even (d-2): r odd; for odd (d-2): r even.
        //   Half-plaquette involves data qubits (r, d-1) and (r+1, d-1).

        // Use a large sentinel value for "outside boundary" in the row/col coord.
        // We use usize::MAX/2 as sentinel to avoid overflow issues.
        const BOUNDARY: usize = usize::MAX / 2;

        // Top boundary X-type: at columns pc = 1, 3, 5, ... (odd pc)
        // Connects data qubits (0, pc) and (0, pc+1).
        // E.g., d=3: pc=1 → connects (0,1)-(0,2).
        for pc in 0..d - 1 {
            if pc % 2 == 1 {
                x_ancillas.push((BOUNDARY, 2 * pc + 1));
            }
        }

        // Bottom boundary X-type: at columns pc = 0, 2, 4, ... (even pc)
        // Connects data qubits (d-1, pc) and (d-1, pc+1).
        // E.g., d=3: pc=0 → connects (2,0)-(2,1).
        for pc in 0..d - 1 {
            if pc % 2 == 0 {
                x_ancillas.push((2 * d, 2 * pc + 1));
            }
        }

        // Left boundary Z-type: at rows pr = 0, 2, 4, ... (even pr)
        // Connects data qubits (pr, 0) and (pr+1, 0).
        for pr in 0..d - 1 {
            if pr % 2 == 0 {
                z_ancillas.push((2 * pr + 1, BOUNDARY));
            }
        }

        // Right boundary Z-type: at rows pr = 1, 3, 5, ... (odd pr)
        // Connects data qubits (pr, d-1) and (pr+1, d-1).
        for pr in 0..d - 1 {
            if pr % 2 == 1 {
                z_ancillas.push((2 * pr + 1, 2 * d));
            }
        }

        Self {
            distance: d,
            data_qubits,
            x_ancillas,
            z_ancillas,
        }
    }

    /// Number of physical data qubits = d².
    pub fn n_data_qubits(&self) -> usize {
        self.distance * self.distance
    }

    /// Code parameters: returns `(n, k, d)` = `(d², 1, d)`.
    pub fn n_k_d(&self) -> (usize, usize, usize) {
        (self.n_data_qubits(), 1, self.distance)
    }

    /// Compute the data qubit neighbors of an ancilla in 2× integer coordinates.
    ///
    /// A boundary sentinel value (`usize::MAX/2` or `2*d`) indicates an edge half-plaquette.
    fn ancilla_neighbors(&self, ar2: usize, ac2: usize) -> Vec<usize> {
        let d = self.distance;
        const BOUNDARY: usize = usize::MAX / 2;
        let mut neighbors = Vec::with_capacity(4);

        // The four adjacent data-qubit positions in 2× coords are at offsets ±1 from (ar2, ac2).
        // Data qubit at 2× coord (dr2, dc2) must satisfy dr2 % 2 == 0, dc2 % 2 == 0,
        // i.e., (ar2 ± 1, ac2 ± 1).

        let row_candidates: [Option<usize>; 2] = if ar2 == BOUNDARY {
            // Top boundary: only row 0 (2× coord = 0)
            [Some(0), None]
        } else if ar2 == 2 * d {
            // Bottom boundary: only row d-1 (2× coord = 2*(d-1))
            [Some(2 * (d - 1)), None]
        } else {
            // Interior: rows ar2-1 and ar2+1 in 2× coords → real rows (ar2-1)/2 and (ar2+1)/2
            [ar2.checked_sub(1), Some(ar2 + 1)]
        };

        let col_candidates: [Option<usize>; 2] = if ac2 == BOUNDARY {
            // Left boundary: only col 0 (2× coord = 0)
            [Some(0), None]
        } else if ac2 == 2 * d {
            // Right boundary: only col d-1 (2× coord = 2*(d-1))
            [Some(2 * (d - 1)), None]
        } else {
            [ac2.checked_sub(1), Some(ac2 + 1)]
        };

        for &maybe_row2 in &row_candidates {
            for &maybe_col2 in &col_candidates {
                if let (Some(row2), Some(col2)) = (maybe_row2, maybe_col2) {
                    // Verify these are even (data qubit positions)
                    if row2 % 2 == 0 && col2 % 2 == 0 {
                        let r = row2 / 2;
                        let c = col2 / 2;
                        if r < d && c < d {
                            let idx = r * d + c;
                            if !neighbors.contains(&idx) {
                                neighbors.push(idx);
                            }
                        }
                    }
                }
            }
        }

        neighbors.sort_unstable();
        neighbors
    }

    /// X stabilizers: each as a sorted list of data qubit indices.
    ///
    /// Weight-4 for interior plaquettes, weight-2 for boundary half-plaquettes.
    pub fn x_stabilizers(&self) -> Vec<Vec<usize>> {
        self.x_ancillas
            .iter()
            .map(|&(ar2, ac2)| self.ancilla_neighbors(ar2, ac2))
            .filter(|nbrs| !nbrs.is_empty())
            .collect()
    }

    /// Z stabilizers: each as a sorted list of data qubit indices.
    pub fn z_stabilizers(&self) -> Vec<Vec<usize>> {
        self.z_ancillas
            .iter()
            .map(|&(ar2, ac2)| self.ancilla_neighbors(ar2, ac2))
            .filter(|nbrs| !nbrs.is_empty())
            .collect()
    }

    /// Logical X operator: left column of data qubits, indices `[0, d, 2d, ...]`.
    ///
    /// In the rotated planar surface code, X-type errors are corrected by Z-stabilizers
    /// whose boundaries are on the left and right edges. A vertical chain of X along the
    /// left column commutes with all stabilizers and anticommutes with the logical Z
    /// (top row) at qubit 0.
    pub fn logical_x_qubits(&self) -> Vec<usize> {
        (0..self.distance)
            .map(|r| r * self.distance)
            .collect()
    }

    /// Logical Z operator: top row of data qubits, indices `[0, 1, ..., d-1]`.
    ///
    /// In the rotated planar surface code, Z-type errors are corrected by X-stabilizers
    /// whose boundaries are on the top and bottom edges. A horizontal chain of Z along the
    /// top row commutes with all stabilizers and anticommutes with the logical X
    /// (left column) at qubit 0.
    pub fn logical_z_qubits(&self) -> Vec<usize> {
        (0..self.distance).collect()
    }

    /// Build a `PauliString` for the logical X operator.
    pub fn logical_x_operator(&self) -> PauliString {
        let mut paulis = vec![Pauli::I; self.n_data_qubits()];
        for idx in self.logical_x_qubits() {
            paulis[idx] = Pauli::X;
        }
        PauliString::new(paulis)
    }

    /// Build a `PauliString` for the logical Z operator.
    pub fn logical_z_operator(&self) -> PauliString {
        let mut paulis = vec![Pauli::I; self.n_data_qubits()];
        for idx in self.logical_z_qubits() {
            paulis[idx] = Pauli::Z;
        }
        PauliString::new(paulis)
    }

    /// Compute the syndrome vector for a given Pauli error.
    ///
    /// Returns `[x_syndromes..., z_syndromes...]`.
    /// - `x_syndrome[i] = true` if error anticommutes with X-stabilizer i (Z or Y error on support).
    /// - `z_syndrome[i] = true` if error anticommutes with Z-stabilizer i (X or Y error on support).
    pub fn syndrome(&self, error: &PauliString) -> QuantRS2Result<Vec<bool>> {
        let x_stabs = self.x_stabilizers();
        let z_stabs = self.z_stabilizers();
        let n_x = x_stabs.len();
        let n_z = z_stabs.len();
        let mut syndrome = vec![false; n_x + n_z];

        // X stabilizer anticommutes with Z and Y errors
        for (i, stab) in x_stabs.iter().enumerate() {
            let anti = stab.iter().any(|&q| {
                q < error.paulis.len() && matches!(error.paulis[q], Pauli::Z | Pauli::Y)
            });
            syndrome[i] = anti;
        }

        // Z stabilizer anticommutes with X and Y errors
        for (i, stab) in z_stabs.iter().enumerate() {
            let anti = stab.iter().any(|&q| {
                q < error.paulis.len() && matches!(error.paulis[q], Pauli::X | Pauli::Y)
            });
            syndrome[n_x + i] = anti;
        }

        Ok(syndrome)
    }

    /// Compute (row, col) coordinates of a data qubit from its linear index.
    pub fn qubit_coords(&self, idx: usize) -> Option<(usize, usize)> {
        let d = self.distance;
        if idx < d * d {
            Some((idx / d, idx % d))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_n_k_d_d3() {
        let code = RotatedSurfaceCode::new(3);
        assert_eq!(code.n_k_d(), (9, 1, 3));
    }

    #[test]
    fn test_n_k_d_d5() {
        let code = RotatedSurfaceCode::new(5);
        assert_eq!(code.n_k_d(), (25, 1, 5));
    }

    #[test]
    fn test_n_k_d_d7() {
        let code = RotatedSurfaceCode::new(7);
        assert_eq!(code.n_k_d(), (49, 1, 7));
    }

    #[test]
    fn test_stabilizer_count_d3() {
        let code = RotatedSurfaceCode::new(3);
        let x_count = code.x_stabilizers().len();
        let z_count = code.z_stabilizers().len();
        assert_eq!(
            x_count + z_count,
            8,
            "d=3: expected 8 = d²-1 stabilizers total, got X={x_count} Z={z_count}"
        );
    }

    #[test]
    fn test_stabilizer_count_d5() {
        let code = RotatedSurfaceCode::new(5);
        let x_count = code.x_stabilizers().len();
        let z_count = code.z_stabilizers().len();
        assert_eq!(
            x_count + z_count,
            24,
            "d=5: expected 24 = d²-1 stabilizers, got X={x_count} Z={z_count}"
        );
    }

    #[test]
    fn test_stabilizer_count_even_split_d3() {
        let code = RotatedSurfaceCode::new(3);
        let x_count = code.x_stabilizers().len();
        let z_count = code.z_stabilizers().len();
        assert_eq!(x_count, 4, "d=3: expected 4 X-stabilizers");
        assert_eq!(z_count, 4, "d=3: expected 4 Z-stabilizers");
    }

    #[test]
    fn test_logical_weight_d3() {
        let code = RotatedSurfaceCode::new(3);
        assert_eq!(
            code.logical_x_qubits().len(),
            3,
            "Logical X weight should be d=3"
        );
        assert_eq!(
            code.logical_z_qubits().len(),
            3,
            "Logical Z weight should be d=3"
        );
    }

    #[test]
    fn test_logical_weight_d5() {
        let code = RotatedSurfaceCode::new(5);
        assert_eq!(code.logical_x_qubits().len(), 5);
        assert_eq!(code.logical_z_qubits().len(), 5);
    }

    #[test]
    fn test_logical_weight_d7() {
        let code = RotatedSurfaceCode::new(7);
        assert_eq!(code.logical_x_qubits().len(), 7);
        assert_eq!(code.logical_z_qubits().len(), 7);
    }

    #[test]
    fn test_logical_anticommute_d3() {
        let code = RotatedSurfaceCode::new(3);
        let lx = code.logical_x_operator();
        let lz = code.logical_z_operator();
        // Logical X (top row) and logical Z (left column) share qubit (0,0).
        // X on qubit 0 and Z on qubit 0 anticommute → overall anticommutation.
        let commutes = lx
            .commutes_with(&lz)
            .expect("commutes_with should succeed");
        assert!(
            !commutes,
            "Logical X and logical Z should anticommute for surface code"
        );
    }

    #[test]
    fn test_syndrome_no_error_d3() {
        let code = RotatedSurfaceCode::new(3);
        let identity = PauliString::identity(code.n_data_qubits());
        let syndrome = code.syndrome(&identity).expect("syndrome should succeed");
        assert!(
            syndrome.iter().all(|&s| !s),
            "Identity error should give all-zero syndrome"
        );
    }

    #[test]
    fn test_syndrome_single_x_error_center_d3() {
        let code = RotatedSurfaceCode::new(3);
        // Single X error on center qubit (1,1) = index 4
        let mut paulis = vec![Pauli::I; code.n_data_qubits()];
        paulis[4] = Pauli::X;
        let error = PauliString::new(paulis);
        let syndrome = code.syndrome(&error).expect("syndrome should succeed");
        // X error at center anticommutes with Z-stabilizers that include qubit 4.
        // The interior Z-stabs [1,2,4,5] and [3,4,6,7] both include qubit 4 → 2 defects.
        let z_start = code.x_stabilizers().len();
        let z_defect_count = syndrome[z_start..].iter().filter(|&&s| s).count();
        assert_eq!(
            z_defect_count, 2,
            "Center X error should trigger 2 Z-syndrome defects"
        );
    }

    #[test]
    fn test_syndrome_single_z_error_center_d3() {
        let code = RotatedSurfaceCode::new(3);
        let mut paulis = vec![Pauli::I; code.n_data_qubits()];
        paulis[4] = Pauli::Z;
        let error = PauliString::new(paulis);
        let syndrome = code.syndrome(&error).expect("syndrome should succeed");
        // Z error at center anticommutes with X-stabilizers that include qubit 4.
        // Interior X-stabs [0,1,3,4] and [4,5,7,8] both include qubit 4 → 2 defects.
        let x_stabs = code.x_stabilizers();
        let x_defect_count = syndrome[..x_stabs.len()].iter().filter(|&&s| s).count();
        assert_eq!(
            x_defect_count, 2,
            "Center Z error should trigger 2 X-syndrome defects"
        );
    }

    #[test]
    fn test_boundary_ancillas_weight_2_d3() {
        let code = RotatedSurfaceCode::new(3);
        let x_stabs = code.x_stabilizers();
        let z_stabs = code.z_stabilizers();
        let x_boundary_count = x_stabs.iter().filter(|s| s.len() == 2).count();
        let z_boundary_count = z_stabs.iter().filter(|s| s.len() == 2).count();
        assert_eq!(x_boundary_count, 2, "d=3 should have 2 X boundary (weight-2) stabilizers");
        assert_eq!(z_boundary_count, 2, "d=3 should have 2 Z boundary (weight-2) stabilizers");
    }

    #[test]
    fn test_all_stabilizers_cover_valid_qubits_d3() {
        let code = RotatedSurfaceCode::new(3);
        let n = code.n_data_qubits();
        for stab in code.x_stabilizers().iter().chain(code.z_stabilizers().iter()) {
            for &q in stab {
                assert!(q < n, "Stabilizer qubit index {q} out of bounds for n={n}");
            }
            assert!(stab.len() >= 2 && stab.len() <= 4, "Stabilizer weight must be 2 or 4");
        }
    }

    #[test]
    fn test_stabilizer_count_d4() {
        let code = RotatedSurfaceCode::new(4);
        let x_count = code.x_stabilizers().len();
        let z_count = code.z_stabilizers().len();
        assert_eq!(
            x_count + z_count,
            15,
            "d=4: expected 15 = d²-1 stabilizers, got X={x_count} Z={z_count}"
        );
    }

    #[test]
    fn test_qubit_coords() {
        let code = RotatedSurfaceCode::new(3);
        assert_eq!(code.qubit_coords(0), Some((0, 0)));
        assert_eq!(code.qubit_coords(4), Some((1, 1)));
        assert_eq!(code.qubit_coords(8), Some((2, 2)));
        assert_eq!(code.qubit_coords(9), None);
    }
}
