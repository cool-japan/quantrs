//! MWPM (Minimum-Weight Perfect Matching) decoder for the rotated surface code.
//!
//! ## Algorithm
//!
//! In the surface code, syndrome ancillas form the nodes of a matching graph, and
//! data qubits are the EDGES. Each data qubit connects exactly the 1 or 2 syndrome
//! ancillas that contain it in their stabilizer support:
//!
//! - If qubit q is in 2 Z-stab supports (A and B), it is the edge A↔B.
//! - If qubit q is in only 1 Z-stab support A, it is the edge A↔virtual_boundary.
//!   The boundary type (top/bottom/left/right) is determined by (row, col) of q.
//!
//! We build this qubit→edge map, run Floyd-Warshall on the ancilla graph to get
//! shortest distances, then run bitmask-DP MWPM on the defect subgraph.
//!
//! Corrections are lifted by walking parent pointers in the ancilla graph.

use super::min_matching::min_weight_perfect_matching;
use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::error_correction::pauli::{Pauli, PauliString};
use crate::error_correction::rotated_surface_code::RotatedSurfaceCode;
use crate::error_correction::SyndromeDecoder;

/// Virtual boundary node indices (past the real ancilla indices).
const VIRT_TOP: usize = 0;
const VIRT_BOTTOM: usize = 1;
const VIRT_LEFT: usize = 2;
const VIRT_RIGHT: usize = 3;

/// MWPM decoder for the rotated planar surface code.
pub struct MwpmSurfaceDecoder {
    /// The associated surface code.
    pub code: RotatedSurfaceCode,
}

impl MwpmSurfaceDecoder {
    /// Create a new MWPM decoder.
    pub fn new(code: RotatedSurfaceCode) -> Self {
        Self { code }
    }

    /// Decode one syndrome type (X or Z errors).
    ///
    /// `syndrome`: boolean vector for one type of stabilizers.
    /// `stabilizers`: the stabilizer support sets corresponding to the syndrome entries.
    /// `correction_type`: which Pauli to apply for corrections.
    fn decode_one_type(
        &self,
        syndrome: &[bool],
        stabilizers: &[Vec<usize>],
        correction_type: Pauli,
    ) -> QuantRS2Result<Vec<(usize, Pauli)>> {
        let d = self.code.distance;
        let n_stabs = stabilizers.len();

        // Find defects
        let defects: Vec<usize> = syndrome
            .iter()
            .enumerate()
            .filter(|(_, &s)| s)
            .map(|(i, _)| i)
            .collect();

        if defects.is_empty() {
            return Ok(Vec::new());
        }

        // Build the ancilla graph:
        //   Nodes: n_stabs real ancillas + 4 virtual boundary nodes
        //   Edges: one per data qubit
        //
        // Virtual node indices (offset past real ancillas):
        //   VIRT_TOP = n_stabs + 0
        //   VIRT_BOTTOM = n_stabs + 1
        //   VIRT_LEFT = n_stabs + 2
        //   VIRT_RIGHT = n_stabs + 3
        let total_nodes = n_stabs + 4;
        let v_top = n_stabs + VIRT_TOP;
        let v_bot = n_stabs + VIRT_BOTTOM;
        let v_left = n_stabs + VIRT_LEFT;
        let v_right = n_stabs + VIRT_RIGHT;

        // qubit_edge[q] = (node_a, node_b): ancilla graph edge for data qubit q.
        let n_qubits = d * d;
        let mut qubit_edge: Vec<(usize, usize)> = vec![(usize::MAX, usize::MAX); n_qubits];

        // For each data qubit, find which stabilizers contain it.
        for q in 0..n_qubits {
            let containing: Vec<usize> = stabilizers
                .iter()
                .enumerate()
                .filter(|(_, s)| s.contains(&q))
                .map(|(i, _)| i)
                .collect();

            let boundary_node = self.qubit_boundary_node(q, d, n_stabs);

            match containing.len() {
                2 => {
                    qubit_edge[q] = (containing[0], containing[1]);
                }
                1 => {
                    qubit_edge[q] = (containing[0], boundary_node);
                }
                0 => {
                    // Qubit not in any stabilizer of this type (shouldn't happen in valid code).
                    qubit_edge[q] = (boundary_node, boundary_node);
                }
                _ => {
                    // Multiple containment: take first two.
                    qubit_edge[q] = (containing[0], containing[1]);
                }
            }
        }

        // Build distance matrix via Floyd-Warshall on the ancilla graph.
        // dist[a][b] = minimum number of data qubits (edge-weight) to go from a to b.
        let mut dist = vec![vec![f64::INFINITY; total_nodes]; total_nodes];
        let mut parent = vec![vec![usize::MAX; total_nodes]; total_nodes];

        for i in 0..total_nodes {
            dist[i][i] = 0.0;
        }

        // Add each qubit as a unit-weight edge
        for q in 0..n_qubits {
            let (a, b) = qubit_edge[q];
            if a == usize::MAX || b == usize::MAX {
                continue;
            }
            if a == b {
                continue;
            }
            if dist[a][b] > 1.0 {
                dist[a][b] = 1.0;
                dist[b][a] = 1.0;
                parent[a][b] = b;
                parent[b][a] = a;
            }
        }

        // Connect virtual boundary nodes to each other with 0-weight "super-boundary" edges.
        // This allows a defect to be matched to ANY boundary (via the super-boundary).
        for &va in &[v_top, v_bot, v_left, v_right] {
            for &vb in &[v_top, v_bot, v_left, v_right] {
                if va != vb {
                    dist[va][vb] = 0.0;
                    dist[vb][va] = 0.0;
                    parent[va][vb] = vb;
                    parent[vb][va] = va;
                }
            }
        }

        // Floyd-Warshall
        for k in 0..total_nodes {
            for i in 0..total_nodes {
                for j in 0..total_nodes {
                    let through_k = dist[i][k] + dist[k][j];
                    if through_k < dist[i][j] {
                        dist[i][j] = through_k;
                        parent[i][j] = parent[i][k];
                    }
                }
            }
        }

        let n_defects = defects.len();

        // Add a single super-boundary node for MWPM to match odd-defect syndromes.
        // Map: MWPM node 0..n_defects → real defect stab, node n_defects → virtual boundary.
        let total_mwpm = if n_defects % 2 != 0 {
            n_defects + 1
        } else {
            n_defects
        };

        if total_mwpm > 24 {
            return Err(QuantRS2Error::InvalidInput(
                "Too many defects for bitmask-DP decoder: use d ≤ 7".to_string(),
            ));
        }

        // Build MWPM edge list (node i = real defect index into `defects`)
        let mut mwpm_edges: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..n_defects {
            // Edges between real defects
            for j in i + 1..n_defects {
                let a = defects[i]; // ancilla index
                let b = defects[j];
                let d_ab = dist[a][b];
                if d_ab.is_finite() {
                    mwpm_edges.push((i, j, d_ab));
                }
            }
            // Edge to virtual super-boundary node (if odd count)
            if total_mwpm > n_defects {
                let a = defects[i];
                // Distance to nearest virtual boundary
                let d_boundary = [v_top, v_bot, v_left, v_right]
                    .iter()
                    .map(|&vb| dist[a][vb])
                    .fold(f64::INFINITY, f64::min);
                if d_boundary.is_finite() {
                    mwpm_edges.push((i, n_defects, d_boundary));
                } else {
                    // Ensure connectivity even if no direct path (shouldn't happen)
                    mwpm_edges.push((i, n_defects, (d as f64) * 2.0));
                }
            }
        }

        // Run MWPM
        let matching = min_weight_perfect_matching(total_mwpm, &mwpm_edges)
            .map_err(|e| QuantRS2Error::InvalidInput(format!("MWPM failed: {e}")))?
            .ok_or_else(|| QuantRS2Error::InvalidInput("No perfect matching found".to_string()))?;

        // Lift matching to data qubit corrections
        let mut corrections: Vec<(usize, Pauli)> = Vec::new();
        let virt_mwpm = n_defects; // index of virtual super-boundary in MWPM

        for (u, v) in &matching {
            let (real_u, real_v_opt) = if *u == virt_mwpm {
                (*v, None)
            } else if *v == virt_mwpm {
                (*u, None)
            } else {
                (*u, Some(*v))
            };

            let a = defects[real_u];

            if let Some(real_v) = real_v_opt {
                // Pair between two real defects
                let b = defects[real_v];
                let path_qubits =
                    self.path_between_nodes(a, b, &parent, &qubit_edge, n_qubits, total_nodes);
                for q in path_qubits {
                    corrections.push((q, correction_type));
                }
            } else {
                // Pair with virtual boundary: find nearest virtual boundary node
                let nearest_virt = [v_top, v_bot, v_left, v_right]
                    .iter()
                    .copied()
                    .min_by(|&va, &vb| {
                        dist[a][va]
                            .partial_cmp(&dist[a][vb])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(v_top);

                let path_qubits = self.path_between_nodes(
                    a,
                    nearest_virt,
                    &parent,
                    &qubit_edge,
                    n_qubits,
                    total_nodes,
                );
                for q in path_qubits {
                    corrections.push((q, correction_type));
                }
            }
        }

        Ok(corrections)
    }

    /// Classify which virtual boundary node qubit `q` connects to when it's a boundary qubit.
    fn qubit_boundary_node(&self, q: usize, d: usize, n_stabs: usize) -> usize {
        let r = q / d;
        let c = q % d;
        // Boundary classification: use row/col position.
        // Corner qubits go to X-type boundaries (top or bottom).
        if r == 0 {
            n_stabs + VIRT_TOP
        } else if r == d - 1 {
            n_stabs + VIRT_BOTTOM
        } else if c == 0 {
            n_stabs + VIRT_LEFT
        } else if c == d - 1 {
            n_stabs + VIRT_RIGHT
        } else {
            // Interior qubit in only 1 stabilizer — shouldn't happen but default to top
            n_stabs + VIRT_TOP
        }
    }

    /// Recover the set of data qubits on the shortest path from ancilla node `a` to node `b`.
    ///
    /// Uses the parent matrix from Floyd-Warshall to trace the path, then maps each
    /// consecutive node pair (u, v) back to the data qubit on that edge.
    fn path_between_nodes(
        &self,
        a: usize,
        b: usize,
        parent: &[Vec<usize>],
        qubit_edge: &[(usize, usize)],
        n_qubits: usize,
        _total_nodes: usize,
    ) -> Vec<usize> {
        if a == b {
            return Vec::new();
        }

        // Trace path using parent pointers
        let mut path_nodes = Vec::new();
        let mut cur = a;
        let mut safety = 0;
        loop {
            path_nodes.push(cur);
            if cur == b {
                break;
            }
            let next = parent[cur][b];
            if next == usize::MAX || next == cur {
                break; // No path found
            }
            cur = next;
            safety += 1;
            if safety > 100 {
                break; // Prevent infinite loop
            }
        }

        // Map consecutive node pairs to data qubits
        let mut qubits = Vec::new();
        for i in 0..path_nodes.len().saturating_sub(1) {
            let u = path_nodes[i];
            let v = path_nodes[i + 1];
            // Find data qubit q such that qubit_edge[q] == (u, v) or (v, u)
            for q in 0..n_qubits {
                let (ea, eb) = qubit_edge[q];
                if (ea == u && eb == v) || (ea == v && eb == u) {
                    qubits.push(q);
                    break;
                }
            }
        }

        qubits
    }
}

impl SyndromeDecoder for MwpmSurfaceDecoder {
    /// Decode a syndrome produced by `RotatedSurfaceCode::syndrome()`.
    ///
    /// The syndrome layout must be `[x_syndrome..., z_syndrome...]` as returned by
    /// `RotatedSurfaceCode::syndrome()`.
    fn decode(&self, syndrome: &[bool]) -> QuantRS2Result<PauliString> {
        let n = self.code.n_data_qubits();
        let x_stabs = self.code.x_stabilizers();
        let z_stabs = self.code.z_stabilizers();
        let n_x = x_stabs.len();
        let n_z = z_stabs.len();

        if syndrome.len() != n_x + n_z {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Syndrome length {} does not match expected {} (n_x={n_x}, n_z={n_z})",
                syndrome.len(),
                n_x + n_z
            )));
        }

        let x_syndrome = &syndrome[..n_x]; // X-stab defects → Z errors
        let z_syndrome = &syndrome[n_x..]; // Z-stab defects → X errors

        let mut error_paulis = vec![Pauli::I; n];

        // Decode X errors (Z-stabilizer defects)
        let x_corrections = self.decode_one_type(z_syndrome, &z_stabs, Pauli::X)?;
        for (qubit, pauli) in x_corrections {
            if qubit < n {
                combine_pauli(&mut error_paulis[qubit], pauli);
            }
        }

        // Decode Z errors (X-stabilizer defects)
        let z_corrections = self.decode_one_type(x_syndrome, &x_stabs, Pauli::Z)?;
        for (qubit, pauli) in z_corrections {
            if qubit < n {
                combine_pauli(&mut error_paulis[qubit], pauli);
            }
        }

        Ok(PauliString::new(error_paulis))
    }
}

/// Combine two Pauli operators on the same qubit.
fn combine_pauli(existing: &mut Pauli, new_pauli: Pauli) {
    *existing = match (*existing, new_pauli) {
        (Pauli::I, p) | (p, Pauli::I) => p,
        (Pauli::X, Pauli::X) | (Pauli::Z, Pauli::Z) | (Pauli::Y, Pauli::Y) => Pauli::I,
        (Pauli::X, Pauli::Z) | (Pauli::Z, Pauli::X) => Pauli::Y,
        (Pauli::X, Pauli::Y) | (Pauli::Y, Pauli::X) => Pauli::Z,
        (Pauli::Z, Pauli::Y) | (Pauli::Y, Pauli::Z) => Pauli::X,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error_correction::rotated_surface_code::RotatedSurfaceCode;

    fn is_logical_error(composed: &PauliString, code: &RotatedSurfaceCode) -> bool {
        let lx = code.logical_x_operator();
        let lz = code.logical_z_operator();
        let anticommutes_lx = composed.commutes_with(&lz).is_ok_and(|c| !c);
        let anticommutes_lz = composed.commutes_with(&lx).is_ok_and(|c| !c);
        anticommutes_lx || anticommutes_lz
    }

    fn make_decoder(d: usize) -> MwpmSurfaceDecoder {
        MwpmSurfaceDecoder::new(RotatedSurfaceCode::new(d))
    }

    #[test]
    fn test_mwpm_no_errors_d3() {
        let decoder = make_decoder(3);
        let n_x = decoder.code.x_stabilizers().len();
        let n_z = decoder.code.z_stabilizers().len();
        let syndrome = vec![false; n_x + n_z];
        let correction = decoder.decode(&syndrome).expect("Decoding should succeed");
        assert_eq!(
            correction.weight(),
            0,
            "No-error syndrome should yield identity correction"
        );
    }

    #[test]
    fn test_mwpm_single_x_error_all_qubits_d3() {
        let code = RotatedSurfaceCode::new(3);
        let decoder = MwpmSurfaceDecoder::new(code.clone());
        let n = code.n_data_qubits();

        for qubit in 0..n {
            let mut paulis = vec![Pauli::I; n];
            paulis[qubit] = Pauli::X;
            let error = PauliString::new(paulis);
            let syndrome = code.syndrome(&error).expect("syndrome ok");
            let correction = decoder.decode(&syndrome).expect("decode ok");
            let composed = error.multiply(&correction).expect("multiply ok");
            assert!(
                !is_logical_error(&composed, &code),
                "MWPM: X error on qubit {qubit} should not cause logical error"
            );
        }
    }

    #[test]
    fn test_mwpm_single_z_error_all_qubits_d3() {
        let code = RotatedSurfaceCode::new(3);
        let decoder = MwpmSurfaceDecoder::new(code.clone());
        let n = code.n_data_qubits();

        for qubit in 0..n {
            let mut paulis = vec![Pauli::I; n];
            paulis[qubit] = Pauli::Z;
            let error = PauliString::new(paulis);
            let syndrome = code.syndrome(&error).expect("syndrome ok");
            let correction = decoder.decode(&syndrome).expect("decode ok");
            let composed = error.multiply(&correction).expect("multiply ok");
            assert!(
                !is_logical_error(&composed, &code),
                "MWPM: Z error on qubit {qubit} should not cause logical error"
            );
        }
    }

    #[test]
    fn test_mwpm_no_errors_d5() {
        let decoder = make_decoder(5);
        let n_x = decoder.code.x_stabilizers().len();
        let n_z = decoder.code.z_stabilizers().len();
        let syndrome = vec![false; n_x + n_z];
        let correction = decoder.decode(&syndrome).expect("decode ok");
        assert_eq!(correction.weight(), 0);
    }

    #[test]
    fn test_mwpm_wrong_syndrome_length() {
        let decoder = make_decoder(3);
        let result = decoder.decode(&[true, false]);
        assert!(result.is_err(), "Wrong syndrome length should return Err");
    }
}
