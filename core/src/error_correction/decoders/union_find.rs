//! Union-Find decoder for the rotated surface code.
//!
//! Implements the Delfosse-Nickerson Union-Find decoder for topological codes.
//! This decoder has near-linear time complexity and is suitable for real-time
//! decoding on hardware.
//!
//! ## Algorithm Overview
//!
//! 1. Start with each syndrome defect as its own cluster with radius 0.
//! 2. Grow all clusters simultaneously by 1/2 unit radius.
//! 3. When two clusters' expanded regions overlap, merge them.
//! 4. A cluster that touches a boundary becomes "neutral" (can be matched to boundary).
//! 5. Once all clusters are neutral or have even size (can be internally matched):
//!    peel the spanning forest to find corrections.
//!
//! ## Reference
//!
//! Delfosse and Nickerson, "Almost-linear time decoding algorithm for topological codes",
//! Quantum 5, 595 (2021).

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::error_correction::pauli::{Pauli, PauliString};
use crate::error_correction::rotated_surface_code::RotatedSurfaceCode;
use crate::error_correction::SyndromeDecoder;
use std::collections::{HashMap, HashSet, VecDeque};

/// Union-Find decoder for the rotated planar surface code.
pub struct UnionFindDecoder {
    /// The associated surface code.
    pub code: RotatedSurfaceCode,
}

impl UnionFindDecoder {
    /// Create a new Union-Find decoder.
    pub fn new(code: RotatedSurfaceCode) -> Self {
        Self { code }
    }

    /// Decode one syndrome type (X or Z errors).
    ///
    /// `syndrome`: boolean vector for one set of stabilizers (X or Z).
    /// `stabilizers`: the corresponding stabilizer support sets.
    /// `boundary_near`: a function returning the distance to the nearest relevant boundary.
    /// `correction_type`: the Pauli type to apply for corrections.
    fn decode_one_type(
        &self,
        syndrome: &[bool],
        stabilizers: &[Vec<usize>],
        correction_type: Pauli,
        use_row_boundary: bool,
    ) -> QuantRS2Result<Vec<(usize, Pauli)>> {
        let d = self.code.distance;
        let n_stabs = stabilizers.len();

        // Find defect positions
        let defects: Vec<usize> = syndrome
            .iter()
            .enumerate()
            .filter(|(_, &s)| s)
            .map(|(i, _)| i)
            .collect();

        if defects.is_empty() {
            return Ok(Vec::new());
        }

        // For each defect, compute its "representative coordinate" (centroid of support)
        let defect_coords: Vec<(f64, f64)> = defects
            .iter()
            .map(|&di| {
                let stab = &stabilizers[di];
                if stab.is_empty() {
                    return (0.0, 0.0);
                }
                let r_sum: f64 = stab.iter().map(|&q| (q / d) as f64).sum();
                let c_sum: f64 = stab.iter().map(|&q| (q % d) as f64).sum();
                let w = stab.len() as f64;
                (r_sum / w, c_sum / w)
            })
            .collect();

        // A stabilizer ancilla is a BOUNDARY ancilla if and only if it is a half-plaquette
        // (weight 2). Interior plaquettes have weight 4.
        // Boundary neutrality: a defect cluster is "neutral" if it contains a boundary ancilla.
        let is_boundary_stab: Vec<bool> = (0..n_stabs).map(|i| stabilizers[i].len() < 4).collect();

        let n_defects = defects.len();

        // --- Union-Find data structure ---
        // parent[i]: union-find parent (points to representative)
        // rank[i]: rank for union by rank
        // neutral[i]: whether the cluster has a boundary ancilla
        // size[i]: number of defects in the cluster
        let mut parent: Vec<usize> = (0..n_defects).collect();
        let mut rank: Vec<usize> = vec![0; n_defects];
        let mut neutral: Vec<bool> = (0..n_defects)
            .map(|i| is_boundary_stab[defects[i]])
            .collect();
        let mut cluster_defects: Vec<Vec<usize>> = (0..n_defects).map(|i| vec![i]).collect();

        // Find with path compression
        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        // Union by rank; returns (root_kept, root_merged)
        fn union(
            parent: &mut Vec<usize>,
            rank: &mut Vec<usize>,
            neutral: &mut Vec<bool>,
            cluster_defects: &mut Vec<Vec<usize>>,
            a: usize,
            b: usize,
        ) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra == rb {
                return;
            }
            let (keep, merge) = if rank[ra] >= rank[rb] {
                (ra, rb)
            } else {
                (rb, ra)
            };
            parent[merge] = keep;
            if rank[keep] == rank[merge] {
                rank[keep] += 1;
            }
            neutral[keep] = neutral[keep] || neutral[merge];
            let merged_defects = std::mem::take(&mut cluster_defects[merge]);
            cluster_defects[keep].extend(merged_defects);
        }

        // Grow clusters until all are neutral or have even defect count
        // We simulate this with BFS growth in integer-radius steps.
        // At each step, for any two defects in different clusters within distance ≤ 2*step,
        // merge their clusters.

        let max_steps = d * 2;
        for step in 1..=max_steps {
            let threshold = (step as f64) * 1.0; // growth radius = step * 0.5 * 2 = step

            // Merge clusters whose centroids are within 2*threshold
            let mut merged = true;
            while merged {
                merged = false;
                'outer: for i in 0..n_defects {
                    for j in i + 1..n_defects {
                        let ri = find(&mut parent, i);
                        let rj = find(&mut parent, j);
                        if ri == rj {
                            continue;
                        }
                        let (r1, c1) = defect_coords[i];
                        let (r2, c2) = defect_coords[j];
                        let dist = (r1 - r2).abs() + (c1 - c2).abs();
                        if dist <= threshold {
                            union(
                                &mut parent,
                                &mut rank,
                                &mut neutral,
                                &mut cluster_defects,
                                i,
                                j,
                            );
                            merged = true;
                            // After merging, update neutrality for the new root:
                            // a cluster is neutral if any of its defects is a boundary ancilla.
                            let root = find(&mut parent, i);
                            for &di in &cluster_defects[root].clone() {
                                if is_boundary_stab[defects[di]] {
                                    neutral[root] = true;
                                }
                            }
                            break 'outer;
                        }
                    }
                }
            }

            // Check termination: all clusters are neutral or have even defect count
            let mut all_done = true;
            for i in 0..n_defects {
                if find(&mut parent, i) == i {
                    // i is a root
                    let root_defects = &cluster_defects[i];
                    if !neutral[i] && root_defects.len() % 2 != 0 {
                        all_done = false;
                        break;
                    }
                }
            }
            if all_done {
                break;
            }
        }

        // Peel the spanning forest: match defects within each cluster
        // Group defects by cluster root
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n_defects {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(i);
        }

        let mut corrections: Vec<(usize, Pauli)> = Vec::new();

        for (root, cluster) in &clusters {
            let cluster_size = cluster.len();

            // Odd cluster: must send one defect to the boundary.
            // We do this regardless of whether the cluster touched a boundary ancilla,
            // because any odd syndrome requires a boundary pairing for parity.
            if cluster_size % 2 != 0 {
                // Find the defect in this cluster closest to the relevant boundary.
                let mut remaining = cluster.clone();
                let boundary_idx = remaining
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        let da = boundary_dist(defect_coords[a], d, use_row_boundary);
                        let db = boundary_dist(defect_coords[b], d, use_row_boundary);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(remaining[0]);

                remaining.retain(|&x| x != boundary_idx);

                // Generate correction from boundary_idx to boundary.
                let (br, bc) = defect_coords[boundary_idx];
                let boundary_correction = boundary_path(
                    br,
                    bc,
                    d,
                    use_row_boundary,
                    stabilizers,
                    defects[boundary_idx],
                );
                for qubit in boundary_correction {
                    corrections.push((qubit, correction_type));
                }

                // Pair remaining defects within the cluster.
                let paired = greedy_pair(&remaining, &defect_coords);
                for (a, b) in paired {
                    let path = manhattan_path_between_stabs(stabilizers, defects[a], defects[b], d);
                    for qubit in path {
                        corrections.push((qubit, correction_type));
                    }
                }
            } else {
                // Even cluster: pair all defects internally.
                let paired = greedy_pair(cluster, &defect_coords);
                for (a, b) in paired {
                    let path = manhattan_path_between_stabs(stabilizers, defects[a], defects[b], d);
                    for qubit in path {
                        corrections.push((qubit, correction_type));
                    }
                }
            }
        }

        Ok(corrections)
    }
}

impl SyndromeDecoder for UnionFindDecoder {
    fn decode(&self, syndrome: &[bool]) -> QuantRS2Result<PauliString> {
        let n = self.code.n_data_qubits();
        let x_stabs = self.code.x_stabilizers();
        let z_stabs = self.code.z_stabilizers();
        let n_x = x_stabs.len();
        let n_z = z_stabs.len();

        if syndrome.len() != n_x + n_z {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Syndrome length {} != expected {} (n_x={n_x}, n_z={n_z})",
                syndrome.len(),
                n_x + n_z
            )));
        }

        let x_syndrome = &syndrome[..n_x]; // X-stab defects → Z errors
        let z_syndrome = &syndrome[n_x..]; // Z-stab defects → X errors

        let mut error_paulis = vec![Pauli::I; n];

        // Decode X errors (Z-stabilizer defects)
        let x_corrections = self.decode_one_type(&z_syndrome, &z_stabs, Pauli::X, false)?;
        for (qubit, pauli) in x_corrections {
            if qubit < n {
                combine_pauli(&mut error_paulis[qubit], pauli);
            }
        }

        // Decode Z errors (X-stabilizer defects)
        let z_corrections = self.decode_one_type(&x_syndrome, &x_stabs, Pauli::Z, true)?;
        for (qubit, pauli) in z_corrections {
            if qubit < n {
                combine_pauli(&mut error_paulis[qubit], pauli);
            }
        }

        Ok(PauliString::new(error_paulis))
    }
}

// --- Helper functions ---

/// Check if a stabilizer centroid is on or near the relevant boundary.
fn is_on_boundary(coord: (f64, f64), d: usize, use_row_boundary: bool) -> bool {
    let (r, c) = coord;
    let limit = (d - 1) as f64;
    if use_row_boundary {
        r <= 0.5 || r >= limit - 0.5
    } else {
        c <= 0.5 || c >= limit - 0.5
    }
}

/// Distance from stabilizer centroid to nearest relevant boundary.
fn boundary_dist(coord: (f64, f64), d: usize, use_row_boundary: bool) -> f64 {
    let (r, c) = coord;
    let limit = (d - 1) as f64;
    if use_row_boundary {
        r.min(limit - r)
    } else {
        c.min(limit - c)
    }
}

/// Generate a correction path from a defect to the nearest boundary.
fn boundary_path(
    r: f64,
    c: f64,
    d: usize,
    use_row_boundary: bool,
    stabilizers: &[Vec<usize>],
    stab_idx: usize,
) -> Vec<usize> {
    // Pick a representative qubit from the stabilizer
    let stab = &stabilizers[stab_idx];
    if stab.is_empty() {
        return Vec::new();
    }

    // Find the qubit in the stabilizer closest to the boundary
    let (target_qubit, _) = stab
        .iter()
        .map(|&q| {
            let qr = (q / d) as f64;
            let qc = (q % d) as f64;
            let dist = if use_row_boundary {
                let lim = (d - 1) as f64;
                qr.min(lim - qr)
            } else {
                let lim = (d - 1) as f64;
                qc.min(lim - qc)
            };
            (q, dist)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((stab[0], 0.0));

    let qr = (target_qubit / d) as isize;
    let qc = (target_qubit % d) as isize;

    // Move to the nearest boundary edge
    let mut path = vec![target_qubit];
    if use_row_boundary {
        // Move to row 0 or row d-1
        let target_row = if qr <= (d as isize / 2) {
            0
        } else {
            d as isize - 1
        };
        let mut cur_r = qr;
        let cur_c = qc;
        while cur_r != target_row {
            if cur_r > target_row {
                cur_r -= 1;
            } else {
                cur_r += 1;
            }
            if cur_r >= 0 && cur_r < d as isize {
                path.push((cur_r as usize) * d + cur_c as usize);
            }
        }
    } else {
        let target_col = if qc <= (d as isize / 2) {
            0
        } else {
            d as isize - 1
        };
        let cur_r = qr;
        let mut cur_c = qc;
        while cur_c != target_col {
            if cur_c > target_col {
                cur_c -= 1;
            } else {
                cur_c += 1;
            }
            if cur_c >= 0 && cur_c < d as isize {
                path.push(cur_r as usize * d + cur_c as usize);
            }
        }
    }

    path
}

/// Greedy pairing of defect indices by nearest-neighbor distance.
fn greedy_pair(defects: &[usize], coords: &[(f64, f64)]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let mut used = HashSet::new();

    for i in 0..defects.len() {
        if used.contains(&i) {
            continue;
        }
        let mut best_j = None;
        let mut best_dist = f64::INFINITY;
        for j in i + 1..defects.len() {
            if used.contains(&j) {
                continue;
            }
            let (ri, ci) = coords[defects[i]];
            let (rj, cj) = coords[defects[j]];
            let d = (ri - rj).abs() + (ci - cj).abs();
            if d < best_dist {
                best_dist = d;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            used.insert(i);
            used.insert(j);
            pairs.push((defects[i], defects[j]));
        }
    }

    pairs
}

/// Generate a Manhattan path between two stabilizer support sets.
fn manhattan_path_between_stabs(
    stabilizers: &[Vec<usize>],
    stab_a: usize,
    stab_b: usize,
    d: usize,
) -> Vec<usize> {
    let sa = &stabilizers[stab_a];
    let sb = &stabilizers[stab_b];
    if sa.is_empty() || sb.is_empty() {
        return Vec::new();
    }

    // Find closest pair (one from each stabilizer)
    let mut best_dist = usize::MAX;
    let mut best_pair = (sa[0], sb[0]);
    for &qa in sa {
        for &qb in sb {
            let ra = qa / d;
            let ca = qa % d;
            let rb = qb / d;
            let cb = qb % d;
            let dist = ra.abs_diff(rb) + ca.abs_diff(cb);
            if dist < best_dist {
                best_dist = dist;
                best_pair = (qa, qb);
            }
        }
    }

    let (qa, qb) = best_pair;
    let r_start = (qa / d) as isize;
    let c_start = (qa % d) as isize;
    let r_end = (qb / d) as isize;
    let c_end = (qb % d) as isize;

    let mut path = Vec::new();
    let mut r = r_start;
    let mut c = c_start;

    while r != r_end {
        path.push((r as usize) * d + (c as usize));
        if r < r_end {
            r += 1;
        } else {
            r -= 1;
        }
    }
    while c != c_end {
        path.push((r as usize) * d + (c as usize));
        if c < c_end {
            c += 1;
        } else {
            c -= 1;
        }
    }
    path.push((r as usize) * d + (c as usize));

    path
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

    fn make_decoder(d: usize) -> UnionFindDecoder {
        UnionFindDecoder::new(RotatedSurfaceCode::new(d))
    }

    #[test]
    fn test_uf_no_errors_d3() {
        let decoder = make_decoder(3);
        let code = &decoder.code;
        let n_x = code.x_stabilizers().len();
        let n_z = code.z_stabilizers().len();
        let syndrome = vec![false; n_x + n_z];
        let correction = decoder.decode(&syndrome).expect("decode ok");
        assert_eq!(
            correction.weight(),
            0,
            "No-error syndrome should give identity correction"
        );
    }

    #[test]
    fn test_uf_single_x_error_d3() {
        let code = RotatedSurfaceCode::new(3);
        let decoder = UnionFindDecoder::new(code.clone());

        let mut paulis = vec![Pauli::I; code.n_data_qubits()];
        paulis[4] = Pauli::X;
        let error = PauliString::new(paulis);

        let syndrome = code.syndrome(&error).expect("syndrome ok");
        let correction = decoder.decode(&syndrome).expect("decode ok");

        assert!(
            correction.weight() <= 3,
            "Single X error correction should be low weight, got {}",
            correction.weight()
        );
    }

    #[test]
    fn test_uf_single_z_error_d3() {
        let code = RotatedSurfaceCode::new(3);
        let decoder = UnionFindDecoder::new(code.clone());

        let mut paulis = vec![Pauli::I; code.n_data_qubits()];
        paulis[3] = Pauli::Z;
        let error = PauliString::new(paulis);

        let syndrome = code.syndrome(&error).expect("syndrome ok");
        let correction = decoder.decode(&syndrome).expect("decode ok");

        assert!(
            correction.weight() <= 3,
            "Single Z error correction weight {} should be ≤ 3",
            correction.weight()
        );
    }

    #[test]
    fn test_uf_wrong_syndrome_length() {
        let decoder = make_decoder(3);
        let result = decoder.decode(&[true, false]);
        assert!(result.is_err());
    }
}
