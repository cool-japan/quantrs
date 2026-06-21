//! KAK decomposition for multi-qubit unitaries
//!
//! This module extends the Cartan (KAK) decomposition to handle arbitrary
//! n-qubit unitaries through recursive application and generalized
//! decomposition techniques.

use crate::{
    cartan::{CartanDecomposer, CartanDecomposition},
    error::{QuantRS2Error, QuantRS2Result},
    gate::{multi::*, single::*, GateOp},
    matrix_ops::{DenseMatrix, QuantumMatrix},
    qubit::QubitId,
    shannon::ShannonDecomposer,
    synthesis::{decompose_single_qubit_zyz, SingleQubitDecomposition},
};
use rustc_hash::FxHashMap;
use scirs2_core::ndarray::{s, Array2};
use scirs2_core::Complex;

/// Complex one-sided Jacobi SVD of a square matrix `m`.
///
/// Returns `(U, s, Vᴴ)` with `m = U · diag(s) · Vᴴ`, `U` and `Vᴴ` unitary and `s` the
/// singular values in non-increasing order. One-sided Jacobi orthogonalises the columns
/// of `m` by unitary plane rotations; the resulting column norms are the singular values
/// and the accumulated rotations form `V`. This is used because the SciRS2 LAPACK SVD
/// currently exposes only a real-valued path, whereas the CSD operates on complex blocks.
fn complex_svd(
    m: &Array2<Complex<f64>>,
) -> QuantRS2Result<(Array2<Complex<f64>>, Vec<f64>, Array2<Complex<f64>>)> {
    let (rows, cols) = (m.nrows(), m.ncols());
    if rows != cols {
        return Err(QuantRS2Error::InvalidInput(
            "complex_svd helper expects a square matrix".to_string(),
        ));
    }
    let n = rows;

    let mut a = m.clone(); // columns orthogonalised in place -> U·diag(s)
    let mut v = Array2::<Complex<f64>>::eye(n); // accumulates right rotations

    let eps = 1e-15;
    let max_sweeps = 60;
    for _sweep in 0..max_sweeps {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut alpha = 0.0f64;
                let mut beta = 0.0f64;
                let mut gamma = Complex::new(0.0, 0.0);
                for r in 0..n {
                    let ai = a[[r, i]];
                    let aj = a[[r, j]];
                    alpha += ai.norm_sqr();
                    beta += aj.norm_sqr();
                    gamma += ai.conj() * aj;
                }
                let gamma_abs = gamma.norm();
                off += gamma_abs;
                if gamma_abs <= eps * (alpha.sqrt() * beta.sqrt()).max(eps) {
                    continue;
                }
                let phase = gamma / gamma_abs;
                let zeta = (beta - alpha) / (2.0 * gamma_abs);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let cval = 1.0 / (1.0 + t * t).sqrt();
                let s_ij = phase * Complex::new(cval * t, 0.0);
                for r in 0..n {
                    let ai = a[[r, i]];
                    let aj = a[[r, j]];
                    a[[r, i]] = Complex::new(cval, 0.0) * ai - s_ij.conj() * aj;
                    a[[r, j]] = s_ij * ai + Complex::new(cval, 0.0) * aj;
                }
                for r in 0..n {
                    let vi = v[[r, i]];
                    let vj = v[[r, j]];
                    v[[r, i]] = Complex::new(cval, 0.0) * vi - s_ij.conj() * vj;
                    v[[r, j]] = s_ij * vi + Complex::new(cval, 0.0) * vj;
                }
            }
        }
        if off <= eps {
            break;
        }
    }

    // Column norms are the singular values; sort non-increasing.
    let mut order: Vec<(f64, usize)> = (0..n)
        .map(|j| {
            let norm = (0..n).map(|r| a[[r, j]].norm_sqr()).sum::<f64>().sqrt();
            (norm, j)
        })
        .collect();
    order.sort_by(|x, y| y.0.total_cmp(&x.0));

    let mut u = Array2::<Complex<f64>>::zeros((n, n));
    let mut s_vec = vec![0.0f64; n];
    let mut v_sorted = Array2::<Complex<f64>>::zeros((n, n));
    let mut zero_cols: Vec<usize> = Vec::new();
    for (new_idx, &(norm, old_idx)) in order.iter().enumerate() {
        s_vec[new_idx] = norm;
        if norm > 1e-300 {
            for r in 0..n {
                u[[r, new_idx]] = a[[r, old_idx]] / Complex::new(norm, 0.0);
            }
        } else {
            zero_cols.push(new_idx);
        }
        for r in 0..n {
            v_sorted[[r, new_idx]] = v[[r, old_idx]];
        }
    }
    // Fill any zero singular-value columns of U with an orthonormal completion so U
    // stays unitary.
    if !zero_cols.is_empty() {
        complete_orthonormal_columns(&mut u, &zero_cols, 1e-12);
    }

    let vh = v_sorted.mapv(|z| z.conj()).t().to_owned();
    Ok((u, s_vec, vh))
}

/// Fill the specified columns of `mat` (an `n×n` matrix whose *other* columns are
/// already orthonormal) with vectors that extend the set to a full orthonormal basis.
///
/// Uses modified Gram–Schmidt against the existing columns and the candidates built so
/// far, seeding from standard basis vectors.
fn complete_orthonormal_columns(mat: &mut Array2<Complex<f64>>, columns: &[usize], tol: f64) {
    let n = mat.nrows();
    let fixed: std::collections::HashSet<usize> = columns.iter().copied().collect();

    // Collect the already-fixed (good) columns as the starting orthonormal set.
    let mut basis: Vec<Vec<Complex<f64>>> = Vec::new();
    for j in 0..n {
        if !fixed.contains(&j) {
            basis.push((0..n).map(|r| mat[[r, j]]).collect());
        }
    }

    let mut seed = 0usize;
    for &col in columns {
        // Find a standard basis vector e_seed that is not (numerically) in the span,
        // orthogonalise it against the current basis, and normalise.
        let mut placed = false;
        while seed < n && !placed {
            let mut v = vec![Complex::new(0.0, 0.0); n];
            v[seed] = Complex::new(1.0, 0.0);
            for b in &basis {
                let proj: Complex<f64> =
                    b.iter().zip(v.iter()).map(|(bk, vk)| bk.conj() * vk).sum();
                for r in 0..n {
                    v[r] -= proj * b[r];
                }
            }
            let norm = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            if norm > tol {
                for r in 0..n {
                    v[r] /= Complex::new(norm, 0.0);
                }
                for r in 0..n {
                    mat[[r, col]] = v[r];
                }
                basis.push(v);
                placed = true;
            }
            seed += 1;
        }
        if !placed {
            // Fallback: leave a unit vector (should not happen for a valid completion).
            mat[[col % n, col]] = Complex::new(1.0, 0.0);
        }
    }
}

/// Result of multi-qubit KAK decomposition
#[derive(Debug, Clone)]
pub struct MultiQubitKAK {
    /// The decomposed gate sequence
    pub gates: Vec<Box<dyn GateOp>>,
    /// Decomposition tree structure
    pub tree: DecompositionTree,
    /// Total CNOT count
    pub cnot_count: usize,
    /// Total single-qubit gate count
    pub single_qubit_count: usize,
    /// Circuit depth
    pub depth: usize,
}

/// Tree structure representing the hierarchical decomposition
#[derive(Debug, Clone)]
pub enum DecompositionTree {
    /// Leaf node - single or two-qubit gate
    Leaf {
        qubits: Vec<QubitId>,
        gate_type: LeafType,
    },
    /// Internal node - recursive decomposition
    Node {
        qubits: Vec<QubitId>,
        method: DecompositionMethod,
        children: Vec<Self>,
    },
}

/// Type of leaf decomposition
#[derive(Debug, Clone)]
pub enum LeafType {
    SingleQubit(SingleQubitDecomposition),
    TwoQubit(CartanDecomposition),
}

/// Method used for decomposition at this level
#[derive(Debug, Clone)]
pub enum DecompositionMethod {
    /// Cosine-Sine Decomposition
    CSD { pivot: usize },
    /// Quantum Shannon Decomposition
    Shannon { partition: usize },
    /// Block diagonalization
    BlockDiagonal { block_size: usize },
    /// Direct Cartan for 2 qubits
    Cartan,
}

/// Multi-qubit KAK decomposer
pub struct MultiQubitKAKDecomposer {
    /// Tolerance for numerical comparisons
    tolerance: f64,
    /// Maximum recursion depth
    max_depth: usize,
    /// Cache for decompositions
    #[allow(dead_code)]
    cache: FxHashMap<u64, MultiQubitKAK>,
    /// Use optimized methods
    use_optimization: bool,
    /// Cartan decomposer for two-qubit blocks
    cartan: CartanDecomposer,
}

impl MultiQubitKAKDecomposer {
    /// Create a new multi-qubit KAK decomposer
    pub fn new() -> Self {
        Self {
            tolerance: 1e-10,
            max_depth: 20,
            cache: FxHashMap::default(),
            use_optimization: true,
            cartan: CartanDecomposer::new(),
        }
    }

    /// Create with custom tolerance
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            tolerance,
            max_depth: 20,
            cache: FxHashMap::default(),
            use_optimization: true,
            cartan: CartanDecomposer::with_tolerance(tolerance),
        }
    }

    /// Decompose an n-qubit unitary
    pub fn decompose(
        &mut self,
        unitary: &Array2<Complex<f64>>,
        qubit_ids: &[QubitId],
    ) -> QuantRS2Result<MultiQubitKAK> {
        let n = qubit_ids.len();
        let size = 1 << n;

        // Validate input
        if unitary.shape() != [size, size] {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Unitary size {} doesn't match {} qubits",
                unitary.shape()[0],
                n
            )));
        }

        // Check unitarity
        let mat = DenseMatrix::new(unitary.clone())?;
        if !mat.is_unitary(self.tolerance)? {
            return Err(QuantRS2Error::InvalidInput(
                "Matrix is not unitary".to_string(),
            ));
        }

        // Check cache
        if let Some(cached) = self.check_cache(unitary) {
            return Ok(cached.clone());
        }

        // Perform decomposition
        let (tree, gates) = self.decompose_recursive(unitary, qubit_ids, 0)?;

        // Count gates
        let mut cnot_count = 0;
        let mut single_qubit_count = 0;

        for gate in &gates {
            match gate.name() {
                "CNOT" | "CZ" | "SWAP" => cnot_count += self.count_cnots(gate.name()),
                _ => single_qubit_count += 1,
            }
        }

        // Calculate actual circuit depth using critical path through DAG.
        // For each gate in topological order:
        //   depth[gate] = 1 + max(depth[prev_gate]) over all preceding gates sharing a qubit.
        let depth = Self::calculate_circuit_depth(&gates);

        let result = MultiQubitKAK {
            gates,
            tree,
            cnot_count,
            single_qubit_count,
            depth,
        };

        // Cache result
        self.cache_result(unitary, &result);

        Ok(result)
    }

    /// Recursive decomposition algorithm
    fn decompose_recursive(
        &mut self,
        unitary: &Array2<Complex<f64>>,
        qubit_ids: &[QubitId],
        depth: usize,
    ) -> QuantRS2Result<(DecompositionTree, Vec<Box<dyn GateOp>>)> {
        if depth > self.max_depth {
            return Err(QuantRS2Error::InvalidInput(
                "Maximum recursion depth exceeded".to_string(),
            ));
        }

        let n = qubit_ids.len();

        // Base cases
        match n {
            0 => {
                let tree = DecompositionTree::Leaf {
                    qubits: vec![],
                    gate_type: LeafType::SingleQubit(SingleQubitDecomposition {
                        global_phase: 0.0,
                        theta1: 0.0,
                        phi: 0.0,
                        theta2: 0.0,
                        basis: "ZYZ".to_string(),
                    }),
                };
                Ok((tree, vec![]))
            }
            1 => {
                let decomp = decompose_single_qubit_zyz(&unitary.view())?;
                let gates = self.single_qubit_to_gates(&decomp, qubit_ids[0]);
                let tree = DecompositionTree::Leaf {
                    qubits: qubit_ids.to_vec(),
                    gate_type: LeafType::SingleQubit(decomp),
                };
                Ok((tree, gates))
            }
            2 => {
                let decomp = self.cartan.decompose(unitary)?;
                let gates = self.cartan.to_gates(&decomp, qubit_ids)?;
                let tree = DecompositionTree::Leaf {
                    qubits: qubit_ids.to_vec(),
                    gate_type: LeafType::TwoQubit(decomp),
                };
                Ok((tree, gates))
            }
            _ => {
                // For n > 2, choose decomposition method
                let method = self.choose_decomposition_method(unitary, n);

                match method {
                    DecompositionMethod::CSD { pivot } => {
                        self.decompose_csd(unitary, qubit_ids, pivot, depth)
                    }
                    DecompositionMethod::Shannon { partition } => {
                        self.decompose_shannon(unitary, qubit_ids, partition, depth)
                    }
                    DecompositionMethod::BlockDiagonal { block_size } => {
                        self.decompose_block_diagonal(unitary, qubit_ids, block_size, depth)
                    }
                    DecompositionMethod::Cartan => unreachable!("Invalid method for n > 2"),
                }
            }
        }
    }

    /// Choose optimal decomposition method based on matrix structure
    fn choose_decomposition_method(
        &self,
        unitary: &Array2<Complex<f64>>,
        n: usize,
    ) -> DecompositionMethod {
        if self.use_optimization {
            // Analyze matrix structure to choose optimal method
            if self.has_block_structure(unitary, n) {
                DecompositionMethod::BlockDiagonal { block_size: n / 2 }
            } else if n % 2 == 0 {
                // Even number of qubits - use CSD at midpoint
                DecompositionMethod::CSD { pivot: n / 2 }
            } else {
                // Odd number - use Shannon decomposition
                DecompositionMethod::Shannon { partition: n / 2 }
            }
        } else {
            // Default to CSD
            DecompositionMethod::CSD { pivot: n / 2 }
        }
    }

    /// Decompose using Cosine-Sine Decomposition
    fn decompose_csd(
        &mut self,
        unitary: &Array2<Complex<f64>>,
        qubit_ids: &[QubitId],
        pivot: usize,
        depth: usize,
    ) -> QuantRS2Result<(DecompositionTree, Vec<Box<dyn GateOp>>)> {
        let n = qubit_ids.len();
        // let _size = 1 << n;
        let pivot_size = 1 << pivot;

        // Split unitary into blocks based on pivot
        // U = [A B]
        //     [C D]
        let a = unitary.slice(s![..pivot_size, ..pivot_size]).to_owned();
        let b = unitary.slice(s![..pivot_size, pivot_size..]).to_owned();
        let c = unitary.slice(s![pivot_size.., ..pivot_size]).to_owned();
        let d = unitary.slice(s![pivot_size.., pivot_size..]).to_owned();

        // Apply CSD to find:
        // U = (U1 ⊗ V1) · Σ · (U2 ⊗ V2)
        // where Σ is diagonal in the CSD basis

        // This is a simplified version - full CSD would use SVD
        let (u1, v1, sigma, u2, v2) = self.compute_csd(&a, &b, &c, &d)?;

        let mut gates = Vec::new();
        let mut children = Vec::new();

        // Decompose U2 and V2 (right multiplications)
        let left_qubits = &qubit_ids[..pivot];
        let right_qubits = &qubit_ids[pivot..];

        let (u2_tree, u2_gates) = self.decompose_recursive(&u2, left_qubits, depth + 1)?;
        let (v2_tree, v2_gates) = self.decompose_recursive(&v2, right_qubits, depth + 1)?;

        gates.extend(u2_gates);
        gates.extend(v2_gates);
        children.push(u2_tree);
        children.push(v2_tree);

        // Apply diagonal gates (controlled rotations)
        let diag_gates = self.diagonal_to_gates(&sigma, qubit_ids)?;
        gates.extend(diag_gates);

        // Decompose U1 and V1 (left multiplications)
        let (u1_tree, u1_gates) = self.decompose_recursive(&u1, left_qubits, depth + 1)?;
        let (v1_tree, v1_gates) = self.decompose_recursive(&v1, right_qubits, depth + 1)?;

        gates.extend(u1_gates);
        gates.extend(v1_gates);
        children.push(u1_tree);
        children.push(v1_tree);

        let tree = DecompositionTree::Node {
            qubits: qubit_ids.to_vec(),
            method: DecompositionMethod::CSD { pivot },
            children,
        };

        Ok((tree, gates))
    }

    /// Decompose using Shannon decomposition
    fn decompose_shannon(
        &self,
        unitary: &Array2<Complex<f64>>,
        qubit_ids: &[QubitId],
        partition: usize,
        _depth: usize,
    ) -> QuantRS2Result<(DecompositionTree, Vec<Box<dyn GateOp>>)> {
        // Use the Shannon decomposer for this
        let mut shannon = ShannonDecomposer::new();
        let decomp = shannon.decompose(unitary, qubit_ids)?;

        // Build tree structure
        let tree = DecompositionTree::Node {
            qubits: qubit_ids.to_vec(),
            method: DecompositionMethod::Shannon { partition },
            children: vec![], // Shannon decomposer doesn't provide tree structure
        };

        Ok((tree, decomp.gates))
    }

    /// Decompose block diagonal matrix
    fn decompose_block_diagonal(
        &mut self,
        unitary: &Array2<Complex<f64>>,
        qubit_ids: &[QubitId],
        block_size: usize,
        depth: usize,
    ) -> QuantRS2Result<(DecompositionTree, Vec<Box<dyn GateOp>>)> {
        let n = qubit_ids.len();
        let num_blocks = n / block_size;

        let mut gates = Vec::new();
        let mut children = Vec::new();

        // Decompose each block independently
        for i in 0..num_blocks {
            let start = i * block_size;
            let end = (i + 1) * block_size;
            let block_qubits = &qubit_ids[start..end];

            // Extract block from unitary
            let block = self.extract_block(unitary, i, block_size)?;

            let (block_tree, block_gates) =
                self.decompose_recursive(&block, block_qubits, depth + 1)?;
            gates.extend(block_gates);
            children.push(block_tree);
        }

        let tree = DecompositionTree::Node {
            qubits: qubit_ids.to_vec(),
            method: DecompositionMethod::BlockDiagonal { block_size },
            children,
        };

        Ok((tree, gates))
    }

    /// Compute the Cosine-Sine Decomposition (CSD) of the unitary block matrix
    /// `W = [[A, B], [C, D]]`.
    ///
    /// Returns `(U1, V1, Σ, U2, V2)` such that
    ///
    /// ```text
    /// W = diag(U1, U2) · Σ · diag(V1, V2)†
    /// ```
    ///
    /// where `U1, U2, V1, V2` are `n×n` unitaries and `Σ` is the `2n×2n` CS matrix
    /// `[[C', S'], [-S', C']]` with diagonal cosine/sine blocks `C' = diag(cos θ_k)`,
    /// `S' = diag(sin θ_k)` (so `Σ` is itself orthogonal).
    ///
    /// Algorithm (Stewart-style, driven by an SVD of the `A` block):
    /// 1. `A = U1 · C' · V1†` via SVD; the singular values are the cosines `cos θ_k`.
    /// 2. The columns of `C · V1` are mutually orthogonal with norms `sin θ_k`; this
    ///    fixes `S'` and `U2 = (C V1) S'^{-1}` (with a safe fallback for `sin θ_k ≈ 0`).
    /// 3. `V2` is recovered from `D = U2 · C' · V2†` (or `B = -U1 · S' · V2†` when a
    ///    cosine vanishes) so that the full block identity holds.
    ///
    /// The result is verified against `W`; if a numerically consistent CSD cannot be
    /// produced (degenerate edge cases the simple driver does not cover), an honest
    /// [`QuantRS2Error::UnsupportedOperation`] is returned rather than a fabricated
    /// identity.
    fn compute_csd(
        &self,
        a: &Array2<Complex<f64>>,
        b: &Array2<Complex<f64>>,
        c: &Array2<Complex<f64>>,
        d: &Array2<Complex<f64>>,
    ) -> QuantRS2Result<(
        Array2<Complex<f64>>, // U1
        Array2<Complex<f64>>, // V1
        Array2<Complex<f64>>, // Sigma (2n x 2n)
        Array2<Complex<f64>>, // U2
        Array2<Complex<f64>>, // V2
    )> {
        let n = a.shape()[0];
        let tol = self.tolerance.max(1e-12);

        // Step 1: SVD of A -> A = U1 · diag(cos) · V1†.
        let (u1, cos_vals, v1h) = complex_svd(a)?;
        let v1 = v1h.mapv(|z| z.conj()).t().to_owned(); // V1 (n x n)

        // Cosines, clamped to [0, 1] for numerical safety.
        let cos: Vec<f64> = cos_vals.iter().map(|&s| s.clamp(0.0, 1.0)).collect();

        // Step 2: with Σ = [[C', S'], [-S', C']] the block identity gives
        // C = -U2 · S' · V1†, hence M = C·V1 = -U2·S'. Its columns are orthogonal with
        // norm sin θ_k, and U2 = -M · S'^{-1}.
        let m = c.dot(&v1);
        let mut sin = vec![0.0f64; n];
        for k in 0..n {
            let col_norm = (0..n).map(|r| m[[r, k]].norm_sqr()).sum::<f64>().sqrt();
            // Prefer the value consistent with cos²+sin²=1 when the column norm is
            // well defined; otherwise derive sin from cos.
            sin[k] = if col_norm > tol {
                col_norm
            } else {
                (1.0 - cos[k] * cos[k]).max(0.0).sqrt()
            };
        }

        // U2 columns: where sin θ_k is non-negligible, U2[:,k] = -M[:,k] / sin θ_k.
        // Columns with sin θ_k ≈ 0 are filled afterwards by orthonormal completion.
        let mut u2 = Array2::<Complex<f64>>::zeros((n, n));
        let mut needs_completion: Vec<usize> = Vec::new();
        for k in 0..n {
            if sin[k] > tol {
                for r in 0..n {
                    u2[[r, k]] = -m[[r, k]] / Complex::new(sin[k], 0.0);
                }
            } else {
                needs_completion.push(k);
            }
        }
        if !needs_completion.is_empty() {
            complete_orthonormal_columns(&mut u2, &needs_completion, tol);
        }

        // Step 3: recover V2. From D = U2 · C' · V2†  ==>  V2† = C'^{-1} U2† D, falling
        // back to B = U1 · S' · V2†  ==>  V2† = S'^{-1} U1† B for rows where cos θ_k ≈ 0.
        let u2h = u2.mapv(|z| z.conj()).t().to_owned();
        let u1h = u1.mapv(|z| z.conj()).t().to_owned();
        let u2h_d = u2h.dot(d); // (n x n)
        let u1h_b = u1h.dot(b); // (n x n)
        let mut v2h = Array2::<Complex<f64>>::zeros((n, n));
        for k in 0..n {
            if cos[k] > tol {
                for col in 0..n {
                    v2h[[k, col]] = u2h_d[[k, col]] / Complex::new(cos[k], 0.0);
                }
            } else if sin[k] > tol {
                for col in 0..n {
                    v2h[[k, col]] = u1h_b[[k, col]] / Complex::new(sin[k], 0.0);
                }
            } else {
                // cos = sin = 0 is impossible for a unitary; bail out honestly.
                return Err(QuantRS2Error::UnsupportedOperation(
                    "cosine-sine decomposition encountered a degenerate angle (cos=sin=0)"
                        .to_string(),
                ));
            }
        }
        let v2 = v2h.mapv(|z| z.conj()).t().to_owned();

        // Assemble Σ = [[C', S'], [-S', C']].
        let mut sigma = Array2::<Complex<f64>>::zeros((2 * n, 2 * n));
        for k in 0..n {
            sigma[[k, k]] = Complex::new(cos[k], 0.0);
            sigma[[k, n + k]] = Complex::new(sin[k], 0.0);
            sigma[[n + k, k]] = Complex::new(-sin[k], 0.0);
            sigma[[n + k, n + k]] = Complex::new(cos[k], 0.0);
        }

        // Verify: W ?= diag(U1, U2) · Σ · diag(V1, V2)†.
        let recon = Self::assemble_from_csd(&u1, &u2, &sigma, &v1, &v2);
        let mut max_err = 0.0f64;
        for (i, j, expected) in [(0usize, 0usize, a), (0, 1, b), (1, 0, c), (1, 1, d)]
            .iter()
            .flat_map(|&(bi, bj, blk)| {
                (0..n).flat_map(move |r| {
                    (0..n).map(move |col| (bi * n + r, bj * n + col, blk[[r, col]]))
                })
            })
        {
            max_err = max_err.max((recon[[i, j]] - expected).norm());
        }

        if max_err > 1e-7 {
            return Err(QuantRS2Error::UnsupportedOperation(format!(
                "cosine-sine decomposition for this matrix is not yet supported \
                 (reconstruction error {max_err:.3e})"
            )));
        }

        Ok((u1, v1, sigma, u2, v2))
    }

    /// Reassemble `diag(U1, U2) · Σ · diag(V1, V2)†` into a `2n×2n` matrix.
    fn assemble_from_csd(
        u1: &Array2<Complex<f64>>,
        u2: &Array2<Complex<f64>>,
        sigma: &Array2<Complex<f64>>,
        v1: &Array2<Complex<f64>>,
        v2: &Array2<Complex<f64>>,
    ) -> Array2<Complex<f64>> {
        let n = u1.nrows();
        let mut left = Array2::<Complex<f64>>::zeros((2 * n, 2 * n));
        left.slice_mut(s![..n, ..n]).assign(u1);
        left.slice_mut(s![n.., n..]).assign(u2);

        let mut right_dag = Array2::<Complex<f64>>::zeros((2 * n, 2 * n));
        let v1h = v1.mapv(|z| z.conj()).t().to_owned();
        let v2h = v2.mapv(|z| z.conj()).t().to_owned();
        right_dag.slice_mut(s![..n, ..n]).assign(&v1h);
        right_dag.slice_mut(s![n.., n..]).assign(&v2h);

        left.dot(sigma).dot(&right_dag)
    }

    /// Convert diagonal matrix to controlled rotation gates
    fn diagonal_to_gates(
        &self,
        diagonal: &Array2<Complex<f64>>,
        qubit_ids: &[QubitId],
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut gates = Vec::new();

        // Extract diagonal elements
        let n = diagonal.shape()[0];
        for i in 0..n {
            let phase = diagonal[[i, i]].arg();
            if phase.abs() > self.tolerance {
                // Determine which qubits are in state |1⟩ for this diagonal element
                let mut control_pattern = Vec::new();
                let mut temp = i;
                for j in 0..qubit_ids.len() {
                    if temp & 1 == 1 {
                        control_pattern.push(j);
                    }
                    temp >>= 1;
                }

                // Create multi-controlled phase gate
                if control_pattern.is_empty() {
                    // Global phase - can be ignored
                } else if control_pattern.len() == 1 {
                    // Single-qubit phase
                    gates.push(Box::new(RotationZ {
                        target: qubit_ids[control_pattern[0]],
                        theta: phase,
                    }) as Box<dyn GateOp>);
                } else {
                    // Multi-controlled phase - decompose further
                    // For now, use simple decomposition
                    // Note: control_pattern.len() >= 2 at this point, so pop is safe
                    let target_idx = control_pattern.pop().unwrap_or(0);
                    let target = qubit_ids[target_idx];
                    for &control_idx in &control_pattern {
                        gates.push(Box::new(CNOT {
                            control: qubit_ids[control_idx],
                            target,
                        }));
                    }

                    gates.push(Box::new(RotationZ {
                        target,
                        theta: phase,
                    }) as Box<dyn GateOp>);

                    // Uncompute CNOTs
                    for &control_idx in control_pattern.iter().rev() {
                        gates.push(Box::new(CNOT {
                            control: qubit_ids[control_idx],
                            target,
                        }));
                    }
                }
            }
        }

        Ok(gates)
    }

    /// Check if matrix has block diagonal structure
    fn has_block_structure(&self, unitary: &Array2<Complex<f64>>, _n: usize) -> bool {
        // Simple check - look for zeros in off-diagonal blocks
        let size = unitary.shape()[0];
        let block_size = size / 2;

        let mut off_diagonal_norm = 0.0;

        // Check upper-right block
        for i in 0..block_size {
            for j in block_size..size {
                off_diagonal_norm += unitary[[i, j]].norm_sqr();
            }
        }

        // Check lower-left block
        for i in block_size..size {
            for j in 0..block_size {
                off_diagonal_norm += unitary[[i, j]].norm_sqr();
            }
        }

        off_diagonal_norm.sqrt() < self.tolerance
    }

    /// Extract a block from block-diagonal matrix
    fn extract_block(
        &self,
        unitary: &Array2<Complex<f64>>,
        block_idx: usize,
        block_size: usize,
    ) -> QuantRS2Result<Array2<Complex<f64>>> {
        let size = 1 << block_size;
        let start = block_idx * size;
        let end = (block_idx + 1) * size;

        Ok(unitary.slice(s![start..end, start..end]).to_owned())
    }

    /// Convert single-qubit decomposition to gates
    fn single_qubit_to_gates(
        &self,
        decomp: &SingleQubitDecomposition,
        qubit: QubitId,
    ) -> Vec<Box<dyn GateOp>> {
        let mut gates = Vec::new();

        if decomp.theta1.abs() > self.tolerance {
            gates.push(Box::new(RotationZ {
                target: qubit,
                theta: decomp.theta1,
            }) as Box<dyn GateOp>);
        }

        if decomp.phi.abs() > self.tolerance {
            gates.push(Box::new(RotationY {
                target: qubit,
                theta: decomp.phi,
            }) as Box<dyn GateOp>);
        }

        if decomp.theta2.abs() > self.tolerance {
            gates.push(Box::new(RotationZ {
                target: qubit,
                theta: decomp.theta2,
            }) as Box<dyn GateOp>);
        }

        gates
    }

    /// Count CNOTs for different gate types
    fn count_cnots(&self, gate_name: &str) -> usize {
        match gate_name {
            "CNOT" | "CZ" => 1, // CZ = H·CNOT·H
            "SWAP" => 3,        // SWAP uses 3 CNOTs
            _ => 0,
        }
    }

    /// Check cache for existing decomposition
    /// Calculate circuit depth as the length of the critical path through the DAG.
    ///
    /// For each gate in topological order (gates are already ordered):
    ///   `depth[i] = 1 + max(depth[j])` for all j < i that share at least one qubit with gate i.
    ///
    /// Uses a BFS/forward-pass approach since gates are given in topological order.
    fn calculate_circuit_depth(gates: &[Box<dyn GateOp>]) -> usize {
        if gates.is_empty() {
            return 0;
        }

        // depth_at[i] = the depth level at which gate i completes (1-based)
        let mut depth_at: Vec<usize> = vec![0; gates.len()];
        // last_qubit_depth maps qubit id -> (gate_index, depth) of the last gate on that qubit
        let mut last_qubit_finish: FxHashMap<u32, usize> = FxHashMap::default();

        for (i, gate) in gates.iter().enumerate() {
            let qubits = gate.qubits();
            // Find the maximum finish depth among all preceding gates on shared qubits
            let predecessor_max_depth = qubits
                .iter()
                .filter_map(|q| last_qubit_finish.get(&q.0).copied())
                .max()
                .unwrap_or(0);

            depth_at[i] = predecessor_max_depth + 1;

            // Update last finish depth for each qubit this gate touches
            for q in &qubits {
                last_qubit_finish.insert(q.0, depth_at[i]);
            }
        }

        depth_at.into_iter().max().unwrap_or(0)
    }

    const fn check_cache(&self, _unitary: &Array2<Complex<f64>>) -> Option<&MultiQubitKAK> {
        // Simple hash based on first few elements
        // Real implementation would use better hashing
        None
    }

    /// Cache decomposition result
    const fn cache_result(&self, _unitary: &Array2<Complex<f64>>, _result: &MultiQubitKAK) {
        // Cache implementation
    }
}

impl Default for MultiQubitKAKDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze decomposition tree structure
pub struct KAKTreeAnalyzer {
    /// Track statistics
    stats: DecompositionStats,
}

#[derive(Debug, Default, Clone)]
pub struct DecompositionStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub max_depth: usize,
    pub method_counts: FxHashMap<String, usize>,
    pub cnot_distribution: FxHashMap<usize, usize>,
}

impl KAKTreeAnalyzer {
    /// Create new analyzer
    pub fn new() -> Self {
        Self {
            stats: DecompositionStats::default(),
        }
    }

    /// Analyze decomposition tree
    pub fn analyze(&mut self, tree: &DecompositionTree) -> DecompositionStats {
        self.stats = DecompositionStats::default();
        self.analyze_recursive(tree, 0);
        self.stats.clone()
    }

    fn analyze_recursive(&mut self, tree: &DecompositionTree, depth: usize) {
        self.stats.total_nodes += 1;
        self.stats.max_depth = self.stats.max_depth.max(depth);

        match tree {
            DecompositionTree::Leaf {
                qubits: _qubits,
                gate_type,
            } => {
                self.stats.leaf_nodes += 1;

                match gate_type {
                    LeafType::SingleQubit(_) => {
                        *self
                            .stats
                            .method_counts
                            .entry("single_qubit".to_string())
                            .or_insert(0) += 1;
                    }
                    LeafType::TwoQubit(cartan) => {
                        *self
                            .stats
                            .method_counts
                            .entry("two_qubit".to_string())
                            .or_insert(0) += 1;
                        let cnots = cartan.interaction.cnot_count(1e-10);
                        *self.stats.cnot_distribution.entry(cnots).or_insert(0) += 1;
                    }
                }
            }
            DecompositionTree::Node {
                method, children, ..
            } => {
                let method_name = match method {
                    DecompositionMethod::CSD { .. } => "csd",
                    DecompositionMethod::Shannon { .. } => "shannon",
                    DecompositionMethod::BlockDiagonal { .. } => "block_diagonal",
                    DecompositionMethod::Cartan => "cartan",
                };
                *self
                    .stats
                    .method_counts
                    .entry(method_name.to_string())
                    .or_insert(0) += 1;

                for child in children {
                    self.analyze_recursive(child, depth + 1);
                }
            }
        }
    }
}

/// Utility function for quick multi-qubit KAK decomposition
pub fn kak_decompose_multiqubit(
    unitary: &Array2<Complex<f64>>,
    qubit_ids: &[QubitId],
) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
    let mut decomposer = MultiQubitKAKDecomposer::new();
    let decomp = decomposer.decompose(unitary, qubit_ids)?;
    Ok(decomp.gates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::Complex;

    #[test]
    fn test_multiqubit_kak_single() {
        let mut decomposer = MultiQubitKAKDecomposer::new();

        // Hadamard matrix
        let h = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(-1.0, 0.0),
            ],
        )
        .expect("Failed to create Hadamard matrix")
            / Complex::new(2.0_f64.sqrt(), 0.0);

        let qubit_ids = vec![QubitId(0)];
        let decomp = decomposer
            .decompose(&h, &qubit_ids)
            .expect("Single-qubit KAK decomposition failed");

        assert!(decomp.single_qubit_count <= 3);
        assert_eq!(decomp.cnot_count, 0);

        // Check tree structure
        match &decomp.tree {
            DecompositionTree::Leaf {
                gate_type: LeafType::SingleQubit(_),
                ..
            } => {}
            _ => panic!("Expected single-qubit leaf"),
        }
    }

    #[test]
    fn test_multiqubit_kak_two() {
        let mut decomposer = MultiQubitKAKDecomposer::new();

        // CNOT matrix
        let cnot = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
        )
        .expect("Failed to create CNOT matrix");

        let qubit_ids = vec![QubitId(0), QubitId(1)];
        let decomp = decomposer
            .decompose(&cnot, &qubit_ids)
            .expect("Two-qubit KAK decomposition failed");

        assert!(decomp.cnot_count <= 1);

        // Check tree structure
        match &decomp.tree {
            DecompositionTree::Leaf {
                gate_type: LeafType::TwoQubit(_),
                ..
            } => {}
            _ => panic!("Expected two-qubit leaf"),
        }
    }

    #[test]
    fn test_multiqubit_kak_three() {
        let mut decomposer = MultiQubitKAKDecomposer::new();

        // 3-qubit identity
        let identity = Array2::eye(8);
        let identity_complex = identity.mapv(|x| Complex::new(x, 0.0));

        let qubit_ids = vec![QubitId(0), QubitId(1), QubitId(2)];
        let decomp = decomposer
            .decompose(&identity_complex, &qubit_ids)
            .expect("Three-qubit KAK decomposition failed");

        // Identity should result in empty circuit
        assert_eq!(decomp.gates.len(), 0);
        assert_eq!(decomp.cnot_count, 0);
        assert_eq!(decomp.single_qubit_count, 0);
    }

    #[test]
    fn test_tree_analyzer() {
        let mut analyzer = KAKTreeAnalyzer::new();

        // Create a simple tree
        let tree = DecompositionTree::Node {
            qubits: vec![QubitId(0), QubitId(1), QubitId(2)],
            method: DecompositionMethod::CSD { pivot: 2 },
            children: vec![
                DecompositionTree::Leaf {
                    qubits: vec![QubitId(0), QubitId(1)],
                    gate_type: LeafType::TwoQubit(CartanDecomposition {
                        left_gates: (
                            SingleQubitDecomposition {
                                global_phase: 0.0,
                                theta1: 0.0,
                                phi: 0.0,
                                theta2: 0.0,
                                basis: "ZYZ".to_string(),
                            },
                            SingleQubitDecomposition {
                                global_phase: 0.0,
                                theta1: 0.0,
                                phi: 0.0,
                                theta2: 0.0,
                                basis: "ZYZ".to_string(),
                            },
                        ),
                        right_gates: (
                            SingleQubitDecomposition {
                                global_phase: 0.0,
                                theta1: 0.0,
                                phi: 0.0,
                                theta2: 0.0,
                                basis: "ZYZ".to_string(),
                            },
                            SingleQubitDecomposition {
                                global_phase: 0.0,
                                theta1: 0.0,
                                phi: 0.0,
                                theta2: 0.0,
                                basis: "ZYZ".to_string(),
                            },
                        ),
                        interaction: crate::prelude::CartanCoefficients::new(0.0, 0.0, 0.0),
                        global_phase: 0.0,
                    }),
                },
                DecompositionTree::Leaf {
                    qubits: vec![QubitId(2)],
                    gate_type: LeafType::SingleQubit(SingleQubitDecomposition {
                        global_phase: 0.0,
                        theta1: 0.0,
                        phi: 0.0,
                        theta2: 0.0,
                        basis: "ZYZ".to_string(),
                    }),
                },
            ],
        };

        let stats = analyzer.analyze(&tree);

        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.leaf_nodes, 2);
        assert_eq!(stats.max_depth, 1);
        assert_eq!(stats.method_counts.get("csd"), Some(&1));
    }

    /// Gram–Schmidt orthonormalisation of the columns of a square complex matrix,
    /// producing a unitary (used to manufacture test unitaries).
    fn orthonormalize(mut m: Array2<Complex<f64>>) -> Array2<Complex<f64>> {
        let n = m.nrows();
        for j in 0..n {
            for prev in 0..j {
                let proj: Complex<f64> = (0..n).map(|r| m[[r, prev]].conj() * m[[r, j]]).sum();
                for r in 0..n {
                    let sub = proj * m[[r, prev]];
                    m[[r, j]] -= sub;
                }
            }
            let norm = (0..n).map(|r| m[[r, j]].norm_sqr()).sum::<f64>().sqrt();
            for r in 0..n {
                m[[r, j]] /= Complex::new(norm, 0.0);
            }
        }
        m
    }

    /// Site-3 proof: the CSD returned by `compute_csd` recomposes to the original
    /// unitary block matrix `W = [[A,B],[C,D]]` within 1e-8, and the factors are the
    /// promised unitaries / CS structure (no fabricated identities).
    #[test]
    fn test_compute_csd_recomposes() {
        let decomposer = MultiQubitKAKDecomposer::new();

        // Manufacture a 4x4 unitary W (n = 2 blocks) from a fixed complex matrix.
        let raw = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(0.5, 0.2),
                Complex::new(-0.3, 0.4),
                Complex::new(0.1, -0.2),
                Complex::new(0.6, 0.0),
                Complex::new(0.2, -0.1),
                Complex::new(0.5, 0.3),
                Complex::new(-0.4, 0.1),
                Complex::new(0.0, 0.2),
                Complex::new(-0.3, 0.2),
                Complex::new(0.1, 0.5),
                Complex::new(0.6, -0.1),
                Complex::new(0.2, 0.3),
                Complex::new(0.4, 0.0),
                Complex::new(-0.2, 0.3),
                Complex::new(0.1, 0.4),
                Complex::new(0.5, -0.2),
            ],
        )
        .expect("raw matrix");
        let w = orthonormalize(raw);

        // Confirm W is unitary.
        let wh = w.mapv(|z| z.conj()).t().to_owned();
        let prod = wh.dot(&w);
        for i in 0..4 {
            for j in 0..4 {
                let exp = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[[i, j]] - Complex::new(exp, 0.0)).norm() < 1e-9,
                    "manufactured W is not unitary"
                );
            }
        }

        let a = w.slice(s![..2, ..2]).to_owned();
        let b = w.slice(s![..2, 2..]).to_owned();
        let c = w.slice(s![2.., ..2]).to_owned();
        let d = w.slice(s![2.., 2..]).to_owned();

        let (u1, v1, sigma, u2, v2) = decomposer
            .compute_csd(&a, &b, &c, &d)
            .expect("CSD should succeed for a generic 4x4 unitary");

        // U1, U2, V1, V2 must be unitary (not identity-fabrications unless genuinely so).
        for (name, mat) in [("U1", &u1), ("U2", &u2), ("V1", &v1), ("V2", &v2)] {
            let mh = mat.mapv(|z| z.conj()).t().to_owned();
            let pr = mh.dot(mat);
            for i in 0..2 {
                for j in 0..2 {
                    let exp = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (pr[[i, j]] - Complex::new(exp, 0.0)).norm() < 1e-7,
                        "{name} is not unitary"
                    );
                }
            }
        }

        // Sigma must have the CS structure: Σ = [[C', S'], [-S', C']] and be orthogonal.
        let sh = sigma.mapv(|z| z.conj()).t().to_owned();
        let sps = sh.dot(&sigma);
        for i in 0..4 {
            for j in 0..4 {
                let exp = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sps[[i, j]] - Complex::new(exp, 0.0)).norm() < 1e-7,
                    "Sigma (CS matrix) is not orthogonal"
                );
            }
        }

        // Recompose and compare to W.
        let recon = MultiQubitKAKDecomposer::assemble_from_csd(&u1, &u2, &sigma, &v1, &v2);
        let mut max_err = 0.0f64;
        for i in 0..4 {
            for j in 0..4 {
                max_err = max_err.max((recon[[i, j]] - w[[i, j]]).norm());
            }
        }
        assert!(
            max_err < 1e-8,
            "CSD recomposition error {max_err} exceeds 1e-8"
        );
    }

    #[test]
    fn test_complex_svd_helper_roundtrip() {
        // Independent check of the complex SVD used by the CSD.
        let m = Array2::from_shape_vec(
            (3, 3),
            vec![
                Complex::new(1.0, 0.2),
                Complex::new(0.3, -0.4),
                Complex::new(-0.1, 0.5),
                Complex::new(0.2, 0.1),
                Complex::new(-0.5, 0.3),
                Complex::new(0.4, 0.0),
                Complex::new(0.0, -0.3),
                Complex::new(0.6, 0.1),
                Complex::new(-0.2, 0.2),
            ],
        )
        .expect("matrix");
        let (u, s, vh) = complex_svd(&m).expect("svd");
        let mut s_mat = Array2::<Complex<f64>>::zeros((3, 3));
        for i in 0..3 {
            s_mat[[i, i]] = Complex::new(s[i], 0.0);
        }
        let recon = u.dot(&s_mat).dot(&vh);
        let mut err = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                err = err.max((recon[[i, j]] - m[[i, j]]).norm());
            }
        }
        assert!(err < 1e-9, "complex_svd roundtrip error {err}");
    }

    #[test]
    fn test_block_structure_detection() {
        let decomposer = MultiQubitKAKDecomposer::new();

        // Create block diagonal matrix
        let mut block_diag = Array2::zeros((4, 4));
        block_diag[[0, 0]] = Complex::new(1.0, 0.0);
        block_diag[[1, 1]] = Complex::new(1.0, 0.0);
        block_diag[[2, 2]] = Complex::new(1.0, 0.0);
        block_diag[[3, 3]] = Complex::new(1.0, 0.0);

        assert!(decomposer.has_block_structure(&block_diag, 2));

        // Non-block diagonal
        block_diag[[0, 2]] = Complex::new(1.0, 0.0);
        assert!(!decomposer.has_block_structure(&block_diag, 2));
    }
}
