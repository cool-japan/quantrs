//! Matrix Product State (MPS) quantum simulator
//!
//! This module implements an efficient quantum simulator using the Matrix Product State
//! representation, which is particularly effective for simulating quantum systems with
//! limited entanglement.

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    prelude::QubitId,
    register::Register,
};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView2};
use scirs2_core::Complex64;

/// MPS tensor for a single qubit
#[derive(Debug, Clone)]
struct MPSTensor {
    /// The tensor data: `left_bond` x physical x `right_bond`
    data: Array3<Complex64>,
    /// Left bond dimension
    left_dim: usize,
    /// Right bond dimension
    right_dim: usize,
}

impl MPSTensor {
    /// Create a new MPS tensor
    fn new(data: Array3<Complex64>) -> Self {
        let shape = data.shape();
        Self {
            left_dim: shape[0],
            right_dim: shape[2],
            data,
        }
    }

    /// Create initial tensor for |0> state
    fn zero_state(is_first: bool, is_last: bool) -> Self {
        let data = if is_first && is_last {
            // Single qubit: 1x2x1 tensor
            let mut tensor = Array3::zeros((1, 2, 1));
            tensor[[0, 0, 0]] = Complex64::new(1.0, 0.0);
            tensor
        } else if is_first {
            // First qubit: 1x2xD tensor
            let mut tensor = Array3::zeros((1, 2, 2));
            tensor[[0, 0, 0]] = Complex64::new(1.0, 0.0);
            tensor
        } else if is_last {
            // Last qubit: Dx2x1 tensor
            let mut tensor = Array3::zeros((2, 2, 1));
            tensor[[0, 0, 0]] = Complex64::new(1.0, 0.0);
            tensor
        } else {
            // Middle qubit: Dx2xD tensor
            let mut tensor = Array3::zeros((2, 2, 2));
            tensor[[0, 0, 0]] = Complex64::new(1.0, 0.0);
            tensor
        };
        Self::new(data)
    }
}

/// Matrix Product State representation of a quantum state
pub struct MPS {
    /// MPS tensors for each qubit
    tensors: Vec<MPSTensor>,
    /// Number of qubits
    num_qubits: usize,
    /// Maximum allowed bond dimension
    max_bond_dim: usize,
    /// SVD truncation threshold
    truncation_threshold: f64,
    /// Current orthogonality center (-1 if not in canonical form)
    orthogonality_center: i32,
}

impl MPS {
    /// Create a new MPS in the |0...0> state
    #[must_use]
    pub fn new(num_qubits: usize, max_bond_dim: usize) -> Self {
        let tensors = (0..num_qubits)
            .map(|i| MPSTensor::zero_state(i == 0, i == num_qubits - 1))
            .collect();

        Self {
            tensors,
            num_qubits,
            max_bond_dim,
            truncation_threshold: 1e-10,
            orthogonality_center: -1,
        }
    }

    /// Set the truncation threshold for SVD
    pub const fn set_truncation_threshold(&mut self, threshold: f64) {
        self.truncation_threshold = threshold;
    }

    /// Move orthogonality center to specified position
    pub fn move_orthogonality_center(&mut self, target: usize) -> QuantRS2Result<()> {
        if target >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(target as u32));
        }

        // If no current center, canonicalize from left
        if self.orthogonality_center < 0 {
            self.left_canonicalize_up_to(target)?;
            self.orthogonality_center = target as i32;
            return Ok(());
        }

        let current = self.orthogonality_center as usize;

        if current < target {
            // Move right
            for i in current..target {
                self.move_center_right(i)?;
            }
        } else if current > target {
            // Move left
            for i in (target + 1..=current).rev() {
                self.move_center_left(i)?;
            }
        }

        self.orthogonality_center = target as i32;
        Ok(())
    }

    /// Left-canonicalize tensors up to position
    fn left_canonicalize_up_to(&mut self, position: usize) -> QuantRS2Result<()> {
        for i in 0..position {
            let tensor = &self.tensors[i];
            let (left_dim, phys_dim, right_dim) = (tensor.left_dim, 2, tensor.right_dim);

            // Reshape to matrix for QR decomposition
            let matrix = tensor
                .data
                .view()
                .into_shape((left_dim * phys_dim, right_dim))?;

            // QR decomposition
            let (q, r) = qr_decomposition(&matrix)?;

            // Update current tensor with Q
            let new_shape = (left_dim, phys_dim, q.shape()[1]);
            self.tensors[i].data = q.into_shape(new_shape)?;
            self.tensors[i].right_dim = new_shape.2;

            // Absorb R into next tensor
            if i + 1 < self.num_qubits {
                let next = &mut self.tensors[i + 1];
                let next_matrix = next
                    .data
                    .view()
                    .into_shape((next.left_dim, 2 * next.right_dim))?;
                let new_matrix = r.dot(&next_matrix);
                next.data = new_matrix.into_shape((r.shape()[0], 2, next.right_dim))?;
                next.left_dim = r.shape()[0];
            }
        }
        Ok(())
    }

    /// Move orthogonality center one position to the right
    fn move_center_right(&mut self, position: usize) -> QuantRS2Result<()> {
        let tensor = &self.tensors[position];
        let (left_dim, phys_dim, right_dim) = (tensor.left_dim, 2, tensor.right_dim);

        // Reshape and QR decompose
        let matrix = tensor
            .data
            .view()
            .into_shape((left_dim * phys_dim, right_dim))?;
        let (q, r) = qr_decomposition(&matrix)?;

        // Update current tensor
        let q_cols = q.shape()[1];
        self.tensors[position].data = q.into_shape((left_dim, phys_dim, q_cols))?;
        self.tensors[position].right_dim = q_cols;

        // Update next tensor
        if position + 1 < self.num_qubits {
            let next = &mut self.tensors[position + 1];
            let next_matrix = next
                .data
                .view()
                .into_shape((next.left_dim, 2 * next.right_dim))?;
            let new_matrix = r.dot(&next_matrix);
            next.data = new_matrix.into_shape((r.shape()[0], 2, next.right_dim))?;
            next.left_dim = r.shape()[0];
        }

        Ok(())
    }

    /// Move orthogonality center one position to the left
    fn move_center_left(&mut self, position: usize) -> QuantRS2Result<()> {
        let tensor = &self.tensors[position];
        let (left_dim, phys_dim, right_dim) = (tensor.left_dim, 2, tensor.right_dim);

        // Reshape and QR decompose from right
        let matrix = tensor
            .data
            .view()
            .permuted_axes([2, 1, 0])
            .into_shape((right_dim * phys_dim, left_dim))?;
        let (q, r) = qr_decomposition(&matrix)?;

        // Update current tensor
        let q_cols = q.shape()[1];
        let q_reshaped = q.into_shape((right_dim, phys_dim, q_cols))?;
        self.tensors[position].data = q_reshaped.permuted_axes([2, 1, 0]);
        self.tensors[position].left_dim = q_cols;

        // Update previous tensor
        if position > 0 {
            let prev = &mut self.tensors[position - 1];
            let prev_matrix = prev
                .data
                .view()
                .into_shape((prev.left_dim * 2, prev.right_dim))?;
            let new_matrix = prev_matrix.dot(&r.t());
            prev.data = new_matrix.into_shape((prev.left_dim, 2, r.shape()[0]))?;
            prev.right_dim = r.shape()[0];
        }

        Ok(())
    }

    /// Apply single-qubit gate
    pub fn apply_single_qubit_gate(
        &mut self,
        gate: &dyn GateOp,
        qubit: usize,
    ) -> QuantRS2Result<()> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }

        // Get gate matrix
        let gate_matrix = gate.matrix()?;
        let gate_array = Array2::from_shape_vec((2, 2), gate_matrix)?;

        // Apply gate to tensor
        let tensor = &mut self.tensors[qubit];
        let mut new_data = Array3::zeros(tensor.data.dim());

        for left in 0..tensor.left_dim {
            for right in 0..tensor.right_dim {
                for i in 0..2 {
                    for j in 0..2 {
                        new_data[[left, i, right]] +=
                            gate_array[[i, j]] * tensor.data[[left, j, right]];
                    }
                }
            }
        }

        tensor.data = new_data;
        Ok(())
    }

    /// Apply two-qubit gate using SVD compression
    pub fn apply_two_qubit_gate(
        &mut self,
        gate: &dyn GateOp,
        qubit1: usize,
        qubit2: usize,
    ) -> QuantRS2Result<()> {
        // Ensure qubits are adjacent
        if (qubit1 as i32 - qubit2 as i32).abs() != 1 {
            return Err(QuantRS2Error::ComputationError(
                "MPS simulator requires adjacent qubits for two-qubit gates".to_string(),
            ));
        }

        let (left_qubit, right_qubit) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };

        // Move orthogonality center to left qubit
        self.move_orthogonality_center(left_qubit)?;

        // Get gate matrix
        let gate_matrix = gate.matrix()?;
        let gate_array = Array2::from_shape_vec((4, 4), gate_matrix)?;

        // Contract the two tensors
        let left_tensor = &self.tensors[left_qubit];
        let right_tensor = &self.tensors[right_qubit];

        let left_dim = left_tensor.left_dim;
        let right_dim = right_tensor.right_dim;

        // Combine tensors
        let mut combined = Array3::<Complex64>::zeros((left_dim, 4, right_dim));
        for l in 0..left_dim {
            for r in 0..right_dim {
                for i in 0..2 {
                    for j in 0..2 {
                        for k in 0..left_tensor.right_dim {
                            combined[[l, i * 2 + j, r]] +=
                                left_tensor.data[[l, i, k]] * right_tensor.data[[k, j, r]];
                        }
                    }
                }
            }
        }

        // Apply gate
        let mut gated = Array3::<Complex64>::zeros((left_dim, 4, right_dim));
        for l in 0..left_dim {
            for r in 0..right_dim {
                for out_idx in 0..4 {
                    for in_idx in 0..4 {
                        gated[[l, out_idx, r]] +=
                            gate_array[[out_idx, in_idx]] * combined[[l, in_idx, r]];
                    }
                }
            }
        }

        // Decompose back using SVD
        let matrix = gated.into_shape((left_dim * 2, 2 * right_dim))?;
        let (u, s, vt) = svd_decomposition(&matrix, self.max_bond_dim, self.truncation_threshold)?;

        // Update tensors
        let new_bond = s.len();
        self.tensors[left_qubit].data = u.into_shape((left_dim, 2, new_bond))?;
        self.tensors[left_qubit].right_dim = new_bond;

        // Convert s to complex diagonal matrix and multiply with vt
        let mut sv = Array2::<Complex64>::zeros((new_bond, vt.shape()[1]));
        for i in 0..new_bond {
            for j in 0..vt.shape()[1] {
                sv[[i, j]] = Complex64::new(s[i], 0.0) * vt[[i, j]];
            }
        }
        self.tensors[right_qubit].data = sv.t().to_owned().into_shape((new_bond, 2, right_dim))?;
        self.tensors[right_qubit].left_dim = new_bond;

        self.orthogonality_center = right_qubit as i32;

        Ok(())
    }

    /// Compute amplitude of a basis state
    pub fn get_amplitude(&self, bitstring: &[bool]) -> QuantRS2Result<Complex64> {
        if bitstring.len() != self.num_qubits {
            return Err(QuantRS2Error::ComputationError(format!(
                "Bitstring length {} doesn't match qubit count {}",
                bitstring.len(),
                self.num_qubits
            )));
        }

        // Contract from left to right
        let mut result = Array2::eye(1);

        for (i, &bit) in bitstring.iter().enumerate() {
            let tensor = &self.tensors[i];
            let idx = i32::from(bit);

            // Extract the matrix for this bit value
            let matrix = tensor.data.slice(s![.., idx, ..]);
            result = result.dot(&matrix);
        }

        Ok(result[[0, 0]])
    }

    /// Sample from the MPS
    #[must_use]
    pub fn sample(&self) -> Vec<bool> {
        use scirs2_core::random::prelude::*;
        let mut rng = thread_rng();
        let mut result = vec![false; self.num_qubits];
        let mut accumulated_matrix = Array2::eye(1);

        for (i, tensor) in self.tensors.iter().enumerate() {
            // Compute probabilities for this qubit
            let mut prob0 = Complex64::new(0.0, 0.0);
            let mut prob1 = Complex64::new(0.0, 0.0);

            // Probability of |0>
            let matrix0 = tensor.data.slice(s![.., 0, ..]);
            let temp0: Array2<Complex64> = accumulated_matrix.dot(&matrix0);

            // Contract with remaining tensors
            let mut right_contract = Array2::eye(temp0.shape()[1]);
            for j in (i + 1)..self.num_qubits {
                let sum_matrix = self.tensors[j].data.slice(s![.., 0, ..]).to_owned()
                    + self.tensors[j].data.slice(s![.., 1, ..]).to_owned();
                right_contract = right_contract.dot(&sum_matrix);
            }

            prob0 = temp0.dot(&right_contract)[[0, 0]];

            // Similar for |1>
            let matrix1 = tensor.data.slice(s![.., 1, ..]);
            let temp1: Array2<Complex64> = accumulated_matrix.dot(&matrix1);
            prob1 = temp1.dot(&right_contract)[[0, 0]];

            // Normalize and sample
            let total = prob0.norm_sqr() + prob1.norm_sqr();
            let threshold = prob0.norm_sqr() / total;

            if rng.random::<f64>() < threshold {
                result[i] = false;
                accumulated_matrix = temp0;
            } else {
                result[i] = true;
                accumulated_matrix = temp1;
            }
        }

        result
    }

    /// Contract the MPS into a dense state vector of `2^n` complex amplitudes.
    ///
    /// The amplitudes use the little-endian convention `amplitude[index]` where bit `q`
    /// of `index` is the computational-basis value of qubit `q`. Each amplitude is the
    /// genuine MPS contraction `A^{s_0} A^{s_1} ... A^{s_{n-1}}` evaluated as a chain of
    /// matrix products, so an entangled MPS produces the corresponding entangled
    /// amplitudes rather than a fabricated product state.
    fn to_statevector(&self) -> QuantRS2Result<Vec<Complex64>> {
        let dim = 1usize << self.num_qubits;
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); dim];

        for (index, amplitude) in amplitudes.iter_mut().enumerate() {
            // Contract the per-qubit slices selected by the bits of `index`.
            let mut accumulated: Array2<Complex64> = Array2::eye(1);
            for qubit in 0..self.num_qubits {
                let bit = (index >> qubit) & 1;
                let slice = self.tensors[qubit].data.slice(s![.., bit as i32, ..]);
                accumulated = accumulated.dot(&slice);
            }
            *amplitude = accumulated[[0, 0]];
        }

        Ok(amplitudes)
    }
}

/// QR decomposition helper
fn qr_decomposition(
    matrix: &ArrayView2<Complex64>,
) -> QuantRS2Result<(Array2<Complex64>, Array2<Complex64>)> {
    // Simple Gram-Schmidt QR decomposition
    let (m, n) = matrix.dim();
    let mut q = Array2::zeros((m, n.min(m)));
    let mut r = Array2::zeros((n.min(m), n));

    for j in 0..n.min(m) {
        let mut v = matrix.column(j).to_owned();

        // Orthogonalize against previous columns
        for i in 0..j {
            let proj = q.column(i).dot(&v);
            r[[i, j]] = proj;
            v -= &(proj * &q.column(i).to_owned());
        }

        let norm = (v.dot(&v)).sqrt();
        if norm.norm() > 1e-10 {
            r[[j, j]] = norm;
            q.column_mut(j).assign(&(v / norm));
        }
    }

    // Copy remaining columns of R
    if n > m {
        for j in m..n {
            for i in 0..m {
                r[[i, j]] = q.column(i).dot(&matrix.column(j));
            }
        }
    }

    Ok((q, r))
}

/// Full (untruncated) reduced complex SVD `A = U · diag(S) · Vt`.
///
/// Implements the one-sided Jacobi SVD algorithm directly on complex matrices. Jacobi
/// rotations are applied to pairs of columns of a working copy of `A` until all columns
/// are mutually orthogonal; the resulting column norms are the singular values, the
/// normalised columns form `U`, and the accumulated rotations form `V` (returned as
/// `Vt = V^H`). This method is numerically robust for all matrix sizes and converges to
/// machine precision, unlike eigendecomposition-of-`A^H A` approaches that can yield
/// non-orthonormal vectors or `NaN` singular values for ill-conditioned inputs.
///
/// Returns `(U, S, Vt)` with `U` of shape `(m, r)`, `S` of length `r`, `Vt` of shape
/// `(r, n)` and `r = min(m, n)`. Singular values are real, non-negative and sorted in
/// descending order.
pub(crate) fn complex_jacobi_svd(
    matrix: &Array2<Complex64>,
) -> QuantRS2Result<(Array2<Complex64>, Array1<f64>, Array2<Complex64>)> {
    let (m, n) = matrix.dim();
    if m == 0 || n == 0 {
        return Err(QuantRS2Error::ComputationError(
            "Cannot compute SVD of an empty matrix".to_string(),
        ));
    }

    // The one-sided Jacobi method orthogonalises the columns of the matrix with the larger
    // number of rows. When n > m we transpose, decompose, and swap U and V at the end so
    // the algorithm always works on a tall-or-square matrix.
    let transposed = n > m;
    let mut work = if transposed {
        // Work on A^H (shape n x m), then U_work plays the role of V and vice versa.
        let mut ah = Array2::<Complex64>::zeros((n, m));
        for i in 0..m {
            for j in 0..n {
                ah[[j, i]] = matrix[[i, j]].conj();
            }
        }
        ah
    } else {
        matrix.clone()
    };

    let rows = work.nrows();
    let cols = work.ncols();

    // Accumulate the right rotations into v (cols x cols), starting from the identity.
    let mut v = Array2::<Complex64>::eye(cols);

    let tolerance = 1e-14_f64;
    let max_sweeps = 60;

    for _sweep in 0..max_sweeps {
        let mut off_diagonal = 0.0_f64;

        for p in 0..cols {
            for q in (p + 1)..cols {
                // Compute the 2x2 Hermitian block of the column Gram matrix:
                //   alpha = <col_p, col_p>, beta = <col_q, col_q>, gamma = <col_p, col_q>.
                let mut alpha = 0.0_f64;
                let mut beta = 0.0_f64;
                let mut gamma = Complex64::new(0.0, 0.0);
                for i in 0..rows {
                    let cp = work[[i, p]];
                    let cq = work[[i, q]];
                    alpha += cp.norm_sqr();
                    beta += cq.norm_sqr();
                    gamma += cp.conj() * cq;
                }

                let gamma_abs = gamma.norm();
                off_diagonal = off_diagonal.max(gamma_abs);
                if gamma_abs <= tolerance * (alpha.sqrt() * beta.sqrt()).max(f64::MIN_POSITIVE) {
                    continue;
                }

                // Complex one-sided Jacobi rotation. Factor out the phase of gamma so the
                // remaining 2x2 problem is real-symmetric, then apply the standard Jacobi
                // angle. The rotation is unitary, preserving the decomposition.
                let phase = if gamma_abs > 0.0 {
                    gamma / Complex64::new(gamma_abs, 0.0)
                } else {
                    Complex64::new(1.0, 0.0)
                };

                let zeta = (beta - alpha) / (2.0 * gamma_abs);
                let sign = if zeta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (zeta.abs() + (zeta * zeta + 1.0).sqrt());
                let cosine = 1.0 / (t * t + 1.0).sqrt();
                let sine = cosine * t;

                let c = Complex64::new(cosine, 0.0);
                let s_pq = phase * Complex64::new(sine, 0.0);
                let s_pq_conj = s_pq.conj();

                // Rotate columns p and q of the working matrix.
                for i in 0..rows {
                    let cp = work[[i, p]];
                    let cq = work[[i, q]];
                    work[[i, p]] = c * cp - s_pq_conj * cq;
                    work[[i, q]] = s_pq * cp + c * cq;
                }
                // Apply the same rotation to the accumulated right-singular-vector matrix.
                for i in 0..cols {
                    let vp = v[[i, p]];
                    let vq = v[[i, q]];
                    v[[i, p]] = c * vp - s_pq_conj * vq;
                    v[[i, q]] = s_pq * vp + c * vq;
                }
            }
        }

        if off_diagonal <= tolerance {
            break;
        }
    }

    // Column norms of the orthogonalised working matrix are the singular values; the
    // normalised columns are the left singular vectors.
    let rank = rows.min(cols);
    let mut singular = Vec::with_capacity(cols);
    let mut u_full = Array2::<Complex64>::zeros((rows, cols));
    for j in 0..cols {
        let mut norm_sq = 0.0_f64;
        for i in 0..rows {
            norm_sq += work[[i, j]].norm_sqr();
        }
        let norm = norm_sq.sqrt();
        singular.push(norm);
        if norm > tolerance {
            for i in 0..rows {
                u_full[[i, j]] = work[[i, j]] / Complex64::new(norm, 0.0);
            }
        }
    }

    // Sort singular values (and corresponding U, V columns) in descending order.
    let mut order: Vec<usize> = (0..cols).collect();
    order.sort_by(|&a, &b| {
        singular[b]
            .partial_cmp(&singular[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut s = Array1::<f64>::zeros(rank);
    let mut u = Array2::<Complex64>::zeros((rows, rank));
    let mut v_sorted = Array2::<Complex64>::zeros((cols, rank));
    for (new_idx, &old_idx) in order.iter().take(rank).enumerate() {
        s[new_idx] = singular[old_idx];
        for i in 0..rows {
            u[[i, new_idx]] = u_full[[i, old_idx]];
        }
        for i in 0..cols {
            v_sorted[[i, new_idx]] = v[[i, old_idx]];
        }
    }

    // For columns with (near-)zero singular value the corresponding U column is zero; fill
    // it with a vector orthonormal to the others so U has orthonormal columns. This keeps
    // the factorisation well-formed for rank-deficient inputs.
    for j in 0..rank {
        if s[j] <= tolerance {
            let mut candidate = Array1::<Complex64>::zeros(rows);
            candidate[j % rows] = Complex64::new(1.0, 0.0);
            for prev in 0..rank {
                if prev == j {
                    continue;
                }
                let mut proj = Complex64::new(0.0, 0.0);
                for i in 0..rows {
                    proj += u[[i, prev]].conj() * candidate[i];
                }
                for i in 0..rows {
                    candidate[i] -= proj * u[[i, prev]];
                }
            }
            let mut norm_sq = 0.0_f64;
            for i in 0..rows {
                norm_sq += candidate[i].norm_sqr();
            }
            let norm = norm_sq.sqrt();
            if norm > tolerance {
                for i in 0..rows {
                    u[[i, j]] = candidate[i] / Complex64::new(norm, 0.0);
                }
            }
        }
    }

    if transposed {
        // We decomposed A^H = U_work · S · V_work^H, hence A = V_work · S · U_work^H.
        // So the true U is v_sorted and the true Vt is u^H.
        let true_u = v_sorted;
        let mut vt = Array2::<Complex64>::zeros((rank, n));
        for i in 0..rank {
            for j in 0..n {
                vt[[i, j]] = u[[j, i]].conj();
            }
        }
        Ok((true_u, s, vt))
    } else {
        // A = U · S · V^H, so Vt = V^H.
        let mut vt = Array2::<Complex64>::zeros((rank, n));
        for i in 0..rank {
            for j in 0..n {
                vt[[i, j]] = v_sorted[[j, i]].conj();
            }
        }
        Ok((u, s, vt))
    }
}

/// SVD decomposition with bond-dimension truncation.
///
/// Computes a real, complex-valued singular value decomposition `A = U · diag(S) · Vt`
/// via [`complex_jacobi_svd`] and truncates the bond to keep the largest singular values
/// (those above `threshold`, capped at `max_bond`). This decomposition controls MPS
/// entanglement representation: returning anything other than the genuine factorization
/// silently corrupts every downstream amplitude.
///
/// Returns `(U, S, Vt)` where `U` has shape `(m, k)`, `S` is the vector of `k` retained
/// singular values (real, non-negative, descending) and `Vt` has shape `(k, n)` such that
/// `U · diag(S) · Vt` reconstructs `A` within numerical tolerance.
fn svd_decomposition(
    matrix: &Array2<Complex64>,
    max_bond: usize,
    threshold: f64,
) -> QuantRS2Result<(Array2<Complex64>, Array1<f64>, Array2<Complex64>)> {
    let (full_u, full_s, full_vt) = complex_jacobi_svd(matrix)?;
    let full_rank = full_s.len();

    // Determine how many singular values to keep: drop those at or below the truncation
    // threshold, then cap at the maximum bond dimension. Always keep at least one so the
    // resulting tensors stay well-formed even for (near-)zero states.
    let mut kept = full_s.iter().filter(|&&value| value > threshold).count();
    kept = kept.min(max_bond).min(full_rank);
    if kept == 0 {
        kept = full_rank.max(1);
    }

    let u = full_u.slice(s![.., ..kept]).to_owned();
    let truncated_s = full_s.slice(s![..kept]).to_owned();
    let vt = full_vt.slice(s![..kept, ..]).to_owned();

    Ok((u, truncated_s, vt))
}

/// MPS quantum simulator
pub struct MPSSimulator {
    /// Maximum bond dimension
    max_bond_dimension: usize,
    /// SVD truncation threshold
    truncation_threshold: f64,
}

impl MPSSimulator {
    /// Create a new MPS simulator
    #[must_use]
    pub const fn new(max_bond_dimension: usize) -> Self {
        Self {
            max_bond_dimension,
            truncation_threshold: 1e-10,
        }
    }

    /// Set the truncation threshold
    pub const fn set_truncation_threshold(&mut self, threshold: f64) {
        self.truncation_threshold = threshold;
    }
}

impl<const N: usize> Simulator<N> for MPSSimulator {
    fn run(&self, circuit: &Circuit<N>) -> QuantRS2Result<Register<N>> {
        // Create initial MPS state in |0...0>.
        let mut mps = MPS::new(N, self.max_bond_dimension);
        mps.set_truncation_threshold(self.truncation_threshold);

        // Apply each circuit gate to the MPS. Single-qubit gates act on the local tensor;
        // two-qubit gates contract, apply, and re-split via the real SVD truncation. Gates
        // acting on more than two qubits or on non-adjacent qubits are not representable by
        // this nearest-neighbour MPS, so we surface an honest error instead of silently
        // skipping them (which would corrupt the resulting state).
        for gate in circuit.gates() {
            let qubits = gate.qubits();
            match qubits.as_slice() {
                [target] => {
                    mps.apply_single_qubit_gate(gate.as_ref(), target.id() as usize)?;
                }
                [first, second] => {
                    mps.apply_two_qubit_gate(
                        gate.as_ref(),
                        first.id() as usize,
                        second.id() as usize,
                    )?;
                }
                _ => {
                    return Err(QuantRS2Error::UnsupportedOperation(format!(
                        "MPS simulator supports only one- and two-qubit gates, but '{}' acts on {} qubits",
                        gate.name(),
                        qubits.len()
                    )));
                }
            }
        }

        // Contract the MPS into a dense state vector and build the register from it.
        let amplitudes = mps.to_statevector()?;
        Register::<N>::with_amplitudes(amplitudes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::single::Hadamard;

    #[test]
    fn test_mps_creation() {
        let mps = MPS::new(4, 10);
        assert_eq!(mps.num_qubits, 4);
        assert_eq!(mps.tensors.len(), 4);
    }

    #[test]
    fn test_single_qubit_gate() {
        let mut mps = MPS::new(1, 10);
        let h = Hadamard {
            target: QubitId::new(0),
        };

        mps.apply_single_qubit_gate(&h, 0)
            .expect("Failed to apply single qubit gate");

        // Check amplitudes
        let amp0 = mps
            .get_amplitude(&[false])
            .expect("Failed to get amplitude for |0>");
        let amp1 = mps
            .get_amplitude(&[true])
            .expect("Failed to get amplitude for |1>");

        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((amp0.re - expected).abs() < 1e-10);
        assert!((amp1.re - expected).abs() < 1e-10);
    }

    #[test]
    fn test_orthogonality_center() {
        let mut mps = MPS::new(5, 10);

        mps.move_orthogonality_center(2)
            .expect("Failed to move orthogonality center to 2");
        assert_eq!(mps.orthogonality_center, 2);

        mps.move_orthogonality_center(4)
            .expect("Failed to move orthogonality center to 4");
        assert_eq!(mps.orthogonality_center, 4);

        mps.move_orthogonality_center(0)
            .expect("Failed to move orthogonality center to 0");
        assert_eq!(mps.orthogonality_center, 0);
    }

    #[test]
    fn test_svd_decomposition_reconstructs_matrix() {
        // A non-trivial, non-Hermitian complex matrix whose SVD is clearly not identity.
        let matrix = Array2::from_shape_vec(
            (3, 3),
            vec![
                Complex64::new(1.0, 0.5),
                Complex64::new(2.0, -1.0),
                Complex64::new(0.0, 3.0),
                Complex64::new(-1.0, 2.0),
                Complex64::new(0.5, 0.5),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(-3.0, 0.0),
                Complex64::new(0.0, -2.0),
            ],
        )
        .expect("failed to build test matrix");

        let (u, s, vt) =
            svd_decomposition(&matrix, 16, 1e-14).expect("real SVD decomposition should succeed");

        // Reconstruct U * diag(S) * Vt and compare to the original within tight tolerance.
        let k = s.len();
        let mut reconstructed = Array2::<Complex64>::zeros(matrix.dim());
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                let mut acc = Complex64::new(0.0, 0.0);
                for r in 0..k {
                    acc += u[[i, r]] * Complex64::new(s[r], 0.0) * vt[[r, j]];
                }
                reconstructed[[i, j]] = acc;
            }
        }

        let mut max_err = 0.0_f64;
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                max_err = max_err.max((reconstructed[[i, j]] - matrix[[i, j]]).norm());
            }
        }
        assert!(
            max_err < 1e-8,
            "SVD reconstruction error too large: {max_err}"
        );

        // Singular values must be non-negative and sorted descending.
        for r in 0..k {
            assert!(s[r] >= -1e-12, "singular value {r} is negative: {}", s[r]);
            if r > 0 {
                assert!(s[r - 1] + 1e-12 >= s[r], "singular values not descending");
            }
        }

        // U must have orthonormal columns (genuine left singular vectors), proving this is
        // a real SVD rather than an arbitrary factorization.
        for a in 0..k {
            for b in 0..k {
                let mut inner = Complex64::new(0.0, 0.0);
                for i in 0..u.nrows() {
                    inner += u[[i, a]].conj() * u[[i, b]];
                }
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (inner.re - expected).abs() < 1e-8 && inner.im.abs() < 1e-8,
                    "U columns not orthonormal at ({a},{b}): {inner:?}"
                );
            }
        }

        // The decomposition must NOT be the fabricated identity-like result: at least one
        // singular value differs from 1 and U is not the identity.
        assert!(
            s.iter().any(|&value| (value - 1.0).abs() > 1e-6),
            "singular values are all ~1 (identity fabrication not fixed)"
        );
        let identity = Array2::<Complex64>::eye(u.nrows());
        let mut differs_from_identity = false;
        for i in 0..u.nrows() {
            for j in 0..k {
                if (u[[i, j]] - identity[[i, j]]).norm() > 1e-6 {
                    differs_from_identity = true;
                }
            }
        }
        assert!(
            differs_from_identity,
            "U equals identity (identity fabrication not fixed)"
        );
    }

    #[test]
    fn test_to_statevector_bell_state() {
        // Build a Bell state (|00> + |11>)/sqrt(2) via H on qubit 0 then CNOT(0, 1).
        let mut mps = MPS::new(2, 16);
        let h = Hadamard {
            target: QubitId::new(0),
        };
        mps.apply_single_qubit_gate(&h, 0)
            .expect("failed to apply Hadamard");

        let cnot = quantrs2_core::gate::multi::CNOT {
            control: QubitId::new(0),
            target: QubitId::new(1),
        };
        mps.apply_two_qubit_gate(&cnot, 0, 1)
            .expect("failed to apply CNOT");

        let state = mps
            .to_statevector()
            .expect("contraction to state vector should succeed");

        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert!((state[0].re - inv_sqrt2).abs() < 1e-10, "amp(|00>) wrong");
        assert!(state[1].norm() < 1e-10, "amp(|01>) should vanish");
        assert!(state[2].norm() < 1e-10, "amp(|10>) should vanish");
        assert!((state[3].re - inv_sqrt2).abs() < 1e-10, "amp(|11>) wrong");

        // Honest check: the contracted state is genuinely entangled, NOT the |00> placeholder.
        let zero_state = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let is_zero_state = state
            .iter()
            .zip(zero_state.iter())
            .all(|(a, b)| (a - b).norm() < 1e-9);
        assert!(
            !is_zero_state,
            "contraction fabricated the |00> placeholder"
        );
    }
}
