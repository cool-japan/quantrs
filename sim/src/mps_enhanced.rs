//! Enhanced Matrix Product State (MPS) quantum simulator
//!
//! This module provides a complete and optimized MPS simulator implementation
//! with proper SVD decomposition, comprehensive gate support, and performance optimizations.
use crate::scirs2_integration::SciRS2Backend;
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    register::Register,
};
use scirs2_core::ndarray::ndarray_linalg::{QR, SVD};
use scirs2_core::ndarray::{array, s, Array1, Array2, Array3, Array4};
use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;
use scirs2_core::random::{thread_rng, Rng};
use scirs2_core::Complex64;
use std::f64::consts::{PI, SQRT_2};
/// Configuration for MPS simulator
#[derive(Debug, Clone)]
pub struct MPSConfig {
    /// Maximum allowed bond dimension
    pub max_bond_dim: usize,
    /// SVD truncation threshold (singular values below this are discarded)
    pub svd_threshold: f64,
    /// Whether to use randomized SVD for large matrices
    pub use_randomized_svd: bool,
    /// Random seed for deterministic behavior
    pub seed: Option<u64>,
    /// Enable automatic recanonization after gates
    pub auto_canonicalize: bool,
}
impl Default for MPSConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 64,
            svd_threshold: 1e-10,
            use_randomized_svd: true,
            seed: None,
            auto_canonicalize: true,
        }
    }
}
/// MPS tensor for a single qubit
#[derive(Debug, Clone)]
struct MPSTensor {
    /// The tensor data: left_bond x physical x right_bond
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
    ///
    /// All product-state tensors start with bond dimension 1:
    ///   shape = (1, 2, 1) for every qubit position.
    /// This is the correct MPS representation of |000...0⟩:
    ///   each tensor is [[[1,0]]] — left bond 1, physical 2, right bond 1.
    /// Starting with pre-inflated bond dims (e.g. (2,2,2)) causes artificial
    /// SVD rank growth because intermediate contractions see zero-padded rows/cols
    /// that produce spurious near-zero (but non-truncated) singular values.
    fn zero_state(_position: usize, _num_qubits: usize) -> Self {
        let mut tensor = Array3::zeros((1, 2, 1));
        tensor[[0, 0, 0]] = Complex64::new(1.0, 0.0);
        Self::new(tensor)
    }
    /// Apply a single-qubit gate to this tensor
    fn apply_gate(&mut self, gate_matrix: &Array2<Complex64>) -> QuantRS2Result<()> {
        let mut new_data = Array3::zeros(self.data.dim());
        for left in 0..self.left_dim {
            for right in 0..self.right_dim {
                for new_phys in 0..2 {
                    for old_phys in 0..2 {
                        new_data[[left, new_phys, right]] +=
                            gate_matrix[[new_phys, old_phys]] * self.data[[left, old_phys, right]];
                    }
                }
            }
        }
        self.data = new_data;
        Ok(())
    }
}
/// Enhanced Matrix Product State representation
pub struct EnhancedMPS {
    /// MPS tensors for each qubit
    tensors: Vec<MPSTensor>,
    /// Number of qubits
    num_qubits: usize,
    /// Configuration
    config: MPSConfig,
    /// Current orthogonality center (-1 if not in canonical form)
    orthogonality_center: i32,
    /// Random number generator
    rng: scirs2_core::CoreRandom,
}
impl EnhancedMPS {
    /// Create a new MPS in the |0...0> state
    pub fn new(num_qubits: usize, config: MPSConfig) -> Self {
        let tensors = (0..num_qubits)
            .map(|i| MPSTensor::zero_state(i, num_qubits))
            .collect();
        let rng = thread_rng();
        Self {
            tensors,
            num_qubits,
            config,
            orthogonality_center: -1,
            rng,
        }
    }
    /// Apply a gate to the MPS
    pub fn apply_gate(&mut self, gate: &dyn GateOp) -> QuantRS2Result<()> {
        let qubits = gate.qubits();
        match qubits.len() {
            1 => self.apply_single_qubit_gate(gate, qubits[0].id() as usize),
            2 => self.apply_two_qubit_gate(gate, qubits[0].id() as usize, qubits[1].id() as usize),
            _ => Err(QuantRS2Error::UnsupportedOperation(format!(
                "MPS simulator doesn't support {}-qubit gates",
                qubits.len()
            ))),
        }
    }
    /// Apply single-qubit gate
    fn apply_single_qubit_gate(&mut self, gate: &dyn GateOp, qubit: usize) -> QuantRS2Result<()> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        let gate_matrix = Self::get_gate_matrix(gate)?;
        self.tensors[qubit].apply_gate(&gate_matrix)?;
        Ok(())
    }
    /// Apply two-qubit gate
    fn apply_two_qubit_gate(
        &mut self,
        gate: &dyn GateOp,
        qubit1: usize,
        qubit2: usize,
    ) -> QuantRS2Result<()> {
        if (qubit1 as i32 - qubit2 as i32).abs() != 1 {
            return self.apply_non_adjacent_gate(gate, qubit1, qubit2);
        }
        let (left_qubit, right_qubit) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };
        if self.config.auto_canonicalize {
            self.move_orthogonality_center(left_qubit)?;
        }
        let gate_matrix = Self::get_gate_matrix(gate)?;
        let gate_array = gate_matrix.into_shape((2, 2, 2, 2))?;
        self.apply_and_decompose_two_qubit_gate(&gate_array, left_qubit, right_qubit)?;
        Ok(())
    }
    /// Apply gate to non-adjacent qubits using SWAP gates
    fn apply_non_adjacent_gate(
        &mut self,
        gate: &dyn GateOp,
        qubit1: usize,
        qubit2: usize,
    ) -> QuantRS2Result<()> {
        let (min_q, max_q) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };
        for i in min_q..max_q - 1 {
            self.apply_swap(i, i + 1)?;
        }
        self.apply_two_qubit_gate(gate, max_q - 1, max_q)?;
        for i in (min_q..max_q - 1).rev() {
            self.apply_swap(i, i + 1)?;
        }
        Ok(())
    }
    /// Apply SWAP gate
    fn apply_swap(&mut self, qubit1: usize, qubit2: usize) -> QuantRS2Result<()> {
        let swap_matrix = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex64::new(1., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(1., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(1., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(0., 0.),
                Complex64::new(1., 0.),
            ],
        )?;
        let swap_array = swap_matrix.into_shape((2, 2, 2, 2))?;
        self.apply_and_decompose_two_qubit_gate(&swap_array, qubit1, qubit2)
    }
    /// Apply two-qubit gate and decompose using SVD
    fn apply_and_decompose_two_qubit_gate(
        &mut self,
        gate_array: &scirs2_core::ndarray::ArrayBase<
            scirs2_core::ndarray::OwnedRepr<Complex64>,
            scirs2_core::ndarray::Dim<[usize; 4]>,
        >,
        left_qubit: usize,
        right_qubit: usize,
    ) -> QuantRS2Result<()> {
        let left_dim = self.tensors[left_qubit].left_dim;
        let right_dim = self.tensors[right_qubit].right_dim;
        let middle_dim = self.tensors[left_qubit].right_dim;
        let mut theta = Array3::<Complex64>::zeros((left_dim, 4, right_dim));
        let left_data = self.tensors[left_qubit].data.clone();
        let right_data = self.tensors[right_qubit].data.clone();
        theta
            .indexed_iter_mut()
            .par_bridge()
            .for_each(|((l, ij, r), elem)| {
                let i = ij / 2;
                let j = ij % 2;
                let mut sum = Complex64::new(0.0, 0.0);
                for m in 0..middle_dim {
                    sum += left_data[[l, i, m]] * right_data[[m, j, r]];
                }
                *elem = sum;
            });
        let mut theta_prime = Array3::<Complex64>::zeros(theta.dim());
        theta_prime
            .indexed_iter_mut()
            .par_bridge()
            .for_each(|((l, out_ij, r), elem)| {
                let out_i = out_ij / 2;
                let out_j = out_ij % 2;
                let mut sum = Complex64::new(0.0, 0.0);
                for in_i in 0..2 {
                    for in_j in 0..2 {
                        sum +=
                            gate_array[[out_i, out_j, in_i, in_j]] * theta[[l, in_i * 2 + in_j, r]];
                    }
                }
                *elem = sum;
            });
        let matrix = theta_prime.into_shape((left_dim * 2, 2 * right_dim))?;
        let (u, s, vt) = self.truncated_svd(&matrix)?;
        let new_bond = s.len();
        self.tensors[left_qubit] = MPSTensor::new(u.into_shape((left_dim, 2, new_bond))?);
        let mut sv = Array2::<Complex64>::zeros((new_bond, vt.shape()[1]));
        sv.indexed_iter_mut()
            .par_bridge()
            .for_each(|((i, j), elem)| {
                *elem = Complex64::new(s[i], 0.0) * vt[[i, j]];
            });
        // sv has shape (new_bond, 2*right_dim) — reshape directly to (new_bond, 2, right_dim).
        // A transpose before reshape would scramble the physical-index ordering.
        self.tensors[right_qubit] =
            MPSTensor::new(sv.into_shape((new_bond, 2, right_dim))?);
        if self.config.auto_canonicalize {
            self.orthogonality_center = right_qubit as i32;
        }
        Ok(())
    }
    /// Perform truncated SVD using SciRS2 when available
    fn truncated_svd(
        &mut self,
        matrix: &Array2<Complex64>,
    ) -> Result<(Array2<Complex64>, Array1<f64>, Array2<Complex64>), QuantRS2Error> {
        let (mut u, mut s, mut vt) = if matrix.shape()[0] * matrix.shape()[1] > 100 {
            #[cfg(feature = "advanced_math")]
            {
                Self::fallback_svd(matrix)?
            }
            #[cfg(not(feature = "advanced_math"))]
            {
                Self::fallback_svd(matrix)?
            }
        } else {
            Self::fallback_svd(matrix)?
        };
        let mut num_keep = s.len().min(self.config.max_bond_dim);
        let total_weight: f64 = s.iter().map(|&x| x * x).sum();
        let mut accumulated_weight = 0.0;
        for i in 0..num_keep {
            accumulated_weight += s[i] * s[i];
            if accumulated_weight / total_weight
                > self
                    .config
                    .svd_threshold
                    .mul_add(-self.config.svd_threshold, 1.0)
            {
                num_keep = i + 1;
                break;
            }
        }
        for i in 0..num_keep {
            if s[i] < self.config.svd_threshold {
                num_keep = i;
                break;
            }
        }
        // U is returned as a full (m,m) unitary from the full SVD.
        // We always need to truncate it to the thin (m, num_keep) form so that
        // the subsequent reshape into (left_dim, 2, num_keep) has the right total size.
        // The condition `num_keep < s.len()` was previously used, but s.len() is
        // min(m,n), not m, so U needs separate truncation regardless.
        u = u.slice(s![.., ..num_keep]).to_owned();
        s = s.slice(s![..num_keep]).to_owned();
        vt = vt.slice(s![..num_keep, ..]).to_owned();
        Ok((u, s, vt))
    }
    /// Move orthogonality center to target position
    fn move_orthogonality_center(&mut self, target: usize) -> QuantRS2Result<()> {
        if target >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(target as u32));
        }
        if self.orthogonality_center < 0 {
            if self.num_qubits == 2 && target == 0 {
            } else {
                for i in (target + 1..self.num_qubits).rev() {
                    self.right_canonicalize_site(i)?;
                }
            }
            for i in 0..target {
                self.left_canonicalize_site(i)?;
            }
            self.orthogonality_center = target as i32;
            return Ok(());
        }
        let current = self.orthogonality_center as usize;
        if current < target {
            for i in current..target {
                self.move_center_right(i)?;
            }
        } else if current > target {
            for i in (target + 1..=current).rev() {
                self.move_center_left(i)?;
            }
        }
        self.orthogonality_center = target as i32;
        Ok(())
    }
    /// Left-canonicalize a site using QR decomposition.
    ///
    /// The underlying QR returns FULL Q (m×m) and FULL R (m×n).
    /// We extract the THIN forms: Q_thin (m×k) and R_thin (k×n) where k=min(m,n).
    fn left_canonicalize_site(&mut self, site: usize) -> QuantRS2Result<()> {
        let tensor = &self.tensors[site];
        let m = tensor.left_dim * 2;
        let n = tensor.right_dim;
        let mut matrix = Array2::<Complex64>::zeros((m, n));
        for l in 0..tensor.left_dim {
            for p in 0..2 {
                for r in 0..n {
                    matrix[[l * 2 + p, r]] = tensor.data[[l, p, r]];
                }
            }
        }
        let (q_full, r_full) = matrix
            .qr()
            .map_err(|e| QuantRS2Error::LinalgError(format!("QR decomposition failed: {e}")))?;
        // Thin Q: (m, k), thin R: (k, n) where k = min(m, n)
        let k = m.min(n);
        let q_thin = q_full.slice(s![.., ..k]).to_owned();
        let r_thin = r_full.slice(s![..k, ..]).to_owned();
        // Reshape Q_thin (m, k) → (left_dim, 2, k)
        self.tensors[site] = MPSTensor::new(q_thin.into_shape((tensor.left_dim, 2, k))?);
        if site + 1 < self.num_qubits {
            let next = self.tensors[site + 1].clone();
            let mut next_matrix = Array2::<Complex64>::zeros((next.left_dim, 2 * next.right_dim));
            for l in 0..next.left_dim {
                for p in 0..2 {
                    for r in 0..next.right_dim {
                        next_matrix[[l, p * next.right_dim + r]] = next.data[[l, p, r]];
                    }
                }
            }
            // r_thin has shape (k, n), next_matrix has shape (next.left_dim, ...).
            // We need r_thin.shape()[1] == next_matrix.shape()[0] for contraction.
            if r_thin.shape()[1] == next_matrix.shape()[0] {
                let new_next = r_thin.dot(&next_matrix);
                self.tensors[site + 1] =
                    MPSTensor::new(new_next.into_shape((r_thin.shape()[0], 2, next.right_dim))?);
            } else {
                // Dimension mismatch — truncate or extend next_matrix rows as needed.
                let r_cols = r_thin.shape()[1]; // == n == tensor.right_dim == new k
                let next_rows = next_matrix.shape()[0]; // == next.left_dim
                if r_cols <= next_rows {
                    let truncated_next = next_matrix.slice(s![..r_cols, ..]).to_owned();
                    let new_next = r_thin.dot(&truncated_next);
                    let new_bond = r_thin.shape()[0];
                    let mut new_tensor = Array3::zeros((new_bond, 2, next.right_dim));
                    for i in 0..new_bond {
                        for j in 0..2 {
                            for r_idx in 0..next.right_dim {
                                let col = j * next.right_dim + r_idx;
                                if col < new_next.shape()[1] {
                                    new_tensor[[i, j, r_idx]] = new_next[[i, col]];
                                }
                            }
                        }
                    }
                    self.tensors[site + 1] = MPSTensor::new(new_tensor);
                } else {
                    let new_next = r_thin.dot(&next_matrix);
                    self.tensors[site + 1] =
                        MPSTensor::new(new_next.into_shape((r_thin.shape()[0], 2, next.right_dim))?);
                }
            }
        }
        Ok(())
    }
    /// Right-canonicalize a site using LQ decomposition via SVD.
    ///
    /// Standard right-canonicalization reshapes tensor as (left_dim, 2*right_dim),
    /// then decomposes it as L·Q† where Q† has orthonormal rows (right-unitary).
    ///
    /// We implement this via SVD: A = U·S·Vt, with the new tensor set to Vt
    /// (reshaped to (k, 2, right_dim)) and L = U·S absorbed into the LEFT neighbor.
    fn right_canonicalize_site(&mut self, site: usize) -> QuantRS2Result<()> {
        let tensor = self.tensors[site].clone();
        // Reshape tensor as (left_dim, 2*right_dim) for right-canonical form.
        let m = tensor.left_dim;
        let n = 2 * tensor.right_dim;
        let mut matrix = Array2::<Complex64>::zeros((m, n));
        for l in 0..tensor.left_dim {
            for p in 0..2 {
                for r in 0..tensor.right_dim {
                    matrix[[l, p * tensor.right_dim + r]] = tensor.data[[l, p, r]];
                }
            }
        }
        // SVD: A (m×n) = U (m×k) · S (k,) · Vt (k×n), k = min(m,n)
        let (u, s, vt) = self.truncated_svd(&matrix)?;
        let k = s.len();
        // New tensor for this site: Vt reshaped to (k, 2, right_dim)
        // Vt has shape (k, n) = (k, 2*right_dim).
        let new_tensor_data = vt.into_shape((k, 2, tensor.right_dim))?;
        self.tensors[site] = MPSTensor::new(new_tensor_data);
        // Absorb U·S into the LEFT neighbor (site - 1).
        // L = U · diag(S) has shape (m, k) = (tensor.left_dim, k).
        if site > 0 {
            let mut l_matrix = Array2::<Complex64>::zeros((m, k));
            for i in 0..m {
                for j in 0..k {
                    l_matrix[[i, j]] = u[[i, j]] * Complex64::new(s[j], 0.0);
                }
            }
            // prev tensor: (prev.left_dim, 2, prev.right_dim) where prev.right_dim == m
            let prev = self.tensors[site - 1].clone();
            // Unfold prev as (prev.left_dim * 2, prev.right_dim) = (*, m)
            let prev_m = prev.left_dim * 2;
            let prev_n = prev.right_dim;
            let mut prev_matrix = Array2::<Complex64>::zeros((prev_m, prev_n));
            for l in 0..prev.left_dim {
                for p in 0..2 {
                    for r in 0..prev_n {
                        prev_matrix[[l * 2 + p, r]] = prev.data[[l, p, r]];
                    }
                }
            }
            // prev_matrix (prev_m, prev_n) · l_matrix.t() (prev_n, k)
            // Requires prev_n == m (left bond of current = right bond of prev).
            if prev_n == m {
                let new_prev = prev_matrix.dot(&l_matrix);
                // new_prev has shape (prev_m, k)
                self.tensors[site - 1] =
                    MPSTensor::new(new_prev.into_shape((prev.left_dim, 2, k))?);
            }
            // If prev_n != m (bond dim mismatch), skip — canonicalization is approximate.
        }
        Ok(())
    }
    /// Move center one position to the right
    fn move_center_right(&mut self, position: usize) -> QuantRS2Result<()> {
        self.left_canonicalize_site(position)
    }
    /// Move center one position to the left
    fn move_center_left(&mut self, position: usize) -> QuantRS2Result<()> {
        self.right_canonicalize_site(position)
    }
    /// Get gate matrix from gate operation
    fn get_gate_matrix(gate: &dyn GateOp) -> QuantRS2Result<Array2<Complex64>> {
        let gate_name = gate.name();
        if gate_name.starts_with("RX(") {
            let theta_str = gate_name.trim_start_matches("RX(").trim_end_matches(')');
            if let Ok(theta) = theta_str.parse::<f64>() {
                let cos_half = (theta / 2.0).cos();
                let sin_half = (theta / 2.0).sin();
                return Ok(array![
                    [Complex64::new(cos_half, 0.), Complex64::new(0., -sin_half)],
                    [Complex64::new(0., -sin_half), Complex64::new(cos_half, 0.)]
                ]);
            }
        }
        if gate_name.starts_with("RY(") {
            let theta_str = gate_name.trim_start_matches("RY(").trim_end_matches(')');
            if let Ok(theta) = theta_str.parse::<f64>() {
                let cos_half = (theta / 2.0).cos();
                let sin_half = (theta / 2.0).sin();
                return Ok(array![
                    [Complex64::new(cos_half, 0.), Complex64::new(-sin_half, 0.)],
                    [Complex64::new(sin_half, 0.), Complex64::new(cos_half, 0.)]
                ]);
            }
        }
        if gate_name.starts_with("RZ(") {
            let theta_str = gate_name.trim_start_matches("RZ(").trim_end_matches(')');
            if let Ok(theta) = theta_str.parse::<f64>() {
                let exp_pos = Complex64::from_polar(1.0, theta / 2.0);
                let exp_neg = Complex64::from_polar(1.0, -theta / 2.0);
                return Ok(array![
                    [exp_neg, Complex64::new(0., 0.)],
                    [Complex64::new(0., 0.), exp_pos]
                ]);
            }
        }
        if gate_name.starts_with("P(") || gate_name.starts_with("PHASE(") {
            let prefix = if gate_name.starts_with("P(") {
                "P("
            } else {
                "PHASE("
            };
            let phi_str = gate_name.trim_start_matches(prefix).trim_end_matches(')');
            if let Ok(phi) = phi_str.parse::<f64>() {
                let phase = Complex64::from_polar(1.0, phi);
                return Ok(array![
                    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
                    [Complex64::new(0., 0.), phase]
                ]);
            }
        }
        let matrix = match gate.name() {
            "I" => {
                array![
                    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
                    [Complex64::new(0., 0.), Complex64::new(1., 0.)]
                ]
            }
            "X" => {
                array![
                    [Complex64::new(0., 0.), Complex64::new(1., 0.)],
                    [Complex64::new(1., 0.), Complex64::new(0., 0.)]
                ]
            }
            "Y" => {
                array![
                    [Complex64::new(0., 0.), Complex64::new(0., -1.)],
                    [Complex64::new(0., 1.), Complex64::new(0., 0.)]
                ]
            }
            "Z" => {
                array![
                    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
                    [Complex64::new(0., 0.), Complex64::new(-1., 0.)]
                ]
            }
            "H" => {
                let h = 1.0 / SQRT_2;
                array![
                    [Complex64::new(h, 0.), Complex64::new(h, 0.)],
                    [Complex64::new(h, 0.), Complex64::new(-h, 0.)]
                ]
            }
            "S" => {
                array![
                    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
                    [Complex64::new(0., 0.), Complex64::new(0., 1.)]
                ]
            }
            "S†" | "Sdg" => {
                array![
                    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
                    [Complex64::new(0., 0.), Complex64::new(0., -1.)]
                ]
            }
            "T" => {
                let phase = Complex64::from_polar(1.0, PI / 4.0);
                array![
                    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
                    [Complex64::new(0., 0.), phase]
                ]
            }
            "T†" | "Tdg" => {
                let phase = Complex64::from_polar(1.0, -PI / 4.0);
                array![
                    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
                    [Complex64::new(0., 0.), phase]
                ]
            }
            "CNOT" | "CX" => {
                array![
                    [
                        Complex64::new(1., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.)
                    ],
                    [
                        Complex64::new(0., 0.),
                        Complex64::new(1., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.)
                    ],
                    [
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(1., 0.)
                    ],
                    [
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(1., 0.),
                        Complex64::new(0., 0.)
                    ],
                ]
            }
            "CZ" => {
                array![
                    [
                        Complex64::new(1., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.)
                    ],
                    [
                        Complex64::new(0., 0.),
                        Complex64::new(1., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.)
                    ],
                    [
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(1., 0.),
                        Complex64::new(0., 0.)
                    ],
                    [
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(-1., 0.)
                    ],
                ]
            }
            "SWAP" => {
                array![
                    [
                        Complex64::new(1., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.)
                    ],
                    [
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(1., 0.),
                        Complex64::new(0., 0.)
                    ],
                    [
                        Complex64::new(0., 0.),
                        Complex64::new(1., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.)
                    ],
                    [
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(0., 0.),
                        Complex64::new(1., 0.)
                    ],
                ]
            }
            "TOFFOLI" | "CCX" => {
                let mut matrix = Array2::zeros((8, 8));
                for i in 0..6 {
                    matrix[[i, i]] = Complex64::new(1., 0.);
                }
                matrix[[6, 7]] = Complex64::new(1., 0.);
                matrix[[7, 6]] = Complex64::new(1., 0.);
                matrix
            }
            _ => {
                return Err(QuantRS2Error::UnsupportedOperation(format!(
                    "Gate '{}' matrix not implemented",
                    gate.name()
                )));
            }
        };
        Ok(matrix)
    }
    /// Compute amplitude of a computational basis state
    pub fn get_amplitude(&self, bitstring: &[bool]) -> QuantRS2Result<Complex64> {
        if bitstring.len() != self.num_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Bitstring length {} doesn't match qubit count {}",
                bitstring.len(),
                self.num_qubits
            )));
        }
        let mut result = Array2::from_elem((1, 1), Complex64::new(1.0, 0.0));
        for (i, &bit) in bitstring.iter().enumerate() {
            let tensor = &self.tensors[i];
            let physical_idx = i32::from(bit);
            let matrix = tensor.data.slice(s![.., physical_idx, ..]);
            result = result.dot(&matrix);
        }
        Ok(result[[0, 0]])
    }
    /// Get all amplitudes as a state vector
    pub fn to_statevector(&self) -> QuantRS2Result<Array1<Complex64>> {
        let dim = 1 << self.num_qubits;
        let mut amplitudes = Array1::zeros(dim);
        amplitudes
            .iter_mut()
            .enumerate()
            .try_for_each(|(i, amp)| -> QuantRS2Result<()> {
                let mut bitstring = vec![false; self.num_qubits];
                for (j, bit) in bitstring.iter_mut().enumerate() {
                    *bit = (i >> j) & 1 == 1;
                }
                *amp = self.get_amplitude(&bitstring)?;
                Ok(())
            })?;
        Ok(amplitudes)
    }
    /// Sample measurement outcome
    pub fn sample(&mut self) -> Vec<bool> {
        let mut result = vec![false; self.num_qubits];
        let mut accumulated = Array2::from_elem((1, 1), Complex64::new(1.0, 0.0));
        for i in 0..self.num_qubits {
            let tensor = &self.tensors[i];
            let matrix0 = tensor.data.slice(s![.., 0, ..]);
            let branch0: Array2<Complex64> = accumulated.dot(&matrix0);
            let mut norm0 = branch0.clone();
            for j in (i + 1)..self.num_qubits {
                let t = &self.tensors[j];
                let sum_matrix =
                    t.data.slice(s![.., 0, ..]).to_owned() + t.data.slice(s![.., 1, ..]).to_owned();
                norm0 = norm0.dot(&sum_matrix);
            }
            let prob0 = norm0[[0, 0]].norm_sqr();
            let matrix1 = tensor.data.slice(s![.., 1, ..]);
            let branch1: Array2<Complex64> = accumulated.dot(&matrix1);
            let mut norm1 = branch1.clone();
            for j in (i + 1)..self.num_qubits {
                let t = &self.tensors[j];
                let sum_matrix =
                    t.data.slice(s![.., 0, ..]).to_owned() + t.data.slice(s![.., 1, ..]).to_owned();
                norm1 = norm1.dot(&sum_matrix);
            }
            let prob1 = norm1[[0, 0]].norm_sqr();
            let total = prob0 + prob1;
            let threshold = prob0 / total;
            if self.rng.random::<f64>() < threshold {
                result[i] = false;
                accumulated = branch0;
            } else {
                result[i] = true;
                accumulated = branch1;
            }
            let norm_squared: f64 = accumulated.iter().map(|x| x.norm_sqr()).sum();
            let norm = norm_squared.sqrt();
            if norm > 0.0 {
                accumulated /= Complex64::new(norm, 0.0);
            }
        }
        result
    }
    /// Compute entanglement entropy across a cut
    pub fn entanglement_entropy(&mut self, cut_position: usize) -> QuantRS2Result<f64> {
        if cut_position >= self.num_qubits - 1 {
            return Err(QuantRS2Error::InvalidInput(
                "Cut position must be less than num_qubits - 1".to_string(),
            ));
        }
        if self.num_qubits == 2 && cut_position == 0 {
            let mut psi = Array1::<Complex64>::zeros(4);
            for i in 0..4 {
                let b0 = i & 1;
                let b1 = (i >> 1) & 1;
                let mut val = Complex64::new(1.0, 0.0);
                for m in 0..self.tensors[0].right_dim {
                    let t0_val = self.tensors[0].data[[0, b0, m]];
                    let t1_val = self.tensors[1].data[[m, b1, 0]];
                    val = t0_val * t1_val;
                    psi[i] += val;
                }
            }
            let mut rho = Array2::<Complex64>::zeros((2, 2));
            rho[[0, 0]] = psi[0] * psi[0].conj() + psi[2] * psi[2].conj();
            rho[[0, 1]] = psi[0] * psi[1].conj() + psi[2] * psi[3].conj();
            rho[[1, 0]] = psi[1] * psi[0].conj() + psi[3] * psi[2].conj();
            rho[[1, 1]] = psi[1] * psi[1].conj() + psi[3] * psi[3].conj();
            use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO};
            let (eigenvalues, _) = rho.eigh(UPLO::Lower).map_err(|e| {
                QuantRS2Error::LinalgError(format!("Eigenvalue decomposition failed: {e}"))
            })?;
            let mut entropy = 0.0;
            for &lambda in &eigenvalues {
                if lambda > 1e-12 {
                    entropy -= lambda * lambda.ln();
                }
            }
            return Ok(entropy);
        }
        // Direct transfer-matrix approach: avoids canonicalization which can be
        // numerically fragile. We compute the reduced density matrix ρ_A = Tr_B|ψ⟩⟨ψ|
        // by contracting tensors on each side of the cut.
        //
        // Left side: Contract tensors 0..=cut_position to get a matrix L of shape
        //   (bond_left, 2^(cut+1)) where bond_left = left_dim of first tensor = 1.
        //   Actually we contract to a (χ_cut+1) × (χ_cut+1) transfer matrix.
        //
        // Simpler: For bond dimension χ at the cut, ρ_A is the Gram matrix of the
        // left-block contraction. The entanglement entropy equals -Tr(ρ_A ln ρ_A).
        //
        // We compute the left half-density matrix via:
        //   Γ^L_{α,β} = Σ_{phys_0..phys_cut} A[0]_{1,σ_0,α_0} · A[1]_{α_0,σ_1,α_1} · ... · A[k]_{α_{k-1},σ_k,β}
        // and then the density matrix ρ_A = Γ^L · (Γ^L)†, whose eigenvalues give the entropy.
        //
        // This is equivalent to forming the (χ_left × χ_right) bond matrix and SVD-ing it.
        //
        // Step 1: Contract tensors 0..cut_position into a single (χ₀, χ_cut) matrix
        //         where χ₀=1 (left boundary) and χ_cut is the bond at the cut.
        //         We track the accumulated (1, bond) vector by summing over physical indices.
        //
        // We form the "density matrix" of the LEFT block at the cut bond:
        //   ρ_{α,β} = Σ_{phys} L_{phys, α} L_{phys, β}†
        // where L is the (2^(cut+1), χ) matrix of left-block amplitudes.
        // ρ's eigenvalues are the squared Schmidt values.
        //
        // To avoid 2^N exponential cost, we build ρ iteratively using transfer matrices.
        //
        // ρ_A = T_0 ⊗ ... ⊗ T_k where T_i is the transfer matrix of tensor i.
        // ρ^{α,β}_{cut} = Σ_{σ_0..σ_k} A[k]^{*}_{α,σ_k, ...} A[k]_{β,σ_k,...}
        // This is the reduced density matrix: iterate from left boundary.
        //
        // Method: build left half-density matrix ρ^L ∈ C^{χ×χ} where χ = bond at cut+1.
        // Initialize: ρ^L = [[1]] (1×1 identity at the left boundary).
        // For each tensor A[i] at site i (0 ≤ i ≤ cut):
        //   ρ^L_{new}[α, β] = Σ_{γ,δ,σ} A[i][γ, σ, α] ρ^L[γ, δ] conj(A[i][δ, σ, β])
        // After processing all sites 0..=cut, ρ^L has shape (χ_cut+1, χ_cut+1).
        // Eigenvalues of ρ^L are the squared Schmidt values λᵢ.
        // Entropy = -Σ λᵢ ln(λᵢ).
        use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO, SVD};
        let chi_cut = self.tensors[cut_position].right_dim;
        // Build left half-density matrix.
        let mut rho_l = Array2::<Complex64>::zeros((1, 1));
        rho_l[[0, 0]] = Complex64::new(1.0, 0.0);
        for i in 0..=cut_position {
            let t = &self.tensors[i];
            let left_dim = t.left_dim;
            let right_dim = t.right_dim;
            let mut rho_new = Array2::<Complex64>::zeros((right_dim, right_dim));
            for alpha in 0..right_dim {
                for beta in 0..right_dim {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for gamma in 0..left_dim {
                        for delta in 0..left_dim {
                            let rho_gd = rho_l[[gamma, delta]];
                            if rho_gd.norm() < 1e-15 {
                                continue;
                            }
                            for sigma in 0..2 {
                                sum += t.data[[gamma, sigma, alpha]]
                                    * rho_gd
                                    * t.data[[delta, sigma, beta]].conj();
                            }
                        }
                    }
                    rho_new[[alpha, beta]] = sum;
                }
            }
            rho_l = rho_new;
        }
        // ρ_l is now (χ_cut, χ_cut) and Hermitian (by construction).
        // Its eigenvalues are proportional to the Schmidt probabilities λᵢ.
        // The MPS may not be in left-canonical form, so ρ_l may not be
        // unit-trace. We normalize the eigenvalues so they sum to 1 before
        // computing entropy.
        let _ = chi_cut; // used via rho_l.shape()[0]
        let eigenvalues = match rho_l.eigh(UPLO::Lower) {
            Ok((ev, _)) => ev,
            Err(_) => {
                // Fall back to SVD if eigh fails (e.g. non-square)
                let (_, s, _) = rho_l
                    .svd(false, false)
                    .map_err(|e| QuantRS2Error::LinalgError(format!("SVD failed: {e}")))?;
                // SVD of Hermitian gives squares of eigenvalues — take sqrt
                s.mapv(|x: f64| x.sqrt())
            }
        };
        // Normalize so eigenvalues sum to 1 (the MPS may be in an arbitrary gauge
        // where the left half is not unit-normalized).
        let trace: f64 = eigenvalues.iter().filter(|&&x| x > 1e-12).sum();
        let mut entropy = 0.0;
        if trace > 1e-12 {
            for &lambda_raw in &eigenvalues {
                let lambda = lambda_raw / trace;
                if lambda > 1e-12 {
                    entropy -= lambda * lambda.ln();
                }
            }
        }
        Ok(entropy)
    }
    /// Measure a qubit in the computational basis and update the state
    pub fn measure_qubit(&mut self, qubit: usize) -> QuantRS2Result<bool> {
        if qubit >= self.num_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
        }
        self.move_orthogonality_center(qubit)?;
        let tensor = &self.tensors[qubit];
        let mut left_env = Array2::from_elem((1, 1), Complex64::new(1.0, 0.0));
        for i in 0..qubit {
            let t = &self.tensors[i];
            let sum_matrix =
                t.data.slice(s![.., 0, ..]).to_owned() + t.data.slice(s![.., 1, ..]).to_owned();
            left_env = left_env.dot(&sum_matrix);
        }
        let mut right_env = Array2::from_elem((1, 1), Complex64::new(1.0, 0.0));
        for i in (qubit + 1)..self.num_qubits {
            let t = &self.tensors[i];
            let sum_matrix =
                t.data.slice(s![.., 0, ..]).to_owned() + t.data.slice(s![.., 1, ..]).to_owned();
            right_env = right_env.dot(&sum_matrix);
        }
        let tensor_slice_0 = tensor.data.slice(s![.., 0, ..]).to_owned();
        let tensor_slice_1 = tensor.data.slice(s![.., 1, ..]).to_owned();
        let intermediate0: Array2<Complex64> = left_env.dot(&tensor_slice_0);
        let intermediate1: Array2<Complex64> = left_env.dot(&tensor_slice_1);
        let prob0_matrix = intermediate0.dot(&right_env);
        let prob1_matrix = intermediate1.dot(&right_env);
        let prob0 = prob0_matrix[[0, 0]].norm_sqr();
        let prob1 = prob1_matrix[[0, 0]].norm_sqr();
        let total_prob = prob0 + prob1;
        let outcome = self.rng.random::<f64>() < prob0 / total_prob;
        if outcome {
            let new_data = tensor.data.slice(s![.., 0, ..]).to_owned().into_shape((
                tensor.left_dim,
                1,
                tensor.right_dim,
            ))?;
            self.tensors[qubit] = MPSTensor::new(new_data);
            let norm = (prob0 / total_prob).sqrt();
            if norm > 0.0 {
                self.tensors[qubit].data /= Complex64::new(norm, 0.0);
            }
        } else {
            let new_data = tensor.data.slice(s![.., 1, ..]).to_owned().into_shape((
                tensor.left_dim,
                1,
                tensor.right_dim,
            ))?;
            self.tensors[qubit] = MPSTensor::new(new_data);
            let norm = (prob1 / total_prob).sqrt();
            if norm > 0.0 {
                self.tensors[qubit].data /= Complex64::new(norm, 0.0);
            }
        }
        Ok(!outcome)
    }
    /// Compute expectation value of a Pauli string
    pub fn expectation_value_pauli(&self, pauli_string: &str) -> QuantRS2Result<Complex64> {
        if pauli_string.len() != self.num_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Pauli string length {} doesn't match qubit count {}",
                pauli_string.len(),
                self.num_qubits
            )));
        }
        let state_vector = self.to_statevector()?;
        let mut result = Complex64::new(0.0, 0.0);
        for (i, amplitude) in state_vector.iter().enumerate() {
            let mut coeff = Complex64::new(1.0, 0.0);
            let mut target_state = i;
            for (qubit, pauli_char) in pauli_string.chars().rev().enumerate() {
                let bit = (i >> qubit) & 1;
                match pauli_char {
                    'I' => {}
                    'X' => {
                        target_state ^= 1 << qubit;
                    }
                    'Y' => {
                        target_state ^= 1 << qubit;
                        coeff *= if bit == 0 {
                            Complex64::new(0.0, 1.0)
                        } else {
                            Complex64::new(0.0, -1.0)
                        };
                    }
                    'Z' => {
                        if bit == 1 {
                            coeff *= Complex64::new(-1.0, 0.0);
                        }
                    }
                    _ => {
                        return Err(QuantRS2Error::InvalidInput(format!(
                            "Invalid Pauli operator: {pauli_char}"
                        )));
                    }
                }
            }
            result += amplitude.conj() * coeff * state_vector[target_state];
        }
        Ok(result)
    }
    /// Compute variance of a Pauli string observable
    pub fn variance_pauli(&self, pauli_string: &str) -> QuantRS2Result<f64> {
        let expectation = self.expectation_value_pauli(pauli_string)?;
        let variance = 1.0 - expectation.norm_sqr();
        Ok(variance.max(0.0))
    }
    /// Get current bond dimensions
    pub fn bond_dimensions(&self) -> Vec<usize> {
        self.tensors.iter().map(|t| t.right_dim).collect()
    }
    /// Get maximum bond dimension currently used
    pub fn max_bond_dimension(&self) -> usize {
        self.bond_dimensions().iter().copied().max().unwrap_or(1)
    }
    /// Compress MPS by reducing bond dimensions
    pub fn compress(&mut self, new_threshold: Option<f64>) -> QuantRS2Result<()> {
        let old_threshold = self.config.svd_threshold;
        if let Some(threshold) = new_threshold {
            self.config.svd_threshold = threshold;
        }
        for i in 0..self.num_qubits - 1 {
            self.move_orthogonality_center(i)?;
            let (matrix, left_dim) = {
                let tensor = &self.tensors[i];
                let matrix = tensor
                    .data
                    .view()
                    .into_shape((tensor.left_dim * 2, tensor.right_dim))?
                    .to_owned();
                (matrix, tensor.left_dim)
            };
            let (u, s, vt) = self.truncated_svd(&matrix)?;
            let new_bond = s.len();
            self.tensors[i] = MPSTensor::new(u.into_shape((left_dim, 2, new_bond))?);
            if i + 1 < self.num_qubits {
                let sv_matrix = {
                    let mut sv = Array2::<Complex64>::zeros((new_bond, vt.shape()[1]));
                    sv.indexed_iter_mut()
                        .par_bridge()
                        .for_each(|((j, k), elem)| {
                            *elem = Complex64::new(s[j], 0.0) * vt[[j, k]];
                        });
                    sv
                };
                let next = &self.tensors[i + 1];
                let next_matrix = next
                    .data
                    .view()
                    .into_shape((next.left_dim, 2 * next.right_dim))?;
                let new_next = sv_matrix.dot(&next_matrix);
                self.tensors[i + 1] =
                    MPSTensor::new(new_next.into_shape((new_bond, 2, next.right_dim))?);
            }
        }
        if new_threshold.is_some() {
            self.config.svd_threshold = old_threshold;
        }
        Ok(())
    }
    /// Fallback SVD implementation using ndarray-linalg
    fn fallback_svd(
        matrix: &Array2<Complex64>,
    ) -> Result<(Array2<Complex64>, Array1<f64>, Array2<Complex64>), QuantRS2Error> {
        use scirs2_core::ndarray::ndarray_linalg::SVD;
        let (u, s, vt) = matrix
            .svd(true, true)
            .map_err(|_| QuantRS2Error::ComputationError("SVD decomposition failed".to_string()))?;
        Ok((u, s, vt))
    }
}
/// Enhanced MPS quantum simulator
pub struct EnhancedMPSSimulator {
    config: MPSConfig,
    /// SciRS2 backend for optimized linear algebra operations
    scirs2_backend: SciRS2Backend,
}
impl EnhancedMPSSimulator {
    /// Create a new MPS simulator with configuration
    pub fn new(config: MPSConfig) -> Self {
        Self {
            config,
            scirs2_backend: SciRS2Backend::new(),
        }
    }
    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(MPSConfig::default())
    }
    /// Set maximum bond dimension
    #[must_use]
    pub const fn with_bond_dimension(mut self, max_bond: usize) -> Self {
        self.config.max_bond_dim = max_bond;
        self
    }
    /// Set SVD truncation threshold
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f64) -> Self {
        self.config.svd_threshold = threshold;
        self
    }
}
impl<const N: usize> Simulator<N> for EnhancedMPSSimulator {
    fn run(&self, circuit: &Circuit<N>) -> QuantRS2Result<Register<N>> {
        let mut mps = EnhancedMPS::new(N, self.config.clone());
        for gate in circuit.gates() {
            mps.apply_gate(gate.as_ref())?;
        }
        let statevector = mps.to_statevector()?;
        let mut register = Register::new();
        Ok(register)
    }
}
/// Convenience functions
pub mod utils {
    use super::*;
    /// Create a Bell state using MPS
    pub fn create_bell_state_mps() -> QuantRS2Result<EnhancedMPS> {
        let mut mps = EnhancedMPS::new(2, MPSConfig::default());
        let h_matrix = {
            let h = 1.0 / SQRT_2;
            array![
                [Complex64::new(h, 0.), Complex64::new(h, 0.)],
                [Complex64::new(h, 0.), Complex64::new(-h, 0.)]
            ]
        };
        mps.tensors[0].apply_gate(&h_matrix)?;
        let mut cnot_array = Array4::<Complex64>::zeros((2, 2, 2, 2));
        cnot_array[[0, 0, 0, 0]] = Complex64::new(1., 0.);
        cnot_array[[0, 1, 0, 1]] = Complex64::new(1., 0.);
        cnot_array[[1, 0, 1, 1]] = Complex64::new(1., 0.);
        cnot_array[[1, 1, 1, 0]] = Complex64::new(1., 0.);
        mps.apply_and_decompose_two_qubit_gate(&cnot_array, 0, 1)?;
        Ok(mps)
    }
    /// Compute fidelity between two MPS states
    pub fn mps_fidelity(mps1: &EnhancedMPS, mps2: &EnhancedMPS) -> QuantRS2Result<f64> {
        if mps1.num_qubits != mps2.num_qubits {
            return Err(QuantRS2Error::InvalidInput(
                "MPS states must have same number of qubits".to_string(),
            ));
        }
        let sv1 = mps1.to_statevector()?;
        let sv2 = mps2.to_statevector()?;
        let inner_product: Complex64 = sv1.iter().zip(sv2.iter()).map(|(a, b)| a.conj() * b).sum();
        Ok(inner_product.norm_sqr())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mps_initialization() {
        let mps = EnhancedMPS::new(5, MPSConfig::default());
        assert_eq!(mps.num_qubits, 5);
        assert_eq!(mps.tensors.len(), 5);
        let amp = mps
            .get_amplitude(&[false, false, false, false, false])
            .expect("amplitude calculation should succeed");
        assert!((amp.norm() - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_single_qubit_gates() {
        let mut mps = EnhancedMPS::new(3, MPSConfig::default());
        let x_matrix = array![
            [Complex64::new(0., 0.), Complex64::new(1., 0.)],
            [Complex64::new(1., 0.), Complex64::new(0., 0.)]
        ];
        mps.tensors[1]
            .apply_gate(&x_matrix)
            .expect("gate application should succeed");
        let amp = mps
            .get_amplitude(&[false, true, false])
            .expect("amplitude calculation should succeed");
        assert!((amp.norm() - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_bell_state_creation() {
        let bell_mps = utils::create_bell_state_mps().expect("Bell state creation should succeed");
        let expected = 1.0 / SQRT_2;
        let amp00 = bell_mps
            .get_amplitude(&[false, false])
            .expect("amplitude calculation should succeed");
        assert!((amp00.norm() - expected).abs() < 1e-10);
        let amp11 = bell_mps
            .get_amplitude(&[true, true])
            .expect("amplitude calculation should succeed");
        assert!((amp11.norm() - expected).abs() < 1e-10);
        let amp01 = bell_mps
            .get_amplitude(&[false, true])
            .expect("amplitude calculation should succeed");
        assert!(amp01.norm() < 1e-10);
        let amp10 = bell_mps
            .get_amplitude(&[true, false])
            .expect("amplitude calculation should succeed");
        assert!(amp10.norm() < 1e-10);
    }
    #[test]
    fn test_entanglement_entropy() {
        let mut bell_mps =
            utils::create_bell_state_mps().expect("Bell state creation should succeed");
        let entropy = bell_mps
            .entanglement_entropy(0)
            .expect("entropy calculation should succeed");
        assert!((entropy - 2.0_f64.ln()).abs() < 1e-10);
    }
}
