//! Tensor Network representations for quantum circuits
//!
//! This module provides tensor network representations and operations for quantum circuits,
//! leveraging SciRS2 for efficient tensor manipulations and contractions.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    linalg_stubs::svd,
    register::Register,
};
use scirs2_core::ndarray::{Array, Array2, ArrayD, IxDyn};
use scirs2_core::Complex;
// use scirs2_linalg::svd;
use std::collections::{HashMap, HashSet};

/// Type alias for complex numbers
type Complex64 = Complex<f64>;

/// A tensor in the network
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Unique identifier for the tensor
    pub id: usize,
    /// The tensor data
    pub data: ArrayD<Complex64>,
    /// Labels for each index of the tensor
    pub indices: Vec<String>,
    /// Shape of the tensor
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor
    pub fn new(id: usize, data: ArrayD<Complex64>, indices: Vec<String>) -> Self {
        let shape = data.shape().to_vec();
        Self {
            id,
            data,
            indices,
            shape,
        }
    }

    /// Create a tensor from a 2D array (matrix)
    pub fn from_matrix(
        id: usize,
        matrix: Array2<Complex64>,
        in_idx: String,
        out_idx: String,
    ) -> Self {
        let shape = matrix.shape().to_vec();
        let data = matrix.into_dyn();
        Self {
            id,
            data,
            indices: vec![in_idx, out_idx],
            shape,
        }
    }

    /// Create a qubit tensor in |0⟩ state
    pub fn qubit_zero(id: usize, idx: String) -> Self {
        let mut data = Array::zeros(IxDyn(&[2]));
        data[[0]] = Complex64::new(1.0, 0.0);
        Self {
            id,
            data,
            indices: vec![idx],
            shape: vec![2],
        }
    }

    /// Create a qubit tensor in |1⟩ state
    pub fn qubit_one(id: usize, idx: String) -> Self {
        let mut data = Array::zeros(IxDyn(&[2]));
        data[[1]] = Complex64::new(1.0, 0.0);
        Self {
            id,
            data,
            indices: vec![idx],
            shape: vec![2],
        }
    }

    /// Create a tensor from an ndarray with specified indices
    pub fn from_array<D>(
        array: scirs2_core::ndarray::ArrayBase<scirs2_core::ndarray::OwnedRepr<Complex64>, D>,
        indices: Vec<usize>,
    ) -> Self
    where
        D: scirs2_core::ndarray::Dimension,
    {
        let shape = array.shape().to_vec();
        let data = array.into_dyn();
        let index_labels: Vec<String> = indices.iter().map(|i| format!("idx_{i}")).collect();
        Self {
            id: 0, // Default ID
            data,
            indices: index_labels,
            shape,
        }
    }

    /// Get the rank (number of indices) of the tensor
    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    /// Get a reference to the tensor data
    pub const fn tensor(&self) -> &ArrayD<Complex64> {
        &self.data
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Contract this tensor with another over specified indices
    pub fn contract(&self, other: &Self, self_idx: &str, other_idx: &str) -> QuantRS2Result<Self> {
        // Find the positions of the indices to contract
        let self_pos = self
            .indices
            .iter()
            .position(|s| s == self_idx)
            .ok_or_else(|| {
                QuantRS2Error::InvalidInput(format!("Index {self_idx} not found in tensor"))
            })?;
        let other_pos = other
            .indices
            .iter()
            .position(|s| s == other_idx)
            .ok_or_else(|| {
                QuantRS2Error::InvalidInput(format!("Index {other_idx} not found in tensor"))
            })?;

        // Check dimensions match
        if self.shape[self_pos] != other.shape[other_pos] {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Cannot contract indices with different dimensions: {} vs {}",
                self.shape[self_pos], other.shape[other_pos]
            )));
        }

        // Perform tensor contraction using einsum-like operation
        let contracted = self.contract_indices(&other, self_pos, other_pos)?;

        // Build new index list
        let mut new_indices = Vec::new();
        for (i, idx) in self.indices.iter().enumerate() {
            if i != self_pos {
                new_indices.push(idx.clone());
            }
        }
        for (i, idx) in other.indices.iter().enumerate() {
            if i != other_pos {
                new_indices.push(idx.clone());
            }
        }

        Ok(Self::new(
            self.id.max(other.id) + 1,
            contracted,
            new_indices,
        ))
    }

    /// Perform the actual index contraction
    fn contract_indices(
        &self,
        other: &Self,
        self_idx: usize,
        other_idx: usize,
    ) -> QuantRS2Result<ArrayD<Complex64>> {
        // Reshape tensors for matrix multiplication
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();

        // Calculate dimensions for reshaping
        let mut self_left_dims = 1;
        let mut self_right_dims = 1;
        for i in 0..self_idx {
            self_left_dims *= self_shape[i];
        }
        for i in (self_idx + 1)..self_shape.len() {
            self_right_dims *= self_shape[i];
        }

        let mut other_left_dims = 1;
        let mut other_right_dims = 1;
        for i in 0..other_idx {
            other_left_dims *= other_shape[i];
        }
        for i in (other_idx + 1)..other_shape.len() {
            other_right_dims *= other_shape[i];
        }

        let contract_dim = self_shape[self_idx];

        // Reshape to matrices
        let self_mat = self
            .data
            .view()
            .into_shape_with_order((self_left_dims, contract_dim * self_right_dims))
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {e}")))?
            .to_owned();
        let other_mat = other
            .data
            .view()
            .into_shape_with_order((other_left_dims * contract_dim, other_right_dims))
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {e}")))?
            .to_owned();

        // Perform contraction via matrix multiplication
        let _result_mat: Array2<Complex64> = Array2::zeros((
            self_left_dims * self_right_dims,
            other_left_dims * other_right_dims,
        ));

        // This is a simplified contraction - a full implementation would be more efficient
        let mut result_vec = Vec::new();
        for i in 0..self_left_dims {
            for j in 0..self_right_dims {
                for k in 0..other_left_dims {
                    for l in 0..other_right_dims {
                        let mut sum = Complex64::new(0.0, 0.0);
                        for c in 0..contract_dim {
                            // Commented out - index calculations unused
                            // let _ = i * contract_dim * self_right_dims + c * self_right_dims + j;
                            // let _ = k * contract_dim * other_right_dims + c * other_right_dims + l;
                            sum += self_mat[[i, c * self_right_dims + j]]
                                * other_mat[[k * contract_dim + c, l]];
                        }
                        result_vec.push(sum);
                    }
                }
            }
        }

        // Build result shape
        let mut result_shape = Vec::new();
        for i in 0..self_idx {
            result_shape.push(self_shape[i]);
        }
        for i in (self_idx + 1)..self_shape.len() {
            result_shape.push(self_shape[i]);
        }
        for i in 0..other_idx {
            result_shape.push(other_shape[i]);
        }
        for i in (other_idx + 1)..other_shape.len() {
            result_shape.push(other_shape[i]);
        }

        ArrayD::from_shape_vec(IxDyn(&result_shape), result_vec)
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {e}")))
    }

    /// Apply SVD decomposition to split tensor along specified index
    pub fn svd_decompose(
        &self,
        idx: usize,
        max_rank: Option<usize>,
    ) -> QuantRS2Result<(Self, Self)> {
        if idx >= self.rank() {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Index {} out of bounds for tensor with rank {}",
                idx,
                self.rank()
            )));
        }

        // Reshape tensor into matrix
        let shape = self.data.shape();
        let mut left_dim = 1;
        let mut right_dim = 1;

        for i in 0..=idx {
            left_dim *= shape[i];
        }
        for i in (idx + 1)..shape.len() {
            right_dim *= shape[i];
        }

        // Convert to matrix
        let matrix = self
            .data
            .view()
            .into_shape_with_order((left_dim, right_dim))
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {e}")))?
            .to_owned();

        // Perform SVD using SciRS2
        let real_matrix = matrix.mapv(|c| c.re);
        let (u, s, vt) = svd(&real_matrix.view(), false, None)
            .map_err(|e| QuantRS2Error::ComputationError(format!("SVD failed: {e:?}")))?;

        // Determine rank to keep
        let rank = if let Some(max_r) = max_rank {
            max_r.min(s.len())
        } else {
            s.len()
        };

        // Truncate based on rank
        let u_trunc = u.slice(scirs2_core::ndarray::s![.., ..rank]).to_owned();
        let s_trunc = s.slice(scirs2_core::ndarray::s![..rank]).to_owned();
        let vt_trunc = vt.slice(scirs2_core::ndarray::s![..rank, ..]).to_owned();

        // Create S matrix
        let mut s_mat = Array2::zeros((rank, rank));
        for i in 0..rank {
            s_mat[[i, i]] = Complex64::new(s_trunc[i].sqrt(), 0.0);
        }

        // Multiply U * sqrt(S) and sqrt(S) * V^T
        let left_data = u_trunc.mapv(|x| Complex64::new(x, 0.0)).dot(&s_mat);
        let right_data = s_mat.dot(&vt_trunc.mapv(|x| Complex64::new(x, 0.0)));

        // Create new tensors with appropriate shapes and indices
        let mut left_indices = self.indices[..=idx].to_vec();
        left_indices.push(format!("bond_{}", self.id));

        let mut right_indices = vec![format!("bond_{}", self.id)];
        right_indices.extend_from_slice(&self.indices[(idx + 1)..]);

        let left_tensor = Self::new(self.id * 2, left_data.into_dyn(), left_indices);

        let right_tensor = Self::new(self.id * 2 + 1, right_data.into_dyn(), right_indices);

        Ok((left_tensor, right_tensor))
    }
}

/// Edge in the tensor network
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorEdge {
    /// First tensor ID
    pub tensor1: usize,
    /// Index on first tensor
    pub index1: String,
    /// Second tensor ID
    pub tensor2: usize,
    /// Index on second tensor
    pub index2: String,
}

/// Tensor network representation
#[derive(Debug)]
pub struct TensorNetwork {
    /// Tensors in the network
    pub tensors: HashMap<usize, Tensor>,
    /// Edges connecting tensors
    pub edges: Vec<TensorEdge>,
    /// Open indices (not connected to other tensors)
    pub open_indices: HashMap<usize, Vec<String>>,
    /// Next available tensor ID
    next_id: usize,
}

impl TensorNetwork {
    /// Create a new empty tensor network
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            edges: Vec::new(),
            open_indices: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add a tensor to the network
    pub fn add_tensor(&mut self, tensor: Tensor) -> usize {
        let id = tensor.id;
        self.open_indices.insert(id, tensor.indices.clone());
        self.tensors.insert(id, tensor);
        self.next_id = self.next_id.max(id + 1);
        id
    }

    /// Connect two tensor indices
    pub fn connect(
        &mut self,
        tensor1: usize,
        index1: String,
        tensor2: usize,
        index2: String,
    ) -> QuantRS2Result<()> {
        // Verify tensors exist
        if !self.tensors.contains_key(&tensor1) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Tensor {tensor1} not found"
            )));
        }
        if !self.tensors.contains_key(&tensor2) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Tensor {tensor2} not found"
            )));
        }

        // Verify indices exist and match dimensions
        let t1 = &self.tensors[&tensor1];
        let t2 = &self.tensors[&tensor2];

        let idx1_pos = t1
            .indices
            .iter()
            .position(|s| s == &index1)
            .ok_or_else(|| {
                QuantRS2Error::InvalidInput(format!("Index {index1} not found in tensor {tensor1}"))
            })?;
        let idx2_pos = t2
            .indices
            .iter()
            .position(|s| s == &index2)
            .ok_or_else(|| {
                QuantRS2Error::InvalidInput(format!("Index {index2} not found in tensor {tensor2}"))
            })?;

        if t1.shape[idx1_pos] != t2.shape[idx2_pos] {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Connected indices must have same dimension: {} vs {}",
                t1.shape[idx1_pos], t2.shape[idx2_pos]
            )));
        }

        // Add edge
        self.edges.push(TensorEdge {
            tensor1,
            index1: index1.clone(),
            tensor2,
            index2: index2.clone(),
        });

        // Remove from open indices
        if let Some(indices) = self.open_indices.get_mut(&tensor1) {
            indices.retain(|s| s != &index1);
        }
        if let Some(indices) = self.open_indices.get_mut(&tensor2) {
            indices.retain(|s| s != &index2);
        }

        Ok(())
    }

    /// Find optimal contraction order using greedy algorithm
    pub fn find_contraction_order(&self) -> Vec<(usize, usize)> {
        // Simple greedy algorithm: contract pairs that minimize intermediate tensor size
        let mut remaining_tensors: HashSet<_> = self.tensors.keys().copied().collect();
        let mut order = Vec::new();

        // Build adjacency list
        let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
        for edge in &self.edges {
            adjacency
                .entry(edge.tensor1)
                .or_insert_with(Vec::new)
                .push(edge.tensor2);
            adjacency
                .entry(edge.tensor2)
                .or_insert_with(Vec::new)
                .push(edge.tensor1);
        }

        while remaining_tensors.len() > 1 {
            let mut best_pair = None;
            let mut min_cost = usize::MAX;

            // Consider all pairs of connected tensors
            for &t1 in &remaining_tensors {
                if let Some(neighbors) = adjacency.get(&t1) {
                    for &t2 in neighbors {
                        if t2 > t1 && remaining_tensors.contains(&t2) {
                            // Estimate cost as product of remaining dimensions
                            let cost = self.estimate_contraction_cost(t1, t2);
                            if cost < min_cost {
                                min_cost = cost;
                                best_pair = Some((t1, t2));
                            }
                        }
                    }
                }
            }

            if let Some((t1, t2)) = best_pair {
                order.push((t1, t2));
                remaining_tensors.remove(&t1);
                remaining_tensors.remove(&t2);

                // Add a virtual tensor representing the contraction result
                let virtual_id = self.next_id + order.len();
                remaining_tensors.insert(virtual_id);

                // Update adjacency for virtual tensor
                let mut virtual_neighbors = HashSet::new();
                if let Some(n1) = adjacency.get(&t1) {
                    virtual_neighbors.extend(
                        n1.iter()
                            .filter(|&&n| n != t2 && remaining_tensors.contains(&n)),
                    );
                }
                if let Some(n2) = adjacency.get(&t2) {
                    virtual_neighbors.extend(
                        n2.iter()
                            .filter(|&&n| n != t1 && remaining_tensors.contains(&n)),
                    );
                }
                adjacency.insert(virtual_id, virtual_neighbors.into_iter().collect());
            } else {
                break;
            }
        }

        order
    }

    /// Estimate the computational cost of contracting two tensors
    const fn estimate_contraction_cost(&self, _t1: usize, _t2: usize) -> usize {
        // Cost is roughly the product of all dimensions in the result
        // This is a simplified estimate
        1000 // Placeholder
    }

    /// Contract the entire network to a single tensor
    pub fn contract_all(&mut self) -> QuantRS2Result<Tensor> {
        if self.tensors.is_empty() {
            return Err(QuantRS2Error::InvalidInput(
                "Cannot contract empty tensor network".into(),
            ));
        }

        if self.tensors.len() == 1 {
            return self
                .tensors
                .values()
                .next()
                .map(|t| t.clone())
                .ok_or_else(|| {
                    QuantRS2Error::InvalidInput("Single tensor expected but not found".into())
                });
        }

        // Find contraction order
        let order = self.find_contraction_order();

        // Execute contractions
        let mut tensor_map = self.tensors.clone();
        let mut next_id = self.next_id;

        for (t1_id, t2_id) in order {
            // Find the edge connecting these tensors
            let edge = self
                .edges
                .iter()
                .find(|e| {
                    (e.tensor1 == t1_id && e.tensor2 == t2_id)
                        || (e.tensor1 == t2_id && e.tensor2 == t1_id)
                })
                .ok_or_else(|| QuantRS2Error::InvalidInput("Tensors not connected".into()))?;

            let t1 = tensor_map
                .remove(&t1_id)
                .ok_or_else(|| QuantRS2Error::InvalidInput("Tensor not found".into()))?;
            let t2 = tensor_map
                .remove(&t2_id)
                .ok_or_else(|| QuantRS2Error::InvalidInput("Tensor not found".into()))?;

            // Contract tensors
            let contracted = if edge.tensor1 == t1_id {
                t1.contract(&t2, &edge.index1, &edge.index2)?
            } else {
                t1.contract(&t2, &edge.index2, &edge.index1)?
            };

            // Add result back
            let mut new_tensor = contracted;
            new_tensor.id = next_id;
            tensor_map.insert(next_id, new_tensor);
            next_id += 1;
        }

        // Return the final tensor
        tensor_map
            .into_values()
            .next()
            .ok_or_else(|| QuantRS2Error::InvalidInput("Contraction failed".into()))
    }

    /// Decompose the (contracted) network into a Matrix Product State (MPS).
    ///
    /// The network is first contracted to a single tensor whose open indices are the
    /// physical legs (each assumed dimension 2). A left-to-right sweep of singular-value
    /// decompositions then factors that tensor into a chain of rank-3 site tensors
    /// `A[0], …, A[n-1]` with bond indices between neighbours. Singular values are kept
    /// up to `max_bond_dim` (when supplied), giving an exact MPS when the bond
    /// dimension is unrestricted and an optimal truncation otherwise.
    ///
    /// The returned tensors carry indices `["phys_k", "bond_{k-1}", "bond_k"]` (the
    /// boundary bonds are dimension 1), so contracting the chain reproduces the
    /// original full tensor (up to the truncation error).
    ///
    /// Uses a complex one-sided Jacobi SVD (see [`Self::complex_svd`]) since the
    /// SciRS2 LAPACK SVD currently exposes only the real-valued path.
    ///
    /// Note: the network must contract to a single tensor whose open legs are the
    /// physical sites in order. Disconnected networks (e.g. an un-entangled product of
    /// independent qubit lines) are limited by [`Self::contract_all`], which returns a
    /// single connected component; build the network with entangling links between the
    /// sites to be represented (as a real circuit does) for a faithful MPS.
    pub fn to_mps(&self, max_bond_dim: Option<usize>) -> QuantRS2Result<Vec<Tensor>> {
        // Contract the network to a single tensor (operate on a clone: to_mps is &self).
        let mut work = TensorNetwork {
            tensors: self.tensors.clone(),
            edges: self.edges.clone(),
            open_indices: self.open_indices.clone(),
            next_id: self.next_id,
        };
        let full = work.contract_all()?;

        // Flatten the full tensor into a vector using the *same* extraction as
        // `to_statevector` (`into_raw_vec`), so the MPS represents exactly the state
        // that `to_statevector` exposes. We treat the flattened amplitudes as a chain
        // of qubits (physical dimension 2); the contracted tensor's reported axis
        // layout may merge legs, so we derive the site count from the amplitude count.
        let total: usize = full.shape.iter().product();
        let flat: Vec<Complex64> = full.data.clone().into_raw_vec_and_offset().0;
        if flat.len() != total {
            return Err(QuantRS2Error::ComputationError(format!(
                "contracted tensor buffer length {} does not match element count {total}",
                flat.len()
            )));
        }

        // Determine the number of qubit sites: total must be a power of two.
        if total == 0 || (total & (total - 1)) != 0 {
            return Err(QuantRS2Error::UnsupportedOperation(format!(
                "MPS construction expects a qubit state (2^n amplitudes); got {total}"
            )));
        }
        let n_sites = total.trailing_zeros() as usize;
        if n_sites == 0 {
            return Err(QuantRS2Error::InvalidInput(
                "cannot build an MPS from a scalar (rank-0) tensor".into(),
            ));
        }
        let phys_dims: Vec<usize> = vec![2usize; n_sites];

        let mut mps = Vec::with_capacity(n_sites);

        // `psi` holds the remaining (left_bond * rest) matrix as a flat row-major
        // buffer with `left_bond` rows; initially left_bond = 1.
        let mut left_bond = 1usize;
        let mut psi = flat;
        let mut remaining = total; // = product of physical dims not yet split off

        for site in 0..n_sites {
            let d = phys_dims[site];
            remaining /= d;
            // Reshape psi (left_bond x (d*remaining)) into a matrix M of shape
            // (left_bond*d, remaining) so the SVD separates this site from the rest.
            let rows = left_bond * d;
            let cols = remaining;
            let mut m = Array2::<Complex64>::zeros((rows, cols));
            for lb in 0..left_bond {
                for phys in 0..d {
                    for rc in 0..cols {
                        // psi index: ((lb)*d + phys)*cols + rc  (row-major over [lb, phys, rc])
                        let src = (lb * d + phys) * cols + rc;
                        m[[lb * d + phys, rc]] = psi[src];
                    }
                }
            }

            if site == n_sites - 1 {
                // Last site: no further splitting; the whole matrix is the final
                // tensor with right bond dimension 1.
                let right_bond = 1usize;
                // rows = left_bond * d, cols should be 1 here.
                let data = Array::from_shape_vec(
                    IxDyn(&[left_bond, d, right_bond]),
                    (0..left_bond * d * right_bond)
                        .map(|idx| {
                            let lb = idx / d;
                            let phys = idx % d;
                            m[[lb * d + phys, 0]]
                        })
                        .collect(),
                )
                .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {e}")))?;
                mps.push(Tensor::new(
                    site,
                    data,
                    vec![
                        format!("bond_{site}"),
                        format!("phys_{site}"),
                        format!("bond_{}", site + 1),
                    ],
                ));
                break;
            }

            // SVD: M = U S V^H.
            let (u, s, vh) = Self::complex_svd(&m)?;

            // Determine kept rank (truncate tiny singular values and cap at max_bond_dim).
            let mut rank = s.len();
            let tol = 1e-12 * s.first().copied().unwrap_or(0.0).max(1.0);
            while rank > 1 && s[rank - 1] <= tol {
                rank -= 1;
            }
            if let Some(max_b) = max_bond_dim {
                rank = rank.min(max_b.max(1));
            }
            rank = rank.max(1);

            // Site tensor A[site] = U[:, :rank] reshaped to (left_bond, d, rank).
            let mut a_data = Array::zeros(IxDyn(&[left_bond, d, rank]));
            for lb in 0..left_bond {
                for phys in 0..d {
                    for r in 0..rank {
                        a_data[[lb, phys, r]] = u[[lb * d + phys, r]];
                    }
                }
            }
            mps.push(Tensor::new(
                site,
                a_data,
                vec![
                    format!("bond_{site}"),
                    format!("phys_{site}"),
                    format!("bond_{}", site + 1),
                ],
            ));

            // Form the remainder S[:rank] * V^H[:rank, :] as the new psi
            // (shape rank x cols), which becomes the next iteration's left part.
            let mut new_psi = vec![Complex64::new(0.0, 0.0); rank * cols];
            for r in 0..rank {
                let sigma = Complex64::new(s[r], 0.0);
                for c in 0..cols {
                    new_psi[r * cols + c] = sigma * vh[[r, c]];
                }
            }
            psi = new_psi;
            left_bond = rank;
        }

        Ok(mps)
    }

    /// Apply a Matrix Product Operator (MPO) to the specified physical qubits.
    ///
    /// Honest status: this `TensorNetwork` stores a general tensor network, not an MPS
    /// state, so there is no canonical MPS chain for an MPO to act on in place. Applying
    /// an MPO correctly requires first bringing the state into MPS form (see
    /// [`Self::to_mps`]) and contracting the operator legs site-by-site. Rather than
    /// silently doing nothing (the previous behaviour), this returns an explicit error.
    pub fn apply_mpo(&mut self, _mpo: &[Tensor], _qubits: &[usize]) -> QuantRS2Result<()> {
        Err(QuantRS2Error::UnsupportedOperation(
            "MPO application requires MPS form; call to_mps first".into(),
        ))
    }

    /// Complex one-sided Jacobi SVD: returns `(U, s, Vᴴ)` with `M = U·diag(s)·Vᴴ`,
    /// `U` (m×k) and `Vᴴ` (k×n) having orthonormal rows/columns and `s` the singular
    /// values in non-increasing order (`k = min(m, n)`).
    ///
    /// One-sided Jacobi rotates pairs of columns of `M` until they are mutually
    /// orthogonal; the column norms are then the singular values and the accumulated
    /// rotations form `V`. The method is numerically robust, handles repeated/zero
    /// singular values gracefully, and works directly on complex data (unlike the
    /// real-only LAPACK path currently exposed by SciRS2).
    fn complex_svd(
        m: &Array2<Complex64>,
    ) -> QuantRS2Result<(Array2<Complex64>, Vec<f64>, Array2<Complex64>)> {
        let (rows, cols) = (m.nrows(), m.ncols());

        // Work on whichever orientation has at least as many rows as columns so that
        // the column-orthogonalisation has full column rank handling; transpose back
        // afterwards if needed.
        let transposed = rows < cols;
        let a0 = if transposed {
            m.mapv(|z| z.conj()).t().to_owned() // (cols x rows)
        } else {
            m.clone()
        };
        let (p, q) = (a0.nrows(), a0.ncols()); // p >= q

        let mut a = a0; // columns will be orthogonalised in place
        let mut v = Array2::<Complex64>::eye(q); // accumulates right rotations

        let max_sweeps = 60;
        let eps = 1e-15;
        for _sweep in 0..max_sweeps {
            let mut off = 0.0_f64;
            for i in 0..q {
                for j in (i + 1)..q {
                    // Compute the 2x2 Hermitian block of A^H A restricted to cols i, j.
                    let mut alpha = 0.0_f64; // <a_i, a_i>
                    let mut beta = 0.0_f64; // <a_j, a_j>
                    let mut gamma = Complex64::new(0.0, 0.0); // <a_i, a_j>
                    for r in 0..p {
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

                    // Jacobi rotation that diagonalises [[alpha, gamma],[gamma*, beta]].
                    // Phase factor to make the off-diagonal real-positive.
                    let phase = gamma / gamma_abs;
                    let zeta = (beta - alpha) / (2.0 * gamma_abs);
                    let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let sgn = c * t; // real sine magnitude
                    let s_ij = phase * Complex64::new(sgn, 0.0);

                    // Apply rotation to columns i, j of A:
                    //   a_i' =  c·a_i - conj(s)·a_j
                    //   a_j' =  s·a_i +      c·a_j
                    for r in 0..p {
                        let ai = a[[r, i]];
                        let aj = a[[r, j]];
                        a[[r, i]] = Complex64::new(c, 0.0) * ai - s_ij.conj() * aj;
                        a[[r, j]] = s_ij * ai + Complex64::new(c, 0.0) * aj;
                    }
                    // Accumulate into V (same rotation on its columns).
                    for r in 0..q {
                        let vi = v[[r, i]];
                        let vj = v[[r, j]];
                        v[[r, i]] = Complex64::new(c, 0.0) * vi - s_ij.conj() * vj;
                        v[[r, j]] = s_ij * vi + Complex64::new(c, 0.0) * vj;
                    }
                }
            }
            if off <= eps {
                break;
            }
        }

        // Singular values are the column norms of the orthogonalised A; U columns are
        // the normalised columns.
        let mut sigma: Vec<(f64, usize)> = (0..q)
            .map(|j| {
                let norm = (0..p).map(|r| a[[r, j]].norm_sqr()).sum::<f64>().sqrt();
                (norm, j)
            })
            .collect();
        // Sort singular values in non-increasing order.
        sigma.sort_by(|x, y| y.0.total_cmp(&x.0));

        let k = q; // number of singular values for the (p x q), p>=q orientation
        let mut u_mat = Array2::<Complex64>::zeros((p, k));
        let mut s_vec = vec![0.0_f64; k];
        let mut v_sorted = Array2::<Complex64>::zeros((q, k));
        for (new_idx, &(norm, old_idx)) in sigma.iter().enumerate() {
            s_vec[new_idx] = norm;
            if norm > 1e-300 {
                for r in 0..p {
                    u_mat[[r, new_idx]] = a[[r, old_idx]] / Complex64::new(norm, 0.0);
                }
            } else {
                // Degenerate/zero column: leave U column zero (its singular value is 0).
                u_mat[[0.min(p - 1), new_idx]] = Complex64::new(0.0, 0.0);
            }
            for r in 0..q {
                v_sorted[[r, new_idx]] = v[[r, old_idx]];
            }
        }

        // Reassemble in the original orientation.
        if transposed {
            // Original M = (a0)^H. With a0 = U_a S V_a^H we get
            // M = V_a S U_a^H, i.e. U_M = V_a, V_M^H = U_a^H.
            let u_m = v_sorted; // (q x k) = (rows? ) ; careful with shapes below
            let vh_m = u_mat.mapv(|z| z.conj()).t().to_owned(); // (k x p)
            Ok((u_m, s_vec, vh_m))
        } else {
            let vh_m = v_sorted.mapv(|z| z.conj()).t().to_owned(); // (k x q)
            Ok((u_mat, s_vec, vh_m))
        }
    }

    /// Get a reference to the tensors in the network
    pub fn tensors(&self) -> Vec<&Tensor> {
        self.tensors.values().collect()
    }

    /// Get a reference to a tensor by ID
    pub fn tensor(&self, id: usize) -> Option<&Tensor> {
        self.tensors.get(&id)
    }
}

/// Builder for quantum circuits as tensor networks
pub struct TensorNetworkBuilder {
    network: TensorNetwork,
    qubit_indices: HashMap<usize, String>,
    current_indices: HashMap<usize, String>,
}

impl TensorNetworkBuilder {
    /// Create a new tensor network builder for n qubits
    pub fn new(num_qubits: usize) -> Self {
        let mut network = TensorNetwork::new();
        let mut qubit_indices = HashMap::new();
        let mut current_indices = HashMap::new();

        // Initialize qubits in |0⟩ state
        for i in 0..num_qubits {
            let idx = format!("q{i}_0");
            let tensor = Tensor::qubit_zero(i, idx.clone());
            network.add_tensor(tensor);
            qubit_indices.insert(i, idx.clone());
            current_indices.insert(i, idx);
        }

        Self {
            network,
            qubit_indices,
            current_indices,
        }
    }

    /// Apply a single-qubit gate
    pub fn apply_single_qubit_gate(
        &mut self,
        gate: &dyn GateOp,
        qubit: usize,
    ) -> QuantRS2Result<()> {
        let matrix_vec = gate.matrix()?;
        let matrix = Array2::from_shape_vec((2, 2), matrix_vec)
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {e}")))?;

        // Create gate tensor
        let in_idx = self.current_indices[&qubit].clone();
        let out_idx = format!("q{}_{}", qubit, self.network.next_id);
        let gate_tensor = Tensor::from_matrix(
            self.network.next_id,
            matrix,
            in_idx.clone(),
            out_idx.clone(),
        );

        // Add to network
        let gate_id = self.network.add_tensor(gate_tensor);

        // Connect to previous tensor on this qubit
        if let Some(prev_tensor) = self.find_tensor_with_index(&in_idx) {
            self.network
                .connect(prev_tensor, in_idx.clone(), gate_id, in_idx)?;
        }

        // Update current index
        self.current_indices.insert(qubit, out_idx);

        Ok(())
    }

    /// Apply a two-qubit gate
    pub fn apply_two_qubit_gate(
        &mut self,
        gate: &dyn GateOp,
        qubit1: usize,
        qubit2: usize,
    ) -> QuantRS2Result<()> {
        let matrix_vec = gate.matrix()?;
        let matrix = Array2::from_shape_vec((4, 4), matrix_vec)
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {e}")))?;

        // Reshape to rank-4 tensor
        let tensor_data = matrix
            .into_shape_with_order((2, 2, 2, 2))
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {e}")))?
            .into_dyn();

        // Create indices
        let in1_idx = self.current_indices[&qubit1].clone();
        let in2_idx = self.current_indices[&qubit2].clone();
        let out1_idx = format!("q{}_{}", qubit1, self.network.next_id);
        let out2_idx = format!("q{}_{}", qubit2, self.network.next_id);

        let gate_tensor = Tensor::new(
            self.network.next_id,
            tensor_data,
            vec![
                in1_idx.clone(),
                in2_idx.clone(),
                out1_idx.clone(),
                out2_idx.clone(),
            ],
        );

        // Add to network
        let gate_id = self.network.add_tensor(gate_tensor);

        // Connect to previous tensors
        if let Some(prev1) = self.find_tensor_with_index(&in1_idx) {
            self.network
                .connect(prev1, in1_idx.clone(), gate_id, in1_idx)?;
        }
        if let Some(prev2) = self.find_tensor_with_index(&in2_idx) {
            self.network
                .connect(prev2, in2_idx.clone(), gate_id, in2_idx)?;
        }

        // Update current indices
        self.current_indices.insert(qubit1, out1_idx);
        self.current_indices.insert(qubit2, out2_idx);

        Ok(())
    }

    /// Find tensor that has the given index as output
    fn find_tensor_with_index(&self, index: &str) -> Option<usize> {
        for (id, tensor) in &self.network.tensors {
            if tensor.indices.iter().any(|idx| idx == index) {
                return Some(*id);
            }
        }
        None
    }

    /// Build the final tensor network
    pub fn build(self) -> TensorNetwork {
        self.network
    }

    /// Contract the network and return the quantum state
    #[must_use]
    pub fn to_statevector(&mut self) -> QuantRS2Result<Vec<Complex64>> {
        let final_tensor = self.network.contract_all()?;
        Ok(final_tensor.data.into_raw_vec_and_offset().0)
    }
}

/// Quantum circuit simulation using tensor networks
pub struct TensorNetworkSimulator {
    /// Maximum bond dimension for MPS
    max_bond_dim: usize,
    /// Use SVD compression
    use_compression: bool,
    /// Parallelization threshold
    parallel_threshold: usize,
}

impl TensorNetworkSimulator {
    /// Create a new tensor network simulator
    pub const fn new() -> Self {
        Self {
            max_bond_dim: 64,
            use_compression: true,
            parallel_threshold: 1000,
        }
    }

    /// Set maximum bond dimension
    #[must_use]
    pub const fn with_max_bond_dim(mut self, dim: usize) -> Self {
        self.max_bond_dim = dim;
        self
    }

    /// Enable or disable compression
    #[must_use]
    pub const fn with_compression(mut self, compress: bool) -> Self {
        self.use_compression = compress;
        self
    }

    /// Simulate a quantum circuit
    pub fn simulate<const N: usize>(
        &self,
        gates: &[Box<dyn GateOp>],
    ) -> QuantRS2Result<Register<N>> {
        let mut builder = TensorNetworkBuilder::new(N);

        // Apply gates
        for gate in gates {
            let qubits = gate.qubits();
            match qubits.len() {
                1 => builder.apply_single_qubit_gate(gate.as_ref(), qubits[0].0 as usize)?,
                2 => builder.apply_two_qubit_gate(
                    gate.as_ref(),
                    qubits[0].0 as usize,
                    qubits[1].0 as usize,
                )?,
                _ => {
                    return Err(QuantRS2Error::UnsupportedOperation(format!(
                        "Gates with {} qubits not supported in tensor network",
                        qubits.len()
                    )))
                }
            }
        }

        // Contract to get statevector
        let amplitudes = builder.to_statevector()?;
        Register::with_amplitudes(amplitudes)
    }
}

/// Optimized contraction strategies
pub mod contraction_optimization {
    use super::*;

    /// Dynamic programming algorithm for optimal contraction order
    pub struct DynamicProgrammingOptimizer {
        memo: HashMap<Vec<usize>, (usize, Vec<(usize, usize)>)>,
    }

    impl DynamicProgrammingOptimizer {
        pub fn new() -> Self {
            Self {
                memo: HashMap::new(),
            }
        }

        /// Find optimal contraction order using dynamic programming
        pub fn optimize(&mut self, network: &TensorNetwork) -> Vec<(usize, usize)> {
            let tensor_ids: Vec<_> = network.tensors.keys().copied().collect();
            self.find_optimal_order(&tensor_ids, network).1
        }

        fn find_optimal_order(
            &mut self,
            tensors: &[usize],
            network: &TensorNetwork,
        ) -> (usize, Vec<(usize, usize)>) {
            if tensors.len() <= 1 {
                return (0, vec![]);
            }

            let key = tensors.to_vec();
            if let Some(result) = self.memo.get(&key) {
                return result.clone();
            }

            let mut best_cost = usize::MAX;
            let mut best_order = vec![];

            // Try all possible pairings
            for i in 0..tensors.len() {
                for j in (i + 1)..tensors.len() {
                    // Check if tensors are connected
                    if self.are_connected(tensors[i], tensors[j], network) {
                        let cost = network.estimate_contraction_cost(tensors[i], tensors[j]);

                        // Remaining tensors after contraction
                        let mut remaining = vec![];
                        for (k, &t) in tensors.iter().enumerate() {
                            if k != i && k != j {
                                remaining.push(t);
                            }
                        }
                        remaining.push(network.next_id + remaining.len()); // Virtual tensor

                        let (sub_cost, sub_order) = self.find_optimal_order(&remaining, network);
                        let total_cost = cost + sub_cost;

                        if total_cost < best_cost {
                            best_cost = total_cost;
                            best_order = vec![(tensors[i], tensors[j])];
                            best_order.extend(sub_order);
                        }
                    }
                }
            }

            self.memo.insert(key, (best_cost, best_order.clone()));
            (best_cost, best_order)
        }

        fn are_connected(&self, t1: usize, t2: usize, network: &TensorNetwork) -> bool {
            network.edges.iter().any(|e| {
                (e.tensor1 == t1 && e.tensor2 == t2) || (e.tensor1 == t2 && e.tensor2 == t1)
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let data = ArrayD::zeros(IxDyn(&[2, 2]));
        let tensor = Tensor::new(0, data, vec!["in".to_string(), "out".to_string()]);
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.shape, vec![2, 2]);
    }

    #[test]
    fn test_qubit_tensors() {
        let t0 = Tensor::qubit_zero(0, "q0".to_string());
        assert_eq!(t0.data[[0]], Complex64::new(1.0, 0.0));
        assert_eq!(t0.data[[1]], Complex64::new(0.0, 0.0));

        let t1 = Tensor::qubit_one(1, "q1".to_string());
        assert_eq!(t1.data[[0]], Complex64::new(0.0, 0.0));
        assert_eq!(t1.data[[1]], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_tensor_network_builder() {
        let builder = TensorNetworkBuilder::new(2);
        assert_eq!(builder.network.tensors.len(), 2);
    }

    /// Contract a returned MPS chain back into the full dense tensor (row-major flat
    /// vector over the physical indices). Each site tensor has indices
    /// `[bond_left, phys, bond_right]` with boundary bonds of dimension 1.
    fn contract_mps(mps: &[Tensor]) -> Vec<Complex64> {
        // psi is a flat (left_bond x phys_so_far) buffer; start with left_bond = 1 and
        // a single scalar 1.0.
        let mut acc: Vec<Complex64> = vec![Complex64::new(1.0, 0.0)];
        let mut left_bond = 1usize;
        for t in mps {
            let lb = t.shape[0];
            let d = t.shape[1];
            let rb = t.shape[2];
            assert_eq!(lb, left_bond, "bond mismatch while contracting MPS");
            let cols = acc.len() / left_bond; // physical entries accumulated so far
                                              // new_acc has shape (rb x (cols*d)) flattened row-major over [rb, cols, d].
            let mut new_acc = vec![Complex64::new(0.0, 0.0); rb * cols * d];
            for r in 0..rb {
                for cidx in 0..cols {
                    for phys in 0..d {
                        let mut sum = Complex64::new(0.0, 0.0);
                        for l in 0..lb {
                            // acc indexed row-major over [l, cidx]
                            let a_val = acc[l * cols + cidx];
                            let t_val = t.data[[l, phys, r]];
                            sum += a_val * t_val;
                        }
                        new_acc[(r * cols + cidx) * d + phys] = sum;
                    }
                }
            }
            acc = new_acc;
            left_bond = rb;
        }
        acc
    }

    /// Build a single-tensor network wrapping a known statevector `amps` over
    /// `n` qubits (shape `[2; n]`). `contract_all` returns such a single tensor
    /// verbatim (deterministically), so this isolates the MPS decomposition from the
    /// network contraction engine.
    fn single_tensor_network(amps: Vec<Complex64>, n: usize) -> TensorNetwork {
        let shape: Vec<usize> = vec![2usize; n];
        let data = Array::from_shape_vec(IxDyn(&shape), amps).expect("state tensor");
        let indices: Vec<String> = (0..n).map(|i| format!("phys_{i}")).collect();
        let mut net = TensorNetwork::new();
        net.add_tensor(Tensor::new(0, data, indices));
        net
    }

    #[test]
    fn test_to_mps_reconstructs_bell_state() {
        // Site-6 proof: to_mps reconstructs an entangled (bond-dim-2) Bell state.
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let bell = vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ];
        let net = single_tensor_network(bell.clone(), 2);
        let mps = net.to_mps(None).expect("to_mps");
        assert_eq!(mps.len(), 2, "one MPS tensor per qubit");
        // Genuine entanglement => inner bond dimension 2.
        assert_eq!(mps[0].shape[2], 2, "Bell state needs bond dimension 2");

        let recon = contract_mps(&mps);
        let err: f64 = recon
            .iter()
            .zip(bell.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-9, "Bell MPS reconstruction error {err}");
    }

    #[test]
    fn test_to_mps_reconstructs_ghz_state() {
        // 3-qubit GHZ = (|000> + |111>)/sqrt2.
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let mut ghz = vec![Complex64::new(0.0, 0.0); 8];
        ghz[0] = Complex64::new(inv_sqrt2, 0.0);
        ghz[7] = Complex64::new(inv_sqrt2, 0.0);
        let net = single_tensor_network(ghz.clone(), 3);
        let mps = net.to_mps(None).expect("to_mps");
        assert_eq!(mps.len(), 3, "one MPS tensor per qubit");

        let recon = contract_mps(&mps);
        let err: f64 = recon
            .iter()
            .zip(ghz.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-9, "GHZ MPS reconstruction error {err}");
    }

    #[test]
    fn test_to_mps_reconstructs_generic_state() {
        // A generic normalised 2-qubit complex state (no special structure).
        let raw = [
            Complex64::new(0.3, 0.1),
            Complex64::new(-0.2, 0.4),
            Complex64::new(0.5, -0.25),
            Complex64::new(0.1, 0.35),
        ];
        let norm = raw.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let state: Vec<Complex64> = raw.iter().map(|z| z / norm).collect();
        let net = single_tensor_network(state.clone(), 2);
        let mps = net.to_mps(None).expect("to_mps");
        let recon = contract_mps(&mps);
        let err: f64 = recon
            .iter()
            .zip(state.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-9, "generic MPS reconstruction error {err}");
    }

    #[test]
    fn test_to_mps_truncation_keeps_bond_dim() {
        // With max_bond_dim = 1 the Bell state cannot be represented exactly, but the
        // call must still succeed and cap every bond dimension at 1 (lossy truncation),
        // proving the truncation path is real (not ignored).
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let bell = vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ];
        let net = single_tensor_network(bell, 2);
        let mps = net.to_mps(Some(1)).expect("to_mps truncated");
        for t in &mps {
            assert!(
                t.shape[0] <= 1 && t.shape[2] <= 1,
                "bond dimension exceeded max_bond_dim=1: {:?}",
                t.shape
            );
        }
    }

    #[test]
    fn test_apply_mpo_honest_error() {
        // Site-6: apply_mpo must report an honest error rather than silently no-op.
        let mut network = TensorNetwork::new();
        let result = network.apply_mpo(&[], &[0]);
        assert!(matches!(
            result,
            Err(QuantRS2Error::UnsupportedOperation(_))
        ));
    }

    #[test]
    fn test_complex_svd_roundtrip() {
        // Validate the complex Jacobi SVD: M ≈ U diag(s) V^H with orthonormal factors.
        let m = Array2::from_shape_vec(
            (3, 2),
            vec![
                Complex64::new(1.0, 0.5),
                Complex64::new(-0.3, 0.2),
                Complex64::new(0.4, -0.1),
                Complex64::new(0.7, 0.0),
                Complex64::new(-0.2, 0.9),
                Complex64::new(0.1, 0.1),
            ],
        )
        .expect("matrix");
        let (u, s, vh) = TensorNetwork::complex_svd(&m).expect("svd");
        // Reconstruct.
        let k = s.len();
        let mut s_mat = Array2::<Complex64>::zeros((k, k));
        for i in 0..k {
            s_mat[[i, i]] = Complex64::new(s[i], 0.0);
        }
        let recon = u.dot(&s_mat).dot(&vh);
        let err: f64 = recon
            .iter()
            .zip(m.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-9, "complex SVD reconstruction error {err}");
        // Singular values non-increasing and non-negative.
        for i in 1..s.len() {
            assert!(s[i] <= s[i - 1] + 1e-12);
            assert!(s[i] >= -1e-12);
        }
    }

    #[test]
    fn test_network_connection() {
        let mut network = TensorNetwork::new();

        let t1 = Tensor::qubit_zero(0, "q0".to_string());
        let t2 = Tensor::qubit_zero(1, "q1".to_string());

        let id1 = network.add_tensor(t1);
        let id2 = network.add_tensor(t2);

        // Should fail - indices don't exist on these tensors
        assert!(network
            .connect(id1, "bond".to_string(), id2, "bond".to_string())
            .is_err());
    }
}
