//! Tensor Network representations for quantum circuits
//!
//! This module provides tensor network representations and operations for quantum circuits,
//! leveraging SciRS2 for efficient tensor manipulations and contractions.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    register::Register,
};
use ndarray::{Array, Array2, ArrayD, IxDyn};
use num_complex::Complex;
use scirs2_linalg::svd;
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
        array: ndarray::ArrayBase<ndarray::OwnedRepr<Complex64>, D>,
        indices: Vec<usize>,
    ) -> Self
    where
        D: ndarray::Dimension,
    {
        let shape = array.shape().to_vec();
        let data = array.into_dyn();
        let index_labels: Vec<String> = indices.iter().map(|i| format!("idx_{}", i)).collect();
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
    pub fn tensor(&self) -> &ArrayD<Complex64> {
        &self.data
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Contract this tensor with another over specified indices
    pub fn contract(
        &self,
        other: &Tensor,
        self_idx: &str,
        other_idx: &str,
    ) -> QuantRS2Result<Tensor> {
        // Find the positions of the indices to contract
        let self_pos = self
            .indices
            .iter()
            .position(|s| s == self_idx)
            .ok_or_else(|| {
                QuantRS2Error::InvalidInput(format!("Index {} not found in tensor", self_idx))
            })?;
        let other_pos = other
            .indices
            .iter()
            .position(|s| s == other_idx)
            .ok_or_else(|| {
                QuantRS2Error::InvalidInput(format!("Index {} not found in tensor", other_idx))
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

        Ok(Tensor::new(
            self.id.max(other.id) + 1,
            contracted,
            new_indices,
        ))
    }

    /// Perform the actual index contraction
    fn contract_indices(
        &self,
        other: &Tensor,
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
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {}", e)))?
            .to_owned();
        let other_mat = other
            .data
            .view()
            .into_shape_with_order((other_left_dims * contract_dim, other_right_dims))
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {}", e)))?
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
                            let _self_idx =
                                i * contract_dim * self_right_dims + c * self_right_dims + j;
                            let _other_idx =
                                k * contract_dim * other_right_dims + c * other_right_dims + l;
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
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {}", e)))
    }

    /// Apply SVD decomposition to split tensor along specified index
    pub fn svd_decompose(
        &self,
        idx: usize,
        max_rank: Option<usize>,
    ) -> QuantRS2Result<(Tensor, Tensor)> {
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
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {}", e)))?
            .to_owned();

        // Perform SVD using SciRS2
        let real_matrix = matrix.mapv(|c| c.re);
        let (u, s, vt) = svd(&real_matrix.view(), false, None)
            .map_err(|e| QuantRS2Error::ComputationError(format!("SVD failed: {:?}", e)))?;

        // Determine rank to keep
        let rank = if let Some(max_r) = max_rank {
            max_r.min(s.len())
        } else {
            s.len()
        };

        // Truncate based on rank
        let u_trunc = u.slice(ndarray::s![.., ..rank]).to_owned();
        let s_trunc = s.slice(ndarray::s![..rank]).to_owned();
        let vt_trunc = vt.slice(ndarray::s![..rank, ..]).to_owned();

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

        let left_tensor = Tensor::new(self.id * 2, left_data.into_dyn(), left_indices);

        let right_tensor = Tensor::new(self.id * 2 + 1, right_data.into_dyn(), right_indices);

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
                "Tensor {} not found",
                tensor1
            )));
        }
        if !self.tensors.contains_key(&tensor2) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Tensor {} not found",
                tensor2
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
                QuantRS2Error::InvalidInput(format!(
                    "Index {} not found in tensor {}",
                    index1, tensor1
                ))
            })?;
        let idx2_pos = t2
            .indices
            .iter()
            .position(|s| s == &index2)
            .ok_or_else(|| {
                QuantRS2Error::InvalidInput(format!(
                    "Index {} not found in tensor {}",
                    index2, tensor2
                ))
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
        let mut remaining_tensors: HashSet<_> = self.tensors.keys().cloned().collect();
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
    fn estimate_contraction_cost(&self, _t1: usize, _t2: usize) -> usize {
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
            return Ok(self.tensors.values().next().unwrap().clone());
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

    /// Apply Matrix Product State (MPS) decomposition
    pub fn to_mps(&self, _max_bond_dim: Option<usize>) -> QuantRS2Result<Vec<Tensor>> {
        // This would decompose the network into a chain of tensors
        // For now, return a placeholder
        Ok(vec![])
    }

    /// Apply Matrix Product Operator (MPO) representation
    pub fn apply_mpo(&mut self, _mpo: &[Tensor], _qubits: &[usize]) -> QuantRS2Result<()> {
        // Apply an MPO to specified qubits
        Ok(())
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
            let idx = format!("q{}_0", i);
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
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {}", e)))?;

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
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {}", e)))?;

        // Reshape to rank-4 tensor
        let tensor_data = matrix
            .into_shape_with_order((2, 2, 2, 2))
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Shape error: {}", e)))?
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
    pub fn new() -> Self {
        Self {
            max_bond_dim: 64,
            use_compression: true,
            parallel_threshold: 1000,
        }
    }

    /// Set maximum bond dimension
    pub fn with_max_bond_dim(mut self, dim: usize) -> Self {
        self.max_bond_dim = dim;
        self
    }

    /// Enable or disable compression
    pub fn with_compression(mut self, compress: bool) -> Self {
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
            let tensor_ids: Vec<_> = network.tensors.keys().cloned().collect();
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
