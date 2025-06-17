//! Matrix Product State (MPS) quantum simulator
//!
//! This module implements an efficient quantum simulator using the Matrix Product State
//! representation, which is particularly effective for simulating quantum systems with
//! limited entanglement.

use ndarray::{s, Array1, Array2, Array3, ArrayView2};
use num_complex::Complex64;
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    prelude::QubitId,
    register::Register,
};

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
    pub fn set_truncation_threshold(&mut self, threshold: f64) {
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
            let idx = if bit { 1 } else { 0 };

            // Extract the matrix for this bit value
            let matrix = tensor.data.slice(s![.., idx, ..]);
            result = result.dot(&matrix);
        }

        Ok(result[[0, 0]])
    }

    /// Sample from the MPS
    pub fn sample(&self) -> Vec<bool> {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();
        let mut result = vec![false; self.num_qubits];
        let mut accumulated_matrix = Array2::eye(1);

        for i in 0..self.num_qubits {
            let tensor = &self.tensors[i];

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

            if rng.gen::<f64>() < threshold {
                result[i] = false;
                accumulated_matrix = temp0;
            } else {
                result[i] = true;
                accumulated_matrix = temp1;
            }
        }

        result
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
            v = v - &(proj * &q.column(i).to_owned());
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

/// SVD decomposition with truncation
fn svd_decomposition(
    matrix: &Array2<Complex64>,
    max_bond: usize,
    threshold: f64,
) -> QuantRS2Result<(Array2<Complex64>, Array1<f64>, Array2<Complex64>)> {
    // Placeholder - in real implementation would use proper SVD
    // For now, return identity-like decomposition
    let (m, n) = matrix.dim();
    let k = m.min(n).min(max_bond);

    let u = Array2::eye(m).slice(s![.., ..k]).to_owned();
    let s = Array1::ones(k);
    let vt = Array2::eye(n).slice(s![..k, ..]).to_owned();

    Ok((u, s, vt))
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
    pub fn new(max_bond_dimension: usize) -> Self {
        Self {
            max_bond_dimension,
            truncation_threshold: 1e-10,
        }
    }

    /// Set the truncation threshold
    pub fn set_truncation_threshold(&mut self, threshold: f64) {
        self.truncation_threshold = threshold;
    }
}

impl<const N: usize> Simulator<N> for MPSSimulator {
    fn run(&self, circuit: &Circuit<N>) -> QuantRS2Result<Register<N>> {
        // Create initial MPS state
        let mut mps = MPS::new(N, self.max_bond_dimension);
        mps.set_truncation_threshold(self.truncation_threshold);

        // Get gate sequence from circuit
        // Note: This is a placeholder - would need actual circuit introspection
        // For now, return a register in |0> state
        Ok(Register::new())
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

        mps.apply_single_qubit_gate(&h, 0).unwrap();

        // Check amplitudes
        let amp0 = mps.get_amplitude(&[false]).unwrap();
        let amp1 = mps.get_amplitude(&[true]).unwrap();

        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((amp0.re - expected).abs() < 1e-10);
        assert!((amp1.re - expected).abs() < 1e-10);
    }

    #[test]
    fn test_orthogonality_center() {
        let mut mps = MPS::new(5, 10);

        mps.move_orthogonality_center(2).unwrap();
        assert_eq!(mps.orthogonality_center, 2);

        mps.move_orthogonality_center(4).unwrap();
        assert_eq!(mps.orthogonality_center, 4);

        mps.move_orthogonality_center(0).unwrap();
        assert_eq!(mps.orthogonality_center, 0);
    }
}
