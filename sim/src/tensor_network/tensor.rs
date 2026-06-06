//! Tensor representation for quantum states and operations
//!
//! This module provides a tensor-based representation for quantum states
//! and operations used in the tensor network simulator.

use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::{Array, Array1, Array2, ArrayD, Dimension, IxDyn};
use scirs2_core::Complex64;

/// A tensor representing a quantum state or operation
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The tensor data
    pub data: ArrayD<Complex64>,

    /// The tensor rank (number of indices)
    pub rank: usize,

    /// The dimensions of each index
    pub dimensions: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor from a multi-dimensional array
    pub fn new(data: ArrayD<Complex64>) -> Self {
        let dimensions = data.shape().to_vec();
        let rank = dimensions.len();

        Self {
            data,
            rank,
            dimensions,
        }
    }

    /// Create a tensor from a matrix (gate)
    pub fn from_matrix(matrix: &[Complex64], dim: usize) -> Self {
        // Determine the shape based on the matrix size and dimension
        let _n = (matrix.len() as f64).sqrt() as usize;

        // Reshape the matrix into a multi-dimensional array
        let mut shape = Vec::new();
        for _ in 0..dim {
            shape.push(2); // Each qubit has dimension 2
        }

        // Create the tensor data
        let mut data = ArrayD::zeros(IxDyn(&shape));

        // Fill the tensor with matrix elements
        let flat_data = data
            .as_slice_mut()
            .expect("Tensor data should be contiguous in memory");
        for (i, val) in matrix.iter().enumerate() {
            if i < flat_data.len() {
                flat_data[i] = *val;
            }
        }

        Self::new(data)
    }

    /// Create a tensor representing the |0⟩ state
    pub fn qubit_zero() -> Self {
        let data = Array::from_shape_vec(
            IxDyn(&[2]),
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        )
        .expect("Valid shape for qubit |0> state");

        Self::new(data)
    }

    /// Create a tensor representing the |1⟩ state
    pub fn qubit_one() -> Self {
        let data = Array::from_shape_vec(
            IxDyn(&[2]),
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        )
        .expect("Valid shape for qubit |1> state");

        Self::new(data)
    }

    /// Create a tensor representing the |+⟩ state
    pub fn qubit_plus() -> Self {
        let data = Array::from_shape_vec(
            IxDyn(&[2]),
            vec![
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            ],
        )
        .expect("Valid shape for qubit |+> state");

        Self::new(data)
    }

    /// Contract this tensor with another tensor along specified axes.
    ///
    /// Performs the Einstein summation over one pair of indices:
    ///   result[i₀,…,iₙ₋₁, j₀,…,jₘ₋₁] = Σₖ self[…k…] * other[…k…]
    /// where `k` runs over `self.dimensions[self_axis]` (= `other.dimensions[other_axis]`).
    ///
    /// The output shape is `self.dimensions` with `self_axis` removed, followed by
    /// `other.dimensions` with `other_axis` removed.
    pub fn contract(
        &self,
        other: &Self,
        self_axis: usize,
        other_axis: usize,
    ) -> QuantRS2Result<Self> {
        // Validate axis indices
        if self_axis >= self.rank || other_axis >= other.rank {
            return Err(QuantRS2Error::CircuitValidationFailed(format!(
                "Invalid contraction axes: {self_axis} and {other_axis}"
            )));
        }

        // Validate axis dimensions match (contraction is only valid when dims agree)
        if self.dimensions[self_axis] != other.dimensions[other_axis] {
            return Err(QuantRS2Error::CircuitValidationFailed(format!(
                "Mismatched dimensions for contraction: {} and {}",
                self.dimensions[self_axis], other.dimensions[other_axis]
            )));
        }

        let _contract_dim = self.dimensions[self_axis];

        // Build the result dimensions:
        //   all self dims except self_axis, then all other dims except other_axis
        let self_outer_dims: Vec<usize> = self
            .dimensions
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != self_axis)
            .map(|(_, &d)| d)
            .collect();
        let other_outer_dims: Vec<usize> = other
            .dimensions
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != other_axis)
            .map(|(_, &d)| d)
            .collect();

        let mut result_dims = self_outer_dims.clone();
        result_dims.extend_from_slice(&other_outer_dims);

        // For scalar output (both tensors were rank-1 vectors)
        let result_is_scalar = result_dims.is_empty();

        let result_shape = if result_is_scalar {
            IxDyn(&[1usize])
        } else {
            IxDyn(result_dims.as_slice())
        };

        let mut result_data = ArrayD::zeros(result_shape);

        // Perform contraction via explicit index iteration.
        // This is O(N_self * N_other) but is simple and correct for any rank,
        // which is appropriate for small quantum-circuit tensors (dim 2–16).
        for (self_idx, self_val) in self.data.indexed_iter() {
            let self_raw = self_idx.slice();
            let k = self_raw[self_axis];

            // Build the partial result index from self (excluding self_axis)
            let self_outer_idx: Vec<usize> = self_raw
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != self_axis)
                .map(|(_, &v)| v)
                .collect();

            for (other_idx, other_val) in other.data.indexed_iter() {
                let other_raw = other_idx.slice();
                if other_raw[other_axis] != k {
                    continue;
                }

                // Build the partial result index from other (excluding other_axis)
                let other_outer_idx: Vec<usize> = other_raw
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != other_axis)
                    .map(|(_, &v)| v)
                    .collect();

                // Concatenate to get full result index
                let mut res_idx = self_outer_idx.clone();
                res_idx.extend_from_slice(&other_outer_idx);

                let target = if result_is_scalar {
                    &mut result_data[IxDyn(&[0usize])]
                } else {
                    &mut result_data[IxDyn(res_idx.as_slice())]
                };
                *target += *self_val * *other_val;
            }
        }

        // Unwrap scalar result back to empty shape
        let final_data = if result_is_scalar {
            let scalar_val = result_data[IxDyn(&[0usize])];
            ArrayD::from_elem(IxDyn(&[]), scalar_val)
        } else {
            result_data
        };

        let result_rank = result_dims.len();
        Ok(Self {
            data: final_data,
            dimensions: result_dims,
            rank: result_rank,
        })
    }

    /// Perform SVD decomposition on this tensor, splitting it into two lower-rank tensors.
    ///
    /// The tensor is logically reshaped into a matrix by grouping `left_axes` into rows
    /// and `right_axes` into columns. The SVD is then computed and the result is split
    /// into two tensors:
    ///
    /// - `left_tensor`: shape `(*left_dims, bond_dim)` — absorbs U * diag(S)
    /// - `right_tensor`: shape `(bond_dim, *right_dims)` — contains Vᴴ
    ///
    /// `max_bond_dim` caps how many singular values are kept (bond dimension).
    pub fn svd(
        &self,
        left_axes: &[usize],
        right_axes: &[usize],
        max_bond_dim: usize,
    ) -> QuantRS2Result<(Self, Self)> {
        use scirs2_core::ndarray::ndarray_linalg::SVD;

        // ---- validation -------------------------------------------------------
        let total_axes = left_axes.len() + right_axes.len();
        if total_axes != self.rank {
            return Err(QuantRS2Error::CircuitValidationFailed(format!(
                "SVD: left_axes ({}) + right_axes ({}) must equal tensor rank ({})",
                left_axes.len(),
                right_axes.len(),
                self.rank
            )));
        }
        // Check no duplicates and all in range
        {
            let mut seen = vec![false; self.rank];
            for &ax in left_axes.iter().chain(right_axes.iter()) {
                if ax >= self.rank {
                    return Err(QuantRS2Error::CircuitValidationFailed(format!(
                        "SVD: axis {ax} out of range for rank-{} tensor",
                        self.rank
                    )));
                }
                if seen[ax] {
                    return Err(QuantRS2Error::CircuitValidationFailed(format!(
                        "SVD: duplicate axis {ax}"
                    )));
                }
                seen[ax] = true;
            }
        }
        if max_bond_dim == 0 {
            return Err(QuantRS2Error::CircuitValidationFailed(
                "SVD: max_bond_dim must be >= 1".to_string(),
            ));
        }

        // ---- compute row/col sizes for the reshaped matrix --------------------
        let left_dims: Vec<usize> = left_axes.iter().map(|&ax| self.dimensions[ax]).collect();
        let right_dims: Vec<usize> = right_axes.iter().map(|&ax| self.dimensions[ax]).collect();

        let left_size: usize = left_dims.iter().product::<usize>().max(1);
        let right_size: usize = right_dims.iter().product::<usize>().max(1);

        // ---- permute and reshape to matrix (left_size, right_size) ------------
        // Build a permutation: left_axes first, then right_axes
        let permutation: Vec<usize> = left_axes.iter().chain(right_axes.iter()).copied().collect();

        // Collect self.data into standard layout after permuting axes
        let perm_data: ArrayD<Complex64> = {
            // permuted_axes on a dynamic array view returns a view with reordered axes
            let view = self.data.view();
            let permuted = view.permuted_axes(permutation.as_slice());
            // Force into owned contiguous array (standard layout)
            permuted.as_standard_layout().into_owned()
        };

        // Reshape to 2D matrix by collecting the permuted data into a flat vec,
        // then building an Array2.  This approach avoids ndarray dimensionality
        // conversion subtleties with IxDyn vs. Ix2.
        let flat: Vec<Complex64> = perm_data.into_raw_vec_and_offset().0;
        let matrix: Array2<Complex64> = Array2::from_shape_vec((left_size, right_size), flat)
            .map_err(|e| {
                QuantRS2Error::CircuitValidationFailed(format!("SVD reshape to matrix failed: {e}"))
            })?;

        // ---- SVD via OxiBLAS/ndarray_linalg -----------------------------------
        // SVD trait: (U, S, Vt) where U is (m,k), S is (k,), Vt is (k,n)
        // with compute_u=true, compute_vt=true and thin=true (economy SVD)
        let (u_full, s_full, vt_full) = matrix.svd(true, true).map_err(|e| {
            QuantRS2Error::CircuitValidationFailed(format!("SVD computation failed: {e}"))
        })?;

        // ---- truncation -------------------------------------------------------
        let rank_cap = left_size.min(right_size);
        let bond_dim = max_bond_dim.min(rank_cap).min(s_full.len());
        let bond_dim = bond_dim.max(1);

        // Keep only the top `bond_dim` singular triplets
        let s_trunc: Array1<f64> = s_full
            .slice(scirs2_core::ndarray::s![..bond_dim])
            .to_owned();
        let u_trunc: Array2<Complex64> = u_full
            .slice(scirs2_core::ndarray::s![.., ..bond_dim])
            .to_owned();
        let vt_trunc: Array2<Complex64> = vt_full
            .slice(scirs2_core::ndarray::s![..bond_dim, ..])
            .to_owned();

        // ---- build left tensor: U * diag(S), shape (*left_dims, bond_dim) ----
        // Absorb singular values into U columns
        let mut us: Array2<Complex64> = u_trunc;
        for j in 0..bond_dim {
            let sigma = Complex64::new(s_trunc[j], 0.0);
            for i in 0..left_size {
                us[[i, j]] *= sigma;
            }
        }

        let mut left_shape = left_dims.clone();
        left_shape.push(bond_dim);
        // Flatten us to a vec, then rebuild as ArrayD with the desired shape.
        // OxiBLAS returns U/Vᴴ as column-major (Fortran) arrays, so we must
        // flatten in logical row-major order (not raw memory order) to match
        // `Array::from_shape_vec`, which interprets the vec as row-major.
        let us_flat: Vec<Complex64> = us.as_standard_layout().iter().copied().collect();
        let left_data: ArrayD<Complex64> =
            Array::from_shape_vec(IxDyn(left_shape.as_slice()), us_flat).map_err(|e| {
                QuantRS2Error::CircuitValidationFailed(format!("SVD left reshape failed: {e}"))
            })?;
        let left_rank = left_shape.len();

        // ---- build right tensor: Vᴴ, shape (bond_dim, *right_dims) -----------
        let mut right_shape = vec![bond_dim];
        right_shape.extend_from_slice(&right_dims);
        // Same column-major → row-major fix as for the left tensor above.
        let vt_flat: Vec<Complex64> = vt_trunc.as_standard_layout().iter().copied().collect();
        let right_data: ArrayD<Complex64> =
            Array::from_shape_vec(IxDyn(right_shape.as_slice()), vt_flat).map_err(|e| {
                QuantRS2Error::CircuitValidationFailed(format!("SVD right reshape failed: {e}"))
            })?;
        let right_rank = right_shape.len();

        let left_tensor = Self {
            data: left_data,
            dimensions: left_shape,
            rank: left_rank,
        };
        let right_tensor = Self {
            data: right_data,
            dimensions: right_shape,
            rank: right_rank,
        };

        Ok((left_tensor, right_tensor))
    }
}

/// A reference to a specific tensor and one of its indices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorIndex {
    /// The ID of the tensor
    pub tensor_id: usize,

    /// The index within the tensor
    pub index: usize,
}
