use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::parallel_ops::{
    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use scirs2_core::Complex64;

use quantrs2_core::qubit::QubitId;

/// Calculate the kronecker product of two matrices
#[must_use]
pub fn kron(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let rows = a_shape[0] * b_shape[0];
    let cols = a_shape[1] * b_shape[1];

    let mut result = Array2::zeros((rows, cols));

    for i in 0..a_shape[0] {
        for j in 0..a_shape[1] {
            let a_val = a[[i, j]];
            let i_offset = i * b_shape[0];
            let j_offset = j * b_shape[1];

            for k in 0..b_shape[0] {
                for l in 0..b_shape[1] {
                    result[[i_offset + k, j_offset + l]] = a_val * b[[k, l]];
                }
            }
        }
    }

    result
}

/// Calculate the tensor product of two state vectors
#[must_use]
pub fn tensor_product(a: &Array1<Complex64>, b: &Array1<Complex64>) -> Array1<Complex64> {
    let a_len = a.len();
    let b_len = b.len();
    let result_len = a_len * b_len;

    let mut result = Array1::zeros(result_len);

    for i in 0..a_len {
        let a_val = a[i];
        let offset = i * b_len;

        for j in 0..b_len {
            result[offset + j] = a_val * b[j];
        }
    }

    result
}

/// Calculate the bit representation of an integer
#[must_use]
pub fn int_to_bits(n: usize, num_bits: usize) -> Vec<u8> {
    let mut bits = vec![0; num_bits];
    for i in 0..num_bits {
        bits[num_bits - 1 - i] = ((n >> i) & 1) as u8;
    }
    bits
}

/// Calculate the integer representation of a bit string
#[must_use]
pub fn bits_to_int(bits: &[u8]) -> usize {
    bits.iter().fold(0, |acc, &bit| (acc << 1) | bit as usize)
}

/// Compute the index with a bit flipped at the specified position
#[must_use]
pub const fn flip_bit(index: usize, pos: usize) -> usize {
    index ^ (1 << pos)
}

/// Compute the index with a controlled bit flip
///
/// If the control bit at `ctrl_pos` is 1, then the target bit at `target_pos` is flipped.
#[must_use]
pub const fn controlled_flip(index: usize, ctrl_pos: usize, target_pos: usize) -> usize {
    if (index >> ctrl_pos) & 1 == 1 {
        flip_bit(index, target_pos)
    } else {
        index
    }
}

/// Compute the global index for a multi-qubit system
///
/// Given a list of qubit indices, compute the global index into the state vector.
#[must_use]
pub fn compute_index(qubit_indices: &[QubitId], state_bits: &[u8]) -> usize {
    assert!(
        (qubit_indices.len() == state_bits.len()),
        "Mismatch between qubit indices and state bits"
    );

    let mut index = 0;
    for (&qubit, &bit) in qubit_indices.iter().zip(state_bits.iter()) {
        let q = qubit.id() as usize;
        if bit != 0 {
            index |= 1 << q;
        }
    }

    index
}

/// Parallel map over state vector elements with index
///
/// Apply a function to each element of the state vector in parallel.
/// The function is passed the index and current value of each element.
pub fn par_indexed_map<F>(state: &mut [Complex64], f: F)
where
    F: Fn(usize, Complex64) -> Complex64 + Sync,
{
    state.par_iter_mut().enumerate().for_each(|(i, v)| {
        *v = f(i, *v);
    });
}

/// Convert a matrix representation of a gate (row-major) to a 2D ndarray
#[must_use]
pub fn gate_vec_to_array2(matrix: &[Complex64], dim: usize) -> Array2<Complex64> {
    let mut result = Array2::zeros((dim, dim));

    for i in 0..dim {
        for j in 0..dim {
            result[[i, j]] = matrix[i * dim + j];
        }
    }

    result
}
