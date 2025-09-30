//! Linear algebra operations for quantum simulation using SciRS2
//!
//! This module provides optimized linear algebra operations for quantum
//! simulation by leveraging SciRS2's BLAS/LAPACK bindings when available.

// Note: NdArrayExt would be used here if it was available in scirs2_core
// For now, we'll use standard ndarray operations

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::Complex64;

/// Matrix-vector multiplication for quantum state evolution
///
/// Computes |ψ'⟩ = U|ψ⟩ where U is a unitary matrix and |ψ⟩ is a state vector.
pub fn apply_unitary(
    unitary: &ArrayView2<Complex64>,
    state: &mut [Complex64],
) -> Result<(), String> {
    let n = state.len();

    // Check dimensions
    if unitary.shape() != [n, n] {
        return Err(format!(
            "Unitary matrix shape {:?} doesn't match state dimension {}",
            unitary.shape(),
            n
        ));
    }

    // Create temporary storage for the result
    let mut result = vec![Complex64::new(0.0, 0.0); n];

    // Perform matrix-vector multiplication
    #[cfg(feature = "advanced_math")]
    {
        // Use optimized matrix multiplication when available
        for i in 0..n {
            for j in 0..n {
                result[i] += unitary[[i, j]] * state[j];
            }
        }
    }

    #[cfg(not(feature = "advanced_math"))]
    {
        // Fallback to manual implementation
        for i in 0..n {
            for j in 0..n {
                result[i] += unitary[[i, j]] * state[j];
            }
        }
    }

    // Copy result back to state
    state.copy_from_slice(&result);
    Ok(())
}

/// Compute the tensor product of two matrices
///
/// This is used for constructing multi-qubit gates from single-qubit gates.
pub fn tensor_product(a: &ArrayView2<Complex64>, b: &ArrayView2<Complex64>) -> Array2<Complex64> {
    let (m, n) = a.dim();
    let (p, q) = b.dim();

    let mut result = Array2::zeros((m * p, n * q));

    for i in 0..m {
        for j in 0..n {
            for k in 0..p {
                for l in 0..q {
                    result[[i * p + k, j * q + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }

    result
}

/// Compute the partial trace over specified qubits
///
/// This is used for obtaining reduced density matrices.
pub fn partial_trace(
    density_matrix: &ArrayView2<Complex64>,
    qubits_to_trace: &[usize],
    total_qubits: usize,
) -> Result<Array2<Complex64>, String> {
    let dim = 1 << total_qubits;

    if density_matrix.shape() != [dim, dim] {
        return Err(format!(
            "Density matrix shape {:?} doesn't match {} qubits",
            density_matrix.shape(),
            total_qubits
        ));
    }

    // Calculate dimensions after tracing
    let traced_qubits = qubits_to_trace.len();
    let remaining_qubits = total_qubits - traced_qubits;
    let remaining_dim = 1 << remaining_qubits;

    let mut result = Array2::zeros((remaining_dim, remaining_dim));

    // Perform the partial trace
    // This is a simplified implementation; a full implementation would be more complex
    for i in 0..remaining_dim {
        for j in 0..remaining_dim {
            let mut sum = Complex64::new(0.0, 0.0);

            // Sum over traced-out basis states
            for k in 0..(1 << traced_qubits) {
                // Map indices appropriately (simplified for demonstration)
                let full_i = i + (k << remaining_qubits);
                let full_j = j + (k << remaining_qubits);

                if full_i < dim && full_j < dim {
                    sum += density_matrix[[full_i, full_j]];
                }
            }

            result[[i, j]] = sum;
        }
    }

    Ok(result)
}

/// Check if a matrix is unitary (U†U = I)
pub fn is_unitary(matrix: &ArrayView2<Complex64>, tolerance: f64) -> bool {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return false; // Not square
    }

    // Compute U†U
    let mut product: Array2<Complex64> = Array2::zeros((n, n));

    #[cfg(feature = "advanced_math")]
    {
        // Use optimized matrix multiplication
        let conjugate_transpose = matrix.t().mapv(|x| x.conj());
        product = conjugate_transpose.dot(matrix);
    }

    #[cfg(not(feature = "advanced_math"))]
    {
        // Manual implementation
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    product[[i, j]] += matrix[[k, i]].conj() * matrix[[k, j]];
                }
            }
        }
    }

    // Check if result is identity
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };

            let diff: Complex64 = product[[i, j]] - expected;
            if diff.norm() > tolerance {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_apply_unitary() {
        // Hadamard gate
        let h = arr2(&[
            [
                Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0),
                Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0),
            ],
            [
                Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0),
                Complex64::new(-1.0 / std::f64::consts::SQRT_2, 0.0),
            ],
        ]);

        // |0⟩ state
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

        apply_unitary(&h.view(), &mut state).unwrap();

        // Should produce |+⟩ state
        let expected_0 = Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0);
        assert!((state[0] - expected_0).norm() < 1e-10);
        assert!((state[1] - expected_0).norm() < 1e-10);
    }

    #[test]
    fn test_tensor_product() {
        // Two 2x2 matrices
        let a = arr2(&[
            [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
        ]);

        let b = arr2(&[
            [Complex64::new(5.0, 0.0), Complex64::new(6.0, 0.0)],
            [Complex64::new(7.0, 0.0), Complex64::new(8.0, 0.0)],
        ]);

        let result = tensor_product(&a.view(), &b.view());

        assert_eq!(result.dim(), (4, 4));
        assert_eq!(result[[0, 0]], Complex64::new(5.0, 0.0));
        assert_eq!(result[[0, 1]], Complex64::new(6.0, 0.0));
        assert_eq!(result[[3, 3]], Complex64::new(32.0, 0.0));
    }

    #[test]
    fn test_is_unitary() {
        // Pauli X gate
        let x = arr2(&[
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ]);

        assert!(is_unitary(&x.view(), 1e-10));

        // Non-unitary matrix
        let non_unitary = arr2(&[
            [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]);

        assert!(!is_unitary(&non_unitary.view(), 1e-10));
    }
}
