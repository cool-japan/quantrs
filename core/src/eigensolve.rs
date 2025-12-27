//! Eigenvalue decomposition for quantum gates
//!
//! This module provides eigenvalue decomposition specifically optimized
//! for unitary matrices (quantum gates). It uses the QR algorithm with
//! shifts for efficient computation.

use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64 as Complex;
use std::f64::EPSILON;

/// Result of eigenvalue decomposition
#[derive(Debug, Clone)]
pub struct EigenDecomposition {
    /// Eigenvalues
    pub eigenvalues: Array1<Complex>,
    /// Eigenvectors as columns
    pub eigenvectors: Array2<Complex>,
}

/// Compute eigenvalue decomposition of a unitary matrix
pub fn eigen_decompose_unitary(
    matrix: &Array2<Complex>,
    tolerance: f64,
) -> QuantRS2Result<EigenDecomposition> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(QuantRS2Error::InvalidInput(
            "Matrix must be square".to_string(),
        ));
    }

    // For small matrices, use analytical solutions
    match n {
        1 => eigen_1x1(matrix),
        2 => eigen_2x2(matrix, tolerance),
        _ => eigen_general(matrix, tolerance),
    }
}

/// Analytical eigendecomposition for 1x1 matrix
fn eigen_1x1(matrix: &Array2<Complex>) -> QuantRS2Result<EigenDecomposition> {
    let eigenvalues = Array1::from_vec(vec![matrix[(0, 0)]]);
    let eigenvectors = Array2::from_shape_vec((1, 1), vec![Complex::new(1.0, 0.0)])
        .map_err(|e| QuantRS2Error::ComputationError(e.to_string()))?;

    Ok(EigenDecomposition {
        eigenvalues,
        eigenvectors,
    })
}

/// Analytical eigendecomposition for 2x2 matrix
fn eigen_2x2(matrix: &Array2<Complex>, tolerance: f64) -> QuantRS2Result<EigenDecomposition> {
    let a = matrix[(0, 0)];
    let b = matrix[(0, 1)];
    let c = matrix[(1, 0)];
    let d = matrix[(1, 1)];

    // Characteristic polynomial: det(A - λI) = 0
    // λ² - (a+d)λ + (ad-bc) = 0
    let trace = a + d;
    let det = a * d - b * c;

    // Solve quadratic equation
    let discriminant = trace * trace - 4.0 * det;
    let sqrt_disc = discriminant.sqrt();

    let lambda1 = (trace + sqrt_disc) / 2.0;
    let lambda2 = (trace - sqrt_disc) / 2.0;

    let eigenvalues = Array1::from_vec(vec![lambda1, lambda2]);

    // Find eigenvectors
    let mut eigenvectors = Array2::zeros((2, 2));

    // For each eigenvalue, solve (A - λI)v = 0
    for (i, &lambda) in eigenvalues.iter().enumerate() {
        // Use the first row of (A - λI) to find eigenvector
        // let _a_minus_lambda = a - lambda;

        if b.norm() > tolerance {
            // v = [b, λ - a]ᵀ (normalized)
            let v1 = b;
            let v2 = lambda - a;
            let norm = (v1.norm_sqr() + v2.norm_sqr()).sqrt();
            eigenvectors[(0, i)] = v1 / norm;
            eigenvectors[(1, i)] = v2 / norm;
        } else if c.norm() > tolerance {
            // Use second row: v = [λ - d, c]ᵀ (normalized)
            let v1 = lambda - d;
            let v2 = c;
            let norm = (v1.norm_sqr() + v2.norm_sqr()).sqrt();
            eigenvectors[(0, i)] = v1 / norm;
            eigenvectors[(1, i)] = v2 / norm;
        } else {
            // Matrix is diagonal
            eigenvectors[(i, i)] = Complex::new(1.0, 0.0);
        }
    }

    Ok(EigenDecomposition {
        eigenvalues,
        eigenvectors,
    })
}

/// General eigendecomposition using QR algorithm with shifts
fn eigen_general(matrix: &Array2<Complex>, tolerance: f64) -> QuantRS2Result<EigenDecomposition> {
    let n = matrix.nrows();
    let max_iterations = 1000;

    // Start with Hessenberg reduction for efficiency
    let mut h = matrix.clone();
    hessenberg_reduction(&mut h)?;

    // QR algorithm with shifts
    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = Array2::eye(n);

    for _ in 0..max_iterations {
        // Check for convergence
        let mut converged = true;
        for i in 1..n {
            if h[(i, i - 1)].norm() > tolerance {
                converged = false;
                break;
            }
        }

        if converged {
            break;
        }

        // Wilkinson shift for faster convergence
        let shift = wilkinson_shift(&h, n);

        // QR decomposition of H - σI
        let mut h_shifted = h.clone();
        for i in 0..n {
            h_shifted[(i, i)] -= shift;
        }
        let (q, r) = qr_decompose(&h_shifted)?;

        // Update H = RQ + σI
        h = r.dot(&q);
        for i in 0..n {
            h[(i, i)] += shift;
        }

        // Accumulate eigenvectors
        eigenvectors = eigenvectors.dot(&q);
    }

    // Extract eigenvalues from diagonal
    for i in 0..n {
        eigenvalues.push(h[(i, i)]);
    }

    // Refine eigenvectors using inverse iteration
    let refined_eigenvectors = refine_eigenvectors(matrix, &eigenvalues, tolerance)?;

    Ok(EigenDecomposition {
        eigenvalues: Array1::from_vec(eigenvalues),
        eigenvectors: refined_eigenvectors,
    })
}

/// Hessenberg reduction using Householder reflections
fn hessenberg_reduction(matrix: &mut Array2<Complex>) -> QuantRS2Result<()> {
    let n = matrix.nrows();

    for k in 0..n - 2 {
        // Compute Householder vector for column k
        let mut x = Array1::zeros(n - k - 1);
        for i in k + 1..n {
            x[i - k - 1] = matrix[(i, k)];
        }

        let alpha = x.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if alpha < EPSILON {
            continue;
        }

        let mut v = x.clone();
        v[0] += alpha * Complex::new(x[0].re.signum(), x[0].im.signum());
        let v_norm = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if v_norm < EPSILON {
            continue;
        }
        for i in 0..v.len() {
            v[i] /= v_norm;
        }

        // Apply Householder reflection: H = I - 2vv*
        // Update A = HAH*
        for j in k..n {
            let mut sum = Complex::new(0.0, 0.0);
            for i in k + 1..n {
                sum += v[i - k - 1].conj() * matrix[(i, j)];
            }
            for i in k + 1..n {
                matrix[(i, j)] -= 2.0 * v[i - k - 1] * sum;
            }
        }

        for i in 0..n {
            let mut sum = Complex::new(0.0, 0.0);
            for j in k + 1..n {
                sum += matrix[(i, j)] * v[j - k - 1];
            }
            for j in k + 1..n {
                matrix[(i, j)] -= 2.0 * sum * v[j - k - 1].conj();
            }
        }
    }

    Ok(())
}

/// QR decomposition using Givens rotations
fn qr_decompose(matrix: &Array2<Complex>) -> QuantRS2Result<(Array2<Complex>, Array2<Complex>)> {
    let n = matrix.nrows();
    let mut r = matrix.clone();
    let mut q = Array2::eye(n);

    // Use Givens rotations to zero out below-diagonal elements
    for j in 0..n - 1 {
        for i in (j + 1)..n {
            if r[(i, j)].norm() < EPSILON {
                continue;
            }

            // Compute Givens rotation
            let (c, s) = givens_rotation(r[(j, j)], r[(i, j)]);

            // Apply rotation to R
            for k in j..n {
                let rjk = r[(j, k)];
                let rik = r[(i, k)];
                r[(j, k)] = c * rjk + s * rik;
                r[(i, k)] = -s.conj() * rjk + c * rik;
            }

            // Apply rotation to Q
            for k in 0..n {
                let qkj = q[(k, j)];
                let qki = q[(k, i)];
                q[(k, j)] = c * qkj + s * qki;
                q[(k, i)] = -s.conj() * qkj + c * qki;
            }
        }
    }

    Ok((q, r))
}

/// Compute Givens rotation coefficients
fn givens_rotation(a: Complex, b: Complex) -> (f64, Complex) {
    let r = (a.norm_sqr() + b.norm_sqr()).sqrt();
    if r < EPSILON {
        (1.0, Complex::new(0.0, 0.0))
    } else {
        let c = a.norm() / r;
        let s = (a.conj() * b) / (a.norm() * r);
        (c, s)
    }
}

/// Wilkinson shift for QR algorithm
fn wilkinson_shift(matrix: &Array2<Complex>, n: usize) -> Complex {
    if n < 2 {
        return Complex::new(0.0, 0.0);
    }

    // Get 2x2 trailing submatrix
    let a = matrix[(n - 2, n - 2)];
    let b = matrix[(n - 2, n - 1)];
    let c = matrix[(n - 1, n - 2)];
    let d = matrix[(n - 1, n - 1)];

    // Compute eigenvalue of 2x2 matrix closest to d
    let trace = a + d;
    let det = a * d - b * c;
    let discriminant = trace * trace - 4.0 * det;
    let sqrt_disc = discriminant.sqrt();

    let lambda1 = (trace + sqrt_disc) / 2.0;
    let lambda2 = (trace - sqrt_disc) / 2.0;

    // Choose shift closest to bottom-right element
    if (lambda1 - d).norm() < (lambda2 - d).norm() {
        lambda1
    } else {
        lambda2
    }
}

/// Refine eigenvectors using inverse iteration
fn refine_eigenvectors(
    matrix: &Array2<Complex>,
    eigenvalues: &[Complex],
    tolerance: f64,
) -> QuantRS2Result<Array2<Complex>> {
    let n = matrix.nrows();
    let mut eigenvectors = Array2::zeros((n, n));

    for (i, &lambda) in eigenvalues.iter().enumerate() {
        // Start with random vector
        let mut v = Array1::from_vec(vec![Complex::new(1.0, 0.0); n]);
        v[i] = Complex::new(1.0, 0.0); // Bias towards standard basis

        // Normalize
        let norm = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm > EPSILON {
            for i in 0..v.len() {
                v[i] /= norm;
            }
        }

        // Inverse iteration: solve (A - λI)v_new = v_old
        let shifted = matrix - lambda * Array2::eye(n);

        for _ in 0..10 {
            // Use QR decomposition to solve the system
            let v_new = solve_system(&shifted, &v, tolerance)?;

            // Normalize
            let norm = v_new.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            if norm < EPSILON {
                break;
            }

            let v_normalized = &v_new / norm;

            // Check convergence
            let diff: f64 = (&v_normalized - &v)
                .iter()
                .map(|x| x.norm_sqr())
                .sum::<f64>()
                .sqrt();
            if diff < tolerance {
                v = v_normalized;
                break;
            }

            v = v_normalized;
        }

        // Store eigenvector
        for j in 0..n {
            eigenvectors[(j, i)] = v[j];
        }
    }

    Ok(eigenvectors)
}

/// Solve linear system using QR decomposition
fn solve_system(
    a: &Array2<Complex>,
    b: &Array1<Complex>,
    tolerance: f64,
) -> QuantRS2Result<Array1<Complex>> {
    let n = a.nrows();

    // Add small regularization for numerical stability
    let mut regularized = a.clone();
    for i in 0..n {
        regularized[(i, i)] += Complex::new(tolerance, 0.0);
    }

    // QR decomposition
    let (q, r) = qr_decompose(&regularized)?;

    // Solve Rx = Q*b using back substitution
    let qtb = q.t().dot(b);
    let mut x = Array1::zeros(n);

    for i in (0..n).rev() {
        let mut sum = qtb[i];
        for j in i + 1..n {
            sum -= r[(i, j)] * x[j];
        }

        if r[(i, i)].norm() > tolerance {
            x[i] = sum / r[(i, i)];
        } else {
            x[i] = Complex::new(0.0, 0.0);
        }
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_eigen_2x2_pauli_x() {
        // Pauli X matrix
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
        )
        .expect("Failed to create Pauli X matrix");

        let result = eigen_decompose_unitary(&matrix, 1e-10).expect("Eigendecomposition failed");

        // Pauli X has eigenvalues ±1
        let mut eigenvalues: Vec<_> = result.eigenvalues.iter().map(|&x| x.re).collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        assert!((eigenvalues[0] - (-1.0)).abs() < 1e-10);
        assert!((eigenvalues[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigen_2x2_rotation() {
        // Rotation matrix R(θ)
        let theta = PI / 4.0;
        let c = theta.cos();
        let s = theta.sin();

        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(c, 0.0),
                Complex::new(-s, 0.0),
                Complex::new(s, 0.0),
                Complex::new(c, 0.0),
            ],
        )
        .expect("Failed to create rotation matrix");

        let result = eigen_decompose_unitary(&matrix, 1e-10).expect("Eigendecomposition failed");

        // Rotation matrix has eigenvalues e^(±iθ)
        for eigenvalue in result.eigenvalues.iter() {
            assert!((eigenvalue.norm() - 1.0).abs() < 1e-10);
        }

        // Check that phases are ±θ
        let phases: Vec<_> = result.eigenvalues.iter().map(|&x| x.arg()).collect();
        assert!(phases.iter().any(|&p| (p - theta).abs() < 1e-10));
        assert!(phases.iter().any(|&p| (p + theta).abs() < 1e-10));
    }

    #[test]
    fn test_eigenvector_orthogonality() {
        // Hadamard matrix
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(sqrt2_inv, 0.0),
                Complex::new(sqrt2_inv, 0.0),
                Complex::new(sqrt2_inv, 0.0),
                Complex::new(-sqrt2_inv, 0.0),
            ],
        )
        .expect("Failed to create Hadamard matrix");

        let result = eigen_decompose_unitary(&matrix, 1e-10).expect("Eigendecomposition failed");

        // Check eigenvectors are orthonormal
        let v1 = result.eigenvectors.column(0);
        let v2 = result.eigenvectors.column(1);

        // Check normalization
        assert!((v1.dot(&v1).norm() - 1.0).abs() < 1e-10);
        assert!((v2.dot(&v2).norm() - 1.0).abs() < 1e-10);

        // Check orthogonality
        assert!(v1.dot(&v2).norm() < 1e-10);
    }
}
