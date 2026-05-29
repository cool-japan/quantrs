//! Integration tests for real tensor contraction and SVD in QuantRS2.
//!
//! These tests verify that `Tensor::contract` performs correct tensor contraction
//! (not just a placeholder clone) and that `Tensor::svd` produces a valid
//! decomposition with proper truncation semantics.

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::Complex64;

use quantrs2_sim::tensor_network::tensor::Tensor;

// ============================================================
// Helpers
// ============================================================

/// Build a Tensor from a flat Vec of Complex64 values and a given shape.
fn make_tensor(shape: &[usize], values: Vec<Complex64>) -> Tensor {
    let data = Array::from_shape_vec(IxDyn(shape), values)
        .expect("shape / values length mismatch in make_tensor");
    Tensor::new(data)
}

/// Purely-real shorthand: wrap f64 as Complex64.
fn c(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

// ============================================================
// Contraction tests
// ============================================================

/// Contracting two rank-1 vectors along axis 0 of each should give the dot product
/// (a rank-0 scalar tensor).
#[test]
fn test_contract_vector_dot_product() {
    // v = [1, 2, 3],  w = [4, 5, 6]
    // v · w = 1*4 + 2*5 + 3*6 = 32
    let v = make_tensor(&[3], vec![c(1.0), c(2.0), c(3.0)]);
    let w = make_tensor(&[3], vec![c(4.0), c(5.0), c(6.0)]);

    let result = v.contract(&w, 0, 0).expect("contraction failed");

    // Scalar: rank 0, empty dimensions
    assert_eq!(result.rank, 0, "dot product should be scalar (rank 0)");
    assert!(result.dimensions.is_empty(), "scalar has no dimensions");

    let val = result.data[IxDyn(&[])];
    assert!((val - c(32.0)).norm() < 1e-10, "expected 32, got {val}");
}

/// Contracting a 2×3 matrix A with a 3×4 matrix B along axis 1 of A and axis 0 of B
/// should equal matrix multiplication (A @ B), producing a 2×4 result.
#[test]
fn test_contract_matrix_multiplication() {
    // A = [[1, 2, 3],
    //      [4, 5, 6]]          (2×3)
    //
    // B = [[1, 0, 0, 0],
    //      [0, 1, 0, 0],
    //      [0, 0, 1, 0]]       (3×4)
    //
    // A @ B = [[1, 2, 3, 0],
    //          [4, 5, 6, 0]]

    let a_vals: Vec<Complex64> = vec![1, 2, 3, 4, 5, 6]
        .into_iter()
        .map(|x| c(x as f64))
        .collect();
    let b_vals: Vec<Complex64> = vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        .into_iter()
        .map(|x| c(x as f64))
        .collect();

    let a = make_tensor(&[2, 3], a_vals);
    let b = make_tensor(&[3, 4], b_vals);

    let result = a
        .contract(&b, 1, 0)
        .expect("matrix-multiply contraction failed");

    assert_eq!(
        result.rank, 2,
        "result of 2D × 2D contraction should be rank 2"
    );
    assert_eq!(result.dimensions, vec![2, 4], "result shape should be 2×4");

    // Expected: A @ B
    let expected = [
        [c(1.0), c(2.0), c(3.0), c(0.0)],
        [c(4.0), c(5.0), c(6.0), c(0.0)],
    ];
    for i in 0..2 {
        for j in 0..4 {
            let got = result.data[IxDyn(&[i, j])];
            let exp = expected[i][j];
            assert!(
                (got - exp).norm() < 1e-10,
                "mismatch at [{i},{j}]: expected {exp}, got {got}"
            );
        }
    }
}

/// Contracting along a non-0/1 axis in higher-rank tensors.
#[test]
fn test_contract_rank3_tensors() {
    // A has shape [2, 2, 2], all ones
    // B has shape [2, 2, 2], all ones
    // Contracting axis 2 of A with axis 0 of B:
    //   result[i,j,l,m] = Σ_k A[i,j,k] * B[k,l,m]
    // Since all elements are 1 and the contracted dim is 2:
    //   result[i,j,l,m] = 2  for all i,j,l,m

    let a = make_tensor(&[2, 2, 2], vec![c(1.0); 8]);
    let b = make_tensor(&[2, 2, 2], vec![c(1.0); 8]);

    let result = a.contract(&b, 2, 0).expect("rank-3 contraction failed");

    assert_eq!(result.rank, 4, "result should be rank 4");
    assert_eq!(result.dimensions, vec![2, 2, 2, 2]);

    for (_, val) in result.data.indexed_iter() {
        assert!((val - c(2.0)).norm() < 1e-10, "expected 2.0, got {val}");
    }
}

/// Contracting along axis 0 of A and axis 1 of B (non-standard order).
#[test]
fn test_contract_non_trivial_axis_selection() {
    // A[i,j] contracted on axis 0 with B[k,i] on axis 1 →
    //   result[j,k] = Σ_i A[i,j] * B[k,i]
    //
    // A = [[1,2],[3,4]]    (2×2)  row-major: A[0,0]=1, A[0,1]=2, A[1,0]=3, A[1,1]=4
    // B = [[5,6],[7,8]]    (2×2)  row-major: B[0,0]=5, B[0,1]=6, B[1,0]=7, B[1,1]=8
    // result[0,0] = A[0,0]*B[0,0] + A[1,0]*B[0,1] = 1*5 + 3*6 = 23
    // result[0,1] = A[0,0]*B[1,0] + A[1,0]*B[1,1] = 1*7 + 3*8 = 31
    // result[1,0] = A[0,1]*B[0,0] + A[1,1]*B[0,1] = 2*5 + 4*6 = 34
    // result[1,1] = A[0,1]*B[1,0] + A[1,1]*B[1,1] = 2*7 + 4*8 = 46

    let a = make_tensor(&[2, 2], vec![c(1.0), c(2.0), c(3.0), c(4.0)]);
    let b = make_tensor(&[2, 2], vec![c(5.0), c(6.0), c(7.0), c(8.0)]);

    let result = a
        .contract(&b, 0, 1)
        .expect("non-trivial axis contraction failed");

    assert_eq!(result.rank, 2);
    assert_eq!(result.dimensions, vec![2, 2]);

    let expected = [[c(23.0), c(31.0)], [c(34.0), c(46.0)]];
    for i in 0..2 {
        for j in 0..2 {
            let got = result.data[IxDyn(&[i, j])];
            assert!(
                (got - expected[i][j]).norm() < 1e-10,
                "mismatch at [{i},{j}]: expected {}, got {got}",
                expected[i][j]
            );
        }
    }
}

/// Contracting tensors with mismatched dimensions should return an error.
#[test]
fn test_contract_dimension_mismatch_returns_error() {
    let a = make_tensor(&[2, 3], vec![c(0.0); 6]);
    let b = make_tensor(&[4, 3], vec![c(0.0); 12]);

    // axis 1 of a has dim 3, axis 0 of b has dim 4 → mismatch
    let result = a.contract(&b, 1, 0);
    assert!(result.is_err(), "expected error for dimension mismatch");
}

/// Out-of-range axis should return an error.
#[test]
fn test_contract_out_of_range_axis_returns_error() {
    let a = make_tensor(&[2, 2], vec![c(0.0); 4]);
    let b = make_tensor(&[2, 2], vec![c(0.0); 4]);

    let result = a.contract(&b, 5, 0);
    assert!(result.is_err(), "expected error for out-of-range self_axis");

    let result2 = a.contract(&b, 0, 5);
    assert!(
        result2.is_err(),
        "expected error for out-of-range other_axis"
    );
}

// ============================================================
// SVD tests
// ============================================================

/// Full SVD of a 4×4 matrix and reconstruction via U * diag(S) * Vᴴ should
/// recover the original matrix to machine precision.
#[test]
fn test_svd_reconstruction() {
    #[rustfmt::skip]
    let vals: Vec<Complex64> = vec![
        c(4.0), c(3.0), c(2.0), c(1.0),
        c(3.0), c(4.0), c(3.0), c(2.0),
        c(2.0), c(3.0), c(4.0), c(3.0),
        c(1.0), c(2.0), c(3.0), c(4.0),
    ];
    let t = make_tensor(&[4, 4], vals.clone());

    // SVD with full bond dimension (no truncation)
    let (left, right) = t.svd(&[0], &[1], 4).expect("SVD failed");

    // left has shape [4, bond_dim], right has shape [bond_dim, 4]
    // Reconstruct by contracting left and right along their bond index
    let reconstructed = left
        .contract(&right, left.rank - 1, 0)
        .expect("SVD reconstruction contraction failed");

    assert_eq!(reconstructed.rank, 2);
    assert_eq!(reconstructed.dimensions, vec![4, 4]);

    for (idx, orig_val) in t.data.indexed_iter() {
        let rec_val = reconstructed.data[idx.clone()];
        assert!(
            (orig_val - rec_val).norm() < 1e-8,
            "SVD reconstruction mismatch at {idx:?}: original {orig_val}, reconstructed {rec_val}"
        );
    }
}

/// SVD with max_bond_dim < full rank should truncate singular values.
#[test]
fn test_svd_truncation_reduces_bond_dimension() {
    // Diagonal matrix with known singular values [10, 3, 0.1, 0.01]
    let diag_vals = [10.0_f64, 3.0, 0.1, 0.01];
    let mut vals = vec![c(0.0); 16];
    for (i, &d) in diag_vals.iter().enumerate() {
        vals[i * 4 + i] = c(d);
    }
    let t = make_tensor(&[4, 4], vals);

    // Truncate to top-2 singular values
    let (left, right) = t.svd(&[0], &[1], 2).expect("truncated SVD failed");

    // The bond dimension of the output tensors should be 2
    let left_bond = *left.dimensions.last().expect("left tensor has dimensions");
    let right_bond = right.dimensions[0];
    assert_eq!(left_bond, 2, "left tensor bond dim should be 2");
    assert_eq!(right_bond, 2, "right tensor bond dim should be 2");

    // Reconstruct and check that large singular values are preserved
    let reconstructed = left
        .contract(&right, left.rank - 1, 0)
        .expect("reconstruction contraction failed");

    // The (0,0) and (1,1) entries should be approximately 10 and 3 respectively
    let v00 = reconstructed.data[IxDyn(&[0, 0])];
    let v11 = reconstructed.data[IxDyn(&[1, 1])];
    assert!(
        (v00 - c(10.0)).norm() < 0.5,
        "expected ~10 at [0,0], got {v00}"
    );
    assert!(
        (v11 - c(3.0)).norm() < 0.5,
        "expected ~3 at [1,1], got {v11}"
    );
}

/// SVD of a rectangular 2×4 matrix should produce correctly shaped tensors.
#[test]
fn test_svd_rectangular_matrix() {
    let vals: Vec<Complex64> = (1..=8).map(|x| c(x as f64)).collect();
    let t = make_tensor(&[2, 4], vals);

    let (left, right) = t
        .svd(&[0], &[1], 8)
        .expect("SVD of rectangular matrix failed");

    // Bond dim is min(2, 4, max_bond_dim) = 2
    let bond = *left.dimensions.last().expect("left has dims");
    assert!(bond <= 2, "bond dim should be <= min(m,n) = 2, got {bond}");
    assert_eq!(right.dimensions[0], bond, "right bond dim should match");
    assert_eq!(left.dimensions[0], 2, "left outer dim should be 2");
    assert_eq!(right.dimensions[1], 4, "right outer dim should be 4");
}

/// SVD of a rank-3 tensor split across two axes on the left and one on the right.
#[test]
fn test_svd_rank3_tensor() {
    let vals: Vec<Complex64> = (0..16).map(|i| c((i as f64 + 1.0) * 0.5)).collect();
    let t = make_tensor(&[2, 2, 4], vals);

    // Split: left_axes = [0, 1] (dims 2,2 → row size 4), right_axes = [2] (dim 4 → col size 4)
    let (left, right) = t.svd(&[0, 1], &[2], 4).expect("rank-3 SVD failed");

    // left shape: [2, 2, bond_dim]
    // right shape: [bond_dim, 4]
    assert_eq!(left.rank, 3, "left should be rank 3");
    assert_eq!(right.rank, 2, "right should be rank 2");
    assert_eq!(left.dimensions[0], 2);
    assert_eq!(left.dimensions[1], 2);
    assert_eq!(right.dimensions[1], 4);
    assert_eq!(
        *left.dimensions.last().unwrap(),
        right.dimensions[0],
        "bond dims must match"
    );
}

/// Invalid axis in SVD should return an error.
#[test]
fn test_svd_invalid_axes_returns_error() {
    let t = make_tensor(&[2, 2], vec![c(0.0); 4]);

    // Wrong total number of axes (right_axes is empty → total=1 != rank=2)
    let result = t.svd(&[0], &[], 2);
    assert!(result.is_err(), "expected error: wrong total axes count");

    // Out of range
    let result3 = t.svd(&[0], &[5], 2);
    assert!(result3.is_err(), "expected error: axis out of range");
}

/// SVD with max_bond_dim = 0 should return an error.
#[test]
fn test_svd_zero_bond_dim_returns_error() {
    let t = make_tensor(&[2, 2], vec![c(0.0); 4]);
    let result = t.svd(&[0], &[1], 0);
    assert!(result.is_err(), "expected error for max_bond_dim = 0");
}
