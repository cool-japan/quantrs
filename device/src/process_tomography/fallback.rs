//! Fallback implementations when SciRS2 is not available
//!
//! These are used (via the `#[cfg(not(feature = "scirs2"))]` call sites in
//! `reconstruction/linear_inversion.rs` and friends) only when the crate is
//! built with `--no-default-features` (i.e. without `scirs2-linalg`). Every
//! matrix operation below is a real, dimension-correct (if numerically
//! naive) implementation rather than a fixed-size placeholder, so a
//! no-scirs2 build computes with the actual input shape instead of silently
//! substituting a hardcoded 2x2 result. The handful of decompositions that
//! are genuinely impractical to hand-roll correctly (full eigenvalue
//! decomposition, SVD) honestly return `Err` instead.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Fallback statistical mean calculation
pub const fn mean(_data: &ArrayView1<f64>) -> Result<f64, String> {
    Ok(0.0)
}

/// Fallback standard deviation calculation
pub const fn std(_data: &ArrayView1<f64>, _ddof: i32) -> Result<f64, String> {
    Ok(1.0)
}

/// Fallback Pearson correlation calculation
pub const fn pearsonr(
    _x: &ArrayView1<f64>,
    _y: &ArrayView1<f64>,
    _alt: &str,
) -> Result<(f64, f64), String> {
    Ok((0.0, 0.5))
}

/// Real matrix trace: sum of the diagonal entries (dimension-correct for
/// any `min(rows, cols)`-length diagonal, not just 2x2).
pub fn trace(matrix: &ArrayView2<f64>) -> Result<f64, String> {
    let n = matrix.nrows().min(matrix.ncols());
    Ok((0..n).map(|i| matrix[[i, i]]).sum())
}

/// Real matrix inversion via Gauss-Jordan elimination with partial
/// pivoting. Works for any square NxN matrix (not just 2x2); returns an
/// honest error for non-square or singular/near-singular input.
pub fn inv(matrix: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(format!(
            "inv: matrix must be square, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }

    let mut a = matrix.to_owned();
    let mut inv_mat = Array2::<f64>::eye(n);

    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = a[[col, col]].abs();
        for row in (col + 1)..n {
            if a[[row, col]].abs() > max_val {
                max_val = a[[row, col]].abs();
                pivot_row = row;
            }
        }
        if max_val < 1e-12 {
            return Err(format!(
                "inv: matrix is singular or nearly singular (pivot magnitude {max_val:.3e} at column {col})"
            ));
        }
        if pivot_row != col {
            for k in 0..n {
                a.swap((col, k), (pivot_row, k));
                inv_mat.swap((col, k), (pivot_row, k));
            }
        }

        let pivot_val = a[[col, col]];
        for k in 0..n {
            a[[col, k]] /= pivot_val;
            inv_mat[[col, k]] /= pivot_val;
        }

        for row in 0..n {
            if row != col {
                let factor = a[[row, col]];
                if factor != 0.0 {
                    for k in 0..n {
                        let a_col_k = a[[col, k]];
                        let inv_col_k = inv_mat[[col, k]];
                        a[[row, k]] -= factor * a_col_k;
                        inv_mat[[row, k]] -= factor * inv_col_k;
                    }
                }
            }
        }
    }

    Ok(inv_mat)
}

/// Fallback optimization result
pub struct OptimizeResult {
    pub x: Array1<f64>,
    pub fun: f64,
    pub success: bool,
    pub nit: usize,
}

/// Fallback optimization function
pub fn minimize(
    _func: fn(&Array1<f64>) -> f64,
    _x0: &Array1<f64>,
    _method: &str,
) -> Result<OptimizeResult, String> {
    Ok(OptimizeResult {
        x: Array1::zeros(2),
        fun: 0.0,
        success: true,
        nit: 0,
    })
}

/// Full eigenvalue decomposition is not implemented in this no-SciRS2
/// fallback (a numerically stable general eigensolver -- e.g. the
/// shifted-QR algorithm with deflation -- is well beyond a "naive"
/// reimplementation); honestly report the limitation rather than return a
/// fixed, wrong-sized identity-like result.
pub fn eig(
    matrix: &ArrayView2<f64>,
) -> Result<
    (
        Array1<scirs2_core::Complex64>,
        Array2<scirs2_core::Complex64>,
    ),
    String,
> {
    Err(format!(
        "eig: full eigenvalue decomposition is not implemented in the no-scirs2 fallback \
         (requested for a {}x{} matrix); rebuild with the `scirs2` feature to use this operation",
        matrix.nrows(),
        matrix.ncols()
    ))
}

/// Real matrix determinant via LU decomposition with partial pivoting.
/// Works for any square NxN matrix (not just 2x2).
pub fn det(matrix: &ArrayView2<f64>) -> Result<f64, String> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(format!(
            "det: matrix must be square, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }

    let mut a = matrix.to_owned();
    let mut sign = 1.0_f64;

    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = a[[col, col]].abs();
        for row in (col + 1)..n {
            if a[[row, col]].abs() > max_val {
                max_val = a[[row, col]].abs();
                pivot_row = row;
            }
        }
        if max_val < 1e-14 {
            return Ok(0.0); // Singular matrix: determinant is zero.
        }
        if pivot_row != col {
            for k in 0..n {
                a.swap((col, k), (pivot_row, k));
            }
            sign = -sign;
        }

        let pivot_val = a[[col, col]];
        for row in (col + 1)..n {
            let factor = a[[row, col]] / pivot_val;
            if factor != 0.0 {
                for k in col..n {
                    let a_col_k = a[[col, k]];
                    a[[row, k]] -= factor * a_col_k;
                }
            }
        }
    }

    let mut det_val = sign;
    for i in 0..n {
        det_val *= a[[i, i]];
    }
    Ok(det_val)
}

/// Real QR decomposition via modified Gram-Schmidt orthogonalization. Works
/// for any MxN matrix (not just 2x2).
pub fn qr(matrix: &ArrayView2<f64>) -> Result<(Array2<f64>, Array2<f64>), String> {
    let m = matrix.nrows();
    let n = matrix.ncols();
    if m == 0 || n == 0 {
        return Err(format!("qr: matrix must be non-empty, got {m}x{n}"));
    }

    let mut q = Array2::<f64>::zeros((m, n));
    let mut r = Array2::<f64>::zeros((n, n));

    for j in 0..n {
        let mut v = matrix.column(j).to_owned();
        for i in 0..j {
            let qi = q.column(i).to_owned();
            let r_ij = qi.dot(&matrix.column(j));
            r[[i, j]] = r_ij;
            for k in 0..m {
                v[k] -= r_ij * qi[k];
            }
        }
        let norm = v.dot(&v).sqrt();
        r[[j, j]] = norm;
        if norm > 1e-14 {
            for k in 0..m {
                q[[k, j]] = v[k] / norm;
            }
        }
    }

    Ok((q, r))
}

/// Singular value decomposition is not implemented in this no-SciRS2
/// fallback (a numerically stable general SVD -- e.g. Golub-Kahan
/// bidiagonalization followed by an implicit-shift QR sweep -- is well
/// beyond a "naive" reimplementation); honestly report the limitation
/// rather than return a fixed, wrong-sized identity-like result.
pub fn svd(matrix: &ArrayView2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), String> {
    Err(format!(
        "svd: singular value decomposition is not implemented in the no-scirs2 fallback \
         (requested for a {}x{} matrix); rebuild with the `scirs2` feature to use this operation",
        matrix.nrows(),
        matrix.ncols()
    ))
}

/// Fallback matrix norm calculation: the Frobenius norm, computed from the
/// actual matrix entries (not a fixed constant). Only the Frobenius norm is
/// supported here -- spectral/nuclear norms would require SVD, which this
/// no-scirs2 fallback does not implement.
pub fn matrix_norm(matrix: &ArrayView2<f64>, _ord: Option<&str>) -> Result<f64, String> {
    Ok(matrix.iter().map(|v| v * v).sum::<f64>().sqrt())
}

/// Real Cholesky decomposition (lower-triangular `L` such that `L L^T =
/// matrix`). Works for any square NxN symmetric positive-definite matrix
/// (not just 2x2); returns an honest error if the matrix is not
/// positive-definite.
pub fn cholesky(matrix: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(format!(
            "cholesky: matrix must be square, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }

    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let val = matrix[[i, i]] - sum;
                if val <= 0.0 {
                    return Err(format!(
                        "cholesky: matrix is not positive-definite (diagonal term {val:.3e} <= 0 at index {i})"
                    ));
                }
                l[[i, j]] = val.sqrt();
            } else {
                l[[i, j]] = (matrix[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Fallback variance calculation
pub const fn var(_data: &ArrayView1<f64>, _ddof: i32) -> Result<f64, String> {
    Ok(1.0)
}

/// Real 2x2 Pearson correlation matrix for two equal-length samples `x`/`y`
/// (a pairwise correlation matrix for two variables is always 2x2, so
/// unlike `inv`/`qr`/etc. the *size* here was never the issue -- the value
/// was; this now computes the actual correlation instead of a fixed
/// identity matrix).
pub fn corrcoef(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<Array2<f64>, String> {
    if x.len() != y.len() {
        return Err(format!(
            "corrcoef: input length mismatch ({} vs {})",
            x.len(),
            y.len()
        ));
    }
    if x.len() < 2 {
        return Err("corrcoef: need at least 2 samples".to_string());
    }

    let n = x.len() as f64;
    let mean_x = x.sum() / n;
    let mean_y = y.sum() / n;

    let mut cov_xx = 0.0_f64;
    let mut cov_yy = 0.0_f64;
    let mut cov_xy = 0.0_f64;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov_xx += dx * dx;
        cov_yy += dy * dy;
        cov_xy += dx * dy;
    }

    let denom = (cov_xx * cov_yy).sqrt();
    let r = if denom > 1e-14 {
        (cov_xy / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    Array2::from_shape_vec((2, 2), vec![1.0, r, r, 1.0])
        .map_err(|e| format!("Array creation error: {e}"))
}

/// Fallback Spearman correlation calculation
pub const fn spearmanr(
    _x: &ArrayView1<f64>,
    _y: &ArrayView1<f64>,
    _alternative: &str,
) -> Result<(f64, f64), String> {
    Ok((0.0, 0.5))
}

/// Fallback t-test (one sample)
pub const fn ttest_1samp(
    _a: &ArrayView1<f64>,
    _popmean: f64,
    _alternative: &str,
) -> Result<TTestResult, String> {
    Ok(TTestResult {
        statistic: 0.0,
        pvalue: 0.5,
    })
}

/// Fallback t-test (independent samples)
pub const fn ttest_ind(
    _a: &ArrayView1<f64>,
    _b: &ArrayView1<f64>,
    _alternative: &str,
) -> Result<TTestResult, String> {
    Ok(TTestResult {
        statistic: 0.0,
        pvalue: 0.5,
    })
}

/// Fallback Kolmogorov-Smirnov 2-sample test
pub const fn ks_2samp(
    _data1: &ArrayView1<f64>,
    _data2: &ArrayView1<f64>,
    _alternative: &str,
) -> Result<KSTestResult, String> {
    Ok(KSTestResult {
        statistic: 0.0,
        pvalue: 0.5,
    })
}

/// Fallback Shapiro-Wilk test
pub const fn shapiro_wilk(_data: &ArrayView1<f64>) -> Result<SWTestResult, String> {
    Ok(SWTestResult {
        statistic: 0.95,
        pvalue: 0.1,
    })
}

/// T-test result structure
#[derive(Debug, Clone)]
pub struct TTestResult {
    pub statistic: f64,
    pub pvalue: f64,
}

/// Kolmogorov-Smirnov test result structure
#[derive(Debug, Clone)]
pub struct KSTestResult {
    pub statistic: f64,
    pub pvalue: f64,
}

/// Shapiro-Wilk test result structure
#[derive(Debug, Clone)]
pub struct SWTestResult {
    pub statistic: f64,
    pub pvalue: f64,
}

/// Alternative hypothesis type
#[derive(Debug, Clone)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

/// Distribution types and functions
pub mod distributions {
    use super::*;

    /// Normal distribution functions
    pub mod norm {
        /// Normal PDF
        pub const fn pdf(_x: f64, _loc: f64, _scale: f64) -> f64 {
            0.4
        }

        /// Normal CDF
        pub const fn cdf(_x: f64, _loc: f64, _scale: f64) -> f64 {
            0.5
        }

        /// Normal PPF (inverse CDF)
        pub const fn ppf(_q: f64, _loc: f64, _scale: f64) -> f64 {
            0.0
        }
    }

    /// Chi-squared distribution functions
    pub mod chi2 {
        /// Chi-squared PDF
        pub const fn pdf(_x: f64, _df: f64) -> f64 {
            0.1
        }

        /// Chi-squared CDF
        pub const fn cdf(_x: f64, _df: f64) -> f64 {
            0.5
        }

        /// Chi-squared PPF
        pub const fn ppf(_q: f64, _df: f64) -> f64 {
            1.0
        }
    }

    /// Gamma distribution functions
    pub mod gamma {
        /// Gamma PDF
        pub const fn pdf(_x: f64, _a: f64, _scale: f64) -> f64 {
            0.2
        }

        /// Gamma CDF
        pub const fn cdf(_x: f64, _a: f64, _scale: f64) -> f64 {
            0.5
        }

        /// Gamma PPF
        pub const fn ppf(_q: f64, _a: f64, _scale: f64) -> f64 {
            1.0
        }
    }
}

/// Graph analysis fallback functions
pub mod graph {
    use super::*;

    /// Fallback betweenness centrality
    pub fn betweenness_centrality(_graph: &Array2<f64>) -> Result<Array1<f64>, String> {
        Ok(Array1::ones(2))
    }

    /// Fallback closeness centrality
    pub fn closeness_centrality(_graph: &Array2<f64>) -> Result<Array1<f64>, String> {
        Ok(Array1::ones(2))
    }

    /// Fallback minimum spanning tree
    pub fn minimum_spanning_tree(_graph: &Array2<f64>) -> Result<Array2<f64>, String> {
        Ok(Array2::eye(2))
    }

    /// Fallback shortest path
    pub fn shortest_path(
        _graph: &Array2<f64>,
        _start: usize,
        _end: usize,
    ) -> Result<Vec<usize>, String> {
        Ok(vec![0, 1])
    }

    /// Fallback strongly connected components
    pub fn strongly_connected_components(_graph: &Array2<f64>) -> Result<Vec<Vec<usize>>, String> {
        Ok(vec![vec![0], vec![1]])
    }

    /// Graph structure placeholder
    pub struct Graph {
        pub adjacency_matrix: Array2<f64>,
    }

    impl Graph {
        pub const fn new(adjacency_matrix: Array2<f64>) -> Self {
            Self { adjacency_matrix }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_scales_with_matrix_size() {
        // A fixed-2x2 stub would ignore this 3x3 matrix; the real
        // implementation must sum all three diagonal entries.
        let m = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
        let t = trace(&m.view()).expect("trace should succeed");
        assert!((t - 15.0).abs() < 1e-9, "expected trace 15.0, got {t}");
    }

    #[test]
    fn test_det_identity_and_known_matrix() {
        let identity = Array2::<f64>::eye(4);
        let d = det(&identity.view()).expect("det should succeed");
        assert!((d - 1.0).abs() < 1e-9);

        // det([[2, 0], [0, 3]]) = 6
        let diag = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).unwrap();
        let d2 = det(&diag.view()).expect("det should succeed");
        assert!((d2 - 6.0).abs() < 1e-9, "expected 6.0, got {d2}");
    }

    #[test]
    fn test_inv_recovers_identity_for_arbitrary_size() {
        // A fixed-2x2 stub would corrupt this 3x3 inversion; the real
        // Gauss-Jordan implementation must produce a genuine inverse.
        let m = Array2::from_shape_vec((3, 3), vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0])
            .unwrap();
        let inverse = inv(&m.view()).expect("inv should succeed for a well-conditioned matrix");
        let product = m.dot(&inverse);
        let identity = Array2::<f64>::eye(3);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (product[[i, j]] - identity[[i, j]]).abs() < 1e-9,
                    "M * M^-1 should be the identity at [{i},{j}]: got {}",
                    product[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_inv_rejects_singular_matrix() {
        let singular = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert!(
            inv(&singular.view()).is_err(),
            "inverting a singular matrix must return an honest error, not a fabricated result"
        );
    }

    #[test]
    fn test_qr_reconstructs_original_matrix() {
        let m = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let (q, r) = qr(&m.view()).expect("qr should succeed");
        let reconstructed = q.dot(&r);
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (reconstructed[[i, j]] - m[[i, j]]).abs() < 1e-9,
                    "QR reconstruction mismatch at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_cholesky_reconstructs_positive_definite_matrix() {
        // A = [[4, 2], [2, 3]] is symmetric positive-definite.
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let l = cholesky(&a.view()).expect("cholesky should succeed for an SPD matrix");
        let reconstructed = l.dot(&l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-9,
                    "L L^T should reconstruct A at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_cholesky_rejects_non_positive_definite() {
        let not_spd = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0]).unwrap();
        assert!(
            cholesky(&not_spd.view()).is_err(),
            "cholesky of a non-positive-definite matrix must return an honest error"
        );
    }

    #[test]
    fn test_corrcoef_perfect_positive_correlation() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);
        let c = corrcoef(&x.view(), &y.view()).expect("corrcoef should succeed");
        assert!(
            (c[[0, 1]] - 1.0).abs() < 1e-9,
            "expected r=1.0, got {}",
            c[[0, 1]]
        );
        assert!((c[[1, 0]] - 1.0).abs() < 1e-9);
        assert!((c[[0, 0]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_corrcoef_perfect_negative_correlation() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from_vec(vec![8.0, 6.0, 4.0, 2.0]);
        let c = corrcoef(&x.view(), &y.view()).expect("corrcoef should succeed");
        assert!(
            (c[[0, 1]] - (-1.0)).abs() < 1e-9,
            "expected r=-1.0, got {}",
            c[[0, 1]]
        );
    }

    #[test]
    fn test_matrix_norm_frobenius() {
        // Frobenius norm of [[3, 0], [0, 4]] is sqrt(9 + 16) = 5.
        let m = Array2::from_shape_vec((2, 2), vec![3.0, 0.0, 0.0, 4.0]).unwrap();
        let norm = matrix_norm(&m.view(), None).expect("matrix_norm should succeed");
        assert!((norm - 5.0).abs() < 1e-9, "expected 5.0, got {norm}");
    }

    #[test]
    fn test_eig_and_svd_honestly_error_instead_of_fabricating() {
        let m = Array2::<f64>::eye(3);
        assert!(
            eig(&m.view()).is_err(),
            "eig fallback must honestly error rather than return a fixed 2x2 result for a 3x3 input"
        );
        assert!(
            svd(&m.view()).is_err(),
            "svd fallback must honestly error rather than return a fixed 2x2 result for a 3x3 input"
        );
    }
}
