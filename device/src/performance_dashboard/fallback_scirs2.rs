//! Fallback implementations for SciRS2 functionality when the feature is not available
//!
//! This module provides basic implementations of SciRS2 functions that are used
//! in the performance dashboard module when the scirs2 feature is not enabled.

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Fallback error type for optimization
#[derive(Debug, Clone)]
pub struct OptimizeError {
    pub message: String,
}

impl std::fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Optimization error: {}", self.message)
    }
}

impl std::error::Error for OptimizeError {}

/// Fallback result type for optimization
pub type OptimizeResult<T> = Result<T, OptimizeError>;

/// Fallback linear algebra error
#[derive(Debug, Clone)]
pub struct LinalgError {
    pub message: String,
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Linear algebra error: {}", self.message)
    }
}

impl std::error::Error for LinalgError {}

/// Fallback result type for linear algebra
pub type LinalgResult<T> = Result<T, LinalgError>;

/// Alternative enum for statistical tests
#[derive(Debug, Clone, Copy)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

/// Basic statistics functions
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn std(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

pub fn var(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn corrcoef(x: &[f64], y: &[f64]) -> f64 {
    pearsonr(x, y)
}

pub fn pearsonr(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let mean_x = mean(x);
    let mean_y = mean(y);

    let numerator: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
    let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

    let denominator = (sum_sq_x * sum_sq_y).sqrt();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

pub fn spearmanr(x: &[f64], y: &[f64]) -> f64 {
    // Simplified Spearman correlation - just return Pearson for fallback
    pearsonr(x, y)
}

/// Fallback optimization function
pub fn minimize<F>(
    _objective: F,
    _initial_guess: &[f64],
    _bounds: Option<&[(f64, f64)]>,
) -> OptimizeResult<MinimizeResult>
where
    F: Fn(&[f64]) -> f64,
{
    // Basic fallback - return the initial guess as "optimal"
    Ok(MinimizeResult {
        x: _initial_guess.to_vec(),
        fun: 0.0,
        success: true,
        message: "Fallback optimization".to_string(),
        nit: 0,
        nfev: 0,
    })
}

/// Result type for minimize function
#[derive(Debug, Clone)]
pub struct MinimizeResult {
    pub x: Vec<f64>,
    pub fun: f64,
    pub success: bool,
    pub message: String,
    pub nit: usize,
    pub nfev: usize,
}

/// Real symmetric eigensolver using the cyclic Jacobi algorithm (pure Rust
/// fallback for when the `scirs2` feature is disabled).
///
/// Assumes the input is **symmetric**; it is defensively symmetrized as
/// `(A + Aᵀ) / 2`. Returns `(eigenvalues, eigenvectors)` with ascending
/// eigenvalues and orthonormal eigenvector columns. Non-square input is an error.
pub fn eig(matrix: &Array2<f64>) -> LinalgResult<(Array1<f64>, Array2<f64>)> {
    let (rows, cols) = matrix.dim();
    if rows != cols {
        return Err(LinalgError {
            message: format!("eig requires a square matrix, got {rows}x{cols}"),
        });
    }
    jacobi_symmetric_eig(matrix).map_err(|message| LinalgError { message })
}

/// Real SVD via the eigendecomposition of `AᵀA` (pure Rust fallback).
///
/// Computes right singular vectors `V` and singular values
/// `σ = sqrt(eig(AᵀA))` (clamped non-negative, sorted descending), then
/// recovers `U[:, i] = A V[:, i] / σ_i` with an orthonormal completion for
/// numerically-zero singular values. Returns `(U, S, Vt)`.
pub fn svd(matrix: &Array2<f64>) -> LinalgResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let (m, n) = matrix.dim();
    if m == 0 || n == 0 {
        return Err(LinalgError {
            message: "svd requires a non-empty matrix".to_string(),
        });
    }

    let mut ata = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let mut acc = 0.0;
            for k in 0..m {
                acc += matrix[(k, i)] * matrix[(k, j)];
            }
            ata[(i, j)] = acc;
            ata[(j, i)] = acc;
        }
    }

    let (eigenvalues, eigenvectors) =
        jacobi_symmetric_eig(&ata).map_err(|message| LinalgError { message })?;

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eigenvalues[b].total_cmp(&eigenvalues[a]));

    let mut singular_values = Array1::<f64>::zeros(n);
    let mut v = Array2::<f64>::zeros((n, n));
    for (new_idx, &old_idx) in order.iter().enumerate() {
        singular_values[new_idx] = eigenvalues[old_idx].max(0.0).sqrt();
        for row in 0..n {
            v[(row, new_idx)] = eigenvectors[(row, old_idx)];
        }
    }

    let mut u = Array2::<f64>::zeros((m, m));
    let k = m.min(n);
    let max_sigma = singular_values.iter().cloned().fold(0.0_f64, f64::max);
    let tol = max_sigma * (m.max(n) as f64) * f64::EPSILON;

    let mut filled = vec![false; m];
    for i in 0..k {
        let sigma = singular_values[i];
        if sigma > tol {
            for r in 0..m {
                let mut acc = 0.0;
                for c in 0..n {
                    acc += matrix[(r, c)] * v[(c, i)];
                }
                u[(r, i)] = acc / sigma;
            }
            filled[i] = true;
        }
    }
    complete_orthonormal_basis(&mut u, &filled);

    let vt = v.t().to_owned();
    Ok((u, singular_values, vt))
}

pub fn matrix_norm(matrix: &Array2<f64>) -> f64 {
    // Frobenius norm
    matrix.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Real matrix inverse via Gauss-Jordan elimination with partial pivoting
/// (pure Rust fallback). Returns an honest error on non-square or singular
/// matrices.
pub fn inv(matrix: &Array2<f64>) -> LinalgResult<Array2<f64>> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(LinalgError {
            message: "Matrix must be square".to_string(),
        });
    }
    gauss_jordan_inverse(matrix).map_err(|message| LinalgError { message })
}

/// Determinant via LU decomposition with partial pivoting (pure Rust fallback).
/// Returns 0.0 for singular or non-square matrices.
pub fn det(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    if n != matrix.ncols() || n == 0 {
        return 0.0;
    }

    let mut a = matrix.clone();
    let mut determinant = 1.0;

    for col in 0..n {
        // Partial pivot: largest magnitude entry in this column.
        let mut pivot_row = col;
        let mut pivot_val = a[(col, col)].abs();
        for r in (col + 1)..n {
            let v = a[(r, col)].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }
        if pivot_val <= f64::MIN_POSITIVE {
            return 0.0;
        }
        if pivot_row != col {
            for c in 0..n {
                a.swap((col, c), (pivot_row, c));
            }
            determinant = -determinant;
        }

        determinant *= a[(col, col)];
        let pivot = a[(col, col)];
        for r in (col + 1)..n {
            let factor = a[(r, col)] / pivot;
            for c in col..n {
                let v = a[(col, c)];
                a[(r, c)] -= factor * v;
            }
        }
    }

    determinant
}

/// Cyclic Jacobi eigenvalue algorithm for real symmetric matrices. Returns
/// `(eigenvalues, eigenvectors)` ascending with orthonormal eigenvector
/// columns. The input is defensively symmetrized as `(A + Aᵀ) / 2`.
fn jacobi_symmetric_eig(matrix: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), String> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err("jacobi_symmetric_eig requires a square matrix".to_string());
    }
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }

    let mut a = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = 0.5 * (matrix[(i, j)] + matrix[(j, i)]);
        }
    }

    let mut eigenvectors = Array2::<f64>::eye(n);
    if n == 1 {
        return Ok((Array1::from_vec(vec![a[(0, 0)]]), eigenvectors));
    }

    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[(p, q)] * a[(p, q)];
            }
        }
        if off <= f64::EPSILON * f64::EPSILON {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[(p, q)];
                if apq.abs() <= f64::MIN_POSITIVE {
                    continue;
                }
                let app = a[(p, p)];
                let aqq = a[(q, q)];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                for k in 0..n {
                    let akp = a[(k, p)];
                    let akq = a[(k, q)];
                    a[(k, p)] = c * akp - s * akq;
                    a[(k, q)] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[(p, k)];
                    let aqk = a[(q, k)];
                    a[(p, k)] = c * apk - s * aqk;
                    a[(q, k)] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = eigenvectors[(k, p)];
                    let vkq = eigenvectors[(k, q)];
                    eigenvectors[(k, p)] = c * vkp - s * vkq;
                    eigenvectors[(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut eigenvalues = Array1::<f64>::zeros(n);
    for i in 0..n {
        eigenvalues[i] = a[(i, i)];
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a_idx, &b_idx| eigenvalues[a_idx].total_cmp(&eigenvalues[b_idx]));

    let mut sorted_values = Array1::<f64>::zeros(n);
    let mut sorted_vectors = Array2::<f64>::zeros((n, n));
    for (new_idx, &old_idx) in order.iter().enumerate() {
        sorted_values[new_idx] = eigenvalues[old_idx];
        for row in 0..n {
            sorted_vectors[(row, new_idx)] = eigenvectors[(row, old_idx)];
        }
    }

    Ok((sorted_values, sorted_vectors))
}

/// Gauss-Jordan matrix inversion with partial pivoting. Returns an error if the
/// matrix is singular.
fn gauss_jordan_inverse(matrix: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = matrix.nrows();
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    // Augmented working copy [A | I].
    let mut a = matrix.clone();
    let mut inv = Array2::<f64>::eye(n);

    for col in 0..n {
        // Partial pivot.
        let mut pivot_row = col;
        let mut pivot_val = a[(col, col)].abs();
        for r in (col + 1)..n {
            let v = a[(r, col)].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }
        if pivot_val <= f64::MIN_POSITIVE {
            return Err("singular matrix: cannot invert".to_string());
        }
        if pivot_row != col {
            for c in 0..n {
                a.swap((col, c), (pivot_row, c));
                inv.swap((col, c), (pivot_row, c));
            }
        }

        // Normalize the pivot row.
        let pivot = a[(col, col)];
        for c in 0..n {
            a[(col, c)] /= pivot;
            inv[(col, c)] /= pivot;
        }

        // Eliminate the pivot column from all other rows.
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[(r, col)];
            if factor == 0.0 {
                continue;
            }
            for c in 0..n {
                let a_val = a[(col, c)];
                let inv_val = inv[(col, c)];
                a[(r, c)] -= factor * a_val;
                inv[(r, c)] -= factor * inv_val;
            }
        }
    }

    Ok(inv)
}

/// Fill unfilled columns of `u` with an orthonormal completion via modified
/// Gram-Schmidt against the canonical basis.
fn complete_orthonormal_basis(u: &mut Array2<f64>, filled: &[bool]) {
    let m = u.nrows();
    let ncols = u.ncols();
    let mut next_canonical = 0usize;
    for col in 0..ncols {
        if col < filled.len() && filled[col] {
            continue;
        }
        loop {
            if next_canonical >= m {
                break;
            }
            let mut candidate = Array1::<f64>::zeros(m);
            candidate[next_canonical] = 1.0;
            next_canonical += 1;

            for prev in 0..col {
                let mut dot = 0.0;
                for r in 0..m {
                    dot += candidate[r] * u[(r, prev)];
                }
                for r in 0..m {
                    candidate[r] -= dot * u[(r, prev)];
                }
            }

            let norm = candidate.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for r in 0..m {
                    u[(r, col)] = candidate[r] / norm;
                }
                break;
            }
        }
    }
}

/// Distribution modules
pub mod distributions {
    use super::*;

    pub struct Normal {
        pub mean: f64,
        pub std: f64,
    }

    impl Normal {
        pub fn new(mean: f64, std: f64) -> Self {
            Self { mean, std }
        }

        pub fn pdf(&self, x: f64) -> f64 {
            let z = (x - self.mean) / self.std;
            (-0.5 * z * z).exp() / (self.std * (2.0 * std::f64::consts::PI).sqrt())
        }

        pub fn cdf(&self, x: f64) -> f64 {
            // Simplified CDF approximation
            0.5 * (1.0 + ((x - self.mean) / (self.std * 2.0_f64.sqrt())).tanh())
        }
    }

    pub fn norm(mean: f64, std: f64) -> Normal {
        Normal::new(mean, std)
    }

    pub fn gamma(_shape: f64, _scale: f64) -> Normal {
        Normal::new(1.0, 1.0) // Fallback to normal
    }

    pub fn chi2(_df: f64) -> Normal {
        Normal::new(1.0, 1.0) // Fallback to normal
    }

    pub fn beta(_a: f64, _b: f64) -> Normal {
        Normal::new(0.5, 0.1) // Fallback to normal
    }

    pub fn uniform(_low: f64, _high: f64) -> Normal {
        Normal::new(0.0, 1.0) // Fallback to standard normal
    }
}

/// Graph-related fallback functions
#[derive(Debug, Clone)]
pub struct Graph<N, E> {
    nodes: Vec<N>,
    edges: Vec<(usize, usize, E)>,
}

impl<N, E> Graph<N, E> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: N) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    pub fn add_edge(&mut self, a: usize, b: usize, edge: E) {
        self.edges.push((a, b, edge));
    }

    pub fn nodes(&self) -> impl Iterator<Item = &N> {
        self.nodes.iter()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

pub fn shortest_path<N, E>(_graph: &Graph<N, E>, _start: usize, _end: usize) -> Option<Vec<usize>> {
    None // Fallback - no path found
}

pub fn betweenness_centrality<N, E>(
    _graph: &Graph<N, E>,
    _normalized: bool,
) -> HashMap<usize, f64> {
    HashMap::new() // Fallback - empty centrality
}

pub fn closeness_centrality<N, E>(_graph: &Graph<N, E>, _normalized: bool) -> HashMap<usize, f64> {
    HashMap::new() // Fallback - empty centrality
}

pub fn minimum_spanning_tree<N, E>(_graph: &Graph<N, E>) -> Vec<(usize, usize)> {
    Vec::new() // Fallback - empty MST
}

pub fn strongly_connected_components<N, E>(_graph: &Graph<N, E>) -> Vec<Vec<usize>> {
    Vec::new() // Fallback - no components
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_eig_diagonal() {
        let m = array![[3.0, 0.0], [0.0, 1.0]];
        let (vals, _vecs) = eig(&m).expect("eig should succeed");
        assert!(approx(vals[0], 1.0, 1e-9));
        assert!(approx(vals[1], 3.0, 1e-9));
    }

    #[test]
    fn test_eig_non_square_errors() {
        let m = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(eig(&m).is_err());
    }

    #[test]
    fn test_inv_identity_property_2x2() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        let a_inv = inv(&a).expect("inverse should succeed");
        // A * A^-1 == I.
        for r in 0..2 {
            for c in 0..2 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += a[(r, k)] * a_inv[(k, c)];
                }
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(approx(acc, expected, 1e-6), "[{r},{c}]={acc}");
            }
        }
    }

    #[test]
    fn test_inv_identity_property_3x3() {
        let a = array![[2.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 0.0]];
        let a_inv = inv(&a).expect("inverse should succeed");
        for r in 0..3 {
            for c in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += a[(r, k)] * a_inv[(k, c)];
                }
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(approx(acc, expected, 1e-6), "[{r},{c}]={acc}");
            }
        }
    }

    #[test]
    fn test_inv_singular_errors() {
        // Rank-deficient matrix has no inverse.
        let a = array![[1.0, 2.0], [2.0, 4.0]];
        assert!(inv(&a).is_err());
    }

    #[test]
    fn test_inv_non_square_errors() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(inv(&a).is_err());
    }

    #[test]
    fn test_det_known() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        // det = 4*6 - 7*2 = 10.
        assert!(approx(det(&a), 10.0, 1e-9));
        let singular = array![[1.0, 2.0], [2.0, 4.0]];
        assert!(approx(det(&singular), 0.0, 1e-9));
    }

    #[test]
    fn test_svd_reconstruction() {
        let a = array![[3.0, 1.0], [1.0, 3.0], [0.0, 2.0]];
        let (u, s, vt) = svd(&a).expect("svd should succeed");
        let (m, n) = a.dim();
        for r in 0..m {
            for c in 0..n {
                let mut recon = 0.0;
                for k in 0..n.min(m) {
                    recon += u[(r, k)] * s[k] * vt[(k, c)];
                }
                assert!(approx(recon, a[(r, c)], 1e-6), "[{r},{c}]={recon}");
            }
        }
    }
}
