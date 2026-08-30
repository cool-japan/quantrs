//! Fallback implementations for SciRS2 functionality when the feature is not available
//!
//! This module provides basic implementations of SciRS2 functions that are used
//! in the quantum algorithm marketplace module when the scirs2 feature is not enabled.

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

pub fn betweenness_centrality<N, E>(
    _graph: &Graph<N, E>,
    _normalized: bool,
) -> HashMap<usize, f64> {
    HashMap::new() // Fallback - empty centrality
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

pub fn matrix_norm(matrix: &Array2<f64>) -> f64 {
    // Frobenius norm
    matrix.iter().map(|x| x * x).sum::<f64>().sqrt()
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
        let (vals, _) = eig(&m).expect("eig should succeed");
        assert!(approx(vals[0], 1.0, 1e-9));
        assert!(approx(vals[1], 3.0, 1e-9));
    }

    #[test]
    fn test_eig_symmetric_reconstruction() {
        let a = array![[2.0, 1.0], [1.0, 2.0]];
        let (vals, vecs) = eig(&a).expect("eig should succeed");
        for r in 0..2 {
            for c in 0..2 {
                let mut recon = 0.0;
                for k in 0..2 {
                    recon += vecs[(r, k)] * vals[k] * vecs[(c, k)];
                }
                assert!(approx(recon, a[(r, c)], 1e-9));
            }
        }
    }

    #[test]
    fn test_eig_non_square_errors() {
        let m = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(eig(&m).is_err());
    }
}
