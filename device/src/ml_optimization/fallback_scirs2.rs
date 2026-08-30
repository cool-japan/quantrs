//! Fallback implementations for SciRS2 functionality when the feature is not available
//!
//! This module provides basic implementations of SciRS2 functions that are used
//! in the ML optimization module when the scirs2 feature is not enabled.

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

/// Real symmetric eigensolver using the cyclic Jacobi eigenvalue algorithm.
///
/// This is a pure-Rust fallback used when the `scirs2` feature is disabled.
/// It assumes the input matrix is **symmetric** (e.g. covariance or Fisher
/// information matrices). For robustness the input is symmetrized as
/// `(A + Aᵀ) / 2` before the iteration begins.
///
/// Returns `(eigenvalues, eigenvectors)` where the columns of the
/// eigenvector matrix are orthonormal. A non-square input is an honest error.
pub fn eig(matrix: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), String> {
    let (rows, cols) = matrix.dim();
    if rows != cols {
        return Err(format!("eig requires a square matrix, got {rows}x{cols}"));
    }
    let (eigenvalues, eigenvectors) = jacobi_symmetric_eig(matrix)?;
    Ok((eigenvalues, eigenvectors))
}

/// Real singular value decomposition via the eigendecomposition of `AᵀA`.
///
/// Pure-Rust fallback used when the `scirs2` feature is disabled. Computes the
/// right singular vectors `V` and singular values `σ = sqrt(eig(AᵀA))`
/// (clamped to be non-negative, sorted descending), then recovers the left
/// singular vectors via `U[:, i] = A V[:, i] / σ_i`. Columns whose singular
/// value is numerically zero are filled with an orthonormal completion.
///
/// Returns `(U, S, Vt)` with `Vt = Vᵀ`.
pub fn svd(matrix: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), String> {
    let (m, n) = matrix.dim();
    if m == 0 || n == 0 {
        return Err("svd requires a non-empty matrix".to_string());
    }

    // Form A^T A (n x n, symmetric positive semi-definite).
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

    // Eigendecomposition (ascending eigenvalues). Reverse to descending so the
    // largest singular values come first.
    let (eigenvalues, eigenvectors) = jacobi_symmetric_eig(&ata)?;

    // Singular values are sqrt of (clamped) eigenvalues, sorted descending.
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

    // Left singular vectors: U[:, i] = A V[:, i] / sigma_i.
    let mut u = Array2::<f64>::zeros((m, m));
    let k = m.min(n);
    // Threshold for treating a singular value as zero, relative to the largest.
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

    // Complete U to an orthonormal basis (for zero / missing columns) using
    // modified Gram-Schmidt against the already-filled columns.
    complete_orthonormal_basis(&mut u, &filled);

    let vt = v.t().to_owned();
    Ok((u, singular_values, vt))
}

pub fn matrix_norm(matrix: &Array2<f64>) -> f64 {
    // Frobenius norm
    matrix.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Cyclic Jacobi eigenvalue algorithm for real symmetric matrices.
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvalues sorted in ascending
/// order and the corresponding orthonormal eigenvectors stored as columns.
/// The input is defensively symmetrized as `(A + Aᵀ) / 2`.
fn jacobi_symmetric_eig(matrix: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), String> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err("jacobi_symmetric_eig requires a square matrix".to_string());
    }
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }

    // Defensive symmetrization: a = (A + A^T) / 2.
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
        // Sum of squares of off-diagonal elements.
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

                // Compute the Jacobi rotation (cos theta, sin theta).
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply rotation to rows/columns p and q of A.
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

                // Accumulate the rotation into the eigenvector matrix.
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

    // Sort eigenvalues ascending and reorder eigenvectors accordingly.
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

/// Fill the unfilled columns of `u` with an orthonormal completion of the
/// already-filled columns, using modified Gram-Schmidt against the canonical
/// basis. `filled[i]` indicates that column `i` already holds a unit vector.
fn complete_orthonormal_basis(u: &mut Array2<f64>, filled: &[bool]) {
    let m = u.nrows();
    let ncols = u.ncols();

    // Collect indices of columns that still need to be generated.
    let mut next_canonical = 0usize;
    for col in 0..ncols {
        if col < filled.len() && filled[col] {
            continue;
        }
        // Find a canonical basis vector not yet (numerically) in the span.
        loop {
            if next_canonical >= m {
                // No more canonical directions; leave as zero column.
                break;
            }
            let mut candidate = Array1::<f64>::zeros(m);
            candidate[next_canonical] = 1.0;
            next_canonical += 1;

            // Orthogonalize against all previously established columns.
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

/// Statistical test results
#[derive(Debug, Clone)]
pub struct TTestResult {
    pub statistic: f64,
    pub pvalue: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

pub const fn ttest_1samp(data: &[f64], _popmean: f64) -> TTestResult {
    TTestResult {
        statistic: 0.0,
        pvalue: 0.5,
    }
}

pub const fn ttest_ind(data1: &[f64], data2: &[f64]) -> TTestResult {
    TTestResult {
        statistic: 0.0,
        pvalue: 0.5,
    }
}

pub const fn ks_2samp(data1: &[f64], data2: &[f64]) -> TTestResult {
    TTestResult {
        statistic: 0.0,
        pvalue: 0.5,
    }
}

pub const fn shapiro_wilk(data: &[f64]) -> TTestResult {
    TTestResult {
        statistic: 0.0,
        pvalue: 0.5,
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
        pub const fn new(mean: f64, std: f64) -> Self {
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

    pub const fn norm(mean: f64, std: f64) -> Normal {
        Normal::new(mean, std)
    }

    pub const fn gamma(_shape: f64, _scale: f64) -> Normal {
        Normal::new(1.0, 1.0) // Fallback to normal
    }

    pub const fn chi2(_df: f64) -> Normal {
        Normal::new(1.0, 1.0) // Fallback to normal
    }

    pub const fn beta(_a: f64, _b: f64) -> Normal {
        Normal::new(0.5, 0.1) // Fallback to normal
    }

    pub const fn uniform(_low: f64, _high: f64) -> Normal {
        Normal::new(0.0, 1.0) // Fallback to standard normal
    }
}

/// Graph-related fallback functions
#[derive(Debug, Clone)]
pub struct Graph<N, E> {
    nodes: Vec<N>,
    edges: Vec<(usize, usize, E)>,
}

impl<N, E> Default for Graph<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> Graph<N, E> {
    pub const fn new() -> Self {
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

pub const fn shortest_path<N, E>(
    _graph: &Graph<N, E>,
    _start: usize,
    _end: usize,
) -> Option<Vec<usize>> {
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

pub const fn minimum_spanning_tree<N, E>(_graph: &Graph<N, E>) -> Vec<(usize, usize)> {
    Vec::new() // Fallback - empty MST
}

pub const fn strongly_connected_components<N, E>(_graph: &Graph<N, E>) -> Vec<Vec<usize>> {
    Vec::new() // Fallback - no components
}

/// Clustering fit result
#[derive(Debug, Clone)]
pub struct KMeansResult {
    pub labels: Vec<usize>,
    pub centers: Array2<f64>,
    pub silhouette_score: f64,
    pub inertia: f64,
}

/// Basic KMeans clustering fallback implementation (real Lloyd's algorithm).
#[derive(Debug, Clone)]
pub struct KMeans {
    pub n_clusters: usize,
    /// Centroids learned by the most recent `fit` call (column = feature).
    fitted_centers: Option<Array2<f64>>,
}

impl KMeans {
    pub const fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            fitted_centers: None,
        }
    }

    pub fn fit(&mut self, data: &Array2<f64>) -> Result<KMeansResult, String> {
        let n_points = data.nrows();
        let n_features = data.ncols();

        if self.n_clusters == 0 {
            return Err("KMeans requires n_clusters >= 1".to_string());
        }
        if n_points == 0 {
            return Err("KMeans requires a non-empty dataset".to_string());
        }
        if n_points < self.n_clusters {
            return Err(format!(
                "KMeans requires at least n_clusters ({}) data points, got {}",
                self.n_clusters, n_points
            ));
        }

        // Deterministic k-means++ style initialization seeded from the data.
        let mut centers = kmeans_plus_plus_init(data, self.n_clusters);

        let mut labels = vec![0usize; n_points];
        let max_iters = 100;

        for _ in 0..max_iters {
            // Assignment step: nearest centroid by squared Euclidean distance.
            let mut changed = false;
            for p in 0..n_points {
                let mut best = 0usize;
                let mut best_dist = f64::INFINITY;
                for c in 0..self.n_clusters {
                    let mut dist = 0.0;
                    for f in 0..n_features {
                        let diff = data[(p, f)] - centers[(c, f)];
                        dist += diff * diff;
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best = c;
                    }
                }
                if labels[p] != best {
                    labels[p] = best;
                    changed = true;
                }
            }

            // Update step: recompute centroids as the mean of assigned points.
            let mut sums = Array2::<f64>::zeros((self.n_clusters, n_features));
            let mut counts = vec![0usize; self.n_clusters];
            for p in 0..n_points {
                let c = labels[p];
                counts[c] += 1;
                for f in 0..n_features {
                    sums[(c, f)] += data[(p, f)];
                }
            }
            for c in 0..self.n_clusters {
                if counts[c] > 0 {
                    let inv = 1.0 / counts[c] as f64;
                    for f in 0..n_features {
                        centers[(c, f)] = sums[(c, f)] * inv;
                    }
                } else {
                    // Re-seed an empty cluster onto the point farthest from its
                    // assigned centroid to avoid a degenerate (collapsed) cluster.
                    if let Some(far) = farthest_point(data, &centers, &labels) {
                        for f in 0..n_features {
                            centers[(c, f)] = data[(far, f)];
                        }
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        // Inertia: sum of squared distances of points to their centroid.
        let mut inertia = 0.0;
        for p in 0..n_points {
            let c = labels[p];
            for f in 0..n_features {
                let diff = data[(p, f)] - centers[(c, f)];
                inertia += diff * diff;
            }
        }

        let silhouette_score = silhouette(data, &labels, self.n_clusters);

        self.fitted_centers = Some(centers.clone());

        Ok(KMeansResult {
            labels,
            centers,
            silhouette_score,
            inertia,
        })
    }

    pub fn predict(&self, data: &Array2<f64>) -> Result<Array1<usize>, String> {
        let centers = self.fitted_centers.as_ref().ok_or_else(|| {
            "KMeans::predict called before fit; no centroids available".to_string()
        })?;
        let n_features = centers.ncols();
        if data.ncols() != n_features {
            return Err(format!(
                "KMeans::predict feature mismatch: model has {} features, data has {}",
                n_features,
                data.ncols()
            ));
        }
        let n_points = data.nrows();
        let mut labels = Array1::<usize>::zeros(n_points);
        for p in 0..n_points {
            let mut best = 0usize;
            let mut best_dist = f64::INFINITY;
            for c in 0..centers.nrows() {
                let mut dist = 0.0;
                for f in 0..n_features {
                    let diff = data[(p, f)] - centers[(c, f)];
                    dist += diff * diff;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            labels[p] = best;
        }
        Ok(labels)
    }

    pub fn fit_predict(&mut self, data: &Array2<f64>) -> Result<Array1<usize>, String> {
        let result = self.fit(data)?;
        Ok(Array1::from_vec(result.labels))
    }
}

/// Deterministic k-means++ initialization seeded from the data itself.
///
/// The first centroid is the data point closest to the global mean (a
/// reproducible, data-driven choice). Each subsequent centroid is the point
/// that maximizes the minimum squared distance to the already-chosen centroids
/// (the deterministic "farthest-point" variant of k-means++ — no RNG required).
fn kmeans_plus_plus_init(data: &Array2<f64>, k: usize) -> Array2<f64> {
    let n_points = data.nrows();
    let n_features = data.ncols();
    let mut centers = Array2::<f64>::zeros((k, n_features));

    if n_points == 0 || k == 0 {
        return centers;
    }

    // First centroid: the point nearest the global mean.
    let mut mean = Array1::<f64>::zeros(n_features);
    for p in 0..n_points {
        for f in 0..n_features {
            mean[f] += data[(p, f)];
        }
    }
    for f in 0..n_features {
        mean[f] /= n_points as f64;
    }

    let mut first = 0usize;
    let mut first_dist = f64::INFINITY;
    for p in 0..n_points {
        let mut dist = 0.0;
        for f in 0..n_features {
            let diff = data[(p, f)] - mean[f];
            dist += diff * diff;
        }
        if dist < first_dist {
            first_dist = dist;
            first = p;
        }
    }
    for f in 0..n_features {
        centers[(0, f)] = data[(first, f)];
    }
    let mut chosen = vec![first];

    // Remaining centroids via deterministic farthest-point selection.
    for c in 1..k {
        let mut best_point = 0usize;
        let mut best_min_dist = -1.0;
        for p in 0..n_points {
            if chosen.contains(&p) {
                continue;
            }
            let mut min_dist = f64::INFINITY;
            for &cc in &chosen {
                let mut dist = 0.0;
                for f in 0..n_features {
                    let diff = data[(p, f)] - data[(cc, f)];
                    dist += diff * diff;
                }
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            if min_dist > best_min_dist {
                best_min_dist = min_dist;
                best_point = p;
            }
        }
        for f in 0..n_features {
            centers[(c, f)] = data[(best_point, f)];
        }
        chosen.push(best_point);
    }

    centers
}

/// Index of the point with the largest squared distance to its assigned
/// centroid. Used to re-seed empty clusters during Lloyd iteration.
fn farthest_point(data: &Array2<f64>, centers: &Array2<f64>, labels: &[usize]) -> Option<usize> {
    let n_points = data.nrows();
    let n_features = data.ncols();
    let mut best = None;
    let mut best_dist = -1.0;
    for p in 0..n_points {
        let c = labels[p];
        let mut dist = 0.0;
        for f in 0..n_features {
            let diff = data[(p, f)] - centers[(c, f)];
            dist += diff * diff;
        }
        if dist > best_dist {
            best_dist = dist;
            best = Some(p);
        }
    }
    best
}

/// Mean silhouette coefficient over all samples (Euclidean distance).
///
/// Returns 0.0 when the score is undefined (fewer than two clusters or
/// singleton clusters), matching the conventional silhouette definition.
fn silhouette(data: &Array2<f64>, labels: &[usize], n_clusters: usize) -> f64 {
    let n_points = data.nrows();
    if n_clusters < 2 || n_points < 2 {
        return 0.0;
    }
    let n_features = data.ncols();

    let dist = |a: usize, b: usize| -> f64 {
        let mut acc = 0.0;
        for f in 0..n_features {
            let diff = data[(a, f)] - data[(b, f)];
            acc += diff * diff;
        }
        acc.sqrt()
    };

    let mut total = 0.0;
    for i in 0..n_points {
        let ci = labels[i];

        // a(i): mean intra-cluster distance.
        let mut a_sum = 0.0;
        let mut a_count = 0usize;
        for j in 0..n_points {
            if j != i && labels[j] == ci {
                a_sum += dist(i, j);
                a_count += 1;
            }
        }
        // Singleton cluster contributes silhouette 0 by convention.
        if a_count == 0 {
            continue;
        }
        let a_i = a_sum / a_count as f64;

        // b(i): minimum mean distance to any other cluster.
        let mut b_i = f64::INFINITY;
        for other in 0..n_clusters {
            if other == ci {
                continue;
            }
            let mut b_sum = 0.0;
            let mut b_count = 0usize;
            for j in 0..n_points {
                if labels[j] == other {
                    b_sum += dist(i, j);
                    b_count += 1;
                }
            }
            if b_count > 0 {
                let mean_other = b_sum / b_count as f64;
                if mean_other < b_i {
                    b_i = mean_other;
                }
            }
        }
        if b_i.is_finite() {
            let denom = a_i.max(b_i);
            if denom > 0.0 {
                total += (b_i - a_i) / denom;
            }
        }
    }

    total / n_points as f64
}

/// Other ML algorithm fallbacks
#[derive(Debug, Clone)]
pub struct DBSCAN;

impl Default for DBSCAN {
    fn default() -> Self {
        Self::new()
    }
}

impl DBSCAN {
    pub const fn new() -> Self {
        Self
    }
    pub fn fit_predict(&mut self, _data: &Array2<f64>) -> Result<Array1<i32>, String> {
        let n_points = _data.nrows();
        Ok(Array1::zeros(n_points)) // All points in cluster 0
    }
}

#[derive(Debug, Clone)]
pub struct IsolationForest;

impl Default for IsolationForest {
    fn default() -> Self {
        Self::new()
    }
}

impl IsolationForest {
    pub const fn new() -> Self {
        Self
    }
    pub const fn fit(&mut self, _data: &Array2<f64>) -> Result<(), String> {
        Ok(())
    }
    pub fn predict(&self, _data: &Array2<f64>) -> Result<Array1<i32>, String> {
        let n_points = _data.nrows();
        Ok(Array1::ones(n_points)) // All points are inliers (1)
    }
    pub fn decision_function(&self, _data: &Array2<f64>) -> Result<Array1<f64>, String> {
        let n_points = _data.nrows();
        Ok(Array1::ones(n_points) * 0.5) // Neutral anomaly scores
    }
}

pub fn train_test_split<T: Clone>(
    data: &Array2<T>,
    targets: &Array1<T>,
    test_size: f64,
) -> (Array2<T>, Array2<T>, Array1<T>, Array1<T>) {
    let n = data.nrows();
    let test_n = (n as f64 * test_size) as usize;
    let train_n = n - test_n;

    // Simple split without shuffling for fallback
    let x_train = data
        .slice(scirs2_core::ndarray::s![0..train_n, ..])
        .to_owned();
    let x_test = data
        .slice(scirs2_core::ndarray::s![train_n.., ..])
        .to_owned();
    let y_train = targets
        .slice(scirs2_core::ndarray::s![0..train_n])
        .to_owned();
    let y_test = targets
        .slice(scirs2_core::ndarray::s![train_n..])
        .to_owned();

    (x_train, x_test, y_train, y_test)
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
        // eig of diag([3, 1]) must return eigenvalues {1, 3} (ascending).
        let m = array![[3.0, 0.0], [0.0, 1.0]];
        let (vals, vecs) = eig(&m).expect("eig should succeed");
        assert!(approx(vals[0], 1.0, 1e-9), "got {}", vals[0]);
        assert!(approx(vals[1], 3.0, 1e-9), "got {}", vals[1]);
        // Eigenvectors orthonormal: V^T V == I.
        for i in 0..2 {
            for j in 0..2 {
                let mut dot = 0.0;
                for k in 0..2 {
                    dot += vecs[(k, i)] * vecs[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx(dot, expected, 1e-9));
            }
        }
    }

    #[test]
    fn test_eig_symmetric_reconstruction() {
        // For symmetric A: A == V diag(lambda) V^T.
        let a = array![[2.0, 1.0], [1.0, 2.0]];
        let (vals, vecs) = eig(&a).expect("eig should succeed");
        // Eigenvalues of [[2,1],[1,2]] are 1 and 3.
        assert!(approx(vals[0], 1.0, 1e-9));
        assert!(approx(vals[1], 3.0, 1e-9));
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

    #[test]
    fn test_svd_reconstruction() {
        // U diag(S) Vt should reconstruct A.
        let a = array![[3.0, 1.0], [1.0, 3.0], [0.0, 2.0]];
        let (u, s, vt) = svd(&a).expect("svd should succeed");
        let (m, n) = a.dim();
        for r in 0..m {
            for c in 0..n {
                let mut recon = 0.0;
                for k in 0..n.min(m) {
                    recon += u[(r, k)] * s[k] * vt[(k, c)];
                }
                assert!(
                    approx(recon, a[(r, c)], 1e-6),
                    "recon[{r},{c}]={recon} expected {}",
                    a[(r, c)]
                );
            }
        }
        // Singular values must be non-negative and sorted descending.
        for k in 1..s.len() {
            assert!(s[k] <= s[k - 1] + 1e-12);
            assert!(s[k] >= -1e-12);
        }
    }

    #[test]
    fn test_svd_singular_values_known() {
        // Diagonal matrix => singular values are |diagonal| sorted descending.
        let a = array![[2.0, 0.0], [0.0, 5.0]];
        let (_, s, _) = svd(&a).expect("svd should succeed");
        assert!(approx(s[0], 5.0, 1e-9));
        assert!(approx(s[1], 2.0, 1e-9));
    }

    #[test]
    fn test_kmeans_two_clusters() {
        // Two well-separated clusters around (0,0) and (10,10).
        let data = array![
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.2],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.8, 10.2],
        ];
        let mut km = KMeans::new(2);
        let result = km.fit(&data).expect("kmeans fit should succeed");

        // The first three points share a label distinct from the last three.
        let l0 = result.labels[0];
        assert_eq!(result.labels[1], l0);
        assert_eq!(result.labels[2], l0);
        let l1 = result.labels[3];
        assert_eq!(result.labels[4], l1);
        assert_eq!(result.labels[5], l1);
        assert_ne!(l0, l1, "clusters must be separated");

        // Centers near the true cluster means (0,0) and (10,10).
        let mut near_origin = false;
        let mut near_ten = false;
        for c in 0..2 {
            let cx = result.centers[(c, 0)];
            let cy = result.centers[(c, 1)];
            if approx(cx, 0.0, 0.5) && approx(cy, 0.0, 0.5) {
                near_origin = true;
            }
            if approx(cx, 10.0, 0.5) && approx(cy, 10.0, 0.5) {
                near_ten = true;
            }
        }
        assert!(near_origin && near_ten, "centers must match true means");

        // Inertia must be small and finite for tight, well-separated clusters.
        assert!(result.inertia.is_finite());
        assert!(result.inertia < 1.0, "inertia {} too large", result.inertia);

        // Well-separated clusters => silhouette close to 1.
        assert!(
            result.silhouette_score > 0.8,
            "silhouette {} too low",
            result.silhouette_score
        );
    }

    #[test]
    fn test_kmeans_predict_matches_fit() {
        let data = array![[0.0, 0.0], [0.2, 0.1], [9.0, 9.0], [9.1, 8.9],];
        let mut km = KMeans::new(2);
        let fitted = km.fit(&data).expect("fit should succeed");
        let predicted = km.predict(&data).expect("predict should succeed");
        for i in 0..data.nrows() {
            assert_eq!(predicted[i], fitted.labels[i]);
        }
    }

    #[test]
    fn test_kmeans_predict_before_fit_errors() {
        let km = KMeans::new(2);
        let data = array![[0.0, 0.0]];
        assert!(km.predict(&data).is_err());
    }

    #[test]
    fn test_kmeans_too_few_points_errors() {
        let data = array![[0.0, 0.0]];
        let mut km = KMeans::new(3);
        assert!(km.fit(&data).is_err());
    }
}
