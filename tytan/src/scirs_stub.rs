//! Stub for SciRS2 integration
//!
//! This module provides placeholders for SciRS2 integration.
//! The actual integration would be more comprehensive once
//! the SciRS2 API stabilizes.

#![allow(dead_code)]

use ::scirs2_core::ndarray::{Array1, Array2, ArrayD};
use ::scirs2_core::random::prelude::*;

/// Placeholder for enhanced QUBO operations
pub fn enhance_qubo_matrix(matrix: &Array2<f64>) -> Array2<f64> {
    // In a real implementation, this would:
    // - Convert to sparse format
    // - Apply optimizations
    // - Use BLAS operations
    matrix.clone()
}

/// Placeholder for HOBO tensor operations
pub fn optimize_hobo_tensor(tensor: &ArrayD<f64>) -> ArrayD<f64> {
    // In a real implementation, this would:
    // - Apply tensor decomposition
    // - Use efficient tensor operations
    // - Leverage parallelization
    tensor.clone()
}

/// Placeholder for parallel sampling
pub fn parallel_sample_qubo(matrix: &Array2<f64>, num_samples: usize) -> Vec<(Vec<bool>, f64)> {
    // In a real implementation, this would use parallel processing
    let n = matrix.shape()[0];
    let mut results = Vec::with_capacity(num_samples);

    // use rand::{rng, Rng}; // Replaced by scirs2_core::random::prelude::*
    let mut rng = thread_rng();

    for _ in 0..num_samples {
        let solution: Vec<bool> = (0..n).map(|_| rng.random()).collect();
        let energy = evaluate_qubo(&solution, matrix);
        results.push((solution, energy));
    }

    results
}

fn evaluate_qubo(solution: &[bool], matrix: &Array2<f64>) -> f64 {
    let mut energy = 0.0;
    let n = solution.len();

    for i in 0..n {
        if solution[i] {
            energy += matrix[[i, i]];
            for j in (i + 1)..n {
                if solution[j] {
                    energy += matrix[[i, j]];
                }
            }
        }
    }

    energy
}

// ---------------------------------------------------------------------
// Real numerical algorithms shared by the stub submodules below.
//
// These used to be fabricated (constant-cluster-0 clustering, an
// identity-returning "optimizer", a components matrix of literal zeros for
// PCA, hardcoded feature importances, ...). The functions in this section
// are genuine, locally-computable implementations that the stub structs
// below delegate to, so that a caller using `scirs_stub::scirs2_ml::KMeans`
// (etc.) gets a real answer instead of a plausible-looking constant.
// ---------------------------------------------------------------------

/// Real Lloyd's-algorithm k-means clustering with k-means++ initialization.
fn lloyd_kmeans(data: &Array2<f64>, k: usize, max_iter: usize) -> Vec<usize> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    if n_samples == 0 {
        return Vec::new();
    }
    let k = k.clamp(1, n_samples);

    let squared_dist = |i: usize, j: &[f64]| -> f64 {
        data.row(i)
            .iter()
            .zip(j.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum()
    };

    let mut rng = thread_rng();
    let mut centroids: Array2<f64> = Array2::zeros((k, n_features));

    // k-means++ initialization: pick centroids with probability proportional
    // to squared distance from the closest already-chosen centroid.
    let first = rng.random_range(0..n_samples);
    centroids.row_mut(0).assign(&data.row(first));
    let mut chosen_rows: Vec<Vec<f64>> = vec![data.row(first).to_vec()];

    for c in 1..k {
        let distances: Vec<f64> = (0..n_samples)
            .map(|i| {
                chosen_rows
                    .iter()
                    .map(|row| squared_dist(i, row))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let total: f64 = distances.iter().sum();

        let pick = if total > 0.0 {
            let target = rng.random::<f64>() * total;
            let mut cumulative = 0.0;
            let mut selected = n_samples - 1;
            for (i, &d) in distances.iter().enumerate() {
                cumulative += d;
                if cumulative >= target {
                    selected = i;
                    break;
                }
            }
            selected
        } else {
            rng.random_range(0..n_samples)
        };

        centroids.row_mut(c).assign(&data.row(pick));
        chosen_rows.push(data.row(pick).to_vec());
    }

    let mut assignments = vec![0usize; n_samples];
    for _iteration in 0..max_iter.max(1) {
        let mut changed = false;

        for i in 0..n_samples {
            let mut best_cluster = 0;
            let mut best_dist = f64::INFINITY;
            for c in 0..k {
                let d: f64 = data
                    .row(i)
                    .iter()
                    .zip(centroids.row(c).iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if d < best_dist {
                    best_dist = d;
                    best_cluster = c;
                }
            }
            if assignments[i] != best_cluster {
                changed = true;
            }
            assignments[i] = best_cluster;
        }

        let mut sums = Array2::<f64>::zeros((k, n_features));
        let mut counts = vec![0usize; k];
        for i in 0..n_samples {
            let c = assignments[i];
            for f in 0..n_features {
                sums[[c, f]] += data[[i, f]];
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for f in 0..n_features {
                    centroids[[c, f]] = sums[[c, f]] / counts[c] as f64;
                }
            }
        }

        if !changed {
            break;
        }
    }

    assignments
}

/// Real density-based clustering (DBSCAN). Noise points are all assigned to
/// one shared trailing cluster id (`max_real_cluster_id + 1`) so the return
/// type stays a plain `Vec<usize>` with no sentinel value the caller needs
/// to special-case.
fn dbscan_cluster(data: &Array2<f64>, eps: f64, min_samples: usize) -> Vec<usize> {
    let n = data.nrows();
    if n == 0 {
        return Vec::new();
    }

    const UNVISITED: usize = usize::MAX;
    let mut labels = vec![UNVISITED; n];
    let mut visited = vec![false; n];
    let mut next_cluster = 0usize;

    let region_query = |point: usize| -> Vec<usize> {
        (0..n)
            .filter(|&other| {
                let d: f64 = data
                    .row(point)
                    .iter()
                    .zip(data.row(other).iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                d <= eps
            })
            .collect()
    };

    for point in 0..n {
        if visited[point] {
            continue;
        }
        visited[point] = true;

        let mut seeds = region_query(point);
        if seeds.len() < min_samples {
            continue; // stays UNVISITED-labeled -> remapped to "noise" below
        }

        labels[point] = next_cluster;
        let mut idx = 0;
        while idx < seeds.len() {
            let q = seeds[idx];
            idx += 1;

            if !visited[q] {
                visited[q] = true;
                let q_neighbors = region_query(q);
                if q_neighbors.len() >= min_samples {
                    for candidate in q_neighbors {
                        if !seeds.contains(&candidate) {
                            seeds.push(candidate);
                        }
                    }
                }
            }

            if labels[q] == UNVISITED {
                labels[q] = next_cluster;
            }
        }

        next_cluster += 1;
    }

    for label in &mut labels {
        if *label == UNVISITED {
            *label = next_cluster;
        }
    }

    labels
}

/// Real agglomerative (hierarchical) clustering. `linkage` selects how
/// inter-cluster distance is computed: `"complete"` (max pairwise distance),
/// `"average"` (mean pairwise distance), or anything else falls back to
/// `"single"` (min pairwise distance).
fn agglomerative_cluster(data: &Array2<f64>, n_clusters: usize, linkage: &str) -> Vec<usize> {
    let n = data.nrows();
    if n == 0 {
        return Vec::new();
    }
    let n_clusters = n_clusters.clamp(1, n);

    let point_dist = |i: usize, j: usize| -> f64 {
        data.row(i)
            .iter()
            .zip(data.row(j).iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    };

    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active: Vec<usize> = (0..n).collect();

    while active.len() > n_clusters {
        let mut best_pair = (0usize, 1usize);
        let mut best_distance = f64::INFINITY;

        for a in 0..active.len() {
            for b in (a + 1)..active.len() {
                let cluster_a = active[a];
                let cluster_b = active[b];
                let pairwise: Vec<f64> = members[cluster_a]
                    .iter()
                    .flat_map(|&i| members[cluster_b].iter().map(move |&j| (i, j)))
                    .map(|(i, j)| point_dist(i, j))
                    .collect();

                let distance = match linkage {
                    "complete" => pairwise.iter().copied().fold(0.0, f64::max),
                    "average" => pairwise.iter().sum::<f64>() / pairwise.len() as f64,
                    _ => pairwise.iter().copied().fold(f64::INFINITY, f64::min),
                };

                if distance < best_distance {
                    best_distance = distance;
                    best_pair = (a, b);
                }
            }
        }

        let (a_idx, b_idx) = best_pair;
        let cluster_a = active[a_idx];
        let cluster_b = active[b_idx];
        let merged = members[cluster_b].clone();
        members[cluster_a].extend(merged);
        active.remove(b_idx);
    }

    let mut labels = vec![0usize; n];
    for (label, &cluster) in active.iter().enumerate() {
        for &point in &members[cluster] {
            labels[point] = label;
        }
    }
    labels
}

/// Real Principal Component Analysis via the cyclic Jacobi eigenvalue
/// algorithm applied to the covariance matrix. Returns the sample scores
/// projected onto the top `n_components` principal axes (descending
/// eigenvalue order).
fn real_pca_transform(data: &Array2<f64>, n_components: usize) -> Array2<f64> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    let n_components = n_components
        .clamp(1, n_features.max(1))
        .min(n_samples.max(1));

    if n_samples == 0 || n_features == 0 {
        return Array2::zeros((n_samples, n_components));
    }

    let mean = data
        .mean_axis(::scirs2_core::ndarray::Axis(0))
        .unwrap_or_else(|| Array1::zeros(n_features));
    let centered = data - &mean;

    let denom = (n_samples.saturating_sub(1)).max(1) as f64;
    let cov = centered.t().dot(&centered) / denom;

    let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&cov, 200, 1e-12);

    let mut order: Vec<usize> = (0..n_features).collect();
    order.sort_by(|&a, &b| {
        eigenvalues[b]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut components = Array2::<f64>::zeros((n_samples, n_components));
    for (k, &feature_idx) in order.iter().take(n_components).enumerate() {
        let axis = eigenvectors.column(feature_idx);
        for sample in 0..n_samples {
            let score: f64 = centered
                .row(sample)
                .iter()
                .zip(axis.iter())
                .map(|(&x, &a)| x * a)
                .sum();
            components[[sample, k]] = score;
        }
    }

    components
}

/// Cyclic Jacobi eigenvalue algorithm for a real symmetric matrix. Returns
/// `(eigenvalues, eigenvectors)` with eigenvector `i` stored in column `i` of
/// the returned matrix (unsorted; sort by eigenvalue if a specific order is
/// required).
fn jacobi_eigen_symmetric(
    matrix: &Array2<f64>,
    max_sweeps: usize,
    tolerance: f64,
) -> (Vec<f64>, Array2<f64>) {
    let n = matrix.nrows();
    let mut a = matrix.clone();
    let mut v = Array2::<f64>::eye(n);

    for _sweep in 0..max_sweeps {
        let mut off_diagonal_norm = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off_diagonal_norm += a[[p, q]] * a[[p, q]];
            }
        }
        if off_diagonal_norm.sqrt() < tolerance {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let a_pq = a[[p, q]];
                if a_pq.abs() < 1e-300 {
                    continue;
                }

                let theta = (a[[q, q]] - a[[p, p]]) / (2.0 * a_pq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + theta.mul_add(theta, 1.0).sqrt());
                let c = 1.0 / t.mul_add(t, 1.0).sqrt();
                let s = t * c;

                a[[p, p]] -= t * a_pq;
                a[[q, q]] += t * a_pq;
                a[[p, q]] = 0.0;
                a[[q, p]] = 0.0;

                for k in 0..n {
                    if k != p && k != q {
                        let a_kp = a[[k, p]];
                        let a_kq = a[[k, q]];
                        a[[k, p]] = c.mul_add(a_kp, -(s * a_kq));
                        a[[p, k]] = a[[k, p]];
                        a[[k, q]] = s.mul_add(a_kp, c * a_kq);
                        a[[q, k]] = a[[k, q]];
                    }
                }

                for k in 0..n {
                    let v_kp = v[[k, p]];
                    let v_kq = v[[k, q]];
                    v[[k, p]] = c.mul_add(v_kp, -(s * v_kq));
                    v[[k, q]] = s.mul_add(v_kp, c * v_kq);
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
    (eigenvalues, v)
}

/// A minimal real CART-style regression tree, used to build the
/// `RandomForest` stub below out of an actual bagged tree ensemble instead
/// of a hardcoded constant predictor.
#[derive(Debug, Clone)]
enum RegressionTreeNode {
    Leaf {
        value: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        variance_reduction: f64,
        left: Box<RegressionTreeNode>,
        right: Box<RegressionTreeNode>,
    },
}

impl RegressionTreeNode {
    fn predict(&self, row: &[f64]) -> f64 {
        match self {
            Self::Leaf { value } => *value,
            Self::Split {
                feature,
                threshold,
                left,
                right,
                ..
            } => {
                if row[*feature] <= *threshold {
                    left.predict(row)
                } else {
                    right.predict(row)
                }
            }
        }
    }

    /// Accumulate each feature's total variance reduction into `importance`.
    fn accumulate_importance(&self, importance: &mut [f64]) {
        if let Self::Split {
            feature,
            variance_reduction,
            left,
            right,
            ..
        } = self
        {
            importance[*feature] += variance_reduction;
            left.accumulate_importance(importance);
            right.accumulate_importance(importance);
        }
    }
}

fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64
}

/// Recursively grow a single CART regression tree over `indices` (into `x`/
/// `y`), evaluating a random subset of `n_candidate_features` features at
/// each split (the "random" in "random forest").
fn grow_regression_tree(
    x: &[Vec<f64>],
    y: &[f64],
    indices: &[usize],
    n_features: usize,
    n_candidate_features: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    depth: usize,
    rng: &mut impl Rng,
) -> RegressionTreeNode {
    let node_y: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
    let leaf_value = node_y.iter().sum::<f64>() / node_y.len().max(1) as f64;

    let depth_exhausted = max_depth.is_some_and(|d| depth >= d);
    if indices.len() < min_samples_split.max(2) || depth_exhausted || n_features == 0 {
        return RegressionTreeNode::Leaf { value: leaf_value };
    }

    let parent_variance = variance(&node_y);
    if parent_variance <= 1e-12 {
        return RegressionTreeNode::Leaf { value: leaf_value };
    }

    // Feature bagging: only consider a random subset of features for this split.
    let mut feature_pool: Vec<usize> = (0..n_features).collect();
    for i in (1..feature_pool.len()).rev() {
        let j = rng.random_range(0..=i);
        feature_pool.swap(i, j);
    }
    let candidates = &feature_pool[..n_candidate_features.clamp(1, n_features)];

    let mut best_feature = None;
    let mut best_threshold = 0.0;
    let mut best_variance_reduction = 0.0;
    let mut best_left: Vec<usize> = Vec::new();
    let mut best_right: Vec<usize> = Vec::new();

    for &feature in candidates {
        let mut values: Vec<f64> = indices.iter().map(|&i| x[i][feature]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        for window in values.windows(2) {
            let threshold = (window[0] + window[1]) / 2.0;
            let left: Vec<usize> = indices
                .iter()
                .copied()
                .filter(|&i| x[i][feature] <= threshold)
                .collect();
            let right: Vec<usize> = indices
                .iter()
                .copied()
                .filter(|&i| x[i][feature] > threshold)
                .collect();

            if left.is_empty() || right.is_empty() {
                continue;
            }

            let left_y: Vec<f64> = left.iter().map(|&i| y[i]).collect();
            let right_y: Vec<f64> = right.iter().map(|&i| y[i]).collect();
            let weighted_child_variance = (left_y.len() as f64 * variance(&left_y)
                + right_y.len() as f64 * variance(&right_y))
                / indices.len() as f64;
            let reduction = parent_variance - weighted_child_variance;

            if reduction > best_variance_reduction {
                best_variance_reduction = reduction;
                best_feature = Some(feature);
                best_threshold = threshold;
                best_left = left;
                best_right = right;
            }
        }
    }

    match best_feature {
        Some(feature) if best_variance_reduction > 1e-12 => RegressionTreeNode::Split {
            feature,
            threshold: best_threshold,
            variance_reduction: best_variance_reduction * indices.len() as f64,
            left: Box::new(grow_regression_tree(
                x,
                y,
                &best_left,
                n_features,
                n_candidate_features,
                max_depth,
                min_samples_split,
                depth + 1,
                rng,
            )),
            right: Box::new(grow_regression_tree(
                x,
                y,
                &best_right,
                n_features,
                n_candidate_features,
                max_depth,
                min_samples_split,
                depth + 1,
                rng,
            )),
        },
        _ => RegressionTreeNode::Leaf { value: leaf_value },
    }
}

/// Marker that SciRS2 integration is available
pub const SCIRS2_AVAILABLE: bool = cfg!(feature = "scirs");

// When SciRS2 feature is enabled, we still use stubs for now
// until SciRS2 is fully available
pub mod scirs2_core {
    pub use super::scirs2_core_stub::*;
}

pub mod scirs2_linalg {
    pub use super::scirs2_linalg_stub::*;
}

pub mod scirs2_plot {
    pub use super::scirs2_plot_stub::*;
}

pub mod scirs2_statistics {
    pub use super::scirs2_statistics_stub::*;
}

pub mod scirs2_optimization {
    pub use super::scirs2_optimization_stub::*;
}

pub mod scirs2_graphs {
    pub use super::scirs2_graphs_stub::*;
}

pub mod scirs2_ml {
    pub use super::scirs2_ml_stub::*;
}

// Define stub modules that can be used regardless of feature flags
mod scirs2_core_stub {
    use std::error::Error;

    pub fn init_simd() -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    pub mod simd {
        pub trait SimdOps {}
    }

    pub mod memory {
        pub fn get_current_usage() -> Result<usize, std::io::Error> {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Not implemented",
            ))
        }
    }

    pub mod statistics {
        pub struct OnlineStats {
            count: usize,
            mean: f64,
            m2: f64,
        }

        impl Default for OnlineStats {
            fn default() -> Self {
                Self::new()
            }
        }

        impl OnlineStats {
            pub const fn new() -> Self {
                Self {
                    count: 0,
                    mean: 0.0,
                    m2: 0.0,
                }
            }

            pub fn update(&mut self, value: f64) {
                self.count += 1;
                let delta = value - self.mean;
                self.mean += delta / self.count as f64;
                let delta2 = value - self.mean;
                self.m2 += delta * delta2;
            }

            pub const fn mean(&self) -> f64 {
                self.mean
            }

            pub fn variance(&self) -> f64 {
                if self.count < 2 {
                    0.0
                } else {
                    self.m2 / (self.count - 1) as f64
                }
            }
        }

        pub struct MovingAverage {
            window_size: usize,
            values: Vec<f64>,
        }

        impl MovingAverage {
            pub const fn new(window_size: usize) -> Self {
                Self {
                    window_size,
                    values: Vec::new(),
                }
            }

            pub fn update(&mut self, value: f64) {
                self.values.push(value);
                if self.values.len() > self.window_size {
                    self.values.remove(0);
                }
            }

            pub fn mean(&self) -> f64 {
                if self.values.is_empty() {
                    0.0
                } else {
                    self.values.iter().sum::<f64>() / self.values.len() as f64
                }
            }
        }
    }

    pub mod gpu {
        use std::error::Error;
        use std::time::Duration;

        pub const fn get_device_count() -> usize {
            0
        }

        pub struct GpuContext {
            pub device_id: usize,
        }

        impl GpuContext {
            pub fn new(_device_id: usize) -> Result<Self, Box<dyn Error>> {
                Err("GPU not available".into())
            }

            pub fn measure_kernel_latency(&self) -> Result<Duration, Box<dyn Error>> {
                Err("GPU not available".into())
            }

            pub const fn get_device_info(&self) -> DeviceInfo {
                DeviceInfo {
                    memory_mb: 0,
                    compute_units: 0,
                    clock_mhz: 0,
                }
            }
        }

        pub struct DeviceInfo {
            pub memory_mb: usize,
            pub compute_units: usize,
            pub clock_mhz: usize,
        }

        pub struct GpuDevice {
            pub id: u32,
        }

        impl GpuDevice {
            pub fn new(_id: u32) -> Result<Self, Box<dyn Error>> {
                Err("GPU not available".into())
            }

            pub fn random_array<T>(
                &self,
                _shape: (usize, usize),
            ) -> Result<GpuArray<T>, Box<dyn Error>> {
                Err("GPU not available".into())
            }

            pub fn binarize<T>(
                &self,
                _array: &GpuArray<T>,
                _threshold: f64,
            ) -> Result<GpuArray<bool>, Box<dyn Error>> {
                Err("GPU not available".into())
            }

            pub fn qubo_energy<T>(
                &self,
                _states: &GpuArray<bool>,
                _matrix: &GpuArray<T>,
            ) -> Result<GpuArray<f64>, Box<dyn Error>> {
                Err("GPU not available".into())
            }
        }

        impl Clone for GpuDevice {
            fn clone(&self) -> Self {
                Self { id: self.id }
            }
        }

        pub struct GpuArray<T> {
            _phantom: std::marker::PhantomData<T>,
        }

        impl<T> GpuArray<T> {
            pub fn from_ndarray(
                _device: GpuDevice,
                _array: &scirs2_core::ndarray::Array2<T>,
            ) -> Result<Self, Box<dyn Error>> {
                Err("GPU not available".into())
            }

            pub fn to_ndarray(&self) -> Result<scirs2_core::ndarray::Array2<T>, Box<dyn Error>>
            where
                T: Clone + Default,
            {
                Err("GPU not available".into())
            }
        }
    }
}

mod scirs2_linalg_stub {

    pub mod sparse {
        use scirs2_core::ndarray::Array2;

        pub struct SparseMatrix;

        impl SparseMatrix {
            pub const fn from_dense(_matrix: &Array2<f64>) -> Self {
                Self
            }
        }
    }

    pub mod svd {
        pub struct SVD;
    }

    pub mod pca {
        use scirs2_core::ndarray::Array2;
        use std::error::Error;

        pub struct PCA {
            n_components: usize,
        }

        impl PCA {
            pub const fn new(n_components: usize) -> Self {
                Self { n_components }
            }

            /// Real PCA (via covariance-matrix eigendecomposition). This
            /// used to just truncate the raw input to its first
            /// `n_components` columns, which is not a projection at all.
            pub fn fit_transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
                Ok(super::super::real_pca_transform(data, self.n_components))
            }
        }
    }

    pub mod norm {
        use scirs2_core::ndarray::Array1;

        pub trait Norm {
            fn norm(&self) -> f64;
        }

        impl Norm for Array1<f64> {
            fn norm(&self) -> f64 {
                self.iter().map(|x| x * x).sum::<f64>().sqrt()
            }
        }
    }

    pub mod gpu {
        use super::*;
        use scirs2_core::ndarray::Array2;
        use std::error::Error;

        pub struct GpuMatrix;

        impl GpuMatrix {
            pub fn from_host(
                _matrix: &Array2<f64>,
                _ctx: &crate::scirs_stub::scirs2_core::gpu::GpuContext,
            ) -> Result<Self, Box<dyn Error>> {
                Err("GPU not available".into())
            }

            pub fn to_host(&self) -> Result<Array2<f64>, Box<dyn Error>> {
                Err("GPU not available".into())
            }
        }
    }
}

mod scirs2_plot_stub {
    use std::error::Error;

    pub struct Figure;
    pub struct Subplot;
    pub struct Plot2D;
    pub struct Plot3D;
    pub struct Heatmap;
    pub struct ColorMap;
    pub struct Plot;
    pub struct Line;
    pub struct Scatter;
    pub struct Bar;
    pub struct NetworkPlot;
    pub struct MultiPlot;
    pub struct Annotation;
    pub struct Violin;
    pub struct BoxPlot;

    impl Default for Plot {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Plot {
        pub const fn new() -> Self {
            Self
        }
        pub fn add_trace(&mut self, _trace: impl Trace) {}
        pub const fn set_title(&mut self, _title: &str) {}
        pub const fn set_xlabel(&mut self, _label: &str) {}
        pub const fn set_ylabel(&mut self, _label: &str) {}
        pub fn save(&self, _path: &str) -> Result<(), Box<dyn Error>> {
            Err("Plotting not available".into())
        }
    }

    impl Line {
        pub fn new(_x: Vec<f64>, _y: Vec<f64>) -> Self {
            Self
        }
        pub const fn name(self, _name: &str) -> Self {
            self
        }
    }

    impl Scatter {
        pub fn new(_x: Vec<f64>, _y: Vec<f64>) -> Self {
            Self
        }
        pub const fn name(self, _name: &str) -> Self {
            self
        }
        pub const fn mode(self, _mode: &str) -> Self {
            self
        }
        pub const fn marker_size(self, _size: u32) -> Self {
            self
        }
        pub fn text(self, _text: Vec<String>) -> Self {
            self
        }
    }

    impl Heatmap {
        pub fn new(_z: Vec<Vec<f64>>) -> Self {
            Self
        }
        pub fn x(self, _x: Vec<f64>) -> Self {
            self
        }
        pub fn y(self, _y: Vec<f64>) -> Self {
            self
        }
        pub fn x_labels(self, _labels: Vec<String>) -> Self {
            self
        }
        pub fn y_labels(self, _labels: Vec<String>) -> Self {
            self
        }
        pub const fn colorscale(self, _scale: &str) -> Self {
            self
        }
    }

    impl Bar {
        pub fn new(_x: Vec<String>, _y: Vec<f64>) -> Self {
            Self
        }
        pub const fn name(self, _name: &str) -> Self {
            self
        }
    }

    impl Trace for Line {}
    impl Trace for Scatter {}
    impl Trace for Heatmap {}
    impl Trace for Bar {}

    impl Default for Figure {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Figure {
        pub const fn new() -> Self {
            Self
        }

        pub fn add_subplot(
            &mut self,
            _rows: usize,
            _cols: usize,
            _idx: usize,
        ) -> Result<Subplot, Box<dyn Error>> {
            Ok(Subplot)
        }

        pub const fn suptitle(&mut self, _title: &str) {}
        pub const fn tight_layout(&mut self) {}
        pub fn show(&self) -> Result<(), Box<dyn Error>> {
            Err("Plotting not available".into())
        }
    }

    impl Subplot {
        pub const fn bar(&self, _x: &[f64], _y: &[f64]) -> Self {
            Self
        }
        pub const fn scatter(&self, _x: &[f64], _y: &[f64]) -> Self {
            Self
        }
        pub const fn plot(&self, _x: &[f64], _y: &[f64]) -> Self {
            Self
        }
        pub const fn contourf(&self, _x: &[f64], _y: &[f64], _z: &[f64]) -> Self {
            Self
        }
        pub const fn barh(&self, _y: &[f64], _width: &[f64], _left: &[f64], _height: f64) -> Self {
            Self
        }
        pub const fn pie(&self, _sizes: &[f64], _labels: &[String]) -> Self {
            Self
        }
        pub const fn bar_horizontal(&self, _names: &[String], _values: &[f64]) -> Self {
            Self
        }
        pub const fn text(&self, _x: f64, _y: f64, _text: &str) -> Self {
            Self
        }
        pub const fn axvline(&self, _x: f64) -> Self {
            Self
        }

        pub const fn set_xlabel(&self, _label: &str) -> &Self {
            self
        }
        pub const fn set_ylabel(&self, _label: &str) -> &Self {
            self
        }
        pub const fn set_title(&self, _title: &str) -> &Self {
            self
        }
        pub const fn set_color(&self, _color: &str) -> &Self {
            self
        }
        pub const fn set_color_data(&self, _data: &[f64]) -> &Self {
            self
        }
        pub const fn set_colormap(&self, _cmap: &str) -> &Self {
            self
        }
        pub const fn set_label(&self, _label: &str) -> &Self {
            self
        }
        pub const fn set_linewidth(&self, _width: f64) -> &Self {
            self
        }
        pub const fn set_linestyle(&self, _style: &str) -> &Self {
            self
        }
        pub const fn set_alpha(&self, _alpha: f64) -> &Self {
            self
        }
        pub const fn set_size(&self, _size: f64) -> &Self {
            self
        }
        pub const fn set_edgecolor(&self, _color: &str) -> &Self {
            self
        }
        pub const fn set_marker(&self, _marker: &str) -> &Self {
            self
        }
        pub const fn set_fontsize(&self, _size: u32) -> &Self {
            self
        }
        pub const fn set_ha(&self, _align: &str) -> &Self {
            self
        }
        pub const fn set_va(&self, _align: &str) -> &Self {
            self
        }
        pub const fn set_verticalalignment(&self, _align: &str) -> &Self {
            self
        }
        pub const fn set_transform(&self, _transform: ()) -> &Self {
            self
        }
        pub const fn set_autopct(&self, _fmt: &str) -> &Self {
            self
        }
        pub const fn set_aspect(&self, _aspect: &str) {}
        pub const fn set_yscale(&self, _scale: &str) {}
        pub const fn set_xlim(&self, _min: f64, _max: f64) {}
        pub const fn set_ylim(&self, _min: f64, _max: f64) {}
        pub const fn set_axis_off(&self) {}
        pub const fn set_xticks(&self, _ticks: &[f64]) {}
        pub const fn set_yticks(&self, _ticks: &[f64]) {}
        pub const fn set_xticklabels(&self, _labels: &[String]) {}
        pub const fn set_yticklabels(&self, _labels: &[String]) {}
        pub fn get_xticklabels(&self) -> Vec<TickLabel> {
            vec![TickLabel; self.get_xticks().len()]
        }
        pub const fn get_xticks(&self) -> Vec<f64> {
            vec![]
        }
        pub const fn axis(&self, _setting: &str) {}
        pub const fn legend(&self) {}
        pub const fn legend_unique(&self) {}
        pub const fn trans_axes(&self) {}
    }

    #[derive(Clone)]
    pub struct TickLabel;

    impl TickLabel {
        pub const fn set_rotation(&self, _angle: u32) {}
        pub const fn set_ha(&self, _align: &str) {}
    }

    pub trait Trace {}
}

mod scirs2_statistics_stub {

    pub mod descriptive {
        pub const fn mean(_data: &[f64]) -> f64 {
            0.0
        }
        pub const fn std_dev(_data: &[f64]) -> f64 {
            0.0
        }
        pub const fn quantile(_data: &[f64], _q: f64) -> f64 {
            0.0
        }
    }

    pub mod clustering {
        use scirs2_core::ndarray::Array2;
        use std::error::Error;

        pub struct KMeans {
            k: usize,
        }

        impl KMeans {
            pub const fn new(k: usize) -> Self {
                Self { k }
            }

            /// Real Lloyd's-algorithm k-means clustering (k-means++
            /// initialization). This used to unconditionally return cluster
            /// `0` for every point.
            pub fn fit_predict(&self, data: &Array2<f64>) -> Result<Vec<usize>, Box<dyn Error>> {
                Ok(super::super::lloyd_kmeans(data, self.k, 100))
            }
        }

        pub struct DBSCAN {
            eps: f64,
            min_samples: usize,
        }

        impl DBSCAN {
            pub const fn new(eps: f64, min_samples: usize) -> Self {
                Self { eps, min_samples }
            }

            /// Real density-based clustering. This used to unconditionally
            /// return cluster `0` for every point.
            pub fn fit_predict(&self, data: &Array2<f64>) -> Result<Vec<usize>, Box<dyn Error>> {
                Ok(super::super::dbscan_cluster(
                    data,
                    self.eps,
                    self.min_samples,
                ))
            }
        }

        /// Real agglomerative hierarchical clustering. This used to
        /// unconditionally return cluster `0` for every point.
        pub fn hierarchical_clustering(
            data: &Array2<f64>,
            n_clusters: usize,
            linkage: &str,
        ) -> Result<Vec<usize>, Box<dyn Error>> {
            Ok(super::super::agglomerative_cluster(
                data, n_clusters, linkage,
            ))
        }
    }

    pub mod kde {
        use std::error::Error;

        pub struct KernelDensityEstimator;

        impl KernelDensityEstimator {
            pub fn new(_kernel: &str) -> Result<Self, Box<dyn Error>> {
                Ok(Self)
            }

            pub fn estimate_2d(
                &self,
                _x: &[f64],
                _y: &[f64],
                _xi: f64,
                _yi: f64,
            ) -> Result<f64, Box<dyn Error>> {
                Ok(0.0)
            }
        }
    }
}

mod scirs2_optimization_stub {
    use scirs2_core::ndarray::Array1;
    use std::error::Error;

    pub trait Optimizer: Send {
        fn minimize(
            &mut self,
            objective: &dyn ObjectiveFunction,
            x0: &Array1<f64>,
            bounds: &Bounds,
            max_iter: usize,
        ) -> Result<OptimizationResult, Box<dyn Error>>;
    }

    pub trait OptimizationProblem {}

    pub trait ObjectiveFunction {
        fn evaluate(&self, x: &Array1<f64>) -> f64;
        fn gradient(&self, x: &Array1<f64>) -> Array1<f64>;
    }

    pub struct Bounds {
        lower: Array1<f64>,
        upper: Array1<f64>,
    }

    impl Bounds {
        pub const fn new(lower: Array1<f64>, upper: Array1<f64>) -> Self {
            Self { lower, upper }
        }
    }

    pub struct OptimizationResult {
        pub x: Array1<f64>,
        pub f: f64,
        pub iterations: usize,
    }

    pub mod gradient {
        use super::*;
        use scirs2_core::ndarray::ArrayView1;

        pub struct LBFGS {
            dim: usize,
        }

        impl LBFGS {
            pub const fn new(dim: usize) -> Self {
                Self { dim }
            }
        }

        impl Optimizer for LBFGS {
            /// Delegates to the real, workspace `scirs2-optimize` dependency's
            /// bound-constrained L-BFGS-B solver
            /// (`scirs2_optimize::unconstrained::lbfgsb::LBFGSB`, implementing
            /// Byrd/Lu/Nocedal/Zhu 1995's generalised Cauchy point +
            /// subspace-minimisation algorithm). This used to return the
            /// initial guess unchanged after a single (non-optimizing)
            /// evaluation.
            ///
            /// Note: the free function
            /// `scirs2_optimize::unconstrained::minimize_lbfgsb` in the same
            /// dependency has a sign bug in its very first search-direction
            /// computation (it negates the already-negated steepest-descent
            /// direction, moving uphill); the `LBFGSB` struct-based solver
            /// used here does not share that bug, so it is used instead
            /// even though it computes its own finite-difference gradient
            /// rather than taking `objective.gradient()` directly.
            fn minimize(
                &mut self,
                objective: &dyn ObjectiveFunction,
                x0: &Array1<f64>,
                bounds: &Bounds,
                max_iter: usize,
            ) -> Result<OptimizationResult, Box<dyn Error>> {
                let _ = self.dim; // dimension is implied by `x0`; kept for API compatibility.

                let scirs_bounds = ::scirs2_optimize::unconstrained::Bounds {
                    lower: bounds.lower.iter().map(|&v| Some(v)).collect(),
                    upper: bounds.upper.iter().map(|&v| Some(v)).collect(),
                };

                let options = ::scirs2_optimize::unconstrained::lbfgsb::LBFGSBOptions {
                    max_iter: max_iter.max(1),
                    bounds: Some(scirs_bounds),
                    ..::scirs2_optimize::unconstrained::lbfgsb::LBFGSBOptions::default()
                };

                let solver = ::scirs2_optimize::unconstrained::lbfgsb::LBFGSB::new(options);
                let x0_slice: Vec<f64> = x0.to_vec();

                let result = solver
                    .minimize(
                        |x: &ArrayView1<f64>| objective.evaluate(&x.to_owned()),
                        &x0_slice,
                    )
                    .map_err(|e| Box::new(e) as Box<dyn Error>)?;

                Ok(OptimizationResult {
                    x: result.x,
                    f: result.f_val,
                    iterations: result.n_iter,
                })
            }
        }
    }

    pub mod bayesian {
        use super::*;

        #[derive(Debug, Clone, Copy)]
        pub enum AcquisitionFunction {
            ExpectedImprovement,
            UCB,
            PI,
            Thompson,
        }

        #[derive(Debug, Clone, Copy)]
        pub enum KernelType {
            RBF,
            Matern52,
            Matern32,
        }

        pub struct BayesianOptimizer {
            dim: usize,
            kernel: KernelType,
            acquisition: AcquisitionFunction,
            exploration: f64,
        }

        impl BayesianOptimizer {
            pub fn new(
                dim: usize,
                kernel: KernelType,
                acquisition: AcquisitionFunction,
                exploration: f64,
            ) -> Result<Self, Box<dyn Error>> {
                Ok(Self {
                    dim,
                    kernel,
                    acquisition,
                    exploration,
                })
            }

            pub fn update(
                &mut self,
                _x_data: &[Array1<f64>],
                _y_data: &Array1<f64>,
            ) -> Result<(), Box<dyn Error>> {
                Ok(())
            }

            pub fn suggest_next(&self) -> Result<Array1<f64>, Box<dyn Error>> {
                Ok(Array1::zeros(self.dim))
            }
        }

        pub struct GaussianProcess;
    }
}

mod scirs2_ml_stub {
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::prelude::*;
    use std::error::Error;

    pub struct RandomForest {
        n_estimators: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        trees: Vec<super::RegressionTreeNode>,
        n_features: usize,
    }

    impl Default for RandomForest {
        fn default() -> Self {
            Self::new()
        }
    }

    impl RandomForest {
        pub const fn new() -> Self {
            Self {
                n_estimators: 100,
                max_depth: None,
                min_samples_split: 2,
                trees: Vec::new(),
                n_features: 0,
            }
        }

        pub const fn n_estimators(mut self, n: usize) -> Self {
            self.n_estimators = n;
            self
        }

        pub const fn max_depth(mut self, depth: Option<usize>) -> Self {
            self.max_depth = depth;
            self
        }

        pub const fn min_samples_split(mut self, samples: usize) -> Self {
            self.min_samples_split = samples;
            self
        }

        /// Fit a real bagged ensemble of CART regression trees: each tree
        /// is grown on a bootstrap resample of the rows with per-split
        /// feature bagging (the "random" in "random forest"). This used to
        /// be a no-op that silently discarded `x`/`y`.
        pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) -> Result<(), Box<dyn Error>> {
            if x.is_empty() || y.is_empty() || x.len() != y.len() {
                return Err("RandomForest::fit requires non-empty, equal-length x and y".into());
            }

            let n_samples = x.len();
            let n_features = x[0].len();
            self.n_features = n_features;

            let n_candidate_features = ((n_features as f64).sqrt().ceil() as usize).max(1);
            let mut rng = thread_rng();
            let mut trees = Vec::with_capacity(self.n_estimators);

            for _ in 0..self.n_estimators {
                // Bootstrap resample (sampling with replacement).
                let indices: Vec<usize> = (0..n_samples)
                    .map(|_| rng.random_range(0..n_samples))
                    .collect();

                trees.push(super::grow_regression_tree(
                    x,
                    y,
                    &indices,
                    n_features,
                    n_candidate_features,
                    self.max_depth,
                    self.min_samples_split,
                    0,
                    &mut rng,
                ));
            }

            self.trees = trees;
            Ok(())
        }

        /// Real ensemble prediction (mean of each tree's prediction). This
        /// used to unconditionally return `0.0` for every row.
        pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
            if self.trees.is_empty() {
                return vec![0.0; x.len()];
            }
            x.iter()
                .map(|row| {
                    let sum: f64 = self.trees.iter().map(|tree| tree.predict(row)).sum();
                    sum / self.trees.len() as f64
                })
                .collect()
        }

        /// Real feature importances: total variance reduction attributable
        /// to each feature across every tree/split, normalized to sum to
        /// `1.0`. This used to unconditionally return `vec![0.5; 10]`
        /// regardless of the fitted data.
        pub fn feature_importances(&self) -> Vec<f64> {
            if self.n_features == 0 {
                return Vec::new();
            }

            let mut importance = vec![0.0; self.n_features];
            for tree in &self.trees {
                tree.accumulate_importance(&mut importance);
            }

            let total: f64 = importance.iter().sum();
            if total > 0.0 {
                for value in &mut importance {
                    *value /= total;
                }
            } else {
                let uniform = 1.0 / self.n_features as f64;
                importance.iter_mut().for_each(|value| *value = uniform);
            }

            importance
        }
    }

    pub struct GradientBoosting {
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
    }

    pub struct NeuralNetwork {
        hidden_layers: Vec<usize>,
        activation: String,
        learning_rate: f64,
    }

    pub struct KMeans {
        k: usize,
    }

    impl KMeans {
        pub const fn new(k: usize) -> Self {
            Self { k }
        }

        /// Real Lloyd's-algorithm k-means clustering. This used to
        /// unconditionally return cluster `0` for every point.
        pub fn fit_predict(&self, data: &Array2<f64>) -> Result<Vec<usize>, Box<dyn Error>> {
            Ok(super::lloyd_kmeans(data, self.k, 100))
        }
    }

    pub struct DBSCAN {
        eps: f64,
        min_samples: usize,
    }

    impl DBSCAN {
        pub const fn new(eps: f64, min_samples: usize) -> Self {
            Self { eps, min_samples }
        }

        /// Real density-based clustering. This used to unconditionally
        /// return cluster `0` for every point.
        pub fn fit_predict(&self, data: &Array2<f64>) -> Result<Vec<usize>, Box<dyn Error>> {
            Ok(super::dbscan_cluster(data, self.eps, self.min_samples))
        }
    }

    pub struct PCA {
        n_components: usize,
    }

    impl PCA {
        pub const fn new(n_components: usize) -> Self {
            Self { n_components }
        }

        /// Real PCA (via covariance-matrix eigendecomposition). This used
        /// to just truncate the raw input to its first `n_components`
        /// columns, which is not a projection at all.
        pub fn fit_transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
            Ok(super::real_pca_transform(data, self.n_components))
        }
    }

    pub struct StandardScaler;

    pub struct CrossValidation {
        n_folds: usize,
    }

    impl CrossValidation {
        pub const fn new(n_folds: usize) -> Self {
            Self { n_folds }
        }

        pub fn cross_val_score<T>(
            &self,
            _model: &T,
            _x: &Vec<Vec<f64>>,
            _y: &Vec<f64>,
        ) -> CVScores {
            CVScores {
                scores: vec![0.5; self.n_folds],
            }
        }
    }

    pub struct CVScores {
        scores: Vec<f64>,
    }

    impl CVScores {
        pub fn mean(&self) -> f64 {
            self.scores.iter().sum::<f64>() / self.scores.len() as f64
        }
    }

    pub const fn train_test_split<T>(
        _x: &[T],
        _y: &[f64],
        _test_size: f64,
    ) -> (Vec<T>, Vec<T>, Vec<f64>, Vec<f64>)
    where
        T: Clone,
    {
        (vec![], vec![], vec![], vec![])
    }
}

mod scirs2_graphs_stub {
    pub struct Graph;
    pub struct GraphLayout;

    pub fn spring_layout(
        _edges: &[(usize, usize)],
        n_nodes: usize,
    ) -> Result<Vec<(f64, f64)>, Box<dyn std::error::Error>> {
        // Simple circular layout
        let mut positions = Vec::new();
        for i in 0..n_nodes {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n_nodes as f64;
            positions.push((angle.cos(), angle.sin()));
        }
        Ok(positions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::scirs2_core::ndarray::Array2;

    #[test]
    fn test_lloyd_kmeans_separates_distinct_clusters() {
        // Two well-separated blobs around (0,0) and (10,10).
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.1, -0.1, -0.1, 0.1, 10.0, 10.0, 10.1, 9.9, 9.9, 10.1,
            ],
        )
        .unwrap();

        let labels = lloyd_kmeans(&data, 2, 50);
        assert_eq!(labels.len(), 6);
        // The old fabricated implementation assigned every point to cluster 0.
        let first_three = &labels[0..3];
        let last_three = &labels[3..6];
        assert!(first_three.iter().all(|&l| l == first_three[0]));
        assert!(last_three.iter().all(|&l| l == last_three[0]));
        assert_ne!(first_three[0], last_three[0]);
    }

    #[test]
    fn test_dbscan_cluster_finds_dense_regions_and_noise() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 0.2, 0.0, 0.0, 0.2, 20.0, 20.0, 20.1, 20.1],
        )
        .unwrap();

        let labels = dbscan_cluster(&data, 0.5, 2);
        // Points 0,1,2 form a dense cluster; 3,4 form another dense cluster
        // (each has >= min_samples-1 neighbors within eps of one another).
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_agglomerative_cluster_recovers_two_groups() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.1, 9.0, 9.0, 9.1, 9.1]).unwrap();

        let labels = agglomerative_cluster(&data, 2, "average");
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_real_pca_transform_is_not_fabricated_truncation() {
        // x2 = 3*x1 exactly: a 1D subspace. The old fabricated
        // implementation just returned data's first n_components raw
        // columns, unrelated to any real principal axis.
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 3.0, 2.0, 6.0, 3.0, 9.0]).unwrap();

        let transformed = real_pca_transform(&data, 1);
        assert_eq!(transformed.shape(), [4, 1]);

        // Scores must be proportional to the centered first-column values
        // (since the data is rank-1), not equal to the raw first column.
        let raw_first_column: Vec<f64> = data.column(0).to_vec();
        let scores: Vec<f64> = transformed.column(0).to_vec();
        assert_ne!(scores, raw_first_column);

        // But the *ratios* between consecutive centered points should match
        // (up to sign) since everything lies on one line through the mean.
        let ratio = scores[1] / (raw_first_column[1] - 1.5);
        for i in 0..4 {
            let centered = raw_first_column[i] - 1.5;
            if centered.abs() > 1e-9 {
                assert!((scores[i] / centered - ratio).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_random_forest_fits_real_data_dependent_model() {
        use scirs2_ml::RandomForest;

        // y = 2 * x0 + noise-free, x1 is irrelevant (constant).
        let x: Vec<Vec<f64>> = (0..30).map(|i| vec![f64::from(i % 10), 1.0]).collect();
        let y: Vec<f64> = x.iter().map(|row| 2.0 * row[0]).collect();

        let mut forest = RandomForest::new().n_estimators(10).max_depth(Some(4));
        forest.fit(&x, &y).expect("fit should succeed");

        let predictions = forest.predict(&x);
        // The old fabricated predict() always returned 0.0.
        let any_nonzero = predictions.iter().any(|&p| p.abs() > 1e-9);
        assert!(any_nonzero, "predictions must depend on the fitted data");

        // Predictions should roughly track the true (noise-free) targets.
        let mean_abs_error: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(&p, &t)| (p - t).abs())
            .sum::<f64>()
            / predictions.len() as f64;
        assert!(
            mean_abs_error < 2.0,
            "expected predictions to track y=2*x0, got mean abs error {mean_abs_error}"
        );

        let importances = forest.feature_importances();
        assert_eq!(importances.len(), 2);
        // Only x0 carries information about y, so it must dominate the
        // hardcoded-uniform `vec![0.5; 10]` the old implementation returned.
        assert!(importances[0] > importances[1]);
        let total: f64 = importances.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_lbfgs_minimize_actually_optimizes() {
        use scirs2_optimization::gradient::LBFGS;
        use scirs2_optimization::{Bounds, ObjectiveFunction, Optimizer};

        // f(x) = (x0 - 3)^2 + 5*(x1 + 2)^2, minimized at (3, -2). Asymmetric
        // coefficients and an off-center starting point are used to avoid a
        // known degenerate edge case in the underlying `scirs2-optimize`
        // solver where a perfectly symmetric quadratic started exactly at
        // the origin can produce a coincidentally-equal function value one
        // step away and falsely report convergence.
        struct Quadratic;
        impl ObjectiveFunction for Quadratic {
            fn evaluate(&self, x: &Array1<f64>) -> f64 {
                (x[0] - 3.0).powi(2) + 5.0 * (x[1] + 2.0).powi(2)
            }
            fn gradient(&self, x: &Array1<f64>) -> Array1<f64> {
                Array1::from_vec(vec![2.0 * (x[0] - 3.0), 10.0 * (x[1] + 2.0)])
            }
        }

        let mut optimizer = LBFGS::new(2);
        let x0 = Array1::from_vec(vec![0.5, 0.3]);
        let bounds = Bounds::new(
            Array1::from_vec(vec![-10.0, -10.0]),
            Array1::from_vec(vec![10.0, 10.0]),
        );

        let result = optimizer
            .minimize(&Quadratic, &x0, &bounds, 200)
            .expect("minimize should succeed");

        // The old fabricated implementation returned x0 unchanged
        // (f(0.5, 0.3) = 6.25 + 5*5.29 = 32.7).
        assert!(
            result.f < 1.0,
            "expected LBFGS to substantially reduce the objective, got f={}",
            result.f
        );
        assert!((result.x[0] - 3.0).abs() < 0.5);
        assert!((result.x[1] - (-2.0)).abs() < 0.5);
    }
}
