//! Sklearn-compatible clustering algorithms

use super::{SklearnClusterer, SklearnEstimator};
use crate::clustering::core::QuantumClusterer;
use crate::error::{MLError, Result};
use crate::simulator_backends::{SimulatorBackend, StatevectorBackend};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::Arc;

/// Quantum K-Means (sklearn-compatible)
pub struct QuantumKMeans {
    /// Internal clusterer
    clusterer: Option<QuantumClusterer>,
    /// Number of clusters
    n_clusters: usize,
    /// Maximum iterations
    max_iter: usize,
    /// Tolerance
    tol: f64,
    /// Random state
    random_state: Option<u64>,
    /// Backend
    backend: Arc<dyn SimulatorBackend>,
    /// Fitted flag
    fitted: bool,
    /// Cluster centers
    cluster_centers_: Option<Array2<f64>>,
    /// Labels
    labels_: Option<Array1<i32>>,
}

impl QuantumKMeans {
    /// Create new Quantum K-Means
    pub fn new(n_clusters: usize) -> Self {
        Self {
            clusterer: None,
            n_clusters,
            max_iter: 300,
            tol: 1e-4,
            random_state: None,
            backend: Arc::new(StatevectorBackend::new(10)),
            fitted: false,
            cluster_centers_: None,
            labels_: None,
        }
    }

    /// Set maximum iterations
    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn set_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random state
    pub fn set_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

impl SklearnEstimator for QuantumKMeans {
    #[allow(non_snake_case)]
    fn fit(&mut self, X: &Array2<f64>, _y: Option<&Array1<f64>>) -> Result<()> {
        let config = crate::clustering::config::QuantumClusteringConfig {
            algorithm: crate::clustering::config::ClusteringAlgorithm::QuantumKMeans,
            n_clusters: self.n_clusters,
            max_iterations: self.max_iter,
            tolerance: self.tol,
            num_qubits: 4,
            random_state: self.random_state,
        };
        let mut clusterer = QuantumClusterer::new(config);

        let result = clusterer.fit_predict(X)?;
        // Convert usize to i32 for sklearn compatibility
        let result_i32 = result.mapv(|x| x as i32);
        self.labels_ = Some(result_i32);
        self.cluster_centers_ = None; // TODO: Get cluster centers from clusterer

        self.clusterer = Some(clusterer);
        self.fitted = true;

        Ok(())
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("n_clusters".to_string(), self.n_clusters.to_string());
        params.insert("max_iter".to_string(), self.max_iter.to_string());
        params.insert("tol".to_string(), self.tol.to_string());
        if let Some(rs) = self.random_state {
            params.insert("random_state".to_string(), rs.to_string());
        }
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "n_clusters" => {
                    self.n_clusters = value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid n_clusters: {}", value))
                    })?;
                }
                "max_iter" => {
                    self.max_iter = value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid max_iter: {}", value))
                    })?;
                }
                "tol" => {
                    self.tol = value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid tol: {}", value))
                    })?;
                }
                "random_state" => {
                    self.random_state = Some(value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid random_state: {}", value))
                    })?);
                }
                _ => {
                    // Skip unknown parameters
                }
            }
        }
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl SklearnClusterer for QuantumKMeans {
    #[allow(non_snake_case)]
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained("Model not trained".to_string()));
        }

        let clusterer = self
            .clusterer
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Clusterer not initialized".to_string()))?;
        let result = clusterer.predict(X)?;
        // Convert usize to i32 for sklearn compatibility
        Ok(result.mapv(|x| x as i32))
    }

    fn cluster_centers(&self) -> Option<&Array2<f64>> {
        self.cluster_centers_.as_ref()
    }
}

/// DBSCAN clustering algorithm
pub struct DBSCAN {
    /// Epsilon - neighborhood radius
    eps: f64,
    /// Minimum samples for core points
    min_samples: usize,
    /// Fitted labels
    labels: Option<Array1<i32>>,
    /// Core sample indices
    core_sample_indices: Vec<usize>,
}

impl DBSCAN {
    /// Create new DBSCAN
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            labels: None,
            core_sample_indices: Vec::new(),
        }
    }

    /// Set eps
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set min_samples
    pub fn min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Get labels
    pub fn labels(&self) -> Option<&Array1<i32>> {
        self.labels.as_ref()
    }

    /// Get core sample indices
    pub fn core_sample_indices(&self) -> &[usize] {
        &self.core_sample_indices
    }

    /// Compute distance matrix
    #[allow(non_snake_case)]
    fn compute_distances(&self, X: &Array2<f64>) -> Array2<f64> {
        let n = X.nrows();
        let mut distances = Array2::zeros((n, n));

        for i in 0..n {
            for j in i + 1..n {
                let mut dist = 0.0;
                for k in 0..X.ncols() {
                    let diff = X[[i, k]] - X[[j, k]];
                    dist += diff * diff;
                }
                let dist = dist.sqrt();
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        distances
    }

    /// Get number of clusters found
    pub fn n_clusters(&self) -> Option<usize> {
        self.labels.as_ref().map(|labels| {
            let max_label = labels.iter().max().copied().unwrap_or(-1);
            if max_label >= 0 {
                (max_label + 1) as usize
            } else {
                0
            }
        })
    }

    /// Fit the model (internal)
    #[allow(non_snake_case)]
    fn fit_internal(&mut self, X: &Array2<f64>) -> Result<()> {
        let n = X.nrows();
        let distances = self.compute_distances(X);

        // Find neighbors for each point
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in 0..n {
                if i != j && distances[[i, j]] <= self.eps {
                    neighbors[i].push(j);
                }
            }
        }

        // Identify core points
        self.core_sample_indices.clear();
        for (i, n_neighbors) in neighbors.iter().enumerate() {
            if n_neighbors.len() >= self.min_samples {
                self.core_sample_indices.push(i);
            }
        }

        // Label points
        let mut labels = Array1::from_elem(n, -1_i32); // -1 = noise
        let mut visited = vec![false; n];
        let mut cluster_id = 0_i32;

        for &core_idx in &self.core_sample_indices {
            if visited[core_idx] {
                continue;
            }

            // BFS to expand cluster
            let mut stack = vec![core_idx];
            while let Some(idx) = stack.pop() {
                if visited[idx] {
                    continue;
                }
                visited[idx] = true;
                labels[idx] = cluster_id;

                // If this is a core point, expand
                if neighbors[idx].len() >= self.min_samples {
                    for &neighbor in &neighbors[idx] {
                        if !visited[neighbor] {
                            stack.push(neighbor);
                        }
                    }
                }
            }
            cluster_id += 1;
        }

        self.labels = Some(labels);
        Ok(())
    }
}

impl SklearnEstimator for DBSCAN {
    #[allow(non_snake_case)]
    fn fit(&mut self, X: &Array2<f64>, _y: Option<&Array1<f64>>) -> Result<()> {
        self.fit_internal(X)
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("eps".to_string(), self.eps.to_string());
        params.insert("min_samples".to_string(), self.min_samples.to_string());
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "eps" => {
                    self.eps = value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid eps: {}", value))
                    })?;
                }
                "min_samples" => {
                    self.min_samples = value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid min_samples: {}", value))
                    })?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.labels.is_some()
    }
}

impl SklearnClusterer for DBSCAN {
    #[allow(non_snake_case)]
    fn predict(&self, _X: &Array2<f64>) -> Result<Array1<i32>> {
        // For DBSCAN, predict returns labels from fit
        // New points would need special handling
        self.labels
            .clone()
            .ok_or_else(|| MLError::ModelNotTrained("DBSCAN not fitted".to_string()))
    }
}

/// Agglomerative Clustering
pub struct AgglomerativeClustering {
    /// Number of clusters
    n_clusters: usize,
    /// Linkage type
    linkage: String,
    /// Fitted labels
    labels: Option<Array1<i32>>,
}

impl AgglomerativeClustering {
    /// Create new AgglomerativeClustering
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: "ward".to_string(),
            labels: None,
        }
    }

    /// Set linkage
    pub fn linkage(mut self, linkage: &str) -> Self {
        self.linkage = linkage.to_string();
        self
    }

    /// Get number of clusters
    pub fn get_n_clusters(&self) -> Option<usize> {
        if self.labels.is_some() {
            Some(self.n_clusters)
        } else {
            None
        }
    }

    /// Fit internal
    #[allow(non_snake_case)]
    fn fit_internal(&mut self, X: &Array2<f64>) -> Result<()> {
        let n = X.nrows();

        // Compute distance matrix
        let mut distances = Array2::from_elem((n, n), f64::INFINITY);
        for i in 0..n {
            for j in i + 1..n {
                let mut dist = 0.0;
                for k in 0..X.ncols() {
                    let diff = X[[i, k]] - X[[j, k]];
                    dist += diff * diff;
                }
                distances[[i, j]] = dist.sqrt();
                distances[[j, i]] = distances[[i, j]];
            }
            distances[[i, i]] = 0.0;
        }

        // Initialize clusters: each point is its own cluster
        let mut cluster_assignment: Vec<usize> = (0..n).collect();
        let mut active_clusters: Vec<bool> = vec![true; n];
        let mut cluster_sizes: Vec<usize> = vec![1; n];

        // Merge clusters until we have n_clusters
        let mut num_clusters = n;
        while num_clusters > self.n_clusters {
            // Find closest pair of clusters
            let mut min_dist = f64::INFINITY;
            let mut merge_i = 0;
            let mut merge_j = 0;

            for i in 0..n {
                if !active_clusters[i] {
                    continue;
                }
                for j in i + 1..n {
                    if !active_clusters[j] {
                        continue;
                    }
                    if distances[[i, j]] < min_dist {
                        min_dist = distances[[i, j]];
                        merge_i = i;
                        merge_j = j;
                    }
                }
            }

            // Merge j into i
            for k in 0..n {
                if cluster_assignment[k] == merge_j {
                    cluster_assignment[k] = merge_i;
                }
            }
            active_clusters[merge_j] = false;
            cluster_sizes[merge_i] += cluster_sizes[merge_j];

            // Update distances (using average linkage as default)
            for k in 0..n {
                if k != merge_i && active_clusters[k] {
                    let new_dist = match self.linkage.as_str() {
                        "single" => distances[[merge_i, k]].min(distances[[merge_j, k]]),
                        "complete" => distances[[merge_i, k]].max(distances[[merge_j, k]]),
                        "average" | _ => {
                            let s_i = cluster_sizes[merge_i] as f64;
                            let s_j = cluster_sizes[merge_j] as f64;
                            (distances[[merge_i, k]] * (s_i - cluster_sizes[merge_j] as f64)
                                + distances[[merge_j, k]] * s_j)
                                / s_i
                        }
                    };
                    distances[[merge_i, k]] = new_dist;
                    distances[[k, merge_i]] = new_dist;
                }
            }

            num_clusters -= 1;
        }

        // Remap cluster labels to 0..n_clusters-1
        let unique_clusters: Vec<usize> = cluster_assignment
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let label_map: std::collections::HashMap<usize, i32> = unique_clusters
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as i32))
            .collect();

        let labels = cluster_assignment
            .iter()
            .map(|&c| *label_map.get(&c).unwrap_or(&0))
            .collect();
        self.labels = Some(Array1::from_vec(labels));

        Ok(())
    }
}

impl SklearnEstimator for AgglomerativeClustering {
    #[allow(non_snake_case)]
    fn fit(&mut self, X: &Array2<f64>, _y: Option<&Array1<f64>>) -> Result<()> {
        self.fit_internal(X)
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("n_clusters".to_string(), self.n_clusters.to_string());
        params.insert("linkage".to_string(), self.linkage.clone());
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "n_clusters" => {
                    self.n_clusters = value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid n_clusters: {}", value))
                    })?;
                }
                "linkage" => {
                    self.linkage = value;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.labels.is_some()
    }
}

impl SklearnClusterer for AgglomerativeClustering {
    #[allow(non_snake_case)]
    fn predict(&self, _X: &Array2<f64>) -> Result<Array1<i32>> {
        self.labels
            .clone()
            .ok_or_else(|| MLError::ModelNotTrained("Not fitted".to_string()))
    }
}
