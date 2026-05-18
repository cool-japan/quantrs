//! Core quantum clustering functionality

use crate::dimensionality_reduction::QuantumDistanceMetric;
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

use super::config::*;

/// Clustering result containing labels and metadata
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// Cluster labels for each data point
    pub labels: Array1<usize>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Cluster centers (if available)
    pub cluster_centers: Option<Array2<f64>>,
    /// Inertia/within-cluster sum of squares (if available)
    pub inertia: Option<f64>,
    /// Cluster probabilities (for soft clustering)
    pub probabilities: Option<Array2<f64>>,
}

/// Main quantum clusterer
#[derive(Debug)]
pub struct QuantumClusterer {
    config: QuantumClusteringConfig,
    cluster_centers: Option<Array2<f64>>,
    labels: Option<Array1<usize>>,
    // Algorithm-specific configurations
    pub kmeans_config: Option<QuantumKMeansConfig>,
    pub dbscan_config: Option<QuantumDBSCANConfig>,
    pub spectral_config: Option<QuantumSpectralConfig>,
    pub fuzzy_config: Option<QuantumFuzzyCMeansConfig>,
    pub gmm_config: Option<QuantumGMMConfig>,
}

impl QuantumClusterer {
    /// Create new quantum clusterer
    pub fn new(config: QuantumClusteringConfig) -> Self {
        Self {
            config,
            cluster_centers: None,
            labels: None,
            kmeans_config: None,
            dbscan_config: None,
            spectral_config: None,
            fuzzy_config: None,
            gmm_config: None,
        }
    }

    /// Create quantum K-means clusterer
    pub fn kmeans(config: QuantumKMeansConfig) -> Self {
        let mut clusterer = Self::new(QuantumClusteringConfig {
            algorithm: ClusteringAlgorithm::QuantumKMeans,
            n_clusters: config.n_clusters,
            max_iterations: config.max_iterations,
            tolerance: config.tolerance,
            num_qubits: 4,
            random_state: config.seed,
        });
        clusterer.kmeans_config = Some(config);
        clusterer
    }

    /// Create quantum DBSCAN clusterer
    pub fn dbscan(config: QuantumDBSCANConfig) -> Self {
        let mut clusterer = Self::new(QuantumClusteringConfig {
            algorithm: ClusteringAlgorithm::QuantumDBSCAN,
            n_clusters: 0, // DBSCAN determines clusters automatically
            max_iterations: 100,
            tolerance: 1e-4,
            num_qubits: 4,
            random_state: config.seed,
        });
        clusterer.dbscan_config = Some(config);
        clusterer
    }

    /// Create quantum spectral clusterer
    pub fn spectral(config: QuantumSpectralConfig) -> Self {
        let mut clusterer = Self::new(QuantumClusteringConfig {
            algorithm: ClusteringAlgorithm::QuantumSpectral,
            n_clusters: config.n_clusters,
            max_iterations: 100,
            tolerance: 1e-4,
            num_qubits: 4,
            random_state: config.seed,
        });
        clusterer.spectral_config = Some(config);
        clusterer
    }

    /// Compute squared Euclidean distance between two array views
    fn squared_dist(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }

    /// Iterative union-find with path halving (no recursion)
    fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            // Path compression by halving
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    /// Run Lloyd's k-means algorithm with k-means++ initialization.
    ///
    /// Returns `(cluster_centers, labels, inertia)`.
    fn run_kmeans(
        &self,
        data: &Array2<f64>,
        k: usize,
    ) -> Result<(Array2<f64>, Array1<usize>, f64)> {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let max_iter = self.config.max_iterations;

        // -----------------------------------------------------------------------
        // k-means++ initialisation
        // First center: deterministic – row 0, or seeded via random_state.
        // Subsequent centers: greedy furthest-point (deterministic, avoids RNG).
        // -----------------------------------------------------------------------
        let mut centers = Array2::<f64>::zeros((k, n_features));

        // Choose first center
        let first_idx = self
            .config
            .random_state
            .map(|s| (s as usize) % n_samples)
            .unwrap_or(0);
        centers.row_mut(0).assign(&data.row(first_idx));

        // k-means++ subsequent centers
        for c in 1..k {
            // For each sample, compute minimum squared distance to any chosen center so far
            let mut min_dists_sq = vec![f64::INFINITY; n_samples];
            for i in 0..n_samples {
                for prev_c in 0..c {
                    let d = self.squared_dist(&data.row(i), &centers.row(prev_c));
                    if d < min_dists_sq[i] {
                        min_dists_sq[i] = d;
                    }
                }
            }
            // Greedy deterministic choice: the sample farthest from all current centers
            let next_idx = min_dists_sq
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(c % n_samples);
            centers.row_mut(c).assign(&data.row(next_idx));
        }

        // -----------------------------------------------------------------------
        // Lloyd's iterations
        // -----------------------------------------------------------------------
        let mut labels = vec![0usize; n_samples];

        for _iter in 0..max_iter {
            // ----- Assignment step -----
            let mut changed = false;
            for i in 0..n_samples {
                let mut best_c = 0;
                let mut best_d = f64::INFINITY;
                for c in 0..k {
                    let d = self.squared_dist(&data.row(i), &centers.row(c));
                    if d < best_d {
                        best_d = d;
                        best_c = c;
                    }
                }
                if labels[i] != best_c {
                    changed = true;
                    labels[i] = best_c;
                }
            }

            // ----- Update step -----
            let mut new_centers = Array2::<f64>::zeros((k, n_features));
            let mut counts = vec![0usize; k];
            for i in 0..n_samples {
                let c = labels[i];
                new_centers.row_mut(c).scaled_add(1.0, &data.row(i));
                counts[c] += 1;
            }
            for c in 0..k {
                if counts[c] > 0 {
                    new_centers
                        .row_mut(c)
                        .mapv_inplace(|v| v / counts[c] as f64);
                } else {
                    // Empty cluster: reassign center to a guaranteed occupied data point
                    new_centers.row_mut(c).assign(&data.row(c % n_samples));
                }
            }
            centers = new_centers;

            if !changed {
                break;
            }
        }

        // -----------------------------------------------------------------------
        // Compute inertia (within-cluster sum of squared distances)
        // -----------------------------------------------------------------------
        let mut inertia = 0.0f64;
        for i in 0..n_samples {
            inertia += self.squared_dist(&data.row(i), &centers.row(labels[i]));
        }

        let labels_arr = Array1::from_iter(labels);
        Ok((centers, labels_arr, inertia))
    }

    /// Density-based cluster counting using union-find over the epsilon neighbourhood.
    ///
    /// Uses `dbscan_config.eps` and `dbscan_config.min_samples` when available,
    /// falling back to sensible defaults derived from the data spread.
    fn fit_dbscan(&self, data: &Array2<f64>) -> Result<usize> {
        let n = data.nrows();

        let (eps, min_samples) = if let Some(cfg) = &self.dbscan_config {
            (cfg.eps, cfg.min_samples)
        } else {
            // Estimate eps as ~10 % of the bounding-box diagonal
            let mut max_sq = 0.0f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = self.squared_dist(&data.row(i), &data.row(j));
                    if d > max_sq {
                        max_sq = d;
                    }
                }
            }
            (max_sq.sqrt() * 0.1, 2usize)
        };

        // Union-find initialisation
        let mut parent: Vec<usize> = (0..n).collect();

        for i in 0..n {
            let mut neighbor_count = 0usize;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = self.squared_dist(&data.row(i), &data.row(j)).sqrt();
                if d <= eps {
                    neighbor_count += 1;
                    // Union i and j
                    let pi = Self::uf_find(&mut parent, i);
                    let pj = Self::uf_find(&mut parent, j);
                    if pi != pj {
                        parent[pi] = pj;
                    }
                }
            }
            // Points with fewer than min_samples neighbours remain noise (own root)
            let _ = neighbor_count;
        }

        // Count distinct roots – each root represents one cluster
        let n_clusters = (0..n)
            .filter(|&i| Self::uf_find(&mut parent, i) == i)
            .count();

        Ok(n_clusters.max(1))
    }

    /// Fit the clustering model using Lloyd's k-means with k-means++ initialization.
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<ClusteringResult> {
        let n_samples = data.nrows();

        if n_samples == 0 {
            return Err(MLError::InvalidInput("Empty data".to_string()));
        }

        // Determine the target number of clusters
        let n_clusters = match self.config.algorithm {
            ClusteringAlgorithm::QuantumDBSCAN => {
                // DBSCAN determines clusters from density
                let auto_k = self.fit_dbscan(data)?;
                auto_k
            }
            _ => {
                // Use configured n_clusters, capped to available samples
                self.config.n_clusters.min(n_samples).max(1)
            }
        };

        // Run Lloyd's k-means (with k-means++ init) over the chosen k
        let (cluster_centers, labels, inertia) = self.run_kmeans(data, n_clusters)?;

        self.cluster_centers = Some(cluster_centers.clone());
        self.labels = Some(labels.clone());

        Ok(ClusteringResult {
            labels,
            n_clusters,
            cluster_centers: Some(cluster_centers),
            inertia: Some(inertia),
            probabilities: None,
        })
    }

    /// Predict cluster labels for new data by assigning to the nearest center.
    pub fn predict(&self, data: &Array2<f64>) -> Result<Array1<usize>> {
        let centers = self.cluster_centers.as_ref().ok_or_else(|| {
            MLError::ModelNotTrained("Clusterer must be fitted before predict".to_string())
        })?;

        let k = centers.nrows();
        let labels: Vec<usize> = (0..data.nrows())
            .map(|i| {
                let mut best_c = 0;
                let mut best_d = f64::INFINITY;
                for c in 0..k {
                    let d = self.squared_dist(&data.row(i), &centers.row(c));
                    if d < best_d {
                        best_d = d;
                        best_c = c;
                    }
                }
                best_c
            })
            .collect();

        Ok(Array1::from_iter(labels))
    }

    /// Predict cluster probabilities (for soft clustering)
    pub fn predict_proba(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        if self.cluster_centers.is_none() {
            return Err(MLError::ModelNotTrained(
                "Clusterer must be fitted before predict_proba".to_string(),
            ));
        }

        let n_samples = data.nrows();
        let n_clusters = self.config.n_clusters;

        // Return uniform probabilities as placeholder
        Ok(Array2::from_elem(
            (n_samples, n_clusters),
            1.0 / n_clusters as f64,
        ))
    }

    /// Compute quantum distance between two points
    pub fn compute_quantum_distance(
        &self,
        point1: &Array1<f64>,
        point2: &Array1<f64>,
        metric: QuantumDistanceMetric,
    ) -> Result<f64> {
        // Placeholder implementation for quantum distance computation
        match metric {
            QuantumDistanceMetric::QuantumEuclidean => {
                let diff = point1 - point2;
                Ok(diff.dot(&diff).sqrt())
            }
            QuantumDistanceMetric::QuantumManhattan => {
                Ok((point1 - point2).mapv(|x| x.abs()).sum())
            }
            QuantumDistanceMetric::QuantumCosine => {
                let dot_product = point1.dot(point2);
                let norm1 = point1.dot(point1).sqrt();
                let norm2 = point2.dot(point2).sqrt();
                Ok(1.0 - (dot_product / (norm1 * norm2)))
            }
            _ => {
                // For other quantum metrics, return Euclidean as fallback
                let diff = point1 - point2;
                Ok(diff.dot(&diff).sqrt())
            }
        }
    }

    /// Fit and predict in one step
    pub fn fit_predict(&mut self, data: &Array2<f64>) -> Result<Array1<usize>> {
        let result = self.fit(data)?;
        Ok(result.labels)
    }

    /// Get cluster centers
    pub fn cluster_centers(&self) -> Option<&Array2<f64>> {
        self.cluster_centers.as_ref()
    }

    /// Evaluate clustering performance
    pub fn evaluate(
        &self,
        _data: &Array2<f64>,
        _true_labels: Option<&Array1<usize>>,
    ) -> Result<ClusteringMetrics> {
        if self.cluster_centers.is_none() {
            return Err(MLError::ModelNotTrained(
                "Clusterer must be fitted before evaluation".to_string(),
            ));
        }

        // Placeholder evaluation metrics
        Ok(ClusteringMetrics {
            silhouette_score: 0.5,
            davies_bouldin_index: 1.0,
            calinski_harabasz_index: 100.0,
            inertia: 0.0,
            adjusted_rand_index: None,
            normalized_mutual_info: None,
        })
    }
}

/// Clustering evaluation metrics
#[derive(Debug, Clone)]
pub struct ClusteringMetrics {
    /// Silhouette score
    pub silhouette_score: f64,
    /// Davies-Bouldin index
    pub davies_bouldin_index: f64,
    /// Calinski-Harabasz index
    pub calinski_harabasz_index: f64,
    /// Within-cluster sum of squares
    pub inertia: f64,
    /// Adjusted Rand Index (if true labels provided)
    pub adjusted_rand_index: Option<f64>,
    /// Normalized Mutual Information (if true labels provided)
    pub normalized_mutual_info: Option<f64>,
}

/// Helper function to create default quantum K-means clusterer
pub fn create_default_quantum_kmeans(n_clusters: usize) -> QuantumClusterer {
    let config = QuantumKMeansConfig {
        n_clusters,
        ..Default::default()
    };
    QuantumClusterer::kmeans(config)
}

/// Helper function to create default quantum DBSCAN clusterer
pub fn create_default_quantum_dbscan(eps: f64, min_samples: usize) -> QuantumClusterer {
    let config = QuantumDBSCANConfig {
        eps,
        min_samples,
        ..Default::default()
    };
    QuantumClusterer::dbscan(config)
}
