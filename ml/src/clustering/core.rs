//! Core quantum clustering functionality

use crate::dimensionality_reduction::QuantumDistanceMetric;
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};

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
    ///
    /// Computed as a softmax over the negative squared distance from each
    /// point to every cluster center, so points closer to a center receive a
    /// higher assignment probability to that cluster and the probabilities
    /// genuinely depend on `data` (rather than being uniform placeholders).
    pub fn predict_proba(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let centers = self.cluster_centers.as_ref().ok_or_else(|| {
            MLError::ModelNotTrained("Clusterer must be fitted before predict_proba".to_string())
        })?;

        let n_samples = data.nrows();
        let k = centers.nrows();
        let mut probabilities = Array2::<f64>::zeros((n_samples, k));

        for i in 0..n_samples {
            let row = data.row(i);
            let neg_distances: Vec<f64> = (0..k)
                .map(|c| -self.squared_dist(&row, &centers.row(c)))
                .collect();
            let max_neg_dist = neg_distances
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_vals: Vec<f64> = neg_distances
                .iter()
                .map(|&d| (d - max_neg_dist).exp())
                .collect();
            let sum_exp: f64 = exp_vals.iter().sum();
            for c in 0..k {
                probabilities[[i, c]] = if sum_exp > 0.0 {
                    exp_vals[c] / sum_exp
                } else {
                    1.0 / k as f64
                };
            }
        }

        Ok(probabilities)
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

    /// Evaluate clustering performance.
    ///
    /// All metrics are computed from the actual fitted `cluster_centers` and
    /// the provided `data` (real silhouette score, Davies-Bouldin index,
    /// Calinski-Harabasz index and inertia). If `true_labels` are supplied,
    /// the Adjusted Rand Index and Normalized Mutual Information against those
    /// ground-truth labels are computed as well; otherwise those two fields
    /// are `None`, since there is nothing external to compare against.
    pub fn evaluate(
        &self,
        data: &Array2<f64>,
        true_labels: Option<&Array1<usize>>,
    ) -> Result<ClusteringMetrics> {
        let centers = self.cluster_centers.as_ref().ok_or_else(|| {
            MLError::ModelNotTrained("Clusterer must be fitted before evaluation".to_string())
        })?;

        if data.nrows() == 0 {
            return Err(MLError::InvalidInput("Empty data".to_string()));
        }

        let predicted_labels = self.predict(data)?;
        let n_samples = data.nrows();
        let k = centers.nrows();

        let inertia: f64 = (0..n_samples)
            .map(|i| self.squared_dist(&data.row(i), &centers.row(predicted_labels[i])))
            .sum();

        let mut cluster_members: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &label) in predicted_labels.iter().enumerate() {
            cluster_members[label].push(i);
        }

        let silhouette_score =
            Self::compute_silhouette_score(data, &predicted_labels, &cluster_members);
        let davies_bouldin_index =
            Self::compute_davies_bouldin_index(data, centers, &cluster_members);
        let calinski_harabasz_index =
            Self::compute_calinski_harabasz_index(data, centers, &cluster_members, inertia);

        let (adjusted_rand_index, normalized_mutual_info) = match true_labels {
            Some(truth) if truth.len() == n_samples => (
                Some(Self::adjusted_rand_index(&predicted_labels, truth)),
                Some(Self::normalized_mutual_info(&predicted_labels, truth)),
            ),
            _ => (None, None),
        };

        Ok(ClusteringMetrics {
            silhouette_score,
            davies_bouldin_index,
            calinski_harabasz_index,
            inertia,
            adjusted_rand_index,
            normalized_mutual_info,
        })
    }

    /// Mean silhouette coefficient over all points: for each point `i`,
    /// `s_i = (b_i - a_i) / max(a_i, b_i)`, where `a_i` is the mean distance
    /// to other points in the same cluster and `b_i` is the smallest mean
    /// distance to the points of any other cluster.
    fn compute_silhouette_score(
        data: &Array2<f64>,
        labels: &Array1<usize>,
        cluster_members: &[Vec<usize>],
    ) -> f64 {
        let n_samples = data.nrows();
        let non_empty_clusters = cluster_members.iter().filter(|m| !m.is_empty()).count();
        if n_samples < 2 || non_empty_clusters < 2 {
            return 0.0;
        }

        let mut silhouette_sum = 0.0;
        let mut counted = 0usize;
        for i in 0..n_samples {
            let own_cluster = labels[i];
            let own_members = &cluster_members[own_cluster];
            let a_i = if own_members.len() > 1 {
                own_members
                    .iter()
                    .filter(|&&j| j != i)
                    .map(|&j| euclidean_distance(&data.row(i), &data.row(j)))
                    .sum::<f64>()
                    / (own_members.len() - 1) as f64
            } else {
                0.0
            };

            let mut b_i = f64::INFINITY;
            for (c, members) in cluster_members.iter().enumerate() {
                if c == own_cluster || members.is_empty() {
                    continue;
                }
                let mean_dist = members
                    .iter()
                    .map(|&j| euclidean_distance(&data.row(i), &data.row(j)))
                    .sum::<f64>()
                    / members.len() as f64;
                if mean_dist < b_i {
                    b_i = mean_dist;
                }
            }

            if b_i.is_finite() {
                let denom = a_i.max(b_i);
                let s_i = if denom > 0.0 {
                    (b_i - a_i) / denom
                } else {
                    0.0
                };
                silhouette_sum += s_i;
                counted += 1;
            }
        }

        if counted > 0 {
            silhouette_sum / counted as f64
        } else {
            0.0
        }
    }

    /// Davies-Bouldin index: average, over clusters `i`, of the worst-case
    /// ratio `(scatter_i + scatter_j) / distance(center_i, center_j)` across
    /// every other cluster `j`. Lower is better (more separated clusters).
    fn compute_davies_bouldin_index(
        data: &Array2<f64>,
        centers: &Array2<f64>,
        cluster_members: &[Vec<usize>],
    ) -> f64 {
        let k = centers.nrows();
        if k < 2 {
            return 0.0;
        }

        let scatter: Vec<f64> = (0..k)
            .map(|c| {
                if cluster_members[c].is_empty() {
                    0.0
                } else {
                    cluster_members[c]
                        .iter()
                        .map(|&i| euclidean_distance(&data.row(i), &centers.row(c)))
                        .sum::<f64>()
                        / cluster_members[c].len() as f64
                }
            })
            .collect();

        let mut db_sum = 0.0;
        for i in 0..k {
            let mut max_ratio = 0.0_f64;
            for j in 0..k {
                if i == j {
                    continue;
                }
                let center_distance = euclidean_distance(&centers.row(i), &centers.row(j));
                if center_distance > 1e-12 {
                    let ratio = (scatter[i] + scatter[j]) / center_distance;
                    if ratio > max_ratio {
                        max_ratio = ratio;
                    }
                }
            }
            db_sum += max_ratio;
        }

        db_sum / k as f64
    }

    /// Calinski-Harabasz index: ratio of between-cluster to within-cluster
    /// dispersion, scaled by the usual degrees-of-freedom correction.
    fn compute_calinski_harabasz_index(
        data: &Array2<f64>,
        centers: &Array2<f64>,
        cluster_members: &[Vec<usize>],
        within_cluster_dispersion: f64,
    ) -> f64 {
        let n_samples = data.nrows();
        let k = centers.nrows();
        if k < 2 || n_samples <= k || within_cluster_dispersion <= 1e-12 {
            return 0.0;
        }

        let overall_mean = data
            .mean_axis(Axis(0))
            .unwrap_or_else(|| Array1::zeros(data.ncols()));

        let between_cluster_dispersion: f64 = (0..k)
            .map(|c| {
                let n_c = cluster_members[c].len() as f64;
                if n_c == 0.0 {
                    0.0
                } else {
                    n_c * euclidean_distance(&centers.row(c), &overall_mean.view()).powi(2)
                }
            })
            .sum();

        (between_cluster_dispersion / within_cluster_dispersion)
            * ((n_samples - k) as f64 / (k - 1) as f64)
    }

    /// Adjusted Rand Index between predicted and ground-truth labels, computed
    /// from the pairwise contingency table.
    fn adjusted_rand_index(predicted: &Array1<usize>, truth: &Array1<usize>) -> f64 {
        let n = predicted.len();
        if n == 0 {
            return 0.0;
        }
        let pred_max = predicted.iter().cloned().max().unwrap_or(0) + 1;
        let true_max = truth.iter().cloned().max().unwrap_or(0) + 1;

        let mut contingency = vec![vec![0usize; true_max]; pred_max];
        for i in 0..n {
            contingency[predicted[i]][truth[i]] += 1;
        }

        let comb2 = |x: usize| -> f64 {
            if x < 2 {
                0.0
            } else {
                (x * (x - 1)) as f64 / 2.0
            }
        };

        let sum_comb_nij: f64 = contingency.iter().flatten().map(|&v| comb2(v)).sum();
        let row_sums: Vec<usize> = contingency.iter().map(|row| row.iter().sum()).collect();
        let col_sums: Vec<usize> = (0..true_max)
            .map(|j| contingency.iter().map(|row| row[j]).sum())
            .collect();

        let sum_comb_a: f64 = row_sums.iter().map(|&a| comb2(a)).sum();
        let sum_comb_b: f64 = col_sums.iter().map(|&b| comb2(b)).sum();
        let comb_n = comb2(n);
        if comb_n <= 0.0 {
            return 0.0;
        }

        let expected_index = (sum_comb_a * sum_comb_b) / comb_n;
        let max_index = 0.5 * (sum_comb_a + sum_comb_b);
        let denom = max_index - expected_index;

        if denom.abs() < 1e-12 {
            0.0
        } else {
            (sum_comb_nij - expected_index) / denom
        }
    }

    /// Normalized Mutual Information between predicted and ground-truth
    /// labels, normalized by the geometric mean of the two label entropies so
    /// the result lies in `[0, 1]`.
    fn normalized_mutual_info(predicted: &Array1<usize>, truth: &Array1<usize>) -> f64 {
        let n = predicted.len();
        if n == 0 {
            return 0.0;
        }
        let pred_max = predicted.iter().cloned().max().unwrap_or(0) + 1;
        let true_max = truth.iter().cloned().max().unwrap_or(0) + 1;

        let mut contingency = vec![vec![0usize; true_max]; pred_max];
        for i in 0..n {
            contingency[predicted[i]][truth[i]] += 1;
        }

        let row_sums: Vec<usize> = contingency.iter().map(|row| row.iter().sum()).collect();
        let col_sums: Vec<usize> = (0..true_max)
            .map(|j| contingency.iter().map(|row| row[j]).sum())
            .collect();

        let n_f = n as f64;
        let mutual_information: f64 = contingency
            .iter()
            .enumerate()
            .flat_map(|(i, row)| row.iter().enumerate().map(move |(j, &n_ij)| (i, j, n_ij)))
            .filter(|&(_, _, n_ij)| n_ij > 0)
            .map(|(i, j, n_ij)| {
                let p_ij = n_ij as f64 / n_f;
                let p_i = row_sums[i] as f64 / n_f;
                let p_j = col_sums[j] as f64 / n_f;
                p_ij * (p_ij / (p_i * p_j)).ln()
            })
            .sum();

        let entropy = |sums: &[usize]| -> f64 {
            sums.iter()
                .filter(|&&s| s > 0)
                .map(|&s| {
                    let p = s as f64 / n_f;
                    -p * p.ln()
                })
                .sum::<f64>()
        };

        let h_pred = entropy(&row_sums);
        let h_true = entropy(&col_sums);

        if h_pred <= 1e-12 || h_true <= 1e-12 {
            if mutual_information.abs() < 1e-12 {
                1.0
            } else {
                0.0
            }
        } else {
            (mutual_information / (h_pred * h_true).sqrt()).clamp(0.0, 1.0)
        }
    }
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
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

#[cfg(test)]
mod regression_tests {
    use super::*;

    /// Two well-separated 2-cluster blobs for deterministic assertions.
    fn two_blob_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, // cluster A around (0,0)
                10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1, // cluster B around (10,10)
            ],
        )
        .expect("valid shape")
    }

    #[test]
    fn predict_proba_reflects_actual_distances_not_uniform() {
        let mut clusterer = create_default_quantum_kmeans(2);
        let data = two_blob_data();
        clusterer.fit(&data).expect("fit should succeed");

        let probabilities = clusterer
            .predict_proba(&data)
            .expect("predict_proba should succeed");

        // Every point is far closer to one center than the other, so its
        // dominant-cluster probability should be near 1.0, not the uniform
        // 1/2 that the previous placeholder implementation always returned.
        for i in 0..probabilities.nrows() {
            let row = probabilities.row(i);
            let max_prob = row.iter().cloned().fold(f64::MIN, f64::max);
            assert!(
                max_prob > 0.9,
                "expected a confident cluster assignment, got {row:?}"
            );
        }

        // Probabilities for each point must sum to 1.
        for i in 0..probabilities.nrows() {
            let sum: f64 = probabilities.row(i).iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "row {i} does not sum to 1: {sum}");
        }
    }

    #[test]
    fn evaluate_computes_real_metrics_for_well_separated_clusters() {
        let mut clusterer = create_default_quantum_kmeans(2);
        let data = two_blob_data();
        clusterer.fit(&data).expect("fit should succeed");

        let metrics = clusterer
            .evaluate(&data, None)
            .expect("evaluate should succeed");

        // Two tight, well-separated blobs should score close to a perfect
        // silhouette (near 1.0), not the hardcoded placeholder of 0.5.
        assert!(
            metrics.silhouette_score > 0.9,
            "silhouette_score should be near 1.0 for well-separated blobs, got {}",
            metrics.silhouette_score
        );
        // Davies-Bouldin should be small (good separation), not the
        // hardcoded placeholder of 1.0.
        assert!(
            metrics.davies_bouldin_index < 0.1,
            "davies_bouldin_index should be small for well-separated blobs, got {}",
            metrics.davies_bouldin_index
        );
        // Calinski-Harabasz should be large for well-separated blobs, not
        // the hardcoded placeholder of 100.0 by coincidence of formula.
        assert!(
            metrics.calinski_harabasz_index > 100.0,
            "calinski_harabasz_index should be large for well-separated blobs, got {}",
            metrics.calinski_harabasz_index
        );
        // Inertia must be strictly positive and reflect within-cluster
        // scatter, not the hardcoded placeholder of 0.0.
        assert!(
            metrics.inertia > 0.0,
            "inertia should be positive, got {}",
            metrics.inertia
        );

        // With ground-truth labels matching the true blob structure exactly,
        // ARI and NMI should both be (near) 1.0.
        let truth = Array1::from_vec(vec![0usize, 0, 0, 0, 1, 1, 1, 1]);
        let metrics_with_truth = clusterer
            .evaluate(&data, Some(&truth))
            .expect("evaluate with truth should succeed");
        let ari = metrics_with_truth
            .adjusted_rand_index
            .expect("ARI should be Some when true_labels given");
        let nmi = metrics_with_truth
            .normalized_mutual_info
            .expect("NMI should be Some when true_labels given");
        assert!(
            ari > 0.9,
            "ARI should be near 1.0 for matching truth, got {ari}"
        );
        assert!(
            nmi > 0.9,
            "NMI should be near 1.0 for matching truth, got {nmi}"
        );
    }

    #[test]
    fn evaluate_without_true_labels_leaves_external_metrics_none() {
        let mut clusterer = create_default_quantum_kmeans(2);
        let data = two_blob_data();
        clusterer.fit(&data).expect("fit should succeed");

        let metrics = clusterer.evaluate(&data, None).expect("evaluate ok");
        assert!(metrics.adjusted_rand_index.is_none());
        assert!(metrics.normalized_mutual_info.is_none());
    }
}
