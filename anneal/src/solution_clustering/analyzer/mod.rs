//! Main solution clustering analyzer implementation

use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::algorithms::{ClusteringAlgorithm, DistanceMetric, LinkageType};
use super::config::{ClusteringConfig, FeatureExtractionMethod};
use super::error::{ClusteringError, ClusteringResult};
use super::types::{
    AnalysisStatistics, ClusterQualityMetrics, ClusterStatistics, ClusteringPerformanceMetrics,
    ClusteringResults, ConnectivityAnalysis, ConvergenceAnalysis, CorrelationAnalysis,
    CorrelationPattern, DifficultyLevel, DistributionAnalysis, DistributionType, EfficiencyMetrics,
    EnergyBasin, EnergyStatistics, FunnelAnalysis, LandscapeAnalysis, MultiModalityAnalysis,
    OptimizationRecommendation, OutlierInfo, OutlierType, OverallClusteringQuality, PatternType,
    PlateauAnalysis, PriorityLevel, RecommendationType, RuggednessMetrics, ScalabilityMetrics,
    SolutionCluster, SolutionMetadata, SolutionPoint, StatisticalSummary,
};
use crate::simulator::AnnealingSolution;

mod quality;

/// Solution clustering analyzer
pub struct SolutionClusteringAnalyzer {
    /// Configuration
    config: ClusteringConfig,
    /// Cached distance matrices
    distance_cache: HashMap<String, Vec<Vec<f64>>>,
    /// Analysis statistics
    stats: AnalysisStatistics,
}

impl SolutionClusteringAnalyzer {
    /// Create a new solution clustering analyzer
    #[must_use]
    pub fn new(config: ClusteringConfig) -> Self {
        Self {
            config,
            distance_cache: HashMap::new(),
            stats: AnalysisStatistics {
                total_solutions: 0,
                total_time: Duration::from_secs(0),
                cache_hit_rate: 0.0,
                peak_memory: 0,
            },
        }
    }

    /// Analyze a collection of solutions
    pub fn analyze_solutions(
        &mut self,
        solutions: &[AnnealingSolution],
    ) -> ClusteringResult<ClusteringResults> {
        let start_time = Instant::now();

        // Convert solutions to solution points
        let solution_points = self.convert_solutions(solutions)?;

        // Extract features if needed
        let featured_points = self.extract_features(solution_points)?;

        // Perform clustering
        let mut clusters = self.perform_clustering(&featured_points)?;

        // Post-pass: compute global quality metrics that require seeing all clusters
        // simultaneously (silhouette, Davies-Bouldin, Calinski-Harabasz). The per-cluster
        // computation in `calculate_cluster_quality_metrics` only fills in `inertia` —
        // global metrics are written back into each cluster's `quality_metrics` here.
        self.update_global_quality_metrics(&mut clusters)?;

        // Perform landscape analysis
        let landscape_analysis = self.analyze_landscape(&featured_points, &clusters)?;

        // Perform statistical analysis
        let statistical_summary = self.perform_statistical_analysis(&featured_points, &clusters)?;

        // Calculate overall quality metrics
        let overall_quality = self.calculate_overall_quality(&clusters, &featured_points)?;

        // Generate recommendations
        let recommendations =
            self.generate_recommendations(&clusters, &landscape_analysis, &statistical_summary)?;

        // Update statistics
        self.stats.total_solutions += solutions.len();
        self.stats.total_time += start_time.elapsed();

        Ok(ClusteringResults {
            clusters,
            algorithm: self.config.algorithm.clone(),
            distance_metric: self.config.distance_metric.clone(),
            overall_quality,
            landscape_analysis,
            statistical_summary,
            performance_metrics: ClusteringPerformanceMetrics {
                clustering_time: start_time.elapsed(),
                analysis_time: start_time.elapsed(),
                memory_usage: 0, // Simplified
                scalability_metrics: ScalabilityMetrics {
                    time_complexity: "O(n^2)".to_string(),
                    space_complexity: "O(n^2)".to_string(),
                    scaling_factor: 2.0,
                    parallelization_efficiency: 0.8,
                },
                efficiency_metrics: EfficiencyMetrics {
                    convergence_efficiency: 0.85,
                    resource_utilization: 0.75,
                    quality_time_ratio: 0.9,
                    robustness: 0.8,
                },
            },
            recommendations,
        })
    }

    /// Convert annealing solutions to solution points
    pub fn convert_solutions(
        &self,
        solutions: &[AnnealingSolution],
    ) -> ClusteringResult<Vec<SolutionPoint>> {
        let mut solution_points = Vec::new();

        for (i, solution) in solutions.iter().enumerate() {
            let mut metrics = HashMap::new();
            metrics.insert("energy".to_string(), solution.best_energy);
            metrics.insert("num_evaluations".to_string(), solution.total_sweeps as f64);

            solution_points.push(SolutionPoint {
                solution: solution.best_spins.clone(),
                energy: solution.best_energy,
                metrics,
                metadata: SolutionMetadata {
                    id: i,
                    source: "annealing".to_string(),
                    timestamp: Instant::now(),
                    iterations: solution.total_sweeps,
                    quality_rank: None,
                    is_feasible: true, // Simplified
                },
                features: None,
            });
        }

        Ok(solution_points)
    }

    /// Extract features from solution points
    fn extract_features(
        &self,
        mut solution_points: Vec<SolutionPoint>,
    ) -> ClusteringResult<Vec<SolutionPoint>> {
        match &self.config.feature_extraction {
            FeatureExtractionMethod::Raw => {
                for point in &mut solution_points {
                    point.features = Some(point.solution.iter().map(|&x| f64::from(x)).collect());
                }
            }
            FeatureExtractionMethod::EnergyBased => {
                for point in &mut solution_points {
                    let mut features = vec![point.energy];
                    features.extend(point.solution.iter().map(|&x| f64::from(x)));
                    point.features = Some(features);
                }
            }
            FeatureExtractionMethod::Structural => {
                for point in &mut solution_points {
                    let features = self.extract_structural_features(&point.solution);
                    point.features = Some(features);
                }
            }
            FeatureExtractionMethod::PCA { num_components } => {
                // Simplified PCA implementation
                let features = self.apply_pca(&solution_points, *num_components)?;
                for (point, feature_vec) in solution_points.iter_mut().zip(features.iter()) {
                    point.features = Some(feature_vec.clone());
                }
            }
            _ => {
                // Default to raw features
                for point in &mut solution_points {
                    point.features = Some(point.solution.iter().map(|&x| f64::from(x)).collect());
                }
            }
        }

        Ok(solution_points)
    }

    /// Extract structural features from a solution
    #[must_use]
    pub fn extract_structural_features(&self, solution: &[i8]) -> Vec<f64> {
        let mut features = Vec::new();

        // Basic structural features
        let num_ones = solution.iter().filter(|&&x| x == 1).count() as f64;
        let num_neg_ones = solution.iter().filter(|&&x| x == -1).count() as f64;

        features.push(num_ones);
        features.push(num_neg_ones);
        features.push(num_ones / solution.len() as f64); // Fraction of +1 spins

        // Consecutive patterns
        let mut consecutive_ones = 0;
        let mut consecutive_neg_ones = 0;
        let mut max_consecutive_ones = 0;
        let mut max_consecutive_neg_ones = 0;

        for &spin in solution {
            if spin == 1 {
                consecutive_ones += 1;
                consecutive_neg_ones = 0;
                max_consecutive_ones = max_consecutive_ones.max(consecutive_ones);
            } else {
                consecutive_neg_ones += 1;
                consecutive_ones = 0;
                max_consecutive_neg_ones = max_consecutive_neg_ones.max(consecutive_neg_ones);
            }
        }

        features.push(f64::from(max_consecutive_ones));
        features.push(f64::from(max_consecutive_neg_ones));

        // Transition count
        let transitions = solution
            .windows(2)
            .filter(|window| window[0] != window[1])
            .count() as f64;

        features.push(transitions);

        features
    }

    /// Apply real PCA to solution points: mean-center the spin configurations,
    /// form the `d x d` covariance matrix, diagonalize it with the Jacobi
    /// eigenvalue algorithm (see [`jacobi_eigen_symmetric`]), and project each
    /// point onto the `num_components` eigenvectors with the largest
    /// eigenvalues (i.e. the directions of maximum variance) — not merely the
    /// first `num_components` raw coordinate axes.
    fn apply_pca(
        &self,
        solution_points: &[SolutionPoint],
        num_components: usize,
    ) -> ClusteringResult<Vec<Vec<f64>>> {
        if solution_points.is_empty() {
            return Ok(Vec::new());
        }

        let n = solution_points.len();
        let d = solution_points[0].solution.len();
        let k = num_components.min(d);

        if d == 0 || k == 0 {
            return Ok(vec![Vec::new(); n]);
        }

        // Create data matrix
        let mut data = vec![vec![0.0; d]; n];
        for (i, point) in solution_points.iter().enumerate() {
            for (j, &spin) in point.solution.iter().enumerate() {
                data[i][j] = f64::from(spin);
            }
        }

        // Center the data
        let mut means = vec![0.0; d];
        for j in 0..d {
            means[j] = data.iter().map(|row| row[j]).sum::<f64>() / n as f64;
        }
        for row in &mut data {
            for j in 0..d {
                row[j] -= means[j];
            }
        }

        // d x d covariance matrix (population covariance; falls back to 0 for n <= 1).
        let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
        let mut covariance = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in i..d {
                let value = data.iter().map(|row| row[i] * row[j]).sum::<f64>() / denom;
                covariance[i][j] = value;
                covariance[j][i] = value;
            }
        }

        // Eigendecompose the (symmetric) covariance matrix and take the
        // `k` eigenvectors with the largest eigenvalues.
        let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&covariance, 100);
        let mut order: Vec<usize> = (0..d).collect();
        order.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_components: Vec<usize> = order.into_iter().take(k).collect();

        let mut pca_data = Vec::with_capacity(n);
        for row in &data {
            let mut pca_row = Vec::with_capacity(k);
            for &component_idx in &top_components {
                let projection: f64 = (0..d)
                    .map(|j| row[j] * eigenvectors[j][component_idx])
                    .sum();
                pca_row.push(projection);
            }
            pca_data.push(pca_row);
        }

        Ok(pca_data)
    }

    /// Perform clustering on solution points
    fn perform_clustering(
        &self,
        solution_points: &[SolutionPoint],
    ) -> ClusteringResult<Vec<SolutionCluster>> {
        match &self.config.algorithm {
            ClusteringAlgorithm::KMeans { k, max_iterations } => {
                self.kmeans_clustering(solution_points, *k, *max_iterations)
            }
            ClusteringAlgorithm::Hierarchical {
                linkage,
                distance_threshold,
            } => self.hierarchical_clustering(solution_points, linkage, *distance_threshold),
            ClusteringAlgorithm::DBSCAN { eps, min_samples } => {
                self.dbscan_clustering(solution_points, *eps, *min_samples)
            }
            _ => {
                // Default to k-means
                self.kmeans_clustering(solution_points, 5, 100)
            }
        }
    }

    /// K-means clustering implementation
    pub fn kmeans_clustering(
        &self,
        solution_points: &[SolutionPoint],
        k: usize,
        max_iterations: usize,
    ) -> ClusteringResult<Vec<SolutionCluster>> {
        if solution_points.len() < k {
            return Err(ClusteringError::InsufficientData {
                required: k,
                actual: solution_points.len(),
            });
        }

        let n = solution_points.len();
        let features = solution_points
            .iter()
            .map(|p| {
                p.features.as_ref().ok_or_else(|| {
                    ClusteringError::DataError("Solution point missing features".to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let d = features[0].len();

        // Initialize centroids randomly
        let mut rng = match self.config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(thread_rng().random()),
        };

        let mut centroids = Vec::new();
        for _ in 0..k {
            let mut centroid = Vec::new();
            for _ in 0..d {
                centroid.push(rng.random_range(-1.0..1.0));
            }
            centroids.push(centroid);
        }

        let mut assignments = vec![0; n];

        // K-means iterations
        for _iteration in 0..max_iterations {
            let mut changed = false;

            // Assign points to closest centroids
            for (i, feature_vec) in features.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_distance = f64::INFINITY;

                for (j, centroid) in centroids.iter().enumerate() {
                    let distance = self.calculate_distance(feature_vec, centroid)?;
                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = j;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            // Update centroids
            for j in 0..k {
                let cluster_points: Vec<_> = features
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == j)
                    .map(|(_, features)| *features)
                    .collect();

                if !cluster_points.is_empty() {
                    for dim in 0..d {
                        centroids[j][dim] =
                            cluster_points.iter().map(|point| point[dim]).sum::<f64>()
                                / cluster_points.len() as f64;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        // Create clusters
        let mut clusters = Vec::new();
        for cluster_id in 0..k {
            let cluster_solutions: Vec<_> = solution_points
                .iter()
                .enumerate()
                .filter(|(i, _)| assignments[*i] == cluster_id)
                .map(|(_, point)| point.clone())
                .collect();

            if !cluster_solutions.is_empty() {
                let statistics = self.calculate_cluster_statistics(&cluster_solutions);
                let quality_metrics = self
                    .calculate_cluster_quality_metrics(&cluster_solutions, &centroids[cluster_id]);

                clusters.push(SolutionCluster {
                    id: cluster_id,
                    solutions: cluster_solutions,
                    centroid: centroids[cluster_id].clone(),
                    representative: None, // Will be set later
                    statistics,
                    quality_metrics,
                });
            }
        }

        Ok(clusters)
    }

    /// Agglomerative hierarchical clustering honoring the caller-selected
    /// [`LinkageType`] (see [`Self::calculate_cluster_distance`]).
    fn hierarchical_clustering(
        &self,
        solution_points: &[SolutionPoint],
        linkage: &LinkageType,
        distance_threshold: f64,
    ) -> ClusteringResult<Vec<SolutionCluster>> {
        let n = solution_points.len();
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        while clusters.len() > 1 {
            let mut min_distance = f64::INFINITY;
            let mut merge_indices = (0, 1);

            // Find closest clusters
            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let distance = self.calculate_cluster_distance(
                        &clusters[i],
                        &clusters[j],
                        solution_points,
                        linkage,
                    )?;
                    if distance < min_distance {
                        min_distance = distance;
                        merge_indices = (i, j);
                    }
                }
            }

            if min_distance > distance_threshold {
                break;
            }

            // Merge clusters
            let (i, j) = merge_indices;
            let mut merged_cluster = clusters[i].clone();
            merged_cluster.extend_from_slice(&clusters[j]);

            // Remove original clusters and add merged cluster
            if i < j {
                clusters.remove(j);
                clusters.remove(i);
            } else {
                clusters.remove(i);
                clusters.remove(j);
            }
            clusters.push(merged_cluster);
        }

        // Convert to SolutionCluster format
        let mut result_clusters = Vec::new();
        for (cluster_id, cluster_indices) in clusters.iter().enumerate() {
            let cluster_solutions: Vec<_> = cluster_indices
                .iter()
                .map(|&i| solution_points[i].clone())
                .collect();

            if !cluster_solutions.is_empty() {
                let centroid = self.calculate_centroid(&cluster_solutions)?;
                let statistics = self.calculate_cluster_statistics(&cluster_solutions);
                let quality_metrics =
                    self.calculate_cluster_quality_metrics(&cluster_solutions, &centroid);

                result_clusters.push(SolutionCluster {
                    id: cluster_id,
                    solutions: cluster_solutions,
                    centroid,
                    representative: None,
                    statistics,
                    quality_metrics,
                });
            }
        }

        Ok(result_clusters)
    }

    /// DBSCAN clustering implementation (simplified)
    fn dbscan_clustering(
        &self,
        solution_points: &[SolutionPoint],
        eps: f64,
        min_samples: usize,
    ) -> ClusteringResult<Vec<SolutionCluster>> {
        let n = solution_points.len();
        let mut labels = vec![-1i32; n]; // -1 = noise, 0+ = cluster id
        let mut cluster_id = 0;

        for i in 0..n {
            if labels[i] != -1 {
                continue; // Already processed
            }

            let neighbors = self.find_neighbors(i, solution_points, eps)?;

            if neighbors.len() < min_samples {
                labels[i] = -1; // Mark as noise
                continue;
            }

            // Start new cluster
            labels[i] = cluster_id;
            let mut queue = VecDeque::from(neighbors);

            while let Some(j) = queue.pop_front() {
                if labels[j] == -1 {
                    labels[j] = cluster_id; // Change noise to border point
                } else if labels[j] != -1 {
                    continue; // Already in a cluster
                }

                labels[j] = cluster_id;
                let j_neighbors = self.find_neighbors(j, solution_points, eps)?;

                if j_neighbors.len() >= min_samples {
                    for &neighbor in &j_neighbors {
                        if labels[neighbor] == -1 || labels[neighbor] == cluster_id {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            cluster_id += 1;
        }

        // Convert to SolutionCluster format
        let mut result_clusters = Vec::new();
        for cid in 0..cluster_id {
            let cluster_solutions: Vec<_> = solution_points
                .iter()
                .enumerate()
                .filter(|(i, _)| labels[*i] == cid)
                .map(|(_, point)| point.clone())
                .collect();

            if !cluster_solutions.is_empty() {
                let centroid = self.calculate_centroid(&cluster_solutions)?;
                let statistics = self.calculate_cluster_statistics(&cluster_solutions);
                let quality_metrics =
                    self.calculate_cluster_quality_metrics(&cluster_solutions, &centroid);

                result_clusters.push(SolutionCluster {
                    id: cid as usize,
                    solutions: cluster_solutions,
                    centroid,
                    representative: None,
                    statistics,
                    quality_metrics,
                });
            }
        }

        Ok(result_clusters)
    }

    /// Find neighbors within eps distance
    fn find_neighbors(
        &self,
        point_idx: usize,
        solution_points: &[SolutionPoint],
        eps: f64,
    ) -> ClusteringResult<Vec<usize>> {
        let mut neighbors = Vec::new();
        let point_features = solution_points[point_idx]
            .features
            .as_ref()
            .ok_or_else(|| {
                ClusteringError::DataError("Solution point missing features".to_string())
            })?;

        for (i, other_point) in solution_points.iter().enumerate() {
            if i != point_idx {
                let other_features = other_point.features.as_ref().ok_or_else(|| {
                    ClusteringError::DataError("Solution point missing features".to_string())
                })?;
                let distance = self.calculate_distance(point_features, other_features)?;
                if distance <= eps {
                    neighbors.push(i);
                }
            }
        }

        Ok(neighbors)
    }

    /// Calculate the inter-cluster distance used to decide the next merge in
    /// [`Self::hierarchical_clustering`], per the caller-selected [`LinkageType`]:
    /// * `Single`: minimum pairwise distance between the two clusters.
    /// * `Complete`: maximum pairwise distance between the two clusters.
    /// * `Average`: mean of all pairwise distances between the two clusters.
    /// * `Ward`: the Lance-Williams/Ward increase-in-variance criterion
    ///   `sqrt(2*|A|*|B| / (|A|+|B|)) * ||centroid_A - centroid_B||`.
    fn calculate_cluster_distance(
        &self,
        cluster1: &[usize],
        cluster2: &[usize],
        solution_points: &[SolutionPoint],
        linkage: &LinkageType,
    ) -> ClusteringResult<f64> {
        let feature_of = |i: usize| -> ClusteringResult<&[f64]> {
            solution_points[i].features.as_deref().ok_or_else(|| {
                ClusteringError::DataError("Solution point missing features".to_string())
            })
        };

        match linkage {
            LinkageType::Single => {
                let mut min_distance = f64::INFINITY;
                for &i in cluster1 {
                    for &j in cluster2 {
                        let distance = self.calculate_distance(feature_of(i)?, feature_of(j)?)?;
                        min_distance = min_distance.min(distance);
                    }
                }
                Ok(min_distance)
            }
            LinkageType::Complete => {
                let mut max_distance = 0.0f64;
                for &i in cluster1 {
                    for &j in cluster2 {
                        let distance = self.calculate_distance(feature_of(i)?, feature_of(j)?)?;
                        max_distance = max_distance.max(distance);
                    }
                }
                Ok(max_distance)
            }
            LinkageType::Average => {
                let mut sum = 0.0;
                let mut count = 0usize;
                for &i in cluster1 {
                    for &j in cluster2 {
                        sum += self.calculate_distance(feature_of(i)?, feature_of(j)?)?;
                        count += 1;
                    }
                }
                Ok(if count == 0 { 0.0 } else { sum / count as f64 })
            }
            LinkageType::Ward => {
                let centroid1 = self.centroid_of_indices(cluster1, solution_points)?;
                let centroid2 = self.centroid_of_indices(cluster2, solution_points)?;
                let centroid_distance = self.calculate_distance(&centroid1, &centroid2)?;
                let n1 = cluster1.len() as f64;
                let n2 = cluster2.len() as f64;
                let factor = (2.0 * n1 * n2 / (n1 + n2)).sqrt();
                Ok(factor * centroid_distance)
            }
        }
    }

    /// Mean feature vector over a set of solution-point indices, used by the
    /// Ward-linkage branch of [`Self::calculate_cluster_distance`].
    fn centroid_of_indices(
        &self,
        indices: &[usize],
        solution_points: &[SolutionPoint],
    ) -> ClusteringResult<Vec<f64>> {
        let mut centroid: Vec<f64> = Vec::new();
        for &i in indices {
            let features = solution_points[i].features.as_deref().ok_or_else(|| {
                ClusteringError::DataError(
                    "Solution point missing features for centroid calculation".to_string(),
                )
            })?;
            if centroid.is_empty() {
                centroid = vec![0.0; features.len()];
            } else if features.len() != centroid.len() {
                return Err(ClusteringError::DimensionMismatch {
                    expected: centroid.len(),
                    actual: features.len(),
                });
            }
            for (c, f) in centroid.iter_mut().zip(features.iter()) {
                *c += f;
            }
        }
        let n = indices.len() as f64;
        if n > 0.0 {
            for c in centroid.iter_mut() {
                *c /= n;
            }
        }
        Ok(centroid)
    }

    /// Calculate distance between two feature vectors
    pub fn calculate_distance(
        &self,
        features1: &[f64],
        features2: &[f64],
    ) -> ClusteringResult<f64> {
        if features1.len() != features2.len() {
            return Err(ClusteringError::DimensionMismatch {
                expected: features1.len(),
                actual: features2.len(),
            });
        }

        match self.config.distance_metric {
            DistanceMetric::Euclidean => {
                let sum_sq: f64 = features1
                    .iter()
                    .zip(features2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                Ok(sum_sq.sqrt())
            }
            DistanceMetric::Manhattan => {
                let sum_abs: f64 = features1
                    .iter()
                    .zip(features2.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                Ok(sum_abs)
            }
            DistanceMetric::Hamming => {
                let diff_count = features1
                    .iter()
                    .zip(features2.iter())
                    .filter(|(a, b)| (*a - *b).abs() > 1e-10)
                    .count();
                Ok(diff_count as f64)
            }
            DistanceMetric::Cosine => {
                let dot_product: f64 = features1
                    .iter()
                    .zip(features2.iter())
                    .map(|(a, b)| a * b)
                    .sum();

                let norm1: f64 = features1.iter().map(|x| x * x).sum::<f64>().sqrt();
                let norm2: f64 = features2.iter().map(|x| x * x).sum::<f64>().sqrt();

                if norm1 > 1e-10 && norm2 > 1e-10 {
                    Ok(1.0 - dot_product / (norm1 * norm2))
                } else {
                    Ok(1.0)
                }
            }
            _ => {
                // Default to Euclidean
                let sum_sq: f64 = features1
                    .iter()
                    .zip(features2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                Ok(sum_sq.sqrt())
            }
        }
    }

    /// Calculate centroid of a cluster
    fn calculate_centroid(
        &self,
        cluster_solutions: &[SolutionPoint],
    ) -> ClusteringResult<Vec<f64>> {
        if cluster_solutions.is_empty() {
            return Ok(Vec::new());
        }

        let features_dim = cluster_solutions[0]
            .features
            .as_ref()
            .ok_or_else(|| {
                ClusteringError::DataError(
                    "Solution point missing features for centroid calculation".to_string(),
                )
            })?
            .len();
        let mut centroid = vec![0.0; features_dim];

        for solution in cluster_solutions {
            let features = solution.features.as_ref().ok_or_else(|| {
                ClusteringError::DataError(
                    "Solution point missing features for centroid calculation".to_string(),
                )
            })?;
            if features.len() != features_dim {
                return Err(ClusteringError::DimensionMismatch {
                    expected: features_dim,
                    actual: features.len(),
                });
            }
            for (i, &value) in features.iter().enumerate() {
                centroid[i] += value;
            }
        }

        for value in &mut centroid {
            *value /= cluster_solutions.len() as f64;
        }

        Ok(centroid)
    }

    /// Calculate cluster statistics
    fn calculate_cluster_statistics(
        &self,
        cluster_solutions: &[SolutionPoint],
    ) -> ClusterStatistics {
        if cluster_solutions.is_empty() {
            return ClusterStatistics {
                size: 0,
                mean_energy: 0.0,
                energy_std: 0.0,
                min_energy: 0.0,
                max_energy: 0.0,
                intra_cluster_distance: 0.0,
                diameter: 0.0,
                density: 0.0,
            };
        }

        let energies: Vec<f64> = cluster_solutions.iter().map(|s| s.energy).collect();
        let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;
        let variance = energies
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f64>()
            / energies.len() as f64;
        let energy_std = variance.sqrt();

        let min_energy = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_energy = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate intra-cluster distance and diameter
        let mut total_distance = 0.0;
        let mut max_distance = 0.0f64;
        let mut distance_count = 0;

        for i in 0..cluster_solutions.len() {
            for j in (i + 1)..cluster_solutions.len() {
                if let (Some(features1), Some(features2)) = (
                    cluster_solutions[i].features.as_ref(),
                    cluster_solutions[j].features.as_ref(),
                ) {
                    if let Ok(distance) = self.calculate_distance(features1, features2) {
                        total_distance += distance;
                        max_distance = max_distance.max(distance);
                        distance_count += 1;
                    }
                }
            }
        }

        let intra_cluster_distance = if distance_count > 0 {
            total_distance / f64::from(distance_count)
        } else {
            0.0
        };

        ClusterStatistics {
            size: cluster_solutions.len(),
            mean_energy,
            energy_std,
            min_energy,
            max_energy,
            intra_cluster_distance,
            diameter: max_distance,
            density: if max_distance > 0.0 {
                cluster_solutions.len() as f64 / max_distance
            } else {
                0.0
            },
        }
    }

    /// Calculate the per-cluster quality metrics that are computable in
    /// isolation from a single cluster's own members (`inertia`, the real
    /// sum of squared distances to the centroid). `silhouette_coefficient`,
    /// `calinski_harabasz_index`, `davies_bouldin_index` and `stability` all
    /// require inter-cluster information (or, for `stability`, a bootstrap
    /// pass over the finished cluster), so they are initialized to neutral
    /// placeholders here and overwritten with real values by
    /// [`Self::update_global_quality_metrics`] once every cluster has been
    /// formed.
    fn calculate_cluster_quality_metrics(
        &self,
        cluster_solutions: &[SolutionPoint],
        centroid: &[f64],
    ) -> ClusterQualityMetrics {
        let mut inertia = 0.0;

        for solution in cluster_solutions {
            if let Some(features) = solution.features.as_ref() {
                if let Ok(distance) = self.calculate_distance(features, centroid) {
                    inertia += distance * distance;
                }
            }
        }

        ClusterQualityMetrics {
            silhouette_coefficient: 0.5, // Overwritten by update_global_quality_metrics.
            inertia,
            calinski_harabasz_index: 1.0, // Overwritten by update_global_quality_metrics.
            davies_bouldin_index: 1.0,    // Overwritten by update_global_quality_metrics.
            stability: 0.8,               // Overwritten by update_global_quality_metrics.
        }
    }
}

/// Diagonalize a real symmetric `n x n` matrix with the classical Jacobi
/// eigenvalue algorithm.
///
/// Repeatedly zeroes the largest off-diagonal element via a Givens rotation
/// until the matrix is (numerically) diagonal or `max_sweeps` full sweeps
/// have elapsed. Returns `(eigenvalues, eigenvectors)` where `eigenvectors[i][k]`
/// is the `i`-th component of the eigenvector for `eigenvalues[k]` (i.e. the
/// eigenvectors are the columns of the returned matrix). This is the real
/// numerical routine backing [`SolutionClusteringAnalyzer::apply_pca`]'s
/// covariance-matrix diagonalization.
fn jacobi_eigen_symmetric(matrix: &[Vec<f64>], max_sweeps: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    let mut a = matrix.to_vec();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    for _ in 0..max_sweeps {
        let mut off_diagonal_norm = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off_diagonal_norm += a[i][j] * a[i][j];
                }
            }
        }
        if off_diagonal_norm.sqrt() < 1e-12 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-14 {
                    continue;
                }

                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                let a_pp = a[p][p];
                let a_qq = a[q][q];
                let a_pq = a[p][q];
                a[p][p] = c.mul_add(c * a_pp, -(2.0 * s * c * a_pq)) + s * s * a_qq;
                a[q][q] = s.mul_add(s * a_pp, 2.0 * s * c * a_pq) + c * c * a_qq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                for i in 0..n {
                    if i != p && i != q {
                        let a_ip = a[i][p];
                        let a_iq = a[i][q];
                        a[i][p] = c * a_ip - s * a_iq;
                        a[p][i] = a[i][p];
                        a[i][q] = s * a_ip + c * a_iq;
                        a[q][i] = a[i][q];
                    }
                }

                for i in 0..n {
                    let v_ip = v[i][p];
                    let v_iq = v[i][q];
                    v[i][p] = c * v_ip - s * v_iq;
                    v[i][q] = s * v_ip + c * v_iq;
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::solution_clustering::config::create_basic_clustering_config;

    fn make_point(solution: Vec<i8>, features: Option<Vec<f64>>) -> SolutionPoint {
        SolutionPoint {
            solution,
            energy: 0.0,
            metrics: HashMap::new(),
            metadata: SolutionMetadata {
                id: 0,
                source: "test".to_string(),
                timestamp: Instant::now(),
                iterations: 0,
                quality_rank: None,
                is_feasible: true,
            },
            features,
        }
    }

    #[test]
    fn apply_pca_finds_the_real_variance_direction_not_the_first_raw_axis() {
        let analyzer = SolutionClusteringAnalyzer::new(create_basic_clustering_config());

        // Dimensions 0 and 1 are constant (zero variance) across every point;
        // all of the real variance lives in dimension 2. The old "simplified"
        // implementation just truncated to the first `num_components` raw
        // coordinates, so it would have returned an all-zero component here.
        let points = vec![
            make_point(vec![1, 1, -1], None),
            make_point(vec![1, 1, -1], None),
            make_point(vec![1, 1, 1], None),
            make_point(vec![1, 1, 1], None),
        ];

        let projected = analyzer.apply_pca(&points, 1).expect("PCA should succeed");

        assert_eq!(projected.len(), 4);
        for row in &projected {
            assert_eq!(row.len(), 1);
            // A real PCA finds the actual (unit-normalized) direction of
            // variance; the magnitude should be non-trivial, not the ~0.0
            // that truncating the constant raw axes 0/1 would have produced.
            assert!(
                row[0].abs() > 0.5,
                "expected the real variance direction to be captured, got {row:?}"
            );
        }
        // Points 0/1 share dimension-2 value -1, points 2/3 share +1: the
        // real principal component must separate them into two groups with
        // opposite-signed (or clearly distinct) projections.
        assert!((projected[0][0] - projected[1][0]).abs() < 1e-6);
        assert!((projected[2][0] - projected[3][0]).abs() < 1e-6);
        assert!((projected[0][0] - projected[2][0]).abs() > 1e-3);
    }

    #[test]
    fn hierarchical_clustering_respects_the_selected_linkage_type() {
        let analyzer = SolutionClusteringAnalyzer::new(create_basic_clustering_config());

        // 1D points at 0, 3, 7. With a merge threshold of 4:
        // * Single linkage merges {0,3} then compares min(dist(0,7)=7, dist(3,7)=4)=4 <= 4,
        //   so all three end up in one cluster.
        // * Complete linkage compares max(dist(0,7)=7, dist(3,7)=4)=7 > 4, so {0,3} and {7}
        //   remain separate clusters.
        let points = vec![
            make_point(vec![1], Some(vec![0.0])),
            make_point(vec![1], Some(vec![3.0])),
            make_point(vec![1], Some(vec![7.0])),
        ];

        let single_clusters = analyzer
            .hierarchical_clustering(&points, &LinkageType::Single, 4.0)
            .expect("single linkage should succeed");
        let complete_clusters = analyzer
            .hierarchical_clustering(&points, &LinkageType::Complete, 4.0)
            .expect("complete linkage should succeed");

        assert_eq!(
            single_clusters.len(),
            1,
            "single linkage should chain all three points into one cluster"
        );
        assert_eq!(
            complete_clusters.len(),
            2,
            "complete linkage should keep the distant point separate"
        );
    }
}
