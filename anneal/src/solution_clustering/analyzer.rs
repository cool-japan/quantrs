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
    DifficultyLevel, DistributionAnalysis, DistributionType, EfficiencyMetrics, EnergyBasin,
    EnergyStatistics, FunnelAnalysis, LandscapeAnalysis, MultiModalityAnalysis,
    OptimizationRecommendation, OverallClusteringQuality, PlateauAnalysis, PriorityLevel,
    RecommendationType, RuggednessMetrics, ScalabilityMetrics, SolutionCluster, SolutionMetadata,
    SolutionPoint, StatisticalSummary,
};
use crate::simulator::AnnealingSolution;

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

    /// Apply PCA to solution points (simplified implementation)
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

        for i in 0..n {
            for j in 0..d {
                data[i][j] -= means[j];
            }
        }

        // Simplified PCA: just take first num_components dimensions
        let mut pca_data = Vec::new();
        for i in 0..n {
            let mut pca_row = Vec::new();
            for j in 0..num_components.min(d) {
                pca_row.push(data[i][j]);
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

    /// Hierarchical clustering implementation (simplified)
    fn hierarchical_clustering(
        &self,
        solution_points: &[SolutionPoint],
        _linkage: &LinkageType,
        distance_threshold: f64,
    ) -> ClusteringResult<Vec<SolutionCluster>> {
        // Simplified implementation using single linkage
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

    /// Calculate distance between cluster indices
    fn calculate_cluster_distance(
        &self,
        cluster1: &[usize],
        cluster2: &[usize],
        solution_points: &[SolutionPoint],
    ) -> ClusteringResult<f64> {
        let mut min_distance = f64::INFINITY;

        for &i in cluster1 {
            for &j in cluster2 {
                let features1 = solution_points[i].features.as_ref().ok_or_else(|| {
                    ClusteringError::DataError("Solution point missing features".to_string())
                })?;
                let features2 = solution_points[j].features.as_ref().ok_or_else(|| {
                    ClusteringError::DataError("Solution point missing features".to_string())
                })?;
                let distance = self.calculate_distance(features1, features2)?;
                min_distance = min_distance.min(distance);
            }
        }

        Ok(min_distance)
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

    /// Calculate cluster quality metrics
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
            silhouette_coefficient: 0.5, // Simplified
            inertia,
            calinski_harabasz_index: 1.0, // Simplified
            davies_bouldin_index: 1.0,    // Simplified
            stability: 0.8,               // Simplified
        }
    }

    /// Analyze the solution landscape
    fn analyze_landscape(
        &self,
        solution_points: &[SolutionPoint],
        clusters: &[SolutionCluster],
    ) -> ClusteringResult<LandscapeAnalysis> {
        let energy_statistics = self.calculate_energy_statistics(solution_points);
        let basins = self.detect_energy_basins(solution_points, clusters);
        let connectivity = self.analyze_connectivity(solution_points);
        let multi_modality = self.analyze_multi_modality(solution_points);
        let ruggedness = self.calculate_ruggedness_metrics(solution_points);
        let funnel_analysis = self.analyze_funnel_structure(solution_points, clusters);

        Ok(LandscapeAnalysis {
            energy_statistics,
            basins,
            connectivity,
            multi_modality,
            ruggedness,
            funnel_analysis,
        })
    }

    /// Calculate energy statistics
    #[must_use]
    pub fn calculate_energy_statistics(
        &self,
        solution_points: &[SolutionPoint],
    ) -> EnergyStatistics {
        let energies: Vec<f64> = solution_points.iter().map(|s| s.energy).collect();

        if energies.is_empty() {
            return EnergyStatistics {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                percentiles: Vec::new(),
                skewness: 0.0,
                kurtosis: 0.0,
                num_distinct_energies: 0,
            };
        }

        let mut sorted_energies = energies.clone();
        sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = energies.iter().sum::<f64>() / energies.len() as f64;
        let variance =
            energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / energies.len() as f64;
        let std_dev = variance.sqrt();

        let min = sorted_energies[0];
        let max = sorted_energies[sorted_energies.len() - 1];

        // Calculate percentiles
        let percentiles = vec![
            sorted_energies[sorted_energies.len() * 25 / 100],
            sorted_energies[sorted_energies.len() * 50 / 100],
            sorted_energies[sorted_energies.len() * 75 / 100],
        ];

        // Calculate skewness and kurtosis (simplified)
        let skewness = if std_dev > 1e-10 {
            energies
                .iter()
                .map(|e| ((e - mean) / std_dev).powi(3))
                .sum::<f64>()
                / energies.len() as f64
        } else {
            0.0
        };

        let kurtosis = if std_dev > 1e-10 {
            energies
                .iter()
                .map(|e| ((e - mean) / std_dev).powi(4))
                .sum::<f64>()
                / energies.len() as f64
                - 3.0
        } else {
            0.0
        };

        let mut sorted_energies = energies;
        sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_energies.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        let num_distinct_energies = sorted_energies.len();

        EnergyStatistics {
            mean,
            std_dev,
            min,
            max,
            percentiles,
            skewness,
            kurtosis,
            num_distinct_energies,
        }
    }

    /// Detect energy basins in the landscape.
    ///
    /// Each cluster is treated as a basin. Per-basin depth is computed against
    /// the global minimum energy across `solution_points` (so the deepest basin
    /// has depth `0`, and shallower basins have positive depth — interpreted as
    /// "energy above the global minimum"). Width is the basin's energy range.
    /// `escape_barrier` is left at `0.0` since a real estimate requires an
    /// inter-basin transition graph that is not maintained here.
    pub(crate) fn detect_energy_basins(
        &self,
        solution_points: &[SolutionPoint],
        clusters: &[SolutionCluster],
    ) -> Vec<EnergyBasin> {
        let mut basins = Vec::new();

        let global_min = solution_points
            .iter()
            .map(|s| s.energy)
            .fold(f64::INFINITY, f64::min);

        for (basin_id, cluster) in clusters.iter().enumerate() {
            let energies: Vec<f64> = cluster.solutions.iter().map(|s| s.energy).collect();

            if !energies.is_empty() {
                let min_energy = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_energy = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let depth = if global_min.is_finite() {
                    (min_energy - global_min).max(0.0)
                } else {
                    0.0
                };

                basins.push(EnergyBasin {
                    id: basin_id,
                    solutions: cluster.solutions.iter().map(|s| s.metadata.id).collect(),
                    min_energy,
                    size: cluster.solutions.len(),
                    depth,
                    width: max_energy - min_energy,
                    escape_barrier: 0.0, // Requires an inter-basin transition graph.
                });
            }
        }

        basins
    }

    /// Analyze connectivity of the solution landscape via single-link clustering
    /// over an epsilon-Hamming-neighbour graph.
    ///
    /// Two solutions are considered "connected" when the Hamming distance between
    /// their spin vectors is at most `eps_hamming`. Connected components are then
    /// found with a union-find pass, yielding `num_components` and
    /// `largest_component_size`. `average_path_length`, `clustering_coefficient`,
    /// and `diameter` are still simplified estimates.
    pub(crate) fn analyze_connectivity(
        &self,
        solution_points: &[SolutionPoint],
    ) -> ConnectivityAnalysis {
        let n = solution_points.len();
        if n == 0 {
            return ConnectivityAnalysis {
                num_components: 0,
                largest_component_size: 0,
                average_path_length: 0.0,
                clustering_coefficient: 0.0,
                diameter: 0,
            };
        }
        if n == 1 {
            return ConnectivityAnalysis {
                num_components: 1,
                largest_component_size: 1,
                average_path_length: 0.0,
                clustering_coefficient: 0.0,
                diameter: 0,
            };
        }

        // Use a Hamming-neighbour threshold of 1 by default. Any two solutions
        // differing in a single spin are direct neighbours; chains of such
        // single-flip moves form a connected component.
        let eps_hamming: usize = 1;

        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let a = &solution_points[i].solution;
                let b = &solution_points[j].solution;
                if a.len() != b.len() {
                    continue;
                }
                let hd = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
                if hd <= eps_hamming {
                    let ra = find(&mut parent, i);
                    let rb = find(&mut parent, j);
                    if ra != rb {
                        parent[ra] = rb;
                    }
                }
            }
        }

        let mut sizes: HashMap<usize, usize> = HashMap::new();
        for i in 0..n {
            let r = find(&mut parent, i);
            *sizes.entry(r).or_insert(0) += 1;
        }

        let num_components = sizes.len();
        let largest_component_size = sizes.values().copied().max().unwrap_or(0);

        // Conservative simplified estimates for the remaining fields.
        let average_path_length = if num_components == 0 {
            0.0
        } else {
            (n as f64 / num_components as f64).sqrt()
        };

        ConnectivityAnalysis {
            num_components,
            largest_component_size,
            average_path_length,
            clustering_coefficient: 0.3, // Heuristic — full computation is out of scope here.
            diameter: largest_component_size.saturating_sub(1),
        }
    }

    /// Analyze multi-modality of the energy landscape via 1D histogram peak detection.
    ///
    /// The energy values are bucketed into `min(20, ceil(sqrt(n)))` equal-width
    /// bins between `min(E)` and `max(E)`. A bin is a mode when its count is
    /// strictly greater than both neighbour bins; the first and last bins are
    /// modes when their count exceeds their single neighbour. The mode energy
    /// is the bin centre; mode strength is the bin's relative population.
    /// Inter-mode distances use the centre-to-centre absolute energy gap.
    pub(crate) fn analyze_multi_modality(
        &self,
        solution_points: &[SolutionPoint],
    ) -> MultiModalityAnalysis {
        let energies: Vec<f64> = solution_points.iter().map(|s| s.energy).collect();
        let n = energies.len();

        if n == 0 {
            return MultiModalityAnalysis {
                num_modes: 0,
                mode_energies: Vec::new(),
                mode_strengths: Vec::new(),
                inter_mode_distances: Vec::new(),
                multi_modality_index: 0.0,
            };
        }

        let min_e = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_e = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Degenerate range: a single mode at the common energy.
        if (max_e - min_e).abs() < 1e-12 {
            return MultiModalityAnalysis {
                num_modes: 1,
                mode_energies: vec![min_e],
                mode_strengths: vec![1.0],
                inter_mode_distances: vec![vec![0.0]],
                multi_modality_index: 0.0,
            };
        }

        let num_bins = ((n as f64).sqrt().ceil() as usize).clamp(2, 20);
        let bin_width = (max_e - min_e) / num_bins as f64;
        let mut counts = vec![0usize; num_bins];
        for &e in &energies {
            let mut idx = ((e - min_e) / bin_width).floor() as isize;
            if idx < 0 {
                idx = 0;
            }
            let mut idx = idx as usize;
            if idx >= num_bins {
                idx = num_bins - 1;
            }
            counts[idx] += 1;
        }

        // Detect peaks (strict local maxima with non-zero population).
        let mut mode_bins: Vec<usize> = Vec::new();
        for i in 0..num_bins {
            if counts[i] == 0 {
                continue;
            }
            let left_ok = i == 0 || counts[i] > counts[i - 1];
            let right_ok = i + 1 == num_bins || counts[i] > counts[i + 1];
            if left_ok && right_ok {
                mode_bins.push(i);
            }
        }

        // If the histogram is perfectly flat or monotone, fall back to "the
        // single most populous bin is the (only) mode" — guarantees at least one.
        if mode_bins.is_empty() {
            let (best_idx, _) = counts
                .iter()
                .enumerate()
                .max_by_key(|(_, c)| **c)
                .unwrap_or((0, &0));
            mode_bins.push(best_idx);
        }

        let mode_energies: Vec<f64> = mode_bins
            .iter()
            .map(|&b| min_e + (b as f64 + 0.5) * bin_width)
            .collect();
        let total = n as f64;
        let mode_strengths: Vec<f64> = mode_bins
            .iter()
            .map(|&b| counts[b] as f64 / total)
            .collect();

        // Symmetric inter-mode distance matrix in energy units.
        let m = mode_energies.len();
        let mut inter_mode_distances = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                inter_mode_distances[i][j] = (mode_energies[i] - mode_energies[j]).abs();
            }
        }

        // Multi-modality index: (m - 1) / m saturates toward 1 as more modes
        // are detected, 0 when there is just one. Capped at 1.
        let multi_modality_index = if m <= 1 {
            0.0
        } else {
            ((m - 1) as f64 / m as f64).min(1.0)
        };

        MultiModalityAnalysis {
            num_modes: m,
            mode_energies,
            mode_strengths,
            inter_mode_distances,
            multi_modality_index,
        }
    }

    /// Calculate ruggedness metrics for the solution landscape.
    ///
    /// Computes the lag-k autocorrelation of the energy sequence (treating the
    /// solution index order as a synthetic walk through the landscape) up to
    /// `max_lag = min(5, n - 1)`:
    ///
    /// ```text
    ///                Σ_{i=0}^{n-1-k} (e_i - μ)(e_{i+k} - μ)
    /// rho(k)  =  ─────────────────────────────────────────
    ///                       Σ_{i=0}^{n-1} (e_i - μ)^2
    /// ```
    ///
    /// The ruggedness coefficient is `1 - rho(1)`: smooth landscapes have
    /// `rho(1) ~= 1` and small ruggedness; rugged landscapes have low/negative
    /// `rho(1)` and ruggedness near or above 1.
    ///
    /// `epistasis` and `neutrality` are heuristic placeholders pending a true
    /// landscape walk infrastructure.
    pub(crate) fn calculate_ruggedness_metrics(
        &self,
        solution_points: &[SolutionPoint],
    ) -> RuggednessMetrics {
        let n = solution_points.len();
        if n < 2 {
            return RuggednessMetrics {
                autocorrelation: Vec::new(),
                ruggedness_coefficient: 0.0,
                num_local_optima: 0,
                epistasis: 0.0,
                neutrality: 0.0,
            };
        }

        let energies: Vec<f64> = solution_points.iter().map(|s| s.energy).collect();
        let mean = energies.iter().sum::<f64>() / n as f64;
        let denom: f64 = energies.iter().map(|e| (e - mean).powi(2)).sum();

        let max_lag = 5.min(n - 1);
        let mut autocorrelation = Vec::with_capacity(max_lag);
        if denom < 1e-12 {
            // Constant energy series — autocorrelation is conventionally 1 at every lag.
            for _ in 0..max_lag {
                autocorrelation.push(1.0);
            }
        } else {
            for k in 1..=max_lag {
                let mut num = 0.0;
                for i in 0..(n - k) {
                    num += (energies[i] - mean) * (energies[i + k] - mean);
                }
                autocorrelation.push(num / denom);
            }
        }

        let ruggedness_coefficient = autocorrelation
            .first()
            .map(|rho1| (1.0 - rho1).max(0.0))
            .unwrap_or(0.0);

        // Local optima along the index-ordered walk: positions where the energy
        // is strictly less than both neighbours (a 1D minimum). This is a real
        // count over the available data, not a placeholder.
        let mut num_local_optima = 0;
        for i in 1..(n - 1) {
            if energies[i] < energies[i - 1] && energies[i] < energies[i + 1] {
                num_local_optima += 1;
            }
        }

        RuggednessMetrics {
            autocorrelation,
            ruggedness_coefficient,
            num_local_optima,
            // Pending a proper neighbour-graph walk; left as documented heuristics.
            epistasis: 0.3,
            neutrality: 0.1,
        }
    }

    /// Analyze funnel structure
    fn analyze_funnel_structure(
        &self,
        _solution_points: &[SolutionPoint],
        clusters: &[SolutionCluster],
    ) -> FunnelAnalysis {
        // Simplified funnel analysis
        FunnelAnalysis {
            num_funnels: clusters.len(),
            funnel_depths: clusters.iter().map(|c| c.statistics.energy_std).collect(),
            funnel_widths: clusters.iter().map(|c| c.statistics.diameter).collect(),
            global_funnel: Some(0), // Simplified
            competition_index: 0.5, // Simplified
        }
    }

    /// Perform statistical analysis
    fn perform_statistical_analysis(
        &self,
        solution_points: &[SolutionPoint],
        clusters: &[SolutionCluster],
    ) -> ClusteringResult<StatisticalSummary> {
        let cluster_size_distribution = clusters.iter().map(|c| c.statistics.size).collect();

        let energy_distribution = DistributionAnalysis {
            distribution_type: DistributionType::Normal, // Simplified
            parameters: HashMap::from([("mean".to_string(), 0.0), ("std".to_string(), 1.0)]),
            goodness_of_fit: 0.8,
            confidence_intervals: vec![(0.1, 0.9)],
        };

        let convergence_analysis = ConvergenceAnalysis {
            trajectory_clusters: Vec::new(), // Simplified
            convergence_rates: vec![0.1, 0.2, 0.15],
            plateau_analysis: PlateauAnalysis {
                num_plateaus: 2,
                plateau_durations: vec![10, 15],
                plateau_energies: vec![-1.0, -0.5],
                escape_probabilities: vec![0.3, 0.7],
            },
            premature_convergence: false,
            diversity_evolution: vec![1.0, 0.8, 0.6, 0.4, 0.2],
        };

        let correlation_analysis = CorrelationAnalysis {
            variable_correlations: vec![
                vec![1.0; solution_points[0].solution.len()];
                solution_points[0].solution.len()
            ],
            energy_correlations: vec![0.1; solution_points[0].solution.len()],
            significant_correlations: Vec::new(),
            correlation_patterns: Vec::new(),
        };

        let outliers = Vec::new(); // Simplified

        Ok(StatisticalSummary {
            cluster_size_distribution,
            energy_distribution,
            convergence_analysis,
            correlation_analysis,
            outliers,
        })
    }

    /// Calculate overall clustering quality.
    ///
    /// The overall silhouette score is the size-weighted mean of the per-cluster
    /// silhouette coefficients (which are themselves means of per-point
    /// silhouettes), matching scikit-learn's `silhouette_score` convention. This
    /// is fed by the real values written by [`Self::update_global_quality_metrics`].
    fn calculate_overall_quality(
        &self,
        clusters: &[SolutionCluster],
        solution_points: &[SolutionPoint],
    ) -> ClusteringResult<OverallClusteringQuality> {
        let silhouette_score = if clusters.is_empty() {
            0.0
        } else {
            let total_points: usize = clusters.iter().map(|c| c.solutions.len()).sum();
            if total_points == 0 {
                0.0
            } else {
                clusters
                    .iter()
                    .map(|c| c.quality_metrics.silhouette_coefficient * c.solutions.len() as f64)
                    .sum::<f64>()
                    / total_points as f64
            }
        };

        let inter_cluster_separation = self.calculate_inter_cluster_separation(clusters)?;
        let cluster_cohesion = self.calculate_cluster_cohesion(clusters);

        Ok(OverallClusteringQuality {
            silhouette_score,
            adjusted_rand_index: None,
            normalized_mutual_information: None,
            inter_cluster_separation,
            cluster_cohesion,
            num_clusters: clusters.len(),
            optimal_num_clusters: self.estimate_optimal_clusters(solution_points)?,
        })
    }

    /// Calculate inter-cluster separation
    fn calculate_inter_cluster_separation(
        &self,
        clusters: &[SolutionCluster],
    ) -> ClusteringResult<f64> {
        if clusters.len() < 2 {
            return Ok(0.0);
        }

        let mut total_separation = 0.0;
        let mut count = 0;

        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let distance =
                    self.calculate_distance(&clusters[i].centroid, &clusters[j].centroid)?;
                total_separation += distance;
                count += 1;
            }
        }

        Ok(total_separation / f64::from(count))
    }

    /// Calculate cluster cohesion
    fn calculate_cluster_cohesion(&self, clusters: &[SolutionCluster]) -> f64 {
        if clusters.is_empty() {
            return 0.0;
        }

        clusters
            .iter()
            .map(|c| 1.0 / (1.0 + c.statistics.intra_cluster_distance))
            .sum::<f64>()
            / clusters.len() as f64
    }

    /// Estimate optimal number of clusters
    fn estimate_optimal_clusters(
        &self,
        solution_points: &[SolutionPoint],
    ) -> ClusteringResult<usize> {
        // Simplified elbow method
        let max_k = solution_points.len().min(10);
        let mut inertias = Vec::new();

        for k in 1..=max_k {
            if let Ok(clusters) = self.kmeans_clustering(solution_points, k, 50) {
                let total_inertia: f64 = clusters.iter().map(|c| c.quality_metrics.inertia).sum();
                inertias.push(total_inertia);
            }
        }

        // Find elbow (simplified)
        let optimal_k = if inertias.len() >= 3 {
            let mut max_diff = 0.0;
            let mut optimal = 1;

            for i in 1..inertias.len() - 1 {
                let diff = 2.0f64.mul_add(-inertias[i], inertias[i - 1]) + inertias[i + 1];
                if diff > max_diff {
                    max_diff = diff;
                    optimal = i + 1;
                }
            }
            optimal
        } else {
            inertias.len()
        };

        Ok(optimal_k)
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        clusters: &[SolutionCluster],
        landscape_analysis: &LandscapeAnalysis,
        _statistical_summary: &StatisticalSummary,
    ) -> ClusteringResult<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        // Recommendation based on cluster quality
        if clusters
            .iter()
            .any(|c| c.quality_metrics.silhouette_coefficient < 0.3)
        {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::ParameterTuning,
                description: "Low cluster quality detected. Consider tuning annealing parameters or using different initialization strategies.".to_string(),
                expected_improvement: 0.2,
                difficulty: DifficultyLevel::Easy,
                priority: PriorityLevel::High,
                evidence: vec!["Low silhouette coefficients in multiple clusters".to_string()],
            });
        }

        // Recommendation based on energy landscape
        if landscape_analysis.multi_modality.num_modes > 3 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::MultiStart,
                description: "Multiple modes detected in energy landscape. Consider using multi-start optimization or parallel runs.".to_string(),
                expected_improvement: 0.3,
                difficulty: DifficultyLevel::Moderate,
                priority: PriorityLevel::Medium,
                evidence: vec![format!("{} modes detected", landscape_analysis.multi_modality.num_modes)],
            });
        }

        // Recommendation based on cluster sizes
        let cluster_sizes: Vec<usize> = clusters.iter().map(|c| c.statistics.size).collect();
        let size_variance = cluster_sizes
            .iter()
            .map(|&size| {
                (size as f64
                    - cluster_sizes.iter().sum::<usize>() as f64 / cluster_sizes.len() as f64)
                    .powi(2)
            })
            .sum::<f64>()
            / cluster_sizes.len() as f64;

        if size_variance > 100.0 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::AlgorithmModification,
                description: "Highly unbalanced cluster sizes suggest potential convergence issues. Consider adjusting cooling schedule or using adaptive algorithms.".to_string(),
                expected_improvement: 0.15,
                difficulty: DifficultyLevel::Moderate,
                priority: PriorityLevel::Medium,
                evidence: vec![format!("Cluster size variance: {:.2}", size_variance)],
            });
        }

        Ok(recommendations)
    }

    /// Update global cluster quality metrics in a post-pass over all clusters.
    ///
    /// Silhouette, Davies-Bouldin, and Calinski-Harabasz indices all require
    /// inter-cluster information that is unavailable when each cluster is built
    /// in isolation by [`Self::kmeans_clustering`], [`Self::hierarchical_clustering`]
    /// or [`Self::dbscan_clustering`]. This pass walks every cluster simultaneously
    /// and writes the real values back into each cluster's
    /// [`super::types::ClusterQualityMetrics`].
    ///
    /// Definitions used:
    /// * Silhouette `s(i) = (b - a) / max(a, b)` where `a` is the mean intra-cluster
    ///   distance from point `i` and `b` is the smallest mean distance from `i` to
    ///   any other cluster. The per-cluster `silhouette_coefficient` is the mean of
    ///   `s(i)` over points in the cluster.
    /// * Davies-Bouldin per-cluster: `max_{j != i} ((sigma_i + sigma_j) / d(c_i, c_j))`
    ///   where `sigma_k` is the cluster's mean distance to its centroid and
    ///   `d(c_i, c_j)` is the distance between centroids.
    /// * Calinski-Harabasz: a single global value
    ///   `(BSS / (k - 1)) / (WSS / (n - k))` written into every cluster — this is the
    ///   conventional convention since the index is global, not per-cluster.
    pub(crate) fn update_global_quality_metrics(
        &self,
        clusters: &mut [SolutionCluster],
    ) -> ClusteringResult<()> {
        let k = clusters.len();
        if k == 0 {
            return Ok(());
        }

        // Single-cluster degenerate case: silhouette and DB are undefined; leave
        // sensible neutral values and CH at 0.
        if k == 1 {
            for c in clusters.iter_mut() {
                c.quality_metrics.silhouette_coefficient = 0.0;
                c.quality_metrics.davies_bouldin_index = 0.0;
                c.quality_metrics.calinski_harabasz_index = 0.0;
            }
            return Ok(());
        }

        // ---- Silhouette coefficient ----
        let mut per_cluster_silhouettes = vec![0.0f64; k];
        let mut per_cluster_counts = vec![0usize; k];

        for ci in 0..k {
            for point in &clusters[ci].solutions {
                let features = point.features.as_ref().ok_or_else(|| {
                    ClusteringError::DataError(
                        "Solution point missing features for silhouette calculation".to_string(),
                    )
                })?;

                // Mean intra-cluster distance `a`. Singletons contribute s(i)=0.
                let a = if clusters[ci].solutions.len() <= 1 {
                    0.0
                } else {
                    let mut sum = 0.0;
                    let mut count = 0usize;
                    for other in &clusters[ci].solutions {
                        if std::ptr::eq(other as *const _, point as *const _) {
                            continue;
                        }
                        let other_feat = other.features.as_ref().ok_or_else(|| {
                            ClusteringError::DataError(
                                "Solution point missing features for silhouette calculation"
                                    .to_string(),
                            )
                        })?;
                        sum += self.calculate_distance(features, other_feat)?;
                        count += 1;
                    }
                    if count == 0 {
                        0.0
                    } else {
                        sum / count as f64
                    }
                };

                // Minimum mean distance to any other cluster `b`.
                let mut b = f64::INFINITY;
                for cj in 0..k {
                    if cj == ci || clusters[cj].solutions.is_empty() {
                        continue;
                    }
                    let mut sum = 0.0;
                    let mut count = 0usize;
                    for other in &clusters[cj].solutions {
                        let other_feat = other.features.as_ref().ok_or_else(|| {
                            ClusteringError::DataError(
                                "Solution point missing features for silhouette calculation"
                                    .to_string(),
                            )
                        })?;
                        sum += self.calculate_distance(features, other_feat)?;
                        count += 1;
                    }
                    if count > 0 {
                        let mean = sum / count as f64;
                        if mean < b {
                            b = mean;
                        }
                    }
                }

                let s = if !b.is_finite() {
                    0.0
                } else if clusters[ci].solutions.len() <= 1 {
                    // Convention: singleton silhouette is 0.
                    0.0
                } else {
                    let denom = a.max(b);
                    if denom < 1e-12 {
                        0.0
                    } else {
                        (b - a) / denom
                    }
                };

                per_cluster_silhouettes[ci] += s;
                per_cluster_counts[ci] += 1;
            }
        }

        for ci in 0..k {
            let mean_s = if per_cluster_counts[ci] == 0 {
                0.0
            } else {
                per_cluster_silhouettes[ci] / per_cluster_counts[ci] as f64
            };
            clusters[ci].quality_metrics.silhouette_coefficient = mean_s;
        }

        // ---- Davies-Bouldin index ----
        // sigma_i = mean distance from each point in cluster i to that cluster's centroid.
        let mut sigma = vec![0.0f64; k];
        for ci in 0..k {
            if clusters[ci].solutions.is_empty() {
                continue;
            }
            let centroid = clusters[ci].centroid.clone();
            if centroid.is_empty() {
                continue;
            }
            let mut sum = 0.0;
            let mut count = 0usize;
            for point in &clusters[ci].solutions {
                let features = point.features.as_ref().ok_or_else(|| {
                    ClusteringError::DataError(
                        "Solution point missing features for Davies-Bouldin calculation"
                            .to_string(),
                    )
                })?;
                if features.len() == centroid.len() {
                    sum += self.calculate_distance(features, &centroid)?;
                    count += 1;
                }
            }
            sigma[ci] = if count == 0 { 0.0 } else { sum / count as f64 };
        }

        for ci in 0..k {
            let mut max_ratio = 0.0f64;
            let centroid_i = &clusters[ci].centroid;
            if centroid_i.is_empty() {
                clusters[ci].quality_metrics.davies_bouldin_index = 0.0;
                continue;
            }
            for cj in 0..k {
                if cj == ci {
                    continue;
                }
                let centroid_j = &clusters[cj].centroid;
                if centroid_j.is_empty() || centroid_i.len() != centroid_j.len() {
                    continue;
                }
                let d_ij = self.calculate_distance(centroid_i, centroid_j)?;
                if d_ij < 1e-12 {
                    // Coincident centroids: treat as the worst case to penalise.
                    max_ratio = f64::INFINITY;
                    break;
                }
                let ratio = (sigma[ci] + sigma[cj]) / d_ij;
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
            clusters[ci].quality_metrics.davies_bouldin_index = if max_ratio.is_finite() {
                max_ratio
            } else {
                0.0
            };
        }

        // ---- Calinski-Harabasz index ----
        // BSS = sum_i n_i * d(c_i, overall_centroid)^2
        // WSS = sum_i sum_x in C_i d(x, c_i)^2  (== sum of inertias)
        // CH  = (BSS / (k - 1)) / (WSS / (n - k))
        let n_total: usize = clusters.iter().map(|c| c.solutions.len()).sum();
        let dim = clusters
            .iter()
            .find(|c| !c.centroid.is_empty())
            .map(|c| c.centroid.len())
            .unwrap_or(0);

        let ch_value = if dim == 0 || n_total <= k {
            0.0
        } else {
            // Overall centroid is the size-weighted average of cluster centroids.
            let mut overall = vec![0.0f64; dim];
            for c in clusters.iter() {
                if c.centroid.len() == dim {
                    let w = c.solutions.len() as f64;
                    for d in 0..dim {
                        overall[d] += c.centroid[d] * w;
                    }
                }
            }
            if n_total > 0 {
                for d in 0..dim {
                    overall[d] /= n_total as f64;
                }
            }

            let mut bss = 0.0f64;
            for c in clusters.iter() {
                if c.centroid.len() != dim {
                    continue;
                }
                let dist = self.calculate_distance(&c.centroid, &overall)?;
                bss += (c.solutions.len() as f64) * dist * dist;
            }

            let wss: f64 = clusters.iter().map(|c| c.quality_metrics.inertia).sum();

            let denom_top = (k as f64 - 1.0).max(1.0);
            let denom_bot = (n_total as f64 - k as f64).max(1.0);
            if wss < 1e-12 {
                0.0
            } else {
                (bss / denom_top) / (wss / denom_bot)
            }
        };

        for c in clusters.iter_mut() {
            c.quality_metrics.calinski_harabasz_index = ch_value;
        }

        Ok(())
    }
}
