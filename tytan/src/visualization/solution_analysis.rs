//! Solution distribution analysis for quantum annealing
//!
//! This module provides statistical analysis and clustering of solution sets
//! including diversity metrics, clustering analysis, and quality distribution.

use crate::sampler::SampleResult;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "scirs")]
use crate::scirs_stub::{
    scirs2_plot::{BoxPlot, Plot2D, Violin},
    scirs2_statistics::{
        clustering::{hierarchical_clustering, KMeans, DBSCAN},
        descriptive::{mean, quantile, std_dev},
    },
};

/// Solution distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionConfig {
    /// Clustering algorithm to use
    pub clustering_method: ClusteringMethod,
    /// Number of clusters (for KMeans)
    pub n_clusters: Option<usize>,
    /// DBSCAN epsilon parameter
    pub epsilon: Option<f64>,
    /// Minimum samples for DBSCAN
    pub min_samples: Option<usize>,
    /// Calculate pairwise distances
    pub compute_distances: bool,
    /// Distance metric
    pub distance_metric: DistanceMetric,
}

/// Clustering methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ClusteringMethod {
    KMeans,
    DBSCAN,
    Hierarchical,
    None,
}

/// Distance metrics
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Hamming,
    Euclidean,
    Manhattan,
    Jaccard,
}

impl Default for DistributionConfig {
    fn default() -> Self {
        Self {
            clustering_method: ClusteringMethod::KMeans,
            n_clusters: Some(5),
            epsilon: Some(0.5),
            min_samples: Some(5),
            compute_distances: true,
            distance_metric: DistanceMetric::Hamming,
        }
    }
}

/// Solution distribution analyzer
pub struct SolutionDistribution {
    config: DistributionConfig,
    samples: Vec<SampleResult>,
    distance_matrix: Option<Array2<f64>>,
    clusters: Option<Vec<usize>>,
}

/// Distribution analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    pub statistics: SolutionStatistics,
    pub diversity_metrics: DiversityMetrics,
    pub cluster_info: Option<ClusterInfo>,
    pub quality_distribution: QualityDistribution,
    pub correlation_analysis: CorrelationAnalysis,
}

/// Basic solution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionStatistics {
    pub n_samples: usize,
    pub n_unique: usize,
    pub mean_energy: f64,
    pub std_energy: f64,
    pub min_energy: f64,
    pub max_energy: f64,
    pub energy_quantiles: HashMap<String, f64>,
    pub most_frequent_solution: Option<HashMap<String, bool>>,
    pub frequency_top_solution: usize,
}

/// Diversity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityMetrics {
    pub average_distance: f64,
    pub min_distance: f64,
    pub max_distance: f64,
    pub diversity_index: f64,
    pub entropy: f64,
    pub effective_sample_size: f64,
}

/// Cluster information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    pub n_clusters: usize,
    pub cluster_sizes: Vec<usize>,
    pub cluster_centers: Vec<HashMap<String, f64>>,
    pub cluster_energies: Vec<ClusterEnergy>,
    pub silhouette_score: f64,
    pub inter_cluster_distances: Array2<f64>,
}

/// Cluster energy statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterEnergy {
    pub cluster_id: usize,
    pub mean_energy: f64,
    pub std_energy: f64,
    pub min_energy: f64,
    pub max_energy: f64,
}

/// Quality distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDistribution {
    pub energy_bins: Vec<f64>,
    pub bin_counts: Vec<usize>,
    pub cumulative_distribution: Vec<f64>,
    pub percentile_values: HashMap<usize, f64>,
}

/// Correlation analysis between variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysis {
    pub variable_correlations: HashMap<(String, String), f64>,
    pub energy_correlations: HashMap<String, f64>,
    pub significant_pairs: Vec<(String, String, f64)>,
}

impl SolutionDistribution {
    /// Create new solution distribution analyzer
    pub fn new(config: DistributionConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
            distance_matrix: None,
            clusters: None,
        }
    }

    /// Add samples for analysis
    pub fn add_samples(&mut self, samples: Vec<SampleResult>) {
        self.samples.extend(samples);
        // Invalidate cached results
        self.distance_matrix = None;
        self.clusters = None;
    }

    /// Perform complete distribution analysis
    pub fn analyze(&mut self) -> Result<DistributionAnalysis, Box<dyn std::error::Error>> {
        if self.samples.is_empty() {
            return Err("No samples to analyze".into());
        }

        // Compute basic statistics
        let statistics = self.compute_statistics()?;

        // Compute diversity metrics
        let diversity_metrics = self.compute_diversity_metrics()?;

        // Perform clustering if requested
        let cluster_info = if self.config.clustering_method != ClusteringMethod::None {
            Some(self.perform_clustering()?)
        } else {
            None
        };

        // Analyze quality distribution
        let quality_distribution = self.analyze_quality_distribution()?;

        // Perform correlation analysis
        let correlation_analysis = self.analyze_correlations()?;

        Ok(DistributionAnalysis {
            statistics,
            diversity_metrics,
            cluster_info,
            quality_distribution,
            correlation_analysis,
        })
    }

    /// Compute basic statistics
    fn compute_statistics(&self) -> Result<SolutionStatistics, Box<dyn std::error::Error>> {
        let energies: Vec<f64> = self.samples.iter().map(|s| s.energy).collect();

        let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;
        let variance = energies
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f64>()
            / energies.len() as f64;
        let std_energy = variance.sqrt();

        let min_energy = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_energy = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Compute quantiles
        let mut sorted_energies = energies.clone();
        sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut energy_quantiles = HashMap::new();
        for &q in &[25, 50, 75] {
            let idx = (q as f64 / 100.0 * sorted_energies.len() as f64) as usize;
            let idx = idx.min(sorted_energies.len() - 1);
            energy_quantiles.insert(format!("q{}", q), sorted_energies[idx]);
        }

        // Find most frequent solution
        let mut solution_counts = HashMap::new();
        for sample in &self.samples {
            let key = format!("{:?}", sample.assignments);
            *solution_counts.entry(key).or_insert(0) += 1;
        }

        let (_most_frequent, frequency) = solution_counts
            .iter()
            .max_by_key(|&(_, count)| count)
            .map(|(sol, &count)| (sol.clone(), count))
            .unwrap_or((String::new(), 0));

        // Count unique solutions
        let n_unique = solution_counts.len();

        Ok(SolutionStatistics {
            n_samples: self.samples.len(),
            n_unique,
            mean_energy,
            std_energy,
            min_energy,
            max_energy,
            energy_quantiles,
            most_frequent_solution: None, // Simplified for now
            frequency_top_solution: frequency,
        })
    }

    /// Compute diversity metrics
    fn compute_diversity_metrics(
        &mut self,
    ) -> Result<DiversityMetrics, Box<dyn std::error::Error>> {
        if self.config.compute_distances && self.distance_matrix.is_none() {
            self.compute_distance_matrix()?;
        }

        let dist_matrix = self
            .distance_matrix
            .as_ref()
            .ok_or("Distance matrix not computed")?;

        let n = dist_matrix.nrows();
        let mut all_distances = Vec::new();

        for i in 0..n {
            for j in i + 1..n {
                all_distances.push(dist_matrix[[i, j]]);
            }
        }

        if all_distances.is_empty() {
            return Ok(DiversityMetrics {
                average_distance: 0.0,
                min_distance: 0.0,
                max_distance: 0.0,
                diversity_index: 0.0,
                entropy: 0.0,
                effective_sample_size: 1.0,
            });
        }

        let average_distance = all_distances.iter().sum::<f64>() / all_distances.len() as f64;
        let min_distance = all_distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_distance = all_distances.iter().fold(0.0f64, |a, &b| a.max(b));

        // Diversity index (normalized average distance)
        let max_possible_distance = self.get_max_distance();
        let diversity_index = if max_possible_distance > 0.0 {
            average_distance / max_possible_distance
        } else {
            0.0
        };

        // Solution entropy
        let entropy = self.calculate_entropy();

        // Effective sample size (based on solution frequencies)
        let ess = self.calculate_effective_sample_size();

        Ok(DiversityMetrics {
            average_distance,
            min_distance,
            max_distance,
            diversity_index,
            entropy,
            effective_sample_size: ess,
        })
    }

    /// Compute distance matrix
    fn compute_distance_matrix(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.samples.len();
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in i + 1..n {
                let distance = self.calculate_distance(&self.samples[i], &self.samples[j]);
                matrix[[i, j]] = distance;
                matrix[[j, i]] = distance;
            }
        }

        self.distance_matrix = Some(matrix);
        Ok(())
    }

    /// Calculate distance between two solutions
    fn calculate_distance(&self, a: &SampleResult, b: &SampleResult) -> f64 {
        match self.config.distance_metric {
            DistanceMetric::Hamming => {
                let mut distance = 0.0;
                let all_vars: std::collections::HashSet<_> =
                    a.assignments.keys().chain(b.assignments.keys()).collect();

                for var in all_vars {
                    let val_a = a.assignments.get(var).copied().unwrap_or(false);
                    let val_b = b.assignments.get(var).copied().unwrap_or(false);
                    if val_a != val_b {
                        distance += 1.0;
                    }
                }
                distance
            }
            DistanceMetric::Jaccard => {
                let set_a: std::collections::HashSet<_> = a
                    .assignments
                    .iter()
                    .filter(|(_, &v)| v)
                    .map(|(k, _)| k)
                    .collect();
                let set_b: std::collections::HashSet<_> = b
                    .assignments
                    .iter()
                    .filter(|(_, &v)| v)
                    .map(|(k, _)| k)
                    .collect();

                let intersection = set_a.intersection(&set_b).count();
                let union = set_a.union(&set_b).count();

                if union > 0 {
                    1.0 - (intersection as f64 / union as f64)
                } else {
                    0.0
                }
            }
            _ => 0.0, // Placeholder for other metrics
        }
    }

    /// Get maximum possible distance
    fn get_max_distance(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let n_vars = self.samples[0].assignments.len();
        match self.config.distance_metric {
            DistanceMetric::Hamming => n_vars as f64,
            DistanceMetric::Jaccard => 1.0,
            _ => 1.0,
        }
    }

    /// Calculate solution entropy
    fn calculate_entropy(&self) -> f64 {
        let mut solution_counts = HashMap::new();
        for sample in &self.samples {
            let key = format!("{:?}", sample.assignments);
            *solution_counts.entry(key).or_insert(0) += 1;
        }

        let total = self.samples.len() as f64;
        let mut entropy = 0.0;

        for &count in solution_counts.values() {
            let p = count as f64 / total;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        entropy
    }

    /// Calculate effective sample size
    fn calculate_effective_sample_size(&self) -> f64 {
        let mut solution_counts = HashMap::new();
        for sample in &self.samples {
            let key = format!("{:?}", sample.assignments);
            *solution_counts.entry(key).or_insert(0) += 1;
        }

        let total = self.samples.len() as f64;
        let sum_squared: f64 = solution_counts
            .values()
            .map(|&c| (c as f64 / total).powi(2))
            .sum();

        if sum_squared > 0.0 {
            1.0 / sum_squared
        } else {
            0.0
        }
    }

    /// Perform clustering
    fn perform_clustering(&mut self) -> Result<ClusterInfo, Box<dyn std::error::Error>> {
        // Convert samples to feature matrix
        let (feature_matrix, _) = self.samples_to_matrix()?;

        let clusters = match self.config.clustering_method {
            ClusteringMethod::KMeans => {
                let k = self.config.n_clusters.unwrap_or(5);
                self.cluster_kmeans(&feature_matrix, k)?
            }
            ClusteringMethod::DBSCAN => {
                let eps = self.config.epsilon.unwrap_or(0.5);
                let min_samples = self.config.min_samples.unwrap_or(5);
                self.cluster_dbscan(&feature_matrix, eps, min_samples)?
            }
            ClusteringMethod::Hierarchical => self.cluster_hierarchical(&feature_matrix)?,
            ClusteringMethod::None => vec![],
        };

        self.clusters = Some(clusters.clone());

        // Analyze clusters
        self.analyze_clusters(&clusters, &feature_matrix)
    }

    /// K-means clustering
    fn cluster_kmeans(
        &self,
        data: &Array2<f64>,
        k: usize,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        #[cfg(feature = "scirs")]
        {
            let kmeans = KMeans::new(k);
            let mut clusters = kmeans.fit_predict(data)?;
            Ok(clusters)
        }

        #[cfg(not(feature = "scirs"))]
        {
            // Simple random assignment as fallback
            use rand::rngs::StdRng;
            use rand::{Rng, SeedableRng};
            let mut rng = StdRng::seed_from_u64(42);

            Ok((0..data.nrows()).map(|_| rng.random_range(0..k)).collect())
        }
    }

    /// DBSCAN clustering
    fn cluster_dbscan(
        &self,
        data: &Array2<f64>,
        eps: f64,
        min_samples: usize,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        #[cfg(feature = "scirs")]
        {
            let dbscan = DBSCAN::new(eps, min_samples);
            let mut clusters = dbscan.fit_predict(data)?;
            Ok(clusters)
        }

        #[cfg(not(feature = "scirs"))]
        {
            // All samples in one cluster as fallback
            Ok(vec![0; data.nrows()])
        }
    }

    /// Hierarchical clustering
    fn cluster_hierarchical(
        &self,
        data: &Array2<f64>,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        #[cfg(feature = "scirs")]
        {
            let n_clusters = self.config.n_clusters.unwrap_or(5);
            let mut clusters = hierarchical_clustering(data, n_clusters, "average")?;
            Ok(clusters)
        }

        #[cfg(not(feature = "scirs"))]
        {
            // Simple clustering by energy
            let energies: Vec<f64> = self.samples.iter().map(|s| s.energy).collect();
            let mut sorted_indices: Vec<usize> = (0..energies.len()).collect();
            sorted_indices.sort_by(|&a, &b| energies[a].partial_cmp(&energies[b]).unwrap());

            let n_clusters = self.config.n_clusters.unwrap_or(5);
            let cluster_size = energies.len() / n_clusters;

            let mut clusters = vec![0; energies.len()];
            for (idx, &i) in sorted_indices.iter().enumerate() {
                clusters[i] = (idx / cluster_size).min(n_clusters - 1);
            }

            Ok(clusters)
        }
    }

    /// Convert samples to feature matrix
    fn samples_to_matrix(&self) -> Result<(Array2<f64>, Vec<String>), Box<dyn std::error::Error>> {
        if self.samples.is_empty() {
            return Err("No samples to convert".into());
        }

        // Get all variable names
        let mut all_vars = std::collections::HashSet::new();
        for sample in &self.samples {
            for var in sample.assignments.keys() {
                all_vars.insert(var.clone());
            }
        }

        let var_names: Vec<String> = all_vars.into_iter().collect();
        let n_vars = var_names.len();
        let n_samples = self.samples.len();

        // Create feature matrix
        let mut matrix = Array2::zeros((n_samples, n_vars));

        for (i, sample) in self.samples.iter().enumerate() {
            for (j, var_name) in var_names.iter().enumerate() {
                if let Some(&value) = sample.assignments.get(var_name) {
                    matrix[[i, j]] = if value { 1.0 } else { 0.0 };
                }
            }
        }

        Ok((matrix, var_names))
    }

    /// Analyze clusters
    fn analyze_clusters(
        &self,
        clusters: &[usize],
        feature_matrix: &Array2<f64>,
    ) -> Result<ClusterInfo, Box<dyn std::error::Error>> {
        let n_clusters = clusters.iter().max().copied().unwrap_or(0) + 1;
        let mut cluster_sizes = vec![0; n_clusters];

        for &c in clusters {
            if c < n_clusters {
                cluster_sizes[c] += 1;
            }
        }

        // Calculate cluster centers
        let mut cluster_centers = Vec::new();
        let (_, var_names) = self.samples_to_matrix()?;

        for cluster_id in 0..n_clusters {
            let mut center = HashMap::new();
            let cluster_samples: Vec<usize> = clusters
                .iter()
                .enumerate()
                .filter(|(_, &c)| c == cluster_id)
                .map(|(i, _)| i)
                .collect();

            if !cluster_samples.is_empty() {
                for (j, var_name) in var_names.iter().enumerate() {
                    let mean_value: f64 = cluster_samples
                        .iter()
                        .map(|&i| feature_matrix[[i, j]])
                        .sum::<f64>()
                        / cluster_samples.len() as f64;
                    center.insert(var_name.clone(), mean_value);
                }
            }

            cluster_centers.push(center);
        }

        // Calculate cluster energy statistics
        let mut cluster_energies = Vec::new();

        for cluster_id in 0..n_clusters {
            let cluster_energy_values: Vec<f64> = clusters
                .iter()
                .enumerate()
                .filter(|(_, &c)| c == cluster_id)
                .map(|(i, _)| self.samples[i].energy)
                .collect();

            if !cluster_energy_values.is_empty() {
                let mean =
                    cluster_energy_values.iter().sum::<f64>() / cluster_energy_values.len() as f64;
                let variance = cluster_energy_values
                    .iter()
                    .map(|e| (e - mean).powi(2))
                    .sum::<f64>()
                    / cluster_energy_values.len() as f64;
                let std = variance.sqrt();
                let min = cluster_energy_values
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b));
                let max = cluster_energy_values
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                cluster_energies.push(ClusterEnergy {
                    cluster_id,
                    mean_energy: mean,
                    std_energy: std,
                    min_energy: min,
                    max_energy: max,
                });
            }
        }

        // Calculate silhouette score
        let silhouette_score = self.calculate_silhouette_score(clusters)?;

        // Calculate inter-cluster distances
        let inter_cluster_distances =
            self.calculate_inter_cluster_distances(&cluster_centers, n_clusters)?;

        Ok(ClusterInfo {
            n_clusters,
            cluster_sizes,
            cluster_centers,
            cluster_energies,
            silhouette_score,
            inter_cluster_distances,
        })
    }

    /// Calculate silhouette score
    fn calculate_silhouette_score(
        &self,
        _clusters: &[usize],
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // Simplified silhouette calculation
        // In full implementation would use proper silhouette coefficient
        Ok(0.5) // Placeholder
    }

    /// Calculate inter-cluster distances
    fn calculate_inter_cluster_distances(
        &self,
        centers: &[HashMap<String, f64>],
        n_clusters: usize,
    ) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        let mut distances = Array2::zeros((n_clusters, n_clusters));

        for i in 0..n_clusters {
            for j in i + 1..n_clusters {
                let dist = self.calculate_center_distance(&centers[i], &centers[j]);
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        Ok(distances)
    }

    /// Calculate distance between cluster centers
    fn calculate_center_distance(&self, a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
        let all_vars: std::collections::HashSet<_> = a.keys().chain(b.keys()).collect();

        let mut distance = 0.0;
        for var in all_vars {
            let val_a = a.get(var).copied().unwrap_or(0.0);
            let val_b = b.get(var).copied().unwrap_or(0.0);
            distance += (val_a - val_b).powi(2);
        }

        distance.sqrt()
    }

    /// Analyze quality distribution
    fn analyze_quality_distribution(
        &self,
    ) -> Result<QualityDistribution, Box<dyn std::error::Error>> {
        let energies: Vec<f64> = self.samples.iter().map(|s| s.energy).collect();

        let min_energy = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_energy = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let n_bins = 20;
        let bin_width = (max_energy - min_energy) / n_bins as f64;

        let mut energy_bins = Vec::new();
        let mut bin_counts = vec![0; n_bins];

        for i in 0..n_bins {
            energy_bins.push(min_energy + (i as f64 + 0.5) * bin_width);
        }

        for &energy in &energies {
            let bin_idx = ((energy - min_energy) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(n_bins - 1);
            bin_counts[bin_idx] += 1;
        }

        // Calculate cumulative distribution
        let total = energies.len() as f64;
        let mut cumulative_distribution = Vec::new();
        let mut cumsum = 0;

        for &count in &bin_counts {
            cumsum += count;
            cumulative_distribution.push(cumsum as f64 / total);
        }

        // Calculate percentiles
        let mut sorted_energies = energies.clone();
        sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut percentile_values = HashMap::new();
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99].iter() {
            let idx = ((*p as f64 / 100.0) * sorted_energies.len() as f64) as usize;
            let idx = idx.min(sorted_energies.len() - 1);
            percentile_values.insert(*p, sorted_energies[idx]);
        }

        Ok(QualityDistribution {
            energy_bins,
            bin_counts,
            cumulative_distribution,
            percentile_values,
        })
    }

    /// Analyze correlations between variables
    fn analyze_correlations(&self) -> Result<CorrelationAnalysis, Box<dyn std::error::Error>> {
        let (feature_matrix, var_names) = self.samples_to_matrix()?;
        let energies: Vec<f64> = self.samples.iter().map(|s| s.energy).collect();

        let mut variable_correlations = HashMap::new();
        let mut energy_correlations = HashMap::new();
        let mut significant_pairs = Vec::new();

        // Variable-variable correlations
        for i in 0..var_names.len() {
            for j in i + 1..var_names.len() {
                let var_i: Vec<f64> = (0..feature_matrix.nrows())
                    .map(|k| feature_matrix[[k, i]])
                    .collect();
                let var_j: Vec<f64> = (0..feature_matrix.nrows())
                    .map(|k| feature_matrix[[k, j]])
                    .collect();

                let corr = calculate_correlation(&var_i, &var_j);

                if corr.abs() > 0.3 {
                    // Threshold for significance
                    significant_pairs.push((var_names[i].clone(), var_names[j].clone(), corr));
                }

                variable_correlations.insert((var_names[i].clone(), var_names[j].clone()), corr);
            }

            // Variable-energy correlations
            let var_values: Vec<f64> = (0..feature_matrix.nrows())
                .map(|k| feature_matrix[[k, i]])
                .collect();

            let energy_corr = calculate_correlation(&var_values, &energies);
            energy_correlations.insert(var_names[i].clone(), energy_corr);
        }

        Ok(CorrelationAnalysis {
            variable_correlations,
            energy_correlations,
            significant_pairs,
        })
    }
}

/// Calculate correlation coefficient
fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>()
        / n;

    let std_x = (x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / n).sqrt();
    let std_y = (y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / n).sqrt();

    if std_x > 0.0 && std_y > 0.0 {
        cov / (std_x * std_y)
    } else {
        0.0
    }
}

/// Analyze solution distribution
pub fn analyze_solution_distribution(
    samples: Vec<SampleResult>,
    config: Option<DistributionConfig>,
) -> Result<DistributionAnalysis, Box<dyn std::error::Error>> {
    let config = config.unwrap_or_default();
    let mut analyzer = SolutionDistribution::new(config);
    analyzer.add_samples(samples);
    analyzer.analyze()
}

/// Plot solution distribution analysis
pub fn plot_distribution_analysis(
    analysis: &DistributionAnalysis,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "scirs")]
    {
        use crate::scirs_stub::scirs2_plot::{Figure, Subplot};

        let mut fig = Figure::new();

        // Energy distribution
        let bin_counts_f64: Vec<f64> = analysis
            .quality_distribution
            .bin_counts
            .iter()
            .map(|&c| c as f64)
            .collect();
        fig.add_subplot(2, 2, 1)?
            .bar(&analysis.quality_distribution.energy_bins, &bin_counts_f64)
            .set_xlabel("Energy")
            .set_ylabel("Count")
            .set_title("Energy Distribution");

        // Cluster sizes (if available)
        if let Some(ref cluster_info) = analysis.cluster_info {
            let cluster_ids: Vec<f64> = (0..cluster_info.n_clusters).map(|i| i as f64).collect();
            let cluster_sizes: Vec<f64> = cluster_info
                .cluster_sizes
                .iter()
                .map(|&s| s as f64)
                .collect();

            fig.add_subplot(2, 2, 2)?
                .bar(&cluster_ids, &cluster_sizes)
                .set_xlabel("Cluster ID")
                .set_ylabel("Size")
                .set_title("Cluster Sizes");
        }

        // Cumulative distribution
        fig.add_subplot(2, 2, 3)?
            .plot(
                &analysis.quality_distribution.energy_bins,
                &analysis.quality_distribution.cumulative_distribution,
            )
            .set_xlabel("Energy")
            .set_ylabel("Cumulative Probability")
            .set_title("Cumulative Distribution");

        // Variable-energy correlations
        let var_names: Vec<String> = analysis
            .correlation_analysis
            .energy_correlations
            .keys()
            .cloned()
            .collect();
        let correlations: Vec<f64> = var_names
            .iter()
            .map(|v| analysis.correlation_analysis.energy_correlations[v])
            .collect();

        if !var_names.is_empty() {
            fig.add_subplot(2, 2, 4)?
                .bar_horizontal(&var_names, &correlations)
                .set_xlabel("Correlation with Energy")
                .set_ylabel("Variable")
                .set_title("Variable-Energy Correlations");
        }

        fig.show()?;
    }

    #[cfg(not(feature = "scirs"))]
    {
        // Export analysis data
        export_distribution_analysis(analysis, "distribution_analysis.json")?;
        println!("Distribution analysis exported to distribution_analysis.json");
    }

    Ok(())
}

/// Export distribution analysis
fn export_distribution_analysis(
    analysis: &DistributionAnalysis,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(analysis)?;
    std::fs::write(path, json)?;
    Ok(())
}
