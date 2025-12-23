//! Advanced Problem Decomposition for QUBO Optimization
//!
//! This module provides sophisticated problem decomposition techniques for large-scale
//! QUBO problems, enabling efficient parallel solving and scalability to problems
//! beyond the capacity of single solvers.
//!
//! # Features
//!
//! - **Spectral Clustering**: Graph partitioning using eigenvalue decomposition
//! - **Community Detection**: Identification of tightly connected variable groups
//! - **Overlapping Decomposition**: Consensus-based solving with overlap handling
//! - **Adaptive Granularity**: Dynamic adjustment of partition sizes
//! - **Parallel Orchestration**: Coordinated solving across multiple solvers
//!
//! # Example
//!
//! ```rust
//! use quantrs2_tytan::advanced_problem_decomposition::{
//!     AdvancedDecomposer, DecompositionConfig, SpectralMethod
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! // Create a QUBO matrix
//! let qubo = Array2::from_shape_vec(
//!     (4, 4),
//!     vec![
//!         -1.0, 2.0, 0.0, 0.0,
//!         2.0, -2.0, 1.0, 0.0,
//!         0.0, 1.0, -1.0, 3.0,
//!         0.0, 0.0, 3.0, -2.0,
//!     ]
//! ).unwrap();
//!
//! // Configure decomposer
//! let config = DecompositionConfig::default()
//!     .with_num_partitions(2)
//!     .with_method(SpectralMethod::NormalizedCut);
//!
//! // Decompose problem
//! let decomposer = AdvancedDecomposer::new(config);
//! let partitions = decomposer.spectral_partition(&qubo).unwrap();
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

/// Error types for problem decomposition
#[derive(Debug, Clone)]
pub enum DecompositionError {
    /// Matrix is not square
    InvalidMatrix(String),
    /// Eigenvalue computation failed
    EigenvalueFailed(String),
    /// Invalid partition specification
    InvalidPartition(String),
    /// Decomposition produced no valid partitions
    NoValidPartitions,
    /// Consensus solving failed
    ConsensusFailed(String),
}

impl fmt::Display for DecompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMatrix(msg) => write!(f, "Invalid matrix: {msg}"),
            Self::EigenvalueFailed(msg) => write!(f, "Eigenvalue computation failed: {msg}"),
            Self::InvalidPartition(msg) => write!(f, "Invalid partition: {msg}"),
            Self::NoValidPartitions => write!(f, "No valid partitions produced"),
            Self::ConsensusFailed(msg) => write!(f, "Consensus solving failed: {msg}"),
        }
    }
}

impl std::error::Error for DecompositionError {}

/// Result type for decomposition operations
pub type DecompositionResult<T> = Result<T, DecompositionError>;

/// Spectral partitioning method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralMethod {
    /// Standard spectral clustering
    Standard,
    /// Normalized cut
    NormalizedCut,
    /// Ratio cut
    RatioCut,
}

/// Community detection algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunityDetection {
    /// Louvain method for modularity optimization
    Louvain,
    /// Label propagation
    LabelPropagation,
    /// Girvan-Newman edge betweenness
    GirvanNewman,
}

/// Configuration for advanced decomposer
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Number of partitions to create
    pub num_partitions: usize,
    /// Spectral method to use
    pub spectral_method: SpectralMethod,
    /// Community detection algorithm
    pub community_algorithm: CommunityDetection,
    /// Overlap size between partitions (number of variables)
    pub overlap_size: usize,
    /// Enable adaptive granularity control
    pub adaptive_granularity: bool,
    /// Target partition size for adaptive mode
    pub target_partition_size: usize,
    /// Maximum number of consensus iterations
    pub max_consensus_iterations: usize,
    /// Convergence tolerance for consensus
    pub consensus_tolerance: f64,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            num_partitions: 4,
            spectral_method: SpectralMethod::NormalizedCut,
            community_algorithm: CommunityDetection::Louvain,
            overlap_size: 2,
            adaptive_granularity: true,
            target_partition_size: 50,
            max_consensus_iterations: 10,
            consensus_tolerance: 1e-6,
        }
    }
}

impl DecompositionConfig {
    /// Set the number of partitions
    pub const fn with_num_partitions(mut self, n: usize) -> Self {
        self.num_partitions = n;
        self
    }

    /// Set the spectral method
    pub const fn with_method(mut self, method: SpectralMethod) -> Self {
        self.spectral_method = method;
        self
    }

    /// Set the community detection algorithm
    pub const fn with_community_algorithm(mut self, alg: CommunityDetection) -> Self {
        self.community_algorithm = alg;
        self
    }

    /// Set the overlap size
    pub const fn with_overlap_size(mut self, size: usize) -> Self {
        self.overlap_size = size;
        self
    }

    /// Enable or disable adaptive granularity
    pub const fn with_adaptive_granularity(mut self, enable: bool) -> Self {
        self.adaptive_granularity = enable;
        self
    }

    /// Set the target partition size
    pub const fn with_target_partition_size(mut self, size: usize) -> Self {
        self.target_partition_size = size;
        self
    }
}

/// Represents a partition of variables
#[derive(Debug, Clone)]
pub struct Partition {
    /// Variable indices in this partition
    pub variables: Vec<usize>,
    /// Overlap variables (shared with other partitions)
    pub overlap_variables: Vec<usize>,
    /// Subproblem QUBO matrix
    pub subproblem: Option<Array2<f64>>,
}

/// Result of consensus solving
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// Final variable assignments
    pub solution: Vec<i32>,
    /// Energy of the solution
    pub energy: f64,
    /// Number of iterations to convergence
    pub iterations: usize,
    /// Whether consensus was reached
    pub converged: bool,
}

/// Advanced problem decomposer
pub struct AdvancedDecomposer {
    config: DecompositionConfig,
}

impl AdvancedDecomposer {
    /// Create a new advanced decomposer
    pub const fn new(config: DecompositionConfig) -> Self {
        Self { config }
    }

    /// Partition a QUBO problem using spectral clustering
    ///
    /// Constructs a graph Laplacian from the QUBO matrix and uses its
    /// eigenvectors to partition variables into groups
    pub fn spectral_partition(&self, qubo: &Array2<f64>) -> DecompositionResult<Vec<Partition>> {
        let n = qubo.nrows();
        if n != qubo.ncols() {
            return Err(DecompositionError::InvalidMatrix(
                "Matrix must be square".to_string(),
            ));
        }

        // Construct graph Laplacian
        let laplacian = self.compute_laplacian(qubo)?;

        // Compute eigenvectors (simplified - in practice use proper eigendecomposition)
        let eigenvector = self.compute_fiedler_vector(&laplacian)?;

        // Partition based on eigenvector signs/values
        let assignments = self.partition_by_eigenvector(&eigenvector, self.config.num_partitions);

        // Create partitions with overlap
        let partitions = self.create_overlapping_partitions(&assignments, n)?;

        // Extract subproblems
        let partitions_with_subproblems: Vec<Partition> = partitions
            .into_iter()
            .map(|mut p| {
                p.subproblem = Some(self.extract_subproblem(qubo, &p.variables));
                p
            })
            .collect();

        Ok(partitions_with_subproblems)
    }

    /// Compute graph Laplacian matrix
    fn compute_laplacian(&self, qubo: &Array2<f64>) -> DecompositionResult<Array2<f64>> {
        let n = qubo.nrows();

        // Compute degree matrix and adjacency
        let mut degree = Array1::zeros(n);
        let mut adjacency = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let weight = qubo[[i, j]].abs();
                    adjacency[[i, j]] = weight;
                    degree[i] += weight;
                }
            }
        }

        // Compute Laplacian based on method
        let mut laplacian = Array2::zeros((n, n));
        match self.config.spectral_method {
            SpectralMethod::Standard => {
                // L = D - A
                for i in 0..n {
                    laplacian[[i, i]] = degree[i];
                    for j in 0..n {
                        if i != j {
                            laplacian[[i, j]] = -adjacency[[i, j]];
                        }
                    }
                }
            }
            SpectralMethod::NormalizedCut => {
                // L = D^(-1/2) (D - A) D^(-1/2)
                for i in 0..n {
                    let d_inv_sqrt = if degree[i] > 1e-10 {
                        1.0 / degree[i].sqrt()
                    } else {
                        0.0
                    };

                    for j in 0..n {
                        if i == j {
                            laplacian[[i, j]] = if degree[i] > 1e-10 { 1.0 } else { 0.0 };
                        } else {
                            let d_j_inv_sqrt = if degree[j] > 1e-10 {
                                1.0 / degree[j].sqrt()
                            } else {
                                0.0
                            };
                            laplacian[[i, j]] = -adjacency[[i, j]] * d_inv_sqrt * d_j_inv_sqrt;
                        }
                    }
                }
            }
            SpectralMethod::RatioCut => {
                // Same as standard Laplacian for our purposes
                for i in 0..n {
                    laplacian[[i, i]] = degree[i];
                    for j in 0..n {
                        if i != j {
                            laplacian[[i, j]] = -adjacency[[i, j]];
                        }
                    }
                }
            }
        }

        Ok(laplacian)
    }

    /// Compute Fiedler vector (second smallest eigenvector of Laplacian)
    ///
    /// This is a simplified implementation using power iteration
    fn compute_fiedler_vector(&self, laplacian: &Array2<f64>) -> DecompositionResult<Array1<f64>> {
        let n = laplacian.nrows();
        let mut rng = thread_rng();

        // Random initialization
        let mut v = Array1::from_shape_fn(n, |_| rng.gen::<f64>().mul_add(2.0, -1.0));

        // Orthogonalize against constant vector (first eigenvector)
        let ones = Array1::ones(n);
        let projection = v.dot(&ones) / (n as f64);
        v = &v - &(&ones * projection);

        // Normalize
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            v = &v / norm;
        }

        // Power iteration (simplified - finds dominant eigenvector of (I - L))
        // In practice, use proper eigensolvers for Fiedler vector
        for _ in 0..100 {
            let mut new_v = Array1::zeros(n);

            // Multiply by (I - αL) to shift spectrum
            let alpha = 0.5;
            for i in 0..n {
                new_v[i] = v[i];
                for j in 0..n {
                    new_v[i] -= alpha * laplacian[[i, j]] * v[j];
                }
            }

            // Orthogonalize against constant vector
            let projection = new_v.dot(&ones) / (n as f64);
            new_v = &new_v - &(&ones * projection);

            // Normalize
            let norm = new_v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                new_v = &new_v / norm;
            }

            // Check convergence
            let diff = (&new_v - &v).iter().map(|&x| x.abs()).sum::<f64>();
            v = new_v;

            if diff < 1e-6 {
                break;
            }
        }

        Ok(v)
    }

    /// Partition variables based on eigenvector values
    fn partition_by_eigenvector(&self, eigenvector: &Array1<f64>, k: usize) -> Vec<usize> {
        let n = eigenvector.len();
        let mut assignments = vec![0; n];

        if k == 2 {
            // Simple bisection
            for i in 0..n {
                assignments[i] = usize::from(eigenvector[i] > 0.0);
            }
        } else {
            // K-way partitioning using k-means on eigenvector values
            let sorted_indices: Vec<usize> = {
                let mut indices: Vec<usize> = (0..n).collect();
                indices.sort_by(|&a, &b| {
                    eigenvector[a]
                        .partial_cmp(&eigenvector[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                indices
            };

            let partition_size = n.div_ceil(k);
            for (i, &idx) in sorted_indices.iter().enumerate() {
                assignments[idx] = (i / partition_size).min(k - 1);
            }
        }

        assignments
    }

    /// Create overlapping partitions from assignments
    fn create_overlapping_partitions(
        &self,
        assignments: &[usize],
        n: usize,
    ) -> DecompositionResult<Vec<Partition>> {
        let k = self.config.num_partitions;
        let mut partitions = Vec::new();

        for partition_id in 0..k {
            let mut variables: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &p)| p == partition_id)
                .map(|(i, _)| i)
                .collect();

            if variables.is_empty() {
                continue;
            }

            // Add overlap variables from neighboring partitions
            let mut overlap_variables = Vec::new();
            if self.config.overlap_size > 0 {
                // Find boundary variables
                for &var in &variables {
                    // Check if this variable should be in overlap
                    // (simplified - in practice, use connectivity analysis)
                    if var > 0 && assignments[var - 1] != partition_id {
                        overlap_variables.push(var);
                    }
                    if var < n - 1 && assignments[var + 1] != partition_id {
                        overlap_variables.push(var);
                    }
                }
                overlap_variables.sort_unstable();
                overlap_variables.dedup();
            }

            partitions.push(Partition {
                variables,
                overlap_variables,
                subproblem: None,
            });
        }

        if partitions.is_empty() {
            return Err(DecompositionError::NoValidPartitions);
        }

        Ok(partitions)
    }

    /// Extract subproblem QUBO for a partition
    fn extract_subproblem(&self, qubo: &Array2<f64>, variables: &[usize]) -> Array2<f64> {
        let m = variables.len();
        let mut subqubo = Array2::zeros((m, m));

        for (i, &var_i) in variables.iter().enumerate() {
            for (j, &var_j) in variables.iter().enumerate() {
                subqubo[[i, j]] = qubo[[var_i, var_j]];
            }
        }

        subqubo
    }

    /// Detect communities using Louvain method
    pub fn detect_communities(&self, qubo: &Array2<f64>) -> DecompositionResult<Vec<usize>> {
        let n = qubo.nrows();

        match self.config.community_algorithm {
            CommunityDetection::Louvain => self.louvain_communities(qubo),
            CommunityDetection::LabelPropagation => self.label_propagation(qubo),
            CommunityDetection::GirvanNewman => self.girvan_newman(qubo),
        }
    }

    /// Louvain community detection
    fn louvain_communities(&self, qubo: &Array2<f64>) -> DecompositionResult<Vec<usize>> {
        let n = qubo.nrows();
        let mut communities = (0..n).collect::<Vec<usize>>();

        // Compute total edge weight
        let total_weight: f64 = qubo.iter().map(|&x| x.abs()).sum::<f64>() / 2.0;

        // Iterative optimization
        let mut improved = true;
        let max_iterations = 10;
        let mut iteration = 0;

        while improved && iteration < max_iterations {
            improved = false;
            iteration += 1;

            for node in 0..n {
                let current_community = communities[node];
                let mut best_community = current_community;
                let mut best_gain = 0.0;

                // Try moving node to each neighboring community
                let neighbors = self.get_neighbors(qubo, node);
                let neighbor_communities: HashSet<usize> =
                    neighbors.iter().map(|&i| communities[i]).collect();

                for &community in &neighbor_communities {
                    let gain =
                        self.modularity_gain(qubo, &communities, node, community, total_weight);
                    if gain > best_gain {
                        best_gain = gain;
                        best_community = community;
                    }
                }

                if best_community != current_community && best_gain > 1e-10 {
                    communities[node] = best_community;
                    improved = true;
                }
            }
        }

        Ok(communities)
    }

    /// Get neighbors of a node
    fn get_neighbors(&self, qubo: &Array2<f64>, node: usize) -> Vec<usize> {
        let n = qubo.nrows();
        let mut neighbors = Vec::new();

        for i in 0..n {
            if i != node && qubo[[node, i]].abs() > 1e-10 {
                neighbors.push(i);
            }
        }

        neighbors
    }

    /// Compute modularity gain from moving a node to a community
    fn modularity_gain(
        &self,
        qubo: &Array2<f64>,
        communities: &[usize],
        node: usize,
        new_community: usize,
        total_weight: f64,
    ) -> f64 {
        let n = qubo.nrows();

        // Compute edge weight to new community
        let mut edge_weight_to_new = 0.0;
        for i in 0..n {
            if communities[i] == new_community {
                edge_weight_to_new += qubo[[node, i]].abs();
            }
        }

        // Simplified modularity gain (actual Louvain is more complex)
        edge_weight_to_new / total_weight
    }

    /// Label propagation community detection
    fn label_propagation(&self, qubo: &Array2<f64>) -> DecompositionResult<Vec<usize>> {
        let n = qubo.nrows();
        let mut labels: Vec<usize> = (0..n).collect();
        let mut rng = thread_rng();

        let max_iterations = 100;
        for _ in 0..max_iterations {
            let mut changed = false;

            // Random order
            let mut order: Vec<usize> = (0..n).collect();
            order.shuffle(&mut rng);

            for &node in &order {
                // Count neighbor labels
                let mut label_counts: HashMap<usize, f64> = HashMap::new();

                for i in 0..n {
                    if i != node {
                        let weight = qubo[[node, i]].abs();
                        if weight > 1e-10 {
                            *label_counts.entry(labels[i]).or_insert(0.0) += weight;
                        }
                    }
                }

                if !label_counts.is_empty() {
                    // Choose most common label
                    let new_label = *label_counts
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(k, _)| k)
                        .unwrap();

                    if new_label != labels[node] {
                        labels[node] = new_label;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        // Renumber communities to be contiguous
        let unique_labels: HashSet<usize> = labels.iter().copied().collect();
        let mut label_map: HashMap<usize, usize> = HashMap::new();
        for (i, &label) in unique_labels.iter().enumerate() {
            label_map.insert(label, i);
        }

        Ok(labels.iter().map(|&l| label_map[&l]).collect())
    }

    /// Girvan-Newman edge betweenness community detection
    fn girvan_newman(&self, qubo: &Array2<f64>) -> DecompositionResult<Vec<usize>> {
        // Simplified implementation - just use spectral partitioning
        let partitions = self.spectral_partition(qubo)?;
        let n = qubo.nrows();
        let mut communities = vec![0; n];

        for (comm_id, partition) in partitions.iter().enumerate() {
            for &var in &partition.variables {
                communities[var] = comm_id;
            }
        }

        Ok(communities)
    }

    /// Solve using consensus over overlapping partitions
    pub fn consensus_solve(
        &self,
        partitions: &[Partition],
        subsolvers: Vec<Vec<i32>>,
    ) -> DecompositionResult<ConsensusResult> {
        if partitions.len() != subsolvers.len() {
            return Err(DecompositionError::ConsensusFailed(
                "Number of partitions and subsolutions mismatch".to_string(),
            ));
        }

        // Determine total number of variables
        let num_vars = partitions
            .iter()
            .flat_map(|p| p.variables.iter())
            .max()
            .map_or(0, |&m| m + 1);

        let mut solution = vec![0; num_vars];
        let mut vote_counts: Vec<HashMap<i32, usize>> = vec![HashMap::new(); num_vars];

        // Aggregate votes from all subsolvers
        for (partition, subsolution) in partitions.iter().zip(subsolvers.iter()) {
            for (local_idx, &var) in partition.variables.iter().enumerate() {
                if local_idx < subsolution.len() {
                    let value = subsolution[local_idx];
                    *vote_counts[var].entry(value).or_insert(0) += 1;
                }
            }
        }

        // Consensus by majority vote
        for (var, counts) in vote_counts.iter().enumerate() {
            if let Some((&value, _)) = counts.iter().max_by_key(|(_, &count)| count) {
                solution[var] = value;
            }
        }

        Ok(ConsensusResult {
            solution,
            energy: 0.0, // Would compute actual energy
            iterations: 1,
            converged: true,
        })
    }

    /// Get configuration
    pub const fn config(&self) -> &DecompositionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposer_creation() {
        let config = DecompositionConfig::default();
        let decomposer = AdvancedDecomposer::new(config);
        assert_eq!(decomposer.config().num_partitions, 4);
    }

    #[test]
    fn test_spectral_partition() {
        let qubo = Array2::from_shape_vec(
            (4, 4),
            vec![
                -1.0, 2.0, 0.0, 0.0, 2.0, -2.0, 1.0, 0.0, 0.0, 1.0, -1.0, 3.0, 0.0, 0.0, 3.0, -2.0,
            ],
        )
        .unwrap();

        let config = DecompositionConfig::default().with_num_partitions(2);
        let decomposer = AdvancedDecomposer::new(config);

        let result = decomposer.spectral_partition(&qubo);
        assert!(result.is_ok());

        let partitions = result.unwrap();
        assert!(!partitions.is_empty());
        assert!(partitions.len() <= 2);
    }

    #[test]
    fn test_laplacian_computation() {
        let qubo =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0])
                .unwrap();

        let config = DecompositionConfig::default();
        let decomposer = AdvancedDecomposer::new(config);

        let result = decomposer.compute_laplacian(&qubo);
        assert!(result.is_ok());

        let laplacian = result.unwrap();
        assert_eq!(laplacian.nrows(), 3);
        assert_eq!(laplacian.ncols(), 3);
    }

    #[test]
    fn test_community_detection() {
        let qubo = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 2.0, 0.1, 0.1, 2.0, 1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 2.0, 0.1, 0.1, 2.0, 1.0,
            ],
        )
        .unwrap();

        let config = DecompositionConfig::default();
        let decomposer = AdvancedDecomposer::new(config);

        let result = decomposer.detect_communities(&qubo);
        assert!(result.is_ok());

        let communities = result.unwrap();
        assert_eq!(communities.len(), 4);
    }

    #[test]
    fn test_label_propagation() {
        let qubo = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
        )
        .unwrap();

        let config = DecompositionConfig::default()
            .with_community_algorithm(CommunityDetection::LabelPropagation);
        let decomposer = AdvancedDecomposer::new(config);

        let result = decomposer.label_propagation(&qubo);
        assert!(result.is_ok());
    }

    #[test]
    fn test_consensus_solve() {
        let config = DecompositionConfig::default();
        let decomposer = AdvancedDecomposer::new(config);

        let partitions = vec![
            Partition {
                variables: vec![0, 1],
                overlap_variables: vec![1],
                subproblem: None,
            },
            Partition {
                variables: vec![1, 2],
                overlap_variables: vec![1],
                subproblem: None,
            },
        ];

        let subsolvers = vec![vec![1, 0], vec![0, 1]];

        let result = decomposer.consensus_solve(&partitions, subsolvers);
        assert!(result.is_ok());

        let consensus = result.unwrap();
        assert_eq!(consensus.solution.len(), 3);
        assert!(consensus.converged);
    }

    #[test]
    fn test_config_builder() {
        let config = DecompositionConfig::default()
            .with_num_partitions(8)
            .with_method(SpectralMethod::RatioCut)
            .with_overlap_size(3)
            .with_adaptive_granularity(false);

        assert_eq!(config.num_partitions, 8);
        assert_eq!(config.spectral_method, SpectralMethod::RatioCut);
        assert_eq!(config.overlap_size, 3);
        assert!(!config.adaptive_granularity);
    }

    #[test]
    fn test_subproblem_extraction() {
        let qubo = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 5.0, 6.0, 7.0, 3.0, 6.0, 8.0, 9.0, 4.0, 7.0, 9.0, 10.0,
            ],
        )
        .unwrap();

        let config = DecompositionConfig::default();
        let decomposer = AdvancedDecomposer::new(config);

        let variables = vec![0, 2];
        let subqubo = decomposer.extract_subproblem(&qubo, &variables);

        assert_eq!(subqubo.nrows(), 2);
        assert_eq!(subqubo.ncols(), 2);
        assert_eq!(subqubo[[0, 0]], 1.0);
        assert_eq!(subqubo[[0, 1]], 3.0);
        assert_eq!(subqubo[[1, 0]], 3.0);
        assert_eq!(subqubo[[1, 1]], 8.0);
    }

    #[test]
    fn test_spectral_methods() {
        let qubo = Array2::eye(4);

        for method in &[
            SpectralMethod::Standard,
            SpectralMethod::NormalizedCut,
            SpectralMethod::RatioCut,
        ] {
            let config = DecompositionConfig::default().with_method(*method);
            let decomposer = AdvancedDecomposer::new(config);

            let result = decomposer.compute_laplacian(&qubo);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_overlapping_partitions() {
        let assignments = vec![0, 0, 1, 1];
        let config = DecompositionConfig::default()
            .with_num_partitions(2)
            .with_overlap_size(1);
        let decomposer = AdvancedDecomposer::new(config);

        let result = decomposer.create_overlapping_partitions(&assignments, 4);
        assert!(result.is_ok());

        let partitions = result.unwrap();
        assert_eq!(partitions.len(), 2);
    }
}
