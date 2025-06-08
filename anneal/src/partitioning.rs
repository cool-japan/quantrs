//! Graph partitioning algorithms for quantum annealing
//!
//! This module implements spectral and other graph partitioning methods
//! to decompose large QUBO problems into smaller subproblems that can
//! fit on quantum annealing hardware.

use std::collections::{HashMap, HashSet, VecDeque};
use num_complex::Complex64;
use crate::ising::{IsingError, IsingResult};

/// Represents a partition of variables into groups
#[derive(Debug, Clone)]
pub struct Partition {
    /// Maps variable index to partition group
    pub assignment: HashMap<usize, usize>,
    /// Number of partitions
    pub num_partitions: usize,
    /// Quality metric (e.g., edge cut)
    pub quality: f64,
}

impl Partition {
    /// Create a new partition
    pub fn new(num_partitions: usize) -> Self {
        Self {
            assignment: HashMap::new(),
            num_partitions,
            quality: 0.0,
        }
    }

    /// Get variables in a specific partition
    pub fn get_partition(&self, partition_id: usize) -> Vec<usize> {
        self.assignment
            .iter()
            .filter_map(|(&var, &part)| {
                if part == partition_id {
                    Some(var)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Calculate edge cut for given edges
    pub fn calculate_edge_cut(&mut self, edges: &[(usize, usize, f64)]) -> f64 {
        let mut cut_weight = 0.0;
        
        for &(u, v, weight) in edges {
            if let (Some(&p1), Some(&p2)) = (self.assignment.get(&u), self.assignment.get(&v)) {
                if p1 != p2 {
                    cut_weight += weight.abs();
                }
            }
        }
        
        self.quality = cut_weight;
        cut_weight
    }
}

/// Spectral partitioning using eigendecomposition
#[derive(Clone, Debug)]
pub struct SpectralPartitioner {
    /// Number of eigenvectors to use
    pub num_eigenvectors: usize,
    /// Maximum iterations for eigensolvers
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for SpectralPartitioner {
    fn default() -> Self {
        Self {
            num_eigenvectors: 2,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

impl SpectralPartitioner {
    /// Partition a graph using spectral methods
    pub fn partition_graph(
        &self,
        num_vars: usize,
        edges: &[(usize, usize, f64)],
        num_partitions: usize,
    ) -> IsingResult<Partition> {
        if num_partitions < 2 {
            return Err(IsingError::InvalidValue(
                "Number of partitions must be at least 2".to_string()
            ));
        }

        // Build graph Laplacian
        let laplacian = build_laplacian(num_vars, edges);
        
        // Compute eigenvectors using power iteration (simplified approach)
        let eigenvectors = compute_eigenvectors_power_iteration(
            &laplacian,
            self.num_eigenvectors.min(num_partitions),
            self.max_iterations,
            self.tolerance,
        )?;
        
        // Use k-means clustering on eigenvectors to assign partitions
        let mut partition = Partition::new(num_partitions);
        
        if num_partitions == 2 {
            // For bipartition, use median split on Fiedler vector for balanced partitions
            if eigenvectors.len() > 1 {
                let fiedler = &eigenvectors[1]; // Second smallest eigenvalue's eigenvector
                
                // Create indices sorted by Fiedler vector values
                let mut sorted_indices: Vec<(usize, f64)> = fiedler.iter()
                    .enumerate()
                    .map(|(i, &v)| (i, v))
                    .collect();
                sorted_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                
                // Assign first half to partition 0, second half to partition 1
                for i in 0..sorted_indices.len() / 2 {
                    partition.assignment.insert(sorted_indices[i].0, 0);
                }
                for i in sorted_indices.len() / 2..sorted_indices.len() {
                    partition.assignment.insert(sorted_indices[i].0, 1);
                }
            } else {
                // Fallback: simple split
                for var in 0..num_vars {
                    partition.assignment.insert(var, if var < num_vars / 2 { 0 } else { 1 });
                }
            }
        } else {
            // For k-way partition, use k-means on eigenvector coordinates
            let assignments = kmeans_clustering(&eigenvectors, num_vars, num_partitions);
            
            for (var, &cluster) in assignments.iter().enumerate() {
                partition.assignment.insert(var, cluster);
            }
        }
        
        // Calculate partition quality
        partition.calculate_edge_cut(edges);
        
        Ok(partition)
    }
}

/// Build graph Laplacian matrix
fn build_laplacian(num_vars: usize, edges: &[(usize, usize, f64)]) -> Vec<Vec<f64>> {
    let mut laplacian = vec![vec![0.0; num_vars]; num_vars];
    
    // Build adjacency and degree
    for &(u, v, weight) in edges {
        if u != v && u < num_vars && v < num_vars {
            let w = weight.abs();
            laplacian[u][v] -= w;
            laplacian[v][u] -= w;
            laplacian[u][u] += w;
            laplacian[v][v] += w;
        }
    }
    
    laplacian
}

/// Compute eigenvectors using power iteration (simplified)
fn compute_eigenvectors_power_iteration(
    laplacian: &[Vec<f64>],
    num_eigenvectors: usize,
    max_iterations: usize,
    tolerance: f64,
) -> IsingResult<Vec<Vec<f64>>> {
    let n = laplacian.len();
    let mut eigenvectors = Vec::new();
    
    // First eigenvector is constant (for connected graphs)
    let first = vec![1.0 / (n as f64).sqrt(); n];
    eigenvectors.push(first);
    
    // Compute remaining eigenvectors using deflation
    for k in 1..num_eigenvectors.min(n) {
        let mut v = vec![0.0; n];
        
        // Initialize with random values
        for i in 0..n {
            v[i] = ((i + k) as f64).sin();
        }
        
        // Orthogonalize against previous eigenvectors
        for prev in &eigenvectors {
            let dot = dot_product(&v, prev);
            for i in 0..n {
                v[i] -= dot * prev[i];
            }
        }
        
        // Normalize
        normalize_vector(&mut v);
        
        // Power iteration with deflation
        for _ in 0..max_iterations {
            let mut v_new = matrix_vector_multiply(laplacian, &v);
            
            // Apply inverse iteration (simplified - just negate for smallest eigenvalues)
            for i in 0..n {
                v_new[i] = -v_new[i];
            }
            
            // Orthogonalize
            for prev in &eigenvectors {
                let dot = dot_product(&v_new, prev);
                for i in 0..n {
                    v_new[i] -= dot * prev[i];
                }
            }
            
            // Normalize
            normalize_vector(&mut v_new);
            
            // Check convergence
            let mut diff = 0.0;
            for i in 0..n {
                diff += (v_new[i] - v[i]).abs();
            }
            
            v = v_new;
            
            if diff < tolerance {
                break;
            }
        }
        
        eigenvectors.push(v);
    }
    
    Ok(eigenvectors)
}

/// Simple k-means clustering
fn kmeans_clustering(
    eigenvectors: &[Vec<f64>],
    num_points: usize,
    k: usize,
) -> Vec<usize> {
    let num_features = eigenvectors.len();
    let mut assignments = vec![0; num_points];
    let mut centroids = vec![vec![0.0; num_features]; k];
    
    // Initialize centroids
    for i in 0..k {
        let point_idx = (i * num_points) / k;
        for j in 0..num_features {
            centroids[i][j] = eigenvectors[j][point_idx];
        }
    }
    
    // K-means iterations
    for _ in 0..100 {
        let old_assignments = assignments.clone();
        
        // Assign points to nearest centroid
        for point in 0..num_points {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;
            
            for cluster in 0..k {
                let mut dist = 0.0;
                for feature in 0..num_features {
                    let diff = eigenvectors[feature][point] - centroids[cluster][feature];
                    dist += diff * diff;
                }
                
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = cluster;
                }
            }
            
            assignments[point] = best_cluster;
        }
        
        // Update centroids
        for cluster in 0..k {
            for feature in 0..num_features {
                centroids[cluster][feature] = 0.0;
            }
            
            let mut count = 0;
            for (point, &assigned) in assignments.iter().enumerate() {
                if assigned == cluster {
                    for feature in 0..num_features {
                        centroids[cluster][feature] += eigenvectors[feature][point];
                    }
                    count += 1;
                }
            }
            
            if count > 0 {
                for feature in 0..num_features {
                    centroids[cluster][feature] /= count as f64;
                }
            }
        }
        
        // Check convergence
        if assignments == old_assignments {
            break;
        }
    }
    
    assignments
}

/// Helper functions
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalize_vector(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn matrix_vector_multiply(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix.iter()
        .map(|row| dot_product(row, vector))
        .collect()
}

/// Kernighan-Lin graph partitioning algorithm
#[derive(Clone, Debug)]
pub struct KernighanLinPartitioner {
    /// Maximum number of improvement iterations
    pub max_iterations: usize,
    /// Random seed for initialization
    pub seed: Option<u64>,
}

impl Default for KernighanLinPartitioner {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            seed: None,
        }
    }
}

impl KernighanLinPartitioner {
    /// Partition graph into two parts using Kernighan-Lin algorithm
    pub fn bipartition(
        &self,
        num_vars: usize,
        edges: &[(usize, usize, f64)],
    ) -> IsingResult<Partition> {
        // Build adjacency lists
        let mut adj: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
        for i in 0..num_vars {
            adj.insert(i, Vec::new());
        }
        
        for &(u, v, weight) in edges {
            if u != v {
                adj.get_mut(&u).unwrap().push((v, weight));
                adj.get_mut(&v).unwrap().push((u, weight));
            }
        }
        
        // Initialize partition (balanced)
        let mut partition = Partition::new(2);
        for i in 0..num_vars {
            partition.assignment.insert(i, if i < num_vars / 2 { 0 } else { 1 });
        }
        
        // Kernighan-Lin iterations
        let mut improved = true;
        let mut iteration = 0;
        
        while improved && iteration < self.max_iterations {
            improved = false;
            let mut gains = Vec::new();
            let mut swapped = HashSet::new();
            
            // Calculate gains for all possible swaps
            for u in 0..num_vars {
                if swapped.contains(&u) {
                    continue;
                }
                
                let u_part = partition.assignment[&u];
                
                for v in (u + 1)..num_vars {
                    if swapped.contains(&v) {
                        continue;
                    }
                    
                    let v_part = partition.assignment[&v];
                    
                    if u_part != v_part {
                        // Calculate gain from swapping u and v
                        let gain = calculate_swap_gain(&adj, &partition.assignment, u, v);
                        gains.push((gain, u, v));
                    }
                }
            }
            
            // Sort by gain (descending)
            gains.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            
            // Apply best swaps
            let mut cumulative_gain = 0.0;
            let mut best_gain = 0.0;
            let mut best_swaps = Vec::new();
            let mut temp_swaps = Vec::new();
            
            for (gain, u, v) in gains {
                if swapped.contains(&u) || swapped.contains(&v) {
                    continue;
                }
                
                cumulative_gain += gain;
                temp_swaps.push((u, v));
                swapped.insert(u);
                swapped.insert(v);
                
                if cumulative_gain > best_gain {
                    best_gain = cumulative_gain;
                    best_swaps = temp_swaps.clone();
                }
                
                // Apply swap temporarily
                let u_part = partition.assignment[&u];
                let v_part = partition.assignment[&v];
                partition.assignment.insert(u, v_part);
                partition.assignment.insert(v, u_part);
            }
            
            // Revert to best configuration
            for (u, v) in &temp_swaps {
                if !best_swaps.contains(&(*u, *v)) {
                    let u_part = partition.assignment[u];
                    let v_part = partition.assignment[v];
                    partition.assignment.insert(*u, v_part);
                    partition.assignment.insert(*v, u_part);
                }
            }
            
            if best_gain > 1e-6 {
                improved = true;
            }
            
            iteration += 1;
        }
        
        // Calculate final quality
        partition.calculate_edge_cut(edges);
        
        Ok(partition)
    }
}

/// Calculate gain from swapping two nodes
fn calculate_swap_gain(
    adj: &HashMap<usize, Vec<(usize, f64)>>,
    assignment: &HashMap<usize, usize>,
    u: usize,
    v: usize,
) -> f64 {
    let u_part = assignment[&u];
    let v_part = assignment[&v];
    
    let mut gain = 0.0;
    
    // Calculate current cut contribution
    for &(neighbor, weight) in &adj[&u] {
        let n_part = assignment[&neighbor];
        if n_part == u_part {
            gain -= weight; // Internal edge becomes external
        } else {
            gain += weight; // External edge becomes internal
        }
    }
    
    for &(neighbor, weight) in &adj[&v] {
        let n_part = assignment[&neighbor];
        if n_part == v_part {
            gain -= weight; // Internal edge becomes external
        } else {
            gain += weight; // External edge becomes internal
        }
    }
    
    // Adjust for direct edge between u and v
    if let Some(&(_, weight)) = adj[&u].iter().find(|(n, _)| *n == v) {
        gain += 2.0 * weight; // This edge changes twice
    }
    
    gain
}

/// Recursive bisection partitioner
pub struct RecursiveBisectionPartitioner {
    /// Base partitioner for bisection
    pub bisection_method: BipartitionMethod,
    /// Balance constraint (ratio of partition sizes)
    pub balance_ratio: f64,
}

#[derive(Clone)]
pub enum BipartitionMethod {
    Spectral(SpectralPartitioner),
    KernighanLin(KernighanLinPartitioner),
}

impl Default for RecursiveBisectionPartitioner {
    fn default() -> Self {
        Self {
            bisection_method: BipartitionMethod::Spectral(SpectralPartitioner::default()),
            balance_ratio: 1.1, // Allow 10% imbalance
        }
    }
}

impl RecursiveBisectionPartitioner {
    /// Partition graph into k parts using recursive bisection
    pub fn partition(
        &self,
        num_vars: usize,
        edges: &[(usize, usize, f64)],
        num_partitions: usize,
    ) -> IsingResult<Partition> {
        if num_partitions == 1 {
            let mut partition = Partition::new(1);
            for i in 0..num_vars {
                partition.assignment.insert(i, 0);
            }
            return Ok(partition);
        }
        
        // Start with all variables in one partition
        let mut current_partitions = vec![HashSet::new(); 1];
        for i in 0..num_vars {
            current_partitions[0].insert(i);
        }
        
        // Recursively bisect until we have enough partitions
        while current_partitions.len() < num_partitions {
            // Find largest partition to split
            let (largest_idx, _) = current_partitions
                .iter()
                .enumerate()
                .max_by_key(|(_, p)| p.len())
                .unwrap();
            
            let to_split = current_partitions.remove(largest_idx);
            
            // Extract subgraph
            let subgraph_vars: Vec<usize> = to_split.iter().copied().collect();
            let var_map: HashMap<usize, usize> = subgraph_vars
                .iter()
                .enumerate()
                .map(|(new, &old)| (old, new))
                .collect();
            
            let subgraph_edges: Vec<(usize, usize, f64)> = edges
                .iter()
                .filter_map(|&(u, v, w)| {
                    if let (Some(&new_u), Some(&new_v)) = (var_map.get(&u), var_map.get(&v)) {
                        Some((new_u, new_v, w))
                    } else {
                        None
                    }
                })
                .collect();
            
            // Bisect the subgraph
            let bipartition = match &self.bisection_method {
                BipartitionMethod::Spectral(sp) => {
                    sp.partition_graph(subgraph_vars.len(), &subgraph_edges, 2)?
                }
                BipartitionMethod::KernighanLin(kl) => {
                    kl.bipartition(subgraph_vars.len(), &subgraph_edges)?
                }
            };
            
            // Create two new partitions
            let mut part0 = HashSet::new();
            let mut part1 = HashSet::new();
            
            for (i, &original_var) in subgraph_vars.iter().enumerate() {
                if bipartition.assignment[&i] == 0 {
                    part0.insert(original_var);
                } else {
                    part1.insert(original_var);
                }
            }
            
            current_partitions.push(part0);
            current_partitions.push(part1);
        }
        
        // Build final partition assignment
        let mut partition = Partition::new(num_partitions);
        for (part_id, part_vars) in current_partitions.iter().enumerate() {
            for &var in part_vars {
                partition.assignment.insert(var, part_id);
            }
        }
        
        // Calculate quality
        partition.calculate_edge_cut(edges);
        
        Ok(partition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_bipartition() {
        // Create a simple graph: 0-1-2-3 (path)
        let edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
        ];
        
        let partitioner = SpectralPartitioner::default();
        let partition = partitioner.partition_graph(4, &edges, 2).unwrap();
        
        // Check that we have a valid bipartition
        assert_eq!(partition.num_partitions, 2);
        assert_eq!(partition.assignment.len(), 4);
        
        // The optimal cut should separate the path in the middle
        let cut = partition.quality;
        println!("Spectral partition cut: {}", cut);
        println!("Partition assignment: {:?}", partition.assignment);
        // For a path graph, any bipartition will cut at least one edge
        assert!(cut >= 1.0);
    }

    #[test]
    fn test_kernighan_lin() {
        // Create a graph with two clear clusters
        let edges = vec![
            // Cluster 1
            (0, 1, 2.0),
            (0, 2, 2.0),
            (1, 2, 2.0),
            // Cluster 2
            (3, 4, 2.0),
            (3, 5, 2.0),
            (4, 5, 2.0),
            // Weak connection between clusters
            (2, 3, 0.5),
        ];
        
        let partitioner = KernighanLinPartitioner::default();
        let partition = partitioner.bipartition(6, &edges).unwrap();
        
        // Should find the natural clustering
        assert_eq!(partition.num_partitions, 2);
        assert_eq!(partition.assignment.len(), 6);
        
        // The cut should be small (ideally just the weak edge)
        assert!(partition.quality < 1.0);
    }

    #[test]
    fn test_recursive_bisection() {
        // Create a grid graph
        let mut edges = Vec::new();
        let n = 4; // 4x4 grid
        
        for i in 0..n {
            for j in 0..n {
                let node = i * n + j;
                // Right neighbor
                if j < n - 1 {
                    edges.push((node, node + 1, 1.0));
                }
                // Bottom neighbor
                if i < n - 1 {
                    edges.push((node, node + n, 1.0));
                }
            }
        }
        
        let partitioner = RecursiveBisectionPartitioner::default();
        let partition = partitioner.partition(n * n, &edges, 4).unwrap();
        
        // Should create 4 partitions
        assert_eq!(partition.num_partitions, 4);
        assert_eq!(partition.assignment.len(), n * n);
        
        // Each partition should have roughly equal size
        let mut sizes = vec![0; 4];
        for &part in partition.assignment.values() {
            sizes[part] += 1;
        }
        
        println!("Partition sizes: {:?}", sizes);
        for size in sizes {
            assert!(size >= 2 && size <= 6); // Allow more imbalance for small graphs
        }
    }
}