//! Problem decomposition methods for large-scale optimization.
//!
//! This module provides various decomposition strategies including
//! graph partitioning, hierarchical solving, domain decomposition,
//! and parallel subproblem solving.

#[cfg(feature = "dwave")]
use crate::compile::CompiledModel;
use crate::sampler::simulated_annealing::SASampler;
use crate::sampler::{SampleResult, Sampler, SamplerError, SamplerResult};
use ndarray::{Array, Array1, Array2, IxDyn};
use rand::prelude::*;
use rand::thread_rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

/// Automatic graph partitioner for QUBO problems
pub struct GraphPartitioner {
    /// Partitioning algorithm
    algorithm: PartitioningAlgorithm,
    /// Number of partitions
    num_partitions: usize,
    /// Balance constraint
    balance_factor: f64,
    /// Edge cut minimization weight
    edge_cut_weight: f64,
    /// Use multilevel partitioning
    use_multilevel: bool,
    /// Maximum recursion depth for multilevel algorithms
    max_recursion_depth: usize,
}

#[derive(Debug, Clone)]
pub enum PartitioningAlgorithm {
    /// Kernighan-Lin algorithm
    KernighanLin,
    /// Fiduccia-Mattheyses algorithm
    FiducciaMattheyses,
    /// Spectral partitioning
    Spectral,
    /// METIS-style multilevel
    Multilevel,
    /// Community detection
    CommunityDetection,
    /// Min-cut max-flow
    MinCutMaxFlow,
}

impl GraphPartitioner {
    /// Create new graph partitioner with default settings
    pub fn new() -> Self {
        Self {
            algorithm: PartitioningAlgorithm::Spectral,
            num_partitions: 2,
            balance_factor: 0.1,
            edge_cut_weight: 1.0,
            use_multilevel: true,
            max_recursion_depth: 10,
        }
    }

    /// Create new graph partitioner with specific settings
    pub fn with_config(algorithm: PartitioningAlgorithm, num_partitions: usize) -> Self {
        Self {
            algorithm,
            num_partitions,
            balance_factor: 0.1,
            edge_cut_weight: 1.0,
            use_multilevel: true,
            max_recursion_depth: 10,
        }
    }

    /// Set number of partitions
    pub fn with_num_partitions(mut self, num_partitions: usize) -> Self {
        self.num_partitions = num_partitions;
        self
    }

    /// Set partitioning algorithm
    pub fn with_algorithm(mut self, algorithm: PartitioningAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set balance factor
    pub fn with_balance_factor(mut self, factor: f64) -> Self {
        self.balance_factor = factor;
        self
    }

    /// Set edge cut weight
    pub fn with_edge_cut_weight(mut self, weight: f64) -> Self {
        self.edge_cut_weight = weight;
        self
    }

    /// Simple partition method that returns subproblems
    pub fn partition(&self, qubo: &Array2<f64>) -> Result<Vec<Subproblem>, String> {
        // Create a simple variable map
        let n = qubo.shape()[0];
        let mut var_map = HashMap::new();
        for i in 0..n {
            var_map.insert(format!("x{}", i), i);
        }

        let partitioning = self.partition_qubo(qubo, &var_map)?;
        Ok(partitioning.subproblems)
    }

    /// Partition QUBO problem
    pub fn partition_qubo(
        &self,
        qubo: &Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<Partitioning, String> {
        // Build graph from QUBO
        let graph = self.build_graph_from_qubo(qubo)?;

        // Apply partitioning algorithm
        let partition_assignment = match self.algorithm {
            PartitioningAlgorithm::KernighanLin => self.kernighan_lin_partition(&graph)?,
            PartitioningAlgorithm::Spectral => self.spectral_partition(&graph)?,
            PartitioningAlgorithm::Multilevel => self.multilevel_partition_with_depth(&graph, 0)?,
            _ => {
                // Default to spectral
                self.spectral_partition(&graph)?
            }
        };

        // Extract subproblems
        let subproblems = self.extract_subproblems(qubo, var_map, &partition_assignment)?;

        // Compute partition metrics
        let metrics = self.compute_partition_metrics(&graph, &partition_assignment);

        let coupling_terms = self.extract_coupling_terms(qubo, &partition_assignment)?;

        Ok(Partitioning {
            partition_assignment,
            subproblems,
            coupling_terms,
            metrics,
        })
    }

    /// Build graph from QUBO matrix
    fn build_graph_from_qubo(&self, qubo: &Array2<f64>) -> Result<Graph, String> {
        let n = qubo.shape()[0];
        let mut edges = Vec::new();
        let mut node_weights = vec![1.0; n];

        for i in 0..n {
            // Node weight from diagonal
            node_weights[i] = qubo[[i, i]].abs();

            for j in i + 1..n {
                if qubo[[i, j]].abs() > 1e-10 {
                    edges.push(Edge {
                        from: i,
                        to: j,
                        weight: qubo[[i, j]].abs(),
                    });
                }
            }
        }

        Ok(Graph {
            num_nodes: n,
            edges,
            node_weights,
        })
    }

    /// Kernighan-Lin partitioning
    fn kernighan_lin_partition(&self, graph: &Graph) -> Result<Vec<usize>, String> {
        let n = graph.num_nodes;
        let mut partition = vec![0; n];

        // Initialize random bisection
        let mut rng = thread_rng();
        for i in 0..n / 2 {
            partition[i] = 1;
        }
        partition.shuffle(&mut rng);

        // Iterative improvement with maximum iterations
        let max_iterations = 100; // Prevent infinite loops
        let min_gain_threshold = 1e-10; // Minimum meaningful gain

        for _iteration in 0..max_iterations {
            // Compute gains for all swaps
            let mut best_swap = None;
            let mut best_gain = 0.0;

            for i in 0..n {
                for j in i + 1..n {
                    if partition[i] != partition[j] {
                        let gain = self.compute_swap_gain(graph, &partition, i, j);
                        if gain > best_gain && gain > min_gain_threshold {
                            best_gain = gain;
                            best_swap = Some((i, j));
                        }
                    }
                }
            }

            // Apply best swap or terminate
            if let Some((i, j)) = best_swap {
                partition.swap(i, j);
            } else {
                break; // No beneficial swaps found
            }
        }

        // Extend to k-way partitioning
        if self.num_partitions > 2 {
            self.extend_to_kway(graph, partition)
        } else {
            Ok(partition)
        }
    }

    /// Compute gain from swapping two nodes
    fn compute_swap_gain(&self, graph: &Graph, partition: &[usize], i: usize, j: usize) -> f64 {
        let mut gain = 0.0;

        // External and internal costs
        for edge in &graph.edges {
            if edge.from == i || edge.to == i {
                let other = if edge.from == i { edge.to } else { edge.from };
                if partition[other] == partition[i] {
                    gain -= edge.weight; // Becomes external
                } else {
                    gain += edge.weight; // Becomes internal
                }
            }

            if edge.from == j || edge.to == j {
                let other = if edge.from == j { edge.to } else { edge.from };
                if partition[other] == partition[j] {
                    gain -= edge.weight; // Becomes external
                } else {
                    gain += edge.weight; // Becomes internal
                }
            }
        }

        gain
    }

    /// Spectral partitioning using eigenvectors
    fn spectral_partition(&self, graph: &Graph) -> Result<Vec<usize>, String> {
        // Build Laplacian matrix
        let laplacian = self.build_laplacian(graph)?;

        // Compute eigenvectors (simplified - use second smallest eigenvalue)
        let eigenvector = self.compute_fiedler_vector(&laplacian)?;

        // Partition based on eigenvector
        let mut partition = vec![0; graph.num_nodes];
        let median = self.find_median(&eigenvector);

        for (i, &value) in eigenvector.iter().enumerate() {
            partition[i] = if value > median { 1 } else { 0 };
        }

        // Refine with local search
        self.refine_partition(graph, partition)
    }

    /// Build graph Laplacian
    fn build_laplacian(&self, graph: &Graph) -> Result<Array2<f64>, String> {
        let n = graph.num_nodes;
        let mut laplacian = Array2::zeros((n, n));

        // Add edge weights (negative for adjacency)
        for edge in &graph.edges {
            laplacian[[edge.from, edge.to]] = -edge.weight;
            laplacian[[edge.to, edge.from]] = -edge.weight;
        }

        // Add degree on diagonal
        for i in 0..n {
            let degree: f64 = graph
                .edges
                .iter()
                .filter(|e| e.from == i || e.to == i)
                .map(|e| e.weight)
                .sum();
            laplacian[[i, i]] = degree;
        }

        Ok(laplacian)
    }

    /// Compute Fiedler vector (second smallest eigenvector)
    fn compute_fiedler_vector(&self, laplacian: &Array2<f64>) -> Result<Vec<f64>, String> {
        // Simplified: use power iteration
        let n = laplacian.shape()[0];
        let mut vector = vec![1.0; n];
        let mut rng = thread_rng();

        // Randomize initial vector
        for v in vector.iter_mut() {
            *v = rng.gen_range(-1.0..1.0);
        }

        // Power iteration (simplified)
        for _ in 0..100 {
            let mut new_vector = vec![0.0; n];

            for i in 0..n {
                for j in 0..n {
                    new_vector[i] += laplacian[[i, j]] * vector[j];
                }
            }

            // Normalize
            let norm: f64 = new_vector.iter().map(|x| x * x).sum::<f64>().sqrt();
            for v in new_vector.iter_mut() {
                *v /= norm;
            }

            vector = new_vector;
        }

        Ok(vector)
    }

    /// Find median value
    fn find_median(&self, values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    }

    /// Refine partition using local search
    fn refine_partition(
        &self,
        graph: &Graph,
        mut partition: Vec<usize>,
    ) -> Result<Vec<usize>, String> {
        let mut improved = true;

        while improved {
            improved = false;

            for i in 0..graph.num_nodes {
                let current_cut = self.compute_node_cut_cost(graph, &partition, i);

                // Try moving to other partition
                let old_part = partition[i];
                partition[i] = 1 - old_part;

                let new_cut = self.compute_node_cut_cost(graph, &partition, i);

                if new_cut < current_cut && self.is_balanced(&partition) {
                    improved = true;
                } else {
                    partition[i] = old_part;
                }
            }
        }

        Ok(partition)
    }

    /// Compute cut cost for a node
    fn compute_node_cut_cost(&self, graph: &Graph, partition: &[usize], node: usize) -> f64 {
        graph
            .edges
            .iter()
            .filter(|e| e.from == node || e.to == node)
            .map(|e| {
                let other = if e.from == node { e.to } else { e.from };
                if partition[node] != partition[other] {
                    e.weight
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Check if partition is balanced
    fn is_balanced(&self, partition: &[usize]) -> bool {
        let counts = self.count_partition_sizes(partition);
        let avg = partition.len() as f64 / counts.len() as f64;

        counts
            .iter()
            .all(|&count| ((count as f64 - avg).abs() / avg) <= self.balance_factor)
    }

    /// Count partition sizes
    fn count_partition_sizes(&self, partition: &[usize]) -> Vec<usize> {
        let max_part = *partition.iter().max().unwrap_or(&0);
        let mut counts = vec![0; max_part + 1];

        for &p in partition {
            counts[p] += 1;
        }

        counts
    }

    /// Multilevel partitioning
    fn multilevel_partition_with_depth(
        &self,
        graph: &Graph,
        depth: usize,
    ) -> Result<Vec<usize>, String> {
        // Check recursion depth limit
        if depth >= self.max_recursion_depth {
            return self.spectral_partition(graph);
        }

        // Coarsening phase
        let (coarse_graph, mapping) = self.coarsen_graph(graph)?;

        // Partition coarse graph
        let coarse_partition = if coarse_graph.num_nodes > 100 {
            self.multilevel_partition_with_depth(&coarse_graph, depth + 1)?
        } else {
            self.spectral_partition(&coarse_graph)?
        };

        // Uncoarsening and refinement
        self.uncoarsen_partition(graph, &coarse_partition, &mapping)
    }

    fn multilevel_partition(&self, graph: &Graph) -> Result<Vec<usize>, String> {
        self.multilevel_partition_with_depth(graph, 0)
    }

    /// Coarsen graph by matching
    fn coarsen_graph(&self, graph: &Graph) -> Result<(Graph, Vec<usize>), String> {
        let mut matched = vec![false; graph.num_nodes];
        let mut mapping = vec![0; graph.num_nodes];
        let mut coarse_nodes = 0;

        // Heavy edge matching
        let mut edges_sorted = graph.edges.clone();
        edges_sorted.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());

        for edge in edges_sorted {
            if !matched[edge.from] && !matched[edge.to] {
                matched[edge.from] = true;
                matched[edge.to] = true;
                mapping[edge.from] = coarse_nodes;
                mapping[edge.to] = coarse_nodes;
                coarse_nodes += 1;
            }
        }

        // Map unmatched nodes
        for i in 0..graph.num_nodes {
            if !matched[i] {
                mapping[i] = coarse_nodes;
                coarse_nodes += 1;
            }
        }

        // Build coarse graph
        let coarse_graph = self.build_coarse_graph(graph, &mapping, coarse_nodes)?;

        Ok((coarse_graph, mapping))
    }

    /// Build coarse graph from mapping
    fn build_coarse_graph(
        &self,
        graph: &Graph,
        mapping: &[usize],
        num_coarse_nodes: usize,
    ) -> Result<Graph, String> {
        let mut coarse_weights = vec![0.0; num_coarse_nodes];
        let mut coarse_edges: HashMap<(usize, usize), f64> = HashMap::new();

        // Aggregate node weights
        for (i, &coarse_id) in mapping.iter().enumerate() {
            coarse_weights[coarse_id] += graph.node_weights[i];
        }

        // Aggregate edge weights
        for edge in &graph.edges {
            let coarse_from = mapping[edge.from];
            let coarse_to = mapping[edge.to];

            if coarse_from != coarse_to {
                let key = if coarse_from < coarse_to {
                    (coarse_from, coarse_to)
                } else {
                    (coarse_to, coarse_from)
                };

                *coarse_edges.entry(key).or_insert(0.0) += edge.weight;
            }
        }

        let edges = coarse_edges
            .into_iter()
            .map(|((from, to), weight)| Edge { from, to, weight })
            .collect();

        Ok(Graph {
            num_nodes: num_coarse_nodes,
            edges,
            node_weights: coarse_weights,
        })
    }

    /// Uncoarsen partition
    fn uncoarsen_partition(
        &self,
        fine_graph: &Graph,
        coarse_partition: &[usize],
        mapping: &[usize],
    ) -> Result<Vec<usize>, String> {
        let mut fine_partition = vec![0; fine_graph.num_nodes];

        // Project partition
        for (i, &coarse_id) in mapping.iter().enumerate() {
            fine_partition[i] = coarse_partition[coarse_id];
        }

        // Refine
        self.refine_partition(fine_graph, fine_partition)
    }

    /// Extend bisection to k-way partition
    fn extend_to_kway(
        &self,
        graph: &Graph,
        mut partition: Vec<usize>,
    ) -> Result<Vec<usize>, String> {
        if self.num_partitions <= 2 {
            return Ok(partition);
        }

        // Recursive bisection
        for part in 0..self.num_partitions.ilog2() {
            let mut new_partition = partition.clone();

            // Bisect each existing partition
            for p in 0..(1 << part) {
                let nodes: Vec<_> = (0..graph.num_nodes)
                    .filter(|&i| partition[i] == p)
                    .collect();

                if nodes.len() > 1 {
                    // Create subgraph and partition
                    let subgraph = self.extract_subgraph(graph, &nodes)?;
                    let sub_partition = self.kernighan_lin_partition(&subgraph)?;

                    // Map back
                    for (i, &node) in nodes.iter().enumerate() {
                        if sub_partition[i] == 1 {
                            new_partition[node] = p + (1 << part);
                        }
                    }
                }
            }

            partition = new_partition;
        }

        Ok(partition)
    }

    /// Extract subgraph
    fn extract_subgraph(&self, graph: &Graph, nodes: &[usize]) -> Result<Graph, String> {
        let node_map: HashMap<usize, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();

        let edges = graph
            .edges
            .iter()
            .filter_map(|edge| {
                if let (Some(&from), Some(&to)) = (node_map.get(&edge.from), node_map.get(&edge.to))
                {
                    Some(Edge {
                        from,
                        to,
                        weight: edge.weight,
                    })
                } else {
                    None
                }
            })
            .collect();

        let node_weights = nodes.iter().map(|&i| graph.node_weights[i]).collect();

        Ok(Graph {
            num_nodes: nodes.len(),
            edges,
            node_weights,
        })
    }

    /// Extract subproblems from partitioning
    fn extract_subproblems(
        &self,
        qubo: &Array2<f64>,
        var_map: &HashMap<String, usize>,
        partition: &[usize],
    ) -> Result<Vec<Subproblem>, String> {
        let mut subproblems = Vec::new();
        let num_parts = *partition.iter().max().unwrap_or(&0) + 1;

        for p in 0..num_parts {
            // Get variables in this partition
            let part_vars: Vec<_> = (0..qubo.shape()[0])
                .filter(|&i| partition[i] == p)
                .collect();

            if part_vars.is_empty() {
                continue;
            }

            // Extract submatrix
            let size = part_vars.len();
            let mut sub_qubo = Array2::zeros((size, size));

            for (i, &vi) in part_vars.iter().enumerate() {
                for (j, &vj) in part_vars.iter().enumerate() {
                    sub_qubo[[i, j]] = qubo[[vi, vj]];
                }
            }

            // Create variable mapping
            let mut sub_var_map = HashMap::new();
            let reverse_map: HashMap<usize, String> =
                var_map.iter().map(|(k, v)| (*v, k.clone())).collect();

            for (i, &vi) in part_vars.iter().enumerate() {
                if let Some(var_name) = reverse_map.get(&vi) {
                    sub_var_map.insert(var_name.clone(), i);
                }
            }

            subproblems.push(Subproblem {
                id: p,
                qubo: sub_qubo,
                var_map: sub_var_map,
                original_indices: part_vars.clone(),
                variables: part_vars,
            });
        }

        Ok(subproblems)
    }

    /// Extract coupling terms between partitions
    fn extract_coupling_terms(
        &self,
        qubo: &Array2<f64>,
        partition: &[usize],
    ) -> Result<Vec<CouplingTerm>, String> {
        let mut coupling_terms = Vec::new();

        for i in 0..qubo.shape()[0] {
            for j in i + 1..qubo.shape()[1] {
                if partition[i] != partition[j] && qubo[[i, j]].abs() > 1e-10 {
                    coupling_terms.push(CouplingTerm {
                        var1: i,
                        var2: j,
                        partition1: partition[i],
                        partition2: partition[j],
                        weight: qubo[[i, j]],
                    });
                }
            }
        }

        Ok(coupling_terms)
    }

    /// Compute partition quality metrics
    fn compute_partition_metrics(&self, graph: &Graph, partition: &[usize]) -> PartitionMetrics {
        let edge_cut = graph
            .edges
            .iter()
            .filter(|e| partition[e.from] != partition[e.to])
            .map(|e| e.weight)
            .sum();

        let partition_sizes = self.count_partition_sizes(partition);
        let avg_size = graph.num_nodes as f64 / partition_sizes.len() as f64;
        let balance = partition_sizes
            .iter()
            .map(|&size| ((size as f64 - avg_size) / avg_size).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let modularity = self.compute_modularity(graph, partition);

        PartitionMetrics {
            edge_cut,
            balance,
            modularity,
            num_partitions: partition_sizes.len(),
        }
    }

    /// Compute modularity of partition
    fn compute_modularity(&self, graph: &Graph, partition: &[usize]) -> f64 {
        let total_weight: f64 = graph.edges.iter().map(|e| e.weight).sum();
        let mut modularity = 0.0;

        for edge in &graph.edges {
            if partition[edge.from] == partition[edge.to] {
                modularity += edge.weight / total_weight;

                // Subtract expected edges
                let deg_i: f64 = graph
                    .edges
                    .iter()
                    .filter(|e| e.from == edge.from || e.to == edge.from)
                    .map(|e| e.weight)
                    .sum();
                let deg_j: f64 = graph
                    .edges
                    .iter()
                    .filter(|e| e.from == edge.to || e.to == edge.to)
                    .map(|e| e.weight)
                    .sum();

                modularity -= (deg_i * deg_j) / (2.0 * total_weight * total_weight);
            }
        }

        modularity
    }
}

#[derive(Debug, Clone)]
struct Graph {
    num_nodes: usize,
    edges: Vec<Edge>,
    node_weights: Vec<f64>,
}

#[derive(Debug, Clone)]
struct Edge {
    from: usize,
    to: usize,
    weight: f64,
}

#[derive(Debug, Clone)]
pub struct Partitioning {
    pub partition_assignment: Vec<usize>,
    pub subproblems: Vec<Subproblem>,
    pub coupling_terms: Vec<CouplingTerm>,
    pub metrics: PartitionMetrics,
}

#[derive(Debug, Clone)]
pub struct Subproblem {
    pub id: usize,
    pub qubo: Array2<f64>,
    pub var_map: HashMap<String, usize>,
    pub original_indices: Vec<usize>,
    pub variables: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct CouplingTerm {
    pub var1: usize,
    pub var2: usize,
    pub partition1: usize,
    pub partition2: usize,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct PartitionMetrics {
    pub edge_cut: f64,
    pub balance: f64,
    pub modularity: f64,
    pub num_partitions: usize,
}

#[derive(Debug, Clone)]
pub struct HierarchyLevelInfo {
    pub level: usize,
    pub size: usize,
    pub variables: Vec<usize>,
}

/// Hierarchical problem solver
pub struct HierarchicalSolver<S: Sampler> {
    /// Base sampler
    base_sampler: S,
    /// Coarsening strategy
    coarsening: CoarseningStrategy,
    /// Refinement strategy
    refinement: RefinementStrategy,
    /// Maximum hierarchy levels
    max_levels: usize,
    /// Minimum problem size
    min_size: usize,
}

#[derive(Debug, Clone)]
pub enum CoarseningStrategy {
    /// Variable clustering
    VariableClustering,
    /// Algebraic multigrid
    AlgebraicMultigrid,
    /// Graph matching
    GraphMatching,
    /// Random aggregation
    RandomAggregation,
}

#[derive(Debug, Clone)]
pub enum RefinementStrategy {
    /// Local search
    LocalSearch,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Machine learning guided
    MLGuided,
}

impl<S: Sampler> HierarchicalSolver<S> {
    /// Create new hierarchical solver
    pub fn new(base_sampler: S) -> Self {
        Self {
            base_sampler,
            coarsening: CoarseningStrategy::VariableClustering,
            refinement: RefinementStrategy::LocalSearch,
            max_levels: 5,
            min_size: 10,
        }
    }
}

impl HierarchicalSolver<SASampler> {
    /// Create new hierarchical solver with default sampler
    pub fn with_default_sampler() -> Self {
        Self {
            base_sampler: SASampler::new(Some(42)),
            coarsening: CoarseningStrategy::VariableClustering,
            refinement: RefinementStrategy::LocalSearch,
            max_levels: 5,
            min_size: 10,
        }
    }

    /// Set coarsening strategy
    pub fn with_coarsening(mut self, strategy: CoarseningStrategy) -> Self {
        self.coarsening = strategy;
        self
    }

    /// Set refinement strategy
    pub fn with_refinement(mut self, strategy: RefinementStrategy) -> Self {
        self.refinement = strategy;
        self
    }

    /// Set maximum hierarchy levels
    pub fn with_max_levels(mut self, max_levels: usize) -> Self {
        self.max_levels = max_levels;
        self
    }

    /// Set minimum coarse size
    pub fn with_min_coarse_size(mut self, min_size: usize) -> Self {
        self.min_size = min_size;
        self
    }

    /// Create hierarchy levels for a QUBO problem
    pub fn create_hierarchy(&self, qubo: &Array2<f64>) -> Result<Vec<HierarchyLevelInfo>, String> {
        let mut hierarchy = Vec::new();
        let mut current_size = qubo.shape()[0];

        // Add the original problem as level 0
        hierarchy.push(HierarchyLevelInfo {
            level: 0,
            size: current_size,
            variables: (0..current_size).collect(),
        });

        // Create coarser levels
        for level in 1..self.max_levels {
            if current_size <= self.min_size {
                break;
            }

            // Simple coarsening: reduce size by half
            current_size = (current_size / 2).max(self.min_size);

            hierarchy.push(HierarchyLevelInfo {
                level,
                size: current_size,
                variables: (0..current_size).collect(),
            });
        }

        Ok(hierarchy)
    }

    /// Solve hierarchically
    pub fn solve(
        &mut self,
        qubo: &(Array2<f64>, HashMap<String, usize>),
        shots: usize,
    ) -> Result<HierarchicalResult, String> {
        let (qubo_matrix, var_map) = qubo;

        // Build hierarchy
        let hierarchy = self.build_hierarchy(qubo_matrix, var_map)?;

        // Solve from coarsest to finest
        let mut current_solution: Option<Vec<SampleResult>> = None;

        for level in (0..hierarchy.levels.len()).rev() {
            let level_data = &hierarchy.levels[level];

            // Solve at this level
            let level_solution = if let Some(ref coarse_sol) = current_solution {
                // Refine from coarser solution
                self.refine_solution(
                    &level_data.qubo,
                    &level_data.var_map,
                    coarse_sol,
                    &hierarchy.projections[level],
                    shots,
                )?
            } else {
                // Solve coarsest level directly
                self.base_sampler
                    .run_qubo(
                        &(level_data.qubo.clone(), level_data.var_map.clone()),
                        shots,
                    )
                    .map_err(|e| format!("Sampler error: {:?}", e))?
            };

            current_solution = Some(level_solution);
        }

        Ok(HierarchicalResult {
            solution: current_solution.unwrap(),
            hierarchy,
            refinement_history: Vec::new(),
        })
    }

    /// Build problem hierarchy
    fn build_hierarchy(
        &self,
        qubo: &Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<Hierarchy, String> {
        let mut levels = Vec::new();
        let mut projections = Vec::new();

        let mut current_qubo = qubo.clone();
        let mut current_var_map = var_map.clone();
        let mut current_size = qubo.shape()[0];

        // Add finest level
        levels.push(HierarchyLevel {
            level: 0,
            size: current_size,
            qubo: current_qubo.clone(),
            var_map: current_var_map.clone(),
        });

        // Coarsen until minimum size reached
        let mut level = 1;
        while current_size > self.min_size && level < self.max_levels {
            let (coarse_qubo, coarse_var_map, projection) =
                self.coarsen_problem(&current_qubo, &current_var_map)?;

            projections.push(projection);
            levels.push(HierarchyLevel {
                level,
                size: coarse_qubo.shape()[0],
                qubo: coarse_qubo.clone(),
                var_map: coarse_var_map.clone(),
            });

            current_qubo = coarse_qubo;
            current_var_map = coarse_var_map;
            current_size = current_qubo.shape()[0];
            level += 1;
        }

        Ok(Hierarchy {
            levels,
            projections,
        })
    }

    /// Coarsen problem
    fn coarsen_problem(
        &self,
        qubo: &Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(Array2<f64>, HashMap<String, usize>, Projection), String> {
        match self.coarsening {
            CoarseningStrategy::VariableClustering => {
                self.variable_clustering_coarsen(qubo, var_map)
            }
            _ => {
                // Default to variable clustering
                self.variable_clustering_coarsen(qubo, var_map)
            }
        }
    }

    /// Variable clustering coarsening
    fn variable_clustering_coarsen(
        &self,
        qubo: &Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<(Array2<f64>, HashMap<String, usize>, Projection), String> {
        let n = qubo.shape()[0];

        // Cluster strongly connected variables
        let mut clusters = Vec::new();
        let mut assigned = vec![false; n];

        for i in 0..n {
            if !assigned[i] {
                let mut cluster = vec![i];
                assigned[i] = true;

                // Find strongly connected variables
                for j in i + 1..n {
                    if !assigned[j] && qubo[[i, j]].abs() > 0.5 {
                        cluster.push(j);
                        assigned[j] = true;
                    }
                }

                clusters.push(cluster);
            }
        }

        // Build coarse problem
        let num_clusters = clusters.len();
        let mut coarse_qubo = Array2::zeros((num_clusters, num_clusters));

        for (ci, cluster_i) in clusters.iter().enumerate() {
            for (cj, cluster_j) in clusters.iter().enumerate() {
                let mut weight = 0.0;

                for &i in cluster_i {
                    for &j in cluster_j {
                        weight += qubo[[i, j]];
                    }
                }

                coarse_qubo[[ci, cj]] = weight;
            }
        }

        // Build variable mapping
        let mut coarse_var_map = HashMap::new();
        let reverse_map: HashMap<usize, String> =
            var_map.iter().map(|(k, v)| (*v, k.clone())).collect();

        for (ci, cluster) in clusters.iter().enumerate() {
            let var_name = format!("cluster_{}", ci);
            coarse_var_map.insert(var_name, ci);
        }

        // Build projection
        let projection = Projection {
            fine_to_coarse: (0..n)
                .map(|i| clusters.iter().position(|c| c.contains(&i)).unwrap_or(0))
                .collect(),
            coarse_to_fine: clusters,
        };

        Ok((coarse_qubo, coarse_var_map, projection))
    }

    /// Refine solution from coarser level
    fn refine_solution(
        &mut self,
        qubo: &Array2<f64>,
        var_map: &HashMap<String, usize>,
        coarse_solution: &[SampleResult],
        projection: &Projection,
        shots: usize,
    ) -> Result<Vec<SampleResult>, String> {
        // Project coarse solution to fine level
        let initial_solution = self.project_solution(coarse_solution, projection)?;

        // Apply refinement strategy
        match self.refinement {
            RefinementStrategy::LocalSearch => {
                self.local_search_refinement(qubo, var_map, &initial_solution, shots)
            }
            _ => {
                // Default to local search
                self.local_search_refinement(qubo, var_map, &initial_solution, shots)
            }
        }
    }

    /// Project solution to finer level
    fn project_solution(
        &self,
        coarse_solution: &[SampleResult],
        projection: &Projection,
    ) -> Result<Vec<bool>, String> {
        if let Some(best) = coarse_solution.first() {
            let mut fine_solution = vec![false; projection.fine_to_coarse.len()];

            for (fine_idx, &coarse_idx) in projection.fine_to_coarse.iter().enumerate() {
                let coarse_var = format!("cluster_{}", coarse_idx);
                fine_solution[fine_idx] =
                    best.assignments.get(&coarse_var).copied().unwrap_or(false);
            }

            Ok(fine_solution)
        } else {
            Err("No coarse solution to project".to_string())
        }
    }

    /// Local search refinement
    fn local_search_refinement(
        &mut self,
        qubo: &Array2<f64>,
        var_map: &HashMap<String, usize>,
        initial: &[bool],
        shots: usize,
    ) -> Result<Vec<SampleResult>, String> {
        // Use base sampler with warm start
        // For now, just run sampler normally
        self.base_sampler
            .run_qubo(&(qubo.clone(), var_map.clone()), shots)
            .map_err(|e| format!("Sampler error: {:?}", e))
    }
}

#[derive(Debug, Clone)]
pub struct Hierarchy {
    levels: Vec<HierarchyLevel>,
    projections: Vec<Projection>,
}

#[derive(Debug, Clone)]
struct HierarchyLevel {
    level: usize,
    size: usize,
    qubo: Array2<f64>,
    var_map: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
struct Projection {
    fine_to_coarse: Vec<usize>,
    coarse_to_fine: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    pub solution: Vec<SampleResult>,
    pub hierarchy: Hierarchy,
    pub refinement_history: Vec<RefinementStep>,
}

#[derive(Debug, Clone)]
pub struct RefinementStep {
    level: usize,
    initial_cost: f64,
    final_cost: f64,
    iterations: usize,
}

/// Domain decomposer
pub struct DomainDecomposer {
    /// Decomposition method
    method: DecompositionMethod,
    /// Number of domains
    num_domains: usize,
    /// Overlap between domains
    overlap: usize,
}

impl DomainDecomposer {
    /// Create new domain decomposer
    pub fn new() -> Self {
        Self {
            method: DecompositionMethod::Geometric { dimensions: 2 },
            num_domains: 2,
            overlap: 0,
        }
    }

    /// Set decomposition method
    pub fn with_method(mut self, method: DecompositionMethod) -> Self {
        self.method = method;
        self
    }

    /// Set number of domains
    pub fn with_num_domains(mut self, num_domains: usize) -> Self {
        self.num_domains = num_domains;
        self
    }

    /// Set overlap
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Decompose QUBO into domains
    pub fn decompose(&self, qubo: &Array2<f64>) -> Result<Vec<DecomposedDomain>, String> {
        // Simple domain decomposition based on variable ranges
        let n = qubo.shape()[0];
        let domain_size = (n + self.num_domains - 1) / self.num_domains;
        let mut domains = Vec::new();

        for i in 0..self.num_domains {
            let start = i * domain_size;
            let end = ((i + 1) * domain_size).min(n);

            if start >= n {
                break;
            }

            // Add overlap
            let overlap_start = if i > 0 {
                start.saturating_sub(self.overlap)
            } else {
                start
            };
            let overlap_end = if i < self.num_domains - 1 {
                (end + self.overlap).min(n)
            } else {
                end
            };

            let variables = (overlap_start..overlap_end).collect();

            domains.push(DecomposedDomain {
                id: i,
                variables,
                constraints: Vec::new(),
                boundary_conditions: Vec::new(),
            });
        }

        Ok(domains)
    }
}

#[derive(Debug, Clone)]
pub struct DecomposedDomain {
    pub id: usize,
    pub variables: Vec<usize>,
    pub constraints: Vec<String>,
    pub boundary_conditions: Vec<String>,
}

/// Parallel coordinator for subproblem solving
pub struct ParallelCoordinator {
    pub num_threads: usize,
    coordination_method: CoordinationMethod,
}

impl ParallelCoordinator {
    /// Create new parallel coordinator
    pub fn new() -> Self {
        Self {
            num_threads: 1,
            coordination_method: CoordinationMethod::MasterWorker,
        }
    }

    /// Set number of threads
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Set coordination method
    pub fn with_coordination_method(mut self, method: CoordinationMethod) -> Self {
        self.coordination_method = method;
        self
    }
}

#[derive(Debug, Clone)]
pub enum CoordinationMethod {
    /// Master-worker coordination
    MasterWorker,
    /// Peer-to-peer coordination
    PeerToPeer,
    /// Hierarchical coordination
    Hierarchical,
}

/// Domain decomposition solver
pub struct DomainDecompositionSolver<S: Sampler> {
    /// Base sampler
    base_sampler: S,
    /// Decomposition method
    method: DecompositionMethod,
    /// Coordination strategy
    coordination: CoordinationStrategy,
    /// Maximum iterations
    max_iterations: usize,
}

#[derive(Debug, Clone)]
pub enum DecompositionMethod {
    /// Geometric decomposition
    Geometric { dimensions: usize },
    /// Algebraic decomposition
    Algebraic,
    /// ADMM decomposition
    ADMM,
    /// Physics-based decomposition
    PhysicsBased { boundary_conditions: String },
    /// Data-driven decomposition
    DataDriven,
}

#[derive(Debug, Clone)]
pub enum CoordinationStrategy {
    /// Alternating direction method of multipliers
    ADMM { rho: f64 },
    /// Dual decomposition
    DualDecomposition { step_size: f64 },
    /// Consensus optimization
    Consensus { weight: f64 },
    /// Message passing
    MessagePassing { damping: f64 },
}

impl<S: Sampler + Clone + Send + Sync> DomainDecompositionSolver<S> {
    /// Create new domain decomposition solver
    pub fn new(base_sampler: S, method: DecompositionMethod) -> Self {
        Self {
            base_sampler,
            method,
            coordination: CoordinationStrategy::ADMM { rho: 1.0 },
            max_iterations: 100,
        }
    }

    /// Set coordination strategy
    pub fn with_coordination(mut self, strategy: CoordinationStrategy) -> Self {
        self.coordination = strategy;
        self
    }

    /// Solve using domain decomposition
    pub fn solve(
        &mut self,
        qubo: &(Array2<f64>, HashMap<String, usize>),
        num_domains: usize,
        shots: usize,
    ) -> Result<DomainDecompositionResult, String> {
        let (qubo_matrix, var_map) = qubo;

        // Decompose into domains
        let domains = self.decompose_into_domains(qubo_matrix, var_map, num_domains)?;

        // Initialize coordination variables
        let mut coordination_state = self.initialize_coordination(&domains)?;

        // Iterative solving with coordination
        for iteration in 0..self.max_iterations {
            // Solve subdomains in parallel
            let subdomain_solutions =
                self.solve_subdomains_parallel(&domains, &coordination_state, shots)?;

            // Update coordination
            let converged =
                self.update_coordination(&mut coordination_state, &subdomain_solutions, &domains)?;

            if converged {
                break;
            }
        }

        // Merge solutions
        let final_solution = self.merge_solutions(&domains, &coordination_state)?;

        Ok(DomainDecompositionResult {
            solution: final_solution,
            domains,
            iterations: coordination_state.iteration,
            converged: coordination_state.converged,
        })
    }

    /// Decompose problem into domains
    fn decompose_into_domains(
        &self,
        qubo: &Array2<f64>,
        var_map: &HashMap<String, usize>,
        num_domains: usize,
    ) -> Result<Vec<Domain>, String> {
        // Use graph partitioner
        let partitioner = GraphPartitioner::new()
            .with_algorithm(PartitioningAlgorithm::Multilevel)
            .with_num_partitions(num_domains);

        let partitioning = partitioner.partition_qubo(qubo, var_map)?;

        // Convert to domains
        let mut domains = Vec::new();

        for subproblem in partitioning.subproblems {
            // Identify boundary variables
            let boundary_vars: HashSet<_> = partitioning
                .coupling_terms
                .iter()
                .filter(|ct| ct.partition1 == subproblem.id || ct.partition2 == subproblem.id)
                .flat_map(|ct| vec![ct.var1, ct.var2])
                .filter(|&v| subproblem.original_indices.contains(&v))
                .collect();

            domains.push(Domain {
                id: subproblem.id,
                qubo: subproblem.qubo,
                var_map: subproblem.var_map,
                boundary_vars: boundary_vars.into_iter().collect(),
                neighbors: self.find_neighbors(&partitioning.coupling_terms, subproblem.id),
            });
        }

        Ok(domains)
    }

    /// Find neighboring domains
    fn find_neighbors(&self, coupling_terms: &[CouplingTerm], domain_id: usize) -> Vec<usize> {
        let neighbors: HashSet<_> = coupling_terms
            .iter()
            .filter_map(|ct| {
                if ct.partition1 == domain_id {
                    Some(ct.partition2)
                } else if ct.partition2 == domain_id {
                    Some(ct.partition1)
                } else {
                    None
                }
            })
            .collect();

        neighbors.into_iter().collect()
    }

    /// Initialize coordination state
    fn initialize_coordination(&self, domains: &[Domain]) -> Result<CoordinationState, String> {
        match &self.coordination {
            CoordinationStrategy::ADMM { rho } => {
                let mut lagrange_multipliers = HashMap::new();
                let mut consensus_variables = HashMap::new();

                for domain in domains {
                    for &boundary_var in &domain.boundary_vars {
                        lagrange_multipliers.insert((domain.id, boundary_var), 0.0);
                        consensus_variables.insert(boundary_var, false);
                    }
                }

                Ok(CoordinationState {
                    lagrange_multipliers: Some(lagrange_multipliers),
                    consensus_variables: Some(consensus_variables),
                    dual_variables: None,
                    messages: None,
                    iteration: 0,
                    converged: false,
                })
            }
            _ => Ok(CoordinationState::default()),
        }
    }

    /// Solve subdomains in parallel
    fn solve_subdomains_parallel(
        &self,
        domains: &[Domain],
        coordination: &CoordinationState,
        shots: usize,
    ) -> Result<Vec<SubdomainSolution>, String> {
        let solutions: Vec<_> = domains
            .par_iter()
            .map(|domain| self.solve_single_subdomain(domain, coordination, shots))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(solutions)
    }

    /// Solve single subdomain
    fn solve_single_subdomain(
        &self,
        domain: &Domain,
        coordination: &CoordinationState,
        shots: usize,
    ) -> Result<SubdomainSolution, String> {
        // Modify QUBO with coordination terms
        let modified_qubo = self.add_coordination_terms(domain, coordination)?;

        // Solve
        let results = self
            .base_sampler
            .run_qubo(&(modified_qubo, domain.var_map.clone()), shots)
            .map_err(|e| format!("Sampler error: {:?}", e))?;

        Ok(SubdomainSolution {
            domain_id: domain.id,
            results,
        })
    }

    /// Add coordination terms to subdomain QUBO
    fn add_coordination_terms(
        &self,
        domain: &Domain,
        coordination: &CoordinationState,
    ) -> Result<Array2<f64>, String> {
        let mut modified_qubo = domain.qubo.clone();

        match &self.coordination {
            CoordinationStrategy::ADMM { rho } => {
                if let (Some(lagrange), Some(consensus)) = (
                    &coordination.lagrange_multipliers,
                    &coordination.consensus_variables,
                ) {
                    // Add augmented Lagrangian terms
                    for &boundary_var in &domain.boundary_vars {
                        if let Some(local_idx) =
                            domain.var_map.values().find(|&&v| v == boundary_var)
                        {
                            let lambda = lagrange.get(&(domain.id, boundary_var)).unwrap_or(&0.0);
                            let z = if *consensus.get(&boundary_var).unwrap_or(&false) {
                                1.0
                            } else {
                                0.0
                            };

                            // Add (rho/2)||x - z||^2 + lambda^T(x - z)
                            modified_qubo[[*local_idx, *local_idx]] +=
                                rho + 2.0 * lambda * (1.0 - z);
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(modified_qubo)
    }

    /// Update coordination variables
    fn update_coordination(
        &self,
        state: &mut CoordinationState,
        solutions: &[SubdomainSolution],
        domains: &[Domain],
    ) -> Result<bool, String> {
        match &self.coordination {
            CoordinationStrategy::ADMM { rho } => {
                self.update_admm_coordination(state, solutions, domains, *rho)
            }
            _ => Ok(false),
        }
    }

    /// Update ADMM coordination
    fn update_admm_coordination(
        &self,
        state: &mut CoordinationState,
        solutions: &[SubdomainSolution],
        domains: &[Domain],
        rho: f64,
    ) -> Result<bool, String> {
        if let (Some(lagrange), Some(consensus)) = (
            &mut state.lagrange_multipliers,
            &mut state.consensus_variables,
        ) {
            let mut max_residual = 0.0_f64;

            // Update consensus variables (z-update)
            for domain in domains {
                for &boundary_var in &domain.boundary_vars {
                    if let Some(best) = solutions
                        .iter()
                        .find(|s| s.domain_id == domain.id)
                        .and_then(|s| s.results.first())
                    {
                        // Average over all domains containing this variable
                        let mut sum = 0.0;
                        let mut count = 0;

                        for d in domains {
                            if d.boundary_vars.contains(&boundary_var) {
                                if let Some(sol) = solutions
                                    .iter()
                                    .find(|s| s.domain_id == d.id)
                                    .and_then(|s| s.results.first())
                                {
                                    // Map boundary var to local var
                                    // Simplified: assume direct mapping
                                    sum += if sol.assignments.values().any(|&v| v) {
                                        1.0
                                    } else {
                                        0.0
                                    };
                                    count += 1;
                                }
                            }
                        }

                        consensus.insert(boundary_var, sum / count as f64 > 0.5);
                    }
                }
            }

            // Update Lagrange multipliers (dual update)
            for domain in domains {
                for &boundary_var in &domain.boundary_vars {
                    if let Some(best) = solutions
                        .iter()
                        .find(|s| s.domain_id == domain.id)
                        .and_then(|s| s.results.first())
                    {
                        let x = if best.assignments.values().any(|&v| v) {
                            1.0_f64
                        } else {
                            0.0_f64
                        };
                        let z = if *consensus.get(&boundary_var).unwrap_or(&false) {
                            1.0_f64
                        } else {
                            0.0_f64
                        };

                        let residual = x - z;
                        max_residual = max_residual.max(residual.abs());

                        let lambda = lagrange.get(&(domain.id, boundary_var)).unwrap_or(&0.0);
                        lagrange.insert((domain.id, boundary_var), lambda + rho * residual);
                    }
                }
            }

            state.iteration += 1;
            state.converged = max_residual < 1e-3;

            Ok(state.converged)
        } else {
            Ok(false)
        }
    }

    /// Merge solutions from domains
    fn merge_solutions(
        &self,
        domains: &[Domain],
        coordination: &CoordinationState,
    ) -> Result<Vec<SampleResult>, String> {
        // For now, return consensus solution
        if let Some(consensus) = &coordination.consensus_variables {
            let mut assignments = HashMap::new();

            for (var_idx, value) in consensus {
                assignments.insert(format!("x{}", var_idx), *value);
            }

            Ok(vec![SampleResult {
                assignments,
                energy: 0.0, // Would need to compute
                occurrences: 1,
            }])
        } else {
            Err("No consensus variables available".to_string())
        }
    }
}

#[derive(Debug, Clone)]
pub struct Domain {
    id: usize,
    qubo: Array2<f64>,
    var_map: HashMap<String, usize>,
    boundary_vars: Vec<usize>,
    neighbors: Vec<usize>,
}

#[derive(Debug, Clone, Default)]
struct CoordinationState {
    lagrange_multipliers: Option<HashMap<(usize, usize), f64>>,
    consensus_variables: Option<HashMap<usize, bool>>,
    dual_variables: Option<HashMap<usize, f64>>,
    messages: Option<HashMap<(usize, usize), f64>>,
    iteration: usize,
    converged: bool,
}

#[derive(Debug, Clone)]
struct SubdomainSolution {
    domain_id: usize,
    results: Vec<SampleResult>,
}

#[derive(Debug, Clone)]
pub struct DomainDecompositionResult {
    pub solution: Vec<SampleResult>,
    pub domains: Vec<Domain>,
    pub iterations: usize,
    pub converged: bool,
}

/// Parallel subproblem solver with load balancing
pub struct ParallelSubproblemSolver<S: Sampler + Clone + Send + Sync> {
    /// Base sampler
    base_sampler: S,
    /// Number of workers
    num_workers: usize,
    /// Load balancing strategy
    load_balancing: LoadBalancingStrategy,
    /// Communication pattern
    communication: CommunicationPattern,
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Static assignment
    Static,
    /// Dynamic work stealing
    WorkStealing,
    /// Guided self-scheduling
    Guided { chunk_size: usize },
    /// Adaptive based on runtime
    Adaptive,
}

#[derive(Debug, Clone)]
pub enum CommunicationPattern {
    /// Master-worker
    MasterWorker,
    /// All-to-all
    AllToAll,
    /// Hierarchical
    Hierarchical { levels: usize },
    /// Asynchronous
    Asynchronous,
}

impl<S: Sampler + Clone + Send + Sync + 'static> ParallelSubproblemSolver<S> {
    /// Create new parallel solver
    pub fn new(base_sampler: S, num_workers: usize) -> Self {
        Self {
            base_sampler,
            num_workers,
            load_balancing: LoadBalancingStrategy::WorkStealing,
            communication: CommunicationPattern::MasterWorker,
        }
    }

    /// Solve subproblems in parallel
    pub fn solve_parallel(
        &self,
        subproblems: Vec<Subproblem>,
        shots_per_subproblem: usize,
    ) -> Result<ParallelResult, String> {
        match self.load_balancing {
            LoadBalancingStrategy::WorkStealing => {
                self.work_stealing_solve(subproblems, shots_per_subproblem)
            }
            _ => self.static_solve(subproblems, shots_per_subproblem),
        }
    }

    /// Static parallel solving
    fn static_solve(
        &self,
        subproblems: Vec<Subproblem>,
        shots: usize,
    ) -> Result<ParallelResult, String> {
        let solutions: Vec<_> = subproblems
            .par_iter()
            .map(|subproblem| {
                let results = self
                    .base_sampler
                    .run_qubo(
                        &(subproblem.qubo.clone(), subproblem.var_map.clone()),
                        shots,
                    )
                    .map_err(|e| format!("Sampler error: {:?}", e))?;

                Ok(SubproblemResult {
                    subproblem_id: subproblem.id,
                    results,
                    solve_time: std::time::Duration::from_secs(1), // Placeholder
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(ParallelResult {
            subproblem_results: solutions,
            total_time: std::time::Duration::from_secs(1),
            efficiency: 0.9,
        })
    }

    /// Work stealing parallel solving
    fn work_stealing_solve(
        &self,
        subproblems: Vec<Subproblem>,
        shots: usize,
    ) -> Result<ParallelResult, String> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};

        let work_queue = Arc::new(Mutex::new(subproblems));
        let results = Arc::new(Mutex::new(Vec::new()));
        let completed = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..self.num_workers)
            .map(|_| {
                let queue = Arc::clone(&work_queue);
                let res = Arc::clone(&results);
                let comp = Arc::clone(&completed);
                let sampler = self.base_sampler.clone();

                std::thread::spawn(move || loop {
                    let subproblem = {
                        let mut q = queue.lock().unwrap();
                        q.pop()
                    };

                    if let Some(sp) = subproblem {
                        let start = std::time::Instant::now();

                        if let Ok(solution) =
                            sampler.run_qubo(&(sp.qubo.clone(), sp.var_map.clone()), shots)
                        {
                            let mut r = res.lock().unwrap();
                            r.push(SubproblemResult {
                                subproblem_id: sp.id,
                                results: solution,
                                solve_time: start.elapsed(),
                            });

                            comp.fetch_add(1, Ordering::Relaxed);
                        }
                    } else {
                        break;
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let solutions = Arc::try_unwrap(results).unwrap().into_inner().unwrap();

        Ok(ParallelResult {
            subproblem_results: solutions,
            total_time: std::time::Duration::from_secs(1),
            efficiency: 0.95,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SubproblemResult {
    pub subproblem_id: usize,
    pub results: Vec<SampleResult>,
    pub solve_time: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct ParallelResult {
    pub subproblem_results: Vec<SubproblemResult>,
    pub total_time: std::time::Duration,
    pub efficiency: f64,
}

/// Constraint satisfaction decomposer
pub struct ConstraintSatisfactionDecomposer {
    /// Decomposition strategy
    strategy: CSPDecompositionStrategy,
    /// Variable ordering heuristic
    variable_ordering: VariableOrderingHeuristic,
    /// Constraint propagation level
    propagation_level: PropagationLevel,
    /// Maximum cluster size
    max_cluster_size: usize,
}

#[derive(Debug, Clone)]
pub enum CSPDecompositionStrategy {
    /// Tree decomposition
    TreeDecomposition,
    /// Hypertree decomposition
    HypertreeDecomposition { width: usize },
    /// Cycle cutset decomposition
    CycleCutset,
    /// Constraint clustering
    ConstraintClustering,
    /// Functional decomposition
    FunctionalDecomposition,
}

#[derive(Debug, Clone)]
pub enum VariableOrderingHeuristic {
    /// Minimum remaining values
    MRV,
    /// Degree heuristic
    Degree,
    /// Minimum width ordering
    MinWidth,
    /// Maximum cardinality search
    MaxCardinality,
    /// Dynamic ordering
    Dynamic,
}

#[derive(Debug, Clone)]
pub enum PropagationLevel {
    /// No propagation
    None,
    /// Node consistency
    NodeConsistency,
    /// Arc consistency
    ArcConsistency,
    /// Path consistency
    PathConsistency,
    /// Generalized arc consistency
    GAC,
}

/// CSP representation
#[derive(Debug, Clone)]
pub struct CSPProblem {
    /// Variables with their domains
    pub variables: HashMap<String, VariableInfo>,
    /// Constraints
    pub constraints: Vec<CSPConstraint>,
    /// Constraint graph
    pub constraint_graph: ConstraintGraph,
}

#[derive(Debug, Clone)]
pub struct VariableInfo {
    pub name: String,
    pub domain: Vec<i32>,
    pub current_domain: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct CSPConstraint {
    pub id: usize,
    pub scope: Vec<String>,
    pub constraint_type: ConstraintType,
    pub tuples: Option<Vec<Vec<i32>>>, // For extensional constraints
}

#[derive(Debug, Clone)]
pub enum ConstraintType {
    /// All different constraint
    AllDifferent,
    /// Linear constraint
    Linear { coefficients: Vec<f64>, op: String, rhs: f64 },
    /// Table constraint (extensional)
    Table { allowed: bool },
    /// Global cardinality
    GlobalCardinality { values: Vec<i32>, bounds: Vec<(usize, usize)> },
    /// Cumulative constraint
    Cumulative { capacity: i32 },
    /// Element constraint
    Element,
    /// Custom constraint
    Custom { name: String },
}

#[derive(Debug, Clone)]
pub struct ConstraintGraph {
    /// Adjacency list representation
    pub adjacency: HashMap<String, HashSet<String>>,
    /// Constraint hypergraph
    pub hyperedges: Vec<HashSet<String>>,
}

impl ConstraintSatisfactionDecomposer {
    /// Create new CSP decomposer
    pub fn new() -> Self {
        Self {
            strategy: CSPDecompositionStrategy::TreeDecomposition,
            variable_ordering: VariableOrderingHeuristic::MinWidth,
            propagation_level: PropagationLevel::ArcConsistency,
            max_cluster_size: 10,
        }
    }

    /// Set decomposition strategy
    pub fn with_strategy(mut self, strategy: CSPDecompositionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set variable ordering heuristic
    pub fn with_variable_ordering(mut self, ordering: VariableOrderingHeuristic) -> Self {
        self.variable_ordering = ordering;
        self
    }

    /// Decompose CSP into subproblems
    pub fn decompose(&self, csp: &CSPProblem) -> Result<CSPDecomposition, String> {
        match self.strategy {
            CSPDecompositionStrategy::TreeDecomposition => {
                self.tree_decomposition(csp)
            }
            CSPDecompositionStrategy::ConstraintClustering => {
                self.constraint_clustering_decomposition(csp)
            }
            CSPDecompositionStrategy::CycleCutset => {
                self.cycle_cutset_decomposition(csp)
            }
            _ => {
                // Default to tree decomposition
                self.tree_decomposition(csp)
            }
        }
    }

    /// Tree decomposition of CSP
    fn tree_decomposition(&self, csp: &CSPProblem) -> Result<CSPDecomposition, String> {
        // Build primal graph
        let primal_graph = self.build_primal_graph(csp);
        
        // Find tree decomposition
        let tree_dec = self.find_tree_decomposition(&primal_graph)?;
        
        // Extract clusters
        let clusters = self.extract_clusters_from_tree(&tree_dec, csp)?;
        
        // Build cluster tree
        let cluster_tree = self.build_cluster_tree(&clusters, &tree_dec)?;
        
        let separator_sets = self.compute_separator_sets(&cluster_tree);
        Ok(CSPDecomposition {
            clusters,
            cluster_tree,
            separator_sets,
            width: tree_dec.width,
        })
    }

    /// Build primal graph from CSP
    fn build_primal_graph(&self, csp: &CSPProblem) -> Graph {
        let mut edges = Vec::new();
        let var_to_idx: HashMap<&str, usize> = csp.variables
            .keys()
            .enumerate()
            .map(|(i, v)| (v.as_str(), i))
            .collect();
        
        // Add edge between variables that appear in same constraint
        for constraint in &csp.constraints {
            let vars_in_constraint: Vec<_> = constraint.scope
                .iter()
                .filter_map(|v| var_to_idx.get(v.as_str()))
                .collect();
            
            // Add all pairs as edges
            for i in 0..vars_in_constraint.len() {
                for j in i+1..vars_in_constraint.len() {
                    edges.push(Edge {
                        from: *vars_in_constraint[i],
                        to: *vars_in_constraint[j],
                        weight: 1.0,
                    });
                }
            }
        }
        
        Graph {
            num_nodes: csp.variables.len(),
            edges,
            node_weights: vec![1.0; csp.variables.len()],
        }
    }

    /// Find tree decomposition using minimum width heuristic
    fn find_tree_decomposition(&self, graph: &Graph) -> Result<TreeDecomposition, String> {
        // Use greedy minimum width algorithm
        let mut remaining_nodes: HashSet<_> = (0..graph.num_nodes).collect();
        let mut tree_nodes = Vec::new();
        let mut elimination_order = Vec::new();
        let mut current_graph = graph.clone();
        let mut max_bag_size = 0;
        
        while !remaining_nodes.is_empty() {
            // Choose node to eliminate based on heuristic
            let node = self.choose_node_to_eliminate(&current_graph, &remaining_nodes)?;
            elimination_order.push(node);
            
            // Find neighbors in current graph
            let neighbors = self.get_neighbors(&current_graph, node);
            
            // Create tree node (bag) with node and its neighbors
            let mut bag = neighbors.clone();
            bag.insert(node);
            max_bag_size = max_bag_size.max(bag.len());
            
            tree_nodes.push(TreeNode {
                id: tree_nodes.len(),
                bag,
                children: Vec::new(),
            });
            
            // Make neighbors form a clique
            self.make_clique(&mut current_graph, &neighbors);
            
            // Remove node from graph
            remaining_nodes.remove(&node);
            current_graph = self.remove_node_from_graph(&current_graph, node);
        }
        
        Ok(TreeDecomposition {
            nodes: tree_nodes,
            width: max_bag_size - 1, // Tree width is max bag size - 1
            root: 0,
        })
    }

    /// Choose node to eliminate based on heuristic
    fn choose_node_to_eliminate(&self, graph: &Graph, remaining: &HashSet<usize>) -> Result<usize, String> {
        match self.variable_ordering {
            VariableOrderingHeuristic::MinWidth => {
                // Choose node that creates smallest clique when eliminated
                remaining.iter()
                    .min_by_key(|&&node| {
                        let neighbors = self.get_neighbors(graph, node);
                        neighbors.len()
                    })
                    .copied()
                    .ok_or_else(|| "No nodes remaining".to_string())
            }
            _ => {
                // Default: choose arbitrary node
                remaining.iter().next().copied()
                    .ok_or_else(|| "No nodes remaining".to_string())
            }
        }
    }

    /// Get neighbors of a node in graph
    fn get_neighbors(&self, graph: &Graph, node: usize) -> HashSet<usize> {
        graph.edges
            .iter()
            .filter_map(|edge| {
                if edge.from == node {
                    Some(edge.to)
                } else if edge.to == node {
                    Some(edge.from)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Make a set of nodes form a clique
    fn make_clique(&self, graph: &mut Graph, nodes: &HashSet<usize>) {
        let node_vec: Vec<_> = nodes.iter().copied().collect();
        for i in 0..node_vec.len() {
            for j in i+1..node_vec.len() {
                // Check if edge already exists
                let exists = graph.edges.iter().any(|e| {
                    (e.from == node_vec[i] && e.to == node_vec[j]) ||
                    (e.from == node_vec[j] && e.to == node_vec[i])
                });
                
                if !exists {
                    graph.edges.push(Edge {
                        from: node_vec[i],
                        to: node_vec[j],
                        weight: 1.0,
                    });
                }
            }
        }
    }

    /// Remove node from graph
    fn remove_node_from_graph(&self, graph: &Graph, node: usize) -> Graph {
        Graph {
            num_nodes: graph.num_nodes,
            edges: graph.edges
                .iter()
                .filter(|e| e.from != node && e.to != node)
                .cloned()
                .collect(),
            node_weights: graph.node_weights.clone(),
        }
    }

    /// Extract clusters from tree decomposition
    fn extract_clusters_from_tree(&self, tree_dec: &TreeDecomposition, csp: &CSPProblem) -> Result<Vec<CSPCluster>, String> {
        let var_names: Vec<_> = csp.variables.keys().cloned().collect();
        let mut clusters = Vec::new();
        
        for tree_node in &tree_dec.nodes {
            let cluster_vars: Vec<_> = tree_node.bag
                .iter()
                .filter_map(|&idx| var_names.get(idx).cloned())
                .collect();
            
            // Find constraints that are fully contained in this cluster
            let cluster_constraints: Vec<_> = csp.constraints
                .iter()
                .filter(|c| c.scope.iter().all(|v| cluster_vars.contains(v)))
                .cloned()
                .collect();
            
            clusters.push(CSPCluster {
                id: tree_node.id,
                variables: cluster_vars,
                constraints: cluster_constraints,
                subproblem: None,
            });
        }
        
        Ok(clusters)
    }

    /// Build cluster tree from tree decomposition
    fn build_cluster_tree(&self, clusters: &[CSPCluster], tree_dec: &TreeDecomposition) -> Result<ClusterTree, String> {
        // For now, create a simple linear tree
        // TODO: Implement proper tree construction from tree decomposition
        let mut edges = Vec::new();
        for i in 1..clusters.len() {
            edges.push((i-1, i));
        }
        
        Ok(ClusterTree {
            clusters: clusters.len(),
            edges,
            root: 0,
        })
    }

    /// Compute separator sets between clusters
    fn compute_separator_sets(&self, cluster_tree: &ClusterTree) -> HashMap<(usize, usize), HashSet<String>> {
        // TODO: Implement separator set computation
        HashMap::new()
    }

    /// Constraint clustering decomposition
    fn constraint_clustering_decomposition(&self, csp: &CSPProblem) -> Result<CSPDecomposition, String> {
        // Group constraints by their scope overlap
        let mut clusters = Vec::new();
        let mut assigned_constraints = vec![false; csp.constraints.len()];
        let mut cluster_id = 0;
        
        for (i, constraint) in csp.constraints.iter().enumerate() {
            if assigned_constraints[i] {
                continue;
            }
            
            // Start new cluster with this constraint
            let mut cluster_vars: HashSet<_> = constraint.scope.iter().cloned().collect();
            let mut cluster_constraints = vec![constraint.clone()];
            assigned_constraints[i] = true;
            
            // Add related constraints
            for (j, other_constraint) in csp.constraints.iter().enumerate() {
                if assigned_constraints[j] {
                    continue;
                }
                
                // Check overlap
                let overlap: HashSet<_> = other_constraint.scope
                    .iter()
                    .filter(|v| cluster_vars.contains(*v))
                    .cloned()
                    .collect();
                
                if !overlap.is_empty() && cluster_vars.len() + other_constraint.scope.len() - overlap.len() <= self.max_cluster_size {
                    // Add to cluster
                    cluster_vars.extend(other_constraint.scope.iter().cloned());
                    cluster_constraints.push(other_constraint.clone());
                    assigned_constraints[j] = true;
                }
            }
            
            clusters.push(CSPCluster {
                id: cluster_id,
                variables: cluster_vars.into_iter().collect(),
                constraints: cluster_constraints,
                subproblem: None,
            });
            cluster_id += 1;
        }
        
        // Build cluster tree (for now, star topology)
        let edges = (1..clusters.len()).map(|i| (0, i)).collect();
        
        Ok(CSPDecomposition {
            clusters,
            cluster_tree: ClusterTree {
                clusters: cluster_id,
                edges,
                root: 0,
            },
            separator_sets: HashMap::new(),
            width: self.max_cluster_size,
        })
    }

    /// Cycle cutset decomposition
    fn cycle_cutset_decomposition(&self, csp: &CSPProblem) -> Result<CSPDecomposition, String> {
        // Find cycle cutset
        let primal_graph = self.build_primal_graph(csp);
        let cutset = self.find_cycle_cutset(&primal_graph)?;
        
        // Create two clusters: cutset variables and remaining variables
        let cutset_vars: HashSet<_> = cutset.iter()
            .filter_map(|&idx| csp.variables.keys().nth(idx))
            .cloned()
            .collect();
        
        let remaining_vars: HashSet<_> = csp.variables.keys()
            .filter(|v| !cutset_vars.contains(*v))
            .cloned()
            .collect();
        
        let mut clusters = vec![
            CSPCluster {
                id: 0,
                variables: cutset_vars.into_iter().collect(),
                constraints: Vec::new(),
                subproblem: None,
            },
            CSPCluster {
                id: 1,
                variables: remaining_vars.into_iter().collect(),
                constraints: Vec::new(),
                subproblem: None,
            },
        ];
        
        // Assign constraints to clusters
        for constraint in &csp.constraints {
            let in_cutset = constraint.scope.iter().all(|v| clusters[0].variables.contains(v));
            let in_remaining = constraint.scope.iter().all(|v| clusters[1].variables.contains(v));
            
            if in_cutset {
                clusters[0].constraints.push(constraint.clone());
            } else if in_remaining {
                clusters[1].constraints.push(constraint.clone());
            }
            // Constraints spanning both clusters need special handling
        }
        
        Ok(CSPDecomposition {
            clusters,
            cluster_tree: ClusterTree {
                clusters: 2,
                edges: vec![(0, 1)],
                root: 0,
            },
            separator_sets: HashMap::new(),
            width: cutset.len(),
        })
    }

    /// Find cycle cutset using greedy heuristic
    fn find_cycle_cutset(&self, graph: &Graph) -> Result<HashSet<usize>, String> {
        let mut cutset = HashSet::new();
        let mut remaining_graph = graph.clone();
        
        while self.has_cycle(&remaining_graph) {
            // Choose node with highest degree
            let node = (0..remaining_graph.num_nodes)
                .filter(|n| !cutset.contains(n))
                .max_by_key(|&n| self.get_neighbors(&remaining_graph, n).len())
                .ok_or_else(|| "No nodes remaining".to_string())?;
            
            cutset.insert(node);
            remaining_graph = self.remove_node_from_graph(&remaining_graph, node);
        }
        
        Ok(cutset)
    }

    /// Check if graph has a cycle using DFS
    fn has_cycle(&self, graph: &Graph) -> bool {
        let mut visited = vec![false; graph.num_nodes];
        let mut rec_stack = vec![false; graph.num_nodes];
        
        for node in 0..graph.num_nodes {
            if !visited[node] && self.has_cycle_dfs(graph, node, &mut visited, &mut rec_stack, None) {
                return true;
            }
        }
        
        false
    }

    /// DFS helper for cycle detection
    fn has_cycle_dfs(&self, graph: &Graph, node: usize, visited: &mut [bool], rec_stack: &mut [bool], parent: Option<usize>) -> bool {
        visited[node] = true;
        rec_stack[node] = true;
        
        for neighbor in self.get_neighbors(graph, node) {
            if !visited[neighbor] {
                if self.has_cycle_dfs(graph, neighbor, visited, rec_stack, Some(node)) {
                    return true;
                }
            } else if rec_stack[neighbor] && Some(neighbor) != parent {
                return true;
            }
        }
        
        rec_stack[node] = false;
        false
    }
}

/// Tree decomposition structure
#[derive(Debug, Clone)]
struct TreeDecomposition {
    nodes: Vec<TreeNode>,
    width: usize,
    root: usize,
}

#[derive(Debug, Clone)]
struct TreeNode {
    id: usize,
    bag: HashSet<usize>,
    children: Vec<usize>,
}

/// CSP decomposition result
#[derive(Debug, Clone)]
pub struct CSPDecomposition {
    pub clusters: Vec<CSPCluster>,
    pub cluster_tree: ClusterTree,
    pub separator_sets: HashMap<(usize, usize), HashSet<String>>,
    pub width: usize,
}

#[derive(Debug, Clone)]
pub struct CSPCluster {
    pub id: usize,
    pub variables: Vec<String>,
    pub constraints: Vec<CSPConstraint>,
    pub subproblem: Option<CSPProblem>,
}

#[derive(Debug, Clone)]
pub struct ClusterTree {
    pub clusters: usize,
    pub edges: Vec<(usize, usize)>,
    pub root: usize,
}

/// CSP solver using decomposition
pub struct DecompositionCSPSolver<S: Sampler> {
    /// Base sampler for solving subproblems
    base_sampler: S,
    /// Decomposer
    decomposer: ConstraintSatisfactionDecomposer,
    /// Message passing algorithm
    message_passing: MessagePassingAlgorithm,
}

#[derive(Debug, Clone)]
pub enum MessagePassingAlgorithm {
    /// Belief propagation
    BeliefPropagation,
    /// Max-product
    MaxProduct,
    /// Sum-product
    SumProduct,
    /// Junction tree algorithm
    JunctionTree,
}

impl<S: Sampler> DecompositionCSPSolver<S> {
    /// Create new decomposition-based CSP solver
    pub fn new(base_sampler: S) -> Self {
        Self {
            base_sampler,
            decomposer: ConstraintSatisfactionDecomposer::new(),
            message_passing: MessagePassingAlgorithm::JunctionTree,
        }
    }

    /// Solve CSP using decomposition
    pub fn solve(&mut self, csp: &CSPProblem, shots: usize) -> Result<CSPSolution, String> {
        // Decompose problem
        let decomposition = self.decomposer.decompose(csp)?;
        
        // Solve each cluster
        let mut cluster_solutions = Vec::new();
        for cluster in &decomposition.clusters {
            let solution = self.solve_cluster(cluster, shots)?;
            cluster_solutions.push(solution);
        }
        
        // Combine solutions using message passing
        let combined_solution = self.combine_solutions(
            &cluster_solutions,
            &decomposition.cluster_tree,
            &decomposition.separator_sets,
        )?;
        
        Ok(combined_solution)
    }

    /// Solve a single cluster
    fn solve_cluster(&mut self, cluster: &CSPCluster, shots: usize) -> Result<ClusterSolution, String> {
        // Convert cluster to QUBO if possible
        // For now, return a dummy solution
        Ok(ClusterSolution {
            cluster_id: cluster.id,
            assignments: HashMap::new(),
            cost: 0.0,
        })
    }

    /// Combine cluster solutions
    fn combine_solutions(
        &self,
        cluster_solutions: &[ClusterSolution],
        cluster_tree: &ClusterTree,
        separator_sets: &HashMap<(usize, usize), HashSet<String>>,
    ) -> Result<CSPSolution, String> {
        // TODO: Implement message passing to combine solutions
        Ok(CSPSolution {
            assignments: HashMap::new(),
            satisfied_constraints: Vec::new(),
            violated_constraints: Vec::new(),
            objective_value: 0.0,
        })
    }
}

#[derive(Debug, Clone)]
struct ClusterSolution {
    cluster_id: usize,
    assignments: HashMap<String, i32>,
    cost: f64,
}

#[derive(Debug, Clone)]
pub struct CSPSolution {
    pub assignments: HashMap<String, i32>,
    pub satisfied_constraints: Vec<usize>,
    pub violated_constraints: Vec<usize>,
    pub objective_value: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::SASampler;

    #[test]
    #[ignore]
    fn test_graph_partitioner() {
        let mut qubo = Array2::zeros((4, 4));
        qubo[[0, 1]] = -1.0;
        qubo[[1, 0]] = -1.0;
        qubo[[2, 3]] = -1.0;
        qubo[[3, 2]] = -1.0;
        qubo[[1, 2]] = -0.5;
        qubo[[2, 1]] = -0.5;

        let mut var_map = HashMap::new();
        for i in 0..4 {
            var_map.insert(format!("x{}", i), i);
        }

        let partitioner = GraphPartitioner::new()
            .with_algorithm(PartitioningAlgorithm::Spectral)
            .with_num_partitions(2);

        let result = partitioner.partition_qubo(&qubo, &var_map);
        assert!(result.is_ok());

        let partitioning = result.unwrap();
        assert_eq!(partitioning.subproblems.len(), 2);
    }

    #[test]
    fn test_hierarchical_solver() {
        let mut solver = HierarchicalSolver::new(SASampler::new(Some(42)));

        let mut qubo = Array2::eye(4);
        let mut var_map = HashMap::new();
        for i in 0..4 {
            var_map.insert(format!("x{}", i), i);
        }

        let result = solver.solve(&(qubo, var_map), 10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_csp_decomposer() {
        let decomposer = ConstraintSatisfactionDecomposer::new();
        
        // Create a simple CSP
        let mut variables = HashMap::new();
        variables.insert("x".to_string(), VariableInfo {
            name: "x".to_string(),
            domain: vec![0, 1],
            current_domain: vec![0, 1],
        });
        variables.insert("y".to_string(), VariableInfo {
            name: "y".to_string(),
            domain: vec![0, 1],
            current_domain: vec![0, 1],
        });
        variables.insert("z".to_string(), VariableInfo {
            name: "z".to_string(),
            domain: vec![0, 1],
            current_domain: vec![0, 1],
        });
        
        let constraints = vec![
            CSPConstraint {
                id: 0,
                scope: vec!["x".to_string(), "y".to_string()],
                constraint_type: ConstraintType::AllDifferent,
                tuples: None,
            },
            CSPConstraint {
                id: 1,
                scope: vec!["y".to_string(), "z".to_string()],
                constraint_type: ConstraintType::AllDifferent,
                tuples: None,
            },
        ];
        
        let mut adjacency = HashMap::new();
        adjacency.insert("x".to_string(), HashSet::from(["y".to_string()]));
        adjacency.insert("y".to_string(), HashSet::from(["x".to_string(), "z".to_string()]));
        adjacency.insert("z".to_string(), HashSet::from(["y".to_string()]));
        
        let csp = CSPProblem {
            variables,
            constraints,
            constraint_graph: ConstraintGraph {
                adjacency,
                hyperedges: vec![HashSet::from(["x".to_string(), "y".to_string()]), 
                                HashSet::from(["y".to_string(), "z".to_string()])],
            },
        };
        
        let result = decomposer.decompose(&csp);
        assert!(result.is_ok());
        
        let decomposition = result.unwrap();
        assert!(!decomposition.clusters.is_empty());
    }

    #[test]
    fn test_constraint_clustering() {
        let decomposer = ConstraintSatisfactionDecomposer::new()
            .with_strategy(CSPDecompositionStrategy::ConstraintClustering);
        
        // Create CSP with multiple constraints
        let mut variables = HashMap::new();
        for i in 0..5 {
            let name = format!("x{}", i);
            variables.insert(name.clone(), VariableInfo {
                name: name.clone(),
                domain: vec![0, 1, 2],
                current_domain: vec![0, 1, 2],
            });
        }
        
        let constraints = vec![
            CSPConstraint {
                id: 0,
                scope: vec!["x0".to_string(), "x1".to_string()],
                constraint_type: ConstraintType::AllDifferent,
                tuples: None,
            },
            CSPConstraint {
                id: 1,
                scope: vec!["x1".to_string(), "x2".to_string()],
                constraint_type: ConstraintType::AllDifferent,
                tuples: None,
            },
            CSPConstraint {
                id: 2,
                scope: vec!["x3".to_string(), "x4".to_string()],
                constraint_type: ConstraintType::AllDifferent,
                tuples: None,
            },
        ];
        
        let csp = CSPProblem {
            variables,
            constraints,
            constraint_graph: ConstraintGraph {
                adjacency: HashMap::new(),
                hyperedges: Vec::new(),
            },
        };
        
        let result = decomposer.decompose(&csp);
        assert!(result.is_ok());
        
        let decomposition = result.unwrap();
        // Should create at least 2 clusters due to disconnected constraints
        assert!(decomposition.clusters.len() >= 2);
    }
}
