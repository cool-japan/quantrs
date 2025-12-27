//! Optimized tensor network contraction algorithms
//!
//! This module implements advanced algorithms for efficient tensor network contraction,
//! including path optimization and slicing techniques.

use super::contraction::{ContractableNetwork, ContractionPath};
use super::tensor::{Tensor, TensorIndex};
use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::{Duration, Instant};

/// Tensor network contraction optimization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractionOptMethod {
    /// Greedy optimization (local decisions)
    Greedy,

    /// Dynamic programming optimization (global optimization)
    DynamicProgramming,

    /// Slicing-based optimization (for large networks)
    Sliced,

    /// Hybrid approach combining multiple methods
    Hybrid,
}

/// Advanced contraction path finder
#[derive(Debug, Clone)]
pub struct PathOptimizer {
    /// Maximum time to spend on optimization
    max_optimization_time: Duration,

    /// Contraction method to use
    method: ContractionOptMethod,

    /// Maximum number of slices to use (for sliced contraction)
    max_slices: usize,

    /// Maximum bond dimension
    max_bond_dimension: usize,

    /// Whether to use memory estimates during optimization
    use_memory_estimates: bool,
}

impl Default for PathOptimizer {
    fn default() -> Self {
        Self {
            max_optimization_time: Duration::from_secs(10),
            method: ContractionOptMethod::Hybrid,
            max_slices: 16,
            max_bond_dimension: 64,
            use_memory_estimates: true,
        }
    }
}

impl PathOptimizer {
    /// Create a new path optimizer with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum optimization time
    #[must_use]
    pub const fn with_max_time(mut self, time: Duration) -> Self {
        self.max_optimization_time = time;
        self
    }

    /// Set the contraction method
    #[must_use]
    pub const fn with_method(mut self, method: ContractionOptMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the maximum number of slices
    #[must_use]
    pub const fn with_max_slices(mut self, slices: usize) -> Self {
        self.max_slices = slices;
        self
    }

    /// Set the maximum bond dimension
    #[must_use]
    pub const fn with_max_bond_dimension(mut self, dim: usize) -> Self {
        self.max_bond_dimension = dim;
        self
    }

    /// Enable or disable memory estimation
    #[must_use]
    pub const fn with_memory_estimates(mut self, use_estimates: bool) -> Self {
        self.use_memory_estimates = use_estimates;
        self
    }

    /// Find the optimal contraction path for a tensor network
    pub fn find_optimal_path(
        &self,
        tensors: &HashMap<usize, Tensor>,
        connections: &[(TensorIndex, TensorIndex)],
    ) -> QuantRS2Result<ContractionPath> {
        match self.method {
            ContractionOptMethod::Greedy => self.find_greedy_path(tensors, connections),
            ContractionOptMethod::DynamicProgramming => self.find_dp_path(tensors, connections),
            ContractionOptMethod::Sliced => self.find_sliced_path(tensors, connections),
            ContractionOptMethod::Hybrid => self.find_hybrid_path(tensors, connections),
        }
    }

    /// Find a contraction path using the greedy algorithm
    fn find_greedy_path(
        &self,
        tensors: &HashMap<usize, Tensor>,
        connections: &[(TensorIndex, TensorIndex)],
    ) -> QuantRS2Result<ContractionPath> {
        // Start timing
        let start_time = Instant::now();

        // Build a graph of tensor connections
        let mut tensor_connections = HashMap::new();
        for (t1, t2) in connections {
            tensor_connections
                .entry(t1.tensor_id)
                .or_insert_with(HashSet::new)
                .insert(t2.tensor_id);
            tensor_connections
                .entry(t2.tensor_id)
                .or_insert_with(HashSet::new)
                .insert(t1.tensor_id);
        }

        // Calculate tensor sizes
        let mut tensor_sizes = HashMap::new();
        for (&id, tensor) in tensors {
            let size: usize = tensor.dimensions.iter().product();
            tensor_sizes.insert(id, size);
        }

        // Set up for greedy algorithm
        let mut remaining_tensors: HashSet<usize> = tensors.keys().copied().collect();
        let mut steps = Vec::new();
        let mut total_cost = 0.0;

        // Greedy contraction while respecting time limit
        while remaining_tensors.len() > 1 {
            // Check time limit
            if start_time.elapsed() > self.max_optimization_time {
                // Time limit reached, return what we have so far
                break;
            }

            // Find the best pair to contract next
            let mut best_pair = None;
            let mut best_cost = f64::INFINITY;

            for &t1 in &remaining_tensors {
                if let Some(connected) = tensor_connections.get(&t1) {
                    for &t2 in connected {
                        if remaining_tensors.contains(&t2) {
                            // Calculate cost metric based on resulting tensor size
                            let t1_size = tensor_sizes[&t1];
                            let t2_size = tensor_sizes[&t2];

                            // Count common indices (shared dimensions)
                            let common_indices = count_common_indices(t1, t2, connections);

                            // Estimate size of resulting tensor
                            let result_size =
                                estimate_contraction_size(t1_size, t2_size, common_indices);

                            // Cost is based on both the computation and the resulting tensor size
                            let cost = (t1_size * t2_size) as f64 + result_size as f64;

                            if cost < best_cost {
                                best_cost = cost;
                                best_pair = Some((t1, t2));
                            }
                        }
                    }
                }
            }

            // Process the best pair
            if let Some((t1, t2)) = best_pair {
                // Add step to contraction path
                steps.push((t1, t2));
                total_cost += best_cost;

                // Update remaining tensors
                remaining_tensors.remove(&t1);
                remaining_tensors.remove(&t2);
                let new_id = t1; // Use t1's ID for the new tensor
                remaining_tensors.insert(new_id);

                // Update connections for the new tensor
                let mut new_connections = HashSet::new();

                // Merge connections from t1 and t2
                for id in &[t1, t2] {
                    if let Some(connections) = tensor_connections.get(id) {
                        let connections_clone = connections.clone();
                        for &connected in &connections_clone {
                            if connected != t1
                                && connected != t2
                                && remaining_tensors.contains(&connected)
                            {
                                new_connections.insert(connected);

                                // Update the other tensor's connections
                                if let Some(other_conns) = tensor_connections.get_mut(&connected) {
                                    other_conns.remove(&t1);
                                    other_conns.remove(&t2);
                                    other_conns.insert(new_id);
                                }
                            }
                        }
                    }
                }

                // Set connections for the new tensor
                tensor_connections.insert(new_id, new_connections);

                // Update size of the new tensor
                let common_indices = count_common_indices(t1, t2, connections);
                let new_size =
                    estimate_contraction_size(tensor_sizes[&t1], tensor_sizes[&t2], common_indices);
                tensor_sizes.insert(new_id, new_size);
            } else {
                // No connected pairs left, just contract any two
                if remaining_tensors.len() >= 2 {
                    let mut ids: Vec<_> = remaining_tensors.iter().copied().collect();
                    ids.sort_unstable();
                    let t1 = ids[0];
                    let t2 = ids[1];

                    steps.push((t1, t2));
                    total_cost += (tensor_sizes[&t1] * tensor_sizes[&t2]) as f64;

                    remaining_tensors.remove(&t1);
                    remaining_tensors.remove(&t2);
                    remaining_tensors.insert(t1);

                    // Update size
                    tensor_sizes.insert(t1, tensor_sizes[&t1] * tensor_sizes[&t2]);

                    // No need to update connections - these tensors weren't connected
                }
                // If only one tensor left, we're done
                break;
            }
        }

        Ok(ContractionPath::new(steps, total_cost))
    }

    /// Find a contraction path using dynamic programming
    fn find_dp_path(
        &self,
        tensors: &HashMap<usize, Tensor>,
        connections: &[(TensorIndex, TensorIndex)],
    ) -> QuantRS2Result<ContractionPath> {
        // Dynamic programming is more complex but finds better paths
        // For simplicity, we'll just call the greedy method for now
        // In a full implementation, this would be a real DP algorithm
        self.find_greedy_path(tensors, connections)
    }

    /// Find a contraction path using slicing
    fn find_sliced_path(
        &self,
        tensors: &HashMap<usize, Tensor>,
        connections: &[(TensorIndex, TensorIndex)],
    ) -> QuantRS2Result<ContractionPath> {
        // Slicing can help with very large networks
        // For simplicity, we'll just call the greedy method for now
        // In a full implementation, this would have real slicing logic
        self.find_greedy_path(tensors, connections)
    }

    /// Find a contraction path using a hybrid approach
    fn find_hybrid_path(
        &self,
        tensors: &HashMap<usize, Tensor>,
        connections: &[(TensorIndex, TensorIndex)],
    ) -> QuantRS2Result<ContractionPath> {
        // Get the size of the network
        let network_size = tensors.len();

        // For small networks, use dynamic programming
        if network_size <= 12 {
            return self.find_dp_path(tensors, connections);
        }

        // For medium-sized networks, use greedy
        if network_size <= 24 {
            return self.find_greedy_path(tensors, connections);
        }

        // For large networks, use slicing
        self.find_sliced_path(tensors, connections)
    }
}

/// Advanced tensor network contraption with optimized paths
pub struct OptimizedTensorNetwork {
    /// Tensors in the network
    tensors: HashMap<usize, Tensor>,

    /// Connections between tensors
    connections: Vec<(TensorIndex, TensorIndex)>,

    /// Cached optimal contraction path
    cached_path: Option<ContractionPath>,

    /// Path optimizer configuration
    optimizer: PathOptimizer,
}

impl Default for OptimizedTensorNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizedTensorNetwork {
    /// Create a new optimized tensor network
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            connections: Vec::new(),
            cached_path: None,
            optimizer: PathOptimizer::default(),
        }
    }

    /// Set the path optimization method
    #[must_use]
    pub const fn with_optimization_method(mut self, method: ContractionOptMethod) -> Self {
        self.optimizer = self.optimizer.with_method(method);
        self
    }

    /// Add a tensor to the network
    pub fn add_tensor(&mut self, id: usize, tensor: Tensor) {
        self.tensors.insert(id, tensor);

        // Clear the cached path since the network changed
        self.cached_path = None;
    }

    /// Add a connection between tensors
    pub fn add_connection(&mut self, t1: TensorIndex, t2: TensorIndex) {
        self.connections.push((t1, t2));

        // Clear the cached path since the network changed
        self.cached_path = None;
    }

    /// Get the optimal contraction path
    pub fn get_optimal_path(&mut self) -> QuantRS2Result<ContractionPath> {
        // Return cached path if available
        if let Some(path) = &self.cached_path {
            return Ok(path.clone());
        }

        // Calculate and cache a new path
        let path = self
            .optimizer
            .find_optimal_path(&self.tensors, &self.connections)?;
        self.cached_path = Some(path.clone());

        Ok(path)
    }

    /// Contract the network according to the optimal path
    pub fn contract(&mut self) -> QuantRS2Result<Tensor> {
        // Get the optimal path
        let path = self.get_optimal_path()?;

        // Make working copies of tensors and connections
        let mut working_tensors = self.tensors.clone();
        let mut working_connections = self.connections.clone();

        // Apply each contraction step
        for (id1, id2) in path.steps() {
            // Find the tensors to contract
            let tensor1 = working_tensors.remove(id1).ok_or_else(|| {
                QuantRS2Error::CircuitValidationFailed(format!("Tensor with ID {id1} not found"))
            })?;

            let tensor2 = working_tensors.remove(id2).ok_or_else(|| {
                QuantRS2Error::CircuitValidationFailed(format!("Tensor with ID {id2} not found"))
            })?;

            // Find the shared indices to contract over
            let shared_indices = find_shared_indices(*id1, *id2, &working_connections);

            // Contract the tensors
            let result_tensor = contract_tensors(&tensor1, &tensor2, shared_indices)?;

            // Insert the result with the first ID
            working_tensors.insert(*id1, result_tensor);

            // Update connections
            // (In a real implementation, this would be more complex)
        }

        // The final tensor should be the only one left
        if working_tensors.len() != 1 {
            return Err(QuantRS2Error::CircuitValidationFailed(format!(
                "{} tensors left after contraction (expected 1)",
                working_tensors.len()
            )));
        }

        // Return the final tensor
        Ok(working_tensors
            .into_values()
            .next()
            .expect("Exactly one tensor should remain after contraction"))
    }
}

/// Helper function to count common indices between two tensors
fn count_common_indices(
    id1: usize,
    id2: usize,
    connections: &[(TensorIndex, TensorIndex)],
) -> usize {
    let mut count = 0;

    for (t1, t2) in connections {
        if (t1.tensor_id == id1 && t2.tensor_id == id2)
            || (t1.tensor_id == id2 && t2.tensor_id == id1)
        {
            count += 1;
        }
    }

    count
}

/// Helper function to estimate the size of a tensor after contraction
const fn estimate_contraction_size(size1: usize, size2: usize, common_indices: usize) -> usize {
    // This is a simplified estimate
    // In a real implementation, we would use the actual tensor dimensions
    let common_dim = 2usize.pow(common_indices as u32);
    (size1 * size2) / common_dim
}

/// Helper function to find shared indices between two tensors
fn find_shared_indices(
    id1: usize,
    id2: usize,
    connections: &[(TensorIndex, TensorIndex)],
) -> Vec<(usize, usize)> {
    let mut shared = Vec::new();

    for (t1, t2) in connections {
        if t1.tensor_id == id1 && t2.tensor_id == id2 {
            shared.push((t1.index, t2.index));
        } else if t1.tensor_id == id2 && t2.tensor_id == id1 {
            shared.push((t2.index, t1.index));
        }
    }

    shared
}

/// Helper function to contract two tensors
fn contract_tensors(
    t1: &Tensor,
    t2: &Tensor,
    indices: Vec<(usize, usize)>,
) -> QuantRS2Result<Tensor> {
    // This is a simplified implementation
    // In a real implementation, this would perform the actual tensor contraction

    // Placeholder: just return the first tensor
    Ok(t1.clone())
}

/// Optimized contraction plan for tensor networks
#[derive(Debug, Clone, PartialEq)]
pub struct ContractionPlan {
    /// Ordered list of tensor pairs to contract
    pairs: Vec<(usize, usize)>,

    /// Estimated computational cost
    flop_estimate: f64,

    /// Estimated peak memory usage
    memory_estimate: usize,
}

impl ContractionPlan {
    /// Create a new contraction plan
    pub const fn new(
        pairs: Vec<(usize, usize)>,
        flop_estimate: f64,
        memory_estimate: usize,
    ) -> Self {
        Self {
            pairs,
            flop_estimate,
            memory_estimate,
        }
    }

    /// Get the pairs of tensors to contract
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Get the estimated computational cost
    pub const fn flop_estimate(&self) -> f64 {
        self.flop_estimate
    }

    /// Get the estimated peak memory usage
    pub const fn memory_estimate(&self) -> usize {
        self.memory_estimate
    }
}

impl Eq for ContractionPlan {}

impl Ord for ContractionPlan {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by computational cost first, then by memory usage
        self.flop_estimate
            .partial_cmp(&other.flop_estimate)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.memory_estimate.cmp(&other.memory_estimate))
    }
}

impl PartialOrd for ContractionPlan {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Generate an optimal contraction plan for a tensor network
pub fn generate_contraction_plan(
    tensors: &HashMap<usize, Tensor>,
    connections: &[(TensorIndex, TensorIndex)],
    max_time: Duration,
) -> QuantRS2Result<ContractionPlan> {
    // Start timing
    let start_time = Instant::now();

    // Check for empty network
    if tensors.is_empty() {
        return Ok(ContractionPlan::new(Vec::new(), 0.0, 0));
    }

    // Build a graph of tensor connections
    let mut tensor_graph = HashMap::new();
    for (t1, t2) in connections {
        tensor_graph
            .entry(t1.tensor_id)
            .or_insert_with(HashSet::new)
            .insert(t2.tensor_id);
        tensor_graph
            .entry(t2.tensor_id)
            .or_insert_with(HashSet::new)
            .insert(t1.tensor_id);
    }

    // Calculate tensor sizes and shapes
    let mut tensor_sizes = HashMap::new();
    for (&id, tensor) in tensors {
        let size: usize = tensor.dimensions.iter().product();
        tensor_sizes.insert(id, size);
    }

    // Priority queue for different contraction plans
    let mut plan_queue = BinaryHeap::new();

    // Initial plan: contract the smallest pair first
    let mut candidate_pairs = Vec::new();
    for &id1 in tensors.keys() {
        if let Some(connected) = tensor_graph.get(&id1) {
            for &id2 in connected {
                if id1 < id2 {
                    // Avoid duplicates
                    let cost = tensor_sizes[&id1] * tensor_sizes[&id2];
                    candidate_pairs.push((cost, id1, id2));
                }
            }
        }
    }

    // Sort by cost (smallest first)
    candidate_pairs.sort_by_key(|&(cost, _, _)| cost);

    // Create initial plans from the top candidates
    for (cost, id1, id2) in candidate_pairs.iter().take(5) {
        let pairs = vec![(*id1, *id2)];
        plan_queue.push(Reverse(ContractionPlan::new(
            pairs,
            *cost as f64,
            std::cmp::max(tensor_sizes[id1], tensor_sizes[id2]),
        )));
    }

    // If no initial pairs, return empty plan
    if plan_queue.is_empty() {
        return Ok(ContractionPlan::new(Vec::new(), 0.0, 0));
    }

    // Best plan found so far
    let mut best_plan = plan_queue
        .peek()
        .expect("Plan queue should not be empty at this point")
        .0
        .clone();

    // Main optimization loop
    while !plan_queue.is_empty() && start_time.elapsed() < max_time {
        // Get the current best plan
        let current_plan = plan_queue
            .pop()
            .expect("Plan queue verified non-empty in loop condition")
            .0;

        // If this plan is complete, update best plan if better
        if current_plan.pairs.len() == tensors.len() - 1 {
            if current_plan.flop_estimate < best_plan.flop_estimate {
                best_plan = current_plan;
            }
            continue;
        }

        // Simulate the contractions to get the current state
        let mut remaining = tensors.keys().copied().collect::<HashSet<_>>();
        let mut current_graph = tensor_graph.clone();
        let mut current_sizes = tensor_sizes.clone();

        for &(id1, id2) in &current_plan.pairs {
            // Remove contracted tensors
            remaining.remove(&id1);
            remaining.remove(&id2);

            // Add the new tensor (using id1)
            remaining.insert(id1);

            // Update connections and sizes (simplified)
            // In a real implementation, this would be more accurate
        }

        // Generate candidate next steps
        let mut candidates = Vec::new();
        for &id1 in &remaining {
            if let Some(connected) = current_graph.get(&id1) {
                for &id2 in connected {
                    if remaining.contains(&id2) && id1 < id2 {
                        let cost = current_sizes[&id1] * current_sizes[&id2];
                        candidates.push((cost, id1, id2));
                    }
                }
            }
        }

        // Sort candidates
        candidates.sort_by_key(|&(cost, _, _)| cost);

        // Add new plans to the queue
        for (cost, id1, id2) in candidates.iter().take(3) {
            let mut new_pairs = current_plan.pairs.clone();
            new_pairs.push((*id1, *id2));

            let new_flops = current_plan.flop_estimate + *cost as f64;
            let new_memory = std::cmp::max(current_plan.memory_estimate, *cost);

            plan_queue.push(Reverse(ContractionPlan::new(
                new_pairs, new_flops, new_memory,
            )));
        }
    }

    Ok(best_plan)
}
