//! Enhanced hardware topology analysis with SciRS2-style graph algorithms.
//!
//! This module provides advanced topology analysis capabilities including
//! graph metrics, community detection, and optimal qubit allocation strategies.

use std::collections::{HashMap, HashSet, VecDeque};
use crate::topology::{HardwareTopology, QubitProperties, GateProperties};
use crate::DeviceResult;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::algo::{bellman_ford, floyd_warshall};
use petgraph::visit::EdgeRef;

/// Advanced topology analyzer with SciRS2-style algorithms
pub struct TopologyAnalyzer {
    /// The hardware topology to analyze
    topology: HardwareTopology,
    /// Cached distance matrix
    distance_matrix: Option<Vec<Vec<f64>>>,
    /// Cached betweenness centrality
    betweenness: Option<HashMap<u32, f64>>,
    /// Cached clustering coefficients
    clustering: Option<HashMap<u32, f64>>,
}

/// Analysis results for hardware topology
#[derive(Debug, Clone)]
pub struct TopologyAnalysis {
    /// Graph diameter (maximum shortest path)
    pub diameter: usize,
    /// Average shortest path length
    pub average_path_length: f64,
    /// Graph density
    pub density: f64,
    /// Number of connected components
    pub components: usize,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Most central qubits (by betweenness)
    pub central_qubits: Vec<u32>,
    /// Peripheral qubits (far from center)
    pub peripheral_qubits: Vec<u32>,
    /// Qubit communities/clusters
    pub communities: Vec<HashSet<u32>>,
    /// Hardware-specific metrics
    pub hardware_metrics: HardwareMetrics,
}

/// Hardware-specific performance metrics
#[derive(Debug, Clone)]
pub struct HardwareMetrics {
    /// Average gate error rate
    pub avg_gate_error: f64,
    /// Average T1 time
    pub avg_t1: f64,
    /// Average T2 time
    pub avg_t2: f64,
    /// Qubit quality scores (0-1)
    pub qubit_scores: HashMap<u32, f64>,
    /// Connection quality scores
    pub connection_scores: HashMap<(u32, u32), f64>,
    /// Recommended qubits for critical operations
    pub premium_qubits: Vec<u32>,
}

/// Qubit allocation strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AllocationStrategy {
    /// Minimize average distance
    MinimizeDistance,
    /// Maximize qubit quality
    MaximizeQuality,
    /// Balance distance and quality
    Balanced,
    /// Use central qubits
    CentralFirst,
    /// Minimize crosstalk
    MinimizeCrosstalk,
}

impl TopologyAnalyzer {
    /// Create a new topology analyzer
    pub fn new(topology: HardwareTopology) -> Self {
        Self {
            topology,
            distance_matrix: None,
            betweenness: None,
            clustering: None,
        }
    }

    /// Perform comprehensive topology analysis
    pub fn analyze(&mut self) -> DeviceResult<TopologyAnalysis> {
        // Calculate distance matrix if not cached
        if self.distance_matrix.is_none() {
            self.calculate_distance_matrix();
        }

        // Calculate graph metrics
        let diameter = self.calculate_diameter();
        let avg_path_length = self.calculate_average_path_length();
        let density = self.calculate_density();
        let components = self.count_components();
        
        // Calculate centrality measures
        self.calculate_betweenness_centrality();
        let central_qubits = self.find_central_qubits(5);
        let peripheral_qubits = self.find_peripheral_qubits(5);
        
        // Calculate clustering
        self.calculate_clustering_coefficients();
        let clustering_coefficient = self.global_clustering_coefficient();
        
        // Find communities
        let communities = self.detect_communities();
        
        // Calculate hardware metrics
        let hardware_metrics = self.calculate_hardware_metrics();

        Ok(TopologyAnalysis {
            diameter,
            average_path_length: avg_path_length,
            density,
            components,
            clustering_coefficient,
            central_qubits,
            peripheral_qubits,
            communities,
            hardware_metrics,
        })
    }

    /// Calculate all-pairs shortest path distances
    fn calculate_distance_matrix(&mut self) {
        let n = self.topology.num_qubits;
        let mut matrix = vec![vec![f64::INFINITY; n]; n];
        
        // Initialize diagonal
        for i in 0..n {
            matrix[i][i] = 0.0;
        }
        
        // Initialize edges
        for edge in self.topology.connectivity.edge_references() {
            let (a, b) = (edge.source(), edge.target());
            let q1_id = self.topology.connectivity[a];
            let q2_id = self.topology.connectivity[b];
            
            // Find the index in the qubit properties list
            let q1 = self.topology.qubit_properties.iter()
                .position(|q| q.id == q1_id)
                .unwrap_or(q1_id as usize);
            let q2 = self.topology.qubit_properties.iter()
                .position(|q| q.id == q2_id)
                .unwrap_or(q2_id as usize);
                
            let weight = *edge.weight();
            
            matrix[q1][q2] = weight;
            matrix[q2][q1] = weight;
        }
        
        // Floyd-Warshall algorithm
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if matrix[i][k] + matrix[k][j] < matrix[i][j] {
                        matrix[i][j] = matrix[i][k] + matrix[k][j];
                    }
                }
            }
        }
        
        self.distance_matrix = Some(matrix);
    }

    /// Calculate graph diameter
    fn calculate_diameter(&self) -> usize {
        // For graph diameter, we need hop count, not weighted distance
        // Create an unweighted version of the graph
        let n = self.topology.num_qubits;
        let mut hop_matrix = vec![vec![usize::MAX; n]; n];
        
        // Initialize diagonal
        for i in 0..n {
            hop_matrix[i][i] = 0;
        }
        
        // Initialize direct edges with hop count 1
        for edge in self.topology.connectivity.edge_references() {
            let (a, b) = (edge.source(), edge.target());
            let q1_id = self.topology.connectivity[a];
            let q2_id = self.topology.connectivity[b];
            
            // Find the index in the qubit properties list
            let q1 = self.topology.qubit_properties.iter()
                .position(|q| q.id == q1_id)
                .unwrap_or(q1_id as usize);
            let q2 = self.topology.qubit_properties.iter()
                .position(|q| q.id == q2_id)
                .unwrap_or(q2_id as usize);
            
            hop_matrix[q1][q2] = 1;
            hop_matrix[q2][q1] = 1;
        }
        
        // Floyd-Warshall for hop counts
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if hop_matrix[i][k] != usize::MAX && hop_matrix[k][j] != usize::MAX {
                        let new_dist = hop_matrix[i][k] + hop_matrix[k][j];
                        if new_dist < hop_matrix[i][j] {
                            hop_matrix[i][j] = new_dist;
                        }
                    }
                }
            }
        }
        
        // Find maximum hop count
        hop_matrix
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&d| d != usize::MAX)
            .copied()
            .max()
            .unwrap_or(0)
    }

    /// Calculate average shortest path length
    fn calculate_average_path_length(&self) -> f64 {
        let matrix = self.distance_matrix.as_ref().unwrap();
        let n = self.topology.num_qubits;
        let mut sum = 0.0;
        let mut count = 0;
        
        for i in 0..n {
            for j in i + 1..n {
                if matrix[i][j] != f64::INFINITY {
                    sum += matrix[i][j];
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Calculate graph density
    fn calculate_density(&self) -> f64 {
        let n = self.topology.num_qubits as f64;
        let m = self.topology.connectivity.edge_count() as f64;
        
        if n > 1.0 {
            2.0 * m / (n * (n - 1.0))
        } else {
            0.0
        }
    }

    /// Count connected components
    fn count_components(&self) -> usize {
        use petgraph::algo::connected_components;
        connected_components(&self.topology.connectivity)
    }

    /// Calculate betweenness centrality for all nodes
    fn calculate_betweenness_centrality(&mut self) {
        let n = self.topology.num_qubits;
        let mut betweenness = HashMap::new();
        
        // Initialize
        for i in 0..n {
            betweenness.insert(i as u32, 0.0);
        }
        
        // For each pair of nodes
        for s in 0..n {
            // Single-source shortest paths
            let (distances, predecessors) = self.single_source_shortest_paths(s);
            
            // Accumulate betweenness
            let mut delta = vec![0.0; n];
            
            // Process nodes in reverse topological order (by distance)
            let mut nodes_by_distance: Vec<_> = (0..n)
                .filter(|&v| distances[v] != f64::INFINITY)
                .collect();
            nodes_by_distance.sort_by(|&a, &b| distances[b].partial_cmp(&distances[a]).unwrap());
            
            for &w in &nodes_by_distance {
                for &v in &predecessors[w] {
                    let num_paths = if distances[v] + 1.0 == distances[w] { 1.0 } else { 0.0 };
                    delta[v] += num_paths * (1.0 + delta[w]);
                }
                
                if w != s {
                    *betweenness.get_mut(&(w as u32)).unwrap() += delta[w];
                }
            }
        }
        
        // Normalize
        let norm = if n > 2 { 2.0 / ((n - 1) * (n - 2)) as f64 } else { 1.0 };
        for value in betweenness.values_mut() {
            *value *= norm;
        }
        
        self.betweenness = Some(betweenness);
    }

    /// Single-source shortest paths using BFS
    fn single_source_shortest_paths(&self, source: usize) -> (Vec<f64>, Vec<Vec<usize>>) {
        let n = self.topology.num_qubits;
        let mut distances = vec![f64::INFINITY; n];
        let mut predecessors = vec![Vec::new(); n];
        let mut queue = VecDeque::new();
        
        distances[source] = 0.0;
        queue.push_back(source);
        
        while let Some(u) = queue.pop_front() {
            // Find node in graph
            if let Some(node_u) = self.topology.connectivity.node_indices()
                .find(|&n| self.topology.connectivity[n] == u as u32) {
                
                // Check all neighbors
                for neighbor in self.topology.connectivity.neighbors(node_u) {
                    let v = self.topology.connectivity[neighbor] as usize;
                    
                    if distances[v] == f64::INFINITY {
                        distances[v] = distances[u] + 1.0;
                        predecessors[v].push(u);
                        queue.push_back(v);
                    } else if distances[v] == distances[u] + 1.0 {
                        predecessors[v].push(u);
                    }
                }
            }
        }
        
        (distances, predecessors)
    }

    /// Find most central qubits
    fn find_central_qubits(&self, count: usize) -> Vec<u32> {
        let mut qubits: Vec<_> = self.betweenness
            .as_ref()
            .unwrap()
            .iter()
            .map(|(&q, &b)| (q, b))
            .collect();
        
        qubits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        qubits.into_iter()
            .take(count)
            .map(|(q, _)| q)
            .collect()
    }

    /// Find peripheral qubits
    fn find_peripheral_qubits(&self, count: usize) -> Vec<u32> {
        let mut qubits: Vec<_> = self.betweenness
            .as_ref()
            .unwrap()
            .iter()
            .map(|(&q, &b)| (q, b))
            .collect();
        
        qubits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        qubits.into_iter()
            .take(count)
            .map(|(q, _)| q)
            .collect()
    }

    /// Calculate clustering coefficient for each node
    fn calculate_clustering_coefficients(&mut self) {
        let mut clustering = HashMap::new();
        
        for node in self.topology.connectivity.node_indices() {
            let qubit = self.topology.connectivity[node];
            let neighbors: Vec<_> = self.topology.connectivity
                .neighbors(node)
                .collect();
            
            let k = neighbors.len();
            if k < 2 {
                clustering.insert(qubit, 0.0);
                continue;
            }
            
            // Count edges between neighbors
            let mut edges = 0;
            for i in 0..neighbors.len() {
                for j in i + 1..neighbors.len() {
                    if self.topology.connectivity.contains_edge(neighbors[i], neighbors[j]) {
                        edges += 1;
                    }
                }
            }
            
            let coefficient = 2.0 * edges as f64 / (k * (k - 1)) as f64;
            clustering.insert(qubit, coefficient);
        }
        
        self.clustering = Some(clustering);
    }

    /// Calculate global clustering coefficient
    fn global_clustering_coefficient(&self) -> f64 {
        let coefficients = self.clustering.as_ref().unwrap();
        if coefficients.is_empty() {
            return 0.0;
        }
        
        coefficients.values().sum::<f64>() / coefficients.len() as f64
    }

    /// Detect communities using spectral clustering
    fn detect_communities(&self) -> Vec<HashSet<u32>> {
        // Simple community detection using connected components
        // In a real implementation, we'd use more sophisticated algorithms
        use petgraph::algo::tarjan_scc;
        
        let mut communities = Vec::new();
        let sccs = tarjan_scc(&self.topology.connectivity);
        
        for scc in sccs {
            let community: HashSet<u32> = scc
                .into_iter()
                .map(|n| self.topology.connectivity[n])
                .collect();
            communities.push(community);
        }
        
        communities
    }

    /// Calculate hardware-specific metrics
    fn calculate_hardware_metrics(&self) -> HardwareMetrics {
        let mut total_gate_error = 0.0;
        let mut gate_count = 0;
        
        for (_, props) in &self.topology.gate_properties {
            total_gate_error += props.error_rate;
            gate_count += 1;
        }
        
        let avg_gate_error = if gate_count > 0 {
            total_gate_error / gate_count as f64
        } else {
            0.0
        };
        
        // Calculate average coherence times
        let avg_t1 = self.topology.qubit_properties.iter()
            .map(|q| q.t1)
            .sum::<f64>() / self.topology.qubit_properties.len() as f64;
        
        let avg_t2 = self.topology.qubit_properties.iter()
            .map(|q| q.t2)
            .sum::<f64>() / self.topology.qubit_properties.len() as f64;
        
        // Calculate qubit quality scores
        let mut qubit_scores = HashMap::new();
        for qubit in &self.topology.qubit_properties {
            let score = self.calculate_qubit_score(qubit);
            qubit_scores.insert(qubit.id, score);
        }
        
        // Calculate connection quality scores
        let mut connection_scores = HashMap::new();
        for ((q1, q2), props) in &self.topology.gate_properties {
            let score = 1.0 - props.error_rate; // Simple scoring
            connection_scores.insert((*q1, *q2), score);
        }
        
        // Find premium qubits (top 20%)
        let mut sorted_qubits: Vec<_> = qubit_scores.iter()
            .map(|(&id, &score)| (id, score))
            .collect();
        sorted_qubits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let premium_count = (sorted_qubits.len() as f64 * 0.2).ceil() as usize;
        let premium_qubits = sorted_qubits.into_iter()
            .take(premium_count)
            .map(|(id, _)| id)
            .collect();
        
        HardwareMetrics {
            avg_gate_error,
            avg_t1,
            avg_t2,
            qubit_scores,
            connection_scores,
            premium_qubits,
        }
    }

    /// Calculate quality score for a qubit
    fn calculate_qubit_score(&self, qubit: &QubitProperties) -> f64 {
        // Weighted scoring based on various metrics
        let t1_score = (qubit.t1 / 100.0).min(1.0); // Normalize to 100μs
        let t2_score = (qubit.t2 / 100.0).min(1.0);
        let gate_score = 1.0 - qubit.single_qubit_gate_error;
        let readout_score = 1.0 - qubit.readout_error;
        
        // Weighted average
        0.3 * t1_score + 0.3 * t2_score + 0.2 * gate_score + 0.2 * readout_score
    }

    /// Find optimal qubit allocation for a given strategy
    pub fn allocate_qubits(
        &mut self,
        num_logical_qubits: usize,
        strategy: AllocationStrategy,
    ) -> DeviceResult<Vec<u32>> {
        if num_logical_qubits > self.topology.num_qubits {
            return Err(crate::DeviceError::InsufficientQubits {
                required: num_logical_qubits,
                available: self.topology.num_qubits,
            });
        }

        // Ensure analysis is performed
        let analysis = self.analyze()?;
        
        match strategy {
            AllocationStrategy::MinimizeDistance => {
                self.allocate_minimize_distance(num_logical_qubits)
            }
            AllocationStrategy::MaximizeQuality => {
                self.allocate_maximize_quality(num_logical_qubits, &analysis)
            }
            AllocationStrategy::Balanced => {
                self.allocate_balanced(num_logical_qubits, &analysis)
            }
            AllocationStrategy::CentralFirst => {
                Ok(analysis.central_qubits.into_iter()
                    .take(num_logical_qubits)
                    .collect())
            }
            AllocationStrategy::MinimizeCrosstalk => {
                self.allocate_minimize_crosstalk(num_logical_qubits)
            }
        }
    }

    /// Allocate qubits to minimize average distance
    fn allocate_minimize_distance(&self, num_qubits: usize) -> DeviceResult<Vec<u32>> {
        // Find connected subgraph with minimum average distance
        let matrix = self.distance_matrix.as_ref().unwrap();
        let n = self.topology.num_qubits;
        
        let mut best_allocation = Vec::new();
        let mut best_score = f64::INFINITY;
        
        // Try different starting points
        for start in 0..n {
            let mut allocation = vec![start as u32];
            let mut remaining: HashSet<_> = (0..n).filter(|&i| i != start).collect();
            
            // Greedy selection
            while allocation.len() < num_qubits && !remaining.is_empty() {
                let mut best_next = None;
                let mut best_dist = f64::INFINITY;
                
                for &candidate in &remaining {
                    let total_dist: f64 = allocation.iter()
                        .map(|&q| matrix[q as usize][candidate])
                        .sum();
                    
                    if total_dist < best_dist {
                        best_dist = total_dist;
                        best_next = Some(candidate);
                    }
                }
                
                if let Some(next) = best_next {
                    allocation.push(next as u32);
                    remaining.remove(&next);
                }
            }
            
            // Calculate average distance
            let mut total_dist = 0.0;
            let mut count = 0;
            for i in 0..allocation.len() {
                for j in i + 1..allocation.len() {
                    total_dist += matrix[allocation[i] as usize][allocation[j] as usize];
                    count += 1;
                }
            }
            
            let avg_dist = if count > 0 { total_dist / count as f64 } else { 0.0 };
            
            if avg_dist < best_score {
                best_score = avg_dist;
                best_allocation = allocation;
            }
        }
        
        Ok(best_allocation)
    }

    /// Allocate qubits to maximize quality
    fn allocate_maximize_quality(
        &self,
        num_qubits: usize,
        analysis: &TopologyAnalysis,
    ) -> DeviceResult<Vec<u32>> {
        // Simply take the highest quality qubits
        let mut qubits: Vec<_> = analysis.hardware_metrics.qubit_scores
            .iter()
            .map(|(&id, &score)| (id, score))
            .collect();
        
        qubits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(qubits.into_iter()
            .take(num_qubits)
            .map(|(id, _)| id)
            .collect())
    }

    /// Allocate qubits with balanced strategy
    fn allocate_balanced(
        &self,
        num_qubits: usize,
        analysis: &TopologyAnalysis,
    ) -> DeviceResult<Vec<u32>> {
        // Score qubits by quality and centrality
        let mut scores = HashMap::new();
        
        for (&qubit, &quality) in &analysis.hardware_metrics.qubit_scores {
            let centrality = self.betweenness.as_ref().unwrap().get(&qubit).unwrap_or(&0.0);
            // Balanced scoring
            let score = 0.6 * quality + 0.4 * centrality;
            scores.insert(qubit, score);
        }
        
        let mut sorted: Vec<_> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(sorted.into_iter()
            .take(num_qubits)
            .map(|(id, _)| id)
            .collect())
    }

    /// Allocate qubits to minimize crosstalk
    fn allocate_minimize_crosstalk(&self, num_qubits: usize) -> DeviceResult<Vec<u32>> {
        // Select qubits that are well-separated
        let matrix = self.distance_matrix.as_ref().unwrap();
        let n = self.topology.num_qubits;
        
        let mut allocation = Vec::new();
        let mut remaining: HashSet<_> = (0..n).collect();
        
        // Start with a random qubit
        let start = 0;
        allocation.push(start as u32);
        remaining.remove(&start);
        
        // Greedily select qubits that maximize minimum distance
        while allocation.len() < num_qubits && !remaining.is_empty() {
            let mut best_candidate = None;
            let mut best_min_dist = 0.0;
            
            for &candidate in &remaining {
                let min_dist = allocation.iter()
                    .map(|&q| matrix[q as usize][candidate])
                    .fold(f64::INFINITY, f64::min);
                
                if min_dist > best_min_dist {
                    best_min_dist = min_dist;
                    best_candidate = Some(candidate);
                }
            }
            
            if let Some(candidate) = best_candidate {
                allocation.push(candidate as u32);
                remaining.remove(&candidate);
            }
        }
        
        Ok(allocation)
    }

    /// Get recommended paths between qubits
    pub fn recommend_swap_paths(&self, source: u32, target: u32) -> Vec<Vec<u32>> {
        // Find k shortest paths for SWAP operations
        // Note: This doesn't use the distance matrix, it finds actual paths
        let n = self.topology.num_qubits;
        
        // Simple BFS for shortest paths
        let mut paths = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((vec![source], HashSet::new()));
        
        while let Some((path, mut visited)) = queue.pop_front() {
            let current = *path.last().unwrap();
            
            if current == target {
                paths.push(path);
                if paths.len() >= 3 { // Return up to 3 paths
                    break;
                }
                continue;
            }
            
            visited.insert(current);
            
            // Find neighbors
            if let Some(node) = self.topology.connectivity.node_indices()
                .find(|&n| self.topology.connectivity[n] == current) {
                
                for neighbor in self.topology.connectivity.neighbors(node) {
                    let next = self.topology.connectivity[neighbor];
                    if !visited.contains(&next) {
                        let mut new_path = path.clone();
                        new_path.push(next);
                        queue.push_back((new_path, visited.clone()));
                    }
                }
            }
        }
        
        paths
    }
}

/// Create standard hardware topologies
pub fn create_standard_topology(topology_type: &str, size: usize) -> DeviceResult<HardwareTopology> {
    match topology_type {
        "linear" => create_linear_topology(size),
        "grid" => create_grid_topology(size),
        "heavy_hex" => create_heavy_hex_topology(size),
        _ => Err(crate::DeviceError::UnsupportedDevice(
            format!("Unknown topology type: {}", topology_type)
        )),
    }
}

/// Create a linear (chain) topology
fn create_linear_topology(size: usize) -> DeviceResult<HardwareTopology> {
    let mut topology = HardwareTopology::new(size);
    
    // Add qubits
    for i in 0..size {
        let props = QubitProperties {
            id: i as u32,
            index: i as u32,
            t1: 50.0 + (i as f64) * 2.0, // Varying T1
            t2: 30.0 + (i as f64) * 1.5, // Varying T2
            single_qubit_gate_error: 0.001,
            gate_error_1q: 0.001,
            readout_error: 0.01,
            frequency: 5.0 + (i as f64) * 0.01,
        };
        topology.add_qubit(props);
    }
    
    // Add connections
    for i in 0..size - 1 {
        let props = GateProperties {
            error_rate: 0.01,
            duration: 200.0,
            gate_type: "CZ".to_string(),
        };
        topology.add_connection(i as u32, (i + 1) as u32, props);
    }
    
    Ok(topology)
}

/// Create a 2D grid topology
fn create_grid_topology(size: usize) -> DeviceResult<HardwareTopology> {
    let side = (size as f64).sqrt().ceil() as usize;
    let mut topology = HardwareTopology::new(size);
    
    // Add qubits
    for i in 0..size {
        let props = QubitProperties {
            id: i as u32,
            index: i as u32,
            t1: 40.0 + (i as f64) * 1.5,
            t2: 25.0 + (i as f64) * 1.0,
            single_qubit_gate_error: 0.001,
            gate_error_1q: 0.001,
            readout_error: 0.01,
            frequency: 5.0 + (i as f64) * 0.01,
        };
        topology.add_qubit(props);
    }
    
    // Add grid connections
    for row in 0..side {
        for col in 0..side {
            let idx = row * side + col;
            if idx >= size {
                break;
            }
            
            // Connect to right neighbor
            if col + 1 < side && idx + 1 < size {
                let props = GateProperties {
                    error_rate: 0.01,
                    duration: 200.0,
                    gate_type: "CZ".to_string(),
                };
                topology.add_connection(idx as u32, (idx + 1) as u32, props);
            }
            
            // Connect to bottom neighbor
            if row + 1 < side && idx + side < size {
                let props = GateProperties {
                    error_rate: 0.01,
                    duration: 200.0,
                    gate_type: "CZ".to_string(),
                };
                topology.add_connection(idx as u32, (idx + side) as u32, props);
            }
        }
    }
    
    Ok(topology)
}

/// Create a heavy-hexagon topology (IBM-style)
fn create_heavy_hex_topology(size: usize) -> DeviceResult<HardwareTopology> {
    // Simplified heavy-hex for demonstration
    let mut topology = HardwareTopology::new(size);
    
    // Add qubits with varying quality
    for i in 0..size {
        let props = QubitProperties {
            id: i as u32,
            index: i as u32,
            t1: 30.0 + (i as f64) * 3.0,
            t2: 20.0 + (i as f64) * 2.0,
            single_qubit_gate_error: 0.001 + (i as f64) * 0.0001,
            gate_error_1q: 0.001 + (i as f64) * 0.0001,
            readout_error: 0.01 + (i as f64) * 0.001,
            frequency: 5.0 + (i as f64) * 0.01,
        };
        topology.add_qubit(props);
    }
    
    // Create hexagonal connections (simplified)
    for i in 0..size {
        // Connect in a pattern that creates hexagons
        let connections = match i % 6 {
            0 => vec![1, 5],
            1 => vec![0, 2],
            2 => vec![1, 3],
            3 => vec![2, 4],
            4 => vec![3, 5],
            5 => vec![4, 0],
            _ => vec![],
        };
        
        for &j in &connections {
            if i < j && j < size {
                let props = GateProperties {
                    error_rate: 0.01 + (i + j) as f64 * 0.0001,
                    duration: 200.0,
                    gate_type: "CZ".to_string(),
                };
                topology.add_connection(i as u32, j as u32, props);
            }
        }
    }
    
    Ok(topology)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_topology_analysis() {
        let topology = create_linear_topology(5).unwrap();
        let mut analyzer = TopologyAnalyzer::new(topology);
        let analysis = analyzer.analyze().unwrap();
        
        assert_eq!(analysis.diameter, 4); // Linear chain of 5 qubits
        assert_eq!(analysis.components, 1); // Fully connected
        assert!(analysis.density < 0.5); // Sparse connectivity
    }

    #[test]
    fn test_grid_topology_analysis() {
        let topology = create_grid_topology(9).unwrap(); // 3x3 grid
        let mut analyzer = TopologyAnalyzer::new(topology);
        let analysis = analyzer.analyze().unwrap();
        
        assert_eq!(analysis.components, 1);
        assert!(analysis.diameter <= 4); // Maximum distance in 3x3 grid
        // Grid topology has no triangles, so clustering coefficient is 0
        assert_eq!(analysis.clustering_coefficient, 0.0);
    }

    #[test]
    fn test_qubit_allocation() {
        let topology = create_grid_topology(9).unwrap();
        let mut analyzer = TopologyAnalyzer::new(topology);
        
        // Test different allocation strategies
        let alloc1 = analyzer.allocate_qubits(4, AllocationStrategy::MinimizeDistance).unwrap();
        assert_eq!(alloc1.len(), 4);
        
        let alloc2 = analyzer.allocate_qubits(4, AllocationStrategy::MaximizeQuality).unwrap();
        assert_eq!(alloc2.len(), 4);
        
        let alloc3 = analyzer.allocate_qubits(4, AllocationStrategy::CentralFirst).unwrap();
        assert_eq!(alloc3.len(), 4);
    }

    #[test]
    fn test_swap_paths() {
        let topology = create_linear_topology(5).unwrap();
        let analyzer = TopologyAnalyzer::new(topology);
        
        let paths = analyzer.recommend_swap_paths(0, 4);
        assert!(!paths.is_empty());
        assert_eq!(paths[0].len(), 5); // 0 -> 1 -> 2 -> 3 -> 4
    }
}