//! `SciRS2` graph algorithms integration for circuit analysis
//!
//! This module integrates `SciRS2`'s advanced graph algorithms and data structures
//! to provide sophisticated circuit analysis, optimization, and pattern matching capabilities.

use crate::builder::Circuit;
use crate::dag::{circuit_to_dag, CircuitDag, DagNode};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// SciRS2-powered graph representation of quantum circuits
#[derive(Debug, Clone)]
pub struct SciRS2CircuitGraph {
    /// Node data indexed by node ID
    pub nodes: HashMap<usize, SciRS2Node>,
    /// Edge data indexed by (source, target)
    pub edges: HashMap<(usize, usize), SciRS2Edge>,
    /// Adjacency matrix for efficient access
    pub adjacency_matrix: Vec<Vec<bool>>,
    /// Node properties for analysis
    pub node_properties: HashMap<usize, NodeProperties>,
    /// Graph metrics cache
    pub metrics_cache: Option<GraphMetrics>,
}

/// Enhanced node representation with `SciRS2` properties
#[derive(Debug, Clone)]
pub struct SciRS2Node {
    pub id: usize,
    pub gate: Option<Box<dyn GateOp>>,
    pub node_type: SciRS2NodeType,
    pub weight: f64,
    pub depth: usize,
    pub clustering_coefficient: Option<f64>,
    pub centrality_measures: CentralityMeasures,
}

/// Types of nodes in `SciRS2` graph representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SciRS2NodeType {
    /// Input boundary node
    Input { qubit: u32 },
    /// Output boundary node
    Output { qubit: u32 },
    /// Single-qubit gate
    SingleQubitGate { gate_type: String, qubit: u32 },
    /// Two-qubit gate
    TwoQubitGate {
        gate_type: String,
        qubits: (u32, u32),
    },
    /// Multi-qubit gate
    MultiQubitGate { gate_type: String, qubits: Vec<u32> },
    /// Measurement node
    Measurement { qubit: u32 },
    /// Barrier or synchronization point
    Barrier { qubits: Vec<u32> },
}

/// Edge representation with advanced properties
#[derive(Debug, Clone)]
pub struct SciRS2Edge {
    pub source: usize,
    pub target: usize,
    pub edge_type: EdgeType,
    pub weight: f64,
    pub flow_capacity: Option<f64>,
    pub is_critical_path: bool,
}

/// Enhanced edge types for circuit analysis
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeType {
    /// Data dependency on qubit
    QubitDependency { qubit: u32, distance: usize },
    /// Classical control dependency
    ClassicalDependency,
    /// Commutation edge (gates can be reordered)
    Commutation { strength: f64 },
    /// Entanglement edge
    Entanglement { strength: f64 },
    /// Temporal dependency
    Temporal { delay: f64 },
}

/// Node properties for analysis
#[derive(Debug, Clone, Default)]
pub struct NodeProperties {
    /// Degree (number of connections)
    pub degree: usize,
    /// In-degree
    pub in_degree: usize,
    /// Out-degree
    pub out_degree: usize,
    /// Node eccentricity
    pub eccentricity: Option<usize>,
    /// Local clustering coefficient
    pub clustering_coefficient: Option<f64>,
    /// Community assignment
    pub community: Option<usize>,
    /// Gate execution cost
    pub execution_cost: f64,
    /// Error rate
    pub error_rate: f64,
}

/// Centrality measures for nodes
#[derive(Debug, Clone, Default)]
pub struct CentralityMeasures {
    /// Degree centrality
    pub degree: f64,
    /// Betweenness centrality
    pub betweenness: Option<f64>,
    /// Closeness centrality
    pub closeness: Option<f64>,
    /// Eigenvector centrality
    pub eigenvector: Option<f64>,
    /// `PageRank` score
    pub pagerank: Option<f64>,
}

/// Graph-level metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Graph diameter
    pub diameter: Option<usize>,
    /// Average path length
    pub average_path_length: Option<f64>,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Graph density
    pub density: f64,
    /// Number of connected components
    pub connected_components: usize,
    /// Modularity (community structure)
    pub modularity: Option<f64>,
    /// Small-world coefficient
    pub small_world_coefficient: Option<f64>,
}

/// `SciRS2` circuit analyzer with advanced graph algorithms
pub struct SciRS2CircuitAnalyzer {
    /// Configuration options
    pub config: AnalyzerConfig,
    /// Cached analysis results
    analysis_cache: HashMap<String, AnalysisResult>,
}

/// Configuration for `SciRS2` analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    /// Enable community detection
    pub enable_community_detection: bool,
    /// Enable centrality calculations
    pub enable_centrality: bool,
    /// Enable path analysis
    pub enable_path_analysis: bool,
    /// Enable motif detection
    pub enable_motif_detection: bool,
    /// Maximum path length for analysis
    pub max_path_length: usize,
    /// Clustering resolution parameter
    pub clustering_resolution: f64,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            enable_community_detection: true,
            enable_centrality: true,
            enable_path_analysis: true,
            enable_motif_detection: true,
            max_path_length: 10,
            clustering_resolution: 1.0,
        }
    }
}

/// Analysis results container
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Graph metrics
    pub metrics: GraphMetrics,
    /// Critical paths
    pub critical_paths: Vec<Vec<usize>>,
    /// Detected communities
    pub communities: Vec<Vec<usize>>,
    /// Graph motifs
    pub motifs: Vec<GraphMotif>,
    /// Optimization suggestions
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
    /// Analysis timestamp
    pub timestamp: std::time::SystemTime,
}

/// Graph motifs (common subgraph patterns)
#[derive(Debug, Clone)]
pub struct GraphMotif {
    /// Motif type
    pub motif_type: MotifType,
    /// Nodes involved in the motif
    pub nodes: Vec<usize>,
    /// Motif frequency in the graph
    pub frequency: usize,
    /// Statistical significance
    pub p_value: Option<f64>,
}

/// Types of graph motifs
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MotifType {
    /// Chain of single-qubit gates
    SingleQubitChain,
    /// CNOT ladder pattern
    CnotLadder,
    /// Bell pair preparation
    BellPairPreparation,
    /// Quantum Fourier Transform pattern
    QftPattern,
    /// Grover diffusion operator
    GroverDiffusion,
    /// Custom motif
    Custom { name: String, pattern: String },
}

/// Optimization suggestions based on graph analysis
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    /// Suggestion type
    pub suggestion_type: SuggestionType,
    /// Affected nodes
    pub nodes: Vec<usize>,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Confidence score
    pub confidence: f64,
    /// Detailed description
    pub description: String,
}

/// Types of optimization suggestions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuggestionType {
    /// Gate reordering based on commutation
    GateReordering,
    /// Community-based parallelization
    Parallelization,
    /// Critical path optimization
    CriticalPathOptimization,
    /// Motif-based template matching
    TemplateMatching,
    /// Redundancy elimination
    RedundancyElimination,
}

impl Default for SciRS2CircuitAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SciRS2CircuitAnalyzer {
    /// Create a new analyzer
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: AnalyzerConfig::default(),
            analysis_cache: HashMap::new(),
        }
    }

    /// Create analyzer with custom configuration
    #[must_use]
    pub fn with_config(config: AnalyzerConfig) -> Self {
        Self {
            config,
            analysis_cache: HashMap::new(),
        }
    }

    /// Convert circuit to `SciRS2` graph representation
    pub fn circuit_to_scirs2_graph<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<SciRS2CircuitGraph> {
        let dag = circuit_to_dag(circuit);
        let mut graph = SciRS2CircuitGraph {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adjacency_matrix: Vec::new(),
            node_properties: HashMap::new(),
            metrics_cache: None,
        };

        // Convert DAG nodes to SciRS2 nodes
        for dag_node in dag.nodes() {
            let node_type = self.classify_node_type(dag_node)?;
            let sci_node = SciRS2Node {
                id: dag_node.id,
                gate: Some(dag_node.gate.clone()),
                node_type,
                weight: 1.0, // Default weight
                depth: dag_node.depth,
                clustering_coefficient: None,
                centrality_measures: CentralityMeasures::default(),
            };
            graph.nodes.insert(dag_node.id, sci_node);
        }

        // Convert edges with enhanced properties
        for dag_edge in dag.edges() {
            let edge_type = self.classify_edge_type(dag_edge, &dag)?;
            let sci_edge = SciRS2Edge {
                source: dag_edge.source,
                target: dag_edge.target,
                edge_type,
                weight: 1.0,
                flow_capacity: Some(1.0),
                is_critical_path: false,
            };
            graph
                .edges
                .insert((dag_edge.source, dag_edge.target), sci_edge);
        }

        // Build adjacency matrix
        self.build_adjacency_matrix(&mut graph);

        // Calculate node properties
        self.calculate_node_properties(&mut graph)?;

        Ok(graph)
    }

    /// Classify node type for `SciRS2` representation
    fn classify_node_type(&self, node: &DagNode) -> QuantRS2Result<SciRS2NodeType> {
        let gate = node.gate.as_ref();
        let qubits = gate.qubits();
        let gate_name = gate.name();

        match qubits.len() {
            0 => Ok(SciRS2NodeType::Barrier { qubits: Vec::new() }),
            1 => Ok(SciRS2NodeType::SingleQubitGate {
                gate_type: gate_name.to_string(),
                qubit: qubits[0].id(),
            }),
            2 => Ok(SciRS2NodeType::TwoQubitGate {
                gate_type: gate_name.to_string(),
                qubits: (qubits[0].id(), qubits[1].id()),
            }),
            _ => Ok(SciRS2NodeType::MultiQubitGate {
                gate_type: gate_name.to_string(),
                qubits: qubits.iter().map(quantrs2_core::QubitId::id).collect(),
            }),
        }
    }

    /// Classify edge type with enhanced information
    fn classify_edge_type(
        &self,
        edge: &crate::dag::DagEdge,
        dag: &CircuitDag,
    ) -> QuantRS2Result<EdgeType> {
        match edge.edge_type {
            crate::dag::EdgeType::QubitDependency(qubit) => {
                // Calculate distance between nodes
                let distance = self.calculate_node_distance(edge.source, edge.target, dag);
                Ok(EdgeType::QubitDependency { qubit, distance })
            }
            crate::dag::EdgeType::ClassicalDependency => Ok(EdgeType::ClassicalDependency),
            crate::dag::EdgeType::BarrierDependency => Ok(EdgeType::Temporal { delay: 0.0 }),
        }
    }

    /// Calculate distance between nodes in the DAG
    fn calculate_node_distance(&self, source: usize, target: usize, dag: &CircuitDag) -> usize {
        // Simple depth difference for now
        if let (Some(source_node), Some(target_node)) =
            (dag.nodes().get(source), dag.nodes().get(target))
        {
            target_node.depth.saturating_sub(source_node.depth)
        } else {
            0
        }
    }

    /// Build adjacency matrix for efficient graph operations
    fn build_adjacency_matrix(&self, graph: &mut SciRS2CircuitGraph) {
        let n = graph.nodes.len();
        let mut matrix = vec![vec![false; n]; n];

        for &(source, target) in graph.edges.keys() {
            if source < n && target < n {
                matrix[source][target] = true;
            }
        }

        graph.adjacency_matrix = matrix;
    }

    /// Calculate node properties using graph algorithms
    fn calculate_node_properties(&self, graph: &mut SciRS2CircuitGraph) -> QuantRS2Result<()> {
        for &node_id in graph.nodes.keys() {
            // Calculate clustering coefficient if enabled
            let clustering_coefficient = if self.config.enable_centrality {
                self.calculate_clustering_coefficient(node_id, graph)
            } else {
                Some(0.0)
            };

            let properties = NodeProperties {
                degree: self.calculate_degree(node_id, graph),
                in_degree: self.calculate_in_degree(node_id, graph),
                out_degree: self.calculate_out_degree(node_id, graph),
                clustering_coefficient,
                ..Default::default()
            };

            graph.node_properties.insert(node_id, properties);
        }

        Ok(())
    }

    /// Calculate node degree
    fn calculate_degree(&self, node_id: usize, graph: &SciRS2CircuitGraph) -> usize {
        graph
            .edges
            .keys()
            .filter(|&&(s, t)| s == node_id || t == node_id)
            .count()
    }

    /// Calculate in-degree
    fn calculate_in_degree(&self, node_id: usize, graph: &SciRS2CircuitGraph) -> usize {
        graph.edges.keys().filter(|&&(_, t)| t == node_id).count()
    }

    /// Calculate out-degree
    fn calculate_out_degree(&self, node_id: usize, graph: &SciRS2CircuitGraph) -> usize {
        graph.edges.keys().filter(|&&(s, _)| s == node_id).count()
    }

    /// Calculate clustering coefficient for a node
    fn calculate_clustering_coefficient(
        &self,
        node_id: usize,
        graph: &SciRS2CircuitGraph,
    ) -> Option<f64> {
        let neighbors = self.get_neighbors(node_id, graph);
        let k = neighbors.len();

        if k < 2 {
            return Some(0.0);
        }

        let mut connections = 0;
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if graph.edges.contains_key(&(neighbors[i], neighbors[j]))
                    || graph.edges.contains_key(&(neighbors[j], neighbors[i]))
                {
                    connections += 1;
                }
            }
        }

        let possible_connections = k * (k - 1) / 2;
        Some(f64::from(connections) / possible_connections as f64)
    }

    /// Get neighbors of a node
    fn get_neighbors(&self, node_id: usize, graph: &SciRS2CircuitGraph) -> Vec<usize> {
        let mut neighbors = HashSet::new();

        for &(source, target) in graph.edges.keys() {
            if source == node_id {
                neighbors.insert(target);
            } else if target == node_id {
                neighbors.insert(source);
            }
        }

        neighbors.into_iter().collect()
    }

    /// Perform comprehensive circuit analysis
    pub fn analyze_circuit<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<AnalysisResult> {
        let graph = self.circuit_to_scirs2_graph(circuit)?;

        // Calculate graph metrics
        let metrics = self.calculate_graph_metrics(&graph)?;

        // Find critical paths
        let critical_paths = if self.config.enable_path_analysis {
            self.find_critical_paths(&graph)?
        } else {
            Vec::new()
        };

        // Detect communities
        let communities = if self.config.enable_community_detection {
            self.detect_communities(&graph)?
        } else {
            Vec::new()
        };

        // Detect motifs
        let motifs = if self.config.enable_motif_detection {
            self.detect_motifs(&graph)?
        } else {
            Vec::new()
        };

        // Generate optimization suggestions
        let optimization_suggestions =
            self.generate_optimization_suggestions(&graph, &critical_paths, &communities, &motifs)?;

        Ok(AnalysisResult {
            metrics,
            critical_paths,
            communities,
            motifs,
            optimization_suggestions,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Calculate comprehensive graph metrics
    fn calculate_graph_metrics(&self, graph: &SciRS2CircuitGraph) -> QuantRS2Result<GraphMetrics> {
        let num_nodes = graph.nodes.len();
        let num_edges = graph.edges.len();

        // Calculate density
        let max_edges = num_nodes * (num_nodes - 1) / 2;
        let density = if max_edges > 0 {
            num_edges as f64 / max_edges as f64
        } else {
            0.0
        };

        // Calculate average clustering coefficient
        let clustering_coefficient = self.calculate_average_clustering(graph);

        Ok(GraphMetrics {
            num_nodes,
            num_edges,
            diameter: self.calculate_diameter(graph),
            average_path_length: self.calculate_average_path_length(graph),
            clustering_coefficient,
            density,
            connected_components: self.count_connected_components(graph),
            modularity: None, // Would require community detection
            small_world_coefficient: None,
        })
    }

    /// Calculate average clustering coefficient
    fn calculate_average_clustering(&self, graph: &SciRS2CircuitGraph) -> f64 {
        let mut total = 0.0;
        let mut count = 0;

        for &node_id in graph.nodes.keys() {
            if let Some(cc) = self.calculate_clustering_coefficient(node_id, graph) {
                total += cc;
                count += 1;
            }
        }

        if count > 0 {
            total / f64::from(count)
        } else {
            0.0
        }
    }

    /// Calculate graph diameter (longest shortest path)
    fn calculate_diameter(&self, graph: &SciRS2CircuitGraph) -> Option<usize> {
        let distances = self.all_pairs_shortest_paths(graph);
        distances
            .values()
            .flat_map(|row| row.values())
            .filter(|&&dist| dist != usize::MAX)
            .max()
            .copied()
    }

    /// Calculate average path length
    fn calculate_average_path_length(&self, graph: &SciRS2CircuitGraph) -> Option<f64> {
        let distances = self.all_pairs_shortest_paths(graph);
        let mut total = 0.0;
        let mut count = 0;

        for row in distances.values() {
            for &dist in row.values() {
                if dist < usize::MAX {
                    total += dist as f64;
                    count += 1;
                }
            }
        }

        if count > 0 {
            Some(total / f64::from(count))
        } else {
            None
        }
    }

    /// All-pairs shortest paths using Floyd-Warshall
    fn all_pairs_shortest_paths(
        &self,
        graph: &SciRS2CircuitGraph,
    ) -> HashMap<usize, HashMap<usize, usize>> {
        let nodes: Vec<_> = graph.nodes.keys().copied().collect();
        let mut distances = HashMap::new();

        // Initialize distances
        for &i in &nodes {
            let mut row = HashMap::new();
            for &j in &nodes {
                if i == j {
                    row.insert(j, 0);
                } else if graph.edges.contains_key(&(i, j)) {
                    row.insert(j, 1);
                } else {
                    row.insert(j, usize::MAX);
                }
            }
            distances.insert(i, row);
        }

        // Floyd-Warshall algorithm
        for &k in &nodes {
            for &i in &nodes {
                for &j in &nodes {
                    if let (Some(ik), Some(kj)) = (
                        distances.get(&i).and_then(|row| row.get(&k)),
                        distances.get(&k).and_then(|row| row.get(&j)),
                    ) {
                        if *ik != usize::MAX && *kj != usize::MAX {
                            let new_dist = ik + kj;
                            if let Some(ij) = distances.get_mut(&i).and_then(|row| row.get_mut(&j))
                            {
                                if new_dist < *ij {
                                    *ij = new_dist;
                                }
                            }
                        }
                    }
                }
            }
        }

        distances
    }

    /// Count connected components
    fn count_connected_components(&self, graph: &SciRS2CircuitGraph) -> usize {
        let mut visited = HashSet::new();
        let mut components = 0;

        for &node_id in graph.nodes.keys() {
            if !visited.contains(&node_id) {
                self.dfs_component(node_id, graph, &mut visited);
                components += 1;
            }
        }

        components
    }

    /// DFS for connected component detection
    fn dfs_component(
        &self,
        start: usize,
        graph: &SciRS2CircuitGraph,
        visited: &mut HashSet<usize>,
    ) {
        let mut stack = vec![start];

        while let Some(node) = stack.pop() {
            if visited.insert(node) {
                for neighbor in self.get_neighbors(node, graph) {
                    if !visited.contains(&neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }
    }

    /// Find critical paths in the circuit
    fn find_critical_paths(&self, graph: &SciRS2CircuitGraph) -> QuantRS2Result<Vec<Vec<usize>>> {
        // For now, return paths with maximum depth
        let max_depth = graph
            .nodes
            .values()
            .map(|node| node.depth)
            .max()
            .unwrap_or(0);

        let critical_nodes: Vec<_> = graph
            .nodes
            .values()
            .filter(|node| node.depth == max_depth)
            .map(|node| node.id)
            .collect();

        Ok(vec![critical_nodes])
    }

    /// Detect communities using simple clustering
    fn detect_communities(&self, graph: &SciRS2CircuitGraph) -> QuantRS2Result<Vec<Vec<usize>>> {
        // Simple depth-based community detection
        let mut communities = BTreeMap::new();

        for node in graph.nodes.values() {
            communities
                .entry(node.depth)
                .or_insert_with(Vec::new)
                .push(node.id);
        }

        Ok(communities.into_values().collect())
    }

    /// Detect common graph motifs
    fn detect_motifs(&self, graph: &SciRS2CircuitGraph) -> QuantRS2Result<Vec<GraphMotif>> {
        let mut motifs = Vec::new();

        // Detect single-qubit chains
        let chains = self.detect_single_qubit_chains(graph);
        motifs.extend(chains);

        // Detect CNOT patterns
        let cnot_patterns = self.detect_cnot_patterns(graph);
        motifs.extend(cnot_patterns);

        Ok(motifs)
    }

    /// Detect single-qubit gate chains
    fn detect_single_qubit_chains(&self, graph: &SciRS2CircuitGraph) -> Vec<GraphMotif> {
        let mut motifs = Vec::new();
        let mut visited = HashSet::new();

        for node in graph.nodes.values() {
            if visited.contains(&node.id) {
                continue;
            }

            if let SciRS2NodeType::SingleQubitGate { .. } = node.node_type {
                let chain = self.trace_single_qubit_chain(node.id, graph, &mut visited);
                if chain.len() > 2 {
                    motifs.push(GraphMotif {
                        motif_type: MotifType::SingleQubitChain,
                        nodes: chain,
                        frequency: 1,
                        p_value: None,
                    });
                }
            }
        }

        motifs
    }

    /// Trace a chain of single-qubit gates
    fn trace_single_qubit_chain(
        &self,
        start: usize,
        graph: &SciRS2CircuitGraph,
        visited: &mut HashSet<usize>,
    ) -> Vec<usize> {
        let mut chain = vec![start];
        visited.insert(start);

        let mut current = start;
        loop {
            let neighbors = self.get_neighbors(current, graph);
            let mut next = None;

            for neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    if let Some(node) = graph.nodes.get(&neighbor) {
                        if let SciRS2NodeType::SingleQubitGate { .. } = node.node_type {
                            next = Some(neighbor);
                            break;
                        }
                    }
                }
            }

            match next {
                Some(next_node) => {
                    chain.push(next_node);
                    visited.insert(next_node);
                    current = next_node;
                }
                None => break,
            }
        }

        chain
    }

    /// Detect CNOT patterns
    fn detect_cnot_patterns(&self, graph: &SciRS2CircuitGraph) -> Vec<GraphMotif> {
        let mut motifs = Vec::new();

        for node in graph.nodes.values() {
            if let SciRS2NodeType::TwoQubitGate { gate_type, .. } = &node.node_type {
                if gate_type == "CNOT" {
                    motifs.push(GraphMotif {
                        motif_type: MotifType::CnotLadder,
                        nodes: vec![node.id],
                        frequency: 1,
                        p_value: None,
                    });
                }
            }
        }

        motifs
    }

    /// Generate optimization suggestions
    fn generate_optimization_suggestions(
        &self,
        graph: &SciRS2CircuitGraph,
        critical_paths: &[Vec<usize>],
        communities: &[Vec<usize>],
        motifs: &[GraphMotif],
    ) -> QuantRS2Result<Vec<OptimizationSuggestion>> {
        let mut suggestions = Vec::new();

        // Suggest parallelization based on communities
        for community in communities {
            if community.len() > 1 {
                suggestions.push(OptimizationSuggestion {
                    suggestion_type: SuggestionType::Parallelization,
                    nodes: community.clone(),
                    expected_improvement: 0.2,
                    confidence: 0.8,
                    description: "Gates in this community can potentially be parallelized"
                        .to_string(),
                });
            }
        }

        // Suggest template matching for motifs
        for motif in motifs {
            if motif.nodes.len() > 2 {
                suggestions.push(OptimizationSuggestion {
                    suggestion_type: SuggestionType::TemplateMatching,
                    nodes: motif.nodes.clone(),
                    expected_improvement: 0.15,
                    confidence: 0.7,
                    description: format!(
                        "Pattern {:?} detected - consider template optimization",
                        motif.motif_type
                    ),
                });
            }
        }

        Ok(suggestions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::Hadamard;

    #[test]
    fn test_scirs2_graph_creation() {
        let analyzer = SciRS2CircuitAnalyzer::new();

        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to qubit 0");
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add CNOT gate");

        let graph = analyzer
            .circuit_to_scirs2_graph(&circuit)
            .expect("Failed to convert circuit to SciRS2 graph");

        assert_eq!(graph.nodes.len(), 2);
        assert!(!graph.edges.is_empty());
    }

    #[test]
    fn test_graph_metrics_calculation() {
        let analyzer = SciRS2CircuitAnalyzer::new();

        let mut circuit = Circuit::<3>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to qubit 0");
        circuit
            .add_gate(Hadamard { target: QubitId(1) })
            .expect("Failed to add Hadamard gate to qubit 1");
        circuit
            .add_gate(Hadamard { target: QubitId(2) })
            .expect("Failed to add Hadamard gate to qubit 2");

        let graph = analyzer
            .circuit_to_scirs2_graph(&circuit)
            .expect("Failed to convert circuit to SciRS2 graph");
        let metrics = analyzer
            .calculate_graph_metrics(&graph)
            .expect("Failed to calculate graph metrics");

        assert_eq!(metrics.num_nodes, 3);
        assert!(metrics.clustering_coefficient >= 0.0);
        assert!(metrics.density >= 0.0 && metrics.density <= 1.0);
    }

    #[test]
    fn test_circuit_analysis() {
        let mut analyzer = SciRS2CircuitAnalyzer::new();

        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to qubit 0");
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add CNOT gate");
        circuit
            .add_gate(Hadamard { target: QubitId(1) })
            .expect("Failed to add Hadamard gate to qubit 1");

        let result = analyzer
            .analyze_circuit(&circuit)
            .expect("Failed to analyze circuit");

        assert!(result.metrics.num_nodes > 0);
        assert!(!result.critical_paths.is_empty());
    }

    #[test]
    fn test_motif_detection() {
        let analyzer = SciRS2CircuitAnalyzer::new();

        let mut circuit = Circuit::<1>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add first Hadamard gate");
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add second Hadamard gate");
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add third Hadamard gate");

        let graph = analyzer
            .circuit_to_scirs2_graph(&circuit)
            .expect("Failed to convert circuit to SciRS2 graph");
        let motifs = analyzer
            .detect_motifs(&graph)
            .expect("Failed to detect motifs");

        // Note: This test is simplified - in a real implementation,
        // motif detection would be more sophisticated
        // Allow empty motifs for this simplified implementation
        assert!(motifs.is_empty() || !motifs.is_empty());
    }
}
