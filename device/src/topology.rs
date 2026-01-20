//! Hardware topology analysis using `SciRS2` graph algorithms
//!
//! This module provides tools for analyzing quantum hardware topologies,
//! finding optimal qubit mappings, and identifying connectivity constraints.

use petgraph::algo::{connected_components, dijkstra, min_spanning_tree};
use petgraph::data::FromElements;
use petgraph::graph::{NodeIndex, UnGraph};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Error type for topology operations
#[derive(Error, Debug)]
pub enum TopologyError {
    #[error("Unsupported topology: {0}")]
    UnsupportedTopology(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

/// Represents a quantum hardware topology
#[derive(Debug, Clone)]
pub struct HardwareTopology {
    /// Number of physical qubits
    pub num_qubits: usize,
    /// Connectivity graph (undirected for most hardware)
    pub connectivity: UnGraph<u32, f64>,
    /// Qubit properties (T1, T2, gate errors, etc.)
    pub qubit_properties: Vec<QubitProperties>,
    /// Two-qubit gate properties (gate errors, gate times)
    pub gate_properties: HashMap<(u32, u32), GateProperties>,
}

/// Properties of a physical qubit
#[derive(Debug, Clone)]
pub struct QubitProperties {
    /// Qubit index
    pub id: u32,
    /// Qubit index (same as id, for compatibility)
    pub index: u32,
    /// T1 coherence time (microseconds)
    pub t1: f64,
    /// T2 coherence time (microseconds)
    pub t2: f64,
    /// Single-qubit gate error rate
    pub single_qubit_gate_error: f64,
    /// Single-qubit gate error rate (same as above, for compatibility)
    pub gate_error_1q: f64,
    /// Readout error rate
    pub readout_error: f64,
    /// Frequency (GHz)
    pub frequency: f64,
}

/// Properties of a two-qubit gate between qubits
#[derive(Debug, Clone)]
pub struct GateProperties {
    /// Gate error rate
    pub error_rate: f64,
    /// Gate duration (nanoseconds)
    pub duration: f64,
    /// Gate type (e.g., "CZ", "CNOT")
    pub gate_type: String,
}

impl HardwareTopology {
    /// Create a new hardware topology
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            connectivity: UnGraph::new_undirected(),
            qubit_properties: Vec::new(),
            gate_properties: HashMap::new(),
        }
    }

    /// Add a qubit with properties
    pub fn add_qubit(&mut self, properties: QubitProperties) -> NodeIndex {
        let idx = self.connectivity.add_node(properties.index);
        self.qubit_properties.push(properties);
        idx
    }

    /// Add a connection between two qubits
    pub fn add_connection(&mut self, q1: u32, q2: u32, properties: GateProperties) {
        // Find node indices
        let n1 = self
            .connectivity
            .node_indices()
            .find(|&n| self.connectivity[n] == q1);
        let n2 = self
            .connectivity
            .node_indices()
            .find(|&n| self.connectivity[n] == q2);

        if let (Some(n1), Some(n2)) = (n1, n2) {
            // Use error rate as edge weight (lower is better)
            self.connectivity.add_edge(n1, n2, properties.error_rate);
            self.gate_properties.insert((q1, q2), properties.clone());
            self.gate_properties.insert((q2, q1), properties);
        }
    }

    /// Load standard hardware topologies
    pub fn load_standard(topology_type: &str) -> Result<Self, TopologyError> {
        match topology_type {
            "linear" => Ok(Self::linear_topology(5)),
            "grid_2x3" => Ok(Self::grid_topology(2, 3)),
            "ibm_5q" => Ok(Self::ibm_topology()),
            "google_sycamore" => Ok(Self::create_sycamore_subset()),
            _ => Err(TopologyError::UnsupportedTopology(format!(
                "Unknown topology: {topology_type}"
            ))),
        }
    }

    /// Create a linear topology (chain of qubits)
    pub fn linear_topology(n: usize) -> Self {
        let mut topo = Self::new(n);

        // Add qubits
        for i in 0..n {
            topo.add_qubit(QubitProperties {
                id: i as u32,
                index: i as u32,
                t1: 50.0,
                t2: 70.0,
                single_qubit_gate_error: 0.001,
                gate_error_1q: 0.001,
                readout_error: 0.01,
                frequency: 0.1f64.mul_add(i as f64, 5.0),
            });
        }

        // Add connections
        for i in 0..n - 1 {
            topo.add_connection(
                i as u32,
                (i + 1) as u32,
                GateProperties {
                    error_rate: 0.01,
                    duration: 200.0,
                    gate_type: "CZ".to_string(),
                },
            );
        }

        topo
    }

    /// Create a 2D grid topology
    pub fn grid_topology(rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        let mut topo = Self::new(n);

        // Add qubits
        for i in 0..n {
            topo.add_qubit(QubitProperties {
                id: i as u32,
                index: i as u32,
                t1: 5.0f64.mul_add(fastrand::f64(), 45.0),
                t2: 10.0f64.mul_add(fastrand::f64(), 60.0),
                single_qubit_gate_error: 0.001 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                gate_error_1q: 0.001 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                readout_error: 0.01 * 0.3f64.mul_add(fastrand::f64(), 1.0),
                frequency: 0.2f64.mul_add(fastrand::f64(), 5.0),
            });
        }

        // Add horizontal connections
        for r in 0..rows {
            for c in 0..cols - 1 {
                let q1 = (r * cols + c) as u32;
                let q2 = (r * cols + c + 1) as u32;
                topo.add_connection(
                    q1,
                    q2,
                    GateProperties {
                        error_rate: 0.01 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                        duration: 40.0f64.mul_add(fastrand::f64(), 180.0),
                        gate_type: "CZ".to_string(),
                    },
                );
            }
        }

        // Add vertical connections
        for r in 0..rows - 1 {
            for c in 0..cols {
                let q1 = (r * cols + c) as u32;
                let q2 = ((r + 1) * cols + c) as u32;
                topo.add_connection(
                    q1,
                    q2,
                    GateProperties {
                        error_rate: 0.01 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                        duration: 40.0f64.mul_add(fastrand::f64(), 180.0),
                        gate_type: "CZ".to_string(),
                    },
                );
            }
        }

        topo
    }

    /// Create IBM 5-qubit topology (bow-tie shape)
    pub fn ibm_topology() -> Self {
        let mut topo = Self::new(5);

        // Add qubits with realistic properties
        let qubit_props = vec![
            (0, 52.3, 71.2, 0.0008, 0.015),
            (1, 48.7, 68.5, 0.0011, 0.018),
            (2, 55.1, 73.8, 0.0009, 0.012),
            (3, 51.4, 69.9, 0.0010, 0.016),
            (4, 49.8, 67.3, 0.0012, 0.020),
        ];

        for (i, (idx, t1, t2, err_1q, err_ro)) in qubit_props.into_iter().enumerate() {
            topo.add_qubit(QubitProperties {
                id: idx,
                index: idx,
                t1,
                t2,
                single_qubit_gate_error: err_1q,
                gate_error_1q: err_1q,
                readout_error: err_ro,
                frequency: 0.05f64.mul_add(i as f64, 5.0),
            });
        }

        // Add connections (bow-tie pattern)
        let connections = vec![(0, 1, 0.008), (1, 2, 0.012), (1, 3, 0.010), (3, 4, 0.009)];

        for (q1, q2, err) in connections {
            topo.add_connection(
                q1,
                q2,
                GateProperties {
                    error_rate: err,
                    duration: 20.0f64.mul_add(fastrand::f64(), 200.0),
                    gate_type: "CNOT".to_string(),
                },
            );
        }

        topo
    }

    /// Create Google Sycamore-like topology (subset)
    pub fn google_topology() -> Self {
        // Create a 6-qubit subset of Sycamore topology
        let mut topo = Self::new(6);

        // Add qubits
        for i in 0..6 {
            topo.add_qubit(QubitProperties {
                id: i as u32,
                index: i as u32,
                t1: 2.0f64.mul_add(fastrand::f64(), 15.0),
                t2: 3.0f64.mul_add(fastrand::f64(), 20.0),
                single_qubit_gate_error: 0.0015 * 0.1f64.mul_add(fastrand::f64(), 1.0),
                gate_error_1q: 0.0015 * 0.1f64.mul_add(fastrand::f64(), 1.0),
                readout_error: 0.008 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                frequency: 0.1f64.mul_add(fastrand::f64(), 5.5),
            });
        }

        // Add connections in a 2x3 grid pattern with diagonals
        let connections = vec![
            (0, 1),
            (1, 2), // Top row
            (3, 4),
            (4, 5), // Bottom row
            (0, 3),
            (1, 4),
            (2, 5), // Vertical
            (0, 4),
            (1, 5), // Diagonals
        ];

        for (q1, q2) in connections {
            topo.add_connection(
                q1,
                q2,
                GateProperties {
                    error_rate: 0.006 * 0.15f64.mul_add(fastrand::f64(), 1.0),
                    duration: 5.0f64.mul_add(fastrand::f64(), 25.0),
                    gate_type: "sqrt_ISWAP".to_string(),
                },
            );
        }

        topo
    }

    /// Create IBM Heavy-Hex topology
    pub fn from_heavy_hex(num_qubits: usize) -> Self {
        let mut topo = Self::new(num_qubits);

        // Add qubits with IBM-like properties
        for i in 0..num_qubits {
            topo.add_qubit(QubitProperties {
                id: i as u32,
                index: i as u32,
                t1: 10.0f64.mul_add(fastrand::f64(), 50.0),
                t2: 15.0f64.mul_add(fastrand::f64(), 70.0),
                single_qubit_gate_error: 0.001 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                gate_error_1q: 0.001 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                readout_error: 0.015 * 0.3f64.mul_add(fastrand::f64(), 1.0),
                frequency: 0.2f64.mul_add(fastrand::f64(), 5.0),
            });
        }

        // Heavy-hex pattern for IBM quantum computers
        // This is a simplified version - real IBM topologies vary
        if num_qubits >= 27 {
            // Create a heavy-hex lattice pattern
            // Row 0: 0, 1, 2, 3, 4
            for i in 0..4 {
                topo.add_connection(
                    i,
                    i + 1,
                    GateProperties {
                        error_rate: 0.01 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                        duration: 40.0f64.mul_add(fastrand::f64(), 200.0),
                        gate_type: "CNOT".to_string(),
                    },
                );
            }

            // Row 1: 5, 6, 7, 8, 9
            for i in 5..9 {
                topo.add_connection(
                    i,
                    i + 1,
                    GateProperties {
                        error_rate: 0.01 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                        duration: 40.0f64.mul_add(fastrand::f64(), 200.0),
                        gate_type: "CNOT".to_string(),
                    },
                );
            }

            // Row 2: 10, 11, 12, 13, 14
            for i in 10..14 {
                topo.add_connection(
                    i,
                    i + 1,
                    GateProperties {
                        error_rate: 0.01 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                        duration: 40.0f64.mul_add(fastrand::f64(), 200.0),
                        gate_type: "CNOT".to_string(),
                    },
                );
            }

            // Row 3: 15, 16, 17, 18, 19
            for i in 15..19 {
                topo.add_connection(
                    i,
                    i + 1,
                    GateProperties {
                        error_rate: 0.01 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                        duration: 40.0f64.mul_add(fastrand::f64(), 200.0),
                        gate_type: "CNOT".to_string(),
                    },
                );
            }

            // Row 4: 20, 21, 22, 23, 24, 25, 26
            for i in 20..26 {
                topo.add_connection(
                    i,
                    i + 1,
                    GateProperties {
                        error_rate: 0.01 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                        duration: 40.0f64.mul_add(fastrand::f64(), 200.0),
                        gate_type: "CNOT".to_string(),
                    },
                );
            }

            // Vertical connections (heavy edges)
            let vertical_connections = vec![
                (0, 5),
                (2, 7),
                (4, 9),
                (5, 10),
                (7, 12),
                (9, 14),
                (10, 15),
                (12, 17),
                (14, 19),
                (15, 20),
                (16, 21),
                (17, 22),
                (18, 23),
                (19, 24),
            ];

            for (a, b) in vertical_connections {
                if a < num_qubits && b < num_qubits {
                    topo.add_connection(
                        a as u32,
                        b as u32,
                        GateProperties {
                            error_rate: 0.01 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                            duration: 40.0f64.mul_add(fastrand::f64(), 200.0),
                            gate_type: "CNOT".to_string(),
                        },
                    );
                }
            }

            // Diagonal connections (hex pattern)
            let diagonal_connections = vec![(1, 6), (3, 8), (6, 11), (8, 13), (11, 16), (13, 18)];

            for (a, b) in diagonal_connections {
                if a < num_qubits && b < num_qubits {
                    topo.add_connection(
                        a as u32,
                        b as u32,
                        GateProperties {
                            error_rate: 0.01 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                            duration: 40.0f64.mul_add(fastrand::f64(), 200.0),
                            gate_type: "CNOT".to_string(),
                        },
                    );
                }
            }
        }

        topo
    }

    /// Create Google Sycamore-like topology
    pub fn from_sycamore(num_qubits: usize) -> Self {
        let mut topo = Self::new(num_qubits);

        // Add qubits with Google-like properties
        for i in 0..num_qubits {
            topo.add_qubit(QubitProperties {
                id: i as u32,
                index: i as u32,
                t1: 3.0f64.mul_add(fastrand::f64(), 15.0),
                t2: 5.0f64.mul_add(fastrand::f64(), 20.0),
                single_qubit_gate_error: 0.0015 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                gate_error_1q: 0.0015 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                readout_error: 0.008 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                frequency: 0.1f64.mul_add(fastrand::f64(), 5.5),
            });
        }

        // Sycamore has a 2D grid with couplers removed
        // This is a simplified version
        if num_qubits >= 20 {
            // Create a 4x5 grid-like structure
            let rows = 4;
            let cols = 5;

            // Horizontal connections
            for row in 0..rows {
                for col in 0..cols - 1 {
                    let q1 = row * cols + col;
                    let q2 = row * cols + col + 1;
                    if q1 < num_qubits && q2 < num_qubits {
                        topo.add_connection(
                            q1 as u32,
                            q2 as u32,
                            GateProperties {
                                error_rate: 0.006 * 0.15f64.mul_add(fastrand::f64(), 1.0),
                                duration: 12.0,
                                gate_type: "sqrt_ISwap".to_string(),
                            },
                        );
                    }
                }
            }

            // Vertical connections (with some removed for Sycamore pattern)
            for row in 0..rows - 1 {
                for col in 0..cols {
                    // Skip some connections to create Sycamore pattern
                    if (row + col) % 2 == 0 {
                        let q1 = row * cols + col;
                        let q2 = (row + 1) * cols + col;
                        if q1 < num_qubits && q2 < num_qubits {
                            topo.add_connection(
                                q1 as u32,
                                q2 as u32,
                                GateProperties {
                                    error_rate: 0.006 * 0.15f64.mul_add(fastrand::f64(), 1.0),
                                    duration: 12.0,
                                    gate_type: "sqrt_ISwap".to_string(),
                                },
                            );
                        }
                    }
                }
            }
        }

        topo
    }

    /// Create a subset of Google Sycamore topology (3x3 grid with diagonal connections)
    fn create_sycamore_subset() -> Self {
        let mut topo = Self::new(9);

        // Add qubits
        for i in 0..9 {
            topo.add_qubit(QubitProperties {
                id: i as u32,
                index: i as u32,
                t1: 3.0f64.mul_add(fastrand::f64(), 15.0),
                t2: 5.0f64.mul_add(fastrand::f64(), 20.0),
                single_qubit_gate_error: 0.0015 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                gate_error_1q: 0.0015 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                readout_error: 0.03 * 0.2f64.mul_add(fastrand::f64(), 1.0),
                frequency: 0.1f64.mul_add(fastrand::f64(), 5.5),
            });
        }

        // Add grid connections
        let grid_conns = vec![
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8), // horizontal
            (0, 3),
            (3, 6),
            (1, 4),
            (4, 7),
            (2, 5),
            (5, 8), // vertical
        ];

        for (q1, q2) in grid_conns {
            topo.add_connection(
                q1,
                q2,
                GateProperties {
                    error_rate: 0.006 * 0.3f64.mul_add(fastrand::f64(), 1.0),
                    duration: 12.0,
                    gate_type: "sqrt_ISwap".to_string(),
                },
            );
        }

        // Add some diagonal connections (Sycamore feature)
        let diag_conns = vec![(0, 4), (4, 8), (2, 4), (4, 6)];

        for (q1, q2) in diag_conns {
            topo.add_connection(
                q1,
                q2,
                GateProperties {
                    error_rate: 0.008 * 0.3f64.mul_add(fastrand::f64(), 1.0),
                    duration: 12.0,
                    gate_type: "sqrt_ISwap".to_string(),
                },
            );
        }

        topo
    }

    /// Get the number of qubits
    pub const fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get all connected pairs of qubits
    pub fn connectivity(&self) -> Vec<(usize, usize)> {
        let mut connections = Vec::new();
        for edge in self.connectivity.edge_indices() {
            if let Some((a, b)) = self.connectivity.edge_endpoints(edge) {
                let q1 = self.connectivity[a] as usize;
                let q2 = self.connectivity[b] as usize;
                connections.push((q1, q2));
            }
        }
        connections
    }

    /// Check if two qubits are connected
    pub fn are_connected(&self, q1: usize, q2: usize) -> bool {
        // Find node indices for these qubits
        let node1 = self
            .connectivity
            .node_indices()
            .find(|&n| self.connectivity[n] == q1 as u32);
        let node2 = self
            .connectivity
            .node_indices()
            .find(|&n| self.connectivity[n] == q2 as u32);

        if let (Some(n1), Some(n2)) = (node1, node2) {
            self.connectivity.contains_edge(n1, n2)
        } else {
            false
        }
    }

    /// Get shortest path distance between two qubits
    pub fn shortest_path_distance(&self, q1: usize, q2: usize) -> Option<f64> {
        // Find node indices for these qubits
        let node1 = self
            .connectivity
            .node_indices()
            .find(|&n| self.connectivity[n] == q1 as u32)?;
        let node2 = self
            .connectivity
            .node_indices()
            .find(|&n| self.connectivity[n] == q2 as u32)?;

        // Use Dijkstra to find shortest path
        let distances = dijkstra(&self.connectivity, node1, Some(node2), |e| *e.weight());
        distances.get(&node2).copied()
    }

    /// Simple connectivity analysis
    pub fn analyze_connectivity(&self) -> ConnectivityAnalysis {
        let num_edges = self.connectivity.edge_count();
        let mut min_conn = usize::MAX;
        let mut max_conn = 0;
        let mut total_conn = 0;

        for node in self.connectivity.node_indices() {
            let degree = self.connectivity.edges(node).count();
            min_conn = min_conn.min(degree);
            max_conn = max_conn.max(degree);
            total_conn += degree;
        }

        let avg_conn = if self.num_qubits > 0 {
            total_conn as f64 / self.num_qubits as f64
        } else {
            0.0
        };

        ConnectivityAnalysis {
            num_qubits: self.num_qubits,
            num_edges,
            average_connectivity: avg_conn,
            max_connectivity: max_conn,
            min_connectivity: min_conn,
            connected_components: connected_components(&self.connectivity),
        }
    }

    /// Analyze topology properties
    pub fn analyze(&self) -> TopologyAnalysis {
        // Calculate shortest paths between all pairs
        let mut all_distances = vec![vec![None; self.num_qubits]; self.num_qubits];

        for node in self.connectivity.node_indices() {
            let distances = dijkstra(&self.connectivity, node, None, |e| *e.weight());
            for (&target, &dist) in &distances {
                let src_idx = self.connectivity[node] as usize;
                let tgt_idx = self.connectivity[target] as usize;
                all_distances[src_idx][tgt_idx] = Some(dist);
            }
        }

        // Calculate average distance
        let mut total_dist = 0.0;
        let mut count = 0;
        for i in 0..self.num_qubits {
            for j in i + 1..self.num_qubits {
                if let Some(d) = all_distances[i][j] {
                    total_dist += d;
                    count += 1;
                }
            }
        }
        let avg_distance = if count > 0 {
            total_dist / f64::from(count)
        } else {
            0.0
        };

        // Calculate degree distribution
        let mut degree_dist = HashMap::new();
        for node in self.connectivity.node_indices() {
            let degree = self.connectivity.edges(node).count();
            *degree_dist.entry(degree).or_insert(0) += 1;
        }

        // Find most connected qubit
        let most_connected = self
            .connectivity
            .node_indices()
            .max_by_key(|&n| self.connectivity.edges(n).count())
            .map(|n| self.connectivity[n]);

        // Calculate clustering coefficient
        let clustering_coeff = self.calculate_clustering_coefficient();

        // Count connected components
        let num_components = connected_components(&self.connectivity);

        // Find minimum spanning tree weight
        let mst = min_spanning_tree(&self.connectivity);
        let mst_graph = UnGraph::<_, _>::from_elements(mst);
        let mst_weight: f64 = mst_graph.edge_references().map(|e| *e.weight()).sum();

        TopologyAnalysis {
            num_qubits: self.num_qubits,
            num_connections: self.connectivity.edge_count(),
            average_distance: avg_distance,
            max_distance: all_distances
                .iter()
                .flatten()
                .filter_map(|&d| d)
                .fold(0.0, f64::max),
            degree_distribution: degree_dist,
            most_connected_qubit: most_connected,
            clustering_coefficient: clustering_coeff,
            connected_components: num_components,
            mst_weight,
            avg_gate_error: self
                .gate_properties
                .values()
                .map(|p| p.error_rate)
                .sum::<f64>()
                / self.gate_properties.len().max(1) as f64,
            avg_coherence_time: self.qubit_properties.iter().map(|p| p.t2).sum::<f64>()
                / self.qubit_properties.len().max(1) as f64,
        }
    }

    /// Calculate clustering coefficient
    fn calculate_clustering_coefficient(&self) -> f64 {
        let mut total_coeff = 0.0;
        let mut count = 0;

        for node in self.connectivity.node_indices() {
            let neighbors: Vec<_> = self.connectivity.neighbors(node).collect();
            let k = neighbors.len();

            if k >= 2 {
                let mut triangles = 0;
                for i in 0..k {
                    for j in i + 1..k {
                        if self.connectivity.contains_edge(neighbors[i], neighbors[j]) {
                            triangles += 1;
                        }
                    }
                }

                let possible = k * (k - 1) / 2;
                total_coeff += f64::from(triangles) / possible as f64;
                count += 1;
            }
        }

        if count > 0 {
            total_coeff / f64::from(count)
        } else {
            0.0
        }
    }

    /// Find critical qubits (removal increases distances significantly)
    pub fn find_critical_qubits(&self) -> Vec<u32> {
        let base_analysis = self.analyze();
        let mut critical = Vec::new();

        for node in self.connectivity.node_indices() {
            // Create topology without this qubit
            let mut test_graph = self.connectivity.clone();
            test_graph.remove_node(node);

            // Check if it disconnects the graph
            if connected_components(&test_graph) > base_analysis.connected_components {
                critical.push(self.connectivity[node]);
            }
        }

        critical
    }

    /// Find optimal qubit subset for a given circuit size
    pub fn find_optimal_subset(&self, required_qubits: usize) -> Result<Vec<u32>, TopologyError> {
        if required_qubits > self.num_qubits {
            return Err(TopologyError::InvalidConfiguration(format!(
                "Required {} qubits but hardware has only {}",
                required_qubits, self.num_qubits
            )));
        }

        // Score function: minimize average error and maximize connectivity
        let score_subset = |subset: &[u32]| -> f64 {
            let mut score = 0.0;
            let mut connections = 0;

            // Add qubit quality scores
            for &q in subset {
                if let Some(props) = self.qubit_properties.get(q as usize) {
                    score += props.gate_error_1q + props.readout_error;
                }
            }

            // Add connection scores
            for i in 0..subset.len() {
                for j in i + 1..subset.len() {
                    if let Some(props) = self
                        .gate_properties
                        .get(&(subset[i], subset[j]))
                        .or_else(|| self.gate_properties.get(&(subset[j], subset[i])))
                    {
                        score += props.error_rate;
                        connections += 1;
                    }
                }
            }

            // Penalize low connectivity
            if connections < required_qubits - 1 {
                score *= 10.0;
            }

            score
        };

        // Use greedy algorithm to find good subset
        let mut best_subset = Vec::new();
        let mut best_score = f64::INFINITY;

        // Try different starting points
        for start in 0..self.num_qubits.min(5) {
            let mut subset = vec![start as u32];
            let mut remaining: HashSet<_> = (0..self.num_qubits as u32).collect();
            remaining.remove(&(start as u32));

            while subset.len() < required_qubits {
                // Find best qubit to add - prefer connected qubits
                let mut best_add = None;
                let mut best_add_score = f64::INFINITY;

                // First, try to add only connected qubits
                let mut found_connected = false;
                for &candidate in &remaining {
                    // Check if candidate is connected to any qubit in subset
                    let is_connected = subset.iter().any(|&q| {
                        self.gate_properties.contains_key(&(q, candidate))
                            || self.gate_properties.contains_key(&(candidate, q))
                    });

                    if is_connected {
                        found_connected = true;
                        let mut test_subset = subset.clone();
                        test_subset.push(candidate);
                        let score = score_subset(&test_subset);

                        if score < best_add_score {
                            best_add_score = score;
                            best_add = Some(candidate);
                        }
                    }
                }

                // If no connected qubit found, consider all remaining qubits
                if !found_connected {
                    for &candidate in &remaining {
                        let mut test_subset = subset.clone();
                        test_subset.push(candidate);
                        let score = score_subset(&test_subset);

                        if score < best_add_score {
                            best_add_score = score;
                            best_add = Some(candidate);
                        }
                    }
                }

                if let Some(q) = best_add {
                    subset.push(q);
                    remaining.remove(&q);
                } else {
                    break; // No more qubits to add
                }
            }

            let score = score_subset(&subset);
            if score < best_score {
                best_score = score;
                best_subset = subset;
            }
        }

        Ok(best_subset)
    }
}

impl Default for HardwareTopology {
    fn default() -> Self {
        Self {
            num_qubits: 0,
            connectivity: UnGraph::new_undirected(),
            qubit_properties: Vec::new(),
            gate_properties: HashMap::new(),
        }
    }
}

/// Simple connectivity analysis results
#[derive(Debug)]
pub struct ConnectivityAnalysis {
    pub num_qubits: usize,
    pub num_edges: usize,
    pub average_connectivity: f64,
    pub max_connectivity: usize,
    pub min_connectivity: usize,
    pub connected_components: usize,
}

/// Analysis results for a hardware topology
#[derive(Debug)]
pub struct TopologyAnalysis {
    pub num_qubits: usize,
    pub num_connections: usize,
    pub average_distance: f64,
    pub max_distance: f64,
    pub degree_distribution: HashMap<usize, usize>,
    pub most_connected_qubit: Option<u32>,
    pub clustering_coefficient: f64,
    pub connected_components: usize,
    pub mst_weight: f64,
    pub avg_gate_error: f64,
    pub avg_coherence_time: f64,
}

impl TopologyAnalysis {
    /// Generate a report of the analysis
    pub fn report(&self) -> String {
        use std::fmt::Write;
        let mut report = String::new();

        report.push_str("=== Hardware Topology Analysis ===\n");
        let _ = writeln!(report, "Number of qubits: {}", self.num_qubits);
        let _ = writeln!(report, "Number of connections: {}", self.num_connections);
        let _ = writeln!(
            report,
            "Connected components: {}",
            self.connected_components
        );
        report.push_str("\nDistance metrics:\n");
        let _ = writeln!(report, "  Average distance: {:.2}", self.average_distance);
        let _ = writeln!(report, "  Maximum distance: {:.2}", self.max_distance);
        report.push_str("\nConnectivity metrics:\n");
        let _ = writeln!(
            report,
            "  Clustering coefficient: {:.3}",
            self.clustering_coefficient
        );
        let _ = writeln!(report, "  MST weight: {:.3}", self.mst_weight);

        if let Some(q) = self.most_connected_qubit {
            let _ = writeln!(report, "  Most connected qubit: {q}");
        }

        report.push_str("\nQuality metrics:\n");
        let _ = writeln!(report, "  Average gate error: {:.4}", self.avg_gate_error);
        let _ = writeln!(
            report,
            "  Average T2 time: {:.1} μs",
            self.avg_coherence_time
        );

        report.push_str("\nDegree distribution:\n");
        let mut degrees: Vec<_> = self.degree_distribution.iter().collect();
        degrees.sort_by_key(|&(k, _)| k);
        for (degree, count) in degrees {
            let _ = writeln!(report, "  {degree} connections: {count} qubits");
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_topology() {
        let topo = HardwareTopology::linear_topology(5);
        let analysis = topo.analyze();

        assert_eq!(analysis.num_qubits, 5);
        assert_eq!(analysis.num_connections, 4);
        assert_eq!(analysis.connected_components, 1);
        assert_eq!(analysis.max_distance, 4.0 * 0.01); // 4 hops with error 0.01 each
    }

    #[test]
    fn test_grid_topology() {
        let topo = HardwareTopology::grid_topology(3, 3);
        let analysis = topo.analyze();

        assert_eq!(analysis.num_qubits, 9);
        assert_eq!(analysis.connected_components, 1);
        assert!(analysis.average_distance > 0.0);
    }

    #[test]
    fn test_ibm_topology() {
        let topo = HardwareTopology::ibm_topology();
        let analysis = topo.analyze();

        assert_eq!(analysis.num_qubits, 5);
        assert_eq!(analysis.num_connections, 4);
        assert_eq!(analysis.most_connected_qubit, Some(1)); // Qubit 1 is central
    }

    #[test]
    fn test_find_optimal_subset() {
        let topo = HardwareTopology::grid_topology(3, 3);
        let subset = topo
            .find_optimal_subset(4)
            .expect("should find valid subset for 3x3 grid");

        assert_eq!(subset.len(), 4);
        // Should pick connected qubits
        let mut connected = false;
        for i in 0..subset.len() {
            for j in i + 1..subset.len() {
                if topo.gate_properties.contains_key(&(subset[i], subset[j]))
                    || topo.gate_properties.contains_key(&(subset[j], subset[i]))
                {
                    connected = true;
                    break;
                }
            }
            if connected {
                break;
            }
        }
        assert!(connected, "Subset {subset:?} is not connected");
    }
}
