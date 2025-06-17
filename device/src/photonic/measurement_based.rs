//! Measurement-Based Quantum Computing (One-Way Quantum Computing)
//!
//! This module implements measurement-based quantum computing using photonic cluster states,
//! enabling universal quantum computation through adaptive measurements.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;
use thiserror::Error;

use super::continuous_variable::{Complex, GaussianState};
use super::{PhotonicMode, PhotonicSystemType};
use crate::DeviceResult;

/// Errors for measurement-based quantum computing
#[derive(Error, Debug)]
pub enum MBQCError {
    #[error("Invalid cluster state: {0}")]
    InvalidClusterState(String),
    #[error("Measurement pattern invalid: {0}")]
    InvalidMeasurementPattern(String),
    #[error("Node not found in cluster: {0}")]
    NodeNotFound(usize),
    #[error("Measurement outcome not available: {0}")]
    MeasurementNotAvailable(String),
    #[error("Adaptive correction failed: {0}")]
    AdaptiveCorrectionFailed(String),
}

type MBQCResult<T> = Result<T, MBQCError>;

/// Cluster state node representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    /// Node identifier
    pub id: usize,
    /// Physical position (optional for visualization)
    pub position: Option<(f64, f64)>,
    /// Neighboring nodes
    pub neighbors: HashSet<usize>,
    /// Whether this node has been measured
    pub measured: bool,
    /// Measurement outcome (if measured)
    pub measurement_outcome: Option<bool>,
    /// Measurement basis (if measured)
    pub measurement_basis: Option<MeasurementBasis>,
    /// Role in computation
    pub role: NodeRole,
}

/// Role of a node in the cluster state computation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Input qubit
    Input(usize),
    /// Output qubit
    Output(usize),
    /// Computational ancilla
    Computational,
    /// Correction ancilla
    Correction,
}

/// Measurement basis for cluster state measurements
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasurementBasis {
    /// Measurement angle in XY plane
    pub angle: f64,
    /// Whether to include Z component
    pub include_z: bool,
}

impl MeasurementBasis {
    /// Create X basis measurement
    pub fn x() -> Self {
        Self {
            angle: 0.0,
            include_z: false,
        }
    }

    /// Create Y basis measurement
    pub fn y() -> Self {
        Self {
            angle: PI / 2.0,
            include_z: false,
        }
    }

    /// Create Z basis measurement
    pub fn z() -> Self {
        Self {
            angle: 0.0,
            include_z: true,
        }
    }

    /// Create arbitrary angle measurement in XY plane
    pub fn xy_angle(angle: f64) -> Self {
        Self {
            angle,
            include_z: false,
        }
    }

    /// Create measurement basis with angle correction
    pub fn with_correction(angle: f64, corrections: &[bool]) -> Self {
        let mut corrected_angle = angle;
        for &correction in corrections {
            if correction {
                corrected_angle += PI;
            }
        }
        Self {
            angle: corrected_angle,
            include_z: false,
        }
    }
}

/// Cluster state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterState {
    /// Nodes in the cluster
    pub nodes: HashMap<usize, ClusterNode>,
    /// Graph edges (node pairs)
    pub edges: HashSet<(usize, usize)>,
    /// Number of qubits
    pub num_qubits: usize,
    /// Cluster state type
    pub cluster_type: ClusterType,
}

/// Types of cluster states
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClusterType {
    /// Linear cluster (1D)
    Linear,
    /// Square lattice (2D)
    SquareLattice { width: usize, height: usize },
    /// Hexagonal lattice
    HexagonalLattice { radius: usize },
    /// Arbitrary graph
    Arbitrary,
    /// Tree cluster
    Tree { depth: usize },
    /// Complete graph
    Complete,
}

impl ClusterState {
    /// Create a linear cluster state
    pub fn linear(length: usize) -> Self {
        let mut nodes = HashMap::new();
        let mut edges = HashSet::new();

        // Create nodes
        for i in 0..length {
            let role = if i == 0 {
                NodeRole::Input(0)
            } else if i == length - 1 {
                NodeRole::Output(0)
            } else {
                NodeRole::Computational
            };

            let mut neighbors = HashSet::new();
            if i > 0 {
                neighbors.insert(i - 1);
                edges.insert((i - 1, i));
            }
            if i < length - 1 {
                neighbors.insert(i + 1);
            }

            nodes.insert(
                i,
                ClusterNode {
                    id: i,
                    position: Some((i as f64, 0.0)),
                    neighbors,
                    measured: false,
                    measurement_outcome: None,
                    measurement_basis: None,
                    role,
                },
            );
        }

        Self {
            nodes,
            edges,
            num_qubits: length,
            cluster_type: ClusterType::Linear,
        }
    }

    /// Create a 2D square lattice cluster state
    pub fn square_lattice(width: usize, height: usize) -> Self {
        let mut nodes = HashMap::new();
        let mut edges = HashSet::new();

        for i in 0..height {
            for j in 0..width {
                let node_id = i * width + j;
                let mut neighbors = HashSet::new();

                // Add neighbors (up, down, left, right)
                if i > 0 {
                    let neighbor = (i - 1) * width + j;
                    neighbors.insert(neighbor);
                    edges.insert((node_id.min(neighbor), node_id.max(neighbor)));
                }
                if i < height - 1 {
                    let neighbor = (i + 1) * width + j;
                    neighbors.insert(neighbor);
                }
                if j > 0 {
                    let neighbor = i * width + (j - 1);
                    neighbors.insert(neighbor);
                    edges.insert((node_id.min(neighbor), node_id.max(neighbor)));
                }
                if j < width - 1 {
                    let neighbor = i * width + (j + 1);
                    neighbors.insert(neighbor);
                }

                // Determine role
                let role = if i == 0 && j == 0 {
                    NodeRole::Input(0)
                } else if i == height - 1 && j == width - 1 {
                    NodeRole::Output(0)
                } else {
                    NodeRole::Computational
                };

                nodes.insert(
                    node_id,
                    ClusterNode {
                        id: node_id,
                        position: Some((j as f64, i as f64)),
                        neighbors,
                        measured: false,
                        measurement_outcome: None,
                        measurement_basis: None,
                        role,
                    },
                );
            }
        }

        Self {
            nodes,
            edges,
            num_qubits: width * height,
            cluster_type: ClusterType::SquareLattice { width, height },
        }
    }

    /// Add an edge to the cluster state
    pub fn add_edge(&mut self, node1: usize, node2: usize) -> MBQCResult<()> {
        if !self.nodes.contains_key(&node1) || !self.nodes.contains_key(&node2) {
            return Err(MBQCError::NodeNotFound(node1.max(node2)));
        }

        self.edges.insert((node1.min(node2), node1.max(node2)));
        self.nodes.get_mut(&node1).unwrap().neighbors.insert(node2);
        self.nodes.get_mut(&node2).unwrap().neighbors.insert(node1);

        Ok(())
    }

    /// Remove an edge from the cluster state
    pub fn remove_edge(&mut self, node1: usize, node2: usize) -> MBQCResult<()> {
        self.edges.remove(&(node1.min(node2), node1.max(node2)));

        if let Some(node) = self.nodes.get_mut(&node1) {
            node.neighbors.remove(&node2);
        }
        if let Some(node) = self.nodes.get_mut(&node2) {
            node.neighbors.remove(&node1);
        }

        Ok(())
    }

    /// Measure a node in the specified basis
    pub fn measure_node(&mut self, node_id: usize, basis: MeasurementBasis) -> MBQCResult<bool> {
        // Check if node exists and hasn't been measured
        {
            let node = self
                .nodes
                .get(&node_id)
                .ok_or(MBQCError::NodeNotFound(node_id))?;

            if node.measured {
                return Err(MBQCError::InvalidMeasurementPattern(format!(
                    "Node {} already measured",
                    node_id
                )));
            }
        }

        // Simulate measurement outcome (in a real implementation, this would come from hardware)
        let outcome = self.simulate_measurement_outcome(node_id, basis)?;

        // Update node with measurement results
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.measured = true;
            node.measurement_outcome = Some(outcome);
            node.measurement_basis = Some(basis);
        }

        Ok(outcome)
    }

    /// Simulate measurement outcome based on cluster state
    fn simulate_measurement_outcome(
        &self,
        node_id: usize,
        basis: MeasurementBasis,
    ) -> MBQCResult<bool> {
        // Simplified simulation - in practice this would depend on the actual quantum state

        // For demonstration, use a probabilistic outcome
        let probability = match basis.angle {
            a if (a - 0.0).abs() < 1e-6 => 0.5,      // X measurement
            a if (a - PI / 2.0).abs() < 1e-6 => 0.5, // Y measurement
            a if (a - PI).abs() < 1e-6 => 0.3,       // -X measurement
            _ => 0.5,                                // General case
        };

        Ok(rand::random::<f64>() < probability)
    }

    /// Get all unmeasured neighbors of a node
    pub fn unmeasured_neighbors(&self, node_id: usize) -> Vec<usize> {
        if let Some(node) = self.nodes.get(&node_id) {
            node.neighbors
                .iter()
                .filter(|&&neighbor_id| {
                    if let Some(neighbor) = self.nodes.get(&neighbor_id) {
                        !neighbor.measured
                    } else {
                        false
                    }
                })
                .copied()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Check if a measurement pattern is valid (causal)
    pub fn is_measurement_pattern_valid(&self, pattern: &MeasurementPattern) -> bool {
        // Check that measurements respect causality in the cluster
        let mut measured_nodes = HashSet::new();

        for measurement in &pattern.measurements {
            // Check if all dependencies are satisfied
            for &dependency in &measurement.dependencies {
                if !measured_nodes.contains(&dependency) {
                    return false;
                }
            }

            measured_nodes.insert(measurement.node_id);
        }

        true
    }

    /// Get the effective logical state after measurements
    pub fn get_logical_state(&self) -> MBQCResult<LogicalState> {
        let mut logical_bits = Vec::new();

        // Extract outcomes from output nodes
        for node in self.nodes.values() {
            if let NodeRole::Output(index) = node.role {
                if let Some(outcome) = node.measurement_outcome {
                    logical_bits.push((index, outcome));
                } else {
                    return Err(MBQCError::MeasurementNotAvailable(format!(
                        "Output node {} not measured",
                        node.id
                    )));
                }
            }
        }

        // Sort by output index
        logical_bits.sort_by_key(|&(index, _)| index);

        Ok(LogicalState {
            bits: logical_bits.into_iter().map(|(_, bit)| bit).collect(),
            fidelity: self.estimate_logical_fidelity(),
        })
    }

    /// Estimate the fidelity of the logical state
    fn estimate_logical_fidelity(&self) -> f64 {
        // Simplified fidelity estimation based on number of measurements
        let total_nodes = self.nodes.len();
        let measured_nodes = self.nodes.values().filter(|n| n.measured).count();

        // More measurements generally lead to higher fidelity
        0.9 + 0.1 * (measured_nodes as f64 / total_nodes as f64)
    }
}

/// Measurement pattern for MBQC computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementPattern {
    /// Sequence of measurements
    pub measurements: Vec<MeasurementStep>,
    /// Adaptive corrections
    pub corrections: Vec<AdaptiveCorrection>,
}

/// Single measurement step in MBQC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementStep {
    /// Node to measure
    pub node_id: usize,
    /// Measurement basis
    pub basis: MeasurementBasis,
    /// Dependencies (nodes that must be measured first)
    pub dependencies: Vec<usize>,
    /// Whether this measurement is adaptive
    pub adaptive: bool,
}

/// Adaptive correction based on previous measurement outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCorrection {
    /// Target node for correction
    pub target_node: usize,
    /// Nodes whose outcomes determine the correction
    pub condition_nodes: Vec<usize>,
    /// Correction function (angle modification)
    pub correction_type: CorrectionType,
}

/// Types of adaptive corrections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrectionType {
    /// Add π to angle if condition is met
    PiCorrection,
    /// Add π/2 to angle if condition is met
    HalfPiCorrection,
    /// Custom angle correction
    CustomAngle(f64),
    /// Basis change
    BasisChange(MeasurementBasis),
}

/// Logical quantum state after MBQC computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalState {
    /// Logical bit values
    pub bits: Vec<bool>,
    /// Estimated fidelity
    pub fidelity: f64,
}

/// MBQC computation engine
pub struct MBQCComputer {
    /// Current cluster state
    pub cluster: ClusterState,
    /// Measurement history
    pub measurement_history: Vec<(usize, MeasurementBasis, bool)>,
}

impl MBQCComputer {
    /// Create new MBQC computer with given cluster state
    pub fn new(cluster: ClusterState) -> Self {
        Self {
            cluster,
            measurement_history: Vec::new(),
        }
    }

    /// Execute a measurement pattern
    pub fn execute_pattern(&mut self, pattern: &MeasurementPattern) -> MBQCResult<LogicalState> {
        // Validate pattern
        if !self.cluster.is_measurement_pattern_valid(pattern) {
            return Err(MBQCError::InvalidMeasurementPattern(
                "Measurement pattern violates causality".to_string(),
            ));
        }

        // Execute measurements in order
        for measurement in &pattern.measurements {
            let mut basis = measurement.basis;

            // Apply adaptive corrections if needed
            if measurement.adaptive {
                basis = self.apply_adaptive_corrections(measurement, &pattern.corrections)?;
            }

            let outcome = self.cluster.measure_node(measurement.node_id, basis)?;
            self.measurement_history
                .push((measurement.node_id, basis, outcome));
        }

        // Extract logical state
        self.cluster.get_logical_state()
    }

    /// Apply adaptive corrections to measurement basis
    fn apply_adaptive_corrections(
        &self,
        measurement: &MeasurementStep,
        corrections: &[AdaptiveCorrection],
    ) -> MBQCResult<MeasurementBasis> {
        let mut basis = measurement.basis;

        for correction in corrections {
            if correction.target_node == measurement.node_id {
                // Check if correction condition is met
                let condition_met =
                    self.evaluate_correction_condition(&correction.condition_nodes)?;

                if condition_met {
                    basis = self.apply_correction(basis, &correction.correction_type);
                }
            }
        }

        Ok(basis)
    }

    /// Evaluate correction condition based on measurement outcomes
    fn evaluate_correction_condition(&self, condition_nodes: &[usize]) -> MBQCResult<bool> {
        let mut parity = false;

        for &node_id in condition_nodes {
            let node = self
                .cluster
                .nodes
                .get(&node_id)
                .ok_or(MBQCError::NodeNotFound(node_id))?;

            let outcome = node
                .measurement_outcome
                .ok_or(MBQCError::MeasurementNotAvailable(format!(
                    "Node {} not measured",
                    node_id
                )))?;

            parity ^= outcome;
        }

        Ok(parity)
    }

    /// Apply correction to measurement basis
    fn apply_correction(
        &self,
        basis: MeasurementBasis,
        correction: &CorrectionType,
    ) -> MeasurementBasis {
        match correction {
            CorrectionType::PiCorrection => MeasurementBasis {
                angle: basis.angle + PI,
                include_z: basis.include_z,
            },
            CorrectionType::HalfPiCorrection => MeasurementBasis {
                angle: basis.angle + PI / 2.0,
                include_z: basis.include_z,
            },
            CorrectionType::CustomAngle(angle) => MeasurementBasis {
                angle: basis.angle + angle,
                include_z: basis.include_z,
            },
            CorrectionType::BasisChange(new_basis) => *new_basis,
        }
    }

    /// Implement a logical Hadamard gate using MBQC
    pub fn logical_hadamard_gate(
        &self,
        input_node: usize,
        output_node: usize,
    ) -> MBQCResult<MeasurementPattern> {
        let measurements = vec![MeasurementStep {
            node_id: input_node,
            basis: MeasurementBasis::xy_angle(PI / 4.0), // π/4 measurement for Hadamard
            dependencies: vec![],
            adaptive: false,
        }];

        Ok(MeasurementPattern {
            measurements,
            corrections: vec![],
        })
    }

    /// Implement a logical CNOT gate using MBQC
    pub fn logical_cnot_gate(
        &self,
        control_input: usize,
        target_input: usize,
        control_output: usize,
        target_output: usize,
        ancilla_nodes: &[usize],
    ) -> MBQCResult<MeasurementPattern> {
        if ancilla_nodes.len() < 2 {
            return Err(MBQCError::InvalidMeasurementPattern(
                "CNOT requires at least 2 ancilla nodes".to_string(),
            ));
        }

        let measurements = vec![
            // Measure ancilla nodes to implement CNOT
            MeasurementStep {
                node_id: ancilla_nodes[0],
                basis: MeasurementBasis::x(),
                dependencies: vec![],
                adaptive: false,
            },
            MeasurementStep {
                node_id: ancilla_nodes[1],
                basis: MeasurementBasis::x(),
                dependencies: vec![ancilla_nodes[0]],
                adaptive: true,
            },
        ];

        let corrections = vec![AdaptiveCorrection {
            target_node: ancilla_nodes[1],
            condition_nodes: vec![control_input],
            correction_type: CorrectionType::PiCorrection,
        }];

        Ok(MeasurementPattern {
            measurements,
            corrections,
        })
    }

    /// Get computation statistics
    pub fn get_statistics(&self) -> MBQCStatistics {
        let total_nodes = self.cluster.nodes.len();
        let measured_nodes = self.cluster.nodes.values().filter(|n| n.measured).count();
        let unmeasured_nodes = total_nodes - measured_nodes;

        let input_nodes = self
            .cluster
            .nodes
            .values()
            .filter(|n| matches!(n.role, NodeRole::Input(_)))
            .count();
        let output_nodes = self
            .cluster
            .nodes
            .values()
            .filter(|n| matches!(n.role, NodeRole::Output(_)))
            .count();

        MBQCStatistics {
            total_nodes,
            measured_nodes,
            unmeasured_nodes,
            input_nodes,
            output_nodes,
            total_edges: self.cluster.edges.len(),
            measurement_history_length: self.measurement_history.len(),
        }
    }
}

/// Statistics for MBQC computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MBQCStatistics {
    pub total_nodes: usize,
    pub measured_nodes: usize,
    pub unmeasured_nodes: usize,
    pub input_nodes: usize,
    pub output_nodes: usize,
    pub total_edges: usize,
    pub measurement_history_length: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_cluster_creation() {
        let cluster = ClusterState::linear(5);
        assert_eq!(cluster.num_qubits, 5);
        assert_eq!(cluster.nodes.len(), 5);
        assert_eq!(cluster.edges.len(), 4);

        // Check connectivity
        assert!(cluster.nodes[&0].neighbors.contains(&1));
        assert!(cluster.nodes[&2].neighbors.contains(&1));
        assert!(cluster.nodes[&2].neighbors.contains(&3));
    }

    #[test]
    fn test_square_lattice_creation() {
        let cluster = ClusterState::square_lattice(3, 3);
        assert_eq!(cluster.num_qubits, 9);
        assert_eq!(cluster.nodes.len(), 9);

        // Check corner node has 2 neighbors
        assert_eq!(cluster.nodes[&0].neighbors.len(), 2);

        // Check center node has 4 neighbors
        assert_eq!(cluster.nodes[&4].neighbors.len(), 4);
    }

    #[test]
    fn test_measurement() {
        let mut cluster = ClusterState::linear(3);
        let outcome = cluster.measure_node(1, MeasurementBasis::x()).unwrap();

        assert!(cluster.nodes[&1].measured);
        assert_eq!(cluster.nodes[&1].measurement_outcome, Some(outcome));
    }

    #[test]
    fn test_mbqc_computer() {
        let cluster = ClusterState::linear(3);
        let mut computer = MBQCComputer::new(cluster);

        let pattern = MeasurementPattern {
            measurements: vec![
                MeasurementStep {
                    node_id: 1,
                    basis: MeasurementBasis::x(),
                    dependencies: vec![],
                    adaptive: false,
                },
                MeasurementStep {
                    node_id: 2,
                    basis: MeasurementBasis::z(),
                    dependencies: vec![],
                    adaptive: false,
                },
            ],
            corrections: vec![],
        };

        let result = computer.execute_pattern(&pattern);
        assert!(result.is_ok());
    }

    #[test]
    fn test_logical_hadamard() {
        let cluster = ClusterState::linear(3);
        let computer = MBQCComputer::new(cluster);

        let pattern = computer.logical_hadamard_gate(0, 2).unwrap();
        assert_eq!(pattern.measurements.len(), 1);
        assert!((pattern.measurements[0].basis.angle - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_correction() {
        let mut cluster = ClusterState::linear(4);
        cluster.measure_node(0, MeasurementBasis::x()).unwrap();

        let computer = MBQCComputer::new(cluster);
        let corrections = vec![AdaptiveCorrection {
            target_node: 2,
            condition_nodes: vec![0],
            correction_type: CorrectionType::PiCorrection,
        }];

        let condition_met = computer.evaluate_correction_condition(&[0]).unwrap();
        assert!(condition_met == computer.cluster.nodes[&0].measurement_outcome.unwrap());
    }
}
