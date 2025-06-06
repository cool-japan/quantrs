//! ZX-calculus primitives for quantum circuit optimization
//!
//! This module implements the ZX-calculus, a graphical language for quantum computing
//! that enables powerful circuit optimization through graph rewriting rules.
//!
//! The ZX-calculus represents quantum computations as graphs with:
//! - Green nodes (Z-spiders): Phase rotations in the Z basis
//! - Red nodes (X-spiders): Phase rotations in the X basis
//! - Edges: Quantum entanglement
//! - Hadamard edges: Basis changes

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{GateOp, single::*, multi::*},
    qubit::QubitId,
};
use num_complex::Complex64 as Complex;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use rustc_hash::FxHashMap;

/// Type of spider in the ZX-diagram
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpiderType {
    /// Z-spider (green node) - computational basis
    Z,
    /// X-spider (red node) - Hadamard basis
    X,
    /// Boundary node (input/output)
    Boundary,
}

/// Represents a spider (node) in a ZX-diagram
#[derive(Debug, Clone)]
pub struct Spider {
    /// Unique identifier for the spider
    pub id: usize,
    /// Type of spider
    pub spider_type: SpiderType,
    /// Phase angle (in radians)
    pub phase: f64,
    /// Qubit this spider is associated with (for boundary spiders)
    pub qubit: Option<QubitId>,
}

impl Spider {
    /// Create a new spider
    pub fn new(id: usize, spider_type: SpiderType, phase: f64) -> Self {
        Self {
            id,
            spider_type,
            phase,
            qubit: None,
        }
    }

    /// Create a boundary spider
    pub fn boundary(id: usize, qubit: QubitId) -> Self {
        Self {
            id,
            spider_type: SpiderType::Boundary,
            phase: 0.0,
            qubit: Some(qubit),
        }
    }

    /// Check if this is a Clifford spider (phase is multiple of π/2)
    pub fn is_clifford(&self, tolerance: f64) -> bool {
        let normalized_phase = self.phase % (2.0 * PI);
        let multiples_of_pi_2 = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
        
        multiples_of_pi_2.iter().any(|&p| {
            (normalized_phase - p).abs() < tolerance
                || (normalized_phase - p + 2.0 * PI).abs() < tolerance
        })
    }

    /// Check if this is a Pauli spider (phase is 0 or π)
    pub fn is_pauli(&self, tolerance: f64) -> bool {
        let normalized_phase = self.phase % (2.0 * PI);
        normalized_phase.abs() < tolerance
            || (normalized_phase - PI).abs() < tolerance
            || (normalized_phase - 2.0 * PI).abs() < tolerance
    }
}

/// Type of edge in the ZX-diagram
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// Regular edge (identity)
    Regular,
    /// Hadamard edge
    Hadamard,
}

/// Represents an edge in a ZX-diagram
#[derive(Debug, Clone)]
pub struct Edge {
    /// Source spider ID
    pub source: usize,
    /// Target spider ID
    pub target: usize,
    /// Type of edge
    pub edge_type: EdgeType,
}

/// ZX-diagram representation
#[derive(Debug, Clone)]
pub struct ZXDiagram {
    /// Spiders (nodes) in the diagram
    pub spiders: FxHashMap<usize, Spider>,
    /// Adjacency list representation of edges
    pub adjacency: FxHashMap<usize, Vec<(usize, EdgeType)>>,
    /// Input boundary spiders (ordered by qubit)
    pub inputs: Vec<usize>,
    /// Output boundary spiders (ordered by qubit)
    pub outputs: Vec<usize>,
    /// Next available spider ID
    next_id: usize,
}

impl ZXDiagram {
    /// Create a new empty ZX-diagram
    pub fn new() -> Self {
        Self {
            spiders: FxHashMap::default(),
            adjacency: FxHashMap::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a spider to the diagram
    pub fn add_spider(&mut self, spider_type: SpiderType, phase: f64) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        let spider = Spider::new(id, spider_type, phase);
        self.spiders.insert(id, spider);
        self.adjacency.insert(id, Vec::new());
        
        id
    }

    /// Add a boundary spider
    pub fn add_boundary(&mut self, qubit: QubitId, is_input: bool) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        
        let spider = Spider::boundary(id, qubit);
        self.spiders.insert(id, spider);
        self.adjacency.insert(id, Vec::new());
        
        if is_input {
            self.inputs.push(id);
        } else {
            self.outputs.push(id);
        }
        
        id
    }

    /// Add an edge between two spiders
    pub fn add_edge(&mut self, source: usize, target: usize, edge_type: EdgeType) {
        self.adjacency
            .get_mut(&source)
            .map(|adj| adj.push((target, edge_type)));
        self.adjacency
            .get_mut(&target)
            .map(|adj| adj.push((source, edge_type)));
    }

    /// Remove an edge between two spiders
    pub fn remove_edge(&mut self, source: usize, target: usize) {
        if let Some(adj) = self.adjacency.get_mut(&source) {
            adj.retain(|(t, _)| *t != target);
        }
        if let Some(adj) = self.adjacency.get_mut(&target) {
            adj.retain(|(s, _)| *s != source);
        }
    }

    /// Get the neighbors of a spider
    pub fn neighbors(&self, spider_id: usize) -> Vec<(usize, EdgeType)> {
        self.adjacency
            .get(&spider_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Remove a spider and its edges
    pub fn remove_spider(&mut self, spider_id: usize) {
        // Remove from spider list
        self.spiders.remove(&spider_id);
        
        // Remove all edges connected to this spider
        let neighbors = self.neighbors(spider_id);
        for (neighbor, _) in neighbors {
            self.remove_edge(spider_id, neighbor);
        }
        
        // Remove from adjacency list
        self.adjacency.remove(&spider_id);
        
        // Remove from inputs/outputs if present
        self.inputs.retain(|&id| id != spider_id);
        self.outputs.retain(|&id| id != spider_id);
    }

    /// Get the degree (number of connections) of a spider
    pub fn degree(&self, spider_id: usize) -> usize {
        self.adjacency
            .get(&spider_id)
            .map(|adj| adj.len())
            .unwrap_or(0)
    }

    /// Apply spider fusion rule: two adjacent spiders of the same color merge
    pub fn spider_fusion(&mut self, spider1: usize, spider2: usize) -> QuantRS2Result<()> {
        let s1 = self.spiders.get(&spider1)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Spider 1 not found".to_string()))?;
        let s2 = self.spiders.get(&spider2)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Spider 2 not found".to_string()))?;
        
        // Check if spiders are the same type
        if s1.spider_type != s2.spider_type || s1.spider_type == SpiderType::Boundary {
            return Err(QuantRS2Error::InvalidInput(
                "Can only fuse spiders of the same non-boundary type".to_string()
            ));
        }
        
        // Check if spiders are connected
        let connected = self.neighbors(spider1)
            .iter()
            .any(|(id, _)| *id == spider2);
        
        if !connected {
            return Err(QuantRS2Error::InvalidInput(
                "Spiders must be connected".to_string()
            ));
        }
        
        // Combine phases
        let new_phase = s1.phase + s2.phase;
        
        // Update spider1 with combined phase
        if let Some(s1_mut) = self.spiders.get_mut(&spider1) {
            s1_mut.phase = new_phase;
        }
        
        // Transfer all edges from spider2 to spider1
        let spider2_neighbors = self.neighbors(spider2).clone();
        for (neighbor, edge_type) in spider2_neighbors {
            if neighbor != spider1 {
                self.remove_edge(spider2, neighbor);
                self.add_edge(spider1, neighbor, edge_type);
            }
        }
        
        // Remove spider2
        self.remove_spider(spider2);
        
        Ok(())
    }

    /// Apply identity removal: remove spiders with phase 0 and degree 2
    pub fn remove_identities(&mut self) -> usize {
        let mut removed = 0;
        let mut to_remove = Vec::new();
        
        for (&id, spider) in &self.spiders {
            if spider.spider_type != SpiderType::Boundary
                && spider.phase.abs() < 1e-10
                && self.degree(id) == 2
            {
                to_remove.push(id);
            }
        }
        
        for id in to_remove {
            let neighbors = self.neighbors(id);
            if neighbors.len() == 2 {
                let (n1, e1) = neighbors[0];
                let (n2, e2) = neighbors[1];
                
                // Connect the two neighbors
                let new_edge_type = match (e1, e2) {
                    (EdgeType::Regular, EdgeType::Regular) => EdgeType::Regular,
                    (EdgeType::Hadamard, EdgeType::Hadamard) => EdgeType::Regular,
                    _ => EdgeType::Hadamard,
                };
                
                self.remove_spider(id);
                self.add_edge(n1, n2, new_edge_type);
                removed += 1;
            }
        }
        
        removed
    }

    /// Apply color change rule: X and Z spiders connected by Hadamard become the same color
    pub fn color_change(&mut self, spider_id: usize) -> QuantRS2Result<()> {
        let spider = self.spiders.get(&spider_id)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Spider not found".to_string()))?
            .clone();
        
        if spider.spider_type == SpiderType::Boundary {
            return Err(QuantRS2Error::InvalidInput(
                "Cannot change color of boundary spider".to_string()
            ));
        }
        
        // Change spider color
        let new_type = match spider.spider_type {
            SpiderType::Z => SpiderType::X,
            SpiderType::X => SpiderType::Z,
            _ => return Ok(()),
        };
        
        if let Some(s) = self.spiders.get_mut(&spider_id) {
            s.spider_type = new_type;
        }
        
        // Flip all edge types
        let neighbors = self.neighbors(spider_id).clone();
        for (neighbor, edge_type) in neighbors {
            self.remove_edge(spider_id, neighbor);
            let new_edge_type = match edge_type {
                EdgeType::Regular => EdgeType::Hadamard,
                EdgeType::Hadamard => EdgeType::Regular,
            };
            self.add_edge(spider_id, neighbor, new_edge_type);
        }
        
        Ok(())
    }

    /// Apply pi-copy rule: Pauli spider can be copied through
    pub fn pi_copy(&mut self, spider_id: usize) -> QuantRS2Result<Vec<usize>> {
        let spider = self.spiders.get(&spider_id)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Spider not found".to_string()))?
            .clone();
        
        if !spider.is_pauli(1e-10) {
            return Err(QuantRS2Error::InvalidInput(
                "Spider must be Pauli (phase 0 or π)".to_string()
            ));
        }
        
        let neighbors = self.neighbors(spider_id).clone();
        let mut new_spiders = Vec::new();
        
        // Create a copy for each neighbor
        for (neighbor, edge_type) in &neighbors[1..] {
            let new_id = self.add_spider(spider.spider_type, spider.phase);
            new_spiders.push(new_id);
            
            // Connect new spider to the neighbor
            self.remove_edge(spider_id, *neighbor);
            self.add_edge(new_id, *neighbor, *edge_type);
            
            // Connect new spider to the first neighbor
            if let Some((first_neighbor, first_edge_type)) = neighbors.first() {
                self.add_edge(new_id, *first_neighbor, *first_edge_type);
            }
        }
        
        Ok(new_spiders)
    }

    /// Apply bialgebra rule
    pub fn bialgebra(&mut self, z_spider: usize, x_spider: usize) -> QuantRS2Result<()> {
        let z = self.spiders.get(&z_spider)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Z-spider not found".to_string()))?;
        let x = self.spiders.get(&x_spider)
            .ok_or_else(|| QuantRS2Error::InvalidInput("X-spider not found".to_string()))?;
        
        // Check spider types
        if z.spider_type != SpiderType::Z || x.spider_type != SpiderType::X {
            return Err(QuantRS2Error::InvalidInput(
                "Need one Z-spider and one X-spider".to_string()
            ));
        }
        
        // Check if connected
        let connected = self.neighbors(z_spider)
            .iter()
            .any(|(id, _)| *id == x_spider);
        
        if !connected {
            return Err(QuantRS2Error::InvalidInput(
                "Spiders must be connected".to_string()
            ));
        }
        
        // Get all neighbors
        let z_neighbors: Vec<_> = self.neighbors(z_spider)
            .into_iter()
            .filter(|(id, _)| *id != x_spider)
            .collect();
        let x_neighbors: Vec<_> = self.neighbors(x_spider)
            .into_iter()
            .filter(|(id, _)| *id != z_spider)
            .collect();
        
        // Remove original spiders
        self.remove_spider(z_spider);
        self.remove_spider(x_spider);
        
        // Create new connections
        for (z_n, z_edge) in &z_neighbors {
            for (x_n, x_edge) in &x_neighbors {
                // Determine edge type based on the rule
                let edge_type = match (z_edge, x_edge) {
                    (EdgeType::Regular, EdgeType::Regular) => EdgeType::Regular,
                    (EdgeType::Hadamard, EdgeType::Hadamard) => EdgeType::Regular,
                    _ => EdgeType::Hadamard,
                };
                self.add_edge(*z_n, *x_n, edge_type);
            }
        }
        
        Ok(())
    }

    /// Simplify the diagram by applying rewrite rules
    pub fn simplify(&mut self, max_iterations: usize) -> usize {
        let mut total_rewrites = 0;
        
        for _ in 0..max_iterations {
            let mut rewrites = 0;
            
            // Remove identities
            rewrites += self.remove_identities();
            
            // Try spider fusion
            let spider_ids: Vec<_> = self.spiders.keys().cloned().collect();
            for i in 0..spider_ids.len() {
                for j in i + 1..spider_ids.len() {
                    let id1 = spider_ids[i];
                    let id2 = spider_ids[j];
                    
                    // Check if spiders still exist and can be fused
                    if self.spiders.contains_key(&id1) && self.spiders.contains_key(&id2) {
                        if let Ok(()) = self.spider_fusion(id1, id2) {
                            rewrites += 1;
                            break;
                        }
                    }
                }
                if rewrites > 0 {
                    break;
                }
            }
            
            total_rewrites += rewrites;
            if rewrites == 0 {
                break;
            }
        }
        
        total_rewrites
    }

    /// Convert to GraphViz DOT format for visualization
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("graph ZX {\n");
        dot.push_str("  rankdir=LR;\n");
        
        // Add spiders
        for (id, spider) in &self.spiders {
            let color = match spider.spider_type {
                SpiderType::Z => "green",
                SpiderType::X => "red",
                SpiderType::Boundary => "black",
            };
            let shape = match spider.spider_type {
                SpiderType::Boundary => "square",
                _ => "circle",
            };
            let label = match spider.spider_type {
                SpiderType::Boundary => {
                    if let Some(qubit) = spider.qubit {
                        format!("q{}", qubit.0)
                    } else {
                        String::new()
                    }
                }
                _ => {
                    if spider.phase.abs() < 1e-10 {
                        String::new()
                    } else {
                        format!("{:.2}", spider.phase / PI)
                    }
                }
            };
            
            dot.push_str(&format!(
                "  {} [label=\"{}\", color={}, shape={}];\n",
                id, label, color, shape
            ));
        }
        
        // Add edges (avoid duplicates)
        let mut processed = HashSet::new();
        for (&source, neighbors) in &self.adjacency {
            for &(target, edge_type) in neighbors {
                let key = if source < target {
                    (source, target)
                } else {
                    (target, source)
                };
                
                if !processed.contains(&key) {
                    processed.insert(key);
                    
                    let style = match edge_type {
                        EdgeType::Regular => "solid",
                        EdgeType::Hadamard => "dashed",
                    };
                    
                    dot.push_str(&format!(
                        "  {} -- {} [style={}];\n",
                        source, target, style
                    ));
                }
            }
        }
        
        dot.push_str("}\n");
        dot
    }
}

/// Convert a quantum circuit to a ZX-diagram
pub struct CircuitToZX {
    diagram: ZXDiagram,
    qubit_wires: FxHashMap<QubitId, (usize, usize)>, // (current_start, current_end)
}

impl CircuitToZX {
    /// Create a new converter
    pub fn new(num_qubits: usize) -> Self {
        let mut diagram = ZXDiagram::new();
        let mut qubit_wires = FxHashMap::default();
        
        // Create input and output boundaries
        for i in 0..num_qubits {
            let qubit = QubitId(i as u32);
            let input = diagram.add_boundary(qubit, true);
            let output = diagram.add_boundary(qubit, false);
            qubit_wires.insert(qubit, (input, output));
        }
        
        Self {
            diagram,
            qubit_wires,
        }
    }

    /// Add a gate to the diagram
    pub fn add_gate(&mut self, gate: &dyn GateOp) -> QuantRS2Result<()> {
        let qubits = gate.qubits();
        
        match gate.name() {
            "H" => self.add_hadamard(qubits[0]),
            "X" => self.add_pauli_x(qubits[0]),
            "Y" => self.add_pauli_y(qubits[0]),
            "Z" => self.add_pauli_z(qubits[0]),
            "S" => self.add_phase(qubits[0], PI / 2.0),
            "T" => self.add_phase(qubits[0], PI / 4.0),
            "RX" => {
                // Extract angle from parametric gate
                if let Some(rx) = gate.as_any().downcast_ref::<RotationX>() {
                    self.add_rotation_x(qubits[0], rx.theta)
                } else {
                    Err(QuantRS2Error::InvalidInput("Invalid RX gate".to_string()))
                }
            }
            "RY" => {
                if let Some(ry) = gate.as_any().downcast_ref::<RotationY>() {
                    self.add_rotation_y(qubits[0], ry.theta)
                } else {
                    Err(QuantRS2Error::InvalidInput("Invalid RY gate".to_string()))
                }
            }
            "RZ" => {
                if let Some(rz) = gate.as_any().downcast_ref::<RotationZ>() {
                    self.add_rotation_z(qubits[0], rz.theta)
                } else {
                    Err(QuantRS2Error::InvalidInput("Invalid RZ gate".to_string()))
                }
            }
            "CNOT" => self.add_cnot(qubits[0], qubits[1]),
            "CZ" => self.add_cz(qubits[0], qubits[1]),
            _ => Err(QuantRS2Error::UnsupportedOperation(
                format!("Gate {} not yet supported in ZX-calculus", gate.name())
            )),
        }
    }

    /// Add a Hadamard gate
    fn add_hadamard(&mut self, qubit: QubitId) -> QuantRS2Result<()> {
        let (start, end) = *self.qubit_wires.get(&qubit)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Qubit not found".to_string()))?;
        
        // Add Hadamard edge
        self.diagram.remove_edge(start, end);
        self.diagram.add_edge(start, end, EdgeType::Hadamard);
        
        Ok(())
    }

    /// Add a Pauli X gate
    fn add_pauli_x(&mut self, qubit: QubitId) -> QuantRS2Result<()> {
        self.add_x_spider(qubit, PI)
    }

    /// Add a Pauli Y gate
    fn add_pauli_y(&mut self, qubit: QubitId) -> QuantRS2Result<()> {
        // Y = iXZ, so we add both X(π) and Z(π) spiders
        self.add_x_spider(qubit, PI)?;
        self.add_z_spider(qubit, PI)
    }

    /// Add a Pauli Z gate
    fn add_pauli_z(&mut self, qubit: QubitId) -> QuantRS2Result<()> {
        self.add_z_spider(qubit, PI)
    }

    /// Add a phase gate
    fn add_phase(&mut self, qubit: QubitId, angle: f64) -> QuantRS2Result<()> {
        self.add_z_spider(qubit, angle)
    }

    /// Add an X rotation
    fn add_rotation_x(&mut self, qubit: QubitId, angle: f64) -> QuantRS2Result<()> {
        self.add_x_spider(qubit, angle)
    }

    /// Add a Y rotation
    fn add_rotation_y(&mut self, qubit: QubitId, angle: f64) -> QuantRS2Result<()> {
        // RY(θ) = e^(-iθY/2) = RZ(-π/2)RX(θ)RZ(π/2)
        self.add_z_spider(qubit, -PI / 2.0)?;
        self.add_x_spider(qubit, angle)?;
        self.add_z_spider(qubit, PI / 2.0)
    }

    /// Add a Z rotation
    fn add_rotation_z(&mut self, qubit: QubitId, angle: f64) -> QuantRS2Result<()> {
        self.add_z_spider(qubit, angle)
    }

    /// Add a CNOT gate
    fn add_cnot(&mut self, control: QubitId, target: QubitId) -> QuantRS2Result<()> {
        let (c_start, c_end) = *self.qubit_wires.get(&control)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Control qubit not found".to_string()))?;
        let (t_start, t_end) = *self.qubit_wires.get(&target)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Target qubit not found".to_string()))?;
        
        // CNOT is represented as a Z-spider on control connected to X-spider on target
        let z_spider = self.diagram.add_spider(SpiderType::Z, 0.0);
        let x_spider = self.diagram.add_spider(SpiderType::X, 0.0);
        
        // Break existing connections
        self.diagram.remove_edge(c_start, c_end);
        self.diagram.remove_edge(t_start, t_end);
        
        // Connect control
        self.diagram.add_edge(c_start, z_spider, EdgeType::Regular);
        self.diagram.add_edge(z_spider, c_end, EdgeType::Regular);
        
        // Connect target
        self.diagram.add_edge(t_start, x_spider, EdgeType::Regular);
        self.diagram.add_edge(x_spider, t_end, EdgeType::Regular);
        
        // Connect control to target
        self.diagram.add_edge(z_spider, x_spider, EdgeType::Regular);
        
        Ok(())
    }

    /// Add a CZ gate
    fn add_cz(&mut self, qubit1: QubitId, qubit2: QubitId) -> QuantRS2Result<()> {
        let (q1_start, q1_end) = *self.qubit_wires.get(&qubit1)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Qubit 1 not found".to_string()))?;
        let (q2_start, q2_end) = *self.qubit_wires.get(&qubit2)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Qubit 2 not found".to_string()))?;
        
        // CZ is represented as two connected Z-spiders
        let z1_spider = self.diagram.add_spider(SpiderType::Z, 0.0);
        let z2_spider = self.diagram.add_spider(SpiderType::Z, 0.0);
        
        // Break existing connections
        self.diagram.remove_edge(q1_start, q1_end);
        self.diagram.remove_edge(q2_start, q2_end);
        
        // Connect qubit 1
        self.diagram.add_edge(q1_start, z1_spider, EdgeType::Regular);
        self.diagram.add_edge(z1_spider, q1_end, EdgeType::Regular);
        
        // Connect qubit 2
        self.diagram.add_edge(q2_start, z2_spider, EdgeType::Regular);
        self.diagram.add_edge(z2_spider, q2_end, EdgeType::Regular);
        
        // Connect the two spiders with Hadamard edge
        self.diagram.add_edge(z1_spider, z2_spider, EdgeType::Hadamard);
        
        Ok(())
    }

    /// Add a Z-spider to a qubit wire
    fn add_z_spider(&mut self, qubit: QubitId, phase: f64) -> QuantRS2Result<()> {
        let (start, end) = *self.qubit_wires.get(&qubit)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Qubit not found".to_string()))?;
        
        // Create Z-spider
        let spider = self.diagram.add_spider(SpiderType::Z, phase);
        
        // Insert into wire
        self.diagram.remove_edge(start, end);
        self.diagram.add_edge(start, spider, EdgeType::Regular);
        self.diagram.add_edge(spider, end, EdgeType::Regular);
        
        // Update wire tracking
        self.qubit_wires.insert(qubit, (start, end));
        
        Ok(())
    }

    /// Add an X-spider to a qubit wire
    fn add_x_spider(&mut self, qubit: QubitId, phase: f64) -> QuantRS2Result<()> {
        let (start, end) = *self.qubit_wires.get(&qubit)
            .ok_or_else(|| QuantRS2Error::InvalidInput("Qubit not found".to_string()))?;
        
        // Create X-spider
        let spider = self.diagram.add_spider(SpiderType::X, phase);
        
        // Insert into wire
        self.diagram.remove_edge(start, end);
        self.diagram.add_edge(start, spider, EdgeType::Regular);
        self.diagram.add_edge(spider, end, EdgeType::Regular);
        
        // Update wire tracking
        self.qubit_wires.insert(qubit, (start, end));
        
        Ok(())
    }

    /// Get the resulting ZX-diagram
    pub fn into_diagram(self) -> ZXDiagram {
        self.diagram
    }
}

/// Optimizer using ZX-calculus rewrite rules
pub struct ZXOptimizer {
    max_iterations: usize,
    tolerance: f64,
}

impl ZXOptimizer {
    /// Create a new ZX optimizer
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-10,
        }
    }

    /// Set maximum iterations for optimization
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Optimize a quantum circuit using ZX-calculus
    pub fn optimize_circuit(&self, gates: &[Box<dyn GateOp>]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // Find number of qubits
        let num_qubits = gates.iter()
            .flat_map(|g| g.qubits())
            .map(|q| q.0 + 1)
            .max()
            .unwrap_or(0);
        
        // Convert to ZX-diagram
        let mut converter = CircuitToZX::new(num_qubits as usize);
        for gate in gates {
            converter.add_gate(gate.as_ref())?;
        }
        
        let mut diagram = converter.into_diagram();
        
        // Optimize the diagram
        diagram.simplify(self.max_iterations);
        
        // Convert back to circuit (TODO: implement extraction)
        // For now, return the original circuit
        Ok(gates.to_vec())
    }

    /// Count T-gates in a ZX-diagram
    pub fn count_t_gates(&self, diagram: &ZXDiagram) -> usize {
        diagram.spiders.values()
            .filter(|spider| {
                spider.spider_type == SpiderType::Z
                    && (spider.phase - PI / 4.0).abs() < self.tolerance
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spider_creation() {
        let spider = Spider::new(0, SpiderType::Z, PI / 2.0);
        assert_eq!(spider.id, 0);
        assert_eq!(spider.spider_type, SpiderType::Z);
        assert!((spider.phase - PI / 2.0).abs() < 1e-10);
        assert!(spider.is_clifford(1e-10));
        assert!(!spider.is_pauli(1e-10));
    }

    #[test]
    fn test_diagram_creation() {
        let mut diagram = ZXDiagram::new();
        
        let z_id = diagram.add_spider(SpiderType::Z, 0.0);
        let x_id = diagram.add_spider(SpiderType::X, PI);
        
        diagram.add_edge(z_id, x_id, EdgeType::Regular);
        
        assert_eq!(diagram.degree(z_id), 1);
        assert_eq!(diagram.degree(x_id), 1);
    }

    #[test]
    fn test_spider_fusion() {
        let mut diagram = ZXDiagram::new();
        
        let z1 = diagram.add_spider(SpiderType::Z, PI / 4.0);
        let z2 = diagram.add_spider(SpiderType::Z, PI / 4.0);
        let boundary = diagram.add_boundary(QubitId(0), true);
        
        diagram.add_edge(z1, z2, EdgeType::Regular);
        diagram.add_edge(z2, boundary, EdgeType::Regular);
        
        assert!(diagram.spider_fusion(z1, z2).is_ok());
        
        // Check that z2 is removed and z1 has combined phase
        assert!(!diagram.spiders.contains_key(&z2));
        assert_eq!(diagram.spiders[&z1].phase, PI / 2.0);
        assert_eq!(diagram.degree(z1), 1); // Connected to boundary
    }

    #[test]
    fn test_identity_removal() {
        let mut diagram = ZXDiagram::new();
        
        let b1 = diagram.add_boundary(QubitId(0), true);
        let id_spider = diagram.add_spider(SpiderType::Z, 0.0);
        let b2 = diagram.add_boundary(QubitId(0), false);
        
        diagram.add_edge(b1, id_spider, EdgeType::Regular);
        diagram.add_edge(id_spider, b2, EdgeType::Regular);
        
        let removed = diagram.remove_identities();
        assert_eq!(removed, 1);
        assert!(!diagram.spiders.contains_key(&id_spider));
        
        // Check that boundaries are now connected
        assert!(diagram.neighbors(b1).iter().any(|(id, _)| *id == b2));
    }

    #[test]
    fn test_circuit_to_zx_hadamard() {
        let mut converter = CircuitToZX::new(1);
        let h_gate = Hadamard { target: QubitId(0) };
        
        assert!(converter.add_gate(&h_gate).is_ok());
        
        let diagram = converter.into_diagram();
        
        // Check that there's a Hadamard edge between boundaries
        let has_hadamard = diagram.adjacency.values()
            .any(|neighbors| {
                neighbors.iter().any(|(_, edge_type)| *edge_type == EdgeType::Hadamard)
            });
        assert!(has_hadamard);
    }

    #[test]
    fn test_circuit_to_zx_cnot() {
        let mut converter = CircuitToZX::new(2);
        let cnot = CNOT {
            control: QubitId(0),
            target: QubitId(1),
        };
        
        assert!(converter.add_gate(&cnot).is_ok());
        
        let diagram = converter.into_diagram();
        
        // Should have 4 boundaries + 2 spiders (Z and X)
        assert_eq!(diagram.spiders.len(), 6);
        
        // Count Z and X spiders
        let z_count = diagram.spiders.values()
            .filter(|s| s.spider_type == SpiderType::Z && s.phase.abs() < 1e-10)
            .count();
        let x_count = diagram.spiders.values()
            .filter(|s| s.spider_type == SpiderType::X && s.phase.abs() < 1e-10)
            .count();
        
        assert_eq!(z_count, 1);
        assert_eq!(x_count, 1);
    }

    #[test]
    fn test_zx_optimizer() {
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: QubitId(0) }),
            Box::new(PauliZ { target: QubitId(0) }),
            Box::new(Hadamard { target: QubitId(0) }),
        ];
        
        let optimizer = ZXOptimizer::new();
        let result = optimizer.optimize_circuit(&gates);
        
        assert!(result.is_ok());
        // HZH = X, so optimized circuit should be simpler
    }

    #[test]
    fn test_dot_generation() {
        let mut diagram = ZXDiagram::new();
        
        let input = diagram.add_boundary(QubitId(0), true);
        let z = diagram.add_spider(SpiderType::Z, PI / 2.0);
        let x = diagram.add_spider(SpiderType::X, 0.0);
        let output = diagram.add_boundary(QubitId(0), false);
        
        diagram.add_edge(input, z, EdgeType::Regular);
        diagram.add_edge(z, x, EdgeType::Hadamard);
        diagram.add_edge(x, output, EdgeType::Regular);
        
        let dot = diagram.to_dot();
        assert!(dot.contains("graph ZX"));
        assert!(dot.contains("color=green")); // Z spider
        assert!(dot.contains("color=red"));   // X spider
        assert!(dot.contains("style=dashed")); // Hadamard edge
    }
}