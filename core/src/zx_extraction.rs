//! Circuit extraction from ZX-diagrams
//!
//! This module provides algorithms to extract quantum circuits from
//! optimized ZX-diagrams, completing the optimization pipeline.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{multi::*, single::*, GateOp},
    qubit::QubitId,
    zx_calculus::{CircuitToZX, EdgeType, Spider, SpiderType, ZXDiagram, ZXOptimizer},
};
use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;

/// Represents a layer of gates in the extracted circuit
#[derive(Debug, Clone)]
struct GateLayer {
    /// Gates in this layer (can be applied in parallel)
    gates: Vec<Box<dyn GateOp>>,
}

/// Circuit extractor from ZX-diagrams
pub struct ZXExtractor {
    diagram: ZXDiagram,
    /// Maps spider IDs to their positions in the circuit
    spider_positions: FxHashMap<usize, (usize, usize)>, // (layer, position_in_layer)
    /// Layers of the circuit
    layers: Vec<GateLayer>,
}

impl ZXExtractor {
    /// Create a new extractor from a ZX-diagram
    pub fn new(diagram: ZXDiagram) -> Self {
        Self {
            diagram,
            spider_positions: FxHashMap::default(),
            layers: Vec::new(),
        }
    }

    /// Extract a quantum circuit from the ZX-diagram
    pub fn extract_circuit(&mut self) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // First, perform graph analysis to understand the structure
        self.analyze_diagram()?;

        // Extract gates layer by layer
        self.extract_gates()?;

        // Flatten layers into a single circuit
        let mut circuit = Vec::new();
        for layer in &self.layers {
            circuit.extend(layer.gates.clone());
        }

        Ok(circuit)
    }

    /// Analyze the diagram structure
    fn analyze_diagram(&mut self) -> QuantRS2Result<()> {
        // Find the flow from inputs to outputs using BFS
        let inputs = self.diagram.inputs.clone();
        let outputs = self.diagram.outputs.clone();

        // Create a topological ordering of spiders
        let topo_order = self.topological_sort(&inputs, &outputs)?;

        // Assign positions to spiders
        for (layer_idx, spider_id) in topo_order.iter().enumerate() {
            self.spider_positions.insert(*spider_id, (layer_idx, 0));
        }

        Ok(())
    }

    /// Perform topological sort from inputs to outputs
    fn topological_sort(&self, inputs: &[usize], outputs: &[usize]) -> QuantRS2Result<Vec<usize>> {
        let mut in_degree: FxHashMap<usize, usize> = FxHashMap::default();
        let mut adjacency: FxHashMap<usize, Vec<usize>> = FxHashMap::default();

        // Build directed graph (from inputs to outputs)
        for &spider_id in self.diagram.spiders.keys() {
            in_degree.insert(spider_id, 0);
            adjacency.insert(spider_id, Vec::new());
        }

        // Count in-degrees and build adjacency
        for (&source, neighbors) in &self.diagram.adjacency {
            for &(target, _) in neighbors {
                // Only count edges going "forward" (avoid double counting)
                if self.is_forward_edge(source, target, inputs, outputs) {
                    *in_degree.get_mut(&target).unwrap() += 1;
                    adjacency.get_mut(&source).unwrap().push(target);
                }
            }
        }

        // Kahn's algorithm for topological sort
        let mut queue: VecDeque<usize> = inputs.iter().cloned().collect();
        let mut topo_order = Vec::new();

        while let Some(spider) = queue.pop_front() {
            topo_order.push(spider);

            if let Some(neighbors) = adjacency.get(&spider) {
                for &neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(&neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        if topo_order.len() != self.diagram.spiders.len() {
            return Err(QuantRS2Error::InvalidInput(
                "ZX-diagram contains cycles or disconnected components".to_string(),
            ));
        }

        Ok(topo_order)
    }

    /// Determine if an edge goes "forward" in the circuit
    fn is_forward_edge(
        &self,
        source: usize,
        target: usize,
        inputs: &[usize],
        outputs: &[usize],
    ) -> bool {
        // More sophisticated flow analysis
        // Boundary nodes have clear direction
        if inputs.contains(&source) && !inputs.contains(&target) {
            return true;
        }
        if outputs.contains(&target) && !outputs.contains(&source) {
            return true;
        }

        // For non-boundary nodes, check if source is closer to inputs
        let source_dist = self.distance_from_inputs(source, inputs);
        let target_dist = self.distance_from_inputs(target, inputs);

        source_dist < target_dist
    }

    /// Calculate minimum distance from a spider to any input
    fn distance_from_inputs(&self, spider: usize, inputs: &[usize]) -> usize {
        if inputs.contains(&spider) {
            return 0;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        for &input in inputs {
            queue.push_back((input, 0));
            visited.insert(input);
        }

        while let Some((current, dist)) = queue.pop_front() {
            if current == spider {
                return dist;
            }

            for (neighbor, _) in self.diagram.neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }

        usize::MAX // Not reachable from inputs
    }

    /// Extract gates from the analyzed diagram
    fn extract_gates(&mut self) -> QuantRS2Result<()> {
        // Group spiders by qubit
        let qubit_spiders = self.group_by_qubit()?;

        // Process each qubit line
        for (qubit, spider_chain) in qubit_spiders {
            self.extract_single_qubit_gates(&qubit, &spider_chain)?;
        }

        // Extract two-qubit gates
        self.extract_two_qubit_gates()?;

        Ok(())
    }

    /// Group spiders by their associated qubit
    fn group_by_qubit(&self) -> QuantRS2Result<FxHashMap<QubitId, Vec<usize>>> {
        let mut qubit_spiders: FxHashMap<QubitId, Vec<usize>> = FxHashMap::default();

        // Start from input boundaries
        for &input_id in &self.diagram.inputs {
            if let Some(input_spider) = self.diagram.spiders.get(&input_id) {
                if let Some(qubit) = input_spider.qubit {
                    let chain = self.trace_qubit_line(input_id)?;
                    qubit_spiders.insert(qubit, chain);
                }
            }
        }

        Ok(qubit_spiders)
    }

    /// Trace a qubit line from input to output
    fn trace_qubit_line(&self, start: usize) -> QuantRS2Result<Vec<usize>> {
        let mut chain = vec![start];
        let mut current = start;
        let mut visited = HashSet::new();
        visited.insert(start);

        // Follow the qubit line
        while !self.diagram.outputs.contains(&current) {
            let neighbors = self.diagram.neighbors(current);

            // Find the next spider in the chain (not visited, not backwards)
            let next = neighbors
                .iter()
                .find(|(id, _)| !visited.contains(id) && self.is_on_qubit_line(*id))
                .map(|(id, _)| *id);

            if let Some(next_id) = next {
                chain.push(next_id);
                visited.insert(next_id);
                current = next_id;
            } else {
                break;
            }
        }

        Ok(chain)
    }

    /// Check if a spider is on a qubit line (not a control/target connection)
    fn is_on_qubit_line(&self, spider_id: usize) -> bool {
        // Boundary spiders are always on qubit lines
        if let Some(spider) = self.diagram.spiders.get(&spider_id) {
            spider.spider_type == SpiderType::Boundary || self.diagram.degree(spider_id) <= 2
        } else {
            false
        }
    }

    /// Extract single-qubit gates from a chain of spiders
    fn extract_single_qubit_gates(
        &mut self,
        qubit: &QubitId,
        spider_chain: &[usize],
    ) -> QuantRS2Result<()> {
        let mut i = 0;

        while i < spider_chain.len() {
            let spider_id = spider_chain[i];

            if let Some(spider) = self.diagram.spiders.get(&spider_id) {
                match spider.spider_type {
                    SpiderType::Z if spider.phase.abs() > 1e-10 => {
                        // Z rotation
                        let gate: Box<dyn GateOp> = Box::new(RotationZ {
                            target: *qubit,
                            theta: spider.phase,
                        });
                        self.add_gate_to_layer(gate, i);
                    }
                    SpiderType::X if spider.phase.abs() > 1e-10 => {
                        // X rotation
                        let gate: Box<dyn GateOp> = Box::new(RotationX {
                            target: *qubit,
                            theta: spider.phase,
                        });
                        self.add_gate_to_layer(gate, i);
                    }
                    _ => {}
                }

                // Check for Hadamard edges
                if i + 1 < spider_chain.len() {
                    let next_id = spider_chain[i + 1];
                    let edge_type = self.get_edge_type(spider_id, next_id);

                    if edge_type == Some(EdgeType::Hadamard) {
                        let gate: Box<dyn GateOp> = Box::new(Hadamard { target: *qubit });
                        self.add_gate_to_layer(gate, i);
                    }
                }
            }

            i += 1;
        }

        Ok(())
    }

    /// Extract two-qubit gates from spider connections
    fn extract_two_qubit_gates(&mut self) -> QuantRS2Result<()> {
        let mut processed = HashSet::new();

        for (&spider_id, spider) in &self.diagram.spiders.clone() {
            if processed.contains(&spider_id) {
                continue;
            }

            // Look for patterns that represent two-qubit gates
            if self.diagram.degree(spider_id) > 2 {
                // This spider connects multiple qubits
                let neighbors = self.diagram.neighbors(spider_id);

                // Check for CNOT pattern (Z spider connected to X spider)
                if spider.spider_type == SpiderType::Z && spider.phase.abs() < 1e-10 {
                    for &(neighbor_id, edge_type) in &neighbors {
                        if let Some(neighbor) = self.diagram.spiders.get(&neighbor_id) {
                            if neighbor.spider_type == SpiderType::X
                                && neighbor.phase.abs() < 1e-10
                                && edge_type == EdgeType::Regular
                            {
                                // Found CNOT pattern
                                if let (Some(control_qubit), Some(target_qubit)) = (
                                    self.get_spider_qubit(spider_id),
                                    self.get_spider_qubit(neighbor_id),
                                ) {
                                    let gate: Box<dyn GateOp> = Box::new(CNOT {
                                        control: control_qubit,
                                        target: target_qubit,
                                    });

                                    let layer = self
                                        .spider_positions
                                        .get(&spider_id)
                                        .map(|(l, _)| *l)
                                        .unwrap_or(0);

                                    self.add_gate_to_layer(gate, layer);
                                    processed.insert(spider_id);
                                    processed.insert(neighbor_id);
                                }
                            }
                        }
                    }
                }

                // Check for CZ pattern (two Z spiders connected with Hadamard)
                if spider.spider_type == SpiderType::Z && spider.phase.abs() < 1e-10 {
                    for &(neighbor_id, edge_type) in &neighbors {
                        if let Some(neighbor) = self.diagram.spiders.get(&neighbor_id) {
                            if neighbor.spider_type == SpiderType::Z
                                && neighbor.phase.abs() < 1e-10
                                && edge_type == EdgeType::Hadamard
                            {
                                // Found CZ pattern
                                if let (Some(qubit1), Some(qubit2)) = (
                                    self.get_spider_qubit(spider_id),
                                    self.get_spider_qubit(neighbor_id),
                                ) {
                                    let gate: Box<dyn GateOp> = Box::new(CZ {
                                        control: qubit1,
                                        target: qubit2,
                                    });

                                    let layer = self
                                        .spider_positions
                                        .get(&spider_id)
                                        .map(|(l, _)| *l)
                                        .unwrap_or(0);

                                    self.add_gate_to_layer(gate, layer);
                                    processed.insert(spider_id);
                                    processed.insert(neighbor_id);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the qubit associated with a spider
    fn get_spider_qubit(&self, spider_id: usize) -> Option<QubitId> {
        // First check if it's a boundary spider
        if let Some(spider) = self.diagram.spiders.get(&spider_id) {
            if let Some(qubit) = spider.qubit {
                return Some(qubit);
            }
        }

        // Otherwise, trace back to find the input boundary
        self.find_connected_boundary(spider_id)
    }

    /// Find a boundary spider connected to this spider
    fn find_connected_boundary(&self, spider_id: usize) -> Option<QubitId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(spider_id);
        visited.insert(spider_id);

        while let Some(current) = queue.pop_front() {
            if let Some(spider) = self.diagram.spiders.get(&current) {
                if let Some(qubit) = spider.qubit {
                    return Some(qubit);
                }
            }

            for (neighbor, _) in self.diagram.neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }

    /// Get the edge type between two spiders
    fn get_edge_type(&self, spider1: usize, spider2: usize) -> Option<EdgeType> {
        self.diagram
            .neighbors(spider1)
            .iter()
            .find(|(id, _)| *id == spider2)
            .map(|(_, edge_type)| *edge_type)
    }

    /// Add a gate to the appropriate layer
    fn add_gate_to_layer(&mut self, gate: Box<dyn GateOp>, layer_idx: usize) {
        // Ensure we have enough layers
        while self.layers.len() <= layer_idx {
            self.layers.push(GateLayer { gates: Vec::new() });
        }

        self.layers[layer_idx].gates.push(gate);
    }
}

/// Complete ZX-calculus optimization pipeline
pub struct ZXPipeline {
    optimizer: ZXOptimizer,
}

impl ZXPipeline {
    /// Create a new ZX optimization pipeline
    pub fn new() -> Self {
        Self {
            optimizer: ZXOptimizer::new(),
        }
    }

    /// Optimize a circuit using ZX-calculus
    pub fn optimize(&self, gates: &[Box<dyn GateOp>]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // Find number of qubits
        let num_qubits = gates
            .iter()
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
        let rewrites = diagram.simplify(100);
        println!("Applied {} ZX-calculus rewrites", rewrites);

        // Extract optimized circuit
        let mut extractor = ZXExtractor::new(diagram);
        extractor.extract_circuit()
    }

    /// Compare T-count before and after optimization
    pub fn compare_t_count(
        &self,
        original: &[Box<dyn GateOp>],
        optimized: &[Box<dyn GateOp>],
    ) -> (usize, usize) {
        let count_t = |gates: &[Box<dyn GateOp>]| {
            gates
                .iter()
                .filter(|g| {
                    g.name() == "T"
                        || (g.name() == "RZ" && {
                            if let Some(rz) = g.as_any().downcast_ref::<RotationZ>() {
                                (rz.theta - PI / 4.0).abs() < 1e-10
                            } else {
                                false
                            }
                        })
                })
                .count()
        };

        (count_t(original), count_t(optimized))
    }
}

impl Default for ZXPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::single::*;

    #[test]
    fn test_circuit_extraction_identity() {
        // Create a simple diagram with just boundaries
        let mut diagram = ZXDiagram::new();
        let input = diagram.add_boundary(QubitId(0), true);
        let output = diagram.add_boundary(QubitId(0), false);
        diagram.add_edge(input, output, EdgeType::Regular);

        let mut extractor = ZXExtractor::new(diagram);
        let circuit = extractor.extract_circuit().unwrap();

        // Should extract empty circuit (identity)
        assert_eq!(circuit.len(), 0);
    }

    #[test]
    fn test_circuit_extraction_single_gate() {
        // Create diagram with single Z rotation
        let mut diagram = ZXDiagram::new();
        let input = diagram.add_boundary(QubitId(0), true);
        let z_spider = diagram.add_spider(SpiderType::Z, PI / 2.0);
        let output = diagram.add_boundary(QubitId(0), false);

        diagram.add_edge(input, z_spider, EdgeType::Regular);
        diagram.add_edge(z_spider, output, EdgeType::Regular);

        let mut extractor = ZXExtractor::new(diagram);
        let circuit = extractor.extract_circuit().unwrap();

        // Should extract one RZ gate
        assert_eq!(circuit.len(), 1);
        assert_eq!(circuit[0].name(), "RZ");
    }

    #[test]
    fn test_zx_pipeline_optimization() {
        // Create a circuit that can be optimized: HZH = X
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: QubitId(0) }),
            Box::new(PauliZ { target: QubitId(0) }),
            Box::new(Hadamard { target: QubitId(0) }),
        ];

        let pipeline = ZXPipeline::new();
        let optimized = pipeline.optimize(&gates).unwrap();

        // The optimized circuit should be simpler
        assert!(optimized.len() <= gates.len());
    }

    #[test]
    fn test_t_count_reduction() {
        // Create a circuit with T gates
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(RotationZ {
                target: QubitId(0),
                theta: PI / 4.0,
            }), // T gate
            Box::new(RotationZ {
                target: QubitId(0),
                theta: PI / 4.0,
            }), // T gate
        ];

        let pipeline = ZXPipeline::new();
        let optimized = pipeline.optimize(&gates).unwrap();

        let (original_t, optimized_t) = pipeline.compare_t_count(&gates, &optimized);

        // Two T gates should fuse into S gate (or equivalent)
        assert_eq!(original_t, 2);
        assert!(optimized_t <= original_t);
    }
}
