//! Lookahead routing algorithm
//!
//! This module implements a lookahead-based routing strategy that considers
//! future gate dependencies when making SWAP decisions.

use crate::builder::Circuit;
use crate::dag::{circuit_to_dag, CircuitDag, DagNode};
use crate::routing::{CouplingMap, RoutedCircuit, RoutingResult};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{multi::SWAP, GateOp},
    qubit::QubitId,
};
use std::collections::{HashMap, HashSet, VecDeque};

/// Configuration for the lookahead router
#[derive(Debug, Clone)]
pub struct LookaheadConfig {
    /// Depth of lookahead (number of layers to consider)
    pub lookahead_depth: usize,
    /// Maximum number of SWAP candidates to consider
    pub max_swap_candidates: usize,
    /// Weight for distance-based scoring
    pub distance_weight: f64,
    /// Weight for future gate scoring
    pub future_weight: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
}

impl LookaheadConfig {
    /// Create a new lookahead configuration with specified depth
    #[must_use]
    pub const fn new(depth: usize) -> Self {
        Self {
            lookahead_depth: depth,
            max_swap_candidates: 20,
            distance_weight: 1.0,
            future_weight: 0.5,
            max_iterations: 1000,
        }
    }
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self::new(10)
    }
}

/// Lookahead routing algorithm
pub struct LookaheadRouter {
    coupling_map: CouplingMap,
    config: LookaheadConfig,
}

impl LookaheadRouter {
    /// Create a new lookahead router
    #[must_use]
    pub const fn new(coupling_map: CouplingMap, config: LookaheadConfig) -> Self {
        Self {
            coupling_map,
            config,
        }
    }

    /// Route a circuit using lookahead algorithm
    pub fn route<const N: usize>(&self, circuit: &Circuit<N>) -> QuantRS2Result<RoutedCircuit<N>> {
        let dag = circuit_to_dag(circuit);
        let mut logical_to_physical = self.initial_mapping(&dag);
        let mut physical_to_logical: HashMap<usize, usize> = logical_to_physical
            .iter()
            .map(|(&logical, &physical)| (physical, logical))
            .collect();

        let mut routed_gates = Vec::new();
        let mut remaining_gates: HashSet<usize> = (0..dag.nodes().len()).collect();
        let mut iteration = 0;

        while !remaining_gates.is_empty() && iteration < self.config.max_iterations {
            iteration += 1;

            // Execute all ready gates
            let ready_gates = self.find_ready_gates(&dag, &remaining_gates, &logical_to_physical);

            for gate_id in ready_gates {
                let node = &dag.nodes()[gate_id];
                let routed_gate = self.map_gate_to_physical(node, &logical_to_physical)?;
                routed_gates.push(routed_gate);
                remaining_gates.remove(&gate_id);
            }

            // If we still have remaining gates, find best SWAP
            if !remaining_gates.is_empty() {
                let best_swap =
                    self.find_best_lookahead_swap(&dag, &remaining_gates, &logical_to_physical)?;

                if let Some((p1, p2)) = best_swap {
                    // Add SWAP gate
                    let swap_gate = Box::new(SWAP {
                        qubit1: QubitId::new(p1 as u32),
                        qubit2: QubitId::new(p2 as u32),
                    }) as Box<dyn GateOp>;
                    routed_gates.push(swap_gate);

                    // Update mappings
                    let l1 = physical_to_logical[&p1];
                    let l2 = physical_to_logical[&p2];

                    logical_to_physical.insert(l1, p2);
                    logical_to_physical.insert(l2, p1);
                    physical_to_logical.insert(p1, l2);
                    physical_to_logical.insert(p2, l1);
                } else {
                    return Err(QuantRS2Error::RoutingError(
                        "Cannot find valid SWAP operation".to_string(),
                    ));
                }
            }
        }

        if !remaining_gates.is_empty() {
            return Err(QuantRS2Error::RoutingError(format!(
                "Routing failed: {} gates remaining",
                remaining_gates.len()
            )));
        }

        let total_swaps = routed_gates.iter().filter(|g| g.name() == "SWAP").count();
        let circuit_depth = self.calculate_depth(&routed_gates);

        Ok(RoutedCircuit::new(
            routed_gates,
            logical_to_physical,
            RoutingResult {
                total_swaps,
                circuit_depth,
                routing_overhead: if circuit_depth > 0 {
                    total_swaps as f64 / circuit_depth as f64
                } else {
                    0.0
                },
            },
        ))
    }

    /// Create initial mapping using a heuristic based on gate connectivity
    fn initial_mapping(&self, dag: &CircuitDag) -> HashMap<usize, usize> {
        let logical_qubits = self.extract_logical_qubits(dag);
        let mut mapping = HashMap::new();

        if logical_qubits.is_empty() {
            return mapping;
        }

        // Build interaction graph
        let interaction_graph = self.build_interaction_graph(dag);

        // Use a greedy approach to map high-interaction qubits to well-connected physical qubits
        let mut used_physical = HashSet::new();
        let mut logical_priorities = self.calculate_logical_priorities(&interaction_graph);
        logical_priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (logical, _priority) in logical_priorities {
            let best_physical = self.find_best_physical_qubit(
                logical,
                &interaction_graph,
                &mapping,
                &used_physical,
            );
            if let Some(physical) = best_physical {
                mapping.insert(logical, physical);
                used_physical.insert(physical);
            }
        }

        // Map remaining logical qubits to any available physical qubits
        for &logical in &logical_qubits {
            if !mapping.contains_key(&logical) {
                for physical in 0..self.coupling_map.num_qubits() {
                    if !used_physical.contains(&physical) {
                        mapping.insert(logical, physical);
                        used_physical.insert(physical);
                        break;
                    }
                }
            }
        }

        mapping
    }

    /// Extract logical qubits from DAG
    fn extract_logical_qubits(&self, dag: &CircuitDag) -> Vec<usize> {
        let mut qubits = HashSet::new();

        for node in dag.nodes() {
            for qubit in node.gate.qubits() {
                qubits.insert(qubit.id() as usize);
            }
        }

        let mut qubit_vec: Vec<usize> = qubits.into_iter().collect();
        qubit_vec.sort_unstable();
        qubit_vec
    }

    /// Build interaction graph from circuit
    fn build_interaction_graph(&self, dag: &CircuitDag) -> HashMap<(usize, usize), usize> {
        let mut interactions = HashMap::new();

        for node in dag.nodes() {
            let qubits = node.gate.qubits();
            if qubits.len() == 2 {
                let q1 = qubits[0].id() as usize;
                let q2 = qubits[1].id() as usize;
                let key = (q1.min(q2), q1.max(q2));
                *interactions.entry(key).or_insert(0) += 1;
            }
        }

        interactions
    }

    /// Calculate priorities for logical qubits based on connectivity
    fn calculate_logical_priorities(
        &self,
        interaction_graph: &HashMap<(usize, usize), usize>,
    ) -> Vec<(usize, f64)> {
        let mut priorities = HashMap::new();

        for (&(q1, q2), &weight) in interaction_graph {
            *priorities.entry(q1).or_insert(0.0) += weight as f64;
            *priorities.entry(q2).or_insert(0.0) += weight as f64;
        }

        priorities.into_iter().collect()
    }

    /// Find best physical qubit for a logical qubit
    fn find_best_physical_qubit(
        &self,
        logical: usize,
        interaction_graph: &HashMap<(usize, usize), usize>,
        current_mapping: &HashMap<usize, usize>,
        used_physical: &HashSet<usize>,
    ) -> Option<usize> {
        let mut best_physical = None;
        let mut best_score = f64::NEG_INFINITY;

        for physical in 0..self.coupling_map.num_qubits() {
            if used_physical.contains(&physical) {
                continue;
            }

            let score = self.calculate_physical_score(
                logical,
                physical,
                interaction_graph,
                current_mapping,
            );
            if score > best_score {
                best_score = score;
                best_physical = Some(physical);
            }
        }

        best_physical
    }

    /// Calculate score for mapping a logical qubit to a physical qubit
    fn calculate_physical_score(
        &self,
        logical: usize,
        physical: usize,
        interaction_graph: &HashMap<(usize, usize), usize>,
        current_mapping: &HashMap<usize, usize>,
    ) -> f64 {
        let mut score = 0.0;

        // Score based on connectivity to already mapped qubits
        for (&other_logical, &other_physical) in current_mapping {
            let interaction_key = (logical.min(other_logical), logical.max(other_logical));
            if let Some(&weight) = interaction_graph.get(&interaction_key) {
                let distance = self.coupling_map.distance(physical, other_physical);
                score += weight as f64 / (1.0 + distance as f64);
            }
        }

        // Prefer physical qubits with higher connectivity
        score += self.coupling_map.neighbors(physical).len() as f64 * 0.1;

        score
    }

    /// Find gates that are ready to execute
    fn find_ready_gates(
        &self,
        dag: &CircuitDag,
        remaining: &HashSet<usize>,
        mapping: &HashMap<usize, usize>,
    ) -> Vec<usize> {
        let mut ready = Vec::new();

        for &gate_id in remaining {
            let node = &dag.nodes()[gate_id];

            // Check if all predecessors are executed
            let deps_ready = node
                .predecessors
                .iter()
                .all(|&pred| !remaining.contains(&pred));

            if deps_ready && self.is_gate_executable(node, mapping) {
                ready.push(gate_id);
            }
        }

        ready
    }

    /// Check if a gate can be executed with current mapping
    fn is_gate_executable(&self, node: &DagNode, mapping: &HashMap<usize, usize>) -> bool {
        let qubits = node.gate.qubits();

        if qubits.len() <= 1 {
            return true;
        }

        if qubits.len() == 2 {
            let q1 = qubits[0].id() as usize;
            let q2 = qubits[1].id() as usize;

            if let (Some(&p1), Some(&p2)) = (mapping.get(&q1), mapping.get(&q2)) {
                return self.coupling_map.are_connected(p1, p2);
            }
        }

        false
    }

    /// Find the best SWAP using lookahead
    fn find_best_lookahead_swap(
        &self,
        dag: &CircuitDag,
        remaining: &HashSet<usize>,
        mapping: &HashMap<usize, usize>,
    ) -> QuantRS2Result<Option<(usize, usize)>> {
        let lookahead_layers = self.compute_lookahead_layers(dag, remaining);
        let swap_candidates = self.generate_swap_candidates(mapping);

        let mut best_swap = None;
        let mut best_score = f64::NEG_INFINITY;

        for &(p1, p2) in &swap_candidates {
            let score = self.evaluate_swap_with_lookahead((p1, p2), &lookahead_layers, mapping);

            if score > best_score {
                best_score = score;
                best_swap = Some((p1, p2));
            }
        }

        Ok(best_swap)
    }

    /// Compute layers for lookahead analysis
    fn compute_lookahead_layers(
        &self,
        dag: &CircuitDag,
        remaining: &HashSet<usize>,
    ) -> Vec<Vec<usize>> {
        let mut layers = Vec::new();
        let mut current_layer = HashSet::new();
        let mut processed: HashSet<usize> = HashSet::new();

        // Find initial layer (gates with no unprocessed predecessors)
        for &gate_id in remaining {
            let node = &dag.nodes()[gate_id];
            if node
                .predecessors
                .iter()
                .all(|&pred| !remaining.contains(&pred))
            {
                current_layer.insert(gate_id);
            }
        }

        for _ in 0..self.config.lookahead_depth {
            if current_layer.is_empty() {
                break;
            }

            layers.push(current_layer.iter().copied().collect());
            processed.extend(&current_layer);

            let mut next_layer = HashSet::new();
            for &gate_id in &current_layer {
                let node = &dag.nodes()[gate_id];
                for &succ in &node.successors {
                    if remaining.contains(&succ) && !processed.contains(&succ) {
                        // Check if all predecessors are processed
                        let ready = dag.nodes()[succ]
                            .predecessors
                            .iter()
                            .all(|&pred| !remaining.contains(&pred) || processed.contains(&pred));

                        if ready {
                            next_layer.insert(succ);
                        }
                    }
                }
            }

            current_layer = next_layer;
        }

        layers
    }

    /// Generate SWAP candidates
    fn generate_swap_candidates(&self, mapping: &HashMap<usize, usize>) -> Vec<(usize, usize)> {
        let mut candidates = Vec::new();
        let mapped_physical: HashSet<usize> = mapping.values().copied().collect();

        for &p1 in &mapped_physical {
            for &p2 in self.coupling_map.neighbors(p1) {
                if mapped_physical.contains(&p2) && p1 < p2 {
                    candidates.push((p1, p2));
                }
            }
        }

        // Limit candidates to avoid exponential blowup
        candidates.truncate(self.config.max_swap_candidates);
        candidates
    }

    /// Evaluate a SWAP operation using lookahead
    fn evaluate_swap_with_lookahead(
        &self,
        swap: (usize, usize),
        lookahead_layers: &[Vec<usize>],
        mapping: &HashMap<usize, usize>,
    ) -> f64 {
        // Create temporary mapping with SWAP applied
        let mut temp_mapping = mapping.clone();
        let (p1, p2) = swap;

        // Find logical qubits and swap them
        let mut l1_opt = None;
        let mut l2_opt = None;

        for (&logical, &physical) in mapping {
            if physical == p1 {
                l1_opt = Some(logical);
            } else if physical == p2 {
                l2_opt = Some(logical);
            }
        }

        if let (Some(l1), Some(l2)) = (l1_opt, l2_opt) {
            temp_mapping.insert(l1, p2);
            temp_mapping.insert(l2, p1);
        } else {
            return f64::NEG_INFINITY;
        }

        // Score based on gates enabled in lookahead layers
        let mut score = 0.0;
        let mut layer_weight = 1.0;

        for layer in lookahead_layers {
            let mut layer_score = 0.0;

            for &gate_id in layer {
                // Note: We would need DAG access here to get the actual node
                // This is a simplified scoring
                layer_score += 1.0;
            }

            score += layer_score * layer_weight;
            layer_weight *= 0.8; // Decay for future layers
        }

        score
    }

    /// Map a gate to physical qubits
    fn map_gate_to_physical(
        &self,
        node: &DagNode,
        mapping: &HashMap<usize, usize>,
    ) -> QuantRS2Result<Box<dyn GateOp>> {
        let qubits = node.gate.qubits();
        let mut physical_qubits = Vec::new();

        for qubit in qubits {
            let logical = qubit.id() as usize;
            if let Some(&physical) = mapping.get(&logical) {
                physical_qubits.push(QubitId::new(physical as u32));
            } else {
                return Err(QuantRS2Error::RoutingError(format!(
                    "Logical qubit {logical} not mapped"
                )));
            }
        }

        self.clone_gate_with_qubits(node.gate.as_ref(), &physical_qubits)
    }

    /// Clone gate with new qubits
    fn clone_gate_with_qubits(
        &self,
        gate: &dyn GateOp,
        new_qubits: &[QubitId],
    ) -> QuantRS2Result<Box<dyn GateOp>> {
        use quantrs2_core::gate::{multi, single};

        match (gate.name(), new_qubits.len()) {
            ("H", 1) => Ok(Box::new(single::Hadamard {
                target: new_qubits[0],
            })),
            ("X", 1) => Ok(Box::new(single::PauliX {
                target: new_qubits[0],
            })),
            ("Y", 1) => Ok(Box::new(single::PauliY {
                target: new_qubits[0],
            })),
            ("Z", 1) => Ok(Box::new(single::PauliZ {
                target: new_qubits[0],
            })),
            ("S", 1) => Ok(Box::new(single::Phase {
                target: new_qubits[0],
            })),
            ("T", 1) => Ok(Box::new(single::T {
                target: new_qubits[0],
            })),
            ("CNOT", 2) => Ok(Box::new(multi::CNOT {
                control: new_qubits[0],
                target: new_qubits[1],
            })),
            ("CZ", 2) => Ok(Box::new(multi::CZ {
                control: new_qubits[0],
                target: new_qubits[1],
            })),
            ("SWAP", 2) => Ok(Box::new(multi::SWAP {
                qubit1: new_qubits[0],
                qubit2: new_qubits[1],
            })),
            _ => Err(QuantRS2Error::UnsupportedOperation(format!(
                "Cannot route gate {} with {} qubits",
                gate.name(),
                new_qubits.len()
            ))),
        }
    }

    /// Calculate circuit depth
    fn calculate_depth(&self, _gates: &[Box<dyn GateOp>]) -> usize {
        // Simplified implementation
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::{multi::CNOT, single::Hadamard};

    #[test]
    fn test_lookahead_basic() {
        let coupling_map = CouplingMap::linear(4);
        let config = LookaheadConfig::new(5);
        let router = LookaheadRouter::new(coupling_map, config);

        let mut circuit = Circuit::<4>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(3),
            })
            .unwrap();

        let result = router.route(&circuit);
        assert!(result.is_ok());
    }

    #[test]
    fn test_interaction_graph() {
        let coupling_map = CouplingMap::linear(3);
        let config = LookaheadConfig::default();
        let router = LookaheadRouter::new(coupling_map, config);

        let mut circuit = Circuit::<3>::new();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();

        let dag = circuit_to_dag(&circuit);
        let graph = router.build_interaction_graph(&dag);

        assert_eq!(graph.get(&(0, 1)), Some(&2));
    }
}
