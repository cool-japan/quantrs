//! SABRE (SWAP-based BidiREctional) routing algorithm
//!
//! Based on the paper "Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices"
//! by Gushu Li et al. This implementation provides efficient routing for quantum circuits
//! on limited connectivity devices.

use crate::builder::Circuit;
use crate::dag::{circuit_to_dag, CircuitDag, DagNode};
use crate::routing::{CouplingMap, RoutedCircuit, RoutingResult};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{multi::SWAP, GateOp},
    qubit::QubitId,
};
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::collections::{HashMap, HashSet, VecDeque};

/// Configuration for the SABRE router
#[derive(Debug, Clone)]
pub struct SabreConfig {
    /// Maximum number of iterations for the routing process
    pub max_iterations: usize,
    /// Number of lookahead layers to consider
    pub lookahead_depth: usize,
    /// Decay factor for distance calculation
    pub decay_factor: f64,
    /// Weight for extended set calculation
    pub extended_set_weight: f64,
    /// Maximum number of SWAP insertions per iteration
    pub max_swaps_per_iteration: usize,
    /// Enable stochastic tie-breaking
    pub stochastic: bool,
}

impl Default for SabreConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            lookahead_depth: 20,
            decay_factor: 0.001,
            extended_set_weight: 0.5,
            max_swaps_per_iteration: 10,
            stochastic: false,
        }
    }
}

impl SabreConfig {
    /// Create a basic configuration with minimal overhead
    pub fn basic() -> Self {
        Self {
            max_iterations: 100,
            lookahead_depth: 5,
            decay_factor: 0.01,
            extended_set_weight: 0.3,
            max_swaps_per_iteration: 5,
            stochastic: false,
        }
    }

    /// Create a stochastic configuration for multiple trials
    pub fn stochastic() -> Self {
        Self {
            stochastic: true,
            ..Default::default()
        }
    }
}

/// SABRE routing algorithm implementation
pub struct SabreRouter {
    coupling_map: CouplingMap,
    config: SabreConfig,
}

impl SabreRouter {
    /// Create a new SABRE router
    pub fn new(coupling_map: CouplingMap, config: SabreConfig) -> Self {
        Self {
            coupling_map,
            config,
        }
    }

    /// Route a circuit using the SABRE algorithm
    pub fn route<const N: usize>(&self, circuit: &Circuit<N>) -> QuantRS2Result<RoutedCircuit<N>> {
        let dag = circuit_to_dag(circuit);
        let mut logical_to_physical = self.initial_mapping(&dag);
        let mut physical_to_logical: HashMap<usize, usize> = logical_to_physical
            .iter()
            .map(|(&logical, &physical)| (physical, logical))
            .collect();

        let mut routed_gates = Vec::new();
        let mut executable = self.find_executable_gates(&dag, &logical_to_physical);
        let mut remaining_gates: HashSet<usize> = (0..dag.nodes().len()).collect();
        let mut iteration = 0;

        while !remaining_gates.is_empty() && iteration < self.config.max_iterations {
            iteration += 1;

            // Execute all possible gates
            while let Some(gate_id) = executable.pop() {
                if remaining_gates.contains(&gate_id) {
                    let node = &dag.nodes()[gate_id];
                    let routed_gate = self.map_gate_to_physical(node, &logical_to_physical)?;
                    routed_gates.push(routed_gate);
                    remaining_gates.remove(&gate_id);

                    // Update executable gates
                    for &succ in &node.successors {
                        if remaining_gates.contains(&succ)
                            && self.is_gate_executable(&dag.nodes()[succ], &logical_to_physical)
                        {
                            executable.push(succ);
                        }
                    }
                }
            }

            // If no more gates can be executed, insert SWAPs
            if !remaining_gates.is_empty() {
                let swaps = self.find_best_swaps(&dag, &remaining_gates, &logical_to_physical)?;

                if swaps.is_empty() {
                    return Err(QuantRS2Error::RoutingError(
                        "Cannot find valid SWAP operations".to_string(),
                    ));
                }

                // Apply SWAPs
                for (p1, p2) in swaps {
                    // Add SWAP gate to routed circuit
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
                }

                // Update executable gates after SWAP
                executable = self.find_executable_gates_from_remaining(
                    &dag,
                    &remaining_gates,
                    &logical_to_physical,
                );
            }
        }

        if !remaining_gates.is_empty() {
            return Err(QuantRS2Error::RoutingError(format!(
                "Routing failed: {} gates remaining after {} iterations",
                remaining_gates.len(),
                iteration
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

    /// Create initial mapping using a simple heuristic
    fn initial_mapping(&self, dag: &CircuitDag) -> HashMap<usize, usize> {
        let mut mapping = HashMap::new();
        let logical_qubits = self.extract_logical_qubits(dag);

        // Simple strategy: map to the first available physical qubits
        for (i, &logical) in logical_qubits.iter().enumerate() {
            if i < self.coupling_map.num_qubits() {
                mapping.insert(logical, i);
            }
        }

        mapping
    }

    /// Extract logical qubits from the DAG
    fn extract_logical_qubits(&self, dag: &CircuitDag) -> Vec<usize> {
        let mut qubits = HashSet::new();

        for node in dag.nodes() {
            for qubit in node.gate.qubits() {
                qubits.insert(qubit.id() as usize);
            }
        }

        let mut qubit_vec: Vec<usize> = qubits.into_iter().collect();
        qubit_vec.sort();
        qubit_vec
    }

    /// Find gates that can be executed with current mapping
    fn find_executable_gates(
        &self,
        dag: &CircuitDag,
        mapping: &HashMap<usize, usize>,
    ) -> Vec<usize> {
        let mut executable = Vec::new();

        for node in dag.nodes() {
            if node.predecessors.is_empty() && self.is_gate_executable(node, mapping) {
                executable.push(node.id);
            }
        }

        executable
    }

    /// Find executable gates from remaining set
    fn find_executable_gates_from_remaining(
        &self,
        dag: &CircuitDag,
        remaining: &HashSet<usize>,
        mapping: &HashMap<usize, usize>,
    ) -> Vec<usize> {
        let mut executable = Vec::new();

        for &gate_id in remaining {
            let node = &dag.nodes()[gate_id];

            // Check if all predecessors are executed
            let ready = node
                .predecessors
                .iter()
                .all(|&pred| !remaining.contains(&pred));

            if ready && self.is_gate_executable(node, mapping) {
                executable.push(gate_id);
            }
        }

        executable
    }

    /// Check if a gate can be executed with current mapping
    fn is_gate_executable(&self, node: &DagNode, mapping: &HashMap<usize, usize>) -> bool {
        let qubits = node.gate.qubits();

        if qubits.len() <= 1 {
            return true; // Single-qubit gates are always executable
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

    /// Map a logical gate to physical qubits
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
                    "Logical qubit {} not mapped to physical qubit",
                    logical
                )));
            }
        }

        // Clone the gate with new physical qubits
        // This is a simplified implementation - in practice, we'd need to handle each gate type
        self.clone_gate_with_qubits(node.gate.as_ref(), &physical_qubits)
    }

    /// Clone a gate with new qubits (simplified implementation)
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

    /// Find the best SWAP operations to enable more gates
    fn find_best_swaps(
        &self,
        dag: &CircuitDag,
        remaining_gates: &HashSet<usize>,
        mapping: &HashMap<usize, usize>,
    ) -> QuantRS2Result<Vec<(usize, usize)>> {
        let front_layer = self.get_front_layer(dag, remaining_gates);
        let extended_set = self.get_extended_set(dag, &front_layer);

        let mut swap_scores = HashMap::new();

        // Score all possible SWAPs
        for &p1 in &self.get_mapped_physical_qubits(mapping) {
            for &p2 in self.coupling_map.neighbors(p1) {
                if p1 < p2 {
                    // Avoid duplicate pairs
                    let score =
                        self.calculate_swap_score((p1, p2), &front_layer, &extended_set, mapping);
                    swap_scores.insert((p1, p2), score);
                }
            }
        }

        if swap_scores.is_empty() {
            return Ok(Vec::new());
        }

        // Select best SWAP(s)
        let mut sorted_swaps: Vec<_> = swap_scores.into_iter().collect();

        if self.config.stochastic {
            // Stochastic selection from top candidates
            let mut rng = thread_rng();
            sorted_swaps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_candidates = sorted_swaps.len().min(5);

            if top_candidates > 0 {
                let idx = rng.gen_range(0..top_candidates);
                Ok(vec![sorted_swaps[idx].0])
            } else {
                Ok(Vec::new())
            }
        } else {
            // Deterministic selection of best SWAP
            sorted_swaps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            if !sorted_swaps.is_empty() {
                Ok(vec![sorted_swaps[0].0])
            } else {
                Ok(Vec::new())
            }
        }
    }

    /// Get the front layer of executable gates
    fn get_front_layer(&self, dag: &CircuitDag, remaining: &HashSet<usize>) -> HashSet<usize> {
        let mut front_layer = HashSet::new();

        for &gate_id in remaining {
            let node = &dag.nodes()[gate_id];

            // Check if all predecessors are executed
            let ready = node
                .predecessors
                .iter()
                .all(|&pred| !remaining.contains(&pred));

            if ready {
                front_layer.insert(gate_id);
            }
        }

        front_layer
    }

    /// Get extended set for lookahead
    fn get_extended_set(&self, dag: &CircuitDag, front_layer: &HashSet<usize>) -> HashSet<usize> {
        let mut extended_set = front_layer.clone();
        let mut to_visit = VecDeque::new();

        for &gate_id in front_layer {
            to_visit.push_back((gate_id, 0));
        }

        while let Some((gate_id, depth)) = to_visit.pop_front() {
            if depth >= self.config.lookahead_depth {
                continue;
            }

            let node = &dag.nodes()[gate_id];
            for &succ in &node.successors {
                if !extended_set.contains(&succ) {
                    extended_set.insert(succ);
                    to_visit.push_back((succ, depth + 1));
                }
            }
        }

        extended_set
    }

    /// Get currently mapped physical qubits
    fn get_mapped_physical_qubits(&self, mapping: &HashMap<usize, usize>) -> Vec<usize> {
        mapping.values().cloned().collect()
    }

    /// Calculate score for a SWAP operation
    fn calculate_swap_score(
        &self,
        swap: (usize, usize),
        front_layer: &HashSet<usize>,
        extended_set: &HashSet<usize>,
        mapping: &HashMap<usize, usize>,
    ) -> f64 {
        // Create temporary mapping with the SWAP applied
        let mut temp_mapping = mapping.clone();
        let (p1, p2) = swap;

        // Find logical qubits mapped to these physical qubits
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
            return -1.0; // Invalid SWAP
        }

        // Count newly executable gates in front layer
        let mut score = 0.0;

        // TODO: Implement proper gate execution checking with temp_mapping
        // This is a simplified version

        score
    }

    /// Calculate circuit depth
    fn calculate_depth(&self, gates: &[Box<dyn GateOp>]) -> usize {
        // Simplified depth calculation
        // In practice, would need to track dependencies properly
        gates.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::{multi::CNOT, single::Hadamard};

    #[test]
    fn test_sabre_basic() {
        let coupling_map = CouplingMap::linear(3);
        let config = SabreConfig::basic();
        let router = SabreRouter::new(coupling_map, config);

        let mut circuit = Circuit::<3>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(2),
            })
            .unwrap();

        let result = router.route(&circuit);
        assert!(result.is_ok());
    }

    #[test]
    fn test_initial_mapping() {
        let coupling_map = CouplingMap::linear(5);
        let config = SabreConfig::default();
        let router = SabreRouter::new(coupling_map, config);

        let mut circuit = Circuit::<3>::new();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();

        let dag = circuit_to_dag(&circuit);
        let mapping = router.initial_mapping(&dag);

        assert!(mapping.contains_key(&0));
        assert!(mapping.contains_key(&1));
    }
}
