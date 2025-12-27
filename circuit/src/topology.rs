//! Enhanced topological sorting and dependency analysis for quantum circuits.
//!
//! This module provides advanced topological analysis capabilities including
//! multiple sorting strategies, dependency chains, and critical path analysis.
use crate::builder::Circuit;
use crate::commutation::CommutationAnalyzer;
use crate::dag::{circuit_to_dag, CircuitDag, DagNode};
use quantrs2_core::{gate::GateOp, qubit::QubitId};
use std::collections::{HashMap, HashSet, VecDeque};
/// Result of topological analysis
#[derive(Debug, Clone)]
pub struct TopologicalAnalysis {
    /// Standard topological order
    pub topological_order: Vec<usize>,
    /// Reverse topological order
    pub reverse_order: Vec<usize>,
    /// Layers of gates that can be executed in parallel
    pub parallel_layers: Vec<Vec<usize>>,
    /// Critical path through the circuit
    pub critical_path: Vec<usize>,
    /// Gate priorities based on criticality
    pub gate_priorities: HashMap<usize, f64>,
    /// Dependency chains for each qubit
    pub qubit_chains: HashMap<u32, Vec<usize>>,
    /// Circuit depth
    pub depth: usize,
    /// Circuit width (max parallel gates)
    pub width: usize,
}
/// Strategy for topological sorting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologicalStrategy {
    /// Standard Kahn's algorithm
    Standard,
    /// Prioritize gates on critical path
    CriticalPath,
    /// Minimize circuit depth
    MinDepth,
    /// Maximize parallelism
    MaxParallel,
    /// Prioritize by gate type
    GateTypePriority,
    /// Custom priority function
    Custom,
}
/// Advanced topological analyzer
pub struct TopologicalAnalyzer {
    /// Commutation analyzer for optimization
    commutation_analyzer: CommutationAnalyzer,
}
impl TopologicalAnalyzer {
    /// Create a new topological analyzer
    #[must_use]
    pub fn new() -> Self {
        Self {
            commutation_analyzer: CommutationAnalyzer::new(),
        }
    }
    /// Perform comprehensive topological analysis
    #[must_use]
    pub fn analyze<const N: usize>(&self, circuit: &Circuit<N>) -> TopologicalAnalysis {
        let dag = circuit_to_dag(circuit);
        let topological_order =
            Self::topological_sort_with_priorities(&dag, TopologicalStrategy::Standard);
        let reverse_order = Self::reverse_topological_sort(&dag);
        let parallel_layers = self.find_parallel_layers(&dag);
        let critical_path = dag.critical_path();
        let gate_priorities = Self::calculate_gate_priorities(&dag, &critical_path);
        let qubit_chains = Self::find_qubit_chains(&dag);
        let depth = dag.max_depth() + 1;
        let width = parallel_layers
            .iter()
            .map(std::vec::Vec::len)
            .max()
            .unwrap_or(0);
        TopologicalAnalysis {
            topological_order,
            reverse_order,
            parallel_layers,
            critical_path,
            gate_priorities,
            qubit_chains,
            depth,
            width,
        }
    }
    /// Perform topological sort with specific strategy
    #[must_use]
    pub fn sort_with_strategy<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        strategy: TopologicalStrategy,
    ) -> Vec<usize> {
        let dag = circuit_to_dag(circuit);
        Self::topological_sort_with_priorities(&dag, strategy)
    }
    /// Topological sort with priority-based tie breaking
    fn topological_sort_with_priorities(
        dag: &CircuitDag,
        strategy: TopologicalStrategy,
    ) -> Vec<usize> {
        let nodes = dag.nodes();
        let n = nodes.len();
        if n == 0 {
            return Vec::new();
        }
        let mut in_degree = vec![0; n];
        for node in nodes {
            in_degree[node.id] = node.predecessors.len();
        }
        let priority_fn: Box<dyn Fn(usize) -> f64> = match strategy {
            TopologicalStrategy::Standard => Box::new(|id| -(id as f64)),
            TopologicalStrategy::CriticalPath => {
                let critical_set: HashSet<_> = dag.critical_path().into_iter().collect();
                Box::new(move |id| {
                    if critical_set.contains(&id) {
                        1000.0
                    } else {
                        0.0
                    }
                })
            }
            TopologicalStrategy::MinDepth => Box::new(move |id| -(nodes[id].depth as f64)),
            TopologicalStrategy::MaxParallel => Box::new(move |id| {
                let parallel_count = dag.parallel_nodes(id).len();
                parallel_count as f64
            }),
            TopologicalStrategy::GateTypePriority => Box::new(move |id| {
                let gate = &nodes[id].gate;
                match gate.qubits().len() {
                    1 => 100.0,
                    2 => 50.0,
                    _ => 0.0,
                }
            }),
            TopologicalStrategy::Custom => Box::new(|_| 0.0),
        };
        let mut ready_nodes = Vec::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                ready_nodes.push((priority_fn(i), i));
            }
        }
        let mut sorted = Vec::new();
        while !ready_nodes.is_empty() {
            ready_nodes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let (_, node_id) = ready_nodes.remove(0);
            sorted.push(node_id);
            for &succ in &nodes[node_id].successors {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    ready_nodes.push((priority_fn(succ), succ));
                }
            }
        }
        sorted
    }
    /// Reverse topological sort
    fn reverse_topological_sort(dag: &CircuitDag) -> Vec<usize> {
        let nodes = dag.nodes();
        let n = nodes.len();
        if n == 0 {
            return Vec::new();
        }
        let mut out_degree = vec![0; n];
        for node in nodes {
            out_degree[node.id] = node.successors.len();
        }
        let mut queue = VecDeque::new();
        for i in 0..n {
            if out_degree[i] == 0 {
                queue.push_back(i);
            }
        }
        let mut sorted = Vec::new();
        while let Some(node_id) = queue.pop_front() {
            sorted.push(node_id);
            for &pred in &nodes[node_id].predecessors {
                out_degree[pred] -= 1;
                if out_degree[pred] == 0 {
                    queue.push_back(pred);
                }
            }
        }
        sorted.reverse();
        sorted
    }
    /// Find layers of gates that can be executed in parallel
    fn find_parallel_layers(&self, dag: &CircuitDag) -> Vec<Vec<usize>> {
        let max_depth = dag.max_depth();
        let mut layers = Vec::new();
        for depth in 0..=max_depth {
            let layer = dag.nodes_at_depth(depth);
            if !layer.is_empty() {
                layers.push(layer);
            }
        }
        self.optimize_parallel_layers(dag, layers)
    }
    /// Optimize parallel layers using commutation
    fn optimize_parallel_layers(
        &self,
        dag: &CircuitDag,
        mut layers: Vec<Vec<usize>>,
    ) -> Vec<Vec<usize>> {
        let nodes = dag.nodes();
        for i in 0..layers.len() {
            if i + 1 < layers.len() {
                let mut gates_to_move = Vec::new();
                for &gate_id in &layers[i + 1] {
                    let gate = &nodes[gate_id].gate;
                    let can_move = layers[i].iter().all(|&other_id| {
                        let other_gate = &nodes[other_id].gate;
                        self.commutation_analyzer
                            .gates_commute(gate.as_ref(), other_gate.as_ref())
                    });
                    if can_move {
                        gates_to_move.push(gate_id);
                    }
                }
                for gate_id in gates_to_move {
                    layers[i + 1].retain(|&x| x != gate_id);
                    layers[i].push(gate_id);
                }
            }
        }
        layers.retain(|layer| !layer.is_empty());
        layers
    }
    /// Calculate gate priorities based on criticality
    fn calculate_gate_priorities(dag: &CircuitDag, critical_path: &[usize]) -> HashMap<usize, f64> {
        let mut priorities = HashMap::new();
        let nodes = dag.nodes();
        let critical_set: HashSet<_> = critical_path.iter().copied().collect();
        for node in nodes {
            let mut priority = 0.0;
            if critical_set.contains(&node.id) {
                priority += 100.0;
            }
            priority += (nodes.len() - node.depth) as f64;
            priority += node.successors.len() as f64 * 10.0;
            match node.gate.qubits().len() {
                1 => priority += 5.0,
                2 => priority += 3.0,
                _ => priority += 1.0,
            }
            priorities.insert(node.id, priority);
        }
        priorities
    }
    /// Find dependency chains for each qubit
    fn find_qubit_chains(dag: &CircuitDag) -> HashMap<u32, Vec<usize>> {
        let mut chains = HashMap::new();
        let nodes = dag.nodes();
        let mut qubit_nodes: HashMap<u32, Vec<usize>> = HashMap::new();
        for node in nodes {
            for qubit in node.gate.qubits() {
                qubit_nodes.entry(qubit.id()).or_default().push(node.id);
            }
        }
        for (qubit, mut node_ids) in qubit_nodes {
            node_ids.sort_by_key(|&id| nodes[id].depth);
            chains.insert(qubit, node_ids);
        }
        chains
    }
    /// Find the longest dependency chain in the circuit
    #[must_use]
    pub fn find_longest_chain<const N: usize>(&self, circuit: &Circuit<N>) -> Vec<usize> {
        let dag = circuit_to_dag(circuit);
        dag.critical_path()
    }
    /// Find independent gate sets
    #[must_use]
    pub fn find_independent_sets<const N: usize>(&self, circuit: &Circuit<N>) -> Vec<Vec<usize>> {
        let dag = circuit_to_dag(circuit);
        let nodes = dag.nodes();
        let mut independent_sets = Vec::new();
        let mut remaining: HashSet<usize> = (0..nodes.len()).collect();
        while !remaining.is_empty() {
            let mut current_set = Vec::new();
            let mut to_remove = Vec::new();
            for &node_id in &remaining {
                let is_independent = current_set
                    .iter()
                    .all(|&other_id| dag.are_independent(node_id, other_id));
                if is_independent {
                    current_set.push(node_id);
                    to_remove.push(node_id);
                }
            }
            for node_id in to_remove {
                remaining.remove(&node_id);
            }
            if !current_set.is_empty() {
                independent_sets.push(current_set);
            }
        }
        independent_sets
    }
    /// Compute the dependency matrix
    #[must_use]
    pub fn dependency_matrix<const N: usize>(&self, circuit: &Circuit<N>) -> Vec<Vec<bool>> {
        let dag = circuit_to_dag(circuit);
        let n = dag.nodes().len();
        let mut matrix = vec![vec![false; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    matrix[i][j] = !dag.paths_between(j, i).is_empty();
                }
            }
        }
        matrix
    }
}
impl Default for TopologicalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
/// Extension methods for circuits
impl<const N: usize> Circuit<N> {
    /// Perform topological analysis
    #[must_use]
    pub fn topological_analysis(&self) -> TopologicalAnalysis {
        let analyzer = TopologicalAnalyzer::new();
        analyzer.analyze(self)
    }
    /// Get topological order with specific strategy
    #[must_use]
    pub fn topological_sort(&self, strategy: TopologicalStrategy) -> Vec<usize> {
        let analyzer = TopologicalAnalyzer::new();
        analyzer.sort_with_strategy(self, strategy)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::{Hadamard, PauliX};
    #[test]
    fn test_topological_analysis() {
        let mut circuit = Circuit::<3>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to qubit 0");
        circuit
            .add_gate(Hadamard { target: QubitId(1) })
            .expect("Failed to add Hadamard gate to qubit 1");
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add CNOT gate from qubit 0 to qubit 1");
        circuit
            .add_gate(PauliX { target: QubitId(2) })
            .expect("Failed to add PauliX gate to qubit 2");
        circuit
            .add_gate(CNOT {
                control: QubitId(1),
                target: QubitId(2),
            })
            .expect("Failed to add CNOT gate from qubit 1 to qubit 2");
        let analysis = circuit.topological_analysis();
        assert_eq!(analysis.topological_order.len(), 5);
        assert!(analysis.depth > 0);
        assert!(analysis.width > 0);
        assert!(!analysis.critical_path.is_empty());
    }
    #[test]
    fn test_parallel_layers() {
        let mut circuit = Circuit::<4>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate to qubit 0");
        circuit
            .add_gate(Hadamard { target: QubitId(1) })
            .expect("Failed to add Hadamard gate to qubit 1");
        circuit
            .add_gate(Hadamard { target: QubitId(2) })
            .expect("Failed to add Hadamard gate to qubit 2");
        circuit
            .add_gate(Hadamard { target: QubitId(3) })
            .expect("Failed to add Hadamard gate to qubit 3");
        let analyzer = TopologicalAnalyzer::new();
        let analysis = analyzer.analyze(&circuit);
        assert_eq!(analysis.parallel_layers.len(), 1);
        assert_eq!(analysis.parallel_layers[0].len(), 4);
    }
    #[test]
    fn test_qubit_chains() {
        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add first Hadamard gate to qubit 0");
        circuit
            .add_gate(PauliX { target: QubitId(0) })
            .expect("Failed to add PauliX gate to qubit 0");
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add second Hadamard gate to qubit 0");
        let analysis = circuit.topological_analysis();
        assert_eq!(analysis.qubit_chains[&0].len(), 3);
    }
    #[test]
    fn test_sorting_strategies() {
        let mut circuit = Circuit::<3>::new();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .expect("Failed to add first CNOT gate");
        circuit
            .add_gate(Hadamard { target: QubitId(2) })
            .expect("Failed to add Hadamard gate to qubit 2");
        circuit
            .add_gate(CNOT {
                control: QubitId(1),
                target: QubitId(2),
            })
            .expect("Failed to add second CNOT gate");
        let analyzer = TopologicalAnalyzer::new();
        let standard = analyzer.sort_with_strategy(&circuit, TopologicalStrategy::Standard);
        let critical = analyzer.sort_with_strategy(&circuit, TopologicalStrategy::CriticalPath);
        let gate_type =
            analyzer.sort_with_strategy(&circuit, TopologicalStrategy::GateTypePriority);
        assert_eq!(standard.len(), 3);
        assert_eq!(critical.len(), 3);
        assert_eq!(gate_type.len(), 3);
    }
}
