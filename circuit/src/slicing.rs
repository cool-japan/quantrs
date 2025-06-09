//! Circuit slicing for parallel execution.
//!
//! This module provides functionality to slice quantum circuits into
//! smaller subcircuits that can be executed in parallel or distributed
//! across multiple quantum processors.

use std::collections::{HashMap, HashSet, VecDeque};

use quantrs2_core::{gate::GateOp, qubit::QubitId};

use crate::builder::Circuit;
use crate::commutation::CommutationAnalyzer;
use crate::dag::{circuit_to_dag, CircuitDag};

/// A slice of a circuit that can be executed independently
#[derive(Debug, Clone)]
pub struct CircuitSlice {
    /// Unique identifier for this slice
    pub id: usize,
    /// Gates in this slice (indices into original circuit)
    pub gate_indices: Vec<usize>,
    /// Qubits used in this slice
    pub qubits: HashSet<u32>,
    /// Dependencies on other slices (slice IDs)
    pub dependencies: HashSet<usize>,
    /// Slices that depend on this one
    pub dependents: HashSet<usize>,
    /// Depth of this slice in the dependency graph
    pub depth: usize,
}

/// Strategy for slicing circuits
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlicingStrategy {
    /// Slice by maximum number of qubits per slice
    MaxQubits(usize),
    /// Slice by maximum number of gates per slice
    MaxGates(usize),
    /// Slice by circuit depth
    DepthBased(usize),
    /// Slice to minimize communication between slices
    MinCommunication,
    /// Slice for load balancing across processors
    LoadBalanced(usize), // number of processors
    /// Custom slicing based on qubit connectivity
    ConnectivityBased,
}

/// Result of circuit slicing
#[derive(Debug)]
pub struct SlicingResult {
    /// The slices
    pub slices: Vec<CircuitSlice>,
    /// Communication cost between slices (number of qubits)
    pub communication_cost: usize,
    /// Maximum parallel depth
    pub parallel_depth: usize,
    /// Slice scheduling order
    pub schedule: Vec<Vec<usize>>, // Groups of slices that can run in parallel
}

/// Circuit slicer
pub struct CircuitSlicer {
    /// Commutation analyzer for optimization
    commutation_analyzer: CommutationAnalyzer,
}

impl CircuitSlicer {
    /// Create a new circuit slicer
    pub fn new() -> Self {
        Self {
            commutation_analyzer: CommutationAnalyzer::new(),
        }
    }

    /// Slice a circuit according to the given strategy
    pub fn slice_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        strategy: SlicingStrategy,
    ) -> SlicingResult {
        match strategy {
            SlicingStrategy::MaxQubits(max_qubits) => self.slice_by_max_qubits(circuit, max_qubits),
            SlicingStrategy::MaxGates(max_gates) => self.slice_by_max_gates(circuit, max_gates),
            SlicingStrategy::DepthBased(max_depth) => self.slice_by_depth(circuit, max_depth),
            SlicingStrategy::MinCommunication => self.slice_min_communication(circuit),
            SlicingStrategy::LoadBalanced(num_processors) => {
                self.slice_load_balanced(circuit, num_processors)
            }
            SlicingStrategy::ConnectivityBased => self.slice_by_connectivity(circuit),
        }
    }

    /// Slice circuit limiting qubits per slice
    fn slice_by_max_qubits<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        max_qubits: usize,
    ) -> SlicingResult {
        let mut slices = Vec::new();
        let mut current_slice = CircuitSlice {
            id: 0,
            gate_indices: Vec::new(),
            qubits: HashSet::new(),
            dependencies: HashSet::new(),
            dependents: HashSet::new(),
            depth: 0,
        };

        // Track which slice each qubit was last used in
        let mut qubit_last_slice: HashMap<u32, usize> = HashMap::new();

        for (gate_idx, gate) in circuit.gates().iter().enumerate() {
            let gate_qubits: HashSet<u32> = gate.qubits().iter().map(|q| q.id()).collect();

            // Check if adding this gate would exceed qubit limit
            let combined_qubits: HashSet<u32> =
                current_slice.qubits.union(&gate_qubits).cloned().collect();

            if !current_slice.gate_indices.is_empty() && combined_qubits.len() > max_qubits {
                // Need to start a new slice
                let slice_id = slices.len();
                current_slice.id = slice_id;

                // Update dependencies based on qubit usage
                for &qubit in &current_slice.qubits {
                    qubit_last_slice.insert(qubit, slice_id);
                }

                slices.push(current_slice);

                // Start new slice
                current_slice = CircuitSlice {
                    id: slice_id + 1,
                    gate_indices: vec![gate_idx],
                    qubits: gate_qubits.clone(),
                    dependencies: HashSet::new(),
                    dependents: HashSet::new(),
                    depth: 0,
                };

                // Add dependencies from previous slices
                for &qubit in &gate_qubits {
                    if let Some(&prev_slice) = qubit_last_slice.get(&qubit) {
                        current_slice.dependencies.insert(prev_slice);
                        slices[prev_slice].dependents.insert(slice_id + 1);
                    }
                }
            } else {
                // Add gate to current slice
                current_slice.gate_indices.push(gate_idx);
                current_slice.qubits.extend(gate_qubits);
            }
        }

        // Don't forget the last slice
        if !current_slice.gate_indices.is_empty() {
            let slice_id = slices.len();
            current_slice.id = slice_id;
            slices.push(current_slice);
        }

        // Calculate depths and schedule
        self.calculate_depths_and_schedule(slices)
    }

    /// Slice circuit limiting gates per slice
    fn slice_by_max_gates<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        max_gates: usize,
    ) -> SlicingResult {
        let mut slices = Vec::new();
        let gates = circuit.gates();

        // Simple slicing by gate count
        for (chunk_idx, chunk) in gates.chunks(max_gates).enumerate() {
            let mut slice = CircuitSlice {
                id: chunk_idx,
                gate_indices: Vec::new(),
                qubits: HashSet::new(),
                dependencies: HashSet::new(),
                dependents: HashSet::new(),
                depth: 0,
            };

            let base_idx = chunk_idx * max_gates;
            for (local_idx, gate) in chunk.iter().enumerate() {
                slice.gate_indices.push(base_idx + local_idx);
                slice.qubits.extend(gate.qubits().iter().map(|q| q.id()));
            }

            slices.push(slice);
        }

        // Add dependencies based on qubit usage
        self.add_qubit_dependencies(&mut slices, gates);

        // Calculate depths and schedule
        self.calculate_depths_and_schedule(slices)
    }

    /// Slice circuit by depth levels
    fn slice_by_depth<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        max_depth: usize,
    ) -> SlicingResult {
        let dag = circuit_to_dag(circuit);
        let mut slices = Vec::new();

        // Group gates by depth levels
        let max_circuit_depth = dag.max_depth();
        for depth_start in (0..=max_circuit_depth).step_by(max_depth) {
            let depth_end = (depth_start + max_depth).min(max_circuit_depth + 1);

            let mut slice = CircuitSlice {
                id: slices.len(),
                gate_indices: Vec::new(),
                qubits: HashSet::new(),
                dependencies: HashSet::new(),
                dependents: HashSet::new(),
                depth: depth_start / max_depth,
            };

            // Collect all gates in this depth range
            for depth in depth_start..depth_end {
                for &node_id in &dag.nodes_at_depth(depth) {
                    slice.gate_indices.push(node_id);
                    let node = &dag.nodes()[node_id];
                    slice
                        .qubits
                        .extend(node.gate.qubits().iter().map(|q| q.id()));
                }
            }

            if !slice.gate_indices.is_empty() {
                slices.push(slice);
            }
        }

        // Dependencies are implicit in depth-based slicing
        for i in 1..slices.len() {
            slices[i].dependencies.insert(i - 1);
            slices[i - 1].dependents.insert(i);
        }

        self.calculate_depths_and_schedule(slices)
    }

    /// Slice to minimize communication between slices
    fn slice_min_communication<const N: usize>(&self, circuit: &Circuit<N>) -> SlicingResult {
        // Use spectral clustering approach
        let gates = circuit.gates();
        let n_gates = gates.len();

        // Build adjacency matrix based on qubit sharing
        let mut adjacency = vec![vec![0.0; n_gates]; n_gates];

        for i in 0..n_gates {
            for j in i + 1..n_gates {
                let qubits_i: HashSet<u32> = gates[i].qubits().iter().map(|q| q.id()).collect();
                let qubits_j: HashSet<u32> = gates[j].qubits().iter().map(|q| q.id()).collect();

                let shared_qubits = qubits_i.intersection(&qubits_j).count();
                if shared_qubits > 0 {
                    adjacency[i][j] = shared_qubits as f64;
                    adjacency[j][i] = shared_qubits as f64;
                }
            }
        }

        // Simple clustering: greedy approach
        let num_slices = (n_gates as f64).sqrt().ceil() as usize;
        let mut slices = Vec::new();
        let mut assigned = vec![false; n_gates];

        // Create initial clusters
        for slice_id in 0..num_slices {
            let mut slice = CircuitSlice {
                id: slice_id,
                gate_indices: Vec::new(),
                qubits: HashSet::new(),
                dependencies: HashSet::new(),
                dependents: HashSet::new(),
                depth: 0,
            };

            // Find unassigned gate with highest connectivity to slice
            for gate_idx in 0..n_gates {
                if !assigned[gate_idx] {
                    // Compute affinity to current slice
                    let affinity = slice
                        .gate_indices
                        .iter()
                        .map(|&idx| adjacency[gate_idx][idx])
                        .sum::<f64>();

                    // Add to slice if first gate or has affinity
                    if slice.gate_indices.is_empty() || affinity > 0.0 {
                        slice.gate_indices.push(gate_idx);
                        slice
                            .qubits
                            .extend(gates[gate_idx].qubits().iter().map(|q| q.id()));
                        assigned[gate_idx] = true;

                        // Limit slice size
                        if slice.gate_indices.len() >= n_gates / num_slices {
                            break;
                        }
                    }
                }
            }

            if !slice.gate_indices.is_empty() {
                slices.push(slice);
            }
        }

        // Assign remaining gates
        for gate_idx in 0..n_gates {
            if !assigned[gate_idx] {
                // Add to slice with highest affinity
                let mut best_slice = 0;
                let mut best_affinity = 0.0;

                for (slice_idx, slice) in slices.iter().enumerate() {
                    let affinity = slice
                        .gate_indices
                        .iter()
                        .map(|&idx| adjacency[gate_idx][idx])
                        .sum::<f64>();

                    if affinity > best_affinity {
                        best_affinity = affinity;
                        best_slice = slice_idx;
                    }
                }

                slices[best_slice].gate_indices.push(gate_idx);
                slices[best_slice]
                    .qubits
                    .extend(gates[gate_idx].qubits().iter().map(|q| q.id()));
            }
        }

        // Add dependencies
        self.add_qubit_dependencies(&mut slices, gates);

        self.calculate_depths_and_schedule(slices)
    }

    /// Slice for load balancing across processors
    fn slice_load_balanced<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        num_processors: usize,
    ) -> SlicingResult {
        let gates = circuit.gates();
        let gates_per_processor = (gates.len() + num_processors - 1) / num_processors;

        // Use max gates strategy with balanced load
        self.slice_by_max_gates(circuit, gates_per_processor)
    }

    /// Slice based on qubit connectivity
    fn slice_by_connectivity<const N: usize>(&self, circuit: &Circuit<N>) -> SlicingResult {
        // Group gates by connected components of qubits
        let gates = circuit.gates();
        let mut slices: Vec<CircuitSlice> = Vec::new();
        let mut gate_to_slice: HashMap<usize, usize> = HashMap::new();

        for (gate_idx, gate) in gates.iter().enumerate() {
            let gate_qubits: HashSet<u32> = gate.qubits().iter().map(|q| q.id()).collect();

            // Find slices that share qubits with this gate
            let mut connected_slices: Vec<usize> = Vec::new();
            for (slice_idx, slice) in slices.iter().enumerate() {
                if !slice.qubits.is_disjoint(&gate_qubits) {
                    connected_slices.push(slice_idx);
                }
            }

            if connected_slices.is_empty() {
                // Create new slice
                let slice_id = slices.len();
                let slice = CircuitSlice {
                    id: slice_id,
                    gate_indices: vec![gate_idx],
                    qubits: gate_qubits,
                    dependencies: HashSet::new(),
                    dependents: HashSet::new(),
                    depth: 0,
                };
                slices.push(slice);
                gate_to_slice.insert(gate_idx, slice_id);
            } else if connected_slices.len() == 1 {
                // Add to existing slice
                let slice_idx = connected_slices[0];
                slices[slice_idx].gate_indices.push(gate_idx);
                slices[slice_idx].qubits.extend(gate_qubits);
                gate_to_slice.insert(gate_idx, slice_idx);
            } else {
                // Merge slices
                let main_slice = connected_slices[0];
                slices[main_slice].gate_indices.push(gate_idx);
                slices[main_slice].qubits.extend(gate_qubits);
                gate_to_slice.insert(gate_idx, main_slice);

                // Merge other slices into main
                for &slice_idx in connected_slices[1..].iter().rev() {
                    let slice = slices.remove(slice_idx);
                    let gate_indices = slice.gate_indices.clone();
                    slices[main_slice].gate_indices.extend(slice.gate_indices);
                    slices[main_slice].qubits.extend(slice.qubits);

                    // Update gate mappings
                    for &g_idx in &gate_indices {
                        gate_to_slice.insert(g_idx, main_slice);
                    }
                }
            }
        }

        // Renumber slices
        for (new_id, slice) in slices.iter_mut().enumerate() {
            slice.id = new_id;
        }

        // Add dependencies based on gate order
        self.add_order_dependencies(&mut slices, gates, &gate_to_slice);

        self.calculate_depths_and_schedule(slices)
    }

    /// Add dependencies based on qubit usage
    fn add_qubit_dependencies(&self, slices: &mut [CircuitSlice], gates: &[Box<dyn GateOp>]) {
        let mut qubit_last_slice: HashMap<u32, usize> = HashMap::new();

        for slice in slices.iter_mut() {
            for &gate_idx in &slice.gate_indices {
                let gate_qubits = gates[gate_idx].qubits();

                // Check dependencies
                for qubit in gate_qubits {
                    if let Some(&prev_slice) = qubit_last_slice.get(&qubit.id()) {
                        if prev_slice != slice.id {
                            slice.dependencies.insert(prev_slice);
                        }
                    }
                }
            }

            // Update last slice for qubits
            for &qubit in &slice.qubits {
                qubit_last_slice.insert(qubit, slice.id);
            }
        }

        // Add dependent relationships
        for i in 0..slices.len() {
            let deps: Vec<usize> = slices[i].dependencies.iter().cloned().collect();
            for dep in deps {
                slices[dep].dependents.insert(i);
            }
        }
    }

    /// Add dependencies based on gate ordering
    fn add_order_dependencies(
        &self,
        slices: &mut [CircuitSlice],
        gates: &[Box<dyn GateOp>],
        gate_to_slice: &HashMap<usize, usize>,
    ) {
        for (gate_idx, gate) in gates.iter().enumerate() {
            let slice_idx = gate_to_slice[&gate_idx];
            let gate_qubits: HashSet<u32> = gate.qubits().iter().map(|q| q.id()).collect();

            // Look for earlier gates on same qubits
            for prev_idx in 0..gate_idx {
                let prev_slice = gate_to_slice[&prev_idx];
                if prev_slice != slice_idx {
                    let prev_qubits: HashSet<u32> =
                        gates[prev_idx].qubits().iter().map(|q| q.id()).collect();

                    if !gate_qubits.is_disjoint(&prev_qubits) {
                        slices[slice_idx].dependencies.insert(prev_slice);
                        slices[prev_slice].dependents.insert(slice_idx);
                    }
                }
            }
        }
    }

    /// Calculate slice depths and parallel schedule
    fn calculate_depths_and_schedule(&self, mut slices: Vec<CircuitSlice>) -> SlicingResult {
        // Calculate depths using topological sort
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        for slice in &slices {
            in_degree.insert(slice.id, slice.dependencies.len());
        }

        let mut queue = VecDeque::new();
        let mut schedule = Vec::new();
        let mut depths = HashMap::new();

        // Initialize with slices having no dependencies
        for slice in &slices {
            if slice.dependencies.is_empty() {
                queue.push_back(slice.id);
                depths.insert(slice.id, 0);
            }
        }

        // Process slices level by level
        while !queue.is_empty() {
            let mut current_level = Vec::new();
            let level_size = queue.len();

            for _ in 0..level_size {
                let slice_id = queue.pop_front().unwrap();
                current_level.push(slice_id);

                // Update dependents
                if let Some(slice) = slices.iter().find(|s| s.id == slice_id) {
                    for &dep_id in &slice.dependents {
                        *in_degree.get_mut(&dep_id).unwrap() -= 1;

                        if in_degree[&dep_id] == 0 {
                            queue.push_back(dep_id);
                            depths.insert(dep_id, depths[&slice_id] + 1);
                        }
                    }
                }
            }

            schedule.push(current_level);
        }

        // Update slice depths
        for slice in &mut slices {
            slice.depth = depths.get(&slice.id).copied().unwrap_or(0);
        }

        // Calculate communication cost
        let communication_cost = self.calculate_communication_cost(&slices);

        SlicingResult {
            slices,
            communication_cost,
            parallel_depth: schedule.len(),
            schedule,
        }
    }

    /// Calculate total communication cost between slices
    fn calculate_communication_cost(&self, slices: &[CircuitSlice]) -> usize {
        let mut total_cost = 0;

        for slice in slices {
            for &dep_id in &slice.dependencies {
                if let Some(dep_slice) = slices.iter().find(|s| s.id == dep_id) {
                    // Count shared qubits
                    let shared: HashSet<_> = slice.qubits.intersection(&dep_slice.qubits).collect();
                    total_cost += shared.len();
                }
            }
        }

        total_cost
    }
}

impl Default for CircuitSlicer {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for circuit slicing
impl<const N: usize> Circuit<N> {
    /// Slice this circuit using the given strategy
    pub fn slice(&self, strategy: SlicingStrategy) -> SlicingResult {
        let slicer = CircuitSlicer::new();
        slicer.slice_circuit(self, strategy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::{Hadamard, PauliX};

    #[test]
    fn test_slice_by_max_qubits() {
        let mut circuit = Circuit::<4>::new();

        // Create a circuit that uses all 4 qubits
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit.add_gate(Hadamard { target: QubitId(1) }).unwrap();
        circuit.add_gate(Hadamard { target: QubitId(2) }).unwrap();
        circuit.add_gate(Hadamard { target: QubitId(3) }).unwrap();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();
        circuit
            .add_gate(CNOT {
                control: QubitId(2),
                target: QubitId(3),
            })
            .unwrap();

        let slicer = CircuitSlicer::new();
        let result = slicer.slice_circuit(&circuit, SlicingStrategy::MaxQubits(2));

        // Should create multiple slices
        assert!(result.slices.len() >= 2);

        // Each slice should use at most 2 qubits
        for slice in &result.slices {
            assert!(slice.qubits.len() <= 2);
        }
    }

    #[test]
    fn test_slice_by_max_gates() {
        let mut circuit = Circuit::<3>::new();

        // Add 6 gates
        for i in 0..6 {
            circuit
                .add_gate(Hadamard {
                    target: QubitId((i % 3) as u32),
                })
                .unwrap();
        }

        let slicer = CircuitSlicer::new();
        let result = slicer.slice_circuit(&circuit, SlicingStrategy::MaxGates(2));

        // Should create 3 slices
        assert_eq!(result.slices.len(), 3);

        // Each slice should have at most 2 gates
        for slice in &result.slices {
            assert!(slice.gate_indices.len() <= 2);
        }
    }

    #[test]
    fn test_slice_dependencies() {
        let mut circuit = Circuit::<2>::new();

        // Create dependent gates
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit.add_gate(Hadamard { target: QubitId(1) }).unwrap();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();
        circuit.add_gate(PauliX { target: QubitId(0) }).unwrap();

        let slicer = CircuitSlicer::new();
        let result = slicer.slice_circuit(&circuit, SlicingStrategy::MaxGates(2));

        // Check dependencies exist
        let mut has_dependencies = false;
        for slice in &result.slices {
            if !slice.dependencies.is_empty() {
                has_dependencies = true;
                break;
            }
        }
        assert!(has_dependencies);
    }

    #[test]
    fn test_parallel_schedule() {
        let mut circuit = Circuit::<4>::new();

        // Create gates that can be parallel
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit.add_gate(Hadamard { target: QubitId(1) }).unwrap();
        circuit.add_gate(Hadamard { target: QubitId(2) }).unwrap();
        circuit.add_gate(Hadamard { target: QubitId(3) }).unwrap();

        let slicer = CircuitSlicer::new();
        let result = slicer.slice_circuit(&circuit, SlicingStrategy::MaxQubits(1));

        // All H gates can be executed in parallel
        assert_eq!(result.parallel_depth, 1);
        assert_eq!(result.schedule[0].len(), 4);
    }
}
