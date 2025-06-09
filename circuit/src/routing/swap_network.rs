//! SWAP network generation and optimization
//!
//! This module provides utilities for generating and optimizing SWAP networks
//! used in quantum circuit routing.

use crate::routing::CouplingMap;
use quantrs2_core::{error::QuantRS2Result, gate::multi::SWAP, qubit::QubitId};
use std::collections::{HashMap, HashSet, VecDeque};

/// A layer of SWAP operations that can be executed in parallel
#[derive(Debug, Clone)]
pub struct SwapLayer {
    /// SWAP operations in this layer
    pub swaps: Vec<(usize, usize)>,
    /// Layer depth in the circuit
    pub depth: usize,
}

impl SwapLayer {
    /// Create a new SWAP layer
    pub fn new(depth: usize) -> Self {
        Self {
            swaps: Vec::new(),
            depth,
        }
    }

    /// Add a SWAP operation to this layer
    pub fn add_swap(&mut self, qubit1: usize, qubit2: usize) {
        self.swaps.push((qubit1, qubit2));
    }

    /// Check if two qubits are involved in any SWAP in this layer
    pub fn involves_qubits(&self, qubit1: usize, qubit2: usize) -> bool {
        self.swaps
            .iter()
            .any(|&(q1, q2)| (q1 == qubit1 || q1 == qubit2) || (q2 == qubit1 || q2 == qubit2))
    }

    /// Get all qubits involved in this layer
    pub fn qubits(&self) -> HashSet<usize> {
        let mut qubits = HashSet::new();
        for &(q1, q2) in &self.swaps {
            qubits.insert(q1);
            qubits.insert(q2);
        }
        qubits
    }

    /// Check if a SWAP can be added without conflicts
    pub fn can_add_swap(&self, qubit1: usize, qubit2: usize) -> bool {
        !self.involves_qubits(qubit1, qubit2)
    }
}

/// SWAP network for routing quantum circuits
#[derive(Debug, Clone)]
pub struct SwapNetwork {
    /// Layers of SWAP operations
    pub layers: Vec<SwapLayer>,
    /// Coupling map for the device
    coupling_map: CouplingMap,
}

impl SwapNetwork {
    /// Create a new SWAP network
    pub fn new(coupling_map: CouplingMap) -> Self {
        Self {
            layers: Vec::new(),
            coupling_map,
        }
    }

    /// Add a new layer
    pub fn add_layer(&mut self, layer: SwapLayer) {
        self.layers.push(layer);
    }

    /// Generate SWAP network to route between two permutations
    pub fn generate_routing_network(
        &mut self,
        initial_mapping: &HashMap<usize, usize>,
        target_mapping: &HashMap<usize, usize>,
    ) -> QuantRS2Result<()> {
        // Convert mappings to permutation arrays for easier manipulation
        let mut current_perm = self.mapping_to_permutation(initial_mapping);
        let target_perm = self.mapping_to_permutation(target_mapping);

        let mut depth = 0;

        while current_perm != target_perm {
            let mut layer = SwapLayer::new(depth);
            let mut swaps_added = false;

            // Find profitable SWAPs for this layer
            for i in 0..current_perm.len() {
                if current_perm[i] != target_perm[i] {
                    // Find where the correct value is located
                    if let Some(j) = current_perm.iter().position(|&x| x == target_perm[i]) {
                        if i != j
                            && self.coupling_map.are_connected(i, j)
                            && layer.can_add_swap(i, j)
                        {
                            layer.add_swap(i, j);
                            current_perm.swap(i, j);
                            swaps_added = true;
                        }
                    }
                }
            }

            // If no direct SWAPs found, use routing SWAPs
            if !swaps_added {
                if let Some((i, j)) = self.find_routing_swap(&current_perm, &target_perm, &layer) {
                    layer.add_swap(i, j);
                    current_perm.swap(i, j);
                    swaps_added = true;
                }
            }

            if swaps_added {
                self.add_layer(layer);
                depth += 1;
            } else {
                break; // No progress possible
            }

            // Safety check to avoid infinite loops
            if depth > current_perm.len() * 2 {
                break;
            }
        }

        Ok(())
    }

    /// Convert mapping to permutation array
    fn mapping_to_permutation(&self, mapping: &HashMap<usize, usize>) -> Vec<usize> {
        let mut perm = vec![0; self.coupling_map.num_qubits()];

        for (&logical, &physical) in mapping {
            if physical < perm.len() {
                perm[physical] = logical;
            }
        }

        perm
    }

    /// Find a routing SWAP that makes progress toward the target
    fn find_routing_swap(
        &self,
        current: &[usize],
        target: &[usize],
        layer: &SwapLayer,
    ) -> Option<(usize, usize)> {
        let mut best_swap = None;
        let mut best_score = -1;

        for i in 0..current.len() {
            for &j in self.coupling_map.neighbors(i) {
                if i < j && layer.can_add_swap(i, j) {
                    let score = self.evaluate_swap_progress(current, target, i, j);
                    if score > best_score {
                        best_score = score;
                        best_swap = Some((i, j));
                    }
                }
            }
        }

        best_swap
    }

    /// Evaluate how much progress a SWAP makes toward the target
    fn evaluate_swap_progress(
        &self,
        current: &[usize],
        target: &[usize],
        i: usize,
        j: usize,
    ) -> i32 {
        let mut score = 0;

        // Check if swapping brings elements closer to their targets
        let target_pos_i = target.iter().position(|&x| x == current[i]);
        let target_pos_j = target.iter().position(|&x| x == current[j]);

        if let Some(target_i) = target_pos_i {
            let dist_before = self.coupling_map.distance(i, target_i);
            let dist_after = self.coupling_map.distance(j, target_i);
            if dist_after < dist_before {
                score += dist_before as i32 - dist_after as i32;
            }
        }

        if let Some(target_j) = target_pos_j {
            let dist_before = self.coupling_map.distance(j, target_j);
            let dist_after = self.coupling_map.distance(i, target_j);
            if dist_after < dist_before {
                score += dist_before as i32 - dist_after as i32;
            }
        }

        score
    }

    /// Optimize the SWAP network by removing redundant operations
    pub fn optimize(&mut self) {
        self.remove_redundant_swaps();
        self.merge_consecutive_layers();
        self.reorder_swaps_for_parallelism();
    }

    /// Remove redundant SWAP operations
    fn remove_redundant_swaps(&mut self) {
        // Track the effect of all SWAPs
        let mut net_swaps: HashMap<(usize, usize), usize> = HashMap::new();

        for layer in &self.layers {
            for &(q1, q2) in &layer.swaps {
                let key = (q1.min(q2), q1.max(q2));
                *net_swaps.entry(key).or_insert(0) += 1;
            }
        }

        // Remove SWAPs that appear an even number of times (cancel out)
        let cancelled_swaps: HashSet<(usize, usize)> = net_swaps
            .iter()
            .filter(|(_, &count)| count % 2 == 0)
            .map(|(&key, _)| key)
            .collect();

        for layer in &mut self.layers {
            layer.swaps.retain(|&(q1, q2)| {
                let key = (q1.min(q2), q1.max(q2));
                !cancelled_swaps.contains(&key)
            });
        }

        // Remove empty layers
        self.layers.retain(|layer| !layer.swaps.is_empty());
    }

    /// Merge consecutive layers if possible
    fn merge_consecutive_layers(&mut self) {
        let mut i = 0;
        while i + 1 < self.layers.len() {
            let can_merge = {
                let layer1_qubits = self.layers[i].qubits();
                let layer2_qubits = self.layers[i + 1].qubits();
                layer1_qubits.is_disjoint(&layer2_qubits)
            };

            if can_merge {
                // Merge layer i+1 into layer i
                let mut layer2 = self.layers.remove(i + 1);
                self.layers[i].swaps.append(&mut layer2.swaps);
            } else {
                i += 1;
            }
        }
    }

    /// Reorder SWAPs within layers for better parallelism
    fn reorder_swaps_for_parallelism(&mut self) {
        for layer in &mut self.layers {
            // Sort SWAPs by some heuristic (e.g., by first qubit index)
            layer.swaps.sort_by_key(|&(q1, q2)| q1.min(q2));
        }
    }

    /// Convert the SWAP network to a sequence of SWAP gates
    pub fn to_swap_gates(&self) -> Vec<SWAP> {
        let mut gates = Vec::new();

        for layer in &self.layers {
            for &(q1, q2) in &layer.swaps {
                gates.push(SWAP {
                    qubit1: QubitId::new(q1 as u32),
                    qubit2: QubitId::new(q2 as u32),
                });
            }
        }

        gates
    }

    /// Get the total number of SWAP operations
    pub fn total_swaps(&self) -> usize {
        self.layers.iter().map(|layer| layer.swaps.len()).sum()
    }

    /// Get the depth of the SWAP network
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Check if the network is empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty() || self.layers.iter().all(|layer| layer.swaps.is_empty())
    }

    /// Generate a minimal SWAP network using bubble sort approach
    pub fn bubble_sort_network(
        &mut self,
        initial_mapping: &HashMap<usize, usize>,
        target_mapping: &HashMap<usize, usize>,
    ) -> QuantRS2Result<()> {
        let mut current_perm = self.mapping_to_permutation(initial_mapping);
        let target_perm = self.mapping_to_permutation(target_mapping);

        let mut depth = 0;
        let mut changed = true;

        while changed && current_perm != target_perm {
            changed = false;
            let mut layer = SwapLayer::new(depth);

            for i in 0..current_perm.len().saturating_sub(1) {
                if current_perm[i] != target_perm[i] {
                    // Look for adjacent swaps that make progress
                    if self.coupling_map.are_connected(i, i + 1) && layer.can_add_swap(i, i + 1) {
                        // Check if swapping makes progress
                        if self.should_swap_for_target(&current_perm, &target_perm, i, i + 1) {
                            layer.add_swap(i, i + 1);
                            current_perm.swap(i, i + 1);
                            changed = true;
                        }
                    }
                }
            }

            if !layer.swaps.is_empty() {
                self.add_layer(layer);
                depth += 1;
            }
        }

        Ok(())
    }

    /// Check if swapping two adjacent positions makes progress toward target
    fn should_swap_for_target(
        &self,
        current: &[usize],
        target: &[usize],
        i: usize,
        j: usize,
    ) -> bool {
        if i >= current.len() || j >= current.len() || i >= target.len() || j >= target.len() {
            return false;
        }

        // Check if the swap brings either element closer to its target position
        (current[i] == target[j] && current[j] != target[j])
            || (current[j] == target[i] && current[i] != target[i])
    }
}

/// Utilities for generating common SWAP networks
pub mod networks {
    use super::*;

    /// Generate a SWAP network for reversing a linear array
    pub fn linear_reversal(num_qubits: usize) -> SwapNetwork {
        let coupling_map = CouplingMap::linear(num_qubits);
        let mut network = SwapNetwork::new(coupling_map);

        // Simple reversal using bubble sort pattern
        for layer_idx in 0..num_qubits {
            let mut layer = SwapLayer::new(layer_idx);

            for i in (layer_idx..num_qubits - 1).step_by(2) {
                layer.add_swap(i, i + 1);
            }

            if !layer.swaps.is_empty() {
                network.add_layer(layer);
            }
        }

        network
    }

    /// Generate a SWAP network for circular rotation
    pub fn circular_rotation(num_qubits: usize, steps: usize) -> SwapNetwork {
        let coupling_map = CouplingMap::ring(num_qubits);
        let mut network = SwapNetwork::new(coupling_map);

        let effective_steps = steps % num_qubits;

        for step in 0..effective_steps {
            let mut layer = SwapLayer::new(step);

            // Rotate by swapping adjacent elements
            for i in 0..num_qubits - 1 {
                layer.add_swap(i, i + 1);
            }

            network.add_layer(layer);
        }

        network
    }

    /// Generate a SWAP network for random permutation
    pub fn random_permutation(
        coupling_map: CouplingMap,
        initial: &HashMap<usize, usize>,
        target: &HashMap<usize, usize>,
    ) -> QuantRS2Result<SwapNetwork> {
        let mut network = SwapNetwork::new(coupling_map);
        network.generate_routing_network(initial, target)?;
        network.optimize();
        Ok(network)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_layer() {
        let mut layer = SwapLayer::new(0);

        assert!(layer.can_add_swap(0, 1));
        layer.add_swap(0, 1);
        assert!(!layer.can_add_swap(0, 2)); // Conflicts with qubit 0
        assert!(layer.can_add_swap(2, 3));

        let qubits = layer.qubits();
        assert!(qubits.contains(&0));
        assert!(qubits.contains(&1));
    }

    #[test]
    fn test_swap_network_generation() {
        let coupling_map = CouplingMap::linear(4);
        let mut network = SwapNetwork::new(coupling_map);

        let initial: HashMap<usize, usize> =
            [(0, 0), (1, 1), (2, 2), (3, 3)].iter().cloned().collect();
        let target: HashMap<usize, usize> =
            [(0, 3), (1, 2), (2, 1), (3, 0)].iter().cloned().collect();

        let result = network.generate_routing_network(&initial, &target);
        assert!(result.is_ok());
        assert!(!network.is_empty());
    }

    #[test]
    fn test_linear_reversal_network() {
        let network = networks::linear_reversal(4);
        assert!(!network.is_empty());
        assert!(network.total_swaps() > 0);
    }

    #[test]
    fn test_network_optimization() {
        let coupling_map = CouplingMap::linear(3);
        let mut network = SwapNetwork::new(coupling_map);

        // Add redundant SWAPs (same SWAP twice should cancel)
        let mut layer1 = SwapLayer::new(0);
        layer1.add_swap(0, 1);
        network.add_layer(layer1);

        let mut layer2 = SwapLayer::new(1);
        layer2.add_swap(0, 1); // This should cancel with the first one
        network.add_layer(layer2);

        network.optimize();
        assert!(network.is_empty()); // Should be empty after optimization
    }
}
