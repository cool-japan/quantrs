//! Quantum Circuit Optimization Pipeline
//!
//! Advanced circuit optimization using gate fusion, parallelization detection,
//! and circuit depth reduction techniques.

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::optimizations_stable::gate_fusion::{
    apply_gate_fusion, FusedGateSequence, GateType, QuantumGate,
};
use std::collections::{HashMap, HashSet};

/// Optimization levels for circuits
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationLevel {
    None,
    Basic,      // Gate fusion only
    Standard,   // Gate fusion + dead code elimination
    Aggressive, // All optimizations + circuit synthesis
}

/// Circuit performance metrics
#[derive(Debug, Clone, Default)]
pub struct CircuitMetrics {
    pub total_gates: usize,
    pub single_qubit_gates: usize,
    pub two_qubit_gates: usize,
    pub multi_qubit_gates: usize,
    pub circuit_depth: usize,
    pub parallelizable_operations: usize,
    pub critical_path_length: usize,
    pub estimated_execution_time_ns: u64,
}

/// Quantum circuit representation for optimization
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    pub gates: Vec<QuantumGate>,
    pub num_qubits: usize,
    pub qubit_map: HashMap<usize, String>, // Physical to logical mapping
}

impl QuantumCircuit {
    /// Create a new quantum circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
            qubit_map: HashMap::new(),
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: QuantumGate) -> QuantRS2Result<()> {
        // Validate gate qubits are within circuit bounds
        for &qubit in &gate.qubits {
            if qubit >= self.num_qubits {
                return Err(QuantRS2Error::InvalidQubitId(qubit as u32));
            }
        }

        self.gates.push(gate);
        Ok(())
    }

    /// Calculate circuit metrics
    pub fn calculate_metrics(&self) -> CircuitMetrics {
        let mut metrics = CircuitMetrics::default();
        metrics.total_gates = self.gates.len();

        // Count gate types
        for gate in &self.gates {
            match gate.num_qubits() {
                1 => metrics.single_qubit_gates += 1,
                2 => metrics.two_qubit_gates += 1,
                _ => metrics.multi_qubit_gates += 1,
            }
        }

        // Calculate circuit depth and parallelization opportunities
        let depth_analysis = self.analyze_circuit_depth();
        metrics.circuit_depth = depth_analysis.depth;
        metrics.parallelizable_operations = depth_analysis.parallel_ops;
        metrics.critical_path_length = depth_analysis.critical_path;

        // Estimate execution time (rough approximation)
        // Single-qubit: 10ns, Two-qubit: 100ns, Multi-qubit: 1000ns
        metrics.estimated_execution_time_ns = (metrics.single_qubit_gates * 10) as u64
            + (metrics.two_qubit_gates * 100) as u64
            + (metrics.multi_qubit_gates * 1000) as u64;

        metrics
    }

    /// Analyze circuit depth and parallelization opportunities
    fn analyze_circuit_depth(&self) -> DepthAnalysis {
        let mut qubit_last_used = vec![0usize; self.num_qubits];
        let mut max_depth = 0;
        let mut parallel_groups = Vec::new();
        let mut current_parallel_group = Vec::new();

        for (gate_idx, gate) in self.gates.iter().enumerate() {
            // Find the earliest time this gate can be executed
            let earliest_time = gate
                .qubits
                .iter()
                .map(|&q| qubit_last_used[q])
                .max()
                .unwrap_or(0);

            // Update qubit usage times
            for &qubit in &gate.qubits {
                qubit_last_used[qubit] = earliest_time + 1;
            }

            max_depth = max_depth.max(earliest_time + 1);

            // Check if this gate can run in parallel with previous gates
            if current_parallel_group.is_empty()
                || self.can_run_in_parallel(gate, &current_parallel_group)
            {
                current_parallel_group.push(gate_idx);
            } else {
                if current_parallel_group.len() > 1 {
                    parallel_groups.push(current_parallel_group.clone());
                }
                current_parallel_group = vec![gate_idx];
            }
        }

        // Add final group if it has parallelizable operations
        if current_parallel_group.len() > 1 {
            parallel_groups.push(current_parallel_group);
        }

        let parallel_ops = parallel_groups
            .iter()
            .map(|g| g.len())
            .sum::<usize>()
            .saturating_sub(parallel_groups.len());

        DepthAnalysis {
            depth: max_depth,
            parallel_ops,
            critical_path: max_depth, // Simplified: actual critical path would need more analysis
        }
    }

    /// Check if a gate can run in parallel with a group of gates
    fn can_run_in_parallel(&self, gate: &QuantumGate, group_indices: &[usize]) -> bool {
        let gate_qubits: HashSet<usize> = gate.qubits.iter().copied().collect();

        for &idx in group_indices {
            let other_gate = &self.gates[idx];
            let other_qubits: HashSet<usize> = other_gate.qubits.iter().copied().collect();

            // Gates can't run in parallel if they share qubits
            if !gate_qubits.is_disjoint(&other_qubits) {
                return false;
            }
        }

        true
    }

    /// Remove redundant gates (identity elimination)
    pub fn eliminate_redundant_gates(&mut self) -> usize {
        let mut eliminated_count = 0;
        let mut new_gates = Vec::new();

        let mut i = 0;
        while i < self.gates.len() {
            let current_gate = &self.gates[i];

            // Check for identity patterns
            if i + 1 < self.gates.len() {
                let next_gate = &self.gates[i + 1];

                // Check for self-inverse gates on same qubits
                if self.are_inverse_gates(current_gate, next_gate) {
                    eliminated_count += 2;
                    i += 2; // Skip both gates
                    continue;
                }
            }

            new_gates.push(current_gate.clone());
            i += 1;
        }

        self.gates = new_gates;
        eliminated_count
    }

    /// Check if two gates are inverses of each other
    fn are_inverse_gates(&self, gate1: &QuantumGate, gate2: &QuantumGate) -> bool {
        // Must act on same qubits
        if gate1.qubits != gate2.qubits {
            return false;
        }

        // Check for known inverse pairs
        match (&gate1.gate_type, &gate2.gate_type) {
            (GateType::PauliX, GateType::PauliX)
            | (GateType::PauliY, GateType::PauliY)
            | (GateType::PauliZ, GateType::PauliZ)
            | (GateType::Hadamard, GateType::Hadamard) => true,

            (GateType::RX(a1), GateType::RX(a2))
            | (GateType::RY(a1), GateType::RY(a2))
            | (GateType::RZ(a1), GateType::RZ(a2)) => {
                // Check if angles sum to 2π (modulo 2π)
                let angle1 = (*a1 as f64) / 1_000_000.0;
                let angle2 = (*a2 as f64) / 1_000_000.0;
                let sum = (angle1 + angle2) % (2.0 * std::f64::consts::PI);
                sum.abs() < 1e-10 || 2.0f64.mul_add(-std::f64::consts::PI, sum).abs() < 1e-10
            }

            _ => false,
        }
    }

    /// Convert gates to optimal sequences
    pub fn optimize_gate_sequences(&mut self) -> QuantRS2Result<usize> {
        // Group gates by qubits they act on
        let mut qubit_sequences: HashMap<Vec<usize>, Vec<QuantumGate>> = HashMap::new();

        for gate in &self.gates {
            let mut qubits = gate.qubits.clone();
            qubits.sort_unstable();
            qubit_sequences
                .entry(qubits)
                .or_insert_with(Vec::new)
                .push(gate.clone());
        }

        let mut total_optimizations = 0;
        let mut new_gates = Vec::new();

        // Optimize each sequence independently
        for (qubits, gates) in qubit_sequences {
            let fused_sequences = apply_gate_fusion(gates)?;

            for sequence in fused_sequences {
                if sequence.gates.len() > 1 {
                    total_optimizations += sequence.gates.len() - 1;
                }
                // Add the optimized gates back
                new_gates.extend(sequence.gates);
            }
        }

        self.gates = new_gates;
        Ok(total_optimizations)
    }
}

/// Circuit depth analysis result
#[derive(Debug, Clone)]
struct DepthAnalysis {
    depth: usize,
    parallel_ops: usize,
    critical_path: usize,
}

/// Comprehensive circuit optimizer
pub struct CircuitOptimizer {
    optimization_level: OptimizationLevel,
    statistics: CircuitOptimizationStats,
}

/// Optimization statistics
#[derive(Debug, Clone, Default)]
pub struct CircuitOptimizationStats {
    pub circuits_optimized: usize,
    pub total_gates_eliminated: usize,
    pub total_depth_reduction: usize,
    pub average_speedup: f64,
}

impl CircuitOptimizer {
    /// Create a new circuit optimizer
    pub fn new(level: OptimizationLevel) -> Self {
        Self {
            optimization_level: level,
            statistics: CircuitOptimizationStats::default(),
        }
    }

    /// Optimize a quantum circuit
    pub fn optimize(&mut self, mut circuit: QuantumCircuit) -> QuantRS2Result<QuantumCircuit> {
        if self.optimization_level == OptimizationLevel::None {
            return Ok(circuit);
        }

        let original_metrics = circuit.calculate_metrics();

        // Phase 1: Gate fusion and sequence optimization
        if self.optimization_level >= OptimizationLevel::Basic {
            circuit.optimize_gate_sequences()?;
        }

        // Phase 2: Dead code elimination
        if self.optimization_level >= OptimizationLevel::Standard {
            let eliminated = circuit.eliminate_redundant_gates();
            self.statistics.total_gates_eliminated += eliminated;
        }

        // Phase 3: Advanced optimizations
        if self.optimization_level == OptimizationLevel::Aggressive {
            self.apply_advanced_optimizations(&mut circuit)?;
        }

        // Update statistics
        let final_metrics = circuit.calculate_metrics();
        self.statistics.circuits_optimized += 1;
        self.statistics.total_depth_reduction += original_metrics
            .circuit_depth
            .saturating_sub(final_metrics.circuit_depth);

        let speedup = if final_metrics.estimated_execution_time_ns > 0 {
            original_metrics.estimated_execution_time_ns as f64
                / final_metrics.estimated_execution_time_ns as f64
        } else {
            1.0
        };

        self.statistics.average_speedup = self
            .statistics
            .average_speedup
            .mul_add((self.statistics.circuits_optimized - 1) as f64, speedup)
            / self.statistics.circuits_optimized as f64;

        Ok(circuit)
    }

    /// Apply advanced optimization techniques
    fn apply_advanced_optimizations(&self, circuit: &mut QuantumCircuit) -> QuantRS2Result<()> {
        // Commutation-based optimization
        self.optimize_commuting_gates(circuit)?;

        // Circuit synthesis optimization
        self.synthesize_efficient_sequences(circuit)?;

        Ok(())
    }

    /// Optimize commuting gates by reordering for better parallelization
    fn optimize_commuting_gates(&self, circuit: &mut QuantumCircuit) -> QuantRS2Result<()> {
        // Find commuting gate pairs and reorder for optimal parallelization
        let mut optimized = false;

        // Simple bubble-sort style optimization for commuting gates
        for i in 0..circuit.gates.len() {
            for j in (i + 1)..circuit.gates.len() {
                if self.gates_commute(&circuit.gates[i], &circuit.gates[j])
                    && self.should_swap_for_optimization(&circuit.gates[i], &circuit.gates[j])
                {
                    circuit.gates.swap(i, j);
                    optimized = true;
                }
            }
        }

        Ok(())
    }

    /// Check if two gates commute
    fn gates_commute(&self, gate1: &QuantumGate, gate2: &QuantumGate) -> bool {
        let qubits1: HashSet<usize> = gate1.qubits.iter().copied().collect();
        let qubits2: HashSet<usize> = gate2.qubits.iter().copied().collect();

        // Gates on disjoint qubit sets always commute
        if qubits1.is_disjoint(&qubits2) {
            return true;
        }

        // Some specific gate pairs commute even on same qubits
        match (&gate1.gate_type, &gate2.gate_type) {
            (GateType::PauliZ, GateType::RZ(_)) | (GateType::RZ(_), GateType::PauliZ) => true,
            _ => false,
        }
    }

    /// Determine if swapping gates would improve optimization
    const fn should_swap_for_optimization(
        &self,
        _gate1: &QuantumGate,
        _gate2: &QuantumGate,
    ) -> bool {
        // Simplified heuristic: prefer grouping similar gates together
        false // Conservative approach for now
    }

    /// Synthesize efficient gate sequences using known optimizations
    fn synthesize_efficient_sequences(&self, circuit: &mut QuantumCircuit) -> QuantRS2Result<()> {
        // Apply known synthesis rules (e.g., Solovay-Kitaev approximations)
        // This is a simplified version - real synthesis would be much more complex

        // Look for inefficient rotation sequences
        self.optimize_rotation_sequences(circuit)?;

        Ok(())
    }

    /// Optimize sequences of rotation gates
    fn optimize_rotation_sequences(&self, circuit: &mut QuantumCircuit) -> QuantRS2Result<()> {
        let mut new_gates = Vec::new();
        let mut i = 0;

        while i < circuit.gates.len() {
            // Look for consecutive rotations on same axis and qubit
            let current_gate = &circuit.gates[i];

            if let Some(optimized_sequence) = self.find_optimizable_rotation_sequence(circuit, i) {
                new_gates.extend(optimized_sequence.gates);
                i += optimized_sequence.original_length;
            } else {
                new_gates.push(current_gate.clone());
                i += 1;
            }
        }

        circuit.gates = new_gates;
        Ok(())
    }

    /// Find and optimize rotation sequences
    fn find_optimizable_rotation_sequence(
        &self,
        circuit: &QuantumCircuit,
        start_idx: usize,
    ) -> Option<OptimizedSequence> {
        let start_gate = &circuit.gates[start_idx];

        // Look for consecutive rotations of the same type on same qubit
        match &start_gate.gate_type {
            GateType::RX(_) | GateType::RY(_) | GateType::RZ(_) => {
                let mut total_angle = 0u64;
                let mut count = 0;

                for gate in &circuit.gates[start_idx..] {
                    if gate.gate_type == start_gate.gate_type && gate.qubits == start_gate.qubits {
                        if let Some(angle) = self.extract_rotation_angle(&gate.gate_type) {
                            total_angle = (total_angle + angle) % (2 * 1_000_000 * 314_159); // 2π in quantized units
                            count += 1;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                if count > 1 {
                    // Create optimized single rotation
                    let optimized_gate_type = match start_gate.gate_type {
                        GateType::RX(_) => GateType::RX(total_angle),
                        GateType::RY(_) => GateType::RY(total_angle),
                        GateType::RZ(_) => GateType::RZ(total_angle),
                        _ => unreachable!(),
                    };

                    if let Ok(optimized_gate) =
                        QuantumGate::new(optimized_gate_type, start_gate.qubits.clone())
                    {
                        return Some(OptimizedSequence {
                            gates: vec![optimized_gate],
                            original_length: count,
                        });
                    }
                }
            }
            _ => {}
        }

        None
    }

    /// Extract rotation angle from gate type
    const fn extract_rotation_angle(&self, gate_type: &GateType) -> Option<u64> {
        match gate_type {
            GateType::RX(angle) | GateType::RY(angle) | GateType::RZ(angle) => Some(*angle),
            _ => None,
        }
    }

    /// Get optimization statistics
    pub const fn get_statistics(&self) -> &CircuitOptimizationStats {
        &self.statistics
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.statistics = CircuitOptimizationStats::default();
    }
}

/// Optimized gate sequence
#[derive(Debug, Clone)]
struct OptimizedSequence {
    gates: Vec<QuantumGate>,
    original_length: usize,
}

/// Optimize a circuit with specified level
pub fn optimize_circuit(
    circuit: QuantumCircuit,
    level: OptimizationLevel,
) -> QuantRS2Result<QuantumCircuit> {
    let mut optimizer = CircuitOptimizer::new(level);
    optimizer.optimize(circuit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_creation() {
        let mut circuit = QuantumCircuit::new(2);
        let gate =
            QuantumGate::new(GateType::Hadamard, vec![0]).expect("Failed to create Hadamard gate");

        assert!(circuit.add_gate(gate).is_ok());
        assert_eq!(circuit.gates.len(), 1);
    }

    #[test]
    fn test_invalid_qubit_rejection() {
        let mut circuit = QuantumCircuit::new(2);
        let invalid_gate = QuantumGate::new(GateType::Hadamard, vec![3])
            .expect("Failed to create gate with invalid qubit"); // Qubit 3 doesn't exist

        assert!(circuit.add_gate(invalid_gate).is_err());
    }

    #[test]
    fn test_redundant_gate_elimination() {
        let mut circuit = QuantumCircuit::new(1);

        // Add two X gates (should cancel out)
        circuit
            .add_gate(
                QuantumGate::new(GateType::PauliX, vec![0])
                    .expect("Failed to create first PauliX gate"),
            )
            .expect("Failed to add first PauliX gate");
        circuit
            .add_gate(
                QuantumGate::new(GateType::PauliX, vec![0])
                    .expect("Failed to create second PauliX gate"),
            )
            .expect("Failed to add second PauliX gate");

        let eliminated = circuit.eliminate_redundant_gates();
        assert_eq!(eliminated, 2);
        assert_eq!(circuit.gates.len(), 0);
    }

    #[test]
    fn test_circuit_metrics() {
        let mut circuit = QuantumCircuit::new(2);

        circuit
            .add_gate(
                QuantumGate::new(GateType::Hadamard, vec![0])
                    .expect("Failed to create Hadamard gate"),
            )
            .expect("Failed to add Hadamard gate");
        circuit
            .add_gate(
                QuantumGate::new(GateType::CNOT, vec![0, 1]).expect("Failed to create CNOT gate"),
            )
            .expect("Failed to add CNOT gate");

        let metrics = circuit.calculate_metrics();
        assert_eq!(metrics.total_gates, 2);
        assert_eq!(metrics.single_qubit_gates, 1);
        assert_eq!(metrics.two_qubit_gates, 1);
        assert!(metrics.estimated_execution_time_ns > 0);
    }

    #[test]
    fn test_circuit_optimization() {
        let mut circuit = QuantumCircuit::new(1);

        // Add redundant gates
        circuit
            .add_gate(
                QuantumGate::new(GateType::Hadamard, vec![0])
                    .expect("Failed to create first Hadamard gate"),
            )
            .expect("Failed to add first Hadamard gate");
        circuit
            .add_gate(
                QuantumGate::new(GateType::Hadamard, vec![0])
                    .expect("Failed to create second Hadamard gate"),
            )
            .expect("Failed to add second Hadamard gate");
        circuit
            .add_gate(
                QuantumGate::new(GateType::PauliX, vec![0]).expect("Failed to create PauliX gate"),
            )
            .expect("Failed to add PauliX gate");

        let optimized = optimize_circuit(circuit, OptimizationLevel::Standard)
            .expect("Failed to optimize circuit");

        // Should eliminate the two Hadamards, leaving only X
        assert_eq!(optimized.gates.len(), 1);
        assert_eq!(optimized.gates[0].gate_type, GateType::PauliX);
    }

    #[test]
    fn test_gate_commutation() {
        let optimizer = CircuitOptimizer::new(OptimizationLevel::Aggressive);

        let gate1 =
            QuantumGate::new(GateType::PauliZ, vec![0]).expect("Failed to create PauliZ gate");
        let gate2 =
            QuantumGate::new(GateType::RZ(1_570_796), vec![0]).expect("Failed to create RZ gate"); // pi/2

        assert!(optimizer.gates_commute(&gate1, &gate2));

        let gate3 =
            QuantumGate::new(GateType::PauliX, vec![0]).expect("Failed to create PauliX gate");
        assert!(!optimizer.gates_commute(&gate1, &gate3)); // X and Z don't commute
    }

    #[test]
    fn test_parallel_gate_detection() {
        let mut circuit = QuantumCircuit::new(2);

        // Add gates on different qubits (should be parallelizable)
        circuit
            .add_gate(
                QuantumGate::new(GateType::Hadamard, vec![0])
                    .expect("Failed to create Hadamard gate on qubit 0"),
            )
            .expect("Failed to add Hadamard gate on qubit 0");
        circuit
            .add_gate(
                QuantumGate::new(GateType::Hadamard, vec![1])
                    .expect("Failed to create Hadamard gate on qubit 1"),
            )
            .expect("Failed to add Hadamard gate on qubit 1");

        let metrics = circuit.calculate_metrics();
        assert!(metrics.parallelizable_operations > 0);
    }
}
