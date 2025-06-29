//! Noise-aware circuit optimization
//!
//! This module provides optimization passes that consider quantum device noise
//! characteristics when making optimization decisions.

use crate::builder::Circuit;
use crate::optimization::passes::OptimizationPassExt;
use crate::optimization::{CircuitMetrics, CostModel, OptimizationPass};
use crate::routing::CouplingMap;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Noise model for quantum devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseModel {
    /// Single-qubit gate error rates (per qubit)
    pub single_qubit_errors: HashMap<usize, f64>,
    /// Two-qubit gate error rates (per qubit pair)
    pub two_qubit_errors: HashMap<(usize, usize), f64>,
    /// T1 coherence times (microseconds)
    pub t1_times: HashMap<usize, f64>,
    /// T2 coherence times (microseconds)
    pub t2_times: HashMap<usize, f64>,
    /// Readout fidelities
    pub readout_fidelities: HashMap<usize, f64>,
    /// Gate execution times (nanoseconds)
    pub gate_times: HashMap<String, f64>,
    /// Crosstalk matrix
    pub crosstalk_matrix: Option<Vec<Vec<f64>>>,
}

impl NoiseModel {
    /// Create a new empty noise model
    pub fn new() -> Self {
        Self {
            single_qubit_errors: HashMap::new(),
            two_qubit_errors: HashMap::new(),
            t1_times: HashMap::new(),
            t2_times: HashMap::new(),
            readout_fidelities: HashMap::new(),
            gate_times: HashMap::new(),
            crosstalk_matrix: None,
        }
    }

    /// Create a uniform noise model for testing
    pub fn uniform(num_qubits: usize) -> Self {
        let mut model = Self::new();

        // Default error rates
        let single_error = 1e-3;
        let two_qubit_error = 1e-2;
        let t1 = 100.0; // microseconds
        let t2 = 50.0; // microseconds
        let readout_fidelity = 0.99;

        for i in 0..num_qubits {
            model.single_qubit_errors.insert(i, single_error);
            model.t1_times.insert(i, t1);
            model.t2_times.insert(i, t2);
            model.readout_fidelities.insert(i, readout_fidelity);

            for j in (i + 1)..num_qubits {
                model.two_qubit_errors.insert((i, j), two_qubit_error);
            }
        }

        // Default gate times (nanoseconds)
        model.gate_times.insert("H".to_string(), 20.0);
        model.gate_times.insert("X".to_string(), 20.0);
        model.gate_times.insert("Y".to_string(), 20.0);
        model.gate_times.insert("Z".to_string(), 0.0); // Virtual Z
        model.gate_times.insert("S".to_string(), 0.0);
        model.gate_times.insert("T".to_string(), 0.0);
        model.gate_times.insert("CNOT".to_string(), 200.0);
        model.gate_times.insert("CZ".to_string(), 200.0);
        model.gate_times.insert("SWAP".to_string(), 600.0); // 3 CNOTs

        model
    }

    /// Create a realistic noise model based on IBM devices
    pub fn ibm_like(num_qubits: usize) -> Self {
        let mut model = Self::new();

        // IBM-like parameters
        for i in 0..num_qubits {
            model.single_qubit_errors.insert(i, 1e-4); // Good single-qubit gates
            model.t1_times.insert(i, 100.0 + (i as f64 * 10.0)); // Varying T1
            model.t2_times.insert(i, 80.0 + (i as f64 * 5.0)); // Varying T2
            model
                .readout_fidelities
                .insert(i, 0.95 + (i as f64 * 0.01).min(0.04));

            for j in (i + 1)..num_qubits {
                // Two-qubit errors vary by connectivity
                let error = if (i as isize - j as isize).abs() == 1 {
                    5e-3 // Adjacent qubits
                } else {
                    1e-2 // Non-adjacent (if connected)
                };
                model.two_qubit_errors.insert((i, j), error);
            }
        }

        // IBM-like gate times
        model.gate_times.insert("H".to_string(), 35.0);
        model.gate_times.insert("X".to_string(), 35.0);
        model.gate_times.insert("Y".to_string(), 35.0);
        model.gate_times.insert("Z".to_string(), 0.0);
        model.gate_times.insert("S".to_string(), 0.0);
        model.gate_times.insert("T".to_string(), 0.0);
        model.gate_times.insert("CNOT".to_string(), 500.0);
        model.gate_times.insert("CZ".to_string(), 300.0);
        model.gate_times.insert("SWAP".to_string(), 1500.0);

        model
    }

    /// Get error rate for a single-qubit gate
    pub fn single_qubit_error(&self, qubit: usize) -> f64 {
        self.single_qubit_errors
            .get(&qubit)
            .copied()
            .unwrap_or(1e-3)
    }

    /// Get error rate for a two-qubit gate
    pub fn two_qubit_error(&self, q1: usize, q2: usize) -> f64 {
        let key = (q1.min(q2), q1.max(q2));
        self.two_qubit_errors.get(&key).copied().unwrap_or(1e-2)
    }

    /// Get T1 time for a qubit
    pub fn t1_time(&self, qubit: usize) -> f64 {
        self.t1_times.get(&qubit).copied().unwrap_or(100.0)
    }

    /// Get T2 time for a qubit
    pub fn t2_time(&self, qubit: usize) -> f64 {
        self.t2_times.get(&qubit).copied().unwrap_or(50.0)
    }

    /// Get gate execution time
    pub fn gate_time(&self, gate_name: &str) -> f64 {
        self.gate_times.get(gate_name).copied().unwrap_or(100.0)
    }

    /// Calculate error probability for a gate
    pub fn gate_error_probability(&self, gate: &dyn GateOp) -> f64 {
        let qubits = gate.qubits();
        match qubits.len() {
            1 => self.single_qubit_error(qubits[0].id() as usize),
            2 => self.two_qubit_error(qubits[0].id() as usize, qubits[1].id() as usize),
            _ => 0.1, // Multi-qubit gates are expensive
        }
    }
}

impl Default for NoiseModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Noise-aware cost model that considers device characteristics
#[derive(Debug, Clone)]
pub struct NoiseAwareCostModel {
    noise_model: NoiseModel,
    coupling_map: Option<CouplingMap>,
    /// Weight for error rate in cost calculation
    pub error_weight: f64,
    /// Weight for execution time in cost calculation
    pub time_weight: f64,
    /// Weight for coherence effects
    pub coherence_weight: f64,
}

impl NoiseAwareCostModel {
    /// Create a new noise-aware cost model
    pub fn new(noise_model: NoiseModel) -> Self {
        Self {
            noise_model,
            coupling_map: None,
            error_weight: 1000.0,
            time_weight: 1.0,
            coherence_weight: 100.0,
        }
    }

    /// Set the coupling map for connectivity analysis
    pub fn with_coupling_map(mut self, coupling_map: CouplingMap) -> Self {
        self.coupling_map = Some(coupling_map);
        self
    }

    /// Calculate cost for a single gate
    pub fn gate_cost(&self, gate: &dyn GateOp) -> f64 {
        let error_prob = self.noise_model.gate_error_probability(gate);
        let exec_time = self.noise_model.gate_time(gate.name());

        // Base cost from error probability and execution time
        let mut cost = self.error_weight * error_prob + self.time_weight * exec_time;

        // Add coherence penalty for long operations
        if exec_time > 100.0 {
            let qubits = gate.qubits();
            for qubit in qubits {
                let t2 = self.noise_model.t2_time(qubit.id() as usize);
                let coherence_penalty = exec_time / (t2 * 1000.0); // Convert μs to ns
                cost += self.coherence_weight * coherence_penalty;
            }
        }

        cost
    }

    /// Calculate total circuit cost
    pub fn circuit_cost<const N: usize>(&self, circuit: &Circuit<N>) -> f64 {
        let mut total_cost = 0.0;
        let mut total_time = 0.0;

        // Simple sequential cost model (no parallelism analysis)
        for gate in circuit.gates() {
            total_cost += self.gate_cost(gate.as_ref());
            total_time += self.noise_model.gate_time(gate.name());
        }

        // Add coherence penalties for total execution time
        if total_time > 0.0 {
            for i in 0..N {
                let t2 = self.noise_model.t2_time(i);
                let coherence_penalty = total_time / (t2 * 1000.0);
                total_cost += self.coherence_weight * coherence_penalty;
            }
        }

        total_cost
    }
}

impl CostModel for NoiseAwareCostModel {
    fn gate_cost(&self, gate: &dyn GateOp) -> f64 {
        let error_prob = self.noise_model.gate_error_probability(gate);
        let exec_time = self.noise_model.gate_time(gate.name());

        // Base cost from error probability and execution time
        let mut cost = self.error_weight * error_prob + self.time_weight * exec_time;

        // Add coherence penalty for long operations
        if exec_time > 100.0 {
            let qubits = gate.qubits();
            for qubit in qubits {
                let t2 = self.noise_model.t2_time(qubit.id() as usize);
                let coherence_penalty = exec_time / (t2 * 1000.0); // Convert μs to ns
                cost += self.coherence_weight * coherence_penalty;
            }
        }

        cost
    }

    fn circuit_cost_from_gates(&self, gates: &[Box<dyn GateOp>]) -> f64 {
        let mut total_cost = 0.0;
        let mut total_time = 0.0;

        // Simple sequential cost model (no parallelism analysis)
        for gate in gates {
            total_cost += self.gate_cost(gate.as_ref());
            total_time += self.noise_model.gate_time(gate.name());
        }

        total_cost
    }

    fn weights(&self) -> super::cost_model::CostWeights {
        super::cost_model::CostWeights {
            gate_count: 1.0,
            execution_time: self.time_weight,
            error_rate: self.error_weight,
            circuit_depth: 1.0,
        }
    }

    fn is_native(&self, gate: &dyn GateOp) -> bool {
        // Consider basic gates as native
        matches!(
            gate.name(),
            "H" | "X" | "Y" | "Z" | "S" | "T" | "CNOT" | "CZ"
        )
    }
}

/// Optimization pass that reduces circuit depth to minimize decoherence
#[derive(Debug, Clone)]
pub struct CoherenceOptimization {
    noise_model: NoiseModel,
    max_parallel_gates: usize,
}

impl CoherenceOptimization {
    /// Create a new coherence optimization pass
    pub fn new(noise_model: NoiseModel) -> Self {
        Self {
            noise_model,
            max_parallel_gates: 10,
        }
    }

    /// Analyze parallelizable gates
    fn find_parallel_gates<const N: usize>(&self, circuit: &Circuit<N>) -> Vec<Vec<usize>> {
        let gates = circuit.gates();
        let mut parallel_groups = Vec::new();
        let mut used_qubits = vec![false; N];
        let mut current_group = Vec::new();

        for (i, gate) in gates.iter().enumerate() {
            let gate_qubits: Vec<_> = gate.qubits().iter().map(|q| q.id() as usize).collect();

            // Check if this gate conflicts with current group
            let conflicts = gate_qubits.iter().any(|&q| used_qubits[q]);

            if conflicts || current_group.len() >= self.max_parallel_gates {
                // Start new group
                if !current_group.is_empty() {
                    parallel_groups.push(current_group);
                    current_group = Vec::new();
                    used_qubits.fill(false);
                }
            }

            // Add gate to current group
            current_group.push(i);
            for &q in &gate_qubits {
                used_qubits[q] = true;
            }
        }

        if !current_group.is_empty() {
            parallel_groups.push(current_group);
        }

        parallel_groups
    }
}

impl OptimizationPass for CoherenceOptimization {
    fn name(&self) -> &str {
        "CoherenceOptimization"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        _cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // For now, return the original gates
        // TODO: Implement gate reordering based on parallel groups
        Ok(gates)
    }

    fn should_apply(&self) -> bool {
        true
    }
}

/// Optimization pass that prioritizes low-noise qubit assignments
#[derive(Debug, Clone)]
pub struct NoiseAwareMapping {
    noise_model: NoiseModel,
    coupling_map: CouplingMap,
}

impl NoiseAwareMapping {
    /// Create a new noise-aware mapping pass
    pub fn new(noise_model: NoiseModel, coupling_map: CouplingMap) -> Self {
        Self {
            noise_model,
            coupling_map,
        }
    }

    /// Score a qubit assignment based on noise characteristics
    fn score_assignment(&self, logical_qubits: &[usize], physical_qubits: &[usize]) -> f64 {
        let mut score = 0.0;

        // Prefer qubits with lower error rates
        for (&logical, &physical) in logical_qubits.iter().zip(physical_qubits.iter()) {
            score += 1.0 / (1.0 + self.noise_model.single_qubit_error(physical));
        }

        // Prefer assignments that minimize two-qubit gate errors
        for i in 0..logical_qubits.len() {
            for j in (i + 1)..logical_qubits.len() {
                let p1 = physical_qubits[i];
                let p2 = physical_qubits[j];

                if self.coupling_map.are_connected(p1, p2) {
                    let error = self.noise_model.two_qubit_error(p1, p2);
                    score += 1.0 / (1.0 + error);
                }
            }
        }

        score
    }
}

impl OptimizationPass for NoiseAwareMapping {
    fn name(&self) -> &str {
        "NoiseAwareMapping"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        _cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // For now, return the original gates
        // TODO: Implement qubit remapping based on noise characteristics
        Ok(gates)
    }

    fn should_apply(&self) -> bool {
        true
    }
}

/// Optimization pass that inserts dynamical decoupling sequences
#[derive(Debug, Clone)]
pub struct DynamicalDecoupling {
    noise_model: NoiseModel,
    /// Minimum idle time to insert decoupling (nanoseconds)
    pub min_idle_time: f64,
    /// Decoupling sequence type
    pub sequence_type: DecouplingSequence,
}

/// Types of dynamical decoupling sequences
#[derive(Debug, Clone)]
pub enum DecouplingSequence {
    /// XY-4 sequence
    XY4,
    /// CPMG sequence
    CPMG,
    /// XY-8 sequence
    XY8,
}

impl DynamicalDecoupling {
    /// Create a new dynamical decoupling pass
    pub fn new(noise_model: NoiseModel) -> Self {
        Self {
            noise_model,
            min_idle_time: 1000.0, // 1 microsecond
            sequence_type: DecouplingSequence::XY4,
        }
    }

    /// Calculate idle times between gates
    fn analyze_idle_times<const N: usize>(&self, circuit: &Circuit<N>) -> Vec<(usize, f64)> {
        let mut idle_times = Vec::new();

        // Simplified analysis - in practice would need detailed scheduling
        for (i, gate) in circuit.gates().iter().enumerate() {
            let exec_time = self.noise_model.gate_time(gate.name());

            // Check for potential idle time after this gate
            if i + 1 < circuit.num_gates() {
                // Simplified: assume some idle time exists
                idle_times.push((i, 500.0)); // 500ns idle time
            }
        }

        idle_times
    }
}

impl OptimizationPass for DynamicalDecoupling {
    fn name(&self) -> &str {
        "DynamicalDecoupling"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        _cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // For now, return the original gates
        // TODO: Insert decoupling sequences during idle periods
        Ok(gates)
    }

    fn should_apply(&self) -> bool {
        true
    }
}

/// Comprehensive noise-aware optimization pass manager
#[derive(Debug)]
pub struct NoiseAwareOptimizer {
    noise_model: NoiseModel,
    coupling_map: Option<CouplingMap>,
    cost_model: NoiseAwareCostModel,
}

impl NoiseAwareOptimizer {
    /// Create a new noise-aware optimizer
    pub fn new(noise_model: NoiseModel) -> Self {
        let cost_model = NoiseAwareCostModel::new(noise_model.clone());

        Self {
            noise_model,
            coupling_map: None,
            cost_model,
        }
    }

    /// Set the coupling map
    pub fn with_coupling_map(mut self, coupling_map: CouplingMap) -> Self {
        self.cost_model = self.cost_model.with_coupling_map(coupling_map.clone());
        self.coupling_map = Some(coupling_map);
        self
    }

    /// Get all noise-aware optimization passes
    pub fn get_passes(&self) -> Vec<Box<dyn OptimizationPass>> {
        let mut passes: Vec<Box<dyn OptimizationPass>> = Vec::new();

        // Add coherence optimization
        passes.push(Box::new(CoherenceOptimization::new(
            self.noise_model.clone(),
        )));

        // Add noise-aware mapping if coupling map is available
        if let Some(ref coupling_map) = self.coupling_map {
            passes.push(Box::new(NoiseAwareMapping::new(
                self.noise_model.clone(),
                coupling_map.clone(),
            )));
        }

        // Add dynamical decoupling
        passes.push(Box::new(DynamicalDecoupling::new(self.noise_model.clone())));

        passes
    }

    /// Optimize a circuit with noise awareness
    pub fn optimize<const N: usize>(&self, circuit: &Circuit<N>) -> QuantRS2Result<Circuit<N>> {
        // For now, return the original circuit
        // TODO: Implement proper circuit optimization using passes
        Ok(circuit.clone())
    }

    /// Estimate circuit fidelity
    pub fn estimate_fidelity<const N: usize>(&self, circuit: &Circuit<N>) -> f64 {
        let mut total_error_prob = 0.0;

        for gate in circuit.gates() {
            let error_prob = self.noise_model.gate_error_probability(gate.as_ref());
            total_error_prob += error_prob;
        }

        // Simple first-order approximation
        (1.0 - total_error_prob).max(0.0)
    }

    /// Get the noise model
    pub fn noise_model(&self) -> &NoiseModel {
        &self.noise_model
    }

    /// Get the cost model
    pub fn cost_model(&self) -> &NoiseAwareCostModel {
        &self.cost_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::{multi::CNOT, single::Hadamard};

    #[test]
    fn test_noise_model_creation() {
        let model = NoiseModel::uniform(4);

        assert_eq!(model.single_qubit_error(0), 1e-3);
        assert_eq!(model.two_qubit_error(0, 1), 1e-2);
        assert_eq!(model.t1_time(0), 100.0);
        assert_eq!(model.gate_time("CNOT"), 200.0);
    }

    #[test]
    fn test_ibm_noise_model() {
        let model = NoiseModel::ibm_like(3);

        assert_eq!(model.single_qubit_error(0), 1e-4);
        assert_eq!(model.gate_time("H"), 35.0);
        assert!(model.t1_time(1) > model.t1_time(0));
    }

    #[test]
    fn test_noise_aware_cost_model() {
        let noise_model = NoiseModel::uniform(4);
        let cost_model = NoiseAwareCostModel::new(noise_model);

        let h_gate = Hadamard { target: QubitId(0) };
        let cnot_gate = CNOT {
            control: QubitId(0),
            target: QubitId(1),
        };

        let h_cost = cost_model.gate_cost(&h_gate);
        let cnot_cost = cost_model.gate_cost(&cnot_gate);

        // CNOT should be more expensive than Hadamard
        assert!(cnot_cost > h_cost);
    }

    #[test]
    fn test_coherence_optimization() {
        let noise_model = NoiseModel::uniform(4);
        let optimizer = CoherenceOptimization::new(noise_model.clone());

        let mut circuit = Circuit::<4>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();

        let cost_model = NoiseAwareCostModel::new(noise_model);
        let result = optimizer.apply(&circuit, &cost_model);
        assert!(result.is_ok());
    }

    #[test]
    fn test_noise_aware_optimizer() {
        let noise_model = NoiseModel::uniform(4);
        let optimizer = NoiseAwareOptimizer::new(noise_model);

        let mut circuit = Circuit::<4>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();

        let optimized = optimizer.optimize(&circuit).unwrap();
        let fidelity = optimizer.estimate_fidelity(&optimized);

        assert!(fidelity > 0.9); // Should have high fidelity for simple circuit
        assert!(fidelity <= 1.0);
    }
}
