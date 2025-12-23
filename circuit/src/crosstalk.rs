//! Cross-talk aware scheduling for quantum circuits
//!
//! This module provides scheduling algorithms that minimize unwanted
//! interactions between qubits during parallel gate execution.

use crate::builder::Circuit;
use crate::dag::{circuit_to_dag, CircuitDag, DagNode};
use crate::slicing::{CircuitSlice, SlicingStrategy};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Cross-talk model for quantum devices
#[derive(Debug, Clone)]
pub struct CrosstalkModel {
    /// Cross-talk coefficients between qubit pairs during simultaneous operations
    /// Key: ((q1, q2), (q3, q4)) - crosstalk when operating on (q1,q2) and (q3,q4) simultaneously
    pub crosstalk_coefficients: HashMap<((usize, usize), (usize, usize)), f64>,
    /// Single-qubit gate crosstalk
    pub single_qubit_crosstalk: HashMap<(usize, usize), f64>,
    /// Threshold for significant crosstalk
    pub threshold: f64,
    /// Device connectivity
    pub coupling_map: Vec<(usize, usize)>,
}

impl Default for CrosstalkModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CrosstalkModel {
    /// Create a new crosstalk model
    #[must_use]
    pub fn new() -> Self {
        Self {
            crosstalk_coefficients: HashMap::new(),
            single_qubit_crosstalk: HashMap::new(),
            threshold: 0.01,
            coupling_map: Vec::new(),
        }
    }

    /// Create a uniform crosstalk model for testing
    #[must_use]
    pub fn uniform(num_qubits: usize, crosstalk_rate: f64) -> Self {
        let mut model = Self::new();

        // Add crosstalk between all neighboring qubit pairs
        for i in 0..num_qubits {
            for j in (i + 1)..num_qubits {
                let dist = f64::from((i as i32 - j as i32).abs());
                let crosstalk = crosstalk_rate / dist; // Decreases with distance

                // Single-qubit crosstalk
                model.single_qubit_crosstalk.insert((i, j), crosstalk);
                model.single_qubit_crosstalk.insert((j, i), crosstalk);

                // Two-qubit gate crosstalk (higher)
                for k in 0..num_qubits {
                    for l in (k + 1)..num_qubits {
                        if (i, j) != (k, l) {
                            let key = ((i.min(j), i.max(j)), (k.min(l), k.max(l)));
                            model.crosstalk_coefficients.insert(key, crosstalk * 2.0);
                        }
                    }
                }
            }
        }

        // Linear coupling map
        for i in 0..(num_qubits - 1) {
            model.coupling_map.push((i, i + 1));
        }

        model
    }

    /// Load from device characterization data
    #[must_use]
    pub fn from_characterization(data: &CrosstalkCharacterization) -> Self {
        let mut model = Self::new();

        // Load measured crosstalk values
        model
            .crosstalk_coefficients
            .clone_from(&data.measured_crosstalk);
        model.threshold = data.significance_threshold;
        model.coupling_map.clone_from(&data.device_coupling);

        // Compute single-qubit crosstalk from measurements
        for (&(q1, q2), &value) in &data.single_qubit_measurements {
            model.single_qubit_crosstalk.insert((q1, q2), value);
        }

        model
    }

    /// Get crosstalk between two gate operations
    pub fn get_crosstalk(&self, gate1: &dyn GateOp, gate2: &dyn GateOp) -> f64 {
        let qubits1 = gate1.qubits();
        let qubits2 = gate2.qubits();

        // Check if gates share qubits (cannot be parallel)
        for q1 in &qubits1 {
            for q2 in &qubits2 {
                if q1.id() == q2.id() {
                    return f64::INFINITY; // Cannot execute in parallel
                }
            }
        }

        // Calculate crosstalk based on gate types
        match (qubits1.len(), qubits2.len()) {
            (1, 1) => {
                // Single-qubit gates
                let q1 = qubits1[0].id() as usize;
                let q2 = qubits2[0].id() as usize;
                self.single_qubit_crosstalk
                    .get(&(q1, q2))
                    .copied()
                    .unwrap_or(0.0)
            }
            (2, 2) => {
                // Two-qubit gates
                let pair1 = (qubits1[0].id() as usize, qubits1[1].id() as usize);
                let pair2 = (qubits2[0].id() as usize, qubits2[1].id() as usize);
                let key = (
                    (pair1.0.min(pair1.1), pair1.0.max(pair1.1)),
                    (pair2.0.min(pair2.1), pair2.0.max(pair2.1)),
                );
                self.crosstalk_coefficients
                    .get(&key)
                    .copied()
                    .unwrap_or(0.0)
            }
            (1, 2) | (2, 1) => {
                // Mixed single and two-qubit gates
                // Use average of single-qubit crosstalk values
                let mut total = 0.0;
                let mut count = 0;
                for q1 in &qubits1 {
                    for q2 in &qubits2 {
                        let key = (q1.id() as usize, q2.id() as usize);
                        if let Some(&crosstalk) = self.single_qubit_crosstalk.get(&key) {
                            total += crosstalk;
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    total / f64::from(count)
                } else {
                    0.0
                }
            }
            _ => 0.0, // Multi-qubit gates - simplified
        }
    }

    /// Check if two gates can be executed in parallel
    pub fn can_parallelize(&self, gate1: &dyn GateOp, gate2: &dyn GateOp) -> bool {
        self.get_crosstalk(gate1, gate2) < self.threshold
    }
}

/// Measured crosstalk characterization data
#[derive(Debug, Clone)]
pub struct CrosstalkCharacterization {
    pub measured_crosstalk: HashMap<((usize, usize), (usize, usize)), f64>,
    pub single_qubit_measurements: HashMap<(usize, usize), f64>,
    pub significance_threshold: f64,
    pub device_coupling: Vec<(usize, usize)>,
}

/// Scheduling result with crosstalk information
#[derive(Debug)]
pub struct CrosstalkSchedule<const N: usize> {
    /// Scheduled time slices
    pub time_slices: Vec<TimeSlice>,
    /// Original circuit
    pub circuit: Circuit<N>,
    /// Total crosstalk error
    pub total_crosstalk: f64,
    /// Execution time estimate
    pub execution_time: f64,
}

/// Time slice containing parallel gates
#[derive(Debug, Clone)]
pub struct TimeSlice {
    /// Gates to execute in this time slice
    pub gates: Vec<usize>, // Indices into circuit gates
    /// Maximum crosstalk in this slice
    pub max_crosstalk: f64,
    /// Duration of this slice
    pub duration: f64,
}

/// Cross-talk aware scheduler
pub struct CrosstalkScheduler {
    model: CrosstalkModel,
    strategy: SchedulingStrategy,
}

#[derive(Debug, Clone)]
pub enum SchedulingStrategy {
    /// Minimize total crosstalk
    MinimizeCrosstalk,
    /// Minimize execution time with crosstalk constraint
    MinimizeTime { max_crosstalk: f64 },
    /// Balance between crosstalk and time
    Balanced {
        time_weight: f64,
        crosstalk_weight: f64,
    },
}

impl CrosstalkScheduler {
    /// Create a new scheduler
    #[must_use]
    pub const fn new(model: CrosstalkModel) -> Self {
        Self {
            model,
            strategy: SchedulingStrategy::MinimizeCrosstalk,
        }
    }

    /// Set scheduling strategy
    #[must_use]
    pub const fn with_strategy(mut self, strategy: SchedulingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Schedule a circuit
    pub fn schedule<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<CrosstalkSchedule<N>> {
        let dag = circuit_to_dag(circuit);

        match self.strategy {
            SchedulingStrategy::MinimizeCrosstalk => self.schedule_min_crosstalk(&dag, circuit),
            SchedulingStrategy::MinimizeTime { max_crosstalk } => {
                self.schedule_min_time(&dag, circuit, max_crosstalk)
            }
            SchedulingStrategy::Balanced {
                time_weight,
                crosstalk_weight,
            } => self.schedule_balanced(&dag, circuit, time_weight, crosstalk_weight),
        }
    }

    /// Schedule to minimize total crosstalk
    fn schedule_min_crosstalk<const N: usize>(
        &self,
        dag: &CircuitDag,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<CrosstalkSchedule<N>> {
        let gates = circuit.gates();
        let mut time_slices = Vec::new();
        let mut scheduled = vec![false; gates.len()];
        let mut total_crosstalk = 0.0;

        // Get topological order
        let topo_order = dag
            .topological_sort()
            .map_err(QuantRS2Error::InvalidInput)?;

        while scheduled.iter().any(|&s| !s) {
            let mut current_slice = TimeSlice {
                gates: Vec::new(),
                max_crosstalk: 0.0,
                duration: 0.0,
            };

            // Find gates that can be scheduled
            let ready_gates: Vec<usize> = (0..gates.len())
                .filter(|&gate_idx| {
                    !scheduled[gate_idx] && self.dependencies_met(&scheduled, dag, gate_idx)
                })
                .collect();

            // Greedily add gates with minimal crosstalk
            for &gate_idx in &ready_gates {
                let gate = &gates[gate_idx];

                // Check crosstalk with already scheduled gates in this slice
                let mut slice_crosstalk: f64 = 0.0;
                let mut can_add = true;

                for &other_idx in &current_slice.gates {
                    let other_gate = &gates[other_idx];
                    let crosstalk = self.model.get_crosstalk(gate.as_ref(), other_gate.as_ref());

                    if crosstalk == f64::INFINITY {
                        can_add = false;
                        break;
                    }

                    slice_crosstalk = slice_crosstalk.max(crosstalk);
                }

                if can_add && slice_crosstalk < self.model.threshold {
                    current_slice.gates.push(gate_idx);
                    current_slice.max_crosstalk = current_slice.max_crosstalk.max(slice_crosstalk);
                    scheduled[gate_idx] = true;

                    // Update duration (simplified - assume all gates take 100ns)
                    current_slice.duration = 100.0;
                }
            }

            // If no gates could be added, forcibly add one
            if current_slice.gates.is_empty() && !ready_gates.is_empty() {
                let gate_idx = ready_gates[0];
                current_slice.gates.push(gate_idx);
                scheduled[gate_idx] = true;
                current_slice.duration = 100.0;
            }

            total_crosstalk += current_slice.max_crosstalk;
            time_slices.push(current_slice);
        }

        let execution_time = time_slices.iter().map(|s| s.duration).sum();

        Ok(CrosstalkSchedule {
            time_slices,
            circuit: circuit.clone(),
            total_crosstalk,
            execution_time,
        })
    }

    /// Schedule to minimize execution time with crosstalk constraint
    fn schedule_min_time<const N: usize>(
        &self,
        dag: &CircuitDag,
        circuit: &Circuit<N>,
        max_crosstalk: f64,
    ) -> QuantRS2Result<CrosstalkSchedule<N>> {
        // Similar to min_crosstalk but prioritizes parallelism
        // up to the crosstalk threshold
        self.schedule_min_crosstalk(dag, circuit) // Simplified for now
    }

    /// Balanced scheduling
    fn schedule_balanced<const N: usize>(
        &self,
        dag: &CircuitDag,
        circuit: &Circuit<N>,
        time_weight: f64,
        crosstalk_weight: f64,
    ) -> QuantRS2Result<CrosstalkSchedule<N>> {
        // Optimize weighted combination of time and crosstalk
        self.schedule_min_crosstalk(dag, circuit) // Simplified for now
    }

    /// Check if all dependencies of a gate have been scheduled
    fn dependencies_met(&self, scheduled: &[bool], dag: &CircuitDag, gate_idx: usize) -> bool {
        // Find the node corresponding to this gate
        for node in dag.nodes() {
            if node.id == gate_idx {
                // Check all predecessors
                return node.predecessors.iter().all(|&pred_id| {
                    if pred_id < scheduled.len() {
                        scheduled[pred_id]
                    } else {
                        true // Input/Output nodes
                    }
                });
            }
        }
        false
    }
}

/// Analyze crosstalk in a circuit
pub struct CrosstalkAnalyzer {
    model: CrosstalkModel,
}

impl CrosstalkAnalyzer {
    /// Create a new analyzer
    #[must_use]
    pub const fn new(model: CrosstalkModel) -> Self {
        Self { model }
    }

    /// Analyze potential crosstalk in a circuit
    #[must_use]
    pub fn analyze<const N: usize>(&self, circuit: &Circuit<N>) -> CrosstalkAnalysis {
        let gates = circuit.gates();
        let mut problematic_pairs = Vec::new();
        let mut crosstalk_graph = HashMap::new();

        // Check all gate pairs
        for i in 0..gates.len() {
            for j in (i + 1)..gates.len() {
                let crosstalk = self
                    .model
                    .get_crosstalk(gates[i].as_ref(), gates[j].as_ref());

                if crosstalk > self.model.threshold && crosstalk < f64::INFINITY {
                    problematic_pairs.push((i, j, crosstalk));
                }

                if crosstalk > 0.0 && crosstalk < f64::INFINITY {
                    crosstalk_graph.insert((i, j), crosstalk);
                }
            }
        }

        // Sort by crosstalk value
        problematic_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        let max_crosstalk = problematic_pairs.first().map_or(0.0, |p| p.2);

        CrosstalkAnalysis {
            total_gates: gates.len(),
            problematic_pairs,
            crosstalk_graph,
            max_crosstalk,
        }
    }

    /// Suggest gate reordering to reduce crosstalk
    pub fn suggest_reordering<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<Vec<GateReordering>> {
        let analysis = self.analyze(circuit);
        let mut suggestions = Vec::new();

        // For each problematic pair, suggest moving gates apart
        for (i, j, crosstalk) in analysis.problematic_pairs.iter().take(5) {
            suggestions.push(GateReordering {
                gate1: *i,
                gate2: *j,
                reason: format!("High crosstalk: {crosstalk:.4}"),
                expected_improvement: crosstalk * 0.5, // Estimate
            });
        }

        Ok(suggestions)
    }
}

/// Crosstalk analysis results
#[derive(Debug)]
pub struct CrosstalkAnalysis {
    pub total_gates: usize,
    pub problematic_pairs: Vec<(usize, usize, f64)>, // (gate1, gate2, crosstalk)
    pub crosstalk_graph: HashMap<(usize, usize), f64>,
    pub max_crosstalk: f64,
}

/// Suggested gate reordering
#[derive(Debug)]
pub struct GateReordering {
    pub gate1: usize,
    pub gate2: usize,
    pub reason: String,
    pub expected_improvement: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::Hadamard;

    #[test]
    fn test_crosstalk_model() {
        let model = CrosstalkModel::uniform(4, 0.05);

        // Check single-qubit crosstalk
        assert!(model.single_qubit_crosstalk.get(&(0, 1)).is_some());

        // Check two-qubit crosstalk
        assert!(!model.crosstalk_coefficients.is_empty());
    }

    #[test]
    fn test_gate_crosstalk() {
        let model = CrosstalkModel::uniform(4, 0.05);

        let gate1 = Hadamard { target: QubitId(0) };
        let gate2 = Hadamard { target: QubitId(1) };
        let gate3 = Hadamard { target: QubitId(0) }; // Same qubit

        let crosstalk = model.get_crosstalk(&gate1, &gate2);
        assert!(crosstalk > 0.0 && crosstalk < 1.0);

        let crosstalk_same = model.get_crosstalk(&gate1, &gate3);
        assert_eq!(crosstalk_same, f64::INFINITY); // Cannot parallelize
    }

    #[test]
    fn test_scheduling() {
        let model = CrosstalkModel::uniform(4, 0.05);
        let scheduler = CrosstalkScheduler::new(model);

        let mut circuit = Circuit::<4>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();
        circuit.add_gate(Hadamard { target: QubitId(1) }).unwrap();
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();

        let schedule = scheduler.schedule(&circuit).unwrap();
        assert!(schedule.time_slices.len() >= 2); // H gates might be parallel
    }

    #[test]
    fn test_crosstalk_analysis() {
        let model = CrosstalkModel::uniform(4, 0.05);
        let analyzer = CrosstalkAnalyzer::new(model);

        let mut circuit = Circuit::<4>::new();
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

        let analysis = analyzer.analyze(&circuit);
        assert_eq!(analysis.total_gates, 2);
        assert!(analysis.max_crosstalk > 0.0);
    }
}
