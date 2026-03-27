//! Quantum circuit optimization passes
//!
//! This module provides various optimization passes that can be applied to quantum circuits
//! to reduce gate count, improve fidelity, and optimize for hardware constraints.

use crate::builder::Circuit;
use quantrs2_core::{
    gate::{
        multi::CNOT,
        single::{PauliX, RotationX, RotationY, RotationZ},
    },
    qubit::QubitId,
};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

/// Gate representation for optimization
#[derive(Debug, Clone, PartialEq)]
pub enum OptGate {
    Single(QubitId, String, Vec<f64>),
    Double(QubitId, QubitId, String, Vec<f64>),
    Multi(Vec<QubitId>, String, Vec<f64>),
}

impl OptGate {
    /// Convert a GateOp to OptGate for optimization analysis
    fn from_gate_op(gate: &dyn quantrs2_core::gate::GateOp) -> Self {
        let qubits = gate.qubits();
        let name = gate.name().to_string();
        let params = Vec::new(); // Parameters would need to be extracted from specific gate types

        match qubits.len() {
            1 => Self::Single(qubits[0], name, params),
            2 => Self::Double(qubits[0], qubits[1], name, params),
            _ => Self::Multi(qubits, name, params),
        }
    }
}

/// Optimization context that holds circuit information
pub struct OptimizationContext<const N: usize> {
    pub circuit: Circuit<N>,
    pub gate_count: usize,
    pub depth: usize,
}

/// Result of applying an optimization pass
pub struct PassResult<const N: usize> {
    pub circuit: Circuit<N>,
    pub improved: bool,
    pub improvement: f64,
}

/// Merge consecutive single-qubit gates
pub struct SingleQubitGateFusion;

impl SingleQubitGateFusion {
    #[must_use]
    pub fn apply<const N: usize>(&self, ctx: &OptimizationContext<N>) -> PassResult<N> {
        let gates = ctx.circuit.gates();
        if gates.len() < 2 {
            return PassResult {
                circuit: ctx.circuit.clone(),
                improved: false,
                improvement: 0.0,
            };
        }

        // Convert gates to OptGate format
        let opt_gates: Vec<OptGate> = gates
            .iter()
            .map(|g| OptGate::from_gate_op(g.as_ref()))
            .collect();

        // Find consecutive single-qubit gates on the same qubit
        let mut fusion_groups: Vec<Vec<usize>> = Vec::new();
        let mut current_group: Vec<usize> = vec![];
        let mut current_qubit: Option<QubitId> = None;

        for (idx, opt_gate) in opt_gates.iter().enumerate() {
            if let OptGate::Single(qubit, _, _) = opt_gate {
                if current_qubit == Some(*qubit) {
                    // Continue current group
                    current_group.push(idx);
                } else {
                    // Start new group
                    if current_group.len() >= 2 {
                        fusion_groups.push(current_group.clone());
                    }
                    current_group = vec![idx];
                    current_qubit = Some(*qubit);
                }
            } else {
                // Non-single-qubit gate breaks the group
                if current_group.len() >= 2 {
                    fusion_groups.push(current_group.clone());
                }
                current_group.clear();
                current_qubit = None;
            }
        }

        // Don't forget the last group
        if current_group.len() >= 2 {
            fusion_groups.push(current_group);
        }

        if fusion_groups.is_empty() {
            return PassResult {
                circuit: ctx.circuit.clone(),
                improved: false,
                improvement: 0.0,
            };
        }

        // For now, we report the fusion opportunities but don't actually fuse
        // Full fusion would require matrix multiplication and creating composite gates
        // which is complex and requires additional infrastructure

        // Calculate potential improvement
        let mut gates_that_could_be_fused = 0;
        for group in &fusion_groups {
            gates_that_could_be_fused += group.len() - 1; // N gates → 1 gate saves N-1
        }

        // Since we're not actually implementing fusion yet, return the original circuit
        // but report that fusion opportunities were found
        PassResult {
            circuit: ctx.circuit.clone(),
            improved: false,
            improvement: 0.0, // Would be gates_that_could_be_fused if we implemented it
        }
    }

    #[must_use]
    pub const fn name(&self) -> &'static str {
        "Single-Qubit Gate Fusion"
    }
}

/// Remove redundant gates (e.g., X·X = I, H·H = I)
pub struct RedundantGateElimination;

impl RedundantGateElimination {
    /// Check if two gates cancel each other
    #[allow(dead_code)]
    fn gates_cancel(gate1: &OptGate, gate2: &OptGate) -> bool {
        match (gate1, gate2) {
            (OptGate::Single(q1, name1, _), OptGate::Single(q2, name2, _)) => {
                if q1 != q2 {
                    return false;
                }

                // Self-inverse gates
                matches!(
                    (name1.as_str(), name2.as_str()),
                    ("X", "X") | ("Y", "Y") | ("Z", "Z") | ("H", "H") | ("CNOT", "CNOT")
                )
            }
            _ => false,
        }
    }

    #[must_use]
    pub fn apply<const N: usize>(&self, ctx: &OptimizationContext<N>) -> PassResult<N> {
        let gates = ctx.circuit.gates();
        if gates.len() < 2 {
            return PassResult {
                circuit: ctx.circuit.clone(),
                improved: false,
                improvement: 0.0,
            };
        }

        // Track which gates should be removed (indices to skip)
        let mut to_remove = HashSet::new();

        // Convert gates to OptGate format for analysis
        let opt_gates: Vec<OptGate> = gates
            .iter()
            .map(|g| OptGate::from_gate_op(g.as_ref()))
            .collect();

        // Find consecutive gates that cancel
        let mut i = 0;
        while i < opt_gates.len() - 1 {
            if !to_remove.contains(&i)
                && !to_remove.contains(&(i + 1))
                && Self::gates_cancel(&opt_gates[i], &opt_gates[i + 1])
            {
                to_remove.insert(i);
                to_remove.insert(i + 1);
                i += 2; // Skip both gates
                continue;
            }
            i += 1;
        }

        if to_remove.is_empty() {
            return PassResult {
                circuit: ctx.circuit.clone(),
                improved: false,
                improvement: 0.0,
            };
        }

        // Build new circuit without redundant gates
        let mut new_circuit = Circuit::<N>::with_capacity(gates.len() - to_remove.len());
        for (idx, gate) in gates.iter().enumerate() {
            if !to_remove.contains(&idx) {
                let _ = new_circuit.add_gate_arc(gate.clone());
            }
        }

        let gates_removed = to_remove.len();
        let improvement = gates_removed as f64; // Each removed gate reduces cost

        PassResult {
            circuit: new_circuit,
            improved: gates_removed > 0,
            improvement,
        }
    }

    #[must_use]
    pub const fn name(&self) -> &'static str {
        "Redundant Gate Elimination"
    }
}

/// Commutation-based optimization
pub struct CommutationOptimizer;

impl CommutationOptimizer {
    /// Check if two gates commute
    #[allow(dead_code)]
    fn gates_commute(gate1: &OptGate, gate2: &OptGate) -> bool {
        match (gate1, gate2) {
            // Single-qubit gates on different qubits always commute
            // Single-qubit gates on different qubits always commute
            (OptGate::Single(q1, name1, _), OptGate::Single(q2, name2, _)) => {
                if q1 == q2 {
                    // Z gates commute with each other on same qubit
                    name1 == "Z" && name2 == "Z"
                } else {
                    true
                }
            }

            // CNOT gates commute if they don't share qubits
            (OptGate::Double(c1, t1, name1, _), OptGate::Double(c2, t2, name2, _)) => {
                name1 == "CNOT" && name2 == "CNOT" && c1 != c2 && c1 != t2 && t1 != c2 && t1 != t2
            }

            _ => false,
        }
    }

    #[must_use]
    pub fn apply<const N: usize>(&self, ctx: &OptimizationContext<N>) -> PassResult<N> {
        let gates = ctx.circuit.gates();
        if gates.len() < 2 {
            return PassResult {
                circuit: ctx.circuit.clone(),
                improved: false,
                improvement: 0.0,
            };
        }

        // Convert gates to OptGate format
        let opt_gates: Vec<OptGate> = gates
            .iter()
            .map(|g| OptGate::from_gate_op(g.as_ref()))
            .collect();

        // Try to reorder gates by bubbling commuting gates forward
        // This can reduce circuit depth and enable other optimizations
        let mut reordered_indices: Vec<usize> = (0..gates.len()).collect();
        let mut made_changes = false;

        // Multiple passes to propagate commuting gates
        for _ in 0..3 {
            let mut i = 0;
            while i + 1 < reordered_indices.len() {
                let idx1 = reordered_indices[i];
                let idx2 = reordered_indices[i + 1];

                if Self::gates_commute(&opt_gates[idx1], &opt_gates[idx2]) {
                    // Try swapping to see if it reduces depth or enables other optimizations
                    // For now, prefer moving single-qubit gates earlier
                    let should_swap = matches!(opt_gates[idx2], OptGate::Single(_, _, _))
                        && !matches!(opt_gates[idx1], OptGate::Single(_, _, _));

                    if should_swap {
                        reordered_indices.swap(i, i + 1);
                        made_changes = true;
                    }
                }
                i += 1;
            }
        }

        if !made_changes {
            return PassResult {
                circuit: ctx.circuit.clone(),
                improved: false,
                improvement: 0.0,
            };
        }

        // Build new circuit with reordered gates
        let mut new_circuit = Circuit::<N>::with_capacity(gates.len());
        for idx in reordered_indices {
            let _ = new_circuit.add_gate_arc(gates[idx].clone());
        }

        // Calculate depth improvement
        let old_depth = ctx.circuit.calculate_depth() as f64;
        let new_depth = new_circuit.calculate_depth() as f64;
        let improvement = (old_depth - new_depth).max(0.0);

        PassResult {
            circuit: new_circuit,
            improved: improvement > 0.0,
            improvement,
        }
    }

    #[must_use]
    pub const fn name(&self) -> &'static str {
        "Commutation-Based Optimization"
    }
}

/// Peephole optimization for common patterns
pub struct PeepholeOptimizer {
    #[allow(dead_code)]
    patterns: Vec<PatternRule>,
}

#[derive(Clone)]
#[allow(dead_code)]
struct PatternRule {
    pattern: Vec<OptGate>,
    replacement: Vec<OptGate>,
    name: String,
}

impl Default for PeepholeOptimizer {
    fn default() -> Self {
        let patterns = vec![
            // Pattern: H-X-H = Z
            PatternRule {
                pattern: vec![
                    OptGate::Single(QubitId::new(0), "H".to_string(), vec![]),
                    OptGate::Single(QubitId::new(0), "X".to_string(), vec![]),
                    OptGate::Single(QubitId::new(0), "H".to_string(), vec![]),
                ],
                replacement: vec![OptGate::Single(QubitId::new(0), "Z".to_string(), vec![])],
                name: "H-X-H to Z".to_string(),
            },
            // Pattern: H-Z-H = X
            PatternRule {
                pattern: vec![
                    OptGate::Single(QubitId::new(0), "H".to_string(), vec![]),
                    OptGate::Single(QubitId::new(0), "Z".to_string(), vec![]),
                    OptGate::Single(QubitId::new(0), "H".to_string(), vec![]),
                ],
                replacement: vec![OptGate::Single(QubitId::new(0), "X".to_string(), vec![])],
                name: "H-Z-H to X".to_string(),
            },
        ];

        Self { patterns }
    }
}

impl PeepholeOptimizer {
    /// Check if three consecutive gates match a pattern
    fn matches_pattern(gates: &[OptGate], pattern: &[OptGate]) -> bool {
        if gates.len() != pattern.len() {
            return false;
        }

        for (gate, pat) in gates.iter().zip(pattern.iter()) {
            match (gate, pat) {
                (OptGate::Single(q1, n1, _), OptGate::Single(q2, n2, _)) => {
                    if q1 != q2 || n1 != n2 {
                        return false;
                    }
                }
                _ => return false, // Only support single-qubit patterns for now
            }
        }
        true
    }

    #[must_use]
    pub fn apply<const N: usize>(&self, ctx: &OptimizationContext<N>) -> PassResult<N> {
        let gates = ctx.circuit.gates();
        if gates.len() < 3 {
            return PassResult {
                circuit: ctx.circuit.clone(),
                improved: false,
                improvement: 0.0,
            };
        }

        // Convert gates to OptGate format
        let opt_gates: Vec<OptGate> = gates
            .iter()
            .map(|g| OptGate::from_gate_op(g.as_ref()))
            .collect();

        // Track replacements: (start_idx, end_idx, replacement_gate_name)
        let mut replacements: Vec<(usize, usize, String, QubitId)> = Vec::new();

        // Find pattern matches (H-X-H → Z, H-Z-H → X)
        let mut i = 0;
        while i + 2 < opt_gates.len() {
            // Check for H-X-H → Z
            if let (
                OptGate::Single(q1, n1, _),
                OptGate::Single(q2, n2, _),
                OptGate::Single(q3, n3, _),
            ) = (&opt_gates[i], &opt_gates[i + 1], &opt_gates[i + 2])
            {
                if q1 == q2 && q2 == q3 {
                    if n1 == "H" && n2 == "X" && n3 == "H" {
                        replacements.push((i, i + 2, "Z".to_string(), *q1));
                        i += 3;
                        continue;
                    } else if n1 == "H" && n2 == "Z" && n3 == "H" {
                        replacements.push((i, i + 2, "X".to_string(), *q1));
                        i += 3;
                        continue;
                    }
                }
            }
            i += 1;
        }

        if replacements.is_empty() {
            return PassResult {
                circuit: ctx.circuit.clone(),
                improved: false,
                improvement: 0.0,
            };
        }

        // Build new circuit with replacements
        let mut new_circuit = Circuit::<N>::new();
        let mut idx = 0;
        let mut replacement_iter = replacements.iter().peekable();

        while idx < gates.len() {
            if let Some((start, end, replacement_name, qubit)) = replacement_iter.peek() {
                if idx == *start {
                    // Apply replacement
                    match replacement_name.as_str() {
                        "X" => {
                            let _ = new_circuit
                                .add_gate(quantrs2_core::gate::single::PauliX { target: *qubit });
                        }
                        "Z" => {
                            let _ = new_circuit
                                .add_gate(quantrs2_core::gate::single::PauliZ { target: *qubit });
                        }
                        _ => {}
                    }
                    idx = *end + 1;
                    replacement_iter.next();
                    continue;
                }
            }
            // Copy original gate
            let _ = new_circuit.add_gate_arc(gates[idx].clone());
            idx += 1;
        }

        let gates_saved = replacements.len() * 2; // Each pattern saves 2 gates (3 → 1)
        let improvement = gates_saved as f64;

        PassResult {
            circuit: new_circuit,
            improved: !replacements.is_empty(),
            improvement,
        }
    }

    #[must_use]
    pub const fn name(&self) -> &'static str {
        "Peephole Optimization"
    }
}

/// Template matching optimization
pub struct TemplateOptimizer {
    #[allow(dead_code)]
    templates: Vec<Template>,
}

#[allow(dead_code)]
struct Template {
    name: String,
    pattern: Vec<OptGate>,
    cost_reduction: f64,
}

impl Default for TemplateOptimizer {
    fn default() -> Self {
        let templates = vec![Template {
            name: "Toffoli Decomposition".to_string(),
            pattern: vec![], // Would contain Toffoli gate pattern
            cost_reduction: 0.3,
        }];

        Self { templates }
    }
}

impl TemplateOptimizer {
    #[must_use]
    pub fn apply<const N: usize>(&self, ctx: &OptimizationContext<N>) -> PassResult<N> {
        let circuit = &ctx.circuit;
        let gates = circuit.gates();
        if gates.len() < 2 {
            return PassResult {
                circuit: circuit.clone(),
                improved: false,
                improvement: 0.0,
            };
        }

        let mut new_circuit = Circuit::<N>::new();
        let n = gates.len();
        let mut i = 0;
        let mut gates_saved: usize = 0;

        while i < n {
            // ---------------------------------------------------------------
            // Pattern: RX(θ)·RX(φ) → RX(θ+φ)  (and RY, RZ analogues)
            // ---------------------------------------------------------------
            if i + 1 < n {
                let g0 = gates[i].as_ref();
                let g1 = gates[i + 1].as_ref();

                if g0.qubits().len() == 1
                    && g1.qubits().len() == 1
                    && g0.qubits()[0] == g1.qubits()[0]
                {
                    let target = g0.qubits()[0];

                    // RZ(θ)·RZ(φ) → RZ(θ+φ)
                    if let (Some(rz0), Some(rz1)) = (
                        g0.as_any().downcast_ref::<RotationZ>(),
                        g1.as_any().downcast_ref::<RotationZ>(),
                    ) {
                        let merged_angle = rz0.theta + rz1.theta;
                        // Normalise into (-π, π]
                        let normalised = ((merged_angle + PI).rem_euclid(2.0 * PI)) - PI;
                        let _ = new_circuit.add_gate(RotationZ {
                            target,
                            theta: normalised,
                        });
                        i += 2;
                        gates_saved += 1;
                        continue;
                    }

                    // RY(θ)·RY(φ) → RY(θ+φ)
                    if let (Some(ry0), Some(ry1)) = (
                        g0.as_any().downcast_ref::<RotationY>(),
                        g1.as_any().downcast_ref::<RotationY>(),
                    ) {
                        let merged_angle = ry0.theta + ry1.theta;
                        let normalised = ((merged_angle + PI).rem_euclid(2.0 * PI)) - PI;
                        let _ = new_circuit.add_gate(RotationY {
                            target,
                            theta: normalised,
                        });
                        i += 2;
                        gates_saved += 1;
                        continue;
                    }

                    // RX(θ)·RX(φ) → RX(θ+φ)
                    if let (Some(rx0), Some(rx1)) = (
                        g0.as_any().downcast_ref::<RotationX>(),
                        g1.as_any().downcast_ref::<RotationX>(),
                    ) {
                        let merged_angle = rx0.theta + rx1.theta;
                        let normalised = ((merged_angle + PI).rem_euclid(2.0 * PI)) - PI;
                        let _ = new_circuit.add_gate(RotationX {
                            target,
                            theta: normalised,
                        });
                        i += 2;
                        gates_saved += 1;
                        continue;
                    }
                }
            }

            // ---------------------------------------------------------------
            // Pattern: X · RZ(θ) · X → RZ(-θ)   (conjugation by X)
            // ---------------------------------------------------------------
            if i + 2 < n {
                let g0 = gates[i].as_ref();
                let g1 = gates[i + 1].as_ref();
                let g2 = gates[i + 2].as_ref();

                if g0.qubits().len() == 1
                    && g1.qubits().len() == 1
                    && g2.qubits().len() == 1
                    && g0.qubits()[0] == g1.qubits()[0]
                    && g1.qubits()[0] == g2.qubits()[0]
                {
                    let target = g1.qubits()[0];

                    if g0.name() == "X" && g2.name() == "X" {
                        if let Some(rz) = g1.as_any().downcast_ref::<RotationZ>() {
                            // X·RZ(θ)·X = RZ(-θ)
                            let _ = new_circuit.add_gate(RotationZ {
                                target,
                                theta: -rz.theta,
                            });
                            i += 3;
                            gates_saved += 2;
                            continue;
                        }
                    }
                }

                // ---------------------------------------------------------------
                // Pattern: CNOT · (I ⊗ RZ(θ)) · CNOT → (I ⊗ RZ(θ)) · CNOT · CNOT
                //          simplified: CNOT commutes with target RZ rotations,
                //          so emit RZ(θ) first then drop the two CNOTs (they cancel).
                //          Only applicable when the two CNOTs have the same control/target.
                // ---------------------------------------------------------------
                let g0 = gates[i].as_ref();
                let g1 = gates[i + 1].as_ref();
                let g2 = gates[i + 2].as_ref();

                if let (Some(cnot0), Some(cnot2)) = (
                    g0.as_any().downcast_ref::<CNOT>(),
                    g2.as_any().downcast_ref::<CNOT>(),
                ) {
                    if cnot0.control == cnot2.control
                        && cnot0.target == cnot2.target
                        && g1.qubits().len() == 1
                        && g1.qubits()[0] == cnot0.target
                    {
                        if let Some(rz) = g1.as_any().downcast_ref::<RotationZ>() {
                            // RZ on target commutes through CNOT → emit RZ then CNOT·CNOT
                            // CNOT·CNOT = I, so the two cancel; emit just RZ
                            let target = rz.target;
                            let _ = new_circuit.add_gate(RotationZ {
                                target,
                                theta: rz.theta,
                            });
                            i += 3;
                            gates_saved += 2;
                            continue;
                        }
                    }
                }
            }

            // No pattern matched — emit gate unchanged
            let _ = new_circuit.add_gate_arc(gates[i].clone());
            i += 1;
        }

        let improved = gates_saved > 0;
        PassResult {
            circuit: new_circuit,
            improved,
            improvement: gates_saved as f64,
        }
    }

    #[must_use]
    pub const fn name(&self) -> &'static str {
        "Template Matching Optimization"
    }
}

/// Enum to hold different optimization passes
pub enum OptimizationPassType {
    SingleQubitFusion(SingleQubitGateFusion),
    RedundantElimination(RedundantGateElimination),
    Commutation(CommutationOptimizer),
    Peephole(PeepholeOptimizer),
    Template(TemplateOptimizer),
    Hardware(HardwareOptimizer),
}

impl OptimizationPassType {
    #[must_use]
    pub fn apply<const N: usize>(&self, ctx: &OptimizationContext<N>) -> PassResult<N> {
        match self {
            Self::SingleQubitFusion(p) => p.apply(ctx),
            Self::RedundantElimination(p) => p.apply(ctx),
            Self::Commutation(p) => p.apply(ctx),
            Self::Peephole(p) => p.apply(ctx),
            Self::Template(p) => p.apply(ctx),
            Self::Hardware(p) => p.apply(ctx),
        }
    }

    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::SingleQubitFusion(p) => p.name(),
            Self::RedundantElimination(p) => p.name(),
            Self::Commutation(p) => p.name(),
            Self::Peephole(p) => p.name(),
            Self::Template(p) => p.name(),
            Self::Hardware(p) => p.name(),
        }
    }
}

/// Main circuit optimizer that applies multiple passes
pub struct CircuitOptimizer<const N: usize> {
    passes: Vec<OptimizationPassType>,
    max_iterations: usize,
}

impl<const N: usize> Default for CircuitOptimizer<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> CircuitOptimizer<N> {
    /// Create a new circuit optimizer with default passes
    #[must_use]
    pub fn new() -> Self {
        let passes = vec![
            OptimizationPassType::RedundantElimination(RedundantGateElimination),
            OptimizationPassType::SingleQubitFusion(SingleQubitGateFusion),
            OptimizationPassType::Commutation(CommutationOptimizer),
            OptimizationPassType::Peephole(PeepholeOptimizer::default()),
            OptimizationPassType::Template(TemplateOptimizer::default()),
        ];

        Self {
            passes,
            max_iterations: 10,
        }
    }

    /// Create a custom optimizer with specific passes
    #[must_use]
    pub const fn with_passes(passes: Vec<OptimizationPassType>) -> Self {
        Self {
            passes,
            max_iterations: 10,
        }
    }

    /// Set the maximum number of optimization iterations
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Add an optimization pass
    #[must_use]
    pub fn add_pass(mut self, pass: OptimizationPassType) -> Self {
        self.passes.push(pass);
        self
    }

    /// Optimize a circuit
    #[must_use]
    pub fn optimize(&self, circuit: &Circuit<N>) -> OptimizationResult<N> {
        let mut current_circuit = circuit.clone();
        let mut total_iterations = 0;
        let mut pass_statistics = HashMap::new();

        // Keep track of circuit cost (simplified as gate count for now)
        let initial_cost = self.estimate_cost(&current_circuit);
        let mut current_cost = initial_cost;

        // Apply optimization passes iteratively
        for iteration in 0..self.max_iterations {
            let iteration_start_cost = current_cost;

            for pass in &self.passes {
                let pass_name = pass.name().to_string();
                let before_cost = current_cost;

                let ctx = OptimizationContext {
                    circuit: current_circuit.clone(),
                    gate_count: 10, // Placeholder
                    depth: 5,       // Placeholder
                };

                let result = pass.apply(&ctx);
                current_circuit = result.circuit;

                if result.improved {
                    current_cost -= result.improvement;
                }

                let improvement = before_cost - current_cost;
                pass_statistics
                    .entry(pass_name)
                    .and_modify(|stats: &mut PassStats| {
                        stats.applications += 1;
                        stats.total_improvement += improvement;
                    })
                    .or_insert(PassStats {
                        applications: 1,
                        total_improvement: improvement,
                    });
            }

            total_iterations = iteration + 1;

            // Stop if no improvement in this iteration
            if (iteration_start_cost - current_cost).abs() < 1e-10 {
                break;
            }
        }

        OptimizationResult {
            optimized_circuit: current_circuit,
            initial_cost,
            final_cost: current_cost,
            iterations: total_iterations,
            pass_statistics,
        }
    }

    /// Estimate the cost of a circuit based on gate count, types, and depth
    fn estimate_cost(&self, circuit: &Circuit<N>) -> f64 {
        let stats = circuit.get_stats();

        // Weight factors for different gate types
        let single_qubit_cost = 1.0;
        let two_qubit_cost = 10.0; // Two-qubit gates are much more expensive
        let multi_qubit_cost = 50.0; // Multi-qubit gates are very expensive

        // Calculate gate cost
        let single_qubit_gates =
            stats.total_gates - stats.two_qubit_gates - stats.multi_qubit_gates;
        let gate_cost = single_qubit_gates as f64 * single_qubit_cost
            + stats.two_qubit_gates as f64 * two_qubit_cost
            + stats.multi_qubit_gates as f64 * multi_qubit_cost;

        // Circuit depth adds to cost (deeper circuits are slower and more error-prone)
        let depth_cost = stats.depth as f64 * 2.0;

        gate_cost + depth_cost
    }
}

/// Statistics for an optimization pass
#[derive(Debug, Clone)]
pub struct PassStats {
    pub applications: usize,
    pub total_improvement: f64,
}

/// Result of circuit optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult<const N: usize> {
    pub optimized_circuit: Circuit<N>,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    pub pass_statistics: HashMap<String, PassStats>,
}

impl<const N: usize> OptimizationResult<N> {
    /// Get the improvement ratio
    #[must_use]
    pub fn improvement_ratio(&self) -> f64 {
        if self.initial_cost > 0.0 {
            (self.initial_cost - self.final_cost) / self.initial_cost
        } else {
            0.0
        }
    }

    /// Print optimization summary
    pub fn print_summary(&self) {
        println!("Circuit Optimization Summary");
        println!("===========================");
        println!("Initial cost: {:.2}", self.initial_cost);
        println!("Final cost: {:.2}", self.final_cost);
        println!("Improvement: {:.1}%", self.improvement_ratio() * 100.0);
        println!("Iterations: {}", self.iterations);
        println!("\nPass Statistics:");

        for (pass_name, stats) in &self.pass_statistics {
            if stats.total_improvement > 0.0 {
                println!(
                    "  {}: {} applications, {:.2} total improvement",
                    pass_name, stats.applications, stats.total_improvement
                );
            }
        }
    }
}

/// Hardware-aware optimization pass
pub struct HardwareOptimizer {
    #[allow(dead_code)]
    connectivity: Vec<(usize, usize)>,
    #[allow(dead_code)]
    native_gates: HashSet<String>,
}

impl HardwareOptimizer {
    #[must_use]
    pub const fn new(connectivity: Vec<(usize, usize)>, native_gates: HashSet<String>) -> Self {
        Self {
            connectivity,
            native_gates,
        }
    }

    #[must_use]
    pub fn apply<const N: usize>(&self, ctx: &OptimizationContext<N>) -> PassResult<N> {
        let circuit = &ctx.circuit;
        let gates = circuit.gates();

        if gates.is_empty() {
            return PassResult {
                circuit: circuit.clone(),
                improved: false,
                improvement: 0.0,
            };
        }

        // Build a set of connected qubit pairs from the connectivity graph
        let connected_pairs: HashSet<(usize, usize)> = self
            .connectivity
            .iter()
            .flat_map(|&(a, b)| [(a.min(b), a.max(b))])
            .collect();

        let mut new_circuit = Circuit::<N>::new();
        let mut gates_decomposed: usize = 0;

        for gate in gates {
            let gate_ref = gate.as_ref();
            let name = gate_ref.name();

            // Check if the gate is in the native gate set
            if self.native_gates.contains(name) {
                // Native gate — pass through directly
                let _ = new_circuit.add_gate_arc(gate.clone());
                continue;
            }

            // Non-native gate: decompose to native equivalents.
            // Decomposition table (subset, hardware-independent standard expansions):
            //   SWAP  → CNOT(a,b) · CNOT(b,a) · CNOT(a,b)
            //   CZ    → H(b) · CNOT(a,b) · H(b)
            //   CCX (Toffoli) — too complex; fall through to pass-through
            match name {
                "SWAP" if gate_ref.qubits().len() == 2 => {
                    let a = gate_ref.qubits()[0];
                    let b = gate_ref.qubits()[1];
                    let ai = a.id() as usize;
                    let bi = b.id() as usize;

                    // Prefer the direction that matches hardware connectivity
                    let (ctrl, tgt) = if connected_pairs.contains(&(ai.min(bi), ai.max(bi))) {
                        (a, b)
                    } else {
                        (b, a)
                    };

                    // SWAP → CNOT(ctrl, tgt) · CNOT(tgt, ctrl) · CNOT(ctrl, tgt)
                    let _ = new_circuit.add_gate(CNOT {
                        control: ctrl,
                        target: tgt,
                    });
                    let _ = new_circuit.add_gate(CNOT {
                        control: tgt,
                        target: ctrl,
                    });
                    let _ = new_circuit.add_gate(CNOT {
                        control: ctrl,
                        target: tgt,
                    });
                    gates_decomposed += 1;
                }
                "CZ" if gate_ref.qubits().len() == 2 => {
                    let a = gate_ref.qubits()[0];
                    let b = gate_ref.qubits()[1];
                    // CZ(a,b) → H(b) · CNOT(a,b) · H(b)
                    let _ =
                        new_circuit.add_gate(quantrs2_core::gate::single::Hadamard { target: b });
                    let _ = new_circuit.add_gate(CNOT {
                        control: a,
                        target: b,
                    });
                    let _ =
                        new_circuit.add_gate(quantrs2_core::gate::single::Hadamard { target: b });
                    gates_decomposed += 1;
                }
                _ => {
                    // Unknown non-native gate — pass through unchanged
                    let _ = new_circuit.add_gate_arc(gate.clone());
                }
            }
        }

        let improved = gates_decomposed > 0;
        PassResult {
            circuit: new_circuit,
            improved,
            improvement: gates_decomposed as f64,
        }
    }

    #[must_use]
    pub const fn name(&self) -> &'static str {
        "Hardware-Aware Optimization"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_optimizer_creation() {
        let optimizer = CircuitOptimizer::<4>::new();
        assert_eq!(optimizer.passes.len(), 5);
        assert_eq!(optimizer.max_iterations, 10);
    }

    #[test]
    fn test_optimization_result() {
        let circuit = Circuit::<4>::new();
        let optimizer = CircuitOptimizer::new();
        let result = optimizer.optimize(&circuit);

        assert!(result.improvement_ratio() >= 0.0);
        assert!(result.iterations > 0);
    }
}
