//! Advanced optimization passes: decomposition, cost-based, and two-qubit optimization.

use crate::optimization::cost_model::CostModel;
use crate::optimization::gate_properties::get_gate_properties;
use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use quantrs2_core::gate::{
    multi,
    single::{self},
    GateOp,
};
use quantrs2_core::qubit::QubitId;
use std::collections::{HashMap, HashSet};

use super::rewriting_passes::parallelize_gates;
use super::OptimizationPass;

/// Decomposition optimization - chooses optimal decompositions based on hardware
pub struct DecompositionOptimization {
    target_gate_set: HashSet<String>,
    prefer_native: bool,
}

impl DecompositionOptimization {
    #[must_use]
    pub const fn new(target_gate_set: HashSet<String>, prefer_native: bool) -> Self {
        Self {
            target_gate_set,
            prefer_native,
        }
    }

    #[must_use]
    pub fn for_hardware(hardware: &str) -> Self {
        let target_gate_set = match hardware {
            "ibm" => vec!["X", "Y", "Z", "H", "S", "T", "RZ", "CNOT", "CZ"]
                .into_iter()
                .map(std::string::ToString::to_string)
                .collect(),
            "google" => vec!["X", "Y", "Z", "H", "RZ", "CZ", "SQRT_X"]
                .into_iter()
                .map(std::string::ToString::to_string)
                .collect(),
            _ => vec!["X", "Y", "Z", "H", "S", "T", "RZ", "RX", "RY", "CNOT"]
                .into_iter()
                .map(std::string::ToString::to_string)
                .collect(),
        };

        Self {
            target_gate_set,
            prefer_native: true,
        }
    }
}

impl OptimizationPass for DecompositionOptimization {
    fn name(&self) -> &'static str {
        "Decomposition Optimization"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut optimized_gates = Vec::with_capacity(gates.len() * 2);

        for gate in gates {
            let gate_name = gate.name();
            let gate_qubits = gate.qubits();

            // Check if this gate should be decomposed based on target gate set
            if self.should_decompose(&gate, cost_model) {
                // Decompose complex gates into simpler ones
                match gate_name {
                    "Toffoli" => {
                        if gate_qubits.len() == 3 {
                            // Decompose Toffoli into CNOT and T gates
                            self.decompose_toffoli(&gate_qubits, &mut optimized_gates)?;
                        } else {
                            optimized_gates.push(gate);
                        }
                    }
                    "Fredkin" | "CSWAP" => {
                        if gate_qubits.len() == 3 {
                            // Decompose Fredkin into CNOT gates
                            self.decompose_fredkin(&gate_qubits, &mut optimized_gates)?;
                        } else {
                            optimized_gates.push(gate);
                        }
                    }
                    "SWAP" => {
                        if self.target_gate_set.contains("CNOT") && gate_qubits.len() == 2 {
                            // Decompose SWAP into 3 CNOTs
                            self.decompose_swap(&gate_qubits, &mut optimized_gates)?;
                        } else {
                            optimized_gates.push(gate);
                        }
                    }
                    "CRX" | "CRY" | "CRZ" => {
                        // Decompose controlled rotations if not in target set
                        if !self.target_gate_set.contains(gate_name) && gate_qubits.len() == 2 {
                            self.decompose_controlled_rotation(&gate, &mut optimized_gates)?;
                        } else {
                            optimized_gates.push(gate);
                        }
                    }
                    _ => {
                        // Keep gates that don't need decomposition
                        optimized_gates.push(gate);
                    }
                }
            } else {
                optimized_gates.push(gate);
            }
        }

        Ok(optimized_gates)
    }
}

impl DecompositionOptimization {
    fn should_decompose(&self, gate: &Box<dyn GateOp>, _cost_model: &dyn CostModel) -> bool {
        let gate_name = gate.name();

        // Always decompose if gate is not in target set
        if self.target_gate_set.contains(gate_name) {
            false
        } else {
            // Only decompose gates we know how to decompose
            matches!(
                gate_name,
                "Toffoli" | "Fredkin" | "CSWAP" | "SWAP" | "CRX" | "CRY" | "CRZ"
            )
        }
    }

    fn decompose_toffoli(
        &self,
        qubits: &[QubitId],
        gates: &mut Vec<Box<dyn GateOp>>,
    ) -> QuantRS2Result<()> {
        if qubits.len() != 3 {
            return Err(QuantRS2Error::InvalidInput(
                "Toffoli gate requires exactly 3 qubits".to_string(),
            ));
        }

        let c1 = qubits[0];
        let c2 = qubits[1];
        let target = qubits[2];

        // Standard Toffoli decomposition using CNOT and T gates
        use quantrs2_core::gate::{
            multi::CNOT,
            single::{Hadamard, TDagger, T},
        };

        gates.push(Box::new(Hadamard { target }));
        gates.push(Box::new(CNOT {
            control: c2,
            target,
        }));
        gates.push(Box::new(TDagger { target }));
        gates.push(Box::new(CNOT {
            control: c1,
            target,
        }));
        gates.push(Box::new(T { target }));
        gates.push(Box::new(CNOT {
            control: c2,
            target,
        }));
        gates.push(Box::new(TDagger { target }));
        gates.push(Box::new(CNOT {
            control: c1,
            target,
        }));
        gates.push(Box::new(T { target: c2 }));
        gates.push(Box::new(T { target }));
        gates.push(Box::new(CNOT {
            control: c1,
            target: c2,
        }));
        gates.push(Box::new(Hadamard { target }));
        gates.push(Box::new(T { target: c1 }));
        gates.push(Box::new(TDagger { target: c2 }));
        gates.push(Box::new(CNOT {
            control: c1,
            target: c2,
        }));

        Ok(())
    }

    fn decompose_fredkin(
        &self,
        qubits: &[QubitId],
        gates: &mut Vec<Box<dyn GateOp>>,
    ) -> QuantRS2Result<()> {
        if qubits.len() != 3 {
            return Err(QuantRS2Error::InvalidInput(
                "Fredkin gate requires exactly 3 qubits".to_string(),
            ));
        }

        let control = qubits[0];
        let target1 = qubits[1];
        let target2 = qubits[2];

        // Fredkin decomposition using CNOT gates
        use quantrs2_core::gate::multi::CNOT;

        gates.push(Box::new(CNOT {
            control: target2,
            target: target1,
        }));
        gates.push(Box::new(CNOT {
            control,
            target: target1,
        }));
        gates.push(Box::new(CNOT {
            control: target1,
            target: target2,
        }));
        gates.push(Box::new(CNOT {
            control,
            target: target1,
        }));
        gates.push(Box::new(CNOT {
            control: target2,
            target: target1,
        }));

        Ok(())
    }

    fn decompose_swap(
        &self,
        qubits: &[QubitId],
        gates: &mut Vec<Box<dyn GateOp>>,
    ) -> QuantRS2Result<()> {
        if qubits.len() != 2 {
            return Err(QuantRS2Error::InvalidInput(
                "SWAP gate requires exactly 2 qubits".to_string(),
            ));
        }

        let q1 = qubits[0];
        let q2 = qubits[1];

        // SWAP decomposition using 3 CNOT gates
        use quantrs2_core::gate::multi::CNOT;

        gates.push(Box::new(CNOT {
            control: q1,
            target: q2,
        }));
        gates.push(Box::new(CNOT {
            control: q2,
            target: q1,
        }));
        gates.push(Box::new(CNOT {
            control: q1,
            target: q2,
        }));

        Ok(())
    }

    fn decompose_controlled_rotation(
        &self,
        gate: &Box<dyn GateOp>,
        gates: &mut Vec<Box<dyn GateOp>>,
    ) -> QuantRS2Result<()> {
        let qubits = gate.qubits();
        if qubits.len() != 2 {
            return Err(QuantRS2Error::InvalidInput(
                "Controlled rotation requires exactly 2 qubits".to_string(),
            ));
        }

        let control = qubits[0];
        let target = qubits[1];

        use quantrs2_core::gate::{
            multi::CNOT,
            single::{RotationX, RotationY, RotationZ},
        };

        match gate.name() {
            "CRX" => {
                gates.push(Box::new(RotationX {
                    target,
                    theta: std::f64::consts::PI / 4.0,
                }));
                gates.push(Box::new(CNOT { control, target }));
                gates.push(Box::new(RotationX {
                    target,
                    theta: -std::f64::consts::PI / 4.0,
                }));
                gates.push(Box::new(CNOT { control, target }));
            }
            "CRY" => {
                gates.push(Box::new(RotationY {
                    target,
                    theta: std::f64::consts::PI / 4.0,
                }));
                gates.push(Box::new(CNOT { control, target }));
                gates.push(Box::new(RotationY {
                    target,
                    theta: -std::f64::consts::PI / 4.0,
                }));
                gates.push(Box::new(CNOT { control, target }));
            }
            "CRZ" => {
                gates.push(Box::new(RotationZ {
                    target,
                    theta: std::f64::consts::PI / 4.0,
                }));
                gates.push(Box::new(CNOT { control, target }));
                gates.push(Box::new(RotationZ {
                    target,
                    theta: -std::f64::consts::PI / 4.0,
                }));
                gates.push(Box::new(CNOT { control, target }));
            }
            _ => {
                return Err(QuantRS2Error::UnsupportedOperation(format!(
                    "Unknown controlled rotation gate: {}",
                    gate.name()
                )));
            }
        }

        Ok(())
    }
}

/// Cost-based optimization - minimizes gate count, depth, or error
pub struct CostBasedOptimization {
    optimization_target: CostTarget,
    max_iterations: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum CostTarget {
    GateCount,
    CircuitDepth,
    TotalError,
    ExecutionTime,
    Balanced,
}

impl CostBasedOptimization {
    #[must_use]
    pub const fn new(target: CostTarget, max_iterations: usize) -> Self {
        Self {
            optimization_target: target,
            max_iterations,
        }
    }
}

impl OptimizationPass for CostBasedOptimization {
    fn name(&self) -> &'static str {
        "Cost-Based Optimization"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut best_gates = gates.clone();
        let mut best_cost = self.calculate_cost(&best_gates, cost_model);

        for iteration in 0..self.max_iterations {
            let candidate_gates = self.generate_candidate_solution(&best_gates, iteration)?;
            let candidate_cost = self.calculate_cost(&candidate_gates, cost_model);

            if candidate_cost < best_cost {
                best_gates = candidate_gates;
                best_cost = candidate_cost;
            }
        }

        Ok(best_gates)
    }
}

impl CostBasedOptimization {
    fn calculate_cost(&self, gates: &[Box<dyn GateOp>], cost_model: &dyn CostModel) -> f64 {
        match self.optimization_target {
            CostTarget::GateCount => gates.len() as f64,
            CostTarget::CircuitDepth => self.calculate_depth(gates) as f64,
            CostTarget::TotalError => self.calculate_total_error(gates),
            CostTarget::ExecutionTime => self.calculate_execution_time(gates),
            CostTarget::Balanced => {
                let gate_count = gates.len() as f64;
                let depth = self.calculate_depth(gates) as f64;
                let error = self.calculate_total_error(gates);
                let time = self.calculate_execution_time(gates);

                (0.2 * error).mul_add(1000.0, 0.3f64.mul_add(gate_count, 0.3 * depth))
                    + 0.2 * time / 1000.0
            }
        }
    }

    fn generate_candidate_solution(
        &self,
        gates: &[Box<dyn GateOp>],
        iteration: usize,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut candidate = gates.to_vec();

        match self.optimization_target {
            CostTarget::GateCount => {
                self.cancel_inverse_gates(&mut candidate);
            }
            CostTarget::CircuitDepth => {
                self.parallelize_gates_impl(&mut candidate);
            }
            CostTarget::TotalError => {
                self.reduce_error_gates(&candidate)?;
            }
            CostTarget::ExecutionTime => {
                self.optimize_for_speed(&candidate)?;
            }
            CostTarget::Balanced => match iteration % 4 {
                0 => self.cancel_inverse_gates(&mut candidate),
                1 => self.parallelize_gates_impl(&mut candidate),
                2 => self.reduce_error_gates(&candidate)?,
                3 => self.optimize_for_speed(&candidate)?,
                _ => unreachable!(),
            },
        }

        Ok(candidate)
    }

    fn calculate_depth(&self, gates: &[Box<dyn GateOp>]) -> usize {
        let mut qubit_depths = std::collections::HashMap::new();
        let mut max_depth = 0;

        for gate in gates {
            let gate_qubits = gate.qubits();
            let gate_start_depth = gate_qubits
                .iter()
                .map(|q| qubit_depths.get(&q.id()).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);

            let gate_end_depth = gate_start_depth + 1;

            for qubit in gate_qubits {
                qubit_depths.insert(qubit.id(), gate_end_depth);
            }

            max_depth = max_depth.max(gate_end_depth);
        }

        max_depth
    }

    fn calculate_total_error(&self, gates: &[Box<dyn GateOp>]) -> f64 {
        gates
            .iter()
            .map(|gate| self.estimate_gate_error(gate.name()))
            .sum()
    }

    fn calculate_execution_time(&self, gates: &[Box<dyn GateOp>]) -> f64 {
        gates
            .iter()
            .map(|gate| self.estimate_gate_time(gate.name()))
            .sum()
    }

    fn estimate_gate_error(&self, gate_name: &str) -> f64 {
        match gate_name {
            "H" | "X" | "Y" | "Z" | "S" | "T" => 0.0001,
            "RX" | "RY" | "RZ" => 0.0005,
            "CNOT" | "CX" | "CZ" | "CY" => 0.01,
            "SWAP" | "CRX" | "CRY" | "CRZ" => 0.015,
            "Toffoli" | "Fredkin" => 0.05,
            _ => 0.01,
        }
    }

    fn estimate_gate_time(&self, gate_name: &str) -> f64 {
        match gate_name {
            "H" | "X" | "Y" | "Z" | "S" | "T" | "RX" | "RY" | "RZ" => 50.0,
            "CNOT" | "CX" | "CZ" | "CY" | "SWAP" | "CRX" | "CRY" | "CRZ" => 200.0,
            "Toffoli" | "Fredkin" => 500.0,
            _ => 100.0,
        }
    }

    fn cancel_inverse_gates(&self, gates: &mut Vec<Box<dyn GateOp>>) {
        let mut i = 0;
        while i + 1 < gates.len() {
            if self.are_inverse_gates(&gates[i], &gates[i + 1]) {
                gates.remove(i + 1);
                gates.remove(i);
                i = i.saturating_sub(1);
            } else {
                i += 1;
            }
        }
    }

    fn are_inverse_gates(&self, gate1: &Box<dyn GateOp>, gate2: &Box<dyn GateOp>) -> bool {
        if gate1.qubits() != gate2.qubits() {
            return false;
        }

        match (gate1.name(), gate2.name()) {
            ("H", "H") | ("X", "X") | ("Y", "Y") | ("Z", "Z") => true,
            ("S", "SDG") | ("SDG", "S") => true,
            ("T", "TDG") | ("TDG", "T") => true,
            ("CNOT", "CNOT") | ("CX", "CX") => true,
            _ => false,
        }
    }

    fn parallelize_gates_impl(&self, gates: &mut Vec<Box<dyn GateOp>>) {
        let owned = std::mem::take(gates);
        *gates = parallelize_gates(owned);
    }

    fn reduce_error_gates(&self, gates: &[Box<dyn GateOp>]) -> QuantRS2Result<()> {
        for i in 0..gates.len() {
            if gates[i].name() == "Toffoli" {
                // Could decompose Toffoli to reduce error in some cases
            }
        }
        Ok(())
    }

    fn optimize_for_speed(&self, gates: &[Box<dyn GateOp>]) -> QuantRS2Result<()> {
        for i in 0..gates.len() {
            if gates[i].name() == "Toffoli" {
                // Could use a faster Toffoli implementation if available
            }
        }
        Ok(())
    }
}

/// Two-qubit gate optimization
pub struct TwoQubitOptimization {
    use_kak_decomposition: bool,
    optimize_cnots: bool,
}

impl TwoQubitOptimization {
    #[must_use]
    pub const fn new(use_kak_decomposition: bool, optimize_cnots: bool) -> Self {
        Self {
            use_kak_decomposition,
            optimize_cnots,
        }
    }
}

impl TwoQubitOptimization {
    /// Qubit ids touched by a gate as a `HashSet<u32>`.
    fn qubit_ids(gate: &dyn GateOp) -> HashSet<u32> {
        gate.qubits().into_iter().map(|q| q.id()).collect()
    }

    /// True when at least one gate in `gates[from..to]` touches qubit `qid`.
    fn range_touches(gates: &[Box<dyn GateOp>], from: usize, to: usize, qid: u32) -> bool {
        if from >= to {
            return false;
        }
        gates[from..to]
            .iter()
            .any(|g| g.qubits().iter().any(|q| q.id() == qid))
    }

    /// First index >= `start` (excluding entries in `skip`) that touches qa or qb.
    fn next_on_pair(
        gates: &[Box<dyn GateOp>],
        start: usize,
        skip: &[bool],
        qa: u32,
        qb: u32,
    ) -> Option<usize> {
        for k in start..gates.len() {
            if skip[k] {
                continue;
            }
            let ids = Self::qubit_ids(gates[k].as_ref());
            if ids.contains(&qa) || ids.contains(&qb) {
                return Some(k);
            }
        }
        None
    }

    /// One optimisation sweep. Returns `(new_gates, did_change)`.
    fn sweep(gates: Vec<Box<dyn GateOp>>) -> (Vec<Box<dyn GateOp>>, bool) {
        let n = gates.len();
        let mut skip = vec![false; n];
        let mut swap_at: HashMap<usize, multi::SWAP> = HashMap::new();
        let mut cnot_at: HashMap<usize, multi::CNOT> = HashMap::new();

        // --- Rule 1: CNOT(a,b) immediately followed (on that pair) by CNOT(a,b) ---
        for i in 0..n {
            if skip[i] {
                continue;
            }
            let ci = match gates[i].as_any().downcast_ref::<multi::CNOT>() {
                Some(c) => *c,
                None => continue,
            };
            let (qa, qb) = (ci.control.id(), ci.target.id());
            if let Some(j) = Self::next_on_pair(&gates, i + 1, &skip, qa, qb) {
                if let Some(cj) = gates[j].as_any().downcast_ref::<multi::CNOT>() {
                    if cj.control.id() == qa && cj.target.id() == qb {
                        skip[i] = true;
                        skip[j] = true;
                    }
                }
            }
        }

        // --- Rule 2: CNOT(a,b), CNOT(b,a), CNOT(a,b) → SWAP(a,b) ---
        for i in 0..n {
            if skip[i] {
                continue;
            }
            let c0 = match gates[i].as_any().downcast_ref::<multi::CNOT>() {
                Some(c) => *c,
                None => continue,
            };
            let (qa, qb) = (c0.control.id(), c0.target.id());
            let j1 = match Self::next_on_pair(&gates, i + 1, &skip, qa, qb) {
                Some(k) => k,
                None => continue,
            };
            match gates[j1].as_any().downcast_ref::<multi::CNOT>() {
                Some(c) if c.control.id() == qb && c.target.id() == qa => {}
                _ => continue,
            }
            let j2 = match Self::next_on_pair(&gates, j1 + 1, &skip, qa, qb) {
                Some(k) => k,
                None => continue,
            };
            if let Some(c2) = gates[j2].as_any().downcast_ref::<multi::CNOT>() {
                if c2.control.id() == qa && c2.target.id() == qb {
                    skip[i] = true;
                    skip[j1] = true;
                    skip[j2] = true;
                    swap_at.insert(
                        i,
                        multi::SWAP {
                            qubit1: c0.control,
                            qubit2: c0.target,
                        },
                    );
                }
            }
        }

        // --- Rule 3: H(b) CZ(a,b) H(b) → CNOT(a,b) ---
        for i in 0..n {
            if skip[i] {
                continue;
            }
            let h1 = match gates[i].as_any().downcast_ref::<single::Hadamard>() {
                Some(h) => *h,
                None => continue,
            };
            let qb = h1.target.id();
            let j = match Self::next_on_pair(&gates, i + 1, &skip, qb, qb) {
                Some(k) => k,
                None => continue,
            };
            let cz = match gates[j].as_any().downcast_ref::<multi::CZ>() {
                Some(c) if c.control.id() == qb || c.target.id() == qb => *c,
                _ => continue,
            };
            let qa = if cz.control.id() == qb {
                cz.target.id()
            } else {
                cz.control.id()
            };
            let k = match Self::next_on_pair(&gates, j + 1, &skip, qb, qb) {
                Some(k) => k,
                None => continue,
            };
            match gates[k].as_any().downcast_ref::<single::Hadamard>() {
                Some(h) if h.target.id() == qb => {}
                _ => continue,
            }
            if Self::range_touches(&gates, i + 1, j, qa)
                || Self::range_touches(&gates, j + 1, k, qa)
            {
                continue;
            }
            let (ctrl, tgt) = if cz.control.id() == qa {
                (cz.control, cz.target)
            } else {
                (cz.target, cz.control)
            };
            skip[i] = true;
            skip[j] = true;
            skip[k] = true;
            cnot_at.insert(
                i,
                multi::CNOT {
                    control: ctrl,
                    target: tgt,
                },
            );
        }

        let any_change = skip.iter().any(|&s| s);
        if !any_change {
            return (gates, false);
        }

        let mut result: Vec<Box<dyn GateOp>> = Vec::with_capacity(n);
        for idx in 0..n {
            if !skip[idx] {
                result.push(gates[idx].clone());
            } else if let Some(&swap) = swap_at.get(&idx) {
                result.push(Box::new(swap));
            } else if let Some(&cnot) = cnot_at.get(&idx) {
                result.push(Box::new(cnot));
            }
            // otherwise: gate is cancelled — emit nothing
        }
        (result, true)
    }

    /// Apply all two-qubit optimisations to convergence.
    fn optimize_two_qubit(gates: Vec<Box<dyn GateOp>>) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut work = gates;
        loop {
            let (next, changed) = Self::sweep(work);
            work = next;
            if !changed {
                break;
            }
        }
        Ok(work)
    }
}

impl OptimizationPass for TwoQubitOptimization {
    fn name(&self) -> &'static str {
        "Two-Qubit Optimization"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        _cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        if !self.optimize_cnots {
            return Ok(gates);
        }
        Self::optimize_two_qubit(gates)
    }
}
