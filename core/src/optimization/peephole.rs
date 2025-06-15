//! Peephole optimization for quantum circuits
//!
//! This module implements peephole optimization, which looks for small patterns
//! of gates that can be simplified or eliminated.

use crate::error::QuantRS2Result;
use crate::gate::{multi::*, single::*, GateOp};
use std::f64::consts::PI;

use super::{gates_can_commute, OptimizationPass};

/// Peephole optimization pass
pub struct PeepholeOptimizer {
    /// Enable rotation merging
    pub merge_rotations: bool,
    /// Enable identity removal
    pub remove_identities: bool,
    /// Enable gate commutation
    pub enable_commutation: bool,
    /// Tolerance for identifying zero rotations
    pub zero_tolerance: f64,
}

impl Default for PeepholeOptimizer {
    fn default() -> Self {
        Self {
            merge_rotations: true,
            remove_identities: true,
            enable_commutation: true,
            zero_tolerance: 1e-10,
        }
    }
}

impl PeepholeOptimizer {
    /// Create a new peephole optimizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a rotation angle is effectively zero
    fn is_zero_rotation(&self, angle: f64) -> bool {
        let normalized = angle % (2.0 * PI);
        normalized.abs() < self.zero_tolerance
            || (normalized - 2.0 * PI).abs() < self.zero_tolerance
    }

    /// Try to simplify a window of gates
    #[allow(dead_code)]
    fn simplify_window(
        &self,
        window: &[Box<dyn GateOp>],
    ) -> QuantRS2Result<Option<Vec<Box<dyn GateOp>>>> {
        match window.len() {
            2 => self.simplify_pair(&window[0], &window[1]),
            3 => self.simplify_triple(&window[0], &window[1], &window[2]),
            _ => Ok(None),
        }
    }

    /// Simplify a pair of gates
    fn simplify_pair(
        &self,
        gate1: &Box<dyn GateOp>,
        gate2: &Box<dyn GateOp>,
    ) -> QuantRS2Result<Option<Vec<Box<dyn GateOp>>>> {
        // Handle rotation merging
        if self.merge_rotations && gate1.qubits() == gate2.qubits() && gate1.qubits().len() == 1 {
            let qubit = gate1.qubits()[0];

            match (gate1.name(), gate2.name()) {
                ("RX", "RX") => {
                    if let (Some(rx1), Some(rx2)) = (
                        gate1.as_any().downcast_ref::<RotationX>(),
                        gate2.as_any().downcast_ref::<RotationX>(),
                    ) {
                        let combined_angle = rx1.theta + rx2.theta;
                        if self.is_zero_rotation(combined_angle) {
                            return Ok(Some(vec![])); // Remove both
                        }
                        return Ok(Some(vec![Box::new(RotationX {
                            target: qubit,
                            theta: combined_angle,
                        })]));
                    }
                }
                ("RY", "RY") => {
                    if let (Some(ry1), Some(ry2)) = (
                        gate1.as_any().downcast_ref::<RotationY>(),
                        gate2.as_any().downcast_ref::<RotationY>(),
                    ) {
                        let combined_angle = ry1.theta + ry2.theta;
                        if self.is_zero_rotation(combined_angle) {
                            return Ok(Some(vec![])); // Remove both
                        }
                        return Ok(Some(vec![Box::new(RotationY {
                            target: qubit,
                            theta: combined_angle,
                        })]));
                    }
                }
                ("RZ", "RZ") => {
                    if let (Some(rz1), Some(rz2)) = (
                        gate1.as_any().downcast_ref::<RotationZ>(),
                        gate2.as_any().downcast_ref::<RotationZ>(),
                    ) {
                        let combined_angle = rz1.theta + rz2.theta;
                        if self.is_zero_rotation(combined_angle) {
                            return Ok(Some(vec![])); // Remove both
                        }
                        return Ok(Some(vec![Box::new(RotationZ {
                            target: qubit,
                            theta: combined_angle,
                        })]));
                    }
                }
                _ => {}
            }
        }

        // Handle special patterns
        if gate1.qubits() == gate2.qubits() {
            match (gate1.name(), gate2.name()) {
                // T gate combinations
                ("T", "T") => {
                    // T² = S
                    return Ok(Some(vec![Box::new(Phase {
                        target: gate1.qubits()[0],
                    })]));
                }
                ("T†", "T†") => {
                    // T†² = S†
                    return Ok(Some(vec![Box::new(PhaseDagger {
                        target: gate1.qubits()[0],
                    })]));
                }

                // S and T combinations
                ("S", "T") | ("T", "S") => {
                    // S·T = T·S = T³ (since T² = S)
                    let qubit = gate1.qubits()[0];
                    return Ok(Some(vec![
                        Box::new(T { target: qubit }),
                        Box::new(T { target: qubit }),
                        Box::new(T { target: qubit }),
                    ]));
                }

                _ => {}
            }
        }

        // Try commutation if enabled
        if self.enable_commutation && gates_can_commute(gate1.as_ref(), gate2.as_ref()) {
            // Return in swapped order if it might help later optimizations
            // This is a heuristic - we prefer to move single-qubit gates earlier
            if gate1.qubits().len() > gate2.qubits().len() {
                return Ok(Some(vec![gate2.clone_gate(), gate1.clone_gate()]));
            }
        }

        Ok(None)
    }

    /// Simplify a triple of gates
    fn simplify_triple(
        &self,
        gate1: &Box<dyn GateOp>,
        gate2: &Box<dyn GateOp>,
        gate3: &Box<dyn GateOp>,
    ) -> QuantRS2Result<Option<Vec<Box<dyn GateOp>>>> {
        // CX-Rz-CX pattern (controlled rotation)
        if gate1.name() == "CNOT" && gate3.name() == "CNOT" && gate2.name() == "RZ" {
            if let (Some(cx1), Some(cx2), Some(rz)) = (
                gate1.as_any().downcast_ref::<CNOT>(),
                gate3.as_any().downcast_ref::<CNOT>(),
                gate2.as_any().downcast_ref::<RotationZ>(),
            ) {
                // Check if it's the controlled-Rz pattern
                if cx1.control == cx2.control && cx1.target == cx2.target && rz.target == cx1.target
                {
                    // This is equivalent to a CRZ gate
                    return Ok(Some(vec![Box::new(CRZ {
                        control: cx1.control,
                        target: cx1.target,
                        theta: rz.theta,
                    })]));
                }
            }
        }

        // H-X-H = Z pattern
        if gate1.name() == "H"
            && gate2.name() == "X"
            && gate3.name() == "H"
            && gate1.qubits() == gate2.qubits()
            && gate2.qubits() == gate3.qubits()
        {
            return Ok(Some(vec![Box::new(PauliZ {
                target: gate1.qubits()[0],
            })]));
        }

        // H-Z-H = X pattern
        if gate1.name() == "H"
            && gate2.name() == "Z"
            && gate3.name() == "H"
            && gate1.qubits() == gate2.qubits()
            && gate2.qubits() == gate3.qubits()
        {
            return Ok(Some(vec![Box::new(PauliX {
                target: gate1.qubits()[0],
            })]));
        }

        // X-Y-X = -Y pattern
        if gate1.name() == "X"
            && gate2.name() == "Y"
            && gate3.name() == "X"
            && gate1.qubits() == gate2.qubits()
            && gate2.qubits() == gate3.qubits()
        {
            let qubit = gate1.qubits()[0];
            return Ok(Some(vec![
                Box::new(PauliY { target: qubit }),
                Box::new(PauliZ { target: qubit }), // Global phase -1
            ]));
        }

        Ok(None)
    }

    /// Remove identity rotations
    fn remove_identity_rotations(&self, gates: Vec<Box<dyn GateOp>>) -> Vec<Box<dyn GateOp>> {
        gates
            .into_iter()
            .filter(|gate| match gate.name() {
                "RX" => {
                    if let Some(rx) = gate.as_any().downcast_ref::<RotationX>() {
                        !self.is_zero_rotation(rx.theta)
                    } else {
                        true
                    }
                }
                "RY" => {
                    if let Some(ry) = gate.as_any().downcast_ref::<RotationY>() {
                        !self.is_zero_rotation(ry.theta)
                    } else {
                        true
                    }
                }
                "RZ" => {
                    if let Some(rz) = gate.as_any().downcast_ref::<RotationZ>() {
                        !self.is_zero_rotation(rz.theta)
                    } else {
                        true
                    }
                }
                _ => true,
            })
            .collect()
    }
}

impl OptimizationPass for PeepholeOptimizer {
    fn optimize(&self, gates: Vec<Box<dyn GateOp>>) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut current = gates;
        let mut changed = true;
        let max_iterations = 10; // Prevent infinite loops
        let mut iterations = 0;

        while changed && iterations < max_iterations {
            changed = false;
            let mut optimized = Vec::new();
            let mut i = 0;

            while i < current.len() {
                // Try triple patterns first
                if i + 2 < current.len() {
                    if let Some(simplified) =
                        self.simplify_triple(&current[i], &current[i + 1], &current[i + 2])?
                    {
                        optimized.extend(simplified);
                        i += 3;
                        changed = true;
                        continue;
                    }
                }

                // Try pair patterns
                if i + 1 < current.len() {
                    if let Some(simplified) = self.simplify_pair(&current[i], &current[i + 1])? {
                        optimized.extend(simplified);
                        i += 2;
                        changed = true;
                        continue;
                    }
                }

                // No pattern matched, keep the gate
                optimized.push(current[i].clone_gate());
                i += 1;
            }

            current = optimized;
            iterations += 1;
        }

        // Final pass to remove identity rotations
        if self.remove_identities {
            current = self.remove_identity_rotations(current);
        }

        Ok(current)
    }

    fn name(&self) -> &str {
        "Peephole Optimization"
    }
}

/// Specialized optimizer for T-count reduction
pub struct TCountOptimizer {
    /// Maximum search depth for optimization
    pub max_depth: usize,
}

impl Default for TCountOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TCountOptimizer {
    pub fn new() -> Self {
        Self { max_depth: 4 }
    }

    /// Count T gates in a sequence
    fn count_t_gates(gates: &[Box<dyn GateOp>]) -> usize {
        gates
            .iter()
            .filter(|g| g.name() == "T" || g.name() == "T†")
            .count()
    }

    /// Try to reduce T-count by recognizing special patterns
    fn reduce_t_count(
        &self,
        gates: &[Box<dyn GateOp>],
    ) -> QuantRS2Result<Option<Vec<Box<dyn GateOp>>>> {
        // Pattern: T-S-T = S-T-S (both have T-count 2, but might enable other optimizations)
        if gates.len() >= 3 {
            for i in 0..gates.len() - 2 {
                if gates[i].name() == "T"
                    && gates[i + 1].name() == "S"
                    && gates[i + 2].name() == "T"
                    && gates[i].qubits() == gates[i + 1].qubits()
                    && gates[i + 1].qubits() == gates[i + 2].qubits()
                {
                    let qubit = gates[i].qubits()[0];
                    let mut result = Vec::new();

                    // Copy gates before pattern
                    for j in 0..i {
                        result.push(gates[j].clone_gate());
                    }

                    // Replace pattern
                    result.push(Box::new(Phase { target: qubit }) as Box<dyn GateOp>);
                    result.push(Box::new(T { target: qubit }) as Box<dyn GateOp>);
                    result.push(Box::new(Phase { target: qubit }) as Box<dyn GateOp>);

                    // Copy gates after pattern
                    for j in i + 3..gates.len() {
                        result.push(gates[j].clone_gate());
                    }

                    return Ok(Some(result));
                }
            }
        }

        Ok(None)
    }
}

impl OptimizationPass for TCountOptimizer {
    fn optimize(&self, gates: Vec<Box<dyn GateOp>>) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let original_t_count = Self::count_t_gates(&gates);

        if let Some(optimized) = self.reduce_t_count(&gates)? {
            let new_t_count = Self::count_t_gates(&optimized);
            if new_t_count < original_t_count {
                return Ok(optimized);
            }
        }

        Ok(gates)
    }

    fn name(&self) -> &str {
        "T-Count Optimization"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::QubitId;

    #[test]
    fn test_rotation_merging() {
        let optimizer = PeepholeOptimizer::new();
        let qubit = QubitId(0);

        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(RotationZ {
                target: qubit,
                theta: PI / 4.0,
            }),
            Box::new(RotationZ {
                target: qubit,
                theta: PI / 4.0,
            }),
        ];

        let result = optimizer.optimize(gates).unwrap();
        assert_eq!(result.len(), 1);

        if let Some(rz) = result[0].as_any().downcast_ref::<RotationZ>() {
            assert!((rz.theta - PI / 2.0).abs() < 1e-10);
        } else {
            panic!("Expected RotationZ");
        }
    }

    #[test]
    fn test_zero_rotation_removal() {
        let optimizer = PeepholeOptimizer::new();
        let qubit = QubitId(0);

        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(RotationX {
                target: qubit,
                theta: PI,
            }),
            Box::new(RotationX {
                target: qubit,
                theta: PI,
            }),
        ];

        let result = optimizer.optimize(gates).unwrap();
        assert_eq!(result.len(), 0); // 2π rotation should be removed
    }

    #[test]
    fn test_cnot_rz_pattern() {
        let optimizer = PeepholeOptimizer::new();
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(CNOT {
                control: q0,
                target: q1,
            }),
            Box::new(RotationZ {
                target: q1,
                theta: PI / 4.0,
            }),
            Box::new(CNOT {
                control: q0,
                target: q1,
            }),
        ];

        let result = optimizer.optimize(gates).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name(), "CRZ");
    }

    #[test]
    fn test_h_x_h_pattern() {
        let optimizer = PeepholeOptimizer::new();
        let qubit = QubitId(0);

        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: qubit }),
            Box::new(PauliX { target: qubit }),
            Box::new(Hadamard { target: qubit }),
        ];

        let result = optimizer.optimize(gates).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name(), "Z");
    }

    #[test]
    fn test_t_gate_combination() {
        let optimizer = PeepholeOptimizer::new();
        let qubit = QubitId(0);

        let gates: Vec<Box<dyn GateOp>> =
            vec![Box::new(T { target: qubit }), Box::new(T { target: qubit })];

        let result = optimizer.optimize(gates).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name(), "S");
    }
}
