//! Basic optimization passes: gate cancellation, commutation, merging, and rotation merging.

use crate::optimization::cost_model::CostModel;
use crate::optimization::gate_properties::CommutationTable;
use quantrs2_core::error::QuantRS2Result;
use quantrs2_core::gate::{
    multi,
    single::{self, RotationX, RotationY, RotationZ},
    GateOp,
};
use quantrs2_core::qubit::QubitId;
use std::collections::HashSet;
use std::f64::consts::PI;

use super::OptimizationPass;

/// Gate cancellation pass - removes redundant gates
pub struct GateCancellation {
    aggressive: bool,
}

impl GateCancellation {
    #[must_use]
    pub const fn new(aggressive: bool) -> Self {
        Self { aggressive }
    }
}

impl OptimizationPass for GateCancellation {
    fn name(&self) -> &'static str {
        "Gate Cancellation"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        _cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut optimized = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            if i + 1 < gates.len() {
                let gate1 = &gates[i];
                let gate2 = &gates[i + 1];

                // Check if gates act on the same qubits
                if gate1.qubits() == gate2.qubits() && gate1.name() == gate2.name() {
                    // Check for self-inverse gates (H, X, Y, Z)
                    match gate1.name() {
                        "H" | "X" | "Y" | "Z" => {
                            // These gates cancel when applied twice - skip both
                            i += 2;
                            continue;
                        }
                        "RX" | "RY" | "RZ" => {
                            // Check if rotations cancel
                            if let (Some(rx1), Some(rx2)) = (
                                gate1.as_any().downcast_ref::<single::RotationX>(),
                                gate2.as_any().downcast_ref::<single::RotationX>(),
                            ) {
                                let combined_angle = rx1.theta + rx2.theta;
                                // Check if the combined rotation is effectively zero
                                if (combined_angle % (2.0 * PI)).abs() < 1e-10 {
                                    i += 2;
                                    continue;
                                }
                            } else if let (Some(ry1), Some(ry2)) = (
                                gate1.as_any().downcast_ref::<single::RotationY>(),
                                gate2.as_any().downcast_ref::<single::RotationY>(),
                            ) {
                                let combined_angle = ry1.theta + ry2.theta;
                                if (combined_angle % (2.0 * PI)).abs() < 1e-10 {
                                    i += 2;
                                    continue;
                                }
                            } else if let (Some(rz1), Some(rz2)) = (
                                gate1.as_any().downcast_ref::<single::RotationZ>(),
                                gate2.as_any().downcast_ref::<single::RotationZ>(),
                            ) {
                                let combined_angle = rz1.theta + rz2.theta;
                                if (combined_angle % (2.0 * PI)).abs() < 1e-10 {
                                    i += 2;
                                    continue;
                                }
                            }
                        }
                        "CNOT" => {
                            // CNOT is self-inverse
                            if let (Some(cnot1), Some(cnot2)) = (
                                gate1.as_any().downcast_ref::<multi::CNOT>(),
                                gate2.as_any().downcast_ref::<multi::CNOT>(),
                            ) {
                                if cnot1.control == cnot2.control && cnot1.target == cnot2.target {
                                    i += 2;
                                    continue;
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // Look for more complex cancellations if aggressive mode is enabled
                if self.aggressive && i + 2 < gates.len() {
                    // Check for patterns like X-Y-X-Y or Z-H-Z-H
                    let gate3 = &gates[i + 2];
                    if gate1.qubits() == gate3.qubits()
                        && gate1.name() == gate3.name()
                        && i + 3 < gates.len()
                    {
                        let gate4 = &gates[i + 3];
                        if gate2.qubits() == gate4.qubits()
                            && gate2.name() == gate4.name()
                            && gate1.qubits() == gate2.qubits()
                        {
                            // Pattern detected, check if it simplifies
                            match (gate1.name(), gate2.name()) {
                                ("X", "Y") | ("Y", "X") | ("Z", "H") | ("H", "Z") => {
                                    // These patterns can sometimes simplify
                                    // For now, we'll keep them as they might not always cancel
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            // If we didn't skip, add the gate to optimized list
            optimized.push(gates[i].clone());
            i += 1;
        }

        Ok(optimized)
    }
}

/// Gate commutation pass - reorders gates to enable other optimizations
pub struct GateCommutation {
    max_lookahead: usize,
    commutation_table: CommutationTable,
}

impl GateCommutation {
    #[must_use]
    pub fn new(max_lookahead: usize) -> Self {
        Self {
            max_lookahead,
            commutation_table: CommutationTable::new(),
        }
    }
}

impl GateCommutation {
    /// Check if two gates commute based on commutation rules
    fn gates_commute(&self, gate1: &dyn GateOp, gate2: &dyn GateOp) -> bool {
        // Use commutation table if available
        if self.commutation_table.commutes(gate1.name(), gate2.name()) {
            return true;
        }

        // Additional commutation rules
        match (gate1.name(), gate2.name()) {
            // Pauli gates commutation
            ("X", "X") | ("Y", "Y") | ("Z", "Z") => true,
            ("I", _) | (_, "I") => true,

            // Phase/T gates commute with Z
            ("S" | "T", "Z") | ("Z", "S" | "T") => true,

            // Same-axis rotations commute
            ("RX", "RX") | ("RY", "RY") | ("RZ", "RZ") => true,

            // RZ commutes with Z-like gates
            ("RZ", "Z" | "S" | "T") | ("Z" | "S" | "T", "RZ") => true,

            _ => false,
        }
    }

    /// Check if swapping gates at position i would enable optimizations
    fn would_benefit_from_swap(&self, gates: &[Box<dyn GateOp>], i: usize) -> bool {
        if i + 2 >= gates.len() {
            return false;
        }

        let gate1 = &gates[i];
        let gate2 = &gates[i + 1];
        let gate3 = &gates[i + 2];

        // Check if swapping would create cancellation opportunities
        if gate1.name() == gate3.name() && gate1.qubits() == gate3.qubits() {
            // After swap, gate2 and gate3 (originally gate1) would be adjacent
            match gate3.name() {
                "H" | "X" | "Y" | "Z" => return true,
                _ => {}
            }
        }

        // Check if swapping would enable rotation merging
        if gate2.name() == gate3.name() && gate2.qubits() == gate3.qubits() {
            match gate2.name() {
                "RX" | "RY" | "RZ" => return true,
                _ => {}
            }
        }

        false
    }
}

impl OptimizationPass for GateCommutation {
    fn name(&self) -> &'static str {
        "Gate Commutation"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        _cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        if gates.len() < 2 {
            return Ok(gates);
        }

        let mut optimized = gates;
        // Bound the number of outer iterations to prevent oscillation.
        // Each pass does at most one forward scan; repeated passes let reordering
        // propagate, but the bound ensures we always terminate.
        let max_outer = self.max_lookahead * 2 + 1;
        let mut outer_iter = 0;
        let mut changed = true;

        // Keep trying to commute gates until no more changes or the iteration
        // bound is reached.
        while changed && outer_iter < max_outer {
            changed = false;
            outer_iter += 1;
            let mut i = 0;

            while i < optimized.len().saturating_sub(1) {
                let can_swap = {
                    let gate1 = &optimized[i];
                    let gate2 = &optimized[i + 1];

                    // Check if gates act on different qubits (always commute)
                    let qubits1: HashSet<_> = gate1.qubits().into_iter().collect();
                    let qubits2: HashSet<_> = gate2.qubits().into_iter().collect();

                    if qubits1.is_disjoint(&qubits2) {
                        // Gates on disjoint qubits: only swap when it would enable
                        // further optimisations (not just because they commute).
                        self.would_benefit_from_swap(&optimized, i)
                    } else if qubits1 == qubits2 {
                        // Same qubit set: only swap when a downstream gate of the
                        // same type exists that could later cancel or merge.
                        // Swapping two identical same-qubit gates is always a no-op,
                        // so guard against that first.
                        if gate1.name() == gate2.name() {
                            // Identical gate names on same qubits: swapping achieves
                            // nothing useful — skip to avoid oscillation.
                            false
                        } else {
                            self.gates_commute(gate1.as_ref(), gate2.as_ref())
                        }
                    } else {
                        // Overlapping but not identical qubit sets
                        false
                    }
                };

                if can_swap {
                    optimized.swap(i, i + 1);
                    changed = true;
                }
                // Always advance forward to avoid cycling on the same pair.
                i += 1;

                // Limit lookahead to prevent excessive computation
                if i >= self.max_lookahead {
                    break;
                }
            }
        }

        Ok(optimized)
    }
}

/// Gate merging pass - combines adjacent gates
pub struct GateMerging {
    merge_rotations: bool,
    merge_threshold: f64,
}

impl GateMerging {
    #[must_use]
    pub const fn new(merge_rotations: bool, merge_threshold: f64) -> Self {
        Self {
            merge_rotations,
            merge_threshold,
        }
    }
}

impl OptimizationPass for GateMerging {
    fn name(&self) -> &'static str {
        "Gate Merging"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        _cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut optimized = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            if i + 1 < gates.len() && self.merge_rotations {
                let gate1 = &gates[i];
                let gate2 = &gates[i + 1];

                // Try to merge rotation gates
                if gate1.qubits() == gate2.qubits() {
                    let merged = match (gate1.name(), gate2.name()) {
                        // Same-axis rotations can be directly merged
                        ("RX", "RX") | ("RY", "RY") | ("RZ", "RZ") => {
                            // Already handled by RotationMerging pass, skip here
                            None
                        }
                        // Different axis rotations might be mergeable using Euler decomposition
                        ("RZ" | "RY", "RX") | ("RX" | "RY", "RZ") | ("RX" | "RZ", "RY")
                            if self.merge_threshold > 0.0 =>
                        {
                            // Complex merging would require matrix multiplication
                            // For now, skip this advanced optimization
                            None
                        }
                        // Phase gates (S, T) can sometimes be merged with RZ
                        ("S" | "T", "RZ") | ("RZ", "S" | "T") => {
                            // S = RZ(π/2), T = RZ(π/4)
                            // These could be merged but need special handling
                            None
                        }
                        _ => None,
                    };

                    if let Some(merged_gate) = merged {
                        optimized.push(merged_gate);
                        i += 2;
                        continue;
                    }
                }
            }

            // Check for special merging patterns
            if i + 1 < gates.len() {
                let gate1 = &gates[i];
                let gate2 = &gates[i + 1];

                // H-Z-H = X, H-X-H = Z (basis change)
                if i + 2 < gates.len() {
                    let gate3 = &gates[i + 2];
                    if gate1.name() == "H"
                        && gate3.name() == "H"
                        && gate1.qubits() == gate2.qubits()
                        && gate2.qubits() == gate3.qubits()
                    {
                        match gate2.name() {
                            "Z" => {
                                // H-Z-H = X
                                optimized.push(Box::new(single::PauliX {
                                    target: gate1.qubits()[0],
                                })
                                    as Box<dyn GateOp>);
                                i += 3;
                                continue;
                            }
                            "X" => {
                                // H-X-H = Z
                                optimized.push(Box::new(single::PauliZ {
                                    target: gate1.qubits()[0],
                                })
                                    as Box<dyn GateOp>);
                                i += 3;
                                continue;
                            }
                            _ => {}
                        }
                    }
                }
            }

            // If no merging happened, keep the original gate
            optimized.push(gates[i].clone());
            i += 1;
        }

        Ok(optimized)
    }
}

/// Rotation merging pass - specifically merges rotation gates
pub struct RotationMerging {
    tolerance: f64,
}

impl RotationMerging {
    #[must_use]
    pub const fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Check if angle is effectively zero (or 2π multiple)
    fn is_zero_rotation(&self, angle: f64) -> bool {
        let normalized = angle % (2.0 * PI);
        normalized.abs() < self.tolerance || 2.0f64.mul_add(-PI, normalized).abs() < self.tolerance
    }

    /// Merge two rotation angles
    fn merge_angles(&self, angle1: f64, angle2: f64) -> f64 {
        let merged = angle1 + angle2;
        let normalized = merged % (2.0 * PI);
        if normalized > PI {
            2.0f64.mul_add(-PI, normalized)
        } else if normalized < -PI {
            2.0f64.mul_add(PI, normalized)
        } else {
            normalized
        }
    }
}

impl OptimizationPass for RotationMerging {
    fn name(&self) -> &'static str {
        "Rotation Merging"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        _cost_model: &dyn CostModel,
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut optimized = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            if i + 1 < gates.len() {
                let gate1 = &gates[i];
                let gate2 = &gates[i + 1];

                // Check if both gates are rotations on the same qubit and axis
                if gate1.qubits() == gate2.qubits() && gate1.name() == gate2.name() {
                    match gate1.name() {
                        "RX" => {
                            if let (Some(rx1), Some(rx2)) = (
                                gate1.as_any().downcast_ref::<single::RotationX>(),
                                gate2.as_any().downcast_ref::<single::RotationX>(),
                            ) {
                                let merged_angle = self.merge_angles(rx1.theta, rx2.theta);
                                if self.is_zero_rotation(merged_angle) {
                                    // Skip both gates if the merged rotation is effectively zero
                                    i += 2;
                                    continue;
                                }
                                // Create a new merged rotation gate
                                optimized.push(Box::new(single::RotationX {
                                    target: rx1.target,
                                    theta: merged_angle,
                                })
                                    as Box<dyn GateOp>);
                                i += 2;
                                continue;
                            }
                        }
                        "RY" => {
                            if let (Some(ry1), Some(ry2)) = (
                                gate1.as_any().downcast_ref::<single::RotationY>(),
                                gate2.as_any().downcast_ref::<single::RotationY>(),
                            ) {
                                let merged_angle = self.merge_angles(ry1.theta, ry2.theta);
                                if self.is_zero_rotation(merged_angle) {
                                    i += 2;
                                    continue;
                                }
                                optimized.push(Box::new(single::RotationY {
                                    target: ry1.target,
                                    theta: merged_angle,
                                })
                                    as Box<dyn GateOp>);
                                i += 2;
                                continue;
                            }
                        }
                        "RZ" => {
                            if let (Some(rz1), Some(rz2)) = (
                                gate1.as_any().downcast_ref::<single::RotationZ>(),
                                gate2.as_any().downcast_ref::<single::RotationZ>(),
                            ) {
                                let merged_angle = self.merge_angles(rz1.theta, rz2.theta);
                                if self.is_zero_rotation(merged_angle) {
                                    i += 2;
                                    continue;
                                }
                                optimized.push(Box::new(single::RotationZ {
                                    target: rz1.target,
                                    theta: merged_angle,
                                })
                                    as Box<dyn GateOp>);
                                i += 2;
                                continue;
                            }
                        }
                        _ => {}
                    }
                }
            }

            // If we didn't merge, keep the original gate
            optimized.push(gates[i].clone());
            i += 1;
        }

        Ok(optimized)
    }
}
