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
use scirs2_core::Complex64;
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

    /// Multiply two 2x2 matrices stored row-major as 4-element slices.
    fn mul_2x2(a: &[Complex64; 4], b: &[Complex64; 4]) -> [Complex64; 4] {
        [
            a[0] * b[0] + a[1] * b[2],
            a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3],
        ]
    }

    /// Extract a single-qubit gate's matrix as a fixed 2x2 array.
    ///
    /// Returns `None` if the gate cannot produce a 2x2 matrix (e.g. its matrix
    /// computation fails or it is not a single-qubit gate), so callers fall back
    /// to keeping the gates untouched.
    fn single_qubit_matrix(gate: &dyn GateOp) -> Option<[Complex64; 4]> {
        let data = gate.matrix().ok()?;
        if data.len() != 4 {
            return None;
        }
        Some([data[0], data[1], data[2], data[3]])
    }

    /// Check whether four single-qubit gates, applied in order
    /// `g1` then `g2` then `g3` then `g4`, compose to the identity up to an
    /// unobservable global phase.
    ///
    /// The combined unitary is `U = M4 · M3 · M2 · M1` (gates apply left-to-right
    /// in circuit order, so later gates left-multiply). `U` equals identity up to
    /// global phase iff it is diagonal with both diagonal entries equal and of
    /// unit modulus. Removing such a block is always physically correct.
    fn four_gate_block_is_identity(
        g1: &dyn GateOp,
        g2: &dyn GateOp,
        g3: &dyn GateOp,
        g4: &dyn GateOp,
    ) -> bool {
        let (Some(m1), Some(m2), Some(m3), Some(m4)) = (
            Self::single_qubit_matrix(g1),
            Self::single_qubit_matrix(g2),
            Self::single_qubit_matrix(g3),
            Self::single_qubit_matrix(g4),
        ) else {
            return false;
        };

        // Compose in application order: U = M4 * (M3 * (M2 * M1)).
        let u = Self::mul_2x2(&m2, &m1);
        let u = Self::mul_2x2(&m3, &u);
        let u = Self::mul_2x2(&m4, &u);

        const TOL: f64 = 1e-10;

        // Off-diagonals must vanish.
        if u[1].norm() > TOL || u[2].norm() > TOL {
            return false;
        }
        // Diagonal entries must be equal (a single global phase) ...
        if (u[0] - u[3]).norm() > TOL {
            return false;
        }
        // ... and that phase must have unit modulus (genuinely a phase).
        (u[0].norm() - 1.0).abs() <= TOL
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

                // Aggressive mode: cancel any block of four consecutive
                // single-qubit gates on the *same* qubit whose combined unitary
                // is the identity (up to an unobservable global phase). Unlike a
                // name-pattern heuristic, this is verified by actually
                // multiplying the gate matrices, so it can never remove gates
                // that do not truly cancel.
                if self.aggressive && i + 3 < gates.len() {
                    let gate3 = &gates[i + 2];
                    let gate4 = &gates[i + 3];

                    let same_single_qubit = gate1.qubits() == gate2.qubits()
                        && gate2.qubits() == gate3.qubits()
                        && gate3.qubits() == gate4.qubits()
                        && gate1.qubits().len() == 1;

                    if same_single_qubit
                        && Self::four_gate_block_is_identity(
                            gate1.as_ref(),
                            gate2.as_ref(),
                            gate3.as_ref(),
                            gate4.as_ref(),
                        )
                    {
                        // The four gates compose to identity: drop all of them.
                        i += 4;
                        continue;
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

#[cfg(test)]
mod basic_passes_tests {
    use super::*;
    use crate::optimization::cost_model::{AbstractCostModel, CostWeights};
    use quantrs2_core::gate::single::{Hadamard, PauliX, PauliY, PauliZ};

    fn cost() -> AbstractCostModel {
        AbstractCostModel::new(CostWeights::default())
    }

    #[test]
    fn test_aggressive_cancels_xyxy_block() {
        // X-Y-X-Y on the same qubit composes to -I (identity up to global
        // phase), so all four must be removed. Pairwise cancellation does NOT
        // catch this (adjacent gates differ), exercising the matrix-based block
        // check.
        let pass = GateCancellation::new(true);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(PauliX { target: QubitId(0) }),
            Box::new(PauliY { target: QubitId(0) }),
            Box::new(PauliX { target: QubitId(0) }),
            Box::new(PauliY { target: QubitId(0) }),
        ];

        let result = pass
            .apply_to_gates(gates, &cost())
            .expect("cancellation pass must succeed");
        assert!(
            result.is_empty(),
            "XYXY block should fully cancel, got {} gates",
            result.len()
        );
    }

    #[test]
    fn test_aggressive_preserves_non_cancelling_block() {
        // H-X-Y-Z does NOT compose to identity, so every gate must be kept.
        let pass = GateCancellation::new(true);
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: QubitId(0) }),
            Box::new(PauliX { target: QubitId(0) }),
            Box::new(PauliY { target: QubitId(0) }),
            Box::new(PauliZ { target: QubitId(0) }),
        ];

        let result = pass
            .apply_to_gates(gates, &cost())
            .expect("cancellation pass must succeed");
        assert_eq!(
            result.len(),
            4,
            "non-cancelling block must be preserved entirely"
        );
    }

    #[test]
    fn test_block_identity_check_direct() {
        // Direct unit check of the matrix-based predicate.
        let x = PauliX { target: QubitId(0) };
        let y = PauliY { target: QubitId(0) };
        let z = PauliZ { target: QubitId(0) };
        let h = Hadamard { target: QubitId(0) };

        // XYXY = -I (cancels).
        assert!(GateCancellation::four_gate_block_is_identity(
            &x, &y, &x, &y
        ));
        // HXYZ is not a global phase times identity.
        assert!(!GateCancellation::four_gate_block_is_identity(
            &h, &x, &y, &z
        ));
    }
}
