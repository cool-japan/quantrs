//! Individual optimization passes
//!
//! This module implements various optimization passes that can be applied to quantum circuits.

use crate::builder::Circuit;
use crate::optimization::gate_properties::{get_gate_properties, CommutationTable};
use crate::optimization::cost_model::CostModel;
use quantrs2_core::error::{QuantRS2Result, QuantRS2Error};
use quantrs2_core::gate::{GateOp, single::{self, RotationX, RotationY, RotationZ, PauliX, PauliZ}, multi};
use quantrs2_core::qubit::QubitId;
use quantrs2_core::decomposition::{GateDecomposable, decompose_controlled_rotation};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

/// Trait for optimization passes (object-safe version)
pub trait OptimizationPass: Send + Sync {
    /// Name of the optimization pass
    fn name(&self) -> &str;
    
    /// Apply the optimization pass to a gate list
    fn apply_to_gates(&self, gates: Vec<Box<dyn GateOp>>, cost_model: &dyn CostModel) -> QuantRS2Result<Vec<Box<dyn GateOp>>>;
    
    /// Check if this pass should be applied
    fn should_apply(&self) -> bool {
        true
    }
}

/// Extension trait for circuit operations
pub trait OptimizationPassExt<const N: usize> {
    fn apply(&self, circuit: &Circuit<N>, cost_model: &dyn CostModel) -> QuantRS2Result<Circuit<N>>;
    fn should_apply_to_circuit(&self, circuit: &Circuit<N>) -> bool;
}

impl<T: OptimizationPass + ?Sized, const N: usize> OptimizationPassExt<N> for T {
    fn apply(&self, circuit: &Circuit<N>, cost_model: &dyn CostModel) -> QuantRS2Result<Circuit<N>> {
        // TODO: Convert circuit to gates, apply pass, convert back
        Ok(circuit.clone())
    }
    
    fn should_apply_to_circuit(&self, _circuit: &Circuit<N>) -> bool {
        self.should_apply()
    }
}

/// Gate cancellation pass - removes redundant gates
pub struct GateCancellation {
    aggressive: bool,
}

impl GateCancellation {
    pub fn new(aggressive: bool) -> Self {
        Self { aggressive }
    }
}

impl OptimizationPass for GateCancellation {
    fn name(&self) -> &str {
        "Gate Cancellation"
    }
    
    fn apply_to_gates(&self, gates: Vec<Box<dyn GateOp>>, _cost_model: &dyn CostModel) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
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
                                gate2.as_any().downcast_ref::<single::RotationX>()
                            ) {
                                let combined_angle = rx1.theta + rx2.theta;
                                // Check if the combined rotation is effectively zero
                                if (combined_angle % (2.0 * PI)).abs() < 1e-10 {
                                    i += 2;
                                    continue;
                                }
                            } else if let (Some(ry1), Some(ry2)) = (
                                gate1.as_any().downcast_ref::<single::RotationY>(),
                                gate2.as_any().downcast_ref::<single::RotationY>()
                            ) {
                                let combined_angle = ry1.theta + ry2.theta;
                                if (combined_angle % (2.0 * PI)).abs() < 1e-10 {
                                    i += 2;
                                    continue;
                                }
                            } else if let (Some(rz1), Some(rz2)) = (
                                gate1.as_any().downcast_ref::<single::RotationZ>(),
                                gate2.as_any().downcast_ref::<single::RotationZ>()
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
                                gate2.as_any().downcast_ref::<multi::CNOT>()
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
                    if gate1.qubits() == gate3.qubits() && 
                       gate1.name() == gate3.name() &&
                       i + 3 < gates.len() {
                        let gate4 = &gates[i + 3];
                        if gate2.qubits() == gate4.qubits() && 
                           gate2.name() == gate4.name() &&
                           gate1.qubits() == gate2.qubits() {
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
            ("S", "Z") | ("Z", "S") | ("T", "Z") | ("Z", "T") => true,
            
            // Same-axis rotations commute
            ("RX", "RX") | ("RY", "RY") | ("RZ", "RZ") => true,
            
            // RZ commutes with Z-like gates
            ("RZ", "Z") | ("Z", "RZ") | ("RZ", "S") | ("S", "RZ") | 
            ("RZ", "T") | ("T", "RZ") => true,
            
            _ => false
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
    fn name(&self) -> &str {
        "Gate Commutation"
    }
    
    fn apply_to_gates(&self, gates: Vec<Box<dyn GateOp>>, _cost_model: &dyn CostModel) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        if gates.len() < 2 {
            return Ok(gates);
        }
        
        let mut optimized = gates;
        let mut changed = true;
        
        // Keep trying to commute gates until no more changes
        while changed {
            changed = false;
            let mut i = 0;
            
            while i < optimized.len().saturating_sub(1) {
                let can_swap = {
                    let gate1 = &optimized[i];
                    let gate2 = &optimized[i + 1];
                    
                    // Check if gates act on different qubits (always commute)
                    let qubits1: HashSet<_> = gate1.qubits().into_iter().collect();
                    let qubits2: HashSet<_> = gate2.qubits().into_iter().collect();
                    
                    if qubits1.is_disjoint(&qubits2) {
                        // Gates on disjoint qubits always commute
                        // Check if swapping would enable optimizations
                        self.would_benefit_from_swap(&optimized, i)
                    } else if qubits1 == qubits2 {
                        // Gates on same qubits - check commutation rules
                        self.gates_commute(gate1.as_ref(), gate2.as_ref())
                    } else {
                        // Overlapping but not identical qubit sets
                        false
                    }
                };
                
                if can_swap {
                    optimized.swap(i, i + 1);
                    changed = true;
                    // Don't increment i to check if we can swap further back
                    if i > 0 {
                        i -= 1;
                    }
                } else {
                    i += 1;
                }
                
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
    pub fn new(merge_rotations: bool, merge_threshold: f64) -> Self {
        Self {
            merge_rotations,
            merge_threshold,
        }
    }
}

impl OptimizationPass for GateMerging {
    fn name(&self) -> &str {
        "Gate Merging"
    }
    
    fn apply_to_gates(&self, gates: Vec<Box<dyn GateOp>>, _cost_model: &dyn CostModel) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
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
                        ("RZ", "RX") | ("RX", "RZ") | ("RY", "RX") | ("RX", "RY") | 
                        ("RZ", "RY") | ("RY", "RZ") if self.merge_threshold > 0.0 => {
                            // Complex merging would require matrix multiplication
                            // For now, skip this advanced optimization
                            None
                        }
                        // Phase gates (S, T) can sometimes be merged with RZ
                        ("S", "RZ") | ("RZ", "S") | ("T", "RZ") | ("RZ", "T") => {
                            // S = RZ(π/2), T = RZ(π/4)
                            // These could be merged but need special handling
                            None
                        }
                        _ => None
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
                    if gate1.name() == "H" && gate3.name() == "H" && 
                       gate1.qubits() == gate2.qubits() && gate2.qubits() == gate3.qubits() {
                        match gate2.name() {
                            "Z" => {
                                // H-Z-H = X
                                optimized.push(Box::new(single::PauliX {
                                    target: gate1.qubits()[0],
                                }) as Box<dyn GateOp>);
                                i += 3;
                                continue;
                            }
                            "X" => {
                                // H-X-H = Z
                                optimized.push(Box::new(single::PauliZ {
                                    target: gate1.qubits()[0],
                                }) as Box<dyn GateOp>);
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
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }
    
    /// Check if angle is effectively zero (or 2π multiple)
    fn is_zero_rotation(&self, angle: f64) -> bool {
        let normalized = angle % (2.0 * PI);
        normalized.abs() < self.tolerance || (normalized - 2.0 * PI).abs() < self.tolerance
    }
    
    /// Merge two rotation angles
    fn merge_angles(&self, angle1: f64, angle2: f64) -> f64 {
        let merged = angle1 + angle2;
        let normalized = merged % (2.0 * PI);
        if normalized > PI {
            normalized - 2.0 * PI
        } else if normalized < -PI {
            normalized + 2.0 * PI
        } else {
            normalized
        }
    }
}

impl OptimizationPass for RotationMerging {
    fn name(&self) -> &str {
        "Rotation Merging"
    }
    
    fn apply_to_gates(&self, gates: Vec<Box<dyn GateOp>>, _cost_model: &dyn CostModel) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
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
                                gate2.as_any().downcast_ref::<single::RotationX>()
                            ) {
                                let merged_angle = self.merge_angles(rx1.theta, rx2.theta);
                                if self.is_zero_rotation(merged_angle) {
                                    // Skip both gates if the merged rotation is effectively zero
                                    i += 2;
                                    continue;
                                } else {
                                    // Create a new merged rotation gate
                                    optimized.push(Box::new(single::RotationX {
                                        target: rx1.target,
                                        theta: merged_angle,
                                    }) as Box<dyn GateOp>);
                                    i += 2;
                                    continue;
                                }
                            }
                        }
                        "RY" => {
                            if let (Some(ry1), Some(ry2)) = (
                                gate1.as_any().downcast_ref::<single::RotationY>(),
                                gate2.as_any().downcast_ref::<single::RotationY>()
                            ) {
                                let merged_angle = self.merge_angles(ry1.theta, ry2.theta);
                                if self.is_zero_rotation(merged_angle) {
                                    i += 2;
                                    continue;
                                } else {
                                    optimized.push(Box::new(single::RotationY {
                                        target: ry1.target,
                                        theta: merged_angle,
                                    }) as Box<dyn GateOp>);
                                    i += 2;
                                    continue;
                                }
                            }
                        }
                        "RZ" => {
                            if let (Some(rz1), Some(rz2)) = (
                                gate1.as_any().downcast_ref::<single::RotationZ>(),
                                gate2.as_any().downcast_ref::<single::RotationZ>()
                            ) {
                                let merged_angle = self.merge_angles(rz1.theta, rz2.theta);
                                if self.is_zero_rotation(merged_angle) {
                                    i += 2;
                                    continue;
                                } else {
                                    optimized.push(Box::new(single::RotationZ {
                                        target: rz1.target,
                                        theta: merged_angle,
                                    }) as Box<dyn GateOp>);
                                    i += 2;
                                    continue;
                                }
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

/// Decomposition optimization - chooses optimal decompositions based on hardware
pub struct DecompositionOptimization {
    target_gate_set: HashSet<String>,
    prefer_native: bool,
}

impl DecompositionOptimization {
    pub fn new(target_gate_set: HashSet<String>, prefer_native: bool) -> Self {
        Self {
            target_gate_set,
            prefer_native,
        }
    }
    
    pub fn for_hardware(hardware: &str) -> Self {
        let target_gate_set = match hardware {
            "ibm" => vec!["X", "Y", "Z", "H", "S", "T", "RZ", "CNOT", "CZ"]
                .into_iter().map(|s| s.to_string()).collect(),
            "google" => vec!["X", "Y", "Z", "H", "RZ", "CZ", "SQRT_X"]
                .into_iter().map(|s| s.to_string()).collect(),
            _ => vec!["X", "Y", "Z", "H", "S", "T", "RZ", "RX", "RY", "CNOT"]
                .into_iter().map(|s| s.to_string()).collect(),
        };
        
        Self {
            target_gate_set,
            prefer_native: true,
        }
    }
}

impl OptimizationPass for DecompositionOptimization {
    fn name(&self) -> &str {
        "Decomposition Optimization"
    }
    
    fn apply_to_gates(&self, gates: Vec<Box<dyn GateOp>>, cost_model: &dyn CostModel) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // TODO: Implement decomposition optimization
        Ok(gates)
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
    pub fn new(target: CostTarget, max_iterations: usize) -> Self {
        Self {
            optimization_target: target,
            max_iterations,
        }
    }
}

impl OptimizationPass for CostBasedOptimization {
    fn name(&self) -> &str {
        "Cost-Based Optimization"
    }
    
    fn apply_to_gates(&self, gates: Vec<Box<dyn GateOp>>, cost_model: &dyn CostModel) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // TODO: Implement cost-based optimization
        Ok(gates)
    }
}

/// Two-qubit gate optimization
pub struct TwoQubitOptimization {
    use_kak_decomposition: bool,
    optimize_cnots: bool,
}

impl TwoQubitOptimization {
    pub fn new(use_kak_decomposition: bool, optimize_cnots: bool) -> Self {
        Self {
            use_kak_decomposition,
            optimize_cnots,
        }
    }
}

impl OptimizationPass for TwoQubitOptimization {
    fn name(&self) -> &str {
        "Two-Qubit Optimization"
    }
    
    fn apply_to_gates(&self, gates: Vec<Box<dyn GateOp>>, _cost_model: &dyn CostModel) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // TODO: Implement two-qubit optimization
        Ok(gates)
    }
}

/// Peephole optimization - looks at small windows of gates for local optimizations
pub struct PeepholeOptimization {
    window_size: usize,
    patterns: Vec<PeepholePattern>,
}

#[derive(Clone)]
pub struct PeepholePattern {
    name: String,
    window_size: usize,
    matcher: fn(&[Box<dyn GateOp>]) -> Option<Vec<Box<dyn GateOp>>>,
}

impl PeepholeOptimization {
    pub fn new(window_size: usize) -> Self {
        let patterns = vec![
            // Pattern: X-Y-X = -Y
            PeepholePattern {
                name: "X-Y-X to -Y".to_string(),
                window_size: 3,
                matcher: |gates| {
                    if gates.len() >= 3 {
                        let g0 = &gates[0];
                        let g1 = &gates[1];
                        let g2 = &gates[2];
                        
                        if g0.name() == "X" && g2.name() == "X" && g1.name() == "Y" &&
                           g0.qubits() == g1.qubits() && g1.qubits() == g2.qubits() {
                            // X-Y-X = -Y, we can return Y with a phase
                            return Some(vec![g1.clone()]);
                        }
                    }
                    None
                },
            },
            // Pattern: H-S-H = X·RZ(π/2)·X
            PeepholePattern {
                name: "H-S-H simplification".to_string(),
                window_size: 3,
                matcher: |gates| {
                    if gates.len() >= 3 {
                        let g0 = &gates[0];
                        let g1 = &gates[1];
                        let g2 = &gates[2];
                        
                        if g0.name() == "H" && g2.name() == "H" && g1.name() == "S" &&
                           g0.qubits() == g1.qubits() && g1.qubits() == g2.qubits() {
                            let target = g0.qubits()[0];
                            return Some(vec![
                                Box::new(single::PauliX { target }) as Box<dyn GateOp>,
                                Box::new(single::RotationZ { target, theta: PI / 2.0 }) as Box<dyn GateOp>,
                                Box::new(single::PauliX { target }) as Box<dyn GateOp>,
                            ]);
                        }
                    }
                    None
                },
            },
            // Pattern: RZ-RX-RZ (Euler decomposition check)
            PeepholePattern {
                name: "Euler angle optimization".to_string(),
                window_size: 3,
                matcher: |gates| {
                    if gates.len() >= 3 {
                        let g0 = &gates[0];
                        let g1 = &gates[1];
                        let g2 = &gates[2];
                        
                        if g0.name() == "RZ" && g1.name() == "RX" && g2.name() == "RZ" &&
                           g0.qubits() == g1.qubits() && g1.qubits() == g2.qubits() {
                            // Check if this is an inefficient decomposition
                            if let (Some(rz1), Some(rx), Some(rz2)) = (
                                g0.as_any().downcast_ref::<single::RotationZ>(),
                                g1.as_any().downcast_ref::<single::RotationX>(),
                                g2.as_any().downcast_ref::<single::RotationZ>()
                            ) {
                                // If middle rotation is small, might be numerical error
                                if rx.theta.abs() < 1e-10 {
                                    // Combine the two RZ rotations
                                    let combined_angle = rz1.theta + rz2.theta;
                                    if combined_angle.abs() < 1e-10 {
                                        return Some(vec![]); // Identity
                                    } else {
                                        return Some(vec![
                                            Box::new(single::RotationZ {
                                                target: rz1.target,
                                                theta: combined_angle,
                                            }) as Box<dyn GateOp>
                                        ]);
                                    }
                                }
                            }
                        }
                    }
                    None
                },
            },
            // Pattern: CNOT-RZ-CNOT (phase gadget)
            PeepholePattern {
                name: "Phase gadget optimization".to_string(),
                window_size: 3,
                matcher: |gates| {
                    if gates.len() >= 3 {
                        let g0 = &gates[0];
                        let g1 = &gates[1];
                        let g2 = &gates[2];
                        
                        if g0.name() == "CNOT" && g2.name() == "CNOT" && g1.name() == "RZ" {
                            if let (Some(cnot1), Some(rz), Some(cnot2)) = (
                                g0.as_any().downcast_ref::<multi::CNOT>(),
                                g1.as_any().downcast_ref::<single::RotationZ>(),
                                g2.as_any().downcast_ref::<multi::CNOT>()
                            ) {
                                // Check if it's the same CNOT structure
                                if cnot1.control == cnot2.control && 
                                   cnot1.target == cnot2.target &&
                                   rz.target == cnot1.target {
                                    // This is a controlled-RZ, keep as is for now
                                    // In future, could replace with native CRZ if available
                                    return None;
                                }
                            }
                        }
                    }
                    None
                },
            },
            // Pattern: Hadamard ladder reduction
            PeepholePattern {
                name: "Hadamard ladder".to_string(),
                window_size: 4,
                matcher: |gates| {
                    if gates.len() >= 4 {
                        // H-CNOT-H-CNOT pattern
                        if gates[0].name() == "H" && gates[1].name() == "CNOT" &&
                           gates[2].name() == "H" && gates[3].name() == "CNOT" {
                            // Check qubit connectivity
                            let h1_target = gates[0].qubits()[0];
                            let h2_target = gates[2].qubits()[0];
                            
                            if let (Some(cnot1), Some(cnot2)) = (
                                gates[1].as_any().downcast_ref::<multi::CNOT>(),
                                gates[3].as_any().downcast_ref::<multi::CNOT>()
                            ) {
                                if h1_target == cnot1.control && h2_target == cnot2.control &&
                                   cnot1.target == cnot2.target {
                                    // Can sometimes be simplified
                                    return None; // Keep for now, needs deeper analysis
                                }
                            }
                        }
                    }
                    None
                },
            },
        ];
        
        Self {
            window_size,
            patterns,
        }
    }
    
    /// Apply patterns to a window of gates
    fn apply_patterns(&self, window: &[Box<dyn GateOp>]) -> Option<Vec<Box<dyn GateOp>>> {
        for pattern in &self.patterns {
            if window.len() >= pattern.window_size {
                if let Some(replacement) = (pattern.matcher)(window) {
                    return Some(replacement);
                }
            }
        }
        None
    }
}

impl OptimizationPass for PeepholeOptimization {
    fn name(&self) -> &str {
        "Peephole Optimization"
    }
    
    fn apply_to_gates(&self, gates: Vec<Box<dyn GateOp>>, _cost_model: &dyn CostModel) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut optimized = Vec::new();
        let mut i = 0;
        
        while i < gates.len() {
            // Try to match patterns starting at position i
            let mut matched = false;
            
            // Try different window sizes up to the configured maximum
            for window_size in (2..=self.window_size).rev() {
                if i + window_size <= gates.len() {
                    let window = &gates[i..i + window_size];
                    
                    if let Some(replacement) = self.apply_patterns(window) {
                        // Apply the optimization
                        optimized.extend(replacement);
                        i += window_size;
                        matched = true;
                        break;
                    }
                }
            }
            
            // If no pattern matched, keep the original gate
            if !matched {
                optimized.push(gates[i].clone());
                i += 1;
            }
        }
        
        Ok(optimized)
    }
}

/// Template matching optimization
pub struct TemplateMatching {
    templates: Vec<CircuitTemplate>,
}

#[derive(Clone)]
pub struct CircuitTemplate {
    name: String,
    pattern: Vec<String>, // Simplified representation
    replacement: Vec<String>,
    cost_reduction: f64,
}

impl TemplateMatching {
    pub fn new() -> Self {
        let templates = vec![
            CircuitTemplate {
                name: "H-X-H to Z".to_string(),
                pattern: vec!["H".to_string(), "X".to_string(), "H".to_string()],
                replacement: vec!["Z".to_string()],
                cost_reduction: 2.0,
            },
            CircuitTemplate {
                name: "CNOT-H-CNOT to CZ".to_string(),
                pattern: vec!["CNOT".to_string(), "H".to_string(), "CNOT".to_string()],
                replacement: vec!["CZ".to_string()],
                cost_reduction: 1.5,
            },
            CircuitTemplate {
                name: "Double CNOT elimination".to_string(),
                pattern: vec!["CNOT".to_string(), "CNOT".to_string()],
                replacement: vec![],
                cost_reduction: 2.0,
            },
        ];
        
        Self { templates }
    }
    
    pub fn with_templates(templates: Vec<CircuitTemplate>) -> Self {
        Self { templates }
    }
}

impl OptimizationPass for TemplateMatching {
    fn name(&self) -> &str {
        "Template Matching"
    }
    
    fn apply_to_gates(&self, gates: Vec<Box<dyn GateOp>>, cost_model: &dyn CostModel) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut optimized = gates;
        let mut changed = true;
        
        while changed {
            changed = false;
            let original_cost = cost_model.gates_cost(&optimized);
            
            for template in &self.templates {
                let result = self.apply_template(template, optimized.clone())?;
                let new_cost = cost_model.gates_cost(&result);
                
                if new_cost < original_cost {
                    optimized = result;
                    changed = true;
                    break;
                }
            }
        }
        
        Ok(optimized)
    }
}

impl TemplateMatching {
    /// Apply a single template to the gates
    fn apply_template(&self, template: &CircuitTemplate, gates: Vec<Box<dyn GateOp>>) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut result = Vec::new();
        let mut i = 0;
        
        while i < gates.len() {
            if let Some(replacement) = self.match_pattern_at_position(template, &gates, i)? {
                // Add replacement gates
                result.extend(replacement);
                i += template.pattern.len();
            } else {
                // Keep original gate
                result.push(gates[i].clone());
                i += 1;
            }
        }
        
        Ok(result)
    }
    
    /// Try to match a pattern starting at a specific position
    fn match_pattern_at_position(&self, template: &CircuitTemplate, gates: &[Box<dyn GateOp>], start: usize) -> QuantRS2Result<Option<Vec<Box<dyn GateOp>>>> {
        if start + template.pattern.len() > gates.len() {
            return Ok(None);
        }
        
        // Check if pattern matches and collect qubits
        let mut qubit_mapping = HashMap::new();
        let mut all_qubits = Vec::new();
        let mut is_match = true;
        
        for (i, pattern_gate) in template.pattern.iter().enumerate() {
            let gate = &gates[start + i];
            
            // Check if the gate name matches
            if !self.gate_matches_pattern(gate.as_ref(), pattern_gate, &mut qubit_mapping) {
                is_match = false;
                break;
            }
            
            // Collect qubits from this gate
            for qubit in gate.qubits() {
                if !all_qubits.contains(&qubit) {
                    all_qubits.push(qubit);
                }
            }
        }
        
        if !is_match {
            return Ok(None);
        }
        
        // Check if all gates operate on the same qubit(s) for single-qubit patterns
        if template.pattern.iter().all(|p| p == "H" || p == "X" || p == "Y" || p == "Z" || p == "S" || p == "T") {
            // For single-qubit gates, check they all operate on the same qubit
            let first_qubit = gates[start].qubits();
            if first_qubit.len() != 1 {
                return Ok(None);
            }
            
            for i in 1..template.pattern.len() {
                let gate_qubits = gates[start + i].qubits();
                if gate_qubits != first_qubit {
                    return Ok(None);
                }
            }
        }
        
        // Store qubits in mapping for replacement generation
        qubit_mapping.insert("qubits".to_string(), all_qubits);
        
        // Generate replacement gates
        self.generate_replacement_gates(template, &qubit_mapping)
    }
    
    /// Check if a gate matches a pattern element
    fn gate_matches_pattern(&self, gate: &dyn GateOp, pattern: &str, qubit_mapping: &mut HashMap<String, Vec<QubitId>>) -> bool {
        // For now, use simple name matching
        // Later we can add more sophisticated pattern matching
        gate.name() == pattern
    }
    
    /// Parse a pattern string like "H(q0)" or "CNOT(q0,q1)"
    fn parse_pattern(&self, pattern: &str) -> Option<(String, String)> {
        if let Some(open_paren) = pattern.find('(') {
            if let Some(close_paren) = pattern.find(')') {
                let gate_name = pattern[..open_paren].to_string();
                let qubit_pattern = pattern[open_paren + 1..close_paren].to_string();
                return Some((gate_name, qubit_pattern));
            }
        }
        None
    }
    
    /// Generate replacement gates based on the template and qubit mapping
    fn generate_replacement_gates(&self, template: &CircuitTemplate, qubit_mapping: &HashMap<String, Vec<QubitId>>) -> QuantRS2Result<Option<Vec<Box<dyn GateOp>>>> {
        let mut replacement_gates = Vec::new();
        
        // For simple patterns, just use the first qubit found in the mapping
        let qubits: Vec<QubitId> = qubit_mapping.values().flat_map(|v| v.iter().cloned()).collect();
        let mut unique_qubits: Vec<QubitId> = Vec::new();
        for qubit in qubits {
            if !unique_qubits.contains(&qubit) {
                unique_qubits.push(qubit);
            }
        }
        
        for replacement_pattern in &template.replacement {
            if let Some(gate) = self.create_simple_gate(replacement_pattern, &unique_qubits)? {
                replacement_gates.push(gate);
            }
        }
        
        Ok(Some(replacement_gates))
    }
    
    /// Create a simple gate from pattern and available qubits
    fn create_simple_gate(&self, pattern: &str, qubits: &[QubitId]) -> QuantRS2Result<Option<Box<dyn GateOp>>> {
        if qubits.is_empty() {
            return Ok(None);
        }
        
        match pattern {
            "H" => Ok(Some(Box::new(single::Hadamard { target: qubits[0] }))),
            "X" => Ok(Some(Box::new(single::PauliX { target: qubits[0] }))),
            "Y" => Ok(Some(Box::new(single::PauliY { target: qubits[0] }))),
            "Z" => Ok(Some(Box::new(single::PauliZ { target: qubits[0] }))),
            "S" => Ok(Some(Box::new(single::Phase { target: qubits[0] }))),
            "T" => Ok(Some(Box::new(single::T { target: qubits[0] }))),
            "CNOT" if qubits.len() >= 2 => Ok(Some(Box::new(multi::CNOT { 
                control: qubits[0], 
                target: qubits[1] 
            }))),
            "CZ" if qubits.len() >= 2 => Ok(Some(Box::new(multi::CZ { 
                control: qubits[0], 
                target: qubits[1] 
            }))),
            "SWAP" if qubits.len() >= 2 => Ok(Some(Box::new(multi::SWAP { 
                qubit1: qubits[0], 
                qubit2: qubits[1] 
            }))),
            _ => Ok(None),
        }
    }
    
    /// Create a gate from a pattern string and qubit mapping
    fn create_gate_from_pattern(&self, pattern: &str, qubit_mapping: &HashMap<String, Vec<QubitId>>) -> QuantRS2Result<Option<Box<dyn GateOp>>> {
        if let Some((gate_name, qubit_pattern)) = self.parse_pattern(pattern) {
            if let Some(qubits) = qubit_mapping.get(&qubit_pattern) {
                return Ok(Some(self.create_gate(&gate_name, qubits)?));
            }
        } else {
            // Try to find any single qubit for simple patterns
            if let Some((_, qubits)) = qubit_mapping.iter().next() {
                if !qubits.is_empty() {
                    return Ok(Some(self.create_gate(pattern, &[qubits[0]])?));
                }
            }
        }
        
        Ok(None)
    }
    
    /// Create a gate instance from name and qubits
    fn create_gate(&self, gate_name: &str, qubits: &[QubitId]) -> QuantRS2Result<Box<dyn GateOp>> {
        match (gate_name, qubits.len()) {
            ("H", 1) => Ok(Box::new(single::Hadamard { target: qubits[0] })),
            ("X", 1) => Ok(Box::new(single::PauliX { target: qubits[0] })),
            ("Y", 1) => Ok(Box::new(single::PauliY { target: qubits[0] })),
            ("Z", 1) => Ok(Box::new(single::PauliZ { target: qubits[0] })),
            ("S", 1) => Ok(Box::new(single::Phase { target: qubits[0] })),
            ("T", 1) => Ok(Box::new(single::T { target: qubits[0] })),
            ("CNOT", 2) => Ok(Box::new(multi::CNOT { 
                control: qubits[0], 
                target: qubits[1] 
            })),
            ("CZ", 2) => Ok(Box::new(multi::CZ { 
                control: qubits[0], 
                target: qubits[1] 
            })),
            ("SWAP", 2) => Ok(Box::new(multi::SWAP { 
                qubit1: qubits[0], 
                qubit2: qubits[1] 
            })),
            _ => Err(QuantRS2Error::UnsupportedOperation(
                format!("Cannot create gate {} with {} qubits", gate_name, qubits.len())
            ))
        }
    }
    
    /// Create an advanced template matcher with more sophisticated patterns
    pub fn with_advanced_templates() -> Self {
        let templates = vec![
            // Basis change patterns
            CircuitTemplate {
                name: "H-Z-H to X".to_string(),
                pattern: vec!["H".to_string(), "Z".to_string(), "H".to_string()],
                replacement: vec!["X".to_string()],
                cost_reduction: 2.0,
            },
            CircuitTemplate {
                name: "H-X-H to Z".to_string(),
                pattern: vec!["H".to_string(), "X".to_string(), "H".to_string()],
                replacement: vec!["Z".to_string()],
                cost_reduction: 2.0,
            },
            // CNOT patterns
            CircuitTemplate {
                name: "CNOT-CNOT elimination".to_string(),
                pattern: vec!["CNOT".to_string(), "CNOT".to_string()],
                replacement: vec![],
                cost_reduction: 2.0,
            },
            // Phase gate optimizations
            CircuitTemplate {
                name: "S-S to Z".to_string(),
                pattern: vec!["S".to_string(), "S".to_string()],
                replacement: vec!["Z".to_string()],
                cost_reduction: 1.0,
            },
            CircuitTemplate {
                name: "T-T-T-T to Identity".to_string(),
                pattern: vec!["T".to_string(), "T".to_string(), "T".to_string(), "T".to_string()],
                replacement: vec![],
                cost_reduction: 4.0,
            },
            // Two-qubit patterns
            CircuitTemplate {
                name: "CNOT-H-CNOT to CZ".to_string(),
                pattern: vec!["CNOT".to_string(), "H".to_string(), "CNOT".to_string()],
                replacement: vec!["CZ".to_string()],
                cost_reduction: 1.0,
            },
            // Fredkin decomposition optimization
            CircuitTemplate {
                name: "SWAP via 3 CNOTs".to_string(),
                pattern: vec!["CNOT".to_string(), "CNOT".to_string(), "CNOT".to_string()],
                replacement: vec!["SWAP".to_string()],
                cost_reduction: 0.5, // SWAP might be native on some hardware
            },
        ];
        
        Self { templates }
    }
    
    /// Create a template matcher for specific hardware
    pub fn for_hardware(hardware: &str) -> Self {
        let templates = match hardware {
            "ibm" => vec![
                CircuitTemplate {
                    name: "H-Z-H to X".to_string(),
                    pattern: vec!["H".to_string(), "Z".to_string(), "H".to_string()],
                    replacement: vec!["X".to_string()],
                    cost_reduction: 2.0,
                },
                CircuitTemplate {
                    name: "CNOT-CNOT elimination".to_string(),
                    pattern: vec!["CNOT".to_string(), "CNOT".to_string()],
                    replacement: vec![],
                    cost_reduction: 2.0,
                },
            ],
            "google" => vec![
                CircuitTemplate {
                    name: "CNOT to CZ with Hadamards".to_string(),
                    pattern: vec!["CNOT".to_string()],
                    replacement: vec!["H".to_string(), "CZ".to_string(), "H".to_string()],
                    cost_reduction: -0.5, // CZ might be more native
                },
            ],
            _ => Self::new().templates,
        };
        
        Self { templates }
    }
}

impl Default for TemplateMatching {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit rewriting using equivalence rules
pub struct CircuitRewriting {
    rules: Vec<RewriteRule>,
    max_rewrites: usize,
}

#[derive(Clone)]
pub struct RewriteRule {
    name: String,
    condition: fn(&[Box<dyn GateOp>]) -> bool,
    rewrite: fn(&[Box<dyn GateOp>]) -> Vec<Box<dyn GateOp>>,
}

impl CircuitRewriting {
    pub fn new(max_rewrites: usize) -> Self {
        let rules = vec![
            // Add rewrite rules here
        ];
        
        Self {
            rules,
            max_rewrites,
        }
    }
}

impl OptimizationPass for CircuitRewriting {
    fn name(&self) -> &str {
        "Circuit Rewriting"
    }
    
    fn apply_to_gates(&self, gates: Vec<Box<dyn GateOp>>, _cost_model: &dyn CostModel) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // TODO: Implement circuit rewriting
        Ok(gates)
    }
}

/// Helper functions for optimization passes
pub mod utils {
    use super::*;
    
    /// Check if two gates cancel each other
    pub fn gates_cancel(gate1: &dyn GateOp, gate2: &dyn GateOp) -> bool {
        if gate1.name() != gate2.name() || gate1.qubits() != gate2.qubits() {
            return false;
        }
        
        let props = get_gate_properties(gate1);
        props.is_self_inverse
    }
    
    /// Check if a gate is effectively identity
    pub fn is_identity_gate(gate: &dyn GateOp, tolerance: f64) -> bool {
        match gate.name() {
            "RX" | "RY" | "RZ" => {
                // Check if rotation angle is effectively 0
                if let Ok(matrix) = gate.matrix() {
                    // Check diagonal elements are close to 1
                    (matrix[0].re - 1.0).abs() < tolerance && matrix[0].im.abs() < tolerance
                } else {
                    false
                }
            }
            _ => false,
        }
    }
    
    /// Calculate circuit depth
    pub fn calculate_depth(gates: &[Box<dyn GateOp>]) -> usize {
        let mut qubit_depths: HashMap<u32, usize> = HashMap::new();
        let mut max_depth = 0;
        
        for gate in gates {
            let gate_qubits = gate.qubits();
            let current_depth = gate_qubits
                .iter()
                .map(|q| qubit_depths.get(&q.id()).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            
            let new_depth = current_depth + 1;
            for qubit in gate_qubits {
                qubit_depths.insert(qubit.id(), new_depth);
            }
            
            max_depth = max_depth.max(new_depth);
        }
        
        max_depth
    }
}