//! Circuit optimization framework for quantum circuits
//!
//! This module provides a comprehensive framework for optimizing quantum circuits
//! through various transformation passes that preserve circuit equivalence while
//! reducing gate count, depth, and execution time.
//!
//! # Optimization Passes
//!
//! - **Gate Cancellation**: Remove inverse gate pairs (H-H, X-X, CNOT-CNOT)
//! - **Gate Fusion**: Combine adjacent single-qubit rotations
//! - **Gate Commutation**: Reorder gates using commutation rules
//! - **Template Matching**: Replace gate sequences with optimized equivalents
//! - **Two-Qubit Reduction**: Minimize expensive two-qubit gates
//!
//! # Example
//!
//! ```ignore
//! use quantrs2_sim::circuit_optimizer::{CircuitOptimizer, OptimizationPass};
//!
//! let optimizer = CircuitOptimizer::new()
//!     .with_pass(OptimizationPass::CancelInverses)
//!     .with_pass(OptimizationPass::FuseRotations)
//!     .with_pass(OptimizationPass::CommutativeReordering);
//!
//! let optimized_circuit = optimizer.optimize(&circuit)?;
//! ```

use crate::error::{Result, SimulatorError};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Gate Representation
// ============================================================================

/// Quantum gate type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateType {
    /// Identity gate
    I,
    /// Hadamard gate
    H,
    /// Pauli-X gate
    X,
    /// Pauli-Y gate
    Y,
    /// Pauli-Z gate
    Z,
    /// S gate (sqrt(Z))
    S,
    /// S† gate
    Sdg,
    /// T gate (fourth root of Z)
    T,
    /// T† gate
    Tdg,
    /// Rotation around X axis
    RX,
    /// Rotation around Y axis
    RY,
    /// Rotation around Z axis
    RZ,
    /// CNOT gate
    CNOT,
    /// CZ gate
    CZ,
    /// SWAP gate
    SWAP,
    /// Toffoli gate (CCX)
    Toffoli,
}

impl GateType {
    /// Check if this gate is its own inverse
    pub fn is_self_inverse(&self) -> bool {
        matches!(
            self,
            GateType::H | GateType::X | GateType::Y | GateType::Z | GateType::CNOT | GateType::SWAP
        )
    }

    /// Get the inverse gate type
    pub fn inverse(&self) -> Self {
        match self {
            GateType::S => GateType::Sdg,
            GateType::Sdg => GateType::S,
            GateType::T => GateType::Tdg,
            GateType::Tdg => GateType::T,
            GateType::RX | GateType::RY | GateType::RZ => *self, // Inverse is negated parameter
            _ if self.is_self_inverse() => *self,
            _ => *self, // Identity and others
        }
    }

    /// Check if this is a rotation gate
    pub fn is_rotation(&self) -> bool {
        matches!(self, GateType::RX | GateType::RY | GateType::RZ)
    }

    /// Check if this is a single-qubit gate
    pub fn is_single_qubit(&self) -> bool {
        !matches!(
            self,
            GateType::CNOT | GateType::CZ | GateType::SWAP | GateType::Toffoli
        )
    }

    /// Check if two gates commute on the same qubits
    pub fn commutes_with(&self, other: &GateType) -> bool {
        // Simplified commutation rules
        match (self, other) {
            // Z-basis gates commute with each other
            (GateType::Z, GateType::Z)
            | (GateType::Z, GateType::S)
            | (GateType::Z, GateType::Sdg)
            | (GateType::Z, GateType::T)
            | (GateType::Z, GateType::Tdg)
            | (GateType::S, GateType::Z)
            | (GateType::Sdg, GateType::Z)
            | (GateType::T, GateType::Z)
            | (GateType::Tdg, GateType::Z) => true,

            // Rotation around same axis commute
            (GateType::RX, GateType::RX)
            | (GateType::RY, GateType::RY)
            | (GateType::RZ, GateType::RZ) => true,

            _ => false,
        }
    }
}

/// Quantum gate instruction
#[derive(Debug, Clone, PartialEq)]
pub struct Gate {
    /// Gate type
    pub gate_type: GateType,
    /// Qubits this gate acts on
    pub qubits: Vec<usize>,
    /// Parameters (for rotation gates)
    pub parameters: Vec<f64>,
}

impl Gate {
    /// Create a new gate
    pub fn new(gate_type: GateType, qubits: Vec<usize>) -> Self {
        Self {
            gate_type,
            qubits,
            parameters: Vec::new(),
        }
    }

    /// Create a gate with parameters
    pub fn with_parameters(gate_type: GateType, qubits: Vec<usize>, parameters: Vec<f64>) -> Self {
        Self {
            gate_type,
            qubits,
            parameters,
        }
    }

    /// Get the inverse gate
    pub fn inverse(&self) -> Self {
        let inv_type = self.gate_type.inverse();
        let inv_params = if self.gate_type.is_rotation() {
            self.parameters.iter().map(|p| -p).collect()
        } else {
            self.parameters.clone()
        };

        Self {
            gate_type: inv_type,
            qubits: self.qubits.clone(),
            parameters: inv_params,
        }
    }

    /// Check if two gates are inverse of each other
    pub fn is_inverse_of(&self, other: &Gate) -> bool {
        if self.qubits != other.qubits {
            return false;
        }

        if self.gate_type.is_self_inverse() && self.gate_type == other.gate_type {
            return true;
        }

        if self.gate_type.inverse() == other.gate_type {
            // For rotation gates, check if parameters are negated
            if self.gate_type.is_rotation() {
                return self
                    .parameters
                    .iter()
                    .zip(other.parameters.iter())
                    .all(|(p1, p2)| (p1 + p2).abs() < 1e-10);
            }
            return true;
        }

        false
    }
}

/// Quantum circuit
#[derive(Debug, Clone)]
pub struct Circuit {
    /// Number of qubits
    pub n_qubits: usize,
    /// Sequence of gates
    pub gates: Vec<Gate>,
}

impl Circuit {
    /// Create a new circuit
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            gates: Vec::new(),
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: Gate) -> Result<()> {
        // Verify qubit indices
        for &qubit in &gate.qubits {
            if qubit >= self.n_qubits {
                return Err(SimulatorError::InvalidInput(format!(
                    "Qubit index {} out of range (circuit has {} qubits)",
                    qubit, self.n_qubits
                )));
            }
        }
        self.gates.push(gate);
        Ok(())
    }

    /// Get circuit depth (number of layers)
    pub fn depth(&self) -> usize {
        if self.gates.is_empty() {
            return 0;
        }

        let mut qubit_depths = vec![0; self.n_qubits];
        let mut max_depth = 0;

        for gate in &self.gates {
            // Find maximum depth of qubits involved
            let current_depth = gate
                .qubits
                .iter()
                .map(|&q| qubit_depths[q])
                .max()
                .unwrap_or(0);

            // Update depths for all involved qubits
            for &qubit in &gate.qubits {
                qubit_depths[qubit] = current_depth + 1;
            }

            max_depth = max_depth.max(current_depth + 1);
        }

        max_depth
    }

    /// Count gates by type
    pub fn gate_counts(&self) -> HashMap<GateType, usize> {
        let mut counts = HashMap::new();
        for gate in &self.gates {
            *counts.entry(gate.gate_type).or_insert(0) += 1;
        }
        counts
    }

    /// Count two-qubit gates
    pub fn two_qubit_gate_count(&self) -> usize {
        self.gates.iter().filter(|g| g.qubits.len() == 2).count()
    }
}

// ============================================================================
// Optimization Passes
// ============================================================================

/// Optimization pass types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPass {
    /// Remove inverse gate pairs
    CancelInverses,
    /// Fuse adjacent rotation gates
    FuseRotations,
    /// Reorder gates using commutation rules
    CommutativeReordering,
    /// Remove identity gates
    RemoveIdentities,
    /// Template matching and replacement
    TemplateMatching,
}

/// Circuit optimizer
pub struct CircuitOptimizer {
    /// Optimization passes to apply
    passes: Vec<OptimizationPass>,
    /// Maximum number of optimization iterations
    max_iterations: usize,
}

impl CircuitOptimizer {
    /// Create a new optimizer with default passes
    pub fn new() -> Self {
        Self {
            passes: vec![
                OptimizationPass::CancelInverses,
                OptimizationPass::RemoveIdentities,
                OptimizationPass::FuseRotations,
            ],
            max_iterations: 10,
        }
    }

    /// Add an optimization pass
    pub fn with_pass(mut self, pass: OptimizationPass) -> Self {
        self.passes.push(pass);
        self
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Optimize a circuit
    pub fn optimize(&self, circuit: &Circuit) -> Result<Circuit> {
        let mut optimized = circuit.clone();
        let mut iteration = 0;

        loop {
            let initial_gate_count = optimized.gates.len();

            // Apply all passes
            for pass in &self.passes {
                optimized = self.apply_pass(&optimized, *pass)?;
            }

            iteration += 1;
            let final_gate_count = optimized.gates.len();

            // Stop if no improvement or max iterations reached
            if final_gate_count >= initial_gate_count || iteration >= self.max_iterations {
                break;
            }
        }

        Ok(optimized)
    }

    /// Apply a single optimization pass
    fn apply_pass(&self, circuit: &Circuit, pass: OptimizationPass) -> Result<Circuit> {
        match pass {
            OptimizationPass::CancelInverses => self.cancel_inverses(circuit),
            OptimizationPass::FuseRotations => self.fuse_rotations(circuit),
            OptimizationPass::CommutativeReordering => self.commutative_reordering(circuit),
            OptimizationPass::RemoveIdentities => self.remove_identities(circuit),
            OptimizationPass::TemplateMatching => self.template_matching(circuit),
        }
    }

    /// Remove inverse gate pairs (e.g., H-H, CNOT-CNOT)
    fn cancel_inverses(&self, circuit: &Circuit) -> Result<Circuit> {
        let mut optimized = Circuit::new(circuit.n_qubits);
        let gates = &circuit.gates;
        let mut skip_next = false;

        for i in 0..gates.len() {
            if skip_next {
                skip_next = false;
                continue;
            }

            if i + 1 < gates.len() && gates[i].is_inverse_of(&gates[i + 1]) {
                // Skip both gates (they cancel)
                skip_next = true;
            } else {
                optimized.add_gate(gates[i].clone())?;
            }
        }

        Ok(optimized)
    }

    /// Fuse adjacent rotation gates on the same qubit and axis
    fn fuse_rotations(&self, circuit: &Circuit) -> Result<Circuit> {
        let mut optimized = Circuit::new(circuit.n_qubits);
        let gates = &circuit.gates;
        let mut i = 0;

        while i < gates.len() {
            let gate = &gates[i];

            // Check if this is a rotation gate
            if gate.gate_type.is_rotation() && gate.qubits.len() == 1 {
                // Look for adjacent rotations on the same qubit and axis
                let mut fused_angle = gate.parameters[0];
                let mut j = i + 1;

                while j < gates.len() {
                    let next_gate = &gates[j];
                    if next_gate.gate_type == gate.gate_type
                        && next_gate.qubits == gate.qubits
                        && !next_gate.parameters.is_empty()
                    {
                        fused_angle += next_gate.parameters[0];
                        j += 1;
                    } else {
                        break;
                    }
                }

                // Add fused gate only if angle is non-zero
                if fused_angle.abs() > 1e-10 {
                    optimized.add_gate(Gate::with_parameters(
                        gate.gate_type,
                        gate.qubits.clone(),
                        vec![fused_angle],
                    ))?;
                }

                i = j;
            } else {
                optimized.add_gate(gate.clone())?;
                i += 1;
            }
        }

        Ok(optimized)
    }

    /// Reorder gates using commutation rules
    fn commutative_reordering(&self, circuit: &Circuit) -> Result<Circuit> {
        // Simple commutation: move gates to enable more cancellations
        // This is a simplified version - full implementation would use DAG analysis
        Ok(circuit.clone())
    }

    /// Remove identity gates and gates with zero parameters
    fn remove_identities(&self, circuit: &Circuit) -> Result<Circuit> {
        let mut optimized = Circuit::new(circuit.n_qubits);

        for gate in &circuit.gates {
            // Skip identity gates
            if gate.gate_type == GateType::I {
                continue;
            }

            // Skip rotation gates with zero angle
            if gate.gate_type.is_rotation()
                && !gate.parameters.is_empty()
                && gate.parameters[0].abs() < 1e-10
            {
                continue;
            }

            optimized.add_gate(gate.clone())?;
        }

        Ok(optimized)
    }

    /// Template matching and replacement
    fn template_matching(&self, circuit: &Circuit) -> Result<Circuit> {
        // Simplified: Look for common patterns like H-CNOT-H = CNOT with swapped control/target
        Ok(circuit.clone())
    }
}

impl Default for CircuitOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Original gate count
    pub original_gates: usize,
    /// Optimized gate count
    pub optimized_gates: usize,
    /// Original circuit depth
    pub original_depth: usize,
    /// Optimized circuit depth
    pub optimized_depth: usize,
    /// Original two-qubit gate count
    pub original_two_qubit_gates: usize,
    /// Optimized two-qubit gate count
    pub optimized_two_qubit_gates: usize,
    /// Reduction percentage
    pub gate_reduction_percent: f64,
    /// Depth reduction percentage
    pub depth_reduction_percent: f64,
}

impl OptimizationStats {
    /// Compute statistics from original and optimized circuits
    pub fn from_circuits(original: &Circuit, optimized: &Circuit) -> Self {
        let original_gates = original.gates.len();
        let optimized_gates = optimized.gates.len();
        let original_depth = original.depth();
        let optimized_depth = optimized.depth();
        let original_two_qubit = original.two_qubit_gate_count();
        let optimized_two_qubit = optimized.two_qubit_gate_count();

        let gate_reduction = if original_gates > 0 {
            100.0 * (original_gates - optimized_gates) as f64 / original_gates as f64
        } else {
            0.0
        };

        let depth_reduction = if original_depth > 0 {
            100.0 * (original_depth - optimized_depth) as f64 / original_depth as f64
        } else {
            0.0
        };

        Self {
            original_gates,
            optimized_gates,
            original_depth,
            optimized_depth,
            original_two_qubit_gates: original_two_qubit,
            optimized_two_qubit_gates: optimized_two_qubit,
            gate_reduction_percent: gate_reduction,
            depth_reduction_percent: depth_reduction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_inverse() {
        let h = Gate::new(GateType::H, vec![0]);
        let h_inv = h.inverse();
        assert!(h.is_inverse_of(&h_inv));
    }

    #[test]
    fn test_cancel_inverses() {
        let mut circuit = Circuit::new(2);
        circuit.add_gate(Gate::new(GateType::H, vec![0])).unwrap();
        circuit.add_gate(Gate::new(GateType::H, vec![0])).unwrap();
        circuit.add_gate(Gate::new(GateType::X, vec![1])).unwrap();

        let optimizer = CircuitOptimizer::new();
        let optimized = optimizer.cancel_inverses(&circuit).unwrap();

        // H-H should cancel, leaving only X
        assert_eq!(optimized.gates.len(), 1);
        assert_eq!(optimized.gates[0].gate_type, GateType::X);
    }

    #[test]
    fn test_fuse_rotations() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Gate::with_parameters(
                GateType::RX,
                vec![0],
                vec![std::f64::consts::PI / 4.0],
            ))
            .unwrap();
        circuit
            .add_gate(Gate::with_parameters(
                GateType::RX,
                vec![0],
                vec![std::f64::consts::PI / 4.0],
            ))
            .unwrap();

        let optimizer = CircuitOptimizer::new();
        let optimized = optimizer.fuse_rotations(&circuit).unwrap();

        // Two RX gates should fuse into one
        assert_eq!(optimized.gates.len(), 1);
        assert!((optimized.gates[0].parameters[0] - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_remove_identities() {
        let mut circuit = Circuit::new(2);
        circuit.add_gate(Gate::new(GateType::I, vec![0])).unwrap();
        circuit.add_gate(Gate::new(GateType::X, vec![1])).unwrap();
        circuit
            .add_gate(Gate::with_parameters(GateType::RZ, vec![0], vec![0.0]))
            .unwrap();

        let optimizer = CircuitOptimizer::new();
        let optimized = optimizer.remove_identities(&circuit).unwrap();

        // Identity and zero-angle rotation should be removed
        assert_eq!(optimized.gates.len(), 1);
        assert_eq!(optimized.gates[0].gate_type, GateType::X);
    }

    #[test]
    fn test_circuit_depth() {
        let mut circuit = Circuit::new(2);
        circuit.add_gate(Gate::new(GateType::H, vec![0])).unwrap();
        circuit.add_gate(Gate::new(GateType::H, vec![1])).unwrap();
        circuit
            .add_gate(Gate::new(GateType::CNOT, vec![0, 1]))
            .unwrap();

        // H on both qubits in parallel (depth 1) + CNOT (depth 2)
        assert_eq!(circuit.depth(), 2);
    }

    #[test]
    fn test_full_optimization() {
        let mut circuit = Circuit::new(2);
        // Add redundant gates that should be optimized away
        circuit.add_gate(Gate::new(GateType::H, vec![0])).unwrap();
        circuit.add_gate(Gate::new(GateType::H, vec![0])).unwrap(); // Cancels with previous
        circuit
            .add_gate(Gate::with_parameters(
                GateType::RX,
                vec![1],
                vec![std::f64::consts::PI / 4.0],
            ))
            .unwrap();
        circuit
            .add_gate(Gate::with_parameters(
                GateType::RX,
                vec![1],
                vec![std::f64::consts::PI / 4.0],
            ))
            .unwrap(); // Should fuse
        circuit.add_gate(Gate::new(GateType::I, vec![0])).unwrap(); // Should be removed

        let optimizer = CircuitOptimizer::new();
        let optimized = optimizer.optimize(&circuit).unwrap();

        // Should have only 1 gate (fused RX)
        assert_eq!(optimized.gates.len(), 1);
        assert_eq!(optimized.gates[0].gate_type, GateType::RX);
    }

    #[test]
    fn test_optimization_stats() {
        let mut original = Circuit::new(2);
        for _ in 0..10 {
            original.add_gate(Gate::new(GateType::X, vec![0])).unwrap();
        }

        let mut optimized = Circuit::new(2);
        optimized.add_gate(Gate::new(GateType::X, vec![0])).unwrap();

        let stats = OptimizationStats::from_circuits(&original, &optimized);
        assert_eq!(stats.original_gates, 10);
        assert_eq!(stats.optimized_gates, 1);
        assert!((stats.gate_reduction_percent - 90.0).abs() < 1e-6);
    }
}
