//! Controlled gate operations
//!
//! This module provides support for creating controlled versions of arbitrary quantum gates,
//! including multi-controlled gates and phase-controlled operations.

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::gate::GateOp;
use crate::qubit::QubitId;
use scirs2_core::Complex64;
use std::any::Any;
use std::fmt::Debug;

/// A controlled gate wrapper that adds control qubits to any gate
#[derive(Debug)]
pub struct ControlledGate {
    /// The control qubits
    controls: Vec<QubitId>,
    /// The base gate to be controlled
    base_gate: Box<dyn GateOp>,
    /// Phase to apply when all controls are |1⟩
    control_phase: Complex64,
}

impl ControlledGate {
    /// Create a new controlled gate
    pub fn new(controls: Vec<QubitId>, base_gate: Box<dyn GateOp>) -> Self {
        Self {
            controls,
            base_gate,
            control_phase: Complex64::new(1.0, 0.0),
        }
    }

    /// Create a new phase-controlled gate
    pub fn with_phase(
        controls: Vec<QubitId>,
        base_gate: Box<dyn GateOp>,
        phase: Complex64,
    ) -> Self {
        Self {
            controls,
            base_gate,
            control_phase: phase,
        }
    }

    /// Get the control qubits
    pub fn controls(&self) -> &[QubitId] {
        &self.controls
    }

    /// Get the target qubits
    pub fn targets(&self) -> Vec<QubitId> {
        self.base_gate.qubits()
    }

    /// Check if a qubit is a control
    pub fn is_control(&self, qubit: QubitId) -> bool {
        self.controls.contains(&qubit)
    }

    /// Check if a qubit is a target
    pub fn is_target(&self, qubit: QubitId) -> bool {
        self.base_gate.qubits().contains(&qubit)
    }
}

impl GateOp for ControlledGate {
    fn name(&self) -> &'static str {
        match self.controls.len() {
            1 => "C",
            2 => "CC",
            _ => "Multi-C",
        }
    }

    fn qubits(&self) -> Vec<QubitId> {
        let mut qubits = self.controls.clone();
        qubits.extend(self.base_gate.qubits());
        qubits
    }

    fn is_parameterized(&self) -> bool {
        self.base_gate.is_parameterized()
    }

    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        let base_matrix = self.base_gate.matrix()?;
        let base_dim = (base_matrix.len() as f64).sqrt() as usize;

        if base_dim * base_dim != base_matrix.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Base gate matrix is not square".to_string(),
            ));
        }

        let num_controls = self.controls.len();
        let control_dim = 1 << num_controls;
        let total_dim = control_dim * base_dim;

        // Create identity matrix of appropriate size
        let mut matrix = vec![Complex64::new(0.0, 0.0); total_dim * total_dim];
        for i in 0..total_dim {
            matrix[i * total_dim + i] = Complex64::new(1.0, 0.0);
        }

        // Apply base gate only when all controls are |1⟩
        let control_mask = control_dim - 1; // All controls = 1

        for i in 0..base_dim {
            for j in 0..base_dim {
                let row = control_mask * base_dim + i;
                let col = control_mask * base_dim + j;
                matrix[row * total_dim + col] = self.control_phase * base_matrix[i * base_dim + j];
            }
        }

        Ok(matrix)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(Self {
            controls: self.controls.clone(),
            base_gate: self.base_gate.clone(),
            control_phase: self.control_phase,
        })
    }
}

/// Multi-controlled gate with optimizations for common patterns
#[derive(Debug)]
pub struct MultiControlledGate {
    /// Positive controls (triggered on |1⟩)
    positive_controls: Vec<QubitId>,
    /// Negative controls (triggered on |0⟩)
    negative_controls: Vec<QubitId>,
    /// The base gate
    base_gate: Box<dyn GateOp>,
}

impl MultiControlledGate {
    /// Create a new multi-controlled gate
    pub fn new(
        positive_controls: Vec<QubitId>,
        negative_controls: Vec<QubitId>,
        base_gate: Box<dyn GateOp>,
    ) -> Self {
        Self {
            positive_controls,
            negative_controls,
            base_gate,
        }
    }

    /// Create with only positive controls
    pub fn positive(controls: Vec<QubitId>, base_gate: Box<dyn GateOp>) -> Self {
        Self {
            positive_controls: controls,
            negative_controls: vec![],
            base_gate,
        }
    }

    /// Get total number of control qubits
    pub fn num_controls(&self) -> usize {
        self.positive_controls.len() + self.negative_controls.len()
    }
}

impl GateOp for MultiControlledGate {
    fn name(&self) -> &'static str {
        match self.num_controls() {
            1 => "C",
            2 => "CC",
            3 => "CCC",
            _ => "Multi-C",
        }
    }

    fn qubits(&self) -> Vec<QubitId> {
        let mut qubits = self.positive_controls.clone();
        qubits.extend(&self.negative_controls);
        qubits.extend(self.base_gate.qubits());
        qubits
    }

    fn is_parameterized(&self) -> bool {
        self.base_gate.is_parameterized()
    }

    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        let base_matrix = self.base_gate.matrix()?;
        let base_dim = (base_matrix.len() as f64).sqrt() as usize;

        let num_controls = self.num_controls();
        let control_dim = 1 << num_controls;
        let total_dim = control_dim * base_dim;

        // Create identity matrix
        let mut matrix = vec![Complex64::new(0.0, 0.0); total_dim * total_dim];
        for i in 0..total_dim {
            matrix[i * total_dim + i] = Complex64::new(1.0, 0.0);
        }

        // Determine the control pattern
        let mut _control_pattern = 0usize;
        for (i, _) in self.positive_controls.iter().enumerate() {
            _control_pattern |= 1 << i;
        }
        // Negative controls start after positive controls
        let neg_offset = self.positive_controls.len();

        // Apply base gate only when control pattern matches
        for ctrl_state in 0..control_dim {
            let mut matches = true;

            // Check positive controls
            for (i, _) in self.positive_controls.iter().enumerate() {
                if (ctrl_state >> i) & 1 != 1 {
                    matches = false;
                    break;
                }
            }

            // Check negative controls
            if matches {
                for (i, _) in self.negative_controls.iter().enumerate() {
                    if (ctrl_state >> (neg_offset + i)) & 1 != 0 {
                        matches = false;
                        break;
                    }
                }
            }

            if matches {
                // Apply base gate to this control subspace
                for i in 0..base_dim {
                    for j in 0..base_dim {
                        let row = ctrl_state * base_dim + i;
                        let col = ctrl_state * base_dim + j;
                        matrix[row * total_dim + col] = base_matrix[i * base_dim + j];
                    }
                }
            }
        }

        Ok(matrix)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(Self {
            positive_controls: self.positive_controls.clone(),
            negative_controls: self.negative_controls.clone(),
            base_gate: self.base_gate.clone(),
        })
    }
}

/// Optimized Toffoli (CCNOT) gate
#[derive(Debug, Clone, Copy)]
pub struct ToffoliGate {
    control1: QubitId,
    control2: QubitId,
    target: QubitId,
}

impl ToffoliGate {
    /// Create a new Toffoli gate
    pub fn new(control1: QubitId, control2: QubitId, target: QubitId) -> Self {
        Self {
            control1,
            control2,
            target,
        }
    }
}

impl GateOp for ToffoliGate {
    fn name(&self) -> &'static str {
        "Toffoli"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.control1, self.control2, self.target]
    }

    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        // Toffoli matrix in computational basis
        let mut matrix = vec![Complex64::new(0.0, 0.0); 64]; // 8x8

        // Identity for most basis states
        for i in 0..8 {
            matrix[i * 8 + i] = Complex64::new(1.0, 0.0);
        }

        // Swap |110⟩ and |111⟩
        matrix[6 * 8 + 6] = Complex64::new(0.0, 0.0);
        matrix[7 * 8 + 7] = Complex64::new(0.0, 0.0);
        matrix[6 * 8 + 7] = Complex64::new(1.0, 0.0);
        matrix[7 * 8 + 6] = Complex64::new(1.0, 0.0);

        Ok(matrix)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(self.clone())
    }
}

/// Optimized Fredkin (CSWAP) gate
#[derive(Debug, Clone, Copy)]
pub struct FredkinGate {
    control: QubitId,
    target1: QubitId,
    target2: QubitId,
}

impl FredkinGate {
    /// Create a new Fredkin gate
    pub fn new(control: QubitId, target1: QubitId, target2: QubitId) -> Self {
        Self {
            control,
            target1,
            target2,
        }
    }
}

impl GateOp for FredkinGate {
    fn name(&self) -> &'static str {
        "Fredkin"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.control, self.target1, self.target2]
    }

    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        // Fredkin matrix in computational basis
        let mut matrix = vec![Complex64::new(0.0, 0.0); 64]; // 8x8

        // Identity for most basis states
        for i in 0..8 {
            matrix[i * 8 + i] = Complex64::new(1.0, 0.0);
        }

        // Swap when control is |1⟩
        // |101⟩ <-> |110⟩
        matrix[5 * 8 + 5] = Complex64::new(0.0, 0.0);
        matrix[6 * 8 + 6] = Complex64::new(0.0, 0.0);
        matrix[5 * 8 + 6] = Complex64::new(1.0, 0.0);
        matrix[6 * 8 + 5] = Complex64::new(1.0, 0.0);

        Ok(matrix)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(self.clone())
    }
}

/// Helper function to create controlled version of any gate
pub fn make_controlled<G: GateOp + 'static>(controls: Vec<QubitId>, gate: G) -> ControlledGate {
    ControlledGate::new(controls, Box::new(gate))
}

/// Helper function to create multi-controlled version with mixed controls
pub fn make_multi_controlled<G: GateOp + 'static>(
    positive_controls: Vec<QubitId>,
    negative_controls: Vec<QubitId>,
    gate: G,
) -> MultiControlledGate {
    MultiControlledGate::new(positive_controls, negative_controls, Box::new(gate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::single::PauliX;

    #[test]
    fn test_controlled_x_gate() {
        let x = PauliX { target: QubitId(1) };
        let cx = make_controlled(vec![QubitId(0)], x);

        assert_eq!(cx.name(), "C");
        assert_eq!(cx.qubits(), vec![QubitId(0), QubitId(1)]);

        let matrix = cx
            .matrix()
            .expect("Failed to get matrix in test_controlled_x_gate");
        assert_eq!(matrix.len(), 16); // 4x4 matrix

        // Check CNOT matrix structure
        assert_eq!(matrix[0], Complex64::new(1.0, 0.0)); // |00⟩ -> |00⟩
        assert_eq!(matrix[5], Complex64::new(1.0, 0.0)); // |01⟩ -> |01⟩
        assert_eq!(matrix[11], Complex64::new(1.0, 0.0)); // |10⟩ -> |11⟩
        assert_eq!(matrix[14], Complex64::new(1.0, 0.0)); // |11⟩ -> |10⟩
    }

    #[test]
    fn test_toffoli_gate() {
        let toffoli = ToffoliGate::new(QubitId(0), QubitId(1), QubitId(2));

        assert_eq!(toffoli.name(), "Toffoli");
        assert_eq!(toffoli.qubits().len(), 3);

        let matrix = toffoli
            .matrix()
            .expect("Failed to get matrix in test_toffoli_gate");
        assert_eq!(matrix.len(), 64); // 8x8 matrix
    }
}
