//! Hardware-efficient ansätze for variational quantum algorithms
//!
//! This module provides pre-built ansatz templates commonly used in VQE, QAOA, and other
//! variational quantum algorithms. These templates are compatible with TorchQuantum.
//!
//! # Ansätze
//!
//! - **RealAmplitudesLayer**: RY rotations with CNOTs (real-valued amplitudes)
//! - **EfficientSU2Layer**: RY + RZ rotations with CNOTs (efficient SU(2) decomposition)
//! - **TwoLocalLayer**: Configurable two-local circuit (rotation + entanglement)
//! - **ExcitationPreservingLayer**: Particle-number conserving ansatz for chemistry
//!
//! # Entanglement Patterns
//!
//! - **Linear**: Qubit i entangles with qubit i+1
//! - **ReverseLinear**: Qubit i entangles with qubit i-1
//! - **Circular**: Linear with wraparound (periodic boundary)
//! - **Full**: All-to-all entanglement
//! - **Custom**: User-defined qubit pairs
//!
//! # Example
//!
//! ```rust
//! use quantrs2_ml::torchquantum::ansatz::{RealAmplitudesLayer, EntanglementPattern};
//! use quantrs2_ml::torchquantum::{TQDevice, TQModule};
//!
//! let mut qdev = TQDevice::new(4);
//! let mut ansatz = RealAmplitudesLayer::new(4, 2, EntanglementPattern::Linear);
//! ansatz.forward(&mut qdev).unwrap();
//! ```

use super::{
    gates::{TQRx, TQRy, TQRz, TQCNOT},
    CType, TQDevice, TQModule, TQOperator, TQParameter,
};
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, ArrayD};

/// Entanglement pattern for two-local circuits
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntanglementPattern {
    /// Linear entanglement: (0,1), (1,2), (2,3), ...
    Linear,
    /// Reverse linear: (n-1, n-2), (n-2, n-3), ...
    ReverseLinear,
    /// Circular: Linear with wraparound (0,1),...,(n-1,0)
    Circular,
    /// Full: All-to-all entanglement
    Full,
    /// Custom qubit pairs
    Custom(Vec<(usize, usize)>),
}

impl EntanglementPattern {
    /// Generate qubit pairs for this pattern
    pub fn generate_pairs(&self, n_wires: usize) -> Vec<(usize, usize)> {
        match self {
            EntanglementPattern::Linear => {
                (0..n_wires.saturating_sub(1)).map(|i| (i, i + 1)).collect()
            }
            EntanglementPattern::ReverseLinear => (1..n_wires).rev().map(|i| (i, i - 1)).collect(),
            EntanglementPattern::Circular => {
                let mut pairs: Vec<(usize, usize)> =
                    (0..n_wires.saturating_sub(1)).map(|i| (i, i + 1)).collect();
                if n_wires > 2 {
                    pairs.push((n_wires - 1, 0));
                }
                pairs
            }
            EntanglementPattern::Full => {
                let mut pairs = Vec::new();
                for i in 0..n_wires {
                    for j in (i + 1)..n_wires {
                        pairs.push((i, j));
                    }
                }
                pairs
            }
            EntanglementPattern::Custom(pairs) => pairs.clone(),
        }
    }
}

/// RealAmplitudes ansatz layer
///
/// Circuit structure per layer:
/// - RY rotations on all qubits
/// - CNOT entanglement according to pattern
///
/// This creates real-valued quantum states (amplitudes are real).
pub struct RealAmplitudesLayer {
    /// Number of qubits
    pub n_wires: usize,
    /// Number of repetitions
    pub reps: usize,
    /// Entanglement pattern
    pub entanglement: EntanglementPattern,
    /// RY rotation gates (n_wires per repetition)
    ry_gates: Vec<Vec<TQRy>>,
    /// Whether to include final rotation layer
    pub final_rotation: bool,
    static_mode: bool,
}

impl RealAmplitudesLayer {
    /// Create a new RealAmplitudes layer
    ///
    /// # Arguments
    ///
    /// * `n_wires` - Number of qubits
    /// * `reps` - Number of repetitions
    /// * `entanglement` - Entanglement pattern
    pub fn new(n_wires: usize, reps: usize, entanglement: EntanglementPattern) -> Self {
        // Create rotation gates for each repetition
        let mut ry_gates = Vec::new();
        for _ in 0..reps {
            let layer: Vec<TQRy> = (0..n_wires).map(|_| TQRy::new(true, true)).collect();
            ry_gates.push(layer);
        }

        // Add final rotation layer
        let final_layer: Vec<TQRy> = (0..n_wires).map(|_| TQRy::new(true, true)).collect();
        ry_gates.push(final_layer);

        Self {
            n_wires,
            reps,
            entanglement,
            ry_gates,
            final_rotation: true,
            static_mode: false,
        }
    }

    /// Create without final rotation layer
    pub fn without_final_rotation(mut self) -> Self {
        self.final_rotation = false;
        self
    }
}

impl TQModule for RealAmplitudesLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        let entanglement_pairs = self.entanglement.generate_pairs(self.n_wires);

        for rep in 0..self.reps {
            // Apply rotation layer
            for (wire, gate) in self.ry_gates[rep].iter_mut().enumerate() {
                gate.apply(qdev, &[wire])?;
            }

            // Apply entanglement layer
            for (control, target) in &entanglement_pairs {
                let mut cnot = TQCNOT::new();
                cnot.apply(qdev, &[*control, *target])?;
            }
        }

        // Final rotation layer (if enabled)
        if self.final_rotation && self.reps < self.ry_gates.len() {
            for (wire, gate) in self.ry_gates[self.reps].iter_mut().enumerate() {
                gate.apply(qdev, &[wire])?;
            }
        }

        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        let num_layers = if self.final_rotation {
            self.reps + 1
        } else {
            self.reps
        };

        self.ry_gates[..num_layers]
            .iter()
            .flat_map(|layer| layer.iter().flat_map(|g| g.parameters()))
            .collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
    }

    fn static_off(&mut self) {
        self.static_mode = false;
    }

    fn name(&self) -> &str {
        "RealAmplitudesLayer"
    }
}

/// EfficientSU2 ansatz layer
///
/// Circuit structure per layer:
/// - RY rotations on all qubits
/// - RZ rotations on all qubits
/// - CNOT entanglement according to pattern
///
/// This provides efficient SU(2) coverage with minimal gates.
pub struct EfficientSU2Layer {
    /// Number of qubits
    pub n_wires: usize,
    /// Number of repetitions
    pub reps: usize,
    /// Entanglement pattern
    pub entanglement: EntanglementPattern,
    /// RY rotation gates
    ry_gates: Vec<Vec<TQRy>>,
    /// RZ rotation gates
    rz_gates: Vec<Vec<TQRz>>,
    /// Whether to include final rotation layer
    pub final_rotation: bool,
    static_mode: bool,
}

impl EfficientSU2Layer {
    /// Create a new EfficientSU2 layer
    pub fn new(n_wires: usize, reps: usize, entanglement: EntanglementPattern) -> Self {
        let mut ry_gates = Vec::new();
        let mut rz_gates = Vec::new();

        for _ in 0..=reps {
            // RY layer
            let ry_layer: Vec<TQRy> = (0..n_wires).map(|_| TQRy::new(true, true)).collect();
            ry_gates.push(ry_layer);

            // RZ layer
            let rz_layer: Vec<TQRz> = (0..n_wires).map(|_| TQRz::new(true, true)).collect();
            rz_gates.push(rz_layer);
        }

        Self {
            n_wires,
            reps,
            entanglement,
            ry_gates,
            rz_gates,
            final_rotation: true,
            static_mode: false,
        }
    }

    pub fn without_final_rotation(mut self) -> Self {
        self.final_rotation = false;
        self
    }
}

impl TQModule for EfficientSU2Layer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        let entanglement_pairs = self.entanglement.generate_pairs(self.n_wires);

        for rep in 0..self.reps {
            // RY rotations
            for (wire, gate) in self.ry_gates[rep].iter_mut().enumerate() {
                gate.apply(qdev, &[wire])?;
            }

            // RZ rotations
            for (wire, gate) in self.rz_gates[rep].iter_mut().enumerate() {
                gate.apply(qdev, &[wire])?;
            }

            // Entanglement
            for (control, target) in &entanglement_pairs {
                let mut cnot = TQCNOT::new();
                cnot.apply(qdev, &[*control, *target])?;
            }
        }

        // Final rotation layer
        if self.final_rotation {
            for (wire, gate) in self.ry_gates[self.reps].iter_mut().enumerate() {
                gate.apply(qdev, &[wire])?;
            }
            for (wire, gate) in self.rz_gates[self.reps].iter_mut().enumerate() {
                gate.apply(qdev, &[wire])?;
            }
        }

        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        let num_layers = if self.final_rotation {
            self.reps + 1
        } else {
            self.reps
        };

        let ry_params = self.ry_gates[..num_layers]
            .iter()
            .flat_map(|layer| layer.iter().flat_map(|g| g.parameters()));

        let rz_params = self.rz_gates[..num_layers]
            .iter()
            .flat_map(|layer| layer.iter().flat_map(|g| g.parameters()));

        ry_params.chain(rz_params).collect()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
    }

    fn static_off(&mut self) {
        self.static_mode = false;
    }

    fn name(&self) -> &str {
        "EfficientSU2Layer"
    }
}

/// Rotation gate type for TwoLocalLayer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationType {
    RX,
    RY,
    RZ,
}

/// TwoLocal configurable ansatz layer
///
/// Flexible two-local circuit with configurable rotation gates.
pub struct TwoLocalLayer {
    /// Number of qubits
    pub n_wires: usize,
    /// Number of repetitions
    pub reps: usize,
    /// Rotation gate types
    pub rotation_gates: Vec<RotationType>,
    /// Entanglement pattern
    pub entanglement: EntanglementPattern,
    /// Parameter storage (flattened)
    parameters: Vec<TQParameter>,
    static_mode: bool,
}

impl TwoLocalLayer {
    /// Create a new TwoLocal layer
    pub fn new(
        n_wires: usize,
        reps: usize,
        rotation_gates: Vec<RotationType>,
        entanglement: EntanglementPattern,
    ) -> Self {
        // Calculate total number of parameters
        let params_per_rep = n_wires * rotation_gates.len();
        let total_params = (reps + 1) * params_per_rep;

        let parameters: Vec<TQParameter> = (0..total_params)
            .map(|i| {
                let param_data = ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[1, 1]));
                TQParameter::new(param_data, format!("param_{}", i))
            })
            .collect();

        Self {
            n_wires,
            reps,
            rotation_gates,
            entanglement,
            parameters,
            static_mode: false,
        }
    }
}

impl TQModule for TwoLocalLayer {
    fn forward(&mut self, qdev: &mut TQDevice) -> Result<()> {
        let entanglement_pairs = self.entanglement.generate_pairs(self.n_wires);
        let params_per_layer = self.n_wires * self.rotation_gates.len();

        for rep in 0..=self.reps {
            let param_offset = rep * params_per_layer;

            // Apply rotation layers
            for (rot_idx, rot_type) in self.rotation_gates.iter().enumerate() {
                for wire in 0..self.n_wires {
                    let param_idx = param_offset + rot_idx * self.n_wires + wire;
                    let param_val = if self.parameters[param_idx].data.len() > 0 {
                        self.parameters[param_idx].data[[0, 0]]
                    } else {
                        0.0
                    };

                    match rot_type {
                        RotationType::RX => {
                            let mut gate = TQRx::new(true, true);
                            gate.apply_with_params(qdev, &[wire], Some(&[param_val]))?;
                        }
                        RotationType::RY => {
                            let mut gate = TQRy::new(true, true);
                            gate.apply_with_params(qdev, &[wire], Some(&[param_val]))?;
                        }
                        RotationType::RZ => {
                            let mut gate = TQRz::new(true, true);
                            gate.apply_with_params(qdev, &[wire], Some(&[param_val]))?;
                        }
                    }
                }
            }

            // Apply entanglement (except after last rep)
            if rep < self.reps {
                for (control, target) in &entanglement_pairs {
                    let mut cnot = TQCNOT::new();
                    cnot.apply(qdev, &[*control, *target])?;
                }
            }
        }

        Ok(())
    }

    fn parameters(&self) -> Vec<TQParameter> {
        self.parameters.clone()
    }

    fn n_wires(&self) -> Option<usize> {
        Some(self.n_wires)
    }

    fn set_n_wires(&mut self, n_wires: usize) {
        self.n_wires = n_wires;
    }

    fn is_static_mode(&self) -> bool {
        self.static_mode
    }

    fn static_on(&mut self) {
        self.static_mode = true;
    }

    fn static_off(&mut self) {
        self.static_mode = false;
    }

    fn name(&self) -> &str {
        "TwoLocalLayer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entanglement_linear() {
        let pattern = EntanglementPattern::Linear;
        let pairs = pattern.generate_pairs(4);
        assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn test_entanglement_circular() {
        let pattern = EntanglementPattern::Circular;
        let pairs = pattern.generate_pairs(4);
        assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 3), (3, 0)]);
    }

    #[test]
    fn test_entanglement_full() {
        let pattern = EntanglementPattern::Full;
        let pairs = pattern.generate_pairs(3);
        assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn test_real_amplitudes_creation() {
        let layer = RealAmplitudesLayer::new(4, 2, EntanglementPattern::Linear);
        assert_eq!(layer.n_wires, 4);
        assert_eq!(layer.reps, 2);

        // Should have 3 layers of rotations (2 reps + 1 final)
        assert_eq!(layer.ry_gates.len(), 3);

        // Each layer has 4 gates
        assert_eq!(layer.ry_gates[0].len(), 4);
    }

    #[test]
    fn test_efficient_su2_creation() {
        let layer = EfficientSU2Layer::new(3, 2, EntanglementPattern::Circular);
        assert_eq!(layer.n_wires, 3);
        assert_eq!(layer.reps, 2);

        // Should have 3 layers of rotations
        assert_eq!(layer.ry_gates.len(), 3);
        assert_eq!(layer.rz_gates.len(), 3);
    }

    #[test]
    fn test_two_local_creation() {
        let rotations = vec![RotationType::RY, RotationType::RZ];
        let layer = TwoLocalLayer::new(4, 2, rotations, EntanglementPattern::Linear);

        assert_eq!(layer.n_wires, 4);
        assert_eq!(layer.reps, 2);

        // (reps + 1) * n_wires * rotation_gates.len()
        // (2 + 1) * 4 * 2 = 24 parameters
        assert_eq!(layer.parameters.len(), 24);
    }

    #[test]
    fn test_real_amplitudes_parameters() {
        let layer = RealAmplitudesLayer::new(3, 2, EntanglementPattern::Linear);

        // 3 layers (2 reps + 1 final) * 3 qubits = 9 parameters
        let params = layer.parameters();
        assert_eq!(params.len(), 9);
    }

    #[test]
    fn test_efficient_su2_parameters() {
        let layer = EfficientSU2Layer::new(3, 2, EntanglementPattern::Linear);

        // 3 layers * 3 qubits * 2 rotations (RY+RZ) = 18 parameters
        let params = layer.parameters();
        assert_eq!(params.len(), 18);
    }
}
