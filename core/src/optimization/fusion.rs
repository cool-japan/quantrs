//! Gate fusion optimization pass
//!
//! This module implements gate fusion, which combines adjacent compatible gates
//! into single operations to reduce circuit depth and improve performance.

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::gate::{multi::*, single::*, GateOp};
use crate::synthesis::{identify_gate, synthesize_unitary};
use scirs2_core::ndarray::Array2;

use super::OptimizationPass;

/// Gate fusion optimization pass
pub struct GateFusion {
    /// Whether to fuse single-qubit gates
    pub fuse_single_qubit: bool,
    /// Whether to fuse two-qubit gates
    pub fuse_two_qubit: bool,
    /// Maximum number of gates to fuse together
    pub max_fusion_size: usize,
    /// Tolerance for gate identification
    pub tolerance: f64,
}

impl Default for GateFusion {
    fn default() -> Self {
        Self {
            fuse_single_qubit: true,
            fuse_two_qubit: true,
            max_fusion_size: 4,
            tolerance: 1e-10,
        }
    }
}

impl GateFusion {
    /// Create a new gate fusion pass
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to fuse a sequence of single-qubit gates
    fn fuse_single_qubit_gates(
        &self,
        gates: &[Box<dyn GateOp>],
    ) -> QuantRS2Result<Option<Box<dyn GateOp>>> {
        if gates.is_empty() {
            return Ok(None);
        }

        // Check all gates act on the same qubit
        let target_qubit = gates[0].qubits()[0];
        if !gates
            .iter()
            .all(|g| g.qubits().len() == 1 && g.qubits()[0] == target_qubit)
        {
            return Ok(None);
        }

        // Compute the combined unitary matrix
        let mut combined = Array2::eye(2);
        for gate in gates {
            let gate_matrix = gate.matrix()?;
            let gate_array = Array2::from_shape_vec((2, 2), gate_matrix)
                .map_err(|e| QuantRS2Error::InvalidInput(e.to_string()))?;
            combined = combined.dot(&gate_array);
        }

        // Try to identify the combined gate
        if let Some(gate_name) = identify_gate(&combined.view(), self.tolerance) {
            // Convert identified gate name to actual gate
            let identified_gate = match gate_name.as_str() {
                "X" => Some(Box::new(PauliX {
                    target: target_qubit,
                }) as Box<dyn GateOp>),
                "Y" => Some(Box::new(PauliY {
                    target: target_qubit,
                }) as Box<dyn GateOp>),
                "Z" => Some(Box::new(PauliZ {
                    target: target_qubit,
                }) as Box<dyn GateOp>),
                "H" => Some(Box::new(Hadamard {
                    target: target_qubit,
                }) as Box<dyn GateOp>),
                "S" => Some(Box::new(Phase {
                    target: target_qubit,
                }) as Box<dyn GateOp>),
                "S†" => Some(Box::new(PhaseDagger {
                    target: target_qubit,
                }) as Box<dyn GateOp>),
                "T" => Some(Box::new(T {
                    target: target_qubit,
                }) as Box<dyn GateOp>),
                "T†" => Some(Box::new(TDagger {
                    target: target_qubit,
                }) as Box<dyn GateOp>),
                "I" | _ => None, // Identity or unknown
            };

            if let Some(gate) = identified_gate {
                return Ok(Some(gate));
            }
        }

        // If we can't identify it, synthesize it
        let synthesized = synthesize_unitary(&combined.view(), &[target_qubit])?;
        if synthesized.len() < gates.len() {
            // Only use synthesis if it reduces gate count
            if synthesized.len() == 1 {
                Ok(synthesized.into_iter().next())
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Try to fuse CNOT gates
    fn fuse_cnot_gates(
        &self,
        gates: &[Box<dyn GateOp>],
    ) -> QuantRS2Result<Option<Vec<Box<dyn GateOp>>>> {
        if gates.len() < 2 {
            return Ok(None);
        }

        let mut fused = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            if i + 1 < gates.len() {
                if let (Some(cnot1), Some(cnot2)) = (
                    gates[i].as_any().downcast_ref::<CNOT>(),
                    gates[i + 1].as_any().downcast_ref::<CNOT>(),
                ) {
                    // Two CNOTs with same control and target cancel
                    if cnot1.control == cnot2.control && cnot1.target == cnot2.target {
                        // Skip both gates
                        i += 2;
                        continue;
                    }
                    // CNOT(a,b) followed by CNOT(b,a) is a SWAP
                    else if cnot1.control == cnot2.target && cnot1.target == cnot2.control {
                        fused.push(Box::new(SWAP {
                            qubit1: cnot1.control,
                            qubit2: cnot1.target,
                        }) as Box<dyn GateOp>);
                        i += 2;
                        continue;
                    }
                }
            }

            // No fusion possible, keep the gate
            fused.push(gates[i].clone_gate());
            i += 1;
        }

        if fused.len() < gates.len() {
            Ok(Some(fused))
        } else {
            Ok(None)
        }
    }

    /// Try to fuse rotation gates
    fn fuse_rotation_gates(
        &self,
        gates: &[Box<dyn GateOp>],
    ) -> QuantRS2Result<Option<Box<dyn GateOp>>> {
        if gates.len() < 2 {
            return Ok(None);
        }

        // Check if all gates are rotations around the same axis on the same qubit
        let first_gate = &gates[0];
        let target_qubit = first_gate.qubits()[0];

        match first_gate.name() {
            "RX" => {
                let mut total_angle = 0.0;
                for gate in gates {
                    if let Some(rx) = gate.as_any().downcast_ref::<RotationX>() {
                        if rx.target != target_qubit {
                            return Ok(None);
                        }
                        total_angle += rx.theta;
                    } else {
                        return Ok(None);
                    }
                }
                Ok(Some(Box::new(RotationX {
                    target: target_qubit,
                    theta: total_angle,
                })))
            }
            "RY" => {
                let mut total_angle = 0.0;
                for gate in gates {
                    if let Some(ry) = gate.as_any().downcast_ref::<RotationY>() {
                        if ry.target != target_qubit {
                            return Ok(None);
                        }
                        total_angle += ry.theta;
                    } else {
                        return Ok(None);
                    }
                }
                Ok(Some(Box::new(RotationY {
                    target: target_qubit,
                    theta: total_angle,
                })))
            }
            "RZ" => {
                let mut total_angle = 0.0;
                for gate in gates {
                    if let Some(rz) = gate.as_any().downcast_ref::<RotationZ>() {
                        if rz.target != target_qubit {
                            return Ok(None);
                        }
                        total_angle += rz.theta;
                    } else {
                        return Ok(None);
                    }
                }
                Ok(Some(Box::new(RotationZ {
                    target: target_qubit,
                    theta: total_angle,
                })))
            }
            _ => Ok(None),
        }
    }

    /// Find fusable gate sequences
    fn find_fusable_sequences(&self, gates: &[Box<dyn GateOp>]) -> Vec<(usize, usize)> {
        let mut sequences = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            // For single-qubit gates, find consecutive gates on same qubit
            if gates[i].qubits().len() == 1 {
                let target_qubit = gates[i].qubits()[0];
                let mut j = i + 1;

                while j < gates.len() && j - i < self.max_fusion_size {
                    if gates[j].qubits().len() == 1 && gates[j].qubits()[0] == target_qubit {
                        j += 1;
                    } else {
                        break;
                    }
                }

                if j > i + 1 {
                    sequences.push((i, j));
                    i = j;
                    continue;
                }
            }

            // For multi-qubit gates, look for specific patterns
            if gates[i].name() == "CNOT" && i + 1 < gates.len() && gates[i + 1].name() == "CNOT" {
                sequences.push((i, i + 2));
                i += 2;
                continue;
            }

            i += 1;
        }

        sequences
    }
}

impl OptimizationPass for GateFusion {
    fn optimize(&self, gates: Vec<Box<dyn GateOp>>) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut optimized = Vec::new();
        let mut processed = vec![false; gates.len()];

        // Find fusable sequences
        let sequences = self.find_fusable_sequences(&gates);

        for (start, end) in sequences {
            let sequence = &gates[start..end];

            // Skip if already processed
            if processed[start] {
                continue;
            }

            // Try different fusion strategies
            let mut fused = false;

            // Try rotation fusion first (most specific)
            if let Some(fused_gate) = self.fuse_rotation_gates(sequence)? {
                optimized.push(fused_gate);
                fused = true;
            }
            // Try CNOT fusion
            else if sequence.iter().all(|g| g.name() == "CNOT") {
                if let Some(fused_gates) = self.fuse_cnot_gates(sequence)? {
                    optimized.extend(fused_gates);
                    fused = true;
                }
            }
            // Try general single-qubit fusion
            else if self.fuse_single_qubit && sequence.iter().all(|g| g.qubits().len() == 1) {
                if let Some(fused_gate) = self.fuse_single_qubit_gates(sequence)? {
                    optimized.push(fused_gate);
                    fused = true;
                }
            }

            // Mark as processed
            if fused {
                for i in start..end {
                    processed[i] = true;
                }
            }
        }

        // Add unfused gates
        for (i, gate) in gates.into_iter().enumerate() {
            if !processed[i] {
                optimized.push(gate);
            }
        }

        Ok(optimized)
    }

    fn name(&self) -> &'static str {
        "Gate Fusion"
    }
}

/// Specialized fusion for Clifford gates
pub struct CliffordFusion {
    #[allow(dead_code)]
    tolerance: f64,
}

impl CliffordFusion {
    pub const fn new() -> Self {
        Self { tolerance: 1e-10 }
    }

    /// Fuse adjacent Clifford gates
    fn fuse_clifford_pair(
        &self,
        gate1: &dyn GateOp,
        gate2: &dyn GateOp,
    ) -> QuantRS2Result<Option<Box<dyn GateOp>>> {
        // Only fuse if gates act on same qubit
        if gate1.qubits() != gate2.qubits() || gate1.qubits().len() != 1 {
            return Ok(None);
        }

        let qubit = gate1.qubits()[0];

        match (gate1.name(), gate2.name()) {
            // Self-inverse gates (H, X, Y, Z, S†S, SS†)
            ("H", "H") | ("X", "X") | ("Y", "Y") | ("Z", "Z") | ("S", "S†") | ("S†", "S") => {
                Ok(None) // Identity - will be removed
            }

            // S gate combinations & Pauli combinations resulting in Z
            ("S", "S") | ("S†", "S†") | ("X", "Y") | ("Y" | "H", "X") => {
                Ok(Some(Box::new(PauliZ { target: qubit }))) // SS/S†S†/XY/YX/HX → Z
            }

            // Pauli combinations
            ("X", "Z") | ("Z", "X") => Ok(Some(Box::new(PauliY { target: qubit }))), // XZ = -iY, ZX = iY
            ("Y" | "H", "Z") | ("Z", "Y") => {
                Ok(Some(Box::new(PauliX { target: qubit }))) // YZ/ZY/HZ → X
            }

            _ => Ok(None),
        }
    }
}

impl OptimizationPass for CliffordFusion {
    fn optimize(&self, gates: Vec<Box<dyn GateOp>>) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut optimized = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            if i + 1 < gates.len() {
                if let Some(fused) =
                    self.fuse_clifford_pair(gates[i].as_ref(), gates[i + 1].as_ref())?
                {
                    optimized.push(fused);
                    i += 2;
                    continue;
                } else if gates[i].qubits() == gates[i + 1].qubits() {
                    // Check if it's identity (would return None from fusion)
                    let combined_is_identity = match (gates[i].name(), gates[i + 1].name()) {
                        ("H", "H")
                        | ("S", "S†")
                        | ("S†", "S")
                        | ("X", "X")
                        | ("Y", "Y")
                        | ("Z", "Z") => true,
                        _ => false,
                    };

                    if combined_is_identity {
                        i += 2;
                        continue;
                    }
                }
            }

            optimized.push(gates[i].clone_gate());
            i += 1;
        }

        Ok(optimized)
    }

    fn name(&self) -> &'static str {
        "Clifford Fusion"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::single::{Hadamard, Phase};
    use crate::prelude::QubitId;

    #[test]
    fn test_rotation_fusion() {
        let fusion = GateFusion::new();
        let qubit = QubitId(0);

        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(RotationZ {
                target: qubit,
                theta: 0.5,
            }),
            Box::new(RotationZ {
                target: qubit,
                theta: 0.3,
            }),
            Box::new(RotationZ {
                target: qubit,
                theta: 0.2,
            }),
        ];

        let result = fusion
            .fuse_rotation_gates(&gates)
            .expect("Failed to fuse rotation gates");
        assert!(result.is_some());

        if let Some(rz) = result
            .expect("Expected a fused gate")
            .as_any()
            .downcast_ref::<RotationZ>()
        {
            assert!((rz.theta - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected RotationZ gate");
        }
    }

    #[test]
    fn test_cnot_cancellation() {
        let fusion = GateFusion::new();
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(CNOT {
                control: q0,
                target: q1,
            }),
            Box::new(CNOT {
                control: q0,
                target: q1,
            }),
        ];

        let result = fusion
            .fuse_cnot_gates(&gates)
            .expect("Failed to fuse CNOT gates");
        assert!(result.is_some());
        assert_eq!(result.expect("Expected fused gate list").len(), 0); // Should cancel
    }

    #[test]
    fn test_cnot_to_swap() {
        let fusion = GateFusion::new();
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(CNOT {
                control: q0,
                target: q1,
            }),
            Box::new(CNOT {
                control: q1,
                target: q0,
            }),
        ];

        let result = fusion
            .fuse_cnot_gates(&gates)
            .expect("Failed to fuse CNOT gates");
        assert!(result.is_some());
        let fused = result.expect("Expected fused gate list");
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].name(), "SWAP");
    }

    #[test]
    fn test_clifford_fusion() {
        let fusion = CliffordFusion::new();
        let qubit = QubitId(0);

        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: qubit }),
            Box::new(Hadamard { target: qubit }),
            Box::new(Phase { target: qubit }),
            Box::new(Phase { target: qubit }),
        ];

        let result = fusion
            .optimize(gates)
            .expect("Failed to optimize Clifford gates");
        // H*H cancels, S*S = Z
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name(), "Z");
    }

    #[test]
    fn test_full_optimization() {
        let mut chain = super::super::OptimizationChain::new();
        chain = chain
            .add_pass(Box::new(CliffordFusion::new()))
            .add_pass(Box::new(GateFusion::new()));

        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: q0 }),
            Box::new(Hadamard { target: q0 }),
            Box::new(CNOT {
                control: q0,
                target: q1,
            }),
            Box::new(CNOT {
                control: q0,
                target: q1,
            }),
            Box::new(RotationZ {
                target: q1,
                theta: 0.5,
            }),
            Box::new(RotationZ {
                target: q1,
                theta: 0.5,
            }),
        ];

        let result = chain
            .optimize(gates)
            .expect("Failed to optimize gate chain");

        // After CliffordFusion: CNOT, CNOT, RZ, RZ (H*H canceled)
        // After GateFusion: RZ (CNOT*CNOT canceled, RZ+RZ fused)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name(), "RZ");

        // Check the fused angle
        if let Some(rz) = result[0].as_any().downcast_ref::<RotationZ>() {
            assert!((rz.theta - 1.0).abs() < 1e-10);
        }
    }
}
