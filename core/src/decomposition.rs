use crate::error::QuantRS2Result;
use crate::gate::{multi, single, GateOp};
use crate::parametric::{
    ParametricGate, ParametricPhaseShift, ParametricRotationX, ParametricRotationY,
    ParametricRotationZ, ParametricU,
};
use crate::qubit::QubitId;
use num_complex::Complex64;
use std::any::Any;
use std::f64::consts::PI;

/// Trait for gate decomposition
pub trait GateDecomposable: GateOp {
    /// Decompose the gate into a sequence of simpler gates
    fn decompose(&self) -> QuantRS2Result<Vec<Box<dyn GateOp>>>;

    /// Check if the gate can be decomposed
    fn is_decomposable(&self) -> bool {
        true
    }
}

/// Trait for gate composition
pub trait GateComposable: GateOp {
    /// Compose this gate with another gate
    fn compose_with(&self, other: &dyn GateOp) -> QuantRS2Result<Box<dyn GateOp>>;

    /// Check if this gate can be composed with another gate
    fn can_compose_with(&self, other: &dyn GateOp) -> bool;
}

/// Decompose a SWAP gate into CNOTs
pub fn decompose_swap(qubit1: QubitId, qubit2: QubitId) -> Vec<Box<dyn GateOp>> {
    vec![
        Box::new(multi::CNOT {
            control: qubit1,
            target: qubit2,
        }),
        Box::new(multi::CNOT {
            control: qubit2,
            target: qubit1,
        }),
        Box::new(multi::CNOT {
            control: qubit1,
            target: qubit2,
        }),
    ]
}

/// Decompose a Toffoli (CCNOT) gate
pub fn decompose_toffoli(
    control1: QubitId,
    control2: QubitId,
    target: QubitId,
) -> Vec<Box<dyn GateOp>> {
    vec![
        Box::new(single::RotationY {
            target,
            theta: PI / 4.0,
        }),
        Box::new(multi::CNOT {
            control: control2,
            target,
        }),
        Box::new(single::RotationY {
            target,
            theta: -PI / 4.0,
        }),
        Box::new(multi::CNOT {
            control: control1,
            target,
        }),
        Box::new(single::RotationY {
            target,
            theta: PI / 4.0,
        }),
        Box::new(multi::CNOT {
            control: control2,
            target,
        }),
        Box::new(single::RotationY {
            target,
            theta: -PI / 4.0,
        }),
        Box::new(multi::CNOT {
            control: control1,
            target: control2,
        }),
        Box::new(single::RotationY {
            target: control2,
            theta: PI / 4.0,
        }),
        Box::new(multi::CNOT {
            control: control1,
            target: control2,
        }),
        Box::new(single::RotationY {
            target: control2,
            theta: -PI / 4.0,
        }),
        Box::new(single::Hadamard { target: control1 }),
        Box::new(single::Hadamard { target: control2 }),
        Box::new(single::Hadamard { target }),
        Box::new(single::RotationX {
            target: control1,
            theta: PI / 4.0,
        }),
        Box::new(single::RotationX {
            target: control2,
            theta: PI / 4.0,
        }),
        Box::new(single::RotationX {
            target,
            theta: PI / 4.0,
        }),
    ]
}

/// Decompose a Fredkin (CSWAP) gate
pub fn decompose_fredkin(
    control: QubitId,
    target1: QubitId,
    target2: QubitId,
) -> Vec<Box<dyn GateOp>> {
    // First, we implement CSWAP using Toffoli
    let mut gates: Vec<Box<dyn GateOp>> = Vec::new();

    // CSWAP can be implemented using:
    // CNOT(t2, t1)
    // CCNOT(c, t1, t2)
    // CNOT(t2, t1)

    gates.push(Box::new(multi::CNOT {
        control: target2,
        target: target1,
    }));

    // Add decomposed Toffoli
    let toffoli_gates = decompose_toffoli(control, target1, target2);
    gates.extend(toffoli_gates);

    gates.push(Box::new(multi::CNOT {
        control: target2,
        target: target1,
    }));

    gates
}

/// Decompose a controlled-rotation gate into single-qubit and CNOT gates
pub fn decompose_controlled_rotation(
    control: QubitId,
    target: QubitId,
    gate_type: &str,
    theta: f64,
) -> Vec<Box<dyn GateOp>> {
    let mut gates: Vec<Box<dyn GateOp>> = Vec::new();

    match gate_type {
        "RX" => {
            // Decompose CRX using:
            // 1. Rz(target, π/2)
            // 2. CNOT(control, target)
            // 3. Ry(target, -θ/2)
            // 4. CNOT(control, target)
            // 5. Ry(target, θ/2)
            // 6. Rz(target, -π/2)
            gates.push(Box::new(single::RotationZ {
                target,
                theta: PI / 2.0,
            }));
            gates.push(Box::new(multi::CNOT { control, target }));
            gates.push(Box::new(single::RotationY {
                target,
                theta: -theta / 2.0,
            }));
            gates.push(Box::new(multi::CNOT { control, target }));
            gates.push(Box::new(single::RotationY {
                target,
                theta: theta / 2.0,
            }));
            gates.push(Box::new(single::RotationZ {
                target,
                theta: -PI / 2.0,
            }));
        }
        "RY" => {
            // Decompose CRY using:
            // 1. CNOT(control, target)
            // 2. Ry(target, -θ/2)
            // 3. CNOT(control, target)
            // 4. Ry(target, θ/2)
            gates.push(Box::new(multi::CNOT { control, target }));
            gates.push(Box::new(single::RotationY {
                target,
                theta: -theta / 2.0,
            }));
            gates.push(Box::new(multi::CNOT { control, target }));
            gates.push(Box::new(single::RotationY {
                target,
                theta: theta / 2.0,
            }));
        }
        "RZ" => {
            // Decompose CRZ using:
            // 1. Rz(target, θ/2)
            // 2. CNOT(control, target)
            // 3. Rz(target, -θ/2)
            // 4. CNOT(control, target)
            gates.push(Box::new(single::RotationZ {
                target,
                theta: theta / 2.0,
            }));
            gates.push(Box::new(multi::CNOT { control, target }));
            gates.push(Box::new(single::RotationZ {
                target,
                theta: -theta / 2.0,
            }));
            gates.push(Box::new(multi::CNOT { control, target }));
        }
        _ => {
            // Default to identity
        }
    }

    gates
}

/// Decompose a U gate into RZ-RY-RZ Euler angle decomposition
pub fn decompose_u_gate(
    target: QubitId,
    theta: f64,
    phi: f64,
    lambda: f64,
) -> Vec<Box<dyn GateOp>> {
    vec![
        Box::new(single::RotationZ {
            target,
            theta: lambda,
        }),
        Box::new(single::RotationY { target, theta }),
        Box::new(single::RotationZ { target, theta: phi }),
    ]
}

/// A generic composite gate that combines multiple gates into one
#[derive(Debug)]
pub struct CompositeGate {
    /// The sequence of gates in this composite
    pub gates: Vec<Box<dyn GateOp>>,

    /// Qubits this gate acts on
    pub qubits: Vec<QubitId>,

    /// Optional name for the composite gate
    pub name: String,
}

impl GateOp for CompositeGate {
    fn name(&self) -> &'static str {
        "Composite" // This is a static string, but we have a name field for dynamic naming
    }

    fn qubits(&self) -> Vec<QubitId> {
        self.qubits.clone()
    }

    fn is_parameterized(&self) -> bool {
        // If any contained gate is parameterized, this composite is parameterized
        self.gates.iter().any(|g| g.is_parameterized())
    }

    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        Err(crate::error::QuantRS2Error::UnsupportedOperation(
            "Direct matrix representation of composite gates not supported. \
             Use gate decomposition."
                .into(),
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl GateDecomposable for CompositeGate {
    fn decompose(&self) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // Create a new vector of trait objects
        let mut result: Vec<Box<dyn GateOp>> = Vec::with_capacity(self.gates.len());
        for gate in &self.gates {
            // We create a new box with a clone of the specific gate type
            // This works because we're cloning the concrete type, not the trait object
            if let Some(h) = gate.as_any().downcast_ref::<single::Hadamard>() {
                result.push(Box::new(single::Hadamard { target: h.target }) as Box<dyn GateOp>);
            } else if let Some(x) = gate.as_any().downcast_ref::<single::PauliX>() {
                result.push(Box::new(single::PauliX { target: x.target }) as Box<dyn GateOp>);
            } else if let Some(y) = gate.as_any().downcast_ref::<single::PauliY>() {
                result.push(Box::new(single::PauliY { target: y.target }) as Box<dyn GateOp>);
            } else if let Some(z) = gate.as_any().downcast_ref::<single::PauliZ>() {
                result.push(Box::new(single::PauliZ { target: z.target }) as Box<dyn GateOp>);
            } else if let Some(cnot) = gate.as_any().downcast_ref::<multi::CNOT>() {
                result.push(Box::new(multi::CNOT {
                    control: cnot.control,
                    target: cnot.target,
                }) as Box<dyn GateOp>);
            } else {
                // For other gate types, we'd need to add them here
                // As a fallback, we'll use a message gate to indicate the issue
                return Err(crate::error::QuantRS2Error::UnsupportedOperation(format!(
                    "Gate type {} not supported for decomposition cloning",
                    gate.name()
                )));
            }
        }
        Ok(result)
    }

    fn is_decomposable(&self) -> bool {
        true
    }
}

/// Implementation of GateDecomposable for Toffoli gate
impl GateDecomposable for multi::Toffoli {
    fn decompose(&self) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        Ok(decompose_toffoli(self.control1, self.control2, self.target))
    }
}

/// Implementation of GateDecomposable for Fredkin gate
impl GateDecomposable for multi::Fredkin {
    fn decompose(&self) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        Ok(decompose_fredkin(self.control, self.target1, self.target2))
    }
}

/// Implementation of GateDecomposable for SWAP gate
impl GateDecomposable for multi::SWAP {
    fn decompose(&self) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        Ok(decompose_swap(self.qubit1, self.qubit2))
    }
}

/// Implementation of GateDecomposable for CRX gate
impl GateDecomposable for multi::CRX {
    fn decompose(&self) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        Ok(decompose_controlled_rotation(
            self.control,
            self.target,
            "RX",
            self.theta,
        ))
    }
}

/// Implementation of GateDecomposable for CRY gate
impl GateDecomposable for multi::CRY {
    fn decompose(&self) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        Ok(decompose_controlled_rotation(
            self.control,
            self.target,
            "RY",
            self.theta,
        ))
    }
}

/// Implementation of GateDecomposable for CRZ gate
impl GateDecomposable for multi::CRZ {
    fn decompose(&self) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        Ok(decompose_controlled_rotation(
            self.control,
            self.target,
            "RZ",
            self.theta,
        ))
    }
}

/// Utility module for gate composition and decomposition
pub mod utils {
    use super::*;

    /// Type alias for a sequence of gates
    pub type GateSequence = Vec<Box<dyn GateOp>>;

    /// Clone a gate by creating a new instance of the same concrete type
    pub fn clone_gate(gate: &dyn GateOp) -> QuantRS2Result<Box<dyn GateOp>> {
        if let Some(h) = gate.as_any().downcast_ref::<single::Hadamard>() {
            Ok(Box::new(single::Hadamard { target: h.target }))
        } else if let Some(x) = gate.as_any().downcast_ref::<single::PauliX>() {
            Ok(Box::new(single::PauliX { target: x.target }))
        } else if let Some(y) = gate.as_any().downcast_ref::<single::PauliY>() {
            Ok(Box::new(single::PauliY { target: y.target }))
        } else if let Some(z) = gate.as_any().downcast_ref::<single::PauliZ>() {
            Ok(Box::new(single::PauliZ { target: z.target }))
        } else if let Some(cnot) = gate.as_any().downcast_ref::<multi::CNOT>() {
            Ok(Box::new(multi::CNOT {
                control: cnot.control,
                target: cnot.target,
            }))
        } else if let Some(rx) = gate.as_any().downcast_ref::<single::RotationX>() {
            Ok(Box::new(single::RotationX {
                target: rx.target,
                theta: rx.theta,
            }))
        } else if let Some(ry) = gate.as_any().downcast_ref::<single::RotationY>() {
            Ok(Box::new(single::RotationY {
                target: ry.target,
                theta: ry.theta,
            }))
        } else if let Some(rz) = gate.as_any().downcast_ref::<single::RotationZ>() {
            Ok(Box::new(single::RotationZ {
                target: rz.target,
                theta: rz.theta,
            }))
        } else if let Some(s) = gate.as_any().downcast_ref::<single::Phase>() {
            Ok(Box::new(single::Phase { target: s.target }))
        } else if let Some(t) = gate.as_any().downcast_ref::<single::T>() {
            Ok(Box::new(single::T { target: t.target }))
        } else if let Some(swap) = gate.as_any().downcast_ref::<multi::SWAP>() {
            Ok(Box::new(multi::SWAP {
                qubit1: swap.qubit1,
                qubit2: swap.qubit2,
            }))
        } else {
            // For unsupported gate types, return an error
            Err(crate::error::QuantRS2Error::UnsupportedOperation(format!(
                "Gate type {} not supported for cloning",
                gate.name()
            )))
        }
    }

    /// Try to optimize a sequence of gates
    pub fn optimize_gate_sequence(gates: &[Box<dyn GateOp>]) -> QuantRS2Result<GateSequence> {
        let mut result: Vec<Box<dyn GateOp>> = Vec::new();
        let mut i = 0;

        while i < gates.len() {
            // Check for cancellations (e.g., H-H, X-X)
            if i + 1 < gates.len() {
                let gate1 = &gates[i];
                let gate2 = &gates[i + 1];

                if let Some(_) = try_cancel_gates(gate1.as_ref(), gate2.as_ref()) {
                    // Gates cancel each other, skip both
                    i += 2;
                    continue;
                }

                if let Some(composed) = try_compose_gates(gate1.as_ref(), gate2.as_ref()) {
                    // Gates can be composed, add the composed gate
                    result.push(composed);
                    i += 2;
                    continue;
                }
            }

            // No optimization found, add the gate as-is by creating a new instance
            // based on the concrete type of the gate
            let gate = &gates[i];
            if let Some(h) = gate.as_any().downcast_ref::<single::Hadamard>() {
                result.push(Box::new(single::Hadamard { target: h.target }) as Box<dyn GateOp>);
            } else if let Some(x) = gate.as_any().downcast_ref::<single::PauliX>() {
                result.push(Box::new(single::PauliX { target: x.target }) as Box<dyn GateOp>);
            } else if let Some(y) = gate.as_any().downcast_ref::<single::PauliY>() {
                result.push(Box::new(single::PauliY { target: y.target }) as Box<dyn GateOp>);
            } else if let Some(z) = gate.as_any().downcast_ref::<single::PauliZ>() {
                result.push(Box::new(single::PauliZ { target: z.target }) as Box<dyn GateOp>);
            } else if let Some(cnot) = gate.as_any().downcast_ref::<multi::CNOT>() {
                result.push(Box::new(multi::CNOT {
                    control: cnot.control,
                    target: cnot.target,
                }) as Box<dyn GateOp>);
            } else {
                // For unsupported gate types, return an error
                return Err(crate::error::QuantRS2Error::UnsupportedOperation(format!(
                    "Gate type {} not supported for optimization cloning",
                    gate.name()
                )));
            }
            i += 1;
        }

        Ok(result)
    }

    /// Try to cancel two gates
    fn try_cancel_gates(gate1: &dyn GateOp, gate2: &dyn GateOp) -> Option<()> {
        // Check if both gates are the same type and act on the same qubits
        if gate1.name() == gate2.name() && gate1.qubits() == gate2.qubits() {
            match gate1.name() {
                // Self-inverse gates that cancel each other when applied twice
                "H" | "X" | "Y" | "Z" | "SWAP" => {
                    return Some(());
                }
                // Gates that are not self-inverse need special handling
                "S" => {
                    if gate2.name() == "S†" {
                        return Some(());
                    }
                }
                "S†" => {
                    if gate2.name() == "S" {
                        return Some(());
                    }
                }
                "T" => {
                    if gate2.name() == "T†" {
                        return Some(());
                    }
                }
                "T†" => {
                    if gate2.name() == "T" {
                        return Some(());
                    }
                }
                // Rotation gates might cancel if they sum to a multiple of 2π
                "RX" | "RY" | "RZ" => {
                    // Would need to check rotation angles for exact cancellation
                }
                _ => {}
            }
        }

        None
    }

    /// Try to compose two gates into a single gate
    fn try_compose_gates(gate1: &dyn GateOp, gate2: &dyn GateOp) -> Option<Box<dyn GateOp>> {
        // Check if both gates are the same type and act on the same qubits
        if gate1.qubits() == gate2.qubits() {
            match (gate1.name(), gate2.name()) {
                // Compose rotation gates
                ("RX", "RX") => {
                    if let (Some(rx1), Some(rx2)) = (
                        gate1.as_any().downcast_ref::<single::RotationX>(),
                        gate2.as_any().downcast_ref::<single::RotationX>(),
                    ) {
                        return Some(Box::new(single::RotationX {
                            target: rx1.target,
                            theta: rx1.theta + rx2.theta,
                        }));
                    }
                }
                ("RY", "RY") => {
                    if let (Some(ry1), Some(ry2)) = (
                        gate1.as_any().downcast_ref::<single::RotationY>(),
                        gate2.as_any().downcast_ref::<single::RotationY>(),
                    ) {
                        return Some(Box::new(single::RotationY {
                            target: ry1.target,
                            theta: ry1.theta + ry2.theta,
                        }));
                    }
                }
                ("RZ", "RZ") => {
                    if let (Some(rz1), Some(rz2)) = (
                        gate1.as_any().downcast_ref::<single::RotationZ>(),
                        gate2.as_any().downcast_ref::<single::RotationZ>(),
                    ) {
                        return Some(Box::new(single::RotationZ {
                            target: rz1.target,
                            theta: rz1.theta + rz2.theta,
                        }));
                    }
                }
                // Add more gate compositions as needed
                _ => {}
            }
        }

        None
    }

    /// Decompose a circuit into a sequence of standard gates
    pub fn decompose_circuit(gates: &[Box<dyn GateOp>]) -> QuantRS2Result<GateSequence> {
        let mut result: Vec<Box<dyn GateOp>> = Vec::new();

        for gate in gates {
            // Try to downcast to specific gate types and then handle decomposition
            if let Some(toff) = gate.as_any().downcast_ref::<multi::Toffoli>() {
                // Decompose Toffoli
                let decomposed = decompose_toffoli(toff.control1, toff.control2, toff.target);
                // Recursively decompose
                let fully_decomposed = decompose_circuit(&decomposed)?;
                result.extend(fully_decomposed);
            } else if let Some(fred) = gate.as_any().downcast_ref::<multi::Fredkin>() {
                // Decompose Fredkin
                let decomposed = decompose_fredkin(fred.control, fred.target1, fred.target2);
                // Recursively decompose
                let fully_decomposed = decompose_circuit(&decomposed)?;
                result.extend(fully_decomposed);
            } else if let Some(swap) = gate.as_any().downcast_ref::<multi::SWAP>() {
                // Decompose SWAP
                let decomposed = decompose_swap(swap.qubit1, swap.qubit2);
                // Recursively decompose
                let fully_decomposed = decompose_circuit(&decomposed)?;
                result.extend(fully_decomposed);
            } else if let Some(crx) = gate.as_any().downcast_ref::<multi::CRX>() {
                // Decompose CRX
                let decomposed =
                    decompose_controlled_rotation(crx.control, crx.target, "RX", crx.theta);
                // Recursively decompose
                let fully_decomposed = decompose_circuit(&decomposed)?;
                result.extend(fully_decomposed);
            } else if let Some(h) = gate.as_any().downcast_ref::<single::Hadamard>() {
                // Basic gates don't need decomposition, just copy them
                result.push(Box::new(single::Hadamard { target: h.target }) as Box<dyn GateOp>);
            } else if let Some(x) = gate.as_any().downcast_ref::<single::PauliX>() {
                result.push(Box::new(single::PauliX { target: x.target }) as Box<dyn GateOp>);
            } else if let Some(cnot) = gate.as_any().downcast_ref::<multi::CNOT>() {
                result.push(Box::new(multi::CNOT {
                    control: cnot.control,
                    target: cnot.target,
                }) as Box<dyn GateOp>);
            } else {
                // For unsupported gate types, return an error
                return Err(crate::error::QuantRS2Error::UnsupportedOperation(format!(
                    "Gate type {} not supported for decomposition",
                    gate.name()
                )));
            }
        }

        Ok(result)
    }
}
