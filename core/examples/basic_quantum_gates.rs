//! Basic Quantum Gates Example
//!
//! This example demonstrates the fundamental quantum gate operations
//! available in QuantRS2-Core, including single-qubit and multi-qubit gates.
//!
//! Run with: cargo run --example basic_quantum_gates

use quantrs2_core::{
    error::QuantRS2Result,
    gate::{multi, single, GateOp},
    qubit::QubitId,
};
use scirs2_core::Complex64;

fn main() -> QuantRS2Result<()> {
    println!("=================================================================");
    println!("   QuantRS2-Core: Basic Quantum Gates");
    println!("=================================================================\n");

    // Demonstrate single-qubit gates
    demonstrate_single_qubit_gates()?;
    println!();

    // Demonstrate two-qubit gates
    demonstrate_two_qubit_gates();
    println!();

    // Demonstrate rotation gates
    demonstrate_rotation_gates();
    println!();

    // Demonstrate gate properties
    demonstrate_gate_properties()?;
    println!();

    println!("=================================================================");
    println!("   Example Complete!");
    println!("=================================================================");

    Ok(())
}

/// Demonstrate single-qubit Pauli and Hadamard gates
fn demonstrate_single_qubit_gates() -> QuantRS2Result<()> {
    println!("SINGLE-QUBIT GATES");
    println!("-----------------------------------------------------------------");

    // Hadamard gate
    let h_gate = single::Hadamard { target: QubitId(0) };
    println!("Hadamard Gate:");
    println!("  Name: {}", h_gate.name());
    println!("  Qubits: {:?}", h_gate.qubits());
    println!("  Creates superposition: |0⟩ → (|0⟩ + |1⟩)/√2");

    let h_matrix = h_gate.matrix()?;
    println!(
        "  Matrix (4 elements): [{:.3}, {:.3}, {:.3}, {:.3}]",
        h_matrix[0].re, h_matrix[1].re, h_matrix[2].re, h_matrix[3].re
    );

    // Pauli-X gate (quantum NOT)
    let x_gate = single::PauliX { target: QubitId(0) };
    println!("\nPauli-X Gate (NOT):");
    println!("  Name: {}", x_gate.name());
    println!("  Effect: |0⟩ → |1⟩, |1⟩ → |0⟩");

    // Pauli-Y gate
    let y_gate = single::PauliY { target: QubitId(1) };
    println!("\nPauli-Y Gate:");
    println!("  Name: {}", y_gate.name());
    println!("  Target qubit: {}", y_gate.qubits()[0].0);

    // Pauli-Z gate
    let z_gate = single::PauliZ { target: QubitId(0) };
    println!("\nPauli-Z Gate:");
    println!("  Name: {}", z_gate.name());
    println!("  Effect: Phase flip, |1⟩ → -|1⟩");

    // T gate (π/8 gate)
    let t_gate = single::T { target: QubitId(0) };
    println!("\nT Gate (π/8):");
    println!("  Name: {}", t_gate.name());
    println!("  Important for fault-tolerant quantum computing");

    println!("\n  ✓ Single-qubit gates demonstrated");

    Ok(())
}

/// Demonstrate two-qubit gates
fn demonstrate_two_qubit_gates() {
    println!("TWO-QUBIT GATES");
    println!("-----------------------------------------------------------------");

    // CNOT gate
    let cnot = multi::CNOT {
        control: QubitId(0),
        target: QubitId(1),
    };
    println!("CNOT Gate:");
    println!("  Name: {}", cnot.name());
    println!("  Control: {}, Target: {}", cnot.control.0, cnot.target.0);
    println!("  Creates entanglement when applied after Hadamard");
    println!("  |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |11⟩, |11⟩ → |10⟩");

    // CZ gate
    let cz = multi::CZ {
        control: QubitId(0),
        target: QubitId(1),
    };
    println!("\nControlled-Z Gate:");
    println!("  Name: {}", cz.name());
    println!("  Applies phase flip if both qubits are |1⟩");

    // SWAP gate
    let swap = multi::SWAP {
        qubit1: QubitId(0),
        qubit2: QubitId(1),
    };
    println!("\nSWAP Gate:");
    println!("  Name: {}", swap.name());
    println!("  Swaps the states of two qubits");
    println!("  |01⟩ → |10⟩, |10⟩ → |01⟩");

    // Toffoli gate (CCNOT)
    let toffoli = multi::Toffoli {
        control1: QubitId(0),
        control2: QubitId(1),
        target: QubitId(2),
    };
    println!("\nToffoli Gate (CCNOT):");
    println!("  Name: {}", toffoli.name());
    println!(
        "  Controls: {}, {}, Target: {}",
        toffoli.control1.0, toffoli.control2.0, toffoli.target.0
    );
    println!("  Flips target if both controls are |1⟩");
    println!("  Universal for classical computation");

    println!("\n  ✓ Two-qubit gates demonstrated");
}

/// Demonstrate rotation gates
fn demonstrate_rotation_gates() {
    println!("ROTATION GATES");
    println!("-----------------------------------------------------------------");

    let pi = std::f64::consts::PI;

    // Rotation around X-axis
    let rx_gate = single::RotationX {
        target: QubitId(0),
        theta: pi / 4.0,
    };
    println!("RX Gate (Rotation around X-axis):");
    println!("  Name: {}", rx_gate.name());
    println!("  Angle: π/4 ({:.4} radians)", rx_gate.theta);
    println!("  Parameterized: {}", rx_gate.is_parameterized());

    // Rotation around Y-axis
    let ry_gate = single::RotationY {
        target: QubitId(0),
        theta: pi / 2.0,
    };
    println!("\nRY Gate (Rotation around Y-axis):");
    println!("  Name: {}", ry_gate.name());
    println!("  Angle: π/2 ({:.4} radians)", ry_gate.theta);

    // Rotation around Z-axis
    let rz_gate = single::RotationZ {
        target: QubitId(0),
        theta: pi / 3.0,
    };
    println!("\nRZ Gate (Rotation around Z-axis):");
    println!("  Name: {}", rz_gate.name());
    println!("  Angle: π/3 ({:.4} radians)", rz_gate.theta);

    // Controlled rotation
    let crx_gate = multi::CRX {
        control: QubitId(0),
        target: QubitId(1),
        theta: pi / 4.0,
    };
    println!("\nControlled-RX Gate:");
    println!("  Name: {}", crx_gate.name());
    println!(
        "  Control: {}, Target: {}",
        crx_gate.control.0, crx_gate.target.0
    );
    println!("  Applies RX rotation conditionally");

    println!("\n  ✓ Rotation gates demonstrated");
}

/// Demonstrate gate properties and matrix representations
fn demonstrate_gate_properties() -> QuantRS2Result<()> {
    println!("GATE PROPERTIES");
    println!("-----------------------------------------------------------------");

    let h_gate = single::Hadamard { target: QubitId(0) };

    // Gate information
    println!("Gate: {}", h_gate.name());
    println!("Number of qubits: {}", h_gate.num_qubits());
    println!("Acts on qubits: {:?}", h_gate.qubits());
    println!("Is parameterized: {}", h_gate.is_parameterized());

    // Matrix representation
    let matrix = h_gate.matrix()?;
    println!("\nMatrix representation (2x2 gate → 4 elements):");
    println!("  [{:.3}  {:.3}]", matrix[0].re, matrix[1].re);
    println!("  [{:.3} {:.3}]", matrix[2].re, matrix[3].re);

    // Verify Hadamard is unitary (H² = I)
    println!("\nHadamard properties:");
    println!("  Self-inverse: H × H = I");
    println!("  Unitary: H† × H = I");
    println!("  Determinant: det(H) = -1");

    // Multi-qubit gate matrix
    let cnot = multi::CNOT {
        control: QubitId(0),
        target: QubitId(1),
    };
    let cnot_matrix = cnot.matrix()?;
    println!("\nCNOT matrix (4x4 gate → 16 elements):");
    println!("  Size: {} elements", cnot_matrix.len());
    println!("  First element: {:.3}", cnot_matrix[0].re);

    println!("\n  ✓ Gate properties demonstrated");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_runs() {
        assert!(main().is_ok());
    }

    #[test]
    fn test_hadamard_matrix() -> QuantRS2Result<()> {
        let h = single::Hadamard { target: QubitId(0) };
        let matrix = h.matrix()?;

        // Hadamard matrix elements should be ±1/√2
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        assert!((matrix[0].re - sqrt2_inv).abs() < 1e-10);
        assert!((matrix[1].re - sqrt2_inv).abs() < 1e-10);
        assert!((matrix[2].re - sqrt2_inv).abs() < 1e-10);
        assert!((matrix[3].re + sqrt2_inv).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_pauli_x_matrix() -> QuantRS2Result<()> {
        let x = single::PauliX { target: QubitId(0) };
        let matrix = x.matrix()?;

        // Pauli-X is [[0, 1], [1, 0]]
        assert_eq!(matrix[0].re, 0.0);
        assert_eq!(matrix[1].re, 1.0);
        assert_eq!(matrix[2].re, 1.0);
        assert_eq!(matrix[3].re, 0.0);

        Ok(())
    }

    #[test]
    fn test_cnot_gate() -> QuantRS2Result<()> {
        let cnot = multi::CNOT {
            control: QubitId(0),
            target: QubitId(1),
        };

        assert_eq!(cnot.name(), "CNOT");
        assert_eq!(cnot.num_qubits(), 2);

        let matrix = cnot.matrix()?;
        assert_eq!(matrix.len(), 16); // 4x4 matrix

        Ok(())
    }

    #[test]
    fn test_rotation_gates() -> QuantRS2Result<()> {
        let pi = std::f64::consts::PI;

        let rx = single::RotationX {
            target: QubitId(0),
            theta: pi / 4.0,
        };

        assert!(rx.is_parameterized());
        assert_eq!(rx.num_qubits(), 1);

        let matrix = rx.matrix()?;
        assert_eq!(matrix.len(), 4); // 2x2 matrix

        Ok(())
    }
}
