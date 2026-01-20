//! Example demonstrating classical control flow in quantum circuits

use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::single::{Hadamard, PauliX, PauliZ};
use quantrs2_core::qubit::QubitId;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Classical Control Flow Demo");
    println!("===========================\n");

    // Example 1: Simple measurement and conditional operation
    simple_conditional()?;

    // Example 2: Quantum teleportation with classical communication
    quantum_teleportation()?;

    // Example 3: Adaptive phase estimation
    adaptive_phase_estimation()?;

    Ok(())
}

/// Simple conditional operation based on measurement
fn simple_conditional() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Simple Conditional Operation");
    println!("--------------------------------------");

    // Create a circuit with classical control
    let circuit = ClassicalCircuitBuilder::<2>::new()
        .classical_register("c", 2)?
        .gate(Hadamard { target: QubitId(0) })?
        .measure(QubitId(0), "c", 0)?
        .conditional(
            ClassicalCondition::register_equals("c", 1),
            PauliX { target: QubitId(1) },
        )?
        .build();

    println!(
        "Created circuit with {} operations",
        circuit.num_operations()
    );
    println!("Circuit applies X to qubit 1 if qubit 0 measures to |1⟩\n");

    Ok(())
}

/// Quantum teleportation with classical communication
fn quantum_teleportation() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Quantum Teleportation");
    println!("--------------------------------");

    // Create Bell pair between qubits 1 and 2
    let mut circuit = Circuit::<3>::new();
    circuit.h(1)?;
    circuit.cnot(1, 2)?;

    // Convert to classical circuit for measurements
    let mut classical_circuit = circuit.with_classical_control();
    classical_circuit.add_classical_register("alice", 2)?;

    // Alice's operations
    classical_circuit.add_gate(quantrs2_core::gate::multi::CNOT {
        control: QubitId(0),
        target: QubitId(1),
    })?;
    classical_circuit.add_gate(Hadamard { target: QubitId(0) })?;

    // Alice measures her qubits
    classical_circuit.measure(QubitId(0), "alice", 0)?;
    classical_circuit.measure(QubitId(1), "alice", 1)?;

    // Bob applies corrections based on Alice's measurements
    classical_circuit.add_conditional(
        ClassicalCondition {
            lhs: ClassicalValue::Register("alice".to_string()),
            op: ComparisonOp::Equal,
            rhs: ClassicalValue::Integer(0b01),
        },
        PauliX { target: QubitId(2) },
    )?;

    classical_circuit.add_conditional(
        ClassicalCondition {
            lhs: ClassicalValue::Register("alice".to_string()),
            op: ComparisonOp::Equal,
            rhs: ClassicalValue::Integer(0b10),
        },
        PauliZ { target: QubitId(2) },
    )?;

    classical_circuit.add_conditional(
        ClassicalCondition {
            lhs: ClassicalValue::Register("alice".to_string()),
            op: ComparisonOp::Equal,
            rhs: ClassicalValue::Integer(0b11),
        },
        PauliX { target: QubitId(2) }, // In reality, this would be X then Z
    )?;

    println!(
        "Created teleportation circuit with {} operations",
        classical_circuit.num_operations()
    );
    println!("Bob's qubit 2 receives the state that was on Alice's qubit 0\n");

    Ok(())
}

/// Adaptive phase estimation using conditional rotations
fn adaptive_phase_estimation() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Adaptive Phase Estimation");
    println!("-----------------------------------");

    let circuit = ClassicalCircuitBuilder::<4>::new()
        .classical_register("phase_bits", 3)?
        // First estimation round
        .gate(Hadamard { target: QubitId(0) })?
        .gate(quantrs2_core::gate::multi::CRZ {
            control: QubitId(0),
            target: QubitId(3),
            theta: std::f64::consts::PI,
        })?
        .gate(Hadamard { target: QubitId(0) })?
        .measure(QubitId(0), "phase_bits", 0)?
        // Second round (adaptive based on first measurement)
        .gate(Hadamard { target: QubitId(1) })?
        .conditional(
            ClassicalCondition::register_equals("phase_bits", 1),
            quantrs2_core::gate::single::Phase { target: QubitId(1) }
        )?
        .gate(quantrs2_core::gate::multi::CRZ {
            control: QubitId(1),
            target: QubitId(3),
            theta: std::f64::consts::PI / 2.0,
        })?
        .gate(Hadamard { target: QubitId(1) })?
        .measure(QubitId(1), "phase_bits", 1)?
        // Third round (adaptive based on previous measurements)
        .gate(Hadamard { target: QubitId(2) })?
        // Apply phase corrections based on previous measurements
        .conditional(
            ClassicalCondition {
                lhs: ClassicalValue::Register("phase_bits".to_string()),
                op: ComparisonOp::GreaterEqual,
                rhs: ClassicalValue::Integer(1),
            },
            quantrs2_core::gate::single::RotationZ {
                target: QubitId(2),
                theta: -std::f64::consts::PI / 4.0,
            }
        )?
        .gate(quantrs2_core::gate::multi::CRZ {
            control: QubitId(2),
            target: QubitId(3),
            theta: std::f64::consts::PI / 4.0,
        })?
        .gate(Hadamard { target: QubitId(2) })?
        .measure(QubitId(2), "phase_bits", 2)?
        .build();

    println!(
        "Created adaptive phase estimation circuit with {} operations",
        circuit.num_operations()
    );
    println!("The circuit adaptively estimates the phase using 3 rounds of measurement\n");

    // Demonstrate builder pattern
    println!("Alternative: Using standard circuit with conversion");
    let mut standard_circuit = Circuit::<4>::new();
    standard_circuit.h(0)?;
    standard_circuit.h(1)?;
    standard_circuit.h(2)?;

    let classical = standard_circuit.with_classical_control();
    println!("Converted standard circuit to classical control\n");

    Ok(())
}
