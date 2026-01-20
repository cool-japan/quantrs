//! Demonstration of mid-circuit measurements and feed-forward
//!
//! This example shows how to use measurements during circuit execution
//! and apply conditional operations based on measurement results.

use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::multi::CNOT;
use quantrs2_core::gate::single::{Hadamard, PauliX, PauliZ};
use quantrs2_core::qubit::QubitId;

fn main() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("=== Mid-Circuit Measurement Demo ===\n");

    demo_basic_measurement()?;
    demo_feed_forward()?;
    demo_quantum_teleportation()?;
    demo_error_correction()?;

    Ok(())
}

fn demo_basic_measurement() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Basic Mid-Circuit Measurement ---");

    let mut circuit = MeasurementCircuit::<3>::new();

    // Create superposition
    circuit.add_gate(Box::new(Hadamard { target: QubitId(0) }))?;

    // Measure qubit 0
    let bit0 = circuit.measure(QubitId(0))?;
    println!("Measured qubit 0 -> classical bit {bit0}");

    // Apply X gate to qubit 1 conditioned on measurement
    let condition = ClassicalCondition::register_equals("default", 1);
    circuit.add_conditional(condition, Box::new(PauliX { target: QubitId(1) }))?;

    println!("Circuit has {} operations", circuit.num_operations());
    println!("Circuit has {} measurements\n", circuit.num_measurements());

    Ok(())
}

fn demo_feed_forward() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Feed-Forward Control ---");

    let (builder, bit) = MeasurementCircuitBuilder::<4>::new()
        // Prepare Bell state
        .gate(Box::new(Hadamard { target: QubitId(0) }))?
        .gate(Box::new(CNOT { control: QubitId(0), target: QubitId(1) }))?
        // Measure first qubit
        .measure(QubitId(0))?;

    let circuit = builder
        // Apply correction based on measurement
        .when(
            ClassicalCondition::register_equals("default", 1),
            Box::new(PauliX { target: QubitId(1) })
        )?
        // Add barrier for synchronization
        .barrier(vec![QubitId(1), QubitId(2)])?
        // Continue with more operations
        .gate(Box::new(Hadamard { target: QubitId(2) }))?
        .build();

    // Analyze dependencies
    let deps = circuit.analyze_dependencies();
    println!("Found {} measurements", deps.num_measurements());
    println!("Has feed-forward: {}", deps.has_feed_forward());

    for (meas_idx, ff_idx) in &deps.feed_forward_deps {
        println!("Operation {ff_idx} depends on measurement {meas_idx}");
    }

    println!();
    Ok(())
}

fn demo_quantum_teleportation() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Quantum Teleportation Protocol ---");

    let mut circuit = MeasurementCircuit::<3>::new();

    // Prepare state to teleport on qubit 0 (|ψ⟩ = α|0⟩ + β|1⟩)
    // For demo, use |+⟩ state
    circuit.add_gate(Box::new(Hadamard { target: QubitId(0) }))?;

    // Create Bell pair between qubits 1 and 2
    circuit.add_gate(Box::new(Hadamard { target: QubitId(1) }))?;
    circuit.add_gate(Box::new(CNOT {
        control: QubitId(1),
        target: QubitId(2),
    }))?;

    println!("Prepared Bell pair and state to teleport");

    // Bell measurement on qubits 0 and 1
    circuit.add_gate(Box::new(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    }))?;
    circuit.add_gate(Box::new(Hadamard { target: QubitId(0) }))?;

    // Measure qubits 0 and 1
    let bit0 = circuit.measure(QubitId(0))?;
    let bit1 = circuit.measure(QubitId(1))?;

    println!("Performed Bell measurement");

    // Apply corrections to qubit 2 based on measurements
    circuit.add_conditional(
        ClassicalCondition::register_equals("default", 1),
        Box::new(PauliX { target: QubitId(2) }),
    )?;

    circuit.add_conditional(
        ClassicalCondition::register_equals("default", 1),
        Box::new(PauliZ { target: QubitId(2) }),
    )?;

    println!("Applied classical corrections");
    println!(
        "Teleportation circuit has {} operations",
        circuit.num_operations()
    );

    println!();
    Ok(())
}

fn demo_error_correction() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Simple Error Correction ---");

    let mut circuit = MeasurementCircuit::<5>::new();

    // Encode logical qubit using repetition code
    // |0⟩ -> |000⟩, |1⟩ -> |111⟩

    // Prepare initial state (for demo, use |1⟩)
    circuit.add_gate(Box::new(PauliX { target: QubitId(0) }))?;

    // Encode
    circuit.add_gate(Box::new(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    }))?;
    circuit.add_gate(Box::new(CNOT {
        control: QubitId(0),
        target: QubitId(2),
    }))?;

    println!("Encoded logical qubit");

    // Simulate error (bit flip on qubit 1)
    circuit.add_gate(Box::new(PauliX { target: QubitId(1) }))?;
    println!("Introduced error on qubit 1");

    // Syndrome measurement using ancillas
    circuit.add_gate(Box::new(CNOT {
        control: QubitId(0),
        target: QubitId(3),
    }))?;
    circuit.add_gate(Box::new(CNOT {
        control: QubitId(1),
        target: QubitId(3),
    }))?;

    circuit.add_gate(Box::new(CNOT {
        control: QubitId(1),
        target: QubitId(4),
    }))?;
    circuit.add_gate(Box::new(CNOT {
        control: QubitId(2),
        target: QubitId(4),
    }))?;

    // Measure syndrome
    let syndrome1 = circuit.measure(QubitId(3))?;
    let syndrome2 = circuit.measure(QubitId(4))?;

    println!("Measured error syndrome");

    // Decode syndrome and apply correction
    // Note: In a full implementation, we would need compound conditions
    // For now, apply simple conditional corrections based on syndrome bits

    // Apply correction based on first syndrome bit
    circuit.add_conditional(
        ClassicalCondition::register_equals("default", 1),
        Box::new(PauliX { target: QubitId(1) }),
    )?;

    println!("Applied error correction");
    println!("Total operations: {}", circuit.num_operations());

    // Reset ancillas for reuse
    circuit.reset(QubitId(3))?;
    circuit.reset(QubitId(4))?;

    println!();
    Ok(())
}

fn demo_measurement_statistics() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Measurement Statistics ---");

    // Create a circuit that demonstrates measurement statistics
    let mut circuit = MeasurementCircuit::<4>::new();

    // Create GHZ state
    circuit.add_gate(Box::new(Hadamard { target: QubitId(0) }))?;
    circuit.add_gate(Box::new(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    }))?;
    circuit.add_gate(Box::new(CNOT {
        control: QubitId(1),
        target: QubitId(2),
    }))?;
    circuit.add_gate(Box::new(CNOT {
        control: QubitId(2),
        target: QubitId(3),
    }))?;

    // Measure all qubits
    let bits: Vec<usize> = (0..4)
        .map(|i| circuit.measure(QubitId(i as u32)).unwrap())
        .collect();

    println!("Created GHZ state and measured all qubits");
    println!("Measurement results stored in bits: {bits:?}");

    // In a real quantum computer, we would run this multiple times
    // and collect statistics showing 50% |0000⟩ and 50% |1111⟩

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measurement_demo() {
        assert!(main().is_ok());
    }
}
