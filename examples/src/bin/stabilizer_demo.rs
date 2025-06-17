//! Demonstration of the Stabilizer Simulator for efficient Clifford circuit simulation
//!
//! The Stabilizer formalism provides exponentially more efficient simulation
//! of quantum circuits composed only of Clifford gates (H, S, CNOT, etc.)
//! compared to full state vector simulation.

use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use quantrs2_core::gate::{single::*, multi::*};
use quantrs2_sim::prelude::*;
use quantrs2_sim::stabilizer::{StabilizerGate, StabilizerSimulator, CliffordCircuitBuilder, is_clifford_circuit};

fn main() {
    println!("=== Stabilizer Simulator Demo ===\n");

    // Example 1: Bell State Preparation
    println!("1. Bell State Preparation:");
    let mut sim = StabilizerSimulator::new(2);

    // Apply H to qubit 0
    sim.apply_gate(StabilizerGate::H(0)).unwrap();
    println!("After H(0): {:?}", sim.get_stabilizers());

    // Apply CNOT(0, 1)
    sim.apply_gate(StabilizerGate::CNOT(0, 1)).unwrap();
    println!("After CNOT(0,1): {:?}", sim.get_stabilizers());
    println!("This represents the Bell state |00⟩ + |11⟩\n");

    // Example 2: GHZ State Preparation
    println!("2. GHZ State Preparation:");
    let mut sim = StabilizerSimulator::new(3);

    sim.apply_gate(StabilizerGate::H(0)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(0, 1)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(1, 2)).unwrap();

    println!("GHZ state stabilizers: {:?}", sim.get_stabilizers());
    println!("This represents the GHZ state |000⟩ + |111⟩\n");

    // Example 3: Error Correction Code (3-qubit bit flip code)
    println!("3. 3-Qubit Bit Flip Code:");
    let mut sim = StabilizerSimulator::new(3);

    // Encode logical |0⟩ as |000⟩
    // The stabilizers are ZZI and IZZ
    sim.apply_gate(StabilizerGate::CNOT(0, 1)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(0, 2)).unwrap();

    println!("Encoded |0⟩ stabilizers: {:?}", sim.get_stabilizers());

    // Apply bit flip error on qubit 1
    sim.apply_gate(StabilizerGate::X(1)).unwrap();
    println!("After X error on qubit 1: {:?}", sim.get_stabilizers());

    // Example 4: Measurement
    println!("\n4. Measurement Example:");
    let mut sim = StabilizerSimulator::new(2);

    // Create superposition
    sim.apply_gate(StabilizerGate::H(0)).unwrap();
    println!("Before measurement: {:?}", sim.get_stabilizers());

    // Measure qubit 0
    let outcome = sim.measure(0).unwrap();
    println!("Measurement outcome: {}", outcome as u8);
    println!("After measurement: {:?}", sim.get_stabilizers());

    // Example 5: Using the Clifford Circuit Builder
    println!("\n5. Clifford Circuit Builder:");
    let circuit = CliffordCircuitBuilder::new(3)
        .h(0)
        .cnot(0, 1)
        .s(1)
        .cnot(1, 2)
        .h(2)
        .run()
        .unwrap();

    println!("Final stabilizers: {:?}", circuit.get_stabilizers());

    // Example 6: Phase Gate Application
    println!("\n6. Phase Gate Example:");
    let mut sim = StabilizerSimulator::new(1);

    // Create |+⟩ state
    sim.apply_gate(StabilizerGate::H(0)).unwrap();
    println!("After H: {:?}", sim.get_stabilizers());

    // Apply S gate
    sim.apply_gate(StabilizerGate::S(0)).unwrap();
    println!("After S: {:?}", sim.get_stabilizers());

    // This creates |+i⟩ = (|0⟩ + i|1⟩)/√2

    // Example 7: Check if circuit is Clifford
    println!("\n7. Clifford Circuit Detection:");

    // Create a simple 2-qubit circuit
    let mut circuit = Circuit::<2>::new();
    circuit.h(0).unwrap();
    circuit.cnot(0, 1).unwrap();

    let is_clifford = is_clifford_circuit(&circuit);
    println!("Is Bell state circuit Clifford? {}", is_clifford);

    // Add a non-Clifford gate
    circuit.rx(0, std::f64::consts::PI / 4.0).unwrap();

    let is_clifford = is_clifford_circuit(&circuit);
    println!("Is circuit with Rx(π/4) Clifford? {}", is_clifford);

    println!("\n=== Demo Complete ===");
}
