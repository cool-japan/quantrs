//! Simple test for the stabilizer simulator

use quantrs2_sim::stabilizer::{StabilizerSimulator, StabilizerGate};

fn main() {
    println!("=== Testing Stabilizer Simulator ===\n");
    
    // Test 1: Bell State
    println!("1. Bell State:");
    let mut sim = StabilizerSimulator::new(2);
    sim.apply_gate(StabilizerGate::H(0)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(0, 1)).unwrap();
    println!("Stabilizers: {:?}\n", sim.get_stabilizers());
    
    // Test 2: GHZ State
    println!("2. GHZ State:");
    let mut sim = StabilizerSimulator::new(3);
    sim.apply_gate(StabilizerGate::H(0)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(0, 1)).unwrap();
    sim.apply_gate(StabilizerGate::CNOT(1, 2)).unwrap();
    println!("Stabilizers: {:?}\n", sim.get_stabilizers());
    
    println!("=== Test Complete ===");
}