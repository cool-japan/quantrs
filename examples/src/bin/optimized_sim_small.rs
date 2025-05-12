//! Demonstration of optimized quantum circuit simulation with Quantrs (Simplified version)
//!
//! This example shows a smaller demo of the optimized simulator implementations.

use std::time::Instant;
use quantrs_core::qubit::QubitId;
use quantrs_circuit::builder::Circuit;
use quantrs_circuit::builder::Simulator;
use quantrs_sim::statevector::StateVectorSimulator;
use quantrs_sim::optimized_simulator::OptimizedSimulator;

fn main() {
    println!("Quantrs Optimized Simulator (Small Demo)");
    println!("=======================================\n");
    
    // Demonstrate a Bell state with different simulators
    demo_bell_state();
    
    // Demonstrate a small circuit with 10 qubits
    demo_small_circuit();
}

/// Demonstrate creation and simulation of a Bell state
fn demo_bell_state() {
    println!("=== Bell State Demo ===");
    
    // Create a Bell state circuit
    let mut circuit = Circuit::<2>::new();
    circuit
        .h(QubitId::new(0)).unwrap()
        .cnot(QubitId::new(0), QubitId::new(1)).unwrap();
    
    // Run with standard simulator
    let standard_sim = StateVectorSimulator::new();
    let start = Instant::now();
    let result = standard_sim.run(&circuit).unwrap();
    let duration = start.elapsed();
    
    println!("Standard simulator: {:.3} seconds", duration.as_secs_f64());
    println!("State vector: {:?}", result.amplitudes());
    
    // Run with optimized simulator
    let optimized_sim = OptimizedSimulator::new();
    let start = Instant::now();
    let result = optimized_sim.run(&circuit).unwrap();
    let duration = start.elapsed();
    
    println!("Optimized simulator: {:.3} seconds", duration.as_secs_f64());
    println!("State vector: {:?}", result.amplitudes());
    
    println!();
}

/// Demonstrate a circuit with 10 qubits
fn demo_small_circuit() {
    println!("=== 10-Qubit Circuit Demo ===");
    
    // Create a quantum circuit with 10 qubits
    let mut circuit = Circuit::<10>::new();
    
    // Apply Hadamard to first 3 qubits only (to keep the output small)
    for i in 0..3 {
        circuit.h(QubitId::new(i as u32)).unwrap();
    }
    
    // Apply CNOT gates between a few qubits
    circuit.cnot(QubitId::new(0), QubitId::new(3)).unwrap();
    circuit.cnot(QubitId::new(1), QubitId::new(4)).unwrap();
    circuit.cnot(QubitId::new(2), QubitId::new(5)).unwrap();
    
    // Run with standard simulator
    let standard_sim = StateVectorSimulator::new();
    let start = Instant::now();
    let result = standard_sim.run(&circuit).unwrap();
    let duration = start.elapsed();
    
    println!("Standard simulator: {:.3} seconds", duration.as_secs_f64());
    println!("State vector size: {}", result.amplitudes().len());
    
    // Print a few of the non-zero amplitudes
    let mut count = 0;
    println!("Non-zero amplitudes:");
    for i in 0..result.amplitudes().len() {
        if result.amplitudes()[i].norm() > 1e-10 {
            println!("  |{:010b}⟩: {}", i, result.amplitudes()[i]);
            count += 1;
            if count >= 8 {
                break;
            }
        }
    }
    
    // Run with optimized simulator
    let optimized_sim = OptimizedSimulator::new();
    let start = Instant::now();
    let result = optimized_sim.run(&circuit).unwrap();
    let duration = start.elapsed();
    
    println!("Optimized simulator: {:.3} seconds", duration.as_secs_f64());
    println!("State vector size: {}", result.amplitudes().len());
    
    // Print a few of the non-zero amplitudes
    let mut count = 0;
    println!("Non-zero amplitudes:");
    for i in 0..result.amplitudes().len() {
        if result.amplitudes()[i].norm() > 1e-10 {
            println!("  |{:010b}⟩: {}", i, result.amplitudes()[i]);
            count += 1;
            if count >= 8 {
                break;
            }
        }
    }
}