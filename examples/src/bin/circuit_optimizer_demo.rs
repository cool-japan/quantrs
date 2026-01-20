//! Demonstration of quantum circuit optimization

use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use std::time::Instant;

fn main() {
    println!("=== Quantum Circuit Optimizer Demo ===\n");

    // Example 1: Basic optimization
    println!("Example 1: Basic Circuit Optimization");
    optimize_basic_circuit();

    println!("\n{}\n", "=".repeat(50));

    // Example 2: Complex circuit optimization
    println!("Example 2: Complex Circuit Optimization");
    optimize_complex_circuit();

    println!("\n{}\n", "=".repeat(50));

    // Example 3: Custom optimization passes
    println!("Example 3: Custom Optimization Passes");
    optimize_with_custom_passes();
}

fn optimize_basic_circuit() {
    // Create a simple circuit with redundant gates
    let mut circuit = Circuit::<4>::new();

    // Add some gates that can be optimized
    circuit
        .h(QubitId::new(0))
        .expect("Failed to apply H gate to qubit 0")
        .x(QubitId::new(0))
        .expect("Failed to apply first X gate to qubit 0")
        .x(QubitId::new(0))
        .expect("Failed to apply second X gate to qubit 0") // X·X = I (redundant)
        .h(QubitId::new(1))
        .expect("Failed to apply first H gate to qubit 1")
        .h(QubitId::new(1))
        .expect("Failed to apply second H gate to qubit 1") // H·H = I (redundant)
        .cnot(QubitId::new(0), QubitId::new(1))
        .expect("Failed to apply CNOT gate")
        .z(QubitId::new(2))
        .expect("Failed to apply first Z gate to qubit 2")
        .z(QubitId::new(2))
        .expect("Failed to apply second Z gate to qubit 2") // Z·Z = I (redundant)
        .s(QubitId::new(3))
        .expect("Failed to apply S gate to qubit 3")
        .sdg(QubitId::new(3))
        .expect("Failed to apply S† gate to qubit 3"); // S·S† = I (redundant)

    println!("Original circuit created with intentional redundancies");

    // Create optimizer
    let optimizer = CircuitOptimizer::<4>::new();

    // Optimize the circuit
    let start = Instant::now();
    let result = optimizer.optimize(&circuit);
    let elapsed = start.elapsed();

    println!("Optimization completed in {elapsed:?}");
    result.print_summary();
}

fn optimize_complex_circuit() {
    // Create a more complex circuit
    let mut circuit = Circuit::<6>::new();

    // Build a circuit with various optimization opportunities
    circuit
        // Prepare Bell pairs
        .h(QubitId::new(0))
        .expect("Failed to apply H gate to qubit 0 for first Bell pair")
        .cnot(QubitId::new(0), QubitId::new(1))
        .expect("Failed to apply CNOT for first Bell pair")
        .h(QubitId::new(2))
        .expect("Failed to apply H gate to qubit 2 for second Bell pair")
        .cnot(QubitId::new(2), QubitId::new(3))
        .expect("Failed to apply CNOT for second Bell pair")
        // Some redundant operations
        .x(QubitId::new(4))
        .expect("Failed to apply first X gate to qubit 4")
        .y(QubitId::new(4))
        .expect("Failed to apply first Y gate to qubit 4")
        .z(QubitId::new(4))
        .expect("Failed to apply first Z gate to qubit 4")
        .x(QubitId::new(4))
        .expect("Failed to apply second X gate to qubit 4")
        .y(QubitId::new(4))
        .expect("Failed to apply second Y gate to qubit 4")
        .z(QubitId::new(4))
        .expect("Failed to apply second Z gate to qubit 4")
        // Gates that can be reordered
        .h(QubitId::new(0))
        .expect("Failed to apply third H gate to qubit 0")
        .z(QubitId::new(1))
        .expect("Failed to apply Z gate to qubit 1")
        .h(QubitId::new(0))
        .expect("Failed to apply fourth H gate to qubit 0") // H·H = I
        // More complex patterns
        .h(QubitId::new(5))
        .expect("Failed to apply first H gate to qubit 5")
        .x(QubitId::new(5))
        .expect("Failed to apply X gate to qubit 5")
        .h(QubitId::new(5))
        .expect("Failed to apply second H gate to qubit 5"); // H·X·H = Z

    println!("Complex circuit created with multiple optimization opportunities");

    // Create optimizer with custom settings
    let optimizer = CircuitOptimizer::<6>::new().with_max_iterations(20);

    // Optimize
    let start = Instant::now();
    let result = optimizer.optimize(&circuit);
    let elapsed = start.elapsed();

    println!("Optimization completed in {elapsed:?}");
    result.print_summary();
}

fn optimize_with_custom_passes() {
    // Create a circuit
    let mut circuit = Circuit::<5>::new();

    // Build circuit
    circuit
        .h(QubitId::new(0))
        .expect("Failed to apply initial H gate to qubit 0")
        .cnot(QubitId::new(0), QubitId::new(1))
        .expect("Failed to apply CNOT between qubits 0 and 1")
        .cnot(QubitId::new(1), QubitId::new(2))
        .expect("Failed to apply CNOT between qubits 1 and 2")
        .cnot(QubitId::new(2), QubitId::new(3))
        .expect("Failed to apply CNOT between qubits 2 and 3")
        .cnot(QubitId::new(3), QubitId::new(4))
        .expect("Failed to apply CNOT between qubits 3 and 4")
        .h(QubitId::new(4))
        .expect("Failed to apply final H gate to qubit 4");

    println!("Circuit created for custom optimization");

    // Create optimizer with only specific passes
    use quantrs2_circuit::optimizer::{
        CommutationOptimizer, OptimizationContext, PassResult, PeepholeOptimizer,
    };

    let optimizer = CircuitOptimizer::<5>::with_passes(vec![
        OptimizationPassType::RedundantElimination(RedundantGateElimination),
        OptimizationPassType::SingleQubitFusion(SingleQubitGateFusion),
        OptimizationPassType::Peephole(PeepholeOptimizer::default()),
        OptimizationPassType::Commutation(CommutationOptimizer),
    ]);

    // Optimize
    let start = Instant::now();
    let result = optimizer.optimize(&circuit);
    let elapsed = start.elapsed();

    println!("Custom optimization completed in {elapsed:?}");
    result.print_summary();
}

// Example of a hardware-specific optimization
fn optimize_for_hardware() {
    use std::collections::HashSet;

    // Define hardware connectivity (linear chain)
    let connectivity = vec![(0, 1), (1, 2), (2, 3), (3, 4)];

    // Define native gates
    let mut native_gates = HashSet::new();
    native_gates.insert("X".to_string());
    native_gates.insert("Y".to_string());
    native_gates.insert("Z".to_string());
    native_gates.insert("H".to_string());
    native_gates.insert("CNOT".to_string());
    native_gates.insert("RZ".to_string());

    // Create hardware-aware optimizer
    let hw_optimizer = HardwareOptimizer::new(connectivity, native_gates);

    let optimizer =
        CircuitOptimizer::<5>::new().add_pass(OptimizationPassType::Hardware(hw_optimizer));

    // Create and optimize circuit
    let circuit = Circuit::<5>::new();
    let result = optimizer.optimize(&circuit);

    println!("\nHardware-specific optimization:");
    result.print_summary();
}
