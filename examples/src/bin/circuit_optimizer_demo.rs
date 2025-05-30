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
    circuit.h(QubitId::new(0)).unwrap()
        .x(QubitId::new(0)).unwrap()
        .x(QubitId::new(0)).unwrap()  // X·X = I (redundant)
        .h(QubitId::new(1)).unwrap()
        .h(QubitId::new(1)).unwrap()  // H·H = I (redundant)
        .cnot(QubitId::new(0), QubitId::new(1)).unwrap()
        .z(QubitId::new(2)).unwrap()
        .z(QubitId::new(2)).unwrap()  // Z·Z = I (redundant)
        .s(QubitId::new(3)).unwrap()
        .sdg(QubitId::new(3)).unwrap(); // S·S† = I (redundant)

    println!("Original circuit created with intentional redundancies");

    // Create optimizer
    let optimizer = CircuitOptimizer::<4>::new();

    // Optimize the circuit
    let start = Instant::now();
    let result = optimizer.optimize(&circuit);
    let elapsed = start.elapsed();

    println!("Optimization completed in {:?}", elapsed);
    result.print_summary();
}

fn optimize_complex_circuit() {
    // Create a more complex circuit
    let mut circuit = Circuit::<6>::new();

    // Build a circuit with various optimization opportunities
    circuit
        // Prepare Bell pairs
        .h(QubitId::new(0)).unwrap()
        .cnot(QubitId::new(0), QubitId::new(1)).unwrap()
        .h(QubitId::new(2)).unwrap()
        .cnot(QubitId::new(2), QubitId::new(3)).unwrap()
        // Some redundant operations
        .x(QubitId::new(4)).unwrap()
        .y(QubitId::new(4)).unwrap()
        .z(QubitId::new(4)).unwrap()
        .x(QubitId::new(4)).unwrap()
        .y(QubitId::new(4)).unwrap()
        .z(QubitId::new(4)).unwrap()
        // Gates that can be reordered
        .h(QubitId::new(0)).unwrap()
        .z(QubitId::new(1)).unwrap()
        .h(QubitId::new(0)).unwrap()  // H·H = I
        // More complex patterns
        .h(QubitId::new(5)).unwrap()
        .x(QubitId::new(5)).unwrap()
        .h(QubitId::new(5)).unwrap(); // H·X·H = Z

    println!("Complex circuit created with multiple optimization opportunities");

    // Create optimizer with custom settings
    let optimizer = CircuitOptimizer::<6>::new().with_max_iterations(20);

    // Optimize
    let start = Instant::now();
    let result = optimizer.optimize(&circuit);
    let elapsed = start.elapsed();

    println!("Optimization completed in {:?}", elapsed);
    result.print_summary();
}

fn optimize_with_custom_passes() {
    // Create a circuit
    let mut circuit = Circuit::<5>::new();

    // Build circuit
    circuit
        .h(QubitId::new(0))
        .unwrap()
        .cnot(QubitId::new(0), QubitId::new(1))
        .unwrap()
        .cnot(QubitId::new(1), QubitId::new(2))
        .unwrap()
        .cnot(QubitId::new(2), QubitId::new(3))
        .unwrap()
        .cnot(QubitId::new(3), QubitId::new(4))
        .unwrap()
        .h(QubitId::new(4))
        .unwrap();

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

    println!("Custom optimization completed in {:?}", elapsed);
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
