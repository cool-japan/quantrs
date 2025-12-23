//! Circuit Optimization Demo
//!
//! This example demonstrates the quantum circuit optimization capabilities
//! of the `QuantRS2` simulation framework, including gate fusion, redundant
//! gate elimination, and parallelization opportunities.

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::gate::multi::*;
use quantrs2_core::gate::single::*;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::circuit_optimization::{CircuitOptimizer, OptimizationConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 QuantRS2 Circuit Optimization Demo");
    println!("=====================================\n");

    // Create a sample circuit with optimization opportunities
    let mut circuit = Circuit::<4>::new();

    // Add redundant gate pairs (H H = I, X X = I)
    circuit.h(QubitId::new(0))?;
    circuit.h(QubitId::new(0))?; // This should be eliminated

    // Add single-qubit gate sequence for fusion
    circuit.x(QubitId::new(1))?;
    circuit.z(QubitId::new(1))?;
    circuit.s(QubitId::new(1))?;

    // Add commuting gates on different qubits (parallelization opportunity)
    circuit.h(QubitId::new(2))?;
    circuit.x(QubitId::new(3))?; // These can be parallelized

    // Add CNOT chain pattern
    circuit.cnot(QubitId::new(0), QubitId::new(1))?;
    circuit.cnot(QubitId::new(1), QubitId::new(2))?;
    circuit.cnot(QubitId::new(0), QubitId::new(1))?;

    // Add more redundant gates
    circuit.y(QubitId::new(2))?;
    circuit.y(QubitId::new(2))?; // Another redundant pair

    println!("📊 Original Circuit Analysis");
    println!("Gate count: {}", circuit.gates().len());

    // Create optimizer with custom configuration
    let config = OptimizationConfig {
        enable_gate_fusion: true,
        enable_redundant_elimination: true,
        enable_commutation_reordering: true,
        enable_single_qubit_optimization: true,
        enable_two_qubit_optimization: true,
        max_passes: 5,
        enable_depth_reduction: true,
    };

    let mut optimizer = CircuitOptimizer::with_config(config);

    // Perform optimization
    println!("\n🚀 Running Circuit Optimization...\n");
    let optimized_circuit = optimizer.optimize(&circuit)?;

    // Display optimization results
    let stats = optimizer.get_statistics();

    println!("{}", stats.generate_report());

    // Demonstrate different optimization strategies
    demonstrate_optimization_strategies()?;

    Ok(())
}

fn demonstrate_optimization_strategies() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Optimization Strategy Comparison");
    println!("===================================\n");

    // Create a more complex test circuit
    let mut complex_circuit = Circuit::<6>::new();

    // Add various gate patterns
    for i in 0..6 {
        complex_circuit.h(QubitId::new(i))?;
        if i > 0 {
            complex_circuit.cnot(QubitId::new(i - 1), QubitId::new(i))?;
        }
    }

    // Add redundant patterns
    complex_circuit.x(QubitId::new(0))?;
    complex_circuit.x(QubitId::new(0))?;
    complex_circuit.z(QubitId::new(1))?;
    complex_circuit.z(QubitId::new(1))?;

    println!("Complex circuit gates: {}", complex_circuit.gates().len());

    // Test different optimization configurations
    let configs = vec![
        (
            "Conservative",
            OptimizationConfig {
                enable_gate_fusion: false,
                enable_redundant_elimination: true,
                enable_commutation_reordering: false,
                enable_single_qubit_optimization: false,
                enable_two_qubit_optimization: false,
                max_passes: 1,
                enable_depth_reduction: false,
            },
        ),
        (
            "Aggressive",
            OptimizationConfig {
                enable_gate_fusion: true,
                enable_redundant_elimination: true,
                enable_commutation_reordering: true,
                enable_single_qubit_optimization: true,
                enable_two_qubit_optimization: true,
                max_passes: 10,
                enable_depth_reduction: true,
            },
        ),
        (
            "Fusion-Only",
            OptimizationConfig {
                enable_gate_fusion: true,
                enable_redundant_elimination: false,
                enable_commutation_reordering: false,
                enable_single_qubit_optimization: true,
                enable_two_qubit_optimization: false,
                max_passes: 3,
                enable_depth_reduction: false,
            },
        ),
    ];

    for (name, config) in configs {
        let mut optimizer = CircuitOptimizer::with_config(config);
        let _optimized = optimizer.optimize(&complex_circuit)?;
        let stats = optimizer.get_statistics();

        println!("📈 {name} Strategy:");
        println!("  Gate Reduction: {:.1}%", stats.gate_count_reduction());
        println!("  Depth Reduction: {:.1}%", stats.depth_reduction());
        println!("  Gates Fused: {}", stats.gates_fused);
        println!(
            "  Redundant Eliminated: {}",
            stats.redundant_gates_eliminated
        );
        println!(
            "  Optimization Time: {:.2}ms",
            stats.optimization_time_ns as f64 / 1_000_000.0
        );
        println!();
    }

    Ok(())
}
