use std::time::Instant;

use quantrs2_circuit::builder::Simulator as CircuitSimulator;
use quantrs2_circuit::prelude::Circuit;
use quantrs2_core::prelude::QubitId;
use quantrs2_sim::prelude::StateVectorSimulator;
use quantrs2_sim::prelude::{ContractionStrategy, TensorNetworkSimulator};
use quantrs2_sim::Simulator as SimSimulator;

// Create a QFT circuit
fn create_qft_circuit(
    num_qubits: usize,
    _params: usize,
) -> Result<Circuit<4>, Box<dyn std::error::Error>> {
    let mut circuit = Circuit::<4>::new();

    // Standard QFT implementation
    for i in 0..num_qubits.min(4) {
        circuit.h(QubitId::new(i as u32))?;

        for j in i + 1..num_qubits.min(4) {
            let angle = std::f64::consts::PI / 2_f64.powi((j - i) as i32);
            circuit.cp(QubitId::new(i as u32), QubitId::new(j as u32), angle)?;
        }
    }

    // Reverse the order of qubits (standard in QFT)
    for i in 0..num_qubits.min(4) / 2 {
        circuit.swap(
            QubitId::new(i as u32),
            QubitId::new((num_qubits.min(4) - i - 1) as u32),
        )?;
    }

    Ok(circuit)
}

// Create a QAOA circuit
fn create_qaoa_circuit(
    num_qubits: usize,
    p: usize,
) -> Result<Circuit<4>, Box<dyn std::error::Error>> {
    let mut circuit = Circuit::<4>::new();

    // Initial state: superposition
    for i in 0..num_qubits.min(4) {
        circuit.h(QubitId::new(i as u32))?;
    }

    // Apply QAOA layers
    for _layer in 0..p {
        // Problem layer (example: MaxCut on linear chain)
        for i in 0..num_qubits.min(4) - 1 {
            circuit.cnot(QubitId::new(i as u32), QubitId::new((i + 1) as u32))?;
            circuit.rz(QubitId::new((i + 1) as u32), 0.1)?; // gamma parameter
            circuit.cnot(QubitId::new(i as u32), QubitId::new((i + 1) as u32))?;
        }

        // Mixer layer
        for i in 0..num_qubits.min(4) {
            circuit.rx(QubitId::new(i as u32), 0.2)?; // beta parameter
        }
    }

    Ok(circuit)
}

// Run a benchmark comparing different simulators and strategies
fn benchmark<F>(name: &str, circuit_fn: F, num_qubits: usize, params: usize)
where
    F: Fn(usize, usize) -> Result<Circuit<4>, Box<dyn std::error::Error>>,
{
    println!("===============================================");
    println!("Benchmarking {name} with {num_qubits} qubits, params: {params}");
    println!("===============================================");

    let circuit = match circuit_fn(num_qubits, params) {
        Ok(c) => c,
        Err(e) => {
            println!("Error creating circuit: {e}");
            return;
        }
    };

    // Standard state vector simulator
    let start = Instant::now();
    let standard_sim = StateVectorSimulator::new();
    let _standard_result = standard_sim
        .run(&circuit)
        .expect("Failed to run circuit with StateVector simulator");
    let standard_duration = start.elapsed();
    println!("StateVector simulator: {standard_duration:?}");

    // Tensor network with default strategy
    let start = Instant::now();
    let mut tensor_sim = TensorNetworkSimulator::new(num_qubits);
    let _tensor_result = tensor_sim
        .run(&circuit)
        .expect("Failed to run circuit with TensorNetwork simulator (default strategy)");
    let tensor_duration = start.elapsed();
    println!("TensorNetwork (default): {tensor_duration:?}");

    // Tensor network with greedy strategy
    let start = Instant::now();
    let mut tensor_sim =
        TensorNetworkSimulator::new(num_qubits).with_strategy(ContractionStrategy::Greedy);
    let _tensor_result = tensor_sim
        .run(&circuit)
        .expect("Failed to run circuit with TensorNetwork simulator (greedy strategy)");
    let greedy_duration = start.elapsed();
    println!("TensorNetwork (greedy): {greedy_duration:?}");

    // Tensor network with optimal strategy for this circuit type
    let start = Instant::now();
    let mut tensor_sim = match name {
        "QFT" => TensorNetworkSimulator::qft(),
        "QAOA" => {
            TensorNetworkSimulator::new(num_qubits).with_strategy(ContractionStrategy::Greedy)
        }
        _ => TensorNetworkSimulator::new(num_qubits).with_strategy(ContractionStrategy::Greedy),
    };
    let _tensor_result = tensor_sim.run(&circuit).expect(&format!(
        "Failed to run circuit with TensorNetwork simulator (optimized strategy for {name})"
    ));
    let optimized_duration = start.elapsed();
    println!("TensorNetwork (optimized): {optimized_duration:?}");

    // Compare performance
    let standard_baseline = standard_duration.as_nanos() as f64;
    let tensor_speedup = standard_baseline / tensor_duration.as_nanos() as f64;
    let greedy_speedup = standard_baseline / greedy_duration.as_nanos() as f64;
    let optimized_speedup = standard_baseline / optimized_duration.as_nanos() as f64;

    println!("\nPerformance Analysis:");
    println!("  Tensor Network (default):  {tensor_speedup:.2}x");
    println!("  Tensor Network (greedy):   {greedy_speedup:.2}x");
    println!("  Tensor Network (optimized): {optimized_speedup:.2}x");

    if tensor_speedup > 1.0 {
        println!("  ✓ Tensor network provides speedup!");
    } else {
        println!("  ⚠ State vector simulator is faster for this problem size");
    }

    println!();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Tensor Network Optimization Comparison");
    println!("=====================================");
    println!("This example demonstrates the performance characteristics");
    println!("of tensor network simulators vs state vector simulators");
    println!("for different quantum circuit types.\n");

    // Small circuit sizes where we can see the difference
    let small_qubits = 4;
    let medium_qubits = 4; // Limited by Circuit<4> constraint

    // QFT Benchmarks
    println!("QUANTUM FOURIER TRANSFORM (QFT) BENCHMARKS");
    println!("------------------------------------------");
    benchmark("QFT", create_qft_circuit, small_qubits, 1);

    // QAOA Benchmarks
    println!("QUANTUM APPROXIMATE OPTIMIZATION ALGORITHM (QAOA) BENCHMARKS");
    println!("------------------------------------------------------------");
    benchmark("QAOA", create_qaoa_circuit, medium_qubits, 2);
    benchmark("QAOA", create_qaoa_circuit, medium_qubits, 4);

    println!("SUMMARY AND INSIGHTS");
    println!("===================");
    println!("Tensor network simulators can provide significant speedups for:");
    println!("• Circuits with limited entanglement");
    println!("• Structured circuits (QFT, QAOA)");
    println!("• Large circuits where state vector simulation is infeasible");
    println!();
    println!("Key optimization strategies:");
    println!("• Greedy contraction: Good general-purpose strategy");
    println!("• Specialized strategies: QFT and QAOA optimized versions");
    println!("• Adaptive strategies: Automatically choose based on circuit structure");
    println!();
    println!("For production use, consider:");
    println!("• Enhanced tensor network simulator for more sophisticated optimization");
    println!("• GPU acceleration for larger circuits");
    println!("• Memory-efficient strategies for very large tensor networks");

    Ok(())
}
