// Tensor Network Simulator Example
//
// This example demonstrates the use of tensor network simulation
// for quantum circuits, which can be more efficient than state vector
// simulation for certain circuit topologies.

use quantrs_circuit::builder::Circuit;
use quantrs_sim::statevector::StateVectorSimulator;
use quantrs_sim::tensor_network::TensorNetworkSimulator;
use std::time::Instant;

fn main() {
    println!("Tensor Network Simulator Example");
    println!("===============================\n");

    println!("This example compares tensor network simulation with state vector simulation");
    println!("for different types of quantum circuits.\n");

    // Linear circuit (good for tensor networks)
    println!("Test 1: Linear Circuit (CNOT chain)");
    println!("----------------------------------");
    compare_linear_circuit();

    // GHZ circuit (tensor network may be less efficient)
    println!("\nTest 2: GHZ State Preparation");
    println!("---------------------------");
    compare_ghz_circuit();

    // Random circuit (general case)
    println!("\nTest 3: Random Circuit");
    println!("--------------------");
    compare_random_circuit();

    println!("\nSummary:");
    println!("-------");
    println!("- Tensor network simulation shines for circuits with limited entanglement");
    println!("- State vector simulation is often faster for highly entangled circuits");
    println!(
        "- Tensor networks can enable simulation of larger qubit counts with lower memory usage"
    );
    println!("  (particularly for circuits with specific structures)");
}

// Compare simulators on a linear circuit (CNOT chain)
fn compare_linear_circuit() {
    // Create a linear circuit with a chain of CNOTs
    let mut circuit = Circuit::<8>::new();

    // Apply H to first qubit
    circuit.h(0).unwrap();

    // Create a chain of CNOTs
    for i in 0..7 {
        circuit.cnot(i, i + 1).unwrap();
    }

    // Compare simulators
    compare_simulators(&circuit, "Linear Circuit");
}

// Compare simulators on a GHZ state preparation circuit
fn compare_ghz_circuit() {
    // Create a GHZ state preparation circuit
    let mut circuit = Circuit::<8>::new();

    // Apply H to first qubit
    circuit.h(0).unwrap();

    // Apply CNOTs from first qubit to all others
    for i in 1..8 {
        circuit.cnot(0, i).unwrap();
    }

    // Compare simulators
    compare_simulators(&circuit, "GHZ Circuit");
}

// Compare simulators on a random circuit
fn compare_random_circuit() {
    // Create a random circuit
    let mut circuit = Circuit::<8>::new();

    // Apply random H gates
    for i in 0..8 {
        circuit.h(i).unwrap();
    }

    // Apply some random gates
    circuit.x(1).unwrap();
    circuit.y(3).unwrap();
    circuit.z(5).unwrap();
    circuit.rx(2, 0.5).unwrap();
    circuit.ry(4, 0.7).unwrap();
    circuit.rz(6, 0.3).unwrap();

    // Apply some random two-qubit gates
    circuit.cnot(0, 1).unwrap();
    circuit.cnot(2, 3).unwrap();
    circuit.cnot(4, 5).unwrap();
    circuit.cnot(6, 7).unwrap();
    circuit.cnot(1, 3).unwrap();
    circuit.cnot(5, 7).unwrap();

    // Compare simulators
    compare_simulators(&circuit, "Random Circuit");
}

// Compare state vector and tensor network simulators
fn compare_simulators<const N: usize>(circuit: &Circuit<N>, name: &str) {
    println!(
        "Circuit: {} (qubits: {}, gates: {})",
        name,
        N,
        circuit.num_gates()
    );

    // Create simulators
    let sv_sim = StateVectorSimulator::new();
    let tn_sim = TensorNetworkSimulator::new();

    // Also create an optimized tensor network simulator
    let opt_tn_sim = TensorNetworkSimulator::new()
        .with_optimization_level(2)
        .with_bond_dimension(32);

    // Run with state vector simulator
    let start = Instant::now();
    let sv_result = circuit.run(sv_sim).unwrap();
    let sv_time = start.elapsed();

    // Run with basic tensor network simulator
    let start = Instant::now();
    let tn_result = circuit.run(tn_sim).unwrap();
    let tn_time = start.elapsed();

    // Run with optimized tensor network simulator
    let start = Instant::now();
    let opt_tn_result = circuit.run(opt_tn_sim).unwrap();
    let opt_tn_time = start.elapsed();

    // Compare results
    println!("State vector time: {:?}", sv_time);
    println!("Basic tensor network time: {:?}", tn_time);
    println!("Optimized tensor network time: {:?}", opt_tn_time);

    // Check if results match
    let sv_probs: Vec<f64> = sv_result
        .amplitudes()
        .iter()
        .map(|a| a.norm_sqr())
        .collect();
    let tn_probs: Vec<f64> = tn_result
        .amplitudes()
        .iter()
        .map(|a| a.norm_sqr())
        .collect();
    let opt_tn_probs: Vec<f64> = opt_tn_result
        .amplitudes()
        .iter()
        .map(|a| a.norm_sqr())
        .collect();

    let basic_matching = sv_probs
        .iter()
        .zip(tn_probs.iter())
        .all(|(a, b)| (a - b).abs() < 1e-10);

    let opt_matching = sv_probs
        .iter()
        .zip(opt_tn_probs.iter())
        .all(|(a, b)| (a - b).abs() < 1e-10);

    println!(
        "Basic TN results match: {}",
        if basic_matching { "Yes" } else { "No" }
    );
    println!(
        "Optimized TN results match: {}",
        if opt_matching { "Yes" } else { "No" }
    );

    // Show speedup or slowdown
    let basic_ratio = sv_time.as_secs_f64() / tn_time.as_secs_f64();
    let opt_ratio = sv_time.as_secs_f64() / opt_tn_time.as_secs_f64();

    if basic_ratio > 1.0 {
        println!(
            "Basic tensor network is {:.2}x faster than state vector",
            basic_ratio
        );
    } else {
        println!(
            "State vector is {:.2}x faster than basic tensor network",
            1.0 / basic_ratio
        );
    }

    if opt_ratio > 1.0 {
        println!(
            "Optimized tensor network is {:.2}x faster than state vector",
            opt_ratio
        );
    } else {
        println!(
            "State vector is {:.2}x faster than optimized tensor network",
            1.0 / opt_ratio
        );
    }

    // Compare basic vs optimized tensor network
    let tn_opt_ratio = tn_time.as_secs_f64() / opt_tn_time.as_secs_f64();
    if tn_opt_ratio > 1.0 {
        println!(
            "Optimization provides {:.2}x speedup for tensor network",
            tn_opt_ratio
        );
    } else {
        println!("Optimization provides no speedup for this circuit type");
    }

    // Estimated memory usage
    let sv_memory = 2_f64.powi(N as i32) * 16.0 / 1024.0 / 1024.0; // MB
    let tn_memory = estimate_tn_memory(circuit);

    println!(
        "Estimated memory: State vector: {:.2} MB, Tensor network: {:.2} MB",
        sv_memory, tn_memory
    );
}

// Estimate memory usage of tensor network simulation
fn estimate_tn_memory<const N: usize>(circuit: &Circuit<N>) -> f64 {
    // This is a very simplified estimate - real implementation would analyze the circuit structure
    let gates = circuit.num_gates() as f64;
    let qubits = N as f64;

    // Rough estimate based on number of tensors and typical bond dimensions
    let tensor_count = qubits + gates;
    let avg_elements = 2_f64.powi(2) * gates + 2.0 * qubits;

    // Complex number is 16 bytes
    (tensor_count * avg_elements * 16.0) / 1024.0 / 1024.0 // MB
}
