use std::time::Instant;

use quantrs_circuit::{Circuit, GateBuilder};
use quantrs_core::QubitId;
use quantrs_sim::{ContractionStrategy, SimulationResult, StateVectorSimulator, TensorNetworkSimulator};

// Create a QFT circuit
fn create_qft_circuit(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);
    
    // Standard QFT implementation
    for i in 0..num_qubits {
        circuit = circuit.h(QubitId::new(i));
        
        for j in i+1..num_qubits {
            let angle = std::f64::consts::PI / 2_f64.powi((j - i) as i32);
            circuit = circuit.cp(angle, QubitId::new(i), QubitId::new(j));
        }
    }
    
    // Reverse the order of qubits (standard in QFT)
    for i in 0..num_qubits/2 {
        circuit = circuit.swap(QubitId::new(i), QubitId::new(num_qubits - i - 1));
    }
    
    circuit
}

// Create a QAOA circuit
fn create_qaoa_circuit(num_qubits: usize, p: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits);
    
    // Initial state: superposition
    for i in 0..num_qubits {
        circuit = circuit.h(QubitId::new(i));
    }
    
    // QAOA layers
    for _ in 0..p {
        // Problem Hamiltonian - ZZ interactions (for example, using nearest-neighbor coupling)
        for i in 0..num_qubits-1 {
            circuit = circuit.cnot(QubitId::new(i), QubitId::new(i+1));
            circuit = circuit.rz(0.1, QubitId::new(i+1));  // gamma parameter
            circuit = circuit.cnot(QubitId::new(i), QubitId::new(i+1));
        }
        
        // Mixer Hamiltonian - X rotations
        for i in 0..num_qubits {
            circuit = circuit.rx(0.2, QubitId::new(i));  // beta parameter
        }
    }
    
    circuit
}

// Run a benchmark comparing different simulators and strategies
fn benchmark<F>(name: &str, circuit_fn: F, num_qubits: usize, params: usize) 
where
    F: Fn(usize, usize) -> Circuit,
{
    println!("===============================================");
    println!("Benchmarking {} with {} qubits, params: {}", name, num_qubits, params);
    println!("===============================================");
    
    let circuit = circuit_fn(num_qubits, params);
    
    // Standard state vector simulator
    let start = Instant::now();
    let standard_sim = StateVectorSimulator::new();
    let _standard_result = standard_sim.run(&circuit).unwrap();
    let standard_duration = start.elapsed();
    println!("StateVector simulator: {:?}", standard_duration);
    
    // Tensor network with default strategy
    let start = Instant::now();
    let tensor_sim = TensorNetworkSimulator::new();
    let _tensor_result = tensor_sim.run(&circuit).unwrap();
    let tensor_duration = start.elapsed();
    println!("TensorNetwork (default): {:?}", tensor_duration);
    
    // Tensor network with greedy strategy
    let start = Instant::now();
    let tensor_sim = TensorNetworkSimulator::with_strategy(ContractionStrategy::Greedy);
    let _tensor_result = tensor_sim.run(&circuit).unwrap();
    let greedy_duration = start.elapsed();
    println!("TensorNetwork (greedy): {:?}", greedy_duration);
    
    // Tensor network with optimal strategy for this circuit type
    let start = Instant::now();
    let tensor_sim = match name {
        "QFT" => TensorNetworkSimulator::qft(),
        "QAOA" => TensorNetworkSimulator::qaoa(),
        _ => TensorNetworkSimulator::with_strategy(ContractionStrategy::Optimal),
    };
    let _tensor_result = tensor_sim.run(&circuit).unwrap();
    let optimal_duration = start.elapsed();
    println!("TensorNetwork (optimal): {:?}", optimal_duration);
    
    // Print performance comparison
    println!("Performance comparison (relative to state vector):");
    println!("Default strategy: {:.2}x", standard_duration.as_secs_f64() / tensor_duration.as_secs_f64());
    println!("Greedy strategy: {:.2}x", standard_duration.as_secs_f64() / greedy_duration.as_secs_f64());
    println!("Optimal strategy: {:.2}x", standard_duration.as_secs_f64() / optimal_duration.as_secs_f64());
    println!();
}

fn main() {
    // Benchmark QFT with increasing qubit counts
    for n in 4..=10 {
        benchmark("QFT", |num_qubits, _| create_qft_circuit(num_qubits), n, 0);
    }
    
    // Benchmark QAOA with increasing qubit counts
    for n in 4..=10 {
        benchmark("QAOA", |num_qubits, p| create_qaoa_circuit(num_qubits, p), n, 2);
    }
    
    // Benchmark QAOA with increasing p parameter
    for p in 1..=5 {
        benchmark("QAOA", |num_qubits, p| create_qaoa_circuit(num_qubits, p), 6, p);
    }
}