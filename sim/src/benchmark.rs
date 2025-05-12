//! Benchmarking utilities for quantum simulators
//!
//! This module provides utilities for benchmarking different simulator
//! implementations and comparing their performance.

use std::time::{Duration, Instant};
use quantrs_circuit::builder::{Circuit, Simulator};
use quantrs_core::{
    register::Register,
    qubit::QubitId,
    error::QuantrsResult,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::statevector::StateVectorSimulator;
use crate::optimized_simulator_simple::OptimizedSimulatorSimple;
use crate::optimized_simulator_chunked::OptimizedSimulatorChunked;
use crate::optimized_simulator::OptimizedSimulator;

/// Benchmark results for a quantum simulator
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the simulator
    pub simulator_name: String,
    /// Number of qubits
    pub num_qubits: usize,
    /// Total number of gates
    pub num_gates: usize,
    /// Execution time
    pub duration: Duration,
    /// Additional notes or context
    pub notes: Option<String>,
    /// Peak memory usage (if available)
    pub peak_memory: Option<usize>,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    pub fn new(
        simulator_name: &str, 
        num_qubits: usize, 
        num_gates: usize, 
        duration: Duration,
        notes: Option<String>,
        peak_memory: Option<usize>,
    ) -> Self {
        Self {
            simulator_name: simulator_name.to_string(),
            num_qubits,
            num_gates,
            duration,
            notes,
            peak_memory,
        }
    }
    
    /// Format the result as a string
    pub fn format(&self) -> String {
        let duration_ms = self.duration.as_millis();
        let rate = if duration_ms > 0 {
            self.num_gates as f64 / (duration_ms as f64 / 1000.0)
        } else {
            f64::INFINITY
        };
        
        let notes = if let Some(ref notes) = self.notes {
            format!(" ({})", notes)
        } else {
            String::new()
        };
        
        let memory_str = if let Some(mem) = self.peak_memory {
            format!(", Memory: {:.2} MB", mem as f64 / (1024.0 * 1024.0))
        } else {
            String::new()
        };
        
        format!(
            "Simulator: {}{}\n  Qubits: {}\n  Gates: {}\n  Time: {} ms\n  Rate: {:.2} gates/s{}",
            self.simulator_name, notes, self.num_qubits, self.num_gates, duration_ms, rate, memory_str
        )
    }
}

/// Run a benchmark on a simulator with a given circuit
pub fn run_benchmark<S, const N: usize>(
    name: &str,
    simulator: &S,
    circuit: &Circuit<N>,
    notes: Option<String>,
) -> QuantrsResult<BenchmarkResult> 
where
    S: Simulator<N>,
{
    let num_qubits = N;
    let num_gates = circuit.gates().len();
    
    let start = Instant::now();
    let _result = simulator.run(circuit)?;
    let duration = start.elapsed();
    
    Ok(BenchmarkResult::new(name, num_qubits, num_gates, duration, notes, None))
}

/// Compare different simulator implementations on the same circuit
pub fn compare_simulators<const N: usize>(circuit: &Circuit<N>) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    // For very large circuits, only test optimized implementations
    if N <= 24 {
        // Run standard simulator
        let standard_sim = StateVectorSimulator::new();
        if let Ok(result) = run_benchmark(
            "Standard Simulator", 
            &standard_sim, 
            circuit, 
            None
        ) {
            results.push(result);
        }
    }
    
    // Run simple optimized simulator
    let simple_opt_sim = OptimizedSimulatorSimple::new();
    if let Ok(result) = run_benchmark(
        "Simple Optimized", 
        &simple_opt_sim, 
        circuit, 
        None
    ) {
        results.push(result);
    }
    
    // Run chunked optimized simulator
    let chunked_sim = OptimizedSimulatorChunked::new();
    if let Ok(result) = run_benchmark(
        "Chunked Optimized", 
        &chunked_sim, 
        circuit, 
        None
    ) {
        results.push(result);
    }
    
    // Run full optimized simulator with automatic selection
    let optimized_sim = OptimizedSimulator::new();
    if let Ok(result) = run_benchmark(
        "Full Optimized", 
        &optimized_sim, 
        circuit, 
        Some("Auto-selection".to_string())
    ) {
        results.push(result);
    }
    
    // For smaller circuits, also test memory-efficient configuration
    if N <= 20 {
        let memory_efficient_sim = OptimizedSimulator::memory_efficient();
        if let Ok(result) = run_benchmark(
            "Memory Efficient", 
            &memory_efficient_sim, 
            circuit, 
            Some("Low memory".to_string())
        ) {
            results.push(result);
        }
    }
    
    results
}

/// Generate a random quantum circuit for benchmarking
pub fn generate_benchmark_circuit<const N: usize>(num_gates: usize, two_qubit_ratio: f64) -> Circuit<N> {
    let mut circuit = Circuit::new();
    let mut rng = ChaCha8Rng::seed_from_u64(42); // Use fixed seed for reproducibility
    
    for _ in 0..num_gates {
        // Decide if this is a single-qubit or two-qubit gate
        let is_two_qubit = rng.gen::<f64>() < two_qubit_ratio;
        
        if is_two_qubit && N > 1 {
            // Select two different qubits
            let qubit1 = QubitId::new(rng.gen_range(0..N as u32));
            let mut qubit2 = QubitId::new(rng.gen_range(0..N as u32));
            while qubit2 == qubit1 {
                qubit2 = QubitId::new(rng.gen_range(0..N as u32));
            }
            
            // Choose a two-qubit gate
            let gate_type = rng.gen_range(0..3);
            match gate_type {
                0 => {
                    let _ = circuit.cnot(qubit1, qubit2);
                },
                1 => {
                    let _ = circuit.cz(qubit1, qubit2);
                },
                _ => {
                    let _ = circuit.swap(qubit1, qubit2);
                },
            }
        } else {
            // Select a qubit
            let qubit = QubitId::new(rng.gen_range(0..N as u32));
            
            // Choose a single-qubit gate
            let gate_type = rng.gen_range(0..7);
            match gate_type {
                0 => {
                    let _ = circuit.h(qubit);
                },
                1 => {
                    let _ = circuit.x(qubit);
                },
                2 => {
                    let _ = circuit.y(qubit);
                },
                3 => {
                    let _ = circuit.z(qubit);
                },
                4 => {
                    let _ = circuit.s(qubit);
                },
                5 => {
                    let _ = circuit.t(qubit);
                },
                _ => {
                    // Random rotation
                    let angle = rng.gen_range(0.0..std::f64::consts::TAU);
                    let rotation_type = rng.gen_range(0..3);
                    match rotation_type {
                        0 => {
                            let _ = circuit.rx(qubit, angle);
                        },
                        1 => {
                            let _ = circuit.ry(qubit, angle);
                        },
                        _ => {
                            let _ = circuit.rz(qubit, angle);
                        },
                    }
                }
            }
        }
    }
    
    circuit
}

/// Run a comprehensive benchmark suite for circuits of different sizes
pub fn run_benchmark_suite() {
    println!("=== Quantrs Simulator Benchmark Suite ===");
    println!("Testing various circuit sizes with different simulator implementations");
    println!();
    
    // Test small circuits (up to 16 qubits)
    benchmark_circuit_size::<16>("Small", 16, 100, 0.3);
    
    // Test medium circuits (up to 20 qubits)
    benchmark_circuit_size::<20>("Medium", 20, 50, 0.3);
    
    // Test large circuits (up to 25 qubits)
    benchmark_circuit_size::<25>("Large", 25, 20, 0.2);
    
    // Test very large circuits for memory-efficient implementation
    benchmark_large_circuit::<28>("Very Large", 28, 10, 0.1);
}

// Helper function to benchmark a specific circuit size
fn benchmark_circuit_size<const N: usize>(
    size_name: &str,
    max_qubits: usize,
    gates_per_qubit: usize,
    two_qubit_ratio: f64,
) {
    println!("\n=== {} Circuit Tests (up to {} qubits) ===", size_name, max_qubits);
    
    // Only proceed if the template parameter is big enough
    if N < max_qubits {
        println!("Cannot benchmark this size - const generic N is too small");
        return;
    }
    
    // Benchmark increasing qubit counts
    for qubits in [max_qubits / 4, max_qubits / 2, (3 * max_qubits) / 4, max_qubits] {
        let num_gates = qubits * gates_per_qubit;
        println!("\nCircuit with {} qubits and {} gates:", qubits, num_gates);
        
        // Generate the circuit
        let circuit = generate_benchmark_circuit::<N>(num_gates, two_qubit_ratio);
        
        // Run all benchmarks
        let results = compare_simulators(&circuit);
        
        // Print results
        for result in &results {
            println!("{}\n", result.format());
        }
    }
}

// Special benchmark for very large circuits
fn benchmark_large_circuit<const N: usize>(
    size_name: &str,
    max_qubits: usize,
    gates_per_qubit: usize,
    two_qubit_ratio: f64,
) {
    println!("\n=== {} Circuit Tests ({} qubits) ===", size_name, max_qubits);
    
    // Only proceed if the template parameter is big enough
    if N < max_qubits {
        println!("Cannot benchmark this size - const generic N is too small");
        return;
    }
    
    let num_gates = max_qubits * gates_per_qubit;
    println!("\nCircuit with {} qubits and {} gates:", max_qubits, num_gates);
    
    // Generate the circuit
    let circuit = generate_benchmark_circuit::<N>(num_gates, two_qubit_ratio);
    
    // Only run with chunked and full optimized simulators
    let mut results = Vec::new();
    
    // Run chunked optimized simulator
    let chunked_sim = OptimizedSimulatorChunked::new();
    if let Ok(result) = run_benchmark(
        "Chunked Optimized", 
        &chunked_sim, 
        &circuit, 
        None
    ) {
        results.push(result);
    }
    
    // Run full optimized simulator
    let optimized_sim = OptimizedSimulator::new();
    if let Ok(result) = run_benchmark(
        "Full Optimized", 
        &optimized_sim, 
        &circuit, 
        Some("Auto-selection".to_string())
    ) {
        results.push(result);
    }
    
    // Print results
    for result in &results {
        println!("{}\n", result.format());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[ignore] // Run only when explicitly requested, not during regular testing
    fn benchmark_small_circuit() {
        const QUBITS: usize = 16;
        let circuit = generate_benchmark_circuit::<QUBITS>(100, 0.3);
        
        let results = compare_simulators(&circuit);
        
        for result in &results {
            println!("{}\n", result.format());
        }
    }
    
    #[test]
    #[ignore] // Run only when explicitly requested, not during regular testing
    fn benchmark_medium_circuit() {
        const QUBITS: usize = 20;
        let circuit = generate_benchmark_circuit::<QUBITS>(50, 0.3);
        
        let results = compare_simulators(&circuit);
        
        for result in &results {
            println!("{}\n", result.format());
        }
    }
    
    #[test]
    #[ignore] // Run only when explicitly requested, not during regular testing
    fn benchmark_large_circuit() {
        const QUBITS: usize = 25;
        let circuit = generate_benchmark_circuit::<QUBITS>(20, 0.2);
        
        // For large circuits, only compare the optimized implementations
        let mut results = Vec::new();
        
        // Run optimized simulators
        let chunked_sim = OptimizedSimulatorChunked::new();
        if let Ok(result) = run_benchmark(
            "Chunked Optimized", 
            &chunked_sim, 
            &circuit, 
            None
        ) {
            results.push(result);
        }
        
        let optimized_sim = OptimizedSimulator::new();
        if let Ok(result) = run_benchmark(
            "Full Optimized", 
            &optimized_sim, 
            &circuit, 
            Some("Auto-selection".to_string())
        ) {
            results.push(result);
        }
        
        for result in &results {
            println!("{}\n", result.format());
        }
    }
    
    #[test]
    #[ignore] // Run only when explicitly requested, not during regular testing
    fn run_full_benchmark_suite() {
        run_benchmark_suite();
    }
}