//! Example for using the GPU-accelerated quantum simulator
//!
//! This example demonstrates how to use the GPU-accelerated simulator
//! to run quantum circuits. It creates a simple circuit and runs it
//! on both the CPU and GPU simulators, comparing the results.

use quantrs_circuit::prelude::*;
use quantrs_sim::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPU-Accelerated Quantum Simulation Example");
    println!("------------------------------------------");

    // Check if GPU acceleration is available
    let gpu_available = quantrs_sim::gpu::GpuStateVectorSimulator::is_available();
    println!("GPU acceleration available: {}", gpu_available);

    if !gpu_available {
        println!("This example requires GPU acceleration. Exiting.");
        return Ok(());
    }

    // Create a GPU simulator
    let gpu_sim = quantrs_sim::gpu::GpuStateVectorSimulator::new_blocking()?;

    // Create a CPU simulator for comparison
    let cpu_sim = quantrs_sim::statevector::StateVectorSimulator::new();

    // Run simulations with increasing qubit counts
    for n_qubits in [5, 10, 15, 20] {
        println!("\nSimulating with {} qubits:", n_qubits);

        match n_qubits {
            5 => run_comparison::<5>(&cpu_sim, &gpu_sim),
            10 => run_comparison::<10>(&cpu_sim, &gpu_sim),
            15 => run_comparison::<15>(&cpu_sim, &gpu_sim),
            20 => run_comparison::<20>(&cpu_sim, &gpu_sim),
            _ => unreachable!(),
        }
    }

    Ok(())
}

/// Run a comparison between CPU and GPU simulators
fn run_comparison<const N: usize>(cpu_sim: &impl Simulator, gpu_sim: &impl Simulator) {
    // Create a circuit with random gates
    let mut circuit = Circuit::<N>::new();

    // Apply Hadamard gates to all qubits
    for i in 0..N {
        circuit = circuit.h(QubitId::new(i));
    }

    // Apply some CNOT gates
    for i in 0..(N - 1) {
        circuit = circuit.cnot(QubitId::new(i), QubitId::new(i + 1));
    }

    // Apply some rotation gates
    for i in 0..N {
        circuit = circuit.rx(QubitId::new(i), std::f64::consts::PI / 4.0);
    }

    // Run on CPU
    let cpu_start = Instant::now();
    let _cpu_result = cpu_sim.run(&circuit);
    let cpu_time = cpu_start.elapsed();

    // Run on GPU
    let gpu_start = Instant::now();
    let _gpu_result = gpu_sim.run(&circuit);
    let gpu_time = gpu_start.elapsed();

    // Print timing comparison
    println!("CPU time: {:?}", cpu_time);
    println!("GPU time: {:?}", gpu_time);
    println!(
        "Speedup: {:.2}x",
        cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
    );

    // Optional: compare first few amplitudes to verify correctness
    /*
    println!("First 4 state amplitudes:");
    for i in 0..4.min(1 << N) {
        println!(
            "  |{:0width$b}⟩: CPU={:.6}, GPU={:.6}",
            i,
            cpu_result.amplitudes[i].norm(),
            gpu_result.amplitudes[i].norm(),
            width = N
        );
    }
    */
}
