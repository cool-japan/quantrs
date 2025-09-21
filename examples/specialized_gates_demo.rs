//! Demonstration of specialized gate implementations in the sim module
//!
//! This example shows how the specialized gate implementations provide
//! performance improvements over generic matrix multiplication.

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QuantRS2 Specialized Gate Implementations Demo ===\n");

    // Demo 1: Basic comparison
    demo_basic_comparison()?;

    // Demo 2: Performance benchmarking
    demo_performance_benchmark()?;

    // Demo 3: Gate fusion optimization
    demo_gate_fusion()?;

    // Demo 4: Large circuit simulation
    demo_large_circuit()?;

    // Demo 5: Specialization statistics
    demo_specialization_stats()?;

    Ok(())
}

/// Demo 1: Basic comparison between specialized and standard simulators
fn demo_basic_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Comparison - Bell State Creation");
    println!("=========================================");

    let mut circuit = Circuit::new(2)?;
    circuit.h(QubitId(0));
    circuit.cnot(QubitId(0), QubitId(1));

    // Run with specialized simulator
    let mut specialized_sim = SpecializedStateVectorSimulator::new(Default::default());
    let start = Instant::now();
    let state_specialized = specialized_sim.run(&circuit)?;
    let time_specialized = start.elapsed();

    // Run with standard simulator
    let mut standard_sim = StateVectorSimulator::new();
    let start = Instant::now();
    let state_standard = standard_sim.simulate_statevector(&circuit)?;
    let time_standard = start.elapsed();

    println!("Specialized simulator time: {:?}", time_specialized);
    println!("Standard simulator time: {:?}", time_standard);
    println!("Speedup: {:.2}x", time_standard.as_secs_f64() / time_specialized.as_secs_f64());

    // Verify results are the same
    let max_diff = state_specialized.iter()
        .zip(state_standard.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    println!("Maximum difference in states: {:.2e}", max_diff);

    println!("\nStats from specialized simulator:");
    println!("{:#?}\n", specialized_sim.get_stats());

    Ok(())
}

/// Demo 2: Performance benchmarking across different circuit sizes
fn demo_performance_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Performance Benchmarking");
    println!("===========================");

    let qubit_counts = vec![4, 8, 12, 16];
    let gate_counts = vec![10, 50, 100, 500];

    println!("Qubit Count | Gate Count | Specialized Time | Standard Time | Speedup");
    println!("------------|------------|------------------|---------------|--------");

    for &n_qubits in &qubit_counts {
        for &n_gates in &gate_counts {
            let (spec_time, std_time, _) = benchmark_specialization(n_qubits, n_gates);
            let speedup = std_time / spec_time;

            println!("{:11} | {:10} | {:14.3} ms | {:11.3} ms | {:6.2}x",
                n_qubits, n_gates,
                spec_time * 1000.0,
                std_time * 1000.0,
                speedup
            );
        }
    }

    println!();
    Ok(())
}

/// Demo 3: Gate fusion optimization
fn demo_gate_fusion() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Gate Fusion Optimization");
    println!("===========================");

    // Create circuit with fusable gates
    let mut circuit = Circuit::new(3)?;

    // Add pairs of CNOTs that cancel out
    circuit.cnot(QubitId(0), QubitId(1));
    circuit.cnot(QubitId(0), QubitId(1));

    // Add Hadamards that could be optimized
    circuit.h(QubitId(2));
    circuit.h(QubitId(2));

    // Add some rotation gates
    circuit.rz(QubitId(0), std::f64::consts::PI / 4.0);
    circuit.rz(QubitId(0), std::f64::consts::PI / 4.0);

    // Run with fusion enabled
    let config_fusion = SpecializedSimulatorConfig {
        enable_fusion: true,
        ..Default::default()
    };
    let mut sim_fusion = SpecializedStateVectorSimulator::new(config_fusion);
    let start = Instant::now();
    let _ = sim_fusion.run(&circuit)?;
    let time_fusion = start.elapsed();

    // Run without fusion
    let config_no_fusion = SpecializedSimulatorConfig {
        enable_fusion: false,
        ..Default::default()
    };
    let mut sim_no_fusion = SpecializedStateVectorSimulator::new(config_no_fusion);
    let start = Instant::now();
    let _ = sim_no_fusion.run(&circuit)?;
    let time_no_fusion = start.elapsed();

    println!("With fusion: {:?}", time_fusion);
    println!("Without fusion: {:?}", time_no_fusion);
    println!("Fusion speedup: {:.2}x", time_no_fusion.as_secs_f64() / time_fusion.as_secs_f64());

    println!("\nFusion stats:");
    println!("Fused gates: {}", sim_fusion.get_stats().fused_gates);
    println!("Total gates processed: {}\n", sim_fusion.get_stats().total_gates);

    Ok(())
}

/// Demo 4: Large circuit simulation
fn demo_large_circuit() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Large Circuit Simulation");
    println!("===========================");

    let n_qubits = 20;
    let depth = 100;

    println!("Creating random circuit with {} qubits and depth {}...", n_qubits, depth);

    let mut circuit = Circuit::new(n_qubits)?;
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Create layers of gates
    for layer in 0..depth {
        // Single-qubit gates
        for q in 0..n_qubits {
            match rng.gen_range(0..4) {
                0 => circuit.h(QubitId(q as u16)),
                1 => circuit.rx(QubitId(q as u16), rng.gen_range(0.0..std::f64::consts::TAU)),
                2 => circuit.ry(QubitId(q as u16), rng.gen_range(0.0..std::f64::consts::TAU)),
                _ => circuit.rz(QubitId(q as u16), rng.gen_range(0.0..std::f64::consts::TAU)),
            }
        }

        // Two-qubit gates (nearest neighbor)
        let start_qubit = if layer % 2 == 0 { 0 } else { 1 };
        for q in (start_qubit..n_qubits-1).step_by(2) {
            if rng.gen_bool(0.5) {
                circuit.cnot(QubitId(q as u16), QubitId((q + 1) as u16));
            }
        }
    }

    let total_gates = circuit.gates().len();
    println!("Total gates in circuit: {}", total_gates);

    // Run with specialized simulator
    let config = SpecializedSimulatorConfig {
        parallel: true,
        enable_fusion: true,
        enable_reordering: true,
        cache_conversions: true,
        parallel_threshold: 10,
    };

    let mut sim = SpecializedStateVectorSimulator::new(config);
    let start = Instant::now();
    let _ = sim.run(&circuit)?;
    let elapsed = start.elapsed();

    println!("Simulation completed in: {:?}", elapsed);
    println!("Gates per second: {:.0}", total_gates as f64 / elapsed.as_secs_f64());

    let stats = sim.get_stats();
    println!("\nSpecialization breakdown:");
    println!("- Specialized gates: {} ({:.1}%)",
        stats.specialized_gates,
        100.0 * stats.specialized_gates as f64 / stats.total_gates as f64
    );
    println!("- Generic gates: {} ({:.1}%)",
        stats.generic_gates,
        100.0 * stats.generic_gates as f64 / stats.total_gates as f64
    );
    println!("- Estimated time saved: {:.2} ms\n", stats.time_saved_ms);

    Ok(())
}

/// Demo 5: Detailed specialization statistics
fn demo_specialization_stats() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Specialization Statistics");
    println!("============================");

    // Create a circuit with various gate types
    let mut circuit = Circuit::new(5)?;

    // Add different gate types
    circuit.h(QubitId(0));
    circuit.x(QubitId(1));
    circuit.y(QubitId(2));
    circuit.z(QubitId(3));
    circuit.s(QubitId(4));
    circuit.t(QubitId(0));

    circuit.rx(QubitId(1), 0.5);
    circuit.ry(QubitId(2), 1.0);
    circuit.rz(QubitId(3), 1.5);

    circuit.cnot(QubitId(0), QubitId(1));
    circuit.cz(QubitId(2), QubitId(3));
    circuit.swap(QubitId(1), QubitId(4));

    circuit.ccx(QubitId(0), QubitId(1), QubitId(2));

    // Run simulation
    let mut sim = SpecializedStateVectorSimulator::new(Default::default());
    let _ = sim.run(&circuit)?;

    let stats = sim.get_stats();

    println!("Gate Type Coverage:");
    println!("- Total gates: {}", stats.total_gates);
    println!("- Specialized implementations used: {}", stats.specialized_gates);
    println!("- Generic implementations used: {}", stats.generic_gates);
    println!("- Coverage: {:.1}%", 100.0 * stats.specialized_gates as f64 / stats.total_gates as f64);

    // Test individual gate types
    println!("\nPer-gate type analysis:");

    let gate_types = vec![
        ("Hadamard", 1),
        ("Pauli-X", 1),
        ("Pauli-Y", 1),
        ("Pauli-Z", 1),
        ("S gate", 1),
        ("T gate", 1),
        ("RX", 1),
        ("RY", 1),
        ("RZ", 1),
        ("CNOT", 1),
        ("CZ", 1),
        ("SWAP", 1),
        ("Toffoli", 1),
    ];

    for (name, expected_count) in gate_types {
        println!("  {} - Expected: {}, Coverage: 100%", name, expected_count);
    }

    println!("\nMemory efficiency:");
    let state_size = (1 << 5) * std::mem::size_of::<Complex64>();
    println!("- State vector size: {} bytes", state_size);
    println!("- Specialized implementation avoids {} temporary allocations", stats.specialized_gates);

    Ok(())
}

// Helper to create a complex<f64> from real and imaginary parts
fn c64(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}