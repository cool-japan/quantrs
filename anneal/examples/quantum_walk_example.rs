#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Example demonstrating quantum walk-based optimization
//!
//! This example shows how to:
//! 1. Use different quantum walk algorithms (CTQW, DTQW, AQWO, QAOA)
//! 2. Configure quantum evolution parameters
//! 3. Apply amplitude amplification for solution enhancement
//! 4. Compare with classical optimization methods

use quantrs2_anneal::{
    ising::IsingModel,
    quantum_walk::{
        AdiabaticHamiltonian, CoinOperator, QuantumState, QuantumWalkAlgorithm, QuantumWalkConfig,
        QuantumWalkOptimizer,
    },
    simulator::{AnnealingParams, QuantumAnnealingSimulator},
};
use scirs2_core::random::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Quantum Walk-Based Optimization Demo ===\n");

    // Example 1: Continuous-time quantum walk (CTQW)
    println!("Example 1: Continuous-Time Quantum Walk");
    continuous_time_walk_example()?;

    // Example 2: Discrete-time quantum walk (DTQW)
    println!("\nExample 2: Discrete-Time Quantum Walk");
    discrete_time_walk_example()?;

    // Example 3: Adiabatic quantum walk optimization (AQWO)
    println!("\nExample 3: Adiabatic Quantum Walk Optimization");
    adiabatic_walk_example()?;

    // Example 4: QAOA with quantum walk
    println!("\nExample 4: QAOA with Quantum Walk");
    qaoa_walk_example()?;

    // Example 5: Amplitude amplification
    println!("\nExample 5: Quantum Walk with Amplitude Amplification");
    amplitude_amplification_example()?;

    // Example 6: Performance comparison
    println!("\nExample 6: Performance Comparison");
    performance_comparison_example()?;

    // Example 7: Quantum state analysis
    println!("\nExample 7: Quantum State Analysis");
    quantum_state_analysis_example()?;

    Ok(())
}

fn continuous_time_walk_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple MaxCut problem
    let mut model = IsingModel::new(4);
    model.set_coupling(0, 1, -1.0)?;
    model.set_coupling(1, 2, -1.0)?;
    model.set_coupling(2, 3, -1.0)?;
    model.set_coupling(3, 0, -1.0)?;

    // Configure continuous-time quantum walk
    let config = QuantumWalkConfig {
        algorithm: QuantumWalkAlgorithm::ContinuousTime {
            evolution_time: 2.0,
            time_steps: 200,
        },
        num_measurements: 500,
        seed: Some(42),
        amplitude_amplification: false,
        ..Default::default()
    };

    let start = Instant::now();
    let mut optimizer = QuantumWalkOptimizer::new(config);
    let result = optimizer.solve(&model)?;
    let runtime = start.elapsed();

    println!("Continuous-Time Quantum Walk Results:");
    println!("  Problem: 4-vertex cycle graph MaxCut");
    println!("  Evolution time: 2.0");
    println!("  Time steps: 200");
    println!("  Best energy: {:.4}", result.best_energy);
    println!("  Best configuration: {:?}", result.best_spins);
    println!("  Runtime: {runtime:.2?}");
    println!("  Measurements: {}", result.total_sweeps);

    // Analyze the solution
    let cut_edges = count_cut_edges(&result.best_spins, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
    println!("  Cut edges: {cut_edges}/4");

    Ok(())
}

fn discrete_time_walk_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a triangular graph problem
    let mut model = IsingModel::new(3);
    model.set_coupling(0, 1, -1.0)?;
    model.set_coupling(1, 2, -1.0)?;
    model.set_coupling(2, 0, -1.0)?;

    // Configure discrete-time quantum walk with Hadamard coin
    let config = QuantumWalkConfig {
        algorithm: QuantumWalkAlgorithm::DiscreteTime {
            coin_operator: CoinOperator::Hadamard,
            steps: 50,
        },
        num_measurements: 300,
        seed: Some(123),
        ..Default::default()
    };

    let start = Instant::now();
    let mut optimizer = QuantumWalkOptimizer::new(config);
    let result = optimizer.solve(&model)?;
    let runtime = start.elapsed();

    println!("Discrete-Time Quantum Walk Results:");
    println!("  Problem: Triangle graph MaxCut");
    println!("  Coin operator: Hadamard");
    println!("  Walk steps: 50");
    println!("  Best energy: {:.4}", result.best_energy);
    println!("  Best configuration: {:?}", result.best_spins);
    println!("  Runtime: {runtime:.2?}");

    // For triangle, theoretical minimum is -1 (can cut at most 2 edges)
    println!("  Theoretical minimum: -1.0");
    println!("  Gap to optimal: {:.4}", result.best_energy - (-1.0));

    Ok(())
}

fn adiabatic_walk_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a more complex problem
    let mut model = IsingModel::new(5);

    // Create a frustrated system
    for i in 0..5 {
        for j in (i + 1)..5 {
            if thread_rng().gen::<f64>() < 0.4 {
                let coupling = if thread_rng().gen::<bool>() {
                    1.0
                } else {
                    -1.0
                };
                model.set_coupling(i, j, coupling)?;
            }
        }
    }

    // Configure adiabatic quantum walk
    let config = QuantumWalkConfig {
        algorithm: QuantumWalkAlgorithm::Adiabatic {
            initial_hamiltonian: AdiabaticHamiltonian::Mixing,
            final_hamiltonian: AdiabaticHamiltonian::Problem,
            evolution_time: 5.0,
            time_steps: 100,
        },
        num_measurements: 800,
        seed: Some(456),
        ..Default::default()
    };

    let start = Instant::now();
    let mut optimizer = QuantumWalkOptimizer::new(config);
    let result = optimizer.solve(&model)?;
    let runtime = start.elapsed();

    println!("Adiabatic Quantum Walk Results:");
    println!("  Problem: 5-qubit random frustrated system");
    println!("  Evolution time: 5.0");
    println!("  Adiabatic steps: 100");
    println!("  Best energy: {:.4}", result.best_energy);
    println!("  Best configuration: {:?}", result.best_spins);
    println!("  Runtime: {runtime:.2?}");

    Ok(())
}

fn qaoa_walk_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a MaxCut problem on a small complete graph
    let mut model = IsingModel::new(4);

    // Complete graph with all negative couplings
    for i in 0..4 {
        for j in (i + 1)..4 {
            model.set_coupling(i, j, -1.0)?;
        }
    }

    // Configure QAOA-style quantum walk
    let beta_schedule = vec![0.1, 0.3, 0.5, 0.7];
    let gamma_schedule = vec![0.2, 0.4, 0.6, 0.8];

    let config = QuantumWalkConfig {
        algorithm: QuantumWalkAlgorithm::QaoaWalk {
            layers: 4,
            beta_schedule: beta_schedule.clone(),
            gamma_schedule: gamma_schedule.clone(),
        },
        num_measurements: 1000,
        seed: Some(789),
        ..Default::default()
    };

    let start = Instant::now();
    let mut optimizer = QuantumWalkOptimizer::new(config);
    let result = optimizer.solve(&model)?;
    let runtime = start.elapsed();

    println!("QAOA Quantum Walk Results:");
    println!("  Problem: Complete graph K4 MaxCut");
    println!("  QAOA layers: 4");
    println!("  Beta schedule: {beta_schedule:?}");
    println!("  Gamma schedule: {gamma_schedule:?}");
    println!("  Best energy: {:.4}", result.best_energy);
    println!("  Best configuration: {:?}", result.best_spins);
    println!("  Runtime: {runtime:.2?}");

    // K4 MaxCut optimal is -4 (can cut 4 out of 6 edges)
    let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let cut_edges = count_cut_edges(&result.best_spins, &edges);
    println!("  Cut edges: {cut_edges}/6");
    println!("  Theoretical maximum: 4 edges");

    Ok(())
}

fn amplitude_amplification_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a problem where amplitude amplification should help
    let mut model = IsingModel::new(6);

    // Create a problem with a unique ground state
    model.set_bias(0, -2.0)?; // Strong bias towards spin down
    for i in 1..6 {
        model.set_coupling(0, i, -0.5)?; // Couple to center qubit
    }

    // Without amplitude amplification
    let config_no_amp = QuantumWalkConfig {
        algorithm: QuantumWalkAlgorithm::ContinuousTime {
            evolution_time: 1.5,
            time_steps: 150,
        },
        num_measurements: 500,
        amplitude_amplification: false,
        seed: Some(42),
        ..Default::default()
    };

    let start = Instant::now();
    let mut optimizer_no_amp = QuantumWalkOptimizer::new(config_no_amp);
    let result_no_amp = optimizer_no_amp.solve(&model)?;
    let time_no_amp = start.elapsed();

    // With amplitude amplification
    let config_with_amp = QuantumWalkConfig {
        algorithm: QuantumWalkAlgorithm::ContinuousTime {
            evolution_time: 1.5,
            time_steps: 150,
        },
        num_measurements: 500,
        amplitude_amplification: true,
        amplification_iterations: 3,
        seed: Some(42),
        ..Default::default()
    };

    let start = Instant::now();
    let mut optimizer_with_amp = QuantumWalkOptimizer::new(config_with_amp);
    let result_with_amp = optimizer_with_amp.solve(&model)?;
    let time_with_amp = start.elapsed();

    println!("Amplitude Amplification Comparison:");
    println!("  Problem: 6-qubit system with unique ground state");

    println!("\nWithout Amplitude Amplification:");
    println!("  Best energy: {:.4}", result_no_amp.best_energy);
    println!("  Runtime: {time_no_amp:.2?}");

    println!("\nWith Amplitude Amplification:");
    println!("  Best energy: {:.4}", result_with_amp.best_energy);
    println!("  Runtime: {time_with_amp:.2?}");

    let improvement = result_no_amp.best_energy - result_with_amp.best_energy;
    println!("\nImprovement: {improvement:.4}");
    println!(
        "Amplification overhead: {:.2?}",
        time_with_amp.checked_sub(time_no_amp).unwrap()
    );

    Ok(())
}

fn performance_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a benchmark problem
    let mut model = IsingModel::new(8);

    // Random spin glass
    for i in 0..8 {
        for j in (i + 1)..8 {
            if thread_rng().gen::<f64>() < 0.3 {
                let coupling = (thread_rng().gen::<f64>() - 0.5) * 2.0;
                model.set_coupling(i, j, coupling)?;
            }
        }
    }

    println!("Performance Comparison on 8-qubit Random Spin Glass:");

    // Quantum walk optimization
    let qw_config = QuantumWalkConfig {
        algorithm: QuantumWalkAlgorithm::ContinuousTime {
            evolution_time: 3.0,
            time_steps: 300,
        },
        num_measurements: 1000,
        seed: Some(42),
        ..Default::default()
    };

    let start = Instant::now();
    let mut qw_optimizer = QuantumWalkOptimizer::new(qw_config);
    let qw_result = qw_optimizer.solve(&model)?;
    let qw_time = start.elapsed();

    // Classical quantum annealing for comparison
    let classical_params = AnnealingParams {
        num_sweeps: 5000,
        num_repetitions: 20,
        seed: Some(42),
        ..Default::default()
    };

    let start = Instant::now();
    let mut classical_solver = QuantumAnnealingSimulator::new(classical_params)?;
    let classical_result = classical_solver.solve(&model)?;
    let classical_time = start.elapsed();

    println!("\nQuantum Walk Results:");
    println!("  Best energy: {:.4}", qw_result.best_energy);
    println!("  Runtime: {qw_time:.2?}");
    println!("  Measurements: {}", qw_result.total_sweeps);

    println!("\nClassical Annealing Results:");
    println!("  Best energy: {:.4}", classical_result.best_energy);
    println!("  Runtime: {classical_time:.2?}");
    println!("  Total sweeps: {}", classical_result.total_sweeps);

    println!("\nComparison:");
    let energy_diff = classical_result.best_energy - qw_result.best_energy;
    let speedup = classical_time.as_secs_f64() / qw_time.as_secs_f64();

    println!("  Energy difference: {energy_diff:.4}");
    println!("  Speedup: {speedup:.2}x");
    println!(
        "  QW advantage: {}",
        if energy_diff > 0.0 {
            "Better energy"
        } else {
            "Faster time"
        }
    );

    Ok(())
}

fn quantum_state_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Quantum State Analysis:");

    // Create and analyze different quantum states
    let num_qubits = 3;

    // Initial state |000⟩
    let initial_state = QuantumState::new(num_qubits);
    println!("  Initial state |000⟩:");
    print_state_info(&initial_state);

    // Uniform superposition state
    let uniform_state = QuantumState::uniform_superposition(num_qubits);
    println!("\n  Uniform superposition state:");
    print_state_info(&uniform_state);

    // Demonstrate state evolution concepts
    println!("\n  State evolution concepts:");
    println!("    - CTQW: Continuous evolution under problem Hamiltonian");
    println!("    - DTQW: Discrete steps with coin + shift operations");
    println!("    - AQWO: Adiabatic interpolation between Hamiltonians");
    println!("    - QAOA: Alternating problem and mixing Hamiltonian layers");

    println!("\n  Measurement and sampling:");
    println!("    - Quantum state contains probability amplitudes");
    println!("    - Measurements collapse to computational basis states");
    println!("    - Multiple measurements needed for optimization");

    Ok(())
}

/// Print information about a quantum state
fn print_state_info(state: &QuantumState) {
    println!("    Number of qubits: {}", state.num_qubits);
    println!("    Number of amplitudes: {}", state.amplitudes.len());

    // Show first few amplitudes
    println!("    Amplitudes:");
    for (i, amp) in state.amplitudes.iter().take(8).enumerate() {
        let bits = state.state_to_bits(i);
        let prob = amp.norm_sqr();
        println!(
            "      |{:?}⟩: ({:.3} + {:.3}i), P = {:.3}",
            bits.iter().map(|&b| i32::from(b > 0)).collect::<Vec<_>>(),
            amp.re,
            amp.im,
            prob
        );
    }

    if state.amplitudes.len() > 8 {
        println!("      ... ({} more states)", state.amplitudes.len() - 8);
    }

    // Calculate and show total probability
    let total_prob: f64 = state
        .amplitudes
        .iter()
        .map(scirs2_core::Complex::norm_sqr)
        .sum();
    println!("    Total probability: {total_prob:.6}");
}

/// Count the number of cut edges in a graph
fn count_cut_edges(spins: &[i8], edges: &[(usize, usize)]) -> usize {
    edges.iter().filter(|&&(i, j)| spins[i] != spins[j]).count()
}
