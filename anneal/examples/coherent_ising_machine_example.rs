#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Example demonstrating Coherent Ising Machine simulation
//!
//! This example shows how to:
//! 1. Configure different network topologies (fully connected, ring, lattice, random)
//! 2. Use various pump power schedules (linear, exponential, sigmoid, custom)
//! 3. Apply different noise models and measurement configurations
//! 4. Solve optimization problems with photonic quantum annealing
//! 5. Analyze optical system performance and convergence behavior
//! 6. Compare CIM performance across different configurations

use quantrs2_anneal::{
    coherent_ising_machine::{
        create_low_noise_cim_config, create_realistic_cim_config, create_standard_cim_config,
        CimConfig, CoherentIsingMachine, ConvergenceConfig, MeasurementConfig, NetworkTopology,
        NoiseConfig, PumpSchedule,
    },
    ising::IsingModel,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Coherent Ising Machine Simulation Demo ===\n");

    // Example 1: Standard CIM configuration
    println!("Example 1: Standard CIM Configuration");
    standard_cim_example()?;

    // Example 2: Different network topologies
    println!("\nExample 2: Network Topology Comparison");
    topology_comparison_example()?;

    // Example 3: Pump schedule optimization
    println!("\nExample 3: Pump Schedule Optimization");
    pump_schedule_example()?;

    // Example 4: Noise resilience study
    println!("\nExample 4: Noise Resilience Study");
    noise_study_example()?;

    // Example 5: Large-scale optimization
    println!("\nExample 5: Large-Scale Optimization Problem");
    large_scale_example()?;

    // Example 6: Convergence analysis
    println!("\nExample 6: Convergence Behavior Analysis");
    convergence_analysis_example()?;

    // Example 7: Custom pump schedule
    println!("\nExample 7: Custom Pump Schedule Design");
    custom_pump_schedule_example()?;

    Ok(())
}

fn standard_cim_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple MaxCut problem on a triangle
    let mut problem = IsingModel::new(3);
    problem.set_coupling(0, 1, -1.0)?; // Negative couplings for MaxCut
    problem.set_coupling(1, 2, -1.0)?;
    problem.set_coupling(2, 0, -1.0)?;

    // Use standard CIM configuration
    let config = create_standard_cim_config(3, 10.0);

    let start = Instant::now();
    let mut cim = CoherentIsingMachine::new(config)?;
    let results = cim.solve(&problem)?;
    let runtime = start.elapsed();

    println!("Standard CIM Results:");
    println!("  Problem: MaxCut on triangle graph (3 vertices)");
    println!("  Network: Fully connected");
    println!("  Pump schedule: Linear (0.5 → 1.5)");
    println!("  Simulation time: 10.0 time units");
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Best solution: {:?}", results.best_solution);
    println!("  Converged: {}", results.converged);
    println!("  Convergence time: {:.3}", results.convergence_time);
    println!("  Runtime: {:.2?}", runtime);

    // Optical system analysis
    println!("\n  Optical system statistics:");
    println!(
        "    Average power: {:.6}",
        results.optical_stats.average_power
    );
    println!(
        "    Power variance: {:.6}",
        results.optical_stats.power_variance
    );
    println!(
        "    Pump efficiency: {:.1}%",
        results.optical_stats.pump_efficiency * 100.0
    );

    // Final optical state
    println!("\n  Final optical amplitudes:");
    for (i, amplitude) in results.final_optical_state.iter().enumerate() {
        println!(
            "    Oscillator {}: magnitude = {:.6}, phase = {:.3} rad",
            i,
            amplitude.magnitude(),
            amplitude.phase()
        );
    }

    // Energy evolution
    if results.energy_history.len() >= 10 {
        println!("\n  Energy evolution:");
        for (i, &energy) in results.energy_history.iter().enumerate() {
            if i % (results.energy_history.len() / 5).max(1) == 0
                || i == results.energy_history.len() - 1
            {
                let time = if let Some(&t) = results.time_points.get(i / 100) {
                    t
                } else {
                    i as f64 * 0.01
                };
                println!("    t = {:.2}: Energy = {:.6}", time, energy);
            }
        }
    }

    Ok(())
}

fn topology_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 4-qubit frustrated system
    let mut problem = IsingModel::new(4);

    // Create frustrated couplings
    problem.set_coupling(0, 1, 1.0)?;
    problem.set_coupling(1, 2, 1.0)?;
    problem.set_coupling(2, 3, 1.0)?;
    problem.set_coupling(3, 0, 1.0)?;
    problem.set_coupling(0, 2, -0.5)?; // Frustrating diagonal

    let topologies = vec![
        ("Fully Connected", NetworkTopology::FullyConnected),
        ("Ring", NetworkTopology::Ring),
        (
            "2×2 Lattice",
            NetworkTopology::Lattice2D {
                width: 2,
                height: 2,
            },
        ),
        (
            "Random (70%)",
            NetworkTopology::Random { connectivity: 0.7 },
        ),
        (
            "Small World",
            NetworkTopology::SmallWorld {
                ring_connectivity: 2,
                rewiring_probability: 0.3,
            },
        ),
    ];

    println!("Network Topology Comparison:");
    println!("  Problem: 4-qubit frustrated square with diagonal coupling");
    println!("  Configuration: Standard CIM with 8 time units");

    for (topology_name, topology) in topologies {
        let mut config = create_standard_cim_config(4, 8.0);
        config.topology = topology;
        config.detailed_logging = false;
        config.seed = Some(42); // Same seed for fair comparison

        let start = Instant::now();
        let mut cim = CoherentIsingMachine::new(config)?;
        let results = cim.solve(&problem)?;
        let runtime = start.elapsed();

        println!("\n  {} Results:", topology_name);
        println!("    Best energy: {:.6}", results.best_energy);
        println!("    Converged: {}", results.converged);
        println!("    Convergence time: {:.3}", results.convergence_time);
        println!("    Runtime: {:.2?}", runtime);
        println!(
            "    Average power: {:.6}",
            results.optical_stats.average_power
        );
        println!(
            "    Power variance: {:.6}",
            results.optical_stats.power_variance
        );

        // Network-specific analysis
        match topology_name {
            "Fully Connected" => {
                println!("    Connectivity: All-to-all (6 couplings)");
            }
            "Ring" => {
                println!("    Connectivity: Nearest neighbors only (4 couplings)");
            }
            "2×2 Lattice" => {
                println!("    Connectivity: 2D grid (4 couplings)");
            }
            "Random (70%)" => {
                println!("    Connectivity: ~70% of possible edges");
            }
            "Small World" => {
                println!("    Connectivity: Ring + 30% rewiring");
            }
            _ => {}
        }
    }

    println!("\n  Analysis:");
    println!("    - Fully connected provides strongest coupling but may over-constrain");
    println!("    - Ring topology matches natural optical fiber loops");
    println!("    - Lattice topology suits 2D spatial problems");
    println!("    - Random networks balance connectivity and flexibility");
    println!("    - Small-world combines local and global connections");

    Ok(())
}

fn pump_schedule_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a test problem
    let mut problem = IsingModel::new(4);
    problem.set_bias(0, 0.5)?;
    problem.set_bias(1, -0.3)?;
    problem.set_bias(2, 0.8)?;
    problem.set_bias(3, -0.2)?;
    problem.set_coupling(0, 1, -0.7)?;
    problem.set_coupling(1, 2, 0.4)?;
    problem.set_coupling(2, 3, -0.6)?;
    problem.set_coupling(0, 3, 0.5)?;

    let schedules = vec![
        (
            "Linear",
            PumpSchedule::Linear {
                initial_power: 0.3,
                final_power: 2.0,
            },
        ),
        (
            "Exponential",
            PumpSchedule::Exponential {
                initial_power: 0.3,
                final_power: 2.0,
                time_constant: 3.0,
            },
        ),
        (
            "Sigmoid",
            PumpSchedule::Sigmoid {
                initial_power: 0.3,
                final_power: 2.0,
                steepness: 5.0,
                midpoint: 0.6,
            },
        ),
    ];

    println!("Pump Schedule Optimization:");
    println!("  Problem: 4-qubit mixed bias and coupling system");
    println!("  Configuration: Standard topology, 12 time units");

    for (schedule_name, schedule) in schedules {
        let mut config = create_standard_cim_config(4, 12.0);
        config.pump_schedule = schedule;
        config.detailed_logging = false;
        config.seed = Some(123);

        let start = Instant::now();
        let mut cim = CoherentIsingMachine::new(config)?;
        let results = cim.solve(&problem)?;
        let runtime = start.elapsed();

        println!("\n  {} Schedule Results:", schedule_name);
        println!("    Best energy: {:.6}", results.best_energy);
        println!("    Converged: {}", results.converged);
        println!("    Convergence time: {:.3}", results.convergence_time);
        println!("    Runtime: {:.2?}", runtime);
        println!(
            "    Final power efficiency: {:.1}%",
            results.performance_metrics.power_efficiency * 100.0
        );

        // Schedule-specific analysis
        match schedule_name {
            "Linear" => {
                println!("    Characteristics: Constant rate, predictable behavior");
            }
            "Exponential" => {
                println!("    Characteristics: Fast initial rise, gradual saturation");
            }
            "Sigmoid" => {
                println!("    Characteristics: Slow start, rapid transition, plateau");
            }
            _ => {}
        }

        // Energy improvement analysis
        if results.energy_history.len() > 1 {
            let initial_energy = results.energy_history[0];
            let final_energy = *results.energy_history.last().unwrap();
            let improvement = initial_energy - final_energy;
            println!("    Energy improvement: {:.6}", improvement);
        }
    }

    println!("\n  Schedule Selection Guidelines:");
    println!("    - Linear: Good general-purpose choice, stable convergence");
    println!("    - Exponential: Fast convergence for well-conditioned problems");
    println!("    - Sigmoid: Better control near threshold, reduced noise impact");

    Ok(())
}

fn noise_study_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a sensitive optimization problem
    let mut problem = IsingModel::new(5);

    // Create a frustrated ring with weak couplings (sensitive to noise)
    for i in 0..5 {
        let j = (i + 1) % 5;
        problem.set_coupling(i, j, 0.2)?; // Weak frustrating couplings
    }

    let noise_levels = vec![
        (
            "Ultra Low Noise",
            NoiseConfig {
                quantum_noise: 0.001,
                phase_noise: 0.0001,
                amplitude_noise: 0.0001,
                temperature: 0.001,
                decoherence_rate: 0.0001,
            },
        ),
        (
            "Low Noise",
            NoiseConfig {
                quantum_noise: 0.01,
                phase_noise: 0.001,
                amplitude_noise: 0.001,
                temperature: 0.01,
                decoherence_rate: 0.001,
            },
        ),
        (
            "Moderate Noise",
            NoiseConfig {
                quantum_noise: 0.05,
                phase_noise: 0.01,
                amplitude_noise: 0.01,
                temperature: 0.1,
                decoherence_rate: 0.01,
            },
        ),
        (
            "High Noise",
            NoiseConfig {
                quantum_noise: 0.1,
                phase_noise: 0.05,
                amplitude_noise: 0.05,
                temperature: 0.5,
                decoherence_rate: 0.05,
            },
        ),
    ];

    println!("Noise Resilience Study:");
    println!("  Problem: 5-qubit frustrated ring with weak couplings");
    println!("  Configuration: Standard CIM, 15 time units");

    for (noise_name, noise_config) in noise_levels {
        let mut config = create_standard_cim_config(5, 15.0);
        config.noise_config = noise_config;
        config.detailed_logging = false;
        config.seed = Some(456);

        let start = Instant::now();
        let mut cim = CoherentIsingMachine::new(config)?;
        let results = cim.solve(&problem)?;
        let runtime = start.elapsed();

        println!("\n  {} Results:", noise_name);
        println!("    Best energy: {:.6}", results.best_energy);
        println!("    Converged: {}", results.converged);
        println!("    Convergence time: {:.3}", results.convergence_time);
        println!("    Runtime: {:.2?}", runtime);

        // Noise impact analysis
        let noise_resilience = results.performance_metrics.noise_resilience;
        println!("    Noise resilience score: {:.3}", noise_resilience);
        println!(
            "    Power variance: {:.6}",
            results.optical_stats.power_variance
        );

        // Phase coherence analysis
        let avg_coherence = results.optical_stats.phase_coherence.iter().sum::<f64>()
            / results.optical_stats.phase_coherence.len() as f64;
        println!("    Average phase coherence: {:.3}", avg_coherence);

        // Energy stability
        if results.energy_history.len() > 10 {
            let last_10_energies = &results.energy_history[results.energy_history.len() - 10..];
            let energy_std = {
                let mean = last_10_energies.iter().sum::<f64>() / 10.0;
                let variance = last_10_energies
                    .iter()
                    .map(|&e| (e - mean).powi(2))
                    .sum::<f64>()
                    / 10.0;
                variance.sqrt()
            };
            println!("    Final energy stability (std): {:.6}", energy_std);
        }
    }

    println!("\n  Noise Impact Summary:");
    println!("    - Quantum noise affects oscillator amplitude fluctuations");
    println!("    - Phase noise disrupts optical interference patterns");
    println!("    - Amplitude noise impacts measurement precision");
    println!("    - Temperature introduces thermal fluctuations");
    println!("    - Decoherence rate affects quantum coherence lifetime");

    Ok(())
}

fn large_scale_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a larger optimization problem
    let problem_size = 8;
    let mut problem = IsingModel::new(problem_size);

    // Create a random graph problem with moderate density
    for i in 0..problem_size {
        // Add biases
        problem.set_bias(i, (i as f64 - 3.5) * 0.1)?;

        for j in (i + 1)..problem_size {
            // Add coupling with 40% probability based on deterministic pattern
            if (i * 7 + j * 11) % 10 < 4 {
                let coupling = if (i + j) % 2 == 0 { 0.3 } else { -0.4 };
                problem.set_coupling(i, j, coupling)?;
            }
        }
    }

    // Use realistic configuration for larger system
    let mut config = create_realistic_cim_config(problem_size);
    config.total_time = 20.0; // Longer simulation for larger system
    config.convergence_config.energy_tolerance = 1e-5;
    config.convergence_config.stagnation_time = 2.0;
    config.seed = Some(789);

    let start = Instant::now();
    let mut cim = CoherentIsingMachine::new(config)?;
    let results = cim.solve(&problem)?;
    let runtime = start.elapsed();

    println!("Large-Scale Optimization Results:");
    println!(
        "  Problem: {}-qubit random graph (~40% edge density)",
        problem_size
    );
    println!("  Configuration: Realistic noise, 20 time units");
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Best solution: {:?}", results.best_solution);
    println!("  Converged: {}", results.converged);
    println!("  Convergence time: {:.3}", results.convergence_time);
    println!("  Runtime: {:.2?}", runtime);

    // Scaling analysis
    println!("\n  Scaling metrics:");
    println!("    Oscillators: {}", problem_size);
    println!("    Simulation time steps: {}", (20.0 / 0.001) as usize);
    println!("    Memory efficiency: Sparse coupling representation");
    println!("    Computational complexity: O(N²) per time step");

    // Performance metrics
    println!("\n  Performance analysis:");
    println!(
        "    Average optical power: {:.6}",
        results.optical_stats.average_power
    );
    println!(
        "    Power efficiency: {:.1}%",
        results.performance_metrics.power_efficiency * 100.0
    );
    println!(
        "    Solution quality score: {:.3}",
        results.performance_metrics.solution_quality
    );
    println!(
        "    Time to convergence: {:.1}%",
        results.performance_metrics.time_to_convergence * 100.0
    );

    // Cross-correlation analysis
    println!("\n  Optical network analysis:");
    let cross_corr_matrix = &results.optical_stats.cross_correlations;
    let avg_cross_correlation = {
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..problem_size {
            for j in (i + 1)..problem_size {
                sum += cross_corr_matrix[i][j];
                count += 1;
            }
        }
        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    };
    println!(
        "    Average cross-correlation: {:.3}",
        avg_cross_correlation
    );

    // Energy landscape exploration
    if results.energy_history.len() > 20 {
        let energy_range = results
            .energy_history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            - results
                .energy_history
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
        println!("    Energy landscape range explored: {:.6}", energy_range);

        // Convergence pattern analysis
        let early_phase = &results.energy_history[0..results.energy_history.len() / 4];
        let late_phase = &results.energy_history[3 * results.energy_history.len() / 4..];

        let early_improvement = early_phase[0] - early_phase[early_phase.len() - 1];
        let late_improvement = late_phase[0] - late_phase[late_phase.len() - 1];

        println!("    Early phase improvement: {:.6}", early_improvement);
        println!("    Late phase improvement: {:.6}", late_improvement);

        if early_improvement > late_improvement * 3.0 {
            println!("    Convergence pattern: Fast initial convergence");
        } else {
            println!("    Convergence pattern: Steady optimization");
        }
    }

    Ok(())
}

fn convergence_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a challenging optimization problem
    let mut problem = IsingModel::new(6);

    // Create a problem with multiple local minima
    for i in 0..6 {
        problem.set_bias(i, (i as i32 - 3) as f64 * 0.2)?;
        for j in (i + 1)..6 {
            if (i + j) % 3 == 0 {
                problem.set_coupling(i, j, if i < j { 0.4 } else { -0.4 })?;
            }
        }
    }

    // Test different convergence criteria
    let convergence_configs = vec![
        (
            "Tight Convergence",
            ConvergenceConfig {
                energy_tolerance: 1e-8,
                stagnation_time: 0.5,
                oscillation_threshold: 0.05,
                phase_stability: 0.005,
            },
        ),
        (
            "Standard Convergence",
            ConvergenceConfig {
                energy_tolerance: 1e-6,
                stagnation_time: 1.0,
                oscillation_threshold: 0.1,
                phase_stability: 0.01,
            },
        ),
        (
            "Relaxed Convergence",
            ConvergenceConfig {
                energy_tolerance: 1e-4,
                stagnation_time: 2.0,
                oscillation_threshold: 0.2,
                phase_stability: 0.02,
            },
        ),
    ];

    println!("Convergence Behavior Analysis:");
    println!("  Problem: 6-qubit multi-modal optimization landscape");
    println!("  Configuration: Low noise CIM, 15 time units");

    for (config_name, conv_config) in convergence_configs {
        let mut config = create_low_noise_cim_config(6);
        config.total_time = 15.0;
        config.convergence_config = conv_config;
        config.detailed_logging = false;
        config.seed = Some(321);

        let start = Instant::now();
        let mut cim = CoherentIsingMachine::new(config)?;
        let results = cim.solve(&problem)?;
        let runtime = start.elapsed();

        println!("\n  {} Results:", config_name);
        println!("    Best energy: {:.6}", results.best_energy);
        println!("    Converged: {}", results.converged);
        println!("    Convergence time: {:.3}", results.convergence_time);
        println!("    Runtime: {:.2?}", runtime);

        // Convergence quality analysis
        if results.energy_history.len() > 5 {
            let final_5_energies = &results.energy_history[results.energy_history.len() - 5..];
            let energy_variance = {
                let mean = final_5_energies.iter().sum::<f64>() / 5.0;
                final_5_energies
                    .iter()
                    .map(|&e| (e - mean).powi(2))
                    .sum::<f64>()
                    / 5.0
            };
            println!("    Final energy variance: {:.8}", energy_variance);

            // Oscillation analysis
            let oscillation_count = final_5_energies
                .windows(2)
                .filter(|pair| (pair[1] - pair[0]).abs() > 1e-6)
                .count();
            println!(
                "    Energy oscillations in final 5 steps: {}",
                oscillation_count
            );
        }

        // Phase stability analysis
        let final_optical_state = &results.final_optical_state;
        let phase_stability = final_optical_state
            .iter()
            .map(|amp| amp.magnitude())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        println!("    Minimum oscillation amplitude: {:.6}", phase_stability);

        // Convergence efficiency
        let efficiency = if results.convergence_time > 0.0 {
            (results.energy_history[0] - results.best_energy) / results.convergence_time
        } else {
            0.0
        };
        println!("    Convergence efficiency: {:.6} energy/time", efficiency);
    }

    println!("\n  Convergence Criteria Guidelines:");
    println!("    - Tight: High precision, longer runtime, best for critical applications");
    println!("    - Standard: Good balance of precision and speed");
    println!("    - Relaxed: Fast convergence, suitable for exploration or large problems");
    println!("\\n  Key Metrics:");
    println!("    - Energy tolerance: Controls energy stability requirement");
    println!("    - Stagnation time: Maximum time without improvement");
    println!("    - Oscillation threshold: Minimum optical power for stability");
    println!("    - Phase stability: Phase coherence requirement");

    Ok(())
}

fn custom_pump_schedule_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a test problem
    let mut problem = IsingModel::new(4);
    problem.set_coupling(0, 1, -1.0)?;
    problem.set_coupling(1, 2, 0.5)?;
    problem.set_coupling(2, 3, -0.7)?;
    problem.set_coupling(0, 3, 0.3)?;
    problem.set_coupling(0, 2, -0.4)?;
    problem.set_coupling(1, 3, 0.6)?;

    // Define custom pump schedules

    // Piecewise linear schedule
    let piecewise_schedule = PumpSchedule::Custom {
        power_function: Box::new(|t| {
            if t < 0.3 {
                0.5 + t * 1.5 // Linear rise
            } else if t < 0.7 {
                1.0 // Constant plateau
            } else {
                1.0 + (t - 0.7) * 3.0 // Steep final rise
            }
        }),
    };

    // Oscillating schedule
    let oscillating_schedule = PumpSchedule::Custom {
        power_function: Box::new(|t| {
            let base = 0.5 + t * 1.5;
            let oscillation = 0.1 * (t * 20.0 * std::f64::consts::PI).sin();
            base + oscillation
        }),
    };

    // Exponential decay to threshold
    let threshold_approach_schedule = PumpSchedule::Custom {
        power_function: Box::new(|t| {
            let threshold = 1.0;
            let max_power = 2.0;
            threshold + (max_power - threshold) * (1.0 - (-5.0 * t).exp())
        }),
    };

    let custom_schedules = vec![
        ("Piecewise Linear", piecewise_schedule),
        ("Oscillating", oscillating_schedule),
        ("Threshold Approach", threshold_approach_schedule),
    ];

    println!("Custom Pump Schedule Design:");
    println!("  Problem: 4-qubit fully connected mixed coupling system");
    println!("  Configuration: Standard CIM, 10 time units");

    for (schedule_name, schedule) in custom_schedules {
        let mut config = create_standard_cim_config(4, 10.0);
        config.pump_schedule = schedule;
        config.detailed_logging = false;
        config.seed = Some(654);

        let start = Instant::now();
        let mut cim = CoherentIsingMachine::new(config)?;
        let results = cim.solve(&problem)?;
        let runtime = start.elapsed();

        println!("\\n  {} Schedule Results:", schedule_name);
        println!("    Best energy: {:.6}", results.best_energy);
        println!("    Converged: {}", results.converged);
        println!("    Convergence time: {:.3}", results.convergence_time);
        println!("    Runtime: {:.2?}", runtime);

        // Schedule-specific analysis
        match schedule_name {
            "Piecewise Linear" => {
                println!("    Strategy: Gentle start, plateau, aggressive finish");
                println!("    Use case: When precise control near threshold is needed");
            }
            "Oscillating" => {
                println!("    Strategy: Continuous small oscillations around linear trend");
                println!("    Use case: Escaping local minima through controlled perturbations");
            }
            "Threshold Approach" => {
                println!("    Strategy: Exponential approach to oscillation threshold");
                println!("    Use case: Maximizing time near critical point");
            }
            _ => {}
        }

        // Performance analysis
        println!(
            "    Power efficiency: {:.1}%",
            results.performance_metrics.power_efficiency * 100.0
        );
        println!(
            "    Average optical power: {:.6}",
            results.optical_stats.average_power
        );

        // Energy progression analysis
        if results.energy_history.len() > 10 {
            let first_half = &results.energy_history[0..results.energy_history.len() / 2];
            let second_half = &results.energy_history[results.energy_history.len() / 2..];

            let first_half_improvement = first_half[0] - first_half[first_half.len() - 1];
            let second_half_improvement = second_half[0] - second_half[second_half.len() - 1];

            println!("    First half improvement: {:.6}", first_half_improvement);
            println!(
                "    Second half improvement: {:.6}",
                second_half_improvement
            );

            if first_half_improvement > second_half_improvement {
                println!("    Pattern: Early optimization dominance");
            } else {
                println!("    Pattern: Late-stage optimization benefits");
            }
        }
    }

    println!("\\n  Custom Schedule Design Principles:");
    println!("    1. Start below oscillation threshold for initialization");
    println!("    2. Gradually approach threshold to avoid premature locking");
    println!("    3. Consider plateaus for system equilibration");
    println!("    4. Use final acceleration for energy minimization");
    println!("    5. Add controlled perturbations to escape local minima");

    println!("\\n  Implementation Guidelines:");
    println!("    - Custom schedules use normalized time t in [0, 1]");
    println!("    - Return power values appropriate for your optical system");
    println!("    - Consider physical constraints (maximum pump power, etc.)");
    println!("    - Test convergence behavior with different schedule shapes");

    Ok(())
}
