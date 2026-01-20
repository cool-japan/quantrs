#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Example demonstrating population annealing
//!
//! This example shows how to:
//! 1. Configure population annealing parameters
//! 2. Solve problems using population annealing
//! 3. Analyze population statistics and convergence
//! 4. Compare with standard simulated annealing

use quantrs2_anneal::{
    ising::IsingModel,
    population_annealing::{MpiConfig, PopulationAnnealingConfig, PopulationAnnealingSimulator},
    simulator::{AnnealingParams, QuantumAnnealingSimulator, TemperatureSchedule},
};
use scirs2_core::random::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Population Annealing Demo ===\n");

    // Example 1: Basic population annealing
    println!("Example 1: Basic Population Annealing");
    basic_population_annealing()?;

    // Example 2: Large population with resampling
    println!("\nExample 2: Large Population with Adaptive Resampling");
    large_population_example()?;

    // Example 3: Comparison with standard annealing
    println!("\nExample 3: Comparison with Standard Annealing");
    comparison_example()?;

    // Example 4: MPI configuration (conceptual)
    println!("\nExample 4: MPI Configuration (Conceptual)");
    mpi_configuration_example()?;

    // Example 5: Complex landscape analysis
    println!("\nExample 5: Complex Landscape Analysis");
    complex_landscape_example()?;

    Ok(())
}

fn basic_population_annealing() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple frustrated system
    let mut model = IsingModel::new(6);

    // Triangle with antiferromagnetic couplings (frustrated)
    model.set_coupling(0, 1, 1.0)?;
    model.set_coupling(1, 2, 1.0)?;
    model.set_coupling(2, 0, 1.0)?;

    // Another triangle
    model.set_coupling(3, 4, 1.0)?;
    model.set_coupling(4, 5, 1.0)?;
    model.set_coupling(5, 3, 1.0)?;

    // Connect the triangles
    model.set_coupling(0, 3, -0.5)?;

    // Configure population annealing
    let config = PopulationAnnealingConfig {
        population_size: 200,
        initial_temperature: 5.0,
        final_temperature: 0.01,
        num_temperature_steps: 50,
        sweeps_per_step: 50,
        resampling_frequency: 5,
        ess_threshold: 0.6,
        seed: Some(42),
        ..Default::default()
    };

    // Run population annealing
    let start = Instant::now();
    let mut simulator = PopulationAnnealingSimulator::new(config)?;
    let result = simulator.solve(&model)?;
    let runtime = start.elapsed();

    println!("Results:");
    println!("  Best energy: {:.4}", result.best_energy);
    println!("  Best configuration: {:?}", result.best_configuration);
    println!("  Runtime: {runtime:.2?}");
    println!("  Number of resamplings: {}", result.num_resamplings);
    println!(
        "  Final ESS: {:.2}",
        result.ess_history.last().unwrap_or(&0.0)
    );

    // Print energy evolution
    println!("\nEnergy evolution:");
    for (i, stats) in result.energy_history.iter().enumerate() {
        if i % 10 == 0 {
            println!(
                "  Step {}: T={:.3}, E_min={:.3}, E_mean={:.3}, ESS={:.1}",
                i,
                stats.temperature,
                stats.min_energy,
                stats.mean_energy,
                stats.effective_sample_size
            );
        }
    }

    Ok(())
}

fn large_population_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a larger random problem
    let model = create_random_spin_glass(12, 0.5)?;

    // Configure with large population
    let config = PopulationAnnealingConfig {
        population_size: 1000,
        initial_temperature: 10.0,
        final_temperature: 0.001,
        num_temperature_steps: 100,
        sweeps_per_step: 20,
        resampling_frequency: 3,
        ess_threshold: 0.5,
        temperature_schedule: TemperatureSchedule::Exponential(4.0),
        seed: Some(123),
        ..Default::default()
    };

    let start = Instant::now();
    let mut simulator = PopulationAnnealingSimulator::new(config)?;
    let result = simulator.solve(&model)?;
    let runtime = start.elapsed();

    println!("Large population results:");
    println!("  Problem size: {} spins", model.num_qubits);
    println!("  Population size: {}", result.final_population.len());
    println!("  Best energy: {:.4}", result.best_energy);
    println!("  Runtime: {runtime:.2?}");
    println!("  Resamplings: {}", result.num_resamplings);

    // Analyze final population diversity
    let final_energies: Vec<f64> = result.final_population.iter().map(|m| m.energy).collect();
    let min_energy = final_energies.iter().copied().fold(f64::INFINITY, f64::min);
    let max_energy = final_energies
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_energy = final_energies.iter().sum::<f64>() / final_energies.len() as f64;

    println!("  Final population diversity:");
    println!("    Energy range: [{min_energy:.4}, {max_energy:.4}]");
    println!("    Mean energy: {mean_energy:.4}");
    println!(
        "    Ground state copies: {}",
        final_energies
            .iter()
            .filter(|&&e| (e - min_energy).abs() < 1e-6)
            .count()
    );

    Ok(())
}

fn comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a challenging MaxCut problem
    let model = create_maxcut_problem(10)?;

    // Population annealing configuration
    let pop_config = PopulationAnnealingConfig {
        population_size: 500,
        initial_temperature: 8.0,
        final_temperature: 0.01,
        num_temperature_steps: 80,
        sweeps_per_step: 30,
        resampling_frequency: 4,
        seed: Some(42),
        ..Default::default()
    };

    // Standard annealing configuration
    let std_config = AnnealingParams {
        initial_temperature: 8.0,
        num_sweeps: 24000, // Equivalent computational cost
        num_repetitions: 20,
        seed: Some(42),
        ..Default::default()
    };

    // Run population annealing
    let start = Instant::now();
    let mut pop_simulator = PopulationAnnealingSimulator::new(pop_config)?;
    let pop_result = pop_simulator.solve(&model)?;
    let pop_time = start.elapsed();

    // Run standard annealing
    let start = Instant::now();
    let mut std_simulator = QuantumAnnealingSimulator::new(std_config)?;
    let std_result = std_simulator.solve(&model)?;
    let std_time = start.elapsed();

    println!("Comparison results:");
    println!("Population Annealing:");
    println!("  Best energy: {:.4}", pop_result.best_energy);
    println!("  Runtime: {pop_time:.2?}");
    println!(
        "  Final ESS: {:.2}",
        pop_result.ess_history.last().unwrap_or(&0.0)
    );

    println!("Standard Annealing:");
    println!("  Best energy: {:.4}", std_result.best_energy);
    println!("  Runtime: {std_time:.2?}");
    println!("  Repetitions: {}", std_result.repetitions);

    let energy_improvement = std_result.best_energy - pop_result.best_energy;
    println!("Population annealing improvement: {energy_improvement:.4}");

    Ok(())
}

fn mpi_configuration_example() -> Result<(), Box<dyn std::error::Error>> {
    // Example of how MPI configuration would work
    let mpi_config = MpiConfig {
        num_processes: 4,
        rank: 0,
        load_balancing: true,
        communication_frequency: 10,
    };

    let config = PopulationAnnealingConfig {
        population_size: 1000,
        mpi_config: Some(mpi_config),
        ..Default::default()
    };

    println!("MPI Configuration:");
    println!(
        "  Number of processes: {}",
        config.mpi_config.as_ref().unwrap().num_processes
    );
    println!(
        "  Load balancing enabled: {}",
        config.mpi_config.as_ref().unwrap().load_balancing
    );
    println!(
        "  Communication frequency: {} steps",
        config.mpi_config.as_ref().unwrap().communication_frequency
    );
    println!(
        "  Population per process: {}",
        config.population_size / config.mpi_config.as_ref().unwrap().num_processes
    );

    println!("\nNote: Actual MPI implementation would require:");
    println!("  - MPI runtime environment");
    println!("  - Process synchronization");
    println!("  - Inter-process population exchange");
    println!("  - Load balancing algorithms");

    Ok(())
}

fn complex_landscape_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a complex energy landscape with multiple local minima
    let model = create_complex_landscape(8)?;

    let config = PopulationAnnealingConfig {
        population_size: 300,
        initial_temperature: 15.0,
        final_temperature: 0.005,
        num_temperature_steps: 120,
        sweeps_per_step: 40,
        resampling_frequency: 6,
        ess_threshold: 0.4,
        seed: Some(789),
        ..Default::default()
    };

    let mut simulator = PopulationAnnealingSimulator::new(config)?;
    let result = simulator.solve(&model)?;

    println!("Complex landscape results:");
    println!("  Best energy found: {:.4}", result.best_energy);
    println!("  Total resamplings: {}", result.num_resamplings);

    // Analyze ESS evolution
    let min_ess = result
        .ess_history
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_ess = result
        .ess_history
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_ess = result.ess_history.iter().sum::<f64>() / result.ess_history.len() as f64;

    println!("  ESS statistics:");
    println!("    Range: [{min_ess:.1}, {max_ess:.1}]");
    println!("    Mean: {mean_ess:.1}");

    // Check for temperature steps with low ESS
    let low_ess_count = result
        .ess_history
        .iter()
        .filter(|&&ess| ess < 150.0)
        .count();
    println!(
        "    Steps with ESS < 150: {}/{}",
        low_ess_count,
        result.ess_history.len()
    );

    Ok(())
}

/// Create a random spin glass model
fn create_random_spin_glass(
    n: usize,
    density: f64,
) -> Result<IsingModel, Box<dyn std::error::Error>> {
    let mut model = IsingModel::new(n);

    for i in 0..n {
        for j in (i + 1)..n {
            if thread_rng().gen::<f64>() < density {
                let coupling = if thread_rng().gen::<bool>() {
                    1.0
                } else {
                    -1.0
                };
                model.set_coupling(i, j, coupling)?;
            }
        }
    }

    Ok(model)
}

/// Create a `MaxCut` problem instance
fn create_maxcut_problem(n: usize) -> Result<IsingModel, Box<dyn std::error::Error>> {
    let mut model = IsingModel::new(n);

    // Create a random graph with negative couplings
    for i in 0..n {
        for j in (i + 1)..n {
            if thread_rng().gen::<f64>() < 0.4 {
                model.set_coupling(i, j, -1.0)?;
            }
        }
    }

    Ok(model)
}

/// Create a complex energy landscape with multiple local minima
fn create_complex_landscape(n: usize) -> Result<IsingModel, Box<dyn std::error::Error>> {
    let mut model = IsingModel::new(n);

    // Add random fields
    for i in 0..n {
        let field = (thread_rng().gen::<f64>() - 0.5) * 2.0;
        model.set_bias(i, field)?;
    }

    // Add competing interactions
    for i in 0..n {
        for j in (i + 1)..n {
            if thread_rng().gen::<f64>() < 0.6 {
                let coupling = (thread_rng().gen::<f64>() - 0.5) * 4.0;
                model.set_coupling(i, j, coupling)?;
            }
        }
    }

    // Add long-range interactions
    for i in 0..n {
        for j in (i + 3)..n {
            if thread_rng().gen::<f64>() < 0.2 {
                let coupling = (thread_rng().gen::<f64>() - 0.5) * 1.0;
                model.set_coupling(i, j, coupling)?;
            }
        }
    }

    Ok(model)
}
