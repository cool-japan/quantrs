#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Example demonstrating Photonic Annealing Systems
//!
//! This example shows how to:
//! 1. Set up different photonic architectures
//! 2. Configure photonic states and pump schedules
//! 3. Use various measurement strategies
//! 4. Analyze photonic annealing results
//! 5. Compare different photonic configurations

use quantrs2_anneal::{
    ising::IsingModel,
    photonic_annealing::{
        create_low_noise_config, create_measurement_based_config, create_realistic_config,
        create_temporal_multiplexing_config, ConnectivityType, InitialStateType,
        MeasurementStrategy, PhotonicAnnealer, PhotonicAnnealingConfig, PhotonicArchitecture,
        PumpPowerSchedule,
    },
};
use scirs2_core::random::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Photonic Annealing Systems Demo ===\n");

    // Example 1: Basic Photonic Annealing
    println!("Example 1: Basic Photonic Annealing");
    basic_photonic_annealing_example()?;

    // Example 2: Temporal Multiplexing Architecture
    println!("\nExample 2: Temporal Multiplexing Architecture");
    temporal_multiplexing_example()?;

    // Example 3: Measurement Strategy Comparison
    println!("\nExample 3: Measurement Strategy Comparison");
    measurement_strategy_comparison()?;

    // Example 4: Squeezed State Optimization
    println!("\nExample 4: Squeezed State Optimization");
    squeezed_state_example()?;

    // Example 5: Measurement-Based Architecture
    println!("\nExample 5: Measurement-Based Architecture");
    measurement_based_example()?;

    // Example 6: Realistic Experimental Setup
    println!("\nExample 6: Realistic Experimental Setup");
    realistic_experimental_example()?;

    Ok(())
}

fn basic_photonic_annealing_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple Ising problem
    let mut ising = IsingModel::new(6);

    // Add biases and couplings for a simple optimization problem
    for i in 0..6 {
        ising.set_bias(i, -0.5)?;
    }

    // Create a ring of couplings
    for i in 0..6 {
        let j = (i + 1) % 6;
        ising.set_coupling(i, j, -1.0)?;
    }

    // Add frustration
    ising.set_coupling(0, 3, 0.5)?;

    // Configure photonic annealing
    let config = PhotonicAnnealingConfig {
        architecture: PhotonicArchitecture::SpatialMultiplexing {
            num_modes: 6,
            connectivity: ConnectivityType::Ring,
        },
        initial_state: InitialStateType::Vacuum,
        pump_schedule: PumpPowerSchedule::Linear {
            initial_power: 0.0,
            final_power: 2.0,
        },
        measurement_strategy: MeasurementStrategy::Homodyne {
            local_oscillator_phase: 0.0,
        },
        num_shots: 1000,
        evolution_time: 0.1,
        time_steps: 100,
        loss_rate: 0.01,
        kerr_strength: 0.1,
        temperature: 0.0,
        quantum_noise: true,
        seed: Some(42),
    };

    let start = Instant::now();
    let mut annealer = PhotonicAnnealer::new(config)?;
    let results = annealer.anneal(&ising)?;
    let runtime = start.elapsed();

    println!("  Architecture: Spatial Multiplexing (Ring)");
    println!("  Number of modes: 6");
    println!("  Evolution time: 0.1 s");
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Best solution: {:?}", results.best_solution);
    println!("  Runtime: {runtime:.2?}");

    // Analyze energy distribution
    println!("\n  Energy distribution:");
    let mut energies: Vec<_> = results.energy_distribution.iter().collect();
    energies.sort_by_key(|(e, _)| *e);

    for (energy, count) in energies.iter().take(5) {
        let actual_energy = **energy as f64 / 1000.0;
        let probability = **count as f64 / results.measurement_outcomes.len() as f64;
        println!("    E = {:.3}: {:.1}%", actual_energy, probability * 100.0);
    }

    // Display metrics
    println!("\n  Performance metrics:");
    println!(
        "    Success probability: {:.1}%",
        results.metrics.success_probability * 100.0
    );
    println!(
        "    Average quality: {:.3}",
        results.metrics.average_quality
    );
    println!(
        "    Quantum advantage estimate: {:.2}x",
        results.metrics.quantum_advantage
    );
    println!(
        "    Photon loss: {:.1}%",
        results.metrics.photon_loss * 100.0
    );
    println!(
        "    Measurement efficiency: {:.1}%",
        results.metrics.measurement_efficiency * 100.0
    );

    Ok(())
}

fn temporal_multiplexing_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a larger Ising problem
    let mut ising = IsingModel::new(20);

    // Random problem instance
    for i in 0..20 {
        ising.set_bias(i, (i as f64 - 10.0) * 0.1)?;
    }

    for i in 0..20 {
        for j in (i + 1)..20 {
            if (i + j) % 3 == 0 {
                ising.set_coupling(i, j, -0.5)?;
            }
        }
    }

    // Temporal multiplexing configuration
    let config = create_temporal_multiplexing_config(20, 10e9); // 10 GHz rep rate
    let mut custom_config = config;
    custom_config.pump_schedule = PumpPowerSchedule::Exponential {
        initial_power: 2.0,
        time_constant: 0.5,
    };
    custom_config.evolution_time = 0.2;
    custom_config.num_shots = 500;

    let start = Instant::now();
    let mut annealer = PhotonicAnnealer::new(custom_config)?;
    let results = annealer.anneal(&ising)?;
    let runtime = start.elapsed();

    println!("  Architecture: Temporal Multiplexing");
    println!("  Time bins: 20");
    println!("  Repetition rate: 10 GHz");
    println!("  Pump schedule: Exponential decay");
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Runtime: {runtime:.2?}");

    // Analyze evolution history
    println!("\n  Evolution analysis:");
    let history = &results.evolution_history;

    if history.times.len() >= 3 {
        let idx_start = 0;
        let idx_mid = history.times.len() / 2;
        let idx_end = history.times.len() - 1;

        println!("    Time    | Mean Photons | Energy Expect | Purity");
        println!("    --------|--------------|---------------|-------");

        for &idx in &[idx_start, idx_mid, idx_end] {
            let mean_photons: f64 = history.photon_numbers[idx].iter().sum::<f64>()
                / history.photon_numbers[idx].len() as f64;
            println!(
                "    {:.4} s | {:.3}        | {:.6}     | {:.3}",
                history.times[idx],
                mean_photons,
                history.energy_expectation[idx],
                history.purity[idx]
            );
        }
    }

    Ok(())
}

fn measurement_strategy_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Create test problem
    let mut ising = IsingModel::new(8);
    for i in 0..8 {
        ising.set_bias(i, if i % 2 == 0 { 1.0 } else { -1.0 })?;
        if i < 7 {
            ising.set_coupling(i, i + 1, -0.8)?;
        }
    }

    let strategies = vec![
        (
            "Homodyne (0°)",
            MeasurementStrategy::Homodyne {
                local_oscillator_phase: 0.0,
            },
        ),
        (
            "Homodyne (45°)",
            MeasurementStrategy::Homodyne {
                local_oscillator_phase: std::f64::consts::PI / 4.0,
            },
        ),
        ("Heterodyne", MeasurementStrategy::Heterodyne),
        (
            "Photon Counting",
            MeasurementStrategy::PhotonCounting { threshold: 0.5 },
        ),
        ("Parity", MeasurementStrategy::Parity),
        (
            "Adaptive",
            MeasurementStrategy::Adaptive {
                feedback_strength: 0.3,
            },
        ),
    ];

    println!("  Comparing measurement strategies:");
    println!("  Strategy         | Best Energy | Success % | Efficiency");
    println!("  -----------------|-------------|-----------|------------");

    for (name, strategy) in strategies {
        let config = PhotonicAnnealingConfig {
            architecture: PhotonicArchitecture::SpatialMultiplexing {
                num_modes: 8,
                connectivity: ConnectivityType::FullyConnected,
            },
            initial_state: InitialStateType::Coherent { alpha: 0.5 },
            measurement_strategy: strategy,
            num_shots: 500,
            evolution_time: 0.05,
            time_steps: 50,
            ..Default::default()
        };

        let mut annealer = PhotonicAnnealer::new(config)?;
        let results = annealer.anneal(&ising)?;

        println!(
            "  {:16} | {:.6}  | {:.1}%     | {:.1}%",
            name,
            results.best_energy,
            results.metrics.success_probability * 100.0,
            results.metrics.measurement_efficiency * 100.0
        );
    }

    Ok(())
}

fn squeezed_state_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a problem that benefits from squeezing
    let mut ising = IsingModel::new(10);

    // Strong correlations benefit from squeezing
    for i in 0..5 {
        ising.set_coupling(i, i + 5, -2.0)?;
    }

    for i in 0..10 {
        ising.set_bias(i, 0.1)?;
    }

    println!("  Comparing initial states:");
    println!("  Initial State    | Best Energy | Mean Photons | Success %");
    println!("  -----------------|-------------|--------------|----------");

    let states = vec![
        ("Vacuum", InitialStateType::Vacuum),
        ("Coherent (α=1)", InitialStateType::Coherent { alpha: 1.0 }),
        (
            "Squeezed (r=0.5)",
            InitialStateType::SqueezedVacuum { squeezing: 0.5 },
        ),
        (
            "Squeezed (r=1.0)",
            InitialStateType::SqueezedVacuum { squeezing: 1.0 },
        ),
        (
            "Thermal (n=0.5)",
            InitialStateType::Thermal { mean_photons: 0.5 },
        ),
    ];

    for (name, initial_state) in states {
        let config = PhotonicAnnealingConfig {
            architecture: PhotonicArchitecture::SpatialMultiplexing {
                num_modes: 10,
                connectivity: ConnectivityType::FullyConnected,
            },
            initial_state,
            pump_schedule: PumpPowerSchedule::Linear {
                initial_power: 0.0,
                final_power: 1.5,
            },
            num_shots: 500,
            evolution_time: 0.1,
            ..Default::default()
        };

        let mut annealer = PhotonicAnnealer::new(config)?;
        let results = annealer.anneal(&ising)?;

        let final_photons = results.final_state.mean_photon_number();

        println!(
            "  {:16} | {:.6}  | {:.3}        | {:.1}%",
            name,
            results.best_energy,
            final_photons,
            results.metrics.success_probability * 100.0
        );
    }

    Ok(())
}

fn measurement_based_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a problem suitable for measurement-based computation
    let mut ising = IsingModel::new(12);

    // Grid-like connectivity
    for i in 0..12 {
        let row = i / 4;
        let col = i % 4;

        // Horizontal connections
        if col < 3 {
            ising.set_coupling(i, i + 1, -1.0)?;
        }

        // Vertical connections
        if row < 2 {
            ising.set_coupling(i, i + 4, -1.0)?;
        }
    }

    // Add some disorder
    for i in 0..12 {
        ising.set_bias(i, (i as f64 - 6.0) * 0.2)?;
    }

    let config = create_measurement_based_config(12);

    let start = Instant::now();
    let mut annealer = PhotonicAnnealer::new(config)?;
    let results = annealer.anneal(&ising)?;
    let runtime = start.elapsed();

    println!("  Architecture: Measurement-Based");
    println!("  Resource state size: 12");
    println!("  Measurement type: Adaptive");
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Runtime: {runtime:.2?}");

    // Analyze measurement outcomes
    println!("\n  Measurement outcome analysis:");
    let mut fidelities: Vec<f64> = results
        .measurement_outcomes
        .iter()
        .map(|o| o.fidelity)
        .collect();
    fidelities.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let avg_fidelity = fidelities.iter().sum::<f64>() / fidelities.len() as f64;
    let max_fidelity = fidelities[0];
    let min_fidelity = fidelities[fidelities.len() - 1];

    println!("    Average fidelity: {avg_fidelity:.3}");
    println!("    Max fidelity: {max_fidelity:.3}");
    println!("    Min fidelity: {min_fidelity:.3}");

    // Solution quality distribution
    let mut solution_qualities: Vec<f64> = results
        .measurement_outcomes
        .iter()
        .map(|o| {
            let worst_energy = results
                .measurement_outcomes
                .iter()
                .map(|o2| o2.energy)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            1.0 - (o.energy - results.best_energy) / (worst_energy - results.best_energy)
        })
        .collect();

    solution_qualities.sort_by(|a, b| b.partial_cmp(a).unwrap());

    println!("\n  Solution quality distribution:");
    let bins = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5];
    for i in 0..bins.len() - 1 {
        let count = solution_qualities
            .iter()
            .filter(|&&q| q <= bins[i] && q > bins[i + 1])
            .count();
        let percentage = count as f64 / solution_qualities.len() as f64 * 100.0;
        println!(
            "    Quality {:.0}%-{:.0}%: {:.1}%",
            bins[i + 1] * 100.0,
            bins[i] * 100.0,
            percentage
        );
    }

    Ok(())
}

fn realistic_experimental_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a realistic problem
    let mut ising = IsingModel::new(16);

    // Random spin glass instance
    let mut rng = scirs2_core::random::ChaCha8Rng::seed_from_u64(123);

    for i in 0..16 {
        ising.set_bias(i, rng.gen_range(-1.0..1.0))?;
    }

    for i in 0..16 {
        for j in (i + 1)..16 {
            if rng.gen::<f64>() < 0.3 {
                // 30% connectivity
                ising.set_coupling(i, j, rng.gen_range(-1.0..1.0))?;
            }
        }
    }

    // Realistic experimental configuration
    let config = create_realistic_config();

    println!("  Realistic experimental configuration:");
    println!("    Loss rate: {} /s", config.loss_rate);
    println!("    Temperature: {} K", config.temperature);
    println!("    Kerr strength: {}", config.kerr_strength);
    println!("    Quantum noise: {}", config.quantum_noise);
    println!("    Number of shots: {}", config.num_shots);

    let start = Instant::now();
    let mut annealer = PhotonicAnnealer::new(config)?;
    let results = annealer.anneal(&ising)?;
    let runtime = start.elapsed();

    println!("\n  Results:");
    println!("    Best energy: {:.6}", results.best_energy);
    println!("    Runtime: {runtime:.2?}");
    println!(
        "    Effective temperature: {:.2e} K",
        results.metrics.effective_temperature
    );

    // Compare with ideal (low-noise) configuration
    println!("\n  Comparison with ideal configuration:");

    let ideal_config = create_low_noise_config();
    let mut ideal_annealer = PhotonicAnnealer::new(ideal_config)?;
    let ideal_results = ideal_annealer.anneal(&ising)?;

    println!("    Ideal best energy: {:.6}", ideal_results.best_energy);
    println!(
        "    Energy gap: {:.6}",
        results.best_energy - ideal_results.best_energy
    );
    println!(
        "    Success probability ratio: {:.2}x",
        ideal_results.metrics.success_probability / results.metrics.success_probability
    );

    // Analyze impact of noise
    println!("\n  Noise impact analysis:");
    println!(
        "    Photon loss: {:.1}%",
        results.metrics.photon_loss * 100.0
    );
    println!(
        "    Final state purity: {:.3}",
        results.final_state.purity()
    );

    // Evolution history comparison
    if !results.evolution_history.purity.is_empty() {
        let initial_purity = results.evolution_history.purity[0];
        let final_purity = results.evolution_history.purity.last().unwrap();
        println!(
            "    Purity degradation: {:.1}%",
            (1.0 - final_purity / initial_purity) * 100.0
        );
    }

    Ok(())
}
