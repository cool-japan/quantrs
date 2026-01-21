#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop,
    clippy::useless_vec,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
//! Quantum Boltzmann Machine Example
//!
//! This example demonstrates quantum Boltzmann machines for unsupervised learning,
//! including RBMs and deep Boltzmann machines.

use quantrs2_ml::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::prelude::*;

fn main() -> Result<()> {
    println!("=== Quantum Boltzmann Machine Demo ===\n");

    // Step 1: Basic Boltzmann machine
    println!("1. Basic Quantum Boltzmann Machine...");
    basic_qbm_demo()?;

    // Step 2: Restricted Boltzmann Machine
    println!("\n2. Quantum Restricted Boltzmann Machine (RBM)...");
    rbm_demo()?;

    // Step 3: Deep Boltzmann Machine
    println!("\n3. Deep Boltzmann Machine...");
    deep_boltzmann_demo()?;

    // Step 4: Energy landscape visualization
    println!("\n4. Energy Landscape Analysis...");
    energy_landscape_demo()?;

    // Step 5: Pattern completion
    println!("\n5. Pattern Completion Demo...");
    pattern_completion_demo()?;

    println!("\n=== Boltzmann Machine Demo Complete ===");

    Ok(())
}

/// Basic Quantum Boltzmann Machine demonstration
fn basic_qbm_demo() -> Result<()> {
    // Create a small QBM
    let mut qbm = QuantumBoltzmannMachine::new(
        4,    // visible units
        2,    // hidden units
        1.0,  // temperature
        0.01, // learning rate
    )?;

    println!("   Created QBM with 4 visible and 2 hidden units");

    // Generate synthetic binary data
    let data = generate_binary_patterns(100, 4);

    // Train the QBM
    println!("   Training on binary patterns...");
    let losses = qbm.train(&data, 50, 10)?;

    println!("   Training complete:");
    println!("   - Initial loss: {:.4}", losses[0]);
    println!("   - Final loss: {:.4}", losses.last().unwrap());

    // Sample from trained model
    let samples = qbm.sample(5)?;
    println!("\n   Generated samples:");
    for (i, sample) in samples.outer_iter().enumerate() {
        print!("   Sample {}: [", i + 1);
        for val in sample {
            print!("{val:.0} ");
        }
        println!("]");
    }

    Ok(())
}

/// RBM demonstration with persistent contrastive divergence
fn rbm_demo() -> Result<()> {
    // Create RBM with annealing
    let annealing = AnnealingSchedule::new(2.0, 0.5, 100);

    let mut rbm = QuantumRBM::new(
        6,    // visible units
        3,    // hidden units
        2.0,  // initial temperature
        0.01, // learning rate
    )?
    .with_annealing(annealing);

    println!("   Created Quantum RBM with annealing schedule");

    // Generate correlated binary data
    let data = generate_correlated_data(200, 6);

    // Train with PCD
    println!("   Training with Persistent Contrastive Divergence...");
    let losses = rbm.train_pcd(
        &data, 100, // epochs
        20,  // batch size
        50,  // persistent chains
    )?;

    // Analyze training
    let improvement = (losses[0] - losses.last().unwrap()) / losses[0] * 100.0;
    println!("   Training statistics:");
    println!("   - Loss reduction: {improvement:.1}%");
    println!("   - Final temperature: 0.5");

    // Test reconstruction
    let test_data = data.slice(s![0..5, ..]).to_owned();
    let reconstructed = rbm.qbm().reconstruct(&test_data)?;

    println!("\n   Reconstruction quality:");
    for i in 0..3 {
        print!("   Original:      [");
        for val in test_data.row(i) {
            print!("{val:.0} ");
        }
        print!("]  →  Reconstructed: [");
        for val in reconstructed.row(i) {
            print!("{val:.0} ");
        }
        println!("]");
    }

    Ok(())
}

/// Deep Boltzmann Machine demonstration
fn deep_boltzmann_demo() -> Result<()> {
    // Create a 3-layer DBM
    let layer_sizes = vec![8, 4, 2];
    let mut dbm = DeepBoltzmannMachine::new(
        layer_sizes.clone(),
        1.0,  // temperature
        0.01, // learning rate
    )?;

    println!("   Created Deep Boltzmann Machine:");
    println!("   - Architecture: {layer_sizes:?}");
    println!("   - Total layers: {}", dbm.rbms().len());

    // Generate hierarchical data
    let data = generate_hierarchical_data(300, 8);

    // Layer-wise pretraining
    println!("\n   Performing layer-wise pretraining...");
    dbm.pretrain(
        &data, 50, // epochs per layer
        30, // batch size
    )?;

    println!("\n   Pretraining complete!");
    println!("   Each layer learned increasingly abstract features");

    Ok(())
}

/// Energy landscape visualization
fn energy_landscape_demo() -> Result<()> {
    // Create small QBM for visualization
    let qbm = QuantumBoltzmannMachine::new(
        2,    // visible units (for 2D visualization)
        1,    // hidden unit
        0.5,  // temperature
        0.01, // learning rate
    )?;

    println!("   Analyzing energy landscape of 2-unit system");

    // Compute energy for all 4 possible states
    let states = [
        Array1::from_vec(vec![0.0, 0.0]),
        Array1::from_vec(vec![0.0, 1.0]),
        Array1::from_vec(vec![1.0, 0.0]),
        Array1::from_vec(vec![1.0, 1.0]),
    ];

    println!("\n   State energies:");
    for (i, state) in states.iter().enumerate() {
        let energy = qbm.energy(state);
        let prob = (-energy / qbm.temperature()).exp();
        println!(
            "   State [{:.0}, {:.0}]: E = {:.3}, P ∝ {:.3}",
            state[0], state[1], energy, prob
        );
    }

    // Show coupling matrix
    println!("\n   Coupling matrix:");
    for i in 0..3 {
        print!("   [");
        for j in 0..3 {
            print!("{:6.3} ", qbm.couplings()[[i, j]]);
        }
        println!("]");
    }

    Ok(())
}

/// Pattern completion demonstration
fn pattern_completion_demo() -> Result<()> {
    // Create RBM
    let mut rbm = QuantumRBM::new(
        8,    // visible units
        4,    // hidden units
        1.0,  // temperature
        0.02, // learning rate
    )?;

    // Train on specific patterns
    let patterns = create_letter_patterns();
    println!("   Training on letter-like patterns...");

    rbm.train_pcd(&patterns, 100, 10, 20)?;

    // Test pattern completion
    println!("\n   Pattern completion test:");

    // Create corrupted patterns
    let mut corrupted = patterns.row(0).to_owned();
    corrupted[3] = 1.0 - corrupted[3]; // Flip one bit
    corrupted[5] = 1.0 - corrupted[5]; // Flip another

    print!("   Corrupted:  [");
    for val in &corrupted {
        print!("{val:.0} ");
    }
    println!("]");

    // Complete pattern
    let completed = complete_pattern(&rbm, &corrupted)?;

    print!("   Completed:  [");
    for val in &completed {
        print!("{val:.0} ");
    }
    println!("]");

    print!("   Original:   [");
    for val in patterns.row(0) {
        print!("{val:.0} ");
    }
    println!("]");

    let accuracy = patterns
        .row(0)
        .iter()
        .zip(completed.iter())
        .filter(|(&a, &b)| (a - b).abs() < 0.5)
        .count() as f64
        / 8.0;

    println!("   Reconstruction accuracy: {:.1}%", accuracy * 100.0);

    Ok(())
}

/// Generate binary patterns
fn generate_binary_patterns(n_samples: usize, n_features: usize) -> Array2<f64> {
    Array2::from_shape_fn((n_samples, n_features), |(_, _)| {
        if thread_rng().gen::<f64>() > 0.5 {
            1.0
        } else {
            0.0
        }
    })
}

/// Generate correlated binary data
fn generate_correlated_data(n_samples: usize, n_features: usize) -> Array2<f64> {
    let mut data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        // Generate correlated features
        let base = if thread_rng().gen::<f64>() > 0.5 {
            1.0
        } else {
            0.0
        };

        for j in 0..n_features {
            if j % 2 == 0 {
                data[[i, j]] = base;
            } else {
                // Correlate with previous feature
                data[[i, j]] = if thread_rng().gen::<f64>() > 0.2 {
                    base
                } else {
                    1.0 - base
                };
            }
        }
    }

    data
}

/// Generate hierarchical data
fn generate_hierarchical_data(n_samples: usize, n_features: usize) -> Array2<f64> {
    let mut data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        // Choose high-level pattern
        let pattern_type = i % 3;

        match pattern_type {
            0 => {
                // Pattern A: alternating
                for j in 0..n_features {
                    data[[i, j]] = (j % 2) as f64;
                }
            }
            1 => {
                // Pattern B: blocks
                for j in 0..n_features {
                    data[[i, j]] = ((j / 2) % 2) as f64;
                }
            }
            _ => {
                // Pattern C: random with structure
                let shift = (thread_rng().gen::<f64>() * 4.0) as usize;
                for j in 0..n_features {
                    data[[i, j]] = if (j + shift) % 3 == 0 { 1.0 } else { 0.0 };
                }
            }
        }

        // Add noise
        for j in 0..n_features {
            if thread_rng().gen::<f64>() < 0.1 {
                data[[i, j]] = 1.0 - data[[i, j]];
            }
        }
    }

    data
}

/// Create letter-like patterns
fn create_letter_patterns() -> Array2<f64> {
    // Simple 8-bit patterns resembling letters
    Array2::from_shape_vec(
        (4, 8),
        vec![
            // Pattern 'L'
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, // Pattern 'T'
            1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // Pattern 'I'
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // Pattern 'H'
            1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0,
        ],
    )
    .unwrap()
}

/// Complete a partial pattern
fn complete_pattern(rbm: &QuantumRBM, partial: &Array1<f64>) -> Result<Array1<f64>> {
    // Use Gibbs sampling to complete pattern
    let mut current = partial.clone();

    for _ in 0..10 {
        let hidden = rbm.qbm().sample_hidden_given_visible(&current.view())?;
        current = rbm.qbm().sample_visible_given_hidden(&hidden)?;
    }

    Ok(current)
}
