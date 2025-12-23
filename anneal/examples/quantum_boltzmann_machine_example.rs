//! Example demonstrating Quantum Restricted Boltzmann Machines
//!
//! This example shows how to:
//! 1. Create and configure different types of RBMs
//! 2. Train RBMs on datasets using quantum annealing
//! 3. Generate samples from learned distributions
//! 4. Perform inference and reconstruction
//! 5. Compare quantum vs classical sampling approaches
//! 6. Analyze training statistics and convergence

use quantrs2_anneal::{
    quantum_boltzmann_machine::{
        create_binary_rbm, create_gaussian_bernoulli_rbm, LayerConfig, QbmTrainingConfig,
        QuantumRestrictedBoltzmannMachine, TrainingSample, UnitType,
    },
    simulator::AnnealingParams,
};
use scirs2_core::random::prelude::*;
use scirs2_core::random::Rng;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Quantum Boltzmann Machine Demo ===\n");

    // Example 1: Binary pattern learning
    println!("Example 1: Binary Pattern Learning");
    binary_pattern_learning_example()?;

    // Example 2: Handwritten digit features (simplified)
    println!("\nExample 2: Handwritten Digit Feature Learning");
    digit_feature_learning_example()?;

    // Example 3: Quantum vs Classical sampling comparison
    println!("\nExample 3: Quantum vs Classical Sampling Comparison");
    quantum_vs_classical_comparison()?;

    // Example 4: Gaussian-Bernoulli RBM for continuous data
    println!("\nExample 4: Gaussian-Bernoulli RBM");
    gaussian_bernoulli_example()?;

    // Example 5: Generative modeling and reconstruction
    println!("\nExample 5: Generative Modeling and Reconstruction");
    generative_modeling_example()?;

    // Example 6: Training parameter sensitivity analysis
    println!("\nExample 6: Training Parameter Sensitivity");
    parameter_sensitivity_analysis()?;

    Ok(())
}

fn binary_pattern_learning_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create simple binary patterns for learning
    let patterns = vec![
        vec![1.0, 1.0, 0.0, 0.0], // Pattern A
        vec![0.0, 1.0, 1.0, 0.0], // Pattern B
        vec![0.0, 0.0, 1.0, 1.0], // Pattern C
        vec![1.0, 0.0, 0.0, 1.0], // Pattern D
        vec![1.0, 1.0, 1.0, 0.0], // Pattern E
        vec![0.0, 1.0, 1.0, 1.0], // Pattern F
    ];

    // Create training dataset with duplicates to reinforce patterns
    let mut training_data = Vec::new();
    for _ in 0..5 {
        // Repeat each pattern 5 times
        for pattern in &patterns {
            training_data.push(TrainingSample::new(pattern.clone()));
        }
    }

    // Configure training with quantum annealing
    let training_config = QbmTrainingConfig {
        learning_rate: 0.05,
        epochs: 50,
        batch_size: 6,
        k_steps: 1,
        persistent_cd: false,
        annealing_params: AnnealingParams {
            num_sweeps: 500,
            num_repetitions: 5,
            ..Default::default()
        },
        seed: Some(42),
        log_frequency: 10,
        ..Default::default()
    };

    // Create and train RBM
    let start = Instant::now();
    let mut rbm = create_binary_rbm(4, 3, training_config)?;
    rbm.train(&training_data)?;
    let training_time = start.elapsed();

    println!("Binary Pattern Learning Results:");
    println!("  Patterns: 6 different 4-bit binary patterns");
    println!(
        "  Training samples: {} (with repetition)",
        training_data.len()
    );
    println!("  RBM architecture: 4 visible × 3 hidden units");
    println!("  Training time: {training_time:.2?}");

    // Test pattern reconstruction
    println!("\n  Pattern reconstruction test:");
    for (i, pattern) in patterns.iter().enumerate() {
        let result = rbm.infer(pattern)?;
        let reconstruction_error: f64 = pattern
            .iter()
            .zip(result.reconstruction.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f64>()
            / pattern.len() as f64;

        println!(
            "    Pattern {}: {:?} → {:?} (error: {:.4})",
            i + 1,
            pattern.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            result
                .reconstruction
                .iter()
                .map(|&x| (x + 0.5) as i32)
                .collect::<Vec<_>>(),
            reconstruction_error
        );
    }

    // Generate new samples
    println!("\n  Generated samples from learned distribution:");
    let generated = rbm.generate_samples(8)?;
    for (i, sample) in generated.iter().enumerate() {
        println!(
            "    Sample {}: {:?}",
            i + 1,
            sample.iter().map(|&x| (x + 0.5) as i32).collect::<Vec<_>>()
        );
    }

    // Show training statistics
    if let Some(stats) = rbm.get_training_stats() {
        println!("\n  Training statistics:");
        println!("    Epochs completed: {}", stats.epochs_completed);
        println!(
            "    Final reconstruction error: {:.6}",
            stats.final_reconstruction_error
        );
        println!("    Converged: {}", stats.converged);
        println!(
            "    Quantum sampling calls: {}",
            stats.quantum_sampling_stats.sampling_calls
        );
        println!(
            "    Classical fallback rate: {:.2}%",
            stats.quantum_sampling_stats.classical_fallback_rate * 100.0
        );
    }

    Ok(())
}

fn digit_feature_learning_example() -> Result<(), Box<dyn std::error::Error>> {
    // Simplified 3x3 digit patterns (0, 1, 2)
    let digit_patterns = vec![
        // Digit 0
        vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        // Digit 1
        vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        // Digit 2
        vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    ];

    // Create training samples
    let training_data: Vec<TrainingSample> = digit_patterns
        .iter()
        .map(|pattern| TrainingSample::new(pattern.clone()))
        .collect();

    // Configure training for feature learning
    let training_config = QbmTrainingConfig {
        learning_rate: 0.03,
        epochs: 80,
        batch_size: 3,
        k_steps: 2,
        persistent_cd: true,
        annealing_params: AnnealingParams {
            num_sweeps: 1000,
            num_repetitions: 8,
            ..Default::default()
        },
        seed: Some(123),
        log_frequency: 20,
        ..Default::default()
    };

    let start = Instant::now();
    let mut rbm = create_binary_rbm(9, 5, training_config)?;
    rbm.train(&training_data)?;
    let training_time = start.elapsed();

    println!("Digit Feature Learning Results:");
    println!("  Dataset: 3×3 simplified digits (0, 1, 2)");
    println!("  Training samples: {}", training_data.len());
    println!("  RBM architecture: 9 visible × 5 hidden units");
    println!("  Training time: {training_time:.2?}");

    // Test digit reconstruction
    println!("\n  Digit reconstruction (3×3 grid):");
    for (i, pattern) in digit_patterns.iter().take(3).enumerate() {
        let result = rbm.infer(pattern)?;

        println!("    Original digit {i}:");
        print_3x3_pattern(pattern);

        println!("    Reconstructed:");
        print_3x3_pattern(&result.reconstruction);

        println!(
            "    Hidden activations: {:?}",
            result
                .hidden_activations
                .iter()
                .map(|&x| format!("{x:.3}"))
                .collect::<Vec<_>>()
        );
        println!("    Free energy: {:.4}", result.free_energy);
        println!();
    }

    // Generate new digit-like patterns
    println!("  Generated digit-like patterns:");
    let generated = rbm.generate_samples(4)?;
    for (i, sample) in generated.iter().enumerate() {
        println!("    Generated pattern {}:", i + 1);
        print_3x3_pattern(sample);
        println!();
    }

    Ok(())
}

fn quantum_vs_classical_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple dataset for comparison
    let training_data = vec![
        TrainingSample::new(vec![1.0, 0.0, 1.0, 0.0]),
        TrainingSample::new(vec![0.0, 1.0, 0.0, 1.0]),
        TrainingSample::new(vec![1.0, 1.0, 0.0, 0.0]),
        TrainingSample::new(vec![0.0, 0.0, 1.0, 1.0]),
    ];

    // Configuration with quantum sampling
    let quantum_config = QbmTrainingConfig {
        learning_rate: 0.04,
        epochs: 30,
        batch_size: 2,
        k_steps: 1,
        annealing_params: AnnealingParams {
            num_sweeps: 300,
            num_repetitions: 3,
            ..Default::default()
        },
        seed: Some(456),
        log_frequency: 10,
        ..Default::default()
    };

    // Create quantum-enabled RBM
    let visible_config =
        LayerConfig::new("visible".to_string(), 4, UnitType::Binary).with_quantum_sampling(true);
    let hidden_config =
        LayerConfig::new("hidden".to_string(), 3, UnitType::Binary).with_quantum_sampling(true);

    let start = Instant::now();
    let mut quantum_rbm = QuantumRestrictedBoltzmannMachine::new(
        visible_config,
        hidden_config,
        quantum_config.clone(),
    )?;
    quantum_rbm.train(&training_data)?;
    let quantum_time = start.elapsed();

    // Create classical RBM (same config but no quantum sampling)
    let visible_config_classical =
        LayerConfig::new("visible".to_string(), 4, UnitType::Binary).with_quantum_sampling(false);
    let hidden_config_classical =
        LayerConfig::new("hidden".to_string(), 3, UnitType::Binary).with_quantum_sampling(false);

    let start = Instant::now();
    let mut classical_rbm = QuantumRestrictedBoltzmannMachine::new(
        visible_config_classical,
        hidden_config_classical,
        quantum_config,
    )?;
    classical_rbm.train(&training_data)?;
    let classical_time = start.elapsed();

    println!("Quantum vs Classical Sampling Comparison:");
    println!("  Dataset: 4 binary patterns (4-bit each)");
    println!("  Architecture: 4 visible × 3 hidden units");

    println!("\n  Training Performance:");
    println!("    Quantum sampling time: {quantum_time:.2?}");
    println!("    Classical sampling time: {classical_time:.2?}");
    println!(
        "    Speedup ratio: {:.2}x",
        classical_time.as_secs_f64() / quantum_time.as_secs_f64()
    );

    // Compare reconstruction quality
    println!("\n  Reconstruction Quality:");
    let test_pattern = vec![1.0, 0.0, 1.0, 0.0];

    let quantum_result = quantum_rbm.infer(&test_pattern)?;
    let classical_result = classical_rbm.infer(&test_pattern)?;

    let quantum_error: f64 = test_pattern
        .iter()
        .zip(quantum_result.reconstruction.iter())
        .map(|(orig, recon)| (orig - recon).powi(2))
        .sum::<f64>()
        / test_pattern.len() as f64;

    let classical_error: f64 = test_pattern
        .iter()
        .zip(classical_result.reconstruction.iter())
        .map(|(orig, recon)| (orig - recon).powi(2))
        .sum::<f64>()
        / test_pattern.len() as f64;

    println!("    Quantum reconstruction error: {quantum_error:.6}");
    println!("    Classical reconstruction error: {classical_error:.6}");
    println!(
        "    Improvement: {:.2}%",
        ((classical_error - quantum_error) / classical_error * 100.0)
    );

    // Compare sampling statistics
    if let Some(q_stats) = quantum_rbm.get_training_stats() {
        println!("\n  Quantum Sampling Statistics:");
        println!(
            "    Total sampling calls: {}",
            q_stats.quantum_sampling_stats.sampling_calls
        );
        println!(
            "    Success rate: {:.2}%",
            q_stats.quantum_sampling_stats.success_rate * 100.0
        );
        println!(
            "    Classical fallback rate: {:.2}%",
            q_stats.quantum_sampling_stats.classical_fallback_rate * 100.0
        );
    }

    Ok(())
}

fn gaussian_bernoulli_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create continuous-valued training data (normalized)
    let continuous_patterns = vec![
        vec![0.8, 0.1, 0.9, 0.2], // High-low pattern
        vec![0.2, 0.9, 0.1, 0.8], // Low-high pattern
        vec![0.7, 0.7, 0.3, 0.3], // High-high, low-low
        vec![0.3, 0.3, 0.7, 0.7], // Low-low, high-high
        vec![0.9, 0.4, 0.6, 0.1], // Mixed pattern 1
        vec![0.1, 0.6, 0.4, 0.9], // Mixed pattern 2
    ];

    // Create training samples (repeat for more data)
    let mut training_data = Vec::new();
    for _ in 0..3 {
        for pattern in &continuous_patterns {
            training_data.push(TrainingSample::new(pattern.clone()));
        }
    }

    // Configure for Gaussian-Bernoulli RBM
    let training_config = QbmTrainingConfig {
        learning_rate: 0.02,
        epochs: 60,
        batch_size: 4,
        k_steps: 2,
        persistent_cd: false,
        annealing_params: AnnealingParams {
            num_sweeps: 800,
            num_repetitions: 6,
            ..Default::default()
        },
        seed: Some(789),
        log_frequency: 15,
        ..Default::default()
    };

    let start = Instant::now();
    let mut rbm = create_gaussian_bernoulli_rbm(4, 4, training_config)?;
    rbm.train(&training_data)?;
    let training_time = start.elapsed();

    println!("Gaussian-Bernoulli RBM Results:");
    println!("  Data type: Continuous-valued patterns (normalized to [0,1])");
    println!("  Training samples: {}", training_data.len());
    println!("  Architecture: 4 Gaussian visible × 4 Bernoulli hidden");
    println!("  Training time: {training_time:.2?}");

    // Test continuous data reconstruction
    println!("\n  Continuous pattern reconstruction:");
    for (i, pattern) in continuous_patterns.iter().take(3).enumerate() {
        let result = rbm.infer(pattern)?;
        let mse: f64 = pattern
            .iter()
            .zip(result.reconstruction.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f64>()
            / pattern.len() as f64;

        println!("    Pattern {}: [{:.2}, {:.2}, {:.2}, {:.2}] → [{:.2}, {:.2}, {:.2}, {:.2}] (MSE: {:.4})",
                 i + 1,
                 pattern[0], pattern[1], pattern[2], pattern[3],
                 result.reconstruction[0], result.reconstruction[1],
                 result.reconstruction[2], result.reconstruction[3],
                 mse);

        println!(
            "      Hidden features: [{:.3}, {:.3}, {:.3}, {:.3}]",
            result.hidden_activations[0],
            result.hidden_activations[1],
            result.hidden_activations[2],
            result.hidden_activations[3]
        );
    }

    // Generate new continuous samples
    println!("\n  Generated continuous samples:");
    let generated = rbm.generate_samples(5)?;
    for (i, sample) in generated.iter().enumerate() {
        println!(
            "    Sample {}: [{:.3}, {:.3}, {:.3}, {:.3}]",
            i + 1,
            sample[0],
            sample[1],
            sample[2],
            sample[3]
        );
    }

    Ok(())
}

fn generative_modeling_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a larger dataset with more complex patterns
    let complex_patterns = vec![
        // Checkerboard patterns
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        // Edge patterns
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        // Center patterns
        vec![0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        // Random-like patterns
        vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    ];

    // Create training dataset with noise and variations
    let mut training_data = Vec::new();
    for _ in 0..4 {
        // Multiple repetitions
        for pattern in &complex_patterns {
            // Add original pattern
            training_data.push(TrainingSample::new(pattern.clone()));

            // Add noisy version
            let mut noisy_pattern = pattern.clone();
            if thread_rng().gen::<f64>() < 0.3 {
                // 30% chance to flip one bit
                let mut rng = thread_rng();
                let flip_idx = rng.gen_range(0..noisy_pattern.len());
                noisy_pattern[flip_idx] = 1.0 - noisy_pattern[flip_idx];
                training_data.push(TrainingSample::new(noisy_pattern));
            }
        }
    }

    // Configure for generative modeling
    let training_config = QbmTrainingConfig {
        learning_rate: 0.03,
        epochs: 100,
        batch_size: 8,
        k_steps: 3,
        persistent_cd: true,
        weight_decay: 0.001,
        momentum: 0.7,
        annealing_params: AnnealingParams {
            num_sweeps: 1200,
            num_repetitions: 10,
            ..Default::default()
        },
        seed: Some(321),
        error_threshold: Some(0.01),
        log_frequency: 25,
        ..Default::default()
    };

    let start = Instant::now();
    let mut rbm = create_binary_rbm(6, 8, training_config)?;
    rbm.train(&training_data)?;
    let training_time = start.elapsed();

    println!("Generative Modeling Results:");
    println!("  Dataset: Complex 6-bit patterns with noise");
    println!(
        "  Training samples: {} (including noisy variants)",
        training_data.len()
    );
    println!("  Architecture: 6 visible × 8 hidden units");
    println!("  Training time: {training_time:.2?}");

    // Test denoising capability
    println!("\n  Denoising capability test:");
    let clean_pattern = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let noisy_pattern = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]; // 2 bits flipped

    let clean_result = rbm.infer(&clean_pattern)?;
    let noisy_result = rbm.infer(&noisy_pattern)?;

    println!(
        "    Clean input:  {:?}",
        clean_pattern.iter().map(|&x| x as i32).collect::<Vec<_>>()
    );
    println!(
        "    Clean output: {:?}",
        clean_result
            .reconstruction
            .iter()
            .map(|&x| (x + 0.5) as i32)
            .collect::<Vec<_>>()
    );

    println!(
        "    Noisy input:  {:?}",
        noisy_pattern.iter().map(|&x| x as i32).collect::<Vec<_>>()
    );
    println!(
        "    Denoised:     {:?}",
        noisy_result
            .reconstruction
            .iter()
            .map(|&x| (x + 0.5) as i32)
            .collect::<Vec<_>>()
    );

    // Generate diverse samples
    println!("\n  Generated diverse patterns:");
    let generated = rbm.generate_samples(10)?;
    for (i, sample) in generated.iter().enumerate() {
        println!(
            "    Pattern {}: {:?}",
            i + 1,
            sample.iter().map(|&x| (x + 0.5) as i32).collect::<Vec<_>>()
        );
    }

    // Analyze learned features
    println!("\n  Feature analysis on original patterns:");
    for (i, pattern) in complex_patterns.iter().take(4).enumerate() {
        let result = rbm.infer(pattern)?;
        let active_features: Vec<usize> = result
            .hidden_activations
            .iter()
            .enumerate()
            .filter(|(_, &activation)| activation > 0.5)
            .map(|(idx, _)| idx)
            .collect();

        println!(
            "    Pattern {}: {:?} → Active features: {:?}",
            i + 1,
            pattern.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            active_features
        );
    }

    Ok(())
}

fn parameter_sensitivity_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Simple test dataset
    let test_data = vec![
        TrainingSample::new(vec![1.0, 0.0, 1.0]),
        TrainingSample::new(vec![0.0, 1.0, 0.0]),
        TrainingSample::new(vec![1.0, 1.0, 0.0]),
        TrainingSample::new(vec![0.0, 0.0, 1.0]),
    ];

    println!("Parameter Sensitivity Analysis:");
    println!("  Test dataset: 4 simple 3-bit patterns");
    println!("  Architecture: 3 visible × 2 hidden units");

    // Test different learning rates
    let learning_rates = vec![0.001, 0.01, 0.05, 0.1];
    println!("\n  Learning rate sensitivity:");

    for &lr in &learning_rates {
        let config = QbmTrainingConfig {
            learning_rate: lr,
            epochs: 20,
            batch_size: 2,
            k_steps: 1,
            annealing_params: AnnealingParams {
                num_sweeps: 200,
                num_repetitions: 3,
                ..Default::default()
            },
            seed: Some(42),
            log_frequency: 1000, // No logging
            ..Default::default()
        };

        let start = Instant::now();
        let mut rbm = create_binary_rbm(3, 2, config)?;
        rbm.train(&test_data)?;
        let time = start.elapsed();

        if let Some(stats) = rbm.get_training_stats() {
            println!(
                "    LR {:.3}: Error = {:.4}, Time = {:.1?}",
                lr, stats.final_reconstruction_error, time
            );
        }
    }

    // Test different hidden layer sizes
    let hidden_sizes = vec![1, 2, 4, 6];
    println!("\n  Hidden layer size sensitivity:");

    for &hidden_size in &hidden_sizes {
        let config = QbmTrainingConfig {
            learning_rate: 0.03,
            epochs: 25,
            batch_size: 2,
            k_steps: 1,
            annealing_params: AnnealingParams {
                num_sweeps: 300,
                num_repetitions: 4,
                ..Default::default()
            },
            seed: Some(42),
            log_frequency: 1000,
            ..Default::default()
        };

        let start = Instant::now();
        let mut rbm = create_binary_rbm(3, hidden_size, config)?;
        rbm.train(&test_data)?;
        let time = start.elapsed();

        if let Some(stats) = rbm.get_training_stats() {
            println!(
                "    {} hidden: Error = {:.4}, Time = {:.1?}",
                hidden_size, stats.final_reconstruction_error, time
            );
        }
    }

    // Test different k-step values for contrastive divergence
    let k_values = vec![1, 2, 5, 10];
    println!("\n  Contrastive divergence k-step sensitivity:");

    for &k in &k_values {
        let config = QbmTrainingConfig {
            learning_rate: 0.03,
            epochs: 15,
            batch_size: 2,
            k_steps: k,
            annealing_params: AnnealingParams {
                num_sweeps: 400,
                num_repetitions: 5,
                ..Default::default()
            },
            seed: Some(42),
            log_frequency: 1000,
            ..Default::default()
        };

        let start = Instant::now();
        let mut rbm = create_binary_rbm(3, 2, config)?;
        rbm.train(&test_data)?;
        let time = start.elapsed();

        if let Some(stats) = rbm.get_training_stats() {
            println!(
                "    CD-{}: Error = {:.4}, Time = {:.1?}",
                k, stats.final_reconstruction_error, time
            );
        }
    }

    println!("\n  Analysis complete. Optimal parameters depend on:");
    println!("    - Dataset complexity and size");
    println!("    - Desired training time vs accuracy trade-off");
    println!("    - Available quantum annealing resources");

    Ok(())
}

/// Helper function to print 3x3 binary pattern
fn print_3x3_pattern(pattern: &[f64]) {
    for row in 0..3 {
        print!("      ");
        for col in 0..3 {
            let idx = row * 3 + col;
            let symbol = if pattern[idx] > 0.5 { "█" } else { "·" };
            print!("{symbol} ");
        }
        println!();
    }
}
