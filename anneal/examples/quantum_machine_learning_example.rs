//! Example demonstrating Quantum Machine Learning with Annealing
//!
//! This example shows how to:
//! 1. Train Variational Quantum Classifiers (VQC)
//! 2. Use Quantum Neural Networks (QNN)
//! 3. Apply Quantum Feature Maps for data encoding
//! 4. Train Quantum Kernel Methods (SVM)
//! 5. Create Quantum Generative Adversarial Networks (QGAN)
//! 6. Implement Quantum Reinforcement Learning
//! 7. Use Quantum Autoencoders for dimensionality reduction
//! 8. Compare quantum vs classical performance

use quantrs2_anneal::{ising::IsingModel, quantum_machine_learning::*};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Quantum Machine Learning with Annealing Demo ===\n");

    // Example 1: Variational Quantum Classifier
    println!("Example 1: Variational Quantum Classifier");
    variational_quantum_classifier_example()?;

    // Example 2: Quantum Neural Network
    println!("\nExample 2: Quantum Neural Network");
    quantum_neural_network_example()?;

    // Example 3: Quantum Feature Maps
    println!("\nExample 3: Quantum Feature Maps");
    quantum_feature_maps_example()?;

    // Example 4: Quantum Kernel Methods
    println!("\nExample 4: Quantum Kernel Methods");
    quantum_kernel_methods_example()?;

    // Example 5: Quantum Generative Adversarial Network
    println!("\nExample 5: Quantum Generative Adversarial Network");
    quantum_gan_example()?;

    // Example 6: Quantum Reinforcement Learning
    println!("\nExample 6: Quantum Reinforcement Learning");
    quantum_reinforcement_learning_example()?;

    // Example 7: Quantum Autoencoder
    println!("\nExample 7: Quantum Autoencoder");
    quantum_autoencoder_example()?;

    // Example 8: Quantum vs Classical Comparison
    println!("\nExample 8: Quantum vs Classical Performance Comparison");
    quantum_classical_comparison_example()?;

    Ok(())
}

fn variational_quantum_classifier_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Training a VQC for binary classification...");

    // Create synthetic dataset for binary classification
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut training_data = Vec::new();

    // Generate training samples
    for _ in 0..50 {
        let x1 = rng.gen_range(-2.0..2.0);
        let x2 = rng.gen_range(-2.0..2.0);
        let x3 = rng.gen_range(-2.0..2.0);
        let x4 = rng.gen_range(-2.0..2.0);

        // Simple classification rule: sum > 0 -> class 1, else class 0
        let label = if x1 + x2 + x3 + x4 > 0.0 { 1 } else { 0 };

        training_data.push(TrainingSample {
            features: vec![x1, x2, x3, x4],
            label,
            weight: 1.0,
        });
    }

    // Create and configure VQC
    let config = VqcConfig {
        max_iterations: 50,
        learning_rate: 0.01,
        num_shots: 256,
        tolerance: 1e-3,
        ..Default::default()
    };

    let start = Instant::now();
    let mut vqc = VariationalQuantumClassifier::new(4, 4, 2, 3, config)?;

    // Train the classifier
    vqc.train(&training_data)?;
    let training_time = start.elapsed();

    // Test the classifier
    let test_sample = vec![0.5, -0.3, 0.8, -0.1];
    let prediction = vqc.predict(&test_sample)?;
    let probabilities = vqc.predict_proba(&test_sample)?;

    println!("    Training completed in {:.2?}", training_time);
    println!("    Training samples: {}", training_data.len());
    println!(
        "    Final loss: {:.6}",
        vqc.training_history.losses.last().unwrap_or(&0.0)
    );
    println!("    Test prediction: class {}", prediction);
    println!(
        "    Class probabilities: [{:.3}, {:.3}]",
        probabilities[0], probabilities[1]
    );

    // Evaluate accuracy on training data
    let mut correct = 0;
    for sample in &training_data[..20] {
        // Test on subset
        let pred = vqc.predict(&sample.features)?;
        if pred == sample.label {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / 20.0;
    println!("    Training accuracy (subset): {:.1}%", accuracy * 100.0);

    Ok(())
}

fn quantum_neural_network_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Training a Quantum Neural Network...");

    // Create synthetic regression dataset
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let mut training_data = Vec::new();

    for _ in 0..30 {
        let x1: f64 = rng.gen_range(-1.0..1.0);
        let x2: f64 = rng.gen_range(-1.0..1.0);
        let x3: f64 = rng.gen_range(-1.0..1.0);

        // Target function: y = sin(x1) + cos(x2) + x3^2
        let y1 = x1.sin() + x2.cos() + x3 * x3;
        let y2 = (x1 * x2).cos() + x3.sin();

        training_data.push((vec![x1, x2, x3], vec![y1, y2]));
    }

    // Create QNN
    let config = QnnConfig {
        learning_rate: 0.02,
        max_epochs: 10,
        batch_size: 16,
        tolerance: 1e-3,
        ..Default::default()
    };

    let start = Instant::now();
    let mut qnn = QuantumNeuralNetwork::new(&[3, 6, 2], config)?;

    // Train the network
    qnn.train(&training_data)?;
    let training_time = start.elapsed();

    // Test the network
    let test_input = vec![0.5, -0.3, 0.8];
    let output = qnn.forward(&test_input)?;

    println!("    Training completed in {:.2?}", training_time);
    println!("    Architecture: 3 -> 6 -> 2");
    println!("    Training samples: {}", training_data.len());
    println!(
        "    Final loss: {:.6}",
        qnn.training_history.losses.last().unwrap_or(&0.0)
    );
    println!(
        "    Test input: [{:.2}, {:.2}, {:.2}]",
        test_input[0], test_input[1], test_input[2]
    );
    println!("    Test output: [{:.3}, {:.3}]", output[0], output[1]);

    // Expected output for comparison
    let expected = vec![
        test_input[0].sin() + test_input[1].cos() + test_input[2] * test_input[2],
        (test_input[0] * test_input[1]).cos() + test_input[2].sin(),
    ];
    println!("    Expected: [{:.3}, {:.3}]", expected[0], expected[1]);

    let error: f64 = output
        .iter()
        .zip(expected.iter())
        .map(|(o, e)| (o - e).abs())
        .sum::<f64>()
        / output.len() as f64;
    println!("    Mean absolute error: {:.3}", error);

    Ok(())
}

fn quantum_feature_maps_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Demonstrating various Quantum Feature Maps...");

    let test_data = vec![0.5, -0.3, 0.8, 0.2];

    // Test different feature map types
    let feature_maps = vec![
        ("Angle Encoding", FeatureMapType::AngleEncoding),
        (
            "Pauli (Linear)",
            FeatureMapType::PauliFeatureMap {
                entanglement: EntanglementType::Linear,
            },
        ),
        (
            "Pauli (Circular)",
            FeatureMapType::PauliFeatureMap {
                entanglement: EntanglementType::Circular,
            },
        ),
        (
            "ZZ Feature Map",
            FeatureMapType::ZZFeatureMap { repetitions: 2 },
        ),
    ];

    println!(
        "    Input data: [{:.2}, {:.2}, {:.2}, {:.2}]",
        test_data[0], test_data[1], test_data[2], test_data[3]
    );
    println!("    Feature Map Results:");

    for (name, map_type) in feature_maps {
        let feature_map = QuantumFeatureMap::new(4, 4, map_type)?;
        let encoded = feature_map.encode(&test_data)?;

        let encoded_str = encoded
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
            .join(", ");

        println!("      {}: [{}]", name, encoded_str);
        println!("        Circuit depth: {}", feature_map.circuit.depth);
        println!("        Parameters: {}", feature_map.circuit.num_parameters);
    }

    // Demonstrate feature map kernel computation
    println!("\n    Quantum Kernel Computation:");
    let feature_map =
        QuantumFeatureMap::new(4, 4, FeatureMapType::ZZFeatureMap { repetitions: 2 })?;

    let x1 = vec![0.5, -0.3, 0.8, 0.2];
    let x2 = vec![0.6, -0.2, 0.7, 0.3];
    let x3 = vec![-0.5, 0.3, -0.8, -0.2];

    // Create a simple kernel method for demonstration
    let kernel_method = QuantumKernelMethod::new(
        feature_map,
        KernelMethodType::SupportVectorMachine { c_parameter: 1.0 },
    );

    let k12 = kernel_method.quantum_kernel(&x1, &x2)?;
    let k13 = kernel_method.quantum_kernel(&x1, &x3)?;
    let k23 = kernel_method.quantum_kernel(&x2, &x3)?;

    println!("      K(x1, x2) = {:.3}", k12);
    println!("      K(x1, x3) = {:.3}", k13);
    println!("      K(x2, x3) = {:.3}", k23);

    Ok(())
}

fn quantum_kernel_methods_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Training Quantum Kernel Methods...");

    // Create synthetic classification dataset
    let mut rng = ChaCha8Rng::seed_from_u64(456);
    let mut training_data = Vec::new();

    for _ in 0..60 {
        let x1 = rng.gen_range(-2.0..2.0);
        let x2 = rng.gen_range(-2.0..2.0);

        // Nonlinear classification boundary: circle
        let label = if x1 * x1 + x2 * x2 < 1.0 { 1.0 } else { -1.0 };

        training_data.push((vec![x1, x2], label));
    }

    // Test different kernel methods
    let kernel_methods = vec![
        (
            "SVM",
            KernelMethodType::SupportVectorMachine { c_parameter: 1.0 },
        ),
        (
            "Ridge Regression",
            KernelMethodType::RidgeRegression {
                regularization: 0.1,
            },
        ),
        ("Gaussian Process", KernelMethodType::GaussianProcess),
    ];

    for (name, method_type) in kernel_methods {
        println!("    Training {} with quantum kernels...", name);

        let feature_map = QuantumFeatureMap::new(
            2,
            3,
            FeatureMapType::PauliFeatureMap {
                entanglement: EntanglementType::Circular,
            },
        )?;

        let start = Instant::now();
        let mut kernel_method = QuantumKernelMethod::new(feature_map, method_type);
        kernel_method.train(training_data.clone())?;
        let training_time = start.elapsed();

        // Test on training data
        let mut correct = 0;
        let test_samples = &training_data[..20];

        for (features, true_label) in test_samples {
            let prediction = kernel_method.predict(features)?;
            let predicted_label = if prediction > 0.0 { 1.0 } else { -1.0 };

            if (predicted_label - *true_label).abs() < 0.1 {
                correct += 1;
            }
        }

        let accuracy = correct as f64 / test_samples.len() as f64;

        println!("      Training time: {:.2?}", training_time);
        println!("      Training samples: {}", training_data.len());
        println!(
            "      Support vectors: {}",
            kernel_method.support_vectors.len()
        );
        println!("      Test accuracy: {:.1}%", accuracy * 100.0);

        // Test prediction
        let test_point = vec![0.5, 0.5];
        let prediction = kernel_method.predict(&test_point)?;
        println!("      Prediction for [0.5, 0.5]: {:.3}", prediction);
    }

    Ok(())
}

fn quantum_gan_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Training a Quantum Generative Adversarial Network...");

    // Create real data samples (2D Gaussian distribution)
    let mut rng = ChaCha8Rng::seed_from_u64(789);
    let mut real_data = Vec::new();

    for _ in 0..100 {
        let x = rng.gen_range(-1.0..1.0);
        let y = rng.gen_range(-1.0..1.0);
        // Transform to create a specific distribution
        let transformed_x = x * 0.8 + y * 0.2;
        let transformed_y = y * 0.8 - x * 0.2;
        real_data.push(vec![transformed_x, transformed_y]);
    }

    // Configure QGAN
    let config = QGanConfig {
        latent_dim: 3,
        data_dim: 2,
        epochs: 20,
        batch_size: 16,
        generator_lr: 0.01,
        discriminator_lr: 0.02,
        seed: Some(42),
    };

    let start = Instant::now();
    let mut qgan = QuantumGAN::new(config)?;

    // Train the QGAN
    qgan.train(&real_data)?;
    let training_time = start.elapsed();

    // Generate samples
    let mut gen_rng = ChaCha8Rng::seed_from_u64(999);
    let generated_samples = qgan.generate_samples(10, &mut gen_rng)?;

    println!("    Training completed in {:.2?}", training_time);
    println!("    Real data samples: {}", real_data.len());
    println!(
        "    Generator architecture: {} -> {} -> {}",
        qgan.config.latent_dim,
        qgan.config.data_dim * 2,
        qgan.config.data_dim
    );

    // Show training progress
    let history = &qgan.training_history;
    if !history.generator_losses.is_empty() {
        println!(
            "    Final generator loss: {:.4}",
            history.generator_losses.last().unwrap()
        );
        println!(
            "    Final discriminator loss: {:.4}",
            history.discriminator_losses.last().unwrap()
        );
    }

    // Display some generated samples
    println!("    Generated samples:");
    for (i, sample) in generated_samples.iter().take(5).enumerate() {
        println!(
            "      Sample {}: [{:.3}, {:.3}]",
            i + 1,
            sample[0],
            sample[1]
        );
    }

    // Calculate statistics
    let real_mean_x: f64 = real_data.iter().map(|s| s[0]).sum::<f64>() / real_data.len() as f64;
    let real_mean_y: f64 = real_data.iter().map(|s| s[1]).sum::<f64>() / real_data.len() as f64;

    let gen_mean_x: f64 =
        generated_samples.iter().map(|s| s[0]).sum::<f64>() / generated_samples.len() as f64;
    let gen_mean_y: f64 =
        generated_samples.iter().map(|s| s[1]).sum::<f64>() / generated_samples.len() as f64;

    println!(
        "    Real data mean: [{:.3}, {:.3}]",
        real_mean_x, real_mean_y
    );
    println!("    Generated mean: [{:.3}, {:.3}]", gen_mean_x, gen_mean_y);

    Ok(())
}

fn quantum_reinforcement_learning_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Demonstrating Quantum Reinforcement Learning...");

    // Simple environment: CartPole-like problem
    // State: [position, velocity, angle, angular_velocity]
    // Actions: 0 (left), 1 (right)

    let config = QRLConfig {
        state_dim: 4,
        action_dim: 2,
        buffer_capacity: 1000,
        learning_rate: 0.01,
        gamma: 0.99,
        epsilon: 0.1,
        use_actor_critic: true,
        seed: Some(42),
    };

    let mut agent = QuantumRLAgent::new(config)?;
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    println!("    Training agent on simplified CartPole environment...");

    // Simulate training episodes
    let num_episodes = 10;
    for episode in 0..num_episodes {
        let mut state = vec![
            rng.gen_range(-0.5..0.5), // position
            rng.gen_range(-0.1..0.1), // velocity
            rng.gen_range(-0.2..0.2), // angle
            rng.gen_range(-0.1..0.1), // angular velocity
        ];

        let mut episode_reward = 0.0;
        let max_steps = 50;

        for step in 0..max_steps {
            // Select action
            let action = agent.select_action(&state, &mut rng)?;

            // Simulate environment step
            let next_state = simulate_cartpole_step(&state, action, &mut rng);
            let reward = calculate_cartpole_reward(&next_state);
            let done = step == max_steps - 1 || next_state[2].abs() > 0.5; // angle threshold

            // Store experience
            agent.store_experience(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });

            episode_reward += reward;
            state = next_state;

            if done {
                break;
            }
        }

        // Train agent
        if episode % 3 == 0 && agent.experience_buffer.len() > 32 {
            agent.train()?;
        }

        agent.stats.episode_rewards.push(episode_reward);

        if episode % 2 == 0 {
            println!(
                "      Episode {}: Reward = {:.2}, Buffer size = {}",
                episode,
                episode_reward,
                agent.experience_buffer.len()
            );
        }
    }

    // Test trained agent
    println!("    Testing trained agent...");
    let test_state = vec![0.1, 0.0, 0.05, 0.0];
    let test_action = agent.select_action(&test_state, &mut rng)?;

    println!(
        "      Test state: [{:.2}, {:.2}, {:.2}, {:.2}]",
        test_state[0], test_state[1], test_state[2], test_state[3]
    );
    println!("      Selected action: {}", test_action);

    // Performance statistics
    let avg_reward: f64 =
        agent.stats.episode_rewards.iter().sum::<f64>() / agent.stats.episode_rewards.len() as f64;
    let max_reward = agent
        .stats
        .episode_rewards
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("      Average episode reward: {:.2}", avg_reward);
    println!("      Best episode reward: {:.2}", max_reward);
    println!(
        "      Experience buffer utilization: {:.1}%",
        agent.experience_buffer.len() as f64 / agent.config.buffer_capacity as f64 * 100.0
    );

    Ok(())
}

fn quantum_autoencoder_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Training a Quantum Autoencoder for dimensionality reduction...");

    // Create high-dimensional data with intrinsic low-dimensional structure
    let mut rng = ChaCha8Rng::seed_from_u64(321);
    let mut training_data = Vec::new();

    for _ in 0..80 {
        // Generate 2D latent variables
        let z1: f64 = rng.gen_range(-1.0..1.0);
        let z2: f64 = rng.gen_range(-1.0..1.0);

        // Map to 8D space with nonlinear transformation
        let data = vec![
            z1,
            z2,
            z1 * z2,
            z1.sin(),
            z2.cos(),
            (z1 + z2) / 2.0,
            z1.powi(2) - z2.powi(2),
            (z1 * z2).tanh(),
        ];

        training_data.push(data);
    }

    // Configure autoencoder
    let config = QAutoencoderConfig {
        input_dim: 8,
        latent_dim: 3,
        learning_rate: 0.02,
        epochs: 30,
        batch_size: 16,
        seed: Some(42),
    };

    let start = Instant::now();
    let mut autoencoder = QuantumAutoencoder::new(config)?;

    // Train the autoencoder
    autoencoder.train(&training_data)?;
    let training_time = start.elapsed();

    // Test the autoencoder
    let test_sample = &training_data[0];
    let encoded = autoencoder.encode(test_sample)?;
    let reconstructed = autoencoder.decode(&encoded)?;

    println!("    Training completed in {:.2?}", training_time);
    println!(
        "    Architecture: 8 -> 3 -> 8 (compression ratio: {:.1}x)",
        8.0 / 3.0
    );
    println!("    Training samples: {}", training_data.len());

    if !autoencoder.training_history.losses.is_empty() {
        println!(
            "    Final reconstruction loss: {:.6}",
            autoencoder.training_history.losses.last().unwrap()
        );
    }

    // Reconstruction quality
    let mut total_error = 0.0;
    let test_samples = &training_data[..10];

    for sample in test_samples {
        let reconstructed = autoencoder.forward(sample)?;
        let error: f64 = sample
            .iter()
            .zip(reconstructed.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f64>()
            .sqrt();
        total_error += error;
    }

    let avg_reconstruction_error = total_error / test_samples.len() as f64;
    println!(
        "    Average reconstruction error: {:.4}",
        avg_reconstruction_error
    );

    // Show example encoding/decoding
    println!("    Example encoding/decoding:");
    let input_str = test_sample
        .iter()
        .map(|x| format!("{:.2}", x))
        .collect::<Vec<_>>()
        .join(", ");
    let encoded_str = encoded
        .iter()
        .map(|x| format!("{:.2}", x))
        .collect::<Vec<_>>()
        .join(", ");
    let reconstructed_str = reconstructed
        .iter()
        .map(|x| format!("{:.2}", x))
        .collect::<Vec<_>>()
        .join(", ");

    println!("      Input (8D):        [{}]", input_str);
    println!("      Encoded (3D):      [{}]", encoded_str);
    println!("      Reconstructed (8D): [{}]", reconstructed_str);

    Ok(())
}

fn quantum_classical_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Comparing Quantum vs Classical ML Performance...");

    // Create benchmark dataset
    let mut rng = ChaCha8Rng::seed_from_u64(555);
    let mut dataset = Vec::new();

    for _ in 0..100 {
        let x1: f64 = rng.gen_range(-1.0..1.0);
        let x2: f64 = rng.gen_range(-1.0..1.0);
        let x3: f64 = rng.gen_range(-1.0..1.0);

        // Complex nonlinear decision boundary
        let label = if x1 * x1 + x2 * x2 - x3.sin() > 0.0 {
            1
        } else {
            0
        };

        dataset.push(TrainingSample {
            features: vec![x1, x2, x3],
            label,
            weight: 1.0,
        });
    }

    let training_data = &dataset[..80];
    let test_data = &dataset[80..];

    println!(
        "    Dataset: {} training, {} test samples",
        training_data.len(),
        test_data.len()
    );

    // Quantum VQC
    println!("\n    Training Quantum VQC...");
    let vqc_config = VqcConfig {
        max_iterations: 50,
        learning_rate: 0.02,
        num_shots: 256,
        ..Default::default()
    };

    let start = Instant::now();
    let mut quantum_classifier = VariationalQuantumClassifier::new(3, 4, 2, 2, vqc_config)?;
    quantum_classifier.train(training_data)?;
    let quantum_training_time = start.elapsed();

    // Test quantum classifier
    let mut quantum_correct = 0;
    for sample in test_data {
        let prediction = quantum_classifier.predict(&sample.features)?;
        if prediction == sample.label {
            quantum_correct += 1;
        }
    }
    let quantum_accuracy = quantum_correct as f64 / test_data.len() as f64;

    // Classical "Neural Network" (simplified)
    println!("    Training Classical NN (simplified)...");
    let start = Instant::now();
    let mut classical_weights = vec![rng.gen_range(-1.0..1.0); 12]; // 3*4 weights

    // Simple gradient descent
    for _ in 0..100 {
        for sample in training_data {
            let prediction = classical_predict(&sample.features, &classical_weights);
            let error = prediction - sample.label as f64;

            // Update weights (simplified)
            for (i, &feature) in sample.features.iter().enumerate() {
                classical_weights[i] -= 0.01 * error * feature;
            }
        }
    }
    let classical_training_time = start.elapsed();

    // Test classical classifier
    let mut classical_correct = 0;
    for sample in test_data {
        let prediction = classical_predict(&sample.features, &classical_weights);
        let predicted_class = if prediction > 0.5 { 1 } else { 0 };
        if predicted_class == sample.label {
            classical_correct += 1;
        }
    }
    let classical_accuracy = classical_correct as f64 / test_data.len() as f64;

    // Comparison results
    println!("\n    Performance Comparison:");
    println!("    ┌─────────────────┬─────────────┬─────────────┬─────────────┐");
    println!("    │ Method          │ Accuracy    │ Train Time  │ Parameters  │");
    println!("    ├─────────────────┼─────────────┼─────────────┼─────────────┤");
    println!(
        "    │ Quantum VQC     │ {:>9.1}%  │ {:>9.2?}  │ {:>9}   │",
        quantum_accuracy * 100.0,
        quantum_training_time,
        quantum_classifier.parameters.len()
    );
    println!(
        "    │ Classical NN    │ {:>9.1}%  │ {:>9.2?}  │ {:>9}   │",
        classical_accuracy * 100.0,
        classical_training_time,
        classical_weights.len()
    );
    println!("    └─────────────────┴─────────────┴─────────────┴─────────────┘");

    // Performance analysis
    let quantum_advantage = quantum_accuracy / classical_accuracy;
    println!("\n    Analysis:");
    println!("      Quantum advantage ratio: {:.2}x", quantum_advantage);

    if quantum_advantage > 1.1 {
        println!("      ✓ Quantum method shows significant advantage");
    } else if quantum_advantage > 0.9 {
        println!("      ≈ Methods show comparable performance");
    } else {
        println!("      ✗ Classical method performs better (quantum needs optimization)");
    }

    // Feature encoding analysis
    println!("\n    Quantum Feature Encoding Analysis:");
    let feature_map = &quantum_classifier.feature_map;
    let test_sample = vec![0.5, -0.3, 0.8];
    let encoded = feature_map.encode(&test_sample)?;

    println!(
        "      Original features: [{:.2}, {:.2}, {:.2}]",
        test_sample[0], test_sample[1], test_sample[2]
    );
    println!(
        "      Quantum encoding: [{:.3}, {:.3}, {:.3}]",
        encoded[0], encoded[1], encoded[2]
    );
    println!("      Feature map type: {:?}", feature_map.map_type);
    println!("      Circuit depth: {}", feature_map.circuit.depth);

    Ok(())
}

// Helper functions

fn simulate_cartpole_step(state: &[f64], action: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let pos = state[0];
    let vel = state[1];
    let angle = state[2];
    let ang_vel = state[3];

    // Simplified CartPole physics
    let force = if action == 1 { 1.0 } else { -1.0 };
    let dt = 0.02;

    let new_ang_vel = ang_vel + (angle.sin() * 9.8 - force * angle.cos()) * dt;
    let new_angle = angle + new_ang_vel * dt;
    let new_vel = vel + force * dt;
    let new_pos = pos + new_vel * dt;

    // Add small amount of noise
    vec![
        new_pos + rng.gen_range(-0.01..0.01),
        new_vel + rng.gen_range(-0.01..0.01),
        new_angle + rng.gen_range(-0.005..0.005),
        new_ang_vel + rng.gen_range(-0.01..0.01),
    ]
}

fn calculate_cartpole_reward(state: &[f64]) -> f64 {
    let pos = state[0];
    let angle = state[2];

    // Reward for keeping pole upright and cart centered
    let angle_reward = 1.0 - angle.abs() / 0.5;
    let position_reward = 1.0 - pos.abs() / 1.0;

    (angle_reward + position_reward) / 2.0
}

fn classical_predict(features: &[f64], weights: &[f64]) -> f64 {
    // Simple linear model with sigmoid activation
    let mut sum = 0.0;
    for (i, &feature) in features.iter().enumerate() {
        if i * 4 < weights.len() {
            sum += feature * weights[i * 4];
        }
    }

    // Sigmoid activation
    1.0 / (1.0 + (-sum).exp())
}
