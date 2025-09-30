//! Quantum Few-Shot Learning Example
//!
//! This example demonstrates how to use quantum few-shot learning algorithms
//! to learn from very limited training examples.

use scirs2_core::ndarray::{Array1, Array2};
use quantrs2_ml::autodiff::optimizers::Adam;
use quantrs2_ml::prelude::*;
use quantrs2_ml::qnn::QNNLayerType;
use scirs2_core::random::prelude::*;

fn main() -> Result<()> {
    println!("=== Quantum Few-Shot Learning Demo ===\n");

    // Step 1: Generate synthetic dataset
    println!("1. Generating synthetic dataset for 5-way classification...");
    let num_samples_per_class = 20;
    let num_classes = 5;
    let num_features = 4;
    let total_samples = num_samples_per_class * num_classes;

    // Generate data with different patterns for each class
    let mut data = Array2::zeros((total_samples, num_features));
    let mut labels = Array1::zeros(total_samples);

    for class_id in 0..num_classes {
        for sample_idx in 0..num_samples_per_class {
            let idx = class_id * num_samples_per_class + sample_idx;

            // Create class-specific patterns
            for feat in 0..num_features {
                data[[idx, feat]] =
                    (class_id as f64 * 0.5 + feat as f64 * 0.3 + sample_idx as f64 * 0.1).sin()
                        + 0.1 * (2.0 * thread_rng().gen::<f64>() - 1.0);
            }
            labels[idx] = class_id;
        }
    }

    println!(
        "   Dataset created: {} samples, {} features, {} classes",
        total_samples, num_features, num_classes
    );

    // Step 2: Create quantum model for few-shot learning
    println!("\n2. Creating quantum neural network...");
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::EntanglementLayer {
            connectivity: "circular".to_string(),
        },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let qnn = QuantumNeuralNetwork::new(layers, 4, num_features, num_classes)?;
    println!("   Quantum model created with {} qubits", qnn.num_qubits);

    // Step 3: Test different few-shot learning methods
    println!("\n3. Testing few-shot learning methods:");

    // Method 1: Prototypical Networks
    println!("\n   a) Prototypical Networks (5-way 3-shot):");
    test_prototypical_networks(&data, &labels, qnn.clone())?;

    // Method 2: MAML
    println!("\n   b) Model-Agnostic Meta-Learning (MAML):");
    test_maml(&data, &labels, qnn.clone())?;

    // Step 4: Compare performance across different shot values
    println!("\n4. Performance comparison across different K-shot values:");
    compare_shot_performance(&data, &labels, qnn.clone())?;

    println!("\n=== Few-Shot Learning Demo Complete ===");

    Ok(())
}

/// Test prototypical networks
fn test_prototypical_networks(
    data: &Array2<f64>,
    labels: &Array1<usize>,
    qnn: QuantumNeuralNetwork,
) -> Result<()> {
    let mut learner = FewShotLearner::new(FewShotMethod::PrototypicalNetworks, qnn);

    // Generate episodes for training
    let num_episodes = 10;
    let mut episodes = Vec::new();

    for _ in 0..num_episodes {
        let episode = FewShotLearner::generate_episode(
            data, labels, 5, // 5-way
            3, // 3-shot
            5, // 5 query examples per class
        )?;
        episodes.push(episode);
    }

    // Train
    let mut optimizer = Adam::new(0.01);
    let accuracies = learner.train(&episodes, &mut optimizer, 20)?;

    // Print results
    println!("   Training completed:");
    println!("   - Initial accuracy: {:.2}%", accuracies[0] * 100.0);
    println!(
        "   - Final accuracy: {:.2}%",
        accuracies.last().unwrap() * 100.0
    );
    println!(
        "   - Improvement: {:.2}%",
        (accuracies.last().unwrap() - accuracies[0]) * 100.0
    );

    Ok(())
}

/// Test MAML
fn test_maml(data: &Array2<f64>, labels: &Array1<usize>, qnn: QuantumNeuralNetwork) -> Result<()> {
    let mut learner = FewShotLearner::new(
        FewShotMethod::MAML {
            inner_steps: 5,
            inner_lr: 0.01,
        },
        qnn,
    );

    // Generate meta-training tasks
    let num_tasks = 20;
    let mut tasks = Vec::new();

    for _ in 0..num_tasks {
        let task = FewShotLearner::generate_episode(
            data, labels, 3, // 3-way (fewer classes for MAML)
            5, // 5-shot
            5, // 5 query examples
        )?;
        tasks.push(task);
    }

    // Meta-train
    let mut meta_optimizer = Adam::new(0.001);
    let losses = learner.train(&tasks, &mut meta_optimizer, 10)?;

    println!("   Meta-training completed:");
    println!("   - Initial loss: {:.4}", losses[0]);
    println!("   - Final loss: {:.4}", losses.last().unwrap());
    println!(
        "   - Convergence rate: {:.2}%",
        (1.0 - losses.last().unwrap() / losses[0]) * 100.0
    );

    Ok(())
}

/// Compare performance across different K-shot values
fn compare_shot_performance(
    data: &Array2<f64>,
    labels: &Array1<usize>,
    qnn: QuantumNeuralNetwork,
) -> Result<()> {
    let k_values = vec![1, 3, 5, 10];

    for k in k_values {
        println!("\n   Testing {}-shot learning:", k);

        let mut learner = FewShotLearner::new(FewShotMethod::PrototypicalNetworks, qnn.clone());

        // Generate episodes
        let mut episodes = Vec::new();
        for _ in 0..5 {
            let episode = FewShotLearner::generate_episode(
                data, labels, 3, // 3-way
                k, // k-shot
                5, // 5 query
            )?;
            episodes.push(episode);
        }

        // Quick training
        let mut optimizer = Adam::new(0.01);
        let accuracies = learner.train(&episodes, &mut optimizer, 10)?;

        println!(
            "     Final accuracy: {:.2}%",
            accuracies.last().unwrap() * 100.0
        );
    }

    Ok(())
}

/// Demonstrate episode structure
fn demonstrate_episode_structure() -> Result<()> {
    println!("\n5. Episode Structure Demonstration:");

    // Create a simple episode manually
    let support_set = vec![
        // Class 0
        (Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]), 0),
        (Array1::from_vec(vec![0.15, 0.25, 0.35, 0.45]), 0),
        // Class 1
        (Array1::from_vec(vec![0.8, 0.7, 0.6, 0.5]), 1),
        (Array1::from_vec(vec![0.85, 0.75, 0.65, 0.55]), 1),
    ];

    let query_set = vec![
        (Array1::from_vec(vec![0.12, 0.22, 0.32, 0.42]), 0),
        (Array1::from_vec(vec![0.82, 0.72, 0.62, 0.52]), 1),
    ];

    let episode = Episode {
        support_set,
        query_set,
        num_classes: 2,
        k_shot: 2,
    };

    println!("   2-way 2-shot episode created");
    println!("   - Support set size: {}", episode.support_set.len());
    println!("   - Query set size: {}", episode.query_set.len());

    Ok(())
}
