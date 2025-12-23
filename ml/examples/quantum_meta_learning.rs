//! Quantum Meta-Learning Example
//!
//! This example demonstrates various quantum meta-learning algorithms including
//! MAML, Reptile, `ProtoMAML`, Meta-SGD, and ANIL for few-shot learning tasks.

use quantrs2_ml::autodiff::optimizers::Adam;
use quantrs2_ml::prelude::*;
use quantrs2_ml::qnn::QNNLayerType;
use scirs2_core::ndarray::{Array1, Array2};

fn main() -> Result<()> {
    println!("=== Quantum Meta-Learning Demo ===\n");

    // Step 1: Basic MAML demonstration
    println!("1. Model-Agnostic Meta-Learning (MAML)...");
    maml_demo()?;

    // Step 2: Reptile algorithm
    println!("\n2. Reptile Algorithm...");
    reptile_demo()?;

    // Step 3: ProtoMAML with prototypical learning
    println!("\n3. ProtoMAML (Prototypical MAML)...");
    protomaml_demo()?;

    // Step 4: Meta-SGD with learnable learning rates
    println!("\n4. Meta-SGD...");
    metasgd_demo()?;

    // Step 5: ANIL (Almost No Inner Loop)
    println!("\n5. ANIL Algorithm...");
    anil_demo()?;

    // Step 6: Continual meta-learning
    println!("\n6. Continual Meta-Learning...");
    continual_meta_learning_demo()?;

    // Step 7: Task distribution analysis
    println!("\n7. Task Distribution Analysis...");
    task_distribution_demo()?;

    println!("\n=== Meta-Learning Demo Complete ===");

    Ok(())
}

/// MAML demonstration
fn maml_demo() -> Result<()> {
    // Create quantum model
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 12 },
        QNNLayerType::EntanglementLayer {
            connectivity: "circular".to_string(),
        },
        QNNLayerType::VariationalLayer { num_params: 12 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let qnn = QuantumNeuralNetwork::new(layers, 4, 4, 3)?;

    // Create MAML learner
    let algorithm = MetaLearningAlgorithm::MAML {
        inner_steps: 5,
        inner_lr: 0.01,
        first_order: true, // Use first-order approximation for efficiency
    };

    let mut meta_learner = QuantumMetaLearner::new(algorithm, qnn);

    println!("   Created MAML meta-learner:");
    println!("   - Inner steps: 5");
    println!("   - Inner learning rate: 0.01");
    println!("   - Using first-order approximation");

    // Generate tasks
    let generator = TaskGenerator::new(4, 3);
    let tasks: Vec<MetaTask> = (0..20)
        .map(|_| generator.generate_rotation_task(30))
        .collect();

    // Meta-train
    println!("\n   Meta-training on 20 rotation tasks...");
    let mut optimizer = Adam::new(0.001);
    meta_learner.meta_train(&tasks, &mut optimizer, 50, 5)?;

    // Test adaptation
    let test_task = generator.generate_rotation_task(20);
    println!("\n   Testing adaptation to new task...");

    let adapted_params = meta_learner.adapt_to_task(&test_task)?;
    println!("   Successfully adapted to new task");
    println!(
        "   Parameter adaptation magnitude: {:.4}",
        (&adapted_params - meta_learner.meta_params())
            .mapv(f64::abs)
            .mean()
            .unwrap()
    );

    Ok(())
}

/// Reptile algorithm demonstration
fn reptile_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 2 },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "Pauli-Z".to_string(),
        },
    ];

    let qnn = QuantumNeuralNetwork::new(layers, 4, 2, 2)?;

    let algorithm = MetaLearningAlgorithm::Reptile {
        inner_steps: 10,
        inner_lr: 0.1,
    };

    let mut meta_learner = QuantumMetaLearner::new(algorithm, qnn);

    println!("   Created Reptile meta-learner:");
    println!("   - Inner steps: 10");
    println!("   - Inner learning rate: 0.1");

    // Generate sinusoid tasks
    let generator = TaskGenerator::new(2, 2);
    let tasks: Vec<MetaTask> = (0..15)
        .map(|_| generator.generate_sinusoid_task(40))
        .collect();

    println!("\n   Meta-training on 15 sinusoid tasks...");
    let mut optimizer = Adam::new(0.001);
    meta_learner.meta_train(&tasks, &mut optimizer, 30, 3)?;

    println!("   Reptile training complete");

    // Analyze task similarities
    println!("\n   Task parameter statistics:");
    for (i, task) in tasks.iter().take(3).enumerate() {
        if let Some(amplitude) = task.metadata.get("amplitude") {
            if let Some(phase) = task.metadata.get("phase") {
                println!("   Task {i}: amplitude={amplitude:.2}, phase={phase:.2}");
            }
        }
    }

    Ok(())
}

/// `ProtoMAML` demonstration
fn protomaml_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 8 },
        QNNLayerType::VariationalLayer { num_params: 16 },
        QNNLayerType::EntanglementLayer {
            connectivity: "full".to_string(),
        },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let qnn = QuantumNeuralNetwork::new(layers, 4, 8, 16)?;

    let algorithm = MetaLearningAlgorithm::ProtoMAML {
        inner_steps: 5,
        inner_lr: 0.01,
        proto_weight: 0.5, // Weight for prototype regularization
    };

    let mut meta_learner = QuantumMetaLearner::new(algorithm, qnn);

    println!("   Created ProtoMAML meta-learner:");
    println!("   - Combines MAML with prototypical networks");
    println!("   - Prototype weight: 0.5");

    // Generate classification tasks
    let generator = TaskGenerator::new(8, 4);
    let tasks: Vec<MetaTask> = (0..10)
        .map(|_| generator.generate_rotation_task(50))
        .collect();

    println!("\n   Meta-training on 4-way classification tasks...");
    let mut optimizer = Adam::new(0.001);
    meta_learner.meta_train(&tasks, &mut optimizer, 40, 2)?;

    println!("   ProtoMAML leverages both gradient-based and metric-based learning");

    Ok(())
}

/// Meta-SGD demonstration
fn metasgd_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 12 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "Pauli-XYZ".to_string(),
        },
    ];

    let qnn = QuantumNeuralNetwork::new(layers, 4, 4, 3)?;

    let algorithm = MetaLearningAlgorithm::MetaSGD { inner_steps: 3 };

    let mut meta_learner = QuantumMetaLearner::new(algorithm, qnn);

    println!("   Created Meta-SGD learner:");
    println!("   - Learns per-parameter learning rates");
    println!("   - Inner steps: 3");

    // Generate diverse tasks
    let generator = TaskGenerator::new(4, 3);
    let mut tasks = Vec::new();

    // Mix different task types
    for i in 0..12 {
        if i % 2 == 0 {
            tasks.push(generator.generate_rotation_task(30));
        } else {
            tasks.push(generator.generate_sinusoid_task(30));
        }
    }

    println!("\n   Meta-training on mixed task distribution...");
    let mut optimizer = Adam::new(0.0005);
    meta_learner.meta_train(&tasks, &mut optimizer, 50, 4)?;

    if let Some(lr) = meta_learner.per_param_lr() {
        println!("\n   Learned per-parameter learning rates:");
        println!(
            "   - Min LR: {:.4}",
            lr.iter().copied().fold(f64::INFINITY, f64::min)
        );
        println!(
            "   - Max LR: {:.4}",
            lr.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
        println!("   - Mean LR: {:.4}", lr.mean().unwrap());
    }

    Ok(())
}

/// ANIL demonstration
fn anil_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 6 },
        QNNLayerType::VariationalLayer { num_params: 12 },
        QNNLayerType::EntanglementLayer {
            connectivity: "circular".to_string(),
        },
        QNNLayerType::VariationalLayer { num_params: 12 },
        QNNLayerType::VariationalLayer { num_params: 6 }, // Final layer (adapted)
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let qnn = QuantumNeuralNetwork::new(layers, 4, 6, 2)?;

    let algorithm = MetaLearningAlgorithm::ANIL {
        inner_steps: 10,
        inner_lr: 0.1,
    };

    let mut meta_learner = QuantumMetaLearner::new(algorithm, qnn);

    println!("   Created ANIL (Almost No Inner Loop) learner:");
    println!("   - Only adapts final layer during inner loop");
    println!("   - More parameter efficient than MAML");
    println!("   - Inner steps: 10");

    // Generate binary classification tasks
    let generator = TaskGenerator::new(6, 2);
    let tasks: Vec<MetaTask> = (0..15)
        .map(|_| generator.generate_rotation_task(40))
        .collect();

    println!("\n   Meta-training on binary classification tasks...");
    let mut optimizer = Adam::new(0.001);
    meta_learner.meta_train(&tasks, &mut optimizer, 40, 5)?;

    println!("   ANIL reduces computational cost while maintaining performance");

    Ok(())
}

/// Continual meta-learning demonstration
fn continual_meta_learning_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let qnn = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    let algorithm = MetaLearningAlgorithm::Reptile {
        inner_steps: 5,
        inner_lr: 0.05,
    };

    let meta_learner = QuantumMetaLearner::new(algorithm, qnn);
    let mut continual_learner = ContinualMetaLearner::new(
        meta_learner,
        10,  // memory capacity
        0.3, // replay ratio
    );

    println!("   Created Continual Meta-Learner:");
    println!("   - Memory capacity: 10 tasks");
    println!("   - Replay ratio: 30%");

    // Generate sequence of tasks
    let generator = TaskGenerator::new(4, 2);

    println!("\n   Learning sequence of tasks...");
    for i in 0..20 {
        let task = if i < 10 {
            generator.generate_rotation_task(30)
        } else {
            generator.generate_sinusoid_task(30)
        };

        continual_learner.learn_task(task)?;

        if i % 5 == 4 {
            println!(
                "   Learned {} tasks, memory contains {} unique tasks",
                i + 1,
                continual_learner.memory_buffer_len()
            );
        }
    }

    println!("\n   Continual learning prevents catastrophic forgetting");

    Ok(())
}

/// Task distribution analysis
fn task_distribution_demo() -> Result<()> {
    println!("   Analyzing task distributions...\n");

    let generator = TaskGenerator::new(4, 3);

    // Generate multiple tasks and analyze their properties
    let mut rotation_tasks = Vec::new();
    let mut sinusoid_tasks = Vec::new();

    for _ in 0..50 {
        rotation_tasks.push(generator.generate_rotation_task(20));
        sinusoid_tasks.push(generator.generate_sinusoid_task(20));
    }

    // Analyze rotation tasks
    println!("   Rotation Task Distribution:");
    let angles: Vec<f64> = rotation_tasks
        .iter()
        .filter_map(|t| t.metadata.get("rotation_angle").copied())
        .collect();

    if !angles.is_empty() {
        let mean_angle = angles.iter().sum::<f64>() / angles.len() as f64;
        println!("   - Mean rotation angle: {mean_angle:.2} rad");
        println!(
            "   - Angle range: [{:.2}, {:.2}] rad",
            angles.iter().copied().fold(f64::INFINITY, f64::min),
            angles.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
    }

    // Analyze sinusoid tasks
    println!("\n   Sinusoid Task Distribution:");
    let amplitudes: Vec<f64> = sinusoid_tasks
        .iter()
        .filter_map(|t| t.metadata.get("amplitude").copied())
        .collect();

    if !amplitudes.is_empty() {
        let mean_amp = amplitudes.iter().sum::<f64>() / amplitudes.len() as f64;
        println!("   - Mean amplitude: {mean_amp:.2}");
        println!(
            "   - Amplitude range: [{:.2}, {:.2}]",
            amplitudes.iter().copied().fold(f64::INFINITY, f64::min),
            amplitudes.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
    }

    // Compare task complexities
    println!("\n   Task Complexity Comparison:");
    println!(
        "   - Rotation tasks: {} training samples each",
        rotation_tasks[0].train_data.len()
    );
    println!(
        "   - Sinusoid tasks: {} training samples each",
        sinusoid_tasks[0].train_data.len()
    );
    println!("   - Both use binary classification for simplicity");

    Ok(())
}
