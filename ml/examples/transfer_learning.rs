//! Quantum Transfer Learning Example
//!
//! This example demonstrates how to use pre-trained quantum models
//! and fine-tune them for new tasks with transfer learning.

use scirs2_core::ndarray::{Array1, Array2};
use quantrs2_ml::autodiff::optimizers::Adam;
use quantrs2_ml::prelude::*;
use quantrs2_ml::qnn::QNNLayerType;

fn main() -> Result<()> {
    println!("=== Quantum Transfer Learning Demo ===\n");

    // Step 1: Load a pre-trained model from the model zoo
    println!("1. Loading pre-trained image classifier...");
    let pretrained = QuantumModelZoo::get_image_classifier()?;

    println!("   Pre-trained model info:");
    println!("   - Task: {}", pretrained.task_description);
    println!(
        "   - Original accuracy: {:.2}%",
        pretrained
            .performance_metrics
            .get("accuracy")
            .unwrap_or(&0.0)
            * 100.0
    );
    println!("   - Number of qubits: {}", pretrained.qnn.num_qubits);

    // Step 2: Create new layers for the target task
    println!("\n2. Creating new layers for text classification task...");
    let new_layers = vec![
        QNNLayerType::VariationalLayer { num_params: 6 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "Pauli-Z".to_string(),
        },
    ];

    // Step 3: Initialize transfer learning with different strategies
    println!("\n3. Testing different transfer learning strategies:");

    // Strategy 1: Fine-tuning
    println!("\n   a) Fine-tuning strategy (train last 2 layers only)");
    let mut transfer_finetune = QuantumTransferLearning::new(
        pretrained.clone(),
        new_layers.clone(),
        TransferStrategy::FineTuning {
            num_trainable_layers: 2,
        },
    )?;

    // Strategy 2: Feature extraction
    println!("   b) Feature extraction strategy (freeze all pre-trained layers)");
    let transfer_feature = QuantumTransferLearning::new(
        pretrained.clone(),
        new_layers.clone(),
        TransferStrategy::FeatureExtraction,
    )?;

    // Strategy 3: Progressive unfreezing
    println!("   c) Progressive unfreezing (unfreeze one layer every 5 epochs)");
    let transfer_progressive = QuantumTransferLearning::new(
        pretrained.clone(),
        new_layers.clone(),
        TransferStrategy::ProgressiveUnfreezing { unfreeze_rate: 5 },
    )?;

    // Step 4: Generate synthetic training data for the new task
    println!("\n4. Generating synthetic training data...");
    let num_samples = 50;
    let num_features = 4;
    let training_data = Array2::from_shape_fn((num_samples, num_features), |(i, j)| {
        (i as f64 * 0.1 + j as f64 * 0.2).sin()
    });
    let labels = Array1::from_shape_fn(num_samples, |i| if i % 2 == 0 { 0.0 } else { 1.0 });

    // Step 5: Train with fine-tuning strategy
    println!("\n5. Training with fine-tuning strategy...");
    let mut optimizer = Adam::new(0.01);

    let result = transfer_finetune.train(
        &training_data,
        &labels,
        &mut optimizer,
        20, // epochs
        10, // batch_size
    )?;

    println!("   Training complete!");
    println!("   - Final loss: {:.4}", result.final_loss);
    println!("   - Accuracy: {:.2}%", result.accuracy * 100.0);

    // Step 6: Extract features using pre-trained layers
    println!("\n6. Extracting features from pre-trained layers...");
    let features = transfer_feature.extract_features(&training_data)?;
    println!("   Extracted feature dimensions: {:?}", features.dim());

    // Step 7: Demonstrate model zoo
    println!("\n7. Available pre-trained models in the zoo:");
    println!("   - Image classifier (4 qubits, MNIST subset)");
    println!("   - Chemistry model (6 qubits, molecular energy)");

    // Load chemistry model
    let chemistry_model = QuantumModelZoo::get_chemistry_model()?;
    println!("\n   Chemistry model info:");
    println!("   - Task: {}", chemistry_model.task_description);
    println!(
        "   - MAE: {:.4}",
        chemistry_model
            .performance_metrics
            .get("mae")
            .unwrap_or(&0.0)
    );
    println!(
        "   - R² score: {:.4}",
        chemistry_model
            .performance_metrics
            .get("r2_score")
            .unwrap_or(&0.0)
    );

    println!("\n=== Transfer Learning Demo Complete ===");

    Ok(())
}

// Helper function to visualize layer configurations
fn print_layer_configs(configs: &[LayerConfig]) {
    for (i, config) in configs.iter().enumerate() {
        println!(
            "   Layer {}: frozen={}, lr_multiplier={:.2}",
            i, config.frozen, config.learning_rate_multiplier
        );
    }
}
