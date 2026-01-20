use scirs2_core::ndarray::{Array1, Array2};
use quantrs2_ml::prelude::*;
use quantrs2_ml::hep::{HEPQuantumClassifier, HEPEncodingMethod, ParticleFeatures, ParticleType};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("QuantRS2 High-Energy Physics Classification Example");
    println!("=================================================");

    // Create a dataset of particle features
    // This simulates data from a particle detector at CERN, etc.
    println!("Generating synthetic particle collision data...");

    let (train_data, train_labels, test_data, test_labels) = generate_collision_dataset(500, 100)?;

    println!("Dataset created:");
    println!("  Training samples: {}", train_data.shape()[0]);
    println!("  Testing samples: {}", test_data.shape()[0]);
    println!("  Features per sample: {}", train_data.shape()[1]);

    // Create a quantum classifier for high-energy physics
    let num_qubits = 8;
    let feature_dim = train_data.shape()[1];
    let num_classes = 2;

    println!("\nCreating HEP quantum classifier:");
    println!("  Qubits: {}", num_qubits);
    println!("  Feature dimension: {}", feature_dim);
    println!("  Encoding method: HybridEncoding");

    let mut classifier = HEPQuantumClassifier::new(
        num_qubits,
        feature_dim,
        num_classes,
        HEPEncodingMethod::HybridEncoding,
        vec!["background".to_string(), "signal".to_string()],
    )?;

    // Train the classifier
    println!("\nTraining classifier...");
    let start = Instant::now();

    let training_result = classifier.train(&train_data, &train_labels, 50, 0.01)?;

    println!("Training completed in {:.2?}", start.elapsed());
    println!("Final training loss: {:.6}", training_result.final_loss);
    println!("Training accuracy: {:.2}%", training_result.accuracy * 100.0);

    // Evaluate on test data
    println!("\nEvaluating on test data...");
    let metrics = classifier.evaluate(&test_data, &test_labels)?;

    println!("Test accuracy: {:.2}%", metrics.accuracy * 100.0);
    println!("Precision: {:.4}", metrics.precision);
    println!("Recall: {:.4}", metrics.recall);
    println!("F1 score: {:.4}", metrics.f1_score);
    println!("Area under ROC curve: {:.4}", metrics.auc);

    // Make some predictions
    println!("\nExample predictions:");
    for i in 0..5 {
        let features = test_data.slice(scirs2_core::ndarray::s![i, ..]).to_owned();
        let (prediction, confidence) = classifier.predict(&features)?;
        let true_label = test_labels[i];

        println!("Sample {}: predicted {} with {:.2}% confidence (true label: {})",
                 i, prediction, confidence * 100.0,
                 if true_label > 0.5 { "signal" } else { "background" });
    }

    // Feature importance analysis
    println!("\nFeature importance analysis:");
    let importance = classifier.feature_importance()?;

    for (i, &imp) in importance.iter().enumerate().take(5) {
        println!("Feature {}: {:.4}", i, imp);
    }

    println!("\nHighest importance feature: {}", importance.iter()
             .enumerate()
             .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
             .map(|(idx, _)| idx)
             .unwrap());

    // Higgs detection example
    println!("\nSpecialized Higgs boson detection example:");
    let higgs_detector = quantrs2_ml::hep::HiggsDetector::new(num_qubits)?;

    // Generate a simulated Higgs event
    let higgs_features = create_higgs_particle()?;
    let background_features = create_background_particle()?;

    // Detect particles
    let higgs_score = higgs_detector.score_particle(&higgs_features)?;
    let background_score = higgs_detector.score_particle(&background_features)?;

    println!("Higgs candidate score: {:.4}", higgs_score);
    println!("Background particle score: {:.4}", background_score);

    if higgs_score > 0.8 {
        println!("✓ Successfully detected Higgs boson candidate!");
    }

    println!("\nHEP classification example completed successfully!");
    Ok(())
}

// Helper function to generate synthetic HEP data
fn generate_collision_dataset(train_size: usize, test_size: usize)
    -> Result<(Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>), Box<dyn std::error::Error>> {

    let feature_dim = 10;
    let mut train_data = Array2::zeros((train_size, feature_dim));
    let mut train_labels = Array1::zeros(train_size);
    let mut test_data = Array2::zeros((test_size, feature_dim));
    let mut test_labels = Array1::zeros(test_size);

    // Generate synthetic training data
    for i in 0..train_size {
        // Generate signal or background (50/50 split)
        let is_signal = i % 2 == 0;
        train_labels[i] = if is_signal { 1.0 } else { 0.0 };

        // Generate synthetic features
        for j in 0..feature_dim {
            if is_signal {
                // Signal particles have distinctive patterns
                train_data[[i, j]] = 0.5 + 0.3 * rand::random::<f64>() * (1.0 + (j as f64).cos());
            } else {
                // Background particles have more random distribution
                train_data[[i, j]] = rand::random::<f64>() * 0.8;
            }
        }

        // Add a small amount of noise
        if rand::random::<f64>() < 0.05 {
            train_data[[i, rand::random::<usize>() % feature_dim]] += 0.2;
        }
    }

    // Generate synthetic test data (similar process)
    for i in 0..test_size {
        let is_signal = i % 2 == 0;
        test_labels[i] = if is_signal { 1.0 } else { 0.0 };

        for j in 0..feature_dim {
            if is_signal {
                test_data[[i, j]] = 0.5 + 0.3 * rand::random::<f64>() * (1.0 + (j as f64).cos());
            } else {
                test_data[[i, j]] = rand::random::<f64>() * 0.8;
            }
        }

        if rand::random::<f64>() < 0.05 {
            test_data[[i, rand::random::<usize>() % feature_dim]] += 0.2;
        }
    }

    Ok((train_data, train_labels, test_data, test_labels))
}

// Helper function to create a Higgs particle
fn create_higgs_particle() -> Result<ParticleFeatures, Box<dyn std::error::Error>> {
    Ok(ParticleFeatures {
        particle_type: ParticleType::Higgs,
        four_momentum: [125.3, 5.2, -7.1, 12.5], // Mass of ~125 GeV
        additional_features: vec![0.9, 0.8, 0.7, 0.8, 0.9, 0.7],
    })
}

// Helper function to create a background particle
fn create_background_particle() -> Result<ParticleFeatures, Box<dyn std::error::Error>> {
    Ok(ParticleFeatures {
        particle_type: ParticleType::Other,
        four_momentum: [60.2, 25.4, 30.1, 41.2], // Different mass profile
        additional_features: vec![0.2, 0.3, 0.1, 0.4, 0.2, 0.3],
    })
}