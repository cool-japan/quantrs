use ndarray::{Array1, Array2};
use quantrs2_ml::hep::{CollisionEvent, HEPQuantumClassifier, ParticleFeatures, ParticleType};
use quantrs2_ml::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("Quantum High-Energy Physics Classification Example");
    println!("=================================================");

    // Create a quantum classifier for high-energy physics data
    let num_qubits = 8;
    let feature_dim = 8;
    let num_classes = 2;

    println!(
        "Creating HEP quantum classifier with {} qubits...",
        num_qubits
    );
    let mut classifier = HEPQuantumClassifier::new(
        num_qubits,
        feature_dim,
        num_classes,
        quantrs2_ml::hep::HEPEncodingMethod::HybridEncoding,
        vec!["background".to_string(), "higgs".to_string()],
    )?;

    // Generate synthetic training data
    println!("Generating synthetic training data...");
    let (training_particles, training_labels) = generate_synthetic_data(500);

    println!("Training quantum classifier...");
    let start = Instant::now();
    let metrics = classifier.train_on_particles(
        &training_particles,
        &training_labels,
        20,   // epochs
        0.05, // learning rate
    )?;

    println!("Training completed in {:.2?}", start.elapsed());
    println!("Final loss: {:.4}", metrics.final_loss);

    // Generate test data
    println!("Generating test data...");
    let (test_particles, test_labels) = generate_synthetic_data(100);

    // Evaluate classifier
    println!("Evaluating classifier...");
    // Convert test data to ndarray format
    let num_samples = test_particles.len();
    let mut test_features = Array2::zeros((num_samples, classifier.feature_dimension));
    let mut test_labels_array = Array1::zeros(num_samples);

    for (i, particle) in test_particles.iter().enumerate() {
        let features = classifier.extract_features(particle)?;
        for j in 0..features.len() {
            test_features[[i, j]] = features[j];
        }
        test_labels_array[i] = test_labels[i] as f64;
    }

    let evaluation = classifier.evaluate(&test_features, &test_labels_array)?;

    println!("Evaluation results:");
    println!("  Overall accuracy: {:.2}%", evaluation.accuracy * 100.0);

    println!("Class accuracies:");
    for (i, &acc) in evaluation.class_accuracies.iter().enumerate() {
        println!("  {}: {:.2}%", evaluation.class_labels[i], acc * 100.0);
    }

    // Create a test collision event
    println!("\nClassifying a test collision event...");
    let event = create_test_collision_event();

    // Run classification
    let classifications = classifier.classify_event(&event)?;

    println!("Event classification results:");
    for (i, (class, confidence)) in classifications.iter().enumerate() {
        println!(
            "  Particle {}: {} (confidence: {:.2})",
            i, class, confidence
        );
    }

    // Create a Higgs detector
    println!("\nCreating Higgs detector...");
    let higgs_detector = quantrs2_ml::hep::HiggsDetector::new(num_qubits)?;

    // Detect Higgs
    let higgs_detections = higgs_detector.detect_higgs(&event)?;

    println!("Higgs detection results:");
    let higgs_count = higgs_detections.iter().filter(|&&x| x).count();
    println!("  Found {} potential Higgs particles", higgs_count);

    Ok(())
}

// Generate synthetic particle data for training/testing
fn generate_synthetic_data(num_samples: usize) -> (Vec<ParticleFeatures>, Vec<usize>) {
    let mut particles = Vec::with_capacity(num_samples);
    let mut labels = Vec::with_capacity(num_samples);

    let particle_types = [
        ParticleType::Electron,
        ParticleType::Muon,
        ParticleType::Photon,
        ParticleType::Quark, // Changed from Proton which doesn't exist
        ParticleType::Higgs,
    ];

    for i in 0..num_samples {
        let is_higgs = i % 5 == 0;
        let particle_type = if is_higgs {
            ParticleType::Higgs
        } else {
            particle_types[i % 4]
        };

        // Generate synthetic four-momentum
        // Higgs particles have higher energy
        let energy_base = if is_higgs { 125.0 } else { 50.0 };
        let energy = energy_base + rand::random::<f64>() * 10.0;
        let px = (rand::random::<f64>() - 0.5) * 20.0;
        let py = (rand::random::<f64>() - 0.5) * 20.0;
        let pz = (rand::random::<f64>() - 0.5) * 50.0;

        // Create additional features
        let mut additional_features = Vec::with_capacity(3);
        for _ in 0..3 {
            additional_features.push(rand::random::<f64>());
        }

        // Create particle features
        let particle = ParticleFeatures {
            particle_type,
            four_momentum: [energy, px, py, pz],
            additional_features,
        };

        particles.push(particle);
        labels.push(if is_higgs { 1 } else { 0 });
    }

    (particles, labels)
}

// Create a test collision event
fn create_test_collision_event() -> CollisionEvent {
    let mut particles = Vec::new();

    // Add an electron
    particles.push(ParticleFeatures {
        particle_type: ParticleType::Electron,
        four_momentum: [50.5, 10.2, -15.7, 45.9],
        additional_features: vec![0.8, 0.2, 0.3],
    });

    // Add a positron
    particles.push(ParticleFeatures {
        particle_type: ParticleType::Electron, // Type is electron, but with opposite charge
        four_momentum: [50.2, -9.7, 14.3, -44.1],
        additional_features: vec![0.7, 0.3, 0.2],
    });

    // Add photons (potential Higgs decay products)
    particles.push(ParticleFeatures {
        particle_type: ParticleType::Photon,
        four_momentum: [62.8, 25.4, 30.1, 41.2],
        additional_features: vec![0.9, 0.1, 0.4],
    });

    particles.push(ParticleFeatures {
        particle_type: ParticleType::Photon,
        four_momentum: [63.2, -24.1, -29.5, -40.8],
        additional_features: vec![0.9, 0.1, 0.5],
    });

    // Create global event features
    let global_features = vec![230.0]; // Total energy

    CollisionEvent {
        particles,
        global_features,
        event_type: Some("potential_higgs".to_string()),
    }
}
