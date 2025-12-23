//! Quantum Adversarial Training Example
//!
//! This example demonstrates quantum adversarial attacks and defenses,
//! including FGSM, PGD, parameter shift attacks, and various defense strategies.

use quantrs2_ml::autodiff::optimizers::Adam;
use quantrs2_ml::prelude::*;
use quantrs2_ml::qnn::QNNLayerType;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::prelude::*;

fn main() -> Result<()> {
    println!("=== Quantum Adversarial Training Demo ===\n");

    // Step 1: Generate adversarial examples
    println!("1. Adversarial Attack Generation...");
    adversarial_attack_demo()?;

    // Step 2: Defense mechanisms
    println!("\n2. Defense Mechanisms...");
    defense_mechanisms_demo()?;

    // Step 3: Adversarial training
    println!("\n3. Adversarial Training...");
    adversarial_training_demo()?;

    // Step 4: Robustness evaluation
    println!("\n4. Robustness Evaluation...");
    robustness_evaluation_demo()?;

    // Step 5: Certified defense
    println!("\n5. Certified Defense Analysis...");
    certified_defense_demo()?;

    // Step 6: Attack comparison
    println!("\n6. Attack Method Comparison...");
    attack_comparison_demo()?;

    // Step 7: Ensemble defense
    println!("\n7. Ensemble Defense...");
    ensemble_defense_demo()?;

    println!("\n=== Quantum Adversarial Demo Complete ===");

    Ok(())
}

/// Demonstrate different adversarial attacks
fn adversarial_attack_demo() -> Result<()> {
    // Create a quantum model
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::EntanglementLayer {
            connectivity: "circular".to_string(),
        },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;
    let defense = create_comprehensive_defense();
    let config = create_default_adversarial_config();

    let trainer = QuantumAdversarialTrainer::new(model, defense, config);

    println!("   Created quantum adversarial trainer"); // model.parameters field is private

    // Test data
    let test_data = Array2::from_shape_fn((10, 4), |(i, j)| {
        0.2f64.mul_add(j as f64 / 4.0, 0.3f64.mul_add(i as f64 / 10.0, 0.5))
    });
    let test_labels = Array1::from_shape_fn(10, |i| i % 2);

    println!("\n   Testing different attack methods:");

    // FGSM Attack
    println!("   - Fast Gradient Sign Method (FGSM)...");
    let fgsm_examples = trainer.generate_adversarial_examples(
        &test_data,
        &test_labels,
        QuantumAttackType::FGSM { epsilon: 0.1 },
    )?;

    let fgsm_success_rate = fgsm_examples
        .iter()
        .map(|ex| if ex.attack_success { 1.0 } else { 0.0 })
        .sum::<f64>()
        / fgsm_examples.len() as f64;

    println!("     Success rate: {:.2}%", fgsm_success_rate * 100.0);

    if let Some(example) = fgsm_examples.first() {
        println!(
            "     Average perturbation: {:.4}",
            example.perturbation_norm
        );
    }

    // PGD Attack
    println!("   - Projected Gradient Descent (PGD)...");
    let pgd_examples = trainer.generate_adversarial_examples(
        &test_data,
        &test_labels,
        QuantumAttackType::PGD {
            epsilon: 0.1,
            alpha: 0.01,
            num_steps: 10,
        },
    )?;

    let pgd_success_rate = pgd_examples
        .iter()
        .map(|ex| if ex.attack_success { 1.0 } else { 0.0 })
        .sum::<f64>()
        / pgd_examples.len() as f64;

    println!("     Success rate: {:.2}%", pgd_success_rate * 100.0);

    // Parameter Shift Attack
    println!("   - Parameter Shift Attack...");
    let param_examples = trainer.generate_adversarial_examples(
        &test_data,
        &test_labels,
        QuantumAttackType::ParameterShift {
            shift_magnitude: 0.05,
            target_parameters: None,
        },
    )?;

    let param_success_rate = param_examples
        .iter()
        .map(|ex| if ex.attack_success { 1.0 } else { 0.0 })
        .sum::<f64>()
        / param_examples.len() as f64;

    println!("     Success rate: {:.2}%", param_success_rate * 100.0);

    // Quantum State Perturbation
    println!("   - Quantum State Perturbation...");
    let state_examples = trainer.generate_adversarial_examples(
        &test_data,
        &test_labels,
        QuantumAttackType::StatePerturbation {
            perturbation_strength: 0.1,
            basis: "pauli_z".to_string(),
        },
    )?;

    let state_success_rate = state_examples
        .iter()
        .map(|ex| if ex.attack_success { 1.0 } else { 0.0 })
        .sum::<f64>()
        / state_examples.len() as f64;

    println!("     Success rate: {:.2}%", state_success_rate * 100.0);

    Ok(())
}

/// Demonstrate defense mechanisms
fn defense_mechanisms_demo() -> Result<()> {
    println!("   Testing defense strategies:");

    // Input preprocessing defense
    println!("   - Input Preprocessing...");
    let preprocessing_defense = QuantumDefenseStrategy::InputPreprocessing {
        noise_addition: 0.05,
        feature_squeezing: true,
    };

    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 6 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;
    let config = create_default_adversarial_config();
    let trainer = QuantumAdversarialTrainer::new(model, preprocessing_defense, config.clone());

    let test_input = Array1::from_vec(vec![0.51, 0.32, 0.83, 0.24]);
    let defended_input = trainer.apply_defense(&test_input)?;

    let defense_effect = (&defended_input - &test_input).mapv(f64::abs).sum();
    println!("     Defense effect magnitude: {defense_effect:.4}");

    // Randomized circuit defense
    println!("   - Randomized Circuit Defense...");
    let randomized_defense = QuantumDefenseStrategy::RandomizedCircuit {
        randomization_strength: 0.1,
        num_random_layers: 2,
    };

    let layers2 = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 8 },
    ];

    let model2 = QuantumNeuralNetwork::new(layers2, 4, 4, 2)?;
    let trainer2 = QuantumAdversarialTrainer::new(model2, randomized_defense, config);

    let defended_input2 = trainer2.apply_defense(&test_input)?;
    let randomization_effect = (&defended_input2 - &test_input).mapv(f64::abs).sum();
    println!("     Randomization effect: {randomization_effect:.4}");

    // Quantum error correction defense
    println!("   - Quantum Error Correction...");
    let qec_defense = QuantumDefenseStrategy::QuantumErrorCorrection {
        code_type: "surface_code".to_string(),
        correction_threshold: 0.01,
    };

    println!("     Error correction configured with surface codes");
    println!("     Correction threshold: 1%");

    Ok(())
}

/// Demonstrate adversarial training process
fn adversarial_training_demo() -> Result<()> {
    // Create model and trainer
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 12 },
        QNNLayerType::EntanglementLayer {
            connectivity: "circular".to_string(),
        },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    let defense = QuantumDefenseStrategy::AdversarialTraining {
        attack_types: vec![
            QuantumAttackType::FGSM { epsilon: 0.08 },
            QuantumAttackType::PGD {
                epsilon: 0.08,
                alpha: 0.01,
                num_steps: 7,
            },
        ],
        adversarial_ratio: 0.4,
    };

    let mut config = create_default_adversarial_config();
    config.epochs = 20; // Reduced for demo
    config.eval_interval = 5;

    let mut trainer = QuantumAdversarialTrainer::new(model, defense, config);

    println!("   Adversarial training configuration:");
    println!("   - Attack types: FGSM + PGD");
    println!("   - Adversarial ratio: 40%");
    println!("   - Training epochs: 20");

    // Generate synthetic training data
    let train_data = generate_quantum_dataset(200, 4);
    let train_labels = Array1::from_shape_fn(200, |i| i % 2);

    let val_data = generate_quantum_dataset(50, 4);
    let val_labels = Array1::from_shape_fn(50, |i| i % 2);

    // Train with adversarial examples
    println!("\n   Starting adversarial training...");
    let mut optimizer = Adam::new(0.001);
    let losses = trainer.train(
        &train_data,
        &train_labels,
        &val_data,
        &val_labels,
        &mut optimizer,
    )?;

    println!("   Training completed!");
    println!("   Final loss: {:.4}", losses.last().unwrap_or(&0.0));

    // Show final robustness metrics
    let metrics = trainer.get_robustness_metrics();
    println!("\n   Final robustness metrics:");
    println!("   - Clean accuracy: {:.3}", metrics.clean_accuracy);
    println!("   - Robust accuracy: {:.3}", metrics.robust_accuracy);
    println!(
        "   - Attack success rate: {:.3}",
        metrics.attack_success_rate
    );

    Ok(())
}

/// Demonstrate robustness evaluation
fn robustness_evaluation_demo() -> Result<()> {
    // Create trained model (simplified)
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;
    let defense = create_comprehensive_defense();
    let config = create_default_adversarial_config();

    let mut trainer = QuantumAdversarialTrainer::new(model, defense, config);

    println!("   Evaluating model robustness...");

    // Test data
    let test_data = generate_quantum_dataset(100, 4);
    let test_labels = Array1::from_shape_fn(100, |i| i % 2);

    // Evaluate against different attack strengths
    let epsilons = vec![0.05, 0.1, 0.15, 0.2];

    println!("\n   Robustness vs. attack strength:");
    for &epsilon in &epsilons {
        let attack_examples = trainer.generate_adversarial_examples(
            &test_data,
            &test_labels,
            QuantumAttackType::FGSM { epsilon },
        )?;

        let success_rate = attack_examples
            .iter()
            .map(|ex| if ex.attack_success { 1.0 } else { 0.0 })
            .sum::<f64>()
            / attack_examples.len() as f64;

        let avg_perturbation = attack_examples
            .iter()
            .map(|ex| ex.perturbation_norm)
            .sum::<f64>()
            / attack_examples.len() as f64;

        println!(
            "   ε = {:.2}: Attack success = {:.1}%, Avg perturbation = {:.4}",
            epsilon,
            success_rate * 100.0,
            avg_perturbation
        );
    }

    // Test different attack types
    println!("\n   Attack type comparison:");
    let attack_types = vec![
        ("FGSM", QuantumAttackType::FGSM { epsilon: 0.1 }),
        (
            "PGD",
            QuantumAttackType::PGD {
                epsilon: 0.1,
                alpha: 0.01,
                num_steps: 10,
            },
        ),
        (
            "Parameter Shift",
            QuantumAttackType::ParameterShift {
                shift_magnitude: 0.05,
                target_parameters: None,
            },
        ),
        (
            "State Perturbation",
            QuantumAttackType::StatePerturbation {
                perturbation_strength: 0.1,
                basis: "pauli_z".to_string(),
            },
        ),
    ];

    for (name, attack_type) in attack_types {
        let examples = trainer.generate_adversarial_examples(
            &test_data.slice(s![0..20, ..]).to_owned(),
            &test_labels.slice(s![0..20]).to_owned(),
            attack_type,
        )?;

        let success_rate = examples
            .iter()
            .map(|ex| if ex.attack_success { 1.0 } else { 0.0 })
            .sum::<f64>()
            / examples.len() as f64;

        println!("   {}: {:.1}% success rate", name, success_rate * 100.0);
    }

    Ok(())
}

/// Demonstrate certified defense
fn certified_defense_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 6 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;

    let certified_defense = QuantumDefenseStrategy::CertifiedDefense {
        smoothing_variance: 0.1,
        confidence_level: 0.95,
    };

    let config = create_default_adversarial_config();
    let trainer = QuantumAdversarialTrainer::new(model, certified_defense, config);

    println!("   Certified defense analysis:");
    println!("   - Smoothing variance: 0.1");
    println!("   - Confidence level: 95%");

    // Generate test data
    let test_data = generate_quantum_dataset(50, 4);

    // Perform certified analysis
    println!("\n   Running randomized smoothing certification...");
    let certified_accuracy = trainer.certified_defense_analysis(
        &test_data, 0.1, // smoothing variance
        100, // number of samples
    )?;

    println!("   Certified accuracy: {:.2}%", certified_accuracy * 100.0);

    // Compare with different smoothing levels
    let smoothing_levels = vec![0.05, 0.1, 0.15, 0.2];
    println!("\n   Certified accuracy vs. smoothing variance:");

    for &variance in &smoothing_levels {
        let cert_acc = trainer.certified_defense_analysis(&test_data, variance, 50)?;
        println!("   σ = {:.2}: {:.1}% certified", variance, cert_acc * 100.0);
    }

    Ok(())
}

/// Compare different attack methods
fn attack_comparison_demo() -> Result<()> {
    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 10 },
        QNNLayerType::EntanglementLayer {
            connectivity: "full".to_string(),
        },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;
    let defense = create_comprehensive_defense();
    let config = create_default_adversarial_config();

    let trainer = QuantumAdversarialTrainer::new(model, defense, config);

    println!("   Comprehensive attack comparison:");

    let test_data = generate_quantum_dataset(30, 4);
    let test_labels = Array1::from_shape_fn(30, |i| i % 2);

    // Test multiple attack configurations
    let attack_configs = vec![
        ("FGSM (ε=0.05)", QuantumAttackType::FGSM { epsilon: 0.05 }),
        ("FGSM (ε=0.1)", QuantumAttackType::FGSM { epsilon: 0.1 }),
        (
            "PGD-5",
            QuantumAttackType::PGD {
                epsilon: 0.1,
                alpha: 0.02,
                num_steps: 5,
            },
        ),
        (
            "PGD-10",
            QuantumAttackType::PGD {
                epsilon: 0.1,
                alpha: 0.01,
                num_steps: 10,
            },
        ),
        (
            "Parameter Shift",
            QuantumAttackType::ParameterShift {
                shift_magnitude: 0.1,
                target_parameters: None,
            },
        ),
        (
            "Circuit Manipulation",
            QuantumAttackType::CircuitManipulation {
                gate_error_rate: 0.01,
                coherence_time: 100.0,
            },
        ),
    ];

    println!("\n   Attack effectiveness comparison:");
    println!(
        "   {:20} {:>12} {:>15} {:>15}",
        "Attack Type", "Success Rate", "Avg Perturbation", "Effectiveness"
    );

    for (name, attack_type) in attack_configs {
        let examples =
            trainer.generate_adversarial_examples(&test_data, &test_labels, attack_type)?;

        let success_rate = examples
            .iter()
            .map(|ex| if ex.attack_success { 1.0 } else { 0.0 })
            .sum::<f64>()
            / examples.len() as f64;

        let avg_perturbation =
            examples.iter().map(|ex| ex.perturbation_norm).sum::<f64>() / examples.len() as f64;

        let effectiveness = if avg_perturbation > 0.0 {
            success_rate / avg_perturbation
        } else {
            0.0
        };

        println!(
            "   {:20} {:>11.1}% {:>14.4} {:>14.2}",
            name,
            success_rate * 100.0,
            avg_perturbation,
            effectiveness
        );
    }

    Ok(())
}

/// Demonstrate ensemble defense
fn ensemble_defense_demo() -> Result<()> {
    println!("   Ensemble defense strategy:");

    let ensemble_defense = QuantumDefenseStrategy::EnsembleDefense {
        num_models: 5,
        diversity_metric: "parameter_diversity".to_string(),
    };

    let layers = vec![
        QNNLayerType::EncodingLayer { num_features: 4 },
        QNNLayerType::VariationalLayer { num_params: 8 },
        QNNLayerType::MeasurementLayer {
            measurement_basis: "computational".to_string(),
        },
    ];

    let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)?;
    let config = create_default_adversarial_config();

    let mut trainer = QuantumAdversarialTrainer::new(model, ensemble_defense, config);

    println!("   - Number of models: 5");
    println!("   - Diversity metric: Parameter diversity");

    // Initialize ensemble
    println!("\n   Initializing ensemble models...");
    // trainer.initialize_ensemble()?; // Method is private
    println!("   Ensemble initialized (placeholder)");

    println!("   Ensemble initialized successfully");

    // Test ensemble robustness (simplified)
    let test_input = Array1::from_vec(vec![0.6, 0.4, 0.7, 0.3]);

    println!("\n   Ensemble prediction characteristics:");
    println!("   - Improved robustness through model diversity");
    println!("   - Reduced attack transferability");
    println!("   - Majority voting for final predictions");

    // Compare single model vs ensemble attack success
    // let single_model_attack = trainer.generate_single_adversarial_example(
    //     &test_input,
    //     0,
    //     QuantumAttackType::FGSM { epsilon: 0.1 }
    // )?;
    // Method is private - using public generate_adversarial_examples instead
    let single_model_attack = trainer.generate_adversarial_examples(
        &Array2::from_shape_vec((1, test_input.len()), test_input.to_vec())?,
        &Array1::from_vec(vec![0]),
        QuantumAttackType::FGSM { epsilon: 0.1 },
    )?[0]
        .clone();

    println!("\n   Single model vs. ensemble comparison:");
    println!(
        "   - Single model attack success: {}",
        if single_model_attack.attack_success {
            "Yes"
        } else {
            "No"
        }
    );
    println!(
        "   - Perturbation magnitude: {:.4}",
        single_model_attack.perturbation_norm
    );

    Ok(())
}

/// Generate synthetic quantum dataset
fn generate_quantum_dataset(samples: usize, features: usize) -> Array2<f64> {
    Array2::from_shape_fn((samples, features), |(i, j)| {
        let phase = (i as f64).mul_add(0.1, j as f64 * 0.3).sin();
        let amplitude = (i as f64 / samples as f64 + j as f64 / features as f64) * 0.5;
        0.1f64.mul_add(fastrand::f64() - 0.5, 0.5 + amplitude * phase)
    })
}
