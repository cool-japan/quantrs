//! Quantum Adversarial Training
//!
//! This module implements adversarial training methods for quantum neural networks,
//! including adversarial attack generation, robust training procedures, and
//! defense mechanisms against quantum adversarial examples.

use crate::autodiff::optimizers::Optimizer;
use crate::error::{MLError, Result};
use crate::optimization::OptimizationMethod;
use crate::qnn::{QNNLayerType, QuantumNeuralNetwork};
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::gate::{
    single::{RotationX, RotationY, RotationZ},
    GateOp,
};
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Types of adversarial attacks for quantum models
#[derive(Debug, Clone)]
pub enum QuantumAttackType {
    /// Fast Gradient Sign Method adapted for quantum circuits
    FGSM { epsilon: f64 },

    /// Projected Gradient Descent attack
    PGD {
        epsilon: f64,
        alpha: f64,
        num_steps: usize,
    },

    /// Quantum Parameter Shift attack
    ParameterShift {
        shift_magnitude: f64,
        target_parameters: Option<Vec<usize>>,
    },

    /// Quantum State Perturbation attack
    StatePerturbation {
        perturbation_strength: f64,
        basis: String,
    },

    /// Quantum Circuit Manipulation attack
    CircuitManipulation {
        gate_error_rate: f64,
        coherence_time: f64,
    },

    /// Universal Adversarial Perturbation for quantum inputs
    UniversalPerturbation {
        perturbation_budget: f64,
        success_rate_threshold: f64,
    },
}

/// Defense strategies against quantum adversarial attacks
#[derive(Debug, Clone)]
pub enum QuantumDefenseStrategy {
    /// Adversarial training with generated examples
    AdversarialTraining {
        attack_types: Vec<QuantumAttackType>,
        adversarial_ratio: f64,
    },

    /// Quantum error correction as defense
    QuantumErrorCorrection {
        code_type: String,
        correction_threshold: f64,
    },

    /// Input preprocessing and sanitization
    InputPreprocessing {
        noise_addition: f64,
        feature_squeezing: bool,
    },

    /// Ensemble defense with multiple quantum models
    EnsembleDefense {
        num_models: usize,
        diversity_metric: String,
    },

    /// Certified defense with provable bounds
    CertifiedDefense {
        smoothing_variance: f64,
        confidence_level: f64,
    },

    /// Randomized circuit defense
    RandomizedCircuit {
        randomization_strength: f64,
        num_random_layers: usize,
    },
}

/// Adversarial example for quantum neural networks
#[derive(Debug, Clone)]
pub struct QuantumAdversarialExample {
    /// Original input
    pub original_input: Array1<f64>,

    /// Adversarial input
    pub adversarial_input: Array1<f64>,

    /// Original prediction
    pub original_prediction: Array1<f64>,

    /// Adversarial prediction
    pub adversarial_prediction: Array1<f64>,

    /// True label
    pub true_label: usize,

    /// Perturbation magnitude
    pub perturbation_norm: f64,

    /// Attack success (caused misclassification)
    pub attack_success: bool,

    /// Attack metadata
    pub metadata: HashMap<String, f64>,
}

/// Quantum adversarial trainer
pub struct QuantumAdversarialTrainer {
    /// Base quantum model
    model: QuantumNeuralNetwork,

    /// Defense strategy
    defense_strategy: QuantumDefenseStrategy,

    /// Training configuration
    config: AdversarialTrainingConfig,

    /// Attack history
    attack_history: Vec<QuantumAdversarialExample>,

    /// Robustness metrics
    robustness_metrics: RobustnessMetrics,

    /// Ensemble models (for ensemble defense)
    ensemble_models: Vec<QuantumNeuralNetwork>,
}

/// Configuration for adversarial training
#[derive(Debug, Clone)]
pub struct AdversarialTrainingConfig {
    /// Number of training epochs
    pub epochs: usize,

    /// Batch size
    pub batch_size: usize,

    /// Learning rate
    pub learning_rate: f64,

    /// Adversarial example generation frequency
    pub adversarial_frequency: usize,

    /// Maximum perturbation budget
    pub max_perturbation: f64,

    /// Robustness evaluation interval
    pub eval_interval: usize,

    /// Early stopping criteria
    pub early_stopping: Option<EarlyStoppingCriteria>,
}

/// Early stopping criteria for adversarial training
#[derive(Debug, Clone)]
pub struct EarlyStoppingCriteria {
    /// Minimum clean accuracy
    pub min_clean_accuracy: f64,

    /// Minimum robust accuracy
    pub min_robust_accuracy: f64,

    /// Patience (epochs without improvement)
    pub patience: usize,
}

/// Robustness metrics
#[derive(Debug, Clone)]
pub struct RobustnessMetrics {
    /// Clean accuracy (on unperturbed data)
    pub clean_accuracy: f64,

    /// Robust accuracy (on adversarial examples)
    pub robust_accuracy: f64,

    /// Average perturbation norm for successful attacks
    pub avg_perturbation_norm: f64,

    /// Attack success rate
    pub attack_success_rate: f64,

    /// Certified accuracy (for certified defenses)
    pub certified_accuracy: Option<f64>,

    /// Per-attack-type metrics
    pub per_attack_metrics: HashMap<String, AttackMetrics>,
}

/// Metrics for specific attack types
#[derive(Debug, Clone)]
pub struct AttackMetrics {
    /// Success rate
    pub success_rate: f64,

    /// Average perturbation
    pub avg_perturbation: f64,

    /// Average confidence drop
    pub avg_confidence_drop: f64,
}

impl QuantumAdversarialTrainer {
    /// Create a new quantum adversarial trainer
    pub fn new(
        model: QuantumNeuralNetwork,
        defense_strategy: QuantumDefenseStrategy,
        config: AdversarialTrainingConfig,
    ) -> Self {
        Self {
            model,
            defense_strategy,
            config,
            attack_history: Vec::new(),
            robustness_metrics: RobustnessMetrics {
                clean_accuracy: 0.0,
                robust_accuracy: 0.0,
                avg_perturbation_norm: 0.0,
                attack_success_rate: 0.0,
                certified_accuracy: None,
                per_attack_metrics: HashMap::new(),
            },
            ensemble_models: Vec::new(),
        }
    }

    /// Train the model with adversarial training
    pub fn train(
        &mut self,
        train_data: &Array2<f64>,
        train_labels: &Array1<usize>,
        val_data: &Array2<f64>,
        val_labels: &Array1<usize>,
        optimizer: &mut dyn Optimizer,
    ) -> Result<Vec<f64>> {
        println!("Starting quantum adversarial training...");

        let mut losses = Vec::new();
        let mut patience_counter = 0;
        let mut best_robust_accuracy = 0.0;

        // Initialize ensemble if needed
        self.initialize_ensemble()?;

        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0;
            let num_batches =
                (train_data.nrows() + self.config.batch_size - 1) / self.config.batch_size;

            for batch_idx in 0..num_batches {
                let batch_start = batch_idx * self.config.batch_size;
                let batch_end = (batch_start + self.config.batch_size).min(train_data.nrows());

                let batch_data = train_data.slice(s![batch_start..batch_end, ..]).to_owned();
                let batch_labels = train_labels.slice(s![batch_start..batch_end]).to_owned();

                // Generate adversarial examples if needed
                let (final_data, final_labels) = if epoch % self.config.adversarial_frequency == 0 {
                    self.generate_adversarial_batch(&batch_data, &batch_labels)?
                } else {
                    (batch_data, batch_labels)
                };

                // Compute loss and update model
                let batch_loss = self.train_batch(&final_data, &final_labels, optimizer)?;
                epoch_loss += batch_loss;
            }

            epoch_loss /= num_batches as f64;
            losses.push(epoch_loss);

            // Evaluate robustness periodically
            if epoch % self.config.eval_interval == 0 {
                self.evaluate_robustness(val_data, val_labels)?;

                println!(
                    "Epoch {}: Loss = {:.4}, Clean Acc = {:.3}, Robust Acc = {:.3}",
                    epoch,
                    epoch_loss,
                    self.robustness_metrics.clean_accuracy,
                    self.robustness_metrics.robust_accuracy
                );

                // Early stopping check
                if let Some(ref criteria) = self.config.early_stopping {
                    if self.robustness_metrics.robust_accuracy > best_robust_accuracy {
                        best_robust_accuracy = self.robustness_metrics.robust_accuracy;
                        patience_counter = 0;
                    } else {
                        patience_counter += 1;
                    }

                    if patience_counter >= criteria.patience {
                        println!("Early stopping triggered at epoch {}", epoch);
                        break;
                    }

                    if self.robustness_metrics.clean_accuracy < criteria.min_clean_accuracy
                        || self.robustness_metrics.robust_accuracy < criteria.min_robust_accuracy
                    {
                        println!("Minimum performance criteria not met, stopping training");
                        break;
                    }
                }
            }
        }

        // Final robustness evaluation
        self.evaluate_robustness(val_data, val_labels)?;

        Ok(losses)
    }

    /// Generate adversarial examples using specified attack
    pub fn generate_adversarial_examples(
        &self,
        data: &Array2<f64>,
        labels: &Array1<usize>,
        attack_type: QuantumAttackType,
    ) -> Result<Vec<QuantumAdversarialExample>> {
        let mut adversarial_examples = Vec::new();

        for (i, (input, &label)) in data.outer_iter().zip(labels.iter()).enumerate() {
            let adversarial_example = self.generate_single_adversarial_example(
                &input.to_owned(),
                label,
                attack_type.clone(),
            )?;

            adversarial_examples.push(adversarial_example);
        }

        Ok(adversarial_examples)
    }

    /// Generate a single adversarial example
    fn generate_single_adversarial_example(
        &self,
        input: &Array1<f64>,
        true_label: usize,
        attack_type: QuantumAttackType,
    ) -> Result<QuantumAdversarialExample> {
        // Get original prediction
        let original_prediction = self.model.forward(input)?;

        let adversarial_input = match attack_type {
            QuantumAttackType::FGSM { epsilon } => self.fgsm_attack(input, true_label, epsilon)?,
            QuantumAttackType::PGD {
                epsilon,
                alpha,
                num_steps,
            } => self.pgd_attack(input, true_label, epsilon, alpha, num_steps)?,
            QuantumAttackType::ParameterShift {
                shift_magnitude,
                target_parameters,
            } => self.parameter_shift_attack(input, shift_magnitude, target_parameters)?,
            QuantumAttackType::StatePerturbation {
                perturbation_strength,
                ref basis,
            } => self.state_perturbation_attack(input, perturbation_strength, basis)?,
            QuantumAttackType::CircuitManipulation {
                gate_error_rate,
                coherence_time,
            } => self.circuit_manipulation_attack(input, gate_error_rate, coherence_time)?,
            QuantumAttackType::UniversalPerturbation {
                perturbation_budget,
                success_rate_threshold,
            } => self.universal_perturbation_attack(input, perturbation_budget)?,
        };

        // Get adversarial prediction
        let adversarial_prediction = self.model.forward(&adversarial_input)?;

        // Compute perturbation norm
        let perturbation = &adversarial_input - input;
        let perturbation_norm = perturbation.mapv(|x| (x as f64).powi(2)).sum().sqrt();

        // Check attack success
        let original_class = original_prediction
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let adversarial_class = adversarial_prediction
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let attack_success = original_class != adversarial_class;

        Ok(QuantumAdversarialExample {
            original_input: input.clone(),
            adversarial_input,
            original_prediction,
            adversarial_prediction,
            true_label,
            perturbation_norm,
            attack_success,
            metadata: HashMap::new(),
        })
    }

    /// Fast Gradient Sign Method (FGSM) for quantum circuits
    fn fgsm_attack(
        &self,
        input: &Array1<f64>,
        true_label: usize,
        epsilon: f64,
    ) -> Result<Array1<f64>> {
        // Compute gradient of loss w.r.t. input
        let gradient = self.compute_input_gradient(input, true_label)?;

        // Apply FGSM perturbation
        let perturbation = gradient.mapv(|g| epsilon * g.signum());
        let adversarial_input = input + &perturbation;

        // Clip to valid range [0, 1] for quantum inputs
        Ok(adversarial_input.mapv(|x| x.max(0.0).min(1.0)))
    }

    /// Projected Gradient Descent (PGD) attack
    fn pgd_attack(
        &self,
        input: &Array1<f64>,
        true_label: usize,
        epsilon: f64,
        alpha: f64,
        num_steps: usize,
    ) -> Result<Array1<f64>> {
        let mut adversarial_input = input.clone();

        for _ in 0..num_steps {
            // Compute gradient
            let gradient = self.compute_input_gradient(&adversarial_input, true_label)?;

            // Take gradient step
            let perturbation = gradient.mapv(|g| alpha * g.signum());
            adversarial_input = &adversarial_input + &perturbation;

            // Project back to epsilon ball
            let total_perturbation = &adversarial_input - input;
            let perturbation_norm = total_perturbation.mapv(|x| (x as f64).powi(2)).sum().sqrt();

            if perturbation_norm > epsilon {
                let scaling = epsilon / perturbation_norm;
                adversarial_input = input + &(total_perturbation * scaling);
            }

            // Clip to valid range
            adversarial_input = adversarial_input.mapv(|x| x.max(0.0).min(1.0));
        }

        Ok(adversarial_input)
    }

    /// Parameter shift attack targeting quantum circuit parameters
    fn parameter_shift_attack(
        &self,
        input: &Array1<f64>,
        shift_magnitude: f64,
        target_parameters: Option<Vec<usize>>,
    ) -> Result<Array1<f64>> {
        // This attack modifies the input to exploit parameter shift rules
        let mut adversarial_input = input.clone();

        // Apply parameter shift-inspired perturbations
        for i in 0..adversarial_input.len() {
            if let Some(ref targets) = target_parameters {
                if !targets.contains(&i) {
                    continue;
                }
            }

            // Use parameter shift rule: f(x + π/2) - f(x - π/2)
            let shift = shift_magnitude * (PI / 2.0);
            adversarial_input[i] += shift * (2.0 * thread_rng().gen::<f64>() - 1.0);
        }

        Ok(adversarial_input.mapv(|x| x.max(0.0).min(1.0)))
    }

    /// Quantum state perturbation attack
    fn state_perturbation_attack(
        &self,
        input: &Array1<f64>,
        perturbation_strength: f64,
        basis: &str,
    ) -> Result<Array1<f64>> {
        let mut adversarial_input = input.clone();

        match basis {
            "pauli_x" => {
                // Apply X-basis perturbations
                for i in 0..adversarial_input.len() {
                    let angle = adversarial_input[i] * PI;
                    let perturbed_angle =
                        angle + perturbation_strength * (2.0 * thread_rng().gen::<f64>() - 1.0);
                    adversarial_input[i] = perturbed_angle / PI;
                }
            }
            "pauli_y" => {
                // Apply Y-basis perturbations
                for i in 0..adversarial_input.len() {
                    adversarial_input[i] +=
                        perturbation_strength * (2.0 * thread_rng().gen::<f64>() - 1.0);
                }
            }
            "pauli_z" | _ => {
                // Apply Z-basis perturbations (default)
                for i in 0..adversarial_input.len() {
                    let phase_shift =
                        perturbation_strength * (2.0 * thread_rng().gen::<f64>() - 1.0);
                    adversarial_input[i] =
                        (adversarial_input[i] + phase_shift / (2.0 * PI)).fract();
                }
            }
        }

        Ok(adversarial_input.mapv(|x| x.max(0.0).min(1.0)))
    }

    /// Circuit manipulation attack (simulating hardware errors)
    fn circuit_manipulation_attack(
        &self,
        input: &Array1<f64>,
        gate_error_rate: f64,
        coherence_time: f64,
    ) -> Result<Array1<f64>> {
        let mut adversarial_input = input.clone();

        // Simulate decoherence effects
        for i in 0..adversarial_input.len() {
            // Apply T1 decay
            let t1_factor = (-1.0 / coherence_time).exp();
            adversarial_input[i] *= t1_factor;

            // Add gate errors
            if thread_rng().gen::<f64>() < gate_error_rate {
                adversarial_input[i] += 0.1 * (2.0 * thread_rng().gen::<f64>() - 1.0);
            }
        }

        Ok(adversarial_input.mapv(|x| x.max(0.0).min(1.0)))
    }

    /// Universal adversarial perturbation attack
    fn universal_perturbation_attack(
        &self,
        input: &Array1<f64>,
        perturbation_budget: f64,
    ) -> Result<Array1<f64>> {
        // Apply a learned universal perturbation (simplified)
        let mut adversarial_input = input.clone();

        // Generate universal perturbation pattern
        for i in 0..adversarial_input.len() {
            let universal_component =
                perturbation_budget * (2.0 * PI * i as f64 / adversarial_input.len() as f64).sin();
            adversarial_input[i] += universal_component;
        }

        Ok(adversarial_input.mapv(|x| x.max(0.0).min(1.0)))
    }

    /// Compute gradient of loss with respect to input
    fn compute_input_gradient(
        &self,
        input: &Array1<f64>,
        true_label: usize,
    ) -> Result<Array1<f64>> {
        // Placeholder for gradient computation
        // In practice, this would use automatic differentiation
        let mut gradient = Array1::zeros(input.len());

        // Finite difference approximation
        let h = 1e-5;
        let original_output = self.model.forward(input)?;
        let original_loss = self.compute_loss(&original_output, true_label);

        for i in 0..input.len() {
            let mut perturbed_input = input.clone();
            perturbed_input[i] += h;

            let perturbed_output = self.model.forward(&perturbed_input)?;
            let perturbed_loss = self.compute_loss(&perturbed_output, true_label);

            gradient[i] = (perturbed_loss - original_loss) / h;
        }

        Ok(gradient)
    }

    /// Compute loss for a given output and true label
    fn compute_loss(&self, output: &Array1<f64>, true_label: usize) -> f64 {
        // Cross-entropy loss
        let predicted_prob = output[true_label].max(1e-10);
        -predicted_prob.ln()
    }

    /// Generate adversarial training batch
    fn generate_adversarial_batch(
        &self,
        data: &Array2<f64>,
        labels: &Array1<usize>,
    ) -> Result<(Array2<f64>, Array1<usize>)> {
        match &self.defense_strategy {
            QuantumDefenseStrategy::AdversarialTraining {
                attack_types,
                adversarial_ratio,
            } => {
                let num_adversarial = (data.nrows() as f64 * adversarial_ratio) as usize;
                let mut combined_data = data.clone();
                let mut combined_labels = labels.clone();

                // Generate adversarial examples
                for i in 0..num_adversarial {
                    let idx = i % data.nrows();
                    let input = data.row(idx).to_owned();
                    let label = labels[idx];

                    // Randomly select attack type
                    let attack_type = attack_types[fastrand::usize(0..attack_types.len())].clone();
                    let adversarial_example =
                        self.generate_single_adversarial_example(&input, label, attack_type)?;

                    // Add to batch (replace original example)
                    combined_data
                        .row_mut(idx)
                        .assign(&adversarial_example.adversarial_input);
                }

                Ok((combined_data, combined_labels))
            }
            _ => Ok((data.clone(), labels.clone())),
        }
    }

    /// Train on a single batch
    fn train_batch(
        &mut self,
        data: &Array2<f64>,
        labels: &Array1<usize>,
        optimizer: &mut dyn Optimizer,
    ) -> Result<f64> {
        // Simplified training step
        let mut total_loss = 0.0;

        for (input, &label) in data.outer_iter().zip(labels.iter()) {
            let output = self.model.forward(&input.to_owned())?;
            let loss = self.compute_loss(&output, label);
            total_loss += loss;

            // Compute gradients and update (simplified)
            // In practice, this would use proper backpropagation
        }

        Ok(total_loss / data.nrows() as f64)
    }

    /// Initialize ensemble for ensemble defense
    fn initialize_ensemble(&mut self) -> Result<()> {
        if let QuantumDefenseStrategy::EnsembleDefense { num_models, .. } = &self.defense_strategy {
            for _ in 0..*num_models {
                // Create model with slight variations
                let model = self.model.clone();
                self.ensemble_models.push(model);
            }
        }
        Ok(())
    }

    /// Evaluate robustness on validation set
    fn evaluate_robustness(
        &mut self,
        val_data: &Array2<f64>,
        val_labels: &Array1<usize>,
    ) -> Result<()> {
        let mut clean_correct = 0;
        let mut robust_correct = 0;
        let mut total_perturbation = 0.0;
        let mut successful_attacks = 0;

        // Test with different attack types
        let test_attacks = vec![
            QuantumAttackType::FGSM { epsilon: 0.1 },
            QuantumAttackType::PGD {
                epsilon: 0.1,
                alpha: 0.01,
                num_steps: 10,
            },
        ];

        for (input, &label) in val_data.outer_iter().zip(val_labels.iter()) {
            let input_owned = input.to_owned();

            // Clean accuracy
            let clean_output = self.model.forward(&input_owned)?;
            let clean_pred = clean_output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            if clean_pred == label {
                clean_correct += 1;
            }

            // Test robustness against attacks
            let mut robust_for_this_input = true;
            for attack_type in &test_attacks {
                let adversarial_example = self.generate_single_adversarial_example(
                    &input_owned,
                    label,
                    attack_type.clone(),
                )?;

                total_perturbation += adversarial_example.perturbation_norm;

                if adversarial_example.attack_success {
                    successful_attacks += 1;
                    robust_for_this_input = false;
                }
            }

            if robust_for_this_input {
                robust_correct += 1;
            }
        }

        let num_samples = val_data.nrows();
        let num_attack_tests = num_samples * test_attacks.len();

        self.robustness_metrics.clean_accuracy = clean_correct as f64 / num_samples as f64;
        self.robustness_metrics.robust_accuracy = robust_correct as f64 / num_samples as f64;
        self.robustness_metrics.avg_perturbation_norm =
            total_perturbation / num_attack_tests as f64;
        self.robustness_metrics.attack_success_rate =
            successful_attacks as f64 / num_attack_tests as f64;

        Ok(())
    }

    /// Apply defense strategy to input
    pub fn apply_defense(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        match &self.defense_strategy {
            QuantumDefenseStrategy::InputPreprocessing {
                noise_addition,
                feature_squeezing,
            } => {
                let mut defended_input = input.clone();

                // Add noise
                for i in 0..defended_input.len() {
                    defended_input[i] += noise_addition * (2.0 * thread_rng().gen::<f64>() - 1.0);
                }

                // Feature squeezing
                if *feature_squeezing {
                    defended_input = defended_input.mapv(|x| (x * 8.0).round() / 8.0);
                }

                Ok(defended_input.mapv(|x| x.max(0.0).min(1.0)))
            }
            QuantumDefenseStrategy::RandomizedCircuit {
                randomization_strength,
                ..
            } => {
                let mut defended_input = input.clone();

                // Add random perturbations to simulate circuit randomization
                for i in 0..defended_input.len() {
                    let random_shift =
                        randomization_strength * (2.0 * thread_rng().gen::<f64>() - 1.0);
                    defended_input[i] += random_shift;
                }

                Ok(defended_input.mapv(|x| x.max(0.0).min(1.0)))
            }
            _ => Ok(input.clone()),
        }
    }

    /// Get robustness metrics
    pub fn get_robustness_metrics(&self) -> &RobustnessMetrics {
        &self.robustness_metrics
    }

    /// Get attack history
    pub fn get_attack_history(&self) -> &[QuantumAdversarialExample] {
        &self.attack_history
    }

    /// Perform certified defense analysis
    pub fn certified_defense_analysis(
        &self,
        data: &Array2<f64>,
        smoothing_variance: f64,
        num_samples: usize,
    ) -> Result<f64> {
        let mut certified_correct = 0;

        for input in data.outer_iter() {
            let input_owned = input.to_owned();

            // Sample multiple noisy versions
            let mut predictions = Vec::new();
            for _ in 0..num_samples {
                let mut noisy_input = input_owned.clone();
                for i in 0..noisy_input.len() {
                    let noise = fastrand::f64() * smoothing_variance;
                    noisy_input[i] += noise;
                }

                let output = self.model.forward(&noisy_input)?;
                let pred = output
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                predictions.push(pred);
            }

            // Check if prediction is certified (majority vote is stable)
            let mut counts = vec![0; 10]; // Assume max 10 classes
            for &pred in &predictions {
                if pred < counts.len() {
                    counts[pred] += 1;
                }
            }

            let max_count = counts.iter().max().unwrap_or(&0);
            let certification_threshold = (num_samples as f64 * 0.6) as usize;

            if *max_count >= certification_threshold {
                certified_correct += 1;
            }
        }

        Ok(certified_correct as f64 / data.nrows() as f64)
    }
}

/// Helper function to create default adversarial training config
pub fn create_default_adversarial_config() -> AdversarialTrainingConfig {
    AdversarialTrainingConfig {
        epochs: 100,
        batch_size: 32,
        learning_rate: 0.001,
        adversarial_frequency: 2,
        max_perturbation: 0.1,
        eval_interval: 10,
        early_stopping: Some(EarlyStoppingCriteria {
            min_clean_accuracy: 0.7,
            min_robust_accuracy: 0.5,
            patience: 20,
        }),
    }
}

/// Helper function to create comprehensive defense strategy
pub fn create_comprehensive_defense() -> QuantumDefenseStrategy {
    QuantumDefenseStrategy::AdversarialTraining {
        attack_types: vec![
            QuantumAttackType::FGSM { epsilon: 0.1 },
            QuantumAttackType::PGD {
                epsilon: 0.1,
                alpha: 0.01,
                num_steps: 7,
            },
            QuantumAttackType::ParameterShift {
                shift_magnitude: 0.05,
                target_parameters: None,
            },
        ],
        adversarial_ratio: 0.5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qnn::QNNLayerType;

    #[test]
    fn test_adversarial_example_creation() {
        let original_input = Array1::from_vec(vec![0.5, 0.3, 0.8, 0.2]);
        let adversarial_input = Array1::from_vec(vec![0.6, 0.4, 0.7, 0.3]);
        let original_prediction = Array1::from_vec(vec![0.8, 0.2]);
        let adversarial_prediction = Array1::from_vec(vec![0.3, 0.7]);

        let perturbation = &adversarial_input - &original_input;
        let perturbation_norm = perturbation.mapv(|x| (x as f64).powi(2)).sum().sqrt();

        let example = QuantumAdversarialExample {
            original_input,
            adversarial_input,
            original_prediction,
            adversarial_prediction,
            true_label: 0,
            perturbation_norm,
            attack_success: true,
            metadata: HashMap::new(),
        };

        assert!(example.attack_success);
        assert!(example.perturbation_norm > 0.0);
    }

    #[test]
    fn test_fgsm_attack() {
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features: 4 },
            QNNLayerType::VariationalLayer { num_params: 8 },
            QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];

        let model = QuantumNeuralNetwork::new(layers, 4, 4, 2).unwrap();
        let defense = create_comprehensive_defense();
        let config = create_default_adversarial_config();

        let trainer = QuantumAdversarialTrainer::new(model, defense, config);

        let input = Array1::from_vec(vec![0.5, 0.3, 0.8, 0.2]);
        let adversarial_input = trainer.fgsm_attack(&input, 0, 0.1).unwrap();

        assert_eq!(adversarial_input.len(), input.len());

        // Check that perturbation exists
        let perturbation = &adversarial_input - &input;
        let perturbation_norm = perturbation.mapv(|x| (x as f64).powi(2)).sum().sqrt();
        assert!(perturbation_norm > 0.0);

        // Check that values are in valid range
        for &val in adversarial_input.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_defense_application() {
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features: 4 },
            QNNLayerType::VariationalLayer { num_params: 8 },
        ];

        let model = QuantumNeuralNetwork::new(layers, 4, 4, 2).unwrap();

        let defense = QuantumDefenseStrategy::InputPreprocessing {
            noise_addition: 0.05,
            feature_squeezing: true,
        };

        let config = create_default_adversarial_config();
        let trainer = QuantumAdversarialTrainer::new(model, defense, config);

        let input = Array1::from_vec(vec![0.51, 0.32, 0.83, 0.24]);
        let defended_input = trainer.apply_defense(&input).unwrap();

        assert_eq!(defended_input.len(), input.len());

        // Check that defense was applied (input changed)
        let difference = (&defended_input - &input).mapv(|x| x.abs()).sum();
        assert!(difference > 0.0);
    }

    #[test]
    fn test_robustness_metrics() {
        let metrics = RobustnessMetrics {
            clean_accuracy: 0.85,
            robust_accuracy: 0.65,
            avg_perturbation_norm: 0.12,
            attack_success_rate: 0.35,
            certified_accuracy: Some(0.55),
            per_attack_metrics: HashMap::new(),
        };

        assert_eq!(metrics.clean_accuracy, 0.85);
        assert_eq!(metrics.robust_accuracy, 0.65);
        assert!(metrics.robust_accuracy < metrics.clean_accuracy);
        assert!(metrics.attack_success_rate < 0.5);
    }
}
