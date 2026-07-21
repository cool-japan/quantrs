//! Quantum Generative Adversarial Networks (qGANs).
//!
//! Provides hybrid classical-quantum and fully-quantum GAN architectures.
//! The generator and discriminator can each be classical networks, quantum
//! circuits, or hybrid combinations, trained via adversarial min-max optimisation.

use crate::error::{MLError, Result};
use crate::qnn::QuantumNeuralNetwork;
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::fmt;

/// Type of generator to use in a quantum GAN
#[derive(Debug, Clone, Copy)]
pub enum GeneratorType {
    /// Pure classical generator
    Classical,

    /// Pure quantum generator
    QuantumOnly,

    /// Hybrid classical-quantum generator
    HybridClassicalQuantum,
}

/// Type of discriminator to use in a quantum GAN
#[derive(Debug, Clone, Copy)]
pub enum DiscriminatorType {
    /// Pure classical discriminator
    Classical,

    /// Pure quantum discriminator
    QuantumOnly,

    /// Hybrid with quantum feature extraction
    HybridQuantumFeatures,

    /// Hybrid with quantum decision function
    HybridQuantumDecision,
}

/// Training metrics for a GAN
#[derive(Debug, Clone)]
pub struct GANTrainingHistory {
    /// Generator loss at each epoch
    pub gen_losses: Vec<f64>,

    /// Discriminator loss at each epoch
    pub disc_losses: Vec<f64>,
}

/// Evaluation metrics for a GAN
#[derive(Debug, Clone)]
pub struct GANEvaluationMetrics {
    /// Accuracy of discriminator on real data
    pub real_accuracy: f64,

    /// Accuracy of discriminator on fake (generated) data
    pub fake_accuracy: f64,

    /// Overall discriminator accuracy
    pub overall_accuracy: f64,

    /// Jensen-Shannon divergence between real and generated distributions
    pub js_divergence: f64,
}

/// Trait for generator models
pub trait Generator {
    /// Generates samples from the latent space
    fn generate(&self, num_samples: usize) -> Result<Array2<f64>>;

    /// Generates samples with specific conditions
    fn generate_conditional(
        &self,
        num_samples: usize,
        conditions: &[(usize, f64)],
    ) -> Result<Array2<f64>>;

    /// Updates the generator based on discriminator feedback
    fn update(
        &mut self,
        latent_vectors: &Array2<f64>,
        discriminator_outputs: &Array1<f64>,
        learning_rate: f64,
    ) -> Result<f64>;
}

/// Trait for discriminator models
pub trait Discriminator {
    /// Discriminates between real and generated samples
    fn discriminate(&self, samples: &Array2<f64>) -> Result<Array1<f64>>;

    /// Predicts probabilities for a batch of samples
    fn predict_batch(&self, samples: &Array2<f64>) -> Result<Array1<f64>> {
        self.discriminate(samples)
    }

    /// Updates the discriminator based on real and generated samples
    fn update(
        &mut self,
        real_samples: &Array2<f64>,
        generated_samples: &Array2<f64>,
        learning_rate: f64,
    ) -> Result<f64>;
}

/// Physics-specific GAN implementations for particle physics simulations
pub mod physics_gan {
    use super::*;

    /// GAN model specialized for particle physics simulations
    pub struct ParticleGAN {
        /// The core quantum GAN implementation
        pub gan: QuantumGAN,

        /// Specialized parameters for physics simulations
        pub physics_params: PhysicsParameters,
    }

    /// Physics-specific parameters for the GAN
    #[derive(Debug, Clone)]
    pub struct PhysicsParameters {
        /// Energy scale for particle simulation
        pub energy_scale: f64,

        /// Momentum conservation factor
        pub momentum_conservation: f64,

        /// Whether to include quantum effects
        pub quantum_effects: bool,
    }

    impl ParticleGAN {
        /// Creates a new particle physics GAN
        pub fn new(
            num_qubits_gen: usize,
            num_qubits_disc: usize,
            latent_dim: usize,
            data_dim: usize,
        ) -> Result<Self> {
            // Create a standard quantum GAN
            let gan = QuantumGAN::new(
                num_qubits_gen,
                num_qubits_disc,
                latent_dim,
                data_dim,
                GeneratorType::HybridClassicalQuantum,
                DiscriminatorType::HybridQuantumFeatures,
            )?;

            // Default physics parameters
            let physics_params = PhysicsParameters {
                energy_scale: 100.0, // GeV
                momentum_conservation: 0.99,
                quantum_effects: true,
            };

            Ok(ParticleGAN {
                gan,
                physics_params,
            })
        }

        /// Trains the particle GAN on real particle data
        pub fn train(
            &mut self,
            particle_data: &Array2<f64>,
            epochs: usize,
        ) -> Result<&GANTrainingHistory> {
            // Use the underlying GAN's training method
            self.gan.train(
                particle_data,
                epochs,
                32,   // batch size
                0.01, // generator learning rate
                0.01, // discriminator learning rate
                1,    // discriminator steps
            )
        }

        /// Generates simulated particle data
        pub fn generate_particles(&self, num_particles: usize) -> Result<Array2<f64>> {
            // Extends basic generation with physics constraints
            let raw_data = self.gan.generate(num_particles)?;

            // In a full implementation, we would apply physics constraints here
            // such as momentum conservation, charge conservation, etc.

            Ok(raw_data)
        }
    }
}

/// Quantum Generator for GAN
#[derive(Debug, Clone)]
pub struct QuantumGenerator {
    /// Number of qubits
    num_qubits: usize,

    /// Dimension of latent space
    latent_dim: usize,

    /// Dimension of output data
    data_dim: usize,

    /// Type of generator
    generator_type: GeneratorType,

    /// Quantum neural network for generation
    qnn: QuantumNeuralNetwork,
}

impl QuantumGenerator {
    /// Creates a new quantum generator
    pub fn new(
        num_qubits: usize,
        latent_dim: usize,
        data_dim: usize,
        generator_type: GeneratorType,
    ) -> Result<Self> {
        // Create a QNN architecture suitable for generation
        let layers = vec![
            crate::qnn::QNNLayerType::EncodingLayer {
                num_features: latent_dim,
            },
            crate::qnn::QNNLayerType::VariationalLayer {
                num_params: 2 * num_qubits,
            },
            crate::qnn::QNNLayerType::EntanglementLayer {
                connectivity: "full".to_string(),
            },
            crate::qnn::QNNLayerType::VariationalLayer {
                num_params: 2 * num_qubits,
            },
            crate::qnn::QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];

        let qnn = QuantumNeuralNetwork::new(layers, num_qubits, latent_dim, data_dim)?;

        Ok(QuantumGenerator {
            num_qubits,
            latent_dim,
            data_dim,
            generator_type,
            qnn,
        })
    }
}

impl QuantumGenerator {
    /// Generate data samples from an explicit batch of latent vectors by
    /// evaluating the generator's quantum neural network.
    ///
    /// Each latent vector is encoded into the QNN circuit, simulated, and its
    /// per-feature Pauli expectation values (in `[-1, 1]`) are affinely mapped
    /// to the `[0, 1]` data range.
    fn generate_from_latent(&self, latent_vectors: &Array2<f64>) -> Result<Array2<f64>> {
        let num_samples = latent_vectors.nrows();
        let mut samples = Array2::zeros((num_samples, self.data_dim));
        for i in 0..num_samples {
            let latent = latent_vectors.row(i).to_owned();
            let output = self.qnn.forward(&latent)?;
            for j in 0..self.data_dim {
                let expectation = if j < output.len() { output[j] } else { 0.0 };
                samples[[i, j]] = (expectation + 1.0) * 0.5;
            }
        }
        Ok(samples)
    }

    /// Least-squares GAN adversarial loss `mean_i (D(G(z_i)) - 1)²` of the
    /// generator against `discriminator` on the latent batch `latent_vectors`.
    fn adversarial_loss(
        &self,
        latent_vectors: &Array2<f64>,
        discriminator: &QuantumDiscriminator,
    ) -> Result<f64> {
        let samples = self.generate_from_latent(latent_vectors)?;
        let outputs = discriminator.discriminate(&samples)?;
        let n = outputs.len();
        if n == 0 {
            return Ok(0.0);
        }
        let loss = outputs.iter().map(|&d| (d - 1.0) * (d - 1.0)).sum::<f64>();
        Ok(loss / n as f64)
    }

    /// Real adversarial update of the generator against a discriminator.
    ///
    /// Minimises the least-squares generator loss `mean_i (D(G(z_i)) - 1)²`
    /// with a central finite-difference gradient (the loss is a non-linear
    /// composition of two quantum circuits, so parameter-shift does not apply
    /// directly), updating the generator's parameters in place.  Returns the
    /// adversarial loss measured *before* the update.
    pub fn adversarial_update(
        &mut self,
        latent_vectors: &Array2<f64>,
        discriminator: &QuantumDiscriminator,
        learning_rate: f64,
    ) -> Result<f64> {
        if latent_vectors.nrows() == 0 {
            return Err(MLError::DataError(
                "adversarial update received an empty latent batch".to_string(),
            ));
        }
        let num_params = self.qnn.parameters.len();
        let epsilon = 1e-3;
        let base_loss = self.adversarial_loss(latent_vectors, discriminator)?;
        let original = self.qnn.parameters.clone();

        let mut gradient = Array1::<f64>::zeros(num_params);
        for j in 0..num_params {
            self.qnn.parameters[j] = original[j] + epsilon;
            let loss_plus = self.adversarial_loss(latent_vectors, discriminator)?;
            self.qnn.parameters[j] = original[j] - epsilon;
            let loss_minus = self.adversarial_loss(latent_vectors, discriminator)?;
            self.qnn.parameters[j] = original[j];
            gradient[j] = (loss_plus - loss_minus) / (2.0 * epsilon);
        }

        for j in 0..num_params {
            self.qnn.parameters[j] = original[j] - learning_rate * gradient[j];
        }
        Ok(base_loss)
    }

    /// Feature-matching loss `mean_i || G(z_i) - prototype ||²` used by the
    /// trait-level [`Generator::update`].
    fn feature_matching_loss(
        &self,
        latent_vectors: &Array2<f64>,
        prototype: &Array1<f64>,
    ) -> Result<f64> {
        let samples = self.generate_from_latent(latent_vectors)?;
        let n = samples.nrows();
        if n == 0 {
            return Ok(0.0);
        }
        let mut total = 0.0;
        for i in 0..n {
            for j in 0..self.data_dim {
                let diff = samples[[i, j]] - prototype[j];
                total += diff * diff;
            }
        }
        Ok(total / n as f64)
    }
}

impl Generator for QuantumGenerator {
    fn generate(&self, num_samples: usize) -> Result<Array2<f64>> {
        // Sample random latent vectors and push them through the quantum
        // generator network.
        let mut latent_vectors = Array2::zeros((num_samples, self.latent_dim));
        for i in 0..num_samples {
            for j in 0..self.latent_dim {
                latent_vectors[[i, j]] = thread_rng().random::<f64>() * 2.0 - 1.0;
            }
        }
        self.generate_from_latent(&latent_vectors)
    }

    fn generate_conditional(
        &self,
        num_samples: usize,
        conditions: &[(usize, f64)],
    ) -> Result<Array2<f64>> {
        // Generate samples
        let mut samples = self.generate(num_samples)?;

        // Apply conditions
        for &(feature_idx, value) in conditions {
            if feature_idx < self.data_dim {
                for i in 0..num_samples {
                    samples[[i, feature_idx]] = value;
                }
            }
        }

        Ok(samples)
    }

    fn update(
        &mut self,
        latent_vectors: &Array2<f64>,
        discriminator_outputs: &Array1<f64>,
        learning_rate: f64,
    ) -> Result<f64> {
        // Feature-matching generator update.
        //
        // The trait signature does not expose the discriminator model, so a
        // full adversarial gradient is not available here (use
        // [`QuantumGenerator::adversarial_update`] / [`QuantumGAN::train`] for
        // that).  Instead we form a realism-weighted prototype from the current
        // batch — samples the discriminator rated as more real receive more
        // weight — and take a real finite-difference gradient step that pulls
        // the generator's output toward that prototype.  Returns the
        // feature-matching loss measured before the update.
        let n = latent_vectors.nrows();
        if n == 0 {
            return Err(MLError::DataError(
                "generator update received an empty latent batch".to_string(),
            ));
        }

        let samples = self.generate_from_latent(latent_vectors)?;
        let weight_sum: f64 = discriminator_outputs.iter().map(|&d| d.max(0.0)).sum();

        let mut prototype = Array1::zeros(self.data_dim);
        if weight_sum > 1e-12 {
            for i in 0..n.min(discriminator_outputs.len()) {
                let weight = discriminator_outputs[i].max(0.0) / weight_sum;
                for j in 0..self.data_dim {
                    prototype[j] += weight * samples[[i, j]];
                }
            }
        } else {
            for i in 0..n {
                for j in 0..self.data_dim {
                    prototype[j] += samples[[i, j]] / n as f64;
                }
            }
        }

        let num_params = self.qnn.parameters.len();
        let epsilon = 1e-3;
        let base_loss = self.feature_matching_loss(latent_vectors, &prototype)?;
        let original = self.qnn.parameters.clone();

        let mut gradient = Array1::<f64>::zeros(num_params);
        for j in 0..num_params {
            self.qnn.parameters[j] = original[j] + epsilon;
            let loss_plus = self.feature_matching_loss(latent_vectors, &prototype)?;
            self.qnn.parameters[j] = original[j] - epsilon;
            let loss_minus = self.feature_matching_loss(latent_vectors, &prototype)?;
            self.qnn.parameters[j] = original[j];
            gradient[j] = (loss_plus - loss_minus) / (2.0 * epsilon);
        }

        for j in 0..num_params {
            self.qnn.parameters[j] = original[j] - learning_rate * gradient[j];
        }
        Ok(base_loss)
    }
}

/// Quantum Discriminator for GAN
#[derive(Debug, Clone)]
pub struct QuantumDiscriminator {
    /// Number of qubits
    num_qubits: usize,

    /// Dimension of input data
    data_dim: usize,

    /// Type of discriminator
    discriminator_type: DiscriminatorType,

    /// Quantum neural network for discrimination
    qnn: QuantumNeuralNetwork,
}

impl QuantumDiscriminator {
    /// Creates a new quantum discriminator
    pub fn new(
        num_qubits: usize,
        data_dim: usize,
        discriminator_type: DiscriminatorType,
    ) -> Result<Self> {
        // Create a QNN architecture suitable for discrimination
        let layers = vec![
            crate::qnn::QNNLayerType::EncodingLayer {
                num_features: data_dim,
            },
            crate::qnn::QNNLayerType::VariationalLayer {
                num_params: 2 * num_qubits,
            },
            crate::qnn::QNNLayerType::EntanglementLayer {
                connectivity: "full".to_string(),
            },
            crate::qnn::QNNLayerType::VariationalLayer {
                num_params: 2 * num_qubits,
            },
            crate::qnn::QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];

        let qnn = QuantumNeuralNetwork::new(
            layers, num_qubits, data_dim, 1, // Binary output (real or fake)
        )?;

        Ok(QuantumDiscriminator {
            num_qubits,
            data_dim,
            discriminator_type,
            qnn,
        })
    }
}

impl QuantumDiscriminator {
    /// Discriminate a single sample, returning the probability (in `[0, 1]`)
    /// that it is real.
    ///
    /// The sample is encoded into the discriminator's quantum neural network;
    /// its single Pauli-Z expectation output (in `[-1, 1]`) is affinely mapped
    /// to a probability.
    fn discriminate_one(&self, sample: &Array1<f64>) -> Result<f64> {
        let output = self.qnn.forward(sample)?;
        if output.is_empty() {
            return Err(MLError::MLOperationError(
                "discriminator QNN produced an empty output".to_string(),
            ));
        }
        Ok((output[0] + 1.0) * 0.5)
    }

    /// Least-squares discrimination loss
    /// `mean_real (D(x) - 1)² + mean_fake (D(x) - 0)²`.
    fn discrimination_loss(
        &self,
        real_samples: &Array2<f64>,
        generated_samples: &Array2<f64>,
    ) -> Result<f64> {
        let n_real = real_samples.nrows();
        let n_fake = generated_samples.nrows();

        let mut real_loss = 0.0;
        for i in 0..n_real {
            let d = self.discriminate_one(&real_samples.row(i).to_owned())?;
            real_loss += (d - 1.0) * (d - 1.0);
        }
        let mut fake_loss = 0.0;
        for i in 0..n_fake {
            let d = self.discriminate_one(&generated_samples.row(i).to_owned())?;
            fake_loss += d * d;
        }

        let mut loss = 0.0;
        if n_real > 0 {
            loss += real_loss / n_real as f64;
        }
        if n_fake > 0 {
            loss += fake_loss / n_fake as f64;
        }
        Ok(loss)
    }
}

impl Discriminator for QuantumDiscriminator {
    fn discriminate(&self, samples: &Array2<f64>) -> Result<Array1<f64>> {
        let num_samples = samples.nrows();
        let mut outputs = Array1::zeros(num_samples);
        for i in 0..num_samples {
            outputs[i] = self.discriminate_one(&samples.row(i).to_owned())?;
        }
        Ok(outputs)
    }

    fn update(
        &mut self,
        real_samples: &Array2<f64>,
        generated_samples: &Array2<f64>,
        learning_rate: f64,
    ) -> Result<f64> {
        // Least-squares GAN discriminator update via parameter-shift gradients.
        //
        // D(x) = (⟨Z⟩(x) + 1) / 2, so ∂D/∂θ = ½ · ∂⟨Z⟩/∂θ where ∂⟨Z⟩/∂θ is the
        // exact parameter-shift gradient.  The least-squares loss gradient for a
        // real sample (target 1) is (D - 1)·∂⟨Z⟩/∂θ and for a fake sample
        // (target 0) is D·∂⟨Z⟩/∂θ, averaged within each class.
        let n_real = real_samples.nrows();
        let n_fake = generated_samples.nrows();
        let num_params = self.qnn.parameters.len();
        let mut gradient = Array1::<f64>::zeros(num_params);

        for i in 0..n_real {
            let x = real_samples.row(i).to_owned();
            let d = self.discriminate_one(&x)?;
            let d_expectation = self.qnn.output_component_gradient(&x, 0)?;
            let coeff = (d - 1.0) / n_real as f64;
            for j in 0..num_params {
                gradient[j] += coeff * d_expectation[j];
            }
        }
        for i in 0..n_fake {
            let x = generated_samples.row(i).to_owned();
            let d = self.discriminate_one(&x)?;
            let d_expectation = self.qnn.output_component_gradient(&x, 0)?;
            let coeff = d / n_fake as f64;
            for j in 0..num_params {
                gradient[j] += coeff * d_expectation[j];
            }
        }

        for j in 0..num_params {
            self.qnn.parameters[j] -= learning_rate * gradient[j];
        }

        // Report the loss after the update so the training history reflects the
        // discriminator's real progress.
        self.discrimination_loss(real_samples, generated_samples)
    }
}

/// Quantum Generative Adversarial Network
#[derive(Debug, Clone)]
pub struct QuantumGAN {
    /// Generator model
    pub generator: QuantumGenerator,

    /// Discriminator model
    pub discriminator: QuantumDiscriminator,

    /// Training history
    pub training_history: GANTrainingHistory,
}

impl QuantumGAN {
    /// Creates a new quantum GAN
    pub fn new(
        num_qubits_gen: usize,
        num_qubits_disc: usize,
        latent_dim: usize,
        data_dim: usize,
        generator_type: GeneratorType,
        discriminator_type: DiscriminatorType,
    ) -> Result<Self> {
        let generator =
            QuantumGenerator::new(num_qubits_gen, latent_dim, data_dim, generator_type)?;

        let discriminator =
            QuantumDiscriminator::new(num_qubits_disc, data_dim, discriminator_type)?;

        let training_history = GANTrainingHistory {
            gen_losses: Vec::new(),
            disc_losses: Vec::new(),
        };

        Ok(QuantumGAN {
            generator,
            discriminator,
            training_history,
        })
    }

    /// Trains the GAN on a dataset
    pub fn train(
        &mut self,
        real_data: &Array2<f64>,
        epochs: usize,
        batch_size: usize,
        gen_learning_rate: f64,
        disc_learning_rate: f64,
        disc_steps: usize,
    ) -> Result<&GANTrainingHistory> {
        let mut gen_losses = Vec::with_capacity(epochs);
        let mut disc_losses = Vec::with_capacity(epochs);

        for _epoch in 0..epochs {
            // Train discriminator for several steps
            let mut disc_loss_sum = 0.0;
            for _step in 0..disc_steps {
                // Generate fake samples
                let fake_samples = self.generator.generate(batch_size)?;

                // Sample real data (random batch)
                let real_batch = sample_batch(real_data, batch_size)?;

                // Update discriminator
                let disc_loss =
                    self.discriminator
                        .update(&real_batch, &fake_samples, disc_learning_rate)?;
                disc_loss_sum += disc_loss;
            }
            let avg_disc_loss = disc_loss_sum / disc_steps as f64;

            // Train generator against the current discriminator with real
            // random latent vectors and a genuine adversarial gradient.
            let latent_dim = self.generator.latent_dim;
            let mut latent_vectors = Array2::zeros((batch_size, latent_dim));
            for i in 0..batch_size {
                for j in 0..latent_dim {
                    latent_vectors[[i, j]] = thread_rng().random::<f64>() * 2.0 - 1.0;
                }
            }
            let gen_loss = self.generator.adversarial_update(
                &latent_vectors,
                &self.discriminator,
                gen_learning_rate,
            )?;

            // Record losses
            gen_losses.push(gen_loss);
            disc_losses.push(avg_disc_loss);
        }

        self.training_history = GANTrainingHistory {
            gen_losses,
            disc_losses,
        };

        Ok(&self.training_history)
    }

    /// Generates samples from the trained generator
    pub fn generate(&self, num_samples: usize) -> Result<Array2<f64>> {
        self.generator.generate(num_samples)
    }

    /// Generates samples with specific conditions
    pub fn generate_conditional(
        &self,
        num_samples: usize,
        conditions: &[(usize, f64)],
    ) -> Result<Array2<f64>> {
        self.generator.generate_conditional(num_samples, conditions)
    }

    /// Evaluates the GAN model
    pub fn evaluate(
        &self,
        real_data: &Array2<f64>,
        num_samples: usize,
    ) -> Result<GANEvaluationMetrics> {
        // Generate fake samples
        let fake_samples = self.generate(num_samples)?;

        // Evaluate discriminator on real data
        let real_preds = self.discriminator.predict_batch(real_data)?;
        let real_correct = real_preds.iter().filter(|&&p| p > 0.5).count();
        let real_accuracy = real_correct as f64 / real_preds.len() as f64;

        // Evaluate discriminator on fake data
        let fake_preds = self.discriminator.predict_batch(&fake_samples)?;
        let fake_correct = fake_preds.iter().filter(|&&p| p < 0.5).count();
        let fake_accuracy = fake_correct as f64 / fake_preds.len() as f64;

        // Overall accuracy
        let overall_correct = real_correct + fake_correct;
        let overall_total = real_preds.len() + fake_preds.len();
        let overall_accuracy = overall_correct as f64 / overall_total as f64;

        // Calculate Jensen-Shannon divergence between real and fake data distributions
        // This is a simplified placeholder calculation
        let js_divergence = calculate_js_divergence(real_data, &fake_samples)?;

        Ok(GANEvaluationMetrics {
            real_accuracy,
            fake_accuracy,
            overall_accuracy,
            js_divergence,
        })
    }
}

/// Calculate Jensen-Shannon divergence between two datasets using histogram estimation.
///
/// For each column (feature dimension), estimates the probability distributions
/// with a fixed-bin histogram, then computes JS = 0.5 * KL(p||m) + 0.5 * KL(q||m)
/// where m = (p + q) / 2.  Results are averaged across columns.
fn calculate_js_divergence(data1: &Array2<f64>, data2: &Array2<f64>) -> Result<f64> {
    if data1.ncols() == 0 || data1.nrows() == 0 || data2.nrows() == 0 {
        return Ok(0.0);
    }

    let n_bins: usize = 20;
    let n_cols = data1.ncols().min(data2.ncols());
    let mut total_js = 0.0;

    for col in 0..n_cols {
        let col1: Vec<f64> = data1.column(col).to_vec();
        let col2: Vec<f64> = data2.column(col).to_vec();

        let min_val = col1
            .iter()
            .chain(col2.iter())
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_val = col1
            .iter()
            .chain(col2.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-14 {
            // All values identical across both datasets → JS divergence is 0
            continue;
        }

        let bin_width = (max_val - min_val) / n_bins as f64;
        let mut hist1 = vec![0.0f64; n_bins];
        let mut hist2 = vec![0.0f64; n_bins];

        for &v in &col1 {
            let bin = ((v - min_val) / bin_width) as usize;
            let bin = bin.min(n_bins - 1);
            hist1[bin] += 1.0;
        }
        for &v in &col2 {
            let bin = ((v - min_val) / bin_width) as usize;
            let bin = bin.min(n_bins - 1);
            hist2[bin] += 1.0;
        }

        let n1 = col1.len() as f64;
        let n2 = col2.len() as f64;
        for i in 0..n_bins {
            hist1[i] /= n1;
            hist2[i] /= n2;
        }

        // JS = 0.5 * KL(p || m) + 0.5 * KL(q || m),  m = (p + q) / 2
        let mut js = 0.0f64;
        for i in 0..n_bins {
            let p = hist1[i];
            let q = hist2[i];
            let m = (p + q) * 0.5;
            if m > 1e-14 {
                if p > 1e-14 {
                    js += 0.5 * p * (p / m).ln();
                }
                if q > 1e-14 {
                    js += 0.5 * q * (q / m).ln();
                }
            }
        }
        total_js += js;
    }

    Ok(if n_cols > 0 {
        total_js / n_cols as f64
    } else {
        0.0
    })
}

// Helper function to sample a random batch from a dataset
fn sample_batch(data: &Array2<f64>, batch_size: usize) -> Result<Array2<f64>> {
    let num_samples = data.nrows();
    let mut batch = Array2::zeros((batch_size.min(num_samples), data.ncols()));

    for i in 0..batch_size.min(num_samples) {
        let idx = fastrand::usize(0..num_samples);
        batch.row_mut(i).assign(&data.row(idx));
    }

    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_js_divergence_identical() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 1.0, 0.5, 0.5, 0.2, 0.8, 0.7, 0.3])
            .expect("array creation failed");
        let js = calculate_js_divergence(&data, &data).expect("divergence failed");
        assert!(js < 0.01, "JS(p,p) should be ≈0, got {js}");
    }

    #[test]
    fn test_js_divergence_bounded() {
        let data1 =
            Array2::from_shape_vec((4, 1), vec![0.0, 0.0, 0.0, 0.0]).expect("array creation");
        let data2 =
            Array2::from_shape_vec((4, 1), vec![1.0, 1.0, 1.0, 1.0]).expect("array creation");
        let js = calculate_js_divergence(&data1, &data2).expect("divergence failed");
        assert!(js >= 0.0 && js <= 1.0, "JS should be in [0, 1], got {js}");
    }
}

impl fmt::Display for GeneratorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeneratorType::Classical => write!(f, "Classical"),
            GeneratorType::QuantumOnly => write!(f, "Quantum Only"),
            GeneratorType::HybridClassicalQuantum => write!(f, "Hybrid Classical-Quantum"),
        }
    }
}

impl fmt::Display for DiscriminatorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiscriminatorType::Classical => write!(f, "Classical"),
            DiscriminatorType::QuantumOnly => write!(f, "Quantum Only"),
            DiscriminatorType::HybridQuantumFeatures => write!(f, "Hybrid with Quantum Features"),
            DiscriminatorType::HybridQuantumDecision => write!(f, "Hybrid with Quantum Decision"),
        }
    }
}
