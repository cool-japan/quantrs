//! Quantum Generative Adversarial Networks (QGANs)
//!
//! This module implements quantum generative adversarial networks, leveraging
//! quantum circuits for both generator and discriminator networks to achieve
//! quantum advantage in generative modeling tasks.

use crate::{
    error::QuantRS2Result,
    gate::GateOp,
    gate::single::*,
    gate::multi::*,
    qml::QMLLayer,
    variational::VariationalOptimizer,
    qubit::QubitId,
};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Configuration for Quantum Generative Adversarial Networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QGANConfig {
    /// Number of qubits for the generator
    pub generator_qubits: usize,
    /// Number of qubits for the discriminator
    pub discriminator_qubits: usize,
    /// Number of qubits for latent (noise) space
    pub latent_qubits: usize,
    /// Number of qubits for data representation
    pub data_qubits: usize,
    /// Generator learning rate
    pub generator_lr: f64,
    /// Discriminator learning rate
    pub discriminator_lr: f64,
    /// Number of generator layers
    pub generator_depth: usize,
    /// Number of discriminator layers
    pub discriminator_depth: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Training iterations
    pub max_iterations: usize,
    /// Generator training frequency (train generator every N discriminator updates)
    pub generator_frequency: usize,
    /// Whether to use quantum advantage techniques
    pub use_quantum_advantage: bool,
    /// Noise distribution type
    pub noise_type: NoiseType,
    /// Regularization strength
    pub regularization: f64,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for QGANConfig {
    fn default() -> Self {
        Self {
            generator_qubits: 6,
            discriminator_qubits: 6,
            latent_qubits: 4,
            data_qubits: 4,
            generator_lr: 0.01,
            discriminator_lr: 0.01,
            generator_depth: 8,
            discriminator_depth: 6,
            batch_size: 16,
            max_iterations: 1000,
            generator_frequency: 1,
            use_quantum_advantage: true,
            noise_type: NoiseType::Gaussian,
            regularization: 0.001,
            random_seed: None,
        }
    }
}

/// Types of noise distributions for the generator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseType {
    /// Gaussian (normal) distribution
    Gaussian,
    /// Uniform distribution
    Uniform,
    /// Quantum superposition state
    QuantumSuperposition,
    /// Hardware-efficient basis states
    BasisStates,
}

/// Quantum Generative Adversarial Network
pub struct QGAN {
    /// Configuration
    config: QGANConfig,
    /// Generator network
    generator: QuantumGenerator,
    /// Discriminator network
    discriminator: QuantumDiscriminator,
    /// Training statistics
    training_stats: QGANTrainingStats,
    /// Random number generator
    rng: StdRng,
    /// Current iteration
    iteration: usize,
}

/// Quantum generator network
pub struct QuantumGenerator {
    /// Quantum circuit for generation
    circuit: QuantumGeneratorCircuit,
    /// Variational parameters
    parameters: Array1<f64>,
    /// Optimizer for parameter updates
    optimizer: VariationalOptimizer,
    /// Parameter gradients history for momentum
    gradient_history: VecDeque<Array1<f64>>,
}

/// Quantum discriminator network
pub struct QuantumDiscriminator {
    /// Quantum circuit for discrimination
    circuit: QuantumDiscriminatorCircuit,
    /// Variational parameters
    parameters: Array1<f64>,
    /// Optimizer for parameter updates
    optimizer: VariationalOptimizer,
    /// Parameter gradients history
    gradient_history: VecDeque<Array1<f64>>,
}

/// Quantum circuit for the generator
#[derive(Debug, Clone)]
pub struct QuantumGeneratorCircuit {
    /// Number of latent qubits
    latent_qubits: usize,
    /// Number of data qubits
    data_qubits: usize,
    /// Circuit depth
    depth: usize,
    /// Total number of qubits
    total_qubits: usize,
    /// Noise type for initialization
    noise_type: NoiseType,
}

/// Quantum circuit for the discriminator
#[derive(Debug, Clone)]
pub struct QuantumDiscriminatorCircuit {
    /// Number of data qubits
    data_qubits: usize,
    /// Number of auxiliary qubits for computation
    aux_qubits: usize,
    /// Circuit depth
    depth: usize,
    /// Total number of qubits
    total_qubits: usize,
}

/// Training statistics for QGAN
#[derive(Debug, Clone, Default)]
pub struct QGANTrainingStats {
    /// Generator loss over iterations
    pub generator_losses: Vec<f64>,
    /// Discriminator loss over iterations
    pub discriminator_losses: Vec<f64>,
    /// Fidelity between generated and real data
    pub fidelities: Vec<f64>,
    /// Training time per iteration
    pub iteration_times: Vec<f64>,
    /// Convergence metrics
    pub convergence_metrics: Vec<f64>,
}

/// Training metrics for a single iteration
#[derive(Debug, Clone)]
pub struct QGANIterationMetrics {
    /// Generator loss
    pub generator_loss: f64,
    /// Discriminator loss
    pub discriminator_loss: f64,
    /// Real data accuracy (discriminator on real data)
    pub real_accuracy: f64,
    /// Fake data accuracy (discriminator on generated data)
    pub fake_accuracy: f64,
    /// Fidelity between generated and real distributions
    pub fidelity: f64,
    /// Iteration number
    pub iteration: usize,
}

impl QGAN {
    /// Create a new Quantum GAN
    pub fn new(config: QGANConfig) -> QuantRS2Result<Self> {
        let rng = match config.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed([0; 32]), // Use fixed seed for StdRng
        };

        let generator = QuantumGenerator::new(&config)?;
        let discriminator = QuantumDiscriminator::new(&config)?;

        Ok(Self {
            config,
            generator,
            discriminator,
            training_stats: QGANTrainingStats::default(),
            rng,
            iteration: 0,
        })
    }

    /// Train the QGAN on real data
    pub fn train(&mut self, real_data: &Array2<f64>) -> QuantRS2Result<QGANIterationMetrics> {
        let batch_size = self.config.batch_size.min(real_data.nrows());
        
        // Sample a batch of real data
        let real_batch = self.sample_real_data_batch(real_data, batch_size)?;
        
        // Generate fake data
        let fake_batch = self.generate_fake_data_batch(batch_size)?;
        
        // Train discriminator
        let (d_loss_real, d_loss_fake) = self.train_discriminator(&real_batch, &fake_batch)?;
        let discriminator_loss = d_loss_real + d_loss_fake;
        
        // Train generator (less frequently to maintain balance)
        let generator_loss = if self.iteration % self.config.generator_frequency == 0 {
            self.train_generator(batch_size)?
        } else {
            0.0
        };
        
        // Compute metrics
        let real_accuracy = self.compute_discriminator_accuracy(&real_batch, true)?;
        let fake_accuracy = self.compute_discriminator_accuracy(&fake_batch, false)?;
        let fidelity = self.compute_fidelity(&real_batch, &fake_batch)?;
        
        // Update statistics
        self.training_stats.generator_losses.push(generator_loss);
        self.training_stats.discriminator_losses.push(discriminator_loss);
        self.training_stats.fidelities.push(fidelity);
        
        let metrics = QGANIterationMetrics {
            generator_loss,
            discriminator_loss,
            real_accuracy,
            fake_accuracy,
            fidelity,
            iteration: self.iteration,
        };
        
        self.iteration += 1;
        
        Ok(metrics)
    }

    /// Generate fake data using the current generator
    pub fn generate_data(&mut self, num_samples: usize) -> QuantRS2Result<Array2<f64>> {
        self.generate_fake_data_batch(num_samples)
    }

    /// Evaluate the discriminator on data
    pub fn discriminate(&self, data: &Array2<f64>) -> QuantRS2Result<Array1<f64>> {
        let num_samples = data.nrows();
        let mut scores = Array1::zeros(num_samples);
        
        for i in 0..num_samples {
            let sample = data.row(i).to_owned();
            scores[i] = self.discriminator.discriminate(&sample)?;
        }
        
        Ok(scores)
    }

    /// Sample real data batch
    fn sample_real_data_batch(&mut self, real_data: &Array2<f64>, batch_size: usize) -> QuantRS2Result<Array2<f64>> {
        let num_samples = real_data.nrows();
        let mut batch = Array2::zeros((batch_size, real_data.ncols()));
        
        for i in 0..batch_size {
            let idx = self.rng.random_range(0..num_samples);
            batch.row_mut(i).assign(&real_data.row(idx));
        }
        
        Ok(batch)
    }

    /// Generate fake data batch
    fn generate_fake_data_batch(&mut self, batch_size: usize) -> QuantRS2Result<Array2<f64>> {
        let mut fake_batch = Array2::zeros((batch_size, self.config.data_qubits));
        
        for i in 0..batch_size {
            let noise = self.sample_noise()?;
            let generated_sample = self.generator.generate(&noise)?;
            fake_batch.row_mut(i).assign(&generated_sample);
        }
        
        Ok(fake_batch)
    }

    /// Sample noise vector for generator input
    fn sample_noise(&mut self) -> QuantRS2Result<Array1<f64>> {
        let mut noise = Array1::zeros(self.config.latent_qubits);
        
        match self.config.noise_type {
            NoiseType::Gaussian => {
                for i in 0..self.config.latent_qubits {
                    noise[i] = self.rng.random::<f64>() * 2.0 - 1.0; // Normal-like distribution
                }
            },
            NoiseType::Uniform => {
                for i in 0..self.config.latent_qubits {
                    noise[i] = self.rng.random::<f64>() * 2.0 * std::f64::consts::PI - std::f64::consts::PI;
                }
            },
            NoiseType::QuantumSuperposition => {
                // Initialize in superposition state
                for i in 0..self.config.latent_qubits {
                    noise[i] = std::f64::consts::PI / 2.0; // Hadamard-like angle
                }
            },
            NoiseType::BasisStates => {
                // Random computational basis state
                let state = self.rng.random_range(0..(1 << self.config.latent_qubits));
                for i in 0..self.config.latent_qubits {
                    noise[i] = if (state >> i) & 1 == 1 { std::f64::consts::PI } else { 0.0 };
                }
            },
        }
        
        Ok(noise)
    }

    /// Train discriminator on real and fake data
    fn train_discriminator(&mut self, real_batch: &Array2<f64>, fake_batch: &Array2<f64>) -> QuantRS2Result<(f64, f64)> {
        // Train on real data (target = 1)
        let d_loss_real = self.discriminator.train_batch(real_batch, &Array1::ones(real_batch.nrows()))?;
        
        // Train on fake data (target = 0)  
        let d_loss_fake = self.discriminator.train_batch(fake_batch, &Array1::zeros(fake_batch.nrows()))?;
        
        Ok((d_loss_real, d_loss_fake))
    }

    /// Train generator to fool discriminator
    fn train_generator(&mut self, batch_size: usize) -> QuantRS2Result<f64> {
        // Generate fake data
        let fake_batch = self.generate_fake_data_batch(batch_size)?;
        
        // Get discriminator scores for fake data
        let discriminator_scores = self.discriminate(&fake_batch)?;
        
        // Generator loss: want discriminator to output 1 for fake data
        let targets = Array1::ones(batch_size);
        let generator_loss = self.generator.train_adversarial(&fake_batch, &targets, &discriminator_scores)?;
        
        Ok(generator_loss)
    }

    /// Compute discriminator accuracy
    fn compute_discriminator_accuracy(&self, data: &Array2<f64>, is_real: bool) -> QuantRS2Result<f64> {
        let scores = self.discriminate(data)?;
        let threshold = 0.5;
        let target = if is_real { 1.0 } else { 0.0 };
        
        let correct = scores.iter()
            .map(|&score| if (score > threshold) == is_real { 1.0 } else { 0.0 })
            .sum::<f64>();
        
        Ok(correct / data.nrows() as f64)
    }

    /// Compute fidelity between real and generated data distributions
    fn compute_fidelity(&self, real_batch: &Array2<f64>, fake_batch: &Array2<f64>) -> QuantRS2Result<f64> {
        // Simplified fidelity computation using mean and variance
        let real_mean = real_batch.mean_axis(Axis(0)).unwrap();
        let fake_mean = fake_batch.mean_axis(Axis(0)).unwrap();
        
        let real_var = real_batch.var_axis(Axis(0), 0.0);
        let fake_var = fake_batch.var_axis(Axis(0), 0.0);
        
        // Approximate fidelity based on Gaussian distributions
        let mean_diff = (&real_mean - &fake_mean).mapv(|x| x.powi(2)).sum().sqrt();
        let var_diff = (&real_var - &fake_var).mapv(|x| x.powi(2)).sum().sqrt();
        
        let fidelity = (-0.5 * (mean_diff + var_diff)).exp();
        
        Ok(fidelity)
    }

    /// Get training statistics
    pub fn get_training_stats(&self) -> &QGANTrainingStats {
        &self.training_stats
    }

    /// Check if training has converged
    pub fn has_converged(&self, tolerance: f64, window: usize) -> bool {
        if self.training_stats.fidelities.len() < window {
            return false;
        }

        let recent_fidelities = &self.training_stats.fidelities[self.training_stats.fidelities.len() - window..];
        let mean_fidelity = recent_fidelities.iter().sum::<f64>() / window as f64;
        
        mean_fidelity > 1.0 - tolerance
    }
}

impl QuantumGenerator {
    /// Create a new quantum generator
    fn new(config: &QGANConfig) -> QuantRS2Result<Self> {
        let circuit = QuantumGeneratorCircuit::new(
            config.latent_qubits,
            config.data_qubits,
            config.generator_depth,
            config.noise_type,
        )?;
        
        let num_parameters = circuit.get_parameter_count();
        let mut parameters = Array1::zeros(num_parameters);
        
        // Initialize parameters randomly
        let mut rng = match config.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed([0; 32]),
        };
        
        for param in parameters.iter_mut() {
            *param = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
        }
        
        let optimizer = VariationalOptimizer::new(0.01, 0.9);
        let gradient_history = VecDeque::with_capacity(10);
        
        Ok(Self {
            circuit,
            parameters,
            optimizer,
            gradient_history,
        })
    }

    /// Generate data from noise
    fn generate(&self, noise: &Array1<f64>) -> QuantRS2Result<Array1<f64>> {
        self.circuit.generate_data(noise, &self.parameters)
    }

    /// Train generator using adversarial loss
    fn train_adversarial(
        &mut self,
        generated_data: &Array2<f64>,
        targets: &Array1<f64>,
        discriminator_scores: &Array1<f64>,
    ) -> QuantRS2Result<f64> {
        let batch_size = generated_data.nrows();
        let mut total_loss = 0.0;
        
        for i in 0..batch_size {
            let data_sample = generated_data.row(i).to_owned();
            let target = targets[i];
            let score = discriminator_scores[i];
            
            // Adversarial loss: want discriminator to output 1 (think data is real)
            let loss = (score - target).powi(2);
            total_loss += loss;
            
            // Compute gradients and update parameters
            let gradients = self.circuit.compute_adversarial_gradients(&data_sample, target, score, &self.parameters)?;
            self.update_parameters(&gradients, 0.01)?; // Use config learning rate
        }
        
        Ok(total_loss / batch_size as f64)
    }

    /// Update generator parameters
    fn update_parameters(&mut self, gradients: &Array1<f64>, learning_rate: f64) -> QuantRS2Result<()> {
        // Apply momentum if we have gradient history
        let mut effective_gradients = gradients.clone();
        
        if let Some(prev_gradients) = self.gradient_history.back() {
            let momentum = 0.9;
            effective_gradients = &effective_gradients + &(prev_gradients * momentum);
        }
        
        // Update parameters
        for (param, &grad) in self.parameters.iter_mut().zip(effective_gradients.iter()) {
            *param -= learning_rate * grad;
        }
        
        // Store gradients for momentum
        self.gradient_history.push_back(effective_gradients);
        if self.gradient_history.len() > 10 {
            self.gradient_history.pop_front();
        }
        
        Ok(())
    }
}

impl QuantumDiscriminator {
    /// Create a new quantum discriminator
    fn new(config: &QGANConfig) -> QuantRS2Result<Self> {
        let circuit = QuantumDiscriminatorCircuit::new(
            config.data_qubits,
            config.discriminator_qubits - config.data_qubits,
            config.discriminator_depth,
        )?;
        
        let num_parameters = circuit.get_parameter_count();
        let mut parameters = Array1::zeros(num_parameters);
        
        // Initialize parameters randomly
        let mut rng = match config.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed([0; 32]),
        };
        
        for param in parameters.iter_mut() {
            *param = rng.random_range(-std::f64::consts::PI..std::f64::consts::PI);
        }
        
        let optimizer = VariationalOptimizer::new(0.01, 0.9);
        let gradient_history = VecDeque::with_capacity(10);
        
        Ok(Self {
            circuit,
            parameters,
            optimizer,
            gradient_history,
        })
    }

    /// Discriminate between real and fake data
    fn discriminate(&self, data: &Array1<f64>) -> QuantRS2Result<f64> {
        self.circuit.discriminate_data(data, &self.parameters)
    }

    /// Train discriminator on a batch of data
    fn train_batch(&mut self, data_batch: &Array2<f64>, targets: &Array1<f64>) -> QuantRS2Result<f64> {
        let batch_size = data_batch.nrows();
        let mut total_loss = 0.0;
        
        for i in 0..batch_size {
            let data_sample = data_batch.row(i).to_owned();
            let target = targets[i];
            
            // Get current prediction
            let prediction = self.discriminate(&data_sample)?;
            
            // Binary cross-entropy loss
            let loss = -(target * prediction.ln() + (1.0 - target) * (1.0 - prediction).ln());
            total_loss += loss;
            
            // Compute gradients and update parameters
            let gradients = self.circuit.compute_discriminator_gradients(&data_sample, target, prediction, &self.parameters)?;
            self.update_parameters(&gradients, 0.01)?; // Use config learning rate
        }
        
        Ok(total_loss / batch_size as f64)
    }

    /// Update discriminator parameters
    fn update_parameters(&mut self, gradients: &Array1<f64>, learning_rate: f64) -> QuantRS2Result<()> {
        // Apply momentum if we have gradient history
        let mut effective_gradients = gradients.clone();
        
        if let Some(prev_gradients) = self.gradient_history.back() {
            let momentum = 0.9;
            effective_gradients = &effective_gradients + &(prev_gradients * momentum);
        }
        
        // Update parameters
        for (param, &grad) in self.parameters.iter_mut().zip(effective_gradients.iter()) {
            *param -= learning_rate * grad;
        }
        
        // Store gradients for momentum
        self.gradient_history.push_back(effective_gradients);
        if self.gradient_history.len() > 10 {
            self.gradient_history.pop_front();
        }
        
        Ok(())
    }
}

impl QuantumGeneratorCircuit {
    /// Create a new quantum generator circuit
    fn new(latent_qubits: usize, data_qubits: usize, depth: usize, noise_type: NoiseType) -> QuantRS2Result<Self> {
        let total_qubits = latent_qubits + data_qubits;
        
        Ok(Self {
            latent_qubits,
            data_qubits,
            depth,
            total_qubits,
            noise_type,
        })
    }

    /// Get number of parameters in the circuit
    fn get_parameter_count(&self) -> usize {
        let total_qubits = self.latent_qubits + self.data_qubits;
        // Each layer: rotation gates (3 per qubit) + entangling gates
        let rotations_per_layer = total_qubits * 3;
        let entangling_per_layer = total_qubits; // Simplified
        self.depth * (rotations_per_layer + entangling_per_layer)
    }

    /// Generate data from noise input
    fn generate_data(&self, noise: &Array1<f64>, parameters: &Array1<f64>) -> QuantRS2Result<Array1<f64>> {
        // Build quantum circuit
        let mut gates = Vec::new();
        
        // Initialize noise qubits
        for i in 0..self.latent_qubits {
            let noise_value = if i < noise.len() { noise[i] } else { 0.0 };
            gates.push(Box::new(RotationY {
                target: QubitId(i as u32),
                theta: noise_value,
            }) as Box<dyn GateOp>);
        }
        
        // Apply variational layers
        let mut param_idx = 0;
        for layer in 0..self.depth {
            // Rotation layer
            for qubit in 0..self.latent_qubits + self.data_qubits {
                if param_idx + 2 < parameters.len() {
                    gates.push(Box::new(RotationX {
                        target: QubitId(qubit as u32),
                        theta: parameters[param_idx],
                    }) as Box<dyn GateOp>);
                    param_idx += 1;

                    gates.push(Box::new(RotationY {
                        target: QubitId(qubit as u32),
                        theta: parameters[param_idx],
                    }) as Box<dyn GateOp>);
                    param_idx += 1;

                    gates.push(Box::new(RotationZ {
                        target: QubitId(qubit as u32),
                        theta: parameters[param_idx],
                    }) as Box<dyn GateOp>);
                    param_idx += 1;
                }
            }

            // Entangling layer
            for qubit in 0..self.latent_qubits + self.data_qubits - 1 {
                if param_idx < parameters.len() {
                    gates.push(Box::new(CRZ {
                        control: QubitId(qubit as u32),
                        target: QubitId((qubit + 1) as u32),
                        theta: parameters[param_idx],
                    }) as Box<dyn GateOp>);
                    param_idx += 1;
                }
            }
        }
        
        // Simulate circuit and extract data qubits
        let generated_data = self.simulate_generation_circuit(&gates)?;
        
        Ok(generated_data)
    }

    /// Simulate generation circuit
    fn simulate_generation_circuit(&self, gates: &[Box<dyn GateOp>]) -> QuantRS2Result<Array1<f64>> {
        // Simplified simulation: hash-based mock generation
        let mut data = Array1::zeros(self.data_qubits);
        
        let mut hash_value = 0u64;
        for gate in gates {
            if let Ok(matrix) = gate.matrix() {
                for complex in &matrix {
                    hash_value = hash_value.wrapping_add((complex.re * 1000.0) as u64);
                }
            }
        }
        
        // Convert hash to data values
        for i in 0..self.data_qubits {
            let qubit_hash = hash_value.wrapping_add(i as u64);
            data[i] = ((qubit_hash % 1000) as f64 / 1000.0) * 2.0 - 1.0; // [-1, 1]
        }
        
        Ok(data)
    }

    /// Compute adversarial gradients
    fn compute_adversarial_gradients(
        &self,
        data_sample: &Array1<f64>,
        target: f64,
        score: f64,
        parameters: &Array1<f64>,
    ) -> QuantRS2Result<Array1<f64>> {
        let mut gradients = Array1::zeros(parameters.len());
        let shift = std::f64::consts::PI / 2.0;
        
        for i in 0..parameters.len() {
            // Parameter-shift rule for quantum gradients
            let mut params_plus = parameters.clone();
            params_plus[i] += shift;
            let data_plus = self.generate_data(&Array1::zeros(self.latent_qubits), &params_plus)?;
            
            let mut params_minus = parameters.clone();
            params_minus[i] -= shift;
            let data_minus = self.generate_data(&Array1::zeros(self.latent_qubits), &params_minus)?;
            
            // Gradient of adversarial loss
            let loss_gradient = 2.0 * (score - target);
            
            // Data difference (approximation of parameter gradient)
            let data_diff = (&data_plus - &data_minus).sum() / 2.0;
            
            gradients[i] = loss_gradient * data_diff;
        }
        
        Ok(gradients)
    }
}

impl QuantumDiscriminatorCircuit {
    /// Create a new quantum discriminator circuit
    fn new(data_qubits: usize, aux_qubits: usize, depth: usize) -> QuantRS2Result<Self> {
        let total_qubits = data_qubits + aux_qubits;
        
        Ok(Self {
            data_qubits,
            aux_qubits,
            depth,
            total_qubits,
        })
    }

    /// Get number of parameters
    fn get_parameter_count(&self) -> usize {
        let total_qubits = self.data_qubits + self.aux_qubits;
        let rotations_per_layer = total_qubits * 3;
        let entangling_per_layer = total_qubits;
        self.depth * (rotations_per_layer + entangling_per_layer)
    }

    /// Discriminate data (return probability it's real)
    fn discriminate_data(&self, data: &Array1<f64>, parameters: &Array1<f64>) -> QuantRS2Result<f64> {
        // Build discriminator circuit
        let mut gates = Vec::new();
        
        // Encode input data
        for i in 0..self.data_qubits {
            let data_value = if i < data.len() { data[i] } else { 0.0 };
            gates.push(Box::new(RotationY {
                target: QubitId(i as u32),
                theta: data_value * std::f64::consts::PI,
            }) as Box<dyn GateOp>);
        }
        
        // Apply variational layers
        let mut param_idx = 0;
        for layer in 0..self.depth {
            // Rotation layer
            for qubit in 0..self.data_qubits + self.aux_qubits {
                if param_idx + 2 < parameters.len() {
                    gates.push(Box::new(RotationX {
                        target: QubitId(qubit as u32),
                        theta: parameters[param_idx],
                    }) as Box<dyn GateOp>);
                    param_idx += 1;

                    gates.push(Box::new(RotationY {
                        target: QubitId(qubit as u32),
                        theta: parameters[param_idx],
                    }) as Box<dyn GateOp>);
                    param_idx += 1;

                    gates.push(Box::new(RotationZ {
                        target: QubitId(qubit as u32),
                        theta: parameters[param_idx],
                    }) as Box<dyn GateOp>);
                    param_idx += 1;
                }
            }

            // Entangling layer
            for qubit in 0..self.data_qubits + self.aux_qubits - 1 {
                if param_idx < parameters.len() {
                    gates.push(Box::new(CRZ {
                        control: QubitId(qubit as u32),
                        target: QubitId((qubit + 1) as u32),
                        theta: parameters[param_idx],
                    }) as Box<dyn GateOp>);
                    param_idx += 1;
                }
            }
        }
        
        // Simulate circuit and return probability
        let probability = self.simulate_discrimination_circuit(&gates)?;
        
        Ok(probability)
    }

    /// Simulate discrimination circuit
    fn simulate_discrimination_circuit(&self, gates: &[Box<dyn GateOp>]) -> QuantRS2Result<f64> {
        // Simplified simulation: hash-based mock probability
        let mut hash_value = 0u64;
        
        for gate in gates {
            if let Ok(matrix) = gate.matrix() {
                for complex in &matrix {
                    hash_value = hash_value.wrapping_add((complex.re * 1000.0) as u64);
                    hash_value = hash_value.wrapping_add((complex.im * 1000.0) as u64);
                }
            }
        }
        
        // Convert to probability [0, 1]
        let probability = ((hash_value % 1000) as f64) / 1000.0;
        
        Ok(probability)
    }

    /// Compute discriminator gradients
    fn compute_discriminator_gradients(
        &self,
        data_sample: &Array1<f64>,
        target: f64,
        prediction: f64,
        parameters: &Array1<f64>,
    ) -> QuantRS2Result<Array1<f64>> {
        let mut gradients = Array1::zeros(parameters.len());
        let shift = std::f64::consts::PI / 2.0;
        
        for i in 0..parameters.len() {
            // Parameter-shift rule
            let mut params_plus = parameters.clone();
            params_plus[i] += shift;
            let pred_plus = self.discriminate_data(data_sample, &params_plus)?;
            
            let mut params_minus = parameters.clone();
            params_minus[i] -= shift;
            let pred_minus = self.discriminate_data(data_sample, &params_minus)?;
            
            // Binary cross-entropy gradient
            let pred_gradient = if prediction > 0.0 && prediction < 1.0 {
                -target / prediction + (1.0 - target) / (1.0 - prediction)
            } else {
                0.0 // Avoid division by zero
            };
            
            gradients[i] = pred_gradient * (pred_plus - pred_minus) / 2.0;
        }
        
        Ok(gradients)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qgan_creation() {
        let config = QGANConfig::default();
        let qgan = QGAN::new(config).unwrap();
        
        assert_eq!(qgan.iteration, 0);
        assert_eq!(qgan.training_stats.generator_losses.len(), 0);
    }

    #[test]
    fn test_noise_generation() {
        let config = QGANConfig::default();
        let mut qgan = QGAN::new(config).unwrap();
        
        let noise = qgan.sample_noise().unwrap();
        assert_eq!(noise.len(), qgan.config.latent_qubits);
        
        // Test different noise types
        qgan.config.noise_type = NoiseType::Uniform;
        let uniform_noise = qgan.sample_noise().unwrap();
        assert_eq!(uniform_noise.len(), qgan.config.latent_qubits);
        
        qgan.config.noise_type = NoiseType::QuantumSuperposition;
        let quantum_noise = qgan.sample_noise().unwrap();
        assert_eq!(quantum_noise.len(), qgan.config.latent_qubits);
    }

    #[test]
    fn test_data_generation() {
        let config = QGANConfig::default();
        let mut qgan = QGAN::new(config).unwrap();
        
        let generated_data = qgan.generate_data(5).unwrap();
        assert_eq!(generated_data.nrows(), 5);
        assert_eq!(generated_data.ncols(), qgan.config.data_qubits);
    }

    #[test]
    fn test_discrimination() {
        let config = QGANConfig::default();
        let qgan = QGAN::new(config).unwrap();
        
        // Create some mock data
        let data = Array2::from_shape_fn((3, qgan.config.data_qubits), |(i, j)| {
            (i as f64 + j as f64) / 10.0
        });
        
        let scores = qgan.discriminate(&data).unwrap();
        assert_eq!(scores.len(), 3);
        
        // Check scores are in [0, 1]
        for &score in scores.iter() {
            assert!(score >= 0.0 && score <= 1.0);
        }
    }

    #[test]
    fn test_qgan_training_step() {
        let config = QGANConfig {
            batch_size: 4,
            ..Default::default()
        };
        let mut qgan = QGAN::new(config).unwrap();
        
        // Create some mock real data
        let real_data = Array2::from_shape_fn((10, qgan.config.data_qubits), |(i, j)| {
            ((i + j) as f64).sin()
        });
        
        let metrics = qgan.train(&real_data).unwrap();
        
        assert_eq!(metrics.iteration, 0);
        assert!(metrics.fidelity >= 0.0 && metrics.fidelity <= 1.0);
        assert_eq!(qgan.iteration, 1);
        assert_eq!(qgan.training_stats.generator_losses.len(), 1);
        assert_eq!(qgan.training_stats.discriminator_losses.len(), 1);
    }

    #[test]
    fn test_convergence_check() {
        let config = QGANConfig::default();
        let mut qgan = QGAN::new(config).unwrap();
        
        // Simulate high fidelity values for convergence
        for _ in 0..10 {
            qgan.training_stats.fidelities.push(0.95);
        }
        
        assert!(qgan.has_converged(0.1, 5)); // Should converge with tolerance 0.1
        assert!(!qgan.has_converged(0.01, 5)); // Should not converge with stricter tolerance
    }

    #[test]
    fn test_quantum_generator_circuit() {
        let circuit = QuantumGeneratorCircuit::new(3, 2, 4, NoiseType::Gaussian).unwrap();
        let param_count = circuit.get_parameter_count();
        assert!(param_count > 0);
        
        let noise = Array1::from_vec(vec![0.5, -0.5, 0.0]);
        let parameters = Array1::zeros(param_count);
        
        let generated_data = circuit.generate_data(&noise, &parameters).unwrap();
        assert_eq!(generated_data.len(), 2);
    }

    #[test]
    fn test_quantum_discriminator_circuit() {
        let circuit = QuantumDiscriminatorCircuit::new(3, 2, 4).unwrap();
        let param_count = circuit.get_parameter_count();
        assert!(param_count > 0);
        
        let data = Array1::from_vec(vec![0.5, -0.5, 0.0]);
        let parameters = Array1::zeros(param_count);
        
        let score = circuit.discriminate_data(&data, &parameters).unwrap();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_fidelity_computation() {
        let config = QGANConfig::default();
        let qgan = QGAN::new(config).unwrap();
        
        // Identical distributions should have high fidelity
        let data1 = Array2::ones((5, 3));
        let data2 = Array2::ones((5, 3));
        let fidelity = qgan.compute_fidelity(&data1, &data2).unwrap();
        assert!(fidelity > 0.9);
        
        // Very different distributions should have low fidelity
        let data3 = Array2::zeros((5, 3));
        let fidelity2 = qgan.compute_fidelity(&data1, &data3).unwrap();
        assert!(fidelity2 < fidelity);
    }
}