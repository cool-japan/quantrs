//! Quantum Boltzmann Machines for machine learning with quantum annealing
//!
//! This module provides implementations of Restricted Boltzmann Machines (RBMs)
//! and other Boltzmann machine variants that leverage quantum annealing for
//! sampling and training, enabling quantum machine learning applications.

use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::SliceRandomExt;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::ising::{IsingError, IsingModel};
use crate::simulator::{AnnealingParams, AnnealingSolution, QuantumAnnealingSimulator};

/// Errors that can occur in quantum Boltzmann machine operations
#[derive(Error, Debug)]
pub enum QbmError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Invalid model configuration
    #[error("Invalid model: {0}")]
    InvalidModel(String),

    /// Training error
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Sampling error
    #[error("Sampling error: {0}")]
    SamplingError(String),

    /// Data format error
    #[error("Data error: {0}")]
    DataError(String),
}

/// Result type for QBM operations
pub type QbmResult<T> = Result<T, QbmError>;

/// Accumulated contrastive-divergence gradient statistics for one mini-batch.
///
/// For a Restricted Boltzmann Machine the log-likelihood gradient of the
/// weights is `⟨v_i h_j⟩_data − ⟨v_i h_j⟩_model`, the visible-bias gradient is
/// `⟨v_i⟩_data − ⟨v_i⟩_model`, and the hidden-bias gradient is
/// `⟨h_j⟩_data − ⟨h_j⟩_model`. The "data" expectations are taken over the
/// positive phase (training sample clamped on the visible units) and the
/// "model" expectations over the negative phase (the reconstruction produced by
/// `k` steps of (quantum) Gibbs sampling). These accumulators sum the per-sample
/// contributions; the caller divides by the batch size to obtain the mean
/// gradient used for the parameter update.
#[derive(Debug, Clone)]
struct CdGradients {
    /// Sum over the batch of `v_i h_j` correlations, positive minus negative.
    weight_grad: Vec<Vec<f64>>,
    /// Sum over the batch of `v_i`, positive minus negative.
    visible_bias_grad: Vec<f64>,
    /// Sum over the batch of `h_j`, positive minus negative.
    hidden_bias_grad: Vec<f64>,
    /// Number of samples accumulated (used to form the mean).
    count: usize,
}

impl CdGradients {
    fn new(num_visible: usize, num_hidden: usize) -> Self {
        Self {
            weight_grad: vec![vec![0.0; num_hidden]; num_visible],
            visible_bias_grad: vec![0.0; num_visible],
            hidden_bias_grad: vec![0.0; num_hidden],
            count: 0,
        }
    }

    /// Accumulate the positive- minus negative-phase contribution of one sample.
    ///
    /// `v_pos`/`h_pos` are the visible data vector and the hidden activation
    /// probabilities of the positive phase; `v_neg`/`h_neg` are the
    /// reconstructed visible vector and hidden activation probabilities of the
    /// negative phase. Using the hidden *probabilities* (rather than sampled
    /// binary states) for the gradient is the standard low-variance estimator
    /// (Hinton, "A Practical Guide to Training RBMs", 2010).
    fn accumulate(&mut self, v_pos: &[f64], h_pos: &[f64], v_neg: &[f64], h_neg: &[f64]) {
        for (i, weight_row) in self.weight_grad.iter_mut().enumerate() {
            let vp = v_pos[i];
            let vn = v_neg[i];
            for (j, w) in weight_row.iter_mut().enumerate() {
                *w += vp.mul_add(h_pos[j], -(vn * h_neg[j]));
            }
        }
        for (i, g) in self.visible_bias_grad.iter_mut().enumerate() {
            *g += v_pos[i] - v_neg[i];
        }
        for (j, g) in self.hidden_bias_grad.iter_mut().enumerate() {
            *g += h_pos[j] - h_neg[j];
        }
        self.count += 1;
    }
}

/// Type of Boltzmann machine unit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitType {
    /// Binary units (0/1 or -1/+1)
    Binary,
    /// Gaussian units for continuous values
    Gaussian,
    /// Softmax units for categorical data
    Softmax,
}

/// Configuration for a Boltzmann machine layer
#[derive(Debug, Clone)]
pub struct LayerConfig {
    /// Number of units in the layer
    pub num_units: usize,

    /// Type of units
    pub unit_type: UnitType,

    /// Layer name
    pub name: String,

    /// Bias initialization range
    pub bias_init_range: (f64, f64),

    /// Whether to use quantum annealing for sampling
    pub quantum_sampling: bool,
}

impl LayerConfig {
    /// Create a new layer configuration
    #[must_use]
    pub const fn new(name: String, num_units: usize, unit_type: UnitType) -> Self {
        Self {
            num_units,
            unit_type,
            name,
            bias_init_range: (-0.1, 0.1),
            quantum_sampling: true,
        }
    }

    /// Set bias initialization range
    #[must_use]
    pub const fn with_bias_range(mut self, min: f64, max: f64) -> Self {
        self.bias_init_range = (min, max);
        self
    }

    /// Enable or disable quantum sampling
    #[must_use]
    pub const fn with_quantum_sampling(mut self, enabled: bool) -> Self {
        self.quantum_sampling = enabled;
        self
    }
}

/// Restricted Boltzmann Machine with quantum annealing support
#[derive(Debug)]
pub struct QuantumRestrictedBoltzmannMachine {
    /// Visible layer configuration
    visible_config: LayerConfig,

    /// Hidden layer configuration
    hidden_config: LayerConfig,

    /// Visible unit biases
    visible_biases: Vec<f64>,

    /// Hidden unit biases
    hidden_biases: Vec<f64>,

    /// Weight matrix (visible x hidden)
    weights: Vec<Vec<f64>>,

    /// Training configuration
    training_config: QbmTrainingConfig,

    /// Random number generator
    rng: ChaCha8Rng,

    /// Training statistics
    training_stats: Option<QbmTrainingStats>,
}

/// Configuration for QBM training
#[derive(Debug, Clone)]
pub struct QbmTrainingConfig {
    /// Learning rate
    pub learning_rate: f64,

    /// Number of training epochs
    pub epochs: usize,

    /// Batch size for training
    pub batch_size: usize,

    /// Number of Gibbs sampling steps for negative phase
    pub k_steps: usize,

    /// Use persistent contrastive divergence
    pub persistent_cd: bool,

    /// Weight decay regularization
    pub weight_decay: f64,

    /// Momentum for parameter updates
    pub momentum: f64,

    /// Annealing parameters for quantum sampling
    pub annealing_params: AnnealingParams,

    /// Random seed
    pub seed: Option<u64>,

    /// Reconstruction error threshold for early stopping
    pub error_threshold: Option<f64>,

    /// Logging frequency (epochs)
    pub log_frequency: usize,
}

impl Default for QbmTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 100,
            batch_size: 32,
            k_steps: 1,
            persistent_cd: false,
            weight_decay: 0.0001,
            momentum: 0.5,
            annealing_params: AnnealingParams::default(),
            seed: None,
            error_threshold: None,
            log_frequency: 10,
        }
    }
}

/// Training statistics for QBM
#[derive(Debug, Clone)]
pub struct QbmTrainingStats {
    /// Training time
    pub total_training_time: Duration,

    /// Reconstruction error per epoch
    pub reconstruction_errors: Vec<f64>,

    /// Free energy difference per epoch
    pub free_energy_diffs: Vec<f64>,

    /// Number of epochs completed
    pub epochs_completed: usize,

    /// Final reconstruction error
    pub final_reconstruction_error: f64,

    /// Convergence achieved
    pub converged: bool,

    /// Quantum sampling statistics
    pub quantum_sampling_stats: QuantumSamplingStats,
}

/// Statistics for quantum sampling in QBM
#[derive(Debug, Clone)]
pub struct QuantumSamplingStats {
    /// Total quantum sampling time
    pub total_sampling_time: Duration,

    /// Number of quantum sampling calls
    pub sampling_calls: usize,

    /// Average annealing energy
    pub average_annealing_energy: f64,

    /// Success rate of quantum sampling
    pub success_rate: f64,

    /// Classical fallback usage percentage
    pub classical_fallback_rate: f64,
}

impl Default for QuantumSamplingStats {
    fn default() -> Self {
        Self {
            total_sampling_time: Duration::from_secs(0),
            sampling_calls: 0,
            average_annealing_energy: 0.0,
            success_rate: 1.0,
            classical_fallback_rate: 0.0,
        }
    }
}

/// Training sample for QBM
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// Input data
    pub data: Vec<f64>,

    /// Optional label (for supervised variants)
    pub label: Option<Vec<f64>>,
}

impl TrainingSample {
    /// Create a new training sample
    #[must_use]
    pub const fn new(data: Vec<f64>) -> Self {
        Self { data, label: None }
    }

    /// Create a labeled training sample
    #[must_use]
    pub const fn labeled(data: Vec<f64>, label: Vec<f64>) -> Self {
        Self {
            data,
            label: Some(label),
        }
    }
}

/// Results from QBM inference
#[derive(Debug, Clone)]
pub struct QbmInferenceResult {
    /// Reconstructed visible units
    pub reconstruction: Vec<f64>,

    /// Hidden unit activations
    pub hidden_activations: Vec<f64>,

    /// Free energy of the configuration
    pub free_energy: f64,

    /// Probability of the input
    pub probability: f64,
}

impl QuantumRestrictedBoltzmannMachine {
    /// Create a new Quantum RBM
    pub fn new(
        visible_config: LayerConfig,
        hidden_config: LayerConfig,
        training_config: QbmTrainingConfig,
    ) -> QbmResult<Self> {
        if visible_config.num_units == 0 || hidden_config.num_units == 0 {
            return Err(QbmError::InvalidModel(
                "Layer sizes must be > 0".to_string(),
            ));
        }

        let rng = match training_config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::seed_from_u64(thread_rng().random()),
        };

        let mut rbm = Self {
            visible_config: visible_config.clone(),
            hidden_config: hidden_config.clone(),
            visible_biases: vec![0.0; visible_config.num_units],
            hidden_biases: vec![0.0; hidden_config.num_units],
            weights: vec![vec![0.0; hidden_config.num_units]; visible_config.num_units],
            training_config,
            rng,
            training_stats: None,
        };

        rbm.initialize_parameters()?;
        Ok(rbm)
    }

    /// Initialize RBM parameters randomly
    fn initialize_parameters(&mut self) -> QbmResult<()> {
        // Initialize visible biases
        let (v_min, v_max) = self.visible_config.bias_init_range;
        for bias in &mut self.visible_biases {
            *bias = self.rng.random_range(v_min..v_max);
        }

        // Initialize hidden biases
        let (h_min, h_max) = self.hidden_config.bias_init_range;
        for bias in &mut self.hidden_biases {
            *bias = self.rng.random_range(h_min..h_max);
        }

        // Initialize weights using Xavier initialization
        let fan_in = self.visible_config.num_units as f64;
        let fan_out = self.hidden_config.num_units as f64;
        let xavier_std = (2.0 / (fan_in + fan_out)).sqrt();

        for i in 0..self.visible_config.num_units {
            for j in 0..self.hidden_config.num_units {
                self.weights[i][j] = self.rng.random_range(-xavier_std..xavier_std);
            }
        }

        Ok(())
    }

    /// Train the RBM on a dataset
    pub fn train(&mut self, dataset: &[TrainingSample]) -> QbmResult<()> {
        if dataset.is_empty() {
            return Err(QbmError::DataError("Dataset is empty".to_string()));
        }

        // Validate data dimensions
        let expected_size = self.visible_config.num_units;
        for (i, sample) in dataset.iter().enumerate() {
            if sample.data.len() != expected_size {
                return Err(QbmError::DataError(format!(
                    "Sample {} has {} features, expected {}",
                    i,
                    sample.data.len(),
                    expected_size
                )));
            }
        }

        println!("Starting QBM training with {} samples", dataset.len());

        let start_time = Instant::now();
        let mut reconstruction_errors = Vec::new();
        let mut free_energy_diffs = Vec::new();
        let mut quantum_stats = QuantumSamplingStats::default();

        // Momentum terms
        let mut weight_momentum =
            vec![vec![0.0; self.hidden_config.num_units]; self.visible_config.num_units];
        let mut visible_bias_momentum = vec![0.0; self.visible_config.num_units];
        let mut hidden_bias_momentum = vec![0.0; self.hidden_config.num_units];

        // Persistent chains for PCD
        let mut persistent_chains = if self.training_config.persistent_cd {
            Some(self.initialize_persistent_chains(self.training_config.batch_size)?)
        } else {
            None
        };

        for epoch in 0..self.training_config.epochs {
            let epoch_start = Instant::now();
            let mut epoch_error = 0.0;
            let mut epoch_free_energy_diff = 0.0;
            let mut num_batches = 0;

            // Shuffle dataset
            let mut shuffled_indices: Vec<usize> = (0..dataset.len()).collect();
            use scirs2_core::random::prelude::*;
            shuffled_indices.shuffle(&mut self.rng);

            // Process batches
            for batch_start in (0..dataset.len()).step_by(self.training_config.batch_size) {
                let batch_end = (batch_start + self.training_config.batch_size).min(dataset.len());
                let batch_indices = &shuffled_indices[batch_start..batch_end];

                let batch_samples: Vec<&TrainingSample> =
                    batch_indices.iter().map(|&i| &dataset[i]).collect();

                // Perform contrastive divergence: this both estimates the batch
                // error/free-energy diagnostics AND accumulates the real
                // positive-minus-negative phase gradient statistics.
                let (batch_error, batch_fe_diff, batch_gradients, batch_quantum_stats) =
                    self.contrastive_divergence_batch(&batch_samples, &mut persistent_chains)?;

                // Apply the real CD gradients with momentum and weight decay.
                self.update_parameters_with_momentum(
                    &batch_gradients,
                    &mut weight_momentum,
                    &mut visible_bias_momentum,
                    &mut hidden_bias_momentum,
                )?;

                epoch_error += batch_error;
                epoch_free_energy_diff += batch_fe_diff;
                quantum_stats.merge(&batch_quantum_stats);
                num_batches += 1;
            }

            let avg_error = epoch_error / f64::from(num_batches);
            let avg_fe_diff = epoch_free_energy_diff / f64::from(num_batches);

            reconstruction_errors.push(avg_error);
            free_energy_diffs.push(avg_fe_diff);

            // Logging
            if epoch % self.training_config.log_frequency == 0 {
                println!(
                    "Epoch {}: Error = {:.6}, FE Diff = {:.6}, Time = {:.2?}",
                    epoch,
                    avg_error,
                    avg_fe_diff,
                    epoch_start.elapsed()
                );
            }

            // Early stopping
            if let Some(threshold) = self.training_config.error_threshold {
                if avg_error < threshold {
                    println!("Converged at epoch {epoch} with error {avg_error:.6}");
                    break;
                }
            }
        }

        let total_time = start_time.elapsed();

        // Store training statistics
        self.training_stats = Some(QbmTrainingStats {
            total_training_time: total_time,
            reconstruction_errors: reconstruction_errors.clone(),
            free_energy_diffs,
            epochs_completed: reconstruction_errors.len(),
            final_reconstruction_error: reconstruction_errors.last().copied().unwrap_or(0.0),
            converged: self.training_config.error_threshold.map_or(false, |t| {
                reconstruction_errors.last().unwrap_or(&f64::INFINITY) < &t
            }),
            quantum_sampling_stats: quantum_stats,
        });

        println!("Training completed in {total_time:.2?}");
        Ok(())
    }

    /// Perform contrastive divergence for a batch.
    ///
    /// Runs the positive phase (sample hidden units given the clamped data) and
    /// the negative phase (`k` steps of Gibbs/quantum sampling) for every sample,
    /// accumulating the CD log-likelihood gradient
    /// `⟨v h⟩_data − ⟨v h⟩_model` (and the analogous bias gradients) into a
    /// [`CdGradients`] that the caller applies. Also returns the mean
    /// reconstruction error and free-energy difference for diagnostics.
    fn contrastive_divergence_batch(
        &mut self,
        batch: &[&TrainingSample],
        persistent_chains: &mut Option<Vec<Vec<f64>>>,
    ) -> QbmResult<(f64, f64, CdGradients, QuantumSamplingStats)> {
        let mut total_error = 0.0;
        let mut total_fe_diff = 0.0;
        let mut quantum_stats = QuantumSamplingStats::default();
        let mut gradients =
            CdGradients::new(self.visible_config.num_units, self.hidden_config.num_units);

        for (i, sample) in batch.iter().enumerate() {
            // Positive phase: hidden activation probabilities given the data.
            let hidden_probs_pos = self.sample_hidden_given_visible(&sample.data)?;

            // Negative phase
            let (visible_recon, hidden_probs_neg, sampling_stats) =
                if self.training_config.persistent_cd {
                    if let Some(ref mut chains) = persistent_chains {
                        let chain_index = i % chains.len();
                        let mut chain = chains[chain_index].clone();
                        for _ in 0..self.training_config.k_steps {
                            let h_probs = self.sample_hidden_given_visible(&chain)?;
                            let h_states = self.sample_binary_units(&h_probs)?;
                            chain = self.sample_visible_given_hidden(&h_states)?;
                        }
                        chains[chain_index] = chain.clone();
                        let h_probs = self.sample_hidden_given_visible(&chain)?;
                        (chain, h_probs, QuantumSamplingStats::default())
                    } else {
                        return Err(QbmError::TrainingError(
                            "Persistent chains not initialized".to_string(),
                        ));
                    }
                } else {
                    // Standard CD-k
                    let mut v_states = sample.data.clone();
                    let mut sampling_stats = QuantumSamplingStats::default();

                    for _ in 0..self.training_config.k_steps {
                        let h_probs = self.sample_hidden_given_visible(&v_states)?;
                        let h_states = if self.hidden_config.quantum_sampling {
                            let (states, stats) = self.quantum_sample_hidden(&h_probs)?;
                            sampling_stats.merge(&stats);
                            states
                        } else {
                            self.sample_binary_units(&h_probs)?
                        };

                        v_states = if self.visible_config.quantum_sampling {
                            let (states, stats) = self.quantum_sample_visible(&h_states)?;
                            sampling_stats.merge(&stats);
                            states
                        } else {
                            self.sample_visible_given_hidden(&h_states)?
                        };
                    }

                    let h_probs_neg = self.sample_hidden_given_visible(&v_states)?;
                    (v_states, h_probs_neg, sampling_stats)
                };

            // Accumulate the contrastive-divergence gradient for this sample.
            // The positive phase uses the clamped data vector and its hidden
            // probabilities; the negative phase uses the reconstruction and its
            // hidden probabilities. Using probabilities (not binary samples) is
            // the standard low-variance gradient estimator.
            gradients.accumulate(
                &sample.data,
                &hidden_probs_pos,
                &visible_recon,
                &hidden_probs_neg,
            );

            // Compute reconstruction error
            let error = sample
                .data
                .iter()
                .zip(visible_recon.iter())
                .map(|(orig, recon)| (orig - recon).powi(2))
                .sum::<f64>()
                / sample.data.len() as f64;

            // Compute free energy difference
            let fe_pos = self.free_energy(&sample.data)?;
            let fe_neg = self.free_energy(&visible_recon)?;
            let fe_diff = fe_pos - fe_neg;

            total_error += error;
            total_fe_diff += fe_diff;
            quantum_stats.merge(&sampling_stats);
        }

        Ok((
            total_error / batch.len() as f64,
            total_fe_diff / batch.len() as f64,
            gradients,
            quantum_stats,
        ))
    }

    /// Update parameters using the real contrastive-divergence gradient.
    ///
    /// Applies, for each parameter `θ`, the momentum update
    /// `Δθ ← momentum·Δθ + learning_rate·g`, then `θ ← θ + Δθ`, where `g` is the
    /// mean CD gradient over the batch. Weights additionally receive L2 weight
    /// decay (`−learning_rate·decay·w`). This is gradient *ascent* on the
    /// log-likelihood: the CD gradient already has the sign
    /// `⟨·⟩_data − ⟨·⟩_model`, so we add it.
    fn update_parameters_with_momentum(
        &mut self,
        gradients: &CdGradients,
        weight_momentum: &mut [Vec<f64>],
        visible_bias_momentum: &mut [f64],
        hidden_bias_momentum: &mut [f64],
    ) -> QbmResult<()> {
        let lr = self.training_config.learning_rate;
        let momentum = self.training_config.momentum;
        let decay = self.training_config.weight_decay;

        // Mean over the batch; guard against an empty batch.
        let inv_count = if gradients.count > 0 {
            1.0 / gradients.count as f64
        } else {
            return Ok(());
        };

        // Update weights with momentum and L2 weight decay.
        for i in 0..self.visible_config.num_units {
            for j in 0..self.hidden_config.num_units {
                let gradient = gradients.weight_grad[i][j] * inv_count;
                weight_momentum[i][j] = momentum.mul_add(weight_momentum[i][j], lr * gradient);
                // w += momentum_term - lr * decay * w
                self.weights[i][j] += weight_momentum[i][j] - lr * decay * self.weights[i][j];
            }
        }

        // Update visible biases (no weight decay on biases, as is conventional).
        for i in 0..self.visible_config.num_units {
            let gradient = gradients.visible_bias_grad[i] * inv_count;
            visible_bias_momentum[i] = momentum.mul_add(visible_bias_momentum[i], lr * gradient);
            self.visible_biases[i] += visible_bias_momentum[i];
        }

        // Update hidden biases.
        for j in 0..self.hidden_config.num_units {
            let gradient = gradients.hidden_bias_grad[j] * inv_count;
            hidden_bias_momentum[j] = momentum.mul_add(hidden_bias_momentum[j], lr * gradient);
            self.hidden_biases[j] += hidden_bias_momentum[j];
        }

        Ok(())
    }

    /// Initialize persistent chains for PCD
    fn initialize_persistent_chains(&mut self, num_chains: usize) -> QbmResult<Vec<Vec<f64>>> {
        let mut chains = Vec::new();

        for _ in 0..num_chains {
            let chain: Vec<f64> = (0..self.visible_config.num_units)
                .map(|_| if self.rng.random_bool(0.5) { 1.0 } else { 0.0 })
                .collect();
            chains.push(chain);
        }

        Ok(chains)
    }

    /// Sample hidden units given visible units
    fn sample_hidden_given_visible(&self, visible: &[f64]) -> QbmResult<Vec<f64>> {
        if visible.len() != self.visible_config.num_units {
            return Err(QbmError::DataError("Visible size mismatch".to_string()));
        }

        let mut hidden_probs = vec![0.0; self.hidden_config.num_units];

        for j in 0..self.hidden_config.num_units {
            let activation = self.hidden_biases[j]
                + visible
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v * self.weights[i][j])
                    .sum::<f64>();

            hidden_probs[j] = match self.hidden_config.unit_type {
                UnitType::Binary => sigmoid(activation),
                UnitType::Gaussian => activation, // Linear for Gaussian
                UnitType::Softmax => activation,  // Will be normalized later
            };
        }

        // Apply softmax normalization if needed
        if self.hidden_config.unit_type == UnitType::Softmax {
            softmax_normalize(&mut hidden_probs);
        }

        Ok(hidden_probs)
    }

    /// Sample visible units given hidden units
    fn sample_visible_given_hidden(&self, hidden: &[f64]) -> QbmResult<Vec<f64>> {
        if hidden.len() != self.hidden_config.num_units {
            return Err(QbmError::DataError("Hidden size mismatch".to_string()));
        }

        let mut visible_probs = vec![0.0; self.visible_config.num_units];

        for i in 0..self.visible_config.num_units {
            let activation = self.visible_biases[i]
                + hidden
                    .iter()
                    .enumerate()
                    .map(|(j, &h)| h * self.weights[i][j])
                    .sum::<f64>();

            visible_probs[i] = match self.visible_config.unit_type {
                UnitType::Binary => sigmoid(activation),
                UnitType::Gaussian => activation,
                UnitType::Softmax => activation,
            };
        }

        if self.visible_config.unit_type == UnitType::Softmax {
            softmax_normalize(&mut visible_probs);
        }

        Ok(visible_probs)
    }

    /// Sample binary units from probabilities
    fn sample_binary_units(&mut self, probabilities: &[f64]) -> QbmResult<Vec<f64>> {
        Ok(probabilities
            .iter()
            .map(|&p| if self.rng.random_bool(p) { 1.0 } else { 0.0 })
            .collect())
    }

    /// Quantum sample hidden units using annealing
    fn quantum_sample_hidden(
        &mut self,
        probabilities: &[f64],
    ) -> QbmResult<(Vec<f64>, QuantumSamplingStats)> {
        let start_time = Instant::now();

        // Create Ising model for sampling
        let mut ising_model = IsingModel::new(probabilities.len());

        // Set biases based on probabilities
        for (i, &prob) in probabilities.iter().enumerate() {
            let bias = -2.0 * (prob.ln() - (1.0 - prob).ln()); // Logit transformation
            ising_model.set_bias(i, bias)?;
        }

        // Sample using quantum annealing
        if let Ok(sample) = self.quantum_annealing_sample(&ising_model) {
            let sampling_time = start_time.elapsed();
            // The annealing energy is the exact Ising energy of the returned
            // sample under the sampling Hamiltonian. With a single annealing
            // call the average over draws is just this value.
            let annealing_energy = ising_model
                .energy(&sample)
                .map_err(|e| QbmError::SamplingError(e.to_string()))?;
            let stats = QuantumSamplingStats {
                total_sampling_time: sampling_time,
                sampling_calls: 1,
                average_annealing_energy: annealing_energy,
                success_rate: 1.0,
                classical_fallback_rate: 0.0,
            };

            // Convert spins to 0/1
            let binary_sample = sample
                .iter()
                .map(|&s| if s > 0 { 1.0 } else { 0.0 })
                .collect();

            Ok((binary_sample, stats))
        } else {
            // Fallback to classical sampling. No annealing took place, so the
            // reported energy is the Ising energy of the classically-drawn
            // configuration under the same Hamiltonian (binary 0/1 mapped to
            // spins -1/+1) — a real quantity, not a fabricated zero.
            let sample = self.sample_binary_units(probabilities)?;
            let spins: Vec<i8> = sample
                .iter()
                .map(|&v| if v > 0.5 { 1 } else { -1 })
                .collect();
            let annealing_energy = ising_model
                .energy(&spins)
                .map_err(|e| QbmError::SamplingError(e.to_string()))?;
            let stats = QuantumSamplingStats {
                total_sampling_time: start_time.elapsed(),
                sampling_calls: 1,
                average_annealing_energy: annealing_energy,
                success_rate: 0.0,
                classical_fallback_rate: 1.0,
            };
            Ok((sample, stats))
        }
    }

    /// Quantum sample visible units using annealing
    fn quantum_sample_visible(
        &mut self,
        hidden_states: &[f64],
    ) -> QbmResult<(Vec<f64>, QuantumSamplingStats)> {
        let visible_probs = self.sample_visible_given_hidden(hidden_states)?;
        self.quantum_sample_hidden(&visible_probs) // Same process
    }

    /// Perform quantum annealing sampling
    fn quantum_annealing_sample(&self, model: &IsingModel) -> QbmResult<Vec<i8>> {
        let mut simulator =
            QuantumAnnealingSimulator::new(self.training_config.annealing_params.clone())
                .map_err(|e| QbmError::SamplingError(e.to_string()))?;

        let result = simulator
            .solve(model)
            .map_err(|e| QbmError::SamplingError(e.to_string()))?;

        Ok(result.best_spins)
    }

    /// Compute free energy of a configuration
    fn free_energy(&self, visible: &[f64]) -> QbmResult<f64> {
        if visible.len() != self.visible_config.num_units {
            return Err(QbmError::DataError("Visible size mismatch".to_string()));
        }

        // Visible bias term
        let visible_term: f64 = visible
            .iter()
            .zip(self.visible_biases.iter())
            .map(|(&v, &b)| v * b)
            .sum();

        // Hidden term (sum of log(1 + exp(activation)) for each hidden unit)
        let hidden_term: f64 = (0..self.hidden_config.num_units)
            .map(|j| {
                let activation = self.hidden_biases[j]
                    + visible
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| v * self.weights[i][j])
                        .sum::<f64>();
                activation.exp().ln_1p()
            })
            .sum();

        Ok(-(visible_term + hidden_term))
    }

    /// Perform inference on input data
    pub fn infer(&mut self, input: &[f64]) -> QbmResult<QbmInferenceResult> {
        if input.len() != self.visible_config.num_units {
            return Err(QbmError::DataError("Input size mismatch".to_string()));
        }

        // Compute hidden activations
        let hidden_probs = self.sample_hidden_given_visible(input)?;
        let hidden_states = self.sample_binary_units(&hidden_probs)?;

        // Reconstruct visible units
        let reconstruction = self.sample_visible_given_hidden(&hidden_states)?;

        // Compute free energy and probability
        let free_energy = self.free_energy(input)?;
        let probability = (-free_energy).exp(); // Unnormalized

        Ok(QbmInferenceResult {
            reconstruction,
            hidden_activations: hidden_probs,
            free_energy,
            probability,
        })
    }

    /// Generate samples from the learned distribution
    pub fn generate_samples(&mut self, num_samples: usize) -> QbmResult<Vec<Vec<f64>>> {
        let mut samples = Vec::new();

        for _ in 0..num_samples {
            // Start with random visible state
            let mut visible: Vec<f64> = (0..self.visible_config.num_units)
                .map(|_| if self.rng.random_bool(0.5) { 1.0 } else { 0.0 })
                .collect();

            // Run Gibbs sampling for burn-in
            for _ in 0..100 {
                let hidden_probs = self.sample_hidden_given_visible(&visible)?;
                let hidden_states = self.sample_binary_units(&hidden_probs)?;
                visible = self.sample_visible_given_hidden(&hidden_states)?;
            }

            samples.push(visible);
        }

        Ok(samples)
    }

    /// Get training statistics
    #[must_use]
    pub const fn get_training_stats(&self) -> Option<&QbmTrainingStats> {
        self.training_stats.as_ref()
    }

    /// Serialize the learned parameters (biases, weights and layer dimensions)
    /// to a JSON file at `path`.
    ///
    /// Returns a [`QbmError::DataError`] if serialization or the file write
    /// fails. The on-disk format round-trips through [`load_model`].
    pub fn save_model(&self, path: &str) -> QbmResult<()> {
        let params = QbmModelParameters {
            num_visible: self.visible_config.num_units,
            num_hidden: self.hidden_config.num_units,
            visible_biases: self.visible_biases.clone(),
            hidden_biases: self.hidden_biases.clone(),
            weights: self.weights.clone(),
        };

        let json = serde_json::to_string_pretty(&params)
            .map_err(|e| QbmError::DataError(format!("Failed to serialize model: {e}")))?;

        std::fs::write(path, json)
            .map_err(|e| QbmError::DataError(format!("Failed to write model to {path}: {e}")))?;

        Ok(())
    }

    /// Load previously-saved parameters from the JSON file at `path`,
    /// overwriting the current biases and weights.
    ///
    /// Returns a [`QbmError::DataError`] on a missing/invalid file or a
    /// [`QbmError::InvalidModel`] if the stored layer dimensions do not match
    /// this machine's configuration (loading mismatched shapes would silently
    /// corrupt the model, so it is rejected).
    pub fn load_model(&mut self, path: &str) -> QbmResult<()> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| QbmError::DataError(format!("Failed to read model from {path}: {e}")))?;

        let params: QbmModelParameters = serde_json::from_str(&contents)
            .map_err(|e| QbmError::DataError(format!("Failed to deserialize model: {e}")))?;

        if params.num_visible != self.visible_config.num_units
            || params.num_hidden != self.hidden_config.num_units
        {
            return Err(QbmError::InvalidModel(format!(
                "Model dimensions ({}x{}) do not match this RBM ({}x{})",
                params.num_visible,
                params.num_hidden,
                self.visible_config.num_units,
                self.hidden_config.num_units
            )));
        }

        if params.visible_biases.len() != self.visible_config.num_units
            || params.hidden_biases.len() != self.hidden_config.num_units
            || params.weights.len() != self.visible_config.num_units
            || params
                .weights
                .iter()
                .any(|row| row.len() != self.hidden_config.num_units)
        {
            return Err(QbmError::InvalidModel(
                "Stored parameter arrays have inconsistent shapes".to_string(),
            ));
        }

        self.visible_biases = params.visible_biases;
        self.hidden_biases = params.hidden_biases;
        self.weights = params.weights;

        Ok(())
    }
}

/// Serializable snapshot of the learned RBM parameters used by
/// [`QuantumRestrictedBoltzmannMachine::save_model`] /
/// [`QuantumRestrictedBoltzmannMachine::load_model`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct QbmModelParameters {
    num_visible: usize,
    num_hidden: usize,
    visible_biases: Vec<f64>,
    hidden_biases: Vec<f64>,
    weights: Vec<Vec<f64>>,
}

impl QuantumSamplingStats {
    /// Merge another stats object into this one
    fn merge(&mut self, other: &Self) {
        self.total_sampling_time += other.total_sampling_time;
        self.sampling_calls += other.sampling_calls;

        if self.sampling_calls > 0 {
            let total_calls = self.sampling_calls as f64;
            self.average_annealing_energy = self.average_annealing_energy.mul_add(
                total_calls - other.sampling_calls as f64,
                other.average_annealing_energy * other.sampling_calls as f64,
            ) / total_calls;

            self.success_rate = self.success_rate.mul_add(
                total_calls - other.sampling_calls as f64,
                other.success_rate * other.sampling_calls as f64,
            ) / total_calls;

            self.classical_fallback_rate = self.classical_fallback_rate.mul_add(
                total_calls - other.sampling_calls as f64,
                other.classical_fallback_rate * other.sampling_calls as f64,
            ) / total_calls;
        }
    }
}

/// Sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply softmax normalization in-place
fn softmax_normalize(values: &mut [f64]) {
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values.iter().map(|&x| (x - max_val).exp()).sum();

    for value in values.iter_mut() {
        *value = (*value - max_val).exp() / sum;
    }
}

/// Helper functions for different QBM variants

/// Create a binary-binary RBM for typical unsupervised learning
pub fn create_binary_rbm(
    num_visible: usize,
    num_hidden: usize,
    training_config: QbmTrainingConfig,
) -> QbmResult<QuantumRestrictedBoltzmannMachine> {
    let visible_config = LayerConfig::new("visible".to_string(), num_visible, UnitType::Binary);
    let hidden_config = LayerConfig::new("hidden".to_string(), num_hidden, UnitType::Binary);

    QuantumRestrictedBoltzmannMachine::new(visible_config, hidden_config, training_config)
}

/// Create a Gaussian-Bernoulli RBM for continuous input data
pub fn create_gaussian_bernoulli_rbm(
    num_visible: usize,
    num_hidden: usize,
    training_config: QbmTrainingConfig,
) -> QbmResult<QuantumRestrictedBoltzmannMachine> {
    let visible_config = LayerConfig::new("visible".to_string(), num_visible, UnitType::Gaussian);
    let hidden_config = LayerConfig::new("hidden".to_string(), num_hidden, UnitType::Binary);

    QuantumRestrictedBoltzmannMachine::new(visible_config, hidden_config, training_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbm_creation() {
        let training_config = QbmTrainingConfig {
            epochs: 10,
            ..Default::default()
        };

        let rbm = create_binary_rbm(4, 3, training_config).expect("failed to create binary RBM");
        assert_eq!(rbm.visible_config.num_units, 4);
        assert_eq!(rbm.hidden_config.num_units, 3);
    }

    #[test]
    fn test_sigmoid_function() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_softmax_normalization() {
        let mut values = vec![1.0, 2.0, 3.0];
        softmax_normalize(&mut values);

        let sum: f64 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(values.iter().all(|&x| x > 0.0 && x < 1.0));
    }

    #[test]
    fn test_training_sample_creation() {
        let sample = TrainingSample::new(vec![1.0, 0.0, 1.0]);
        assert_eq!(sample.data.len(), 3);
        assert!(sample.label.is_none());

        let labeled_sample = TrainingSample::labeled(vec![1.0, 0.0], vec![1.0]);
        assert_eq!(labeled_sample.data.len(), 2);
        assert_eq!(
            labeled_sample
                .label
                .as_ref()
                .expect("label should exist")
                .len(),
            1
        );
    }

    #[test]
    fn test_layer_config() {
        let config = LayerConfig::new("test".to_string(), 10, UnitType::Binary)
            .with_bias_range(-0.5, 0.5)
            .with_quantum_sampling(false);

        assert_eq!(config.num_units, 10);
        assert_eq!(config.unit_type, UnitType::Binary);
        assert_eq!(config.bias_init_range, (-0.5, 0.5));
        assert!(!config.quantum_sampling);
    }
}
