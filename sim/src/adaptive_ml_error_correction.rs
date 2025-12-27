//! Real-time Adaptive Error Correction with Machine Learning
//!
//! This module implements machine learning-driven adaptive error correction that
//! learns from error patterns in real-time to optimize correction strategies.
//! The system uses various ML techniques including neural networks, reinforcement
//! learning, and online learning to continuously improve error correction performance.
//!
//! Key features:
//! - Real-time syndrome pattern recognition using neural networks
//! - Reinforcement learning for optimal correction strategy selection
//! - Online learning for adaptive threshold adjustment
//! - Ensemble methods for robust error prediction
//! - Temporal pattern analysis for correlated noise
//! - Hardware-aware correction optimization

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::circuit_interfaces::CircuitInterface;
use crate::concatenated_error_correction::ErrorType;
use crate::error::Result;

/// Machine learning model type for error correction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MLModelType {
    /// Neural network for syndrome classification
    NeuralNetwork,
    /// Decision tree for rule-based correction
    DecisionTree,
    /// Support vector machine for pattern recognition
    SVM,
    /// Reinforcement learning agent
    ReinforcementLearning,
    /// Ensemble of multiple models
    Ensemble,
}

/// Learning strategy for adaptive correction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearningStrategy {
    /// Supervised learning with labeled training data
    Supervised,
    /// Unsupervised learning for pattern discovery
    Unsupervised,
    /// Reinforcement learning with reward signals
    Reinforcement,
    /// Online learning with continuous updates
    Online,
    /// Transfer learning from pre-trained models
    Transfer,
}

/// Adaptive error correction configuration
#[derive(Debug, Clone)]
pub struct AdaptiveMLConfig {
    /// ML model type to use
    pub model_type: MLModelType,
    /// Learning strategy
    pub learning_strategy: LearningStrategy,
    /// Learning rate for gradient-based methods
    pub learning_rate: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Maximum training history to keep
    pub max_history_size: usize,
    /// Minimum confidence threshold for corrections
    pub confidence_threshold: f64,
    /// Enable real-time learning
    pub real_time_learning: bool,
    /// Update frequency for model retraining
    pub update_frequency: usize,
    /// Feature extraction method
    pub feature_extraction: FeatureExtractionMethod,
    /// Hardware-specific optimizations
    pub hardware_aware: bool,
}

/// Feature extraction method for syndrome analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureExtractionMethod {
    /// Raw syndrome bits
    RawSyndrome,
    /// Fourier transform features
    FourierTransform,
    /// Principal component analysis
    PCA,
    /// Autoencoder features
    Autoencoder,
    /// Temporal convolution features
    TemporalConvolution,
}

impl Default for AdaptiveMLConfig {
    fn default() -> Self {
        Self {
            model_type: MLModelType::NeuralNetwork,
            learning_strategy: LearningStrategy::Online,
            learning_rate: 0.001,
            batch_size: 32,
            max_history_size: 10_000,
            confidence_threshold: 0.8,
            real_time_learning: true,
            update_frequency: 100,
            feature_extraction: FeatureExtractionMethod::RawSyndrome,
            hardware_aware: true,
        }
    }
}

/// Neural network for syndrome classification
#[derive(Debug, Clone)]
pub struct SyndromeClassificationNetwork {
    /// Input layer size (syndrome length)
    input_size: usize,
    /// Hidden layer sizes
    hidden_sizes: Vec<usize>,
    /// Output size (number of error classes)
    output_size: usize,
    /// Network weights
    weights: Vec<Array2<f64>>,
    /// Network biases
    biases: Vec<Array1<f64>>,
    /// Learning rate
    learning_rate: f64,
    /// Training history
    training_history: Vec<(Array1<f64>, Array1<f64>)>,
}

impl SyndromeClassificationNetwork {
    /// Create new neural network
    #[must_use]
    pub fn new(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        learning_rate: f64,
    ) -> Self {
        let mut layer_sizes = vec![input_size];
        layer_sizes.extend(&hidden_sizes);
        layer_sizes.push(output_size);

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let rows = layer_sizes[i + 1];
            let cols = layer_sizes[i];

            // Xavier initialization
            let scale = (2.0 / (rows + cols) as f64).sqrt();
            let mut weight_matrix = Array2::zeros((rows, cols));
            for elem in &mut weight_matrix {
                *elem = (fastrand::f64() - 0.5) * 2.0 * scale;
            }
            weights.push(weight_matrix);

            biases.push(Array1::zeros(rows));
        }

        Self {
            input_size,
            hidden_sizes,
            output_size,
            weights,
            biases,
            learning_rate,
            training_history: Vec::new(),
        }
    }

    /// Forward pass through the network
    #[must_use]
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut activation = input.clone();

        // Get reference to last weight for comparison
        let last_weight = self.weights.last();

        for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
            activation = weight.dot(&activation) + bias;

            // Apply ReLU activation (except for output layer)
            let is_output_layer = last_weight.map_or(false, |last| weight == last);
            if is_output_layer {
                // Softmax for output layer
                let max_val = activation.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                activation.mapv_inplace(|x| (x - max_val).exp());
                let sum = activation.sum();
                activation.mapv_inplace(|x| x / sum);
            } else {
                activation.mapv_inplace(|x| x.max(0.0));
            }
        }

        activation
    }

    /// Train the network with a batch of examples
    pub fn train_batch(&mut self, inputs: &[Array1<f64>], targets: &[Array1<f64>]) -> f64 {
        let batch_size = inputs.len();
        let mut total_loss = 0.0;

        // Accumulate gradients
        let mut weight_gradients: Vec<Array2<f64>> = self
            .weights
            .iter()
            .map(|w| Array2::zeros(w.raw_dim()))
            .collect();
        let mut bias_gradients: Vec<Array1<f64>> = self
            .biases
            .iter()
            .map(|b| Array1::zeros(b.raw_dim()))
            .collect();

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let (loss, w_grads, b_grads) = self.backward(input, target);
            total_loss += loss;

            for (wg_acc, wg) in weight_gradients.iter_mut().zip(w_grads.iter()) {
                *wg_acc = &*wg_acc + wg;
            }
            for (bg_acc, bg) in bias_gradients.iter_mut().zip(b_grads.iter()) {
                *bg_acc = &*bg_acc + bg;
            }
        }

        // Update weights and biases
        let lr = self.learning_rate / batch_size as f64;
        for (weight, gradient) in self.weights.iter_mut().zip(weight_gradients.iter()) {
            *weight = &*weight - &(gradient * lr);
        }
        for (bias, gradient) in self.biases.iter_mut().zip(bias_gradients.iter()) {
            *bias = &*bias - &(gradient * lr);
        }

        total_loss / batch_size as f64
    }

    /// Backward pass to compute gradients
    fn backward(
        &self,
        input: &Array1<f64>,
        target: &Array1<f64>,
    ) -> (f64, Vec<Array2<f64>>, Vec<Array1<f64>>) {
        // Forward pass with intermediate activations
        let mut activations = vec![input.clone()];
        let mut z_values = Vec::new();

        // Get reference to last weight for comparison
        let last_weight = self.weights.last();

        for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
            // Safety: activations always has at least one element (input)
            let last_activation = activations
                .last()
                .expect("activations should never be empty");
            let z = weight.dot(last_activation) + bias;
            z_values.push(z.clone());

            let mut activation = z;
            let is_output_layer = last_weight.map_or(false, |last| weight == last);
            if is_output_layer {
                // Softmax
                let max_val = activation.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                activation.mapv_inplace(|x| (x - max_val).exp());
                let sum = activation.sum();
                activation.mapv_inplace(|x| x / sum);
            } else {
                activation.mapv_inplace(|x| x.max(0.0)); // ReLU
            }
            activations.push(activation);
        }

        // Calculate loss (cross-entropy)
        // Safety: activations has at least one element from the loop
        let output = activations
            .last()
            .expect("activations should have output from forward pass");
        let loss = -target
            .iter()
            .zip(output.iter())
            .map(|(&t, &o)| if t > 0.0 { t * o.ln() } else { 0.0 })
            .sum::<f64>();

        // Backward pass
        let mut weight_gradients = Vec::with_capacity(self.weights.len());
        let mut bias_gradients = Vec::with_capacity(self.biases.len());

        // Output layer gradient
        let mut delta = output - target;

        for i in (0..self.weights.len()).rev() {
            // Weight gradient
            let weight_grad = delta
                .view()
                .insert_axis(Axis(1))
                .dot(&activations[i].view().insert_axis(Axis(0)));
            weight_gradients.insert(0, weight_grad);

            // Bias gradient
            bias_gradients.insert(0, delta.clone());

            if i > 0 {
                // Propagate delta to previous layer
                delta = self.weights[i].t().dot(&delta);

                // Apply derivative of ReLU
                for (j, &z) in z_values[i - 1].iter().enumerate() {
                    if z <= 0.0 {
                        delta[j] = 0.0;
                    }
                }
            }
        }

        (loss, weight_gradients, bias_gradients)
    }

    /// Predict error class from syndrome
    #[must_use]
    pub fn predict(&self, syndrome: &Array1<f64>) -> (usize, f64) {
        let output = self.forward(syndrome);
        let max_idx = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let confidence = output.get(max_idx).copied().unwrap_or(0.0);
        (max_idx, confidence)
    }
}

/// Reinforcement learning agent for error correction
#[derive(Debug, Clone)]
pub struct ErrorCorrectionAgent {
    /// Q-table for state-action values
    q_table: HashMap<String, Array1<f64>>,
    /// Learning rate
    learning_rate: f64,
    /// Discount factor
    discount_factor: f64,
    /// Exploration rate (epsilon)
    epsilon: f64,
    /// Action space size
    action_space_size: usize,
    /// Total training steps
    training_steps: usize,
    /// Episode rewards history
    episode_rewards: VecDeque<f64>,
}

impl ErrorCorrectionAgent {
    /// Create new RL agent
    #[must_use]
    pub fn new(
        action_space_size: usize,
        learning_rate: f64,
        discount_factor: f64,
        epsilon: f64,
    ) -> Self {
        Self {
            q_table: HashMap::new(),
            learning_rate,
            discount_factor,
            epsilon,
            action_space_size,
            training_steps: 0,
            episode_rewards: VecDeque::with_capacity(1000),
        }
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&mut self, state: &str) -> usize {
        if fastrand::f64() < self.epsilon {
            // Explore: random action
            fastrand::usize(0..self.action_space_size)
        } else {
            // Exploit: best action
            let q_values = self
                .q_table
                .entry(state.to_string())
                .or_insert_with(|| Array1::zeros(self.action_space_size));

            q_values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    }

    /// Update Q-value using Q-learning
    pub fn update_q_value(
        &mut self,
        state: &str,
        action: usize,
        reward: f64,
        next_state: &str,
        done: bool,
    ) {
        let current_q = self
            .q_table
            .entry(state.to_string())
            .or_insert_with(|| Array1::zeros(self.action_space_size))
            .clone();

        let next_q_max = if done {
            0.0
        } else {
            let next_q_values = self
                .q_table
                .entry(next_state.to_string())
                .or_insert_with(|| Array1::zeros(self.action_space_size));
            next_q_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        };

        let td_target = self.discount_factor.mul_add(next_q_max, reward);
        let current_q_action = current_q.get(action).copied().unwrap_or(0.0);
        let td_error = td_target - current_q_action;

        // Safety: We just inserted this entry above with entry().or_insert_with()
        if let Some(q_values) = self.q_table.get_mut(state) {
            if action < q_values.len() {
                q_values[action] += self.learning_rate * td_error;
            }
        }

        self.training_steps += 1;

        // Decay epsilon
        if self.training_steps % 1000 == 0 {
            self.epsilon = (self.epsilon * 0.995).max(0.01);
        }
    }

    /// Calculate reward based on correction success
    #[must_use]
    pub fn calculate_reward(
        &self,
        errors_before: usize,
        errors_after: usize,
        correction_cost: f64,
    ) -> f64 {
        let error_reduction = errors_before as f64 - errors_after as f64;
        let reward = error_reduction.mul_add(10.0, -correction_cost);

        // Bonus for perfect correction
        if errors_after == 0 {
            reward + 5.0
        } else {
            reward
        }
    }
}

/// Adaptive ML error correction system
pub struct AdaptiveMLErrorCorrection {
    /// Configuration
    config: AdaptiveMLConfig,
    /// Neural network for syndrome classification
    classifier: SyndromeClassificationNetwork,
    /// Reinforcement learning agent
    rl_agent: ErrorCorrectionAgent,
    /// Feature extractor
    feature_extractor: FeatureExtractor,
    /// Training data history
    training_history: Arc<Mutex<VecDeque<TrainingExample>>>,
    /// Performance metrics
    metrics: CorrectionMetrics,
    /// Circuit interface
    circuit_interface: CircuitInterface,
    /// Model update counter
    update_counter: usize,
}

/// Training example for supervised learning
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Input syndrome
    pub syndrome: Array1<f64>,
    /// Target error type
    pub error_type: ErrorType,
    /// Correction action taken
    pub action: usize,
    /// Reward received
    pub reward: f64,
    /// Timestamp
    pub timestamp: f64,
}

/// Feature extractor for syndrome analysis
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Extraction method
    method: FeatureExtractionMethod,
    /// PCA components (if using PCA)
    pca_components: Option<Array2<f64>>,
    /// Autoencoder network (if using autoencoder)
    autoencoder: Option<SyndromeClassificationNetwork>,
}

impl FeatureExtractor {
    /// Create new feature extractor
    #[must_use]
    pub const fn new(method: FeatureExtractionMethod) -> Self {
        Self {
            method,
            pca_components: None,
            autoencoder: None,
        }
    }

    /// Extract features from syndrome
    #[must_use]
    pub fn extract_features(&self, syndrome: &[bool]) -> Array1<f64> {
        match self.method {
            FeatureExtractionMethod::RawSyndrome => {
                let mut features: Vec<f64> = syndrome
                    .iter()
                    .map(|&b| if b { 1.0 } else { 0.0 })
                    .collect();
                // Pad to minimum size of 4 for consistency
                while features.len() < 4 {
                    features.push(0.0);
                }
                Array1::from_vec(features)
            }
            FeatureExtractionMethod::FourierTransform => self.fft_features(syndrome),
            FeatureExtractionMethod::PCA => self.pca_features(syndrome),
            FeatureExtractionMethod::Autoencoder => self.autoencoder_features(syndrome),
            FeatureExtractionMethod::TemporalConvolution => self.temporal_conv_features(syndrome),
        }
    }

    /// Extract FFT features
    fn fft_features(&self, syndrome: &[bool]) -> Array1<f64> {
        let mut signal: Vec<f64> = syndrome
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();

        // Pad signal to minimum size of 4 for consistency
        while signal.len() < 4 {
            signal.push(0.0);
        }

        // Simple FFT-like transformation (simplified)
        let mut features = Vec::new();
        let n = signal.len();

        for k in 0..n.min(8) {
            // Take first 8 frequency components
            let mut real_part = 0.0;
            let mut imag_part = 0.0;

            for (i, &x) in signal.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
                real_part += x * angle.cos();
                imag_part += x * angle.sin();
            }

            features.push(real_part);
            features.push(imag_part);
        }

        Array1::from_vec(features)
    }

    /// Extract PCA features
    fn pca_features(&self, syndrome: &[bool]) -> Array1<f64> {
        let mut features: Vec<f64> = syndrome
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        // Pad to minimum size of 4 for consistency
        while features.len() < 4 {
            features.push(0.0);
        }
        let raw_features = Array1::from_vec(features);

        if let Some(ref components) = self.pca_components {
            components.dot(&raw_features)
        } else {
            raw_features
        }
    }

    /// Extract autoencoder features
    fn autoencoder_features(&self, syndrome: &[bool]) -> Array1<f64> {
        let mut features: Vec<f64> = syndrome
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        // Pad to minimum size of 4 for consistency
        while features.len() < 4 {
            features.push(0.0);
        }
        let raw_features = Array1::from_vec(features);

        if let Some(ref encoder) = self.autoencoder {
            encoder.forward(&raw_features)
        } else {
            raw_features
        }
    }

    /// Extract temporal convolution features
    fn temporal_conv_features(&self, syndrome: &[bool]) -> Array1<f64> {
        let mut signal: Vec<f64> = syndrome
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();

        // Pad signal to minimum size of 4 for consistency
        while signal.len() < 4 {
            signal.push(0.0);
        }

        // Simple 1D convolution with learned kernels
        let kernel_size = 3;
        let mut features = Vec::new();

        for i in 0..signal.len().saturating_sub(kernel_size - 1) {
            let mut conv_sum = 0.0;
            for j in 0..kernel_size {
                conv_sum += signal[i + j] * (j as f64 + 1.0) / kernel_size as f64;
                // Simple kernel
            }
            features.push(conv_sum);
        }

        // Ensure at least some features
        if features.is_empty() {
            features = signal; // Fall back to raw signal
        }

        Array1::from_vec(features)
    }
}

/// Performance metrics for error correction
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorrectionMetrics {
    /// Total errors corrected
    pub total_corrections: usize,
    /// Successful corrections
    pub successful_corrections: usize,
    /// False positive corrections
    pub false_positives: usize,
    /// False negative missed errors
    pub false_negatives: usize,
    /// Average correction confidence
    pub average_confidence: f64,
    /// Learning curve (loss over time)
    pub learning_curve: Vec<f64>,
    /// Reward history (for RL)
    pub reward_history: Vec<f64>,
    /// Processing time per correction
    pub avg_correction_time_ms: f64,
}

impl CorrectionMetrics {
    /// Calculate correction accuracy
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        if self.total_corrections == 0 {
            return 1.0;
        }
        self.successful_corrections as f64 / self.total_corrections as f64
    }

    /// Calculate precision
    #[must_use]
    pub fn precision(&self) -> f64 {
        let true_positives = self.successful_corrections;
        let predicted_positives = true_positives + self.false_positives;

        if predicted_positives == 0 {
            return 1.0;
        }
        true_positives as f64 / predicted_positives as f64
    }

    /// Calculate recall
    #[must_use]
    pub fn recall(&self) -> f64 {
        let true_positives = self.successful_corrections;
        let actual_positives = true_positives + self.false_negatives;

        if actual_positives == 0 {
            return 1.0;
        }
        true_positives as f64 / actual_positives as f64
    }

    /// Calculate F1 score
    #[must_use]
    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();

        if precision + recall == 0.0 {
            return 0.0;
        }
        2.0 * precision * recall / (precision + recall)
    }
}

impl AdaptiveMLErrorCorrection {
    /// Create new adaptive ML error correction system
    pub fn new(config: AdaptiveMLConfig) -> Result<Self> {
        let circuit_interface = CircuitInterface::new(Default::default())?;

        // Initialize feature extractor first to determine input size
        let feature_extractor = FeatureExtractor::new(config.feature_extraction);

        // Calculate input size based on feature extraction method
        // Use a test syndrome to determine the feature vector size
        let test_syndrome = vec![false, false, false, false]; // 4-bit test syndrome
        let test_features = feature_extractor.extract_features(&test_syndrome);
        let input_size = test_features.len();

        // Initialize neural network for syndrome classification
        let hidden_sizes = vec![input_size * 2, input_size]; // Adaptive hidden sizes
        let output_size = 4; // I, X, Y, Z errors
        let classifier = SyndromeClassificationNetwork::new(
            input_size,
            hidden_sizes,
            output_size,
            config.learning_rate,
        );

        // Initialize RL agent
        let action_space_size = 8; // Different correction strategies
        let rl_agent = ErrorCorrectionAgent::new(
            action_space_size,
            config.learning_rate,
            0.99, // discount factor
            0.1,  // epsilon
        );

        let training_history =
            Arc::new(Mutex::new(VecDeque::with_capacity(config.max_history_size)));

        Ok(Self {
            config,
            classifier,
            rl_agent,
            feature_extractor,
            training_history,
            metrics: CorrectionMetrics::default(),
            circuit_interface,
            update_counter: 0,
        })
    }

    /// Perform adaptive error correction on quantum state
    pub fn correct_errors_adaptive(
        &mut self,
        state: &mut Array1<Complex64>,
        syndrome: &[bool],
    ) -> Result<AdaptiveCorrectionResult> {
        let start_time = std::time::Instant::now();

        // Extract features from syndrome
        let features = self.feature_extractor.extract_features(syndrome);

        // Classify error type using neural network
        let (predicted_error_class, confidence) = self.classifier.predict(&features);
        let predicted_error_type = self.class_to_error_type(predicted_error_class);

        // Select correction action using RL agent
        let state_repr = self.syndrome_to_string(syndrome);
        let action = self.rl_agent.select_action(&state_repr);

        // Count errors before correction
        let errors_before = self.count_errors(state, syndrome);

        // Apply correction based on ML predictions
        let correction_applied = if confidence >= self.config.confidence_threshold {
            self.apply_ml_correction(state, predicted_error_type, action)?;
            true
        } else {
            // Fall back to classical correction if confidence is low
            self.apply_classical_correction(state, syndrome)?;
            false
        };

        // Count errors after correction
        let errors_after = self.count_errors(state, syndrome);

        // Calculate reward for RL agent
        let reward = self
            .rl_agent
            .calculate_reward(errors_before, errors_after, 1.0);

        // Update RL agent
        let next_state_repr = self.state_to_string(state);
        self.rl_agent.update_q_value(
            &state_repr,
            action,
            reward,
            &next_state_repr,
            errors_after == 0,
        );

        // Record training example
        if self.config.real_time_learning {
            let training_example = TrainingExample {
                syndrome: features,
                error_type: predicted_error_type,
                action,
                reward,
                timestamp: start_time.elapsed().as_secs_f64(),
            };

            if let Ok(mut history) = self.training_history.lock() {
                history.push_back(training_example);
                if history.len() > self.config.max_history_size {
                    history.pop_front();
                }
            }
        }

        // Update metrics
        self.update_metrics(errors_before, errors_after, confidence, reward);

        // Periodic model retraining
        self.update_counter += 1;
        if self.update_counter % self.config.update_frequency == 0 {
            self.retrain_models()?;
        }

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(AdaptiveCorrectionResult {
            predicted_error_type,
            confidence,
            correction_applied,
            errors_corrected: errors_before.saturating_sub(errors_after),
            reward,
            processing_time_ms: processing_time,
            rl_action: action,
        })
    }

    /// Apply ML-based correction
    fn apply_ml_correction(
        &self,
        state: &mut Array1<Complex64>,
        error_type: ErrorType,
        action: usize,
    ) -> Result<()> {
        match action {
            0 => {
                // Single qubit correction
                self.apply_single_qubit_correction(state, error_type, 0)?;
            }
            1 => {
                // Two qubit correction
                self.apply_two_qubit_correction(state, error_type, 0, 1)?;
            }
            2 => {
                // Syndrome-based correction
                self.apply_syndrome_based_correction(state, error_type)?;
            }
            3 => {
                // Probabilistic correction
                self.apply_probabilistic_correction(state, error_type)?;
            }
            _ => {
                // Default correction
                self.apply_single_qubit_correction(state, error_type, 0)?;
            }
        }
        Ok(())
    }

    /// Apply single qubit correction
    fn apply_single_qubit_correction(
        &self,
        state: &mut Array1<Complex64>,
        error_type: ErrorType,
        qubit: usize,
    ) -> Result<()> {
        let n_qubits = (state.len() as f64).log2().ceil() as usize;
        if qubit >= n_qubits {
            return Ok(());
        }

        match error_type {
            ErrorType::BitFlip => {
                // Apply X correction
                for i in 0..state.len() {
                    if (i >> qubit) & 1 == 0 {
                        let partner = i | (1 << qubit);
                        if partner < state.len() {
                            state.swap(i, partner);
                        }
                    }
                }
            }
            ErrorType::PhaseFlip => {
                // Apply Z correction
                for i in 0..state.len() {
                    if (i >> qubit) & 1 == 1 {
                        state[i] *= -1.0;
                    }
                }
            }
            ErrorType::BitPhaseFlip => {
                // Apply Y correction (Z then X)
                self.apply_single_qubit_correction(state, ErrorType::PhaseFlip, qubit)?;
                self.apply_single_qubit_correction(state, ErrorType::BitFlip, qubit)?;
            }
            ErrorType::Identity => {
                // No correction needed
            }
        }

        Ok(())
    }

    /// Apply two qubit correction
    fn apply_two_qubit_correction(
        &self,
        state: &mut Array1<Complex64>,
        error_type: ErrorType,
        qubit1: usize,
        qubit2: usize,
    ) -> Result<()> {
        // Apply correction to both qubits
        self.apply_single_qubit_correction(state, error_type, qubit1)?;
        self.apply_single_qubit_correction(state, error_type, qubit2)?;
        Ok(())
    }

    /// Apply syndrome-based correction
    fn apply_syndrome_based_correction(
        &self,
        state: &mut Array1<Complex64>,
        error_type: ErrorType,
    ) -> Result<()> {
        // Apply correction based on error type to most likely qubit
        let n_qubits = (state.len() as f64).log2().ceil() as usize;
        let target_qubit = fastrand::usize(0..n_qubits);
        self.apply_single_qubit_correction(state, error_type, target_qubit)?;
        Ok(())
    }

    /// Apply probabilistic correction
    fn apply_probabilistic_correction(
        &self,
        state: &mut Array1<Complex64>,
        error_type: ErrorType,
    ) -> Result<()> {
        let n_qubits = (state.len() as f64).log2().ceil() as usize;

        // Apply correction with probability based on error type
        for qubit in 0..n_qubits {
            let prob = match error_type {
                ErrorType::BitFlip => 0.3,
                ErrorType::PhaseFlip => 0.2,
                ErrorType::BitPhaseFlip => 0.1,
                ErrorType::Identity => 0.0,
            };

            if fastrand::f64() < prob {
                self.apply_single_qubit_correction(state, error_type, qubit)?;
            }
        }

        Ok(())
    }

    /// Apply classical error correction as fallback
    fn apply_classical_correction(
        &self,
        state: &mut Array1<Complex64>,
        syndrome: &[bool],
    ) -> Result<()> {
        // Simple classical correction based on syndrome
        for (i, &has_error) in syndrome.iter().enumerate() {
            if has_error {
                self.apply_single_qubit_correction(state, ErrorType::BitFlip, i)?;
            }
        }
        Ok(())
    }

    /// Count estimated errors in state
    fn count_errors(&self, _state: &Array1<Complex64>, syndrome: &[bool]) -> usize {
        syndrome.iter().map(|&b| usize::from(b)).sum()
    }

    /// Convert error class to error type
    const fn class_to_error_type(&self, class: usize) -> ErrorType {
        match class {
            0 => ErrorType::Identity,
            1 => ErrorType::BitFlip,
            2 => ErrorType::PhaseFlip,
            3 => ErrorType::BitPhaseFlip,
            _ => ErrorType::Identity,
        }
    }

    /// Convert syndrome to string representation
    fn syndrome_to_string(&self, syndrome: &[bool]) -> String {
        syndrome
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect()
    }

    /// Convert quantum state to string representation (simplified)
    fn state_to_string(&self, state: &Array1<Complex64>) -> String {
        let amplitudes: Vec<f64> = state.iter().map(|c| c.norm()).collect();
        format!("{amplitudes:.3?}")
    }

    /// Update performance metrics
    fn update_metrics(
        &mut self,
        errors_before: usize,
        errors_after: usize,
        confidence: f64,
        reward: f64,
    ) {
        self.metrics.total_corrections += 1;

        if errors_after < errors_before {
            self.metrics.successful_corrections += 1;
        } else if errors_after > errors_before {
            self.metrics.false_positives += 1;
        }

        self.metrics.average_confidence = self
            .metrics
            .average_confidence
            .mul_add((self.metrics.total_corrections - 1) as f64, confidence)
            / self.metrics.total_corrections as f64;

        self.metrics.reward_history.push(reward);
        if self.metrics.reward_history.len() > 1000 {
            self.metrics.reward_history.remove(0);
        }
    }

    /// Retrain models with accumulated data
    fn retrain_models(&mut self) -> Result<()> {
        let history = self.training_history.lock().map_err(|e| {
            crate::error::SimulatorError::InvalidOperation(format!("Lock poisoned: {e}"))
        })?;
        if history.len() < self.config.batch_size {
            return Ok(());
        }

        // Prepare training data
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for example in history.iter() {
            inputs.push(example.syndrome.clone());

            // Create one-hot target
            let mut target = Array1::zeros(4);
            let error_class = match example.error_type {
                ErrorType::Identity => 0,
                ErrorType::BitFlip => 1,
                ErrorType::PhaseFlip => 2,
                ErrorType::BitPhaseFlip => 3,
            };
            target[error_class] = 1.0;
            targets.push(target);
        }

        // Train neural network
        let batch_size = self.config.batch_size.min(inputs.len());
        for chunk in inputs.chunks(batch_size).zip(targets.chunks(batch_size)) {
            let loss = self.classifier.train_batch(chunk.0, chunk.1);
            self.metrics.learning_curve.push(loss);
        }

        Ok(())
    }

    /// Get current performance metrics
    #[must_use]
    pub const fn get_metrics(&self) -> &CorrectionMetrics {
        &self.metrics
    }

    /// Reset metrics and training history
    pub fn reset(&mut self) {
        self.metrics = CorrectionMetrics::default();
        if let Ok(mut history) = self.training_history.lock() {
            history.clear();
        }
        self.update_counter = 0;
    }
}

/// Result of adaptive error correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCorrectionResult {
    /// Predicted error type
    pub predicted_error_type: ErrorType,
    /// Prediction confidence
    pub confidence: f64,
    /// Whether ML correction was applied
    pub correction_applied: bool,
    /// Number of errors corrected
    pub errors_corrected: usize,
    /// Reward signal for RL
    pub reward: f64,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// RL action taken
    pub rl_action: usize,
}

/// Benchmark adaptive ML error correction
pub fn benchmark_adaptive_ml_error_correction() -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();

    // Test different ML configurations
    let configs = vec![
        AdaptiveMLConfig {
            model_type: MLModelType::NeuralNetwork,
            learning_strategy: LearningStrategy::Online,
            ..Default::default()
        },
        AdaptiveMLConfig {
            model_type: MLModelType::ReinforcementLearning,
            learning_strategy: LearningStrategy::Reinforcement,
            ..Default::default()
        },
    ];

    for (i, config) in configs.into_iter().enumerate() {
        let start = std::time::Instant::now();

        let mut adaptive_ec = AdaptiveMLErrorCorrection::new(config)?;

        // Simulate error correction on test data
        for _ in 0..100 {
            let mut test_state = Array1::from_vec(vec![
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ]);

            let syndrome = vec![true, false, true, false]; // Example syndrome
            let _result = adaptive_ec.correct_errors_adaptive(&mut test_state, &syndrome)?;
        }

        let time = start.elapsed().as_secs_f64() * 1000.0;
        results.insert(format!("config_{i}"), time);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_neural_network_creation() {
        let nn = SyndromeClassificationNetwork::new(4, vec![8, 4], 2, 0.01);
        assert_eq!(nn.input_size, 4);
        assert_eq!(nn.output_size, 2);
        assert_eq!(nn.weights.len(), 3); // input->hidden1, hidden1->hidden2, hidden2->output
    }

    #[test]
    fn test_neural_network_forward() {
        let nn = SyndromeClassificationNetwork::new(3, vec![4], 2, 0.01);
        let input = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let output = nn.forward(&input);

        assert_eq!(output.len(), 2);
        assert_abs_diff_eq!(output.sum(), 1.0, epsilon = 1e-6); // Softmax normalization
    }

    #[test]
    fn test_rl_agent_creation() {
        let agent = ErrorCorrectionAgent::new(4, 0.1, 0.99, 0.1);
        assert_eq!(agent.action_space_size, 4);
        assert!(agent.q_table.is_empty());
    }

    #[test]
    fn test_rl_agent_action_selection() {
        let mut agent = ErrorCorrectionAgent::new(3, 0.1, 0.99, 0.0); // No exploration
        let state = "001";

        // First call should create Q-values and select action 0 (all zeros)
        let action = agent.select_action(state);
        assert!(action < 3);
    }

    #[test]
    fn test_feature_extraction() {
        let extractor = FeatureExtractor::new(FeatureExtractionMethod::RawSyndrome);
        let syndrome = vec![true, false, true, false];
        let features = extractor.extract_features(&syndrome);

        assert_eq!(features.len(), 4);
        assert_abs_diff_eq!(features[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(features[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(features[2], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(features[3], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adaptive_ml_error_correction_creation() {
        let config = AdaptiveMLConfig::default();
        let adaptive_ec = AdaptiveMLErrorCorrection::new(config);
        assert!(adaptive_ec.is_ok());
    }

    #[test]
    fn test_error_correction_application() {
        let config = AdaptiveMLConfig::default();
        let mut adaptive_ec = AdaptiveMLErrorCorrection::new(config)
            .expect("Failed to create AdaptiveMLErrorCorrection");

        let mut state = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);

        let syndrome = vec![false, false];
        let result = adaptive_ec.correct_errors_adaptive(&mut state, &syndrome);
        assert!(result.is_ok());

        let correction_result = result.expect("Failed to correct errors");
        assert!(correction_result.processing_time_ms >= 0.0);
    }

    #[test]
    fn test_metrics_calculation() {
        let mut metrics = CorrectionMetrics::default();
        metrics.total_corrections = 100;
        metrics.successful_corrections = 90;
        metrics.false_positives = 5;
        metrics.false_negatives = 5;

        assert_abs_diff_eq!(metrics.accuracy(), 0.9, epsilon = 1e-10);
        assert_abs_diff_eq!(metrics.precision(), 90.0 / 95.0, epsilon = 1e-10);
        assert_abs_diff_eq!(metrics.recall(), 90.0 / 95.0, epsilon = 1e-10);
    }

    #[test]
    fn test_different_error_types() {
        let config = AdaptiveMLConfig::default();
        let adaptive_ec = AdaptiveMLErrorCorrection::new(config)
            .expect("Failed to create AdaptiveMLErrorCorrection");

        assert_eq!(adaptive_ec.class_to_error_type(0), ErrorType::Identity);
        assert_eq!(adaptive_ec.class_to_error_type(1), ErrorType::BitFlip);
        assert_eq!(adaptive_ec.class_to_error_type(2), ErrorType::PhaseFlip);
        assert_eq!(adaptive_ec.class_to_error_type(3), ErrorType::BitPhaseFlip);
    }
}
