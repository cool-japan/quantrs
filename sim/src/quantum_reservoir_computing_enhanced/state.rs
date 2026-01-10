//! State types for Quantum Reservoir Computing
//!
//! This module provides state structs for the QRC framework.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Memory analysis metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Linear memory capacity
    pub linear_capacity: f64,
    /// Nonlinear memory capacity
    pub nonlinear_capacity: f64,
    /// Total memory capacity
    pub total_capacity: f64,
    /// Information processing capacity
    pub processing_capacity: f64,
    /// Temporal correlation length
    pub correlation_length: f64,
    /// Memory decay rate
    pub decay_rate: f64,
    /// Memory efficiency
    pub efficiency: f64,
}

/// Enhanced quantum reservoir state
#[derive(Debug, Clone)]
pub struct QuantumReservoirState {
    /// Current quantum state vector
    pub state_vector: Array1<Complex64>,
    /// Evolution history buffer
    pub state_history: VecDeque<Array1<Complex64>>,
    /// Observable measurements cache
    pub observables: HashMap<String, f64>,
    /// Two-qubit correlation matrix
    pub correlations: Array2<f64>,
    /// Higher-order correlations
    pub higher_order_correlations: HashMap<String, f64>,
    /// Entanglement measures
    pub entanglement_measures: HashMap<String, f64>,
    /// Memory capacity metrics
    pub memory_metrics: MemoryMetrics,
    /// Time index counter
    pub time_index: usize,
    /// Last update timestamp
    pub last_update: f64,
    /// Reservoir activity level
    pub activity_level: f64,
    /// Performance tracking
    pub performance_history: VecDeque<f64>,
}

impl QuantumReservoirState {
    /// Create new enhanced reservoir state
    #[must_use]
    pub fn new(num_qubits: usize, memory_capacity: usize) -> Self {
        let state_size = 1 << num_qubits;
        let mut state_vector = Array1::zeros(state_size);
        state_vector[0] = Complex64::new(1.0, 0.0); // Start in |0...0⟩

        Self {
            state_vector,
            state_history: VecDeque::with_capacity(memory_capacity),
            observables: HashMap::new(),
            correlations: Array2::zeros((num_qubits, num_qubits)),
            higher_order_correlations: HashMap::new(),
            entanglement_measures: HashMap::new(),
            memory_metrics: MemoryMetrics::default(),
            time_index: 0,
            last_update: 0.0,
            activity_level: 0.0,
            performance_history: VecDeque::with_capacity(1000),
        }
    }

    /// Update state and maintain comprehensive history
    pub fn update_state(&mut self, new_state: Array1<Complex64>, timestamp: f64) {
        // Store previous state
        self.state_history.push_back(self.state_vector.clone());
        if self.state_history.len() > self.state_history.capacity() {
            self.state_history.pop_front();
        }

        // Update current state
        self.state_vector = new_state;
        self.time_index += 1;
        self.last_update = timestamp;

        // Update activity level
        self.update_activity_level();
    }

    /// Update reservoir activity level
    fn update_activity_level(&mut self) {
        let activity = self
            .state_vector
            .iter()
            .map(scirs2_core::Complex::norm_sqr)
            .sum::<f64>()
            / self.state_vector.len() as f64;

        // Exponential moving average
        let alpha = 0.1;
        self.activity_level = alpha * activity + (1.0 - alpha) * self.activity_level;
    }

    /// Calculate memory decay
    #[must_use]
    pub fn calculate_memory_decay(&self) -> f64 {
        if self.state_history.len() < 2 {
            return 0.0;
        }

        let mut total_decay = 0.0;
        let current_state = &self.state_vector;

        for (i, past_state) in self.state_history.iter().enumerate() {
            let fidelity = self.calculate_fidelity(current_state, past_state);
            let time_diff = (self.state_history.len() - i) as f64;
            total_decay += fidelity * (-time_diff * 0.1).exp();
        }

        total_decay / self.state_history.len() as f64
    }

    /// Calculate fidelity between two states
    fn calculate_fidelity(&self, state1: &Array1<Complex64>, state2: &Array1<Complex64>) -> f64 {
        let overlap = state1
            .iter()
            .zip(state2.iter())
            .map(|(a, b)| a.conj() * b)
            .sum::<Complex64>();
        overlap.norm_sqr()
    }
}

/// Enhanced training data for reservoir computing
#[derive(Debug, Clone)]
pub struct ReservoirTrainingData {
    /// Input time series
    pub inputs: Vec<Array1<f64>>,
    /// Target outputs
    pub targets: Vec<Array1<f64>>,
    /// Time stamps
    pub timestamps: Vec<f64>,
    /// Additional features
    pub features: Option<Vec<Array1<f64>>>,
    /// Data labels for classification
    pub labels: Option<Vec<usize>>,
    /// Sequence lengths for variable-length sequences
    pub sequence_lengths: Option<Vec<usize>>,
    /// Missing data indicators
    pub missing_mask: Option<Vec<Array1<bool>>>,
    /// Data weights for importance sampling
    pub sample_weights: Option<Vec<f64>>,
    /// Metadata for each sample
    pub metadata: Option<Vec<HashMap<String, String>>>,
}

impl ReservoirTrainingData {
    /// Create new training data
    #[must_use]
    pub const fn new(
        inputs: Vec<Array1<f64>>,
        targets: Vec<Array1<f64>>,
        timestamps: Vec<f64>,
    ) -> Self {
        Self {
            inputs,
            targets,
            timestamps,
            features: None,
            labels: None,
            sequence_lengths: None,
            missing_mask: None,
            sample_weights: None,
            metadata: None,
        }
    }

    /// Add features to training data
    #[must_use]
    pub fn with_features(mut self, features: Vec<Array1<f64>>) -> Self {
        self.features = Some(features);
        self
    }

    /// Add labels for classification
    #[must_use]
    pub fn with_labels(mut self, labels: Vec<usize>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Add sample weights
    #[must_use]
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.sample_weights = Some(weights);
        self
    }

    /// Get data length
    #[must_use]
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Check if data is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    /// Split data into train/test sets
    #[must_use]
    pub fn train_test_split(&self, test_ratio: f64) -> (Self, Self) {
        let test_size = (self.len() as f64 * test_ratio) as usize;
        let train_size = self.len() - test_size;

        let train_data = Self {
            inputs: self.inputs[..train_size].to_vec(),
            targets: self.targets[..train_size].to_vec(),
            timestamps: self.timestamps[..train_size].to_vec(),
            features: self.features.as_ref().map(|f| f[..train_size].to_vec()),
            labels: self.labels.as_ref().map(|l| l[..train_size].to_vec()),
            sequence_lengths: self
                .sequence_lengths
                .as_ref()
                .map(|s| s[..train_size].to_vec()),
            missing_mask: self.missing_mask.as_ref().map(|m| m[..train_size].to_vec()),
            sample_weights: self
                .sample_weights
                .as_ref()
                .map(|w| w[..train_size].to_vec()),
            metadata: self.metadata.as_ref().map(|m| m[..train_size].to_vec()),
        };

        let test_data = Self {
            inputs: self.inputs[train_size..].to_vec(),
            targets: self.targets[train_size..].to_vec(),
            timestamps: self.timestamps[train_size..].to_vec(),
            features: self.features.as_ref().map(|f| f[train_size..].to_vec()),
            labels: self.labels.as_ref().map(|l| l[train_size..].to_vec()),
            sequence_lengths: self
                .sequence_lengths
                .as_ref()
                .map(|s| s[train_size..].to_vec()),
            missing_mask: self.missing_mask.as_ref().map(|m| m[train_size..].to_vec()),
            sample_weights: self
                .sample_weights
                .as_ref()
                .map(|w| w[train_size..].to_vec()),
            metadata: self.metadata.as_ref().map(|m| m[train_size..].to_vec()),
        };

        (train_data, test_data)
    }
}

/// Enhanced training example for reservoir learning
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Input data
    pub input: Array1<f64>,
    /// Reservoir state after processing
    pub reservoir_state: Array1<f64>,
    /// Extracted features
    pub features: Array1<f64>,
    /// Target output
    pub target: Array1<f64>,
    /// Predicted output
    pub prediction: Array1<f64>,
    /// Prediction error
    pub error: f64,
    /// Confidence score
    pub confidence: f64,
    /// Processing timestamp
    pub timestamp: f64,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

/// Enhanced performance metrics for reservoir computing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReservoirMetrics {
    /// Total training examples processed
    pub training_examples: usize,
    /// Current prediction accuracy
    pub prediction_accuracy: f64,
    /// Memory capacity estimate
    pub memory_capacity: f64,
    /// Nonlinear memory capacity
    pub nonlinear_memory_capacity: f64,
    /// Information processing capacity
    pub processing_capacity: f64,
    /// Generalization error
    pub generalization_error: f64,
    /// Echo state property indicator
    pub echo_state_property: f64,
    /// Average processing time per input
    pub avg_processing_time_ms: f64,
    /// Quantum resource utilization
    pub quantum_resource_usage: f64,
    /// Temporal correlation length
    pub temporal_correlation_length: f64,
    /// Reservoir efficiency
    pub reservoir_efficiency: f64,
    /// Adaptation rate
    pub adaptation_rate: f64,
    /// Plasticity level
    pub plasticity_level: f64,
    /// Hardware utilization
    pub hardware_utilization: f64,
    /// Error mitigation overhead
    pub error_mitigation_overhead: f64,
    /// Quantum advantage metric
    pub quantum_advantage: f64,
    /// Computational complexity
    pub computational_complexity: f64,
}

/// Enhanced training result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Training error (RMSE)
    pub training_error: f64,
    /// Test error (RMSE)
    pub test_error: f64,
    /// Training time in milliseconds
    pub training_time_ms: f64,
    /// Number of training examples
    pub num_examples: usize,
    /// Echo state property measure
    pub echo_state_property: f64,
    /// Memory capacity estimate
    pub memory_capacity: f64,
    /// Nonlinear memory capacity
    pub nonlinear_capacity: f64,
    /// Information processing capacity
    pub processing_capacity: f64,
}
